"""Query validation.

Every plan crosses this module before anything is fetched. The validator answers one
question: *is this plan inside the envelope the allowlist defines?* It never repairs
and never guesses -- it reports issues, and the caller decides whether to ask the model
to try again or to fail the request.

Two classes of problem are distinguished, because they mean different things:

``error``
    The plan would produce a request outside the policy. Execution is refused.
``warning``
    The plan is executable but suspicious -- typically a code that is not in our
    terminology subset, which is the signature of an LLM hallucinating an identifier.
    The benchmark counts these separately from outright failures.
"""

from __future__ import annotations

from fhir_healthcare_ai.domain.enums import ResourceType, Severity, StepRole
from fhir_healthcare_ai.domain.query import QueryPlan, QueryStep, SearchParam, ValidationResult
from fhir_healthcare_ai.fhir.allowlist import (
    CLINICAL_CODE_SYSTEMS,
    ParamPolicy,
    ParamType,
    ResourcePolicy,
    allowed_params,
    allowed_resource_names,
    get_policy,
)
from fhir_healthcare_ai.logging_config import get_logger
from fhir_healthcare_ai.terminology import resolve_coding

logger = get_logger(__name__)

MAX_STEPS_PER_PLAN = 8
MAX_PARAMS_PER_STEP = 12

# Parameters that must be expressed through dedicated QueryStep fields. Allowing them
# inside `params` would reopen the door to arbitrary control parameters.
_CONTROL_PARAMS = frozenset(
    {"_include", "_revinclude", "_sort", "_count", "_elements", "_summary", "_query", "_format"}
)

# A patient-scoped search with no selective filter would stream the whole resource type.
_SELECTIVE_PARAMS = frozenset(
    {"_id", "patient", "subject", "encounter", "code", "combo-code", "category", "identifier"}
)

# `analysis.options` is a free-form dict in the schema, so it is allowlisted here like
# everything else a model writes: key -> required value type.
_ANALYSIS_OPTIONS: dict[str, type] = {"require_abnormal": bool}


class QueryValidator:
    """Validates plans and steps against the allowlist."""

    def __init__(
        self,
        *,
        max_page_size: int = 200,
        max_steps: int = MAX_STEPS_PER_PLAN,
        strict_codes: bool = False,
    ) -> None:
        """
        Args:
            max_page_size: Upper bound on ``_count`` regardless of resource policy.
            max_steps: Upper bound on plan size, to cap fan-out.
            strict_codes: Treat codes absent from the terminology subset as errors
                instead of warnings.
        """
        self.max_page_size = max_page_size
        self.max_steps = max_steps
        self.strict_codes = strict_codes

    # -- plan level -----------------------------------------------------------------

    def validate_plan(self, plan: QueryPlan) -> ValidationResult:
        """Validate a whole plan. Safe to call on untrusted, model-generated input."""
        result = ValidationResult()

        if plan.unsupported:
            result.add(
                Severity.INFORMATION,
                "unsupported-question",
                plan.unsupported_reason or "Planner marked the question as unsupported.",
            )
            return result

        if not plan.steps:
            result.add(
                Severity.ERROR, "empty-plan", "Plan contains no retrieval steps.", "plan.steps"
            )
            return result

        if len(plan.steps) > self.max_steps:
            result.add(
                Severity.ERROR,
                "too-many-steps",
                f"Plan has {len(plan.steps)} steps; the limit is {self.max_steps}.",
                "plan.steps",
            )

        if all(step.role is StepRole.EXCLUDE for step in plan.steps):
            result.add(
                Severity.ERROR,
                "exclusion-only-plan",
                "Every step excludes patients, so there is no cohort to exclude them from. "
                "Add a filter step that selects the population first.",
                "plan.steps",
            )

        for key, value in plan.analysis.options.items():
            if key not in _ANALYSIS_OPTIONS:
                result.add(
                    Severity.ERROR,
                    "unknown-analysis-option",
                    f"Analysis option {key!r} is not supported; allowed: "
                    f"{', '.join(sorted(_ANALYSIS_OPTIONS))}.",
                    "plan.analysis.options",
                )
            elif not isinstance(value, _ANALYSIS_OPTIONS[key]):
                result.add(
                    Severity.ERROR,
                    "invalid-analysis-option",
                    f"Analysis option {key!r} must be a {_ANALYSIS_OPTIONS[key].__name__}.",
                    "plan.analysis.options",
                )

        for concept_ref in plan.analysis.concepts:
            if not concept_ref.replace("_", "").isalnum():
                result.add(
                    Severity.ERROR,
                    "invalid-concept",
                    f"Analysis concept {concept_ref!r} is not a valid concept key.",
                    "plan.analysis.concepts",
                )

        for index, step in enumerate(plan.steps):
            step_result = self.validate_step(step, location=f"plan.steps[{index}]")
            result = result.merge(step_result)

        if not result.ok:
            logger.warning(
                "plan validation failed",
                extra={"issue_count": len(result.errors), "question": plan.question[:200]},
            )
        return result

    # -- step level -----------------------------------------------------------------

    def validate_step(self, step: QueryStep, location: str = "step") -> ValidationResult:
        """Validate a single retrieval step."""
        result = ValidationResult()
        policy = get_policy(step.resource_type)
        if policy is None:
            result.add(
                Severity.ERROR,
                "resource-not-allowed",
                (
                    f"Resource {step.resource_type.value!r} is not reachable. "
                    f"Allowed: {', '.join(allowed_resource_names())}."
                ),
                f"{location}.resource_type",
            )
            return result

        if len(step.params) > MAX_PARAMS_PER_STEP:
            result.add(
                Severity.ERROR,
                "too-many-params",
                f"Step has {len(step.params)} parameters; the limit is {MAX_PARAMS_PER_STEP}.",
                f"{location}.params",
            )

        seen: set[str] = set()
        for index, param in enumerate(step.params):
            param_location = f"{location}.params[{index}]"
            if param.key in seen:
                result.add(
                    Severity.WARNING,
                    "duplicate-param",
                    f"Parameter {param.key!r} appears more than once.",
                    param_location,
                )
            seen.add(param.key)
            result = result.merge(
                self._validate_param(param, policy, step.resource_type, param_location)
            )

        result = result.merge(self._validate_controls(step, policy, location))
        result = result.merge(self._validate_selectivity(step, policy, seen, location))
        return result

    # -- internals ------------------------------------------------------------------

    def _validate_param(
        self,
        param: SearchParam,
        policy: ResourcePolicy,
        resource_type: ResourceType,
        location: str,
    ) -> ValidationResult:
        result = ValidationResult()

        if param.name in _CONTROL_PARAMS:
            result.add(
                Severity.ERROR,
                "control-param-in-params",
                (
                    f"{param.name!r} is a control parameter and must be expressed through the "
                    "dedicated step fields (include / revinclude / sort / count)."
                ),
                location,
            )
            return result

        param_policy = policy.param(param.name)
        if param_policy is None:
            result.add(
                Severity.ERROR,
                "param-not-allowed",
                (
                    f"{resource_type.value} does not allow search parameter {param.name!r}. "
                    f"Allowed: {', '.join(allowed_params(resource_type))}."
                ),
                location,
            )
            return result

        if not param_policy.allows_comparator(param.comparator):
            allowed = ", ".join(sorted(c.value for c in param_policy.comparators)) or "none"
            result.add(
                Severity.ERROR,
                "comparator-not-allowed",
                (
                    f"Comparator {param.comparator} is not valid for "
                    f"{resource_type.value}.{param.name} (allowed: {allowed})."
                ),
                location,
            )

        if not param_policy.allows_modifier(param.modifier):
            allowed = ", ".join(sorted(param_policy.modifiers)) or "none"
            result.add(
                Severity.ERROR,
                "modifier-not-allowed",
                (
                    f"Modifier ':{param.modifier}' is not valid for "
                    f"{resource_type.value}.{param.name} (allowed: {allowed})."
                ),
                location,
            )

        if param.system is not None and not param_policy.allows_system(param.system):
            allowed = ", ".join(sorted(param_policy.systems)) or "none"
            result.add(
                Severity.ERROR,
                "system-not-allowed",
                (
                    f"Code system {param.system!r} is not permitted on "
                    f"{resource_type.value}.{param.name} (allowed: {allowed})."
                ),
                location,
            )

        if len(param.values) > param_policy.max_values:
            result.add(
                Severity.ERROR,
                "too-many-values",
                (
                    f"{param.name} has {len(param.values)} values; "
                    f"the limit is {param_policy.max_values}."
                ),
                location,
            )

        pattern = param_policy.pattern()
        for value in param.values:
            if not pattern.fullmatch(value):
                result.add(
                    Severity.ERROR,
                    "invalid-param-value",
                    (
                        f"Value {value!r} is not a valid {param_policy.type.value} for "
                        f"{resource_type.value}.{param.name}."
                    ),
                    location,
                )
                continue
            result = result.merge(self._check_terminology(param, param_policy, value, location))

        if param_policy.type is ParamType.QUANTITY and param.unit is None and param.values:
            result.add(
                Severity.WARNING,
                "quantity-without-unit",
                (
                    f"{param.name} compares a bare number; without a unit the server may "
                    "match values recorded in a different unit."
                ),
                location,
            )

        return result

    def _check_terminology(
        self, param: SearchParam, param_policy: ParamPolicy, value: str, location: str
    ) -> ValidationResult:
        """Flag codes that do not exist in the terminology subset.

        This is the main hallucination detector for Mode 1: a model that invents a
        plausible-looking LOINC code produces a query that is syntactically perfect and
        semantically empty.
        """
        result = ValidationResult()
        if param_policy.type is not ParamType.TOKEN:
            return result
        if not param_policy.systems & CLINICAL_CODE_SYSTEMS:
            return result  # status-style token, already constrained by its value pattern
        if param.name in {"_id", "identifier"}:
            return result

        if resolve_coding(param.system, value) is None:
            severity = Severity.ERROR if self.strict_codes else Severity.WARNING
            result.add(
                severity,
                "unknown-code",
                (
                    f"Code {value!r}"
                    + (f" in system {param.system!r}" if param.system else "")
                    + " is not present in the configured terminology; it may be hallucinated "
                    "or simply outside the loaded subset."
                ),
                location,
            )
        return result

    def _validate_controls(
        self, step: QueryStep, policy: ResourcePolicy, location: str
    ) -> ValidationResult:
        result = ValidationResult()

        for value in step.include:
            if value not in policy.includes:
                result.add(
                    Severity.ERROR,
                    "include-not-allowed",
                    f"_include={value!r} is not permitted on {step.resource_type.value}.",
                    f"{location}.include",
                )
        for value in step.revinclude:
            if value not in policy.revincludes:
                result.add(
                    Severity.ERROR,
                    "revinclude-not-allowed",
                    f"_revinclude={value!r} is not permitted on {step.resource_type.value}.",
                    f"{location}.revinclude",
                )
        if step.sort is not None and step.sort not in policy.sorts:
            result.add(
                Severity.ERROR,
                "sort-not-allowed",
                f"_sort={step.sort!r} is not permitted on {step.resource_type.value}.",
                f"{location}.sort",
            )
        if step.count is not None:
            limit = min(policy.max_count, self.max_page_size)
            if step.count > limit:
                result.add(
                    Severity.ERROR,
                    "count-too-large",
                    f"_count={step.count} exceeds the limit of {limit}.",
                    f"{location}.count",
                )
        return result

    def _validate_selectivity(
        self, step: QueryStep, policy: ResourcePolicy, param_names: set[str], location: str
    ) -> ValidationResult:
        """Refuse searches that would stream an entire resource type."""
        result = ValidationResult()
        if step.resource_type is ResourceType.PATIENT:
            return result  # cohort entry point; bounded by _count and pagination caps
        bare_names = {name.split(":", 1)[0] for name in param_names}
        if not (bare_names & _SELECTIVE_PARAMS) and step.depends_on is None:
            result.add(
                Severity.ERROR,
                "unselective-query",
                (
                    f"{step.resource_type.value} search has no selective filter. Add one of "
                    f"{', '.join(sorted(_SELECTIVE_PARAMS))}, or make the step depend on an "
                    "earlier patient-producing step."
                ),
                location,
            )
        _ = policy
        return result


def format_issues(result: ValidationResult, limit: int = 10) -> str:
    """Human-readable issue list, used as repair feedback to the planner."""
    return "\n".join(f"- {issue}" for issue in result.issues[:limit])
