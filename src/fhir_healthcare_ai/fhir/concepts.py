"""Expands concept keys in a query plan into real terminology codes.

The planner is never asked to produce a LOINC or RxNorm code. It writes ``hba1c``, and
this module turns that into ``http://loinc.org|4548-4,http://loinc.org|17856-6`` before
the plan reaches the validator.

That indirection buys three things:

**Hallucinated codes become impossible on the happy path.** A model cannot misremember a
code it was never asked for. The validator's ``unknown-code`` check still runs, so a
model that ignores the instruction and emits a raw code is still caught -- but the
common case no longer depends on the model's recall of a coding system.

**One concept, every synonym.** HbA1c has several LOINC codes in real data, and a
condition may be recorded with SNOMED CT at one site and ICD-10-CM at another. Expansion
emits every code the concept has in the requested system, so a query written once
matches data coded either way.

**Codes change in one place.** Adding a code to a concept in
:mod:`fhir_healthcare_ai.terminology` immediately widens every plan that references it.

Expansion runs *before* validation, never after. Validating the pre-expansion plan would
check strings the server never sees.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fhir_healthcare_ai.domain.query import QueryPlan, QueryStep, SearchParam
from fhir_healthcare_ai.fhir.allowlist import CLINICAL_CODE_SYSTEMS, ParamType, get_policy
from fhir_healthcare_ai.logging_config import get_logger
from fhir_healthcare_ai.terminology import get_concept, resolve_coding

logger = get_logger(__name__)


@dataclass
class ExpansionReport:
    """What expansion did, so the response can explain the codes it searched for.

    Args:
        expanded: concept key -> the codes it became.
        unresolved: Values that looked like concept keys but matched nothing. Left
            untouched in the plan so the validator can rule on them.
        passthrough: Values that were already valid codes in the terminology.
    """

    expanded: dict[str, list[str]] = field(default_factory=dict)
    unresolved: list[str] = field(default_factory=list)
    passthrough: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.expanded)

    def assumption_lines(self) -> list[str]:
        """Human-readable notes for the response's ``assumptions`` list."""
        lines = []
        for key, codes in sorted(self.expanded.items()):
            concept = get_concept(key)
            label = concept.display if concept else key
            lines.append(f"{label!r} was searched as {len(codes)} code(s): {', '.join(codes)}")
        return lines


class ConceptExpander:
    """Rewrites concept keys into codes within a plan."""

    def __init__(self, *, max_codes_per_param: int = 20) -> None:
        #: Matches the allowlist's own per-parameter value cap, so expansion cannot push
        #: a legal plan over the limit and turn a valid question into a rejection.
        self.max_codes_per_param = max_codes_per_param

    def expand_plan(self, plan: QueryPlan) -> tuple[QueryPlan, ExpansionReport]:
        """Return a copy of the plan with concept keys resolved."""
        report = ExpansionReport()
        steps = [self._expand_step(step, report) for step in plan.steps]
        expanded = plan.model_copy(update={"steps": steps})
        if report.unresolved:
            logger.warning(
                "plan contained values that resolved to no known concept or code",
                extra={"values": sorted(set(report.unresolved))},
            )
        return expanded, report

    def _expand_step(self, step: QueryStep, report: ExpansionReport) -> QueryStep:
        policy = get_policy(step.resource_type)
        if policy is None:
            return step  # not allowed at all; the validator will say so

        params = []
        for param in step.params:
            param_policy = policy.param(param.name)
            if param_policy is None or param_policy.type is not ParamType.TOKEN:
                params.append(param)
                continue
            if not param_policy.systems & CLINICAL_CODE_SYSTEMS:
                params.append(param)  # status-style token, not a clinical code
                continue
            params.append(self._expand_param(param, report))
        return step.model_copy(update={"params": params})

    def _expand_param(self, param: SearchParam, report: ExpansionReport) -> SearchParam:
        codes: list[str] = []
        systems: set[str] = set()

        for value in param.values:
            concept = get_concept(value)
            if concept is not None:
                resolved = self._codes_for(concept_key=value, system=param.system)
                if resolved:
                    report.expanded.setdefault(value, [])
                    for system, code in resolved:
                        if code not in codes:
                            codes.append(code)
                            report.expanded[value].append(code)
                            systems.add(system)
                    continue
                # The concept exists but has no coding in the requested system. Falling
                # back to its own systems is better than searching for a key as a code.
                report.unresolved.append(f"{value} (no coding in {param.system})")
                continue

            if resolve_coding(param.system, value) is not None:
                report.passthrough.append(value)
            else:
                report.unresolved.append(value)
            if value not in codes:
                codes.append(value)

        if codes == list(param.values):
            return param

        if len(codes) > self.max_codes_per_param:
            logger.warning(
                "truncating expanded code list",
                extra={"param": param.name, "count": len(codes)},
            )
            codes = codes[: self.max_codes_per_param]

        update: dict[str, object] = {"values": codes}
        # A multi-system expansion cannot be rendered as one `system|code` prefix, so the
        # system is folded into each value instead and the param-level system is dropped.
        if len(systems) > 1:
            update["values"] = [f"{system}|{code}" for system, code in self._pairs(param, codes)]
            update["system"] = None
        elif systems:
            update["system"] = next(iter(systems))
        return param.model_copy(update=update)

    def _codes_for(self, concept_key: str, system: str | None) -> list[tuple[str, str]]:
        concept = get_concept(concept_key)
        if concept is None:
            return []
        codings = [
            coding
            for coding in concept.codings
            if coding.code and (system is None or coding.system == system)
        ]
        return [(coding.system or "", coding.code or "") for coding in codings if coding.code]

    def _pairs(self, param: SearchParam, codes: list[str]) -> list[tuple[str, str]]:
        """Re-attach each code to the system it came from, for mixed-system expansions."""
        pairs: list[tuple[str, str]] = []
        for value in param.values:
            concept = get_concept(value)
            if concept is None:
                continue
            for coding in concept.codings:
                if coding.code in codes and coding.system:
                    pairs.append((coding.system, coding.code))
        return pairs


def expand_plan(plan: QueryPlan) -> tuple[QueryPlan, ExpansionReport]:
    """Convenience wrapper for the default expander."""
    return ConceptExpander().expand_plan(plan)
