"""Deterministic translation of a validated plan step into executable FHIR searches.

The builder is the only component allowed to produce a URL. It is pure and
side-effect free: same step in, same query out. It re-validates its input even though
the orchestrator already did, because "the caller validated it" is exactly the
assumption that turns a refactor into a security incident.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from fhir_healthcare_ai.domain.query import FHIRQuery, QueryPlan, QueryStep, ValidationResult
from fhir_healthcare_ai.fhir.allowlist import get_policy
from fhir_healthcare_ai.fhir.validator import QueryValidator, format_issues
from fhir_healthcare_ai.logging_config import get_logger

logger = get_logger(__name__)

PATIENT_CHUNK_SIZE = 50


class QueryBuildError(ValueError):
    """Raised when a step cannot be turned into a permitted query."""

    def __init__(self, message: str, validation: ValidationResult | None = None) -> None:
        super().__init__(message)
        self.validation = validation or ValidationResult()


class FHIRQueryBuilder:
    """Turns :class:`QueryStep` objects into :class:`FHIRQuery` objects."""

    def __init__(
        self,
        validator: QueryValidator | None = None,
        *,
        default_count: int = 100,
        max_page_size: int = 200,
    ) -> None:
        self.validator = validator or QueryValidator(max_page_size=max_page_size)
        self.default_count = min(default_count, max_page_size)
        self.max_page_size = max_page_size

    def build_step(
        self, step: QueryStep, patient_ids: Iterable[str] | None = None
    ) -> list[FHIRQuery]:
        """Build the queries for one step.

        A step scoped to many patients is split into several queries, because a
        ``patient=`` list long enough to matter will otherwise blow past server URL
        limits and silently truncate the cohort.
        """
        validation = self.validator.validate_step(step)
        if not validation.ok:
            raise QueryBuildError(
                f"step {step.step_id!r} failed validation:\n{format_issues(validation)}",
                validation,
            )

        base_params = self._base_params(step)
        scoped_ids = sorted(set(patient_ids)) if patient_ids is not None else None

        if not scoped_ids or not step.is_patient_scoped:
            return [
                FHIRQuery(
                    resource_type=step.resource_type,
                    params=tuple(base_params),
                    step_id=step.step_id,
                )
            ]

        queries: list[FHIRQuery] = []
        explicit = {name for name, _ in base_params}
        scope_param = "patient" if "patient" not in explicit else "_id"
        if scope_param == "_id":
            # The step already pins `patient`; re-scoping would silently widen it.
            scope_param = "patient"
            base_params = [(n, v) for n, v in base_params if n != "patient"]

        for chunk in _chunk(scoped_ids, PATIENT_CHUNK_SIZE):
            params = [*base_params, (scope_param, ",".join(chunk))]
            queries.append(
                FHIRQuery(
                    resource_type=step.resource_type,
                    params=tuple(params),
                    step_id=step.step_id,
                )
            )
        return queries

    def build_plan(self, plan: QueryPlan) -> list[FHIRQuery]:
        """Build every independent step of a plan.

        Steps with ``depends_on`` are skipped: their patient scope is only known once
        the parent step has executed, so the orchestrator builds them later.
        """
        validation = self.validator.validate_plan(plan)
        if not validation.ok:
            raise QueryBuildError(
                f"plan failed validation:\n{format_issues(validation)}", validation
            )
        queries: list[FHIRQuery] = []
        for step in plan.steps:
            if step.depends_on is None:
                queries.extend(self.build_step(step))
        return queries

    def _base_params(self, step: QueryStep) -> list[tuple[str, str]]:
        policy = get_policy(step.resource_type)
        if policy is None:  # pragma: no cover - validate_step already refused this
            raise QueryBuildError(f"resource {step.resource_type.value} is not reachable")

        params: list[tuple[str, str]] = [param.as_pair() for param in step.params]

        for include in step.include:
            params.append(("_include", include))
        for revinclude in step.revinclude:
            params.append(("_revinclude", revinclude))
        if step.sort:
            params.append(("_sort", step.sort))

        count = step.count or self.default_count
        params.append(("_count", str(min(count, policy.max_count, self.max_page_size))))
        return params


def _chunk(items: Sequence[str], size: int) -> list[list[str]]:
    return [list(items[i : i + size]) for i in range(0, len(items), size)]
