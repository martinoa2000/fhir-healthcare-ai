"""Query plan and FHIR query models.

The query plan is the *only* thing the LLM is allowed to produce. It is a declarative
description of what to fetch, never a URL and never raw HTTP. Turning a plan into an
actual request is the job of the deterministic query builder, and every plan passes
through the validator before a single byte leaves the process.
"""

from __future__ import annotations

from typing import Annotated, Any
from urllib.parse import quote, urlencode

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from fhir_healthcare_ai.domain.enums import (
    AnalysisType,
    CohortLogic,
    Comparator,
    QueryIntent,
    ResourceType,
    Severity,
    StepRole,
)

ParamName = Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_\-]{0,63}$")]


class SearchParam(BaseModel):
    """One FHIR search parameter.

    Multiple ``values`` are OR-ed by the FHIR server (comma-separated). A ``system``
    turns the values into proper tokens so ``code=4548-4`` cannot silently match a
    same-numbered code in a different code system.
    """

    model_config = ConfigDict(extra="forbid")

    name: ParamName
    values: list[str] = Field(min_length=1)
    comparator: Comparator | None = None
    modifier: str | None = Field(default=None, pattern=r"^[a-zA-Z\-]{1,32}$")
    system: str | None = None
    unit: str | None = None
    unit_system: str | None = None

    @field_validator("values", mode="before")
    @classmethod
    def _coerce_scalar(cls, v: Any) -> Any:
        """Accept ``value="x"`` style scalars from LLM output."""
        if isinstance(v, str | int | float):
            return [str(v)]
        if isinstance(v, list):
            return [str(item) for item in v]
        return v

    @model_validator(mode="before")
    @classmethod
    def _accept_value_alias(cls, data: Any) -> Any:
        if isinstance(data, dict) and "values" not in data and "value" in data:
            data = {**data, "values": data.pop("value")}
        return data

    @property
    def key(self) -> str:
        """Parameter name including any modifier."""
        return f"{self.name}:{self.modifier}" if self.modifier else self.name

    def render_value(self) -> str:
        """Render the right-hand side of the parameter.

        A comparator prefix is only meaningful on ordered parameters, so it is dropped
        for token searches rather than emitted as an invalid ``gt`` inside a code.
        """
        prefix = self.comparator.value if self.comparator else ""
        rendered = []
        for raw in self.values:
            if self.system:
                rendered.append(f"{self.system}|{raw}")
            elif self.unit is not None:
                rendered.append(f"{prefix}{raw}|{self.unit_system or ''}|{self.unit}")
            else:
                rendered.append(f"{prefix}{raw}")
        return ",".join(rendered)

    def as_pair(self) -> tuple[str, str]:
        return self.key, self.render_value()


class QueryStep(BaseModel):
    """A single retrieval step in a plan."""

    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(pattern=r"^[a-z0-9_\-]{1,40}$")
    resource_type: ResourceType
    purpose: str = Field(default="", max_length=500)
    params: list[SearchParam] = Field(default_factory=list)
    include: list[str] = Field(default_factory=list)
    revinclude: list[str] = Field(default_factory=list)
    sort: str | None = None
    count: int | None = Field(default=None, ge=1, le=1000)
    depends_on: str | None = Field(
        default=None,
        description="step_id whose resulting patient set scopes this step via `patient=`.",
    )
    role: StepRole = Field(
        default=StepRole.FILTER,
        description="filter narrows the cohort, context never does, exclude removes patients.",
    )

    @property
    def is_patient_scoped(self) -> bool:
        return self.resource_type is not ResourceType.PATIENT


class AnalysisRequest(BaseModel):
    """Mode 2 analytics to run over the retrieved cohort."""

    model_config = ConfigDict(extra="forbid")

    type: AnalysisType = AnalysisType.NONE
    concepts: list[str] = Field(default_factory=list)
    lookback_days: int | None = Field(default=None, ge=1, le=3650)
    options: dict[str, Any] = Field(default_factory=dict)


class QueryPlan(BaseModel):
    """Structured interpretation of a natural-language clinical question."""

    model_config = ConfigDict(extra="forbid")

    question: str
    intent: QueryIntent = QueryIntent.COHORT_SEARCH
    rationale: str = Field(default="", max_length=2000)
    steps: list[QueryStep] = Field(default_factory=list)
    cohort_logic: CohortLogic = CohortLogic.ALL
    analysis: AnalysisRequest = Field(default_factory=AnalysisRequest)
    assumptions: list[str] = Field(default_factory=list)
    unsupported: bool = Field(
        default=False,
        description="Set when the question cannot be answered with the allowed resources.",
    )
    unsupported_reason: str | None = None

    @model_validator(mode="after")
    def _check_step_ids(self) -> QueryPlan:
        seen: set[str] = set()
        for step in self.steps:
            if step.step_id in seen:
                raise ValueError(f"duplicate step_id: {step.step_id}")
            seen.add(step.step_id)
        for step in self.steps:
            if step.depends_on and step.depends_on not in seen:
                raise ValueError(f"step {step.step_id} depends on unknown step {step.depends_on}")
            if step.depends_on == step.step_id:
                raise ValueError(f"step {step.step_id} depends on itself")
        return self


class FHIRQuery(BaseModel):
    """A concrete, validated, executable FHIR search.

    Produced only by :class:`~fhir_healthcare_ai.fhir.query_builder.FHIRQueryBuilder`.
    """

    model_config = ConfigDict(frozen=True)

    resource_type: ResourceType
    params: tuple[tuple[str, str], ...] = ()
    step_id: str | None = None

    def relative_url(self) -> str:
        """``Observation?code=...&date=ge2025-01-01``"""
        if not self.params:
            return self.resource_type.value
        # ``:`` and ``/`` are legal unencoded in a query value (RFC 3986 pchar), and leaving
        # them alone keeps system URIs readable: ``code=http://loinc.org|4548-4`` rather than
        # ``code=http%3A%2F%2Floinc.org|4548-4``. Every token value is already allowlisted.
        query = urlencode(self.params, quote_via=quote, safe="|,$<>=:/")
        return f"{self.resource_type.value}?{query}"

    def full_url(self, base_url: str) -> str:
        return f"{base_url.rstrip('/')}/{self.relative_url()}"

    def __str__(self) -> str:
        return self.relative_url()


class ValidationIssue(BaseModel):
    """A single validator finding, shaped like a FHIR OperationOutcome issue."""

    model_config = ConfigDict(frozen=True)

    severity: Severity
    code: str
    message: str
    location: str | None = None

    def __str__(self) -> str:
        where = f" at {self.location}" if self.location else ""
        return f"[{self.severity.value}] {self.code}{where}: {self.message}"


class ValidationResult(BaseModel):
    """Outcome of validating a plan or a query."""

    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when nothing at ERROR or FATAL severity was found."""
        return not any(i.severity in (Severity.ERROR, Severity.FATAL) for i in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity in (Severity.ERROR, Severity.FATAL)]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity is Severity.WARNING]

    def add(self, severity: Severity, code: str, message: str, location: str | None = None) -> None:
        self.issues.append(
            ValidationIssue(severity=severity, code=code, message=message, location=location)
        )

    def merge(self, other: ValidationResult) -> ValidationResult:
        return ValidationResult(issues=[*self.issues, *other.issues])
