"""Result models returned by the pipeline.

Every clinical claim the system makes must be traceable to a FHIR resource. That is
what :class:`Evidence` is for: it is not a nicety, it is the reason a reviewer can
audit an answer instead of trusting it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from fhir_healthcare_ai.domain.enums import AbnormalFlag, AnalysisType, ResourceType, RiskBand
from fhir_healthcare_ai.domain.query import FHIRQuery, QueryPlan, ValidationIssue

DISCLAIMER = (
    "Research and engineering demonstration only. Outputs are generated from synthetic "
    "data by automated tooling, are not validated for clinical use, and must not be used "
    "to make or support decisions about the care of any person."
)


class Evidence(BaseModel):
    """A pointer to the exact resource that supports a statement."""

    model_config = ConfigDict(frozen=True)

    patient_id: str
    resource_type: ResourceType
    resource_id: str
    concept: str | None = None
    display: str | None = None
    value: str | None = None
    effective: datetime | None = None
    source_step: str | None = None
    note: str | None = None

    @property
    def reference(self) -> str:
        return f"{self.resource_type.value}/{self.resource_id}"


class AbnormalLabFinding(BaseModel):
    """One out-of-range laboratory result."""

    model_config = ConfigDict(frozen=True)

    concept: str
    display: str | None = None
    value: float | None = None
    unit: str | None = None
    flag: AbnormalFlag
    reference_low: float | None = None
    reference_high: float | None = None
    effective: datetime | None = None
    evidence: Evidence


class RiskAssessment(BaseModel):
    """Model output for one patient. Decision support only."""

    model_config = ConfigDict(frozen=True)

    score: float = Field(ge=0.0, le=1.0)
    band: RiskBand
    model_name: str
    model_version: str
    contributing_factors: list[str] = Field(default_factory=list)
    missing_features: list[str] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER


class PatientAnalysis(BaseModel):
    """Per-patient analytics output."""

    patient_id: str
    age_years: float | None = None
    gender: str | None = None
    features: dict[str, Any] = Field(default_factory=dict)
    abnormal_labs: list[AbnormalLabFinding] = Field(default_factory=list)
    risk: RiskAssessment | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)


class PatientMatch(BaseModel):
    """A patient returned by a cohort search."""

    patient_id: str
    matched_steps: list[str] = Field(default_factory=list)
    age_years: float | None = None
    gender: str | None = None
    summary: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)


class CohortSummary(BaseModel):
    """Aggregate view over the matched cohort."""

    total_patients: int = 0
    total_resources: int = 0
    per_step_counts: dict[str, int] = Field(default_factory=dict)
    statistics: dict[str, Any] = Field(default_factory=dict)


class ExecutionTrace(BaseModel):
    """What actually happened, for debugging and for the audit trail."""

    correlation_id: str
    stages: dict[str, float] = Field(default_factory=dict, description="stage -> milliseconds")
    fhir_requests: int = 0
    resources_fetched: int = 0
    llm_calls: int = 0
    llm_tokens: int = 0
    truncated: bool = False


class QueryResponse(BaseModel):
    """Top-level response to a natural-language clinical question."""

    question: str
    query_plan: QueryPlan
    fhir_queries: list[str] = Field(default_factory=list)
    patients: list[PatientMatch] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    cohort: CohortSummary = Field(default_factory=CohortSummary)
    analysis_type: AnalysisType = AnalysisType.NONE
    analyses: list[PatientAnalysis] = Field(default_factory=list)
    narrative: str | None = None
    validation_issues: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    trace: ExecutionTrace | None = None
    disclaimer: str = DISCLAIMER


class ExecutedQuery(BaseModel):
    """Internal record linking a plan step to the query that ran and what came back."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    step_id: str
    query: FHIRQuery
    resource_count: int = 0
    patient_ids: set[str] = Field(default_factory=set)
    truncated: bool = False
    error: str | None = None
