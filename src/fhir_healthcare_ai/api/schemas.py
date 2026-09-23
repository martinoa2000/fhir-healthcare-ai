"""Request and response bodies that are specific to the HTTP layer.

Pipeline results (:class:`~fhir_healthcare_ai.domain.results.QueryResponse`,
:class:`~fhir_healthcare_ai.domain.results.PatientAnalysis`) are returned as they are;
only what the API itself adds is defined here.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

#: FHIR logical id syntax. Enforced at the edge so a malformed id never reaches a query.
PATIENT_ID_PATTERN = r"^[A-Za-z0-9\-.]{1,64}$"


class QueryRequest(BaseModel):
    """A natural-language clinical question."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        min_length=3,
        max_length=1000,
        examples=["Which patients with elevated HbA1c had a recent medication change?"],
    )
    as_of: date | None = Field(
        default=None,
        description="Reference date for relative windows ('recent', 'last year'). "
        "Fix it to make an answer reproducible; defaults to today (UTC).",
    )
    context: str | None = Field(
        default=None,
        max_length=500,
        description="Extra grounding for the planner, e.g. 'patient id: syn42-pat-0001'.",
    )
    narrate: bool = Field(
        default=True, description="Ask the model for a short prose summary of the findings."
    )


class FHIRStatus(BaseModel):
    base_url: str
    reachable: bool
    in_memory: bool


class LLMStatus(BaseModel):
    configured: str
    active: str
    model: str
    local: bool = Field(description="Inference stays inside the deployment boundary.")
    fallback: bool = Field(description="The configured backend is down; mock is answering.")
    reason: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    version: str
    environment: str
    fhir: FHIRStatus
    llm: LLMStatus


class FeatureDescription(BaseModel):
    name: str
    dtype: str
    description: str
    group: str


class ConceptDescription(BaseModel):
    key: str
    display: str
    unit: str | None = None
    drug_class: str | None = None


class Limits(BaseModel):
    max_patients_per_response: int
    max_page_size: int
    max_pages: int
    max_total_resources: int


class CapabilitiesResponse(BaseModel):
    """What this deployment can be asked, and within which bounds."""

    resources: dict[str, dict[str, Any]]
    concepts: dict[str, list[ConceptDescription]]
    features: list[FeatureDescription]
    example_questions: list[str]
    limits: Limits
    llm: LLMStatus


class ErrorResponse(BaseModel):
    detail: str
    correlation_id: str | None = None
