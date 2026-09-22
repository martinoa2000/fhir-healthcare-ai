"""Normalized clinical models.

These sit between raw FHIR JSON and the analytics layer. A normalized record is
flat, unit-canonicalized and terminology-resolved, so downstream code never has to
know that ``Observation.valueQuantity`` and ``Observation.component[].valueQuantity``
exist, or that the same lab can arrive as LOINC ``4548-4`` or ``17856-6``.

Everything is optional by design: real FHIR servers omit fields constantly, and a
parser that raises on a missing ``effectiveDateTime`` is useless in production.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field

from fhir_healthcare_ai.domain.enums import AbnormalFlag, DatePrecision, ResourceType


class Coding(BaseModel):
    """A single terminology coding."""

    model_config = ConfigDict(frozen=True)

    system: str | None = None
    code: str | None = None
    display: str | None = None

    @property
    def token(self) -> str:
        """Render as a FHIR token (``system|code``)."""
        if self.system and self.code:
            return f"{self.system}|{self.code}"
        return self.code or ""

    def __str__(self) -> str:
        return self.display or self.token or "<unknown>"


class TemporalValue(BaseModel):
    """A FHIR date/dateTime with the precision that was actually supplied."""

    model_config = ConfigDict(frozen=True)

    value: datetime | None = None
    precision: DatePrecision | None = None
    raw: str | None = None


class Quantity(BaseModel):
    """A measured value with both its source and canonical (UCUM-normalized) form."""

    model_config = ConfigDict(frozen=True)

    value: float | None = None
    unit: str | None = None
    system: str | None = None
    code: str | None = None
    comparator: str | None = None  # "<", ">", "<=", ">=" from FHIR Quantity.comparator
    canonical_value: float | None = None
    canonical_unit: str | None = None

    @property
    def is_estimate(self) -> bool:
        """True when the source only bounded the value (e.g. ``<0.1``)."""
        return self.comparator is not None


class ReferenceRange(BaseModel):
    """Observation reference range, when the lab supplied one."""

    model_config = ConfigDict(frozen=True)

    low: float | None = None
    high: float | None = None
    unit: str | None = None
    text: str | None = None


class NormalizedResource(BaseModel):
    """Fields every normalized resource carries."""

    resource_type: ResourceType
    id: str
    patient_id: str | None = None
    encounter_id: str | None = None
    source_codings: list[Coding] = Field(default_factory=list)
    concept: str | None = Field(
        default=None,
        description="Canonical internal concept key (e.g. 'hba1c', 'type_2_diabetes').",
    )
    display: str | None = None
    parse_warnings: list[str] = Field(default_factory=list)

    @property
    def reference(self) -> str:
        """FHIR relative reference, e.g. ``Observation/obs-1``."""
        return f"{self.resource_type.value}/{self.id}"


class NormalizedPatient(NormalizedResource):
    """Demographics."""

    resource_type: ResourceType = ResourceType.PATIENT
    gender: str | None = None
    birth_date: date | None = None
    age_years: float | None = None
    deceased: bool = False
    deceased_date: datetime | None = None
    postal_code: str | None = None
    managing_organization: str | None = None


class NormalizedObservation(NormalizedResource):
    """A lab result, vital sign or other observation."""

    resource_type: ResourceType = ResourceType.OBSERVATION
    status: str | None = None
    category: list[Coding] = Field(default_factory=list)
    effective: TemporalValue | None = None
    issued: datetime | None = None
    quantity: Quantity | None = None
    value_string: str | None = None
    value_boolean: bool | None = None
    value_codeable: Coding | None = None
    interpretation: list[Coding] = Field(default_factory=list)
    reference_range: ReferenceRange | None = None
    abnormal_flag: AbnormalFlag = AbnormalFlag.UNKNOWN
    components: list[NormalizedObservation] = Field(default_factory=list)
    derived_from: list[str] = Field(default_factory=list)

    @property
    def numeric_value(self) -> float | None:
        """Canonical numeric value if one is available."""
        if self.quantity is None:
            return None
        return (
            self.quantity.canonical_value
            if self.quantity.canonical_value is not None
            else self.quantity.value
        )

    @property
    def effective_datetime(self) -> datetime | None:
        return self.effective.value if self.effective else None


class NormalizedCondition(NormalizedResource):
    """A problem-list or encounter-diagnosis condition."""

    resource_type: ResourceType = ResourceType.CONDITION
    clinical_status: str | None = None
    verification_status: str | None = None
    category: list[Coding] = Field(default_factory=list)
    severity: Coding | None = None
    onset: TemporalValue | None = None
    abatement: TemporalValue | None = None
    recorded_date: datetime | None = None

    @property
    def is_active(self) -> bool:
        return (self.clinical_status or "").lower() in {"active", "recurrence", "relapse"}


class NormalizedMedicationRequest(NormalizedResource):
    """A medication order. Medication may be inline or a reference."""

    resource_type: ResourceType = ResourceType.MEDICATION_REQUEST
    status: str | None = None
    intent: str | None = None
    authored_on: datetime | None = None
    medication_reference: str | None = None
    drug_class: str | None = None
    dosage_text: str | None = None
    dose_quantity: Quantity | None = None
    validity_start: datetime | None = None
    validity_end: datetime | None = None
    reason_codings: list[Coding] = Field(default_factory=list)

    @property
    def is_active(self) -> bool:
        return (self.status or "").lower() in {"active", "on-hold"}


class NormalizedEncounter(NormalizedResource):
    """A contact between patient and provider."""

    resource_type: ResourceType = ResourceType.ENCOUNTER
    status: str | None = None
    encounter_class: Coding | None = None
    type_codings: list[Coding] = Field(default_factory=list)
    period_start: datetime | None = None
    period_end: datetime | None = None
    length_days: float | None = None
    reason_codings: list[Coding] = Field(default_factory=list)
    service_provider: str | None = None


class NormalizedDiagnosticReport(NormalizedResource):
    """A grouped report (lab panel, imaging study) pointing at member Observations."""

    resource_type: ResourceType = ResourceType.DIAGNOSTIC_REPORT
    status: str | None = None
    category: list[Coding] = Field(default_factory=list)
    effective: TemporalValue | None = None
    issued: datetime | None = None
    result_references: list[str] = Field(default_factory=list)
    conclusion: str | None = None
    conclusion_codings: list[Coding] = Field(default_factory=list)


class PatientRecord(BaseModel):
    """All normalized resources retrieved for one patient.

    This is the single input to the feature layer. Assembling it is the job of the
    normalization layer, not of the analytics code.
    """

    patient_id: str
    patient: NormalizedPatient | None = None
    observations: list[NormalizedObservation] = Field(default_factory=list)
    conditions: list[NormalizedCondition] = Field(default_factory=list)
    medication_requests: list[NormalizedMedicationRequest] = Field(default_factory=list)
    encounters: list[NormalizedEncounter] = Field(default_factory=list)
    diagnostic_reports: list[NormalizedDiagnosticReport] = Field(default_factory=list)

    def observations_for(self, concept: str) -> list[NormalizedObservation]:
        """Observations matching a canonical concept, oldest first."""
        matches = [o for o in self.observations if o.concept == concept]
        return sorted(matches, key=lambda o: o.effective_datetime or datetime.min)

    def latest_observation(self, concept: str) -> NormalizedObservation | None:
        """Most recent observation for a concept, or None."""
        matches = self.observations_for(concept)
        return matches[-1] if matches else None

    @property
    def resource_count(self) -> int:
        return (
            len(self.observations)
            + len(self.conditions)
            + len(self.medication_requests)
            + len(self.encounters)
            + len(self.diagnostic_reports)
            + (1 if self.patient else 0)
        )


NormalizedObservation.model_rebuild()
