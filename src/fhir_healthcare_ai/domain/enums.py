"""Enumerations shared across the pipeline."""

from __future__ import annotations

from enum import StrEnum


class ResourceType(StrEnum):
    """FHIR R4 resource types this system is allowed to touch.

    Adding a member here is not enough to make a resource reachable: it must also be
    registered in :mod:`fhir_healthcare_ai.fhir.allowlist` with its permitted search
    parameters, and it needs a parser in :mod:`fhir_healthcare_ai.fhir.parsers`.
    """

    PATIENT = "Patient"
    OBSERVATION = "Observation"
    CONDITION = "Condition"
    MEDICATION_REQUEST = "MedicationRequest"
    ENCOUNTER = "Encounter"
    DIAGNOSTIC_REPORT = "DiagnosticReport"
    # Interfaces exist for these; enable them in the allowlist once parsers land.
    PROCEDURE = "Procedure"
    ALLERGY_INTOLERANCE = "AllergyIntolerance"
    CARE_PLAN = "CarePlan"
    IMMUNIZATION = "Immunization"


class Comparator(StrEnum):
    """FHIR search prefixes for ordered (date / quantity / number) parameters."""

    EQ = "eq"
    NE = "ne"
    GT = "gt"
    LT = "lt"
    GE = "ge"
    LE = "le"
    SA = "sa"  # starts after
    EB = "eb"  # ends before
    AP = "ap"  # approximately


class QueryIntent(StrEnum):
    """What the user is asking for, which decides how results are assembled."""

    COHORT_SEARCH = "cohort_search"
    PATIENT_SUMMARY = "patient_summary"
    TREND = "trend"
    COUNT = "count"


class CohortLogic(StrEnum):
    """How the patient sets returned by individual query steps are combined."""

    ALL = "all"  # intersection - patient must satisfy every step
    ANY = "any"  # union - patient must satisfy at least one step


class AnalysisType(StrEnum):
    """Mode 2 analytics tasks."""

    NONE = "none"
    ABNORMAL_LABS = "abnormal_labs"
    RISK_STRATIFICATION = "risk_stratification"
    COHORT_IDENTIFICATION = "cohort_identification"


class RiskBand(StrEnum):
    """Coarse risk bucket. Decision support only - never a diagnosis."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class Severity(StrEnum):
    """Validation issue severity, mirroring FHIR OperationOutcome."""

    INFORMATION = "information"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"


class AbnormalFlag(StrEnum):
    """Direction of a laboratory abnormality."""

    NORMAL = "normal"
    LOW = "low"
    HIGH = "high"
    CRITICAL_LOW = "critical_low"
    CRITICAL_HIGH = "critical_high"
    UNKNOWN = "unknown"


class DatePrecision(StrEnum):
    """Precision recovered from a partial FHIR date/dateTime."""

    YEAR = "year"
    MONTH = "month"
    DAY = "day"
    SECOND = "second"
    MILLISECOND = "millisecond"
