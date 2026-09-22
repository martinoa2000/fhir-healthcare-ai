"""Abnormal laboratory detection.

The classification itself lives in the Observation parser (:func:`classify`), which
knows the precedence rules: an explicit ``interpretation`` from the lab wins, then the
``referenceRange`` the lab attached to the result, then our curated interval. This
module is the layer above: it walks a patient record, applies that classifier, and
turns each hit into an :class:`AbnormalLabFinding` carrying the :class:`Evidence` that
points back at the exact Observation.

Two deliberate choices:

*Sex-specific intervals are used when the patient's gender is known*, because
hemoglobin and creatinine intervals differ enough that ignoring sex would generate
false positives for half the cohort.

*Panel components are expanded.* A blood-pressure panel stores its systolic and
diastolic values as components, so the parser emits them as synthetic observations
and they are screened like any other result.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta

from fhir_healthcare_ai.domain.clinical import NormalizedObservation, PatientRecord
from fhir_healthcare_ai.domain.enums import AbnormalFlag
from fhir_healthcare_ai.domain.results import AbnormalLabFinding, Evidence
from fhir_healthcare_ai.fhir.parsers.observation import classify
from fhir_healthcare_ai.logging_config import get_logger
from fhir_healthcare_ai.terminology import ConceptDefinition, get_concept, resolve_codings

logger = get_logger(__name__)

DEFAULT_WINDOW_DAYS = 180

ABNORMAL_FLAGS: frozenset[AbnormalFlag] = frozenset(
    {
        AbnormalFlag.LOW,
        AbnormalFlag.HIGH,
        AbnormalFlag.CRITICAL_LOW,
        AbnormalFlag.CRITICAL_HIGH,
    }
)

CRITICAL_FLAGS: frozenset[AbnormalFlag] = frozenset(
    {AbnormalFlag.CRITICAL_LOW, AbnormalFlag.CRITICAL_HIGH}
)

# Ordering used when a caller asks for "the most important findings first".
_FLAG_PRIORITY: dict[AbnormalFlag, int] = {
    AbnormalFlag.CRITICAL_HIGH: 0,
    AbnormalFlag.CRITICAL_LOW: 0,
    AbnormalFlag.HIGH: 1,
    AbnormalFlag.LOW: 1,
    AbnormalFlag.UNKNOWN: 2,
    AbnormalFlag.NORMAL: 3,
}


class AbnormalLabDetector:
    """Screens a patient's observations against reference intervals."""

    def __init__(
        self,
        *,
        window_days: int = DEFAULT_WINDOW_DAYS,
        concepts: Sequence[str] | None = None,
        latest_only: bool = True,
        include_unknown: bool = False,
    ) -> None:
        """
        Args:
            window_days: Only results effective within this many days of ``as_of`` are
                screened. Old results are real data but are rarely actionable context.
            concepts: Restrict screening to these concept keys. ``None`` screens every
                observation that maps to a known concept.
            latest_only: Report one finding per concept (the most recent). Set to False
                to get the full abnormal history within the window.
            include_unknown: Emit findings for results that could not be classified.
                Off by default so that the output is not diluted by unmapped codes.
        """
        self.window_days = window_days
        self.concepts = frozenset(concepts) if concepts else None
        self.latest_only = latest_only
        self.include_unknown = include_unknown

    def detect(
        self, record: PatientRecord, as_of: datetime | None = None
    ) -> list[AbnormalLabFinding]:
        """Return the abnormal results for one patient, most severe first."""
        now = as_of or datetime.now(UTC)
        window_start = now - timedelta(days=self.window_days)
        sex = record.patient.gender if record.patient else None

        candidates = [
            obs
            for obs in record.observations
            if _in_window(obs, window_start, now) and _has_value(obs)
        ]
        candidates.sort(key=lambda o: o.effective_datetime or window_start)

        findings: list[AbnormalLabFinding] = []
        seen_concepts: set[str] = set()

        for obs in reversed(candidates):  # newest first, so latest_only keeps the latest
            concept = _concept_for(obs)
            concept_key = concept.key if concept else None
            if self.concepts is not None and concept_key not in self.concepts:
                continue

            flag = classify(obs, concept=concept, sex=sex)
            if flag not in ABNORMAL_FLAGS and not (
                self.include_unknown and flag is AbnormalFlag.UNKNOWN
            ):
                continue

            key = concept_key or f"{obs.resource_type.value}:{obs.id}"
            if self.latest_only and key in seen_concepts:
                continue
            seen_concepts.add(key)

            findings.append(_to_finding(record.patient_id, obs, concept, flag))

        findings.sort(
            key=lambda f: (_FLAG_PRIORITY.get(f.flag, 9), -(_epoch(f.effective))),
        )
        return findings

    def detect_many(
        self, records: Iterable[PatientRecord], as_of: datetime | None = None
    ) -> dict[str, list[AbnormalLabFinding]]:
        now = as_of or datetime.now(UTC)
        return {record.patient_id: self.detect(record, now) for record in records}


def summarize_findings(findings: Sequence[AbnormalLabFinding]) -> str:
    """One human-readable line, e.g. ``HbA1c 9.2 % (high), Potassium 6.1 mmol/L (crit)``."""
    if not findings:
        return "No abnormal results in the screening window."
    parts = []
    for finding in findings:
        label = finding.display or finding.concept
        value = "" if finding.value is None else f" {_trim(finding.value)}"
        unit = f" {finding.unit}" if finding.unit else ""
        parts.append(f"{label}{value}{unit} ({finding.flag.value.replace('_', ' ')})")
    return ", ".join(parts)


def detect_abnormal_labs(
    record: PatientRecord,
    as_of: datetime | None = None,
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> list[AbnormalLabFinding]:
    """Convenience wrapper for the common case."""
    return AbnormalLabDetector(window_days=window_days).detect(record, as_of)


# -- internals ------------------------------------------------------------------------


def _to_finding(
    patient_id: str,
    obs: NormalizedObservation,
    concept: ConceptDefinition | None,
    flag: AbnormalFlag,
) -> AbnormalLabFinding:
    low, high = _interval_bounds(obs, concept)
    value = obs.numeric_value
    unit = obs.quantity.canonical_unit or obs.quantity.unit if obs.quantity else None
    display = obs.display or (concept.display if concept else None)

    evidence = Evidence(
        patient_id=patient_id,
        resource_type=obs.resource_type,
        resource_id=obs.id,
        concept=concept.key if concept else None,
        display=display,
        value=_format_value(obs),
        effective=obs.effective_datetime,
        note=f"classified {flag.value}",
    )
    return AbnormalLabFinding(
        concept=concept.key if concept else (obs.concept or "unmapped"),
        display=display,
        value=value,
        unit=unit,
        flag=flag,
        reference_low=low,
        reference_high=high,
        effective=obs.effective_datetime,
        evidence=evidence,
    )


def _interval_bounds(
    obs: NormalizedObservation, concept: ConceptDefinition | None
) -> tuple[float | None, float | None]:
    """Prefer the range the lab reported; fall back to our curated interval."""
    if obs.reference_range and (
        obs.reference_range.low is not None or obs.reference_range.high is not None
    ):
        return obs.reference_range.low, obs.reference_range.high
    if concept and concept.reference_intervals:
        interval = concept.interval_for(None)
        if interval:
            return interval.low, interval.high
    return None, None


def _concept_for(obs: NormalizedObservation) -> ConceptDefinition | None:
    if obs.concept:
        definition = get_concept(obs.concept)
        if definition:
            return definition
    return resolve_codings(obs.source_codings)


def _in_window(obs: NormalizedObservation, start: datetime, end: datetime) -> bool:
    moment = obs.effective_datetime
    return moment is not None and start <= moment <= end


def _has_value(obs: NormalizedObservation) -> bool:
    return obs.numeric_value is not None or obs.interpretation is not None


def _format_value(obs: NormalizedObservation) -> str | None:
    if obs.quantity is None:
        return obs.value_string
    value = obs.quantity.canonical_value
    unit = obs.quantity.canonical_unit
    if value is None:
        value = obs.quantity.value
        unit = obs.quantity.unit
    if value is None:
        return None
    return f"{_trim(value)} {unit}".strip()


def _trim(value: float) -> str:
    return f"{value:.10g}"


def _epoch(moment: datetime | None) -> float:
    return moment.timestamp() if moment else 0.0
