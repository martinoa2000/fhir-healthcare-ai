"""Turns a :class:`PatientRecord` into a flat, model-ready feature vector.

This is the layer that separates "we passed some JSON to an LLM" from "we built a
clinical analytics pipeline". Everything downstream -- risk models, cohort statistics,
abnormality summaries -- consumes features, never raw FHIR.

Three properties matter more than the individual feature list:

**Missingness is data.** A patient with no HbA1c is not a patient with HbA1c 0. Numeric
features are ``None`` when unknown, and the concepts that were expected but absent are
reported in :attr:`FeatureSet.missing`, so a risk score can say what it could not see.

**Everything is computed as of a fixed instant.** ``as_of`` is threaded through every
window so that a result computed today and the same result recomputed next week from
the same data are identical. Without this, "last 6 months" quietly moves.

**Features are declared, not implied.** :data:`FEATURE_CATALOG` is the contract; the
API exposes it and the README documents it.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from fhir_healthcare_ai.domain.clinical import (
    NormalizedMedicationRequest,
    NormalizedObservation,
    PatientRecord,
)
from fhir_healthcare_ai.domain.enums import AbnormalFlag
from fhir_healthcare_ai.logging_config import get_logger
from fhir_healthcare_ai.terminology import get_concept

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

logger = get_logger(__name__)

# Labs and vitals that get the full numeric treatment (latest / trend / recency).
LAB_FEATURE_CONCEPTS: tuple[str, ...] = (
    "hba1c",
    "glucose",
    "creatinine",
    "egfr",
    "ldl_cholesterol",
    "hdl_cholesterol",
    "triglycerides",
    "potassium",
    "hemoglobin",
    "urine_albumin_creatinine_ratio",
    "systolic_bp",
    "diastolic_bp",
    "bmi",
)

# Conditions exposed as presence flags.
CONDITION_FEATURE_CONCEPTS: tuple[str, ...] = (
    "type_2_diabetes",
    "type_1_diabetes",
    "hypertension",
    "chronic_kidney_disease",
    "heart_failure",
    "hyperlipidemia",
    "obesity",
    "myocardial_infarction",
    "copd",
    "diabetic_nephropathy",
)

# Drug classes exposed as "currently prescribed" flags.
DRUG_CLASS_FEATURES: tuple[str, ...] = (
    "biguanide",
    "sulfonylurea",
    "dpp4_inhibitor",
    "sglt2_inhibitor",
    "glp1_agonist",
    "insulin",
    "ace_inhibitor",
    "arb",
    "statin",
    "beta_blocker",
    "loop_diuretic",
    "thiazide",
)

DIABETES_DRUG_CLASSES: frozenset[str] = frozenset(
    {"biguanide", "sulfonylurea", "dpp4_inhibitor", "sglt2_inhibitor", "glp1_agonist", "insulin"}
)

DEFAULT_LOOKBACK_DAYS = 365
MEDICATION_CHANGE_WINDOW_DAYS = 90
ABNORMAL_LAB_WINDOW_DAYS = 90


@dataclass(frozen=True)
class FeatureSpec:
    """Metadata for one feature. Drives documentation and the ``/capabilities`` API."""

    name: str
    dtype: str
    description: str
    group: str


def describe_features() -> tuple[FeatureSpec, ...]:
    """The full feature contract, in the order the builder emits it."""
    specs: list[FeatureSpec] = [
        FeatureSpec("age_years", "float", "Age in fractional years at as_of.", "demographics"),
        FeatureSpec("is_female", "bool", "Administrative gender is female.", "demographics"),
        FeatureSpec("is_deceased", "bool", "Patient is recorded as deceased.", "demographics"),
        FeatureSpec(
            "condition_count", "int", "Distinct active problem-list conditions.", "conditions"
        ),
    ]
    for concept in LAB_FEATURE_CONCEPTS:
        definition = get_concept(concept)
        label = definition.display if definition else concept
        unit = definition.canonical_unit if definition else None
        suffix = f" ({unit})" if unit else ""
        specs += [
            FeatureSpec(f"{concept}_latest", "float", f"Most recent {label}{suffix}.", "labs"),
            FeatureSpec(f"{concept}_days_since", "float", f"Days since the last {label}.", "labs"),
            FeatureSpec(f"{concept}_count", "int", f"Number of {label} results in window.", "labs"),
            FeatureSpec(f"{concept}_mean", "float", f"Mean {label} in window.", "labs"),
            FeatureSpec(f"{concept}_max", "float", f"Maximum {label} in window.", "labs"),
            FeatureSpec(f"{concept}_min", "float", f"Minimum {label} in window.", "labs"),
            FeatureSpec(f"{concept}_delta", "float", f"Latest minus previous {label}.", "labs"),
            FeatureSpec(
                f"{concept}_slope_per_year",
                "float",
                f"Least-squares slope of {label} per year over the window.",
                "labs",
            ),
        ]
    for concept in CONDITION_FEATURE_CONCEPTS:
        specs += [
            FeatureSpec(f"has_{concept}", "bool", f"Active condition: {concept}.", "conditions"),
            FeatureSpec(
                f"{concept}_years_since_onset",
                "float",
                f"Years since recorded onset of {concept}.",
                "conditions",
            ),
        ]
    for drug_class in DRUG_CLASS_FEATURES:
        specs.append(
            FeatureSpec(
                f"on_{drug_class}", "bool", f"Active order for a {drug_class}.", "medications"
            )
        )
    specs += [
        FeatureSpec(
            "active_medication_count", "int", "Distinct active medication orders.", "medications"
        ),
        FeatureSpec(
            "diabetes_medication_count",
            "int",
            "Distinct active glucose-lowering agents.",
            "medications",
        ),
        FeatureSpec(
            "medication_change_recent",
            "bool",
            f"Any medication started or stopped in the last {MEDICATION_CHANGE_WINDOW_DAYS} days.",
            "medications",
        ),
        FeatureSpec(
            "diabetes_medication_change_recent",
            "bool",
            f"Glucose-lowering therapy changed in the last {MEDICATION_CHANGE_WINDOW_DAYS} days.",
            "medications",
        ),
        FeatureSpec(
            "days_since_medication_change",
            "float",
            "Days since the most recent order or stop.",
            "medications",
        ),
        FeatureSpec(
            "encounter_count", "int", "Encounters within the lookback window.", "utilization"
        ),
        FeatureSpec(
            "inpatient_count", "int", "Inpatient encounters within the window.", "utilization"
        ),
        FeatureSpec(
            "emergency_count", "int", "Emergency encounters within the window.", "utilization"
        ),
        FeatureSpec(
            "days_since_last_encounter",
            "float",
            "Days since the most recent encounter.",
            "utilization",
        ),
        FeatureSpec(
            "inpatient_days",
            "float",
            "Total inpatient length of stay in the window.",
            "utilization",
        ),
        FeatureSpec(
            "abnormal_lab_count_recent",
            "int",
            f"Abnormal results in the last {ABNORMAL_LAB_WINDOW_DAYS} days.",
            "labs",
        ),
        FeatureSpec(
            "critical_lab_count_recent",
            "int",
            f"Critical results in the last {ABNORMAL_LAB_WINDOW_DAYS} days.",
            "labs",
        ),
        FeatureSpec("observation_count", "int", "All observations within the window.", "labs"),
    ]
    return tuple(specs)


@dataclass
class FeatureSet:
    """Computed features for one patient."""

    patient_id: str
    as_of: datetime
    values: dict[str, Any] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    lookback_days: int = DEFAULT_LOOKBACK_DAYS

    def get(self, name: str, default: Any = None) -> Any:
        value = self.values.get(name, default)
        return default if value is None else value

    def numeric(self, name: str) -> float | None:
        value = self.values.get(name)
        return (
            float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None
        )

    def flag(self, name: str) -> bool:
        return bool(self.values.get(name))

    def non_null(self) -> dict[str, Any]:
        """Only the features that were actually computable."""
        return {k: v for k, v in self.values.items() if v is not None}


class FeatureBuilder:
    """Computes a :class:`FeatureSet` from a :class:`PatientRecord`."""

    def __init__(
        self,
        *,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        lab_concepts: Sequence[str] = LAB_FEATURE_CONCEPTS,
        condition_concepts: Sequence[str] = CONDITION_FEATURE_CONCEPTS,
        drug_classes: Sequence[str] = DRUG_CLASS_FEATURES,
    ) -> None:
        self.lookback_days = lookback_days
        self.lab_concepts = tuple(lab_concepts)
        self.condition_concepts = tuple(condition_concepts)
        self.drug_classes = tuple(drug_classes)

    def build(self, record: PatientRecord, as_of: datetime | None = None) -> FeatureSet:
        """Compute every feature for one patient."""
        now = as_of or datetime.now(UTC)
        window_start = now - timedelta(days=self.lookback_days)
        features = FeatureSet(
            patient_id=record.patient_id, as_of=now, lookback_days=self.lookback_days
        )

        self._demographics(record, features)
        self._labs(record, features, now, window_start)
        self._conditions(record, features, now)
        self._medications(record, features, now)
        self._utilization(record, features, now, window_start)
        return features

    def build_many(
        self, records: Sequence[PatientRecord], as_of: datetime | None = None
    ) -> list[FeatureSet]:
        now = as_of or datetime.now(UTC)
        return [self.build(record, now) for record in records]

    # -- groups ---------------------------------------------------------------------

    def _demographics(self, record: PatientRecord, features: FeatureSet) -> None:
        patient = record.patient
        features.values["age_years"] = patient.age_years if patient else None
        features.values["is_female"] = (patient.gender == "female") if patient else None
        features.values["is_deceased"] = patient.deceased if patient else None
        if patient is None:
            features.missing.append("Patient demographics")
        elif patient.age_years is None:
            features.missing.append("birthDate")

    def _labs(
        self,
        record: PatientRecord,
        features: FeatureSet,
        now: datetime,
        window_start: datetime,
    ) -> None:
        by_concept = _index_observations(record)

        for concept in self.lab_concepts:
            series = by_concept.get(concept, [])
            in_window = [o for o in series if _within(o.effective_datetime, window_start, now)]
            values = [v for v in (o.numeric_value for o in in_window) if v is not None]

            latest = in_window[-1] if in_window else None
            previous = in_window[-2] if len(in_window) > 1 else None
            latest_value = latest.numeric_value if latest else None
            previous_value = previous.numeric_value if previous else None

            features.values[f"{concept}_latest"] = latest_value
            features.values[f"{concept}_days_since"] = _days_between(
                latest.effective_datetime if latest else None, now
            )
            features.values[f"{concept}_count"] = len(in_window)
            features.values[f"{concept}_mean"] = (
                round(statistics.fmean(values), 4) if values else None
            )
            features.values[f"{concept}_max"] = max(values) if values else None
            features.values[f"{concept}_min"] = min(values) if values else None
            features.values[f"{concept}_delta"] = (
                round(latest_value - previous_value, 4)
                if latest_value is not None and previous_value is not None
                else None
            )
            features.values[f"{concept}_slope_per_year"] = _slope_per_year(in_window)

            if not in_window:
                features.missing.append(concept)

        recent_start = now - timedelta(days=ABNORMAL_LAB_WINDOW_DAYS)
        recent = [
            o for o in record.observations if _within(o.effective_datetime, recent_start, now)
        ]
        features.values["abnormal_lab_count_recent"] = sum(
            1
            for o in recent
            if o.abnormal_flag
            in (
                AbnormalFlag.HIGH,
                AbnormalFlag.LOW,
                AbnormalFlag.CRITICAL_HIGH,
                AbnormalFlag.CRITICAL_LOW,
            )
        )
        features.values["critical_lab_count_recent"] = sum(
            1
            for o in recent
            if o.abnormal_flag in (AbnormalFlag.CRITICAL_HIGH, AbnormalFlag.CRITICAL_LOW)
        )
        features.values["observation_count"] = sum(
            1 for o in record.observations if _within(o.effective_datetime, window_start, now)
        )

    def _conditions(self, record: PatientRecord, features: FeatureSet, now: datetime) -> None:
        active = [c for c in record.conditions if c.is_active]
        by_concept = {c.concept: c for c in active if c.concept}

        for concept in self.condition_concepts:
            condition = by_concept.get(concept)
            features.values[f"has_{concept}"] = condition is not None
            onset = condition.onset.value if condition and condition.onset else None
            days = _days_between(onset, now)
            features.values[f"{concept}_years_since_onset"] = (
                round(days / 365.25, 2) if days is not None else None
            )

        features.values["condition_count"] = len({c.concept or c.id for c in active})

    def _medications(self, record: PatientRecord, features: FeatureSet, now: datetime) -> None:
        active = [m for m in record.medication_requests if m.is_active]
        active_classes = {m.drug_class for m in active if m.drug_class}

        for drug_class in self.drug_classes:
            features.values[f"on_{drug_class}"] = drug_class in active_classes

        features.values["active_medication_count"] = len({m.concept or m.id for m in active})
        features.values["diabetes_medication_count"] = len(
            {m.concept for m in active if m.drug_class in DIABETES_DRUG_CLASSES and m.concept}
        )

        change_window_start = now - timedelta(days=MEDICATION_CHANGE_WINDOW_DAYS)
        changes = [m for m in record.medication_requests if _is_change(m, change_window_start, now)]
        features.values["medication_change_recent"] = bool(changes)
        features.values["diabetes_medication_change_recent"] = any(
            m.drug_class in DIABETES_DRUG_CLASSES for m in changes
        )

        change_dates = [d for d in (_change_date(m) for m in record.medication_requests) if d]
        features.values["days_since_medication_change"] = (
            _days_between(max(change_dates), now) if change_dates else None
        )

        if not record.medication_requests:
            features.missing.append("MedicationRequest")

    def _utilization(
        self,
        record: PatientRecord,
        features: FeatureSet,
        now: datetime,
        window_start: datetime,
    ) -> None:
        in_window = [e for e in record.encounters if _within(e.period_start, window_start, now)]
        classes = [
            (e.encounter_class.code or "").upper() if e.encounter_class else "" for e in in_window
        ]
        features.values["encounter_count"] = len(in_window)
        features.values["inpatient_count"] = sum(
            1 for c in classes if c in {"IMP", "ACUTE", "NONAC"}
        )
        features.values["emergency_count"] = sum(1 for c in classes if c == "EMER")
        features.values["inpatient_days"] = round(
            sum(
                e.length_days or 0.0
                for e, c in zip(in_window, classes, strict=True)
                if c in {"IMP", "ACUTE", "NONAC"}
            ),
            2,
        )
        starts = [e.period_start for e in record.encounters if e.period_start]
        features.values["days_since_last_encounter"] = (
            _days_between(max(starts), now) if starts else None
        )
        if not record.encounters:
            features.missing.append("Encounter")


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _index_observations(record: PatientRecord) -> dict[str, list[NormalizedObservation]]:
    """Group observations by concept, oldest first, flattening panel components.

    A blood-pressure panel carries no top-level value: systolic and diastolic live in
    ``component``. Flattening here means every downstream consumer sees ``systolic_bp``
    regardless of how the source chose to structure it.
    """
    index: dict[str, list[NormalizedObservation]] = {}
    for observation in record.observations:
        candidates = observation.components or [observation]
        for candidate in candidates:
            if candidate.concept and candidate.numeric_value is not None:
                index.setdefault(candidate.concept, []).append(candidate)
    for series in index.values():
        series.sort(key=lambda o: o.effective_datetime or datetime.min.replace(tzinfo=UTC))
    return index


def _within(moment: datetime | None, start: datetime, end: datetime) -> bool:
    return moment is not None and start <= moment <= end


def _days_between(earlier: datetime | None, later: datetime) -> float | None:
    if earlier is None:
        return None
    return round((later - earlier).total_seconds() / 86400, 2)


def _slope_per_year(observations: list[NormalizedObservation]) -> float | None:
    """Least-squares slope in units per year.

    Needs at least three points spread over time; two points make a line through noise,
    and a single-day cluster makes the denominator explode.
    """
    points = [
        (o.effective_datetime, o.numeric_value)
        for o in observations
        if o.effective_datetime and o.numeric_value is not None
    ]
    if len(points) < 3:
        return None
    origin = points[0][0]
    xs = [(moment - origin).total_seconds() / (365.25 * 86400) for moment, _ in points]
    ys = [value for _, value in points]
    span = max(xs) - min(xs)
    if span < 1 / 365.25:
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return None
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    return round(numerator / denominator, 4)


def _change_date(request: NormalizedMedicationRequest) -> datetime | None:
    """When this order last represented a therapy change.

    A stopped order changes therapy at its validity end; anything else changes therapy
    when it was authored.
    """
    if (request.status or "").lower() in {"stopped", "cancelled", "completed"}:
        return request.validity_end or request.authored_on
    return request.authored_on


def _is_change(request: NormalizedMedicationRequest, window_start: datetime, now: datetime) -> bool:
    return _within(_change_date(request), window_start, now)


def build_feature_frame(
    feature_sets: Sequence[FeatureSet], columns: Sequence[str] | None = None
) -> pd.DataFrame:
    """Stack feature sets into a pandas DataFrame indexed by patient id.

    pandas is imported lazily: the API and the query path never need it, and a 300 ms
    import on every process start is a poor trade for a dependency used by analytics only.
    """
    import pandas as pd

    if not feature_sets:
        return pd.DataFrame(columns=list(columns or []))
    rows = {fs.patient_id: fs.values for fs in feature_sets}
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "patient_id"
    if columns is not None:
        frame = frame.reindex(columns=list(columns))
    return frame
