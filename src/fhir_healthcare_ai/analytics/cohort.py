"""Cohort identification and aggregate statistics.

A multi-step plan produces one patient set per step. Turning those sets into *the*
cohort is the job of :class:`CohortLogic`:

``ALL``
    Intersection. "Patients with elevated HbA1c **and** a recent diabetes medication
    change" means a patient must appear in both step results. This is the default and
    the safer reading of a conjunctive clinical question.

``ANY``
    Union. Used for questions phrased as "patients on metformin or a sulfonylurea".

Each step also carries a :class:`~fhir_healthcare_ai.domain.enums.StepRole`:

``filter``
    The step selects patients and takes part in the ``ALL``/``ANY`` combination. A
    dependent filter step ("... *and* a recent medication change") narrows its parent.

``context``
    The step fetches data about patients who are already selected (a cohort's recent
    labs). It must not narrow the cohort: a patient with no lab in the window is still
    a member, just one with a data gap.

``exclude``
    Every patient the step returns is removed from the cohort. This is how "diabetic
    patients *not* on a statin" is answered, since FHIR search cannot express an absent
    resource.

Aggregate statistics are computed from the feature layer, never from raw resources, and
report ``n`` alongside every statistic because a median over three patients and a
median over three hundred are different claims.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from fhir_healthcare_ai.domain.clinical import PatientRecord
from fhir_healthcare_ai.domain.enums import CohortLogic, StepRole
from fhir_healthcare_ai.domain.results import CohortSummary, Evidence, PatientMatch
from fhir_healthcare_ai.features.builder import FeatureSet
from fhir_healthcare_ai.logging_config import get_logger

logger = get_logger(__name__)

#: Features summarised for every cohort, when present.
DEFAULT_SUMMARY_FEATURES: tuple[str, ...] = (
    "age_years",
    "hba1c_latest",
    "egfr_latest",
    "systolic_bp_latest",
    "ldl_cholesterol_latest",
    "condition_count",
    "active_medication_count",
    "encounter_count",
)

#: Boolean features reported as a count and a proportion.
DEFAULT_SUMMARY_FLAGS: tuple[str, ...] = (
    "is_female",
    "has_type_2_diabetes",
    "has_hypertension",
    "has_chronic_kidney_disease",
    "on_biguanide",
    "on_insulin",
    "on_sglt2_inhibitor",
    "medication_change_recent",
)


@dataclass
class StepResult:
    """Which patients one plan step returned, and what that means for the cohort.

    Args:
        step_id: The plan step this came from.
        patient_ids: Patients the step returned.
        role: How the step's patient set combines into the cohort.
        truncated: The step hit a page or resource cap, so its patient set is partial.
    """

    step_id: str
    patient_ids: set[str] = field(default_factory=set)
    role: StepRole = StepRole.FILTER
    truncated: bool = False

    @property
    def selective(self) -> bool:
        return self.role is StepRole.FILTER


class CohortResolver:
    """Combines per-step patient sets into a single cohort."""

    def __init__(self, logic: CohortLogic = CohortLogic.ALL) -> None:
        self.logic = logic

    def resolve(self, steps: Sequence[StepResult]) -> tuple[set[str], list[str]]:
        """Return the cohort and any warnings about how it was derived."""
        warnings: list[str] = []
        selective = [step for step in steps if step.role is StepRole.FILTER]
        excluding = [step for step in steps if step.role is StepRole.EXCLUDE]

        if not selective:
            warnings.append(
                "no selective step in the plan; cohort is the union of everything retrieved"
            )
            included = [step for step in steps if step.role is not StepRole.EXCLUDE]
            cohort = set().union(*(step.patient_ids for step in included))
            return self._exclude(cohort, excluding, warnings), warnings

        if any(step.truncated for step in selective):
            truncated_ids = [step.step_id for step in selective if step.truncated]
            warnings.append(
                "result truncated for step(s) "
                f"{', '.join(truncated_ids)}; the cohort may be incomplete"
            )

        empty = [step.step_id for step in selective if not step.patient_ids]
        if empty and self.logic is CohortLogic.ALL:
            warnings.append(
                f"step(s) {', '.join(empty)} matched no patients, so the intersection is empty"
            )

        sets = [step.patient_ids for step in selective]
        if self.logic is CohortLogic.ANY:
            cohort = set().union(*sets)
        else:
            cohort = set(sets[0])
            for other in sets[1:]:
                cohort &= other
        return self._exclude(cohort, excluding, warnings), warnings

    @staticmethod
    def _exclude(
        cohort: set[str], excluding: Sequence[StepResult], warnings: list[str]
    ) -> set[str]:
        for step in excluding:
            if step.truncated:
                # A partial exclusion list errs towards *keeping* patients who should
                # have been removed, which is the direction a reviewer must be told about.
                warnings.append(
                    f"exclusion step {step.step_id} was truncated; some patients it should "
                    "have removed may still be listed"
                )
            removed = cohort & step.patient_ids
            if removed:
                warnings.append(
                    f"{len(removed)} patient(s) removed by exclusion step {step.step_id}"
                )
            cohort = cohort - step.patient_ids
        return cohort

    def matched_steps(self, patient_id: str, steps: Sequence[StepResult]) -> list[str]:
        """Which steps a given patient satisfied. Shown per patient in the response."""
        return [
            step.step_id
            for step in steps
            if step.role is not StepRole.EXCLUDE and patient_id in step.patient_ids
        ]


def build_matches(
    patient_ids: Iterable[str],
    records: Mapping[str, PatientRecord],
    steps: Sequence[StepResult] = (),
    *,
    evidence: Mapping[str, Sequence[Evidence]] | None = None,
    summaries: Mapping[str, str] | None = None,
    limit: int | None = None,
) -> list[PatientMatch]:
    """Turn a set of patient ids into ordered, demographically annotated matches."""
    resolver = CohortResolver()
    ordered = sorted(patient_ids)
    if limit is not None:
        ordered = ordered[:limit]

    matches: list[PatientMatch] = []
    for patient_id in ordered:
        record = records.get(patient_id)
        patient = record.patient if record else None
        matches.append(
            PatientMatch(
                patient_id=patient_id,
                matched_steps=resolver.matched_steps(patient_id, steps),
                age_years=patient.age_years if patient else None,
                gender=patient.gender if patient else None,
                summary=(summaries or {}).get(patient_id),
                evidence=list((evidence or {}).get(patient_id, [])),
            )
        )
    return matches


def summarize_cohort(
    feature_sets: Sequence[FeatureSet],
    *,
    records: Mapping[str, PatientRecord] | None = None,
    per_step_counts: Mapping[str, int] | None = None,
    numeric_features: Sequence[str] = DEFAULT_SUMMARY_FEATURES,
    flag_features: Sequence[str] = DEFAULT_SUMMARY_FLAGS,
) -> CohortSummary:
    """Descriptive statistics over the cohort's feature layer.

    Every numeric entry carries its own ``n`` because feature coverage differs between
    features: a cohort of 100 may have age for all of them and eGFR for 40.
    """
    statistics_block: dict[str, Any] = {}

    for name in numeric_features:
        values = [v for v in (fs.numeric(name) for fs in feature_sets) if v is not None]
        if not values:
            continue
        statistics_block[name] = _describe(values)

    flags: dict[str, Any] = {}
    for name in flag_features:
        known = [fs.values.get(name) for fs in feature_sets if fs.values.get(name) is not None]
        if not known:
            continue
        true_count = sum(1 for value in known if bool(value))
        flags[name] = {
            "n": len(known),
            "count": true_count,
            "proportion": round(true_count / len(known), 4),
        }
    if flags:
        statistics_block["flags"] = flags

    coverage = _coverage(feature_sets, numeric_features)
    if coverage:
        statistics_block["feature_coverage"] = coverage

    total_resources = sum(record.resource_count for record in records.values()) if records else 0
    return CohortSummary(
        total_patients=len(feature_sets),
        total_resources=total_resources,
        per_step_counts=dict(per_step_counts or {}),
        statistics=statistics_block,
    )


def _describe(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(values)
    block: dict[str, Any] = {
        "n": len(ordered),
        "mean": round(statistics.fmean(ordered), 4),
        "median": round(statistics.median(ordered), 4),
        "min": round(ordered[0], 4),
        "max": round(ordered[-1], 4),
    }
    if len(ordered) > 1:
        block["stdev"] = round(statistics.stdev(ordered), 4)
    if len(ordered) >= 4:
        quartiles = statistics.quantiles(ordered, n=4, method="inclusive")
        block["p25"] = round(quartiles[0], 4)
        block["p75"] = round(quartiles[2], 4)
    return block


def _coverage(feature_sets: Sequence[FeatureSet], names: Sequence[str]) -> dict[str, float]:
    """Fraction of the cohort for which each feature was computable.

    This is the honesty channel: a statistic over 12% of the cohort should not be read
    the same way as one over 100%.
    """
    if not feature_sets:
        return {}
    total = len(feature_sets)
    return {
        name: round(sum(1 for fs in feature_sets if fs.values.get(name) is not None) / total, 4)
        for name in names
    }
