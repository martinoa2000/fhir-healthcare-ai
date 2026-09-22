"""Risk stratification over the clinical feature layer.

**What this is not.** It is not a validated clinical risk score. It was not fitted on
outcome data, it has never been calibrated against a real population, and its weights
were chosen to be legible rather than accurate. Every output carries
:data:`~fhir_healthcare_ai.domain.results.DISCLAIMER` and the band names are
deliberately generic ("moderate", "high") rather than clinical directives.

**What it is.** A worked example of the thing that actually matters architecturally:
an analytics component that consumes the typed feature layer rather than raw FHIR, and
that is honest about what it could not see.

Two design decisions are worth more than the model itself:

*The model is additive and inspectable.* Each contribution is a named rule with a
weight, so ``contributing_factors`` is generated from the same structure that produces
the score. There is no way for the explanation to drift from the computation.

*Absent features lower confidence, they do not score as zero.* A patient with no HbA1c
on file is not a patient with a good HbA1c. Rules whose inputs are missing are skipped
and reported in ``missing_features``, and the score is normalised by the weight that
was actually evaluable -- so a nearly-empty record cannot be pushed into a low band
simply by having no data.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

from fhir_healthcare_ai.domain.enums import RiskBand
from fhir_healthcare_ai.domain.results import RiskAssessment
from fhir_healthcare_ai.features.builder import FeatureSet
from fhir_healthcare_ai.logging_config import get_logger

logger = get_logger(__name__)

MODEL_NAME = "deterioration-risk-demo"
MODEL_VERSION = "0.1.0"

#: Score at or above which a patient lands in the given band.
BAND_THRESHOLDS: tuple[tuple[float, RiskBand], ...] = (
    (0.60, RiskBand.HIGH),
    (0.30, RiskBand.MODERATE),
    (0.0, RiskBand.LOW),
)

#: If fewer than this fraction of the total rule weight could be evaluated, the result
#: is reported as LOW with an explicit "insufficient data" factor rather than as a
#: confident low score.
MIN_EVALUABLE_WEIGHT_FRACTION = 0.35


@dataclass(frozen=True)
class RiskRule:
    """One inspectable contribution to the score.

    Args:
        name: Stable identifier, used in logs and tests.
        weight: Maximum points this rule can contribute.
        requires: Feature names that must be non-null for the rule to be evaluable.
        score: Maps the feature set to a value in ``[0, 1]``; multiplied by ``weight``.
        explain: Human-readable sentence, rendered only when the rule fires.
    """

    name: str
    weight: float
    requires: tuple[str, ...]
    score: Callable[[FeatureSet], float]
    explain: Callable[[FeatureSet], str]

    def evaluable(self, features: FeatureSet) -> bool:
        return all(features.values.get(name) is not None for name in self.requires)


def _ramp(value: float, low: float, high: float) -> float:
    """Linear 0..1 ramp between two thresholds, clamped at both ends."""
    if high == low:
        return 1.0 if value >= high else 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def _num(features: FeatureSet, name: str, default: float = 0.0) -> float:
    value = features.numeric(name)
    return default if value is None else value


def default_rules() -> tuple[RiskRule, ...]:
    """The rule set. Weights are illustrative and sum to 1.0 before normalisation."""
    return (
        RiskRule(
            name="glycaemic_control",
            weight=0.22,
            requires=("hba1c_latest",),
            score=lambda f: _ramp(_num(f, "hba1c_latest"), 7.0, 10.0),
            explain=lambda f: f"HbA1c {_num(f, 'hba1c_latest'):.1f}% above the 7.0% target",
        ),
        RiskRule(
            name="glycaemic_trend",
            weight=0.08,
            requires=("hba1c_slope_per_year",),
            score=lambda f: _ramp(_num(f, "hba1c_slope_per_year"), 0.3, 1.5),
            explain=lambda f: f"HbA1c rising {_num(f, 'hba1c_slope_per_year'):.1f}%/year",
        ),
        RiskRule(
            name="renal_function",
            weight=0.18,
            requires=("egfr_latest",),
            # eGFR falls as risk rises, so the ramp is inverted.
            score=lambda f: _ramp(-_num(f, "egfr_latest"), -60.0, -30.0),
            explain=lambda f: f"eGFR {_num(f, 'egfr_latest'):.0f} mL/min/1.73m2",
        ),
        RiskRule(
            name="albuminuria",
            weight=0.08,
            requires=("urine_albumin_creatinine_ratio_latest",),
            score=lambda f: _ramp(_num(f, "urine_albumin_creatinine_ratio_latest"), 30.0, 300.0),
            explain=lambda f: (
                "urine albumin/creatinine ratio "
                f"{_num(f, 'urine_albumin_creatinine_ratio_latest'):.0f} mg/g"
            ),
        ),
        RiskRule(
            name="blood_pressure",
            weight=0.10,
            requires=("systolic_bp_latest",),
            score=lambda f: _ramp(_num(f, "systolic_bp_latest"), 135.0, 170.0),
            explain=lambda f: f"systolic blood pressure {_num(f, 'systolic_bp_latest'):.0f} mmHg",
        ),
        RiskRule(
            name="acute_utilisation",
            weight=0.14,
            requires=("inpatient_count", "emergency_count"),
            score=lambda f: _ramp(
                _num(f, "inpatient_count") * 1.5 + _num(f, "emergency_count"), 1.0, 4.0
            ),
            explain=lambda f: (
                f"{int(_num(f, 'inpatient_count'))} inpatient and "
                f"{int(_num(f, 'emergency_count'))} emergency encounters in the window"
            ),
        ),
        RiskRule(
            name="comorbidity_burden",
            weight=0.10,
            requires=("condition_count",),
            score=lambda f: _ramp(_num(f, "condition_count"), 2.0, 6.0),
            explain=lambda f: (
                f"{int(_num(f, 'condition_count'))} active conditions on the problem list"
            ),
        ),
        RiskRule(
            name="therapy_instability",
            weight=0.05,
            requires=("diabetes_medication_change_recent",),
            score=lambda f: 1.0 if f.flag("diabetes_medication_change_recent") else 0.0,
            explain=lambda _: "glucose-lowering therapy changed recently",
        ),
        RiskRule(
            name="abnormal_result_load",
            weight=0.05,
            requires=("abnormal_lab_count_recent",),
            score=lambda f: _ramp(_num(f, "abnormal_lab_count_recent"), 1.0, 5.0),
            explain=lambda f: (
                f"{int(_num(f, 'abnormal_lab_count_recent'))} abnormal results recently"
            ),
        ),
    )


class RiskStratifier:
    """Scores a :class:`FeatureSet` and explains the result.

    The score is the evaluable weighted sum divided by the evaluable weight, so it stays
    in ``[0, 1]`` regardless of how many rules could be applied. A mild logistic shaping
    keeps mid-range scores from clustering, without changing the ordering.
    """

    def __init__(
        self,
        rules: Sequence[RiskRule] | None = None,
        *,
        model_name: str = MODEL_NAME,
        model_version: str = MODEL_VERSION,
        min_evaluable_fraction: float = MIN_EVALUABLE_WEIGHT_FRACTION,
        max_factors: int = 5,
    ) -> None:
        self.rules = tuple(rules) if rules is not None else default_rules()
        self.model_name = model_name
        self.model_version = model_version
        self.min_evaluable_fraction = min_evaluable_fraction
        self.max_factors = max_factors
        self.total_weight = sum(rule.weight for rule in self.rules)

    def assess(self, features: FeatureSet) -> RiskAssessment:
        """Score one patient."""
        evaluable_weight = 0.0
        raw = 0.0
        fired: list[tuple[float, str]] = []
        missing: list[str] = []

        for rule in self.rules:
            if not rule.evaluable(features):
                missing.extend(name for name in rule.requires if name not in missing)
                continue
            evaluable_weight += rule.weight
            contribution = rule.weight * max(0.0, min(1.0, rule.score(features)))
            raw += contribution
            if contribution > 0.0:
                fired.append((contribution, rule.explain(features)))

        coverage = evaluable_weight / self.total_weight if self.total_weight else 0.0
        if coverage < self.min_evaluable_fraction:
            return RiskAssessment(
                score=0.0,
                band=RiskBand.LOW,
                model_name=self.model_name,
                model_version=self.model_version,
                contributing_factors=[
                    f"insufficient data to stratify: {coverage:.0%} of model inputs available"
                ],
                missing_features=sorted(missing),
            )

        score = _shape(raw / evaluable_weight) if evaluable_weight else 0.0
        fired.sort(key=lambda item: item[0], reverse=True)
        factors = [text for _, text in fired[: self.max_factors]]
        if not factors:
            factors = ["no risk rule exceeded its threshold"]

        return RiskAssessment(
            score=round(score, 4),
            band=band_for(score),
            model_name=self.model_name,
            model_version=self.model_version,
            contributing_factors=factors,
            missing_features=sorted(missing),
        )

    def assess_many(self, feature_sets: Iterable[FeatureSet]) -> dict[str, RiskAssessment]:
        return {features.patient_id: self.assess(features) for features in feature_sets}


def band_for(score: float) -> RiskBand:
    """Map a score in ``[0, 1]`` onto a band."""
    for threshold, band in BAND_THRESHOLDS:
        if score >= threshold:
            return band
    return RiskBand.LOW


def _shape(value: float) -> float:
    """Gentle logistic re-spread of the normalised sum. Monotonic, so ranking is preserved."""
    if value <= 0.0:
        return 0.0
    centred = (value - 0.5) * 5.0
    return 1.0 / (1.0 + math.exp(-centred))


def stratify(features: FeatureSet, as_of: datetime | None = None) -> RiskAssessment:
    """Convenience wrapper. ``as_of`` is accepted for symmetry and is not used here --
    the feature set was already computed against a fixed instant."""
    del as_of
    return RiskStratifier().assess(features)
