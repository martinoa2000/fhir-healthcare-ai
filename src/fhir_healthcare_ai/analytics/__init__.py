"""Analytics layer: features in, structured clinical findings out.

Nothing here touches FHIR JSON. Every component consumes the typed feature layer or a
:class:`~fhir_healthcare_ai.domain.clinical.PatientRecord`, which is what makes the
models testable without a server and swappable without touching retrieval.
"""

from fhir_healthcare_ai.analytics.abnormal_labs import (
    AbnormalLabDetector,
    detect_abnormal_labs,
    summarize_findings,
)
from fhir_healthcare_ai.analytics.cohort import (
    CohortResolver,
    StepResult,
    build_matches,
    summarize_cohort,
)
from fhir_healthcare_ai.analytics.risk import (
    MODEL_NAME,
    MODEL_VERSION,
    RiskRule,
    RiskStratifier,
    band_for,
    default_rules,
    stratify,
)

__all__ = [
    "MODEL_NAME",
    "MODEL_VERSION",
    "AbnormalLabDetector",
    "CohortResolver",
    "RiskRule",
    "RiskStratifier",
    "StepResult",
    "band_for",
    "build_matches",
    "default_rules",
    "detect_abnormal_labs",
    "stratify",
    "summarize_cohort",
    "summarize_findings",
]
