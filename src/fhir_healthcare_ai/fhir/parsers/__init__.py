"""Resource parsers and the default registry.

To support a new resource type: implement a :class:`ResourceParser`, add it to
:func:`default_registry`, and add a :class:`~fhir_healthcare_ai.fhir.allowlist.ResourcePolicy`.
Both steps are required -- a parser without a policy is unreachable, and a policy
without a parser retrieves data nothing can read.
"""

from fhir_healthcare_ai.fhir.parsers.base import ParseError, ParserRegistry, ResourceParser
from fhir_healthcare_ai.fhir.parsers.condition import ConditionParser
from fhir_healthcare_ai.fhir.parsers.diagnostic_report import DiagnosticReportParser
from fhir_healthcare_ai.fhir.parsers.encounter import EncounterParser
from fhir_healthcare_ai.fhir.parsers.medication_request import MedicationRequestParser
from fhir_healthcare_ai.fhir.parsers.observation import ObservationParser
from fhir_healthcare_ai.fhir.parsers.patient import PatientParser


def default_registry() -> ParserRegistry:
    """Registry covering every currently reachable resource type."""
    return ParserRegistry(
        [
            PatientParser(),
            ObservationParser(),
            ConditionParser(),
            MedicationRequestParser(),
            EncounterParser(),
            DiagnosticReportParser(),
        ]
    )


__all__ = [
    "ConditionParser",
    "DiagnosticReportParser",
    "EncounterParser",
    "MedicationRequestParser",
    "ObservationParser",
    "ParseError",
    "ParserRegistry",
    "PatientParser",
    "ResourceParser",
    "default_registry",
]
