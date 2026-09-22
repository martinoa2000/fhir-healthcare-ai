"""Assembles a flat stream of FHIR resources into per-patient records.

FHIR search results arrive as a bag of resources in arbitrary order, with references
pointing in every direction. The feature layer wants the opposite shape: one object
per patient, with that patient's observations, conditions, medications and encounters
already attached.

Resources whose subject cannot be determined are dropped, but counted. Silently losing
them would make a cohort look smaller than it is with no way to notice.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from fhir_healthcare_ai.domain.clinical import (
    NormalizedCondition,
    NormalizedDiagnosticReport,
    NormalizedEncounter,
    NormalizedMedicationRequest,
    NormalizedObservation,
    NormalizedPatient,
    NormalizedResource,
    PatientRecord,
)
from fhir_healthcare_ai.domain.enums import ResourceType
from fhir_healthcare_ai.fhir.parsers import ParserRegistry, default_registry
from fhir_healthcare_ai.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class AssemblyStats:
    """What the assembler did, for observability and for the response trace."""

    parsed: int = 0
    unsupported: int = 0
    unparseable: int = 0
    orphaned: int = 0
    per_type: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "parsed": self.parsed,
            "unsupported": self.unsupported,
            "unparseable": self.unparseable,
            "orphaned": self.orphaned,
            "per_type": dict(self.per_type),
        }


class RecordAssembler:
    """Groups normalized resources into :class:`PatientRecord` objects."""

    def __init__(self, registry: ParserRegistry | None = None) -> None:
        self.registry = registry or default_registry()

    def assemble(
        self, resources: Iterable[dict[str, Any]]
    ) -> tuple[dict[str, PatientRecord], AssemblyStats]:
        """Parse and group raw resources.

        Patient resources create records; everything else is attached to the record
        for its subject, creating a stub record if the Patient itself was not fetched
        (which is normal when a search returns only Observations).
        """
        stats = AssemblyStats()
        records: dict[str, PatientRecord] = {}

        for raw in resources:
            if not isinstance(raw, dict):
                stats.unparseable += 1
                continue
            raw_type = raw.get("resourceType")
            if not isinstance(raw_type, str) or raw_type not in _SUPPORTED_NAMES:
                stats.unsupported += 1
                continue

            parsed = self.registry.parse(raw)
            if parsed is None:
                stats.unparseable += 1
                continue

            stats.parsed += 1
            stats.per_type[parsed.resource_type.value] = (
                stats.per_type.get(parsed.resource_type.value, 0) + 1
            )

            patient_id = (
                parsed.id if parsed.resource_type is ResourceType.PATIENT else parsed.patient_id
            )
            if not patient_id:
                stats.orphaned += 1
                continue

            record = records.setdefault(patient_id, PatientRecord(patient_id=patient_id))
            _attach(record, parsed)

        if stats.orphaned:
            logger.warning(
                "dropped resources with no resolvable subject",
                extra={"orphaned": stats.orphaned},
            )
        return records, stats

    def assemble_one(self, patient_id: str, resources: Iterable[dict[str, Any]]) -> PatientRecord:
        """Assemble a single patient's record, ignoring resources for anyone else."""
        records, _ = self.assemble(resources)
        return records.get(patient_id) or PatientRecord(patient_id=patient_id)


def _attach(record: PatientRecord, resource: NormalizedResource) -> None:
    """Place a parsed resource on the right list of its patient's record."""
    match resource:
        case NormalizedPatient():
            record.patient = resource
        case NormalizedObservation():
            record.observations.append(resource)
        case NormalizedCondition():
            record.conditions.append(resource)
        case NormalizedMedicationRequest():
            record.medication_requests.append(resource)
        case NormalizedEncounter():
            record.encounters.append(resource)
        case NormalizedDiagnosticReport():
            record.diagnostic_reports.append(resource)
        case _:  # pragma: no cover - guarded by _SUPPORTED_NAMES
            logger.warning(
                "no slot for normalized resource",
                extra={"resource_type": resource.resource_type.value},
            )


_SUPPORTED_NAMES: frozenset[str] = frozenset(
    {
        ResourceType.PATIENT.value,
        ResourceType.OBSERVATION.value,
        ResourceType.CONDITION.value,
        ResourceType.MEDICATION_REQUEST.value,
        ResourceType.ENCOUNTER.value,
        ResourceType.DIAGNOSTIC_REPORT.value,
    }
)


def assemble_records(
    resources: Iterable[dict[str, Any]], registry: ParserRegistry | None = None
) -> dict[str, PatientRecord]:
    """Convenience wrapper when the statistics are not needed."""
    records, _ = RecordAssembler(registry).assemble(resources)
    return records
