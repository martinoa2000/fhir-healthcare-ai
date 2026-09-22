"""Encounter parser."""

from __future__ import annotations

from typing import Any

from fhir_healthcare_ai.domain.clinical import NormalizedEncounter
from fhir_healthcare_ai.domain.enums import ResourceType
from fhir_healthcare_ai.fhir.parsers.base import ResourceParser
from fhir_healthcare_ai.fhir.parsers.primitives import (
    first_display,
    get_dict,
    get_float,
    get_list,
    get_str,
    parse_codeable_concept,
    parse_coding,
    parse_period,
    parse_reference_id,
)

_LENGTH_UNIT_DAYS: dict[str, float] = {
    "d": 1.0,
    "day": 1.0,
    "days": 1.0,
    "h": 1 / 24,
    "hour": 1 / 24,
    "hours": 1 / 24,
    "min": 1 / 1440,
    "minute": 1 / 1440,
    "minutes": 1 / 1440,
}


class EncounterParser(ResourceParser[NormalizedEncounter]):
    """Parses encounters and derives a length of stay in days.

    ``Encounter.length`` is preferred when present; otherwise the period is used. An
    open encounter (start but no end) has no length, and reporting one as zero would
    quietly understate inpatient stays.
    """

    resource_type = ResourceType.ENCOUNTER

    def _parse(self, resource: dict[str, Any], resource_id: str) -> NormalizedEncounter:
        period_start, period_end = parse_period(get_dict(resource, "period"))
        length_days = _length_days(get_dict(resource, "length"))
        if length_days is None and period_start and period_end:
            length_days = round((period_end - period_start).total_seconds() / 86400, 3)

        type_codings = [
            coding
            for entry in get_list(resource, "type")
            for coding in parse_codeable_concept(entry)[0]
        ]
        reason_codings = [
            coding
            for entry in get_list(resource, "reasonCode")
            for coding in parse_codeable_concept(entry)[0]
        ]
        encounter_class = parse_coding(resource.get("class"))

        return NormalizedEncounter(
            id=resource_id,
            patient_id=self.subject_id(resource),
            source_codings=type_codings,
            display=first_display(type_codings)
            or (encounter_class.display if encounter_class else None),
            status=get_str(resource, "status"),
            encounter_class=encounter_class,
            type_codings=type_codings,
            period_start=period_start,
            period_end=period_end,
            length_days=length_days,
            reason_codings=reason_codings,
            service_provider=parse_reference_id(resource.get("serviceProvider")),
        )


def _length_days(node: dict[str, Any]) -> float | None:
    value = get_float(node, "value")
    if value is None:
        return None
    unit = (get_str(node, "code") or get_str(node, "unit") or "d").lower()
    factor = _LENGTH_UNIT_DAYS.get(unit)
    return round(value * factor, 3) if factor else None
