"""Condition parser."""

from __future__ import annotations

from typing import Any

from fhir_healthcare_ai.domain.clinical import Coding, NormalizedCondition
from fhir_healthcare_ai.domain.enums import ResourceType
from fhir_healthcare_ai.fhir.parsers.base import ResourceParser
from fhir_healthcare_ai.fhir.parsers.primitives import (
    first_display,
    get_dict,
    get_list,
    parse_codeable_concept,
    parse_coding,
    parse_datetime,
    parse_datetime_value,
    parse_period,
    status_code,
)
from fhir_healthcare_ai.terminology import resolve_codings


class ConditionParser(ResourceParser[NormalizedCondition]):
    """Parses problems and diagnoses.

    ``onset[x]`` is a choice of dateTime, Age, Period, Range or string. Only the
    time-based variants are usable for interval logic, so an Age or Range onset is
    recorded as a warning instead of being coerced into a fake date.
    """

    resource_type = ResourceType.CONDITION

    def _parse(self, resource: dict[str, Any], resource_id: str) -> NormalizedCondition:
        warnings: list[str] = []
        codings, text = parse_codeable_concept(get_dict(resource, "code"))
        concept = resolve_codings(codings)
        if concept is None and codings:
            warnings.append(f"unmapped code: {codings[0].token}")

        onset = parse_datetime(resource.get("onsetDateTime"))
        if onset is None:
            start, _ = parse_period(get_dict(resource, "onsetPeriod"))
            if start is not None:
                onset = parse_datetime(start.isoformat())
        if onset is None and any(
            key in resource for key in ("onsetAge", "onsetRange", "onsetString")
        ):
            warnings.append("onset recorded in a non-temporal form; interval filters will skip it")

        abatement = parse_datetime(resource.get("abatementDateTime"))
        if abatement is None:
            start, _ = parse_period(get_dict(resource, "abatementPeriod"))
            if start is not None:
                abatement = parse_datetime(start.isoformat())

        severity_codings, severity_text = parse_codeable_concept(get_dict(resource, "severity"))
        severity = severity_codings[0] if severity_codings else None
        if severity is None and severity_text:
            severity = Coding(display=severity_text)

        return NormalizedCondition(
            id=resource_id,
            patient_id=self.subject_id(resource),
            encounter_id=self.encounter_id(resource),
            source_codings=codings,
            concept=concept.key if concept else None,
            display=first_display(codings, text),
            clinical_status=status_code(resource, "clinicalStatus"),
            verification_status=status_code(resource, "verificationStatus"),
            category=[
                c
                for c in (
                    parse_coding(item)
                    for entry in get_list(resource, "category")
                    for item in get_list(entry, "coding")
                )
                if c
            ],
            severity=severity,
            onset=onset,
            abatement=abatement,
            recorded_date=parse_datetime_value(resource.get("recordedDate")),
            parse_warnings=warnings,
        )
