"""DiagnosticReport parser."""

from __future__ import annotations

from typing import Any

from fhir_healthcare_ai.domain.clinical import NormalizedDiagnosticReport
from fhir_healthcare_ai.domain.enums import ResourceType
from fhir_healthcare_ai.fhir.parsers.base import ResourceParser
from fhir_healthcare_ai.fhir.parsers.primitives import (
    first_display,
    get_dict,
    get_list,
    get_str,
    parse_codeable_concept,
    parse_coding,
    parse_datetime,
    parse_datetime_value,
    parse_period,
    parse_reference_relative,
)
from fhir_healthcare_ai.terminology import resolve_codings


class DiagnosticReportParser(ResourceParser[NormalizedDiagnosticReport]):
    """Parses grouped reports.

    ``result`` entries are kept as references rather than resolved here: whether to
    fetch them is a retrieval decision, and a parser that issues network calls is a
    parser you cannot unit-test.
    """

    resource_type = ResourceType.DIAGNOSTIC_REPORT

    def _parse(self, resource: dict[str, Any], resource_id: str) -> NormalizedDiagnosticReport:
        codings, text = parse_codeable_concept(get_dict(resource, "code"))
        concept = resolve_codings(codings)

        effective = parse_datetime(resource.get("effectiveDateTime"))
        if effective is None:
            start, _ = parse_period(get_dict(resource, "effectivePeriod"))
            if start is not None:
                effective = parse_datetime(start.isoformat())

        conclusion_codings = [
            coding
            for entry in get_list(resource, "conclusionCode")
            for coding in parse_codeable_concept(entry)[0]
        ]

        return NormalizedDiagnosticReport(
            id=resource_id,
            patient_id=self.subject_id(resource),
            encounter_id=self.encounter_id(resource),
            source_codings=codings,
            concept=concept.key if concept else None,
            display=first_display(codings, text),
            status=get_str(resource, "status"),
            category=[
                c
                for c in (
                    parse_coding(item)
                    for entry in get_list(resource, "category")
                    for item in get_list(entry, "coding")
                )
                if c
            ],
            effective=effective,
            issued=parse_datetime_value(resource.get("issued")),
            result_references=[
                ref
                for ref in (parse_reference_relative(item) for item in get_list(resource, "result"))
                if ref
            ],
            conclusion=get_str(resource, "conclusion"),
            conclusion_codings=conclusion_codings,
        )
