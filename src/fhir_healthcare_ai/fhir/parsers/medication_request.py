"""MedicationRequest parser."""

from __future__ import annotations

from typing import Any

from fhir_healthcare_ai.domain.clinical import NormalizedMedicationRequest
from fhir_healthcare_ai.domain.enums import ResourceType
from fhir_healthcare_ai.fhir.parsers.base import ResourceParser
from fhir_healthcare_ai.fhir.parsers.primitives import (
    first_display,
    get_dict,
    get_list,
    get_str,
    parse_codeable_concept,
    parse_datetime_value,
    parse_period,
    parse_quantity,
    parse_reference_relative,
)
from fhir_healthcare_ai.terminology import resolve_codings


class MedicationRequestParser(ResourceParser[NormalizedMedicationRequest]):
    """Parses medication orders.

    ``medication[x]`` may be an inline CodeableConcept or a Reference to a Medication
    resource. When it is a reference, the drug is unknown until that resource is also
    fetched; the reference is preserved so the caller can decide whether to resolve it,
    rather than the record silently claiming "no medication".
    """

    resource_type = ResourceType.MEDICATION_REQUEST

    def _parse(self, resource: dict[str, Any], resource_id: str) -> NormalizedMedicationRequest:
        warnings: list[str] = []

        codings, text = parse_codeable_concept(get_dict(resource, "medicationCodeableConcept"))
        medication_reference = parse_reference_relative(resource.get("medicationReference"))
        if not codings and medication_reference:
            warnings.append(
                "medication given as a reference; resolve the Medication resource to code it"
            )
        concept = resolve_codings(codings)
        if concept is None and codings:
            warnings.append(f"unmapped medication code: {codings[0].token}")

        dosage = next(
            (d for d in get_list(resource, "dosageInstruction") if isinstance(d, dict)), {}
        )
        dose_and_rate = next(
            (d for d in get_list(dosage, "doseAndRate") if isinstance(d, dict)), {}
        )
        validity_start, validity_end = parse_period(
            get_dict(get_dict(resource, "dispenseRequest"), "validityPeriod")
        )

        reason_codings = [
            coding
            for entry in get_list(resource, "reasonCode")
            for coding in parse_codeable_concept(entry)[0]
        ]

        return NormalizedMedicationRequest(
            id=resource_id,
            patient_id=self.subject_id(resource),
            encounter_id=self.encounter_id(resource),
            source_codings=codings,
            concept=concept.key if concept else None,
            display=first_display(codings, text),
            status=get_str(resource, "status"),
            intent=get_str(resource, "intent"),
            authored_on=parse_datetime_value(resource.get("authoredOn")),
            medication_reference=medication_reference,
            drug_class=concept.drug_class if concept else None,
            dosage_text=get_str(dosage, "text"),
            dose_quantity=parse_quantity(get_dict(dose_and_rate, "doseQuantity")),
            validity_start=validity_start,
            validity_end=validity_end,
            reason_codings=reason_codings,
            parse_warnings=warnings,
        )
