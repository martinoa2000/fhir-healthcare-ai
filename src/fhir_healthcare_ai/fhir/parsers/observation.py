"""Observation parser, including components and abnormality flagging."""

from __future__ import annotations

from typing import Any

from fhir_healthcare_ai.domain.clinical import (
    Coding,
    NormalizedObservation,
    Quantity,
    ReferenceRange,
)
from fhir_healthcare_ai.domain.enums import AbnormalFlag, ResourceType
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
    parse_quantity,
    parse_reference_range,
    parse_reference_relative,
)
from fhir_healthcare_ai.terminology import ConceptDefinition, resolve_codings

# v3-ObservationInterpretation codes we act on. Anything else is treated as unknown
# rather than guessed at.
_INTERPRETATION_MAP: dict[str, AbnormalFlag] = {
    "H": AbnormalFlag.HIGH,
    "HH": AbnormalFlag.CRITICAL_HIGH,
    "HU": AbnormalFlag.CRITICAL_HIGH,
    "L": AbnormalFlag.LOW,
    "LL": AbnormalFlag.CRITICAL_LOW,
    "LU": AbnormalFlag.CRITICAL_LOW,
    "N": AbnormalFlag.NORMAL,
    "A": AbnormalFlag.UNKNOWN,  # "abnormal" without direction
}


class ObservationParser(ResourceParser[NormalizedObservation]):
    """Parses labs and vitals.

    Components are parsed recursively so a blood-pressure panel yields usable
    systolic/diastolic records instead of a parent with no value.
    """

    resource_type = ResourceType.OBSERVATION

    def _parse(self, resource: dict[str, Any], resource_id: str) -> NormalizedObservation:
        warnings: list[str] = []
        codings, text = parse_codeable_concept(get_dict(resource, "code"))
        concept = resolve_codings(codings)
        if concept is None and codings:
            warnings.append(f"unmapped code: {codings[0].token}")

        effective = parse_datetime(resource.get("effectiveDateTime"))
        if effective is None:
            start, _ = parse_period(get_dict(resource, "effectivePeriod"))
            if start is not None:
                effective = parse_datetime(start.isoformat())
        if effective is None and resource.get("issued"):
            # Fall back to release time, but record that it is not the clinical time.
            effective = parse_datetime(resource.get("issued"))
            if effective is not None:
                warnings.append("effective[x] missing; using issued as the clinical time")

        quantity, value_string, value_boolean, value_codeable = _parse_value(resource, concept)
        if quantity is not None and quantity.canonical_value is None and quantity.value is not None:
            warnings.append(
                f"could not convert {quantity.unit!r} to the canonical unit for "
                f"{concept.key if concept else 'unmapped concept'}"
            )

        interpretation = [
            c
            for c in (
                parse_coding(item)
                for entry in get_list(resource, "interpretation")
                for item in get_list(entry, "coding")
            )
            if c
        ]
        reference_range = parse_reference_range(
            resource.get("referenceRange"), quantity.unit if quantity else None
        )

        observation = NormalizedObservation(
            id=resource_id,
            patient_id=self.subject_id(resource),
            encounter_id=self.encounter_id(resource),
            status=get_str(resource, "status"),
            source_codings=codings,
            concept=concept.key if concept else None,
            display=first_display(codings, text),
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
            quantity=quantity,
            value_string=value_string,
            value_boolean=value_boolean,
            value_codeable=value_codeable,
            interpretation=interpretation,
            reference_range=reference_range,
            components=self._parse_components(resource, resource_id),
            derived_from=[
                ref
                for ref in (
                    parse_reference_relative(item) for item in get_list(resource, "derivedFrom")
                )
                if ref
            ],
            parse_warnings=warnings,
        )
        return observation.model_copy(update={"abnormal_flag": classify(observation, concept)})

    def _parse_components(
        self, resource: dict[str, Any], parent_id: str
    ) -> list[NormalizedObservation]:
        components: list[NormalizedObservation] = []
        for index, node in enumerate(get_list(resource, "component")):
            if not isinstance(node, dict):
                continue
            codings, text = parse_codeable_concept(get_dict(node, "code"))
            concept = resolve_codings(codings)
            quantity, value_string, value_boolean, value_codeable = _parse_value(node, concept)
            component = NormalizedObservation(
                id=f"{parent_id}#component-{index}",
                patient_id=self.subject_id(resource),
                encounter_id=self.encounter_id(resource),
                status=get_str(resource, "status"),
                source_codings=codings,
                concept=concept.key if concept else None,
                display=first_display(codings, text),
                effective=parse_datetime(resource.get("effectiveDateTime")),
                quantity=quantity,
                value_string=value_string,
                value_boolean=value_boolean,
                value_codeable=value_codeable,
                reference_range=parse_reference_range(
                    node.get("referenceRange"), quantity.unit if quantity else None
                ),
            )
            components.append(
                component.model_copy(update={"abnormal_flag": classify(component, concept)})
            )
        return components


def _parse_value(
    node: dict[str, Any], concept: ConceptDefinition | None
) -> tuple[Quantity | None, str | None, bool | None, Coding | None]:
    """Resolve the ``value[x]`` choice type."""
    quantity = parse_quantity(get_dict(node, "valueQuantity"), concept)
    value_string = get_str(node, "valueString")
    raw_boolean = node.get("valueBoolean")
    value_boolean = raw_boolean if isinstance(raw_boolean, bool) else None

    value_codeable = None
    codeable = get_dict(node, "valueCodeableConcept")
    if codeable:
        codings, text = parse_codeable_concept(codeable)
        value_codeable = codings[0] if codings else (Coding(display=text) if text else None)

    if quantity is None:
        integer_value = node.get("valueInteger")
        if isinstance(integer_value, int) and not isinstance(integer_value, bool):
            quantity = Quantity(value=float(integer_value))

    return quantity, value_string, value_boolean, value_codeable


def classify(
    observation: NormalizedObservation,
    concept: ConceptDefinition | None = None,
    sex: str | None = None,
) -> AbnormalFlag:
    """Decide whether a result is out of range.

    Precedence is deliberate: the laboratory's own interpretation wins, then the
    reference range it supplied, then our curated interval. Overriding a lab's
    interpretation with a generic textbook range would be wrong.
    """
    for coding in observation.interpretation:
        if coding.code and coding.code.upper() in _INTERPRETATION_MAP:
            flag = _INTERPRETATION_MAP[coding.code.upper()]
            if flag is not AbnormalFlag.UNKNOWN:
                return flag

    value = observation.numeric_value
    if value is None:
        return AbnormalFlag.UNKNOWN

    if observation.reference_range and _has_bounds(observation.reference_range):
        return _flag_against(
            value, observation.reference_range.low, observation.reference_range.high
        )

    if concept is not None:
        interval = concept.interval_for(sex)
        if interval is not None:
            flag = _flag_against(value, interval.low, interval.high)
            if (
                flag is AbnormalFlag.HIGH
                and interval.critical_high is not None
                and value >= interval.critical_high
            ):
                return AbnormalFlag.CRITICAL_HIGH
            if (
                flag is AbnormalFlag.LOW
                and interval.critical_low is not None
                and value <= interval.critical_low
            ):
                return AbnormalFlag.CRITICAL_LOW
            return flag

    return AbnormalFlag.UNKNOWN


def _has_bounds(reference_range: ReferenceRange) -> bool:
    return reference_range.low is not None or reference_range.high is not None


def _flag_against(value: float, low: float | None, high: float | None) -> AbnormalFlag:
    if high is not None and value > high:
        return AbnormalFlag.HIGH
    if low is not None and value < low:
        return AbnormalFlag.LOW
    return AbnormalFlag.NORMAL
