"""Patient parser."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

from fhir_healthcare_ai.domain.clinical import NormalizedPatient
from fhir_healthcare_ai.domain.enums import ResourceType
from fhir_healthcare_ai.fhir.parsers.base import ResourceParser
from fhir_healthcare_ai.fhir.parsers.primitives import (
    get_list,
    get_str,
    parse_date,
    parse_datetime_value,
    parse_reference_id,
)


def age_in_years(birth_date: date | None, as_of: date | None = None) -> float | None:
    """Age in fractional years. Returns None when the birth date is unknown."""
    if birth_date is None:
        return None
    reference = as_of or datetime.now(UTC).date()
    days = (reference - birth_date).days
    if days < 0:
        return None
    return round(days / 365.25, 2)


class PatientParser(ResourceParser[NormalizedPatient]):
    """Parses demographics, tolerating the many ways ``deceased[x]`` is expressed."""

    resource_type = ResourceType.PATIENT

    def _parse(self, resource: dict[str, Any], resource_id: str) -> NormalizedPatient:
        warnings: list[str] = []

        birth_date = parse_date(resource.get("birthDate"))
        if birth_date is None and resource.get("birthDate"):
            warnings.append(f"unparseable birthDate: {resource.get('birthDate')!r}")

        deceased_date = parse_datetime_value(resource.get("deceasedDateTime"))
        deceased_flag = resource.get("deceasedBoolean")
        deceased = bool(deceased_flag) or deceased_date is not None

        gender = get_str(resource, "gender")
        if gender:
            gender = gender.lower()

        postal_code = None
        for address in get_list(resource, "address"):
            postal_code = get_str(address, "postalCode")
            if postal_code:
                break

        # An age computed against "today" is wrong for a deceased patient.
        as_of = deceased_date.date() if deceased_date else None

        return NormalizedPatient(
            id=resource_id,
            patient_id=resource_id,
            gender=gender,
            birth_date=birth_date,
            age_years=age_in_years(birth_date, as_of),
            deceased=deceased,
            deceased_date=deceased_date,
            postal_code=postal_code,
            managing_organization=parse_reference_id(resource.get("managingOrganization")),
            display=_display_name(resource),
            parse_warnings=warnings,
        )


def _display_name(resource: dict[str, Any]) -> str | None:
    """Human name, preferring ``official`` use.

    Synthetic data only. Real deployments should strip this before it reaches a model
    or a log line -- see ``docs/security-privacy.md``.
    """
    names = get_list(resource, "name")
    chosen = next((n for n in names if isinstance(n, dict) and n.get("use") == "official"), None)
    chosen = chosen or next((n for n in names if isinstance(n, dict)), None)
    if not chosen:
        return None
    text = get_str(chosen, "text")
    if text:
        return text
    given = " ".join(str(g) for g in get_list(chosen, "given") if isinstance(g, str))
    family = get_str(chosen, "family") or ""
    full = f"{given} {family}".strip()
    return full or None
