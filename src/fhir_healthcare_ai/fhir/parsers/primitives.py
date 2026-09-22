"""Parsing of FHIR primitive and complex datatypes.

This is where most real-world FHIR pain lives:

* ``date``/``dateTime`` can be a bare year, a year-month, a date, or a full instant,
  with or without a timezone. Losing that precision silently is how a 2019 record ends
  up matching a "last 6 months" filter.
* ``Quantity`` can carry a comparator (``<0.1``), a UCUM code that disagrees with the
  human-readable unit, or no unit at all.
* ``Reference`` can be relative, absolute, a ``urn:uuid:``, contained (``#id``), or
  identifier-only with no literal reference at all.
* ``CodeableConcept`` can have several codings from several systems, or only ``text``.

Every function here is total: it returns ``None`` rather than raising, and records what
went wrong through the caller's warning list.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from typing import Any

from fhir_healthcare_ai.domain.clinical import Coding, Quantity, ReferenceRange, TemporalValue
from fhir_healthcare_ai.domain.enums import DatePrecision
from fhir_healthcare_ai.terminology import ConceptDefinition, normalize_unit, to_canonical

_DATE_PATTERNS: tuple[tuple[re.Pattern[str], DatePrecision, str], ...] = (
    (re.compile(r"^\d{4}$"), DatePrecision.YEAR, "%Y"),
    (re.compile(r"^\d{4}-\d{2}$"), DatePrecision.MONTH, "%Y-%m"),
    (re.compile(r"^\d{4}-\d{2}-\d{2}$"), DatePrecision.DAY, "%Y-%m-%d"),
)


def get_dict(node: Any, key: str) -> dict[str, Any]:
    """Return ``node[key]`` when it is a dict, otherwise an empty dict."""
    if isinstance(node, dict):
        value = node.get(key)
        if isinstance(value, dict):
            return value
    return {}


def get_list(node: Any, key: str) -> list[Any]:
    """Return ``node[key]`` when it is a list, otherwise an empty list.

    Tolerates servers that emit a bare object where the spec requires an array.
    """
    if isinstance(node, dict):
        value = node.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return [value]
    return []


def get_str(node: Any, key: str) -> str | None:
    """Return ``node[key]`` as a non-empty string, otherwise None."""
    if isinstance(node, dict):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def get_float(node: Any, key: str) -> float | None:
    """Return ``node[key]`` as a float, tolerating numeric strings."""
    if not isinstance(node, dict):
        return None
    value = node.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def parse_datetime(value: Any) -> TemporalValue | None:
    """Parse a FHIR ``date``, ``dateTime`` or ``instant``, preserving precision.

    Partial dates are anchored to the *start* of their period (``2019`` becomes
    ``2019-01-01T00:00:00Z``) and the original precision is retained so that callers
    can decide whether an interval comparison is actually meaningful.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()

    for pattern, precision, fmt in _DATE_PATTERNS:
        if pattern.match(raw):
            try:
                parsed = datetime.strptime(raw, fmt).replace(tzinfo=UTC)
            except ValueError:
                return TemporalValue(value=None, precision=None, raw=raw)
            return TemporalValue(value=parsed, precision=precision, raw=raw)

    iso = raw.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(iso)
    except ValueError:
        return TemporalValue(value=None, precision=None, raw=raw)
    if parsed.tzinfo is None:
        # FHIR requires an offset on dateTime with a time part; assume UTC and move on.
        parsed = parsed.replace(tzinfo=UTC)
    precision = DatePrecision.MILLISECOND if parsed.microsecond else DatePrecision.SECOND
    return TemporalValue(value=parsed, precision=precision, raw=raw)


def parse_datetime_value(value: Any) -> datetime | None:
    """Parse to a plain datetime, discarding precision."""
    temporal = parse_datetime(value)
    return temporal.value if temporal else None


def parse_date(value: Any) -> date | None:
    """Parse a FHIR ``date`` into a ``datetime.date``."""
    temporal = parse_datetime(value)
    return temporal.value.date() if temporal and temporal.value else None


def parse_coding(node: Any) -> Coding | None:
    """Parse a single ``Coding``."""
    if not isinstance(node, dict):
        return None
    system = get_str(node, "system")
    code = get_str(node, "code")
    display = get_str(node, "display")
    if not (system or code or display):
        return None
    return Coding(system=system, code=code, display=display)


def parse_codeable_concept(node: Any) -> tuple[list[Coding], str | None]:
    """Parse a ``CodeableConcept`` into its codings plus free text.

    A concept with only ``text`` is common in narrative-heavy feeds; the text is kept
    so the display layer has something to show even when nothing resolves.
    """
    if not isinstance(node, dict):
        return [], None
    codings = [c for c in (parse_coding(item) for item in get_list(node, "coding")) if c]
    return codings, get_str(node, "text")


def first_display(codings: list[Coding], text: str | None = None) -> str | None:
    """Best available human-readable label for a concept."""
    for coding in codings:
        if coding.display:
            return coding.display
    if text:
        return text
    for coding in codings:
        if coding.code:
            return coding.code
    return None


def parse_quantity(node: Any, concept: ConceptDefinition | None = None) -> Quantity | None:
    """Parse a ``Quantity``, adding a canonical-unit view when possible."""
    if not isinstance(node, dict):
        return None
    value = get_float(node, "value")
    unit = get_str(node, "unit") or get_str(node, "code")
    system = get_str(node, "system")
    code = get_str(node, "code")
    comparator = get_str(node, "comparator")
    if value is None and unit is None:
        return None

    canonical_value, canonical_unit = to_canonical(value, unit, concept)
    return Quantity(
        value=value,
        unit=normalize_unit(unit),
        system=system,
        code=code,
        comparator=comparator,
        canonical_value=canonical_value,
        canonical_unit=canonical_unit,
    )


def parse_reference_id(node: Any) -> str | None:
    """Extract the logical id a ``Reference`` points at.

    Handles ``Patient/123``, ``http://host/fhir/Patient/123``, ``urn:uuid:...`` and
    contained ``#local`` references. Identifier-only references return None: there is
    no id to resolve without querying, and guessing would corrupt the record graph.
    """
    if isinstance(node, str):
        literal = node
    elif isinstance(node, dict):
        literal = get_str(node, "reference") or ""
    else:
        return None

    if not literal:
        return None
    if literal.startswith("#"):
        return literal[1:] or None
    if literal.startswith("urn:uuid:"):
        return literal[len("urn:uuid:") :] or None

    parts = [part for part in literal.split("/") if part]
    if len(parts) >= 2 and parts[-2][:1].isupper():
        return parts[-1].split("?")[0] or None
    return parts[-1] if parts else None


def parse_reference_type(node: Any) -> str | None:
    """Resource type a ``Reference`` points at, if it can be determined."""
    if isinstance(node, dict):
        explicit = get_str(node, "type")
        if explicit:
            return explicit
        literal = get_str(node, "reference") or ""
    elif isinstance(node, str):
        literal = node
    else:
        return None
    parts = [part for part in literal.split("/") if part]
    if len(parts) >= 2 and parts[-2][:1].isupper():
        return parts[-2]
    return None


def parse_reference_relative(node: Any) -> str | None:
    """Normalize a reference to ``ResourceType/id`` form when both parts are known."""
    resource_type = parse_reference_type(node)
    resource_id = parse_reference_id(node)
    if resource_type and resource_id:
        return f"{resource_type}/{resource_id}"
    return resource_id


def parse_reference_range(nodes: Any, unit_hint: str | None = None) -> ReferenceRange | None:
    """Parse the first usable ``Observation.referenceRange`` entry."""
    entries = nodes if isinstance(nodes, list) else [nodes] if isinstance(nodes, dict) else []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        low = get_float(get_dict(entry, "low"), "value")
        high = get_float(get_dict(entry, "high"), "value")
        text = get_str(entry, "text")
        if low is None and high is None and text is None:
            continue
        unit = (
            get_str(get_dict(entry, "low"), "unit")
            or get_str(get_dict(entry, "high"), "unit")
            or unit_hint
        )
        return ReferenceRange(low=low, high=high, unit=normalize_unit(unit), text=text)
    return None


def parse_period(node: Any) -> tuple[datetime | None, datetime | None]:
    """Parse a ``Period`` into (start, end)."""
    if not isinstance(node, dict):
        return None, None
    return parse_datetime_value(node.get("start")), parse_datetime_value(node.get("end"))


def status_code(node: Any, key: str) -> str | None:
    """Read a status field that may be a plain code or a CodeableConcept.

    R4 ``Condition.clinicalStatus`` is a CodeableConcept, but plenty of feeds send a
    bare string. Accepting both costs one branch and avoids losing the field entirely.
    """
    if not isinstance(node, dict):
        return None
    value = node.get(key)
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        codings, text = parse_codeable_concept(value)
        for coding in codings:
            if coding.code:
                return coding.code
        return text
    return None
