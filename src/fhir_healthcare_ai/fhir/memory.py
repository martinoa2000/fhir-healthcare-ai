"""An in-process FHIR R4 server, just large enough for this project's searches.

It exists for three callers that must not depend on a running HAPI container:

* the test suite, which drives the real :class:`~fhir_healthcare_ai.fhir.client.FHIRClient`
  through it, so pagination, retries and audit events are exercised over actual HTTP
  semantics rather than mocked method calls;
* the benchmark, which needs a reproducible server loaded with a known synthetic
  population to score plans against ground truth;
* ``FHIR_IN_MEMORY=true``, a demo mode that serves a generated population without
  Docker or Java.

It plugs in underneath httpx as a transport, so nothing above the client can tell it
apart from a remote server. It implements the subset of FHIR search that the allowlist
can express -- no more. A parameter it does not know is answered with ``400`` and an
OperationOutcome, the same way HAPI answers an unknown parameter, so a plan that would
fail against a real server also fails here instead of silently matching everything.

This is a test double, not a FHIR server. Do not point anything real at it.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse

import httpx

from fhir_healthcare_ai.config import FHIRSettings
from fhir_healthcare_ai.fhir.client import FHIRClient

DEFAULT_BASE_URL = "http://fhir.in-memory/fhir"
DEFAULT_PAGE_SIZE = 20

Token = tuple[str | None, str, str | None]  # (system, code, display/text)
DateRange = tuple[datetime, datetime]
Quantity = tuple[float, str | None, str | None]  # (value, system, code-or-unit)


class SearchError(ValueError):
    """The search cannot be executed; answered with 400 and an OperationOutcome."""


# -- value extraction -----------------------------------------------------------------


def _codeable_tokens(node: Any) -> list[Token]:
    """Tokens from a CodeableConcept, a Coding, or a list of either."""
    if isinstance(node, list):
        return [token for item in node for token in _codeable_tokens(item)]
    if not isinstance(node, dict):
        return []
    tokens: list[Token] = []
    text = node.get("text")
    for coding in node.get("coding") or []:
        if isinstance(coding, dict) and isinstance(coding.get("code"), str):
            tokens.append((coding.get("system"), coding["code"], coding.get("display") or text))
    if "code" in node and isinstance(node.get("code"), str) and "coding" not in node:
        tokens.append((node.get("system"), node["code"], node.get("display")))
    if not tokens and isinstance(text, str):
        tokens.append((None, "", text))
    return tokens


def _code_tokens(value: Any) -> list[Token]:
    """Tokens from a primitive ``code`` or ``boolean`` element."""
    if isinstance(value, bool):
        return [(None, "true" if value else "false", None)]
    if isinstance(value, str):
        return [(None, value, None)]
    return []


def _identifier_tokens(node: Any) -> list[Token]:
    items = node if isinstance(node, list) else [node]
    return [
        (item.get("system"), item["value"], None)
        for item in items
        if isinstance(item, dict) and isinstance(item.get("value"), str)
    ]


def _references(*nodes: Any) -> list[str]:
    refs: list[str] = []
    for node in nodes:
        items = node if isinstance(node, list) else [node]
        for item in items:
            if isinstance(item, dict) and isinstance(item.get("reference"), str):
                refs.append(item["reference"])
    return refs


def parse_fhir_date(value: str) -> DateRange:
    """The instant range a FHIR date, dateTime or instant denotes.

    ``2024`` covers the whole year, ``2024-03`` the month, a full timestamp the second.
    Searching against ranges rather than points is what makes ``date=ge2024-01-01``
    behave correctly for a resource recorded only as ``2024``.
    """
    text = value.strip()
    try:
        if len(text) == 4:
            start = datetime(int(text), 1, 1, tzinfo=UTC)
            return start, start.replace(year=start.year + 1) - timedelta(microseconds=1)
        if len(text) == 7:
            year, month = int(text[:4]), int(text[5:7])
            start = datetime(year, month, 1, tzinfo=UTC)
            end = datetime(year + month // 12, month % 12 + 1, 1, tzinfo=UTC)
            return start, end - timedelta(microseconds=1)
        if len(text) == 10:
            start = datetime.fromisoformat(text).replace(tzinfo=UTC)
            return start, start + timedelta(days=1) - timedelta(microseconds=1)
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SearchError(f"invalid date value {value!r}") from exc
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    if moment.microsecond:
        return moment, moment
    return moment, moment + timedelta(seconds=1) - timedelta(microseconds=1)


def _date_ranges(*values: Any) -> list[DateRange]:
    ranges: list[DateRange] = []
    for value in values:
        if isinstance(value, str):
            try:
                ranges.append(parse_fhir_date(value))
            except SearchError:
                continue
        elif isinstance(value, dict):  # Period
            start, end = value.get("start"), value.get("end")
            if not isinstance(start, str) and not isinstance(end, str):
                continue
            low = parse_fhir_date(start)[0] if isinstance(start, str) else datetime.min
            high = parse_fhir_date(end)[1] if isinstance(end, str) else datetime.max
            ranges.append((low.replace(tzinfo=UTC), high.replace(tzinfo=UTC)))
    return ranges


def _quantity(node: Any) -> list[Quantity]:
    if not isinstance(node, dict) or not isinstance(node.get("value"), int | float):
        return []
    return [(float(node["value"]), node.get("system"), node.get("code") or node.get("unit"))]


def _components(resource: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in resource.get("component") or [] if isinstance(c, dict)]


@dataclass(frozen=True)
class _Param:
    kind: str  # token | date | quantity | reference | string
    extract: Callable[[dict[str, Any]], list[Any]]
    #: Resource type a bare reference id (``encounter=abc``) is resolved against.
    target: str = "Patient"


def _subject(resource: dict[str, Any]) -> list[str]:
    return _references(resource.get("subject"), resource.get("patient"))


_COMMON: dict[str, _Param] = {
    "_id": _Param("token", lambda r: _code_tokens(r.get("id"))),
    "patient": _Param("reference", _subject),
    "subject": _Param("reference", _subject),
    "encounter": _Param("reference", lambda r: _references(r.get("encounter")), "Encounter"),
    "status": _Param("token", lambda r: _code_tokens(r.get("status"))),
    "category": _Param("token", lambda r: _codeable_tokens(r.get("category"))),
}

_PARAMS: dict[str, dict[str, _Param]] = {
    "Patient": {
        "_id": _COMMON["_id"],
        "active": _Param("token", lambda r: _code_tokens(r.get("active"))),
        "gender": _Param("token", lambda r: _code_tokens(r.get("gender"))),
        "birthdate": _Param("date", lambda r: _date_ranges(r.get("birthDate"))),
        "deceased": _Param(
            "token",
            lambda r: _code_tokens(bool(r.get("deceasedBoolean") or r.get("deceasedDateTime"))),
        ),
        "identifier": _Param("token", lambda r: _identifier_tokens(r.get("identifier"))),
        "address-postalcode": _Param(
            "string",
            lambda r: [
                a["postalCode"]
                for a in r.get("address") or []
                if isinstance(a, dict) and isinstance(a.get("postalCode"), str)
            ],
        ),
    },
    "Observation": {
        **_COMMON,
        "code": _Param("token", lambda r: _codeable_tokens(r.get("code"))),
        "component-code": _Param(
            "token", lambda r: [t for c in _components(r) for t in _codeable_tokens(c.get("code"))]
        ),
        "combo-code": _Param(
            "token",
            lambda r: (
                _codeable_tokens(r.get("code"))
                + [t for c in _components(r) for t in _codeable_tokens(c.get("code"))]
            ),
        ),
        "value-quantity": _Param("quantity", lambda r: _quantity(r.get("valueQuantity"))),
        "component-value-quantity": _Param(
            "quantity",
            lambda r: [q for c in _components(r) for q in _quantity(c.get("valueQuantity"))],
        ),
        "combo-value-quantity": _Param(
            "quantity",
            lambda r: (
                _quantity(r.get("valueQuantity"))
                + [q for c in _components(r) for q in _quantity(c.get("valueQuantity"))]
            ),
        ),
        "date": _Param(
            "date", lambda r: _date_ranges(r.get("effectiveDateTime"), r.get("effectivePeriod"))
        ),
    },
    "Condition": {
        **_COMMON,
        "code": _Param("token", lambda r: _codeable_tokens(r.get("code"))),
        "clinical-status": _Param("token", lambda r: _codeable_tokens(r.get("clinicalStatus"))),
        "verification-status": _Param(
            "token", lambda r: _codeable_tokens(r.get("verificationStatus"))
        ),
        "onset-date": _Param(
            "date", lambda r: _date_ranges(r.get("onsetDateTime"), r.get("onsetPeriod"))
        ),
        "recorded-date": _Param("date", lambda r: _date_ranges(r.get("recordedDate"))),
    },
    "MedicationRequest": {
        **_COMMON,
        "code": _Param("token", lambda r: _codeable_tokens(r.get("medicationCodeableConcept"))),
        "intent": _Param("token", lambda r: _code_tokens(r.get("intent"))),
        "authoredon": _Param("date", lambda r: _date_ranges(r.get("authoredOn"))),
    },
    "Encounter": {
        **_COMMON,
        "class": _Param("token", lambda r: _codeable_tokens(r.get("class"))),
        "type": _Param("token", lambda r: _codeable_tokens(r.get("type"))),
        "reason-code": _Param("token", lambda r: _codeable_tokens(r.get("reasonCode"))),
        "date": _Param("date", lambda r: _date_ranges(r.get("period"))),
    },
    "DiagnosticReport": {
        **_COMMON,
        "code": _Param("token", lambda r: _codeable_tokens(r.get("code"))),
        "date": _Param(
            "date", lambda r: _date_ranges(r.get("effectiveDateTime"), r.get("effectivePeriod"))
        ),
        "issued": _Param("date", lambda r: _date_ranges(r.get("issued"))),
    },
}

#: ``_sort`` key -> the search parameter whose (first) value orders the results.
_SORT_KEYS: dict[str, str] = {
    "date": "date",
    "authoredon": "authoredon",
    "onset-date": "onset-date",
    "recorded-date": "recorded-date",
    "issued": "issued",
    "birthdate": "birthdate",
}

_PREFIXES = ("eq", "ne", "gt", "lt", "ge", "le", "sa", "eb", "ap")


# -- matching -------------------------------------------------------------------------


def _split_prefix(raw: str) -> tuple[str, str]:
    if len(raw) > 2 and raw[:2] in _PREFIXES and not raw[2:3].isalpha():
        return raw[:2], raw[2:]
    return "eq", raw


def _match_token(candidates: list[Token], raw: str, modifier: str | None) -> bool:
    if modifier == "text":
        needle = raw.lower()
        return any(display and needle in display.lower() for _, _, display in candidates)
    if "|" in raw:
        system, code = raw.split("|", 1)
        if system == "":
            return any(s is None and c == code for s, c, _ in candidates)
        return any(s == system and c == code for s, c, _ in candidates)
    return any(c == raw for _, c, _ in candidates)


def _match_date(candidates: list[DateRange], raw: str) -> bool:
    prefix, value = _split_prefix(raw)
    low, high = parse_fhir_date(value)
    for start, end in candidates:
        if prefix == "eq" and start >= low and end <= high:
            return True
        if prefix == "ne" and not (start >= low and end <= high):
            return True
        if prefix in ("gt", "sa") and end > high:
            return True
        if prefix == "ge" and end >= low:
            return True
        if prefix in ("lt", "eb") and start < low:
            return True
        if prefix == "le" and start <= high:
            return True
        if prefix == "ap" and abs((start - low).days) <= 10:
            return True
    return False


def _match_quantity(candidates: list[Quantity], raw: str) -> bool:
    parts = raw.split("|")
    prefix, number = _split_prefix(parts[0])
    try:
        target = float(number)
    except ValueError as exc:
        raise SearchError(f"invalid quantity value {raw!r}") from exc
    system = parts[1] if len(parts) > 1 and parts[1] else None
    unit = parts[2] if len(parts) > 2 and parts[2] else None
    for value, value_system, value_unit in candidates:
        if system and value_system and system != value_system:
            continue
        if unit and value_unit and unit != value_unit:
            continue
        if _compare(prefix, value, target):
            return True
    return False


def _compare(prefix: str, value: float, target: float) -> bool:
    if prefix == "eq":
        return abs(value - target) < 1e-9
    if prefix == "ne":
        return abs(value - target) >= 1e-9
    if prefix in ("gt", "sa"):
        return value > target
    if prefix in ("lt", "eb"):
        return value < target
    if prefix == "ge":
        return value >= target
    if prefix == "le":
        return value <= target
    return abs(value - target) <= abs(target) * 0.1  # ap: within 10%


def _match_reference(candidates: list[str], raw: str, default_type: str) -> bool:
    wanted = raw if "/" in raw else f"{default_type}/{raw}"
    return any(ref == wanted or ref.endswith("/" + wanted) for ref in candidates)


# -- the server -----------------------------------------------------------------------


class InMemoryFHIRServer:
    """A dictionary of resources that answers FHIR REST calls.

    Args:
        base_url: The URL clients are configured with. Pagination links are absolute
            and point back here, which also exercises the client's same-origin check.
        resources: Initial contents.
        read_only: Refuse transaction bundles, as the application's own server should.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        resources: Iterable[dict[str, Any]] = (),
        *,
        read_only: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.read_only = read_only
        self._store: dict[str, dict[str, dict[str, Any]]] = {}
        self._lock = threading.Lock()
        #: Every request received, as ``"GET Observation?code=..."``. Tests assert on it.
        self.requests: list[str] = []
        #: Status codes to return, in order, before serving normally. For retry tests.
        self.fail_next: list[int] = []
        self.load(resources)

    # -- contents -------------------------------------------------------------------

    def load(self, resources: Iterable[dict[str, Any]]) -> int:
        """Insert or replace resources by ``resourceType``/``id``. Returns how many."""
        count = 0
        with self._lock:
            for resource in resources:
                self._put(resource)
                count += 1
        return count

    def _put(self, resource: dict[str, Any]) -> bool:
        resource_type, resource_id = resource.get("resourceType"), resource.get("id")
        if not isinstance(resource_type, str) or not isinstance(resource_id, str):
            raise SearchError("a stored resource needs a resourceType and an id")
        bucket = self._store.setdefault(resource_type, {})
        existed = resource_id in bucket
        bucket[resource_id] = resource
        return existed

    def resources(self, resource_type: str | None = None) -> Iterator[dict[str, Any]]:
        if resource_type is not None:
            yield from self._store.get(resource_type, {}).values()
            return
        for bucket in self._store.values():
            yield from bucket.values()

    def count(self, resource_type: str | None = None) -> int:
        return sum(1 for _ in self.resources(resource_type))

    # -- transport ------------------------------------------------------------------

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def client(self, settings: FHIRSettings | None = None, **kwargs: Any) -> FHIRClient:
        """A real :class:`FHIRClient` wired to this server."""
        resolved = settings or FHIRSettings(base_url=self.base_url, max_retries=0)
        http = httpx.AsyncClient(
            base_url=resolved.base_url,
            transport=self.transport(),
            headers={"Accept": "application/fhir+json"},
        )
        return FHIRClient(resolved, client=http, **kwargs)

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        base_path = urlparse(self.base_url).path.rstrip("/")
        relative = path[len(base_path) :].strip("/") if path.startswith(base_path) else None
        query = request.url.query.decode()
        self.requests.append(f"{request.method} {relative}{'?' + query if query else ''}")

        if self.fail_next:
            status = self.fail_next.pop(0)
            return _outcome(status, "transient", f"injected failure {status}")
        if relative is None:
            return _outcome(404, "not-found", f"unknown base path {path}")

        try:
            if request.method == "GET":
                return self._get(relative, parse_qsl(query, keep_blank_values=True))
            if request.method == "POST" and relative == "":
                return self._transaction(json.loads(request.content or b"{}"))
        except SearchError as exc:
            return _outcome(400, "invalid", str(exc))
        return _outcome(405, "not-supported", f"{request.method} {relative} is not supported")

    def _get(self, relative: str, params: list[tuple[str, str]]) -> httpx.Response:
        if relative == "metadata":
            return _json(200, self.capability_statement())
        parts = relative.split("/")
        if len(parts) == 2:
            resource = self._store.get(parts[0], {}).get(parts[1])
            if resource is None:
                return _outcome(404, "not-found", f"{relative} is not known")
            return _json(200, resource)
        if len(parts) == 1 and parts[0] in _PARAMS:
            return _json(200, self.search(parts[0], params))
        return _outcome(404, "not-found", f"unknown resource type {relative!r}")

    def _transaction(self, bundle: dict[str, Any]) -> httpx.Response:
        if self.read_only:
            return _outcome(403, "forbidden", "this server is read-only")
        if bundle.get("resourceType") != "Bundle":
            raise SearchError("expected a Bundle")
        entries = []
        with self._lock:
            for entry in bundle.get("entry") or []:
                resource = entry.get("resource") or {}
                method = (entry.get("request") or {}).get("method")
                if method != "PUT":
                    raise SearchError(f"unsupported transaction method {method!r}")
                existed = self._put(resource)
                status = "200 OK" if existed else "201 Created"
                entries.append({"response": {"status": status}})
        return _json(
            200, {"resourceType": "Bundle", "type": "transaction-response", "entry": entries}
        )

    # -- search ---------------------------------------------------------------------

    def search(self, resource_type: str, params: list[tuple[str, str]]) -> dict[str, Any]:
        """Execute a search and return one page of a searchset Bundle."""
        known = _PARAMS[resource_type]
        filters: list[tuple[_Param, str, str | None]] = []
        count = DEFAULT_PAGE_SIZE
        offset = 0
        sort: str | None = None
        includes: list[str] = []
        revincludes: list[str] = []

        for key, value in params:
            name, _, modifier = key.partition(":")
            if name == "_count":
                count = max(0, min(int(value), 1000))
            elif name == "_offset":
                offset = max(0, int(value))
            elif name == "_sort":
                sort = value
            elif name == "_include":
                includes.append(value)
            elif name == "_revinclude":
                revincludes.append(value)
            elif name in ("_summary", "_format"):
                continue
            elif name in known:
                if modifier and modifier not in ("not", "text"):
                    raise SearchError(f"modifier :{modifier} is not supported on {name}")
                filters.append((known[name], value, modifier or None))
            else:
                raise SearchError(f"unknown search parameter {name!r} for {resource_type}")

        matches = [r for r in self._store.get(resource_type, {}).values() if _matches(r, filters)]
        matches = self._sorted(resource_type, matches, sort)
        page = matches[offset : offset + count]

        entries: list[dict[str, Any]] = [
            {
                "fullUrl": f"{self.base_url}/{resource_type}/{r['id']}",
                "resource": r,
                "search": {"mode": "match"},
            }
            for r in page
        ]
        for extra in self._included(page, includes) + self._revincluded(
            resource_type, page, revincludes
        ):
            entries.append(
                {
                    "fullUrl": f"{self.base_url}/{extra['resourceType']}/{extra['id']}",
                    "resource": extra,
                    "search": {"mode": "include"},
                }
            )

        bundle: dict[str, Any] = {
            "resourceType": "Bundle",
            "type": "searchset",
            "total": len(matches),
            "link": [{"relation": "self", "url": self._page_url(resource_type, params, offset)}],
            "entry": entries,
        }
        if count and offset + count < len(matches):
            bundle["link"].append(
                {"relation": "next", "url": self._page_url(resource_type, params, offset + count)}
            )
        return bundle

    def _sorted(
        self, resource_type: str, resources: list[dict[str, Any]], sort: str | None
    ) -> list[dict[str, Any]]:
        ordered = sorted(resources, key=lambda r: str(r.get("id")))
        if not sort:
            return ordered
        descending = sort.startswith("-")
        key = sort.lstrip("-")
        if key == "_id":
            return list(reversed(ordered)) if descending else ordered
        param_name = _SORT_KEYS.get(key)
        param = _PARAMS[resource_type].get(param_name or "")
        if param is None or param.kind != "date":
            return ordered  # _lastUpdated and friends: stable id order is good enough here

        floor = datetime.min.replace(tzinfo=UTC)

        def sort_key(resource: dict[str, Any]) -> datetime:
            ranges = param.extract(resource)
            return ranges[0][0] if ranges else floor

        return sorted(ordered, key=sort_key, reverse=descending)

    def _included(self, page: list[dict[str, Any]], includes: list[str]) -> list[dict[str, Any]]:
        found: dict[str, dict[str, Any]] = {}
        for include in includes:
            _, _, field = include.partition(":")
            for resource in page:
                if field in ("patient", "subject"):
                    refs = _subject(resource)
                elif field == "result":
                    refs = _references(resource.get("result"))
                else:
                    refs = _references(resource.get(field))
                for ref in refs:
                    target_type, _, target_id = ref.partition("/")
                    target = self._store.get(target_type, {}).get(target_id)
                    if target is not None:
                        found[ref] = target
        return list(found.values())

    def _revincluded(
        self, resource_type: str, page: list[dict[str, Any]], revincludes: list[str]
    ) -> list[dict[str, Any]]:
        found: dict[str, dict[str, Any]] = {}
        wanted = {f"{resource_type}/{r['id']}" for r in page}
        for revinclude in revincludes:
            source_type, _, field = revinclude.partition(":")
            for resource in self._store.get(source_type, {}).values():
                refs = (
                    _subject(resource)
                    if field in ("patient", "subject")
                    else _references(resource.get(field))
                )
                if wanted.intersection(refs):
                    found[f"{source_type}/{resource['id']}"] = resource
        return list(found.values())

    def _page_url(self, resource_type: str, params: list[tuple[str, str]], offset: int) -> str:
        kept = [(k, v) for k, v in params if k != "_offset"]
        if offset:
            kept.append(("_offset", str(offset)))
        query = urlencode(kept, safe="|,:/")
        return f"{self.base_url}/{resource_type}" + (f"?{query}" if query else "")

    def capability_statement(self) -> dict[str, Any]:
        return {
            "resourceType": "CapabilityStatement",
            "status": "active",
            "kind": "instance",
            "fhirVersion": "4.0.1",
            "format": ["application/fhir+json"],
            "software": {"name": "fhir-healthcare-ai in-memory test server"},
            "rest": [
                {
                    "mode": "server",
                    "resource": [
                        {
                            "type": resource_type,
                            "interaction": [{"code": "read"}, {"code": "search-type"}],
                            "searchParam": [
                                {"name": name, "type": p.kind} for name, p in params.items()
                            ],
                        }
                        for resource_type, params in _PARAMS.items()
                    ],
                }
            ],
        }


def _matches(resource: dict[str, Any], filters: list[tuple[_Param, str, str | None]]) -> bool:
    """Repeated parameters AND together; comma-separated values within one OR together."""
    for param, value, modifier in filters:
        candidates = param.extract(resource)
        values = value.split(",")
        if param.kind == "token":
            hit = any(_match_token(candidates, v, modifier) for v in values)
            hit = not hit if modifier == "not" else hit
        elif param.kind == "date":
            hit = any(_match_date(candidates, v) for v in values)
        elif param.kind == "quantity":
            hit = any(_match_quantity(candidates, v) for v in values)
        elif param.kind == "reference":
            hit = any(_match_reference(candidates, v, param.target) for v in values)
        else:  # string: case-insensitive prefix, per the FHIR spec
            hit = any(str(c).lower().startswith(v.lower()) for c in candidates for v in values)
        if not hit:
            return False
    return True


def _json(status: int, payload: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        status, content=json.dumps(payload), headers={"Content-Type": "application/fhir+json"}
    )


def _outcome(status: int, code: str, message: str) -> httpx.Response:
    return _json(
        status,
        {
            "resourceType": "OperationOutcome",
            "issue": [{"severity": "error", "code": code, "diagnostics": message}],
        },
    )
