"""Async FHIR REST client.

The client's public search API accepts :class:`~fhir_healthcare_ai.domain.query.FHIRQuery`
objects only. There is deliberately no ``get(url)`` method: if arbitrary URLs cannot be
passed in, a compromised or confused caller upstream cannot turn this into a generic
HTTP proxy.

Pagination follows ``Bundle.link[next]``, but only after checking that the link stays on
the configured server. A FHIR server that returns a next-link pointing somewhere else is
either misconfigured or hostile, and following it blindly would be an SSRF primitive.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any
from urllib.parse import urlparse

import httpx

from fhir_healthcare_ai.audit.log import AuditEvent, AuditSink, LoggingAuditSink
from fhir_healthcare_ai.config import FHIRSettings
from fhir_healthcare_ai.domain.enums import ResourceType
from fhir_healthcare_ai.domain.query import FHIRQuery
from fhir_healthcare_ai.logging_config import correlation_id_var, get_logger

logger = get_logger(__name__)

FHIR_JSON = "application/fhir+json"
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class FHIRError(Exception):
    """Base class for FHIR transport failures."""


class FHIRNotFoundError(FHIRError):
    """Resource does not exist."""


class FHIRAuthError(FHIRError):
    """Server rejected our credentials."""


class FHIRServerError(FHIRError):
    """Server returned an error we could not recover from."""


class FHIRWriteForbiddenError(FHIRError):
    """A write was attempted while ``FHIR_ALLOW_WRITE`` is false."""


@dataclass
class SearchResult:
    """Outcome of a (possibly paginated) search."""

    query: FHIRQuery
    matches: list[dict[str, Any]] = field(default_factory=list)
    included: list[dict[str, Any]] = field(default_factory=list)
    total: int | None = None
    pages_fetched: int = 0
    truncated: bool = False
    duration_ms: float = 0.0

    @property
    def all_resources(self) -> list[dict[str, Any]]:
        return [*self.matches, *self.included]

    def __len__(self) -> int:
        return len(self.matches)


class FHIRClient:
    """Thin, safety-constrained wrapper over a FHIR R4 REST endpoint."""

    def __init__(
        self,
        settings: FHIRSettings | None = None,
        *,
        audit_sink: AuditSink | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings or FHIRSettings()
        self.audit = audit_sink or LoggingAuditSink()
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self.settings.base_url,
            timeout=self.settings.timeout_seconds,
            verify=self.settings.verify_ssl,
            headers=self._default_headers(),
            follow_redirects=False,
        )

    def _default_headers(self) -> dict[str, str]:
        headers = {"Accept": FHIR_JSON, "Content-Type": FHIR_JSON}
        if self.settings.auth_token:
            headers["Authorization"] = f"Bearer {self.settings.auth_token}"
        return headers

    async def __aenter__(self) -> FHIRClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # -- reads ----------------------------------------------------------------------

    async def search(self, query: FHIRQuery, max_pages: int | None = None) -> SearchResult:
        """Execute a validated search, following pages up to the configured caps."""
        started = time.perf_counter()
        page_limit = max_pages or self.settings.max_pages
        result = SearchResult(query=query)
        url: str | None = query.relative_url()
        params: dict[str, str] | None = None

        while url and result.pages_fetched < page_limit:
            bundle = await self._request_json("GET", url, params=params)
            params = None
            result.pages_fetched += 1
            if result.total is None:
                total = bundle.get("total")
                result.total = int(total) if isinstance(total, int) else None

            for entry in bundle.get("entry") or []:
                resource = entry.get("resource")
                if not isinstance(resource, dict):
                    continue
                mode = (entry.get("search") or {}).get("mode", "match")
                if mode == "include":
                    result.included.append(resource)
                else:
                    result.matches.append(resource)

            if len(result.matches) >= self.settings.max_total_resources:
                result.truncated = True
                break

            url = self._next_link(bundle)

        if url and result.pages_fetched >= page_limit:
            result.truncated = True

        result.duration_ms = (time.perf_counter() - started) * 1000
        self._audit_search(result)
        return result

    async def read(self, resource_type: ResourceType, resource_id: str) -> dict[str, Any]:
        """Read a single resource by logical id."""
        if not _is_safe_id(resource_id):
            raise FHIRError(f"unsafe resource id: {resource_id!r}")
        return await self._request_json("GET", f"{resource_type.value}/{resource_id}")

    async def capability_statement(self) -> dict[str, Any]:
        """Fetch ``/metadata``. Used by the health check to prove the server is FHIR."""
        return await self._request_json("GET", "metadata", params={"_summary": "true"})

    async def ping(self) -> bool:
        """True when the server answers with a CapabilityStatement."""
        try:
            statement = await self.capability_statement()
        except (FHIRError, httpx.HTTPError):
            return False
        return statement.get("resourceType") == "CapabilityStatement"

    # -- writes (seeding only) ------------------------------------------------------

    async def post_bundle(self, bundle: dict[str, Any]) -> dict[str, Any]:
        """POST a transaction/batch bundle. Refused unless writes are enabled."""
        if not self.settings.allow_write:
            raise FHIRWriteForbiddenError(
                "writes are disabled; set FHIR_ALLOW_WRITE=true to seed data"
            )
        if bundle.get("resourceType") != "Bundle":
            raise FHIRError("post_bundle expects a Bundle resource")
        return await self._request_json("POST", "", json_body=bundle)

    # -- transport ------------------------------------------------------------------

    async def _request_json(
        self,
        method: str,
        path: str,
        params: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        max_attempts = self.settings.max_retries + 1
        last_error: Exception | None = None

        for attempt in range(max_attempts):
            is_last = attempt == max_attempts - 1
            try:
                response = await self._client.request(
                    method, path or "/", params=params, json=json_body
                )
            except httpx.HTTPError as exc:
                last_error = exc
                if is_last:
                    raise FHIRServerError(f"{method} {path} failed: {exc}") from exc
                await asyncio.sleep(0.25 * 2**attempt)
                continue

            if response.status_code in _RETRY_STATUS and not is_last:
                await asyncio.sleep(_retry_delay(response, attempt))
                continue

            return self._handle_response(response, method, path)

        raise FHIRServerError(f"{method} {path} failed after retries: {last_error}")

    def _handle_response(self, response: httpx.Response, method: str, path: str) -> dict[str, Any]:
        if response.status_code == 404:
            raise FHIRNotFoundError(f"{method} {path} -> 404")
        if response.status_code in (401, 403):
            raise FHIRAuthError(f"{method} {path} -> {response.status_code}")
        if response.status_code >= 400:
            raise FHIRServerError(
                f"{method} {path} -> {response.status_code}: {_operation_outcome_text(response)}"
            )
        if not response.content:
            return {}
        try:
            payload = response.json()
        except ValueError as exc:
            raise FHIRServerError(f"{method} {path} returned non-JSON content") from exc
        if not isinstance(payload, dict):
            raise FHIRServerError(f"{method} {path} returned a non-object JSON payload")
        return payload

    def _next_link(self, bundle: dict[str, Any]) -> str | None:
        """Return the next page URL, if it is on our configured server."""
        for link in bundle.get("link") or []:
            if link.get("relation") != "next":
                continue
            url = link.get("url")
            if not isinstance(url, str):
                return None
            if not self._is_same_origin(url):
                logger.warning(
                    "refusing off-origin pagination link",
                    extra={"next_url": url, "base_url": self.settings.base_url},
                )
                return None
            return url
        return None

    def _is_same_origin(self, url: str) -> bool:
        target = urlparse(url)
        if not target.scheme:  # relative link, stays on base_url
            return True
        base = urlparse(self.settings.base_url)
        return (target.scheme, target.netloc) == (base.scheme, base.netloc)

    def _audit_search(self, result: SearchResult) -> None:
        self.audit.record(
            AuditEvent(
                correlation_id=correlation_id_var.get() or "-",
                action="query.executed",
                resource_type=result.query.resource_type.value,
                query=result.query.relative_url(),
                resource_count=len(result.matches),
                duration_ms=round(result.duration_ms, 2),
                details={
                    "pages": result.pages_fetched,
                    "included": len(result.included),
                    "truncated": result.truncated,
                    "step_id": result.query.step_id,
                },
            )
        )


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return min(float(retry_after), 5.0)
        except ValueError:
            pass
    return 0.25 * 2**attempt


def _operation_outcome_text(response: httpx.Response) -> str:
    """Extract a human-readable message from an OperationOutcome, if present."""
    try:
        payload = response.json()
    except ValueError:
        return response.text[:300]
    if isinstance(payload, dict) and payload.get("resourceType") == "OperationOutcome":
        parts = []
        for issue in payload.get("issue") or []:
            detail = (issue.get("details") or {}).get("text") or issue.get("diagnostics")
            if detail:
                parts.append(str(detail))
        if parts:
            return "; ".join(parts)[:500]
    return response.text[:300]


def _is_safe_id(resource_id: str) -> bool:
    """FHIR logical ids are ``[A-Za-z0-9\\-\\.]{1,64}``."""
    if not 1 <= len(resource_id) <= 64:
        return False
    return all(ch.isalnum() or ch in "-." for ch in resource_id)
