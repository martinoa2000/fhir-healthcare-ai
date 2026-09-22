"""FHIRClient over real HTTP semantics, via the in-memory server."""

from __future__ import annotations

import httpx
import pytest

from fhir_healthcare_ai.audit import InMemoryAuditSink
from fhir_healthcare_ai.config import FHIRSettings
from fhir_healthcare_ai.domain.enums import ResourceType
from fhir_healthcare_ai.domain.query import FHIRQuery
from fhir_healthcare_ai.fhir.client import (
    FHIRClient,
    FHIRError,
    FHIRNotFoundError,
    FHIRServerError,
    FHIRWriteForbiddenError,
)
from fhir_healthcare_ai.fhir.memory import InMemoryFHIRServer, parse_fhir_date
from fhir_healthcare_ai.synthetic.generator import SyntheticDataset
from fhir_healthcare_ai.synthetic.loader import push_dataset
from fhir_healthcare_ai.terminology import LOINC


def _query(resource_type: ResourceType, *params: tuple[str, str]) -> FHIRQuery:
    return FHIRQuery(resource_type=resource_type, params=params)


async def test_search_follows_pagination_to_the_end(
    client: FHIRClient, server: InMemoryFHIRServer
) -> None:
    result = await client.search(_query(ResourceType.PATIENT, ("_count", "7")))
    assert len(result.matches) == server.count("Patient")
    assert result.pages_fetched == -(-server.count("Patient") // 7)
    assert not result.truncated


async def test_page_cap_marks_the_result_truncated(
    server: InMemoryFHIRServer, audit: InMemoryAuditSink
) -> None:
    settings = FHIRSettings(base_url=server.base_url, max_pages=2, max_retries=0)
    async with server.client(settings, audit_sink=audit) as capped:
        result = await capped.search(_query(ResourceType.PATIENT, ("_count", "5")))
    assert result.pages_fetched == 2
    assert result.truncated
    assert len(result.matches) == 10


async def test_off_origin_next_link_is_not_followed(server: InMemoryFHIRServer) -> None:
    def hostile(request: httpx.Request) -> httpx.Response:
        bundle = server.search("Patient", [("_count", "3")])
        bundle["link"] = [{"relation": "next", "url": "http://attacker.example/fhir/Patient"}]
        return httpx.Response(200, json=bundle)

    http = httpx.AsyncClient(base_url=server.base_url, transport=httpx.MockTransport(hostile))
    async with FHIRClient(FHIRSettings(base_url=server.base_url), client=http) as fhir:
        result = await fhir.search(_query(ResourceType.PATIENT, ("_count", "3")))
    await http.aclose()
    assert result.pages_fetched == 1
    assert len(result.matches) == 3


async def test_transient_errors_are_retried(server: InMemoryFHIRServer) -> None:
    server.fail_next = [503, 429]
    settings = FHIRSettings(base_url=server.base_url, max_retries=2)
    async with server.client(settings) as retrying:
        result = await retrying.search(_query(ResourceType.PATIENT, ("_count", "100")))
    assert len(result.matches) == server.count("Patient")
    assert server.requests == ["GET Patient?_count=100"] * 3


async def test_retries_are_bounded(server: InMemoryFHIRServer) -> None:
    server.fail_next = [503, 503, 503]
    settings = FHIRSettings(base_url=server.base_url, max_retries=1)
    async with server.client(settings) as retrying:
        with pytest.raises(FHIRServerError):
            await retrying.search(_query(ResourceType.PATIENT))


async def test_unknown_parameter_is_a_server_error_not_a_silent_match(
    client: FHIRClient,
) -> None:
    with pytest.raises(FHIRServerError, match="unknown search parameter"):
        await client.search(_query(ResourceType.OBSERVATION, ("value-string", "x")))


async def test_read_and_not_found(client: FHIRClient, dataset: SyntheticDataset) -> None:
    patient = next(r for r in dataset.resources if r["resourceType"] == "Patient")
    assert (await client.read(ResourceType.PATIENT, patient["id"]))["id"] == patient["id"]
    with pytest.raises(FHIRNotFoundError):
        await client.read(ResourceType.PATIENT, "does-not-exist")
    with pytest.raises(FHIRError, match="unsafe"):
        await client.read(ResourceType.PATIENT, "../metadata")


async def test_writes_are_refused_unless_enabled(client: FHIRClient) -> None:
    with pytest.raises(FHIRWriteForbiddenError):
        await client.post_bundle({"resourceType": "Bundle", "type": "transaction", "entry": []})


async def test_every_search_is_audited(client: FHIRClient, audit: InMemoryAuditSink) -> None:
    await client.search(_query(ResourceType.PATIENT, ("_count", "2")))
    events = audit.by_action("query.executed")
    assert events and events[-1].query == "Patient?_count=2"


async def test_token_date_and_quantity_search(client: FHIRClient) -> None:
    result = await client.search(
        _query(
            ResourceType.OBSERVATION,
            ("code", f"{LOINC}|4548-4"),
            ("value-quantity", "gt7||%"),
            ("date", "ge2025-01-01"),
            ("_count", "200"),
        )
    )
    assert result.matches
    for observation in result.matches:
        assert observation["valueQuantity"]["value"] > 7
        assert observation["effectiveDateTime"] >= "2025"


async def test_component_value_quantity_matches_panel_components(client: FHIRClient) -> None:
    result = await client.search(
        _query(
            ResourceType.OBSERVATION,
            ("code", f"{LOINC}|85354-9"),
            ("component-value-quantity", "ge160||mm[Hg]"),
            ("_count", "200"),
        )
    )
    for panel in result.matches:
        assert any(c["valueQuantity"]["value"] >= 160 for c in panel["component"])


async def test_not_modifier_and_include(client: FHIRClient) -> None:
    result = await client.search(
        _query(
            ResourceType.CONDITION,
            ("clinical-status:not", "active"),
            ("_include", "Condition:patient"),
            ("_count", "200"),
        )
    )
    assert all(c["clinicalStatus"]["coding"][0]["code"] != "active" for c in result.matches)
    assert {r["resourceType"] for r in result.included} <= {"Patient"}


def test_partial_dates_denote_ranges() -> None:
    start, end = parse_fhir_date("2024")
    assert (start.month, start.day, end.month, end.day) == (1, 1, 12, 31)
    start, end = parse_fhir_date("2024-02")
    assert end.day == 29


async def test_seeder_upserts_idempotently(dataset: SyntheticDataset) -> None:
    target = InMemoryFHIRServer()
    settings = FHIRSettings(base_url=target.base_url, allow_write=True, max_retries=0)
    async with target.client(settings) as writer:
        first = await push_dataset(dataset, target.base_url, bundle_size=200, client=writer)
        second = await push_dataset(dataset, target.base_url, bundle_size=200, client=writer)
    assert first.ok and second.ok
    assert first.created == len(dataset.resources) and first.updated == 0
    assert second.updated == len(dataset.resources) and second.created == 0
    assert target.count() == len(dataset.resources)


async def test_read_only_server_refuses_transactions() -> None:
    target = InMemoryFHIRServer(read_only=True)
    settings = FHIRSettings(base_url=target.base_url, allow_write=True, max_retries=0)
    async with target.client(settings) as writer:
        with pytest.raises(FHIRError):
            await writer.post_bundle({"resourceType": "Bundle", "type": "transaction"})
