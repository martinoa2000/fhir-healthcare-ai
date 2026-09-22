"""The HTTP layer, run for real against injected in-memory dependencies."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fhir_healthcare_ai.api.main import create_app
from fhir_healthcare_ai.audit import InMemoryAuditSink
from fhir_healthcare_ai.config import FHIRSettings, Settings
from fhir_healthcare_ai.fhir.memory import InMemoryFHIRServer
from fhir_healthcare_ai.llm.base import LLMMessage, LLMProvider, LLMResponse, LLMUnavailableError
from fhir_healthcare_ai.llm.mock import MockLLMProvider
from fhir_healthcare_ai.synthetic.generator import SyntheticDataset


class DownProvider(LLMProvider):
    name = "vllm"

    async def complete(self, messages: list[LLMMessage], **_: Any) -> LLMResponse:
        raise LLMUnavailableError("connection refused")


def _app(
    dataset: SyntheticDataset,
    *,
    provider: LLMProvider | None = None,
    environment: str = "local",
) -> tuple[TestClient, InMemoryFHIRServer]:
    server = InMemoryFHIRServer(resources=dataset.resources, read_only=True)
    settings = Settings(
        environment=environment,  # type: ignore[arg-type]
        log_json=False,
        log_level="ERROR",
        fhir=FHIRSettings(base_url=server.base_url, max_retries=0),
    )
    audit = InMemoryAuditSink()
    app = create_app(
        settings,
        client=server.client(settings.fhir, audit_sink=audit),
        provider=provider or MockLLMProvider(),
        audit=audit,
    )
    return TestClient(app), server


@pytest.fixture
def api(dataset: SyntheticDataset) -> Iterator[TestClient]:
    client, _ = _app(dataset)
    with client:
        yield client


def test_liveness_and_readiness(api: TestClient) -> None:
    assert api.get("/health/live").json() == {"status": "ok"}
    assert api.get("/health/ready").status_code == 200


def test_health_reports_dependencies(api: TestClient) -> None:
    body = api.get("/health").json()
    assert body["status"] == "ok"
    assert body["fhir"]["reachable"] is True
    assert body["llm"]["active"] == "mock" and body["llm"]["local"] is True


def test_capabilities_expose_the_allowlist_and_features(api: TestClient) -> None:
    body = api.get("/capabilities").json()
    assert "Observation" in body["resources"]
    assert "Binary" not in body["resources"]
    assert any(f["name"] == "hba1c_latest" for f in body["features"])
    assert body["example_questions"]


def test_query_returns_evidence_and_echoes_the_request_id(api: TestClient) -> None:
    response = api.post(
        "/query",
        json={"question": "Which patients have elevated HbA1c?", "as_of": "2026-01-01"},
        headers={"X-Request-ID": "trace-123"},
    )
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "trace-123"
    body = response.json()
    assert body["patients"] and body["evidence"]
    assert body["disclaimer"]
    assert body["trace"]["correlation_id"] == "trace-123"


def test_malformed_request_id_is_replaced_not_echoed(api: TestClient) -> None:
    response = api.get("/health/live", headers={"X-Request-ID": "bad id\r\nX-Evil: 1"})
    assert response.headers["X-Request-ID"] != "bad id\r\nX-Evil: 1"
    assert len(response.headers["X-Request-ID"]) == 32


def test_query_input_is_validated(api: TestClient) -> None:
    assert api.post("/query", json={"question": ""}).status_code == 422
    assert api.post("/query", json={"question": "x" * 1001}).status_code == 422
    extra = {"question": "Which patients have elevated HbA1c?", "url": "http://x"}
    assert api.post("/query", json=extra).status_code == 422


def test_unsupported_question_is_a_successful_refusal(api: TestClient) -> None:
    body = api.post("/query", json={"question": "What is the weather tomorrow?"}).json()
    assert body["query_plan"]["unsupported"] is True
    assert body["patients"] == []


def test_patient_analysis_routes(api: TestClient) -> None:
    ok = api.get("/patient/syn7-pat-0001/analyze", params={"as_of": "2026-01-01"})
    assert ok.status_code == 200 and ok.json()["patient_id"] == "syn7-pat-0001"
    missing = api.get("/patient/nobody/analyze")
    assert missing.status_code == 404 and missing.json()["correlation_id"]
    assert api.get("/patient/bad$id/analyze").status_code == 422


def test_audit_tail_filters_by_request(api: TestClient) -> None:
    api.post(
        "/query",
        json={"question": "Which patients have elevated HbA1c?"},
        headers={"X-Request-ID": "audit-me"},
    )
    events = api.get("/audit", params={"correlation_id": "audit-me"}).json()
    assert events and {e["correlation_id"] for e in events} == {"audit-me"}


def test_audit_tail_is_disabled_in_production(dataset: SyntheticDataset) -> None:
    client, _ = _app(dataset, environment="prod")
    with client:
        assert client.get("/audit").status_code == 404


def test_model_outage_is_a_503(dataset: SyntheticDataset) -> None:
    client, _ = _app(dataset, provider=DownProvider())
    with client:
        response = client.post("/query", json={"question": "Which patients have high HbA1c?"})
    assert response.status_code == 503
    assert "planner is unavailable" in response.json()["detail"]


def test_api_never_writes_to_the_fhir_server(dataset: SyntheticDataset) -> None:
    client, server = _app(dataset)
    with client:
        client.post("/query", json={"question": "Which diabetic patients are not on a statin?"})
        client.get("/patient/syn7-pat-0002/analyze")
    assert server.requests
    assert all(line.startswith("GET ") for line in server.requests)


def test_in_memory_mode_boots_without_injection(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(
        log_json=False,
        log_level="ERROR",
        fhir=FHIRSettings(in_memory=True, in_memory_patients=15),
        llm={"provider": "mock"},  # type: ignore[arg-type]
    )
    with TestClient(create_app(settings)) as client:
        body = client.get("/health").json()
    assert body["fhir"]["in_memory"] is True and body["status"] == "ok"
