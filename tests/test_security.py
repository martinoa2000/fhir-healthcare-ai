"""API-key authentication, per-caller rate limiting and the audited actor."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from fhir_healthcare_ai.api.main import create_app
from fhir_healthcare_ai.api.security import SlidingWindowRateLimiter
from fhir_healthcare_ai.audit import InMemoryAuditSink
from fhir_healthcare_ai.config import FHIRSettings, SecuritySettings, Settings
from fhir_healthcare_ai.fhir.memory import InMemoryFHIRServer
from fhir_healthcare_ai.llm.mock import MockLLMProvider
from fhir_healthcare_ai.logging_config import JSONFormatter, actor_var
from fhir_healthcare_ai.synthetic.generator import SyntheticDataset

ALICE = "alice-secret-3f9a7c"
CI = "ci-secret-b81e04"
KEYS = {"alice": ALICE, "ci": CI}
QUESTION = {"question": "Which patients have elevated HbA1c?", "as_of": "2026-01-01"}


def _app(
    dataset: SyntheticDataset, *, environment: str = "local", **security: Any
) -> tuple[TestClient, InMemoryAuditSink]:
    server = InMemoryFHIRServer(resources=dataset.resources, read_only=True)
    settings = Settings(
        environment=environment,
        log_json=False,
        log_level="ERROR",
        fhir=FHIRSettings(base_url=server.base_url, max_retries=0),
        security=SecuritySettings(**security),
    )
    audit = InMemoryAuditSink()
    app = create_app(
        settings,
        client=server.client(settings.fhir, audit_sink=audit),
        provider=MockLLMProvider(),
        audit=audit,
    )
    return TestClient(app), audit


@pytest.fixture
def secured(dataset: SyntheticDataset) -> Iterator[tuple[TestClient, InMemoryAuditSink]]:
    client, audit = _app(dataset, api_keys=KEYS)
    with client:
        yield client, audit


def _bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


# --------------------------------------------------------------------- authentication


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/health"),
        ("GET", "/capabilities"),
        ("POST", "/query"),
        ("GET", "/patient/syn7-pat-0001/analyze"),
        ("GET", "/audit"),
    ],
)
def test_protected_routes_need_a_key(
    secured: tuple[TestClient, InMemoryAuditSink], method: str, path: str
) -> None:
    client, _ = secured
    response = client.request(method, path, json=QUESTION if method == "POST" else None)
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    body = response.json()
    assert body["detail"] == "missing API key" and body["correlation_id"]


def test_wrong_key_is_rejected(secured: tuple[TestClient, InMemoryAuditSink]) -> None:
    client, _ = secured
    for headers in (_bearer("nope"), {"X-API-Key": "nope"}, _bearer(ALICE + "x")):
        response = client.get("/health", headers=headers)
        assert response.status_code == 401
        assert response.json()["detail"] == "invalid API key"
        assert response.headers["WWW-Authenticate"] == "Bearer"


def test_both_header_styles_are_accepted(secured: tuple[TestClient, InMemoryAuditSink]) -> None:
    client, _ = secured
    assert client.get("/health", headers=_bearer(ALICE)).status_code == 200
    assert client.get("/health", headers={"X-API-Key": CI}).status_code == 200


def test_auth_failure_is_checked_before_body_validation(
    secured: tuple[TestClient, InMemoryAuditSink],
) -> None:
    client, _ = secured
    assert client.post("/query", json={"question": ""}).status_code == 401


def test_probes_docs_and_root_stay_public(secured: tuple[TestClient, InMemoryAuditSink]) -> None:
    client, _ = secured
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 200
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200
    # The UI shell and its assets hold no data and must load before a key can be entered.
    assert client.get("/").status_code == 200
    assert client.get("/ui/app.js").status_code == 200


def test_actor_is_audited_and_the_secret_never_is(
    secured: tuple[TestClient, InMemoryAuditSink],
    caplog: pytest.LogCaptureFixture,
) -> None:
    client, audit = secured
    caplog.set_level(logging.INFO)
    response = client.post(
        "/query", json=QUESTION, headers={**_bearer(ALICE), "X-Request-ID": "who-asked"}
    )
    assert response.status_code == 200

    events = [e for e in audit.events if e.correlation_id == "who-asked"]
    assert events and {e.actor for e in events} == {"alice"}
    trail = "\n".join(e.to_json() for e in audit.events)
    assert ALICE not in trail and CI not in trail
    assert ALICE not in caplog.text

    # The failed attempt is logged without echoing what was presented.
    client.get("/health", headers=_bearer("leaked-guess-123"))
    assert "authentication failed" in caplog.text
    assert "leaked-guess-123" not in caplog.text


def test_json_log_lines_carry_the_actor() -> None:
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "hello", (), None)
    token = actor_var.set("alice")
    try:
        payload = json.loads(JSONFormatter().format(record))
    finally:
        actor_var.reset(token)
    assert payload["actor"] == "alice"
    assert "actor" not in json.loads(JSONFormatter().format(record))


def test_auth_disabled_in_local_works_as_before(dataset: SyntheticDataset) -> None:
    client, audit = _app(dataset)
    with client:
        assert client.get("/health").status_code == 200
        response = client.post("/query", json=QUESTION, headers={"X-Request-ID": "anon"})
        assert response.status_code == 200
    events = [e for e in audit.events if e.correlation_id == "anon"]
    assert events and {e.actor for e in events} == {"anonymous"}


# ---------------------------------------------------------------------- configuration


def test_keys_alone_turn_auth_on_outside_prod() -> None:
    assert Settings().security.auth_enabled is False
    assert Settings(security=SecuritySettings(api_keys=KEYS)).security.auth_enabled is True
    off = SecuritySettings(api_keys=KEYS, require_auth=False)
    assert Settings(security=off).security.auth_enabled is False


def test_prod_without_keys_fails_at_startup() -> None:
    with pytest.raises(ValidationError, match="no API keys are configured"):
        Settings(environment="prod")


def test_prod_cannot_switch_auth_off() -> None:
    with pytest.raises(ValidationError, match="not allowed when ENVIRONMENT=prod"):
        Settings(environment="prod", security=SecuritySettings(api_keys=KEYS, require_auth=False))
    prod = Settings(environment="prod", security=SecuritySettings(api_keys=KEYS))
    assert prod.security.auth_enabled


def test_required_auth_without_keys_is_refused() -> None:
    with pytest.raises(ValidationError, match="no API keys"):
        Settings(security=SecuritySettings(require_auth=True))


def test_keys_are_read_from_json_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEYS", json.dumps(KEYS))
    monkeypatch.setenv("API_RATE_LIMIT_PER_MINUTE", "5")
    settings = Settings(environment="prod")
    assert set(settings.security.api_keys) == {"alice", "ci"}
    assert settings.security.rate_limit_per_minute == 5
    assert ALICE not in repr(settings)


def test_blank_env_values_from_the_example_file_mean_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_REQUIRE_AUTH", "")
    monkeypatch.setenv("API_KEYS", "{}")
    assert Settings().security.auth_enabled is False


@pytest.mark.parametrize(
    "keys",
    [{"bad name": "s3cret"}, {"alice": "   "}, {"alice": "same", "bob": "same"}],
)
def test_malformed_keys_are_refused(keys: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        SecuritySettings(api_keys=keys)


# ----------------------------------------------------------------------- rate limiting


def test_rate_limit_is_per_actor_and_sets_retry_after(dataset: SyntheticDataset) -> None:
    client, _ = _app(dataset, api_keys=KEYS, rate_limit_per_minute=2)
    path = "/patient/syn7-pat-0001/analyze"
    with client:
        for _ in range(2):
            assert client.get(path, headers=_bearer(ALICE)).status_code == 200
        limited = client.get(path, headers=_bearer(ALICE))
        assert limited.status_code == 429
        assert 1 <= int(limited.headers["Retry-After"]) <= 60
        assert limited.json()["correlation_id"]
        # /query shares the budget; another caller has their own.
        assert client.post("/query", json=QUESTION, headers=_bearer(ALICE)).status_code == 429
        assert client.get(path, headers={"X-API-Key": CI}).status_code == 200
        # Cheap routes are not limited.
        assert client.get("/health", headers=_bearer(ALICE)).status_code == 200


def test_rate_limit_applies_to_anonymous_callers_by_address(dataset: SyntheticDataset) -> None:
    client, _ = _app(dataset, rate_limit_per_minute=1)
    with client:
        assert client.post("/query", json=QUESTION).status_code == 200
        assert client.post("/query", json=QUESTION).status_code == 429


def test_rate_limit_zero_disables_it(dataset: SyntheticDataset) -> None:
    client, _ = _app(dataset, rate_limit_per_minute=0)
    with client:
        for _ in range(5):
            assert client.get("/patient/syn7-pat-0001/analyze").status_code == 200


def test_sliding_window_releases_hits_as_they_age() -> None:
    now = [1000.0]
    limiter = SlidingWindowRateLimiter(2, 60.0, clock=lambda: now[0])
    assert limiter.hit("a") is None
    now[0] += 30
    assert limiter.hit("a") is None
    assert limiter.hit("a") == pytest.approx(30.0)
    assert limiter.hit("b") is None
    now[0] += 30.5
    assert limiter.hit("a") is None  # the first hit has left the window
    assert limiter.hit("a") is not None


def test_idle_keys_are_swept() -> None:
    now = [0.0]
    limiter = SlidingWindowRateLimiter(1, 60.0, clock=lambda: now[0], max_keys=2)
    limiter.hit("a")
    limiter.hit("b")
    now[0] += 61
    limiter.hit("c")
    assert set(limiter._hits) == {"c"}
