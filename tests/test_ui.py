"""The browser UI: served from the package, self-contained, and additive to the API."""

from __future__ import annotations

import re
from collections.abc import Iterator
from importlib import resources

import pytest
from fastapi.testclient import TestClient

from fhir_healthcare_ai.api.main import create_app
from fhir_healthcare_ai.audit import InMemoryAuditSink
from fhir_healthcare_ai.config import FHIRSettings, Settings
from fhir_healthcare_ai.fhir.memory import InMemoryFHIRServer
from fhir_healthcare_ai.llm.mock import MockLLMProvider
from fhir_healthcare_ai.synthetic.generator import SyntheticDataset


@pytest.fixture
def api(dataset: SyntheticDataset) -> Iterator[TestClient]:
    server = InMemoryFHIRServer(resources=dataset.resources, read_only=True)
    settings = Settings(
        log_json=False,
        log_level="ERROR",
        fhir=FHIRSettings(base_url=server.base_url, max_retries=0),
    )
    audit = InMemoryAuditSink()
    app = create_app(
        settings,
        client=server.client(settings.fhir, audit_sink=audit),
        provider=MockLLMProvider(),
        audit=audit,
    )
    with TestClient(app) as client:
        yield client


def test_index_serves_the_page(api: TestClient) -> None:
    response = api.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    html = response.text
    assert '<form id="query-form"' in html and 'id="question"' in html
    assert '<form id="patient-form"' in html
    assert "X-Request-ID" in response.headers


def test_page_is_locked_down(api: TestClient) -> None:
    headers = api.get("/").headers
    csp = headers["content-security-policy"]
    assert "script-src 'self'" in csp and "default-src 'none'" in csp
    assert headers["x-content-type-options"] == "nosniff"


def test_every_referenced_asset_resolves(api: TestClient) -> None:
    html = api.get("/").text
    referenced = re.findall(r'(?:src|href)="(/ui/[^"]+)"', html)
    assert {"/ui/app.js", "/ui/app.css"} <= set(referenced)
    expected_types = {".js": "text/javascript", ".css": "text/css", ".woff2": "font/woff2"}
    for path in referenced:
        response = api.get(path)
        assert response.status_code == 200, path
        suffix = path[path.rfind(".") :]
        assert response.headers["content-type"].startswith(expected_types[suffix])
        assert response.content


def test_page_has_no_external_or_inline_code(api: TestClient) -> None:
    """Offline-capable and compatible with the CSP: nothing from a CDN, nothing inline."""
    html = api.get("/").text
    assert not re.search(r'(?:src|href)="(?:https?:)?//', html)
    assert "<script>" not in html and "<style" not in html and " style=" not in html
    js = api.get("/ui/app.js").text
    assert "innerHTML" not in js and "insertAdjacentHTML" not in js


@pytest.mark.parametrize("path", ["/ui/missing.js", "/ui/..%2Fui.py", "/ui/../main.py", "/ui/"])
def test_unknown_assets_are_404(api: TestClient, path: str) -> None:
    assert api.get(path).status_code == 404


def test_assets_are_packaged_with_the_module() -> None:
    """The wheel ships whatever lives in the package directory; the router reads it there."""
    static = resources.files("fhir_healthcare_ai.api").joinpath("static")
    for name in ("index.html", "app.js", "app.css", "atkinson-next.woff2", "atkinson-mono.woff2"):
        assert static.joinpath(name).is_file(), name


def test_ui_is_not_part_of_the_openapi_schema(api: TestClient) -> None:
    paths = api.get("/openapi.json").json()["paths"]
    assert "/" not in paths and not any(p.startswith("/ui") for p in paths)
    assert {"/query", "/health", "/capabilities", "/patient/{patient_id}/analyze"} <= set(paths)


def test_api_routes_are_unaffected(api: TestClient) -> None:
    assert api.get("/health").json()["status"] == "ok"
    assert api.get("/capabilities").json()["example_questions"]
    body = api.post(
        "/query", json={"question": "Which patients have elevated HbA1c?", "as_of": "2026-01-01"}
    ).json()
    assert body["patients"] and body["disclaimer"]
    assert api.get("/patient/syn7-pat-0001/analyze").status_code == 200
    assert api.get("/docs").status_code == 200


def test_fonts_are_bundled_and_allowed_by_the_csp(api: TestClient) -> None:
    css = api.get("/ui/app.css").text
    fonts = re.findall(r'url\("(/ui/[^"]+\.woff2)"\)', css)
    assert fonts
    for path in fonts:
        response = api.get(path)
        assert response.status_code == 200
        assert response.headers["content-type"] == "font/woff2"
        assert response.content[:4] == b"wOF2"
    assert "font-src 'self'" in response.headers["content-security-policy"]


def test_page_keeps_style_out_of_attributes() -> None:
    """Widths are set through the CSSOM, because the CSP forbids inline style attributes."""
    js = resources.files("fhir_healthcare_ai.api").joinpath("static", "app.js").read_text()
    assert 'setAttribute("style"' not in js and ".style.cssText" not in js
