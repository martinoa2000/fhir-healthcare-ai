"""Exports: the pure renderers, then the HTTP route against in-memory dependencies."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from fhir_healthcare_ai.api.main import create_app
from fhir_healthcare_ai.audit import InMemoryAuditSink
from fhir_healthcare_ai.config import FHIRSettings, Settings
from fhir_healthcare_ai.domain.enums import AbnormalFlag, AnalysisType, ResourceType, RiskBand
from fhir_healthcare_ai.domain.query import QueryPlan
from fhir_healthcare_ai.domain.results import (
    AbnormalLabFinding,
    Evidence,
    PatientAnalysis,
    PatientMatch,
    QueryResponse,
    RiskAssessment,
)
from fhir_healthcare_ai.export import (
    ANALYSIS_COLUMNS,
    BASE_COLUMNS,
    EXECUTED_QUERY_EXTENSION,
    to_csv,
    to_fhir_bundle,
    to_fhir_group,
)
from fhir_healthcare_ai.fhir.memory import InMemoryFHIRServer
from fhir_healthcare_ai.llm.mock import MockLLMProvider
from fhir_healthcare_ai.synthetic.generator import SyntheticDataset

QUESTION = "Which patients have elevated HbA1c?"
QUERIES = ["Observation?code=http://loinc.org|4548-4&value-quantity=gt6.5"]


def _evidence(pid: str, rid: str) -> Evidence:
    return Evidence(patient_id=pid, resource_type=ResourceType.OBSERVATION, resource_id=rid)


def _response(
    *, analyses: bool = False, patients: list[PatientMatch] | None = None
) -> QueryResponse:
    patients = patients or [
        PatientMatch(
            patient_id="pat-1",
            age_years=61.25,
            gender="female",
            matched_steps=["hba1c", "meds"],
            evidence=[_evidence("pat-1", "o1"), _evidence("pat-1", "o2")],
        ),
        PatientMatch(patient_id="pat-2", age_years=None, gender=None, matched_steps=["hba1c"]),
    ]
    result = [
        PatientAnalysis(
            patient_id="pat-1",
            risk=RiskAssessment(score=0.72, band=RiskBand.HIGH, model_name="m", model_version="1"),
            abnormal_labs=[
                AbnormalLabFinding(
                    concept=concept, flag=AbnormalFlag.HIGH, evidence=_evidence("pat-1", "o1")
                )
                for concept in ("potassium", "hba1c", "hba1c")
            ],
        )
    ]
    return QueryResponse(
        question=QUESTION,
        query_plan=QueryPlan(question=QUESTION),
        fhir_queries=QUERIES,
        patients=patients,
        analysis_type=AnalysisType.RISK_STRATIFICATION if analyses else AnalysisType.NONE,
        analyses=result if analyses else [],
    )


def _rows(text: str) -> list[list[str]]:
    return list(csv.reader(io.StringIO(text)))


# -- CSV --------------------------------------------------------------------------


def test_csv_has_a_header_and_one_row_per_patient() -> None:
    rows = _rows(to_csv(_response()))
    assert rows[0] == list(BASE_COLUMNS)
    assert rows[1] == ["pat-1", "61.25", "female", "hba1c;meds", "2"]
    assert rows[2] == ["pat-2", "", "", "hba1c", "0"]
    assert len(rows) == 3


def test_csv_adds_analysis_columns_only_when_analyses_exist() -> None:
    rows = _rows(to_csv(_response(analyses=True)))
    assert rows[0] == list(BASE_COLUMNS + ANALYSIS_COLUMNS)
    assert rows[1][-3:] == ["0.72", "high", "hba1c;potassium"]
    assert rows[2][-3:] == ["", "", ""]


def test_csv_neutralises_formula_injection() -> None:
    hostile = PatientMatch(
        patient_id='=HYPERLINK("http://evil")',
        gender="+cmd|' /C calc'!A0",
        matched_steps=["-2+3", "ok"],
    )
    others = [
        PatientMatch(patient_id="@SUM(A1)", gender="\tfemale"),
        PatientMatch(patient_id="\rpat", gender="male"),
    ]
    rows = _rows(to_csv(_response(patients=[hostile, *others])))
    assert rows[1][0] == '\'=HYPERLINK("http://evil")'
    assert rows[1][2] == "'+cmd|' /C calc'!A0"
    assert rows[1][3] == "'-2+3;ok"
    assert rows[2][0] == "'@SUM(A1)" and rows[2][2] == "'\tfemale"
    assert rows[3][0] == "'\rpat"
    for row in rows[1:]:
        assert not any(cell.startswith(("=", "+", "-", "@", "\t", "\r")) for cell in row)


def test_empty_response_is_header_only() -> None:
    empty = QueryResponse(question="weather?", query_plan=QueryPlan(question="weather?"))
    assert _rows(to_csv(empty)) == [list(BASE_COLUMNS)]


# -- FHIR -------------------------------------------------------------------------


def test_group_is_a_valid_actual_person_group() -> None:
    group = to_fhir_group(_response())
    assert group["resourceType"] == "Group"
    assert group["type"] == "person" and group["actual"] is True
    assert group["quantity"] == len(group["member"]) == 2
    assert [m["entity"]["reference"] for m in group["member"]] == ["Patient/pat-1", "Patient/pat-2"]
    assert "characteristic" not in group
    assert QUESTION in group["name"]
    assert group["text"]["status"] == "generated"
    assert group["text"]["div"].startswith('<div xmlns="http://www.w3.org/1999/xhtml">')
    assert any(t["code"] == "HTEST" for t in group["meta"]["tag"])
    executed = [
        e["valueString"] for e in group["extension"] if e["url"] == EXECUTED_QUERY_EXTENSION
    ]
    assert executed == QUERIES


def test_group_narrative_is_escaped() -> None:
    question = "Which <script>alert(1)</script> patients?"
    response = _response().model_copy(update={"question": question})
    assert "<script>" not in to_fhir_group(response)["text"]["div"]


def test_empty_group_has_quantity_zero() -> None:
    empty = QueryResponse(question="weather?", query_plan=QueryPlan(question="weather?"))
    group = to_fhir_group(empty)
    assert group["quantity"] == 0 and group["member"] == []


def test_bundle_is_a_collection_holding_the_group() -> None:
    bundle = to_fhir_bundle(_response())
    assert bundle["resourceType"] == "Bundle" and bundle["type"] == "collection"
    (entry,) = bundle["entry"]
    assert entry["resource"]["resourceType"] == "Group"
    assert entry["fullUrl"] == f"urn:uuid:{entry['resource']['id']}"
    assert bundle["id"] != entry["resource"]["id"]


def test_exports_are_deterministic() -> None:
    assert to_csv(_response(analyses=True)) == to_csv(_response(analyses=True))
    assert json.dumps(to_fhir_group(_response())) == json.dumps(to_fhir_group(_response()))
    assert json.dumps(to_fhir_bundle(_response())) == json.dumps(to_fhir_bundle(_response()))


def test_ids_change_with_the_cohort() -> None:
    fewer = _response(patients=[PatientMatch(patient_id="pat-1")])
    assert to_fhir_group(fewer)["id"] != to_fhir_group(_response())["id"]


# -- API --------------------------------------------------------------------------

BODY = {"question": QUESTION, "as_of": "2026-01-01"}


@pytest.fixture
def api(dataset: SyntheticDataset, audit: InMemoryAuditSink) -> Iterator[TestClient]:
    server = InMemoryFHIRServer(resources=dataset.resources, read_only=True)
    settings = Settings(
        log_json=False,
        log_level="ERROR",
        fhir=FHIRSettings(base_url=server.base_url, max_retries=0),
    )
    app = create_app(
        settings,
        client=server.client(settings.fhir, audit_sink=audit),
        provider=MockLLMProvider(),
        audit=audit,
    )
    with TestClient(app) as client:
        yield client


def test_csv_export_is_a_download(api: TestClient, audit: InMemoryAuditSink) -> None:
    response = api.post("/query/export", params={"format": "csv"}, json=BODY)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert response.headers["content-disposition"] == "attachment; filename=cohort.csv"
    rows = _rows(response.text)
    assert rows[0][: len(BASE_COLUMNS)] == list(BASE_COLUMNS) and len(rows) > 1
    exported = [e for e in audit.events if e.action == "response.exported"]
    assert exported and exported[-1].details["export"] == "csv"


def test_csv_is_the_default_format(api: TestClient) -> None:
    response = api.post("/query/export", json=BODY)
    assert response.headers["content-type"].startswith("text/csv")


def test_group_export_matches_the_query_answer(api: TestClient) -> None:
    answer = api.post("/query", json=BODY).json()
    response = api.post("/query/export", params={"format": "group"}, json=BODY)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/fhir+json")
    group = response.json()
    assert group["resourceType"] == "Group"
    assert group["quantity"] == len(answer["patients"]) == len(group["member"])
    assert {m["entity"]["reference"] for m in group["member"]} == {
        f"Patient/{p['patient_id']}" for p in answer["patients"]
    }
    executed = [
        e["valueString"] for e in group["extension"] if e["url"] == EXECUTED_QUERY_EXTENSION
    ]
    assert executed == answer["fhir_queries"]


def test_bundle_export_is_deterministic(api: TestClient) -> None:
    first = api.post("/query/export", params={"format": "bundle"}, json=BODY)
    second = api.post("/query/export", params={"format": "bundle"}, json=BODY)
    assert first.status_code == 200
    assert first.headers["content-type"].startswith("application/fhir+json")
    assert first.json()["type"] == "collection"
    assert first.content == second.content


def test_unsupported_question_exports_empty(api: TestClient) -> None:
    body = {"question": "What is the weather tomorrow?"}
    csv_response = api.post("/query/export", params={"format": "csv"}, json=body)
    assert csv_response.status_code == 200
    assert _rows(csv_response.text) == [list(BASE_COLUMNS)]
    group = api.post("/query/export", params={"format": "group"}, json=body).json()
    assert group["quantity"] == 0 and group["member"] == []


def test_export_validates_its_inputs(api: TestClient) -> None:
    assert api.post("/query/export", params={"format": "xlsx"}, json=BODY).status_code == 422
    assert api.post("/query/export", json={"question": ""}).status_code == 422
