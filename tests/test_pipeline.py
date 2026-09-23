"""The whole workflow against a seeded population, checked against independent truth."""

from __future__ import annotations

import pytest

from fhir_healthcare_ai.audit import InMemoryAuditSink
from fhir_healthcare_ai.benchmark.cases import (
    Population,
    abnormal_potassium,
    diabetes_without_statin,
    diabetic_cohort,
    diabetic_older_than_65,
    elevated_hba1c,
    elevated_hba1c_recent_med_change,
)
from fhir_healthcare_ai.config import Settings
from fhir_healthcare_ai.fhir.client import FHIRClient
from fhir_healthcare_ai.fhir.memory import InMemoryFHIRServer
from fhir_healthcare_ai.llm.mock import MockLLMProvider
from fhir_healthcare_ai.logging_config import correlation_id_var
from fhir_healthcare_ai.pipeline.orchestrator import PatientNotFoundError, PipelineOrchestrator
from fhir_healthcare_ai.synthetic.generator import SyntheticDataset
from tests.conftest import AS_OF, AS_OF_DATE


@pytest.fixture(scope="module")
def population(dataset: SyntheticDataset) -> Population:
    return Population.from_resources(dataset.resources, AS_OF_DATE)


async def _cohort(orchestrator: PipelineOrchestrator, question: str) -> set[str]:
    response = await orchestrator.answer(question, as_of=AS_OF, narrate=False)
    return {p.patient_id for p in response.patients}


async def test_dependent_filter_narrows_to_the_medication_change(
    orchestrator: PipelineOrchestrator, population: Population
) -> None:
    both = await _cohort(
        orchestrator, "Which patients with elevated HbA1c had a recent medication change?"
    )
    only_a1c = await _cohort(orchestrator, "Which patients have elevated HbA1c?")
    assert both == elevated_hba1c_recent_med_change(population)
    assert only_a1c == elevated_hba1c(population)
    assert both < only_a1c


async def test_negation_excludes_patients_on_an_active_statin(
    orchestrator: PipelineOrchestrator, population: Population
) -> None:
    cohort = await _cohort(orchestrator, "Which diabetic patients are not on a statin?")
    assert cohort == diabetes_without_statin(population)


async def test_demographic_filter_reads_only_the_parent_cohort(
    orchestrator: PipelineOrchestrator, server: InMemoryFHIRServer, population: Population
) -> None:
    cohort = await _cohort(orchestrator, "Which diabetic patients are older than 65?")
    assert cohort == diabetic_older_than_65(population)
    assert cohort < diabetic_cohort(population)
    patient_reads = [r for r in server.requests if r.startswith("GET Patient?")]
    assert patient_reads and all("_id=" in r for r in patient_reads)


async def test_abnormal_screening_uses_reference_intervals(
    orchestrator: PipelineOrchestrator, population: Population
) -> None:
    response = await orchestrator.answer(
        "Which patients had abnormal potassium results?", as_of=AS_OF, narrate=False
    )
    assert {p.patient_id for p in response.patients} == abnormal_potassium(population)
    for analysis in response.analyses:
        assert any(f.concept == "potassium" for f in analysis.abnormal_labs)


async def test_every_match_carries_evidence(orchestrator: PipelineOrchestrator) -> None:
    response = await orchestrator.answer(
        "Which patients have elevated HbA1c?", as_of=AS_OF, narrate=False
    )
    assert response.patients
    for match in response.patients:
        assert match.evidence
        assert all(e.patient_id == match.patient_id for e in match.evidence)
    assert all(q.startswith(("Observation?", "Patient?")) for q in response.fhir_queries)


async def test_cohort_matches_carry_demographics(
    orchestrator: PipelineOrchestrator, dataset: SyntheticDataset
) -> None:
    """A cohort selected by Observations still reports each patient's age and sex."""
    response = await orchestrator.answer(
        "Which patients have elevated HbA1c?", as_of=AS_OF, narrate=False
    )
    genders = {r["id"]: r["gender"] for r in dataset.resources if r["resourceType"] == "Patient"}
    assert response.patients
    for match in response.patients:
        assert match.age_years is not None
        assert match.gender == genders[match.patient_id]


async def test_demographics_fetch_is_scoped_to_the_cohort(
    orchestrator: PipelineOrchestrator,
) -> None:
    response = await orchestrator.answer(
        "Which patients have elevated HbA1c?", as_of=AS_OF, narrate=False
    )
    patient_queries = [q for q in response.fhir_queries if q.startswith("Patient?")]
    assert patient_queries, "demographics were not fetched"
    requested: set[str] = set()
    for query in patient_queries:
        params = dict(part.split("=", 1) for part in query.split("?", 1)[1].split("&"))
        assert set(params) == {"_id", "_count"}
        requested |= set(params["_id"].split(","))
    assert requested == {p.patient_id for p in response.patients}


async def test_demographics_are_not_refetched_when_the_plan_has_them(
    orchestrator: PipelineOrchestrator,
) -> None:
    response = await orchestrator.answer(
        "Summarize the record of patient syn7-pat-0001", as_of=AS_OF, narrate=False
    )
    assert sum(q.startswith("Patient?") for q in response.fhir_queries) == 1
    assert response.patients[0].gender is not None


async def test_response_cap_limits_what_is_shown(
    client: FHIRClient, settings: Settings, audit: InMemoryAuditSink
) -> None:
    capped = settings.model_copy(update={"max_patients_per_response": 3})
    orchestrator = PipelineOrchestrator(
        MockLLMProvider(), client, settings=capped, audit_sink=audit, narrate=False
    )
    response = await orchestrator.answer(
        "Which diabetic patients are at highest risk of deterioration?", as_of=AS_OF
    )
    assert len(response.patients) == 3
    assert response.trace is not None and response.trace.truncated
    assert any("capped at 3" in w for w in response.warnings)


async def test_unsupported_question_touches_no_data(
    orchestrator: PipelineOrchestrator, audit: InMemoryAuditSink
) -> None:
    response = await orchestrator.answer("What's the weather tomorrow?", as_of=AS_OF)
    assert response.query_plan.unsupported
    assert response.patients == [] and response.fhir_queries == []
    assert not audit.by_action("query.executed")


async def test_patient_analysis(orchestrator: PipelineOrchestrator) -> None:
    analysis = await orchestrator.analyze_patient("syn7-pat-0001", as_of=AS_OF)
    assert analysis.patient_id == "syn7-pat-0001"
    assert analysis.age_years is not None
    assert analysis.risk is not None and 0.0 <= analysis.risk.score <= 1.0
    with pytest.raises(PatientNotFoundError):
        await orchestrator.analyze_patient("no-such-patient", as_of=AS_OF)


async def test_audit_events_share_the_request_correlation_id(
    orchestrator: PipelineOrchestrator, audit: InMemoryAuditSink
) -> None:
    token = correlation_id_var.set("req-42")
    try:
        await orchestrator.answer("Which patients have elevated HbA1c?", as_of=AS_OF)
    finally:
        correlation_id_var.reset(token)
    actions = [e.action for e in audit.events if e.correlation_id == "req-42"]
    assert actions[0] == "plan.generated"
    assert "query.executed" in actions
    assert actions[-1] == "response.generated"
