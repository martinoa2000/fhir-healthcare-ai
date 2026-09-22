"""Against a real FHIR server: ``make up`` then ``make test-integration``.

Skipped unless ``FHIR_BASE_URL`` answers with a CapabilityStatement, so the default
test run never needs a container.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest

from fhir_healthcare_ai.config import FHIRSettings
from fhir_healthcare_ai.fhir.client import FHIRClient
from fhir_healthcare_ai.llm.mock import MockLLMProvider
from fhir_healthcare_ai.pipeline.orchestrator import PipelineOrchestrator

pytestmark = pytest.mark.integration


@pytest.fixture
async def live_client() -> AsyncIterator[FHIRClient]:
    base_url = os.environ.get("FHIR_BASE_URL", "http://localhost:8080/fhir")
    async with FHIRClient(FHIRSettings(base_url=base_url, timeout_seconds=10)) as client:
        if not await client.ping():
            pytest.skip(f"no FHIR server at {base_url}")
        yield client


async def test_server_is_seeded(live_client: FHIRClient) -> None:
    from fhir_healthcare_ai.domain.enums import ResourceType
    from fhir_healthcare_ai.domain.query import FHIRQuery

    result = await live_client.search(
        FHIRQuery(resource_type=ResourceType.PATIENT, params=(("_count", "5"),)), max_pages=1
    )
    assert result.matches, "the server is up but empty; run `make seed`"


async def test_documented_questions_answer_against_hapi(live_client: FHIRClient) -> None:
    orchestrator = PipelineOrchestrator(MockLLMProvider(), live_client, narrate=False)
    for question in (
        "Which diabetic patients are not on a statin?",
        "Which patients have reduced kidney function?",
    ):
        response = await orchestrator.answer(question)
        assert not response.query_plan.unsupported
        assert response.patients, f"no patients for {question!r}"
