"""Shared fixtures.

Everything runs against :class:`InMemoryFHIRServer` loaded with a seeded synthetic
population, through the real :class:`FHIRClient`. No network, no container, no API key.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime

import pytest

from fhir_healthcare_ai.audit import InMemoryAuditSink
from fhir_healthcare_ai.config import FHIRSettings, Settings
from fhir_healthcare_ai.fhir.client import FHIRClient
from fhir_healthcare_ai.fhir.memory import InMemoryFHIRServer
from fhir_healthcare_ai.llm.mock import MockLLMProvider
from fhir_healthcare_ai.pipeline.orchestrator import PipelineOrchestrator
from fhir_healthcare_ai.synthetic.generator import SyntheticDataset, SyntheticGenerator

AS_OF_DATE = date(2026, 1, 1)
AS_OF = datetime(2026, 1, 1, tzinfo=UTC)
PATIENTS = 60
SEED = 7


@pytest.fixture(autouse=True)
def _quiet_logs() -> None:
    logging.getLogger().setLevel(logging.ERROR)


@pytest.fixture(scope="session")
def dataset() -> SyntheticDataset:
    return SyntheticGenerator(patients=PATIENTS, seed=SEED, as_of=AS_OF_DATE).generate()


@pytest.fixture
def server(dataset: SyntheticDataset) -> InMemoryFHIRServer:
    return InMemoryFHIRServer(resources=dataset.resources, read_only=True)


@pytest.fixture
def settings(server: InMemoryFHIRServer) -> Settings:
    return Settings(
        max_patients_per_response=1000,
        fhir=FHIRSettings(base_url=server.base_url, max_retries=0),
    )


@pytest.fixture
def audit() -> InMemoryAuditSink:
    return InMemoryAuditSink(max_events=10_000)


@pytest.fixture
async def client(
    server: InMemoryFHIRServer, settings: Settings, audit: InMemoryAuditSink
) -> AsyncIterator[FHIRClient]:
    async with server.client(settings.fhir, audit_sink=audit) as fhir_client:
        yield fhir_client


@pytest.fixture
def orchestrator(
    client: FHIRClient, settings: Settings, audit: InMemoryAuditSink
) -> PipelineOrchestrator:
    return PipelineOrchestrator(
        MockLLMProvider(), client, settings=settings, audit_sink=audit, narrate=False
    )
