"""FastAPI application.

    uvicorn fhir_healthcare_ai.api.main:app

Two modes, both behind the same fixed pipeline:

* ``POST /query`` -- a natural-language question is planned, validated, executed and
  answered with evidence (mode 1: retrieval; mode 2 when the plan asks for analytics).
* ``GET /patient/{id}/analyze`` -- a fixed single-patient analysis. No model involved.

Everything else is operational: health, capabilities, and a debug view of the audit
trail that is switched off in production.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request, Response
from fastapi.responses import JSONResponse

from fhir_healthcare_ai import __version__
from fhir_healthcare_ai.api import ui
from fhir_healthcare_ai.api.schemas import (
    PATIENT_ID_PATTERN,
    CapabilitiesResponse,
    ConceptDescription,
    ErrorResponse,
    FeatureDescription,
    FHIRStatus,
    HealthResponse,
    Limits,
    LLMStatus,
    QueryRequest,
)
from fhir_healthcare_ai.api.security import RATE_LIMITED, authenticate, install_security
from fhir_healthcare_ai.audit import AuditEvent, AuditSink, InMemoryAuditSink, build_audit_sink
from fhir_healthcare_ai.audit.log import CompositeAuditSink
from fhir_healthcare_ai.config import Settings, get_settings
from fhir_healthcare_ai.domain.results import PatientAnalysis, QueryResponse
from fhir_healthcare_ai.features.builder import describe_features
from fhir_healthcare_ai.fhir.allowlist import describe_allowlist
from fhir_healthcare_ai.fhir.client import FHIRClient
from fhir_healthcare_ai.fhir.memory import InMemoryFHIRServer
from fhir_healthcare_ai.llm.base import LLMError, LLMProvider
from fhir_healthcare_ai.llm.factory import ProviderStatus, is_local, resolve_provider
from fhir_healthcare_ai.logging_config import configure_logging, correlation_id_var, get_logger
from fhir_healthcare_ai.pipeline.orchestrator import (
    PatientNotFoundError,
    PipelineError,
    PipelineOrchestrator,
)
from fhir_healthcare_ai.synthetic.generator import SyntheticGenerator
from fhir_healthcare_ai.terminology import ConceptKind, concepts_by_kind

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID = re.compile(r"^[A-Za-z0-9\-_.]{1,64}$")

#: Questions the deterministic planner answers. Shown by ``/capabilities`` and the docs.
EXAMPLE_QUESTIONS: tuple[str, ...] = (
    "Which patients with elevated HbA1c had a recent medication change?",
    "Which diabetic patients are not on a statin?",
    "Which diabetic patients are at highest risk of deterioration?",
    "Which patients have uncontrolled blood pressure?",
    "Which patients have reduced kidney function?",
    "Which patients had abnormal potassium results?",
    "Which patients are on an SGLT2 inhibitor?",
    "Which diabetic patients are on insulin?",
    "Which patients have chronic kidney disease?",
    "Which diabetic patients are older than 65?",
    "Which patients have LDL above 160?",
    "Which patients had an emergency visit in the last year?",
    "Which patients with heart failure were admitted in the last 6 months?",
    "Which hypertensive patients are not on any antihypertensive?",
    "Summarize the record of patient syn42-pat-0001",
)


@dataclass
class AppState:
    """Everything a request needs, built once per process."""

    settings: Settings
    orchestrator: PipelineOrchestrator
    client: FHIRClient
    provider: LLMProvider
    provider_status: ProviderStatus
    audit: AuditSink
    audit_buffer: InMemoryAuditSink | None


def create_app(
    settings: Settings | None = None,
    *,
    client: FHIRClient | None = None,
    provider: LLMProvider | None = None,
    audit: AuditSink | None = None,
) -> FastAPI:
    """Build the application.

    Every dependency can be injected, which is how the tests run the real routes
    against an in-memory FHIR server without monkeypatching. Whatever is injected is
    owned by the caller and is not closed on shutdown.
    """
    resolved = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(resolved.log_level, resolved.log_json)
        state = await _build_state(resolved, client=client, provider=provider, audit=audit)
        app.state.pipeline = state
        logger.info(
            "api ready",
            extra={
                "environment": resolved.environment,
                "fhir_base_url": state.client.settings.base_url,
                "llm_provider": state.provider_status.active,
                "llm_fallback": state.provider_status.fallback,
            },
        )
        try:
            yield
        finally:
            if client is None:
                await state.client.aclose()
            if provider is None:
                await state.provider.aclose()

    app = FastAPI(
        title="fhir-healthcare-ai",
        version=__version__,
        description=(
            "A governed AI layer over HL7 FHIR R4. Questions are planned by a model, "
            "validated against an allowlist, executed read-only and answered with "
            "evidence. **Research and engineering demonstration only -- not for clinical "
            "use.**"
        ),
        lifespan=lifespan,
        responses={500: {"model": ErrorResponse}},
        # App-wide, so every route (including ones added later) is authenticated unless
        # security.PUBLIC_ROUTES exempts it.
        dependencies=[Depends(authenticate)],
    )
    install_security(app, resolved.security)
    app.middleware("http")(_correlation_middleware)
    _register_routes(app)
    app.include_router(ui.router)
    from fhir_healthcare_ai.api.export_routes import router as export_router  # imports this module

    app.include_router(export_router)
    return app


async def _build_state(
    settings: Settings,
    *,
    client: FHIRClient | None,
    provider: LLMProvider | None,
    audit: AuditSink | None,
) -> AppState:
    buffer = None
    if audit is None:
        audit = build_audit_sink(settings.audit_log_path, keep_in_memory=True)
    if isinstance(audit, InMemoryAuditSink):
        buffer = audit
    elif isinstance(audit, CompositeAuditSink):
        buffer = next((s for s in audit.sinks if isinstance(s, InMemoryAuditSink)), None)

    if client is None:
        client = _build_client(settings, audit)

    if provider is None:
        provider, status = await resolve_provider(settings.llm)
    else:
        status = ProviderStatus(
            configured=provider.name,
            active=provider.name,
            model=str(getattr(provider, "model", provider.name)),
            local=is_local(provider.name),
        )

    orchestrator = PipelineOrchestrator(provider, client, settings=settings, audit_sink=audit)
    return AppState(
        settings=settings,
        orchestrator=orchestrator,
        client=client,
        provider=provider,
        provider_status=status,
        audit=audit,
        audit_buffer=buffer,
    )


def _build_client(settings: Settings, audit: AuditSink) -> FHIRClient:
    if not settings.fhir.in_memory:
        return FHIRClient(settings.fhir, audit_sink=audit)
    dataset = SyntheticGenerator(
        patients=settings.fhir.in_memory_patients, seed=settings.fhir.in_memory_seed
    ).generate()
    server = InMemoryFHIRServer(resources=dataset.resources, read_only=True)
    logger.warning(
        "serving an in-memory synthetic population; FHIR_BASE_URL is ignored",
        extra={"patients": dataset.patient_count, "resources": len(dataset.resources)},
    )
    fhir_settings = settings.fhir.model_copy(update={"base_url": server.base_url})
    return server.client(fhir_settings, audit_sink=audit)


async def _correlation_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Bind a request id to the context so every log line and audit event carries it.

    A caller-supplied ``X-Request-ID`` is honoured when it is well-formed, so a trace can
    be followed from an upstream gateway; anything else is replaced, never echoed.
    """
    supplied = request.headers.get(REQUEST_ID_HEADER)
    request_id = supplied if supplied and _REQUEST_ID.match(supplied) else uuid.uuid4().hex
    token = correlation_id_var.set(request_id)
    try:
        response = await call_next(request)
    finally:
        correlation_id_var.reset(token)
    response.headers[REQUEST_ID_HEADER] = request_id
    return response


def _state(request: Request) -> AppState:
    state: AppState = request.app.state.pipeline
    return state


def _as_of(value: date | None) -> datetime | None:
    return datetime(value.year, value.month, value.day, tzinfo=UTC) if value else None


def _llm_status(status: ProviderStatus) -> LLMStatus:
    return LLMStatus(
        configured=status.configured,
        active=status.active,
        model=status.model,
        local=status.local,
        fallback=status.fallback,
        reason=status.reason,
    )


def _pipeline_http_error(exc: PipelineError) -> HTTPException:
    """Map pipeline failures onto status codes a client can act on.

    A model that cannot be reached is the server's problem (503, retry later); a
    question that produced no valid plan is the request's (422, rephrase it).
    """
    cause: BaseException | None = exc
    while cause is not None:
        if isinstance(cause, LLMError):
            return HTTPException(503, f"the query planner is unavailable: {exc}")
        cause = cause.__cause__
    return HTTPException(422, str(exc))


def _register_routes(app: FastAPI) -> None:
    @app.get("/health/live", tags=["operations"], summary="Liveness probe")
    async def live() -> dict[str, str]:
        """Answers as long as the process is serving. Never touches the FHIR server."""
        return {"status": "ok"}

    @app.get(
        "/health/ready",
        tags=["operations"],
        summary="Readiness probe",
        responses={503: {"model": ErrorResponse}},
    )
    async def ready(request: Request) -> dict[str, str]:
        """503 until the FHIR server answers with a CapabilityStatement."""
        if not await _state(request).client.ping():
            raise HTTPException(503, "FHIR server is not reachable")
        return {"status": "ready"}

    @app.get("/health", response_model=HealthResponse, tags=["operations"])
    async def health(request: Request) -> HealthResponse:
        """Dependency status, including whether the planner has fallen back to mock."""
        state = _state(request)
        reachable = await state.client.ping()
        if not reachable:
            status = "unavailable"
        elif state.provider_status.fallback:
            status = "degraded"
        else:
            status = "ok"
        return HealthResponse(
            status=status,
            version=__version__,
            environment=state.settings.environment,
            fhir=FHIRStatus(
                base_url=state.client.settings.base_url,
                reachable=reachable,
                in_memory=state.settings.fhir.in_memory,
            ),
            llm=_llm_status(state.provider_status),
        )

    @app.get("/capabilities", response_model=CapabilitiesResponse, tags=["operations"])
    async def capabilities(request: Request) -> CapabilitiesResponse:
        """The allowlist, concept vocabulary and feature contract this deployment runs."""
        state = _state(request)
        fhir = state.settings.fhir
        return CapabilitiesResponse(
            resources=describe_allowlist(),
            concepts={
                kind.value: [
                    ConceptDescription(
                        key=c.key, display=c.display, unit=c.canonical_unit, drug_class=c.drug_class
                    )
                    for c in concepts_by_kind(kind)
                ]
                for kind in ConceptKind
            },
            features=[
                FeatureDescription(
                    name=spec.name, dtype=spec.dtype, description=spec.description, group=spec.group
                )
                for spec in describe_features()
            ],
            example_questions=list(EXAMPLE_QUESTIONS),
            limits=Limits(
                max_patients_per_response=state.settings.max_patients_per_response,
                max_page_size=fhir.max_page_size,
                max_pages=fhir.max_pages,
                max_total_resources=fhir.max_total_resources,
            ),
            llm=_llm_status(state.provider_status),
        )

    @app.post(
        "/query",
        response_model=QueryResponse,
        tags=["clinical"],
        summary="Answer a natural-language clinical question",
        responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
        dependencies=[RATE_LIMITED],
    )
    async def query(body: QueryRequest, request: Request) -> QueryResponse:
        """Plan, validate, retrieve and analyse. Every claim carries its FHIR evidence."""
        state = _state(request)
        try:
            return await state.orchestrator.answer(
                body.question,
                as_of=_as_of(body.as_of),
                context=body.context,
                narrate=body.narrate,
            )
        except PipelineError as exc:
            raise _pipeline_http_error(exc) from exc

    @app.get(
        "/patient/{patient_id}/analyze",
        response_model=PatientAnalysis,
        tags=["clinical"],
        summary="Analyse one patient's record",
        responses={404: {"model": ErrorResponse}, 502: {"model": ErrorResponse}},
        dependencies=[RATE_LIMITED],
    )
    async def analyze_patient(
        request: Request,
        patient_id: Annotated[str, Path(pattern=PATIENT_ID_PATTERN)],
        as_of: Annotated[date | None, Query()] = None,
    ) -> PatientAnalysis:
        """Features, abnormal labs and a risk estimate. Fixed retrieval; no model."""
        state = _state(request)
        try:
            return await state.orchestrator.analyze_patient(patient_id, as_of=_as_of(as_of))
        except PatientNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except PipelineError as exc:
            raise HTTPException(502, f"the FHIR server could not be read: {exc}") from exc

    @app.get(
        "/audit",
        response_model=list[AuditEvent],
        tags=["operations"],
        summary="Recent audit events (disabled in production)",
        responses={404: {"model": ErrorResponse}},
    )
    async def audit_events(
        request: Request,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
        correlation_id: Annotated[str | None, Query(max_length=64)] = None,
    ) -> list[AuditEvent]:
        """The in-process tail of the audit trail, newest last.

        A debugging aid for local and dev environments only. In production the audit
        trail belongs in the JSONL file or a real audit store, not behind an endpoint.
        """
        state = _state(request)
        if state.settings.environment == "prod" or state.audit_buffer is None:
            raise HTTPException(404, "not available in this environment")
        events = list(state.audit_buffer.events)
        if correlation_id:
            events = [e for e in events if e.correlation_id == correlation_id]
        return events[-limit:]

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "correlation_id": correlation_id_var.get()},
            headers=exc.headers,
        )


app = create_app()
