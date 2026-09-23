"""``POST /query/export``: the answer to a question as a file.

Runs exactly the same pipeline as ``POST /query`` and renders the result with
:mod:`fhir_healthcare_ai.export`. Kept out of :mod:`fhir_healthcare_ai.api.main` so the
export surface can evolve without touching the core routes.
"""

from __future__ import annotations

import json
from typing import Annotated, Literal

from fastapi import APIRouter, Query, Request, Response

from fhir_healthcare_ai.api.main import _as_of, _pipeline_http_error, _state
from fhir_healthcare_ai.api.schemas import ErrorResponse, QueryRequest
from fhir_healthcare_ai.audit import AuditEvent
from fhir_healthcare_ai.domain.query import QueryPlan
from fhir_healthcare_ai.domain.results import QueryResponse
from fhir_healthcare_ai.export import (
    CSV_MEDIA_TYPE,
    FHIR_MEDIA_TYPE,
    to_csv,
    to_fhir_bundle,
    to_fhir_group,
)
from fhir_healthcare_ai.pipeline.orchestrator import PipelineError

ExportFormat = Literal["csv", "group", "bundle"]

CSV_FILENAME = "cohort.csv"

router = APIRouter(tags=["clinical"])


@router.post(
    "/query/export",
    summary="Answer a question and export the cohort as CSV or FHIR",
    response_class=Response,
    responses={
        200: {
            "description": "The cohort as CSV, a FHIR Group, or a FHIR collection Bundle.",
            "content": {CSV_MEDIA_TYPE: {}, FHIR_MEDIA_TYPE: {}},
        },
        503: {"model": ErrorResponse},
    },
)
async def export_query(
    body: QueryRequest,
    request: Request,
    export_format: Annotated[ExportFormat, Query(alias="format")] = "csv",
) -> Response:
    """Same pipeline as ``/query``; the result is rendered for another tool.

    A question the planner refuses, or one that produced no valid plan, exports as an
    empty cohort (header-only CSV, Group with quantity 0) rather than an error: a
    scheduled export job should get a well-formed, empty file it can diff. A planner
    outage is still a 503, because an empty file there would be a false "nobody".
    """
    state = _state(request)
    try:
        response = await state.orchestrator.answer(
            body.question,
            as_of=_as_of(body.as_of),
            context=body.context,
            narrate=False,
        )
    except PipelineError as exc:
        error = _pipeline_http_error(exc)
        if error.status_code == 503:
            raise error from exc
        response = _refused(body.question, str(exc))

    state.audit.record(
        AuditEvent(
            action="response.exported",
            query=body.question,
            patient_ids=[p.patient_id for p in response.patients][:50],
            reason=response.query_plan.unsupported_reason,
            details={"export": export_format, "patients": len(response.patients)},
        )
    )

    if export_format == "csv":
        return Response(
            content=to_csv(response),
            media_type=CSV_MEDIA_TYPE,
            headers={"Content-Disposition": f"attachment; filename={CSV_FILENAME}"},
        )
    resource = to_fhir_group(response) if export_format == "group" else to_fhir_bundle(response)
    return Response(content=json.dumps(resource, ensure_ascii=False), media_type=FHIR_MEDIA_TYPE)


def _refused(question: str, reason: str) -> QueryResponse:
    """An empty answer that still records why it is empty."""
    plan = QueryPlan(question=question, unsupported=True, unsupported_reason=reason[:2000])
    return QueryResponse(question=question, query_plan=plan, warnings=[reason])
