"""The workflow. One fixed sequence, no autonomy.

    Planner -> Query Builder -> Validator -> FHIR Tool -> Normalizer
            -> Features -> Analyzer -> Response Generator

This is deliberately *not* an agent. The model is called exactly twice in the worst
case -- once to plan, once to narrate -- and it never decides what happens next. Control
flow lives here, in Python, where it can be read, tested and reasoned about. A model
that could choose its own next tool call could also choose to loop, to widen a query
after it was validated, or to fetch a resource nobody allowed; none of those failure
modes are reachable from a fixed pipeline.

The model never sees a URL and never produces one. It emits a :class:`QueryPlan`, and
the only component that can turn a plan into an HTTP request is the query builder, which
re-validates first.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from fhir_healthcare_ai.analytics.abnormal_labs import AbnormalLabDetector
from fhir_healthcare_ai.analytics.cohort import (
    CohortResolver,
    StepResult,
    summarize_cohort,
)
from fhir_healthcare_ai.analytics.risk import RiskStratifier
from fhir_healthcare_ai.audit import AuditEvent, AuditSink, LoggingAuditSink
from fhir_healthcare_ai.config import Settings, get_settings
from fhir_healthcare_ai.domain.clinical import PatientRecord
from fhir_healthcare_ai.domain.enums import AnalysisType, QueryIntent, ResourceType, StepRole
from fhir_healthcare_ai.domain.query import QueryPlan, QueryStep, SearchParam
from fhir_healthcare_ai.domain.results import (
    ExecutedQuery,
    ExecutionTrace,
    PatientAnalysis,
    QueryResponse,
)
from fhir_healthcare_ai.features.builder import FeatureBuilder, FeatureSet
from fhir_healthcare_ai.fhir.client import FHIRClient, FHIRError
from fhir_healthcare_ai.fhir.parsers.primitives import parse_reference_id
from fhir_healthcare_ai.fhir.query_builder import FHIRQueryBuilder, QueryBuildError
from fhir_healthcare_ai.llm.base import LLMProvider
from fhir_healthcare_ai.logging_config import correlation_id_var, get_logger
from fhir_healthcare_ai.normalization.assembler import RecordAssembler
from fhir_healthcare_ai.pipeline.planner import PlanningError, PlanningResult, QueryPlanner
from fhir_healthcare_ai.pipeline.response import ResponseGenerator

logger = get_logger(__name__)

#: ``analysis.options`` key: keep only patients with an abnormal result for the concepts.
REQUIRE_ABNORMAL = "require_abnormal"

#: Resources pulled for a single-patient analysis. Ordered so the Patient arrives first
#: and demographics are available to everything that follows.
PATIENT_ANALYSIS_RESOURCES: tuple[ResourceType, ...] = (
    ResourceType.PATIENT,
    ResourceType.OBSERVATION,
    ResourceType.CONDITION,
    ResourceType.MEDICATION_REQUEST,
    ResourceType.ENCOUNTER,
    ResourceType.DIAGNOSTIC_REPORT,
)


class PipelineError(RuntimeError):
    """The pipeline could not produce an answer."""


class PatientNotFoundError(PipelineError):
    """The FHIR server holds no data for the requested patient."""


@dataclass
class Retrieval:
    """Everything the retrieval stage produced."""

    executed: list[ExecutedQuery] = field(default_factory=list)
    step_results: list[StepResult] = field(default_factory=list)
    resources: list[dict[str, Any]] = field(default_factory=list)
    #: ``"Observation/abc"`` -> the step_id that fetched it, so evidence can name its source.
    step_by_resource: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    truncated: bool = False

    @property
    def resource_count(self) -> int:
        return len(self.resources)


class _Stages:
    """Stopwatch for the execution trace."""

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    @asynccontextmanager
    async def __call__(self, name: str) -> Any:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = (time.perf_counter() - started) * 1000


class PipelineOrchestrator:
    """Runs the fixed workflow for both API modes."""

    def __init__(
        self,
        provider: LLMProvider,
        client: FHIRClient,
        *,
        settings: Settings | None = None,
        planner: QueryPlanner | None = None,
        builder: FHIRQueryBuilder | None = None,
        assembler: RecordAssembler | None = None,
        feature_builder: FeatureBuilder | None = None,
        detector: AbnormalLabDetector | None = None,
        stratifier: RiskStratifier | None = None,
        generator: ResponseGenerator | None = None,
        audit_sink: AuditSink | None = None,
        narrate: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.provider = provider
        self.client = client
        self.audit = audit_sink or LoggingAuditSink()
        self.planner = planner or QueryPlanner(provider, audit_sink=self.audit)
        self.builder = builder or FHIRQueryBuilder(max_page_size=self.settings.fhir.max_page_size)
        self.assembler = assembler or RecordAssembler()
        self.features = feature_builder or FeatureBuilder()
        self.detector = detector or AbnormalLabDetector()
        self.stratifier = stratifier or RiskStratifier()
        self.generator = generator or ResponseGenerator(provider if narrate else None)
        self.max_patients = self.settings.max_patients_per_response

    # -- mode 1 + 2: natural-language question ------------------------------------

    async def answer(
        self,
        question: str,
        *,
        as_of: datetime | None = None,
        context: str | None = None,
        narrate: bool = True,
    ) -> QueryResponse:
        """Answer a clinical question end to end."""
        now = as_of or datetime.now(UTC)
        stage = _Stages()
        warnings: list[str] = []

        async with stage("planning"):
            planning = await self._plan(question, context=context, as_of=now)

        plan = planning.plan
        warnings.extend(planning.warnings)
        warnings.extend(planning.assumptions)

        if plan.unsupported:
            return self.generator.unsupported_response(
                question=question,
                plan=plan,
                trace=self._trace(stage, planning, Retrieval()),
                warnings=warnings,
            )

        async with stage("retrieval"):
            retrieval = await self._retrieve(plan)
        warnings.extend(retrieval.errors)

        async with stage("cohort"):
            resolver = CohortResolver(plan.cohort_logic)
            matched_ids, match_warnings = resolver.resolve(retrieval.step_results)
            warnings.extend(match_warnings)
            ordered_ids = sorted(matched_ids)

        async with stage("normalization"):
            records, stats = self.assembler.assemble(retrieval.resources)
            if stats.unparseable:
                warnings.append(f"{stats.unparseable} resource(s) could not be parsed")

        if plan.analysis.options.get(REQUIRE_ABNORMAL):
            async with stage("screening"):
                ordered_ids = self._screen_abnormal(plan, ordered_ids, records, now, warnings)

        # The cap is applied last, so it limits what is *shown*, never which patients were
        # eligible: capping before screening would drop real matches for arbitrary ones.
        if len(ordered_ids) > self.max_patients:
            warnings.append(
                f"{len(ordered_ids)} patients matched; the response is capped at "
                f"{self.max_patients}. Narrow the question for a complete set."
            )
            retrieval.truncated = True
            ordered_ids = ordered_ids[: self.max_patients]

        async with stage("features"):
            feature_sets = {
                pid: self.features.build(records[pid], now) for pid in ordered_ids if pid in records
            }

        async with stage("analysis"):
            analyses = self._analyze(plan, ordered_ids, records, feature_sets, now, retrieval)

        matches = [
            self.generator.build_match(
                pid,
                records.get(pid),
                matched_steps=resolver.matched_steps(pid, retrieval.step_results),
                analysis=next((a for a in analyses if a.patient_id == pid), None),
                step_by_resource=retrieval.step_by_resource,
            )
            for pid in ordered_ids
        ]

        cohort = summarize_cohort(
            [feature_sets[pid] for pid in ordered_ids if pid in feature_sets],
            records={pid: records[pid] for pid in ordered_ids if pid in records},
            per_step_counts={r.step_id: len(r.patient_ids) for r in retrieval.step_results},
        )

        narrative = None
        if narrate:
            async with stage("narrative"):
                narrative = await self.generator.narrate(
                    question, plan=plan, matches=matches, analyses=analyses
                )

        response = self.generator.build_response(
            question=question,
            plan=plan,
            validation=planning.validation,
            executed=retrieval.executed,
            matches=matches,
            analyses=analyses,
            cohort=cohort,
            trace=self._trace(stage, planning, retrieval),
            warnings=warnings,
            narrative=narrative,
        )
        self._audit_response(question, response, retrieval)
        return response

    # -- mode 2 only: one known patient -------------------------------------------

    async def analyze_patient(
        self, patient_id: str, *, as_of: datetime | None = None
    ) -> PatientAnalysis:
        """Analyze a single patient. No model involved: the plan is fixed."""
        now = as_of or datetime.now(UTC)
        plan = _patient_plan(patient_id, self.settings.fhir.max_page_size)
        retrieval = await self._retrieve(plan)
        if retrieval.errors and not retrieval.resources:
            raise PipelineError("; ".join(retrieval.errors))

        record = self.assembler.assemble_one(patient_id, retrieval.resources)
        if record.resource_count == 0:
            raise PatientNotFoundError(f"no data found for patient {patient_id}")

        features = self.features.build(record, now)
        analysis = self.generator.build_analysis(
            patient_id,
            record,
            features,
            abnormal_labs=self.detector.detect(record, now),
            risk=self.stratifier.assess(features),
            step_by_resource=retrieval.step_by_resource,
        )
        self.audit.record(
            AuditEvent(
                action="analysis.executed",
                resource_type=ResourceType.PATIENT.value,
                patient_ids=[patient_id],
                resource_count=record.resource_count,
                details={"analysis": "patient_summary"},
            )
        )
        return analysis

    # -- stages -------------------------------------------------------------------

    def _screen_abnormal(
        self,
        plan: QueryPlan,
        patient_ids: Sequence[str],
        records: dict[str, PatientRecord],
        now: datetime,
        warnings: list[str],
    ) -> list[str]:
        """Keep only patients with at least one out-of-range result.

        This is how "patients with *abnormal* potassium" is answered: the query fetches
        every potassium result in the window, and whether a value is abnormal is decided
        here against the sex-specific reference interval -- a judgement a
        ``value-quantity`` filter cannot make, because the threshold differs per patient.
        """
        screen = AbnormalLabDetector(
            window_days=plan.analysis.lookback_days or self.detector.window_days,
            concepts=plan.analysis.concepts or None,
        )
        kept = [pid for pid in patient_ids if pid in records and screen.detect(records[pid], now)]
        dropped = len(patient_ids) - len(kept)
        if dropped:
            scope = ", ".join(plan.analysis.concepts) or "any screened concept"
            warnings.append(
                f"{dropped} patient(s) had results but none outside the reference interval "
                f"for {scope}, and were left out"
            )
        return kept

    async def _plan(self, question: str, *, context: str | None, as_of: datetime) -> PlanningResult:
        try:
            return await self.planner.plan(question, context=context, as_of=as_of)
        except PlanningError as exc:
            raise PipelineError(str(exc)) from exc

    async def _retrieve(self, plan: QueryPlan) -> Retrieval:
        """Execute the plan's steps, dependencies last.

        A step that fails is recorded and skipped rather than aborting the request: a
        four-step plan where one resource type is unavailable should still return the
        three that worked, clearly marked, instead of nothing at all.
        """
        retrieval = Retrieval()
        completed: dict[str, set[str]] = {}
        budget = self.settings.fhir.max_total_resources

        for step in _in_dependency_order(plan.steps):
            scope: set[str] | None = None
            if step.depends_on:
                scope = completed.get(step.depends_on)
                if not scope:
                    retrieval.errors.append(
                        f"step {step.step_id!r} was skipped: its parent step "
                        f"{step.depends_on!r} matched no patients"
                    )
                    completed[step.step_id] = set()
                    continue

            try:
                queries = self.builder.build_step(step, patient_ids=scope)
            except QueryBuildError as exc:
                retrieval.errors.append(f"step {step.step_id!r} could not be built: {exc}")
                self._audit_refused(step, str(exc))
                completed[step.step_id] = set()
                continue

            patient_ids: set[str] = set()
            for query in queries:
                if retrieval.resource_count >= budget:
                    retrieval.truncated = True
                    retrieval.errors.append(
                        f"stopped after {budget} resources; results are partial"
                    )
                    break
                try:
                    result = await self.client.search(query)
                except FHIRError as exc:
                    retrieval.errors.append(f"step {step.step_id!r} failed: {exc}")
                    retrieval.executed.append(
                        ExecutedQuery(step_id=step.step_id, query=query, error=str(exc))
                    )
                    continue

                resources = result.all_resources
                step_ids = {
                    pid for pid in (_subject_id(raw) for raw in resources) if pid is not None
                }
                patient_ids |= step_ids
                retrieval.resources.extend(resources)
                for raw in resources:
                    reference = _reference_of(raw)
                    if reference:
                        retrieval.step_by_resource.setdefault(reference, step.step_id)
                retrieval.truncated = retrieval.truncated or result.truncated
                retrieval.executed.append(
                    ExecutedQuery(
                        step_id=step.step_id,
                        query=query,
                        resource_count=len(resources),
                        patient_ids=step_ids,
                        truncated=result.truncated,
                    )
                )

            completed[step.step_id] = patient_ids
            retrieval.step_results.append(
                StepResult(
                    step_id=step.step_id,
                    patient_ids=patient_ids,
                    # The plan says what the step means: a dependent `filter` narrows its
                    # parent ("and a recent medication change"), a `context` step only
                    # gathers data and must not drop patients who lack it, an `exclude`
                    # step removes whoever it returns.
                    role=step.role,
                    truncated=retrieval.truncated,
                )
            )
        return retrieval

    def _analyze(
        self,
        plan: QueryPlan,
        patient_ids: Sequence[str],
        records: dict[str, PatientRecord],
        feature_sets: dict[str, FeatureSet],
        now: datetime,
        retrieval: Retrieval,
    ) -> list[PatientAnalysis]:
        analysis_type = plan.analysis.type
        if analysis_type is AnalysisType.NONE and plan.intent is not QueryIntent.PATIENT_SUMMARY:
            return []

        detector = self.detector
        if plan.analysis.lookback_days:
            detector = AbnormalLabDetector(window_days=plan.analysis.lookback_days)

        analyses: list[PatientAnalysis] = []
        for pid in patient_ids:
            record = records.get(pid)
            features = feature_sets.get(pid)
            labs = None
            risk = None
            if analysis_type in (AnalysisType.ABNORMAL_LABS, AnalysisType.RISK_STRATIFICATION):
                labs = detector.detect(record, now) if record else []
            if analysis_type is AnalysisType.RISK_STRATIFICATION and features is not None:
                risk = self.stratifier.assess(features)
            if plan.intent is QueryIntent.PATIENT_SUMMARY:
                labs = (
                    labs if labs is not None else (detector.detect(record, now) if record else [])
                )
            analyses.append(
                self.generator.build_analysis(
                    pid,
                    record,
                    features,
                    abnormal_labs=labs,
                    risk=risk,
                    step_by_resource=retrieval.step_by_resource,
                )
            )

        self.audit.record(
            AuditEvent(
                action="analysis.executed",
                patient_ids=list(patient_ids)[:50],
                resource_count=retrieval.resource_count,
                details={"analysis": analysis_type.value, "patients": len(analyses)},
            )
        )
        return analyses

    # -- plumbing -----------------------------------------------------------------

    def _trace(
        self, stage: _Stages, planning: PlanningResult, retrieval: Retrieval
    ) -> ExecutionTrace:
        return ExecutionTrace(
            correlation_id=correlation_id_var.get() or "-",
            stages={k: round(v, 2) for k, v in stage.timings.items()},
            fhir_requests=len(retrieval.executed),
            resources_fetched=retrieval.resource_count,
            llm_calls=planning.llm_calls,
            llm_tokens=planning.llm_tokens,
            truncated=retrieval.truncated,
        )

    def _audit_refused(self, step: QueryStep, reason: str) -> None:
        self.audit.record(
            AuditEvent(
                action="query.refused",
                outcome="refused",
                resource_type=step.resource_type.value,
                reason=reason[:2000],
                details={"step": step.step_id},
            )
        )

    def _audit_response(self, question: str, response: QueryResponse, retrieval: Retrieval) -> None:
        self.audit.record(
            AuditEvent(
                action="response.generated",
                query=question,
                patient_ids=[p.patient_id for p in response.patients][:50],
                resource_count=retrieval.resource_count,
                duration_ms=sum(response.trace.stages.values()) if response.trace else None,
                details={
                    "queries": len(retrieval.executed),
                    "analysis": response.analysis_type.value,
                    "truncated": retrieval.truncated,
                },
            )
        )


# -- helpers ----------------------------------------------------------------------


def _in_dependency_order(steps: Sequence[QueryStep]) -> list[QueryStep]:
    """Order steps so every dependency runs before its dependents.

    Steps in a dependency cycle are dropped rather than deadlocking the pipeline. The
    plan validator should have caught it; this is the second line of defence.
    """
    remaining = list(steps)
    done: set[str] = set()
    ordered: list[QueryStep] = []
    while remaining:
        ready = [s for s in remaining if s.depends_on is None or s.depends_on in done]
        if not ready:
            logger.error(
                "dropping steps in a dependency cycle",
                extra={"steps": [s.step_id for s in remaining]},
            )
            break
        for step in ready:
            ordered.append(step)
            done.add(step.step_id)
        remaining = [s for s in remaining if s.step_id not in done]
    return ordered


def _subject_id(raw: dict[str, Any]) -> str | None:
    """The patient a raw resource belongs to, without parsing the whole thing."""
    if not isinstance(raw, dict):
        return None
    if raw.get("resourceType") == ResourceType.PATIENT.value:
        resource_id = raw.get("id")
        return resource_id if isinstance(resource_id, str) else None
    for key in ("subject", "patient"):
        node = raw.get(key)
        if node is not None:
            reference = parse_reference_id(node)
            if reference:
                return reference
    return None


def _reference_of(raw: dict[str, Any]) -> str | None:
    if not isinstance(raw, dict):
        return None
    resource_type = raw.get("resourceType")
    resource_id = raw.get("id")
    if isinstance(resource_type, str) and isinstance(resource_id, str):
        return f"{resource_type}/{resource_id}"
    return None


def _patient_plan(patient_id: str, max_count: int) -> QueryPlan:
    """The fixed retrieval plan behind ``/patient/{id}/analyze``.

    Expressed as a :class:`QueryPlan` rather than as direct client calls so that this
    path goes through the same validator and builder as a model-authored plan. There is
    exactly one way to reach the FHIR server, and it is not bypassable from inside.
    """
    steps = [
        QueryStep(
            step_id="patient",
            resource_type=ResourceType.PATIENT,
            purpose="Demographics.",
            params=[SearchParam(name="_id", values=[patient_id])],
            count=1,
        )
    ]
    for resource_type in PATIENT_ANALYSIS_RESOURCES[1:]:
        steps.append(
            QueryStep(
                step_id=resource_type.value.lower().replace("_", "-"),
                resource_type=resource_type,
                purpose=f"{resource_type.value} history.",
                params=[SearchParam(name="patient", values=[patient_id])],
                sort="-date" if resource_type is ResourceType.OBSERVATION else None,
                count=max_count,
                role=StepRole.CONTEXT,
            )
        )
    return QueryPlan(
        question=f"Summarize the record of patient {patient_id}",
        intent=QueryIntent.PATIENT_SUMMARY,
        rationale="Fixed single-patient retrieval; no model interpretation involved.",
        steps=steps,
    )


def summarize_response(response: QueryResponse) -> dict[str, Any]:
    """Compact view of a response, for logs and the benchmark."""
    return {
        "patients": len(response.patients),
        "queries": len(response.fhir_queries),
        "evidence": len(response.evidence),
        "analysis": response.analysis_type.value,
        "warnings": len(response.warnings),
    }
