"""Turns retrieved and analyzed state into the response the caller sees.

The rule this module exists to enforce: **no clinical statement without a resource
behind it**. Every patient in a response carries :class:`Evidence` pointing at the
`ResourceType/id` that put them there, so a reviewer can open the source record rather
than trust the summary.

The narrative is optional and always last. It is generated from findings that already
exist, never from the model's own recollection, and if the provider is unavailable the
response is still complete -- prose is a convenience on top of structured evidence, not
the product.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fhir_healthcare_ai.analytics.cohort import DEFAULT_SUMMARY_FEATURES
from fhir_healthcare_ai.domain.clinical import (
    NormalizedCondition,
    NormalizedMedicationRequest,
    NormalizedObservation,
    PatientRecord,
)
from fhir_healthcare_ai.domain.enums import AnalysisType, ResourceType
from fhir_healthcare_ai.domain.query import QueryPlan, ValidationResult
from fhir_healthcare_ai.domain.results import (
    AbnormalLabFinding,
    CohortSummary,
    Evidence,
    ExecutedQuery,
    ExecutionTrace,
    PatientAnalysis,
    PatientMatch,
    QueryResponse,
)
from fhir_healthcare_ai.features.builder import FeatureSet
from fhir_healthcare_ai.llm.base import LLMError, LLMMessage, LLMProvider
from fhir_healthcare_ai.llm.prompts import build_narrative_prompt
from fhir_healthcare_ai.logging_config import get_logger

logger = get_logger(__name__)

#: Cap on evidence rows emitted per patient. A cohort of 200 patients with every
#: resource attached is a payload nobody reads and a response nobody can render.
MAX_EVIDENCE_PER_PATIENT = 6

#: How many patients the narrative is allowed to describe. Beyond this it summarises
#: the cohort instead of listing people.
MAX_NARRATIVE_PATIENTS = 10


class ResponseGenerator:
    """Assembles :class:`QueryResponse` objects from pipeline state."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        *,
        max_evidence_per_patient: int = MAX_EVIDENCE_PER_PATIENT,
        max_narrative_patients: int = MAX_NARRATIVE_PATIENTS,
    ) -> None:
        self.provider = provider
        self.max_evidence_per_patient = max_evidence_per_patient
        self.max_narrative_patients = max_narrative_patients

    # -- patient-level assembly ---------------------------------------------------

    def build_match(
        self,
        patient_id: str,
        record: PatientRecord | None,
        *,
        matched_steps: list[str],
        analysis: PatientAnalysis | None = None,
        step_by_resource: dict[str, str] | None = None,
    ) -> PatientMatch:
        """One row of a cohort result, with the evidence that justifies it."""
        patient = record.patient if record else None
        evidence = (
            list(analysis.evidence)
            if analysis and analysis.evidence
            else self.collect_evidence(patient_id, record, step_by_resource=step_by_resource)
        )
        return PatientMatch(
            patient_id=patient_id,
            matched_steps=matched_steps,
            age_years=patient.age_years if patient else None,
            gender=patient.gender if patient else None,
            summary=self.summarize_patient(record, analysis),
            evidence=evidence[: self.max_evidence_per_patient],
        )

    def summarize_patient(
        self, record: PatientRecord | None, analysis: PatientAnalysis | None = None
    ) -> str | None:
        """A one-line, factual description. No interpretation, no recommendation."""
        parts: list[str] = []
        if record and record.patient:
            demo = _demographics(record.patient.age_years, record.patient.gender)
            if demo:
                parts.append(demo)
        if record:
            active = [c for c in record.conditions if c.is_active]
            if active:
                names = [c.display or c.concept or "condition" for c in active[:3]]
                more = f" (+{len(active) - 3})" if len(active) > 3 else ""
                parts.append(f"active problems: {', '.join(names)}{more}")
            meds = [m for m in record.medication_requests if m.is_active]
            if meds:
                parts.append(f"{len(meds)} active medication order(s)")
        if analysis and analysis.abnormal_labs:
            flags = ", ".join(
                f"{f.display or f.concept} {_format_value(f.value, f.unit)} ({f.flag.value})"
                for f in analysis.abnormal_labs[:3]
            )
            parts.append(f"out-of-range: {flags}")
        if analysis and analysis.risk:
            parts.append(f"model risk band {analysis.risk.band.value}")
        return "; ".join(parts) if parts else None

    def collect_evidence(
        self,
        patient_id: str,
        record: PatientRecord | None,
        *,
        concepts: set[str] | None = None,
        step_by_resource: dict[str, str] | None = None,
    ) -> list[Evidence]:
        """Pick the resources worth citing for a patient.

        Preference order is most-recent-observation, then active conditions, then active
        medication orders: that is the order a reviewer reads a chart in, and it puts the
        resource that usually decided the match first.
        """
        if record is None:
            return []
        steps = step_by_resource or {}
        evidence: list[Evidence] = []

        observations = [o for o in record.observations if o.numeric_value is not None]
        if concepts:
            observations = [o for o in observations if o.concept in concepts] or observations
        observations.sort(key=lambda o: o.effective_datetime or datetime.min, reverse=True)
        seen_concepts: set[str] = set()
        for obs in observations:
            key = obs.concept or obs.id
            if key in seen_concepts:
                continue
            seen_concepts.add(key)
            evidence.append(_observation_evidence(patient_id, obs, steps.get(obs.reference)))
            if len(evidence) >= self.max_evidence_per_patient:
                return evidence

        for condition in record.conditions:
            if not condition.is_active:
                continue
            evidence.append(
                _condition_evidence(patient_id, condition, steps.get(condition.reference))
            )
            if len(evidence) >= self.max_evidence_per_patient:
                return evidence

        for med in record.medication_requests:
            if not med.is_active:
                continue
            evidence.append(_medication_evidence(patient_id, med, steps.get(med.reference)))
            if len(evidence) >= self.max_evidence_per_patient:
                return evidence

        return evidence

    def build_analysis(
        self,
        patient_id: str,
        record: PatientRecord | None,
        features: FeatureSet | None,
        *,
        abnormal_labs: list[AbnormalLabFinding] | None = None,
        risk: Any = None,
        step_by_resource: dict[str, str] | None = None,
    ) -> PatientAnalysis:
        """Per-patient analytics output, with the evidence each finding rests on."""
        findings = abnormal_labs or []
        patient = record.patient if record else None
        evidence = [f.evidence for f in findings]
        if not evidence:
            evidence = self.collect_evidence(patient_id, record, step_by_resource=step_by_resource)
        return PatientAnalysis(
            patient_id=patient_id,
            age_years=patient.age_years if patient else None,
            gender=patient.gender if patient else None,
            features=features.non_null() if features else {},
            abnormal_labs=findings,
            risk=risk,
            evidence=evidence[: self.max_evidence_per_patient],
            data_gaps=_data_gaps(features, risk),
        )

    # -- top-level assembly -------------------------------------------------------

    def build_response(
        self,
        *,
        question: str,
        plan: QueryPlan,
        validation: ValidationResult,
        executed: list[ExecutedQuery],
        matches: list[PatientMatch],
        analyses: list[PatientAnalysis],
        cohort: CohortSummary,
        trace: ExecutionTrace,
        warnings: list[str] | None = None,
        narrative: str | None = None,
    ) -> QueryResponse:
        evidence: list[Evidence] = []
        for match in matches:
            evidence.extend(match.evidence)
        return QueryResponse(
            question=question,
            query_plan=plan,
            fhir_queries=[executed_query.query.relative_url() for executed_query in executed],
            patients=matches,
            evidence=evidence,
            cohort=cohort,
            analysis_type=plan.analysis.type,
            analyses=analyses,
            narrative=narrative,
            validation_issues=list(validation.issues),
            warnings=list(warnings or []),
            trace=trace,
        )

    def unsupported_response(
        self,
        *,
        question: str,
        plan: QueryPlan,
        trace: ExecutionTrace,
        warnings: list[str] | None = None,
    ) -> QueryResponse:
        """A refusal is a valid answer. It still carries the plan that produced it."""
        return QueryResponse(
            question=question,
            query_plan=plan,
            narrative=plan.unsupported_reason,
            warnings=list(warnings or []),
            trace=trace,
        )

    # -- narrative ----------------------------------------------------------------

    async def narrate(
        self,
        question: str,
        *,
        plan: QueryPlan,
        matches: list[PatientMatch],
        analyses: list[PatientAnalysis],
    ) -> str | None:
        """Ask the model to describe findings that already exist.

        Returns ``None`` rather than raising when the provider is unreachable: a missing
        paragraph must not turn a successful retrieval into a failed request.
        """
        if self.provider is None or not matches:
            return None
        findings = _narrative_findings(matches, analyses, self.max_narrative_patients)
        system, user = build_narrative_prompt(
            question,
            plan_summary=_plan_summary(plan),
            findings=findings,
            patient_count=len(matches),
        )
        try:
            response = await self.provider.complete(
                [LLMMessage("system", system), LLMMessage("user", user)],
                temperature=0.0,
                max_tokens=600,
            )
        except LLMError as exc:
            logger.warning("narrative generation skipped", extra={"error": str(exc)})
            return None
        return response.text.strip() or None


# -- evidence constructors --------------------------------------------------------


def _observation_evidence(
    patient_id: str, obs: NormalizedObservation, step_id: str | None
) -> Evidence:
    unit = obs.quantity.canonical_unit or obs.quantity.unit if obs.quantity else None
    return Evidence(
        patient_id=patient_id,
        resource_type=ResourceType.OBSERVATION,
        resource_id=obs.id,
        concept=obs.concept,
        display=obs.display,
        value=_format_value(obs.numeric_value, unit) or obs.value_string,
        effective=obs.effective_datetime,
        source_step=step_id,
        note=obs.abnormal_flag.value if obs.abnormal_flag.value != "unknown" else None,
    )


def _condition_evidence(
    patient_id: str, condition: NormalizedCondition, step_id: str | None
) -> Evidence:
    return Evidence(
        patient_id=patient_id,
        resource_type=ResourceType.CONDITION,
        resource_id=condition.id,
        concept=condition.concept,
        display=condition.display,
        value=condition.clinical_status,
        effective=condition.onset.value if condition.onset else condition.recorded_date,
        source_step=step_id,
    )


def _medication_evidence(
    patient_id: str, med: NormalizedMedicationRequest, step_id: str | None
) -> Evidence:
    return Evidence(
        patient_id=patient_id,
        resource_type=ResourceType.MEDICATION_REQUEST,
        resource_id=med.id,
        concept=med.concept,
        display=med.display,
        value=med.status,
        effective=med.authored_on,
        source_step=step_id,
        note=med.drug_class,
    )


# -- helpers ----------------------------------------------------------------------


def _demographics(age: float | None, gender: str | None) -> str:
    bits = []
    if age is not None:
        bits.append(f"{age:.0f}y")
    if gender:
        bits.append(gender)
    return " ".join(bits)


def _format_value(value: float | None, unit: str | None) -> str | None:
    if value is None:
        return None
    rendered = f"{value:g}"
    return f"{rendered} {unit}" if unit else rendered


def _data_gaps(features: FeatureSet | None, risk: Any) -> list[str]:
    """Features the answer would have used but could not compute.

    Reported rather than silently imputed: a risk score built on absent data looks
    identical to one built on real data unless the gaps travel with it.
    """
    gaps: list[str] = []
    if risk is not None and getattr(risk, "missing_features", None):
        gaps.extend(risk.missing_features)
    if features is not None:
        for name in DEFAULT_SUMMARY_FEATURES:
            if features.values.get(name) is None and name not in gaps:
                gaps.append(name)
    return gaps


def _plan_summary(plan: QueryPlan) -> str:
    lines = [f"intent: {plan.intent.value}", f"cohort logic: {plan.cohort_logic.value}"]
    for step in plan.steps:
        params = ", ".join(f"{p.key}={p.render_value()}" for p in step.params)
        lines.append(f"- {step.step_id} ({step.resource_type.value}): {params}")
    if plan.analysis.type is not AnalysisType.NONE:
        lines.append(f"analysis: {plan.analysis.type.value}")
    if plan.assumptions:
        lines.append("assumptions: " + "; ".join(plan.assumptions))
    return "\n".join(lines)


def _narrative_findings(
    matches: list[PatientMatch], analyses: list[PatientAnalysis], limit: int
) -> str:
    by_id = {a.patient_id: a for a in analyses}
    lines: list[str] = []
    for match in matches[:limit]:
        bits = [f"patient {match.patient_id}"]
        if match.summary:
            bits.append(match.summary)
        analysis = by_id.get(match.patient_id)
        if analysis and analysis.risk:
            bits.append(
                f"risk {analysis.risk.score:.2f} ({analysis.risk.band.value}); "
                f"drivers: {', '.join(analysis.risk.contributing_factors) or 'none recorded'}"
            )
        lines.append(" | ".join(bits))
    if len(matches) > limit:
        lines.append(f"... and {len(matches) - limit} further patients not listed")
    return "\n".join(lines)
