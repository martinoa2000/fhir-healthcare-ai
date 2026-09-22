"""Mode 1: natural language in, a validated query plan out.

This is the single place where model output becomes something the system will act on,
so it is where the trust boundary sits. The sequence is fixed and every stage can only
narrow what follows:

1. **Prompt** the model with the allowlist and the concept vocabulary.
2. **Parse** the completion into a :class:`QueryPlan`. Pydantic rejects anything that is
   not the expected shape -- a plan is a typed object, never a string the system will
   later interpolate into a URL.
3. **Expand** concept keys into real codes.
4. **Validate** against the allowlist. Errors mean the plan does not run.
5. **Repair** once, feeding the validator's own issues back to the model. One retry,
   not a loop: a model that cannot fix a plan when told exactly what is wrong will not
   fix it on the fifth attempt either, and an unbounded repair loop is an unbounded bill
   and an unbounded latency.

A plan that still fails after the repair attempt is *refused*, and the refusal is
audited. There is no path from here to the FHIR server that skips step 4.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from pydantic import ValidationError

from fhir_healthcare_ai.audit import AuditEvent, AuditSink, LoggingAuditSink
from fhir_healthcare_ai.config import get_settings
from fhir_healthcare_ai.domain.enums import Severity
from fhir_healthcare_ai.domain.query import QueryPlan, ValidationResult
from fhir_healthcare_ai.fhir.concepts import ConceptExpander, ExpansionReport
from fhir_healthcare_ai.fhir.validator import QueryValidator, format_issues
from fhir_healthcare_ai.llm.base import (
    LLMError,
    LLMMessage,
    LLMProvider,
    LLMResponseError,
)
from fhir_healthcare_ai.llm.prompts import build_planning_prompt
from fhir_healthcare_ai.logging_config import get_logger

logger = get_logger(__name__)


class PlanningError(RuntimeError):
    """The planner could not produce a plan the validator would accept."""

    def __init__(self, message: str, validation: ValidationResult | None = None) -> None:
        super().__init__(message)
        self.validation = validation or ValidationResult()


@dataclass
class PlanningResult:
    """A validated plan plus everything needed to explain how it was reached."""

    plan: QueryPlan
    validation: ValidationResult
    expansion: ExpansionReport
    attempts: int = 1
    llm_calls: int = 0
    llm_tokens: int = 0
    duration_ms: float = 0.0
    model: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def assumptions(self) -> list[str]:
        """Model assumptions plus the code expansions, which are assumptions too."""
        return [*self.plan.assumptions, *self.expansion.assumption_lines()]


class QueryPlanner:
    """Turns a clinical question into a plan that is safe to execute."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        validator: QueryValidator | None = None,
        expander: ConceptExpander | None = None,
        audit_sink: AuditSink | None = None,
        max_repair_attempts: int = 1,
    ) -> None:
        self.provider = provider
        self.validator = validator or QueryValidator(
            max_page_size=get_settings().fhir.max_page_size
        )
        self.expander = expander or ConceptExpander()
        self.audit = audit_sink or LoggingAuditSink()
        self.max_repair_attempts = max_repair_attempts

    async def plan(
        self,
        question: str,
        *,
        context: str | None = None,
        as_of: datetime | None = None,
    ) -> PlanningResult:
        """Produce a validated plan, or raise :class:`PlanningError`.

        Args:
            question: The clinician's question. Treated strictly as data.
            context: Extra grounding, e.g. ``"patient id: 123"`` for a summary request.
            as_of: Reference date for relative windows. Fixing it makes planning
                reproducible, which is what lets the benchmark compare runs.
        """
        started = time.perf_counter()
        today = (as_of or datetime.now(UTC)).date().isoformat()

        previous_text: str | None = None
        feedback: str | None = None
        last_validation = ValidationResult()
        llm_calls = 0
        llm_tokens = 0
        model_name: str | None = None

        for attempt in range(self.max_repair_attempts + 1):
            system, user = build_planning_prompt(
                question,
                today=today,
                context=context,
                repair_feedback=feedback,
                previous_plan=previous_text,
            )
            try:
                response = await self.provider.complete(
                    [LLMMessage("system", system), LLMMessage("user", user)],
                    temperature=0.0,
                    json_mode=True,
                )
            except LLMError as exc:
                self._audit_refusal(question, f"llm failure: {exc}")
                raise PlanningError(f"the planner could not reach the model: {exc}") from exc

            llm_calls += 1
            llm_tokens += response.total_tokens
            model_name = response.model
            previous_text = response.text

            try:
                plan = self._parse(response.text, question)
            except PlanningError as exc:
                if attempt >= self.max_repair_attempts:
                    self._audit_refusal(question, str(exc))
                    raise
                feedback = str(exc)
                continue

            if plan.unsupported:
                # An honest refusal is a correct answer, not a failure. It is returned
                # rather than retried: asking again will not add a capability.
                logger.info(
                    "planner declared the question unsupported",
                    extra={"reason": plan.unsupported_reason},
                )
                self._audit_refusal(question, plan.unsupported_reason or "unsupported")
                return PlanningResult(
                    plan=plan,
                    validation=ValidationResult(),
                    expansion=ExpansionReport(),
                    attempts=attempt + 1,
                    llm_calls=llm_calls,
                    llm_tokens=llm_tokens,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    model=model_name,
                )

            expanded, expansion = self.expander.expand_plan(plan)
            validation = self.validator.validate_plan(expanded)
            last_validation = validation

            if validation.ok:
                self._audit_accepted(question, expanded, attempt + 1)
                return PlanningResult(
                    plan=expanded,
                    validation=validation,
                    expansion=expansion,
                    attempts=attempt + 1,
                    llm_calls=llm_calls,
                    llm_tokens=llm_tokens,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    model=model_name,
                    warnings=[issue.message for issue in validation.warnings],
                )

            feedback = format_issues(validation)
            logger.warning(
                "plan rejected by the validator",
                extra={"attempt": attempt + 1, "errors": len(validation.errors)},
            )

        self._audit_refusal(question, format_issues(last_validation))
        raise PlanningError(
            "the generated plan did not pass validation after "
            f"{self.max_repair_attempts + 1} attempt(s)",
            last_validation,
        )

    def _parse(self, text: str, question: str) -> QueryPlan:
        """Parse a completion into a plan, converting pydantic errors into feedback."""
        from fhir_healthcare_ai.llm.base import extract_json_object

        try:
            payload = extract_json_object(text)
        except LLMResponseError as exc:
            raise PlanningError(
                f"the model did not return a JSON object ({exc}). Respond with the plan "
                "object only, no prose and no code fence."
            ) from exc

        # The question is authoritative from the caller, not from the model: a model that
        # paraphrases it must not change what the response claims was asked.
        payload["question"] = question

        try:
            return QueryPlan.model_validate(payload)
        except ValidationError as exc:
            raise PlanningError(_describe_schema_errors(exc)) from exc

    def _audit_accepted(self, question: str, plan: QueryPlan, attempts: int) -> None:
        self.audit.record(
            AuditEvent(
                action="plan.generated",
                outcome="success",
                query=question,
                details={
                    "steps": [step.step_id for step in plan.steps],
                    "intent": plan.intent.value,
                    "attempts": attempts,
                    "provider": self.provider.name,
                },
            )
        )

    def _audit_refusal(self, question: str, reason: str) -> None:
        self.audit.record(
            AuditEvent(
                action="plan.rejected",
                outcome="refused",
                query=question,
                reason=reason[:2000],
                details={"provider": self.provider.name},
            )
        )


def _describe_schema_errors(exc: ValidationError, limit: int = 8) -> str:
    """Render pydantic errors as instructions the model can act on."""
    lines = []
    for error in exc.errors()[:limit]:
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        lines.append(f"- {location}: {error['msg']}")
    remaining = len(exc.errors()) - limit
    if remaining > 0:
        lines.append(f"- ... and {remaining} more")
    return "The JSON did not match the plan schema:\n" + "\n".join(lines)


def severity_counts(validation: ValidationResult) -> dict[str, int]:
    """Issue counts by severity, for the execution trace and the benchmark."""
    counts: dict[str, int] = {}
    for issue in validation.issues:
        key = issue.severity.value if isinstance(issue.severity, Severity) else str(issue.severity)
        counts[key] = counts.get(key, 0) + 1
    return counts
