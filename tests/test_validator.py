"""The allowlist validator is the security boundary; these tests pin its behaviour."""

from __future__ import annotations

import pytest

from fhir_healthcare_ai.domain.enums import ResourceType, StepRole
from fhir_healthcare_ai.domain.query import AnalysisRequest, QueryPlan, QueryStep, SearchParam
from fhir_healthcare_ai.fhir.concepts import ConceptExpander
from fhir_healthcare_ai.fhir.validator import QueryValidator
from fhir_healthcare_ai.terminology import ICD10, LOINC, SNOMED


def _codes(result: object) -> set[str]:
    return {issue.code for issue in result.errors}  # type: ignore[attr-defined]


def _step(**overrides: object) -> QueryStep:
    base: dict[str, object] = {
        "step_id": "labs",
        "resource_type": ResourceType.OBSERVATION,
        "params": [SearchParam(name="code", values=["4548-4"], system=LOINC)],
    }
    base.update(overrides)
    return QueryStep.model_validate(base)


def _plan(*steps: QueryStep, **overrides: object) -> QueryPlan:
    return QueryPlan(question="q", steps=list(steps), **overrides)  # type: ignore[arg-type]


@pytest.fixture
def validator() -> QueryValidator:
    return QueryValidator()


def test_accepts_a_minimal_selective_step(validator: QueryValidator) -> None:
    assert validator.validate_step(_step()).ok


def test_refuses_resource_outside_allowlist(validator: QueryValidator) -> None:
    result = validator.validate_step(_step(resource_type=ResourceType.PROCEDURE))
    assert _codes(result) == {"resource-not-allowed"}


def test_refuses_unknown_search_parameter(validator: QueryValidator) -> None:
    step = _step(params=[SearchParam(name="_query", values=["evil"])])
    assert "control-param-in-params" in _codes(validator.validate_step(step))
    step = _step(params=[SearchParam(name="value-string", values=["x"])])
    assert "param-not-allowed" in _codes(validator.validate_step(step))


@pytest.mark.parametrize("value", ["4548-4&_include=*", "../Binary", "a b", "x\ny"])
def test_refuses_injection_shaped_values(validator: QueryValidator, value: str) -> None:
    step = _step(params=[SearchParam(name="code", values=[value], system=LOINC)])
    assert "invalid-param-value" in _codes(validator.validate_step(step))


def test_refuses_an_unselective_scan(validator: QueryValidator) -> None:
    step = _step(params=[SearchParam(name="status", values=["final"])])
    assert not validator.validate_step(step).ok


def test_multi_system_values_from_the_expander_are_accepted(validator: QueryValidator) -> None:
    step = _step(
        resource_type=ResourceType.CONDITION,
        params=[SearchParam(name="code", values=[f"{SNOMED}|44054006", f"{ICD10}|E11.9"])],
    )
    assert validator.validate_step(step).ok


def test_system_prefixed_value_must_name_an_allowed_system(validator: QueryValidator) -> None:
    step = _step(params=[SearchParam(name="code", values=["http://evil.example|123"])])
    assert "system-not-allowed" in _codes(validator.validate_step(step))


def test_status_tokens_never_accept_a_system_prefix(validator: QueryValidator) -> None:
    step = _step(
        params=[
            SearchParam(name="code", values=["4548-4"], system=LOINC),
            SearchParam(name="status", values=[f"{LOINC}|final"]),
        ]
    )
    assert not validator.validate_step(step).ok


def test_expanded_multi_system_concept_plan_validates(validator: QueryValidator) -> None:
    plan = _plan(
        _step(
            resource_type=ResourceType.CONDITION,
            params=[SearchParam(name="code", values=["type_2_diabetes"])],
        )
    )
    expanded, _ = ConceptExpander().expand_plan(plan)
    systems = {value.split("|", 1)[0] for value in expanded.steps[0].params[0].values}
    assert systems == {SNOMED, ICD10}
    assert validator.validate_plan(expanded).ok


def test_exclusion_only_plan_is_rejected(validator: QueryValidator) -> None:
    plan = _plan(_step(role=StepRole.EXCLUDE))
    assert "exclusion-only-plan" in _codes(validator.validate_plan(plan))


def test_analysis_options_are_allowlisted(validator: QueryValidator) -> None:
    ok = _plan(_step(), analysis=AnalysisRequest(options={"require_abnormal": True}))
    assert validator.validate_plan(ok).ok
    unknown = _plan(_step(), analysis=AnalysisRequest(options={"exec": "rm -rf /"}))
    assert "unknown-analysis-option" in _codes(validator.validate_plan(unknown))
    wrong_type = _plan(_step(), analysis=AnalysisRequest(options={"require_abnormal": "yes"}))
    assert "invalid-analysis-option" in _codes(validator.validate_plan(wrong_type))


def test_empty_and_oversized_plans_are_rejected(validator: QueryValidator) -> None:
    assert "empty-plan" in _codes(validator.validate_plan(_plan()))
    steps = [_step(step_id=f"s{i}") for i in range(validator.max_steps + 1)]
    assert "too-many-steps" in _codes(validator.validate_plan(_plan(*steps)))


def test_plan_model_rejects_broken_dependencies() -> None:
    with pytest.raises(ValueError, match="unknown step"):
        _plan(_step(depends_on="missing"))
    with pytest.raises(ValueError, match="duplicate step_id"):
        _plan(_step(), _step())
