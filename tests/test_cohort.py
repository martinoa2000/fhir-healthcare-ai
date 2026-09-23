"""Cohort resolution: how step roles and cohort logic combine per-step patient sets."""

from __future__ import annotations

from fhir_healthcare_ai.analytics.cohort import CohortResolver, StepResult
from fhir_healthcare_ai.domain.enums import CohortLogic, StepRole


def test_dependent_filter_step_narrows_the_cohort() -> None:
    cohort, _ = CohortResolver().resolve(
        [
            StepResult("elevated_hba1c", {"a", "b", "c"}),
            StepResult("recent_med_change", {"b"}, role=StepRole.FILTER),
        ]
    )
    assert cohort == {"b"}


def test_context_step_never_drops_a_patient() -> None:
    cohort, _ = CohortResolver().resolve(
        [
            StepResult("diabetes", {"a", "b", "c"}),
            StepResult("recent_results", {"a"}, role=StepRole.CONTEXT),
        ]
    )
    assert cohort == {"a", "b", "c"}


def test_exclude_step_removes_its_patients() -> None:
    cohort, warnings = CohortResolver().resolve(
        [
            StepResult("diabetes", {"a", "b", "c"}),
            StepResult("statin_orders", {"b", "z"}, role=StepRole.EXCLUDE),
        ]
    )
    assert cohort == {"a", "c"}
    assert any("removed by exclusion step statin_orders" in w for w in warnings)


def test_truncated_exclusion_is_reported_because_it_errs_towards_inclusion() -> None:
    _, warnings = CohortResolver().resolve(
        [
            StepResult("diabetes", {"a"}),
            StepResult("statin_orders", set(), role=StepRole.EXCLUDE, truncated=True),
        ]
    )
    assert any("was truncated" in w for w in warnings)


def test_any_logic_is_a_union_and_all_is_an_intersection() -> None:
    steps = [StepResult("x", {"a", "b"}), StepResult("y", {"b", "c"})]
    assert CohortResolver(CohortLogic.ANY).resolve(steps)[0] == {"a", "b", "c"}
    assert CohortResolver(CohortLogic.ALL).resolve(steps)[0] == {"b"}


def test_empty_filter_step_explains_an_empty_intersection() -> None:
    cohort, warnings = CohortResolver().resolve([StepResult("x", {"a"}), StepResult("y", set())])
    assert cohort == set()
    assert any("matched no patients" in w for w in warnings)


def test_matched_steps_omit_exclusions() -> None:
    steps = [
        StepResult("x", {"a"}),
        StepResult("ctx", {"a"}, role=StepRole.CONTEXT),
        StepResult("ex", {"a"}, role=StepRole.EXCLUDE),
    ]
    assert CohortResolver().matched_steps("a", steps) == ["x", "ctx"]
