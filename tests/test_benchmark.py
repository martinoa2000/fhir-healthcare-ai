"""The benchmark must be able to fail: errors, refusals and unsafe requests all count."""

from __future__ import annotations

from datetime import date
from typing import Any

from fhir_healthcare_ai.benchmark.cases import CASES
from fhir_healthcare_ai.benchmark.cli import main, render_table
from fhir_healthcare_ai.benchmark.runner import BenchmarkRunner, _safety_violations
from fhir_healthcare_ai.llm.base import LLMMessage, LLMProvider, LLMResponse, LLMUnavailableError

AS_OF = date(2026, 1, 1)


class DownProvider(LLMProvider):
    name = "down"

    async def complete(self, messages: list[LLMMessage], **_: Any) -> LLMResponse:
        raise LLMUnavailableError("offline")


async def test_mock_planner_passes_the_benchmark() -> None:
    report = await BenchmarkRunner(patients=40, seed=3, as_of=AS_OF).run()
    assert report.passed, render_table(report)
    assert report.safety_violations == 0
    assert {c.case_id for c in report.cases} == {c.case_id for c in CASES}


async def test_a_planner_that_answers_nothing_fails() -> None:
    report = await BenchmarkRunner(patients=20, seed=3, as_of=AS_OF, provider=DownProvider()).run()
    assert not report.passed
    assert report.mean_f1 == 0.0
    # Refusing everything is not rewarded on answerable questions, but the refusal
    # cases themselves are still counted as refused.
    assert all(c.refused for c in report.cases if c.kind == "refusal")


def test_safety_check_flags_writes_and_unlisted_resources() -> None:
    assert _safety_violations(["GET Observation?code=x", "GET metadata"]) == []
    violations = _safety_violations(["POST ", "GET Binary/1", "GET Account?_count=1"])
    assert len(violations) == 3


def test_cli_exit_status_and_report(tmp_path: Any, capsys: Any) -> None:
    exit_code = main(["--patients", "30", "--seed", "3", "--output", str(tmp_path)])
    assert exit_code == 0
    assert "PASSED" in capsys.readouterr().out
    assert list(tmp_path.glob("benchmark-mock-*.json"))
