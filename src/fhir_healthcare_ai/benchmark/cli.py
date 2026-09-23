"""``fhir-ai-bench`` - score the planner end to end against synthetic ground truth.

    fhir-ai-bench                                  # mock planner, 120 patients
    fhir-ai-bench --provider vllm                  # the local model, same questions
    fhir-ai-bench --output benchmark-results/      # also write JSON for tracking

Exit status is 0 when the run passes (mean F1 at or above ``--min-f1``, every refusal
case refused, no safety violation) and 1 otherwise, so it can gate CI.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import date
from pathlib import Path
from typing import get_args

from fhir_healthcare_ai.benchmark.runner import DEFAULT_MIN_F1, BenchmarkReport, BenchmarkRunner
from fhir_healthcare_ai.config import LLMProviderName, get_settings
from fhir_healthcare_ai.logging_config import configure_logging

#: Fixed so that two runs on different days score the same population.
DEFAULT_AS_OF = date(2026, 1, 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fhir-ai-bench", description="Benchmark the query planner against ground truth."
    )
    parser.add_argument(
        "--provider",
        choices=get_args(LLMProviderName),
        default="mock",
        help="LLM provider to benchmark (default: mock, the deterministic control arm)",
    )
    parser.add_argument("--patients", type=int, default=120, help="Population size (default: 120)")
    parser.add_argument("--seed", type=int, default=42, help="Generator seed (default: 42)")
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=DEFAULT_AS_OF,
        help=f"Reference date for data and questions (default: {DEFAULT_AS_OF})",
    )
    parser.add_argument(
        "--min-f1",
        type=float,
        default=DEFAULT_MIN_F1,
        help=f"Mean cohort F1 required to pass (default: {DEFAULT_MIN_F1})",
    )
    parser.add_argument("--output", type=Path, default=None, help="Directory for a JSON report")
    parser.add_argument("--json", action="store_true", help="Print the JSON report to stdout")
    return parser


def render_table(report: BenchmarkReport) -> str:
    """A plain-text table a human can read in CI logs."""
    header = (
        f"{'case':<28} {'result':<6} {'exp':>5} {'got':>5} "
        f"{'prec':>6} {'rec':>6} {'f1':>6} {'ms':>8}"
    )
    lines = [
        f"provider={report.provider} model={report.model} patients={report.patients} "
        f"seed={report.seed} as_of={report.as_of}",
        "",
        header,
        "-" * len(header),
    ]
    for case in report.cases:
        status = "PASS" if case.passed else "FAIL"
        if case.kind == "refusal":
            lines.append(
                f"{case.case_id:<28} {status:<6} {'refused' if case.refused else 'ANSWERED':>33}"
                f" {case.latency_ms:>8.1f}"
            )
            continue
        lines.append(
            f"{case.case_id:<28} {status:<6} {_n(case.expected):>5} {_n(case.returned):>5} "
            f"{_f(case.precision):>6} {_f(case.recall):>6} {_f(case.f1):>6} {case.latency_ms:>8.1f}"
        )
        if case.error:
            lines.append(f"    error: {case.error[:160]}")
    for case in report.cases:
        for violation in case.safety_violations:
            lines.append(f"SAFETY {case.case_id}: {violation}")
    summary = report.summary()
    lines += [
        "",
        f"mean F1 {summary['mean_f1']:.4f} (min {report.min_f1})  "
        f"refusals {summary['refusal_accuracy']}  "
        f"safety violations {summary['safety_violations']}  "
        f"-> {'PASSED' if summary['passed'] else 'FAILED'}",
    ]
    return "\n".join(lines)


def _n(value: int | None) -> str:
    return "-" if value is None else str(value)


def _f(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging("WARNING", settings.log_json, stream=sys.stderr)

    runner = BenchmarkRunner(
        patients=args.patients,
        seed=args.seed,
        as_of=args.as_of,
        provider_name=args.provider,
        min_f1=args.min_f1,
    )
    report = asyncio.run(_run(runner))

    payload = report.to_dict()
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / f"benchmark-{report.provider}-{args.as_of.isoformat()}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2) if args.json else render_table(report))
    return 0 if report.passed else 1


async def _run(runner: BenchmarkRunner) -> BenchmarkReport:
    try:
        return await runner.run()
    finally:
        await runner.provider.aclose()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
