"""``fhir-ai-seed`` - generate synthetic patients and load them into a FHIR server.

Two shapes of invocation matter and both must keep working:

    fhir-ai-seed --push --wait-for-server --patients 120 --seed 42 \\
        --fhir-base-url http://fhir:8080/fhir --output /app/data/synthetic
    fhir-ai-seed --patients 20 --seed 42 --output /tmp/seedtest

The first is what docker compose runs on every boot; the second is the offline path a
developer uses with no server anywhere in sight. ``--push`` therefore has to be opt-in:
defaulting it on would make the dry run fail with a connection error on a laptop.

Logging goes through the shared configuration so the compose logs stay parseable;
``print`` appears exactly once, for the end-of-run summary a human reads.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import date
from pathlib import Path

from fhir_healthcare_ai.config import get_settings
from fhir_healthcare_ai.logging_config import configure_logging, get_logger
from fhir_healthcare_ai.synthetic.generator import SyntheticDataset, SyntheticGenerator
from fhir_healthcare_ai.synthetic.loader import (
    DEFAULT_BUNDLE_SIZE,
    DEFAULT_WAIT_TIMEOUT,
    LoadResult,
    SeedError,
    push_dataset,
    write_dataset,
)

logger = get_logger(__name__)

DEFAULT_OUTPUT = Path("data/synthetic")


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="fhir-ai-seed",
        description="Generate synthetic FHIR patients and optionally load them.",
    )
    parser.add_argument(
        "--patients", type=int, default=120, help="How many patients to generate (default: 120)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for the generator's RNG (default: 42)"
    )
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=None,
        help="Anchor date for the clinical timeline (default: today, UTC)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory for NDJSON, bundles and manifest.json (default: ./data/synthetic "
        "unless --push is given)",
    )
    parser.add_argument(
        "--push",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Upsert the dataset into the FHIR server (default: off)",
    )
    parser.add_argument(
        "--fhir-base-url",
        default=settings.fhir.base_url,
        help=f"FHIR R4 base URL (default: {settings.fhir.base_url})",
    )
    parser.add_argument(
        "--wait-for-server",
        action="store_true",
        help="Poll {base}/metadata until the server answers before loading",
    )
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=DEFAULT_WAIT_TIMEOUT,
        help=f"Seconds to wait for the server (default: {DEFAULT_WAIT_TIMEOUT:.0f})",
    )
    parser.add_argument(
        "--bundle-size",
        type=int,
        default=DEFAULT_BUNDLE_SIZE,
        help=f"Resources per transaction bundle (default: {DEFAULT_BUNDLE_SIZE})",
    )
    parser.add_argument("--log-level", default=settings.log_level, help="Logging level")
    parser.add_argument(
        "--log-json",
        action=argparse.BooleanOptionalAction,
        default=settings.log_json,
        help="Emit structured JSON logs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(level=args.log_level, json_output=args.log_json)

    if args.patients < 1:
        logger.error("--patients must be at least 1", extra={"patients": args.patients})
        return 2

    # With neither destination chosen the run would compute a dataset and throw it
    # away, so an absent --output falls back to a local directory.
    output: Path | None = args.output
    if output is None and not args.push:
        output = DEFAULT_OUTPUT

    dataset = SyntheticGenerator(
        patients=args.patients, seed=args.seed, as_of=args.as_of
    ).generate()

    if output is not None:
        write_dataset(dataset, output)

    load: LoadResult | None = None
    if args.push:
        try:
            load = asyncio.run(
                push_dataset(
                    dataset,
                    args.fhir_base_url,
                    bundle_size=args.bundle_size,
                    wait=args.wait_for_server,
                    wait_timeout=args.wait_timeout,
                )
            )
        except SeedError as exc:
            logger.error("seed aborted", extra={"error": str(exc)})
            return 1

    print(_summary(dataset, output, args.fhir_base_url if args.push else None, load))
    return 0 if load is None or load.ok else 1


def _summary(
    dataset: SyntheticDataset, output: Path | None, base_url: str | None, load: LoadResult | None
) -> str:
    counts = ", ".join(
        f"{name} {count}" for name, count in sorted(dataset.counts_by_type().items())
    )
    archetypes = ", ".join(
        f"{name} {count}" for name, count in sorted(dataset.archetype_counts.items())
    )
    lines = [
        f"Generated {dataset.patient_count} patients "
        f"({len(dataset.resources)} resources) with seed {dataset.seed} "
        f"as of {dataset.as_of.isoformat()}",
        f"  resources : {counts}",
        f"  archetypes: {archetypes}",
    ]
    if output is not None:
        lines.append(f"  written to: {output}")
    if load is not None and base_url is not None:
        lines.append(
            f"  loaded to : {base_url} "
            f"({load.created} created, {load.updated} updated, {load.failed} failed, "
            f"{load.bundles} bundles)"
        )
        for failure in load.failures[:5]:
            lines.append(f"    ! {failure}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
