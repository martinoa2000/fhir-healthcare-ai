"""Persists a generated dataset: to a FHIR server, to disk, or both.

The compose stack re-runs the seeder on every ``docker compose up``, so loading has to
be idempotent. Every entry is a ``PUT Type/id`` inside a *transaction* bundle: the id
comes from the generator's seed, so a second run overwrites the same resources instead
of minting new ones. A POST-based load would silently triple the cohort by the third
boot and quietly invalidate every benchmark answer.

Bundles are ordered Patient -> Encounter -> Condition -> MedicationRequest ->
Observation -> DiagnosticReport. HAPI enforces referential integrity, and a bundle is
applied as a unit, so a DiagnosticReport whose ``result`` points at an Observation in a
later bundle would be rejected outright.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fhir_healthcare_ai.config import FHIRSettings
from fhir_healthcare_ai.fhir.client import FHIRClient, FHIRError
from fhir_healthcare_ai.logging_config import get_logger
from fhir_healthcare_ai.synthetic.generator import SyntheticDataset

logger = get_logger(__name__)

DEFAULT_BUNDLE_SIZE = 200
DEFAULT_WAIT_TIMEOUT = 300.0
DEFAULT_POLL_INTERVAL = 3.0

# Dependency order. Anything the generator adds later that is not listed here is
# written last, which is the safe default for a resource nothing else references.
_LOAD_ORDER: tuple[str, ...] = (
    "Patient",
    "Encounter",
    "Condition",
    "MedicationRequest",
    "Observation",
    "DiagnosticReport",
)


class SeedError(RuntimeError):
    """The dataset could not be loaded into the FHIR server."""


@dataclass
class LoadResult:
    """Tally of what the server did with the submitted bundles."""

    bundles: int = 0
    submitted: int = 0
    created: int = 0
    updated: int = 0
    failed: int = 0
    duration_ms: float = 0.0
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.failed == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "bundles": self.bundles,
            "submitted": self.submitted,
            "created": self.created,
            "updated": self.updated,
            "failed": self.failed,
            "duration_ms": round(self.duration_ms, 2),
        }


def seed_settings(base_url: str, *, timeout_seconds: float = 120.0) -> FHIRSettings:
    """Build the settings the seeder writes with.

    ``allow_write`` is set here in code rather than read from ``FHIR_ALLOW_WRITE``.
    Seeding is the one component that is *supposed* to write, and making that explicit
    keeps the env var meaningful for everything else: the query path stays read-only
    even if someone exports the variable in a shared shell, and a misconfigured
    container cannot turn the API into a writer by accident.

    The timeout is raised well above the default because a transaction bundle of a few
    hundred resources takes HAPI far longer to apply than any single search.
    """
    return FHIRSettings(base_url=base_url, allow_write=True, timeout_seconds=timeout_seconds)


def build_transaction_bundles(
    resources: list[dict[str, Any]], bundle_size: int = DEFAULT_BUNDLE_SIZE
) -> list[dict[str, Any]]:
    """Chunk resources into dependency-ordered PUT transaction bundles."""
    if bundle_size < 1:
        raise ValueError("bundle_size must be at least 1")

    ordered = sorted(resources, key=_load_rank)
    bundles: list[dict[str, Any]] = []
    for start in range(0, len(ordered), bundle_size):
        chunk = ordered[start : start + bundle_size]
        bundles.append(
            {
                "resourceType": "Bundle",
                "type": "transaction",
                "entry": [_transaction_entry(resource) for resource in chunk],
            }
        )
    return bundles


async def wait_for_server(
    client: FHIRClient,
    timeout_seconds: float = DEFAULT_WAIT_TIMEOUT,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
) -> bool:
    """Poll ``{base_url}/metadata`` until a CapabilityStatement comes back.

    A cold HAPI needs 60-120s to finish booting, and until then it either refuses the
    connection or answers 503. Polling is bounded so a genuinely dead server fails the
    seed job instead of hanging the compose stack forever.
    """
    deadline = time.monotonic() + timeout_seconds
    attempt = 0
    logger.info(
        "waiting for FHIR server",
        extra={"base_url": client.settings.base_url, "timeout_seconds": timeout_seconds},
    )
    while True:
        attempt += 1
        if await client.ping():
            logger.info(
                "FHIR server is ready",
                extra={"base_url": client.settings.base_url, "attempts": attempt},
            )
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.error(
                "FHIR server did not become ready",
                extra={"base_url": client.settings.base_url, "attempts": attempt},
            )
            return False
        logger.info(
            "FHIR server not ready yet, retrying",
            extra={"attempt": attempt, "seconds_remaining": round(remaining, 1)},
        )
        await asyncio.sleep(min(poll_interval, remaining))


async def push_dataset(
    dataset: SyntheticDataset,
    base_url: str,
    *,
    bundle_size: int = DEFAULT_BUNDLE_SIZE,
    wait: bool = False,
    wait_timeout: float = DEFAULT_WAIT_TIMEOUT,
    client: FHIRClient | None = None,
) -> LoadResult:
    """Upsert every resource in ``dataset`` into the FHIR server at ``base_url``."""
    owns_client = client is None
    active = client or FHIRClient(seed_settings(base_url))
    started = time.perf_counter()
    try:
        if wait and not await wait_for_server(active, timeout_seconds=wait_timeout):
            raise SeedError(f"FHIR server at {base_url} never became ready")

        bundles = build_transaction_bundles(dataset.resources, bundle_size)
        result = LoadResult(bundles=len(bundles), submitted=len(dataset.resources))
        for index, bundle in enumerate(bundles, start=1):
            entries = len(bundle["entry"])
            try:
                response = await active.post_bundle(bundle)
            except FHIRError as exc:
                result.failed += entries
                result.failures.append(f"bundle {index}: {exc}")
                logger.error(
                    "transaction bundle rejected",
                    extra={"bundle": index, "entries": entries, "error": str(exc)},
                )
                continue
            _tally(response, result)
            logger.info(
                "transaction bundle applied",
                extra={
                    "bundle": index,
                    "of": len(bundles),
                    "entries": entries,
                    "created": result.created,
                    "updated": result.updated,
                },
            )
    finally:
        if owns_client:
            await active.aclose()

    result.duration_ms = (time.perf_counter() - started) * 1000
    logger.info("seed load finished", extra=result.as_dict())
    return result


def write_dataset(dataset: SyntheticDataset, output_dir: Path | str) -> Path:
    """Write the dataset to disk as NDJSON plus the bundles that would be POSTed.

    NDJSON is what a human (or a test) wants to grep; the bundles are what the server
    would actually receive, so a failed load can be replayed with curl without
    re-running the generator.
    """
    root = Path(output_dir)
    (root / "bundles").mkdir(parents=True, exist_ok=True)

    for resource_type, items in dataset.by_type().items():
        path = root / f"{resource_type}.ndjson"
        with path.open("w", encoding="utf-8") as handle:
            for resource in items:
                handle.write(json.dumps(resource, separators=(",", ":")) + "\n")

    bundles = build_transaction_bundles(dataset.resources)
    for index, bundle in enumerate(bundles, start=1):
        bundle_path = root / "bundles" / f"transaction-{index:03d}.json"
        bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    manifest = dataset.manifest()
    manifest["bundles"] = len(bundles)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info(
        "wrote synthetic dataset",
        extra={
            "output_dir": str(root),
            "resources": len(dataset.resources),
            "bundles": len(bundles),
        },
    )
    return root


def _load_rank(resource: dict[str, Any]) -> tuple[int, str]:
    """Sort key placing referenced resources ahead of the resources that cite them."""
    resource_type = str(resource.get("resourceType", ""))
    try:
        rank = _LOAD_ORDER.index(resource_type)
    except ValueError:
        rank = len(_LOAD_ORDER)
    return rank, str(resource.get("id", ""))


def _transaction_entry(resource: dict[str, Any]) -> dict[str, Any]:
    resource_type = resource.get("resourceType")
    resource_id = resource.get("id")
    if not isinstance(resource_type, str) or not isinstance(resource_id, str):
        raise ValueError("every seeded resource needs a resourceType and a logical id")
    return {
        "fullUrl": f"{resource_type}/{resource_id}",
        "resource": resource,
        "request": {"method": "PUT", "url": f"{resource_type}/{resource_id}"},
    }


def _tally(response: dict[str, Any], result: LoadResult) -> None:
    """Fold a transaction-response bundle into the running counts.

    HAPI answers a PUT with 201 when the resource is new and 200 when it already
    existed, which is precisely the signal that tells a first boot from a re-run.
    """
    for entry in response.get("entry") or []:
        status = str((entry.get("response") or {}).get("status", ""))
        code = status.split(" ", 1)[0]
        if code.startswith("201"):
            result.created += 1
        elif code.startswith("200"):
            result.updated += 1
        else:
            result.failed += 1
            result.failures.append(status or "missing response status")
