"""Synthetic patient generation and FHIR seeding.

Everything the demo stack knows about patients starts here: the generator builds
archetype-driven, longitudinally consistent records, and the loader upserts them into a
FHIR server idempotently so the stack can be booted repeatedly without the cohort
drifting.
"""

from __future__ import annotations

from fhir_healthcare_ai.synthetic.generator import (
    ARCHETYPES,
    Archetype,
    SyntheticDataset,
    SyntheticGenerator,
)
from fhir_healthcare_ai.synthetic.loader import (
    LoadResult,
    SeedError,
    build_transaction_bundles,
    push_dataset,
    seed_settings,
    wait_for_server,
    write_dataset,
)

__all__ = [
    "ARCHETYPES",
    "Archetype",
    "LoadResult",
    "SeedError",
    "SyntheticDataset",
    "SyntheticGenerator",
    "build_transaction_bundles",
    "push_dataset",
    "seed_settings",
    "wait_for_server",
    "write_dataset",
]
