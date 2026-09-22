"""The synthetic population is the answer key; it has to be reproducible."""

from __future__ import annotations

import json
from datetime import date

from fhir_healthcare_ai.synthetic.cli import main
from fhir_healthcare_ai.synthetic.generator import SyntheticGenerator


def test_same_seed_same_population() -> None:
    first = SyntheticGenerator(patients=15, seed=11, as_of=date(2026, 1, 1)).generate()
    second = SyntheticGenerator(patients=15, seed=11, as_of=date(2026, 1, 1)).generate()
    assert json.dumps(first.resources, sort_keys=True) == json.dumps(
        second.resources, sort_keys=True
    )


def test_different_seed_different_population() -> None:
    first = SyntheticGenerator(patients=15, seed=11, as_of=date(2026, 1, 1)).generate()
    second = SyntheticGenerator(patients=15, seed=12, as_of=date(2026, 1, 1)).generate()
    assert first.resources != second.resources


def test_every_clinical_resource_references_a_generated_patient() -> None:
    dataset = SyntheticGenerator(patients=10, seed=1, as_of=date(2026, 1, 1)).generate()
    patients = {f"Patient/{r['id']}" for r in dataset.resources if r["resourceType"] == "Patient"}
    assert dataset.patient_count == 10
    for resource in dataset.resources:
        if resource["resourceType"] != "Patient":
            assert resource["subject"]["reference"] in patients


def test_offline_seed_cli_writes_ndjson_and_bundles(tmp_path) -> None:  # type: ignore[no-untyped-def]
    exit_code = main(["--patients", "5", "--seed", "1", "--output", str(tmp_path)])
    assert exit_code == 0
    assert (tmp_path / "Patient.ndjson").read_text().count("\n") == 5
    assert list((tmp_path / "bundles").glob("transaction-*.json"))
    assert json.loads((tmp_path / "manifest.json").read_text())["patients"] == 5
