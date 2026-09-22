"""Archetype-driven synthetic patient generator.

A demonstrator is only honest if its data can actually answer the questions the
benchmark asks. Random values sampled independently per resource produce a dataset
where "patients with elevated HbA1c *and* a recent change in diabetes medication" has
either no members or a meaningless set of them. So patients here are drawn from
clinical *archetypes* with fixed prevalence, and every resource for a patient is
derived from one coherent trajectory: HbA1c trends with control status, eGFR declines
in CKD, blood pressure falls once an antihypertensive starts, weight and BMI agree with
height, and lab Observations are grouped under the DiagnosticReport that reported them.

Two deliberate design choices:

**Nothing here knows a code.** Every system/code pair is looked up through
:mod:`fhir_healthcare_ai.terminology`, so the generated data and the query layer can
never drift apart.

**The data is messy on purpose.** Partial dates, interpretation-instead-of-range,
alternative codings and units, missing optional fields and a couple of deceased
patients all appear in a minority of resources, because parsers that are only ever fed
clean data are parsers that have not been tested.

Determinism is over ``(seed, as_of)``: the anchor date defaults to today so that
"recent" stays recent for a demo that is booted months from now, and every value is
drawn from a single :class:`random.Random` threaded through generation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from random import Random
from typing import Any

from fhir_healthcare_ai.logging_config import get_logger
from fhir_healthcare_ai.terminology import (
    CONDITION_CATEGORY,
    CONDITION_CLINICAL,
    CONDITION_VER_STATUS,
    DIAGNOSTIC_SERVICE,
    ENCOUNTER_CLASS,
    INTERPRETATION,
    OBSERVATION_CATEGORY,
    UCUM,
    ConceptDefinition,
    convert_value,
    require_concept,
)

logger = get_logger(__name__)

SYNTHETIC_IDENTIFIER_SYSTEM = "urn:fhir-healthcare-ai:synthetic-mrn"


# --------------------------------------------------------------------------------------
# Trajectories and archetypes
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Trend:
    """A longitudinal series described by where it ends up, not where it started.

    Anchoring on the *latest* value keeps the clinically interesting end of the series
    under control: an "uncontrolled diabetic" is defined by a current HbA1c above 8, and
    back-projecting from there is what makes the earlier points consistent with it.
    """

    latest: tuple[float, float]
    per_year: float = 0.0
    noise: float = 0.0
    floor: float | None = None
    ceiling: float | None = None

    def baseline(self, rng: Random) -> float:
        return rng.uniform(*self.latest)

    def at(self, baseline: float, years_before: float, rng: Random) -> float:
        value = baseline - self.per_year * years_before + _jitter(rng, self.noise)
        if self.floor is not None:
            value = max(value, self.floor)
        if self.ceiling is not None:
            value = min(value, self.ceiling)
        return value


def _jitter(rng: Random, sigma: float) -> float:
    """Gaussian noise truncated at two sigma.

    An untruncated tail occasionally pushes a patient out of the archetype they were
    drawn for, which quietly breaks the ground truth the benchmark depends on.
    """
    if sigma <= 0:
        return 0.0
    return max(-2.0 * sigma, min(2.0 * sigma, rng.gauss(0.0, sigma)))


@dataclass(frozen=True)
class Archetype:
    """One clinical population, its prevalence, and how its measurements behave."""

    key: str
    display: str
    prevalence: float
    age_range: tuple[int, int]
    conditions: tuple[str, ...]
    hba1c: Trend
    egfr: Trend
    systolic: Trend
    ldl: Trend
    bmi: Trend
    medications: tuple[str, ...] = ()
    optional_conditions: tuple[tuple[str, float], ...] = ()
    escalation: tuple[str, ...] = ()
    escalation_probability: float = 0.0
    deprescribed: str | None = None
    bp_response: float = 0.0
    visits: tuple[int, int] = (4, 7)
    inpatient_probability: float = 0.0
    emergency_probability: float = 0.05
    deceased_probability: float = 0.0

    def condition_keys(self) -> tuple[str, ...]:
        return self.conditions + tuple(key for key, _ in self.optional_conditions)

    def has_tag(self, tag: str) -> bool:
        return any(tag in require_concept(key).tags for key in self.condition_keys())


ARCHETYPES: tuple[Archetype, ...] = (
    Archetype(
        key="healthy",
        display="Healthy adult",
        prevalence=0.22,
        age_range=(24, 62),
        conditions=(),
        hba1c=Trend((5.0, 5.6), per_year=0.02, noise=0.12),
        egfr=Trend((88.0, 112.0), per_year=-0.8, noise=3.0),
        systolic=Trend((106.0, 126.0), per_year=0.6, noise=5.0),
        ldl=Trend((72.0, 118.0), per_year=0.8, noise=8.0),
        bmi=Trend((20.5, 27.0), per_year=0.2, noise=0.5),
        optional_conditions=(("hyperlipidemia", 0.12),),
        visits=(3, 5),
        emergency_probability=0.06,
    ),
    Archetype(
        key="controlled_t2dm",
        display="Type 2 diabetes at target",
        prevalence=0.17,
        age_range=(48, 78),
        conditions=("type_2_diabetes", "hyperlipidemia"),
        hba1c=Trend((6.1, 7.1), per_year=-0.18, noise=0.16),
        egfr=Trend((68.0, 96.0), per_year=-1.2, noise=3.0),
        systolic=Trend((116.0, 134.0), per_year=0.4, noise=5.0),
        ldl=Trend((74.0, 118.0), per_year=-3.0, noise=8.0),
        bmi=Trend((25.5, 32.0), per_year=-0.3, noise=0.6),
        medications=("metformin", "atorvastatin"),
        optional_conditions=(("hypertension", 0.45), ("obesity", 0.25)),
        escalation=("sitagliptin", "empagliflozin"),
        escalation_probability=0.15,
        bp_response=8.0,
        visits=(4, 7),
    ),
    Archetype(
        key="uncontrolled_t2dm",
        display="Type 2 diabetes above target with therapy escalation",
        prevalence=0.15,
        age_range=(44, 76),
        conditions=("type_2_diabetes", "obesity", "hyperlipidemia"),
        hba1c=Trend((8.7, 10.9), per_year=0.7, noise=0.18),
        egfr=Trend((58.0, 92.0), per_year=-2.4, noise=3.0),
        systolic=Trend((124.0, 148.0), per_year=0.9, noise=6.0),
        ldl=Trend((102.0, 168.0), per_year=2.0, noise=10.0),
        bmi=Trend((30.5, 40.0), per_year=0.7, noise=0.7),
        medications=("metformin",),
        optional_conditions=(("hypertension", 0.55), ("diabetic_nephropathy", 0.2)),
        escalation=("empagliflozin", "semaglutide", "sitagliptin", "insulin_glargine"),
        escalation_probability=0.85,
        deprescribed="glipizide",
        bp_response=6.0,
        visits=(5, 8),
        emergency_probability=0.18,
    ),
    Archetype(
        key="diabetic_ckd",
        display="Type 2 diabetes with chronic kidney disease",
        prevalence=0.12,
        age_range=(58, 86),
        conditions=("type_2_diabetes", "chronic_kidney_disease", "hypertension"),
        hba1c=Trend((7.1, 8.8), per_year=0.25, noise=0.2),
        egfr=Trend((18.0, 44.0), per_year=-4.5, noise=2.5, floor=12.0),
        systolic=Trend((126.0, 148.0), per_year=0.7, noise=6.0),
        ldl=Trend((70.0, 124.0), per_year=-1.5, noise=9.0),
        bmi=Trend((25.0, 35.0), per_year=-0.4, noise=0.7),
        medications=("insulin_glargine", "lisinopril", "atorvastatin", "furosemide"),
        optional_conditions=(("diabetic_nephropathy", 0.7), ("heart_failure", 0.25)),
        escalation=("dapagliflozin", "empagliflozin"),
        escalation_probability=0.35,
        bp_response=16.0,
        visits=(5, 8),
        inpatient_probability=0.25,
        emergency_probability=0.22,
        deceased_probability=0.06,
    ),
    Archetype(
        key="hypertensive",
        display="Treated essential hypertension",
        prevalence=0.18,
        age_range=(40, 80),
        conditions=("hypertension",),
        hba1c=Trend((5.2, 6.0), per_year=0.05, noise=0.13),
        egfr=Trend((66.0, 98.0), per_year=-1.4, noise=3.0),
        systolic=Trend((124.0, 138.0), per_year=0.3, noise=5.0),
        ldl=Trend((86.0, 142.0), per_year=-1.0, noise=9.0),
        bmi=Trend((24.0, 33.0), per_year=0.1, noise=0.6),
        medications=("amlodipine", "hydrochlorothiazide"),
        optional_conditions=(("hyperlipidemia", 0.5), ("obesity", 0.2), ("copd", 0.1)),
        bp_response=22.0,
        visits=(4, 7),
        emergency_probability=0.1,
    ),
    Archetype(
        key="heart_failure",
        display="Heart failure with recurrent admissions",
        prevalence=0.08,
        age_range=(62, 90),
        conditions=("heart_failure", "hypertension"),
        hba1c=Trend((5.5, 6.6), per_year=0.08, noise=0.15),
        egfr=Trend((38.0, 70.0), per_year=-3.0, noise=3.0, floor=15.0),
        systolic=Trend((102.0, 126.0), per_year=-0.5, noise=6.0),
        ldl=Trend((62.0, 112.0), per_year=-2.0, noise=9.0),
        bmi=Trend((22.0, 31.0), per_year=-0.6, noise=0.8),
        medications=("furosemide", "metoprolol", "lisinopril", "apixaban"),
        optional_conditions=(
            ("myocardial_infarction", 0.4),
            ("chronic_kidney_disease", 0.35),
            ("copd", 0.2),
        ),
        bp_response=10.0,
        visits=(5, 8),
        inpatient_probability=0.55,
        emergency_probability=0.35,
        deceased_probability=0.12,
    ),
    Archetype(
        key="metabolic_risk",
        display="Obesity with prediabetes",
        prevalence=0.08,
        age_range=(30, 64),
        conditions=("obesity",),
        hba1c=Trend((5.8, 6.4), per_year=0.12, noise=0.14),
        egfr=Trend((80.0, 108.0), per_year=-1.0, noise=3.0),
        systolic=Trend((118.0, 138.0), per_year=0.8, noise=5.0),
        ldl=Trend((104.0, 168.0), per_year=1.5, noise=10.0),
        bmi=Trend((31.0, 42.0), per_year=0.8, noise=0.7),
        medications=("atorvastatin",),
        optional_conditions=(("hyperlipidemia", 0.6),),
        visits=(3, 6),
        emergency_probability=0.08,
    ),
)


# --------------------------------------------------------------------------------------
# Dosing and demographics vocabulary (free text only -- never codes)
# --------------------------------------------------------------------------------------

_DOSING: dict[str, tuple[float, str, str]] = {
    "metformin": (500.0, "mg", "500 mg orally twice daily with meals"),
    "glipizide": (5.0, "mg", "5 mg orally once daily before breakfast"),
    "sitagliptin": (100.0, "mg", "100 mg orally once daily"),
    "empagliflozin": (10.0, "mg", "10 mg orally once daily"),
    "dapagliflozin": (10.0, "mg", "10 mg orally once daily"),
    "semaglutide": (0.5, "mg", "0.5 mg subcutaneously once weekly"),
    "liraglutide": (1.2, "mg", "1.2 mg subcutaneously once daily"),
    "insulin_glargine": (20.0, "[IU]", "20 units subcutaneously at bedtime"),
    "lisinopril": (10.0, "mg", "10 mg orally once daily"),
    "losartan": (50.0, "mg", "50 mg orally once daily"),
    "amlodipine": (5.0, "mg", "5 mg orally once daily"),
    "hydrochlorothiazide": (25.0, "mg", "25 mg orally every morning"),
    "metoprolol": (25.0, "mg", "25 mg orally twice daily"),
    "atorvastatin": (40.0, "mg", "40 mg orally at bedtime"),
    "simvastatin": (20.0, "mg", "20 mg orally at bedtime"),
    "furosemide": (40.0, "mg", "40 mg orally every morning"),
    "apixaban": (5.0, "mg", "5 mg orally twice daily"),
    "aspirin": (81.0, "mg", "81 mg orally once daily"),
    "levothyroxine": (75.0, "ug", "75 mcg orally every morning before food"),
}

_GIVEN_FEMALE = (
    "Amara",
    "Beatriz",
    "Chioma",
    "Dorothy",
    "Elena",
    "Farrah",
    "Greta",
    "Hannah",
    "Ingrid",
    "Joanna",
    "Kavita",
    "Lucia",
    "Maria",
    "Nadia",
    "Olive",
    "Priya",
    "Rosa",
    "Saoirse",
    "Tamsin",
    "Ursula",
    "Vera",
    "Wendy",
    "Yara",
    "Zofia",
)
_GIVEN_MALE = (
    "Adeyemi",
    "Bruno",
    "Caleb",
    "Dmitri",
    "Elliot",
    "Farid",
    "Gustav",
    "Hiroshi",
    "Ivan",
    "Jonas",
    "Karim",
    "Lucas",
    "Mateo",
    "Niall",
    "Omar",
    "Pedro",
    "Quentin",
    "Rashid",
    "Stefan",
    "Tobias",
    "Ulrich",
    "Viktor",
    "Wesley",
    "Yusuf",
)
_FAMILY = (
    "Achebe",
    "Bergstrom",
    "Castellano",
    "Donnelly",
    "Eriksen",
    "Fontaine",
    "Gallagher",
    "Haddad",
    "Iversen",
    "Jankowski",
    "Kowalczyk",
    "Larsen",
    "Mwangi",
    "Nakamura",
    "Okafor",
    "Petrova",
    "Quinones",
    "Rasmussen",
    "Silva",
    "Tanaka",
    "Ueda",
    "Vasquez",
    "Whitfield",
    "Yilmaz",
    "Zielinski",
)
_POSTAL_CODES = (
    "02139",
    "02140",
    "02215",
    "10025",
    "10032",
    "11201",
    "19104",
    "20007",
    "30308",
    "48109",
    "60637",
    "94110",
    "94143",
    "98105",
)


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------


@dataclass
class SyntheticDataset:
    """Generated resources plus enough provenance to reproduce them."""

    seed: int
    as_of: date
    archetype_counts: dict[str, int] = field(default_factory=dict)
    resources: list[dict[str, Any]] = field(default_factory=list)

    def by_type(self) -> dict[str, list[dict[str, Any]]]:
        """Resources grouped by ``resourceType``, insertion ordered."""
        grouped: dict[str, list[dict[str, Any]]] = {}
        for resource in self.resources:
            grouped.setdefault(str(resource["resourceType"]), []).append(resource)
        return grouped

    def counts_by_type(self) -> dict[str, int]:
        return {name: len(items) for name, items in self.by_type().items()}

    @property
    def patient_count(self) -> int:
        return sum(1 for r in self.resources if r.get("resourceType") == "Patient")

    def manifest(self) -> dict[str, Any]:
        """Machine-readable description of this run, written next to the data."""
        return {
            "generator": "fhir_healthcare_ai.synthetic",
            "seed": self.seed,
            "as_of": self.as_of.isoformat(),
            "patients": self.patient_count,
            "resource_total": len(self.resources),
            "counts_by_type": self.counts_by_type(),
            "archetypes": dict(sorted(self.archetype_counts.items())),
            "synthetic": True,
        }


def _archetype_plan(archetypes: Sequence[Archetype], total: int) -> list[Archetype]:
    """Apportion a population by prevalence using largest remainders.

    Sampling each patient independently would leave a 20-patient run missing whole
    archetypes, which is exactly the run a developer does first.
    """
    exact = [a.prevalence * total for a in archetypes]
    counts = [math.floor(value) for value in exact]
    shortfall = total - sum(counts)
    order = sorted(range(len(archetypes)), key=lambda i: (-(exact[i] - counts[i]), i))
    for index in order[:shortfall]:
        counts[index] += 1
    plan: list[Archetype] = []
    for archetype, count in zip(archetypes, counts, strict=True):
        plan.extend([archetype] * count)
    return plan


# --------------------------------------------------------------------------------------
# FHIR fragment helpers
# --------------------------------------------------------------------------------------


def _coding(concept: ConceptDefinition, index: int = 0) -> dict[str, Any]:
    coding = concept.codings[index % len(concept.codings)]
    return {"system": coding.system, "code": coding.code, "display": coding.display}


def _codeable(concept: ConceptDefinition, index: int = 0) -> dict[str, Any]:
    return {"coding": [_coding(concept, index)], "text": concept.display}


def _quantity(value: float, unit: str) -> dict[str, Any]:
    return {"value": value, "unit": unit, "system": UCUM, "code": unit}


def _instant(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _category(system: str, code: str, display: str) -> list[dict[str, Any]]:
    return [{"coding": [{"system": system, "code": code, "display": display}]}]


def _interpretation_code(concept: ConceptDefinition, value: float, sex: str) -> tuple[str, str]:
    interval = concept.interval_for(sex)
    if interval is None:
        return "N", "Normal"
    if interval.high is not None and value > interval.high:
        return "H", "High"
    if interval.low is not None and value < interval.low:
        return "L", "Low"
    return "N", "Normal"


def _reference_range(concept: ConceptDefinition, sex: str) -> list[dict[str, Any]] | None:
    interval = concept.interval_for(sex)
    if interval is None:
        return None
    entry: dict[str, Any] = {}
    if interval.low is not None:
        entry["low"] = _quantity(interval.low, interval.unit)
    if interval.high is not None:
        entry["high"] = _quantity(interval.high, interval.unit)
    return [entry] if entry else None


@dataclass
class _Patient:
    """Everything about one synthetic patient that more than one resource needs."""

    id: str
    archetype: Archetype
    sex: str
    birth_date: date
    age_years: float
    height_cm: float
    visits: list[datetime]
    conditions: tuple[str, ...]
    baselines: dict[str, float]
    bp_treatment_start: datetime | None = None
    deceased_at: datetime | None = None
    screen_hba1c: bool = False

    @property
    def reference(self) -> dict[str, str]:
        return {"reference": f"Patient/{self.id}"}

    def has_tag(self, tag: str) -> bool:
        """True when any condition this patient actually has carries ``tag``."""
        return any(tag in require_concept(key).tags for key in self.conditions)


class SyntheticGenerator:
    """Builds a deterministic, clinically coherent FHIR R4 dataset."""

    def __init__(
        self,
        *,
        patients: int = 120,
        seed: int = 42,
        as_of: date | None = None,
        archetypes: Sequence[Archetype] = ARCHETYPES,
    ) -> None:
        if patients < 1:
            raise ValueError("patients must be at least 1")
        self.patients = patients
        self.seed = seed
        self.as_of = as_of or datetime.now(UTC).date()
        self.archetypes = tuple(archetypes)

    def generate(self) -> SyntheticDataset:
        """Generate the full population."""
        rng = Random(self.seed)
        plan = _archetype_plan(self.archetypes, self.patients)
        rng.shuffle(plan)

        resources: list[dict[str, Any]] = []
        counts: dict[str, int] = {}
        for index, archetype in enumerate(plan, start=1):
            counts[archetype.key] = counts.get(archetype.key, 0) + 1
            resources.extend(self._build_patient(index, archetype, rng))

        dataset = SyntheticDataset(
            seed=self.seed, as_of=self.as_of, archetype_counts=counts, resources=resources
        )
        logger.info(
            "generated synthetic population",
            extra={
                "seed": self.seed,
                "as_of": self.as_of.isoformat(),
                "patients": dataset.patient_count,
                "resources": len(resources),
            },
        )
        return dataset

    # -- patient scaffolding --------------------------------------------------------

    def _build_patient(self, index: int, archetype: Archetype, rng: Random) -> list[dict[str, Any]]:
        patient = self._draw_patient(index, archetype, rng)
        return [
            self._patient_resource(patient, index, rng),
            *self._condition_resources(patient, rng),
            *self._encounter_resources(patient, rng),
            # Medications are built before observations because starting an
            # antihypertensive is what bends the blood-pressure series.
            *self._medication_resources(patient, rng),
            *self._clinical_resources(patient, rng),
        ]

    def _draw_patient(self, index: int, archetype: Archetype, rng: Random) -> _Patient:
        sex = "female" if rng.random() < 0.51 else "male"
        age = rng.uniform(*archetype.age_range)
        birth_date = self.as_of - timedelta(days=int(age * 365.25))
        height_cm = round(rng.gauss(164.0 if sex == "female" else 177.0, 6.5), 1)
        visits = self._visit_schedule(archetype, rng)

        conditions = list(archetype.conditions)
        for key, probability in archetype.optional_conditions:
            if rng.random() < probability and key not in conditions:
                conditions.append(key)

        baselines = {
            "hba1c": archetype.hba1c.baseline(rng),
            "egfr": archetype.egfr.baseline(rng),
            "systolic": archetype.systolic.baseline(rng),
            "ldl": archetype.ldl.baseline(rng),
            "bmi": archetype.bmi.baseline(rng),
        }

        patient = _Patient(
            id=f"syn{self.seed}-pat-{index:04d}",
            archetype=archetype,
            sex=sex,
            birth_date=birth_date,
            age_years=age,
            height_cm=height_cm,
            visits=visits,
            conditions=tuple(conditions),
            baselines=baselines,
            screen_hba1c=rng.random() < 0.45,
        )
        patient.deceased_at = self._draw_death(patient, rng)
        return patient

    def _visit_schedule(self, archetype: Archetype, rng: Random) -> list[datetime]:
        """Roughly quarterly visits walking backwards from a recent one."""
        anchor = datetime.combine(self.as_of, time(0, 0), tzinfo=UTC)
        offset = rng.randint(6, 45)
        moments: list[datetime] = []
        for _ in range(rng.randint(*archetype.visits)):
            moments.append(
                anchor
                - timedelta(days=offset)
                + timedelta(hours=rng.randint(8, 16), minutes=rng.choice((0, 15, 30, 45)))
            )
            offset += rng.randint(72, 135)
        return sorted(moments)

    def _draw_death(self, patient: _Patient, rng: Random) -> datetime | None:
        """Kill a patient only when there is room to do so after their last visit."""
        if rng.random() >= patient.archetype.deceased_probability:
            return None
        anchor = datetime.combine(self.as_of, time(0, 0), tzinfo=UTC)
        gap = (anchor - patient.visits[-1]).days
        if gap < 20:
            return None
        return patient.visits[-1] + timedelta(days=rng.randint(5, gap - 5))

    def _years_before(self, moment: datetime) -> float:
        anchor = datetime.combine(self.as_of, time(0, 0), tzinfo=UTC)
        return (anchor - moment).total_seconds() / (365.25 * 86400)

    def _encounter_id(self, patient: _Patient, visit_index: int) -> str:
        return f"{patient.id}-enc-{visit_index:02d}"

    # -- demographics ---------------------------------------------------------------

    def _patient_resource(self, patient: _Patient, index: int, rng: Random) -> dict[str, Any]:
        given = rng.choice(_GIVEN_FEMALE if patient.sex == "female" else _GIVEN_MALE)
        family = rng.choice(_FAMILY)
        resource: dict[str, Any] = {
            "resourceType": "Patient",
            "id": patient.id,
            "identifier": [
                {
                    "system": SYNTHETIC_IDENTIFIER_SYSTEM,
                    "value": f"SYN-{self.seed}-{index:04d}",
                }
            ],
            "active": True,
            "name": [{"use": "official", "family": family, "given": [given]}],
            "gender": patient.sex,
            "birthDate": patient.birth_date.isoformat(),
        }

        # A birth date recorded to the year only is common in registry-sourced feeds.
        if rng.random() < 0.02:
            resource["birthDate"] = str(patient.birth_date.year)
        # Administrative sex is genuinely unknown for a small slice of real patients.
        if rng.random() < 0.03:
            resource["gender"] = "unknown"
        if rng.random() < 0.9:
            resource["address"] = [{"postalCode": rng.choice(_POSTAL_CODES), "country": "US"}]
        if patient.deceased_at is not None:
            resource["deceasedDateTime"] = _instant(patient.deceased_at)
        return resource

    # -- problems -------------------------------------------------------------------

    def _condition_resources(self, patient: _Patient, rng: Random) -> list[dict[str, Any]]:
        resources: list[dict[str, Any]] = []
        for position, key in enumerate(patient.conditions, start=1):
            concept = require_concept(key)
            onset = patient.visits[0] - timedelta(days=rng.randint(200, 4200))
            # A minority of feeds carry the billing code only, with no SNOMED.
            icd_only = rng.random() < 0.2 and len(concept.codings) > 1
            codings = (
                [_coding(concept, len(concept.codings) - 1)]
                if icd_only
                else [_coding(concept, i) for i in range(min(2, len(concept.codings)))]
            )
            resource: dict[str, Any] = {
                "resourceType": "Condition",
                "id": f"{patient.id}-cond-{position:02d}",
                "clinicalStatus": {
                    "coding": [
                        {"system": CONDITION_CLINICAL, "code": "active", "display": "Active"}
                    ]
                },
                "verificationStatus": {
                    "coding": [
                        {
                            "system": CONDITION_VER_STATUS,
                            "code": "confirmed",
                            "display": "Confirmed",
                        }
                    ]
                },
                "category": _category(CONDITION_CATEGORY, "problem-list-item", "Problem List Item"),
                "code": {"coding": codings, "text": concept.display},
                "subject": patient.reference,
                "onsetDateTime": _partial_or_full_date(onset, rng),
                "recordedDate": _instant(onset + timedelta(days=rng.randint(0, 21))),
            }
            resources.append(resource)
        return resources

    # -- encounters -----------------------------------------------------------------

    def _encounter_resources(self, patient: _Patient, rng: Random) -> list[dict[str, Any]]:
        reason: list[dict[str, Any]] | None = None
        if patient.conditions:
            reason = [_codeable(require_concept(patient.conditions[0]))]
        resources: list[dict[str, Any]] = []

        for visit_index, moment in enumerate(patient.visits):
            inpatient = rng.random() < patient.archetype.inpatient_probability
            stay_days = rng.randint(2, 9) if inpatient else 0
            end = moment + (
                timedelta(days=stay_days) if inpatient else timedelta(minutes=rng.randint(20, 55))
            )
            resource: dict[str, Any] = {
                "resourceType": "Encounter",
                "id": self._encounter_id(patient, visit_index),
                "status": "finished",
                "class": (
                    {"system": ENCOUNTER_CLASS, "code": "IMP", "display": "inpatient encounter"}
                    if inpatient
                    else {"system": ENCOUNTER_CLASS, "code": "AMB", "display": "ambulatory"}
                ),
                "subject": patient.reference,
                "period": {"start": _instant(moment), "end": _instant(end)},
            }
            if inpatient and rng.random() < 0.6:
                resource["length"] = {
                    "value": stay_days,
                    "unit": "days",
                    "system": UCUM,
                    "code": "d",
                }
            if reason is not None and rng.random() < 0.8:
                resource["reasonCode"] = reason
            resources.append(resource)

        if rng.random() < patient.archetype.emergency_probability:
            resources.append(self._emergency_encounter(patient, reason, rng))
        return resources

    def _emergency_encounter(
        self, patient: _Patient, reason: list[dict[str, Any]] | None, rng: Random
    ) -> dict[str, Any]:
        moment = patient.visits[-1] - timedelta(days=rng.randint(10, 240))
        resource: dict[str, Any] = {
            "resourceType": "Encounter",
            "id": f"{patient.id}-enc-er",
            "status": "finished",
            "class": {"system": ENCOUNTER_CLASS, "code": "EMER", "display": "emergency"},
            "subject": patient.reference,
            "period": {
                "start": _instant(moment),
                "end": _instant(moment + timedelta(hours=rng.randint(3, 11))),
            },
        }
        if reason is not None:
            resource["reasonCode"] = reason
        return resource

    # -- medications ----------------------------------------------------------------

    def _medication_resources(self, patient: _Patient, rng: Random) -> list[dict[str, Any]]:
        """Long-standing therapy, one optional escalation, one optional deprescription.

        The escalation is authored *at the most recent visit*, which is what makes
        "changed therapy in the last 90 days" a real, answerable question rather than an
        artefact of a random date.
        """
        archetype = patient.archetype
        resources: list[dict[str, Any]] = []
        position = 0

        for key in archetype.medications:
            position += 1
            authored = patient.visits[0] - timedelta(days=rng.randint(120, 1600))
            if require_concept(key).tags & {"antihypertensive", "loop_diuretic"}:
                # Start the blood-pressure agent mid-history so the response is visible.
                mid = patient.visits[len(patient.visits) // 2]
                authored = mid
                if patient.bp_treatment_start is None or mid < patient.bp_treatment_start:
                    patient.bp_treatment_start = mid
            resources.append(
                self._medication_resource(patient, key, position, authored, "active", rng)
            )

        if archetype.escalation and rng.random() < archetype.escalation_probability:
            position += 1
            authored = patient.visits[-1]
            resources.append(
                self._medication_resource(
                    patient,
                    rng.choice(archetype.escalation),
                    position,
                    authored,
                    "active",
                    rng,
                    encounter_index=len(patient.visits) - 1,
                )
            )
            if archetype.deprescribed and rng.random() < 0.45:
                position += 1
                stopped = self._medication_resource(
                    patient,
                    archetype.deprescribed,
                    position,
                    patient.visits[0] - timedelta(days=rng.randint(200, 900)),
                    "stopped",
                    rng,
                )
                stopped["dispenseRequest"] = {
                    "validityPeriod": {
                        "start": stopped["authoredOn"],
                        "end": _instant(patient.visits[-1]),
                    }
                }
                resources.append(stopped)

        if rng.random() < 0.15:
            position += 1
            resources.append(
                self._medication_resource(
                    patient,
                    rng.choice(("aspirin", "levothyroxine")),
                    position,
                    patient.visits[0] - timedelta(days=rng.randint(150, 2000)),
                    "active",
                    rng,
                )
            )
        return resources

    def _medication_resource(
        self,
        patient: _Patient,
        key: str,
        position: int,
        authored: datetime,
        status: str,
        rng: Random,
        encounter_index: int | None = None,
    ) -> dict[str, Any]:
        concept = require_concept(key)
        dose, unit, text = _DOSING[key]
        resource: dict[str, Any] = {
            "resourceType": "MedicationRequest",
            "id": f"{patient.id}-med-{position:02d}",
            "status": status,
            "intent": "order",
            "medicationCodeableConcept": _codeable(concept),
            "subject": patient.reference,
            "authoredOn": _instant(authored),
        }
        if encounter_index is not None:
            resource["encounter"] = {
                "reference": f"Encounter/{self._encounter_id(patient, encounter_index)}"
            }
        # Orders transcribed from paper or from an external pharmacy often arrive with
        # no structured dose at all.
        if rng.random() < 0.85:
            resource["dosageInstruction"] = [
                {"text": text, "doseAndRate": [{"doseQuantity": _quantity(dose, unit)}]}
            ]
        if patient.conditions and rng.random() < 0.5:
            resource["reasonCode"] = [_codeable(require_concept(patient.conditions[0]))]
        return resource

    # -- observations and reports ---------------------------------------------------

    def _clinical_resources(self, patient: _Patient, rng: Random) -> list[dict[str, Any]]:
        resources: list[dict[str, Any]] = []
        total = len(patient.visits)

        for visit_index, moment in enumerate(patient.visits):
            values = self._visit_values(patient, moment, rng)
            encounter = {"reference": f"Encounter/{self._encounter_id(patient, visit_index)}"}
            position = 0

            lab_keys = self._lab_keys(patient, visit_index, total)
            lab_ids: list[str] = []
            for key in lab_keys:
                position += 1
                observation = self._observation(
                    patient,
                    key,
                    values[key],
                    moment,
                    visit_index,
                    position,
                    rng,
                    category=("laboratory", "Laboratory"),
                    encounter=encounter,
                )
                lab_ids.append(str(observation["id"]))
                resources.append(observation)

            position += 1
            resources.append(
                self._blood_pressure(patient, values, moment, visit_index, position, encounter)
            )
            for key in ("body_weight", "bmi"):
                position += 1
                resources.append(
                    self._observation(
                        patient,
                        key,
                        values[key],
                        moment,
                        visit_index,
                        position,
                        rng,
                        category=("vital-signs", "Vital Signs"),
                        encounter=encounter,
                    )
                )
            if visit_index == 0:
                position += 1
                resources.append(
                    self._observation(
                        patient,
                        "body_height",
                        patient.height_cm,
                        moment,
                        visit_index,
                        position,
                        rng,
                        category=("vital-signs", "Vital Signs"),
                        encounter=encounter,
                    )
                )
            if rng.random() < 0.6:
                position += 1
                resources.append(
                    self._observation(
                        patient,
                        "heart_rate",
                        values["heart_rate"],
                        moment,
                        visit_index,
                        position,
                        rng,
                        category=("vital-signs", "Vital Signs"),
                        encounter=encounter,
                    )
                )

            if lab_ids:
                resources.append(
                    self._diagnostic_report(
                        patient, lab_keys, lab_ids, moment, visit_index, encounter, rng
                    )
                )
        return resources

    def _lab_keys(self, patient: _Patient, visit_index: int, total: int) -> list[str]:
        """Which panel was drawn at this visit.

        Alternating panels is what real follow-up looks like, and it gives the feature
        layer series of different lengths to cope with.
        """
        glycemic = patient.has_tag("diabetes") or patient.has_tag("metabolic")
        keys: list[str] = []
        if glycemic or patient.screen_hba1c:
            keys += ["hba1c", "glucose"]
        keys += ["creatinine", "egfr"]
        if visit_index % 2 == 0:
            keys += ["potassium", "sodium"]
            if patient.has_tag("renal") or patient.has_tag("diabetes"):
                keys.append("urine_albumin_creatinine_ratio")
        else:
            keys += ["ldl_cholesterol", "hdl_cholesterol", "triglycerides"]
        if visit_index == total - 1:
            keys += ["hemoglobin", "alt"]
        return keys

    def _visit_values(self, patient: _Patient, moment: datetime, rng: Random) -> dict[str, float]:
        """Every measurement for one visit, derived from one shared physiology.

        All of them are drawn whether or not the visit's panel uses them: the draw order
        has to be independent of the panel for the seed to be reproducible.
        """
        archetype = patient.archetype
        baselines = patient.baselines
        years = self._years_before(moment)

        egfr = archetype.egfr.at(baselines["egfr"], years, rng)
        hba1c = archetype.hba1c.at(baselines["hba1c"], years, rng)
        bmi = archetype.bmi.at(baselines["bmi"], years, rng)
        ldl = archetype.ldl.at(baselines["ldl"], years, rng)
        untreated = archetype.bp_response * (1.0 - _bp_ramp(patient.bp_treatment_start, moment))
        systolic = archetype.systolic.at(baselines["systolic"], years, rng) + untreated

        renal = patient.has_tag("renal")
        height_m = patient.height_cm / 100.0
        age_at_visit = max(18.0, patient.age_years - years)

        hdl = _clamp(62.0 - (bmi - 25.0) * 0.9 + _jitter(rng, 5.0), 26.0, 88.0)
        triglycerides = _clamp(88.0 + (bmi - 25.0) * 7.0 + _jitter(rng, 22.0), 45.0, 430.0)

        values = {
            "hba1c": round(hba1c, 1),
            "glucose": round(max(58.0, 25.0 * hba1c - 42.0 + _jitter(rng, 8.0)), 0),
            "egfr": round(egfr, 0),
            "creatinine": round(_creatinine_from_egfr(egfr, age_at_visit, patient.sex), 2),
            "ldl_cholesterol": round(max(35.0, ldl), 0),
            "hdl_cholesterol": round(hdl, 0),
            "triglycerides": round(triglycerides, 0),
            "potassium": round(rng.uniform(4.3, 5.6) if renal else rng.uniform(3.6, 5.0), 1),
            "sodium": round(rng.uniform(134.0, 144.0), 0),
            "hemoglobin": round(_hemoglobin(patient.sex, renal, rng), 1),
            "alt": round(rng.uniform(28.0, 72.0) if bmi > 32 else rng.uniform(10.0, 42.0), 0),
            "urine_albumin_creatinine_ratio": round(_uacr(patient, egfr, rng), 0),
            "systolic_bp": round(systolic, 0),
            "diastolic_bp": round(systolic * 0.55 + 4.0 + _jitter(rng, 3.0), 0),
            "bmi": round(bmi, 1),
            "body_weight": round(bmi * height_m * height_m, 1),
            "heart_rate": round(_clamp(74.0 + _jitter(rng, 9.0), 48, 118), 0),
        }
        return values

    def _observation(
        self,
        patient: _Patient,
        key: str,
        value: float,
        moment: datetime,
        visit_index: int,
        position: int,
        rng: Random,
        *,
        category: tuple[str, str],
        encounter: dict[str, str],
    ) -> dict[str, Any]:
        concept = require_concept(key)
        coding_index = rng.randrange(len(concept.codings)) if rng.random() < 0.12 else 0
        reported_value, unit = _reported_quantity(concept, value, rng)

        resource: dict[str, Any] = {
            "resourceType": "Observation",
            "id": f"{patient.id}-obs-{visit_index:02d}-{position:02d}",
            "status": "final",
            "category": _category(OBSERVATION_CATEGORY, *category),
            "code": {"coding": [_coding(concept, coding_index)], "text": concept.display},
            "subject": patient.reference,
            "encounter": encounter,
            "effectiveDateTime": _partial_or_full_instant(moment, self.as_of, rng),
            "valueQuantity": _quantity(reported_value, unit),
        }
        if rng.random() < 0.8:
            resource["issued"] = _instant(moment + timedelta(hours=rng.randint(1, 30)))

        # Some laboratories publish their own interpretation and no numeric range; the
        # parser has to trust the flag in that case rather than a textbook interval.
        if rng.random() < 0.15:
            code, display = _interpretation_code(concept, value, patient.sex)
            resource["interpretation"] = [
                {"coding": [{"system": INTERPRETATION, "code": code, "display": display}]}
            ]
        else:
            reference_range = _reference_range(concept, patient.sex)
            if reference_range is not None:
                resource["referenceRange"] = reference_range
        return resource

    def _blood_pressure(
        self,
        patient: _Patient,
        values: dict[str, float],
        moment: datetime,
        visit_index: int,
        position: int,
        encounter: dict[str, str],
    ) -> dict[str, Any]:
        """Blood pressure as a panel with components, never as two loose observations."""
        panel = require_concept("blood_pressure_panel")
        return {
            "resourceType": "Observation",
            "id": f"{patient.id}-obs-{visit_index:02d}-{position:02d}",
            "status": "final",
            "category": _category(OBSERVATION_CATEGORY, "vital-signs", "Vital Signs"),
            "code": _codeable(panel),
            "subject": patient.reference,
            "encounter": encounter,
            "effectiveDateTime": _instant(moment),
            "component": [
                {
                    "code": _codeable(require_concept(component_key)),
                    "valueQuantity": _quantity(
                        values[component_key], require_concept(component_key).canonical_unit or ""
                    ),
                    "referenceRange": _reference_range(require_concept(component_key), patient.sex),
                }
                for component_key in ("systolic_bp", "diastolic_bp")
            ],
        }

    def _diagnostic_report(
        self,
        patient: _Patient,
        lab_keys: Sequence[str],
        lab_ids: Sequence[str],
        moment: datetime,
        visit_index: int,
        encounter: dict[str, str],
        rng: Random,
    ) -> dict[str, Any]:
        lead = require_concept(lab_keys[0])
        resource: dict[str, Any] = {
            "resourceType": "DiagnosticReport",
            "id": f"{patient.id}-rep-{visit_index:02d}",
            "status": "final",
            "category": _category(DIAGNOSTIC_SERVICE, "LAB", "Laboratory"),
            "code": {"coding": [_coding(lead)], "text": f"{lead.display} panel"},
            "subject": patient.reference,
            "encounter": encounter,
            "effectiveDateTime": _instant(moment),
            "issued": _instant(moment + timedelta(hours=rng.randint(2, 28))),
            "result": [{"reference": f"Observation/{lab_id}"} for lab_id in lab_ids],
        }
        if patient.conditions and rng.random() < 0.25:
            concept = require_concept(patient.conditions[0])
            resource["conclusion"] = f"Results reviewed in the context of {concept.display}."
            resource["conclusionCode"] = [_codeable(concept)]
        return resource


# --------------------------------------------------------------------------------------
# Physiology and messiness helpers
# --------------------------------------------------------------------------------------


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _bp_ramp(treatment_start: datetime | None, moment: datetime) -> float:
    """How much of an antihypertensive's effect has been realised by ``moment``.

    Blood pressure does not step down on the day of the prescription; ramping over six
    months is what makes the pre/post difference look like treatment rather than noise.
    """
    if treatment_start is None:
        return 1.0
    if moment < treatment_start:
        return 0.0
    return min(1.0, (moment - treatment_start).days / 180.0)


def _creatinine_from_egfr(egfr: float, age_years: float, sex: str) -> float:
    """Invert MDRD so creatinine and eGFR cannot contradict each other."""
    factor = 175.0 * age_years**-0.203 * (0.742 if sex == "female" else 1.0)
    return _clamp((max(egfr, 5.0) / factor) ** (-1.0 / 1.154), 0.4, 9.0)


def _hemoglobin(sex: str, renal: bool, rng: Random) -> float:
    if renal:
        return rng.uniform(9.6, 12.4)
    return rng.uniform(12.2, 15.4) if sex == "female" else rng.uniform(13.4, 17.0)


def _uacr(patient: _Patient, egfr: float, rng: Random) -> float:
    if patient.has_tag("renal"):
        return _clamp(rng.uniform(120.0, 700.0) * (60.0 / max(egfr, 12.0)), 60.0, 2400.0)
    if patient.has_tag("diabetes"):
        return rng.uniform(10.0, 80.0)
    return rng.uniform(2.0, 22.0)


def _reported_quantity(concept: ConceptDefinition, value: float, rng: Random) -> tuple[float, str]:
    """Occasionally report a value in a non-canonical unit the terminology can convert.

    A pipeline that has only ever seen HbA1c in percent will silently mis-threshold the
    first IFCC-reporting laboratory it meets.
    """
    canonical = concept.canonical_unit or ""
    alternative = _ALTERNATIVE_UNITS.get(concept.key)
    if alternative is None or rng.random() >= 0.06:
        return value, canonical
    converted = convert_value(value, canonical, alternative, concept.key)
    return round(converted, 1), alternative


_ALTERNATIVE_UNITS: dict[str, str] = {
    "hba1c": "mmol/mol",
    "glucose": "mmol/L",
    "creatinine": "umol/L",
    "body_weight": "lb",
    "urine_albumin_creatinine_ratio": "mg/mmol",
}


def _partial_or_full_date(moment: datetime, rng: Random) -> str:
    """Condition onsets are frequently recorded to the year or month only."""
    draw = rng.random()
    if draw < 0.08:
        return str(moment.year)
    if draw < 0.2:
        return moment.strftime("%Y-%m")
    return moment.strftime("%Y-%m-%d")


def _partial_or_full_instant(moment: datetime, as_of: date, rng: Random) -> str:
    """Drop time precision on a minority of older results.

    Only results older than 200 days are degraded: coarsening a recent one would move it
    out of its own window and break the ground truth the benchmark checks.
    """
    age_days = (datetime.combine(as_of, time(0, 0), tzinfo=UTC) - moment).days
    if age_days > 200 and rng.random() < 0.05:
        return moment.strftime("%Y-%m")
    if rng.random() < 0.04:
        return moment.strftime("%Y-%m-%d")
    return _instant(moment)
