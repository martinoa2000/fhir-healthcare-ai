"""Terminology layer: canonical concepts, code systems and unit conversion.

Real clinical data is polysemous. The same HbA1c arrives as LOINC ``4548-4`` from one
lab and ``17856-6`` from another; the same diabetes diagnosis arrives as SNOMED
``44054006`` or ICD-10 ``E11.9``. Downstream code should never care.

This module is the single place that knows about codes. It maps *many* source codings
onto *one* canonical concept key, and it converts values onto one canonical unit so a
threshold like "HbA1c > 7" means the same thing regardless of whether the source
reported percent or mmol/mol.

The content here is a deliberately small, hand-curated subset sized for a demonstrator.
A production deployment would back this with a real terminology server (e.g. FHIR
``$lookup`` / ``$translate`` against a ValueSet-aware service) rather than a dict.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum

from fhir_healthcare_ai.domain.clinical import Coding

# --------------------------------------------------------------------------------------
# Code systems
# --------------------------------------------------------------------------------------

LOINC = "http://loinc.org"
SNOMED = "http://snomed.info/sct"
RXNORM = "http://www.nlm.nih.gov/research/umls/rxnorm"
ICD10 = "http://hl7.org/fhir/sid/icd-10-cm"
UCUM = "http://unitsofmeasure.org"
CONDITION_CLINICAL = "http://terminology.hl7.org/CodeSystem/condition-clinical"
CONDITION_VER_STATUS = "http://terminology.hl7.org/CodeSystem/condition-ver-status"
CONDITION_CATEGORY = "http://terminology.hl7.org/CodeSystem/condition-category"
OBSERVATION_CATEGORY = "http://terminology.hl7.org/CodeSystem/observation-category"
ENCOUNTER_CLASS = "http://terminology.hl7.org/CodeSystem/v3-ActCode"
INTERPRETATION = "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation"
DIAGNOSTIC_SERVICE = "http://terminology.hl7.org/CodeSystem/v2-0074"

KNOWN_SYSTEMS: frozenset[str] = frozenset(
    {
        LOINC,
        SNOMED,
        RXNORM,
        ICD10,
        UCUM,
        CONDITION_CLINICAL,
        CONDITION_VER_STATUS,
        CONDITION_CATEGORY,
        OBSERVATION_CATEGORY,
        ENCOUNTER_CLASS,
        INTERPRETATION,
        DIAGNOSTIC_SERVICE,
    }
)


class ConceptKind(StrEnum):
    """What sort of clinical thing a concept denotes."""

    LAB = "lab"
    VITAL = "vital"
    CONDITION = "condition"
    MEDICATION = "medication"
    PANEL = "panel"


@dataclass(frozen=True)
class ReferenceInterval:
    """A normal range, optionally sex-specific.

    ``critical_*`` bounds mark values that would normally trigger an urgent call-back
    in a real laboratory workflow.
    """

    low: float | None
    high: float | None
    unit: str
    critical_low: float | None = None
    critical_high: float | None = None
    sex: str | None = None  # "male" | "female" | None (applies to all)


@dataclass(frozen=True)
class ConceptDefinition:
    """One canonical clinical concept and every source coding that maps to it."""

    key: str
    display: str
    kind: ConceptKind
    codings: tuple[Coding, ...]
    canonical_unit: str | None = None
    reference_intervals: tuple[ReferenceInterval, ...] = ()
    drug_class: str | None = None
    tags: frozenset[str] = field(default_factory=frozenset)

    def tokens(self, system: str | None = None) -> list[str]:
        """Codes usable as FHIR token search values, optionally filtered by system."""
        return [c.code for c in self.codings if c.code and (system is None or c.system == system)]

    def primary_system(self) -> str | None:
        return self.codings[0].system if self.codings else None

    def interval_for(self, sex: str | None) -> ReferenceInterval | None:
        """Best matching reference interval for a patient's administrative sex."""
        if not self.reference_intervals:
            return None
        if sex:
            for interval in self.reference_intervals:
                if interval.sex == sex.lower():
                    return interval
        for interval in self.reference_intervals:
            if interval.sex is None:
                return interval
        return self.reference_intervals[0]


def _loinc(code: str, display: str) -> Coding:
    return Coding(system=LOINC, code=code, display=display)


def _snomed(code: str, display: str) -> Coding:
    return Coding(system=SNOMED, code=code, display=display)


def _icd10(code: str, display: str) -> Coding:
    return Coding(system=ICD10, code=code, display=display)


def _rxnorm(code: str, display: str) -> Coding:
    return Coding(system=RXNORM, code=code, display=display)


# --------------------------------------------------------------------------------------
# Laboratory concepts
# --------------------------------------------------------------------------------------

_LABS: tuple[ConceptDefinition, ...] = (
    ConceptDefinition(
        key="hba1c",
        display="Hemoglobin A1c",
        kind=ConceptKind.LAB,
        codings=(
            _loinc("4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood"),
            _loinc("17856-6", "Hemoglobin A1c/Hemoglobin.total in Blood by HPLC"),
            _loinc("4549-2", "Hemoglobin A1c/Hemoglobin.total in Blood by Electrophoresis"),
            _loinc("41995-2", "Hemoglobin A1c/Hemoglobin.total in Blood"),
            _loinc("59261-8", "Hemoglobin A1c/Hemoglobin.total in Blood by IFCC"),
        ),
        canonical_unit="%",
        reference_intervals=(ReferenceInterval(low=4.0, high=5.7, unit="%", critical_high=12.0),),
        tags=frozenset({"diabetes", "glycemic_control"}),
    ),
    ConceptDefinition(
        key="glucose",
        display="Glucose",
        kind=ConceptKind.LAB,
        codings=(
            _loinc("2339-0", "Glucose [Mass/volume] in Blood"),
            _loinc("2345-7", "Glucose [Mass/volume] in Serum or Plasma"),
            _loinc("1558-6", "Fasting glucose [Mass/volume] in Serum or Plasma"),
            _loinc("15074-8", "Glucose [Moles/volume] in Blood"),
        ),
        canonical_unit="mg/dL",
        reference_intervals=(
            ReferenceInterval(
                low=70.0, high=99.0, unit="mg/dL", critical_low=45.0, critical_high=400.0
            ),
        ),
        tags=frozenset({"diabetes", "metabolic_panel"}),
    ),
    ConceptDefinition(
        key="creatinine",
        display="Creatinine",
        kind=ConceptKind.LAB,
        codings=(
            _loinc("2160-0", "Creatinine [Mass/volume] in Serum or Plasma"),
            _loinc("38483-4", "Creatinine [Mass/volume] in Blood"),
        ),
        canonical_unit="mg/dL",
        reference_intervals=(
            ReferenceInterval(low=0.7, high=1.3, unit="mg/dL", critical_high=4.0, sex="male"),
            ReferenceInterval(low=0.6, high=1.1, unit="mg/dL", critical_high=4.0, sex="female"),
            ReferenceInterval(low=0.6, high=1.3, unit="mg/dL", critical_high=4.0),
        ),
        tags=frozenset({"renal", "metabolic_panel"}),
    ),
    ConceptDefinition(
        key="egfr",
        display="Estimated glomerular filtration rate",
        kind=ConceptKind.LAB,
        codings=(
            _loinc("33914-3", "Glomerular filtration rate/1.73 sq M.predicted by MDRD"),
            _loinc("48642-3", "GFR/1.73 sq M.predicted among non-blacks by MDRD"),
            _loinc("62238-1", "GFR/1.73 sq M.predicted by CKD-EPI"),
            _loinc("98979-8", "GFR/1.73 sq M.predicted by CKD-EPI 2021"),
        ),
        canonical_unit="mL/min/1.73m2",
        reference_intervals=(
            ReferenceInterval(low=90.0, high=None, unit="mL/min/1.73m2", critical_low=15.0),
        ),
        tags=frozenset({"renal"}),
    ),
    ConceptDefinition(
        key="ldl_cholesterol",
        display="LDL cholesterol",
        kind=ConceptKind.LAB,
        codings=(
            _loinc("13457-7", "Cholesterol in LDL [Mass/volume] calculated"),
            _loinc("18262-6", "Cholesterol in LDL [Mass/volume] direct assay"),
            _loinc("2089-1", "Cholesterol in LDL [Mass/volume] in Serum or Plasma"),
        ),
        canonical_unit="mg/dL",
        reference_intervals=(ReferenceInterval(low=None, high=100.0, unit="mg/dL"),),
        tags=frozenset({"lipids", "cardiovascular"}),
    ),
    ConceptDefinition(
        key="hdl_cholesterol",
        display="HDL cholesterol",
        kind=ConceptKind.LAB,
        codings=(_loinc("2085-9", "Cholesterol in HDL [Mass/volume] in Serum or Plasma"),),
        canonical_unit="mg/dL",
        reference_intervals=(ReferenceInterval(low=40.0, high=None, unit="mg/dL"),),
        tags=frozenset({"lipids", "cardiovascular"}),
    ),
    ConceptDefinition(
        key="total_cholesterol",
        display="Total cholesterol",
        kind=ConceptKind.LAB,
        codings=(_loinc("2093-3", "Cholesterol [Mass/volume] in Serum or Plasma"),),
        canonical_unit="mg/dL",
        reference_intervals=(ReferenceInterval(low=None, high=200.0, unit="mg/dL"),),
        tags=frozenset({"lipids", "cardiovascular"}),
    ),
    ConceptDefinition(
        key="triglycerides",
        display="Triglycerides",
        kind=ConceptKind.LAB,
        codings=(_loinc("2571-8", "Triglyceride [Mass/volume] in Serum or Plasma"),),
        canonical_unit="mg/dL",
        reference_intervals=(ReferenceInterval(low=None, high=150.0, unit="mg/dL"),),
        tags=frozenset({"lipids", "cardiovascular"}),
    ),
    ConceptDefinition(
        key="potassium",
        display="Potassium",
        kind=ConceptKind.LAB,
        codings=(_loinc("2823-3", "Potassium [Moles/volume] in Serum or Plasma"),),
        canonical_unit="mmol/L",
        reference_intervals=(
            ReferenceInterval(
                low=3.5, high=5.1, unit="mmol/L", critical_low=2.8, critical_high=6.2
            ),
        ),
        tags=frozenset({"electrolytes", "metabolic_panel"}),
    ),
    ConceptDefinition(
        key="sodium",
        display="Sodium",
        kind=ConceptKind.LAB,
        codings=(_loinc("2951-2", "Sodium [Moles/volume] in Serum or Plasma"),),
        canonical_unit="mmol/L",
        reference_intervals=(
            ReferenceInterval(
                low=136.0, high=145.0, unit="mmol/L", critical_low=120.0, critical_high=160.0
            ),
        ),
        tags=frozenset({"electrolytes", "metabolic_panel"}),
    ),
    ConceptDefinition(
        key="hemoglobin",
        display="Hemoglobin",
        kind=ConceptKind.LAB,
        codings=(_loinc("718-7", "Hemoglobin [Mass/volume] in Blood"),),
        canonical_unit="g/dL",
        reference_intervals=(
            ReferenceInterval(low=13.5, high=17.5, unit="g/dL", critical_low=7.0, sex="male"),
            ReferenceInterval(low=12.0, high=15.5, unit="g/dL", critical_low=7.0, sex="female"),
            ReferenceInterval(low=12.0, high=17.5, unit="g/dL", critical_low=7.0),
        ),
        tags=frozenset({"hematology", "cbc"}),
    ),
    ConceptDefinition(
        key="alt",
        display="Alanine aminotransferase",
        kind=ConceptKind.LAB,
        codings=(_loinc("1742-6", "Alanine aminotransferase [Enzymatic activity/volume]"),),
        canonical_unit="U/L",
        reference_intervals=(ReferenceInterval(low=7.0, high=56.0, unit="U/L"),),
        tags=frozenset({"hepatic"}),
    ),
    ConceptDefinition(
        key="urine_albumin_creatinine_ratio",
        display="Urine albumin/creatinine ratio",
        kind=ConceptKind.LAB,
        codings=(
            _loinc("9318-7", "Albumin/Creatinine [Mass ratio] in Urine"),
            _loinc("14959-1", "Microalbumin/Creatinine [Mass ratio] in Urine"),
        ),
        canonical_unit="mg/g",
        reference_intervals=(ReferenceInterval(low=None, high=30.0, unit="mg/g"),),
        tags=frozenset({"renal", "diabetes"}),
    ),
)

# --------------------------------------------------------------------------------------
# Vital signs
# --------------------------------------------------------------------------------------

_VITALS: tuple[ConceptDefinition, ...] = (
    ConceptDefinition(
        key="systolic_bp",
        display="Systolic blood pressure",
        kind=ConceptKind.VITAL,
        codings=(_loinc("8480-6", "Systolic blood pressure"),),
        canonical_unit="mm[Hg]",
        reference_intervals=(
            ReferenceInterval(low=90.0, high=130.0, unit="mm[Hg]", critical_high=180.0),
        ),
        tags=frozenset({"cardiovascular", "blood_pressure"}),
    ),
    ConceptDefinition(
        key="diastolic_bp",
        display="Diastolic blood pressure",
        kind=ConceptKind.VITAL,
        codings=(_loinc("8462-4", "Diastolic blood pressure"),),
        canonical_unit="mm[Hg]",
        reference_intervals=(
            ReferenceInterval(low=60.0, high=80.0, unit="mm[Hg]", critical_high=120.0),
        ),
        tags=frozenset({"cardiovascular", "blood_pressure"}),
    ),
    ConceptDefinition(
        key="blood_pressure_panel",
        display="Blood pressure panel",
        kind=ConceptKind.PANEL,
        codings=(_loinc("85354-9", "Blood pressure panel with all children optional"),),
        tags=frozenset({"cardiovascular", "blood_pressure"}),
    ),
    ConceptDefinition(
        key="body_weight",
        display="Body weight",
        kind=ConceptKind.VITAL,
        codings=(_loinc("29463-7", "Body weight"),),
        canonical_unit="kg",
        tags=frozenset({"anthropometry"}),
    ),
    ConceptDefinition(
        key="body_height",
        display="Body height",
        kind=ConceptKind.VITAL,
        codings=(_loinc("8302-2", "Body height"),),
        canonical_unit="cm",
        tags=frozenset({"anthropometry"}),
    ),
    ConceptDefinition(
        key="bmi",
        display="Body mass index",
        kind=ConceptKind.VITAL,
        codings=(_loinc("39156-5", "Body mass index (BMI) [Ratio]"),),
        canonical_unit="kg/m2",
        reference_intervals=(ReferenceInterval(low=18.5, high=25.0, unit="kg/m2"),),
        tags=frozenset({"anthropometry", "metabolic"}),
    ),
    ConceptDefinition(
        key="heart_rate",
        display="Heart rate",
        kind=ConceptKind.VITAL,
        codings=(_loinc("8867-4", "Heart rate"),),
        canonical_unit="/min",
        reference_intervals=(
            ReferenceInterval(low=60.0, high=100.0, unit="/min", critical_high=140.0),
        ),
        tags=frozenset({"cardiovascular"}),
    ),
)

# --------------------------------------------------------------------------------------
# Conditions
# --------------------------------------------------------------------------------------

_CONDITIONS: tuple[ConceptDefinition, ...] = (
    ConceptDefinition(
        key="type_2_diabetes",
        display="Type 2 diabetes mellitus",
        kind=ConceptKind.CONDITION,
        codings=(
            _snomed("44054006", "Type 2 diabetes mellitus"),
            _snomed("422034002", "Diabetic retinopathy associated with type 2 diabetes mellitus"),
            _icd10("E11.9", "Type 2 diabetes mellitus without complications"),
            _icd10("E11.65", "Type 2 diabetes mellitus with hyperglycemia"),
        ),
        tags=frozenset({"diabetes", "chronic"}),
    ),
    ConceptDefinition(
        key="type_1_diabetes",
        display="Type 1 diabetes mellitus",
        kind=ConceptKind.CONDITION,
        codings=(
            _snomed("46635009", "Type 1 diabetes mellitus"),
            _icd10("E10.9", "Type 1 diabetes mellitus without complications"),
        ),
        tags=frozenset({"diabetes", "chronic"}),
    ),
    ConceptDefinition(
        key="hypertension",
        display="Essential hypertension",
        kind=ConceptKind.CONDITION,
        codings=(
            _snomed("59621000", "Essential hypertension"),
            _snomed("38341003", "Hypertensive disorder"),
            _icd10("I10", "Essential (primary) hypertension"),
        ),
        tags=frozenset({"cardiovascular", "chronic"}),
    ),
    ConceptDefinition(
        key="chronic_kidney_disease",
        display="Chronic kidney disease",
        kind=ConceptKind.CONDITION,
        codings=(
            _snomed("709044004", "Chronic kidney disease"),
            _snomed("433144002", "Chronic kidney disease stage 3"),
            _icd10("N18.3", "Chronic kidney disease, stage 3"),
            _icd10("N18.9", "Chronic kidney disease, unspecified"),
        ),
        tags=frozenset({"renal", "chronic"}),
    ),
    ConceptDefinition(
        key="heart_failure",
        display="Heart failure",
        kind=ConceptKind.CONDITION,
        codings=(
            _snomed("84114007", "Heart failure"),
            _icd10("I50.9", "Heart failure, unspecified"),
        ),
        tags=frozenset({"cardiovascular", "chronic"}),
    ),
    ConceptDefinition(
        key="hyperlipidemia",
        display="Hyperlipidemia",
        kind=ConceptKind.CONDITION,
        codings=(
            _snomed("55822004", "Hyperlipidemia"),
            _icd10("E78.5", "Hyperlipidemia, unspecified"),
        ),
        tags=frozenset({"cardiovascular", "metabolic", "chronic"}),
    ),
    ConceptDefinition(
        key="obesity",
        display="Obesity",
        kind=ConceptKind.CONDITION,
        codings=(
            _snomed("414916001", "Obesity"),
            _icd10("E66.9", "Obesity, unspecified"),
        ),
        tags=frozenset({"metabolic", "chronic"}),
    ),
    ConceptDefinition(
        key="myocardial_infarction",
        display="Myocardial infarction",
        kind=ConceptKind.CONDITION,
        codings=(
            _snomed("22298006", "Myocardial infarction"),
            _icd10("I21.9", "Acute myocardial infarction, unspecified"),
        ),
        tags=frozenset({"cardiovascular", "acute"}),
    ),
    ConceptDefinition(
        key="copd",
        display="Chronic obstructive pulmonary disease",
        kind=ConceptKind.CONDITION,
        codings=(
            _snomed("13645005", "Chronic obstructive lung disease"),
            _icd10("J44.9", "Chronic obstructive pulmonary disease, unspecified"),
        ),
        tags=frozenset({"respiratory", "chronic"}),
    ),
    ConceptDefinition(
        key="diabetic_nephropathy",
        display="Diabetic nephropathy",
        kind=ConceptKind.CONDITION,
        codings=(
            _snomed("127013003", "Disorder of kidney due to diabetes mellitus"),
            _icd10("E11.21", "Type 2 diabetes mellitus with diabetic nephropathy"),
        ),
        tags=frozenset({"diabetes", "renal", "chronic"}),
    ),
)

# --------------------------------------------------------------------------------------
# Medications
# --------------------------------------------------------------------------------------


def _med(key: str, display: str, code: str, drug_class: str, *extra_tags: str) -> ConceptDefinition:
    return ConceptDefinition(
        key=key,
        display=display,
        kind=ConceptKind.MEDICATION,
        codings=(_rxnorm(code, display),),
        drug_class=drug_class,
        tags=frozenset({drug_class, *extra_tags}),
    )


_MEDICATIONS: tuple[ConceptDefinition, ...] = (
    _med("metformin", "Metformin", "6809", "biguanide", "diabetes_medication"),
    _med("glipizide", "Glipizide", "4821", "sulfonylurea", "diabetes_medication"),
    _med("sitagliptin", "Sitagliptin", "593411", "dpp4_inhibitor", "diabetes_medication"),
    _med("empagliflozin", "Empagliflozin", "1545653", "sglt2_inhibitor", "diabetes_medication"),
    _med("dapagliflozin", "Dapagliflozin", "1488564", "sglt2_inhibitor", "diabetes_medication"),
    _med("semaglutide", "Semaglutide", "1991302", "glp1_agonist", "diabetes_medication"),
    _med("liraglutide", "Liraglutide", "475968", "glp1_agonist", "diabetes_medication"),
    _med("insulin_glargine", "Insulin glargine", "274783", "insulin", "diabetes_medication"),
    _med("lisinopril", "Lisinopril", "29046", "ace_inhibitor", "antihypertensive"),
    _med("losartan", "Losartan", "52175", "arb", "antihypertensive"),
    _med("amlodipine", "Amlodipine", "17767", "calcium_channel_blocker", "antihypertensive"),
    _med("hydrochlorothiazide", "Hydrochlorothiazide", "5487", "thiazide", "antihypertensive"),
    _med("metoprolol", "Metoprolol", "6918", "beta_blocker", "antihypertensive"),
    _med("atorvastatin", "Atorvastatin", "83367", "statin", "lipid_lowering"),
    _med("simvastatin", "Simvastatin", "36567", "statin", "lipid_lowering"),
    _med("furosemide", "Furosemide", "4603", "loop_diuretic"),
    _med("apixaban", "Apixaban", "1364430", "anticoagulant"),
    _med("aspirin", "Aspirin", "1191", "antiplatelet"),
    _med("levothyroxine", "Levothyroxine", "10582", "thyroid_hormone"),
)


ALL_CONCEPTS: tuple[ConceptDefinition, ...] = _LABS + _VITALS + _CONDITIONS + _MEDICATIONS

_BY_KEY: dict[str, ConceptDefinition] = {c.key: c for c in ALL_CONCEPTS}
_BY_TOKEN: dict[str, ConceptDefinition] = {}
_BY_BARE_CODE: dict[str, ConceptDefinition] = {}
for _concept in ALL_CONCEPTS:
    for _coding in _concept.codings:
        if _coding.system and _coding.code:
            _BY_TOKEN[f"{_coding.system}|{_coding.code}"] = _concept
        if _coding.code:
            _BY_BARE_CODE.setdefault(_coding.code, _concept)


# --------------------------------------------------------------------------------------
# Lookup API
# --------------------------------------------------------------------------------------


def get_concept(key: str) -> ConceptDefinition | None:
    """Look up a concept by its canonical key."""
    return _BY_KEY.get(key)


def require_concept(key: str) -> ConceptDefinition:
    """Look up a concept, raising if it is unknown."""
    concept = _BY_KEY.get(key)
    if concept is None:
        raise KeyError(f"unknown clinical concept: {key!r}")
    return concept


def known_concept_keys() -> tuple[str, ...]:
    """Every concept key, sorted. Used to constrain LLM output."""
    return tuple(sorted(_BY_KEY))


def resolve_coding(system: str | None, code: str | None) -> ConceptDefinition | None:
    """Map a source coding onto a canonical concept.

    Falls back to a system-agnostic code match, which is how data from servers that
    omit ``Coding.system`` still resolves.
    """
    if not code:
        return None
    if system:
        hit = _BY_TOKEN.get(f"{system}|{code}")
        if hit is not None:
            return hit
    return _BY_BARE_CODE.get(code)


def resolve_codings(codings: Iterable[Coding]) -> ConceptDefinition | None:
    """First concept matched by any of the supplied codings."""
    for coding in codings:
        concept = resolve_coding(coding.system, coding.code)
        if concept is not None:
            return concept
    return None


def concepts_by_kind(kind: ConceptKind) -> tuple[ConceptDefinition, ...]:
    return tuple(c for c in ALL_CONCEPTS if c.kind is kind)


def concepts_with_tag(tag: str) -> tuple[ConceptDefinition, ...]:
    return tuple(c for c in ALL_CONCEPTS if tag in c.tags)


def drug_class_of(key: str) -> str | None:
    concept = _BY_KEY.get(key)
    return concept.drug_class if concept else None


# --------------------------------------------------------------------------------------
# Unit conversion
# --------------------------------------------------------------------------------------

_UNIT_ALIASES: dict[str, str] = {
    "%": "%",
    "percent": "%",
    "mg/dl": "mg/dL",
    "mg/dL": "mg/dL",
    "mgs/dl": "mg/dL",
    "mmol/l": "mmol/L",
    "mmol/L": "mmol/L",
    "umol/l": "umol/L",
    "µmol/l": "umol/L",
    "umol/L": "umol/L",
    "mmol/mol": "mmol/mol",
    "g/dl": "g/dL",
    "g/dL": "g/dL",
    "g/l": "g/L",
    "u/l": "U/L",
    "iu/l": "U/L",
    "mm[hg]": "mm[Hg]",
    "mmhg": "mm[Hg]",
    "mm hg": "mm[Hg]",
    "kg": "kg",
    "g": "g",
    "lb": "lb",
    "lbs": "lb",
    "[lb_av]": "lb",
    "cm": "cm",
    "m": "m",
    "in": "in",
    "[in_i]": "in",
    "kg/m2": "kg/m2",
    "kg/m^2": "kg/m2",
    "/min": "/min",
    "bpm": "/min",
    "{beats}/min": "/min",
    "ml/min/1.73m2": "mL/min/1.73m2",
    "ml/min/{1.73_m2}": "mL/min/1.73m2",
    "mg/g": "mg/g",
    "mg/mmol": "mg/mmol",
}

# (from_unit, to_unit) -> multiplicative factor. Concept-specific entries win.
_LINEAR_CONVERSIONS: dict[tuple[str, str], float] = {
    ("g", "kg"): 0.001,
    ("lb", "kg"): 0.45359237,
    ("in", "cm"): 2.54,
    ("m", "cm"): 100.0,
    ("g/L", "g/dL"): 0.1,
    ("umol/L", "mg/dL"): 1 / 88.4,  # creatinine
    ("mg/mmol", "mg/g"): 8.84,
}

_CONCEPT_CONVERSIONS: dict[tuple[str, str, str], float] = {
    ("glucose", "mmol/L", "mg/dL"): 18.0182,
    ("total_cholesterol", "mmol/L", "mg/dL"): 38.67,
    ("ldl_cholesterol", "mmol/L", "mg/dL"): 38.67,
    ("hdl_cholesterol", "mmol/L", "mg/dL"): 38.67,
    ("triglycerides", "mmol/L", "mg/dL"): 88.57,
    ("creatinine", "umol/L", "mg/dL"): 1 / 88.4,
}


def normalize_unit(unit: str | None) -> str | None:
    """Map a free-text unit onto its canonical spelling."""
    if not unit:
        return None
    stripped = unit.strip()
    return _UNIT_ALIASES.get(stripped.lower(), _UNIT_ALIASES.get(stripped, stripped))


class UnitConversionError(ValueError):
    """Raised when a value cannot be expressed in the requested unit."""


def convert_value(
    value: float, from_unit: str | None, to_unit: str | None, concept_key: str | None = None
) -> float:
    """Convert ``value`` between units.

    HbA1c is special-cased because the NGSP (%) and IFCC (mmol/mol) scales are related
    by an affine transform, not a ratio -- multiplying would be silently wrong.
    """
    src = normalize_unit(from_unit)
    dst = normalize_unit(to_unit)
    if dst is None or src is None or src == dst:
        return value

    if concept_key == "hba1c":
        if src == "mmol/mol" and dst == "%":
            return value / 10.929 + 2.15
        if src == "%" and dst == "mmol/mol":
            return (value - 2.15) * 10.929

    if concept_key is not None:
        factor = _CONCEPT_CONVERSIONS.get((concept_key, src, dst))
        if factor is not None:
            return value * factor
        inverse = _CONCEPT_CONVERSIONS.get((concept_key, dst, src))
        if inverse is not None:
            return value / inverse

    factor = _LINEAR_CONVERSIONS.get((src, dst))
    if factor is not None:
        return value * factor
    inverse = _LINEAR_CONVERSIONS.get((dst, src))
    if inverse is not None:
        return value / inverse

    raise UnitConversionError(f"no conversion from {src!r} to {dst!r} (concept={concept_key!r})")


def to_canonical(
    value: float | None, unit: str | None, concept: ConceptDefinition | None
) -> tuple[float | None, str | None]:
    """Best-effort conversion of a value onto its concept's canonical unit.

    Returns the input unchanged when no canonical unit is defined or no conversion is
    known; callers record that as a parse warning rather than dropping the value.
    """
    if value is None or concept is None or concept.canonical_unit is None:
        return value, normalize_unit(unit)
    try:
        converted = convert_value(value, unit, concept.canonical_unit, concept.key)
    except UnitConversionError:
        return None, None
    return converted, concept.canonical_unit
