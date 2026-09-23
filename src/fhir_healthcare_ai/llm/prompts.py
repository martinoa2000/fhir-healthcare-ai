"""Prompt construction for the query planner.

The prompt is *generated from the allowlist*, not hand-written alongside it. If a
resource, search parameter or coding system is added to
:mod:`fhir_healthcare_ai.fhir.allowlist`, the model is told about it on the next call;
if one is removed, the model stops hearing about it. A hand-maintained prompt would
drift from the validator, and the failure mode of that drift is a model confidently
producing plans that are always rejected.

The prompt tells the model what it may emit. It is the validator that decides what
actually runs. Prompt text is guidance; the allowlist is enforcement. Never move a rule
from the validator into the prompt.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from fhir_healthcare_ai.domain.enums import AnalysisType, CohortLogic, QueryIntent, StepRole
from fhir_healthcare_ai.fhir.allowlist import describe_allowlist
from fhir_healthcare_ai.terminology import (
    ConceptDefinition,
    ConceptKind,
    concepts_by_kind,
    get_concept,
)

MAX_STEPS_HINT = 8

SYSTEM_PROMPT = f"""\
You are a FHIR query planner for a clinical analytics system. You translate a clinical \
question in natural language into a structured query plan.

You do NOT have network access, you never write URLs, and you never decide what runs. \
You emit one JSON object describing intent. A separate validator checks it against a \
strict allowlist and refuses anything outside it, so inventing a resource type, a search \
parameter or a code does not widen your access -- it only makes the plan fail.

Rules you must follow:

1. Output exactly one JSON object and nothing else. No prose, no code fences.
2. Use only the resource types and search parameters listed in CAPABILITIES below.
3. Use only the concept keys listed in CONCEPTS below. To search for a clinical concept, \
put its concept key in `values` and the system in `system`; the builder expands the key \
into the correct codes. Never invent a LOINC, SNOMED, RxNorm or ICD-10 code yourself. \
For conditions, omit `system`: diagnoses are coded in SNOMED CT by some sources and \
ICD-10-CM by others, and a search pinned to one system misses the other patients.
4. Every step that searches a patient-scoped resource must include at least one selective \
filter (a code, a category, a date range, or a dependency on an earlier step). A bare \
`Observation?_count=100` is a data scan and will be refused.
5. Use `depends_on` when a step should only run for the patients an earlier step found. \
That is how you express "and", and it is far cheaper than fetching everything twice.
6. Give every step a `role`. "filter" (the default) means a patient must be returned by \
the step to be in the cohort. "context" fetches data about patients already selected \
(their recent labs) and never removes anyone. "exclude" removes every patient the step \
returns; it is the only way to express "without" or "not on", because FHIR search cannot \
filter for an absent resource.
7. `cohort_logic` is "all" when the question joins criteria with "and", "any" when it \
joins them with "or".
8. Use at most {MAX_STEPS_HINT} steps. Fewer, more selective steps beat many broad ones.
9. If the question cannot be answered with the capabilities below, set `unsupported` to \
true and explain why in `unsupported_reason`. Do not approximate. A wrong cohort is worse \
than an honest refusal.
10. Record any clinical assumption you had to make in `assumptions` -- what "elevated" or \
"recent" was taken to mean, which codes stand in for a vague term. These are shown to the \
user, so they must be complete.
11. You are not making a clinical decision. You are retrieving evidence for a human to \
review.
"""


PLAN_SCHEMA: dict[str, object] = {
    "question": "string -- the original question, verbatim",
    "intent": [intent.value for intent in QueryIntent],
    "rationale": "string -- one or two sentences on why this plan answers the question",
    "cohort_logic": [logic.value for logic in CohortLogic],
    "steps": [
        {
            "step_id": "string -- lowercase slug, unique within the plan",
            "resource_type": "string -- must appear in CAPABILITIES",
            "purpose": "string -- what this step contributes",
            "params": [
                {
                    "name": "string -- a search parameter from CAPABILITIES",
                    "values": ["string -- concept key, code, or literal value"],
                    "comparator": "optional -- eq|ne|gt|lt|ge|le|sa|eb|ap",
                    "modifier": "optional -- e.g. 'in', 'not'",
                    "system": "optional -- coding system URI for token searches",
                    "unit": "optional -- UCUM unit for quantity searches",
                }
            ],
            "include": ["optional -- e.g. 'Observation:patient'"],
            "revinclude": ["optional"],
            "sort": "optional -- e.g. '-date'",
            "count": "optional integer",
            "depends_on": "optional -- step_id whose patients scope this step",
            "role": [role.value for role in StepRole],
        }
    ],
    "analysis": {
        "type": [analysis.value for analysis in AnalysisType],
        "concepts": ["optional -- concept keys the analysis should focus on"],
        "lookback_days": "optional integer",
        "options": {
            "require_abnormal": (
                "optional boolean -- keep only patients with at least one result outside "
                "the reference interval for `concepts`. Use it for 'abnormal' or "
                "'out of range' questions instead of guessing a value threshold."
            )
        },
    },
    "assumptions": ["string"],
    "unsupported": "boolean",
    "unsupported_reason": "string or null",
}


FEW_SHOT_QUESTION = "Find patients with elevated HbA1c and a recent change in diabetes medication"

FEW_SHOT_PLAN: dict[str, object] = {
    "question": FEW_SHOT_QUESTION,
    "intent": "cohort_search",
    "rationale": (
        "Select patients by an out-of-target HbA1c result, then restrict to those whose "
        "glucose-lowering therapy was ordered or changed in the last 90 days."
    ),
    "cohort_logic": "all",
    "steps": [
        {
            "step_id": "elevated_hba1c",
            "resource_type": "Observation",
            "purpose": "Patients with an HbA1c result above the 7% target in the last year.",
            "params": [
                {"name": "code", "values": ["hba1c"], "system": "http://loinc.org"},
                {"name": "value-quantity", "values": ["7"], "comparator": "gt", "unit": "%"},
                {"name": "date", "values": ["2024-01-01"], "comparator": "ge"},
            ],
            "sort": "-date",
            "count": 200,
        },
        {
            "step_id": "recent_med_change",
            "resource_type": "MedicationRequest",
            "purpose": "Glucose-lowering orders authored recently for those patients.",
            "depends_on": "elevated_hba1c",
            "role": "filter",
            "params": [
                {
                    "name": "code",
                    "values": ["metformin", "glipizide", "empagliflozin", "insulin_glargine"],
                    "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                },
                {"name": "authoredon", "values": ["2024-10-01"], "comparator": "ge"},
            ],
            "count": 200,
        },
    ],
    "analysis": {"type": "abnormal_labs", "concepts": ["hba1c"], "lookback_days": 365},
    "assumptions": [
        "'elevated HbA1c' was read as a result above 7%, the common general target.",
        "'recent' was read as the last 90 days.",
        "'diabetes medication' covers the glucose-lowering classes in the concept list.",
    ],
    "unsupported": False,
    "unsupported_reason": None,
}


def build_capabilities_block() -> str:
    """The allowlist, rendered for the model."""
    return json.dumps(describe_allowlist(), indent=2, sort_keys=True)


def build_concepts_block() -> str:
    """Concept keys grouped by kind, with the unit and system the model should use."""
    groups: dict[str, list[str]] = {}
    for kind in ConceptKind:
        entries = [_concept_line(concept) for concept in concepts_by_kind(kind)]
        if entries:
            groups[kind.value] = entries
    return json.dumps(groups, indent=2)


def _concept_line(concept: ConceptDefinition) -> str:
    parts = [concept.key, f"= {concept.display}"]
    if concept.canonical_unit:
        parts.append(f"[{concept.canonical_unit}]")
    if concept.drug_class:
        parts.append(f"class={concept.drug_class}")
    return " ".join(parts)


def build_planning_prompt(
    question: str,
    *,
    today: str,
    context: str | None = None,
    repair_feedback: str | None = None,
    previous_plan: str | None = None,
) -> tuple[str, str]:
    """Return ``(system_prompt, user_prompt)`` for one planning call.

    Args:
        question: The clinician's question, passed through unmodified. It is data, and
            it is interpolated into a user turn rather than the system turn so that a
            question containing instruction-shaped text cannot rewrite the rules.
        today: The reference date, so that "recent" and "last year" resolve to fixed
            windows and the same question asked twice gives the same plan.
        context: Optional extra grounding, e.g. a patient id for a summary request.
        repair_feedback: Validator issues from a rejected attempt. Present only on a
            retry, and it is what makes the retry worth making.
        previous_plan: The rejected plan, so the model repairs rather than restarts.
    """
    system = "\n\n".join(
        [
            SYSTEM_PROMPT,
            "CAPABILITIES (the only resources and parameters that exist):\n"
            + build_capabilities_block(),
            "CONCEPTS (the only concept keys you may use in `values`):\n" + build_concepts_block(),
            "OUTPUT SCHEMA:\n" + json.dumps(PLAN_SCHEMA, indent=2),
            "EXAMPLE\nQuestion: "
            + FEW_SHOT_QUESTION
            + "\nPlan:\n"
            + json.dumps(FEW_SHOT_PLAN, indent=2),
        ]
    )

    user_parts = [f"Today's date is {today}.", f"Question: {question}"]
    if context:
        user_parts.append(f"Additional context: {context}")
    if previous_plan and repair_feedback:
        user_parts.append(
            "Your previous plan was REJECTED by the validator:\n"
            + previous_plan
            + "\n\nIssues:\n"
            + repair_feedback
            + "\n\nEmit a corrected plan. Fix only what the issues name. If the question "
            "cannot be expressed within the capabilities, set unsupported to true."
        )
    user_parts.append("Respond with the JSON plan object only.")
    return system, "\n\n".join(user_parts)


def build_narrative_prompt(
    question: str,
    *,
    plan_summary: str,
    findings: str,
    patient_count: int,
) -> tuple[str, str]:
    """Prompt for the optional natural-language summary of a result set.

    The model is given only aggregates and counts that the pipeline already computed.
    It is explicitly forbidden from adding numbers, because a narrative that invents a
    statistic is indistinguishable from one that reports a real one.
    """
    system = (
        "You summarise the results of a clinical data query for a technical reviewer.\n\n"
        "Rules:\n"
        "1. Use only the numbers given to you. Never compute, estimate or infer a new one.\n"
        "2. Never name or describe an individual patient beyond the identifiers provided.\n"
        "3. Never give clinical advice, a diagnosis, or a recommendation for any person.\n"
        "4. State what the query looked for and what it found, including any stated "
        "assumption or data gap.\n"
        "5. Three sentences at most. Plain prose, no headings, no bullet points."
    )
    user = (
        f"Question: {question}\n\n"
        f"Plan: {plan_summary}\n\n"
        f"Patients matched: {patient_count}\n\n"
        f"Findings:\n{findings}\n\n"
        "Write the summary."
    )
    return system, user


def format_concept_keys(keys: Sequence[str]) -> str:
    """Render concept keys with their displays, for assumption text and error messages."""
    rendered = []
    for key in keys:
        concept = get_concept(key)
        rendered.append(f"{key} ({concept.display})" if concept else key)
    return ", ".join(rendered)
