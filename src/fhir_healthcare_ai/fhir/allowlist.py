"""The allowlist: the security boundary between generated plans and the FHIR server.

Nothing reaches the network unless it is described here. This is an allowlist, not a
blocklist, and that asymmetry is the whole point -- a model that invents
``Patient?_query=evil`` or asks for ``Binary`` fails closed because those were never
enumerated, not because someone remembered to forbid them.

Each resource declares:

* which search parameters may be used, and of what type
* which comparators and modifiers are legal per parameter
* which code systems a token parameter may reference
* which ``_include`` / ``_revinclude`` / ``_sort`` values are permitted
* an upper bound on page size

Widening access is a reviewable, one-file diff.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from fhir_healthcare_ai.domain.enums import Comparator, ResourceType
from fhir_healthcare_ai.terminology import ICD10, LOINC, RXNORM, SNOMED

ORDERED_COMPARATORS: frozenset[Comparator] = frozenset(
    {
        Comparator.EQ,
        Comparator.NE,
        Comparator.GT,
        Comparator.LT,
        Comparator.GE,
        Comparator.LE,
        Comparator.SA,
        Comparator.EB,
        Comparator.AP,
    }
)


class ParamType(StrEnum):
    """FHIR search parameter types that this system supports."""

    TOKEN = "token"
    DATE = "date"
    QUANTITY = "quantity"
    NUMBER = "number"
    STRING = "string"
    REFERENCE = "reference"


# Conservative value patterns. These are the last line of defence against injection of
# extra parameters (``&``), path traversal or control characters into a query string.
_VALUE_PATTERNS: dict[ParamType, re.Pattern[str]] = {
    ParamType.TOKEN: re.compile(r"^[A-Za-z0-9\-_.:]{1,64}$"),
    ParamType.DATE: re.compile(
        r"^\d{4}(-\d{2}(-\d{2}(T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+\-]\d{2}:\d{2})?)?)?)?$"
    ),
    ParamType.QUANTITY: re.compile(r"^-?\d+(\.\d+)?$"),
    ParamType.NUMBER: re.compile(r"^-?\d+(\.\d+)?$"),
    ParamType.STRING: re.compile(r"^[\w\s\-.,'()/]{1,100}$", re.UNICODE),
    ParamType.REFERENCE: re.compile(
        r"^[A-Za-z]{1,40}/[A-Za-z0-9\-.]{1,64}$|^[A-Za-z0-9\-.]{1,64}$"
    ),
}

CLINICAL_CODE_SYSTEMS: frozenset[str] = frozenset({LOINC, SNOMED, RXNORM, ICD10})


@dataclass(frozen=True)
class ParamPolicy:
    """What a single search parameter is allowed to look like."""

    name: str
    type: ParamType
    description: str = ""
    comparators: frozenset[Comparator] = field(default_factory=frozenset)
    modifiers: frozenset[str] = field(default_factory=frozenset)
    systems: frozenset[str] = field(default_factory=frozenset)
    max_values: int = 20
    value_pattern: re.Pattern[str] | None = None

    def pattern(self) -> re.Pattern[str]:
        return self.value_pattern or _VALUE_PATTERNS[self.type]

    def allows_comparator(self, comparator: Comparator | None) -> bool:
        if comparator is None:
            return True
        return comparator in self.comparators

    def allows_modifier(self, modifier: str | None) -> bool:
        if modifier is None:
            return True
        return modifier in self.modifiers

    def allows_system(self, system: str | None) -> bool:
        if system is None:
            return True
        return system in self.systems


@dataclass(frozen=True)
class ResourcePolicy:
    """Everything permitted for one resource type."""

    resource_type: ResourceType
    params: dict[str, ParamPolicy]
    includes: frozenset[str] = field(default_factory=frozenset)
    revincludes: frozenset[str] = field(default_factory=frozenset)
    sorts: frozenset[str] = field(default_factory=frozenset)
    max_count: int = 200
    patient_param: str | None = "patient"
    read_enabled: bool = True

    def param(self, name: str) -> ParamPolicy | None:
        return self.params.get(name)


def _token(
    name: str,
    description: str = "",
    *,
    systems: frozenset[str] = CLINICAL_CODE_SYSTEMS,
    modifiers: frozenset[str] = frozenset({"text", "not", "in"}),
) -> ParamPolicy:
    return ParamPolicy(
        name=name,
        type=ParamType.TOKEN,
        description=description,
        systems=systems,
        modifiers=modifiers,
    )


def _status_token(name: str, description: str, allowed: frozenset[str]) -> ParamPolicy:
    """Status-style token parameter whose values come from a closed set."""
    return ParamPolicy(
        name=name,
        type=ParamType.TOKEN,
        description=description,
        systems=frozenset(),
        modifiers=frozenset({"not"}),
        value_pattern=re.compile(rf"^({'|'.join(re.escape(v) for v in sorted(allowed))})$"),
    )


def _date(name: str, description: str = "") -> ParamPolicy:
    return ParamPolicy(
        name=name,
        type=ParamType.DATE,
        description=description,
        comparators=ORDERED_COMPARATORS,
        modifiers=frozenset({"missing"}),
        max_values=2,
    )


def _quantity(name: str, description: str = "") -> ParamPolicy:
    return ParamPolicy(
        name=name,
        type=ParamType.QUANTITY,
        description=description,
        comparators=ORDERED_COMPARATORS,
        max_values=2,
    )


def _reference(name: str, description: str = "") -> ParamPolicy:
    return ParamPolicy(
        name=name,
        type=ParamType.REFERENCE,
        description=description,
        max_values=100,  # cohort fan-out: patient=id1,id2,...
    )


_PATIENT_POLICY = ResourcePolicy(
    resource_type=ResourceType.PATIENT,
    patient_param=None,
    params={
        "_id": ParamPolicy(
            name="_id", type=ParamType.TOKEN, description="Logical id", max_values=100
        ),
        "identifier": _token("identifier", "Business identifier", systems=frozenset()),
        "gender": _status_token(
            "gender", "Administrative gender", frozenset({"male", "female", "other", "unknown"})
        ),
        "birthdate": _date("birthdate", "Date of birth"),
        "deceased": _status_token("deceased", "Deceased flag", frozenset({"true", "false"})),
        "address-postalcode": ParamPolicy(
            name="address-postalcode", type=ParamType.STRING, description="Postal code"
        ),
        "active": _status_token("active", "Record active flag", frozenset({"true", "false"})),
    },
    revincludes=frozenset(
        {
            "Observation:patient",
            "Condition:patient",
            "MedicationRequest:patient",
            "Encounter:patient",
            "DiagnosticReport:patient",
        }
    ),
    sorts=frozenset({"birthdate", "-birthdate", "_id", "_lastUpdated", "-_lastUpdated"}),
)

_OBSERVATION_POLICY = ResourcePolicy(
    resource_type=ResourceType.OBSERVATION,
    params={
        "_id": ParamPolicy(name="_id", type=ParamType.TOKEN, description="Logical id"),
        "patient": _reference("patient", "Subject of the observation"),
        "subject": _reference("subject", "Subject of the observation"),
        "encounter": _reference("encounter", "Encounter the observation belongs to"),
        "code": _token("code", "Observation code (LOINC)"),
        "category": _token(
            "category",
            "Observation category",
            systems=frozenset({"http://terminology.hl7.org/CodeSystem/observation-category"}),
        ),
        "date": _date("date", "Clinically relevant time"),
        "value-quantity": _quantity("value-quantity", "Numeric result"),
        "status": _status_token(
            "status",
            "Observation status",
            frozenset(
                {
                    "registered",
                    "preliminary",
                    "final",
                    "amended",
                    "corrected",
                    "cancelled",
                    "entered-in-error",
                    "unknown",
                }
            ),
        ),
        "combo-code": _token("combo-code", "Code including components"),
        "combo-value-quantity": _quantity("combo-value-quantity", "Value including components"),
        # Blood pressure is a panel whose systolic and diastolic values live in components,
        # never on the Observation itself, so without these a BP question is unanswerable.
        "component-code": _token("component-code", "Code of an observation component"),
        "component-value-quantity": _quantity(
            "component-value-quantity", "Numeric value of an observation component"
        ),
    },
    includes=frozenset({"Observation:patient", "Observation:encounter"}),
    sorts=frozenset({"date", "-date", "patient", "_lastUpdated", "-_lastUpdated"}),
)

_CONDITION_POLICY = ResourcePolicy(
    resource_type=ResourceType.CONDITION,
    params={
        "_id": ParamPolicy(name="_id", type=ParamType.TOKEN, description="Logical id"),
        "patient": _reference("patient", "Who has the condition"),
        "subject": _reference("subject", "Who has the condition"),
        "encounter": _reference("encounter", "Encounter of diagnosis"),
        "code": _token("code", "Condition code (SNOMED CT / ICD-10)"),
        "category": _token(
            "category",
            "problem-list-item | encounter-diagnosis",
            systems=frozenset({"http://terminology.hl7.org/CodeSystem/condition-category"}),
        ),
        "clinical-status": _status_token(
            "clinical-status",
            "Clinical status",
            frozenset({"active", "recurrence", "relapse", "inactive", "remission", "resolved"}),
        ),
        "verification-status": _status_token(
            "verification-status",
            "Verification status",
            frozenset(
                {
                    "unconfirmed",
                    "provisional",
                    "differential",
                    "confirmed",
                    "refuted",
                    "entered-in-error",
                }
            ),
        ),
        "onset-date": _date("onset-date", "Date of onset"),
        "recorded-date": _date("recorded-date", "Date the condition was recorded"),
    },
    includes=frozenset({"Condition:patient", "Condition:encounter"}),
    sorts=frozenset({"onset-date", "-onset-date", "recorded-date", "-recorded-date"}),
)

_MEDICATION_REQUEST_POLICY = ResourcePolicy(
    resource_type=ResourceType.MEDICATION_REQUEST,
    params={
        "_id": ParamPolicy(name="_id", type=ParamType.TOKEN, description="Logical id"),
        "patient": _reference("patient", "Subject of the request"),
        "subject": _reference("subject", "Subject of the request"),
        "encounter": _reference("encounter", "Encounter of the request"),
        "code": _token("code", "Medication code (RxNorm)"),
        "status": _status_token(
            "status",
            "Request status",
            frozenset(
                {
                    "active",
                    "on-hold",
                    "cancelled",
                    "completed",
                    "entered-in-error",
                    "stopped",
                    "draft",
                    "unknown",
                }
            ),
        ),
        "intent": _status_token(
            "intent",
            "Request intent",
            frozenset(
                {
                    "proposal",
                    "plan",
                    "order",
                    "original-order",
                    "reflex-order",
                    "filler-order",
                    "instance-order",
                    "option",
                }
            ),
        ),
        "authoredon": _date("authoredon", "Date the request was authored"),
    },
    includes=frozenset({"MedicationRequest:patient", "MedicationRequest:encounter"}),
    sorts=frozenset({"authoredon", "-authoredon"}),
)

_ENCOUNTER_POLICY = ResourcePolicy(
    resource_type=ResourceType.ENCOUNTER,
    params={
        "_id": ParamPolicy(name="_id", type=ParamType.TOKEN, description="Logical id"),
        "patient": _reference("patient", "Subject of the encounter"),
        "subject": _reference("subject", "Subject of the encounter"),
        "class": _token(
            "class",
            "Encounter class (AMB, IMP, EMER...)",
            systems=frozenset({"http://terminology.hl7.org/CodeSystem/v3-ActCode"}),
        ),
        "type": _token("type", "Encounter type"),
        "status": _status_token(
            "status",
            "Encounter status",
            frozenset(
                {
                    "planned",
                    "arrived",
                    "triaged",
                    "in-progress",
                    "onleave",
                    "finished",
                    "cancelled",
                    "entered-in-error",
                    "unknown",
                }
            ),
        ),
        "date": _date("date", "Encounter period"),
        "reason-code": _token("reason-code", "Reason for the encounter"),
    },
    includes=frozenset({"Encounter:patient"}),
    sorts=frozenset({"date", "-date"}),
)

_DIAGNOSTIC_REPORT_POLICY = ResourcePolicy(
    resource_type=ResourceType.DIAGNOSTIC_REPORT,
    params={
        "_id": ParamPolicy(name="_id", type=ParamType.TOKEN, description="Logical id"),
        "patient": _reference("patient", "Subject of the report"),
        "subject": _reference("subject", "Subject of the report"),
        "encounter": _reference("encounter", "Encounter of the report"),
        "code": _token("code", "Report code (LOINC)"),
        "category": _token(
            "category",
            "Service category",
            systems=frozenset({"http://terminology.hl7.org/CodeSystem/v2-0074"}),
        ),
        "date": _date("date", "Clinically relevant time"),
        "issued": _date("issued", "Date the report was released"),
        "status": _status_token(
            "status",
            "Report status",
            frozenset(
                {
                    "registered",
                    "partial",
                    "preliminary",
                    "final",
                    "amended",
                    "corrected",
                    "appended",
                    "cancelled",
                    "entered-in-error",
                    "unknown",
                }
            ),
        ),
    },
    includes=frozenset({"DiagnosticReport:patient", "DiagnosticReport:result"}),
    sorts=frozenset({"date", "-date", "issued", "-issued"}),
)


RESOURCE_POLICIES: dict[ResourceType, ResourcePolicy] = {
    ResourceType.PATIENT: _PATIENT_POLICY,
    ResourceType.OBSERVATION: _OBSERVATION_POLICY,
    ResourceType.CONDITION: _CONDITION_POLICY,
    ResourceType.MEDICATION_REQUEST: _MEDICATION_REQUEST_POLICY,
    ResourceType.ENCOUNTER: _ENCOUNTER_POLICY,
    ResourceType.DIAGNOSTIC_REPORT: _DIAGNOSTIC_REPORT_POLICY,
}
"""Resources currently reachable.

Procedure, AllergyIntolerance, CarePlan and Immunization are intentionally absent:
:class:`~fhir_healthcare_ai.domain.enums.ResourceType` knows about them so interfaces
can be typed, but they stay unreachable until a policy and a parser are added together.
"""


def get_policy(resource_type: ResourceType) -> ResourcePolicy | None:
    """Policy for a resource type, or None if the resource is not reachable."""
    return RESOURCE_POLICIES.get(resource_type)


def is_allowed(resource_type: ResourceType) -> bool:
    return resource_type in RESOURCE_POLICIES


def allowed_resource_types() -> tuple[ResourceType, ...]:
    return tuple(RESOURCE_POLICIES)


def allowed_resource_names() -> tuple[str, ...]:
    return tuple(rt.value for rt in RESOURCE_POLICIES)


def allowed_params(resource_type: ResourceType) -> tuple[str, ...]:
    policy = get_policy(resource_type)
    return tuple(sorted(policy.params)) if policy else ()


def describe_allowlist() -> dict[str, dict[str, object]]:
    """Machine-readable summary, used in LLM prompts and exposed at ``/capabilities``."""
    summary: dict[str, dict[str, object]] = {}
    for resource_type, policy in RESOURCE_POLICIES.items():
        summary[resource_type.value] = {
            "search_params": {
                name: {
                    "type": param.type.value,
                    "description": param.description,
                    "comparators": sorted(c.value for c in param.comparators),
                    "modifiers": sorted(param.modifiers),
                    "systems": sorted(param.systems),
                }
                for name, param in sorted(policy.params.items())
            },
            "_include": sorted(policy.includes),
            "_revinclude": sorted(policy.revincludes),
            "_sort": sorted(policy.sorts),
            "max_count": policy.max_count,
        }
    return summary
