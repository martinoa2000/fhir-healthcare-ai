"""Parser interface and registry.

Adding a new resource type means writing one :class:`ResourceParser` subclass and
registering it. Nothing else in the pipeline changes -- which is what makes the
"interfaces for Procedure / AllergyIntolerance / CarePlan / Immunization" promise real
rather than aspirational.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, TypeVar

from fhir_healthcare_ai.domain.clinical import NormalizedResource
from fhir_healthcare_ai.domain.enums import ResourceType
from fhir_healthcare_ai.fhir.parsers.primitives import get_str, parse_reference_id
from fhir_healthcare_ai.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=NormalizedResource)


class ParseError(ValueError):
    """Raised only for input that is not a FHIR resource at all."""


class ResourceParser(ABC, Generic[T]):
    """Converts one raw FHIR resource into its normalized form."""

    resource_type: ClassVar[ResourceType]

    def can_parse(self, resource: dict[str, Any]) -> bool:
        return resource.get("resourceType") == self.resource_type.value

    def parse(self, resource: dict[str, Any]) -> T:
        """Parse a resource. Raises :class:`ParseError` only on structural failure."""
        if not isinstance(resource, dict):
            raise ParseError(f"expected a JSON object, got {type(resource).__name__}")
        actual = resource.get("resourceType")
        if actual != self.resource_type.value:
            raise ParseError(f"{type(self).__name__} cannot parse resourceType={actual!r}")
        resource_id = get_str(resource, "id")
        if not resource_id:
            raise ParseError(f"{actual} is missing a logical id")
        return self._parse(resource, resource_id)

    def try_parse(self, resource: dict[str, Any]) -> T | None:
        """Parse, returning None instead of raising. Used for bulk ingest."""
        try:
            return self.parse(resource)
        except (ParseError, ValueError, TypeError):
            logger.warning(
                "skipping unparseable resource",
                extra={
                    "resource_type": resource.get("resourceType"),
                    "resource_id": resource.get("id"),
                },
            )
            return None

    @abstractmethod
    def _parse(self, resource: dict[str, Any], resource_id: str) -> T:
        """Subclass hook. ``resource_id`` is guaranteed non-empty."""

    @staticmethod
    def subject_id(resource: dict[str, Any]) -> str | None:
        """Patient id from ``subject`` or ``patient``, whichever the server used."""
        return parse_reference_id(resource.get("subject")) or parse_reference_id(
            resource.get("patient")
        )

    @staticmethod
    def encounter_id(resource: dict[str, Any]) -> str | None:
        return parse_reference_id(resource.get("encounter")) or parse_reference_id(
            resource.get("context")
        )


class ParserRegistry:
    """Dispatches raw resources to the parser that handles them."""

    def __init__(self, parsers: list[ResourceParser[Any]] | None = None) -> None:
        self._parsers: dict[ResourceType, ResourceParser[Any]] = {}
        for parser in parsers or []:
            self.register(parser)

    def register(self, parser: ResourceParser[Any]) -> None:
        self._parsers[parser.resource_type] = parser

    def get(self, resource_type: ResourceType) -> ResourceParser[Any] | None:
        return self._parsers.get(resource_type)

    def supported(self) -> tuple[ResourceType, ...]:
        return tuple(self._parsers)

    def parse(self, resource: dict[str, Any]) -> NormalizedResource | None:
        """Parse one resource, or return None when the type is unsupported."""
        if not isinstance(resource, dict):
            return None
        raw_type = resource.get("resourceType")
        if not isinstance(raw_type, str):
            return None
        try:
            resource_type = ResourceType(raw_type)
        except ValueError:
            return None
        parser = self._parsers.get(resource_type)
        if parser is None:
            return None
        return parser.try_parse(resource)

    def parse_many(self, resources: list[dict[str, Any]]) -> list[NormalizedResource]:
        """Parse a batch, dropping anything unsupported or malformed."""
        parsed: list[NormalizedResource] = []
        for resource in resources:
            result = self.parse(resource)
            if result is not None:
                parsed.append(result)
        return parsed
