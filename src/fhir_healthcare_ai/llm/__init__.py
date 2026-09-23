"""LLM abstraction.

The model is a component, not the architecture. Everything above this package talks to
:class:`LLMProvider`, so the backend is an environment variable: a local vLLM server, a
HuggingFace model in-process, the deterministic rule-based planner, or a hosted API.

Default is local inference. See :mod:`fhir_healthcare_ai.llm.local`.
"""

from fhir_healthcare_ai.llm.base import (
    LLMError,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    LLMResponseError,
    LLMUnavailableError,
    extract_json_object,
)
from fhir_healthcare_ai.llm.factory import (
    LOCAL_PROVIDERS,
    PROVIDERS,
    ProviderStatus,
    build_provider,
    is_local,
    resolve_provider,
)
from fhir_healthcare_ai.llm.local import HuggingFaceProvider, VLLMProvider
from fhir_healthcare_ai.llm.mock import MockLLMProvider, plan_for
from fhir_healthcare_ai.llm.prompts import build_narrative_prompt, build_planning_prompt

__all__ = [
    "LOCAL_PROVIDERS",
    "PROVIDERS",
    "HuggingFaceProvider",
    "LLMError",
    "LLMMessage",
    "LLMProvider",
    "LLMResponse",
    "LLMResponseError",
    "LLMUnavailableError",
    "MockLLMProvider",
    "ProviderStatus",
    "VLLMProvider",
    "build_narrative_prompt",
    "build_planning_prompt",
    "build_provider",
    "extract_json_object",
    "is_local",
    "plan_for",
    "resolve_provider",
]
