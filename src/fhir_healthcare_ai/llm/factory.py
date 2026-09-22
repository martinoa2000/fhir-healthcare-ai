"""Provider selection.

One function, one dict. The rest of the codebase asks for "the configured provider" and
never learns which one it got, which is the only reason swapping vLLM for a hosted API
is a one-line environment change rather than a refactor.

The registry is keyed by the literal values of
:data:`~fhir_healthcare_ai.config.LLMProviderName`, so adding a backend means adding it
to both and nothing else.
"""

from __future__ import annotations

from collections.abc import Callable

from fhir_healthcare_ai.config import LLMProviderName, LLMSettings, get_settings
from fhir_healthcare_ai.llm.base import LLMProvider, LLMUnavailableError
from fhir_healthcare_ai.llm.local import HuggingFaceProvider, VLLMProvider
from fhir_healthcare_ai.llm.mock import MockLLMProvider
from fhir_healthcare_ai.logging_config import get_logger

logger = get_logger(__name__)

ProviderFactory = Callable[[LLMSettings], LLMProvider]


def _openai(settings: LLMSettings) -> LLMProvider:
    from fhir_healthcare_ai.llm.openai_provider import OpenAIProvider

    return OpenAIProvider(settings)


def _anthropic(settings: LLMSettings) -> LLMProvider:
    from fhir_healthcare_ai.llm.anthropic_provider import AnthropicProvider

    return AnthropicProvider(settings)


PROVIDERS: dict[str, ProviderFactory] = {
    "vllm": lambda settings: VLLMProvider(settings),
    "huggingface": lambda settings: HuggingFaceProvider(settings),
    "mock": lambda _: MockLLMProvider(),
    "openai": _openai,
    "anthropic": _anthropic,
}

#: Backends that keep inference inside the deployment boundary. Surfaced by the API so
#: an operator can see at a glance whether question text is leaving the host.
LOCAL_PROVIDERS: frozenset[str] = frozenset({"vllm", "huggingface", "mock"})


def build_provider(
    settings: LLMSettings | None = None, *, name: LLMProviderName | None = None
) -> LLMProvider:
    """Construct the configured provider.

    Args:
        settings: Overrides the global LLM settings. Mostly used by tests.
        name: Overrides the configured provider name, for the benchmark's A/B runs.

    Raises:
        LLMUnavailableError: The name is not registered, or the backend is installed but
            unusable (missing package, missing key, unreachable server).
    """
    resolved = settings or get_settings().llm
    provider_name = name or resolved.provider
    factory = PROVIDERS.get(provider_name)
    if factory is None:
        raise LLMUnavailableError(
            f"unknown LLM provider {provider_name!r}; available: {', '.join(sorted(PROVIDERS))}"
        )

    try:
        provider = factory(resolved)
    except LLMUnavailableError:
        # A missing package or key is a configuration problem, not a request failure, so
        # it is worth degrading rather than returning 500 for every question. The
        # substitution is logged loudly and reported by /health, never hidden.
        if not (resolved.fallback_to_mock and name is None and provider_name != "mock"):
            raise
        logger.warning(
            "configured llm provider unavailable, falling back to the deterministic planner",
            extra={"provider": provider_name},
        )
        return MockLLMProvider()
    logger.info(
        "llm provider ready",
        extra={
            "provider": provider.name,
            "model": getattr(provider, "model", resolved.model),
            "local": provider_name in LOCAL_PROVIDERS,
        },
    )
    return provider


def is_local(name: str) -> bool:
    """Whether the named provider runs inference inside the deployment boundary."""
    return name in LOCAL_PROVIDERS
