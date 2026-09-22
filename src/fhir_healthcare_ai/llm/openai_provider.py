"""OpenAI backend, and the OpenAI-compatible transport the local backend reuses.

The SDK is imported lazily inside the constructor. Importing it at module scope would
make ``pip install fhir-healthcare-ai`` drag in a dependency that the default
configuration never touches -- the stack ships with ``LLM_PROVIDER=mock`` precisely so
that it runs with no vendor account at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fhir_healthcare_ai.config import LLMSettings
from fhir_healthcare_ai.llm.base import (
    LLMError,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    LLMUnavailableError,
)
from fhir_healthcare_ai.logging_config import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from openai import AsyncOpenAI

logger = get_logger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIProvider(LLMProvider):
    """Chat Completions backend."""

    name = "openai"

    def __init__(self, settings: LLMSettings | None = None, *, client: Any = None) -> None:
        self.settings = settings or LLMSettings()
        self.model = self.settings.model or DEFAULT_MODEL
        self._client: AsyncOpenAI | None = client
        if client is None:
            self._client = self._build_client()

    def _build_client(self) -> AsyncOpenAI:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise LLMUnavailableError(
                "The openai package is not installed. Install it with "
                "`pip install 'fhir-healthcare-ai[openai]'` or set LLM_PROVIDER=mock."
            ) from exc
        if not self.settings.api_key:
            raise LLMUnavailableError(
                "LLM_API_KEY is not set; the openai provider cannot authenticate."
            )
        return AsyncOpenAI(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
            timeout=self.settings.timeout_seconds,
            max_retries=self.settings.max_retries,
        )

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        assert self._client is not None
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [message.as_dict() for message in messages],
            "temperature": (self.settings.temperature if temperature is None else temperature),
            "max_tokens": self.settings.max_tokens if max_tokens is None else max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            completion = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise LLMError(f"openai completion failed: {exc}") from exc

        choice = completion.choices[0]
        usage = completion.usage
        return LLMResponse(
            text=choice.message.content or "",
            model=completion.model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            finish_reason=choice.finish_reason,
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
