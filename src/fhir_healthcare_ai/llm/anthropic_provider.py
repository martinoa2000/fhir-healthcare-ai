"""Anthropic backend.

The Messages API takes the system prompt as a top-level parameter rather than as a
message with ``role: "system"``, so the conversation is split before it is sent. That
split is the only provider-specific behaviour in this file -- everything else the
planner does is identical across backends, which is the point of the abstraction.
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
    from anthropic import AsyncAnthropic

logger = get_logger(__name__)

DEFAULT_MODEL = "claude-opus-4-7"

#: Anthropic has no JSON response mode, so structured output is coaxed by prefilling the
#: assistant turn with an opening brace. The model then has no syntactically valid way to
#: begin with prose, which removes the most common cause of unparseable plans.
JSON_PREFILL = "{"


class AnthropicProvider(LLMProvider):
    """Messages API backend."""

    name = "anthropic"

    def __init__(self, settings: LLMSettings | None = None, *, client: Any = None) -> None:
        self.settings = settings or LLMSettings()
        self.model = self.settings.model or DEFAULT_MODEL
        self._client: AsyncAnthropic | None = client
        if client is None:
            self._client = self._build_client()

    def _build_client(self) -> AsyncAnthropic:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise LLMUnavailableError(
                "The anthropic package is not installed. Install it with "
                "`pip install 'fhir-healthcare-ai[anthropic]'` or set LLM_PROVIDER=mock."
            ) from exc
        if not self.settings.api_key:
            raise LLMUnavailableError(
                "LLM_API_KEY is not set; the anthropic provider cannot authenticate."
            )
        return AsyncAnthropic(
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
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        turns: list[dict[str, str]] = [m.as_dict() for m in messages if m.role != "system"]
        if not turns:
            raise LLMError("anthropic requires at least one non-system message")
        if json_mode:
            turns.append({"role": "assistant", "content": JSON_PREFILL})

        try:
            message = await self._client.messages.create(
                model=self.model,
                system=system or None,
                messages=turns,
                temperature=(self.settings.temperature if temperature is None else temperature),
                max_tokens=self.settings.max_tokens if max_tokens is None else max_tokens,
            )
        except Exception as exc:
            raise LLMError(f"anthropic completion failed: {exc}") from exc

        text = "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        )
        if json_mode:
            text = JSON_PREFILL + text

        return LLMResponse(
            text=text,
            model=message.model,
            prompt_tokens=message.usage.input_tokens,
            completion_tokens=message.usage.output_tokens,
            finish_reason=message.stop_reason,
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
