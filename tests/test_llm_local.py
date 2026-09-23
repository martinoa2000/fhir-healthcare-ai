"""The vLLM provider's request shape, over a mocked OpenAI-compatible server."""

from __future__ import annotations

import json

import httpx

from fhir_healthcare_ai.config import LLMSettings
from fhir_healthcare_ai.llm.base import LLMMessage
from fhir_healthcare_ai.llm.local import DEFAULT_VLLM_MODEL, VLLMProvider


def _provider(settings: LLMSettings, sent: list[dict[str, object]]) -> VLLMProvider:
    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": settings.model,
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            },
        )

    client = httpx.AsyncClient(base_url="http://vllm/v1", transport=httpx.MockTransport(handler))
    return VLLMProvider(settings, client=client)


async def test_defaults_to_a_local_qwen_with_thinking_off() -> None:
    sent: list[dict[str, object]] = []
    settings = LLMSettings(_env_file=None)
    provider = _provider(settings, sent)

    response = await provider.complete([LLMMessage("user", "hi")], json_mode=True)

    assert settings.model == DEFAULT_VLLM_MODEL == "Qwen/Qwen3.8-27B-FP8"
    assert response.text == "{}"
    assert sent[0]["model"] == DEFAULT_VLLM_MODEL
    assert sent[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert sent[0]["response_format"] == {"type": "json_object"}


async def test_thinking_can_be_turned_on() -> None:
    sent: list[dict[str, object]] = []
    provider = _provider(LLMSettings(_env_file=None, enable_thinking=True), sent)

    await provider.complete([LLMMessage("user", "hi")])

    assert sent[0]["chat_template_kwargs"] == {"enable_thinking": True}
    assert "response_format" not in sent[0]
