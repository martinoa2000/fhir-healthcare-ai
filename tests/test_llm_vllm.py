"""vLLM provider: the request it sends, the response it parses, and how it fails."""

from __future__ import annotations

import json
from typing import Any, get_args

import httpx
import pytest

from fhir_healthcare_ai.config import LLMProviderName, LLMSettings
from fhir_healthcare_ai.llm.base import LLMError, LLMMessage, LLMUnavailableError
from fhir_healthcare_ai.llm.factory import PROVIDERS, build_provider, is_local, resolve_provider
from fhir_healthcare_ai.llm.local import VLLMProvider
from fhir_healthcare_ai.llm.mock import MockLLMProvider

BASE_URL = "http://vllm.test/v1"
MESSAGES = [LLMMessage("system", "plan only"), LLMMessage("user", "which patients?")]


def completion(content: str | None = '{"steps": []}', **extra: Any) -> dict[str, Any]:
    return {
        "model": "served-model",
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5},
        **extra,
    }


def provider_for(
    handler: Any, settings: LLMSettings | None = None
) -> tuple[VLLMProvider, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        result: httpx.Response = handler(request)
        return result

    client = httpx.AsyncClient(base_url=BASE_URL, transport=httpx.MockTransport(record))
    return VLLMProvider(settings or LLMSettings(base_url=BASE_URL), client=client), seen


def test_vllm_is_the_only_model_backend() -> None:
    assert set(get_args(LLMProviderName)) == {"vllm", "mock"}
    assert set(PROVIDERS) == {"vllm", "mock"}
    assert all(is_local(name) for name in PROVIDERS)


def test_hosted_providers_are_refused() -> None:
    for name in ("openai", "anthropic", "huggingface"):
        with pytest.raises(LLMUnavailableError, match="unknown LLM provider"):
            build_provider(LLMSettings(), name=name)  # type: ignore[arg-type]


async def test_thinking_mode_follows_the_setting() -> None:
    settings = LLMSettings(base_url=BASE_URL, enable_thinking=True)
    provider, seen = provider_for(lambda _: httpx.Response(200, json=completion()), settings)

    await provider.complete(MESSAGES)

    assert json.loads(seen[0].content)["chat_template_kwargs"] == {"enable_thinking": True}


def test_defaults_point_at_the_local_server() -> None:
    provider = VLLMProvider(LLMSettings(base_url=None, model=""))
    assert provider.base_url == "http://localhost:8001/v1"
    assert provider.model == "Qwen/Qwen3.8-27B-FP8"


async def test_complete_sends_an_openai_compatible_request() -> None:
    settings = LLMSettings(base_url=BASE_URL, model="m", temperature=0.0, max_tokens=64)
    provider, seen = provider_for(lambda _: httpx.Response(200, json=completion()), settings)

    response = await provider.complete(MESSAGES, json_mode=True)

    (request,) = seen
    assert request.method == "POST" and request.url.path == "/v1/chat/completions"
    body = json.loads(request.content)
    assert body == {
        "model": "m",
        "messages": [
            {"role": "system", "content": "plan only"},
            {"role": "user", "content": "which patients?"},
        ],
        "temperature": 0.0,
        "max_tokens": 64,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
    }
    assert response.text == '{"steps": []}'
    assert response.model == "served-model"
    assert (response.prompt_tokens, response.completion_tokens) == (12, 5)
    assert response.finish_reason == "stop"


async def test_call_arguments_override_settings_and_json_mode_is_opt_in() -> None:
    provider, seen = provider_for(lambda _: httpx.Response(200, json=completion()))

    await provider.complete(MESSAGES, temperature=0.7, max_tokens=10)

    body = json.loads(seen[0].content)
    assert body["temperature"] == 0.7 and body["max_tokens"] == 10
    assert "response_format" not in body


async def test_missing_content_and_usage_are_tolerated() -> None:
    data = completion(content=None)
    del data["usage"], data["model"]
    provider, _ = provider_for(lambda _: httpx.Response(200, json=data))

    response = await provider.complete(MESSAGES)

    assert response.text == ""
    assert response.model == provider.model
    assert (response.prompt_tokens, response.completion_tokens) == (0, 0)


async def test_unexpected_response_shape_is_an_error() -> None:
    provider, _ = provider_for(lambda _: httpx.Response(200, json={"choices": []}))
    with pytest.raises(LLMError, match="unexpected response shape"):
        await provider.complete(MESSAGES)


async def test_http_error_status_is_an_error() -> None:
    provider, _ = provider_for(lambda _: httpx.Response(503, text="overloaded"))
    with pytest.raises(LLMError, match="503: overloaded"):
        await provider.complete(MESSAGES)


async def test_connection_failure_means_unavailable() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    provider, _ = provider_for(refuse)
    with pytest.raises(LLMUnavailableError, match="no local model server"):
        await provider.complete(MESSAGES)


async def test_ping_reports_whether_models_are_served() -> None:
    up, seen = provider_for(lambda _: httpx.Response(200, json={"data": []}))
    assert await up.ping()
    assert seen[0].url.path == "/v1/models"

    down, _ = provider_for(lambda _: httpx.Response(500))
    assert not await down.ping()

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    unreachable, _ = provider_for(refuse)
    assert not await unreachable.ping()


async def test_bearer_token_is_placeholder_unless_configured() -> None:
    anonymous = VLLMProvider(LLMSettings(base_url=BASE_URL))
    keyed = VLLMProvider(LLMSettings(base_url=BASE_URL, api_key="s3cret"))
    try:
        assert anonymous._client.headers["Authorization"] == "Bearer local"
        assert keyed._client.headers["Authorization"] == "Bearer s3cret"
    finally:
        await anonymous.aclose()
        await keyed.aclose()


async def test_reachable_server_is_used_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    async def ok(self: VLLMProvider) -> bool:
        return True

    monkeypatch.setattr(VLLMProvider, "ping", ok)
    provider, status = await resolve_provider(LLMSettings(base_url=BASE_URL, model="m"))
    try:
        assert isinstance(provider, VLLMProvider) and not isinstance(provider, MockLLMProvider)
        assert status.active == "vllm" and status.model == "m"
        assert status.local and not status.fallback and status.reason is None
    finally:
        await provider.aclose()
