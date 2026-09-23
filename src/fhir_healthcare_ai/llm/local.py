"""Local model backend: a vLLM server over HTTP.

This is the project's only model backend. Clinical data -- even synthetic data standing
in for clinical data -- should not leave the deployment boundary just to have a search
query written for it, and a platform that only works when it can reach a vendor API is
not a platform a hospital can run.

:class:`VLLMProvider` talks to a vLLM (or any OpenAI-compatible) server over HTTP. The
model is a separate, independently scaled service with continuous batching, and the
application container stays small and free of CUDA. ``docker compose --profile vllm
up`` starts one alongside the stack.

Nothing is sent off the host unless the operator points ``base_url`` somewhere else.
Generation is greedy by default (``temperature=0``) because the planner needs the same
question to yield the same plan -- a benchmark over a sampling decoder measures the
sampler as much as the model.
"""

from __future__ import annotations

from typing import Any

import httpx

from fhir_healthcare_ai.config import LLMSettings
from fhir_healthcare_ai.llm.base import (
    LLMError,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    LLMUnavailableError,
)
from fhir_healthcare_ai.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_VLLM_BASE_URL = "http://localhost:8001/v1"
DEFAULT_VLLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

#: vLLM's OpenAI server accepts any bearer token when started without `--api-key`.
PLACEHOLDER_KEY = "local"


class VLLMProvider(LLMProvider):
    """OpenAI-compatible chat completions against a locally hosted server.

    Deliberately implemented with plain ``httpx`` rather than the OpenAI SDK: the whole
    point of this backend is that it has no vendor dependency, and the OpenAI-compatible
    surface it uses is four JSON fields wide.
    """

    name = "vllm"

    def __init__(
        self,
        settings: LLMSettings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings or LLMSettings()
        self.base_url = (self.settings.base_url or DEFAULT_VLLM_BASE_URL).rstrip("/")
        self.model = self.settings.model or DEFAULT_VLLM_MODEL
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.settings.timeout_seconds,
            headers={
                "Authorization": f"Bearer {self.settings.api_key or PLACEHOLDER_KEY}",
                "Content-Type": "application/json",
            },
        )

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [message.as_dict() for message in messages],
            "temperature": self.settings.temperature if temperature is None else temperature,
            "max_tokens": self.settings.max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }
        if json_mode:
            # vLLM implements guided decoding through this OpenAI-compatible field, so a
            # server that supports it returns syntactically valid JSON by construction.
            # Servers that do not simply ignore the key.
            payload["response_format"] = {"type": "json_object"}

        data = await self._post("/chat/completions", payload)
        try:
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response shape from {self.base_url}: {data!r}") from exc

        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            model=data.get("model", self.model),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            finish_reason=choice.get("finish_reason"),
        )

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._client.post(path, json=payload)
        except httpx.RequestError as exc:
            raise LLMUnavailableError(
                f"no local model server at {self.base_url}: {exc}. Start one with "
                "`docker compose --profile vllm up vllm`, or set LLM_PROVIDER=mock."
            ) from exc
        if response.status_code >= 400:
            raise LLMError(
                f"local model server returned {response.status_code}: {response.text[:300]}"
            )
        result: dict[str, Any] = response.json()
        return result

    async def ping(self) -> bool:
        """True when the server is up and serving a model. Used by ``/health``."""
        try:
            response = await self._client.get("/models")
        except httpx.RequestError:
            return False
        return response.status_code == 200

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
