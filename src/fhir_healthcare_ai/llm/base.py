"""The provider interface every LLM backend implements.

The contract is deliberately narrow: *messages in, text out, plus a token count*. No
tool calling, no function schemas, no provider-specific response objects leak through.

That narrowness is a security property, not an aesthetic one. The planner asks the
model for a JSON object describing a query plan and nothing else; the model has no
channel through which it could request an HTTP call, name a URL, or reach anything
outside the process. Whatever comes back is parsed into a
:class:`~fhir_healthcare_ai.domain.query.QueryPlan` and then has to survive the
allowlist validator before a single byte reaches the FHIR server.

Adding a backend means implementing :meth:`LLMProvider.complete` and registering it in
:mod:`fhir_healthcare_ai.llm.factory`. Nothing else in the codebase changes.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from fhir_healthcare_ai.logging_config import get_logger

logger = get_logger(__name__)

Role = Literal["system", "user", "assistant"]


class LLMError(RuntimeError):
    """Any failure to obtain a usable completion."""


class LLMUnavailableError(LLMError):
    """The backend is not installed, not configured, or not reachable."""


class LLMResponseError(LLMError):
    """A completion came back but could not be used (e.g. unparseable JSON)."""


@dataclass(frozen=True)
class LLMMessage:
    """One turn of a conversation."""

    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    """What a provider returns.

    Args:
        text: The raw completion text.
        model: The model that actually answered, as reported by the backend.
        prompt_tokens / completion_tokens: Best-effort usage, 0 when unreported.
        finish_reason: Backend-specific stop reason, kept for the execution trace.
        raw: The untouched provider payload, for debugging. Never serialized into an
            API response -- it can contain provider metadata we do not want to expose.
    """

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def json_payload(self) -> dict[str, Any]:
        """Parse the completion as a JSON object.

        Models wrap JSON in prose and fenced code blocks no matter how firmly the
        prompt forbids it, so the fence is stripped and the outermost balanced object
        is extracted before parsing. A model that returns a bare array or a scalar is
        a protocol violation and raises -- the planner needs an object.
        """
        return extract_json_object(self.text)


class LLMProvider(ABC):
    """Base class for every backend."""

    name: str = "base"

    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Produce a completion.

        Args:
            messages: The conversation. Exactly one leading system message is expected
                but not required; providers that have no system role fold it into the
                first user turn themselves.
            temperature: Overrides the configured default. The planner always passes 0.
            max_tokens: Overrides the configured default.
            json_mode: Request structured JSON output where the backend supports it.
                Providers that do not support it must still honour the prompt, so
                callers may never assume the response is valid JSON.
        """

    async def aclose(self) -> None:
        """Release any network resources. Safe to call more than once."""
        return None

    async def __aenter__(self) -> LLMProvider:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first well-formed JSON object out of a completion.

    Tries, in order: the whole string, the contents of a fenced block, then the
    outermost brace-balanced span. Brace balancing is string-aware so that a ``}``
    inside a quoted value does not truncate the span.
    """
    candidates: list[str] = [text.strip()]

    fenced = _FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())

    span = _balanced_object(text)
    if span:
        candidates.append(span)

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    preview = text[:200].replace("\n", " ")
    raise LLMResponseError(f"no JSON object found in completion: {preview!r}")


def _balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None
