"""Application configuration, sourced from environment variables or a .env file."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Inference backends. The first three keep the model inside the deployment boundary;
#: the hosted ones exist to show the abstraction holds, not because the demo needs them.
LLMProviderName = Literal["vllm", "huggingface", "mock", "openai", "anthropic"]


class FHIRSettings(BaseSettings):
    """Connection and safety limits for the upstream FHIR server."""

    model_config = SettingsConfigDict(env_prefix="FHIR_", env_file=".env", extra="ignore")

    base_url: str = "http://localhost:8080/fhir"
    timeout_seconds: float = 30.0
    verify_ssl: bool = True
    auth_token: str | None = None
    max_retries: int = Field(default=2, ge=0, le=5)

    # Safety envelope. The query validator refuses anything outside these bounds.
    max_page_size: int = Field(default=200, ge=1, le=1000)
    max_pages: int = Field(default=10, ge=1, le=100)
    max_total_resources: int = Field(default=2000, ge=1)
    allow_write: bool = False

    @field_validator("base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")


class LLMSettings(BaseSettings):
    """Provider-agnostic LLM configuration.

    The default is ``vllm`` against a locally hosted server: clinical text should not
    leave the deployment boundary to have a search query written for it. ``mock`` is the
    deterministic rule-based planner used by CI and by anyone who wants the stack up
    without weights on disk.
    """

    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=".env", extra="ignore")

    provider: LLMProviderName = "vllm"
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    api_key: str | None = None
    #: OpenAI-compatible endpoint for the ``vllm`` provider. Ignored by ``huggingface``,
    #: which loads the model in-process.
    base_url: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1)
    timeout_seconds: float = 60.0
    max_retries: int = Field(default=2, ge=0, le=5)

    #: When the configured local backend is unreachable, fall back to the deterministic
    #: planner instead of failing the request. On by default so that a stack started
    #: without the vLLM profile still answers the documented example questions; turn it
    #: off in any setting where a silent capability drop would be misleading.
    fallback_to_mock: bool = True


class Settings(BaseSettings):
    """Root settings object."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "fhir-healthcare-ai"
    environment: Literal["local", "dev", "prod"] = "local"
    log_level: str = "INFO"
    log_json: bool = True

    # Hard cap on how many patients a single natural-language request may surface.
    max_patients_per_response: int = Field(default=100, ge=1)
    audit_log_path: str | None = None

    fhir: FHIRSettings = Field(default_factory=FHIRSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
