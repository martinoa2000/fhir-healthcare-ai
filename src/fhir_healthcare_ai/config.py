"""Application configuration, sourced from environment variables or a .env file."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Inference backends. Both keep the model inside the deployment boundary: ``vllm`` is a
#: locally hosted model server, ``mock`` the deterministic planner. No hosted API exists.
LLMProviderName = Literal["vllm", "mock"]


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

    #: Serve a generated synthetic population from an in-process server instead of
    #: calling ``base_url``. For demos and CI without Docker; never for real data.
    in_memory: bool = False
    in_memory_patients: int = Field(default=120, ge=1, le=5000)
    in_memory_seed: int = 42

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
    model: str = "mlx-community/Qwen3.5-9B-MLX-4bit"
    #: Bearer token, only for a vLLM server started with ``--api-key``.
    api_key: str | None = None
    #: OpenAI-compatible endpoint of the local vLLM server.
    base_url: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1)
    timeout_seconds: float = 60.0
    max_retries: int = Field(default=2, ge=0, le=5)
    #: Qwen3-family reasoning ("thinking") for the ``vllm`` provider. Off by default: the
    #: planner wants a JSON plan, not a chain of thought, and thinking multiplies latency
    #: and tokens without making the validated plan any safer.
    enable_thinking: bool = False

    #: When the configured local backend is unreachable, fall back to the deterministic
    #: planner instead of failing the request. On by default so that a stack started
    #: without the vLLM profile still answers the documented example questions; turn it
    #: off in any setting where a silent capability drop would be misleading.
    fallback_to_mock: bool = True


#: Key names end up in logs and in every audit event, so they are restricted to
#: identifiers that cannot smuggle structure (whitespace, quotes) into either.
_KEY_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@-]{0,63}$")


class SecuritySettings(BaseSettings):
    """Who may call the API, and how often.

    Keys are *named* so that the audit trail can say who asked a question without ever
    storing the secret that proved it. ``API_KEYS`` is a JSON object of name -> key::

        API_KEYS='{"alice": "<secret>", "ci": "<secret>"}'
    """

    #: ``env_ignore_empty`` so the blank ``API_REQUIRE_AUTH=`` in .env.example means
    #: "unset" rather than failing to parse as a bool.
    model_config = SettingsConfigDict(
        env_prefix="API_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
        env_ignore_empty=True,
    )

    #: The alias is unprefixed so the variable is ``API_KEYS`` rather than
    #: ``API_API_KEYS``, while code still reads the unambiguous ``api_keys``.
    api_keys: dict[str, SecretStr] = Field(default_factory=dict, validation_alias="api_keys")
    #: ``None`` means "decide from context": on when keys are configured, and always on in
    #: prod. :class:`Settings` resolves it to a bool.
    require_auth: bool | None = None
    #: Requests per minute per caller on the expensive routes. 0 disables the limiter.
    rate_limit_per_minute: int = Field(default=60, ge=0)

    @field_validator("api_keys")
    @classmethod
    def _check_keys(cls, v: dict[str, SecretStr]) -> dict[str, SecretStr]:
        secrets: set[str] = set()
        for name, key in v.items():
            if not _KEY_NAME.match(name):
                raise ValueError(f"API key name {name!r} must match {_KEY_NAME.pattern}")
            secret = key.get_secret_value()
            if not secret.strip():
                raise ValueError(f"API key {name!r} is empty")
            # Two names sharing one secret would make the audited actor ambiguous.
            if secret in secrets:
                raise ValueError(f"API key {name!r} reuses another key's secret")
            secrets.add(secret)
        return v

    @property
    def auth_enabled(self) -> bool:
        """Whether requests must present a key."""
        return bool(self.require_auth)


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
    security: SecuritySettings = Field(default_factory=SecuritySettings)

    @model_validator(mode="after")
    def _resolve_security(self) -> Self:
        """Resolve ``require_auth`` and refuse configurations that expose or lock out.

        This runs when settings are built, which for the server is at import of the ASGI
        app, so a production deployment without keys fails before it binds a port rather
        than serving patient data to anyone who can reach it.
        """
        sec = self.security
        require = sec.require_auth
        if self.environment == "prod":
            if require is False:
                raise ValueError("API_REQUIRE_AUTH=false is not allowed when ENVIRONMENT=prod")
            require = True
        elif require is None:
            require = bool(sec.api_keys)
        if require and not sec.api_keys:
            raise ValueError(
                f"authentication is required (ENVIRONMENT={self.environment}) but no API "
                """keys are configured; set API_KEYS='{"<name>": "<secret>"}'"""
            )
        if require != sec.require_auth:
            self.security = sec.model_copy(update={"require_auth": require})
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
