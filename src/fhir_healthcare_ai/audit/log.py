"""Append-only audit trail.

Access logging is a regulatory requirement in most healthcare jurisdictions (HIPAA
§164.312(b) in the US, Article 30 GDPR in the EU), and it is also what makes an AI
system reviewable at all: without it, nobody can answer "why did the model see this
patient's data?" after the fact.

Every FHIR request, every refused query and every model call produces one event. Sinks
are pluggable so a deployment can ship events to stdout, a file, or a real audit store
without touching the pipeline.
"""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from fhir_healthcare_ai.logging_config import actor_var, correlation_id_var, get_logger

logger = get_logger("audit")

AuditAction = Literal[
    "plan.generated",
    "plan.rejected",
    "query.executed",
    "query.refused",
    "query.failed",
    "resources.fetched",
    "analysis.executed",
    "response.generated",
    "llm.call",
]

Outcome = Literal["success", "failure", "refused"]


class AuditEvent(BaseModel):
    """One auditable action."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    #: Defaults to the ambient request id. Requiring callers to pass it explicitly meant
    #: that any forgotten argument produced an event that could not be joined back to the
    #: request that caused it, which defeats the point of the trail.
    correlation_id: str = Field(default_factory=lambda: correlation_id_var.get() or "-")
    action: AuditAction
    outcome: Outcome = "success"
    #: The authenticated API key name bound by the security layer, for the same reason as
    #: ``correlation_id``. ``"system"`` outside a request (startup, CLI tools).
    actor: str = Field(default_factory=lambda: actor_var.get() or "system")
    resource_type: str | None = None
    query: str | None = None
    patient_ids: list[str] = Field(default_factory=list)
    resource_count: int = 0
    duration_ms: float | None = None
    reason: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    def to_json(self) -> str:
        return self.model_dump_json(exclude_none=True)


class AuditSink(ABC):
    """Destination for audit events."""

    @abstractmethod
    def record(self, event: AuditEvent) -> None:
        """Persist one event. Implementations must not raise."""


class LoggingAuditSink(AuditSink):
    """Emit events through the application logger (JSON lines on stdout)."""

    def record(self, event: AuditEvent) -> None:
        logger.info(
            "audit",
            extra={
                "audit_action": event.action,
                "audit_outcome": event.outcome,
                "correlation_id": event.correlation_id,
                "actor": event.actor,
                "resource_type": event.resource_type,
                "query": event.query,
                "resource_count": event.resource_count,
                "patient_count": len(event.patient_ids),
                "duration_ms": event.duration_ms,
                "reason": event.reason,
            },
        )


class FileAuditSink(AuditSink):
    """Append events to a JSONL file. Thread-safe."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, event: AuditEvent) -> None:
        line = event.to_json()
        try:
            with self._lock, self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            logger.exception("failed to write audit event", extra={"path": str(self.path)})


class InMemoryAuditSink(AuditSink):
    """Collect events in memory. Used by tests and by the ``/audit`` debug endpoint."""

    def __init__(self, max_events: int = 1000) -> None:
        self.max_events = max_events
        self.events: list[AuditEvent] = []
        self._lock = threading.Lock()

    def record(self, event: AuditEvent) -> None:
        with self._lock:
            self.events.append(event)
            if len(self.events) > self.max_events:
                del self.events[: len(self.events) - self.max_events]

    def by_action(self, action: AuditAction) -> list[AuditEvent]:
        return [e for e in self.events if e.action == action]

    def clear(self) -> None:
        with self._lock:
            self.events.clear()


class CompositeAuditSink(AuditSink):
    """Fan one event out to several sinks; a failing sink never blocks the others."""

    def __init__(self, *sinks: AuditSink) -> None:
        self.sinks = sinks

    def record(self, event: AuditEvent) -> None:
        for sink in self.sinks:
            try:
                sink.record(event)
            except Exception:
                logger.exception("audit sink failed", extra={"sink": type(sink).__name__})


def build_audit_sink(file_path: str | None = None, keep_in_memory: bool = True) -> AuditSink:
    """Assemble the default sink chain for the application."""
    sinks: list[AuditSink] = [LoggingAuditSink()]
    if file_path:
        sinks.append(FileAuditSink(file_path))
    if keep_in_memory:
        sinks.append(InMemoryAuditSink())
    return CompositeAuditSink(*sinks)


def read_audit_file(path: str | Path) -> list[AuditEvent]:
    """Load a JSONL audit file, skipping malformed lines."""
    events: list[AuditEvent] = []
    file_path = Path(path)
    if not file_path.exists():
        return events
    for line in file_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(AuditEvent.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValueError):
            logger.warning("skipping malformed audit line", extra={"path": str(file_path)})
    return events
