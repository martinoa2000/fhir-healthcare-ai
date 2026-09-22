"""Structured logging setup.

Clinical systems need logs that can be shipped to a SIEM without reparsing, so the
default formatter emits one JSON object per line. Every log record carries a
``correlation_id`` when one is bound to the current context.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from typing import Any, TextIO

correlation_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class JSONFormatter(logging.Formatter):
    """Render log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        cid = correlation_id_var.get()
        if cid:
            payload["correlation_id"] = cid
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in _RESERVED:
                payload[key] = value
        return json.dumps(payload, default=str)


class CorrelationFilter(logging.Filter):
    """Attach the ambient correlation id to plain-text records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id_var.get() or "-"
        return True


def configure_logging(
    level: str = "INFO", json_output: bool = True, stream: TextIO | None = None
) -> None:
    """Install the root handler. Safe to call more than once.

    ``stream`` defaults to stdout, where container log collectors look. CLIs whose
    stdout is the product (``fhir-ai-bench --json``) pass stderr instead.
    """
    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream or sys.stdout)
    if json_output:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-7s [%(correlation_id)s] %(name)s: %(message)s"
            )
        )
        handler.addFilter(CorrelationFilter())
    root.addHandler(handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger."""
    return logging.getLogger(name)
