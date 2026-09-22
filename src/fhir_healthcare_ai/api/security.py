"""Access control for the HTTP layer: API-key authentication and per-caller rate limits.

The pipeline already refuses unsafe *queries*; this module decides who may ask at all.
It is deliberately small:

* **Authentication** is an app-wide FastAPI dependency (:func:`authenticate`), so a
  route added later is protected by default and has to be listed in
  :data:`PUBLIC_ROUTES` to opt out. Probes, the landing page and the OpenAPI docs are
  public; everything else needs a key once ``API_REQUIRE_AUTH`` is on.
* **Identity** is the key's configured *name*. It is bound to
  :data:`~fhir_healthcare_ai.logging_config.actor_var`, which every audit event and log
  line reads, so the trail says who asked without the secret ever being written down.
* **Rate limiting** (:func:`rate_limit`) is attached only to the routes that reach the
  FHIR server and the model, and is keyed by actor (or client address when
  authentication is off).

The limiter lives in process memory. With several workers or replicas each one counts
separately, so the effective limit is multiplied by their number; a deployment that
needs a hard global limit should put a shared store (e.g. Redis) or the gateway in
front of this.
"""

from __future__ import annotations

import hmac
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from fhir_healthcare_ai.config import SecuritySettings
from fhir_healthcare_ai.logging_config import actor_var, get_logger

logger = get_logger(__name__)

API_KEY_HEADER = "X-API-Key"
ANONYMOUS = "anonymous"

#: Route paths (as declared, not as requested) that never require a key. Probes must
#: answer for an orchestrator that holds no credentials; ``/`` is the landing page.
#: ``/docs`` and ``/openapi.json`` are plain Starlette routes that app dependencies do
#: not reach, so they are public without being listed here.
# `/ui/{name}` serves the page's own CSS/JS: they hold no data, and the page cannot send a
# key before it has loaded. Matched on the route template, so no other path slips through.
PUBLIC_ROUTES: frozenset[str] = frozenset({"/", "/ui/{name}", "/health/live", "/health/ready"})

# auto_error=False: a missing credential is reported by us, with the shared error shape
# and a WWW-Authenticate header, rather than by FastAPI's own 403.
_bearer = HTTPBearer(auto_error=False, description="`Authorization: Bearer <API key>`")
_api_key_header = APIKeyHeader(
    name=API_KEY_HEADER, auto_error=False, description="Alternative to the Bearer header."
)


class SlidingWindowRateLimiter:
    """Allow at most ``limit`` hits per ``window`` seconds for each key.

    A sliding log: each key keeps the timestamps of its hits inside the window, so there
    is no burst of ``2 * limit`` at a window boundary as with fixed windows. Memory is
    bounded by ``limit`` timestamps per active key; idle keys are swept once the table
    grows past ``max_keys``.

    Thread-safe. The critical section never awaits, so a plain lock is also safe under
    asyncio and in FastAPI's threadpool.
    """

    def __init__(
        self,
        limit: int,
        window_seconds: float = 60.0,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 10_000,
    ) -> None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        self.limit = limit
        self.window = window_seconds
        self._clock = clock
        self._max_keys = max_keys
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def hit(self, key: str) -> float | None:
        """Record one request for ``key``.

        Returns ``None`` when it is allowed, otherwise the number of seconds until the
        oldest hit leaves the window. A refused request is not recorded, so a client
        that keeps retrying too early is not locked out for longer.
        """
        now = self._clock()
        cutoff = now - self.window
        with self._lock:
            hits = self._hits.get(key)
            if hits is None:
                if len(self._hits) >= self._max_keys:
                    self._sweep(cutoff)
                hits = self._hits[key] = deque()
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.limit:
                return max(hits[0] - cutoff, 0.0)
            hits.append(now)
            return None

    def _sweep(self, cutoff: float) -> None:
        for key in [k for k, hits in self._hits.items() if not hits or hits[-1] <= cutoff]:
            del self._hits[key]


@dataclass
class SecurityState:
    """Per-application security configuration, built once by :func:`install_security`."""

    settings: SecuritySettings
    limiter: SlidingWindowRateLimiter | None
    #: (name, secret) pairs, pre-encoded for constant-time comparison.
    keys: list[tuple[str, bytes]] = field(default_factory=list)

    def identify(self, presented: str) -> str | None:
        """Return the name of the key matching ``presented``, or ``None``.

        Every configured key is compared, with no early exit, so response time does not
        reveal how many keys exist or which one came closest.
        """
        candidate = presented.encode("utf-8")
        match: str | None = None
        for name, secret in self.keys:
            if hmac.compare_digest(candidate, secret):
                match = name
        return match


def install_security(app: FastAPI, settings: SecuritySettings) -> SecurityState:
    """Attach the security state that :func:`authenticate` and :func:`rate_limit` read."""
    limit = settings.rate_limit_per_minute
    state = SecurityState(
        settings=settings,
        limiter=SlidingWindowRateLimiter(limit) if limit > 0 else None,
        keys=[(n, k.get_secret_value().encode("utf-8")) for n, k in settings.api_keys.items()],
    )
    app.state.security = state
    logger.info(
        "api security configured",
        extra={
            "auth_required": settings.auth_enabled,
            "api_key_names": sorted(settings.api_keys),
            "rate_limit_per_minute": limit,
        },
    )
    return state


def _security_state(request: Request) -> SecurityState:
    state: SecurityState | None = getattr(request.app.state, "security", None)
    if state is None:
        # Fail closed: an app that forgot to install security must not serve requests.
        raise RuntimeError("install_security() was not called for this application")
    return state


def _unauthorized(request: Request, detail: str, reason: str) -> HTTPException:
    logger.warning(
        "authentication failed",
        extra={
            "reason": reason,
            "path": request.url.path,
            "client": request.client.host if request.client else None,
        },
    )
    return HTTPException(401, detail, headers={"WWW-Authenticate": "Bearer"})


async def authenticate(
    request: Request,
    bearer: Annotated[HTTPAuthorizationCredentials | None, Security(_bearer)],
    api_key: Annotated[str | None, Security(_api_key_header)],
) -> str:
    """Resolve the caller and bind them to the request context.

    Must stay ``async``: FastAPI runs sync dependencies in a worker thread with a copy
    of the context, and the actor binding would then never reach the endpoint.
    """
    state = _security_state(request)
    route_path = getattr(request.scope.get("route"), "path", request.url.path)
    if not state.settings.auth_enabled or route_path in PUBLIC_ROUTES:
        actor = ANONYMOUS
    else:
        presented = bearer.credentials if bearer else api_key
        if not presented:
            raise _unauthorized(request, "missing API key", "missing")
        name = state.identify(presented)
        if name is None:
            raise _unauthorized(request, "invalid API key", "invalid")
        actor = name
    actor_var.set(actor)
    request.state.actor = actor
    return actor


async def rate_limit(request: Request) -> None:
    """Refuse the request with 429 once its caller exceeds the per-minute budget.

    Attach to a route with ``dependencies=[Depends(rate_limit)]``. It relies on
    :func:`authenticate` having run first, which app-level dependencies guarantee.
    """
    state = _security_state(request)
    if state.limiter is None:
        return
    actor = getattr(request.state, "actor", ANONYMOUS)
    if actor == ANONYMOUS:
        key = f"ip:{request.client.host if request.client else 'unknown'}"
    else:
        key = f"key:{actor}"
    retry_after = state.limiter.hit(key)
    if retry_after is None:
        return
    seconds = max(1, math.ceil(retry_after))
    logger.warning("rate limit exceeded", extra={"path": request.url.path, "retry_after": seconds})
    raise HTTPException(
        429,
        f"rate limit of {state.limiter.limit} requests per minute exceeded",
        headers={"Retry-After": str(seconds)},
    )


#: Convenience for route decorators: ``dependencies=[RATE_LIMITED]``.
RATE_LIMITED = Depends(rate_limit)
