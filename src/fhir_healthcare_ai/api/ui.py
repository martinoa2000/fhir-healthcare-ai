"""A small browser UI over the public API.

The page is static HTML, CSS and vanilla JavaScript shipped inside the package, so it
works offline, needs no build step and is installed with the wheel. It only calls the
same public routes a client would (``/health``, ``/capabilities``, ``/query``,
``/patient/{id}/analyze``); it has no privileged access and adds no server-side logic.

Assets are served from a fixed table rather than a directory mount: a request can only
ever resolve to one of the files listed here, and ``APIRouter.include_router`` does not
carry ``Mount`` routes, so the whole UI stays a single router that ``create_app`` includes.
"""

from __future__ import annotations

from functools import cache
from importlib import resources

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

#: Published name -> media type. Anything not listed here is a 404.
_ASSETS: dict[str, str] = {
    "index.html": "text/html; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "app.css": "text/css; charset=utf-8",
}

#: Everything the page needs comes from this origin, and no inline script or style is
#: used, so the policy can be strict. It is the second line of defence behind the page
#: never inserting response data as HTML.
_CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
    "img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)

_HEADERS = {
    "Content-Security-Policy": _CSP,
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    # The page is tiny and versioned with the package; revalidating keeps an upgraded
    # server from pairing new HTML with a cached old script.
    "Cache-Control": "no-cache",
}

router = APIRouter(include_in_schema=False)


@cache
def _read_asset(name: str) -> bytes:
    """Load a packaged file once. ``importlib.resources`` works for wheels and zips alike."""
    return resources.files("fhir_healthcare_ai.api").joinpath("static", name).read_bytes()


def _asset_response(name: str) -> Response:
    media_type = _ASSETS.get(name)
    if media_type is None:
        raise HTTPException(404, "not found")
    return Response(_read_asset(name), media_type=media_type, headers=_HEADERS)


@router.get("/")
async def index() -> Response:
    """The single-page UI."""
    return _asset_response("index.html")


@router.get("/ui/{name}")
async def asset(name: str) -> Response:
    """A static asset of the UI, from the fixed table above."""
    return _asset_response(name)
