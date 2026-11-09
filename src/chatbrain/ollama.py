"""The one gate in front of Ollama: loopback only, and new enough (rulings R3, C13).

Chat text goes to Ollama, so the client must stay on this PC:

- ``make_client`` builds the only kind of client BrainChat uses: a loopback
  ``base_url`` (``127.0.0.0/8``, ``::1`` or ``localhost``, the same rule as the Label
  Studio client), no proxy settings from the environment (``trust_env=False``: on
  Windows httpx would otherwise read the system proxy from the registry and send
  ``127.0.0.1`` requests through it) and no redirects. Requests use relative paths.
- ``require_loopback`` checks a client without calling it; ``teacher.suggest`` runs it
  before every request.
- ``check`` also asks ``GET /api/version`` and refuses below 0.31.2, the first version
  with structured output while thinking is off (ruling R3). A pre-release such as
  ``0.31.2-rc1`` counts as below its release. ``teacher.run``, the ``suggest`` command
  and the narrator call it first.

Every refusal is an ``OllamaError``; its message names the rule or the exception type,
never a response body.
"""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlsplit

import httpx

DEFAULT_URL = "http://127.0.0.1:11434"
MIN_VERSION = (0, 31, 2)
VERSION_TIMEOUT_S = 10.0

_MIN = ".".join(map(str, MIN_VERSION))
_VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)([-+].*)?", re.ASCII)


class OllamaError(RuntimeError):
    """Ollama can't be used: not loopback, unreachable, too old or missing the model."""


def make_client(base_url: str = DEFAULT_URL, *, transport: httpx.BaseTransport | None = None) -> httpx.Client:
    """A client for the local Ollama at ``base_url`` (loopback only, no proxies, no redirects).

    ``transport`` is for tests (``httpx.MockTransport``). Per-request timeouts are set
    by the caller.
    """
    return httpx.Client(
        base_url=_loopback(base_url), transport=transport, trust_env=False, follow_redirects=False
    )


def require_loopback(client: httpx.Client) -> None:
    """Refuse a client that could send a request off this PC. Makes no request."""
    _loopback(str(client.base_url))
    if client.trust_env:
        raise OllamaError(
            "the Ollama client must not read proxy settings from the environment; build it with ollama.make_client()"
        )
    if client.follow_redirects:
        raise OllamaError("the Ollama client must not follow redirects; build it with ollama.make_client()")


def check(client: httpx.Client) -> None:
    """``require_loopback``, then refuse unless ``GET /api/version`` reports 0.31.2 or newer."""
    require_loopback(client)
    try:
        r = client.get("/api/version", timeout=VERSION_TIMEOUT_S)
    except httpx.HTTPError as exc:
        raise OllamaError(f"Ollama is not reachable at {client.base_url} ({type(exc).__name__})") from None
    if r.status_code != 200:
        raise OllamaError(f"Ollama {_MIN} or newer is needed; /api/version answered HTTP {r.status_code}")
    try:
        doc = r.json()
    except ValueError:
        doc = None
    version = doc.get("version") if isinstance(doc, dict) else None
    parsed = _parse_version(version) if isinstance(version, str) else None
    if parsed is None:
        raise OllamaError(f"Ollama {_MIN} or newer is needed; it reported a version that can't be read")
    if parsed < (*MIN_VERSION, 1):
        raise OllamaError(f"Ollama {_MIN} or newer is needed; this one is {version}")


def _parse_version(text: str) -> tuple[int, int, int, int] | None:
    """``(major, minor, patch, 1)`` for a release, ``(..., 0)`` for a pre-release; None if unreadable."""
    m = _VERSION.fullmatch(text.strip())
    if m is None:
        return None
    release = 0 if (m.group(4) or "").startswith("-") else 1
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), release


def _loopback(base_url: str) -> str:
    """``base_url`` without a trailing slash, or ``OllamaError`` unless it is http(s) on a loopback host."""
    parts = urlsplit(base_url)
    if parts.scheme not in ("http", "https"):
        raise OllamaError("the Ollama URL must be http or https")
    if parts.username is not None or parts.password is not None or parts.query or parts.fragment:
        raise OllamaError("the Ollama URL must not carry a user, a password, a query or a fragment")
    host = parts.hostname
    if not host:
        raise OllamaError("the Ollama URL has no host")
    if host != "localhost":
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = False
        if not loopback:
            raise OllamaError("the Ollama URL must be a loopback address (127.0.0.1, ::1 or localhost)")
    try:
        parts.port  # noqa: B018 - raises ValueError for a bad port
    except ValueError:
        raise OllamaError("the Ollama URL has a bad port") from None
    return base_url.rstrip("/")
