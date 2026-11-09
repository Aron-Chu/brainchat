"""A loopback-only Label Studio client (Label Studio 1.23).

The client refuses any host that isn't a loopback address (``127.0.0.0/8``, ``::1`` or
``localhost``), never reads proxy settings from the environment (``trust_env=False``)
and never follows a redirect, so chat text can only go to the local container.

Auth (checked against the 1.23.2 image source, ruling C23):

- ``pat`` (a personal access token): ``POST /api/token/refresh/`` with
  ``{"refresh": <token>}`` returns ``{"access": ...}``, sent as
  ``Authorization: Bearer <access>``. The access token is refreshed every 4 minutes,
  and once more after a 401.
- ``legacy``: ``Authorization: Token <token>``.

Errors carry the HTTP status or the exception type, never a response body, a URL with
a token or the token itself.

Every export API call leaves a copy holding live text in the folders Label Studio
writes exports to (``<data>/labelstudio/export`` and ``<data>/labelstudio/media/export``).
``export_annotations`` empties both as soon as the answer has been read (ruling B13).
``compact()`` makes deleted task text leave Label Studio's files: it empties the same
folders again, then runs SQLite ``VACUUM`` on Label Studio's database inside the
container with ``docker compose exec``. ``export_copies_left`` holds how many files the
last emptying could not delete (one another program holds open, say); while any is left,
``compact()`` raises ``CompactionFailed`` after its ``VACUUM``, so the compaction stays due
and the next expiry tries again, and the CLI reports the count.

``from_env`` builds the client from the data folder's ``.env`` (``LS_TOKEN``,
``LS_AUTH``, ``LS_URL``); the CLI is the only caller, and only for the real data
folder (ruling C17). Agents never read that file.

Projects are found by exact title (``TITLES``). ``chatbrain demo`` passes
``DEMO_TITLES`` (ruling B20), so its projects, all titled ``BrainChat demo · ...``,
stay apart: a client with the real titles never finds them, so the real store's
``adopt_untracked`` and ``pull`` never read a demo task.
"""

from __future__ import annotations

import ipaddress
import os
import subprocess
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx

from .. import expiry, paths
from .. import scheme as scheme_module
from . import config

DEFAULT_URL = "http://127.0.0.1:14610"
# The real projects already exist in Label Studio under these titles, so they keep the old name.
TITLES = {
    "suggested": "Chat Brain · Suggested",
    "blind": "Chat Brain · Blind",
    "repeat": "Chat Brain · Repeat",
}
KINDS = tuple(TITLES)
DEMO_TITLES = {
    "suggested": "BrainChat demo · Suggested",
    "blind": "BrainChat demo · Blind",
    "repeat": "BrainChat demo · Repeat",
}
ACCESS_TTL_S = 240.0  # PAT access tokens are refreshed after 4 minutes
SERVICE = "chatbrain-ls"
LS_DATABASE = "/label-studio/data/label_studio.sqlite3"
EXPORT_FOLDERS = (("export",), ("media", "export"))
COMPACT_TIMEOUT_S = 600
_PAGE_SIZE = 100
_TASK_KEYS = {"data", "predictions"}
_DATA_KEYS = {"window_id", "messages", "counts_text"}

_VACUUM = (
    "import sqlite3\n"
    f"c = sqlite3.connect({LS_DATABASE!r}, timeout=60, isolation_level=None)\n"
    "c.execute('PRAGMA wal_checkpoint(TRUNCATE)')\n"
    "c.execute('VACUUM')\n"
    "c.execute('PRAGMA wal_checkpoint(TRUNCATE)')\n"
    "c.close()\n"
)


class LabelStudioError(RuntimeError):
    """Label Studio answered with an error status or an unexpected body (status only, never the body)."""


class CompactionFailed(RuntimeError):
    """The VACUUM inside the container did not run to the end."""


class ConfigError(ValueError):
    """The ``.env`` Label Studio settings are unusable. Names the key, never its value."""


def loopback_url(base_url: str) -> str:
    """``base_url`` without a trailing slash, or ``ValueError`` unless it is http(s) on a loopback host."""
    parts = urlsplit(base_url)
    if parts.scheme not in ("http", "https"):
        raise ValueError("Label Studio URL must be http or https")
    if parts.username is not None or parts.password is not None or parts.query or parts.fragment:
        raise ValueError("Label Studio URL must not carry a user, a password, a query or a fragment")
    host = parts.hostname
    if not host:
        raise ValueError("Label Studio URL has no host")
    if host != "localhost":
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = False
        if not loopback:
            raise ValueError("Label Studio URL must be a loopback address (127.0.0.1, ::1 or localhost)")
    try:
        parts.port  # noqa: B018 - raises ValueError for a bad port
    except ValueError:
        raise ValueError("Label Studio URL has a bad port") from None
    return base_url.rstrip("/")


def check_import(kind: str, tasks: list[dict]) -> None:
    """Refuse an import that could show the labeler an earlier answer or send more than labeling needs.

    Every kind: a task holds ``data`` (window ID, messages, counts text) and maybe
    ``predictions``, never ``annotations``. Blind and repeat tasks never hold
    ``predictions``.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("a task must be an object")
        if "annotations" in task:
            raise ValueError("imports never carry annotations")
        if "predictions" in task and kind != "suggested":
            raise ValueError(f"{kind} tasks never carry predictions")
        if not set(task) <= _TASK_KEYS:
            raise ValueError("a task holds only data and predictions")
        data = task.get("data")
        if not isinstance(data, dict) or set(data) != _DATA_KEYS:
            raise ValueError("task data holds exactly window_id, messages and counts_text")


class LabelStudio:
    """A Label Studio client for the three BrainChat projects. It satisfies ``expiry.TaskDeleter``."""

    def __init__(
        self,
        base_url: str = DEFAULT_URL,
        *,
        token: str,
        auth: Literal["pat", "legacy"],
        data_dir: Path,
        compose_file: Path,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 30.0,
        clock: Callable[[], float] | None = None,
        runner: Callable[..., subprocess.CompletedProcess] | None = None,
        titles: Mapping[str, str] | None = None,
    ) -> None:
        self.base_url = loopback_url(base_url)
        if auth not in ("pat", "legacy"):
            raise ValueError("auth must be pat or legacy")
        if not isinstance(token, str) or not token.strip():
            raise ValueError("the Label Studio token is empty")
        self.titles = dict(TITLES if titles is None else titles)
        if set(self.titles) != set(KINDS) or len(set(self.titles.values())) != len(KINDS):
            raise ValueError(f"titles must give each of {', '.join(KINDS)} its own title")
        if any(not isinstance(t, str) or not t.strip() for t in self.titles.values()):
            raise ValueError("a project title is empty")
        self.auth = auth
        self._token = token.strip()
        self.data_dir = Path(data_dir)
        self.compose_file = Path(compose_file)
        self._clock = clock or time.monotonic
        self._runner = runner or subprocess.run
        self._access: str | None = None
        self._access_at = 0.0
        self.export_copies_left = 0  # export copies the last emptying could not delete
        self._http = httpx.Client(
            base_url=self.base_url, transport=transport, timeout=timeout, trust_env=False, follow_redirects=False
        )

    def __repr__(self) -> str:  # never the token
        return f"LabelStudio({self.base_url!r}, auth={self.auth!r})"

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> LabelStudio:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- HTTP -------------------------------------------------------------------------------

    def _authorization(self) -> str:
        if self.auth == "legacy":
            return f"Token {self._token}"
        if self._access is None or self._clock() - self._access_at >= ACCESS_TTL_S:
            self._refresh()
        return f"Bearer {self._access}"

    def _refresh(self) -> None:
        self._access = None
        r = self._http.post("/api/token/refresh/", json={"refresh": self._token})
        if r.status_code != 200:
            raise LabelStudioError(f"token refresh refused (HTTP {r.status_code})")
        try:
            access = r.json().get("access")
        except (ValueError, AttributeError):
            access = None
        if not isinstance(access, str) or not access:
            raise LabelStudioError("token refresh returned no access token")
        self._access = access
        self._access_at = self._clock()

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        r = self._http.request(method, path, headers={"Authorization": self._authorization()}, **kwargs)
        if r.status_code == 401 and self.auth == "pat":
            self._access = None  # the access token may have expired early: refresh once and retry
            r = self._http.request(method, path, headers={"Authorization": self._authorization()}, **kwargs)
        return r

    @staticmethod
    def _ok(r: httpx.Response, what: str) -> httpx.Response:
        if not r.is_success:
            raise LabelStudioError(f"{what}: HTTP {r.status_code}")
        return r

    @staticmethod
    def _json(r: httpx.Response, what: str) -> Any:
        try:
            return r.json()
        except ValueError:
            raise LabelStudioError(f"{what}: the answer is not JSON") from None

    # --- projects ---------------------------------------------------------------------------

    def _title(self, kind: str) -> str:
        _title(kind)  # refuses an unknown kind
        return self.titles[kind]

    def _find(self, kind: str) -> dict | None:
        title = self._title(kind)
        page = 1
        while True:
            r = self._ok(
                self._request("GET", "/api/projects/", params={"title": title, "page": page, "page_size": _PAGE_SIZE}),
                "project list",
            )
            body = self._json(r, "project list")
            results = body.get("results") if isinstance(body, dict) else None
            if not isinstance(results, list):
                raise LabelStudioError("project list: unexpected answer")
            for project in results:  # the title filter matches substrings; the title must match exactly
                if isinstance(project, dict) and project.get("title") == title:
                    return project
            if not body.get("next"):
                return None
            page += 1

    def project_id(self, kind: str) -> int | None:
        """The ID of the project for ``kind``, or None when it doesn't exist yet."""
        project = self._find(kind)
        return None if project is None else int(project["id"])

    def ensure_project(self, kind: str, label_config: str | None = None) -> int:
        """Create the project for ``kind`` if needed and keep its view and settings current.

        ``label_config`` defaults to the view built from ``config/labels_v1.json``.
        """
        xml = config.label_config(scheme_module.load()) if label_config is None else label_config
        wanted = {
            "label_config": xml,
            "maximum_annotations": 1,
            "enable_empty_annotation": False,
            "show_collab_predictions": kind == "suggested",  # prelabels show only in the suggested project
            "show_annotation_history": False,
        }
        project = self._find(kind)
        if project is None:
            r = self._ok(self._request("POST", "/api/projects/", json={"title": self._title(kind), **wanted}), "create project")
            body = self._json(r, "create project")
            return int(body["id"])
        changed = {k: v for k, v in wanted.items() if project.get(k) != v}
        if changed:
            self._ok(self._request("PATCH", f"/api/projects/{int(project['id'])}/", json=changed), "update project")
        return int(project["id"])

    def use_model_version(self, project_id: int, model_version: str) -> None:
        """Show the prelabels of ``model_version`` (Label Studio shows one model version's predictions)."""
        r = self._ok(self._request("GET", f"/api/projects/{int(project_id)}/"), "read project")
        if self._json(r, "read project").get("model_version") != model_version:
            self._ok(
                self._request("PATCH", f"/api/projects/{int(project_id)}/", json={"model_version": model_version}),
                "set model version",
            )

    # --- tasks ----------------------------------------------------------------------------------

    def import_tasks(self, project_id: int, kind: str, tasks: list[dict]) -> list[int]:
        """Import ``tasks`` into the project and return their new task IDs, in order."""
        check_import(kind, tasks)
        if not tasks:
            return []
        r = self._ok(
            self._request(
                "POST", f"/api/projects/{int(project_id)}/import", params={"return_task_ids": "true"}, json=tasks
            ),
            "import",
        )
        body = self._json(r, "import")
        ids = body.get("task_ids") if isinstance(body, dict) else None
        if (
            not isinstance(ids, list)
            or len(ids) != len(tasks)
            or not all(isinstance(i, int) and not isinstance(i, bool) for i in ids)
        ):
            raise LabelStudioError("import: Label Studio did not return one task ID per task")
        return ids

    def export_annotations(self, project_id: int, *, all_tasks: bool = False) -> list[dict]:
        """The project's tasks with their annotations (only annotated tasks unless ``all_tasks``).

        Label Studio keeps a copy of every export under ``<data>/labelstudio/export``,
        and that copy holds live text. Both export folders are emptied as soon as the
        answer has been read, whether the call succeeded or not (ruling B13), so no copy
        waits for the next compaction. A copy that can't be deleted yet is counted in
        ``export_copies_left``.
        """
        params = {"exportType": "JSON"}
        if all_tasks:
            params["download_all_tasks"] = "true"
        try:
            r = self._ok(self._request("GET", f"/api/projects/{int(project_id)}/export", params=params), "export")
            body = self._json(r, "export")
        finally:
            self._empty_export_folders()
        if not isinstance(body, list):
            raise LabelStudioError("export: not a list of tasks")
        return body

    def delete_tasks(self, task_ids: list[int]) -> int:
        """Delete each task and return how many are gone; a task that is already gone (404) counts.

        Any other status counts as not deleted. Connection errors raise.
        """
        gone = 0
        for task_id in task_ids:
            r = self._request("DELETE", f"/api/tasks/{int(task_id)}/")
            if r.is_success or r.status_code == 404:
                gone += 1
        return gone

    def _empty_export_folders(self) -> int:
        """Empty both export folders; how many files could not be deleted (``export_copies_left``)."""
        root = self.data_dir / "labelstudio"
        self.export_copies_left = sum(expiry.empty_folder(root.joinpath(*parts)).failed for parts in EXPORT_FOLDERS)
        return self.export_copies_left

    def compact(self) -> None:
        """Empty Label Studio's export folders, then VACUUM its database inside the container.

        Raises ``CompactionFailed`` when the VACUUM fails, or after it when an export copy is
        still there, so the compaction stays due.
        """
        left = self._empty_export_folders()
        args =["docker", "compose", "-f", str(self.compose_file), "exec", "-T", SERVICE, "python", "-c", _VACUUM]
        env = {**os.environ, paths.ENV_VAR: str(self.data_dir)}
        try:
            done = self._runner(args, env=env, capture_output=True, timeout=COMPACT_TIMEOUT_S, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            raise CompactionFailed(type(exc).__name__) from None
        if done.returncode != 0:
            raise CompactionFailed(f"exit status {done.returncode}")  # stderr is never echoed
        if left:
            raise CompactionFailed("export copy still present")


def _title(kind: str) -> str:
    try:
        return TITLES[kind]
    except KeyError:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}") from None


# --- .env -------------------------------------------------------------------------------------------


def _parse_env(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def from_env(
    env_file: Path, *, data_dir: Path, compose_file: Path, titles: Mapping[str, str] | None = None
) -> LabelStudio | None:
    """The client configured by ``env_file``, or None when the file or ``LS_TOKEN`` is missing.

    ``titles`` picks the projects (default ``TITLES``; the demo passes ``DEMO_TITLES``).

    Keys: ``LS_TOKEN`` (required), ``LS_AUTH`` (``pat`` or ``legacy``, default ``pat``)
    and ``LS_URL`` (default ``http://127.0.0.1:14610``, loopback only). Raises
    ``ConfigError`` naming the bad key, never its value.
    """
    try:
        text = Path(env_file).read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f".env could not be read ({type(exc).__name__})") from None
    values = _parse_env(text)
    token = values.get("LS_TOKEN", "").strip()
    if not token:
        return None
    auth = values.get("LS_AUTH", "").strip().lower() or "pat"
    if auth not in ("pat", "legacy"):
        raise ConfigError("LS_AUTH must be pat or legacy")
    url = values.get("LS_URL", "").strip() or DEFAULT_URL
    try:
        loopback_url(url)
    except ValueError:
        raise ConfigError("LS_URL must be an http(s) loopback address, like http://127.0.0.1:14610") from None
    return LabelStudio(
        url, token=token, auth=auth, data_dir=data_dir, compose_file=compose_file, titles=titles  # type: ignore[arg-type]
    )
