"""Where BrainChat keeps its data, and the checks that keep it out of unsafe places.

Real chat lives only in the data folder named by ``CHATBRAIN_DATA``. That folder
must never sit inside this repo (or hold it), under OneDrive, or inside any git
work tree. ``data_dir`` refuses all of those before it creates anything. ``resolve``
makes the same checks and creates nothing, for code that only needs to know where
the folder is.

Paths are compared as text after ``os.path.realpath`` and ``os.path.normcase``, with
a Windows ``\\\\?\\`` or ``\\\\.\\`` prefix stripped, so case, symlinks, ``..`` and the
extended-length form can't slip a path past the checks. A path segment that ends in
a space or a dot is refused outright: Windows drops those characters, so
``chat-brain.`` can name the repo itself.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path

ENV_VAR = "CHATBRAIN_DATA"
SUBDIRS = ("inbox", "imported", "models", "labelstudio")
# Windows sets these to the user's OneDrive folders; a data folder under any of them is refused.
ONEDRIVE_VARS = ("OneDrive", "OneDriveCommercial", "OneDriveConsumer")

# src/chatbrain/paths.py -> the repo root. The project is installed editable by uv.
REPO_ROOT = Path(__file__).resolve().parents[2]

_PREFIXES = ("\\\\?\\", "\\\\.\\")  # \\?\ (extended-length) and \\.\ (device namespace)
_UNC = "UNC\\"


class UnsafeDataDir(RuntimeError):
    """``CHATBRAIN_DATA`` is unset or points somewhere chat must never be stored."""


def data_dir(env: Mapping[str, str] = os.environ) -> Path:
    """Return the checked data folder from ``env[CHATBRAIN_DATA]``, creating its subfolders.

    Raises ``UnsafeDataDir`` when the variable is unset or relative, when a path
    segment ends in a space or a dot, when the folder is inside the repo or holds it,
    has a ``OneDrive`` path segment, sits under a folder named by ``OneDrive``,
    ``OneDriveCommercial`` or ``OneDriveConsumer`` (read from ``env`` and from the
    process environment), or has a ``.git`` entry in itself or any ancestor. Nothing is
    created on refusal.
    """
    target = resolve(env)
    for name in SUBDIRS:
        (target / name).mkdir(parents=True, exist_ok=True)
    return target


def resolve(env: Mapping[str, str] = os.environ) -> Path:
    """The data folder ``data_dir`` would return, after the same checks, without creating anything."""
    raw = (env.get(ENV_VAR) or "").strip()
    if not raw:
        raise UnsafeDataDir(f"{ENV_VAR} is not set; point it at a folder outside the repo and OneDrive")
    plain = _plain(raw)
    if not Path(plain).is_absolute():
        raise UnsafeDataDir(f"{ENV_VAR} must be an absolute path, not {raw!r}")
    _refuse_trailing_dot_or_space(plain)

    repo_root = Path(REPO_ROOT)
    if not (repo_root / "pyproject.toml").is_file():
        # Fail closed: without the repo root the inside-repo check would be meaningless.
        raise UnsafeDataDir(f"cannot locate the chat-brain repo root (looked at {repo_root})")
    repo = _real(str(repo_root))
    onedrive = _onedrive_folders(env)

    lexical = os.path.abspath(plain)
    resolved = _real(plain)
    for candidate in dict.fromkeys((lexical, resolved)):
        _refuse_unsafe(candidate, repo, onedrive)

    return Path(resolved)


def _refuse_unsafe(path: str, repo: str, onedrive: list[tuple[str, str]]) -> None:
    _refuse_trailing_dot_or_space(path)
    key = os.path.normcase(path)
    if _within(key, os.path.normcase(repo)):
        raise UnsafeDataDir(f"{path} is inside the chat-brain repo {repo}")
    if _within(os.path.normcase(repo), key):
        raise UnsafeDataDir(f"{path} holds the chat-brain repo {repo}")
    for part in Path(path).parts:
        if part.casefold().startswith("onedrive"):
            raise UnsafeDataDir(f"{path} is under OneDrive ({part!r})")
    for var, folder in onedrive:
        if _within(key, os.path.normcase(folder)):
            raise UnsafeDataDir(f"{path} is under OneDrive ({var} is {folder})")
    p = Path(path)
    for folder in (p, *p.parents):
        if (folder / ".git").exists():
            raise UnsafeDataDir(f"{path} is inside a git work tree ({folder})")


def _onedrive_folders(env: Mapping[str, str]) -> list[tuple[str, str]]:
    """(variable, real path) for each OneDrive variable set to an absolute path."""
    found: list[tuple[str, str]] = []
    for var in ONEDRIVE_VARS:
        for source in (env, os.environ):
            value = _plain((source.get(var) or "").strip())
            if value and Path(value).is_absolute():
                found.append((var, _real(value)))
    return list(dict.fromkeys(found))


def _real(path: str) -> str:
    """``os.path.realpath`` of a plain path, with any prefix it adds stripped again."""
    return _plain(os.path.realpath(_plain(path)))


def _plain(path: str) -> str:
    """``path`` with ``/`` as ``\\`` and without a ``\\\\?\\`` or ``\\\\.\\`` prefix (``\\\\?\\UNC\\`` becomes ``\\\\``)."""
    if os.altsep:
        path = path.replace(os.altsep, os.sep)
    for prefix in _PREFIXES:
        if path.startswith(prefix):
            rest = path[len(prefix):]
            if rest[: len(_UNC)].upper() == _UNC:
                return "\\\\" + rest[len(_UNC):]
            return rest
    return path


def _refuse_trailing_dot_or_space(path: str) -> None:
    for part in re.split(r"[\\/]", path):
        if part and part != part.rstrip(". "):
            raise UnsafeDataDir(f"{path}: a path segment ends in a space or a dot ({part!r}); Windows would drop it")


def _within(child: str, parent: str) -> bool:
    """True when ``child`` is ``parent`` or below it (both normalised the same way)."""
    if child == parent:
        return True
    base = parent if parent.endswith(os.sep) else parent + os.sep
    return child.startswith(base)
