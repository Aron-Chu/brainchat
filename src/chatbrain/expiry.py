"""Text expiry: chat text is gone from the store, the files and Label Studio at ``text_expires_at``.

``run`` does, in order:

1. Store: in one transaction, delete the messages of every window whose
   ``text_expires_at <= now`` (``now == text_expires_at`` is expired) or that is
   already ``expired`` (a stray row left by an interrupted run), and set those
   windows ``expired``. In the same transaction, delete every ask whose ``expires_at``
   has come (Task 10b; ``asks_deleted``). Labels, features, assignments, predictions,
   manifests and ``emote_names`` stay.
   Also in that transaction (Slice 1, Task 2; Q10): delete every ``stream_activity`` row
   whose ``expires_at <= now`` (30 days after its stream started) and write its expired
   marker (E2), delete the markers whose stream no window names, and delete the
   ``imported_streams`` rows whose path has no open ``imported_files`` row (E9's safety
   net). Activity counts that have not expired stay, even after their windows' text expired.
   Each Moment's top emote (``moment_emotes``, glance F1) goes once its ``expires_at <= now``
   or its stream has a marker; until then it stays, after the text too.
2. Files (ruling C3): retry deleting import sources the importer could not delete,
   and empty ``imported_dir``. A file found at a logged path is deleted only when it
   is an export whose windows are all in that import's log and, for a ``/2`` file, whose
   every stream is logged for that path from an export at least as new (E9), so a newer
   export that happens to land at the same path is never deleted unimported. Closing a
   path's last open log row drops its ``imported_streams`` rows too. A link inside
   ``imported_dir`` (a symlink or a Windows junction) is removed as a link and never
   followed, and an ``imported_dir`` that is itself a link is left alone, so nothing
   outside the folder is ever deleted.
   Then it counts the files that may still hold chat text (``held_files``): logged import
   sources still on disk after the retry (``sources_left``), and, with ``inbox``, the
   inbox's export and unknown ``.json`` files no import logged (``inbox_files``: a refused
   file, or one never imported). Each also counts as past its text expiry
   (``sources_left_past``, ``inbox_past``) once ``text_due`` has come: the earliest window
   ``text_expires_at`` the file names, or ``exported_at`` + 14 days if sooner, read leniently;
   for a file that names neither, its last change + 14 days. Nothing here deletes an inbox
   file (ruling B10: it may be the only copy); the CLI reports the counts on every command.
3. Word models (C5), in ``models_dir`` (default ``manifest.default_models_dir(store)``, the
   ``models`` folder beside the database). A model's files are always looked for there, as
   ``<model_id>/``, never at a stored absolute path, so a moved or copied data folder only
   ever touches its own files. A model of rung >= 1 holds word features (rung 1: its
   TF-IDF vocabulary). Its ``model.joblib`` is deleted, ``file_deleted_at`` set and the
   model retired once every window in its splits has no text left, or once a deletion
   check removed a message of a split window it saw: that window holds now, or has held
   since a check dropped messages from it (``text_drops``), fewer messages than its
   manifest's ``text_message_counts`` recorded, so the trigger holds even after that
   window's text expired. A manifest that can't be read can't show its words are still
   allowed, so its file goes too. The manifest stays. ``models_deleted`` counts only files
   this run removed: a file already gone while its folder still holds the manifest is
   recorded as deleted without being counted, and a model with neither its file nor its
   manifest in ``models_dir`` is left as it is and counted in ``models_missing`` on every
   run. A file that can't be deleted yet is retired at once (so nothing uses it) and
   retried on the next run; ``models_pending`` counts those. Rung 0 holds counts only and
   is never touched by this step.
   Then leftovers: a ``.writing-*`` file in ``models_dir``, and a
   ``<model_id>/model.joblib`` with no ``manifests`` row or whose row says its file was
   deleted, are what an interrupted write or a restored copy leaves behind. They are
   deleted once last changed at least ``LEFTOVER_MIN_AGE`` before ``now``, so a write still
   running in another process is left alone; ``model_leftovers_deleted`` counts them.
   Links inside ``models_dir`` are never followed.
4. Label Studio (rulings C4, C5, B11): with a ``TaskDeleter``, when any task is due,
   first call ``before_delete`` (the CLI pulls every project's annotations through
   ``lstudio.sync.pull``, so a label made just before expiry isn't lost). That pull
   is best effort: if it raises, the deletes still run, ``ls_pull_error`` holds the
   exception type and ``ls_unsynced_deleted`` counts the deleted tasks whose window
   has no owner label from that project. Then delete every task of a window whose
   ``text_expires_at <= now`` (whatever its ``text_state``) and every task a
   deletion check queued, one at a time, dropping each ``ls_tasks`` row once its
   delete succeeds (the client raised nothing and says it deleted that task); then
   ``compact()`` while a delete has not been compacted yet. The first failure stops
   the Label Studio part and is reported by exception type, or as ``NotDeleted``
   when the client says it deleted nothing; the rest stays pending for the next
   run. Without a client the work stays pending and ``ls_tasks_pending`` counts it.
5. ``VACUUM``, so no deleted page survives in the database file.

The lock is never held while Label Studio is called.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, NamedTuple, Protocol, runtime_checkable

from . import manifest
from .export_format import MAX_TEXT_LIFETIME, ExportRejected, iso, parse_dt, parse_export
from .ingest import close_imported_file, inbox_files, remove_file
from .store import Store

LEFTOVER_MIN_AGE = timedelta(minutes=10)  # a model file changed more recently may be a write in progress


@runtime_checkable
class TaskDeleter(Protocol):
    """What expiry needs from a Label Studio client (Task 6 builds it).

    ``delete_tasks`` deletes the given task IDs and returns how many it deleted; a task
    that is already gone counts as deleted. Expiry passes one ID per call and drops its
    ``ls_tasks`` row only when the call returns 1; any other count keeps it pending.
    ``compact`` reclaims the space the deleted tasks held, so their text is gone from
    Label Studio's own database too.
    """

    def delete_tasks(self, task_ids: list[int]) -> int: ...

    def compact(self) -> None: ...


@dataclass(frozen=True)
class ExpiryResult:
    windows_expired: int
    messages_deleted: int
    files_deleted: int
    ls_tasks_deleted: int
    ls_tasks_pending: int = 0  # Label Studio tasks still to delete
    ls_compaction_pending: bool = False  # deleted, but not compacted yet
    ls_error: str | None = None  # exception type of the Label Studio failure, or "NotDeleted"
    ls_pull_error: str | None = None  # exception type when pulling labels before the deletes failed
    ls_unsynced_deleted: int = 0  # deleted tasks with no stored owner label, counted when that pull failed
    models_deleted: int = 0  # word-model files this run deleted (rung >= 1)
    models_pending: int = 0  # word-model files due for deletion that could not be deleted yet (retired already)
    models_missing: int = 0  # word models due for deletion with neither file nor manifest in models_dir
    model_leftovers_deleted: int = 0  # model files no row accounts for (interrupted writes), deleted
    asks_deleted: int = 0  # asked questions whose expires_at had come
    inbox_files: int = 0  # export or unknown .json files in the inbox no import logged (kept, B10)
    inbox_past: int = 0  # of those, past their text expiry
    sources_left: int = 0  # imported files the import could not delete yet (retried every run)
    sources_left_past: int = 0  # of those, past their text expiry


@dataclass(frozen=True)
class HeldFiles:
    """Files in the data folder that may still hold chat text (see the module docstring, step 2)."""

    inbox_files: int = 0
    inbox_past: int = 0
    sources_left: int = 0
    sources_left_past: int = 0


class Emptied(NamedTuple):
    deleted: int  # files deleted
    failed: int  # files (or unreadable folders) that could not be deleted: they may still hold text


@dataclass(frozen=True)
class _WordModels:
    deleted: int = 0
    pending: int = 0
    missing: int = 0


@dataclass(frozen=True)
class _LabelStudioRun:
    deleted: int = 0
    error: str | None = None
    pull_error: str | None = None
    unsynced_deleted: int = 0


def run(
    store: Store,
    now: datetime,
    imported_dir: Path,
    ls: TaskDeleter | None = None,
    *,
    before_delete: Callable[[], object] | None = None,
    models_dir: Path | None = None,
    inbox: Path | None = None,
) -> ExpiryResult:
    """Expire text at ``now`` (timezone-aware; a naive datetime raises ``ValueError``).

    ``before_delete`` runs only when Label Studio tasks are due, before the first delete.
    ``models_dir`` defaults to ``manifest.default_models_dir(store)``. ``inbox`` (the CLI
    passes ``<data>/inbox``) is only counted, never emptied.
    """
    now_s = iso(now)
    models = manifest.default_models_dir(store) if models_dir is None else Path(models_dir)

    with store.transaction() as conn:
        messages_deleted = conn.execute(
            "DELETE FROM messages WHERE window_id IN"
            " (SELECT window_id FROM windows WHERE text_state = 'expired' OR text_expires_at <= ?)",
            (now_s,),
        ).rowcount
        windows_expired = conn.execute(
            "UPDATE windows SET text_state = 'expired' WHERE text_state = 'present' AND text_expires_at <= ?",
            (now_s,),
        ).rowcount
        asks_deleted = conn.execute("DELETE FROM asks WHERE expires_at <= ?", (now_s,)).rowcount
        # Q10: activity counts go 30 days after their stream started; the marker keeps only the stream ID (E2).
        conn.execute(
            "INSERT OR IGNORE INTO stream_activity_expired (stream_id)"
            " SELECT stream_id FROM stream_activity WHERE expires_at <= ?",
            (now_s,),
        )
        conn.execute("DELETE FROM stream_activity WHERE expires_at <= ?", (now_s,))
        # F1: a Moment's top emote goes with its stream's counts, and with any marked stream.
        conn.execute("DELETE FROM moment_emotes WHERE expires_at <= ?", (now_s,))
        conn.execute("DELETE FROM moment_emotes WHERE stream_id IN (SELECT stream_id FROM stream_activity_expired)")
        conn.execute("DELETE FROM stream_activity_expired WHERE stream_id NOT IN (SELECT stream_id FROM windows)")
        conn.execute(  # E9's safety net: no stream ID or export time outlives the source it guards
            "DELETE FROM imported_streams WHERE path NOT IN (SELECT path FROM imported_files WHERE deleted_at IS NULL)"
        )

    files_deleted = _retry_sources(store, now) + empty_folder(Path(imported_dir)).deleted
    held = held_files(store, inbox, now)
    words = _word_models(store, now_s, models)
    leftovers = _sweep_models(store, models, now)
    lsr = _label_studio(store, now_s, ls, before_delete)

    with store.lock:
        pending = len(_due_tasks(store, now_s))
        compaction_pending = store.conn.execute("SELECT 1 FROM ls_compaction").fetchone() is not None
    store.vacuum()
    return ExpiryResult(
        windows_expired,
        messages_deleted,
        files_deleted,
        lsr.deleted,
        ls_tasks_pending=pending,
        ls_compaction_pending=compaction_pending,
        ls_error=lsr.error,
        ls_pull_error=lsr.pull_error,
        ls_unsynced_deleted=lsr.unsynced_deleted,
        models_deleted=words.deleted,
        models_pending=words.pending,
        models_missing=words.missing,
        model_leftovers_deleted=leftovers,
        asks_deleted=asks_deleted,
        inbox_files=held.inbox_files,
        inbox_past=held.inbox_past,
        sources_left=held.sources_left,
        sources_left_past=held.sources_left_past,
    )


# --- files ---------------------------------------------------------------------------------------


def _retry_sources(store: Store, now: datetime) -> int:
    """Close ``imported_files`` rows whose source is gone, deleting it first when it is safe to."""
    with store.lock:
        rows = store.conn.execute(
            "SELECT rowid, path, window_ids_json FROM imported_files WHERE deleted_at IS NULL ORDER BY rowid"
        ).fetchall()
    deleted = 0
    for log_id, logged, ids_json in rows:
        path = Path(logged)
        if os.path.lexists(path):
            if not _is_logged_export(store, logged, set(_json_list(ids_json))) or not remove_file(path):
                continue
            deleted += 1
        with store.transaction() as conn:
            close_imported_file(conn, log_id, iso(now))
    return deleted


def _is_logged_export(store: Store, logged_path: str, logged_ids: set[str]) -> bool:
    """True when the file at ``logged_path`` is an export that import already read.

    Its windows must all be in that import's log, and every stream it carries must be
    logged in ``imported_streams`` for that path from an export at least as new (E9).
    A ``/1`` file carries no streams, so only its windows are compared.
    """
    try:
        export = parse_export(Path(logged_path).read_bytes())
    except (OSError, ExportRejected):
        return False
    if not {w.window_id for w in export.windows} <= logged_ids:
        return False
    if not export.streams:
        return True
    with store.lock:
        seen = dict(
            store.conn.execute(
                "SELECT stream_id, exported_at FROM imported_streams WHERE path = ?", (logged_path,)
            ).fetchall()
        )
    exported_s = iso(export.exported_at)
    return all(s.stream_id in seen and seen[s.stream_id] >= exported_s for s in export.streams)


def _json_list(raw: str) -> list[str]:
    value = json.loads(raw)
    return [v for v in value if isinstance(v, str)] if isinstance(value, list) else []


def held_files(store: Store, inbox: Path | None, now: datetime) -> HeldFiles:
    """Count the files that may still hold chat text (see the module docstring, step 2); delete nothing."""
    with store.lock:
        rows = store.conn.execute(
            "SELECT path, window_ids_json FROM imported_files WHERE deleted_at IS NULL ORDER BY rowid"
        ).fetchall()
    sources = sources_past = 0
    logged: set[str] = set()
    for logged_path, ids_json in rows:
        path = Path(logged_path)
        if not os.path.lexists(path) or not _is_logged_export(store, logged_path, set(_json_list(ids_json))):
            continue  # gone, or a different file now (the inbox count below sees that one)
        due = text_due(path)
        if due is None:
            continue
        logged.add(os.path.normcase(logged_path))
        sources += 1
        sources_past += due <= now
    waiting = past = 0
    if inbox is not None:
        kinds = inbox_files(Path(inbox))
        for path in kinds["export"] + kinds["unknown"]:
            if os.path.normcase(os.path.realpath(path)) in logged:
                continue  # an imported source still on disk: counted above
            due = text_due(path)
            if due is None:
                continue
            waiting += 1
            past += due <= now
    return HeldFiles(waiting, past, sources, sources_past)


_ALWAYS_DUE = datetime.min.replace(tzinfo=timezone.utc)


def text_due(path: Path) -> datetime | None:
    """When the chat text ``path`` may hold is due to go, read leniently; None when the file is gone.

    The earliest window ``text_expires_at`` the file names, or its ``exported_at`` + 14 days if
    that is sooner; for a file that names neither (unreadable, not JSON, not an export), its
    last change + 14 days.
    """
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError:
        raw = b""  # held open by another program, say: dated by its last change below
    try:
        doc: Any = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        doc = None
    dues: list[datetime] = []
    if isinstance(doc, dict):
        exported = _lenient_dt(doc.get("exported_at"))
        if exported is not None:
            dues.append(exported + MAX_TEXT_LIFETIME)
        windows = doc.get("windows")
        for w in windows if isinstance(windows, list) else []:
            expires = _lenient_dt(w.get("text_expires_at")) if isinstance(w, dict) else None
            if expires is not None:
                dues.append(expires)
    if dues:
        return min(dues)
    try:
        changed = path.stat().st_mtime
    except FileNotFoundError:
        return None
    except OSError:
        return _ALWAYS_DUE  # can't tell when it was written: treat its text as due
    return datetime.fromtimestamp(changed, timezone.utc) + MAX_TEXT_LIFETIME


def _lenient_dt(value: Any) -> datetime | None:
    try:
        return parse_dt(value)
    except ExportRejected:
        return None


def empty_folder(folder: Path) -> Emptied:
    """Delete everything under ``folder``; count the files deleted and those that could not be.

    ``failed`` counts files that are still there (a file another program holds open, say)
    and folders that could not be read, since either may still hold text. A link that
    can't be removed, or an empty folder left behind, holds no text and isn't counted.

    Links (symlinks and Windows junctions) are removed as links and never followed, so
    nothing outside ``folder`` is touched; a ``folder`` that is itself a link is left
    alone. ``os.walk`` isn't used: with ``followlinks=False`` it still descends into
    junctions, because a junction isn't a symlink.
    """
    if _is_link(folder) or not folder.is_dir():
        return Emptied(0, 0)
    return _empty_real_dir(folder)


def _is_link(path: Path) -> bool:
    return os.path.islink(path) or os.path.isjunction(path)


def _empty_real_dir(folder: Path) -> Emptied:
    try:
        with os.scandir(folder) as it:
            entries = list(it)
    except OSError:
        return Emptied(0, 1)  # unreadable: whatever it holds is still there
    deleted = failed = 0
    for entry in entries:
        path = Path(entry.path)
        try:
            if entry.is_junction() or (entry.is_symlink() and entry.is_dir()):
                with contextlib.suppress(OSError):
                    _remove_link(path)  # a link to a folder: drop the link, never descend
            elif entry.is_dir(follow_symlinks=False):
                inner = _empty_real_dir(path)
                deleted += inner.deleted
                failed += inner.failed
                if not inner.failed:
                    with contextlib.suppress(OSError):
                        path.rmdir()
            elif remove_file(path):  # a file, or a link to one (unlink removes the link itself)
                deleted += 1
            else:
                failed += 1
        except OSError:
            failed += 1
    return Emptied(deleted, failed)


def _remove_link(path: Path) -> None:
    try:
        os.unlink(path)
    except OSError:
        os.rmdir(path)  # removes a directory link itself, not its target


# --- word models -----------------------------------------------------------------------------------


def _word_models(store: Store, now_s: str, models_dir: Path) -> _WordModels:
    """Delete the files of word models whose text is gone or changed (see the module docstring, step 3)."""
    with store.lock:
        rows = [
            r[0]
            for r in store.conn.execute(
                "SELECT model_id FROM manifests WHERE rung >= ? AND file_deleted_at IS NULL ORDER BY model_id",
                (manifest.WORD_RUNG,),
            )
        ]
        if not rows:
            return _WordModels()
        states = dict(store.conn.execute("SELECT window_id, text_state FROM windows").fetchall())
        counts = dict(store.conn.execute("SELECT window_id, COUNT(*) FROM messages GROUP BY window_id").fetchall())
        drops = dict(store.conn.execute("SELECT window_id, messages_left FROM text_drops").fetchall())
    deleted = pending = missing = 0
    for model_id in rows:
        folder = manifest.model_folder(models_dir, model_id)
        doc = manifest.read(folder) if folder is not None else None
        if not _word_model_due(doc, states, counts, drops):
            continue
        model_file = folder / manifest.MODEL_NAME if folder is not None else None
        had_file = model_file is not None and os.path.lexists(model_file)
        if not had_file and (folder is None or not os.path.lexists(folder / manifest.MANIFEST_NAME)):
            missing += 1  # nothing of this model here (a copied or partial data folder): nothing to delete or record
            continue
        gone = not had_file or remove_file(model_file)
        with store.transaction() as conn:
            conn.execute(
                "UPDATE manifests SET retired_at = COALESCE(retired_at, ?) WHERE model_id = ?", (now_s, model_id)
            )
            if gone:  # the file is not in this data folder's models/ any more
                conn.execute("UPDATE manifests SET file_deleted_at = ? WHERE model_id = ?", (now_s, model_id))
        deleted += had_file and gone
        pending += not gone
    return _WordModels(deleted, pending, missing)


def _word_model_due(
    doc: dict | None, states: dict[str, str], counts: dict[str, int], drops: dict[str, int]
) -> bool:
    """True when a word model's file must go (see the module docstring, step 3)."""
    splits = doc.get("splits") if doc is not None else None
    seen = doc.get("text_message_counts") if doc is not None else None
    if not isinstance(splits, dict) or not isinstance(seen, dict):
        return True
    windows: list[str] = []
    for ids in splits.values():
        if not isinstance(ids, list) or not all(isinstance(w, str) for w in ids):
            return True
        windows.extend(ids)
    if not any(states.get(w) == "present" for w in windows):
        return True  # every split window's text is gone
    for w, n in seen.items():
        if not isinstance(n, int) or isinstance(n, bool):
            return True
        if states.get(w) == "present" and counts.get(w, 0) < n:
            return True  # a deletion check removed a message this model saw (C5)
        if drops.get(w, n) < n:
            return True  # the same, recorded by the check, whether or not the window's text has expired since
    return False


def _sweep_models(store: Store, models_dir: Path, now: datetime) -> int:
    """Delete model files no row accounts for (see the module docstring, step 3); return how many."""
    try:
        with os.scandir(models_dir) as it:
            entries = list(it)
    except OSError:
        return 0  # no models folder (yet)
    with store.lock:
        file_deleted_at = dict(store.conn.execute("SELECT model_id, file_deleted_at FROM manifests").fetchall())
    cutoff = (now - LEFTOVER_MIN_AGE).timestamp()
    deleted = 0
    for entry in entries:
        if entry.is_junction() or entry.is_symlink():
            continue  # never follow a link
        if entry.name.startswith(manifest.WRITING_PREFIX):
            if entry.is_file(follow_symlinks=False):
                deleted += _remove_if_old(Path(entry.path), cutoff)
        elif manifest.WRITTEN_ID.fullmatch(entry.name) and entry.is_dir(follow_symlinks=False):
            if entry.name not in file_deleted_at or file_deleted_at[entry.name] is not None:
                deleted += _remove_if_old(Path(entry.path) / manifest.MODEL_NAME, cutoff)
    return deleted


def _remove_if_old(path: Path, cutoff: float) -> int:
    """Delete ``path`` when it is there and last changed at or before ``cutoff``; 1 when it was deleted."""
    try:
        if path.lstat().st_mtime > cutoff:
            return 0
    except OSError:
        return 0  # not there
    return int(remove_file(path))


# --- Label Studio ----------------------------------------------------------------------------------


def _due_tasks(store: Store, now_s: str) -> list[int]:
    rows = store.conn.execute(
        "SELECT t.task_id FROM ls_tasks t JOIN windows w ON w.window_id = t.window_id WHERE w.text_expires_at <= ?"
        " UNION SELECT task_id FROM ls_deletions ORDER BY 1",
        (now_s,),
    ).fetchall()
    return [r[0] for r in rows]


def _unsynced(store: Store, task_id: int) -> bool:
    """True when the task's window has no owner label from the task's project."""
    return (
        store.conn.execute(
            "SELECT 1 FROM ls_tasks t WHERE t.task_id = ? AND NOT EXISTS (SELECT 1 FROM labels l"
            " WHERE l.window_id = t.window_id AND l.project = t.project_kind AND l.origin = 'aron')",
            (task_id,),
        ).fetchone()
        is not None
    )


def _label_studio(
    store: Store, now_s: str, ls: TaskDeleter | None, before_delete: Callable[[], object] | None
) -> _LabelStudioRun:
    """Pull (best effort), delete due tasks one at a time, then compact."""
    if ls is None:
        return _LabelStudioRun()
    with store.lock:
        due = _due_tasks(store, now_s)
    pull_error = None
    if due and before_delete is not None:
        try:
            before_delete()
        except Exception as exc:  # B11: the deletes still run; the CLI says what may be lost
            pull_error = type(exc).__name__
    deleted = unsynced = 0

    def result(error: str | None) -> _LabelStudioRun:
        return _LabelStudioRun(deleted, error, pull_error, unsynced)

    for task_id in due:
        with store.lock:
            lost = pull_error is not None and _unsynced(store, task_id)
        try:
            gone = ls.delete_tasks([task_id])
        except Exception as exc:  # unreachable, refused, timed out: stop and keep the rest pending
            return result(type(exc).__name__)
        if gone != 1:  # the client says the task wasn't deleted: its text may still be there
            return result("NotDeleted")
        with store.transaction() as conn:
            conn.execute("DELETE FROM ls_tasks WHERE task_id = ?", (task_id,))  # cascades to ls_deletions
            conn.execute("INSERT OR IGNORE INTO ls_compaction (id, due_at) VALUES (1, ?)", (now_s,))
        deleted += 1
        unsynced += lost
    with store.lock:
        due_compaction = store.conn.execute("SELECT 1 FROM ls_compaction").fetchone() is not None
    if due_compaction:
        try:
            ls.compact()
        except Exception as exc:
            return result(type(exc).__name__)
        with store.transaction() as conn:
            conn.execute("DELETE FROM ls_compaction")
    return result(None)
