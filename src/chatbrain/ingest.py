"""Import exports, apply deletion checks, and the freshness gate.

Import (``import_export``): a whole file imports in one transaction, all or nothing.
Windows whose ``language`` isn't ``en`` and windows whose text has expired at ``now``
(``text_expires_at <= now``) are skipped and counted; a window the store already
holds is counted as a duplicate. Each new window's counted features
(``spread.compute(w).to_json()``) are stored with it. The import is logged in
``imported_files`` under the resolved, case-folded path, and the source file is
deleted as soon as the import commits, whatever happened to its windows (ruling C3):
the store is the only copy of text. A file that can't be deleted then is retried by
``expiry.run``.

The same transaction counts each new Moment's top emote (glance F1, ``faces.top_emote``) into
``moment_emotes`` through ``Store.backfill_moment_emotes``, after the file's activity lines, so
its deadline is the stream's counts' one, or with no counts its earliest window start + 30 days.
A stream whose counts expired (or would expire by ``now``) gets none.

A ``chatbrain-export/2`` file also carries one activity line per stream (Slice 1, Task 2;
Q10). The same transaction stores each through ``Store.put_stream_activity`` (counts
that already expired are skipped and counted, and leave only the expired marker). Only
a stream that a stored window names gets a line, checked after this file's windows are
stored: that window keeps the stream's expired marker (E2), so a later re-export can
never bring swept counts back. A stream with no stored window (none in any file, or only
skipped ones) gets no line and no marker; its counts are only counted as skipped when
they already expired. The import also logs every stream the file carried in
``imported_streams`` under the same path (E9), so expiry never deletes a file at a
logged path before its streams were read. A stream
whose channel differs from a window of the same stream, in the file or already stored,
refuses the whole file (C22). A path's ``imported_streams`` rows go when its last open
``imported_files`` row is closed.

An export also counts as deletion-check coverage (ruling B7): it lists only messages
that still existed at its ``exported_at``, so every window it adds is recorded in
``checked_windows`` as of ``min(exported_at, now)``. A window the store already holds
is covered the same way only when every message the store keeps for it is in the
export; a duplicate missing a stored message can't vouch for it. Coverage only ever
moves forward (the newer of the stored and the new date wins).

Deletion checks (``apply_deletion_check``): a check touches only the windows it
lists (ruling C2). A listed window keeps only the message IDs in its ``present``
list; its features and its Moment tally (F1, or the tally's row when no emote is
left) are recomputed, its Label Studio tasks are queued for deletion,
and ``text_drops`` keeps the fewest messages it has held since (ruling C5), so expiry
finds the word models that saw a dropped message even after the window's text expired.
Windows the check doesn't list are left alone.

Freshness gate (``require_fresh_deletion_check``): labeling, suggesting and training
refuse unless every window with text was listed by a deletion check, or covered by an
export, from within ``max_age`` (B6, B7), and the newest of those is at most ``max_age``
old and no more than 5 minutes ahead of this PC's clock (T1). Refusals carry counts
and dates only.

Nothing here prints or returns chat text, and errors never carry window IDs.
"""

from __future__ import annotations

import json
import os
import sqlite3
import stat
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from . import spread
from .export_format import (
    DELETION_CHECK_FORMAT,
    EXPORT_FORMAT,
    EXPORT_FORMAT_V2,
    EXPORT_FORMATS,
    WINDOW_LIST_FORMAT,
    DeletionCheck,
    ExportRejected,
    iso,
    parse_deletion_check,
    parse_dt,
    parse_export,
    window_list,
)
from .store import ACTIVITY_LIFETIME, ActivityPut, Store

LANGUAGE = "en"
MAX_CHECK_AGE = timedelta(days=7)
CLOCK_SKEW = timedelta(minutes=5)


class StaleDeletionCheck(RuntimeError):
    """No deletion check, an old or future-dated one, or windows with text no recent check listed."""


@dataclass(frozen=True)
class ImportResult:
    windows_added: int
    windows_duplicate: int
    skipped_language: int
    skipped_expired: int
    # False when the import committed but the source file could not be deleted;
    # ``expiry.run`` retries it (its ``imported_files`` row keeps ``deleted_at`` empty).
    source_deleted: bool = True
    # Activity lines from a /2 file: stored (newer than any stored version), and skipped
    # because the counts had already expired at import. A line kept as it was, or one for a
    # stream no stored window names whose counts have not expired, counts in neither.
    streams_written: int = 0
    streams_expired_skipped: int = 0


@dataclass(frozen=True)
class DeletionApplied:
    messages_dropped: int
    windows: frozenset[str]  # windows that lost at least one message


@dataclass(frozen=True)
class InboxChecks:
    """What ``apply_inbox_checks`` did. ``refused`` holds ``"<file name>: <reason>"`` lines."""

    checks_applied: int
    messages_dropped: int
    windows: frozenset[str]
    refused: tuple[str, ...]


# --- import -----------------------------------------------------------------------------------


def import_export(store: Store, path: Path, imported_dir: Path, now: datetime) -> ImportResult:
    """Import one ``chatbrain-export/1`` or ``/2`` file, then delete it.

    A refused file (``ExportRejected``) or a failed import leaves the file where it is
    and the store unchanged: no window, activity line, marker or log row. A ``/2``
    stream whose channel differs from a window of the same stream, in this file or
    already stored, refuses the file (C22). Only a stream a stored window names gets an
    activity line (see the module notes). ``imported_dir`` is kept for the interface:
    under ruling C3 nothing is moved there, and ``expiry.run`` keeps that folder empty.
    """
    del imported_dir  # ruling C3: the source is deleted, never moved
    iso(now)  # refuses a naive datetime before anything is read
    real = Path(os.path.realpath(path))
    export = parse_export(real.read_bytes())
    logged = os.path.normcase(str(real))
    window_channels: dict[str, set[str]] = {}
    for w in export.windows:
        window_channels.setdefault(w.stream_id, set()).add(w.channel)
    for i, s in enumerate(export.streams):
        if window_channels.get(s.stream_id, {s.channel}) != {s.channel}:
            raise ExportRejected(_CHANNEL_DIFFERS.format(i))

    covered_at = iso(min(export.exported_at, now))  # B7: an export is coverage as of when it was made
    added = duplicate = skipped_language = skipped_expired = 0
    streams_written = streams_expired = 0
    with store.transaction() as conn:
        for i, s in enumerate(export.streams):  # C22, against windows already stored
            if conn.execute(
                "SELECT 1 FROM windows WHERE stream_id = ? AND channel != ? LIMIT 1", (s.stream_id, s.channel)
            ).fetchone():
                raise ExportRejected(_CHANNEL_DIFFERS.format(i))
        for w in export.windows:
            if w.language != LANGUAGE:
                skipped_language += 1
            elif w.text_expires_at <= now:
                skipped_expired += 1
            elif store.insert_window(w, export.source, now):
                conn.execute(
                    "INSERT INTO features (window_id, json) VALUES (?, ?)",
                    (w.window_id, spread.compute(w).to_json()),
                )
                _cover(conn, w.window_id, covered_at)
                added += 1
            else:
                duplicate += 1
                kept = {r[0] for r in conn.execute("SELECT message_id FROM messages WHERE window_id = ?", (w.window_id,))}
                if kept <= {m.message_id for m in w.messages}:
                    _cover(conn, w.window_id, covered_at)
        for s in export.streams:
            # Only a stream a stored window names gets a line: the window keeps its expired marker
            # (E2), so swept counts can never be revived. Without one, nothing could keep the marker.
            put: ActivityPut | None
            if conn.execute("SELECT 1 FROM windows WHERE stream_id = ? LIMIT 1", (s.stream_id,)).fetchone():
                put = store.put_stream_activity(s, export.exported_at, now)
            else:
                put = "expired" if s.started_at + ACTIVITY_LIFETIME <= now else None
            streams_written += put == "written"
            streams_expired += put == "expired"
            conn.execute(
                "INSERT INTO imported_streams (path, stream_id, exported_at) VALUES (?, ?, ?)"
                " ON CONFLICT (path, stream_id) DO UPDATE SET exported_at = MAX(exported_at, excluded.exported_at)",
                (logged, s.stream_id, iso(export.exported_at)),
            )
        # F1: each new Moment's top emote, after the streams so its deadline can use this file's
        # counts (and any Moment an older store never counted, while its text is present).
        store.backfill_moment_emotes(now)
        log_id = conn.execute(
            "INSERT INTO imported_files (path, window_ids_json, imported_at, deleted_at) VALUES (?, ?, ?, NULL)",
            (logged, json.dumps([w.window_id for w in export.windows]), iso(now)),
        ).lastrowid

    deleted = remove_file(real)
    if deleted:
        if Path(path) != real and os.path.islink(path):
            remove_file(Path(path))
        mark_file_deleted(store, log_id, now)
    return ImportResult(
        windows_added=added,
        windows_duplicate=duplicate,
        skipped_language=skipped_language,
        skipped_expired=skipped_expired,
        source_deleted=deleted,
        streams_written=streams_written,
        streams_expired_skipped=streams_expired,
    )


_CHANNEL_DIFFERS = "export.streams[{}].channel: differs from the channel of a window of the same stream"


def _cover(conn: sqlite3.Connection, window_id: str, checked_at: str) -> None:
    """Record that ``window_id`` was vouched for at ``checked_at``, keeping the newer date.

    The caller holds ``store.transaction()``.
    """
    conn.execute(
        "INSERT INTO checked_windows (window_id, checked_at) VALUES (?, ?)"
        " ON CONFLICT (window_id) DO UPDATE SET checked_at = MAX(checked_at, excluded.checked_at)",
        (window_id, checked_at),
    )


def remove_file(path: Path) -> bool:
    """Delete ``path``, clearing a read-only flag if that is what stops it. True when gone."""
    try:
        path.unlink(missing_ok=True)
        return True
    except PermissionError:
        try:
            os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
            path.unlink(missing_ok=True)
            return True
        except OSError:
            return False
    except OSError:
        return False


def mark_file_deleted(store: Store, log_id: int, now: datetime) -> None:
    with store.transaction() as conn:
        close_imported_file(conn, log_id, iso(now))


def close_imported_file(conn: sqlite3.Connection, log_id: int, deleted_at: str) -> None:
    """Close one ``imported_files`` row; when no row for its path stays open, drop the path's
    ``imported_streams`` rows too (E9). The caller holds ``store.transaction()``.
    """
    conn.execute("UPDATE imported_files SET deleted_at = ? WHERE rowid = ?", (deleted_at, log_id))
    conn.execute(
        "DELETE FROM imported_streams WHERE path = (SELECT path FROM imported_files WHERE rowid = ?)"
        " AND NOT EXISTS (SELECT 1 FROM imported_files f WHERE f.path = imported_streams.path AND f.deleted_at IS NULL)",
        (log_id,),
    )


# --- deletion checks ------------------------------------------------------------------------------

# A stored caption over a window: one of the ask's result windows, or a window it took a line from.
# secure_delete zeroes the cleared text.
_CLEAR_CAPTIONS = (
    "UPDATE asks SET summary = NULL WHERE summary IS NOT NULL AND ("
    " EXISTS (SELECT 1 FROM json_each(asks.result_window_ids_json) WHERE value = ?)"
    " OR EXISTS (SELECT 1 FROM json_each(asks.query_json, '$.notes.caption_for.msgs') AS c"
    " WHERE json_extract(c.value, '$[0]') = ?))"
)


def apply_deletion_check(store: Store, check: DeletionCheck) -> DeletionApplied:
    """Apply one deletion check in one transaction (rulings C2, C5, B6).

    For each listed window the store holds: drop the messages whose IDs aren't in its
    ``present`` list, recompute its features if anything was dropped, queue its Label
    Studio tasks for deletion, keep in ``text_drops`` the fewest messages the window
    has held since a check dropped one (for word models, C5), clear the stored caption
    (``asks.summary``, Task 10c) of every ask over the window, since a caption can repeat a
    dropped line, and record that this check listed it. Windows whose
    text already expired keep their features. Windows the check doesn't list, and
    IDs the store never held, are ignored.
    """
    checked_at = iso(check.checked_at)
    dropped = 0
    affected: set[str] = set()
    with store.transaction() as conn:
        held = 0
        for window_id, present in sorted(check.present.items()):
            row = conn.execute("SELECT text_state FROM windows WHERE window_id = ?", (window_id,)).fetchone()
            if row is None:
                continue
            held += 1
            _cover(conn, window_id, checked_at)
            if row[0] != "present":
                continue
            stored = [r[0] for r in conn.execute("SELECT message_id FROM messages WHERE window_id = ?", (window_id,))]
            gone = [mid for mid in stored if mid not in present]
            if not gone:
                continue
            conn.executemany(
                "DELETE FROM messages WHERE window_id = ? AND message_id = ?", [(window_id, mid) for mid in gone]
            )
            dropped += len(gone)
            affected.add(window_id)
            conn.execute(
                "INSERT INTO features (window_id, json) VALUES (?, ?)"
                " ON CONFLICT (window_id) DO UPDATE SET json = excluded.json",
                (window_id, spread.compute(store.as_window(window_id)).to_json()),
            )
            store.recount_moment_emote(window_id)  # F1: count again, or drop the row when no emote is left
            conn.execute(
                "INSERT OR IGNORE INTO ls_deletions (task_id) SELECT task_id FROM ls_tasks WHERE window_id = ?",
                (window_id,),
            )
            conn.execute(
                "INSERT INTO text_drops (window_id, messages_left) VALUES (?, ?)"
                " ON CONFLICT (window_id) DO UPDATE SET messages_left = MIN(messages_left, excluded.messages_left)",
                (window_id, len(stored) - len(gone)),
            )
            conn.execute(_CLEAR_CAPTIONS, (window_id, window_id))
        conn.execute(
            "INSERT INTO deletion_checks (checked_at, windows_checked, messages_dropped) VALUES (?, ?, ?)",
            (checked_at, held, dropped),
        )
    return DeletionApplied(dropped, frozenset(affected))


def apply_inbox_checks(store: Store, inbox: Path, *, now: datetime) -> InboxChecks:
    """Apply every deletion-check file in ``inbox`` (oldest ``checked_at`` first) and delete it.

    Export and window-list files are left for their own commands. A check dated more
    than 5 minutes after ``now``, a check that breaks the format, and a ``.json`` file
    that isn't a BrainChat file at all are left in place and listed in ``refused``.
    """
    iso(now)
    good: list[tuple[datetime, str, Path, DeletionCheck]] = []
    refused: list[str] = []
    for path in _json_files(inbox):
        try:
            raw = path.read_bytes()
        except OSError as exc:
            refused.append(f"{path.name}: could not be read ({type(exc).__name__})")
            continue
        kind = _format_of(raw)
        if kind in (*EXPORT_FORMATS, WINDOW_LIST_FORMAT):
            continue
        if kind != DELETION_CHECK_FORMAT:
            refused.append(f"{path.name}: not a BrainChat file")
            continue
        try:
            check = parse_deletion_check(raw)
        except ExportRejected as exc:
            refused.append(f"{path.name}: {exc}")
            continue
        if check.checked_at > now + CLOCK_SKEW:
            refused.append(f"{path.name}: checked_at is more than 5 minutes ahead of this PC's clock")
            continue
        good.append((check.checked_at, path.name, path, check))

    dropped = 0
    affected: set[str] = set()
    for _, _, path, check in sorted(good, key=lambda g: (g[0], g[1])):
        applied = apply_deletion_check(store, check)
        dropped += applied.messages_dropped
        affected |= applied.windows
        remove_file(path)  # if it stays, applying it again later changes nothing
    return InboxChecks(len(good), dropped, frozenset(affected), tuple(refused))


def apply_pending_checks(store: Store, inbox: Path, *, now: datetime) -> frozenset[str]:
    """Apply every waiting deletion-check file in ``inbox`` and return the affected windows (C36).

    Good files are applied first; then any refused file raises ``ExportRejected``
    naming each one, so a broken or future-dated check can't be skipped silently.
    """
    result = apply_inbox_checks(store, inbox, now=now)
    if result.refused:
        raise ExportRejected("deletion-check files refused: " + "; ".join(result.refused))
    return result.windows


def inbox_files(inbox: Path) -> dict[str, list[Path]]:
    """The inbox's ``.json`` files by kind: ``export``, ``deletion-check``, ``window-list``, ``unknown``."""
    kinds: dict[str, list[Path]] = {"export": [], "deletion-check": [], "window-list": [], "unknown": []}
    names = {
        EXPORT_FORMAT: "export",
        EXPORT_FORMAT_V2: "export",
        DELETION_CHECK_FORMAT: "deletion-check",
        WINDOW_LIST_FORMAT: "window-list",
    }
    for path in _json_files(inbox):
        try:
            kind = names.get(_format_of(path.read_bytes()) or "", "unknown")
        except OSError:
            kind = "unknown"
        kinds[kind].append(path)
    return kinds


def _json_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() == ".json" and p.is_file())


def _format_of(raw: bytes) -> str | None:
    """The file's top-level ``format`` string, or None when it isn't a JSON object with one."""
    try:
        doc = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None
    fmt = doc.get("format") if isinstance(doc, dict) else None
    return fmt if isinstance(fmt, str) else None


# --- freshness gate -----------------------------------------------------------------------------------


def require_fresh_deletion_check(store: Store, now: datetime, max_age: timedelta = MAX_CHECK_AGE) -> None:
    """Raise ``StaleDeletionCheck`` unless deletion checks are recent enough to label on.

    Checked in this order:

    1. no deletion check was applied and no import covered a window (B7);
    2. the newest check or export coverage is more than 5 minutes ahead of ``now`` (T1);
    3. a window with text was last listed by a check, or covered by an export, more
       than ``max_age`` before ``now`` (or never): the refusal gives the count only;
    4. the newest check or coverage is more than ``max_age`` old (this matters when no
       window has text).

    Exactly ``max_age`` old still passes.
    """
    now_s = iso(now)
    with store.lock:
        newest_check = store.conn.execute("SELECT MAX(checked_at) FROM deletion_checks").fetchone()[0]
        newest_cover = store.conn.execute("SELECT MAX(checked_at) FROM checked_windows").fetchone()[0]
        if newest_check is None and newest_cover is None:
            raise StaleDeletionCheck(
                "no deletion check has been applied; write a window list and apply the check that answers it"
            )
        newest_s = max(t for t in (newest_check, newest_cover) if t is not None)
        newest = parse_dt(newest_s, "checked_at")
        if newest > now + CLOCK_SKEW:
            raise StaleDeletionCheck(
                f"the newest deletion check or export ({newest_s}) is dated in the future,"
                f" more than 5 minutes after {now_s}"
            )
        uncovered = store.conn.execute(
            "SELECT COUNT(*) FROM windows w LEFT JOIN checked_windows c ON c.window_id = w.window_id"
            " WHERE w.text_state = 'present' AND (c.checked_at IS NULL OR c.checked_at < ?)",
            (iso(now - max_age),),
        ).fetchone()[0]
    if uncovered:
        raise StaleDeletionCheck(
            f"{uncovered} window(s) with text are not listed by a deletion check from the last {_days(max_age)};"
            " write a window list and apply the check that answers it"
        )
    if now - newest > max_age:
        raise StaleDeletionCheck(f"the newest deletion check ({newest_s}) is more than {_days(max_age)} old")


def _days(age: timedelta) -> str:
    return f"{age.days} days" if age == timedelta(days=age.days) else str(age)


# --- window list -------------------------------------------------------------------------------------


def write_window_list(store: Store, out: Path) -> int:
    """Write the ``chatbrain-window-list/1`` request for every window with text; return its size."""
    with store.lock:
        rows = store.conn.execute(
            "SELECT window_id, stream_id, start_at, end_at FROM windows WHERE text_state = 'present'"
            " ORDER BY window_id"
        ).fetchall()
    raw = window_list(
        [(wid, sid, parse_dt(s, "windows.start_at"), parse_dt(e, "windows.end_at")) for wid, sid, s, e in rows]
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, out)
    return len(rows)
