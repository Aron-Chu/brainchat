"""The local SQLite store: windows, their text while it lives, and everything that outlives it.

The connection always runs with ``secure_delete=ON`` (deleted text is overwritten,
not left in free pages), ``journal_mode=DELETE`` (no WAL file keeping old pages
around), ``temp_store=MEMORY`` (sorts and VACUUM's scratch copy stay off disk)
and ``foreign_keys=ON``.

Times are stored as fixed-width UTC text (``export_format.iso``), so SQL can compare
them as strings, for example ``text_expires_at <= ?``.

Text lives only in ``messages.text``. Everything else here outlives it: window
bounds, counted features, labels, suggestions, predictions and manifests.

Value lists that the label scheme owns (answers, not-sure reasons, ``cause_source``)
have no SQL CHECK here: ``scheme.validate_answers`` and ``scheme.validate_cause_source``
check them before a row is written (ruling C1).

One ``Store`` may be shared across threads (FastAPI's threadpool): the connection is
opened with ``check_same_thread=False`` and every method holds ``Store.lock``, one
``threading.RLock``. Code that uses ``Store.conn`` directly from more than one thread
holds ``lock`` too. ``Store.transaction()`` holds the lock for a whole write
transaction and commits once, so several writes land all or nothing.

Deletion and Label Studio bookkeeping (Task 4):

- ``imported_files`` logs every import: the normalised source path, every window ID
  in the file, and ``deleted_at`` once the source file is gone (ruling C3).
- ``checked_windows`` holds, per window, the newest ``checked_at`` of a deletion
  check that listed it; the freshness gate needs every window with text listed by
  a recent check (ruling B6).
- ``ls_deletions`` queues Label Studio tasks whose text a deletion check changed
  (ruling C5); a row goes when its ``ls_tasks`` row does.
- ``ls_compaction`` holds one row while Label Studio deleted something and has not
  been compacted since (ruling C4).
- ``text_drops`` holds, per window a deletion check dropped messages from, the fewest
  messages it has held since (ruling C5). It outlives the window's text, so expiry can
  still tell that a word model saw a message a check removed after that window expired.

Asking (Task 10b):

- ``asks`` holds each asked question's query, thread and result window IDs, never its
  numbers. ``expires_at`` is the earlier of 14 days and the earliest result window's
  text expiry (and, for a quoted phrase, the earliest text expiry of any window holding
  it); ``expiry.run`` deletes due rows. ``update_ask`` rewrites one after the local
  model read it (Task 10c); ``summary`` holds its checked caption. A deletion check that drops
  messages from a window clears the caption of every ask over that window (``ingest``).
- ``emote_names`` holds ``(channel, emote_id, name)`` for every emote seen, filled when a
  window is stored. It outlives text, so emote questions still read after expiry.

Per-stream activity (Slice 1, Task 2; Q10):

- ``stream_activity`` holds one activity line per stream from a ``chatbrain-export/2`` file,
  only for a stream a stored window names (``ingest`` checks this, so that window keeps the
  stream's marker below): people chatting per 10 s bin (``null`` = not recorded), the
  stream's span and title. No chat text, speaker or per-user ID. ``expires_at`` is UTC ``started_at`` + 30 days; every
  version of the stream an import reads can only lower it (E3), and at ``now == expires_at``
  the counts are expired. The reads below leave an expired row out before any sweep;
  ``expiry.run`` deletes it.
- ``stream_activity_expired`` holds only the stream ID of a stream whose counts expired
  (E2), so the page can tell expired counts from a stream that never had any. A stream
  with a marker never gets counts back. ``expiry.run`` deletes a marker once no window
  names its stream.
- ``imported_streams`` holds, per logged import path, each stream that file carried and the
  newest ``exported_at`` logged for it (E9), so expiry never deletes a ``/2`` file at a logged
  path before its streams were read. A path's rows live only while one of its
  ``imported_files`` rows is open.

Moment faces (glance F1, call 2):

- ``moment_emotes`` holds, per Moment window, its top emote from ``faces.top_emote``: the exact
  emote ID as exported, the name at its first line and three counts. No text, speaker or
  per-user ID. One row per Moment, the winner only, even below the face bar (``faces.shows`` is
  applied at render); a Moment with no emote has no row. It is counted while the text is present
  (``backfill_moment_emotes``, from import and page upkeep) and outlives the text.
- ``expires_at`` is the stream's activity deadline, or, with no counts, its earliest window
  start + 30 days: one deadline per stream, only ever lowered (``put_stream_activity`` lowers it,
  and deletes the rows on its expired branch). Reads leave out an expired row or a marked
  stream before any sweep; ``expiry.run`` deletes them. A deletion check recounts the row, or
  deletes it when no emote is left (``recount_moment_emote``).
"""

from __future__ import annotations

import contextlib
import functools
import json
import sqlite3
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Concatenate, Literal, ParamSpec, TypeVar

from .export_format import SOURCES, Emote, Message, StreamActivity, Window, iso, parse_dt
from .faces import Tally, top_emote

P = ParamSpec("P")
R = TypeVar("R")

ActivityPut = Literal["written", "kept", "expired"]
ActivityState = Literal["active", "expired", "missing"]
ACTIVITY_LIFETIME = timedelta(days=30)  # Q10: counts expire 30 days after the stream started (UTC)

SCHEMA = """
CREATE TABLE IF NOT EXISTS windows (
    window_id       TEXT PRIMARY KEY,
    channel         TEXT NOT NULL,
    stream_id       TEXT NOT NULL,
    anchor_kind     TEXT NOT NULL CHECK (anchor_kind IN ('moment', 'ordinary')),
    anchor_at       TEXT NOT NULL,
    start_at        TEXT NOT NULL,
    end_at          TEXT NOT NULL,
    language        TEXT NOT NULL,
    text_expires_at TEXT NOT NULL,
    source          TEXT NOT NULL CHECK (source IN ('vod', 'live')),
    imported_at     TEXT NOT NULL,
    text_state      TEXT NOT NULL DEFAULT 'present' CHECK (text_state IN ('present', 'expired')),
    picked_rare     INTEGER NOT NULL DEFAULT 0 CHECK (picked_rare IN (0, 1))
);

CREATE TABLE IF NOT EXISTS messages (
    window_id   TEXT NOT NULL REFERENCES windows (window_id),
    message_id  TEXT NOT NULL,
    rel_ms      INTEGER NOT NULL,
    speaker     TEXT NOT NULL,
    text        TEXT NOT NULL,
    emotes_json TEXT NOT NULL,
    PRIMARY KEY (window_id, message_id)
);

CREATE TABLE IF NOT EXISTS features (
    window_id TEXT PRIMARY KEY REFERENCES windows (window_id),
    json      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS assignments (
    window_id   TEXT PRIMARY KEY REFERENCES windows (window_id),
    exposure    TEXT NOT NULL CHECK (exposure IN ('blind', 'suggested')),
    assigned_at TEXT NOT NULL,
    seed        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS repeats (
    window_id TEXT PRIMARY KEY REFERENCES windows (window_id),
    due_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS labels (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    window_id          TEXT NOT NULL REFERENCES windows (window_id),
    project            TEXT NOT NULL CHECK (project IN ('blind', 'suggested', 'repeat', 'page')),
    origin             TEXT NOT NULL CHECK (origin IN ('aron', 'synthetic')),
    provenance         TEXT NOT NULL CHECK (provenance IN
                           ('blind', 'accepted', 'corrected', 'repeat', 'corrected-on-page', 'synthetic',
                            'unsuggested')),
    labels_version     TEXT NOT NULL,
    guide_version      TEXT NOT NULL,
    answers_json       TEXT NOT NULL,
    not_sure_json      TEXT NOT NULL DEFAULT '{}',
    cause_note         TEXT,
    cause_source       TEXT,
    suggestion_json    TEXT,
    same_as_suggestion INTEGER CHECK (same_as_suggestion IS NULL OR same_as_suggestion IN (0, 1)),
    teacher_model      TEXT,
    prompt_version     TEXT,
    lead_time_s        REAL,
    ls_annotation_id   INTEGER UNIQUE,
    created_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS labels_window ON labels (window_id);

CREATE TABLE IF NOT EXISTS suggestions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    window_id      TEXT NOT NULL REFERENCES windows (window_id),
    teacher_model  TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    labels_version TEXT NOT NULL,
    answers_json   TEXT,
    error          TEXT,
    created_at     TEXT NOT NULL,
    CHECK ((answers_json IS NULL) != (error IS NULL))
);
CREATE INDEX IF NOT EXISTS suggestions_window ON suggestions (window_id);

CREATE TABLE IF NOT EXISTS predictions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    window_id  TEXT NOT NULL REFERENCES windows (window_id),
    model_id   TEXT NOT NULL,
    question   TEXT NOT NULL,
    value_json TEXT NOT NULL,
    confidence REAL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS predictions_window ON predictions (window_id, model_id);

CREATE TABLE IF NOT EXISTS deletion_checks (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    checked_at       TEXT NOT NULL,
    windows_checked  INTEGER NOT NULL,
    messages_dropped INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS imported_files (
    path            TEXT NOT NULL,
    window_ids_json TEXT NOT NULL,
    imported_at     TEXT NOT NULL,
    deleted_at      TEXT
);

CREATE TABLE IF NOT EXISTS ls_tasks (
    window_id    TEXT NOT NULL REFERENCES windows (window_id),
    project_kind TEXT NOT NULL CHECK (project_kind IN ('suggested', 'blind', 'repeat')),
    task_id      INTEGER NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS ls_tasks_window ON ls_tasks (window_id);

CREATE TABLE IF NOT EXISTS checked_windows (
    window_id  TEXT PRIMARY KEY REFERENCES windows (window_id),
    checked_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ls_deletions (
    task_id INTEGER PRIMARY KEY REFERENCES ls_tasks (task_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS ls_compaction (
    id     INTEGER PRIMARY KEY CHECK (id = 1),
    due_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS text_drops (
    window_id     TEXT PRIMARY KEY REFERENCES windows (window_id),
    messages_left INTEGER NOT NULL CHECK (messages_left >= 0)
);

CREATE TABLE IF NOT EXISTS manifests (
    model_id        TEXT PRIMARY KEY,
    rung            INTEGER NOT NULL,
    origin          TEXT NOT NULL CHECK (origin IN ('aron', 'synthetic')),
    path            TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    retired_at      TEXT,
    file_deleted_at TEXT
);

-- Asking (Task 10b). A row lives at most 14 days, and no longer than the text of the
-- earliest result window that still had text when it was asked (expiry deletes it).
-- A refused question is stored as "[refused: <reason>]", never as typed, and a chip
-- edit stores NULL. No answer numbers are stored: a thread re-runs its queries.
CREATE TABLE IF NOT EXISTS asks (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id              TEXT NOT NULL,
    created_at             TEXT NOT NULL,
    expires_at             TEXT NOT NULL,
    question               TEXT,
    query_json             TEXT NOT NULL,
    follow_up              TEXT NOT NULL CHECK (follow_up IN ('new', 'refine', 'about_result')),
    result_window_ids_json TEXT NOT NULL,
    summary                TEXT
);
CREATE INDEX IF NOT EXISTS asks_thread ON asks (thread_id, id);
CREATE INDEX IF NOT EXISTS asks_expires ON asks (expires_at);

-- Public emote names per channel, kept when text expires so "what does KEKW mean"
-- still works. Channel, emote ID and name only: nothing about who used it.
CREATE TABLE IF NOT EXISTS emote_names (
    channel  TEXT NOT NULL,
    emote_id TEXT NOT NULL,
    name     TEXT NOT NULL,
    PRIMARY KEY (channel, emote_id)
);

-- Per-stream activity (Slice 1, Task 2; Q10): people chatting per 10 s bin, as a JSON
-- array (null = not recorded). Counts only: no chat text, speaker or per-user ID.
-- expires_at = the stream's start + 30 days, only ever lowered; expiry deletes the row then.
-- Column names hold no word a refused question may use (P0 checks the database bytes for
-- "started" and "chatter"), so the span is start_at/end_at, as in windows, and the bins counts_json.
CREATE TABLE IF NOT EXISTS stream_activity (
    stream_id     TEXT PRIMARY KEY,
    channel       TEXT NOT NULL,
    start_at      TEXT NOT NULL,
    end_at        TEXT NOT NULL,
    title         TEXT,
    bin_s         INTEGER NOT NULL CHECK (bin_s = 10),
    counts_json   TEXT NOT NULL,
    exported_at   TEXT NOT NULL,
    imported_at   TEXT NOT NULL,
    expires_at    TEXT NOT NULL
);
-- E2: the expired marker. Only the stream ID, which windows already keep.
CREATE TABLE IF NOT EXISTS stream_activity_expired (
    stream_id TEXT PRIMARY KEY
);
-- E9: which streams a logged export path carried, so expiry never deletes
-- a /2 file (or a newer file at a logged path) before its streams are read.
CREATE TABLE IF NOT EXISTS imported_streams (
    path        TEXT NOT NULL,
    stream_id   TEXT NOT NULL,
    exported_at TEXT NOT NULL,
    PRIMARY KEY (path, stream_id)
);

-- Glance F1 (call 2): each Moment's top emote, a count kept at import by exact emote ID.
-- No text, no speaker, no per-user ID. It lives as long as the stream's activity counts
-- (only ever lowered); expiry deletes it then. Column names avoid every word a refused
-- question may use (P0 checks the database bytes).
CREATE TABLE IF NOT EXISTS moment_emotes (
    window_id  TEXT PRIMARY KEY REFERENCES windows (window_id),
    stream_id  TEXT NOT NULL,
    emote_id   TEXT NOT NULL,
    name       TEXT NOT NULL,
    lines      INTEGER NOT NULL CHECK (lines >= 1),
    of_lines   INTEGER NOT NULL CHECK (of_lines >= lines),
    people     INTEGER NOT NULL CHECK (people >= 1 AND people <= lines),
    expires_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS moment_emotes_expires ON moment_emotes (expires_at);
"""

_DATETIME_COLUMNS = ("anchor_at", "start_at", "end_at", "text_expires_at", "imported_at")
_WINDOW_COLUMNS = (
    "window_id",
    "channel",
    "stream_id",
    "anchor_kind",
    "anchor_at",
    "start_at",
    "end_at",
    "language",
    "text_expires_at",
    "source",
    "imported_at",
    "text_state",
    "picked_rare",
)


@dataclass(frozen=True)
class WindowRow:
    """One ``windows`` row: every ``Window`` field except ``messages``, plus import state."""

    window_id: str
    channel: str
    stream_id: str
    anchor_kind: str
    anchor_at: datetime
    start_at: datetime
    end_at: datetime
    language: str
    text_expires_at: datetime
    source: str
    imported_at: datetime
    text_state: str
    picked_rare: bool


def _locked(fn: Callable[Concatenate[Store, P], R]) -> Callable[Concatenate[Store, P], R]:
    """Run a ``Store`` method while holding the store's one ``RLock``."""

    @functools.wraps(fn)
    def wrapper(self: Store, *args: P.args, **kwargs: P.kwargs) -> R:
        with self.lock:
            return fn(self, *args, **kwargs)

    return wrapper


class Store:
    """The BrainChat SQLite database at ``path``, created on first use.

    Safe to share across threads: every method holds ``lock`` (one ``threading.RLock``).
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.lock = threading.RLock()
        with self.lock:
            self.conn = sqlite3.connect(self.path, check_same_thread=False)
            try:
                self._configure()
                self.conn.executescript(SCHEMA)
                self._backfill_emote_names()
            except BaseException:
                self.conn.close()
                raise

    @_locked
    def _configure(self) -> None:
        c = self.conn
        c.execute("PRAGMA secure_delete = ON")
        mode = c.execute("PRAGMA journal_mode = DELETE").fetchone()[0]
        if str(mode).lower() != "delete":
            raise sqlite3.OperationalError(f"could not set journal_mode=DELETE (got {mode!r})")
        c.execute("PRAGMA temp_store = MEMORY")
        c.execute("PRAGMA foreign_keys = ON")
        if c.execute("PRAGMA secure_delete").fetchone()[0] != 1:
            raise sqlite3.OperationalError("could not set secure_delete=ON")

    @_locked
    def close(self) -> None:
        self.conn.close()

    @contextlib.contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Hold ``lock`` and one write transaction; commit on success, roll back on any error.

        Raises ``RuntimeError`` when a transaction is already open on the connection,
        so a caller can't nest one and commit someone else's half-done writes.
        """
        with self.lock:
            if self.conn.in_transaction:
                raise RuntimeError("Store.transaction() does not nest")
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield self.conn
            except BaseException:
                self.conn.rollback()
                raise
            else:
                self.conn.commit()

    @_locked
    def vacuum(self) -> None:
        """Rebuild the file so no deleted page survives in it (outside any transaction)."""
        if self.conn.in_transaction:
            raise RuntimeError("Store.vacuum() runs outside a transaction")
        self.conn.execute("VACUUM")

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- windows -------------------------------------------------------------------

    @_locked
    def add_window(self, w: Window, source: str, now: datetime) -> bool:
        """Store a window and its messages. Returns False (and changes nothing) if it exists.

        Refuses (``ValueError``) a window whose text has already expired at ``now``,
        so expired text is never written; the importer skips and counts those first.
        """
        with self.transaction():
            return self.insert_window(w, source, now)

    @_locked
    def insert_window(self, w: Window, source: str, now: datetime) -> bool:
        """``add_window`` inside a ``transaction()`` the caller already holds, so a whole
        file can import all or nothing. Raises ``RuntimeError`` outside one."""
        if not self.conn.in_transaction:
            raise RuntimeError("Store.insert_window runs inside Store.transaction()")
        if source not in SOURCES:
            raise ValueError(f"source must be one of {', '.join(SOURCES)}")
        # iso() raises ExportRejected (a ValueError) on any naive datetime.
        imported_at = iso(now)
        row = (
            w.window_id,
            w.channel,
            w.stream_id,
            w.anchor_kind,
            iso(w.anchor_at),
            iso(w.start_at),
            iso(w.end_at),
            w.language,
            iso(w.text_expires_at),
            source,
            imported_at,
            "present",
            0,
        )
        if self.conn.execute("SELECT 1 FROM windows WHERE window_id = ?", (w.window_id,)).fetchone():
            return False
        if w.text_expires_at <= now:
            raise ValueError(f"window {w.window_id}: text already expired at {imported_at}")
        self.conn.execute(
            f"INSERT INTO windows ({', '.join(_WINDOW_COLUMNS)}) VALUES ({', '.join('?' * len(_WINDOW_COLUMNS))})",
            row,
        )
        self.conn.executemany(
            "INSERT INTO messages (window_id, message_id, rel_ms, speaker, text, emotes_json)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            [(w.window_id, m.message_id, m.rel_ms, m.speaker, m.text, _emotes_json(m.emotes)) for m in w.messages],
        )
        names = sorted({(w.channel, e.id, e.name) for m in w.messages for e in m.emotes})
        self.conn.executemany(_EMOTE_NAME_UPSERT, names)
        return True

    @_locked
    def window(self, window_id: str) -> WindowRow:
        """The stored row for ``window_id``. Raises ``KeyError`` if there is none."""
        row = self.conn.execute(
            f"SELECT {', '.join(_WINDOW_COLUMNS)} FROM windows WHERE window_id = ?", (window_id,)
        ).fetchone()
        if row is None:
            raise KeyError(window_id)
        values: dict[str, Any] = dict(zip(_WINDOW_COLUMNS, row))
        for name in _DATETIME_COLUMNS:
            values[name] = parse_dt(values[name], f"windows.{name}")
        values["picked_rare"] = bool(values["picked_rare"])
        return WindowRow(**values)

    @_locked
    def as_window(self, window_id: str) -> Window:
        """Rebuild the ``Window`` from the tables; ``messages=()`` once its text has expired."""
        r = self.window(window_id)
        messages = () if r.text_state == "expired" else tuple(self._messages(window_id))
        return Window(
            window_id=r.window_id,
            channel=r.channel,
            stream_id=r.stream_id,
            anchor_kind=r.anchor_kind,  # type: ignore[arg-type]
            anchor_at=r.anchor_at,
            start_at=r.start_at,
            end_at=r.end_at,
            language=r.language,
            text_expires_at=r.text_expires_at,
            messages=messages,
        )

    @_locked
    def messages(self, window_id: str) -> list[Message]:
        """The window's stored messages, by time then ID. Raises ``KeyError`` for an unknown window."""
        if not self.conn.execute("SELECT 1 FROM windows WHERE window_id = ?", (window_id,)).fetchone():
            raise KeyError(window_id)
        return self._messages(window_id)

    @_locked
    def _messages(self, window_id: str) -> list[Message]:
        rows = self.conn.execute(
            "SELECT message_id, rel_ms, speaker, text, emotes_json FROM messages"
            " WHERE window_id = ? ORDER BY rel_ms, message_id",
            (window_id,),
        ).fetchall()
        return [
            Message(
                message_id=mid,
                rel_ms=rel,
                speaker=spk,
                text=text,
                emotes=tuple(Emote(id=e["id"], name=e["name"]) for e in json.loads(ej)),
            )
            for mid, rel, spk, text, ej in rows
        ]

    @_locked
    def window_ids(self, **filters: Any) -> list[str]:
        """Window IDs, sorted, where every ``column=value`` filter on ``windows`` holds.

        Datetimes are compared as stored UTC text and bools as 0/1. An unknown column
        raises ``ValueError``.
        """
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in filters.items():
            if column not in _WINDOW_COLUMNS:
                raise ValueError(f"windows has no column {column!r}")
            if isinstance(value, datetime):
                value = iso(value)
            elif isinstance(value, bool):
                value = int(value)
            clauses.append(f"{column} = ?")
            params.append(value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.conn.execute(f"SELECT window_id FROM windows{where} ORDER BY window_id", params).fetchall()
        return [r[0] for r in rows]

    # --- per-stream activity (Slice 1, Task 2; Q10) ----------------------------------

    @_locked
    def put_stream_activity(self, a: StreamActivity, exported_at: datetime, now: datetime) -> ActivityPut:
        """Store one stream's activity line from an export made at ``exported_at``.

        Runs inside a ``transaction()`` the caller already holds, like ``insert_window``, and
        raises ``RuntimeError`` outside one. Every version first lowers the stream's deadline
        to the earlier of the stored one and ``a.started_at`` + 30 days (E3). Then:

        1. a stream with a marker, or a deadline at or before ``now``: any stored payload is
           deleted, the marker written, and ``"expired"`` returned;
        2. no stored row, or one from an older export: the payload is written with that
           deadline (``"written"``);
        3. otherwise the stored payload stays and only the lowered deadline is saved (``"kept"``).

        Every version also lowers the stream's ``moment_emotes`` deadlines to that one (F1), and
        branch 1 deletes the stream's rows.

        ``ingest.import_export`` calls it only for a stream a stored window names, so the
        marker it writes is never swept (E2) and the stream's counts can never come back.
        """
        if not self.conn.in_transaction:
            raise RuntimeError("Store.put_stream_activity runs inside Store.transaction()")
        now_s, exported_s = iso(now), iso(exported_at)
        own = iso(a.started_at + ACTIVITY_LIFETIME)
        row = self.conn.execute(
            "SELECT exported_at, expires_at FROM stream_activity WHERE stream_id = ?", (a.stream_id,)
        ).fetchone()
        deadline = own if row is None else min(row[1], own)  # fixed-width UTC text sorts as time does
        marked = self.conn.execute(
            "SELECT 1 FROM stream_activity_expired WHERE stream_id = ?", (a.stream_id,)
        ).fetchone()
        # F1: the stream's Moment faces live no longer than its counts (lowered on every version).
        self.conn.execute(
            "UPDATE moment_emotes SET expires_at = MIN(expires_at, ?) WHERE stream_id = ?", (deadline, a.stream_id)
        )
        if marked or deadline <= now_s:
            self.conn.execute("DELETE FROM stream_activity WHERE stream_id = ?", (a.stream_id,))
            self.conn.execute("DELETE FROM moment_emotes WHERE stream_id = ?", (a.stream_id,))
            self.conn.execute("INSERT OR IGNORE INTO stream_activity_expired (stream_id) VALUES (?)", (a.stream_id,))
            return "expired"
        if row is None or row[0] < exported_s:
            self.conn.execute(
                "INSERT INTO stream_activity (stream_id, channel, start_at, end_at, title, bin_s, counts_json,"
                " exported_at, imported_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT (stream_id) DO UPDATE SET channel = excluded.channel, start_at = excluded.start_at,"
                " end_at = excluded.end_at, title = excluded.title, bin_s = excluded.bin_s,"
                " counts_json = excluded.counts_json, exported_at = excluded.exported_at,"
                " imported_at = excluded.imported_at, expires_at = excluded.expires_at",
                (a.stream_id, a.channel, iso(a.started_at), iso(a.ended_at), a.title, a.bin_s,
                 json.dumps(list(a.chatters), separators=(",", ":")), exported_s, now_s, deadline),
            )  # fmt: skip
            return "written"
        self.conn.execute("UPDATE stream_activity SET expires_at = ? WHERE stream_id = ?", (deadline, a.stream_id))
        return "kept"

    @_locked
    def stream_activity(self, stream_id: str, *, now: datetime) -> StreamActivity | None:
        """The stream's activity line, or None when it has none or its counts expired at ``now``."""
        row = self.conn.execute(
            f"SELECT {_ACTIVITY_SELECT} FROM stream_activity a WHERE a.stream_id = ? AND {_ACTIVITY_LIVE}",
            (stream_id, iso(now)),
        ).fetchone()
        return _activity(row) if row else None

    @_locked
    def channel_activity(self, channel: str, *, now: datetime) -> list[StreamActivity]:
        """The channel's activity lines whose counts have not expired at ``now``, newest ``ended_at`` first."""
        rows = self.conn.execute(
            f"SELECT {_ACTIVITY_SELECT} FROM stream_activity a WHERE a.channel = ? AND {_ACTIVITY_LIVE}"
            " ORDER BY a.end_at DESC, a.stream_id",
            (channel, iso(now)),
        ).fetchall()
        return [_activity(r) for r in rows]

    @_locked
    def activity_state(self, stream_id: str, *, now: datetime) -> ActivityState:
        """``"expired"`` for a marker or counts expired at ``now``, ``"active"`` for live counts,
        ``"missing"`` when the stream never had any (a ``/1`` export, say)."""
        now_s = iso(now)
        if self.conn.execute("SELECT 1 FROM stream_activity_expired WHERE stream_id = ?", (stream_id,)).fetchone():
            return "expired"
        row = self.conn.execute("SELECT expires_at FROM stream_activity WHERE stream_id = ?", (stream_id,)).fetchone()
        if row is None:
            return "missing"
        return "expired" if row[0] <= now_s else "active"

    # --- Moment faces (glance F1, call 2) ------------------------------------------------

    @_locked
    def moment_faces(self, stream_id: str, now: datetime) -> dict[str, Tally]:
        """The stream's Moment tallies live at ``now``, by window ID: not past their deadline, no marker.

        Every tally is returned, below the face bar too; the page applies ``faces.shows``.
        """
        rows = self.conn.execute(
            "SELECT f.window_id, f.emote_id, f.name, f.lines, f.of_lines, f.people FROM moment_emotes f"
            " WHERE f.stream_id = ? AND f.expires_at > ?"
            " AND NOT EXISTS (SELECT 1 FROM stream_activity_expired x WHERE x.stream_id = f.stream_id)"
            " ORDER BY f.window_id",
            (stream_id, iso(now)),
        ).fetchall()
        return {wid: Tally(eid, name, lines, of_lines, people) for wid, eid, name, lines, of_lines, people in rows}

    @_locked
    def backfill_moment_emotes(self, now: datetime) -> int:
        """Count the top emote of every Moment window that still has its text and no tally; return how many.

        Skips a window whose text has expired at ``now`` (even before the sweep), a window with no
        emote, and every window of a stream with a marker or whose faces' deadline is at or before
        ``now``. Each written row gets its stream's one deadline (``_faces_deadline``), and the
        stream's other rows are lowered to it. Idempotent. Runs inside a ``transaction()`` the
        caller holds (import), or in its own (page upkeep).
        """
        if self.conn.in_transaction:
            return self._fill_moment_emotes(now)
        with self.transaction():
            return self._fill_moment_emotes(now)

    @_locked
    def recount_moment_emote(self, window_id: str) -> None:
        """Count a window's tally again from the text it still holds (after a deletion check dropped
        messages): update its row, or delete it when no emote is left. Never adds a row, and never
        changes the deadline. Runs inside a ``transaction()`` the caller already holds."""
        if not self.conn.in_transaction:
            raise RuntimeError("Store.recount_moment_emote runs inside Store.transaction()")
        t = top_emote(self.as_window(window_id))
        if t is None:
            self.conn.execute("DELETE FROM moment_emotes WHERE window_id = ?", (window_id,))
            return
        self.conn.execute(
            "UPDATE moment_emotes SET emote_id = ?, name = ?, lines = ?, of_lines = ?, people = ? WHERE window_id = ?",
            (t.emote_id, t.name, t.lines, t.of_lines, t.people, window_id),
        )

    def _fill_moment_emotes(self, now: datetime) -> int:
        now_s = iso(now)
        todo = self.conn.execute(
            "SELECT w.window_id, w.stream_id FROM windows w WHERE w.anchor_kind = 'moment'"
            " AND w.text_state = 'present' AND w.text_expires_at > ?"
            " AND NOT EXISTS (SELECT 1 FROM moment_emotes f WHERE f.window_id = w.window_id)"
            " AND NOT EXISTS (SELECT 1 FROM stream_activity_expired x WHERE x.stream_id = w.stream_id)"
            " AND EXISTS (SELECT 1 FROM messages m WHERE m.window_id = w.window_id AND m.emotes_json != '[]')"
            " ORDER BY w.window_id",
            (now_s,),
        ).fetchall()
        deadlines: dict[str, str] = {}
        written = 0
        for window_id, stream_id in todo:
            if stream_id not in deadlines:
                deadlines[stream_id] = self._faces_deadline(stream_id)
            deadline = deadlines[stream_id]
            if deadline <= now_s:
                continue
            t = top_emote(self.as_window(window_id))
            if t is None:
                continue
            self.conn.execute(
                "INSERT INTO moment_emotes (window_id, stream_id, emote_id, name, lines, of_lines, people, expires_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (window_id, stream_id, t.emote_id, t.name, t.lines, t.of_lines, t.people, deadline),
            )
            written += 1
        for stream_id, deadline in deadlines.items():  # one deadline per stream, only ever lowered
            if deadline > now_s:
                self.conn.execute(
                    "UPDATE moment_emotes SET expires_at = MIN(expires_at, ?) WHERE stream_id = ?", (deadline, stream_id)
                )
        return written

    def _faces_deadline(self, stream_id: str) -> str:
        """When the stream's faces expire: its activity deadline, or with no counts its earliest
        window start + 30 days; never later than a face of the stream already stored."""
        row = self.conn.execute("SELECT expires_at FROM stream_activity WHERE stream_id = ?", (stream_id,)).fetchone()
        if row is not None:
            deadline = row[0]
        else:
            first = self.conn.execute("SELECT MIN(start_at) FROM windows WHERE stream_id = ?", (stream_id,))
            deadline = iso(parse_dt(first.fetchone()[0], "windows.start_at") + ACTIVITY_LIFETIME)
        stored = self.conn.execute("SELECT MIN(expires_at) FROM moment_emotes WHERE stream_id = ?", (stream_id,))
        lowest = stored.fetchone()[0]
        return deadline if lowest is None else min(deadline, lowest)

    # --- emote names (Task 10b) -----------------------------------------------------

    @_locked
    def _backfill_emote_names(self) -> None:
        """Fill ``emote_names`` once from the text still held, for a store made before it existed."""
        if self.conn.execute("SELECT 1 FROM emote_names LIMIT 1").fetchone():
            return
        if not self.conn.execute("SELECT 1 FROM messages LIMIT 1").fetchone():
            return
        with self.transaction() as conn:
            rows = conn.execute(
                "SELECT w.channel, m.emotes_json FROM messages m JOIN windows w ON w.window_id = m.window_id"
            ).fetchall()
            names = sorted({(ch, e["id"], e["name"]) for ch, ej in rows for e in json.loads(ej)})
            conn.executemany(_EMOTE_NAME_UPSERT, names)

    # --- asks (Task 10b) -------------------------------------------------------------

    @_locked
    def add_ask(
        self,
        *,
        thread_id: str,
        created_at: datetime,
        expires_at: datetime,
        question: str | None,
        query_json: str,
        follow_up: str,
        result_window_ids: list[str],
    ) -> int:
        """Store one ask; returns its ID."""
        with self.transaction() as conn:
            cur = conn.execute(
                "INSERT INTO asks (thread_id, created_at, expires_at, question, query_json, follow_up,"
                " result_window_ids_json, summary) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)",
                (thread_id, iso(created_at), iso(expires_at), question, query_json, follow_up,
                 json.dumps(list(result_window_ids))),
            )  # fmt: skip
            return int(cur.lastrowid or 0)

    @_locked
    def update_ask(
        self,
        ask_id: int,
        *,
        question: str | None,
        query_json: str,
        follow_up: str,
        result_window_ids: list[str],
        expires_at: datetime,
        summary: str | None,
        summary_msgs: Sequence[Sequence[str]] = (),
    ) -> None:
        """Rewrite one ask after the local model read it or captioned it (Task 10c).

        ``secure_delete`` zeroes the bytes the old row held, so a refused question's words are gone.
        ``summary_msgs`` are the ``(window_id, message_id)`` lines the caption was written from: if a
        deletion check dropped any of them while the caption was being written, the caption is not
        stored (checked in this write's own transaction).
        """
        with self.transaction() as conn:
            if summary is not None and not _holds(conn, summary_msgs):
                summary = None
            conn.execute(
                "UPDATE asks SET question = ?, query_json = ?, follow_up = ?, result_window_ids_json = ?,"
                " expires_at = ?, summary = ? WHERE id = ?",
                (question, query_json, follow_up, json.dumps(list(result_window_ids)), iso(expires_at), summary,
                 ask_id),
            )  # fmt: skip

    @_locked
    def holds_messages(self, pairs: Sequence[Sequence[str]]) -> bool:
        """True when every ``(window_id, message_id)`` in ``pairs`` is still stored."""
        return _holds(self.conn, pairs)

    @_locked
    def asks_in_thread(self, thread_id: str, now: datetime) -> list[dict[str, Any]]:
        """The thread's asks that have not expired at ``now``, oldest first."""
        rows = self.conn.execute(
            f"SELECT {', '.join(_ASK_COLUMNS)} FROM asks WHERE thread_id = ? AND expires_at > ? ORDER BY id",
            (thread_id, iso(now)),
        ).fetchall()
        return [dict(zip(_ASK_COLUMNS, r)) for r in rows]

    @_locked
    def ask(self, ask_id: int, now: datetime) -> dict[str, Any] | None:
        """One ask that has not expired at ``now``, or None."""
        row = self.conn.execute(
            f"SELECT {', '.join(_ASK_COLUMNS)} FROM asks WHERE id = ? AND expires_at > ?", (ask_id, iso(now))
        ).fetchone()
        return dict(zip(_ASK_COLUMNS, row)) if row else None

    @_locked
    def recent_threads(self, now: datetime, limit: int = 20) -> list[tuple[str, str | None]]:
        """``(thread_id, first question)`` for threads with an ask live at ``now``, newest first."""
        rows = self.conn.execute(
            "SELECT thread_id, MAX(id) AS last FROM asks WHERE expires_at > ? GROUP BY thread_id"
            " ORDER BY last DESC LIMIT ?",
            (iso(now), limit),
        ).fetchall()
        out: list[tuple[str, str | None]] = []
        for thread_id, _ in rows:
            first = self.conn.execute(
                "SELECT question FROM asks WHERE thread_id = ? AND expires_at > ? AND question IS NOT NULL"
                " ORDER BY id LIMIT 1",
                (thread_id, iso(now)),
            ).fetchone()
            out.append((thread_id, first[0] if first else None))
        return out


_ASK_COLUMNS = (
    "id", "thread_id", "created_at", "expires_at", "question", "query_json", "follow_up",
    "result_window_ids_json", "summary",
)  # fmt: skip
_EMOTE_NAME_UPSERT = (
    "INSERT INTO emote_names (channel, emote_id, name) VALUES (?, ?, ?)"
    " ON CONFLICT (channel, emote_id) DO UPDATE SET name = excluded.name"
)
_ACTIVITY_SELECT = "a.stream_id, a.channel, a.start_at, a.end_at, a.title, a.bin_s, a.counts_json"
# A live row: not expired at the given time, and no marker (whatever that time is).
_ACTIVITY_LIVE = (
    "a.expires_at > ? AND NOT EXISTS (SELECT 1 FROM stream_activity_expired x WHERE x.stream_id = a.stream_id)"
)


def _activity(row: Sequence[Any]) -> StreamActivity:
    stream_id, channel, start_at, end_at, title, bin_s, counts_json = row
    return StreamActivity(
        stream_id=stream_id,
        channel=channel,
        started_at=parse_dt(start_at, "stream_activity.start_at"),
        ended_at=parse_dt(end_at, "stream_activity.end_at"),
        title=title,
        bin_s=bin_s,
        chatters=tuple(json.loads(counts_json)),
    )


def _emotes_json(emotes: tuple[Emote, ...]) -> str:
    return json.dumps([{"id": e.id, "name": e.name} for e in emotes], ensure_ascii=False, separators=(",", ":"))


def _holds(conn: sqlite3.Connection, pairs: Sequence[Sequence[str]]) -> bool:
    """True when every ``(window_id, message_id)`` in ``pairs`` is a stored message."""
    return all(
        conn.execute("SELECT 1 FROM messages WHERE window_id = ? AND message_id = ?", (w, m)).fetchone()
        for w, m in pairs
    )
