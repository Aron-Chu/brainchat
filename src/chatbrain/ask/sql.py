"""Read-only SQL for asking (ruling B14): one connection per answer, one template per intent.

The connection opens the store as ``file:<db>?mode=ro`` (URI) with ``PRAGMA query_only=ON``, then a
deny-by-default authorizer: plain reads of the approved tables and columns below, and the listed
functions. Everything else (ATTACH, PRAGMA, writes, DDL, any other table or column, any other
function) is refused and raises ``Denied``. A progress handler stops a statement after
``DEADLINE_S`` (``TooSlow``) and ``rows`` fetches at most a row cap (``ReadOnly.capped``).

SQL here is built by our code from a validated query object (``query.check``): keys pick fixed
fragments and every value is bound. The typed question never reaches this module; ``phrase``
(a word-for-word part of it, checked by the validator) is a bound ``LIKE`` pattern and reaches
only windows whose text still exists.

``Scope`` is the shared prefix: page ``shown`` (``page.shown.cte``), pivoted to one row per window,
joined with the window and its counts, narrowed by time, channel and kind, then ``keep`` (the
definite filters), ``hit`` (``keep`` and every value filter matched) and ``maybe`` (``keep``, no value
filter contradicted, one or more not sure). A not-sure answer never matches a value.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from datetime import time as dtime
from pathlib import Path
from typing import Any

from ..export_format import iso, parse_dt

DEADLINE_S = 2.0
ROW_CAP = 2000
STEPS = 1000  # virtual-machine steps between deadline checks
AT_SLACK = timedelta(minutes=5)  # "at 22:03" reaches windows within 5 minutes of 22:03
TIME_SPAN = {"24h": timedelta(hours=24), "7d": timedelta(days=7), "14d": timedelta(days=14)}

APPROVED: dict[str, frozenset[str]] = {
    "windows": frozenset(
        ("window_id", "channel", "stream_id", "anchor_kind", "anchor_at", "start_at", "end_at", "text_expires_at",
         "text_state")  # fmt: skip
    ),
    "features": frozenset(("window_id", "json")),
    "labels": frozenset(
        ("id", "window_id", "origin", "labels_version", "answers_json", "not_sure_json", "provenance", "cause_note",
         "created_at")  # fmt: skip
    ),
    "predictions": frozenset(("id", "window_id", "model_id", "question", "value_json", "confidence", "created_at")),
    "messages": frozenset(("window_id", "text")),
    "emote_names": frozenset(("channel", "emote_id", "name")),
    "json_each": frozenset(("key", "value", "type", "atom", "id", "parent", "fullkey", "path", "json", "root")),
}
FUNCTIONS = frozenset(
    ("json_extract", "json_type", "json_each", "json", "json_array_length", "->", "->>", "row_number", "coalesce",
     "count", "sum", "max", "min", "total", "avg", "abs", "julianday", "date", "like", "lower", "ifnull", "iif",
     "nullif", "length", "round")  # fmt: skip
)

# value key -> (scheme question, shown column)
VALUE_COLUMNS = {"doing": ("doing", "v_doing"), "aimed": ("aimed_at", "v_aimed"), "feels": ("feeling", "v_feels"),
                 "tone": ("tone", "v_tone")}  # fmt: skip
IS_SQL = {
    "not_sure": "k.n_unsure > 0",
    "needs_review": "k.review",
    "labeled": "k.labeled",
    "has_cause": "k.has_cause",
    "has_text": "k.has_text",
}
ORDER_SQL = {
    "posters": "posters DESC, anchor_at, window_id",
    "lines": "lines DESC, anchor_at, window_id",
    "rate": "rate DESC, anchor_at, window_id",
    "newest": "anchor_at DESC, window_id",
    "oldest": "anchor_at, window_id",
    "least_sure": "n_unsure DESC, COALESCE(min_conf, 2), posters DESC, anchor_at, window_id",
}


class Denied(Exception):
    """The read-only connection refused a statement."""


class TooSlow(Exception):
    """A statement ran past the deadline and was stopped."""


def _authorize(action: int, arg1: str | None, arg2: str | None, db: str | None, _trigger: str | None) -> int:
    if action == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ:
        cols = APPROVED.get(arg1 or "")
        if cols is not None and db in ("main", None) and (arg2 == "" or arg2 in cols):
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION:
        return sqlite3.SQLITE_OK if (arg2 or "").lower() in FUNCTIONS else sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_DENY


_DENIED_WORDS = ("not authorized", "no such function", "readonly", "read-only", "attempt to write", "prohibited")


class ReadOnly:
    """One read-only connection; ``rows`` runs one statement under the deadline and row cap."""

    def __init__(self, conn: sqlite3.Connection, deadline_s: float, cap: int) -> None:
        self.conn = conn
        self.deadline_s = deadline_s
        self.cap = cap
        self.capped = False
        self.statements = 0  # statements run (the answer's step line counts them)
        self._deadline = 0.0
        conn.set_progress_handler(self._progress, STEPS)

    def _progress(self) -> int:
        return 1 if time.monotonic() > self._deadline else 0

    def rows(self, sql: str, params: Sequence[Any], cap: int | None = None) -> list[tuple[Any, ...]]:
        """At most ``cap`` rows (default the connection's); ``capped`` says whether more were there."""
        limit = self.cap if cap is None else cap
        self.statements += 1
        self._deadline = time.monotonic() + self.deadline_s
        try:
            cur = self.conn.execute(sql, list(params))
            got = cur.fetchmany(limit + 1)
            cur.close()
        except sqlite3.DatabaseError as e:
            msg = str(e).lower()
            if "interrupted" in msg:
                raise TooSlow(f"stopped after {self.deadline_s:g} s") from None
            if any(w in msg for w in _DENIED_WORDS):
                raise Denied(str(e)) from None
            raise
        self.capped = len(got) > limit
        return got[:limit]

    def one(self, sql: str, params: Sequence[Any]) -> tuple[Any, ...]:
        found = self.rows(sql, params, cap=1)
        return found[0] if found else ()


@contextmanager
def readonly(path: str | Path, *, deadline_s: float = DEADLINE_S, cap: int = ROW_CAP) -> Iterator[ReadOnly]:
    """A separate read-only connection to the store at ``path``; closed on exit."""
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=1.0, check_same_thread=False, isolation_level=None)
    try:
        conn.execute("PRAGMA query_only = ON")
        conn.execute("PRAGMA trusted_schema = OFF")
        conn.set_authorizer(_authorize)
        yield ReadOnly(conn, deadline_s, cap)
    finally:
        conn.close()


# --- the shared prefix -------------------------------------------------------------------------


@dataclass
class Scope:
    """``sql`` is a ``WITH`` prefix ending in ``scored``; append a ``SELECT ... FROM scored``."""

    sql: str
    params: list[Any]
    stream_id: str | None = None  # the stream "last stream" means, when the query says so
    instants: list[datetime] = field(default_factory=list)  # "at": the clock time on each date in reach
    period: bool = False  # ``cur`` marks the query's time range; the rest of the scope is "before"

    def rows(self, ro: ReadOnly, select: str, params: Sequence[Any] = (), cap: int | None = None) -> list[tuple]:
        return ro.rows(self.sql + "\n" + select, [*self.params, *params], cap)

    def one(self, ro: ReadOnly, select: str, params: Sequence[Any] = ()) -> tuple:
        found = self.rows(ro, select, params, cap=1)
        return found[0] if found else ()


def last_stream(ro: ReadOnly, channels: Sequence[str]) -> str | None:
    """The newest stream (by its last window's end) among ``channels`` (all when empty)."""
    where, params = ("WHERE channel IN (SELECT value FROM json_each(?))", [json.dumps(list(channels))]) if channels else ("", [])
    row = ro.one(
        f"SELECT stream_id FROM windows {where} GROUP BY stream_id ORDER BY MAX(end_at) DESC, stream_id LIMIT 1", params
    )
    return row[0] if row else None


def _clock(at: str) -> dtime:
    h, m = at.split(":")
    return dtime(int(h), int(m))


def instants(start: datetime, end: datetime, at: str) -> list[datetime]:
    """The UTC instants of local clock time ``at`` on every local date from ``start`` to ``end``."""
    first, last = start.astimezone().date(), end.astimezone().date()
    out: list[datetime] = []
    d: date = first - timedelta(days=1)
    while d <= last:
        local = datetime.combine(d, _clock(at)).astimezone()  # a naive time read as this PC's local time
        out.append(local.astimezone(timezone.utc))
        d += timedelta(days=1)
    return out


def scope(
    ro: ReadOnly,
    shown_cte: tuple[str, list[Any]],
    query: dict[str, Any],
    now: datetime,
    *,
    window_id: str | None = None,
    period: bool = False,
    moments_only: bool = True,
    use_at: bool = True,
) -> Scope:
    """The prefix for ``query`` at ``now``. ``window_id`` narrows it to that one window (any kind).

    ``period`` keeps windows outside the query's time range, marking the range with ``cur``.
    ``use_at`` False leaves ``at`` out of ``keep`` (``instants`` are still worked out), for the
    closest-Moment answers that rank by distance instead.
    """
    cte_sql, cte_params = shown_cte
    now_s = iso(now)
    channels = list(query.get("channel") or [])

    # where the windows come from: one window, or kind + channel + time
    where: list[str] = []
    where_p: list[Any] = []
    time_sql, time_p = "1", []
    stream_id = None
    if window_id is not None:
        where.append("w.window_id = ?")
        where_p.append(window_id)
    else:
        if moments_only:
            where.append("w.anchor_kind = 'moment'")
        if channels:
            where.append("w.channel IN (SELECT value FROM json_each(?))")
            where_p.append(json.dumps(channels))
        t = query.get("time", "last_stream")
        if t == "last_stream":
            stream_id = last_stream(ro, channels)
            time_sql, time_p = ("w.stream_id = ?", [stream_id]) if stream_id else ("0", [])
        elif t in TIME_SPAN:
            time_sql, time_p = "w.anchor_at >= ?", [iso(now - TIME_SPAN[t])]
        if not period:
            where.append(time_sql)
            where_p += time_p

    # "at": the clock time on each local date the scope covers, give or take AT_SLACK
    found: list[datetime] = []
    at = query.get("at")
    if at:
        span = ro.one(
            "SELECT MIN(w.start_at), MAX(w.end_at) FROM windows w WHERE " + " AND ".join(where or ["1"]), where_p
        )
        if span and span[0]:
            found = instants(parse_dt(span[0], "start_at"), parse_dt(span[1], "end_at"), at)

    sel_p: list[Any] = [now_s]
    match_cols = []
    for key, (_, col) in VALUE_COLUMNS.items():
        values = query.get(key) or []
        if values:
            match_cols.append(
                f"CASE WHEN sv.{col} IS NULL THEN NULL ELSE EXISTS (SELECT 1 FROM json_each(sv.{col}) j"
                f" WHERE j.value IN (SELECT value FROM json_each(?))) END AS m_{key}"
            )
            sel_p.append(json.dumps(values))
        else:
            match_cols.append(f"1 AS m_{key}")
    emotes = list(query.get("emote") or [])
    if emotes:
        uses = (
            "COALESCE((SELECT SUM(e.value) FROM json_each(f.json, '$.emote_counts') e JOIN emote_names n"
            " ON n.emote_id = e.key AND n.channel = w.channel WHERE n.name IN (SELECT value FROM json_each(?))), 0)"
        )
        sel_p.append(json.dumps(emotes))
    else:
        uses = "0"
    phrase = query.get("phrase")
    if phrase:
        said = (
            "(w.text_state = 'present' AND w.text_expires_at > ? AND EXISTS (SELECT 1 FROM messages m"
            " WHERE m.window_id = w.window_id AND lower(m.text) LIKE ? ESCAPE '!'))"
        )
        sel_p += [now_s, like_pattern(phrase)]
    else:
        said = "1"
    if at and use_at:
        near = (
            "EXISTS (SELECT 1 FROM json_each(?) r WHERE w.start_at <= json_extract(r.value, '$[1]')"
            " AND w.end_at >= json_extract(r.value, '$[0]'))"
        )
        sel_p.append(json.dumps([[iso(x - AT_SLACK), iso(x + AT_SLACK)] for x in found]))
    else:
        near = "1"
    cur = time_sql if period else "1"
    if period:
        sel_p += time_p

    keep: list[str] = ["k.said", "k.near"]
    keep_p: list[Any] = []
    if query.get("min_posters"):
        keep.append("k.posters >= ?")
        keep_p.append(int(query["min_posters"]))
    if query.get("spread"):
        keep.append("k.spread IN (SELECT value FROM json_each(?))")
        keep_p.append(json.dumps(list(query["spread"])))
    for v in query.get("is") or []:
        keep.append(IS_SQL[v])
    if emotes:
        keep.append("k.uses > 0")

    filtered = [k for k in VALUE_COLUMNS if query.get(k)]
    all_match = " AND ".join(f"COALESCE(k.m_{k}, 0)" for k in filtered) or "1"
    no_clash = " AND ".join(f"(k.m_{k} IS NULL OR k.m_{k})" for k in filtered) or "1"
    any_null = " OR ".join(f"k.m_{k} IS NULL" for k in filtered) or "0"
    q_cols = {col: q for q, col in VALUE_COLUMNS.values()}

    sql = f"""{cte_sql},
sv AS (
    SELECT window_id,
           {", ".join(f"MAX(CASE WHEN question = '{q}' THEN value_json END) AS {col}" for col, q in q_cols.items())},
           SUM(value_json IS NULL) AS n_unsure,
           MIN(CASE WHEN kind = 'model' AND value_json IS NOT NULL THEN confidence END) AS min_conf,
           MAX(kind = 'label') AS labeled,
           MAX(COALESCE(reason = 'needs_review', 0)
               OR (kind = 'label' AND value_json IS NULL
                   AND EXISTS (SELECT 1 FROM json_each(reasons_json) x WHERE x.value = 'needs_review'))) AS review
    FROM shown GROUP BY window_id
),
base AS (
    SELECT w.window_id, w.channel, w.stream_id, w.anchor_kind, w.anchor_at, w.start_at, w.end_at,
           w.text_expires_at, (w.text_state = 'present' AND w.text_expires_at > ?) AS has_text,
           sv.v_doing, sv.v_aimed, sv.v_feels, sv.v_tone, {", ".join(match_cols)},
           sv.n_unsure, sv.min_conf, sv.labeled, sv.review,
           COALESCE(json_extract(f.json, '$.joining_posters'), 0) AS posters,
           COALESCE(json_extract(f.json, '$.joining_lines'), 0) AS lines,
           COALESCE(json_extract(f.json, '$.msg_per_min'), 0) AS rate,
           json_extract(f.json, '$.spread') AS spread,
           EXISTS (SELECT 1 FROM labels l WHERE l.window_id = w.window_id AND l.origin = 'aron'
                   AND l.cause_note IS NOT NULL) AS has_cause,
           {uses} AS uses, {said} AS said, {near} AS near, {cur} AS cur
    FROM windows w JOIN sv ON sv.window_id = w.window_id LEFT JOIN features f ON f.window_id = w.window_id
    WHERE {" AND ".join(where or ["1"])}
),
kept AS (SELECT k.*, ({" AND ".join(keep)}) AS keep FROM base k),
scored AS (
    SELECT k.*, (k.keep AND {all_match}) AS hit,
           (k.keep AND NOT ({all_match}) AND {no_clash} AND ({any_null})) AS maybe
    FROM kept k
)"""
    return Scope(sql, [*cte_params, *sel_p, *where_p, *keep_p], stream_id, found, period)


def like_pattern(phrase: str) -> str:
    """``%phrase%`` for ``LIKE ... ESCAPE '!'``, lower case, with ``%``, ``_`` and ``!`` escaped."""
    s = phrase.lower().replace("!", "!!").replace("%", "!%").replace("_", "!_")
    return f"%{s}%"


# --- one template per intent -------------------------------------------------------------------


@dataclass
class Found:
    """What a list query found: ``ids`` in order, the scope, the matches and the not-sure ones."""

    ids: list[str]
    scope_n: int
    total: int
    maybe: int


def found(ro: ReadOnly, sc: Scope, sort: str, limit: int, *, only_cur: bool = False) -> Found:
    """find_moments, top_moments: matches sorted by ``sort`` (a count for top), then the counts.

    ``limit`` 0 reads the counts only. ``only_cur`` keeps a period scope's own range.
    """
    ids: list[str] = []
    if limit > 0:
        where = "hit AND cur" if only_cur else "hit"
        rows = sc.rows(ro, f"SELECT window_id FROM scored WHERE {where} ORDER BY {ORDER_SQL[sort]} LIMIT ?", [limit])
        ids = [r[0] for r in rows]
    return Found(ids, *_tally(ro, sc))


def first_anchor(ro: ReadOnly, sc: Scope) -> datetime | None:
    """The earliest anchor in the scope (the day a default reading names)."""
    row = sc.one(ro, "SELECT MIN(anchor_at) FROM scored")
    return parse_dt(row[0], "anchor_at") if row and row[0] else None


def _tally(ro: ReadOnly, sc: Scope) -> tuple[int, int, int]:
    row = sc.one(ro, "SELECT COUNT(*), COALESCE(SUM(hit), 0), COALESCE(SUM(maybe), 0) FROM scored")
    return (int(row[0] or 0), int(row[1] or 0), int(row[2] or 0)) if row else (0, 0, 0)


def anchors(ro: ReadOnly, sc: Scope) -> list[tuple[str, datetime]]:
    """Every match's anchor, oldest first (for the closest-Moment answers)."""
    rows = sc.rows(ro, "SELECT window_id, anchor_at FROM scored WHERE hit ORDER BY anchor_at, window_id")
    return [(w, parse_dt(a, "anchor_at")) for w, a in rows]


def nearest(ro: ReadOnly, sc: Scope, n: int) -> list[str]:
    """Up to ``n`` matches nearest ``sc.instants`` by anchor (posters joining breaks ties)."""
    if not sc.instants:
        return []
    got = anchors(ro, sc)
    return [w for w, a in sorted(got, key=lambda x: min(abs((x[1] - t).total_seconds()) for t in sc.instants))[:n]]


def by_value(ro: ReadOnly, sc: Scope, key: str) -> tuple[dict[str, int], int, int, int, int]:
    """count by value: ``({value: k}, n, not sure, maybe, scope)`` over the matches."""
    col = VALUE_COLUMNS[key][1]
    rows = sc.rows(ro, f"SELECT j.value, COUNT(*) FROM scored s, json_each(s.{col}) j WHERE s.hit GROUP BY j.value")
    tally = sc.one(
        ro,
        f"SELECT COALESCE(SUM(hit), 0), COALESCE(SUM(hit AND {col} IS NULL), 0), COALESCE(SUM(maybe), 0), COUNT(*)"
        " FROM scored",
    )
    n, unsure, maybe, scope_n = (int(x or 0) for x in tally) if tally else (0, 0, 0, 0)
    return {str(v): int(c) for v, c in rows}, n, unsure, maybe, scope_n


GROUP_SQL = {
    "channel": "channel",
    "stream": "stream_id",
    "day": "date(anchor_at, 'localtime')",
    "period": "cur",
}


@dataclass
class Group:
    key: Any
    n: int  # windows in the group's scope
    k: int  # matches
    maybe: int
    first: datetime  # its earliest anchor


def groups(ro: ReadOnly, sc: Scope, by: str) -> list[Group]:
    """count by channel, stream, day or period, and trend: per group, matches of all; oldest first."""
    rows = sc.rows(
        ro,
        f"SELECT {GROUP_SQL[by]} AS g, COUNT(*), COALESCE(SUM(hit), 0), COALESCE(SUM(maybe), 0), MIN(anchor_at)"
        " FROM scored GROUP BY g ORDER BY MIN(anchor_at), g",
    )
    return [Group(g, int(n), int(k), int(m), parse_dt(a, "anchor_at")) for g, n, k, m, a in rows]


@dataclass
class EmoteMix:
    n: int  # windows with the emote
    moments: int
    expired: int  # of those, windows whose text has gone (their counts remain)
    mix: dict[str, dict[str, int]]  # question -> {value: windows}
    unsure: dict[str, int]  # question -> windows not sure on it
    ids: list[str]  # examples, most uses first


def emote_mix(ro: ReadOnly, sc: Scope, examples: int = 3) -> EmoteMix:
    """emote_meaning: the windows with the emote, the mix of their shown answers, and examples."""
    keys = ("doing", "feels", "tone")
    row = sc.one(
        ro,
        "SELECT COUNT(*), COALESCE(SUM(anchor_kind = 'moment'), 0), COALESCE(SUM(NOT has_text), 0), "
        + ", ".join(f"COALESCE(SUM({VALUE_COLUMNS[k][1]} IS NULL), 0)" for k in keys)
        + " FROM scored WHERE hit",
    )
    got = [int(x or 0) for x in row] if row else [0] * (3 + len(keys))
    n, moments, expired = got[:3]
    unsure = dict(zip(keys, got[3:]))
    mix: dict[str, dict[str, int]] = {}
    for key in keys:
        col = VALUE_COLUMNS[key][1]
        rows = sc.rows(ro, f"SELECT j.value, COUNT(*) FROM scored s, json_each(s.{col}) j WHERE s.hit GROUP BY j.value")
        mix[key] = {str(v): int(c) for v, c in rows}
    ids = [r[0] for r in sc.rows(ro, "SELECT window_id FROM scored WHERE hit"
                                     " ORDER BY uses DESC, posters DESC, anchor_at, window_id LIMIT ?", [examples])]  # fmt: skip
    return EmoteMix(n, moments, expired, mix, unsure, ids)


def review_count(ro: ReadOnly, sc: Scope) -> int:
    row = sc.one(ro, "SELECT COALESCE(SUM(review), 0) FROM scored")
    return int(row[0] or 0) if row else 0
