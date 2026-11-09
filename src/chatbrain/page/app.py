"""The local Moment page: one stream's glance (its activity line and busiest three), and one Moment.

FastAPI with Jinja templates, served by uvicorn on ``127.0.0.1:14612`` only (``serve``
refuses any other host, and the app answers only to the Host names ``127.0.0.1`` and
``localhost``). It makes no outbound requests: fonts, CSS and JS come from ``static/``.

Routes:

- ``GET /?stream=<stream_id>&thread=<id>``: the glance of one stream (default: the newest, by
  its last window's end): the stream button and its menu (other streams, theme, keys), the
  eyebrow and status line, the activity line (people chatting per 10 s) with the lane of
  numbered Moments, and the busiest three cards (the first three when the counts are off,
  ``glance``); then the answers thread (each ask re-runs its stored query), the ask box with the
  suggestions, and "Asked before". ``doing`` is still checked (a bad value is refused) but no
  longer filters the page;
- ``GET /m/<window_id>``: the Moment as a full page; ``?panel=1`` returns the side lane's
  partial (Task 4, G1): no chart, window meta and id, whole chat line list or per-Moment ask, which
  the full page keeps (G-14);
- ``GET /list?stream=<stream_id>&order=busy|time``: all the stream's Moments, busiest first (G-3)
  or in time order; with the counts off only in time order. ``?panel=1`` returns the lane's partial;
- ``POST /m/<window_id>/fix``: ``question``, then ``value`` (repeatable for doing) or
  ``not_sure`` (repeatable reasons), writes one label row (project ``page``, origin ``aron``,
  provenance ``corrected-on-page``) and redirects back (303). Choosing the value already
  shown saves it too (B29);
- ``POST /m/<window_id>/cause``: ``cause_note`` (1-300 characters once trimmed) and
  ``cause_source`` write a row carrying only the cause note;
- ``POST /ask``: ``q`` (1-300 characters), optional ``thread_id`` and ``window_id`` (a
  per-Moment ask) parses, runs and stores one ask (``chatbrain.ask``), then redirects (303)
  to ``/?thread=<id>#a<ask_id>``; the URL never holds the question;
- ``POST /ask/<ask_id>/edit``: ``tok`` (repeatable), ``drop`` and ``add`` store a chip edit
  or a follow-up as a ``refine`` ask in the same thread;
- ``POST /ask/<ask_id>/model``: runs the one local model step the ask waits on (read the
  question, or write its caption; ``ask.model``) and returns the ask's answer block for the page
  script to swap in;
- ``POST /ask/<ask_id>/stop``: Stop (204): the ask keeps its keyword or template answer.

The label writes refuse (409) while the deletion check is stale
(``ingest.require_fresh_deletion_check``), and every POST that another site sent is refused (403).

What the page shows comes from ``shown``. Chat text shows only while ``text_state`` is
``present`` and ``text_expires_at`` is after the request's clock, so no line renders past
its expiry between the hourly runs (C19). At page start and then hourly the page runs
``expiry.run`` (no Label Studio client: due task deletions stay pending for the next
command) and then saves rung-0 predictions for windows that lack them. These, the two
label writes and the ``asks`` rows are the page's only writes (B24, B31). Asking reads the
store through its own read-only connection (``ask.sql``, B14).

Faces (F1, Task 5): a Moment's top emote typed (``store.moment_faces``, a count with no text or
speaker) shows as a text chip on its card, list row and lane chip, and as one line in the opened
Moment, only while the counts are live, when it passes ``faces.shows`` and the Moment's Doing is
sure. There is no picture (``faces.face_picture`` is the later stage's seam and is never drawn).

With a local model (``narrator``; Task 10c) the page also talks to Ollama on 127.0.0.1 only:
it preloads the model once when the server starts, and the dock shows the amber banner, with
its fix, only when asking falls back to the keyword path.

Every time shown is a local clock time (``datetime.astimezone()``, B28); chat lines and the
chart use offsets from the Moment's anchor.
"""

from __future__ import annotations

import json
import math
import sys
import threading
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote, urlencode

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .. import expiry, faces, ingest, paths
from ..export_format import StreamActivity, iso, parse_dt
from ..scheme import Question, Scheme, validate_answers, validate_cause_source
from ..store import ActivityState, Store
from . import activity as activity_mod
from . import shown as shown_mod
from .shown import Shown

HOST = "127.0.0.1"
PORT = 14612
SERVE_HOSTS = ("127.0.0.1", "localhost")
HOURLY_S = 3600.0
CAUSE_MAX = 300
LINES_ALL = 14  # a window with at most this many lines shows them all
LINES_HEAD = 12  # with no evidence line, the lines shown from the anchor on
EVIDENCE_MAX = 5
BIN_S = 5  # the chart's bar width in seconds
TICK_S = 30  # the chart's axis step in seconds
PAGE_DIR = Path(__file__).parent
REVIEW_EMOTES = paths.REPO_ROOT / "config" / "review_emotes.json"

QUESTION_LABEL = {"doing": "Doing", "aimed_at": "Aimed at", "feeling": "Feels", "tone": "Tone"}
CARD_LABEL = {"feeling": "feels", "tone": "tone"}
# What a reaction was aimed at, as it reads after a value or a time.
AIMED = {
    "streamer": "at the streamer",
    "game_content": "at the game",
    "guest": "at the guest",
    "chat_itself": "at chat itself",
    "tech_problem": "about a tech problem",
    "unclear": "with an unclear aim",
}
TITLE_AIM = {**AIMED, "unclear": "aim unclear"}
NOT_SURE_COLOR = "var(--cb-border-strong)"
# What the glance says in place of the line when a stream has no live counts (Q10's approved copy).
NO_ACTIVITY = {
    "missing": "This stream's export has no activity counts.",
    "expired": "Activity counts expired 30 days after this stream started.",
}
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; font-src 'self';"
        " connect-src 'self'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'; object-src 'none'"
    ),
    "Referrer-Policy": "same-origin",  # no referrer to other sites; same-origin posts keep their Origin
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}


# --- serving -----------------------------------------------------------------------------------------


def serve(app: FastAPI, host: str = HOST, port: int = PORT) -> None:
    """Run the page with uvicorn on this PC only; any host but 127.0.0.1 or localhost raises."""
    if host not in SERVE_HOSTS:
        raise ValueError(f"the page serves on {' or '.join(SERVE_HOSTS)} only")
    uvicorn.run(app, host=host, port=port, access_log=False)


def load_review_emotes(path: Path) -> frozenset[str]:
    """The emote IDs on the review list (``{"ids": [...]}``); a missing or malformed file raises."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = doc.get("ids") if isinstance(doc, dict) else None
    if not isinstance(ids, list) or not all(isinstance(x, str) and x for x in ids):
        raise ValueError(f'{Path(path).name}: needs "ids": a list of emote ID strings')
    return frozenset(ids)


# --- formatting --------------------------------------------------------------------------------------


def clock_text(dt: datetime) -> str:
    """A local clock time, ``20:01:07`` (B28)."""
    return f"{dt.astimezone():%H:%M:%S}"


def hm(dt: datetime) -> str:
    return f"{dt.astimezone():%H:%M}"


def day(dt: datetime) -> str:
    d = dt.astimezone()
    return f"{d:%b} {d.day}"


def offset(seconds: float) -> str:
    """Seconds from the anchor as ``−0:30`` (a true minus sign), ``0:00`` or ``+1:05``."""
    s = int(seconds)  # toward zero
    if s == 0:
        return "0:00"
    sign = "−" if s < 0 else "+"
    return f"{sign}{abs(s) // 60}:{abs(s) % 60:02d}"


def duration(seconds: float) -> str:
    minutes = max(0, int(round(seconds / 60)))
    return f"{minutes // 60} h {minutes % 60} m" if minutes >= 60 else f"{minutes} m"


def doing_color(value: str | None) -> str:
    if value is None:
        return NOT_SURE_COLOR
    return "var(--cb-doing-usual)" if value == "chatting_as_usual" else f"var(--cb-doing-{value})"


def pct(x: float) -> str:
    return f"{max(0.0, min(1.0, x)) * 100:.0f}%"


def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def url(path: str, **params: str | None) -> str:
    """A same-site link: ``path`` plus the params that are set."""
    q = urlencode({k: v for k, v in params.items() if v})
    return f"{path}?{q}" if q else path


def moment_url(window_id: str, **params: str | None) -> str:
    return url(f"/m/{quote(window_id, safe='')}", **params)


# --- what the page reads -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Win:
    window_id: str
    channel: str
    stream_id: str
    kind: str  # "moment" or "ordinary"
    anchor: datetime
    start: datetime
    end: datetime
    expires: datetime
    state: str
    counts: dict[str, Any]

    def visible(self, now: datetime) -> bool:
        """Chat text shows only while it is present and its time isn't up (C19)."""
        return self.state == "present" and self.expires > now

    def count(self, key: str) -> int:
        value = self.counts.get(key)
        return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


@dataclass(frozen=True)
class Stream:
    stream_id: str
    channel: str
    start: datetime
    end: datetime
    moments: int
    keeps_until: datetime | None  # the earliest expiry among windows whose text still shows
    expired_on: datetime  # the latest expiry: when the last text went or goes


@dataclass(frozen=True)
class Cause:
    note: str
    source: str


@dataclass
class Moment:
    """One window as a card or the panel shows it."""

    win: Win
    answers: dict[str, Shown]
    title: str
    visible: bool
    needs_review: bool
    fixed: bool
    cause: Cause | None
    # the panel's extras (``Page.fill_panel``)
    lines: list[dict[str, Any]] = field(default_factory=list)
    chart: dict[str, Any] | None = None
    evidence: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    # its number among the stream's Moments and its chip's shape, and the people chatting at its
    # peak (``_number``); ``peak_known`` is False only when the stream has live counts and they
    # give no peak at the anchor. ``peak_why`` then says why: ``"gap"`` (the anchor's bin is null,
    # "not recorded") or ``"unknown"`` (the anchor lies outside the counts, "coverage unknown").
    # It is ``""`` with a known peak or with no live counts (missing or expired).
    n: int | None = None
    shape: str | None = None
    peak: int | None = None
    peak_known: bool = True
    peak_why: Literal["", "gap", "unknown"] = ""
    # F1 (Task 5): the Moment's live top-emote tally (``_faces``), below the bar too; None without one
    face: faces.Tally | None = None

    @property
    def answered(self) -> int:
        return sum(1 for s in self.answers.values() if s.sure)

    @property
    def first_doing(self) -> str | None:
        d = self.answers.get("doing")
        return d.values[0] if d is not None and d.values else None

    @property
    def face_shown(self) -> bool:
        """The face shows only for a live tally that passes the bar (call 3) on a Moment whose Doing
        is sure: a not-sure Moment shows no face anywhere (G-13), and a weaker emote never stands in."""
        return self.face is not None and faces.shows(self.face) and self.first_doing is not None

    @property
    def face_rest(self) -> str:
        """The face sentence after the emote's name (the name is set apart, so it can be bold)."""
        t = self.face
        if t is None:
            return ""
        return f" · in {t.lines} of {t.of_lines} lines · sent by {t.people} people · the 120 s around {hm(self.win.anchor)}"

    @property
    def face_sentence(self) -> str:
        """``{name} · in {lines} of {of_lines} lines · sent by {people} people · the 120 s around {HH:MM}``."""
        return f"{self.face.name}{self.face_rest}" if self.face is not None else ""

    @property
    def chat(self) -> bool:
        """More than one message: the window has chat to show (else "No chat in this window.")."""
        return self.win.count("n_messages") > 1


_WIN_SQL = (
    "SELECT w.window_id, w.channel, w.stream_id, w.anchor_kind, w.anchor_at, w.start_at, w.end_at,"
    " w.text_expires_at, w.text_state, f.json FROM windows w LEFT JOIN features f ON f.window_id = w.window_id"
)


def _win(row: tuple) -> Win:
    wid, channel, stream, kind, anchor, start, end, expires, state, counts = row
    loaded = json.loads(counts) if counts else {}
    return Win(
        window_id=wid,
        channel=channel,
        stream_id=stream,
        kind=kind,
        anchor=parse_dt(anchor, "anchor_at"),
        start=parse_dt(start, "start_at"),
        end=parse_dt(end, "end_at"),
        expires=parse_dt(expires, "text_expires_at"),
        state=state,
        counts=loaded if isinstance(loaded, dict) else {},
    )


class Refused(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class Page:
    """The page's reads and writes over one store."""

    def __init__(
        self,
        store: Store,
        scheme: Scheme,
        models_dir: Path,
        *,
        imported_dir: Path,
        clock: Callable[[], datetime],
        review_ids: frozenset[str],
        narrator: Any = None,
        cpu_captions: bool = False,
    ) -> None:
        self.store = store
        self.scheme = scheme
        self.models_dir = Path(models_dir)
        self.imported_dir = Path(imported_dir)
        self.clock = clock
        self.review_ids = review_ids
        self.questions: dict[str, Question] = {q.key: q for q in scheme.questions}
        self._models: dict[str, Any] = {}
        self._models_lock = threading.Lock()
        # the local model (Task 10c): ``narrator.Narrator`` or None (``page --no-model``)
        self.narrator = narrator
        self.cpu_captions = cpu_captions  # captions even while the model runs on the CPU
        self.model_lock = threading.Lock()  # one model step (or the preload) at a time
        self.ask_lock = threading.Lock()  # a model step's write and Stop's write
        self.preloader: threading.Thread | None = None
        self._stops: dict[int, threading.Event] = {}
        self._stops_lock = threading.Lock()

    def stop_event(self, ask_id: int, *, create: bool = True) -> threading.Event | None:
        """The event Stop sets for a running model step on ``ask_id``."""
        with self._stops_lock:
            ev = self._stops.get(ask_id)
            if ev is None and create:
                ev = self._stops[ask_id] = threading.Event()
            return ev

    def drop_stop(self, ask_id: int, ev: threading.Event) -> None:
        with self._stops_lock:
            if self._stops.get(ask_id) is ev:
                del self._stops[ask_id]

    def now(self) -> datetime:
        now = self.clock()
        if now.utcoffset() is None:
            raise ValueError("the page clock must be timezone-aware")
        return now

    def plan(self) -> shown_mod.Plan:
        return shown_mod.plan(self.store, self.scheme, self.models_dir)

    # upkeep

    def maintain(self) -> None:
        """Expiry first, then the Moment tallies a store made before F1 lacks, then the rung-0
        predictions windows lack (C19)."""
        now = self.now()
        expiry.run(self.store, now, self.imported_dir, None, models_dir=self.models_dir)
        self.store.backfill_moment_emotes(now)
        shown_mod.backfill_rung0(self.store, self.scheme, self.models_dir, now)

    # reads

    def streams(self, now: datetime) -> list[Stream]:
        """Every stream, newest (by its last window's end) first."""
        sql = (
            "SELECT stream_id, MIN(channel), MIN(start_at), MAX(end_at), SUM(anchor_kind = 'moment'),"
            " MIN(CASE WHEN text_state = 'present' AND text_expires_at > ? THEN text_expires_at END),"
            " MAX(text_expires_at) FROM windows GROUP BY stream_id ORDER BY MAX(end_at) DESC, stream_id"
        )
        with self.store.lock:
            rows = self.store.conn.execute(sql, (iso(now),)).fetchall()
        return [
            Stream(
                stream_id=s, channel=ch, start=parse_dt(a, "start_at"), end=parse_dt(b, "end_at"),
                moments=int(n or 0), keeps_until=parse_dt(k, "text_expires_at") if k else None,
                expired_on=parse_dt(x, "text_expires_at"),
            )  # fmt: skip
            for s, ch, a, b, n, k, x in rows
        ]

    def stream_moments(self, stream_id: str) -> list[Win]:
        return self._wins("w.stream_id = ? AND w.anchor_kind = 'moment'", (stream_id,))

    # activity counts: every read takes the handler's one ``now`` (E5)

    def activity(self, stream_id: str, now: datetime) -> StreamActivity | None:
        """The stream's people-chatting counts, or None when it has none or they expired at ``now``."""
        return self.store.stream_activity(stream_id, now=now)

    def usual(self, channel: str, now: datetime) -> float | None:
        """The channel's usual pace from its live counts only, so an expired sibling never feeds it (E6)."""
        return activity_mod.usual_pace(self.store.channel_activity(channel, now=now))

    def activity_state(self, stream_id: str, now: datetime) -> ActivityState:
        return self.store.activity_state(stream_id, now=now)

    def counts(self, stream_id: str, now: datetime) -> tuple[ActivityState, StreamActivity | None]:
        """The stream's state and its live counts, read together under the store's lock so they agree."""
        with self.store.lock:
            state = self.activity_state(stream_id, now)
            return state, self.activity(stream_id, now) if state == "active" else None

    def window(self, window_id: str) -> Win | None:
        found = self._wins("w.window_id = ?", (window_id,))
        return found[0] if found else None

    def windows(self, ids: list[str]) -> list[Win]:
        """The stored windows among ``ids``, in that order."""
        if not ids:
            return []
        found = self._wins("w.window_id IN (SELECT value FROM json_each(?))", (json.dumps(ids),))
        by_id = {w.window_id: w for w in found}
        return [by_id[i] for i in ids if i in by_id]

    def _wins(self, where: str, params: tuple[Any, ...]) -> list[Win]:
        with self.store.lock:
            rows = self.store.conn.execute(f"{_WIN_SQL} WHERE {where} ORDER BY w.anchor_at, w.window_id", params)
            return [_win(r) for r in rows.fetchall()]

    def moments(self, wins: list[Win], now: datetime, p: shown_mod.Plan) -> list[Moment]:
        if not wins:
            return []
        ids = [w.window_id for w in wins]
        answers = shown_mod.read(self.store, self.scheme, p, ids, review_ids=self.review_ids)
        fixed = self._fixed(ids)
        causes = self._causes(ids)
        out = []
        for w in wins:
            a = answers.get(w.window_id, {})
            out.append(
                Moment(
                    win=w,
                    answers=a,
                    title=self.title(a),
                    visible=w.visible(now),
                    needs_review=any("needs_review" in s.reasons for s in a.values()),
                    fixed=w.window_id in fixed,
                    cause=causes.get(w.window_id),
                )
            )
        return out

    def title(self, answers: Mapping[str, Shown]) -> str:
        """"Laughing, shocked · at the guest": the doing values and the aim, never a cause."""
        d = answers.get("doing")
        a = answers.get("aimed_at")
        if d is not None and d.values:
            words = ", ".join(self.scheme.display.get(v, v) for v in d.values)
            first = words[:1].upper() + words[1:]
        else:
            first = "Not sure what chat was doing"
        aim = TITLE_AIM.get(a.values[0], a.values[0]) if a is not None and a.values else "aim not sure"
        return f"{first} · {aim}"

    def _fixed(self, ids: list[str]) -> set[str]:
        """Windows with a page label that names a question (a cause-only row doesn't count)."""
        sql = (
            "SELECT DISTINCT window_id FROM labels WHERE origin = 'aron' AND provenance = 'corrected-on-page'"
            " AND window_id IN (SELECT value FROM json_each(?))"
            " AND (EXISTS (SELECT 1 FROM json_each(answers_json)) OR EXISTS (SELECT 1 FROM json_each(not_sure_json)))"
        )
        with self.store.lock:
            return {r[0] for r in self.store.conn.execute(sql, (json.dumps(ids),)).fetchall()}

    def _causes(self, ids: list[str]) -> dict[str, Cause]:
        """The owner's latest cause note per window."""
        sql = (
            "SELECT window_id, cause_note, cause_source FROM (SELECT window_id, cause_note, cause_source,"
            " ROW_NUMBER() OVER (PARTITION BY window_id ORDER BY created_at DESC, id DESC) AS rn FROM labels"
            " WHERE origin = 'aron' AND cause_note IS NOT NULL AND window_id IN (SELECT value FROM json_each(?)))"
            " WHERE rn = 1"
        )
        with self.store.lock:
            rows = self.store.conn.execute(sql, (json.dumps(ids),)).fetchall()
        return {w: Cause(note, source or "") for w, note, source in rows}

    # the panel's extras

    def model(self, model_id: str) -> Any:
        """A loaded model for evidence lines, or None when its file won't load (cached either way)."""
        with self._models_lock:
            if model_id not in self._models:
                try:
                    self._models[model_id] = shown_mod.load_model(self.models_dir, model_id)
                except shown_mod.ModelFileError:
                    self._models[model_id] = None
            return self._models[model_id]

    def fill_panel(self, m: Moment, p: shown_mod.Plan, *, chart: bool = False) -> None:
        """Evidence and chat lines, and with ``chart`` the 5 s chart (only the full Moment page draws
        it). Text is read only while it shows."""
        if not m.visible or not m.chat:
            return
        window = self.store.as_window(m.win.window_id)
        msgs = list(window.messages)
        if len(msgs) <= 1:
            return
        anchor_ms = (m.win.anchor - m.win.start).total_seconds() * 1000
        backs: dict[str, list[str]] = {}  # message ID -> the answers its line backs
        by_id = {x.message_id: x for x in msgs}
        for q, s in m.answers.items():
            if not s.sure:
                continue
            model_id = s.model_id if s.kind == "model" else (p.active[q].model_id if q in p.active else None)
            model = self.model(model_id) if model_id else None
            if model is None:
                continue
            ids: list[str] = []
            for v in s.values:
                try:
                    found = _evidence(model, window, q, v)
                except (ValueError, KeyError):
                    continue  # the model doesn't answer this question or value
                for mid in found:
                    if mid not in ids:
                        ids.append(mid)
                    tags = backs.setdefault(mid, [])
                    tag = self.scheme.display.get(v, v)
                    if tag not in tags:
                        tags.append(tag)
            m.evidence[q] = [_line(by_id[i], anchor_ms) for i in ids[:EVIDENCE_MAX] if i in by_id]
        rows = [_line(x, anchor_ms, " · ".join(backs.get(x.message_id, ()))) for x in msgs]
        m.lines = _fold(rows, anchor_ms)
        if chart:
            m.chart = _chart(msgs, m.win, [x for x in msgs if x.message_id in backs])

    # writes

    def exists(self, window_id: str) -> bool:
        with self.store.lock:
            row = self.store.conn.execute("SELECT 1 FROM windows WHERE window_id = ?", (window_id,)).fetchone()
        return row is not None

    def fix(self, window_id: str, form: Mapping[str, list[Any]]) -> None:
        if not self.exists(window_id):
            raise Refused(404, "No such window.")
        questions = form.get("question", [])
        if len(questions) != 1 or questions[0] not in self.questions:
            raise Refused(400, "Not saved: name one question of the scheme.")
        q = self.questions[questions[0]]
        values, reasons = list(form.get("value", [])), list(form.get("not_sure", []))
        if not all(isinstance(x, str) for x in values + reasons):
            raise Refused(400, "Not saved: values must be text.")
        if bool(values) == bool(reasons):
            raise Refused(400, "Not saved: choose values or not-sure reasons (one of the two).")
        answers: dict[str, Any] = {}
        not_sure: dict[str, Any] = {}
        if values:
            if q.kind == "multi":
                answers[q.key] = _in_order(values, q.values)
            elif len(values) == 1:
                answers[q.key] = values[0]
            else:
                raise Refused(400, f"Not saved: {q.key} takes one value.")
        else:
            not_sure[q.key] = _in_order(reasons, self.scheme.not_sure_reasons)
        try:
            validate_answers(self.scheme, answers, not_sure)
        except ValueError as e:
            raise Refused(400, f"Not saved: {e}") from None
        self._write(window_id, answers=answers, not_sure=not_sure)

    def cause(self, window_id: str, form: Mapping[str, list[Any]]) -> None:
        if not self.exists(window_id):
            raise Refused(404, "No such window.")
        notes, sources = form.get("cause_note", []), form.get("cause_source", [])
        if len(notes) != 1 or len(sources) != 1 or not isinstance(notes[0], str) or not isinstance(sources[0], str):
            raise Refused(400, "Not saved: give one cause note and one source.")
        note = notes[0].strip()
        if not 1 <= len(note) <= CAUSE_MAX:
            raise Refused(400, f"Not saved: a cause note is 1 to {CAUSE_MAX} characters.")
        try:
            validate_cause_source(self.scheme, sources[0])
        except ValueError as e:
            raise Refused(400, f"Not saved: {e}") from None
        self._write(window_id, cause_note=note, cause_source=sources[0])

    def _write(
        self,
        window_id: str,
        *,
        answers: dict[str, Any] | None = None,
        not_sure: dict[str, Any] | None = None,
        cause_note: str | None = None,
        cause_source: str | None = None,
    ) -> None:
        now = self.now()
        try:
            ingest.require_fresh_deletion_check(self.store, now)
        except ingest.StaleDeletionCheck as e:
            raise Refused(
                409,
                f"Not saved: the deletion check is stale ({e}). Apply a fresh deletion check"
                " (uv run chatbrain deletions), then save again.",
            ) from None
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO labels (window_id, project, origin, provenance, labels_version, guide_version,"
                " answers_json, not_sure_json, cause_note, cause_source, created_at)"
                " VALUES (?, 'page', 'aron', 'corrected-on-page', ?, ?, ?, ?, ?, ?, ?)",
                (
                    window_id, self.scheme.version, self.scheme.guide_version, json.dumps(answers or {}),
                    json.dumps(not_sure or {}), cause_note, cause_source, iso(now),
                ),
            )  # fmt: skip


def _in_order(items: list[str], order: tuple[str, ...]) -> list[str]:
    """``items`` without repeats, in the scheme's order (unknown ones last, for the validator to refuse)."""
    rank = {v: i for i, v in enumerate(order)}
    return sorted(set(items), key=lambda v: (rank.get(v, len(rank)), v))


def _evidence(model: Any, window: Any, question: str, value: str) -> list[str]:
    from .. import train  # scikit-learn loads only when a model answers

    return train.evidence(model, window, question, value, EVIDENCE_MAX)


def _line(msg: Any, anchor_ms: float, tag: str = "") -> dict[str, Any]:
    return {"off": offset((msg.rel_ms - anchor_ms) / 1000), "rel_ms": msg.rel_ms, "msg": msg, "tag": tag}


def _fold(rows: list[dict[str, Any]], anchor_ms: float) -> list[dict[str, Any]]:
    """Lines to show, with the rest folded: all of them up to ``LINES_ALL``; else each evidence
    line with one line either side; else ``LINES_HEAD`` lines from the anchor on."""
    n = len(rows)
    if n <= LINES_ALL:
        keep = set(range(n))
    else:
        tagged = [i for i, r in enumerate(rows) if r["tag"]]
        if tagged:
            keep = {j for i in tagged for j in (i - 1, i, i + 1) if 0 <= j < n}
        else:
            first = next((i for i, r in enumerate(rows) if r["rel_ms"] >= anchor_ms), n)
            first = min(first, n - LINES_HEAD)
            keep = set(range(first, first + LINES_HEAD))
    out: list[dict[str, Any]] = []
    run: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        if i in keep:
            if run:
                out.append({"fold": run, "first": run[0]["off"], "last": run[-1]["off"]})
                run = []
            out.append(r)
        else:
            run.append(r)
    if run:
        out.append({"fold": run, "first": run[0]["off"], "last": run[-1]["off"]})
    return out


def _chart(msgs: list[Any], w: Win, evidence: list[Any]) -> dict[str, Any]:
    """Messages per ``BIN_S`` seconds across the window. SVG units are seconds across, 46 down."""
    span = max(float(BIN_S), (w.end - w.start).total_seconds())
    n_bins = max(1, math.ceil(span / BIN_S))
    bins = [0] * n_bins
    for x in msgs:
        bins[min(n_bins - 1, max(0, int(x.rel_ms // (BIN_S * 1000))))] += 1
    peak = max(bins)
    width = n_bins * BIN_S
    anchor = (w.anchor - w.start).total_seconds()
    bars = [
        {"x": f"{i * BIN_S + 0.6:g}", "y": f"{42 - b / peak * 38:.1f}", "h": f"{b / peak * 38:.1f}"}
        for i, b in enumerate(bins)
        if b
    ]
    ticks = []
    t = math.ceil(-anchor / TICK_S) * TICK_S
    while anchor + t <= width + 1e-6:
        left = (anchor + t) / width * 100
        ticks.append({"left": f"{left:.1f}", "label": offset(t), "edge": "l" if left < 5 else "r" if left > 95 else ""})
        t += TICK_S
    marks = sorted({f"{x.rel_ms / 1000:.1f}" for x in evidence}, key=float)
    return {"width": width, "anchor": f"{anchor:g}", "bars": bars, "peak": peak, "ticks": ticks, "marks": marks}


def _chip_input(m: Moment) -> tuple[str, datetime, bool, str | None]:
    return (m.win.window_id, m.win.anchor, m.cause is not None, m.first_doing)


def _number(ms: list[Moment], wins: list[Win], act: StreamActivity | None) -> None:
    """Give each of ``ms`` its number, its chip's shape and its peak (C7: the home and Moment handlers share it).

    ``wins`` are all the stream's Moments, unfiltered, so a doing filter never changes a number:
    ``activity.chips`` numbers them in (anchor, window_id) order, the order the activity line's chips
    get. A window outside ``wins`` (an ordinary window opened by its link) is no Moment, so it gets
    no number or shape, only its peak. The peak is ``activity.peak_near`` on the stream's live
    counts ``act``; with none (never imported, or expired) a Moment has no peak and isn't "not
    recorded" or "coverage unknown" either. Without a peak under live counts, ``peak_why`` says
    whether the anchor's bin was not recorded (``"gap"``) or lies outside the counts
    (``"unknown"``)."""
    given = {w.window_id: (w.window_id, w.anchor, False, None) for w in wins}
    given |= {m.win.window_id: _chip_input(m) for m in ms if m.win.window_id in given}
    epoch = datetime.fromtimestamp(0, timezone.utc)  # the span only places chips; numbers and shapes ignore it
    numbered = {c.window_id: c for c in activity_mod.chips(list(given.values()), epoch, epoch)}
    for m in ms:
        chip = numbered.get(m.win.window_id)
        if chip is not None:
            m.n, m.shape = chip.n, chip.shape
        m.peak = activity_mod.peak_near(act, m.win.anchor) if act is not None else None
        m.peak_known = act is None or m.peak is not None
        if act is None or m.peak is not None:
            m.peak_why = ""
        else:
            # no peak with live counts: the anchor's bin is null or outside them ("known" can't
            # lose its peak, since a recorded anchor bin is always in peak_near's span)
            m.peak_why = "unknown" if activity_mod.coverage(act, m.win.anchor) == "unknown" else "gap"


def _faces(page: Page, ms: list[Moment], stream_id: str, state: ActivityState, now: datetime) -> None:
    """Give each of ``ms`` its live top-emote tally (F1): one read for the stream at the handler's
    ``now`` (``Store.moment_faces``: not past its deadline, no expired marker). The tallies live as
    long as the counts, so with the counts expired none is read, even before the sweep deletes them.
    Whether a face shows is ``Moment.face_shown``."""
    got = page.store.moment_faces(stream_id, now) if state != "expired" else {}
    for m in ms:
        m.face = got.get(m.win.window_id)


def _ticks(start: float, w: float) -> list[dict[str, str]]:
    """Axis ticks at local clock times over ``w`` seconds from ``start`` (a timestamp)."""
    step = next((s for s in (60, 300, 600, 900, 1800, 3600, 7200) if w / s <= 6), 10800)
    ticks = []
    t = math.ceil(start / step) * step
    while t <= start + w:
        left = (t - start) / w * 100
        label = hm(datetime.fromtimestamp(t, timezone.utc))
        ticks.append({"left": f"{left:.1f}", "label": label, "edge": "l" if left < 5 else "r" if left > 95 else ""})
        t += step
    return ticks


def _sec(v: float) -> str:
    """Seconds for an SVG number: whole when whole, else one decimal."""
    return f"{v:.1f}".rstrip("0").rstrip(".")


def _activity_line(
    stream: Stream, ms: list[Moment], act: StreamActivity | None, *, activity_state: ActivityState
) -> dict[str, Any]:
    """The glance's activity line: people chatting per 10 s (``viewBox`` 100 high, G1's scale), one
    numbered chip per Moment for the lane under it, its gaps and clock ticks.

    The line shows only for ``"active"`` counts; then the chips, gaps and ticks share one axis, the
    counts' span (``started_at`` to ``started_at + width_s``, C8) widened to reach every Moment's
    anchor. A Moment anchored outside the counts keeps its own place, and the part of the axis the
    counts don't cover is hatched like a null bin but titled "Coverage unknown": only a null bin is
    a known "No chat recorded" gap (R1). Otherwise the chips spread over the windows, with the
    approved line for the state the store gave (``NO_ACTIVITY``), never a guessed one.

    ``spans`` holds the gaps in seconds from ``axis_start`` (for the plot's spoken label);
    ``gaps_pct`` the same in % of the axis."""
    drawn = activity_mod.line(act, height=GL_H) if act is not None and activity_state == "active" else None
    lead = 0.0  # seconds the axis reaches back before the counts' first bin
    if act is not None and drawn is not None:
        start, w = act.started_at, float(drawn.width_s)
        anchors = [m.win.anchor for m in ms]
        axis_start = min([start, *anchors])
        axis_end = max([start + timedelta(seconds=w), *anchors])
        lead = (start - axis_start).total_seconds()
        axis_w = (axis_end - axis_start).total_seconds()
    else:
        start, w = stream.start, max(60.0, (stream.end - stream.start).total_seconds())
        axis_start, axis_w = start, w
    chips = activity_mod.chips([_chip_input(m) for m in ms], axis_start, axis_start + timedelta(seconds=axis_w))
    merged: list[tuple[float, float, str]] = []
    gaps_pct: list[tuple[float, float, str]] = []  # (x_pct, w_pct, kind), kind "gap" or "unknown"
    if drawn is not None:
        # null bins are known gaps ("No chat recorded"); the axis the counts don't cover is unknown
        # ("Coverage unknown", G-1). Touching spans merge only when they are the same kind.
        spans: list[tuple[float, float, str]] = [(lead + a, lead + b, "gap") for a, b in drawn.gaps]
        if lead > 0:
            spans.insert(0, (0.0, lead, "unknown"))
        if axis_w > lead + w:
            spans.append((lead + w, axis_w, "unknown"))
        for a, b, kind in spans:
            if merged and a <= merged[-1][1] and kind == merged[-1][2]:
                merged[-1] = (merged[-1][0], max(b, merged[-1][1]), kind)
            else:
                merged.append((a, b, kind))
        gaps_pct = [(round(a / axis_w * 100, 3), round((b - a) / axis_w * 100, 3), kind) for a, b, kind in merged]
    return {
        "line": drawn,
        "gaps_pct": gaps_pct,
        "spans": merged,
        "chips": chips,
        "lead": lead,
        "axis_start": axis_start,
        "axis_w": axis_w,
        "view_x": _sec(-lead if lead else 0.0),
        "view_w": _sec(axis_w),
        "ticks": _ticks(axis_start.timestamp(), axis_w),
        "note": NO_ACTIVITY.get(activity_state) if drawn is None else None,
    }


# --- the glance (home) ---------------------------------------------------------------------------------

GL_H = 100  # the glance line's viewBox height: y = 100 - sqrt(c) / sqrt(peak) * 96, as G1 draws it
GAP_WORDS = {"gap": "No chat recorded", "unknown": "Coverage unknown"}
GAP_EDGE = 20  # a gap label centred within 20% of either end hangs inward (about 54 px of a 390 px phone plot)
COUNT_WORDS = ("one", "two", "three")
LANE_LABEL = "Moments on the line. Arrow keys move between them."


@dataclass(frozen=True)
class LaneMark:
    """One Moment on the lane under the line: a numbered chip for the busiest three (``top``), a
    small mark for the rest; ``sure`` is False when the doing answer is not sure (a ring). A busiest
    chip whose Moment shows a face (F1) ends its ``label`` with the face sentence, and ``note`` holds
    the sentence as (name, rest) for the desk note its focus shows; small marks never have one."""

    n: int
    window_id: str
    x_pct: float
    top: bool
    sure: bool
    label: str
    note: tuple[str, str] | None = None


@dataclass
class Glance:
    """What the home page shows for one stream (Task 3): the eyebrow, the status line, the line with
    its lane, and the busiest (or first) three Moments as cards."""

    eyebrow: str
    heading: str
    status_bold: str
    status_tail: str
    top: list[Moment]
    counts_ok: bool
    gone: str | None
    lane: list[LaneMark]
    kept: str
    n: int
    kept_menu: str = ""
    made_up: bool = False
    line: Any = None
    view_x: str = "0"
    view_w: str = "60"
    gaps: list[dict[str, Any]] = field(default_factory=list)
    peaks: list[dict[str, Any]] = field(default_factory=list)
    ticks: list[dict[str, str]] = field(default_factory=list)
    plot_label: str = ""
    cards: list[dict[str, Any]] = field(default_factory=list)
    faced: bool = False  # some shown card has a face: the h2 gets " · top emotes typed" (F1, call 6)


def _peak_bin(a: StreamActivity, at: datetime, before_s: int = 30, after_s: int = 90) -> int | None:
    """The bin ``activity.peak_near`` reads its number from (the first largest), or None."""
    if activity_mod.peak_near(a, at, before_s, after_s) is None:
        return None
    off = (at - a.started_at).total_seconds()
    first = max(0, math.floor((off - before_s) / a.bin_s))
    last = min(len(a.chatters) - 1, math.ceil((off + after_s) / a.bin_s) - 1)
    best: int | None = None
    for i in range(first, last + 1):
        c = a.chatters[i]
        if c is not None and (best is None or c > a.chatters[best]):
            best = i
    return best


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def _quote(page: Page, m: Moment, p: shown_mod.Plan) -> dict[str, Any]:
    """The card's one chat line: the first evidence line (Doing first), else the first line from the
    anchor on; or why there is none."""
    if not m.visible:
        return {"expired": day(m.win.expires)}
    if not m.chat:
        return {"none": True}
    page.fill_panel(m, p)
    ev = m.evidence.get("doing") or next((v for v in m.evidence.values() if v), [])
    if ev:
        line = ev[0]
    else:
        rows = [r for x in m.lines for r in (x["fold"] if "fold" in x else [x])]
        if not rows:
            return {"none": True}
        anchor_ms = (m.win.anchor - m.win.start).total_seconds() * 1000
        line = next((r for r in rows if r["rel_ms"] >= anchor_ms), rows[-1])
    return {"speaker": line["msg"].speaker, "text": line["msg"].text}


def glance(page: Page, stream: Stream, now: datetime, p: shown_mod.Plan | None = None) -> Glance:
    """The glance for ``stream`` at ``now``. The busiest three sort by peak people in 10 s, then by
    time, with an unknown peak last (G-3); with the counts missing or expired they are the first
    three by time and the line gives way to the approved Q10 line."""
    p = p if p is not None else page.plan()
    ms = page.moments(page.stream_moments(stream.stream_id), now, p)
    state, act = page.counts(stream.stream_id, now)
    _number(ms, [m.win for m in ms], act)
    _faces(page, ms, stream.stream_id, state, now)
    n = len(ms)
    counts_ok = act is not None and state == "active"
    if counts_ok:
        top = sorted(ms, key=_busy)[:3]
    else:
        top = ms[:3]
    k = len(top)
    lead_word = "Busiest" if counts_ok else "First"
    unsure = sum(1 for m in ms if m.first_doing is None)
    if stream.keeps_until is not None:
        kept = f"chat text kept until {day(stream.keeps_until)}"
    else:
        kept = f"chat text expired {day(stream.expired_on)}"
    g = Glance(
        eyebrow=f"{lead_word} {k} of {n} Moments" if k else "",
        heading=f"{lead_word} {COUNT_WORDS[k - 1]}" if k else "",
        status_bold=f"Not sure about {unsure} of {n} Moments." if unsure else f"Sure about all {n} Moments.",
        status_tail=" No model yet." if p.origin is None else " Guessed on this PC.",
        top=top,
        counts_ok=counts_ok,
        gone=None if counts_ok else NO_ACTIVITY.get(state, NO_ACTIVITY["missing"]),
        lane=[],
        kept=kept,
        n=n,
        kept_menu=kept[:1].upper() + kept[1:],
        made_up=p.made_up,
        faced=any(m.face_shown for m in top),
    )
    if not n:
        return g
    strip = _activity_line(stream, ms, act, activity_state=state)
    top_ids = {m.win.window_id for m in top}
    by_id = {m.win.window_id: m for m in ms}
    for c in strip["chips"]:
        m = by_id[c.window_id]
        sure = m.first_doing is not None
        label = f"Moment {c.n}, {clock_text(m.win.anchor)}" + ("" if sure else ", not sure")
        is_top = c.window_id in top_ids
        note = None
        if is_top and m.face_shown:  # only a busiest chip names its face (F1-desk); the line keeps plain marks
            label += f". {m.face_sentence}"
            note = (m.face.name, m.face_rest)
        g.lane.append(LaneMark(c.n, c.window_id, round(c.x_pct, 2), is_top, sure, label, note))
    if counts_ok and strip["line"] is not None:
        drawn, axis_start, axis_w, lead = strip["line"], strip["axis_start"], strip["axis_w"], strip["lead"]
        g.line, g.view_x, g.view_w, g.ticks = drawn, strip["view_x"], strip["view_w"], strip["ticks"]
        for (left, width, kind), (a, b, _) in zip(strip["gaps_pct"], strip["spans"]):
            mid = round(left + width / 2, 3)
            said_at = clock_text if b - a < 60 else hm  # a gap under a minute keeps its seconds
            g.gaps.append({
                "left": left, "width": width, "kind": kind, "title": GAP_WORDS[kind], "mid": mid,
                # a label near either end of the axis hangs inward, so it never leaves the plot
                "edge": "l" if mid < GAP_EDGE else "r" if mid > 100 - GAP_EDGE else "",
                "from": said_at(axis_start + timedelta(seconds=a)), "to": said_at(axis_start + timedelta(seconds=b)),
            })  # fmt: skip
        for m in top:
            i = _peak_bin(act, m.win.anchor) if act is not None else None
            if i is None or m.peak is None:
                continue
            x = (lead + (i + 0.5) * act.bin_s) / axis_w * 100
            y = GL_H - math.sqrt(m.peak) / math.sqrt(drawn.peak) * (GL_H - 4) if drawn.peak else GL_H
            g.peaks.append({"n": m.n, "x": round(x, 2), "y": round(y, 1)})
        end = axis_start + timedelta(seconds=axis_w)
        said = [f"Activity line of people chatting per 10 s from {hm(axis_start)} to {hm(end)}."]
        times = [hm(m.win.anchor) for m in top if m.peak is not None]
        if len(times) == 1:
            said.append(f"The busiest peak is at {times[0]}.")
        elif times:
            said.append(f"The {COUNT_WORDS[len(times) - 1]} busiest peaks are at {_and(times)}.")
        said += [f"{x['title']} from {x['from']} to {x['to']}." for x in g.gaps]
        g.plot_label = " ".join(said)
    usual = page.usual(stream.channel, now) if counts_ok else None
    for m in top:
        g.cards.append({
            "m": m, "sure": m.first_doing is not None, "yours": _tag(m.answers["doing"]) == "your label",
            "stat": _stat(m, counts_ok, usual), "quote": _quote(page, m, p),
        })  # fmt: skip
    return g


def _stat(m: Moment, counts_ok: bool, usual: float | None) -> dict[str, Any] | None:
    """A card's or list row's stat: the peak and its pace, or why there is none; None with the counts off."""
    if not counts_ok:
        return None
    if m.peak is not None:
        return {"peak": m.peak, "ratio": activity_mod.ratio_text(m.peak / usual) if usual else None}
    return {"word": "coverage unknown" if m.peak_why == "unknown" else "not recorded"}


def _busy(m: Moment) -> tuple[Any, ...]:
    """G-3's order: the most people in 10 s first, then by time, an unknown peak last."""
    return (m.peak is None, -(m.peak or 0), m.win.anchor, m.win.window_id)


THREE = 3  # the opened Moment shows three chat lines


def _three(m: Moment) -> tuple[list[dict[str, Any]], int]:
    """The opened Moment's chat lines and how many the window has: the evidence lines first (up to
    three), filled with the lines from the anchor on, shown in time order."""
    rows = [r for x in m.lines for r in (x["fold"] if "fold" in x else [x])]
    if not rows:
        return [], 0
    anchor_ms = (m.win.anchor - m.win.start).total_seconds() * 1000
    keep = [i for i, r in enumerate(rows) if r["tag"]][:THREE]
    start = next((i for i, r in enumerate(rows) if r["rel_ms"] >= anchor_ms), len(rows))
    start = max(0, min(start, len(rows) - THREE))
    for i in list(range(start, len(rows))) + list(range(start - 1, -1, -1)):
        if len(keep) >= THREE:
            break
        if i not in keep:
            keep.append(i)
    return [rows[i] for i in sorted(keep)], len(rows)


def _tag(s: Shown) -> str | None:
    """The tag after a Moment's title: the owner's own label, a model's sure guess, or none (not sure)."""
    if not s.sure:
        return None
    if s.source == shown_mod.LABEL_SOURCE:
        return "your label"
    return "model guess" if s.kind == "model" else None


def _choices(page: Page, key: str, s: Shown) -> list[dict[str, Any]]:
    """The Fix menu's values for one question, each with the model's number when it gave that value."""
    q = page.questions[key]
    values = q.values + ((q.blank_value,) if q.blank_value else ())
    return [
        {
            "value": v,
            "label": page.scheme.display.get(v, v),
            "current": v in s.values,
            "model": shown_mod.conf_text(s.confidence) if v in s.model_values and s.confidence is not None else None,
        }
        for v in values
    ]


def _counted(m: Moment, scheme: Scheme) -> list[dict[str, Any]]:
    """The Counted rows: counts only, never words (the top emote is not named)."""
    c = m.win
    n = c.count("n_messages")
    posters = c.count("distinct_posters")
    spread = c.counts.get("spread")
    text = scheme.display.get(spread, spread) if isinstance(spread, str) else "none"
    share = c.counts.get("top_emote_share") if spread == "repeated_emotes" else c.counts.get("chant_share")
    if spread in ("repeated_emotes", "chant") and isinstance(share, (int, float)) and n:
        what = "top emote" if spread == "repeated_emotes" else "repeated line"
        text += f" · {what} in {round(share * n)} of {n} lines"
    joining = c.count("joining_lines")
    joined = c.count("joining_posters")
    return [
        {"label": "Spread", "text": text},
        {"label": "Lines joining", "text": f"{joining} of {n}", "bar": joining / n if n else 0.0},
        {"label": "Posters joining", "text": f"{joined} of {posters}", "bar": joined / posters if posters else 0.0},
        {"label": "Text", "text": f"kept until {day(c.expires)}" if m.visible else f"expired {day(c.expires)}"},
    ]


# --- the FastAPI app ---------------------------------------------------------------------------------


def create_app(
    store: Store,
    scheme: Scheme,
    models_dir: Path,
    *,
    imported_dir: Path | None = None,
    clock: Callable[[], datetime] | None = None,
    review_emotes: Path | None = None,
    hourly: bool = True,
    narrator: Any = None,
    cpu_captions: bool = False,
) -> FastAPI:
    """The page over ``store``. Page start (expiry, then the rung-0 backfill) runs here, before
    the app is returned; ``hourly`` starts the thread that repeats both every hour while it runs.

    ``narrator`` (``narrator.Narrator``) adds the local model steps to asking; None keeps the
    keyword path alone. Building the app asks Ollama nothing: the model is preloaded once, in the
    background, when the server starts. ``cpu_captions`` keeps captions on while the model runs
    on the CPU."""
    from .. import ask as ask_mod  # here, not at the top: ``ask`` builds on this module
    from ..ask import model as model_mod
    from ..ask import query as q_mod

    page = Page(
        store,
        scheme,
        models_dir,
        imported_dir=Path(imported_dir) if imported_dir is not None else Path(store.path).parent / "imported",
        clock=clock or (lambda: datetime.now(timezone.utc)),
        review_ids=load_review_emotes(review_emotes if review_emotes is not None else REVIEW_EMOTES),
        narrator=narrator,
        cpu_captions=cpu_captions,
    )
    page.maintain()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        stop = threading.Event()
        if hourly:
            threading.Thread(target=_hourly, args=(page, stop), name="chatbrain-page-hourly", daemon=True).start()
        if page.narrator is not None and page.preloader is None:  # once per app, not per start
            page.preloader = threading.Thread(target=_preload, args=(page,), name="chatbrain-page-preload", daemon=True)
            page.preloader.start()
        try:
            yield
        finally:
            stop.set()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.page = page
    env = Environment(
        loader=FileSystemLoader(PAGE_DIR / "templates"),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters.update(
        clock=clock_text, hm=hm, day=day, conf=shown_mod.conf_text, pct=pct, color=doing_color,
        ratio=activity_mod.ratio_text,
    )  # fmt: skip
    env.globals.update(
        scheme=scheme, display=scheme.display, QUESTION_LABEL=QUESTION_LABEL, CARD_LABEL=CARD_LABEL,
        CAUSE_MAX=CAUSE_MAX, url=url, moment_url=moment_url, plural=plural,
        offset=offset, duration=duration, QUESTION_MAX=ask_mod.QUESTION_MAX,
    )  # fmt: skip

    def render(name: str, status: int = 200, **ctx: Any) -> HTMLResponse:
        return HTMLResponse(env.get_template(name).render(**ctx), status_code=status)

    def refused(e: Refused) -> HTMLResponse:
        title = "Not found" if e.status == 404 else "Not saved" if e.status in (400, 409) else "Refused"
        return render("message.html", e.status, title=title, message=e.message)

    @app.middleware("http")
    async def guard(request: Request, call_next: Callable) -> Response:
        if request.method not in ("GET", "HEAD"):
            origin = request.headers.get("origin")
            site = request.headers.get("sec-fetch-site")
            host = request.headers.get("host", "")
            if origin == "null" and site == "same-origin":
                # A native form post can carry an opaque Origin; Sec-Fetch-Site comes from the browser,
                # which no page can forge, so it vouches for the post.
                origin = None
            if (origin is not None and origin != f"http://{host}") or site not in (None, "same-origin", "none"):
                return PlainTextResponse("Refused: the page takes changes only from itself.", status_code=403)
        response = await call_next(request)
        for k, v in SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        if not request.url.path.startswith("/static/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(SERVE_HOSTS))  # added last, so it runs first
    app.mount("/static", StaticFiles(directory=PAGE_DIR / "static"), name="static")

    @app.get("/favicon.ico")
    def favicon() -> Response:
        return Response(status_code=204)  # no icon: the browser stops asking, and nothing shows

    def doing_filter(doing: str | None) -> str | None:
        if doing and doing not in page.questions["doing"].values:
            raise Refused(400, "No such doing value.")
        return doing or None

    @app.get("/", response_class=HTMLResponse)
    def home(stream: str | None = None, doing: str | None = None, thread: str | None = None) -> HTMLResponse:
        now = page.now()
        try:
            doing = doing_filter(doing)
            if thread is not None and not ask_mod.valid_thread(thread):
                raise Refused(400, "That thread link is broken.")
        except Refused as e:
            return refused(e)
        streams = page.streams(now)
        if not streams:
            return render("home.html", streams=[], channels=[], current=None, made_up=False)
        asked = ask_mod.thread(page, thread) if thread else []
        current = streams[0] if stream is None else next((s for s in streams if s.stream_id == stream), None)
        if current is None:
            return refused(Refused(404, "No such stream."))
        # ``?doing=`` is still checked (a bad value is refused) but no longer filters the glance:
        # the busiest three are the stream's, whatever was looked for (Task 3 ruling).
        p = page.plan()
        return render(
            "home.html",
            streams=streams,
            channels=_by_channel(streams),
            current=current,
            made_up=p.made_up,
            g=glance(page, current, now, p),
            LANE_LABEL=LANE_LABEL,
            thread=asked,
            thread_id=thread if asked else None,
            suggestions=q_mod.suggestions(q_mod.held(store)),
            history=ask_mod.history(page),
            banner=ask_mod.BANNER,
            model=model_mod.dock(page),
        )

    @app.get("/m/{window_id}", response_class=HTMLResponse)
    def moment(window_id: str, panel: int = 0, doing: str | None = None, answer: str | None = None) -> HTMLResponse:
        now = page.now()
        try:
            doing = doing_filter(doing)
            if answer is not None and not ask_mod.valid_thread(answer):
                raise Refused(400, "That thread link is broken.")
        except Refused as e:
            return refused(e)
        win = page.window(window_id)
        if win is None:
            return refused(Refused(404, "No such window."))
        p = page.plan()
        (m,) = page.moments([win], now, p)
        page.fill_panel(m, p, chart=not panel)
        streams = page.streams(now)
        stream = next(s for s in streams if s.stream_id == win.stream_id)
        # previous and next among the stream's Moments, filtered like the list it was opened from
        wins = page.stream_moments(win.stream_id)
        if doing:
            ids = [x.win.window_id for x in page.moments(wins, now, p) if doing in x.answers["doing"].values]
        else:
            ids = [w.window_id for w in wins]
        pos = ids.index(window_id) if window_id in ids else None
        # its number among all the stream's Moments (never the filtered ones) and its peak, at this one ``now``
        state, act = page.counts(win.stream_id, now)
        _number([m], wins, act)
        _faces(page, [m], win.stream_id, state, now)  # an ordinary window has no tally, so no face
        three, of_lines = _three(m)
        ctx = dict(
            m=m,
            stream=stream,
            full=not panel,  # the full page keeps Slice 1's meta, chart, chat lines and ask (G-14)
            answer=answer if panel else None,  # opened from an answer in the lane: "‹ Answer" goes back (F3)
            tag=_tag(m.answers["doing"]),
            # with the counts off, the Q10 line takes the stat slot and joins Counted
            gone=None if act is not None else NO_ACTIVITY.get(state, NO_ACTIVITY["missing"]),
            three=three,
            of_lines=of_lines,
            n_all=len(wins),
            usual=page.usual(stream.channel, now) if act is not None else None,
            doing=doing,
            prev=ids[pos - 1] if pos is not None and pos > 0 else None,
            next=ids[pos + 1] if pos is not None and pos + 1 < len(ids) else None,
            made_up=p.made_up,
            choices={q: _choices(page, q, m.answers[q]) for q in page.questions},
            counted=_counted(m, scheme),
            reasons=[(r, scheme.display.get(r, r)) for r in scheme.not_sure_reasons],
            sources=[(s, scheme.display.get(s, s)) for s in scheme.cause_sources],
            multi={q.key for q in scheme.questions if q.kind == "multi"},
        )
        if panel:
            return render("_moment_panel.html", **ctx)
        return render("moment.html", streams=streams, channels=_by_channel(streams), current=stream, **ctx)

    @app.get("/list", response_class=HTMLResponse)
    def moment_list(stream: str, order: str | None = None, panel: int = 0) -> HTMLResponse:
        """All the stream's Moments (Task 4): busiest first (G-3) or in time order. With the counts
        missing or expired there is no peak to sort by, so the list is in time order with no toggle."""
        now = page.now()
        if order not in (None, "", "busy", "time"):
            return refused(Refused(400, "No such order."))
        streams = page.streams(now)
        current = next((s for s in streams if s.stream_id == stream), None)
        if current is None:
            return refused(Refused(404, "No such stream."))
        p = page.plan()
        ms = page.moments(page.stream_moments(stream), now, p)
        state, act = page.counts(stream, now)
        _number(ms, [m.win for m in ms], act)
        _faces(page, ms, stream, state, now)
        counts_ok = act is not None
        busy = counts_ok and order != "time"
        usual = page.usual(current.channel, now) if counts_ok else None
        rows = [
            {
                # the same tag rule as the opened Moment (``_tag``): no tag on a not-sure Doing
                "m": m, "sure": m.first_doing is not None,
                "yours": _tag(m.answers["doing"]) == "your label", "stat": _stat(m, counts_ok, usual),
            }  # fmt: skip
            for m in (sorted(ms, key=_busy) if busy else ms)
        ]
        # F1: a face column only when some row has a face; then every row keeps its cell, so they line up
        faced = any(m.face_shown for m in ms)
        ctx = dict(
            stream=current, rows=rows, n=len(ms), busy=busy, counts_ok=counts_ok, made_up=p.made_up, faced=faced
        )  # fmt: skip
        if panel:
            return render("_list_panel.html", **ctx)
        return render("list.html", streams=streams, channels=_by_channel(streams), current=current, **ctx)

    async def post(action: Callable[[str, Mapping[str, list[Any]]], None], window_id: str, request: Request) -> Response:
        form = await request.form()
        fields = {k: form.getlist(k) for k in set(form.keys())}
        try:
            await run_in_threadpool(action, window_id, fields)
        except Refused as e:
            return refused(e)
        return RedirectResponse(moment_url(window_id), status_code=303)

    @app.post("/m/{window_id}/fix")
    async def fix(window_id: str, request: Request) -> Response:
        return await post(page.fix, window_id, request)

    @app.post("/m/{window_id}/cause")
    async def cause(window_id: str, request: Request) -> Response:
        return await post(page.cause, window_id, request)

    def one(form: Any, key: str) -> str | None:
        values = form.getlist(key)
        if len(values) > 1 or any(not isinstance(v, str) for v in values):
            raise Refused(400, f"Not asked: one {key} at most.")
        return values[0] if values and values[0] != "" else None

    @app.post("/ask")
    async def ask_post(request: Request) -> Response:
        form = await request.form()
        try:
            q = one(form, "q")
            thread_id, window_id = one(form, "thread_id"), one(form, "window_id")
            thread_id, ask_id = await run_in_threadpool(
                ask_mod.submit, page, q or "", thread_id=thread_id, window_id=window_id
            )
        except Refused as e:
            return refused(e)
        return RedirectResponse(f"/?thread={thread_id}#a{ask_id}", status_code=303)

    @app.post("/ask/{ask_id}/edit")
    async def ask_edit(ask_id: int, request: Request) -> Response:
        form = await request.form()
        try:
            toks = form.getlist("tok")
            if any(not isinstance(t, str) for t in toks):
                raise Refused(400, "Not asked: chips are text.")
            drop, add = one(form, "drop"), one(form, "add")
            thread_id, new_id = await run_in_threadpool(ask_mod.edit, page, ask_id, list(toks), drop=drop, add=add)
        except Refused as e:
            return refused(e)
        return RedirectResponse(f"/?thread={thread_id}#a{new_id}", status_code=303)

    @app.post("/ask/{ask_id}/model", response_class=HTMLResponse)
    async def ask_model(ask_id: int) -> Response:
        try:
            await run_in_threadpool(model_mod.advance, page, ask_id)
            x = await run_in_threadpool(ask_mod.one, page, ask_id)
        except Refused as e:
            return refused(e)
        return render("_ask_one.html", x=x)

    @app.post("/ask/{ask_id}/stop")
    async def ask_stop(ask_id: int) -> Response:
        try:
            await run_in_threadpool(model_mod.stop, page, ask_id)
        except Refused as e:
            return refused(e)
        return Response(status_code=204)

    return app


def _preload(page: Page) -> None:
    """Load the local model once when the page starts (keep_alive 30 minutes), if it can run."""
    nr = page.narrator
    try:
        if not nr.state(fresh=True).ok:
            return
        with page.model_lock:
            nr.preload()
        nr.state(fresh=True)  # now loaded: where it sits decides the CPU line and the timeouts
    except Exception as e:  # keep serving; the first ask wakes the model instead
        print(f"page: model preload failed ({type(e).__name__})", file=sys.stderr)


def _hourly(page: Page, stop: threading.Event) -> None:
    while not stop.wait(HOURLY_S):
        try:
            page.maintain()
        except Exception as e:  # keep serving; the next hour tries again
            print(f"page: hourly expiry and backfill failed ({type(e).__name__})", file=sys.stderr)


def _by_channel(streams: list[Stream]) -> list[tuple[str, list[Stream]]]:
    groups: dict[str, list[Stream]] = {}
    for s in streams:
        groups.setdefault(s.channel, []).append(s)
    return sorted(groups.items())
