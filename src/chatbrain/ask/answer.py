"""One answer: a checked query run read-only, as a line, blocks, chips and two follow-ups (Task 10b).

``run`` takes a query object (``query.check`` has passed it) and answers it from the store with
no model: every number comes from shown answers and counts, read through ``sql.readonly``. The
answer is never stored; a thread re-runs its stored queries on every load, so later labels and
fixes change old answers.

Blocks are plain dicts with a ``kind`` the page renders with one Jinja macro each:

- ``cards``: ``moments`` (``page.app.Moment``) and ``max_posters``;
- ``bars``: ``rows`` of ``{label, k, n, text, v}`` (top: one per Moment; trend: one per stream or day);
- ``count``: ``rows`` of ``{label, k, n, maybe, color, v}``, biggest first, "not sure" last;
- ``mix``: an emote's windows: ``n``, ``moments``, ``ordinary``, ``expired``, and ``groups`` of
  ``(question label, rows)`` for doing, feels and tone (rows as ``count``);
- ``moment``: one Moment ``m`` (panel extras filled), its ``counted`` rows and every ``questions``;
- ``evidence``: one Moment ``m`` (panel extras filled) and the ``questions`` asked about;
- ``notice``: ``text``, plus at most one runnable chip: an ``open`` Moment, or ``run`` =
  ``(label, tokens)`` the page posts like a follow-up (a refusal's nearest thing it can do).

Follow-ups are ``(label, tokens)``: the page posts the tokens to ``/ask/<id>/edit``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from .. import manifest
from ..page import app as app_mod
from ..page import shown as shown_mod
from . import query as q_mod
from . import sql as sql_mod

REFUSALS = ("identity", "traits", "single_message", "as_streamer", "outbound")  # stored as "[refused: …]"
PER_WINDOW = ("moment_detail", "explain_answer")
LEAD = {"posters": "Biggest", "lines": "Most lines", "rate": "Fastest", "newest": "Newest", "oldest": "First",
        "least_sure": "Least sure"}  # fmt: skip
EXAMPLES = 3  # example cards under a count, a mix or a closest-Moments answer
EMOTE_MIN = 5  # fewer windows than this: too few to say what an emote means
ABOUT_LEAD = {"doing": "chat was ", "aimed_at": "aimed at ", "feeling": "chat felt ", "tone": "tone "}
SLOW = "That took too long, so BrainChat stopped it. Try a narrower question."
BANNER = "Asking is limited: the local model isn't running. Words were matched to filters; check the chips."
ON_STREAM = "Chat alone can't show what happened on stream"
CAUSE_UNKNOWN = ON_STREAM + ", so the cause is unknown."
NO_CAUSE = "Cause unknown · no cause note."
VIEWERS = "BrainChat counts people who posted, not viewers."


StepState = Literal["done", "skipped", "failed"]


@dataclass
class Step:
    """One step that really ran (F3, G-9): its line, how it ended, and its time when it was measured.

    ``state`` is ``done``, ``skipped`` (it never ran: the model was off or busy) or ``failed`` (it
    ran and didn't finish). ``ms`` is set only from the time stored with the ask when the step ran;
    a re-run on display never measures it again."""

    text: str
    state: StepState = "done"
    ms: int | None = None


@dataclass
class Answer:
    query: dict[str, Any]
    window_id: str | None = None
    ids: list[str] = field(default_factory=list)  # the result windows, in order (stored with the ask)
    line: list[str] = field(default_factory=list)  # the template line's sentences
    more: str | None = None  # "+N more where BrainChat isn't sure."
    blocks: list[dict[str, Any]] = field(default_factory=list)
    chips: list[q_mod.Chip] = field(default_factory=list)
    follow_ups: list[tuple[str, list[str]]] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)  # runnable questions when the query was unusable
    total: int = 0  # matches
    maybe: int = 0  # not sure on a value filter, counted apart
    scope_n: int = 0  # windows looked at
    noun: str = "Moment"
    steps: list[Step] = field(default_factory=list)
    seconds: float = 0.0  # this run's own time; stored with the ask when it is asked, never shown from a re-run
    made_up: bool = False
    refused: str | None = None
    closest: bool = False  # the asked time had no Moment within 5 min; kept with the ask so a re-run says so
    how: str | None = None  # "keywords" when the local model step fell back (Task 10c)

    @property
    def line_text(self) -> str:
        return " ".join([*self.line, *([self.more] if self.more else [])])

    @property
    def step_text(self) -> str:
        """The settled line (F3): how many steps ran and, when their times were stored, how long."""
        # G-9: only steps that really ran count. A skipped step never ran (the model was off or busy);
        # a failed one ran and didn't finish, so it counts.
        s = app_mod.plural(sum(x.state != "skipped" for x in self.steps), "step")
        timed = [x.ms for x in self.steps if x.ms is not None]
        if timed:
            s += f" · {sum(timed) / 1000:.1f} s"
        return s + (" · made-up labels" if self.made_up else "") + (f" · {self.how}" if self.how else "")


@dataclass
class _Ctx:
    page: Any
    ro: sql_mod.ReadOnly
    cte: tuple[str, list[Any]]
    now: datetime
    plan: shown_mod.Plan

    def scope(self, q: dict[str, Any], **kw: Any) -> sql_mod.Scope:
        return sql_mod.scope(self.ro, self.cte, q, self.now, **kw)

    def moments(self, ids: list[str]) -> list[Any]:
        return self.page.moments(self.page.windows(ids), self.now, self.plan) if ids else []

    def text(self, v: str) -> str:
        return self.page.scheme.display.get(v, v)


def run(
    page: Any,
    query: dict[str, Any],
    *,
    window_id: str | None = None,
    notes: dict[str, Any] | None = None,
    previous_ids: list[str] | None = None,
) -> Answer:
    """Answer ``query`` on ``page`` at its clock. ``window_id`` is the Moment a per-Moment ask is
    about; ``previous_ids`` resolve ``ref`` ("the 2nd one"). ``notes``: ``kept``, ``ignored`` and
    ``didnt_understand`` (chips), ``default_reading`` and ``edited``."""
    notes = dict(notes or {})
    started = time.monotonic()
    now = page.now()
    p = page.plan()
    q = dict(query)
    if window_id is None and q.get("ref") and previous_ids and 0 < q["ref"] <= len(previous_ids):
        window_id = previous_ids[q["ref"] - 1]
    if q["intent"] not in PER_WINDOW:
        window_id = None
    a = Answer(query=q, window_id=window_id, made_up=p.made_up)
    if q["intent"] == "cant_answer" and q.get("reason") in (*REFUSALS, None, "off_topic", "unresolved_ref"):
        a.refused = q["reason"] if q["reason"] in REFUSALS else None
    else:
        a.chips = q_mod.chips(
            q, page.scheme, kept=notes.get("kept", ()), ignored=notes.get("ignored", ()),
            didnt=notes.get("didnt_understand", ()),
        )  # fmt: skip
    cte = shown_mod.cte(p, scheme_version=page.scheme.version, review_ids=page.review_ids)
    statements = 0
    try:
        with sql_mod.readonly(page.store.path) as ro:
            ctx = _Ctx(page, ro, cte, now, p)
            _INTENTS[q["intent"]](ctx, a, notes)
            statements = ro.statements
    except (sql_mod.TooSlow, sql_mod.Denied):
        a.ids, a.line, a.more, a.blocks, a.follow_ups = [], [SLOW], None, [], []
    else:
        if not _one_chip(q):
            a.follow_ups = _two(q, [*a.follow_ups, *_fallbacks(q)])
    a.seconds = time.monotonic() - started
    a.steps = [
        Step("Took the edited chips" if notes.get("edited") else "Matched words to filters (no model)"),
        Step(f"Read the shown answers of {app_mod.plural(a.scope_n, a.noun)}"),
    ]
    if statements:
        a.steps.append(Step(f"Ran {statements} read-only {'query' if statements == 1 else 'queries'}"))
    return a


def ms_of(value: Any) -> int | None:
    """A stored step time in ms, or None when there is none (an ask stored before steps had times)."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


# --- list answers ---------------------------------------------------------------------------------


def _size(win: Any, sort: str) -> str:
    if sort == "lines":
        return f"{app_mod.plural(win.count('joining_lines'), 'line')} joining"
    if sort == "rate":
        rate = win.counts.get("msg_per_min")
        return f"{rate:.0f} messages per minute" if isinstance(rate, (int, float)) else "rate unknown"
    return f"{app_mod.plural(win.count('joining_posters'), 'poster')} joining"


def _list_line(total: int, ms: list[Any], sort: str) -> str:
    if total == 0 or not ms:
        return "0 Moments match."
    first = ms[0]
    where = f"{app_mod.clock_text(first.win.anchor)}, {_size(first.win, sort)}"
    if total == 1:
        return f"1 Moment: {where}."
    return f"{total} Moments match. {LEAD.get(sort, 'Biggest')}: {where}."


def _cards(ms: list[Any]) -> dict[str, Any]:
    return {"kind": "cards", "moments": ms, "max_posters": max([m.win.count("joining_posters") for m in ms] + [1])}


def _maybe(a: Answer) -> None:
    if a.maybe:
        a.more = f"+{a.maybe} more where BrainChat isn't sure."


def _find(ctx: _Ctx, a: Answer, notes: dict[str, Any]) -> None:
    q = a.query
    sc = ctx.scope(q)
    f = sql_mod.found(ctx.ro, sc, q["sort"], q["limit"])
    a.ids, a.total, a.maybe, a.scope_n = f.ids, f.total, f.maybe, f.scope_n
    ms = ctx.moments(f.ids)
    a.line = [_list_line(a.total, ms, q["sort"])]
    if q["intent"] == "top_moments" and ms:
        a.line.append(f"Cause: {ms[0].cause.note}" if ms[0].cause else "Cause unknown: no cause note.")
        values = [_value(m.win, q["sort"]) for m in ms]
        top = max(values + [1e-9])
        a.blocks.append({
            "kind": "bars",
            "rows": [
                {"label": app_mod.clock_text(m.win.anchor), "window_id": m.win.window_id, "k": v, "n": top,
                 "text": _size(m.win, q["sort"]), "v": v / top}  # fmt: skip
                for m, v in zip(ms, values)
            ],
        })
    if ms:
        a.blocks.append(_cards(ms))
    _maybe(a)
    a.follow_ups = _narrow_widen(q, ctx.page.scheme, empty=a.total == 0)


def _value(win: Any, sort: str) -> float:
    if sort == "rate":
        rate = win.counts.get("msg_per_min")
        return float(rate) if isinstance(rate, (int, float)) else 0.0
    return float(win.count("joining_lines" if sort == "lines" else "joining_posters"))


# --- one Moment -----------------------------------------------------------------------------------


def _one_window(ctx: _Ctx, a: Answer, notes: dict[str, Any]) -> Any | None:
    """The Moment a per-window answer is about: the page's, the nearest to ``at``, else the top one.
    A stored ask keeps the Moment it found, so ``notes["closest"]`` repeats the "closest" line."""
    q = a.query
    if a.window_id is not None:
        a.scope_n = 1
        found = ctx.moments([a.window_id])
        if not found:
            a.line = ["That Moment is no longer stored."]
            a.window_id = None
        elif notes.get("closest") and q.get("at"):
            a.line, a.closest = [f"No Moment at {q['at']}; this is the closest."], True
        return found[0] if found else None
    sc = ctx.scope(q, use_at=False)
    a.scope_n = sql_mod.found(ctx.ro, sc, "posters", 0).scope_n
    ids = sql_mod.nearest(ctx.ro, sc, 1) if sc.instants else sql_mod.found(ctx.ro, sc, q["sort"], 1).ids
    if not ids:
        a.line = [f"No Moment near {q['at']}." if q.get("at") else "No Moments in reach."]
        return None
    m = ctx.moments(ids)[0]
    if sc.instants:
        gap = min(abs((m.win.anchor - t).total_seconds()) for t in sc.instants)
        if gap > sql_mod.AT_SLACK.total_seconds():
            a.line, a.closest = [f"No Moment at {q['at']}; this is the closest."], True
    return m


def _asked(q: dict[str, Any]) -> list[str]:
    """The scheme questions a query names (its value keys, then ``about``)."""
    out = [q_mod.KEY_QUESTION[k] for k in q_mod.VALUE_KEYS if q.get(k)]
    if q.get("about") and q["about"] not in out:
        out.append(q["about"])
    return out


def _cause_line(m: Any, asks_cause: bool) -> list[str]:
    """The server's cause sentence: a cause comes only from a cause note (DESIGN-BRIEF §4)."""
    if m.cause is not None:
        return ([ON_STREAM + "."] if asks_cause else []) + [f"Cause note: {m.cause.note}."]
    return [CAUSE_UNKNOWN] if asks_cause else [NO_CAUSE]


def _detail(ctx: _Ctx, a: Answer, notes: dict[str, Any]) -> None:
    m = _one_window(ctx, a, notes)
    if m is None:
        return
    a.ids, a.total = [m.win.window_id], 1
    w = m.win
    ctx.page.fill_panel(m, ctx.plan)  # up to 5 evidence lines per answer, while the text shows
    cause = _cause_line(m, bool(a.query.get("asks_cause")))
    if a.query.get("asks_cause"):
        a.line[:0] = cause[:1]
        cause = cause[1:]
    a.line.append(
        f"{app_mod.clock_text(w.anchor)}: {m.title}. {_size(w, 'posters')}, {_size(w, 'lines')}."
    )
    for qk in _asked(a.query):
        a.line.append(_says(ctx, m, qk))
    a.line += cause
    a.blocks.append({"kind": "moment", "m": m, "counted": app_mod._counted(m, ctx.page.scheme),
                     "questions": list(ctx.page.questions)})  # fmt: skip
    a.window_id = w.window_id
    a.follow_ups = [("why chat thinks so", q_mod.tokens(q_mod.blank() | {"intent": "explain_answer"})), _like(m)]


def _like(m: Any) -> tuple[str, list[str]]:
    """The wider follow-up from one Moment: Moments doing the same, else the channel's biggest."""
    w = m.win
    if m.first_doing:
        return "Moments like this", q_mod.tokens(q_mod.blank() | {"doing": [m.first_doing], "channel": [w.channel]})
    return f"more from {w.channel}", q_mod.tokens(
        q_mod.blank() | {"intent": "top_moments", "channel": [w.channel], "time": "all"}
    )


def _says(ctx: _Ctx, m: Any, qk: str, asked: list[str] | None = None) -> str:
    s = m.answers[qk]
    label = app_mod.QUESTION_LABEL[qk]
    if not s.sure:
        why = f" ({', '.join(s.not_sure)})" if s.not_sure else ""
        return f"{label}: not sure{why}."
    words = ", ".join(ctx.text(v) for v in s.values)
    if s.kind == "model" and s.confidence is not None:
        thr = f", threshold {shown_mod.conf_text(s.threshold)}" if s.threshold is not None else ""
        out = f"{label}: {words}, from {s.source} at {shown_mod.conf_text(s.confidence)}{thr}."
    else:
        out = f"{label}: {words}, from {s.source}."
    missing = [v for v in (asked or []) if v not in s.values]
    if missing:
        out += f" It doesn't say {', '.join(ctx.text(v) for v in missing)}."
    return out


def _report(ctx: _Ctx, qk: str, asked: list[str]) -> str | None:
    """One question's test report from the active model's manifest (ENGINE §2: n, agreement, Wilson)."""
    use = ctx.plan.active.get(qk)
    folder = manifest.model_folder(ctx.page.models_dir, use.model_id) if use is not None else None
    doc = manifest.read(folder) if folder is not None else None
    r = ((doc or {}).get("report") or {}).get(qk)
    if not isinstance(r, dict):
        return None
    label = app_mod.QUESTION_LABEL[qk]
    num = manifest._number
    answered, graded, agreement, ci = r.get("answered"), r.get("n_graded"), r.get("agreement"), r.get("agreement_ci")
    if not num(agreement):
        out = f"{label}: no test answers yet"
    else:
        out = f"{label}: agreement {shown_mod.conf_text(agreement)}"
        if isinstance(ci, list) and len(ci) == 2 and all(num(x) for x in ci):
            out += f" [{shown_mod.conf_text(ci[0])}, {shown_mod.conf_text(ci[1])}]"
        if num(answered):
            out += f" on {int(answered)} answered" + (f" of {int(graded)} graded" if num(graded) else "")
    if use.threshold is not None:
        out += f", threshold {shown_mod.conf_text(use.threshold)}"
    out += f" ({shown_mod.source_of(use.origin, use.rung)})."
    per = r.get("per_value") if isinstance(r.get("per_value"), dict) else {}
    for v in asked:
        row = per.get(v)
        if isinstance(row, list) and len(row) == 3 and all(num(x) for x in row):
            out += (f" {ctx.text(v)}: precision {shown_mod.conf_text(row[0])}, recall {shown_mod.conf_text(row[1])},"
                    f" {int(row[2])} in the test set.")  # fmt: skip
    return out


def _explain(ctx: _Ctx, a: Answer, notes: dict[str, Any]) -> None:
    q = a.query
    questions = _asked(q) or list(ctx.page.questions)
    if a.window_id is None:
        a.follow_ups = [
            ("least sure Moments", q_mod.tokens(q_mod.blank() | {"is": ["not_sure"], "sort": "least_sure",
                                                                 "time": "all"})),  # fmt: skip
            ("Moments you labeled", q_mod.tokens(q_mod.blank() | {"is": ["labeled"], "time": "all"})),
        ]
        if ctx.plan.origin is None:
            text = "No model answers yet, so there is nothing to explain: what the page shows comes from your labels."
            a.line = [text]
            a.blocks.append({"kind": "notice", "text": text, "open": None, "run": None})
            return
        said = [
            _report(ctx, qk, q.get(q_mod.QUESTION_KEY[qk]) or []) or f"{app_mod.QUESTION_LABEL[qk]}: no model report."
            for qk in questions
        ]
        a.line = [*said, "Open a Moment and ask “why” there to see its evidence lines."]
        return
    m = _one_window(ctx, a, notes)
    if m is None:
        return
    ctx.page.fill_panel(m, ctx.plan)
    a.ids, a.total = [m.win.window_id], 1
    if q.get("asks_cause"):
        a.line[:0] = _cause_line(m, True)
    for qk in questions:
        key = q_mod.QUESTION_KEY[qk]
        a.line.append(_says(ctx, m, qk, q.get(key) or []))
    a.blocks.append({"kind": "evidence", "m": m, "questions": questions})
    a.follow_ups = [("what happened here", q_mod.tokens(q_mod.blank() | {"intent": "moment_detail"})), _like(m)]


# --- counts ---------------------------------------------------------------------------------------


def _group_label(ctx: _Ctx, g: sql_mod.Group, by: str, q: dict[str, Any]) -> str:
    if by == "channel":
        return str(g.key)
    if by == "stream":
        return f"{app_mod.day(g.first)} stream"
    if by == "day":
        return app_mod.day(g.first)
    return q_mod.TIME_TEXT[q["time"]] if g.key else "the rest"


def _side(r: dict[str, Any]) -> str:
    """One side of a comparison: "k of n", plus its not-sure count (ENGINE §2)."""
    return f"{r['label']}: {r['k']} of {r['n']}" + (f", {r['maybe']} not sure" if r.get("maybe") else "")


def _count(ctx: _Ctx, a: Answer, notes: dict[str, Any]) -> None:
    q = a.query
    by, about = q.get("by"), q.get("about")
    period = by == "period"
    sc = ctx.scope(q, period=period)
    rows: list[dict[str, Any]] = []
    if by == "value" or (about and not by):
        qk = about or "doing"
        vals, n, unsure, maybe, scope_n = sql_mod.by_value(ctx.ro, sc, q_mod.QUESTION_KEY[qk])
        order = list(ctx.page.questions[qk].values) + [ctx.page.questions[qk].blank_value]
        ranked = sorted(vals.items(), key=lambda kv: (-kv[1], order.index(kv[0]) if kv[0] in order else 99))
        for v, k in ranked:
            rows.append({"label": ctx.text(v), "k": k, "n": n, "maybe": 0,
                         "color": app_mod.doing_color(v) if qk == "doing" else None})  # fmt: skip
        if unsure:
            rows.append({"label": "not sure", "k": unsure, "n": n, "maybe": 0, "color": app_mod.NOT_SURE_COLOR})
        a.total, a.maybe, a.scope_n = n, maybe, scope_n
        if n == 0:
            line = "0 Moments match."
        else:
            named = [f"{r['label']} in {r['k']}" for r in rows if r["label"] != "not sure"][:3]
            body = ", ".join(named[:-1]) + (" and " if len(named) > 1 else "") + named[-1] if named else "nothing sure"
            line = f"Of {app_mod.plural(n, 'Moment')}, {ABOUT_LEAD[qk]}{body}."
            if unsure:
                line += f" Not sure in {unsure}."
    elif by in ("channel", "stream", "day", "period"):
        gs = sql_mod.groups(ctx.ro, sc, by)
        if period:
            gs.sort(key=lambda g: not g.key)  # the asked time first, then the rest
        for g in gs:
            rows.append({"label": _group_label(ctx, g, by, q), "k": g.k, "n": g.n, "maybe": g.maybe, "color": None})
        cur = [g for g in gs if g.key] if period else gs
        a.total = sum(g.k for g in cur)
        a.maybe = sum(g.maybe for g in cur)
        a.scope_n = sum(g.n for g in gs)
        line = "; ".join(_side(r) for r in rows) + "." if rows else "No Moments in reach."
    else:
        f = sql_mod.found(ctx.ro, sc, "posters", 0)
        a.total, a.maybe, a.scope_n = f.total, f.maybe, f.scope_n
        rows.append({"label": "match", "k": f.total, "n": f.scope_n, "maybe": f.maybe, "color": None})
        line = f"{f.total} of {app_mod.plural(f.scope_n, 'Moment')} match."
    for r in rows:
        r["v"] = r["k"] / r["n"] if r["n"] else 0.0
    if notes.get("default_reading"):
        when = (
            f"on {app_mod.day(sql_mod.first_anchor(ctx.ro, sc) or ctx.now)}"
            if q["time"] == "last_stream"
            else f"over {q_mod.TIME_TEXT[q['time']]}"
        )
        a.line = [f"Took the plainest reading: what chat did {when}.", line]
    else:
        a.line = [line]
    a.blocks.append({"kind": "count", "rows": rows})
    if by in ("channel", "stream", "day", "period"):
        a.more = None  # each side already says how many it isn't sure about
    ex = sql_mod.found(ctx.ro, sc, "posters", EXAMPLES, only_cur=period)
    ms = ctx.moments(ex.ids)
    a.ids = ex.ids
    if ms:
        a.blocks.append(_cards(ms))
    if by not in ("channel", "stream", "day", "period"):
        _maybe(a)
    if notes.get("default_reading"):
        base = q_mod.blank() | {"intent": "top_moments", "time": q["time"], "channel": list(q["channel"])}
        a.follow_ups = [
            ("funniest", q_mod.tokens(base | {"doing": ["laughing"]})),
            ("most critical", q_mod.tokens(base | {"feels": ["critical"]})),
        ]
    else:
        a.follow_ups = _narrow_widen(q, ctx.page.scheme)


def _trend(ctx: _Ctx, a: Answer, notes: dict[str, Any]) -> None:
    q = a.query
    by = q.get("by") if q.get("by") in q_mod.TREND_BY else "stream"
    gs = sql_mod.groups(ctx.ro, ctx.scope(q), by)
    rows = [
        {"label": _group_label(ctx, g, by, q), "k": g.k, "n": g.n, "text": f"{g.k} of {g.n}",
         "v": g.k / g.n if g.n else 0.0}  # fmt: skip
        for g in gs
    ]
    a.total, a.maybe, a.scope_n = sum(g.k for g in gs), sum(g.maybe for g in gs), sum(g.n for g in gs)
    for r, g in zip(rows, gs):
        r["maybe"] = g.maybe
    if not gs:
        a.line = ["No Moments in reach."]
    elif len(gs) == 1:
        a.line = [f"Not enough history: only 1 {by} in reach.",
                  f"{gs[0].k} of {app_mod.plural(gs[0].n, 'Moment')} match."]  # fmt: skip
    else:
        last, before = gs[-1], gs[:-1]
        k0, n0 = sum(g.k for g in before), sum(g.n for g in before)
        r1, r0 = (last.k / last.n if last.n else 0.0), (k0 / n0 if n0 else 0.0)
        word = "more often" if r1 > r0 + 0.1 else "less often" if r1 < r0 - 0.1 else "about as often"
        a.line = [f"{rows[-1]['label']}: {last.k} of {last.n} Moments match, {word} than before ({k0} of {n0}).",
                  f"Covers {len(gs)} {by}s."]  # fmt: skip
    a.blocks.append({"kind": "bars", "rows": rows})
    _maybe(a)
    a.follow_ups = _narrow_widen(q, ctx.page.scheme)


# --- emotes ---------------------------------------------------------------------------------------


def _emote(ctx: _Ctx, a: Answer, notes: dict[str, Any]) -> None:
    q = a.query
    name = q["emote"][0]
    a.noun = "window"
    sc = ctx.scope(q, moments_only=False)
    mix = sql_mod.emote_mix(ctx.ro, sc, EXAMPLES)
    a.scope_n = sql_mod.found(ctx.ro, sc, "posters", 0).scope_n
    a.total = mix.n
    held = q_mod.held(ctx.page.store)
    review = any(eid in ctx.page.review_ids for _, eid in held.emotes.get(name, ()))
    head = (
        f"{name} showed up in {app_mod.plural(mix.n, 'window')} ({app_mod.plural(mix.moments, 'Moment')},"
        f" {app_mod.plural(mix.n - mix.moments, 'ordinary window')})."
    )
    if mix.n == 0:
        a.line = [f"{name} didn't show up in any window in reach."]
    elif review:
        a.line = [head, f"{name} is on the review list, so BrainChat doesn't describe what it means."]
    elif mix.n < EMOTE_MIN:
        a.line = [head, "That's too few to say what it means, so: not sure."]
    else:
        ranked = sorted(mix.mix["doing"].items(), key=lambda kv: -kv[1])
        said = [f"{ctx.text(v)} in {k}" for v, k in ranked[:2]]
        line = [head, f"Chat was {' and '.join(said)}." if said else "BrainChat isn't sure what chat was doing."]
        feels = sorted(mix.mix["feels"].items(), key=lambda kv: -kv[1])
        if feels:
            line.append(f"Mostly {ctx.text(feels[0][0])} ({feels[0][1]} of {mix.n}).")
        a.line = line
    if mix.expired:
        a.line.append(f"Text has gone for {mix.expired} of them; their counts remain.")
    if mix.n and not review:
        groups = []
        for key in ("doing", "feels", "tone"):
            rows = [
                {"label": ctx.text(v), "k": k, "n": mix.n, "maybe": 0, "v": k / mix.n,
                 "color": app_mod.doing_color(v) if key == "doing" else None}  # fmt: skip
                for v, k in sorted(mix.mix[key].items(), key=lambda kv: -kv[1])
            ]
            if mix.unsure[key]:
                rows.append({"label": "not sure", "k": mix.unsure[key], "n": mix.n, "maybe": 0,
                             "color": app_mod.NOT_SURE_COLOR, "v": mix.unsure[key] / mix.n})  # fmt: skip
            groups.append((app_mod.QUESTION_LABEL[q_mod.KEY_QUESTION[key]], rows))
        a.blocks.append({"kind": "mix", "n": mix.n, "moments": mix.moments, "ordinary": mix.n - mix.moments,
                         "expired": mix.expired, "groups": groups})  # fmt: skip
    a.ids = mix.ids
    ms = ctx.moments(mix.ids)
    if ms:
        a.blocks.append(_cards(ms))
    a.follow_ups = [
        ("where it showed up", q_mod.tokens(q_mod.blank() | {"emote": [name], "time": q["time"],
                                                             "channel": list(q["channel"])})),  # fmt: skip
    ]
    if q["time"] != "all":
        a.follow_ups.append(("all time", q_mod.tokens(q | {"follow_up": "refine", "time": "all"})))
    else:  # already all time: the same question about another emote held in reach
        chans = set(q["channel"])
        other = next((n for n, pairs in held.emotes.items()
                      if n != name and (not chans or any(c in chans for c, _ in pairs))), None)  # fmt: skip
        if other is not None:
            a.follow_ups.append((f"what {other} means", q_mod.tokens(q | {"follow_up": "refine", "emote": [other]})))


# --- can't answer ---------------------------------------------------------------------------------

CANT = {
    "identity": "BrainChat doesn't track who people are. Speakers are numbered per Moment and can't be linked.",
    "traits": "BrainChat doesn't judge single messages or anything about who people are.",
    "single_message": "BrainChat doesn't judge single messages or anything about who people are.",
    "as_streamer": "BrainChat speaks only about chat as a crowd, never as the streamer.",
    "outbound": "Nothing leaves this PC: BrainChat can't post, send or share anything.",
    "off_topic": "BrainChat only answers questions about chat in your Moments.",
    "unresolved_ref": "There's no earlier answer to point at. Ask about a time, or open a Moment and ask there.",
    None: "BrainChat couldn't turn that into a search. Try one of these:",
}


def _closest(ctx: _Ctx, q: dict[str, Any], n: int) -> tuple[list[str], int, str]:
    """Up to ``n`` Moments and how they were found: nearest ``at`` ("at"), else the biggest that
    match the labels asked ("labels"), else the biggest in reach ("busiest")."""
    sc = ctx.scope(q, use_at=False)
    f = sql_mod.found(ctx.ro, sc, "posters", n)
    if sc.instants:
        ids = sql_mod.nearest(ctx.ro, sc, n)
        if ids:
            return ids, f.scope_n, "at"
    labelled = any(q.get(k) for k in (*q_mod.VALUE_KEYS, "spread", "is", "emote")) or q.get("phrase")
    if f.ids and labelled:
        return f.ids, f.scope_n, "labels"
    wide = q_mod.blank() | {"time": q["time"], "channel": list(q["channel"])}
    w = sql_mod.found(ctx.ro, ctx.scope(wide), "posters", n)
    return w.ids, w.scope_n, "busiest"


def _cant(ctx: _Ctx, a: Answer, notes: dict[str, Any]) -> None:
    q = a.query
    reason = q.get("reason")
    if reason == "on_stream":
        ids, a.scope_n, how = _closest(ctx, q, EXAMPLES)
        a.line = [CAUSE_UNKNOWN if q.get("asks_cause") else ON_STREAM + "."]
        if notes.get("viewers"):
            a.line.append(VIEWERS)
        if ids:
            a.line.append({
                "at": f"Here is what chat did around {q['at']}.",
                "labels": "These are the closest Moments by their labels.",
                "busiest": "Here is where chat was busiest.",
            }[how])
        a.ids = ids
        ms = ctx.moments(ids)
        if ms:
            a.blocks.append(_cards(ms))
        when = {"time": q["time"], "channel": list(q["channel"])}
        if q.get("at"):  # what chat itself did at that time: the one thing chat can show
            a.follow_ups = [(f"what chat did at {q['at']}",
                             q_mod.tokens(q_mod.blank() | {"intent": "moment_detail", "at": q["at"]} | when))]  # fmt: skip
        else:
            a.follow_ups = [("what chat did", q_mod.tokens(q_mod.blank() | {"intent": "count", "about": "doing",
                                                                            "by": "value"} | when))]  # fmt: skip
        if q["time"] != "all":
            a.follow_ups.append(("all time", q_mod.tokens(q | {"time": "all"})))
        return
    a.line = [CANT.get(reason, CANT[None])]
    a.ids = []
    if reason is None:
        a.suggestions = notes.get("suggestions") or _suggest(ctx)
        return
    when = {"time": q["time"]}
    run_q = q_mod.blank() | {"intent": "top_moments"} | when
    label = "biggest Moments"
    if reason == "identity":
        ids, a.scope_n, _ = _closest(ctx, q_mod.blank() | {"at": q.get("at")} | when, 1)
        ms = ctx.moments(ids)
        if ms:
            a.blocks.append({"kind": "notice", "text": None, "open": ms[0], "run": None})
            return
    elif reason in ("traits", "single_message"):
        sc = ctx.scope(q_mod.blank() | when)
        a.scope_n = sql_mod.found(ctx.ro, sc, "posters", 0).scope_n
        n = sql_mod.review_count(ctx.ro, sc)
        if n:
            a.line.append(f"{app_mod.plural(n, 'Moment')} in reach {'is' if n == 1 else 'are'} marked needs review.")
            label, run_q = "Moments marked needs review", q_mod.blank() | {"is": ["needs_review"]} | when
        else:
            label, run_q = "hostile Moments", q_mod.blank() | {"tone": ["hostile"]} | when
        a.line.append("For one message, use the platform's own mod tools.")
    elif reason == "as_streamer":
        label, run_q = "what chat did", q_mod.blank() | {"intent": "count", "about": "doing", "by": "value"} | when
    elif reason == "outbound":
        label, run_q = "Moment times, oldest first", q_mod.blank() | {"sort": "oldest", "limit": 20} | when
    a.blocks.append({"kind": "notice", "text": None, "open": None, "run": (label, q_mod.tokens(run_q))})


def _suggest(ctx: _Ctx) -> list[str]:
    s = q_mod.suggestions(q_mod.held(ctx.page.store))
    return [s[0], s[1], s[4]]


# --- follow-ups -----------------------------------------------------------------------------------


_NARROWEST = ("phrase", "min_posters", "emote", "is", "spread", "tone", "feels", "aimed", "doing", "channel")


def _narrow_widen(q: dict[str, Any], scheme: Any, *, empty: bool = False) -> list[tuple[str, list[str]]]:
    """Follow-ups in order of preference: one narrower, then one wider (DESIGN-BRIEF §4), then the
    other narrower and wider ones. ``run`` keeps the first two. With no match, the first drops the
    narrowest filter instead ("0 Moments match." plus one chip that drops it), then "all time", then
    the next-narrowest filter."""
    base = q | {"follow_up": "refine"}
    wider_time = [("all time", q_mod.tokens(base | {"time": "all"}))] if q["time"] != "all" else []
    if empty:
        drops = []
        for k in _NARROWEST:
            v = q.get(k)
            if v in (None, [], ""):
                continue
            if isinstance(v, list):
                tok, rest = f"{k}:{v[-1]}", base | {k: v[:-1]}
            else:
                tok, rest = f"{k}:{v}", base | {k: None}
            drops.append((f"without {q_mod.chip_text(tok, scheme)}", q_mod.tokens(rest)))
        return [*drops[:1], *wider_time, *drops[1:]]
    narrow = []
    if not q["aimed"]:
        narrow.append(("only at the streamer", q_mod.tokens(base | {"aimed": ["streamer"]})))
    if not q["feels"]:
        narrow.append(("only critical", q_mod.tokens(base | {"feels": ["critical"]})))
    if "labeled" not in q["is"]:
        narrow.append(("only ones you labeled", q_mod.tokens(base | {"is": [*q["is"], "labeled"]})))
    wide = list(wider_time)
    last = next((k for k in reversed(q_mod.VALUE_KEYS) if q[k]), None)
    if last is not None:
        wide.append((f"without {q_mod.KEY_TEXT[last]} {q[last][-1]}".replace("_", " "),
                     q_mod.tokens(base | {last: q[last][:-1]})))  # fmt: skip
    if q["intent"] in ("find_moments", "top_moments") and q["limit"] < 10:
        wide.append(("show 10", q_mod.tokens(base | {"limit": 10})))
    return [*narrow[:1], *wide[:1], *narrow[1:], *wide[1:]]


def _one_chip(q: dict[str, Any]) -> bool:
    """A refusal or a can't-answer other than on-stream: one sentence and one runnable chip (or the
    three suggestions), never a follow-up row (DESIGN-BRIEF §4 "Refusal")."""
    return q["intent"] == "cant_answer" and q.get("reason") != "on_stream"


def _fallbacks(q: dict[str, Any]) -> list[tuple[str, list[str]]]:
    """Follow-ups any answer can offer, in the asked time and channels."""
    when = {"time": q["time"], "channel": list(q["channel"])}
    b = q_mod.blank()
    return [
        ("biggest Moments", q_mod.tokens(b | {"intent": "top_moments"} | when)),
        ("what chat did", q_mod.tokens(b | {"intent": "count", "about": "doing", "by": "value"} | when)),
        ("least sure Moments", q_mod.tokens(b | {"is": ["not_sure"], "sort": "least_sure"} | when)),
    ]


def _two(q: dict[str, Any], options: list[tuple[str, list[str]]]) -> list[tuple[str, list[str]]]:
    """Exactly two follow-ups (DESIGN-BRIEF §4): the first two ``options`` that differ from the
    query itself and from each other. ``options`` ends with ``_fallbacks``, so two always remain."""

    def key(toks: list[str]) -> tuple[str, ...]:
        return tuple(sorted(t for t in toks if not t.startswith("follow_up:")))

    seen, labels, out = {key(q_mod.tokens(q))}, set(), []
    for label, toks in options:
        k = key(toks)
        if k in seen or label in labels:
            continue
        seen.add(k)
        labels.add(label)
        out.append((label, toks))
        if len(out) == 2:
            break
    return out


_INTENTS = {
    "find_moments": _find,
    "top_moments": _find,
    "moment_detail": _detail,
    "explain_answer": _explain,
    "count": _count,
    "trend": _trend,
    "emote_meaning": _emote,
    "cant_answer": _cant,
}
