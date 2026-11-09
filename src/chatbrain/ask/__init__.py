"""Asking about chat on the page: the keyword path (Task 10b) and the local model steps (10c).

A typed question goes through ``keywords.parse`` (the refusal pre-check first), then
``query.check``, then ``answer.run`` over the read-only connection in ``sql``. The page stores
one ``asks`` row per question: the typed text (or ``[refused: <reason>]``, never the text of
a refused question), the checked query, its follow-up kind and the result window IDs. Numbers
are never stored: a thread re-runs every stored query when it is shown.

An ask expires at the earlier of 14 days after it was asked and the earliest text expiry among
its result windows whose text still shows (``expiry.run`` deletes due rows; the thread hides
them before that). A query with a quoted ``phrase`` stores chat text in its question, so its
ask also expires by the earliest text expiry of every window whose kept text holds that phrase,
whatever the query's scope or result cap. Threads are named by a random 16-hex ID; the URL
holds only that.

The keyword path answers and stores every ask at once. With a local model (``page.narrator``),
``model.start`` marks a typed ask as waiting on step 1 and ``model.after_edit`` a chip edit on its
caption; the page script then runs the steps (``model.advance``), which may rewrite the stored ask.
``asks.summary`` holds a checked caption, under the same expiry as its ask.

The log line for an ask names its intent, the validator's fixed codes, the result count and the
time taken, never the question.
"""

from __future__ import annotations

import json
import logging
import re
import secrets
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from ..export_format import iso, parse_dt
from ..page.app import Refused
from . import keywords, model
from . import query as q_mod
from . import sql as sql_mod
from .answer import BANNER, PER_WINDOW, REFUSALS, Answer, Step, ms_of, run

__all__ = ["BANNER", "Answer", "Asked", "Step", "edit", "history", "model", "one", "run", "submit", "thread"]

# The keyword steps' stored times (F3): ``notes["ms"]`` names each step that ran when the ask was
# asked ("words": matching words or taking the edited chips, "run": reading the shown answers) and,
# once a model step ran, "read" and "caption". A thread shows these; it never times a re-run.
_KEYWORD_MS = ("words", "run")


def _ms_since(t: float) -> int:
    return round((time.monotonic() - t) * 1000)

QUESTION_MAX = 300
TOKENS_MAX = 60
TOKEN_MAX = 160
WINDOW_ID_MAX = 200
ASK_DAYS = 14
THREAD_ID = re.compile(r"[0-9a-f]{16}")
_NOT_A_CHIP = ("intent:", "follow_up:", "reason:")

log = logging.getLogger("chatbrain.ask")


@dataclass
class Asked:
    """One ask in a thread, re-run now."""

    id: int
    thread_id: str
    echo: str  # the stored question, "[refused: …]", or what a chip edit changed
    answer: Answer
    refused: bool = False
    tokens: list[str] = field(default_factory=list)  # the query as tokens (the chip row posts them)
    addable: list[tuple[str, list[tuple[str, str]]]] = field(default_factory=list)  # chip menu: group -> (token, text)
    # the local model's parts (``model.view``)
    pending: str | None = None  # the step the page script still posts: "read" or "caption"
    waiting: str | None = None  # the waiting line, e.g. "Reading question · waking the local model…"
    stopped_text: str | None = None  # the waiting line after Stop
    slow: str | None = None  # the CPU line while a step waits
    skeleton: bool = False  # a caption slot waits for step 2
    refined: bool = False  # step 1 swapped the chips
    caption: list[Any] = field(default_factory=list)  # text and (n, window_id) citations
    no_caption: str | None = None  # why there is no caption, when one was wanted


def valid_thread(thread_id: str | None) -> bool:
    return thread_id is not None and THREAD_ID.fullmatch(thread_id) is not None


def ask_expires(page: Any, ids: list[str], created: Any, now: Any | None = None,
                query: dict[str, Any] | None = None) -> Any:  # fmt: skip
    """When an ask expires (B26): 14 days after it was asked, or the earliest text expiry among its
    result windows whose text still shows (at ``now``, default ``created``), if that is sooner.

    With a ``phrase`` in ``query``, also the earliest text expiry of every window whose kept text
    holds the phrase (the result IDs are capped; the phrase is chat text of each of them)."""
    now = created if now is None else now
    out = created + timedelta(days=ASK_DAYS)
    for w in page.windows(ids):
        if w.visible(now):
            out = min(out, w.expires)
    phrase = (query or {}).get("phrase")
    if phrase:
        out = min(out, _phrase_expires(page, phrase, now, out))
    return out


_PHRASE_EXPIRES = (
    "SELECT MIN(w.text_expires_at) FROM windows w WHERE w.text_state = 'present' AND w.text_expires_at > ?"
    " AND EXISTS (SELECT 1 FROM messages m WHERE m.window_id = w.window_id AND lower(m.text) LIKE ? ESCAPE '!')"
)
_TEXT_EXPIRES = "SELECT MIN(text_expires_at) FROM windows WHERE text_state = 'present' AND text_expires_at > ?"


def _phrase_expires(page: Any, phrase: str, now: Any, default: Any) -> Any:
    """The earliest text expiry among windows whose kept text holds ``phrase`` (``default`` if none).

    Read like an answer (B14). If that read is too slow or refused, the earliest text expiry of any
    window with text is used instead: sooner, never later, than the phrase's own."""
    now_s = iso(now)
    with sql_mod.readonly(page.store.path) as ro:
        try:
            row = ro.one(_PHRASE_EXPIRES, [now_s, sql_mod.like_pattern(phrase)])
        except (sql_mod.TooSlow, sql_mod.Denied):
            try:
                row = ro.one(_TEXT_EXPIRES, [now_s])
            except (sql_mod.TooSlow, sql_mod.Denied):
                return now  # can't tell: expire at once rather than outlive the text
    return parse_dt(row[0], "text_expires_at") if row and row[0] else default


def _store(page: Any, *, thread_id: str, now: Any, question: str | None, query: dict[str, Any],
           window_id: str | None, notes: dict[str, Any], ids: list[str]) -> int:  # fmt: skip
    stored = {"query": query, "window_id": window_id, "notes": notes}
    return page.store.add_ask(
        thread_id=thread_id,
        created_at=now,
        expires_at=ask_expires(page, ids, now, query=query),
        question=question,
        query_json=json.dumps(stored, ensure_ascii=False, sort_keys=True),
        follow_up=query["follow_up"],
        result_window_ids=ids,
    )


def submit(page: Any, question: Any, *, thread_id: str | None = None, window_id: str | None = None) -> tuple[str, int]:
    """Parse, check, run and store one typed question; returns ``(thread_id, ask_id)``."""
    started = time.monotonic()
    if not isinstance(question, str) or len(question) > QUESTION_MAX:
        raise Refused(400, f"Not asked: a question is at most {QUESTION_MAX} characters.")
    text = " ".join(question.split())
    if not text:
        raise Refused(400, "Not asked: type a question first.")
    if thread_id is not None and not valid_thread(thread_id):
        raise Refused(400, "Not asked: that thread link is broken.")
    if window_id is not None and (len(window_id) > WINDOW_ID_MAX or not page.exists(window_id)):
        raise Refused(404, "No such window.")
    now = page.now()
    held = q_mod.held(page.store)
    earlier = page.store.asks_in_thread(thread_id, now) if thread_id else []
    last = earlier[-1] if earlier else None
    previous = json.loads(last["query_json"])["query"] if last else None
    previous_ids = json.loads(last["result_window_ids_json"]) if last else []

    words = time.monotonic()
    parsed = keywords.parse(text, page.scheme, held, previous=previous, window_id=window_id)
    # a phrase carried from the earlier ask was checked against that ask's question; a new one must
    # appear in this question word for word
    carried = previous.get("phrase") if previous else None
    checked = q_mod.check(parsed.query, page.scheme, held, question=text, carried_phrase=carried)
    words_ms = _ms_since(words)
    q = checked.query
    refused = parsed.refused
    if refused is None and q["intent"] == "cant_answer" and q["reason"] in REFUSALS:
        refused = q["reason"]

    wid = window_id
    if not refused and q.get("ref") is not None and wid is None:
        if 0 < q["ref"] <= len(previous_ids):
            wid = previous_ids[q["ref"] - 1]
        else:
            q = q_mod.blank() | {"intent": "cant_answer", "reason": "unresolved_ref", "follow_up": q["follow_up"]}
    if q["intent"] not in PER_WINDOW:
        wid = None

    if refused:
        notes: dict[str, Any] = {}  # nothing from the text is kept
    else:
        final = set(q_mod.tokens(q))  # "kept" only for what survived the check
        notes = {"kept": [t for t in parsed.kept if t in final], "ignored": checked.ignored,
                 "didnt_understand": parsed.leftover}  # fmt: skip
        if parsed.default:
            notes["default_reading"] = True
        if parsed.viewers:
            notes["viewers"] = True
    answer = run(page, q, window_id=wid, notes=notes)
    if answer.closest:
        notes["closest"] = True  # the stored ask keeps its Moment; a re-run still says it wasn't at that time
    if not refused:
        notes["ms"] = {"words": words_ms, "run": round(answer.seconds * 1000)}  # a refusal keeps nothing
        model.start(page, notes, answer, window_id)  # step 1 waits for the page script, when the model can run
    ids = [] if refused else answer.ids
    thread = thread_id or secrets.token_hex(8)
    ask_id = _store(
        page, thread_id=thread, now=now, question=f"[refused: {refused}]" if refused else text, query=q,
        window_id=answer.window_id if not refused else None, notes=notes, ids=ids,
    )  # fmt: skip
    log.info(
        "ask intent=%s codes=%s results=%d ms=%d",
        q["intent"] if not refused else f"refused:{refused}",
        ",".join(checked.codes) or "-",
        len(ids),
        round((time.monotonic() - started) * 1000),
    )
    return thread, ask_id


def edit(page: Any, ask_id: int, toks: list[str], *, drop: str | None = None, add: str | None = None) -> tuple[str, int]:
    """A chip edit or a follow-up: the ask's tokens (or ``toks``), less ``drop``, plus ``add``,
    stored as a ``refine`` ask in the same thread with no question."""
    started = time.monotonic()
    if len(toks) > TOKENS_MAX or any(len(t) > TOKEN_MAX for t in [*toks, drop or "", add or ""]):
        raise Refused(400, "Not asked: too many chips.")
    now = page.now()
    row = page.store.ask(ask_id, now)
    if row is None:
        raise Refused(404, "No such ask: it may have expired.")
    stored = json.loads(row["query_json"])
    old = q_mod.tokens(stored["query"])
    new = [t for t in (toks or old) if t != drop]
    if add and add not in new:
        new.append(add)
    took = time.monotonic()
    checked = q_mod.from_tokens(new, page.scheme, q_mod.held(page.store))
    words_ms = _ms_since(took)
    q = checked.query | {"follow_up": "refine"}
    now_toks = q_mod.tokens(q)
    chip = [t for t in now_toks if not t.startswith(_NOT_A_CHIP)]
    was = [t for t in old if not t.startswith(_NOT_A_CHIP)]
    notes = {
        "kept": [t for t in was if t in chip],
        "ignored": checked.ignored,
        "edited": {"added": [t for t in chip if t not in was], "dropped": [t for t in was if t not in chip]},
    }
    wid = stored.get("window_id") if q["intent"] in PER_WINDOW else None
    answer = run(page, q, window_id=wid, notes=notes)
    if answer.closest:
        notes["closest"] = True
    refused = q["intent"] == "cant_answer" and q["reason"] in REFUSALS
    if not refused:
        notes["ms"] = {"words": words_ms, "run": round(answer.seconds * 1000)}
        model.after_edit(page, notes, answer)  # edited chips skip step 1
    ids = [] if refused else answer.ids
    new_id = _store(
        page, thread_id=row["thread_id"], now=now, question=None, query=q, window_id=answer.window_id, notes=notes,
        ids=ids,
    )  # fmt: skip
    log.info(
        "ask intent=%s codes=%s results=%d ms=%d", q["intent"], ",".join(["edit", *checked.codes]), len(ids),
        round((time.monotonic() - started) * 1000),
    )  # fmt: skip
    return row["thread_id"], new_id


def _echo(page: Any, row: dict[str, Any], notes: dict[str, Any]) -> str:
    if row["question"] is not None:
        return row["question"]
    e = notes.get("edited") or {}
    parts = [f"+ {q_mod.chip_text(t, page.scheme)}" for t in e.get("added", [])]
    parts += [f"− {q_mod.chip_text(t, page.scheme)}" for t in e.get("dropped", [])]
    return "Edited the chips" + (": " + "; ".join(parts) if parts else "")


def _addable(page: Any, q: dict[str, Any]) -> list[tuple[str, list[tuple[str, str]]]]:
    """The chip menu: value, ``is`` and time tokens the query doesn't hold yet."""
    have = set(q_mod.tokens(q))
    groups: list[tuple[str, list[tuple[str, str]]]] = []
    for key in q_mod.VALUE_KEYS:
        opts = [(f"{key}:{v}", page.scheme.display.get(v, v)) for v in q_mod.allowed(page.scheme, key)]
        groups.append((q_mod.KEY_TEXT[key], [o for o in opts if o[0] not in have]))
    groups.append(("is", [(f"is:{v}", t) for v, t in q_mod.IS_TEXT.items() if f"is:{v}" not in have]))
    groups.append(("time", [(f"time:{v}", t) for v, t in q_mod.TIME_TEXT.items() if f"time:{v}" not in have]))
    return [(g, opts) for g, opts in groups if opts]


def _asked(page: Any, row: dict[str, Any]) -> Asked:
    """One stored ask, re-run now, with its model parts."""
    stored = json.loads(row["query_json"])
    notes = stored.get("notes") or {}
    q = stored["query"]
    a = run(page, q, window_id=stored.get("window_id"), notes=notes)
    refused = bool(row["question"] and row["question"].startswith("[refused: "))
    ms = notes.get("ms") or {}
    for key, s in zip(_KEYWORD_MS, a.steps):  # the times stored when it was asked; a re-run is never timed
        s.ms = ms_of(ms.get(key))
    x = Asked(
        id=row["id"], thread_id=row["thread_id"], echo=_echo(page, row, notes), answer=a, refused=refused,
        tokens=q_mod.tokens(q), addable=_addable(page, q),
    )  # fmt: skip
    model.view(page, x, row, notes)
    if refused or a.refused:
        a.steps = []  # a refusal shows no steps (F3): nothing from its words ran
    return x


def thread(page: Any, thread_id: str) -> list[Asked]:
    """The thread's live asks, oldest first, each re-run now."""
    return [_asked(page, row) for row in page.store.asks_in_thread(thread_id, page.now())]


def one(page: Any, ask_id: int) -> Asked:
    """One live ask, re-run now (the page script swaps it in after a model step)."""
    row = page.store.ask(ask_id, page.now())
    if row is None:
        raise Refused(404, "No such ask: it may have expired.")
    return _asked(page, row)


def history(page: Any) -> list[tuple[str, str]]:
    """Recent threads, newest first, by their first question (``[refused: …]`` for a refused one)."""
    return [(t, first or "Edited the chips") for t, first in page.store.recent_threads(page.now())]
