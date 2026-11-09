"""The local model steps of an ask (Task 10c): read the question, then write a short caption.

The keyword path (10b) answers first and is stored at once; its chips and cards show while the
model works. ``start`` marks a stored ask as waiting on step 1 when the model can run, and the
page script then posts ``/ask/<id>/model`` once per step (``advance``):

- **read** (step 1): ``Narrator.interpret`` turns the stored question, the previous query and
  the previous rows (refs, never text) into a query object, and ``query.check`` validates it.
  The same chips keep the keyword answer ("same"); a different valid query swaps the chips with
  the "refined" mark and re-runs the SQL ("refined"); an unusable reading keeps the keyword
  chips; a refusal overwrites the stored words with ``[refused: <reason>]``.
- **caption** (step 2): only for one Moment, an emote's meaning or a vague question, and not
  while the model runs on the CPU unless the page was started with ``--cpu-captions``.
  ``Narrator.summarize`` gets the chips in words, numbered facts, the not-sure answers, the cause
  note and the Moment's evidence lines; never the question. A checked caption is stored in
  ``asks.summary`` with the facts and the chat lines (message IDs) it was written from; when a
  later fix changes those facts, or a line is gone, the caption is hidden instead of shown stale.
  A deletion check that drops a window's messages clears the stored caption of every ask over that
  window (``ingest.apply_deletion_check``), and a caption whose line a check dropped while it was
  being written is not stored.

Anything that fails keeps the keyword answer: Ollama off or too old, the model missing, a
base URL that isn't loopback, a step past its deadline (20 s on the GPU, 60 s on the CPU), or
Stop. ``stop`` records the stop in the stored ask first and then ends a running step between
chunks, so a stopped ask never takes a late model result. One model step runs at a time
(``page.model_lock``); a step's write and Stop's write take ``page.ask_lock``.

``notes["model"]`` holds the progress: ``stage`` (``read``, ``caption`` or ``done``), the
``read`` and ``caption`` outcomes, and while step 1 waits the Moment the ask was posted from.
Each step's time is measured when it runs and stored apart, in ``notes["ms"]`` (F3); a thread
shows the stored times and never times a step again.
Log lines carry the step, the outcome, the intent, the validator's codes, the result count and
the time; never the question, a prompt or model output.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .. import narrator as nar
from .. import ollama
from ..export_format import parse_dt
from ..page import app as app_mod
from ..page.app import Refused
from . import query as q_mod
from .answer import ABOUT_LEAD, EMOTE_MIN, PER_WINDOW, REFUSALS, Answer, Step, StepState, _size, ms_of, run

log = logging.getLogger("chatbrain.ask")

WAITING = ("read", "caption")  # the stages a page script still posts for
NO_ASK = "No such ask: it may have expired."
LOCK_POLL_S = 0.2
REFS_MAX = 20  # previous rows sent to step 1 as refs

HEAD_OFF = "Asking is limited: the local model isn't running."
HEAD_TIMEOUT = "Asking is limited: the local model didn't answer in time."
BODY = "Words were matched to filters; check the chips."
SLOW_DOCK = "Local model on the CPU · about half a minute per answer."
CAPTIONS_OFF = " Captions are off."
SLOW_ASK = "Still working · local model on the CPU, about 30 s. The line above is counted and won't change."

READ_STOPPED = "Reading question · stopped. The keyword answer stays."
CAPTION_STOPPED = "Writing summary · stopped. The template answer stays."
STALE = "The caption was for earlier numbers, so it is hidden."
WROTE = "Wrote a caption with the local model, checked sentence by sentence"

_NOT_RUNNING = "The local model wasn't running; words were matched to filters"
_COULDNT = "The local model couldn't read the question; the keyword chips stay"
READ_LINE = {
    "same": "Read the question with the local model: same chips",
    "refined": "Read the question with the local model: refined the chips",
    "refused": "Read the question with the local model: refused",
    "unusable": _COULDNT,
    "failed": _COULDNT,
    "timeout": "The local model didn't answer in time; the keyword answer stays",
    "busy": "The local model didn't answer in time a moment ago; words were matched to filters",
    "off": _NOT_RUNNING,
    "old": _NOT_RUNNING,
    "missing": _NOT_RUNNING,
    "not_loopback": _NOT_RUNNING,
}
FALLBACK = ("unusable", "failed", "timeout", "busy", "off", "old", "missing", "not_loopback", "stopped")
# How a step 1 row ends (F3's icons): it ran and didn't finish, or it never ran; else done.
READ_STATE: dict[str, StepState] = {
    "unusable": "failed", "failed": "failed", "timeout": "failed", "busy": "skipped", "off": "skipped",
    "old": "skipped", "missing": "skipped", "not_loopback": "skipped",
}  # fmt: skip
NO_CAPTION = {
    "off": "No caption while the model is off.",
    "busy": "No caption while the local model is busy.",
    "cpu": "No caption: captions are off while the model runs on the CPU.",
    "none_written": "No caption this time; the template line stays.",
    "timeout": "No caption: the local model didn't answer in time.",
    "stopped": CAPTION_STOPPED,
}
_CITE = re.compile(r"\[(\d+)\]")


# --- the dock --------------------------------------------------------------------------------------


def _fix(problem: str | None, model: str) -> tuple[str, str | None, str] | None:
    """The banner's fix as (text before, code, text after)."""
    return {
        "off": ("Start Ollama with ", "ollama serve", "."),
        "old": ("Update Ollama to 0.31.2 or newer.", None, ""),
        "missing": ("Get the model with ", f"ollama pull {model}", "."),
        "not_loopback": ("BrainChat uses Ollama only on this PC (127.0.0.1).", None, ""),
        "timeout": ("BrainChat asks the model again in half a minute.", None, ""),
    }.get(problem or "")


def dock(page: Any) -> dict[str, Any]:
    """What the dock says about the model: the amber banner only on a fallback (with its fix),
    and the CPU line when the model runs there."""
    nr = page.narrator
    if nr is None:  # ``page --no-model``: 10b's banner
        return {"banner": True, "head": HEAD_OFF, "body": BODY, "fix": None, "slow": None}
    st = nr.state()
    if not st.ok:
        head = HEAD_TIMEOUT if st.problem == "timeout" else HEAD_OFF
        return {"banner": True, "head": head, "body": BODY, "fix": _fix(st.problem, nr.model), "slow": None}
    slow = SLOW_DOCK + ("" if page.cpu_captions else CAPTIONS_OFF) if st.cpu else None
    return {"banner": False, "head": None, "body": None, "fix": None, "slow": slow}


# --- what an ask waits on --------------------------------------------------------------------------


def _read_problem(problem: str | None) -> str:
    return "busy" if problem == "timeout" else (problem or "off")


def _caption_problem(problem: str | None) -> str:
    return "busy" if problem == "timeout" else "off"


def wants_words(answer: Answer, notes: dict[str, Any]) -> bool:
    """A caption helps one Moment, an emote's meaning and a vague question; never a list."""
    if answer.refused or answer.query.get("intent") == "cant_answer":
        return False
    for b in answer.blocks:
        if b["kind"] in ("moment", "evidence"):
            return True
        if b["kind"] == "mix" and b["n"] >= EMOTE_MIN:
            return True
        if b["kind"] == "count" and notes.get("default_reading") and b["rows"] and b["rows"][0]["n"]:
            return True
    return False


def _captions_on(page: Any, st: nar.State) -> bool:
    return not st.cpu or page.cpu_captions


def start(page: Any, notes: dict[str, Any], answer: Answer, window_id: str | None) -> None:
    """A new typed ask: wait on step 1 when the model can run, else record why it didn't."""
    nr = page.narrator
    if nr is None:
        return
    st = nr.state()
    if st.ok:
        m: dict[str, Any] = {"stage": "read"}
        if window_id is not None:
            m["window"] = window_id  # the Moment the ask was posted from, for a refined per-Moment query
    else:
        m = {"stage": "done", "read": _read_problem(st.problem)}
        if wants_words(answer, notes):
            m["caption"] = _caption_problem(st.problem)
    notes["model"] = m


def after_edit(page: Any, notes: dict[str, Any], answer: Answer) -> None:
    """A chip edit skips step 1; it still gets a caption when its answer asks for words."""
    nr = page.narrator
    if nr is None:
        return
    m: dict[str, Any] = {"stage": "done", "read": "edited"}
    if wants_words(answer, notes):
        st = nr.state()
        if not st.ok:
            m["caption"] = _caption_problem(st.problem)
        elif not _captions_on(page, st):
            m["caption"] = "cpu"
        else:
            m["stage"] = "caption"
    notes["model"] = m


# --- step 1 ----------------------------------------------------------------------------------------


def previous_rows(page: Any, ids: list[str]) -> list[str]:
    """The previous answer's rows as refs: ``#2 = fakechan01 21:14 laughing``; no text, no speakers."""
    ids = ids[:REFS_MAX]
    if not ids:
        return []
    by = {m.win.window_id: m for m in page.moments(page.windows(ids), page.now(), page.plan())}
    return [
        nar.ref_row(i, by[w].win.channel, app_mod.hm(by[w].win.anchor), by[w].first_doing)
        for i, w in enumerate(ids, 1)
        if w in by
    ]


def _previous(page: Any, row: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    """The thread's ask before ``row``: its query and result window IDs."""
    before = None
    for r in page.store.asks_in_thread(row["thread_id"], page.now()):
        if r["id"] == row["id"]:
            break
        before = r
    if before is None:
        return None, []
    return json.loads(before["query_json"])["query"], json.loads(before["result_window_ids_json"])


def _same(a: dict[str, Any], b: dict[str, Any]) -> bool:
    def key(q: dict[str, Any]) -> list[str]:
        return sorted(t for t in q_mod.tokens(q) if not t.startswith("follow_up:"))

    return key(a) == key(b)


@dataclass
class _Write:
    """What a step stores: the ask's fields after it (unchanged ones copied from the row)."""

    stored: dict[str, Any]
    question: str | None
    ids: list[str]
    summary: str | None
    outcome: str
    intent: str
    codes: list[str] = field(default_factory=list)


def _read(page: Any, nr: nar.Narrator, row: dict[str, Any], stored: dict[str, Any], ev: threading.Event) -> _Write:
    notes = dict(stored.get("notes") or {})
    m = dict(notes.get("model") or {})
    q0 = stored["query"]
    ids0 = json.loads(row["result_window_ids_json"])
    keep = _Write(stored, row["question"], ids0, row["summary"], "", q0["intent"])
    st = nr.state()
    checked = None
    previous, prev_ids = None, []
    if not st.ok:
        keep.outcome = _read_problem(st.problem)
    else:
        held = q_mod.held(page.store)
        previous, prev_ids = _previous(page, row)
        try:
            raw = nr.interpret(
                row["question"] or "", schema=q_mod.schema(page.store, page.scheme, held), previous=previous,
                previous_rows=previous_rows(page, prev_ids), scheme=page.scheme, stop=ev,
            )  # fmt: skip
        except nar.Stopped:
            keep.outcome = "stopped"
        except nar.StepTimeout:
            nr.note_timeout()
            keep.outcome = "timeout"
        except ollama.OllamaError:
            fresh = nr.state(fresh=True)
            keep.outcome = _read_problem(fresh.problem) if not fresh.ok else "failed"
        except nar.StepFailed:
            keep.outcome = "failed"
        else:
            carried = previous.get("phrase") if previous else None
            checked = q_mod.check(raw, page.scheme, held, question=row["question"], carried_phrase=carried)
            keep.codes = list(checked.codes)

    if checked is not None and "unusable" not in checked.codes:
        q = checked.query
        if q["intent"] == "cant_answer" and q.get("reason") in REFUSALS:
            refused = q_mod.blank() | {
                "intent": "cant_answer", "reason": q["reason"], "at": q.get("at"), "time": q["time"],
                "follow_up": q["follow_up"],
            }  # fmt: skip
            out = {"query": refused, "window_id": None, "notes": {"model": {"stage": "done", "read": "refused"}}}
            return _Write(out, f"[refused: {q['reason']}]", [], None, "refused", f"refused:{q['reason']}", keep.codes)
        if _same(q, q0):
            keep.outcome = "same"
        else:
            return _refined(page, nr, st, notes, m, q, checked, previous, prev_ids, keep)
    elif checked is not None:
        keep.outcome = "unusable"

    # the keyword answer stays
    m = {"stage": "done", "read": keep.outcome}
    if keep.outcome in ("same", "unusable", "failed", "timeout", "off", "old", "missing", "not_loopback", "busy"):
        a = run(page, q0, window_id=stored.get("window_id"), notes=notes)
        if wants_words(a, notes):
            if keep.outcome in ("same", "unusable", "failed"):
                if _captions_on(page, st):
                    m["stage"] = "caption"
                else:
                    m["caption"] = "cpu"
            else:
                m["caption"] = "timeout" if keep.outcome == "timeout" else _caption_problem(st.problem or "off")
    keep.stored = stored | {"notes": notes | {"model": m}}
    return keep


def _refined(
    page: Any, nr: nar.Narrator, st: nar.State, notes: dict[str, Any], m: dict[str, Any], q: dict[str, Any],
    checked: q_mod.Checked, previous: dict[str, Any] | None, prev_ids: list[str], keep: _Write,
) -> _Write:  # fmt: skip
    """Step 1 read the question differently: the model's chips, and the SQL re-run on them."""
    toks = q_mod.tokens(q)
    if previous is not None and q["follow_up"] != "new":
        before, blank = set(q_mod.tokens(previous)), set(q_mod.tokens(q_mod.blank()))
        kept = [t for t in toks if t in before and t not in blank]
    else:
        kept = []
    new: dict[str, Any] = {"kept": kept, "ignored": checked.ignored, "didnt_understand": []}
    if notes.get("viewers"):
        new["viewers"] = True
    if notes.get("default_reading") and q["intent"] == "count" and q.get("about") == "doing" and q.get("by") == "value":
        new["default_reading"] = True
    wid = m.get("window")
    if q.get("ref") is not None and wid is None:
        if 0 < q["ref"] <= len(prev_ids):
            wid = prev_ids[q["ref"] - 1]
        else:
            q = q_mod.blank() | {"intent": "cant_answer", "reason": "unresolved_ref", "follow_up": q["follow_up"]}
    if q["intent"] not in PER_WINDOW:
        wid = None
    a = run(page, q, window_id=wid, notes=new)
    if a.closest:
        new["closest"] = True
    done: dict[str, Any] = {"stage": "done", "read": "refined"}
    if wants_words(a, new):
        if _captions_on(page, st):
            done["stage"] = "caption"
        else:
            done["caption"] = "cpu"
    new["model"] = done
    out = {"query": q, "window_id": a.window_id, "notes": new}
    return _Write(out, keep.question, list(a.ids), keep.summary, "refined", q["intent"], keep.codes)


# --- step 2 ----------------------------------------------------------------------------------------


@dataclass
class CaptionInput:
    chips_in_words: str
    facts: dict[str, str]
    not_sure: dict[str, list[str]]
    cause_note: str | None
    evidence: list[str]
    cites: list[str]  # window IDs: a caption's [n] opens cites[n - 1]
    asks_cause: bool
    msgs: list[list[str]] = field(default_factory=list)  # [window_id, message_id] of each evidence line


def _chips_in_words(answer: Answer) -> str:
    out = []
    for c in answer.chips:
        if c.kind != "filter":
            continue
        out.append("said: a quoted phrase" if c.token.startswith("phrase:") else c.text)  # the phrase is the question's
    return "; ".join(out)


def _cards(answer: Answer) -> list[str]:
    for b in answer.blocks:
        if b["kind"] == "cards":
            return [m.win.window_id for m in b["moments"]]
    return []


def caption_input(page: Any, answer: Answer, notes: dict[str, Any]) -> CaptionInput | None:
    """Step 2's input, built from the answer on screen: never the question, never a speaker."""
    words = _chips_in_words(answer)
    asks_cause = bool(answer.query.get("asks_cause"))
    for b in answer.blocks:
        if b["kind"] in ("moment", "evidence"):
            m = b["m"]
            w = m.win
            facts = {"F1": app_mod.clock_text(w.anchor), "F2": _size(w, "posters"), "F3": _size(w, "lines"),
                     "F4": w.channel}  # fmt: skip
            not_sure: dict[str, list[str]] = {}
            for qk in page.questions:
                s = m.answers.get(qk)
                if s is None:
                    continue
                if s.sure:
                    said = ", ".join(page.scheme.display.get(v, v) for v in s.values)
                    facts[f"F{len(facts) + 1}"] = ABOUT_LEAD.get(qk, f"{qk} ") + said
                else:
                    not_sure[qk] = list(s.not_sure)
            evidence, msgs, seen = [], [], set()
            for qk in page.questions:
                for line in m.evidence.get(qk, ()):
                    msg = line["msg"]
                    if msg.message_id not in seen:
                        seen.add(msg.message_id)
                        evidence.append(msg.text)
                        msgs.append([w.window_id, msg.message_id])
            return CaptionInput(words, facts, not_sure, m.cause.note if m.cause else None,
                                evidence[: nar.EVIDENCE_MAX], [w.window_id], asks_cause,
                                msgs[: nar.EVIDENCE_MAX])  # fmt: skip
        if b["kind"] == "mix":
            n = b["n"]
            facts = {"F1": answer.query["emote"][0], "F2": app_mod.plural(n, "window"),
                     "F3": app_mod.plural(b["moments"], "Moment")}  # fmt: skip
            for label, rows in b["groups"]:
                for r in rows[:2]:
                    facts[f"F{len(facts) + 1}"] = f"{label.lower()}: {r['label']} in {r['k']} of {app_mod.plural(n, 'window')}"
            return CaptionInput(words, facts, {}, None, [], _cards(answer), asks_cause)
        if b["kind"] == "count" and notes.get("default_reading") and b["rows"]:
            n = b["rows"][0]["n"]
            facts = {"F1": app_mod.plural(n, "Moment")}
            named = [r for r in b["rows"] if r["label"] != "not sure"][:4]
            named += [r for r in b["rows"] if r["label"] == "not sure"]
            for r in named:
                facts[f"F{len(facts) + 1}"] = f"{r['label']} in {r['k']} of {app_mod.plural(n, 'Moment')}"
            return CaptionInput(words, facts, {}, None, [], _cards(answer), asks_cause)
    return None


def _caption(page: Any, nr: nar.Narrator, row: dict[str, Any], stored: dict[str, Any], ev: threading.Event) -> _Write:
    notes = dict(stored.get("notes") or {})
    m = dict(notes.get("model") or {})
    q = stored["query"]
    ids = json.loads(row["result_window_ids_json"])
    w = _Write(stored, row["question"], ids, row["summary"], "", q["intent"])
    st = nr.state()
    a = run(page, q, window_id=stored.get("window_id"), notes=notes)
    ci = caption_input(page, a, notes) if wants_words(a, notes) else None
    text = None
    if not st.ok:
        w.outcome = _caption_problem(st.problem)
    elif not _captions_on(page, st):
        w.outcome = "cpu"
    elif ci is None:
        w.outcome = "none_written"
    else:
        try:
            text = nr.summarize(
                chips_in_words=ci.chips_in_words, facts=ci.facts, not_sure=ci.not_sure, cause_note=ci.cause_note,
                evidence=ci.evidence, asks_cause=ci.asks_cause, cites=len(ci.cites), stop=ev,
            )  # fmt: skip
        except nar.Stopped:
            w.outcome = "stopped"
        except nar.StepTimeout:
            w.outcome = "timeout"
        except ollama.OllamaError:
            fresh = nr.state(fresh=True)
            w.outcome = _caption_problem(fresh.problem) if not fresh.ok else "none_written"
        else:
            w.outcome = "written" if text else "none_written"
    new = dict(notes)
    if text:
        assert ci is not None
        w.summary = text
        new["caption_for"] = {"facts": ci.facts, "cites": ci.cites, "msgs": ci.msgs}  # what it was written from
    new["model"] = {"stage": "done", "read": m.get("read", "edited"), "caption": w.outcome}
    w.stored = stored | {"notes": new}
    return w


# --- one step, and Stop ----------------------------------------------------------------------------


def _stage(row: dict[str, Any] | None) -> str | None:
    if row is None:
        return None
    return ((json.loads(row["query_json"]).get("notes") or {}).get("model") or {}).get("stage")


def _rewrite(page: Any, row: dict[str, Any], stored: dict[str, Any], *, question: str | None, ids: list[str],
             summary: str | None) -> None:  # fmt: skip
    from . import ask_expires  # the package's expiry rule (B26); imported here, the package imports this module

    created = parse_dt(row["created_at"], "created_at")
    made = (stored.get("notes") or {}).get("caption_for") or {}
    page.store.update_ask(
        row["id"], question=question, query_json=json.dumps(stored, ensure_ascii=False, sort_keys=True),
        follow_up=stored["query"]["follow_up"], result_window_ids=ids,
        expires_at=ask_expires(page, ids, created, page.now(), query=stored["query"]), summary=summary,
        summary_msgs=made.get("msgs") or (),  # a line a deletion check dropped meanwhile: the caption isn't stored
    )  # fmt: skip


def advance(page: Any, ask_id: int) -> None:
    """Run the one model step ``ask_id`` waits on, if any, and store its result."""
    started = time.monotonic()
    row = page.store.ask(ask_id, page.now())
    if row is None:
        raise Refused(404, NO_ASK)
    nr = page.narrator
    if nr is None or _stage(row) not in WAITING:
        return
    ev = page.stop_event(ask_id)
    try:
        while not page.model_lock.acquire(timeout=LOCK_POLL_S):
            if ev.is_set():
                return  # Stop already stored the ask as stopped
        try:
            with page.ask_lock:
                row = page.store.ask(ask_id, page.now())
            stage = _stage(row)
            if row is None or stage not in WAITING or ev.is_set():
                return
            stored = json.loads(row["query_json"])
            ran = time.monotonic()
            w = (_read if stage == "read" else _caption)(page, nr, row, stored, ev)
            took = round((time.monotonic() - ran) * 1000)
            with page.ask_lock:
                now_row = page.store.ask(ask_id, page.now())
                if now_row is None or _stage(now_row) != stage or ev.is_set():
                    return  # Stop (or expiry) came first: the late result is dropped
                if w.outcome != "refused":  # a refusal keeps nothing, not even times
                    ms = dict((stored.get("notes") or {}).get("ms") or {}) | {stage: took}
                    w.stored = w.stored | {"notes": (w.stored.get("notes") or {}) | {"ms": ms}}
                _rewrite(page, row, w.stored, question=w.question, ids=w.ids, summary=w.summary)
        finally:
            page.model_lock.release()
        log.info(
            "ask model step=%s outcome=%s intent=%s codes=%s results=%d ms=%d", stage, w.outcome, w.intent,
            ",".join(w.codes) or "-", len(w.ids), round((time.monotonic() - started) * 1000),
        )  # fmt: skip
    finally:
        page.drop_stop(ask_id, ev)


def stop(page: Any, ask_id: int) -> None:
    """Stop: store the ask as stopped (the keyword or template answer stays), then end a running
    step between chunks. An ask that waits on nothing is left as it is."""
    with page.ask_lock:
        row = page.store.ask(ask_id, page.now())
        if row is None:
            raise Refused(404, NO_ASK)
        stored = json.loads(row["query_json"])
        notes = dict(stored.get("notes") or {})
        m = notes.get("model") or {}
        stage = m.get("stage")
        if stage not in WAITING:
            return
        if stage == "read":
            notes["model"] = {"stage": "done", "read": "stopped"}
        else:
            notes["model"] = {"stage": "done", "read": m.get("read", "edited"), "caption": "stopped"}
        _rewrite(page, row, stored | {"notes": notes}, question=row["question"],
                 ids=json.loads(row["result_window_ids_json"]), summary=row["summary"])  # fmt: skip
    ev = page.stop_event(ask_id, create=False)
    if ev is not None:
        ev.set()
    log.info("ask model step=%s outcome=stopped", stage)


# --- the page's view of an ask ---------------------------------------------------------------------


def view(page: Any, asked: Any, row: dict[str, Any], notes: dict[str, Any]) -> None:
    """Fill ``asked``'s model parts: the waiting state, the step lines, the refined mark and the
    caption (or why there is none)."""
    m = notes.get("model")
    if not m:
        return
    a = asked.answer
    nr = page.narrator
    stage = m.get("stage")
    read, cap = m.get("read"), m.get("caption")
    ms = notes.get("ms") or {}
    if read in READ_LINE:
        state = READ_STATE.get(read, "done")
        a.steps.append(Step(READ_LINE[read], state, ms_of(ms.get("read")) if state != "skipped" else None))
    if read in FALLBACK:
        a.how = "keywords"
    asked.refined = read == "refined"
    if stage in WAITING and nr is not None:
        st = nr.state()
        asked.pending = stage
        what = "Reading question" if stage == "read" else "Writing summary"
        asked.waiting = what + ("…" if st.loaded else " · waking the local model…")
        asked.stopped_text = READ_STOPPED if stage == "read" else CAPTION_STOPPED
        asked.slow = SLOW_ASK if st.cpu else None
        asked.skeleton = stage == "caption" or (wants_words(a, notes) and _captions_on(page, st))
        return
    if read == "stopped":
        asked.no_caption = READ_STOPPED
        return
    if cap == "written":
        # Shown only while the facts it was written from still hold and every chat line it was
        # written from is still stored; a deletion check clears the stored caption (``ingest``).
        ci = caption_input(page, a, notes) if row["summary"] else None
        made = notes.get("caption_for") or {}
        msgs = made.get("msgs")
        if (ci is not None and made.get("facts") == ci.facts and made.get("cites") == ci.cites
                and msgs is not None and page.store.holds_messages(msgs)):  # fmt: skip
            a.steps.append(Step(WROTE, "done", ms_of(ms.get("caption"))))
            asked.caption = _parts(row["summary"], ci.cites)
        else:
            asked.no_caption = STALE
    elif cap in NO_CAPTION:
        asked.no_caption = NO_CAPTION[cap]


def _parts(summary: str, cites: list[str]) -> list[Any]:
    """The caption as text and ``(n, window_id)`` citations the page links."""
    out: list[Any] = []
    pos = 0
    for c in _CITE.finditer(summary):
        n = int(c.group(1))
        if 1 <= n <= len(cites):
            out.append(summary[pos : c.start()])
            out.append((n, cites[n - 1]))
            pos = c.end()
    out.append(summary[pos:])
    return [p for p in out if p != ""]
