"""Sync between the store and Label Studio: push tasks, pull annotations, adopt lost tasks.

``pull(store, client, scheme, now)`` turns every new annotation of the three projects
into an owner label (rulings C1, C35, B12):

- Skipped: a cancelled annotation (counted in ``skipped``) and an annotation whose
  ``ls_annotation_id`` is already stored, so pulling twice writes nothing twice.
- Refused (``invalid``, as ``(annotation ID, reason)``): a window the store doesn't
  hold; a task tracked in ``ls_tasks`` for another window or project; a blind or
  suggested window without an assignment of that exposure; a repeat without a
  ``repeats`` row; a result that names a field the view doesn't have, gives a field
  twice, gives a question both values and not-sure reasons or neither (a blank
  optional question takes its ``blank_value``), holds a value or reason outside the
  scheme, or has a cause note without a source (or the reverse); a ``created_at``
  without an offset; a ``lead_time`` that isn't a number. Reasons name fields and
  rules only, never a value, a window ID or chat text, because the CLI prints them.
- Written: ``answers`` and ``not_sure`` in scheme order (``validate_answers`` is the
  last check), the annotation's own ``created_at`` and ``lead_time``, and provenance:
  ``blind`` and ``repeat`` from their projects; in the suggested project, ``accepted``
  or ``corrected`` against the window's newest ``suggestions`` row (accepted = both
  dicts equal after filling ``tone``'s blank value), with ``suggestion_json``,
  ``teacher_model`` and ``prompt_version`` copied, or ``unsuggested`` when that row is
  an error or is unusable (C15).

``push(store, client, scheme, kind, limit, now)`` imports up to ``limit`` windows into
one project, one task per import, recording each ``ls_tasks`` row before the next
import (B12). A window is pushed once per project, only while its text lives
(``text_expires_at > now``), only with an assignment, and never after the owner labeled it
in that project:

- blind / suggested: the assignment's exposure picks the project; the suggested project
  also needs a ``suggestions`` row. Only suggested tasks carry a prediction (the newest
  suggestion), and an error-only suggestion is pushed without one.
- repeat: windows whose ``repeats.due_at <= now``, shuffled, never with a prediction.

``adopt_untracked`` records tasks Label Studio holds for known windows that
``ls_tasks`` lost (a crash between an import and its row), so expiry deletes them too;
tasks for windows the store doesn't hold are counted in ``unknown``. ``push`` adopts
before importing and again after a failed import.

The suggestion JSON contract (Task 7 writes it): ``suggestions.answers_json`` is
``{"answers": {...}, "not_sure": {...}}`` in the ``validate_answers`` shape.
"""

from __future__ import annotations

import json
import math
import random
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from .. import spread
from ..export_format import ExportRejected, iso, parse_dt
from ..scheme import Scheme, validate_answers, validate_cause_source
from ..store import Store
from . import config, tasks
from .client import KINDS


class Client(Protocol):
    """What sync needs from ``client.LabelStudio`` (tests pass a fake)."""

    def project_id(self, kind: str) -> int | None: ...

    def ensure_project(self, kind: str, label_config: str | None = None) -> int: ...

    def import_tasks(self, project_id: int, kind: str, tasks: list[dict]) -> list[int]: ...

    def export_annotations(self, project_id: int, *, all_tasks: bool = False) -> list[dict]: ...

    def use_model_version(self, project_id: int, model_version: str) -> None: ...


@dataclass(frozen=True)
class SyncResult:
    written: int
    invalid: list[tuple[int | None, str]] = field(default_factory=list)  # (annotation ID, reason)
    skipped: int = 0  # cancelled annotations


@dataclass(frozen=True)
class Adopted:
    recorded: int
    unknown: int  # tasks whose window the store doesn't hold


@dataclass(frozen=True)
class PushResult:
    pushed: int
    with_prediction: int
    adopted: int = 0
    unknown: int = 0
    error: str | None = None  # exception type of the failure that stopped the push


@dataclass(frozen=True)
class _Suggestion:
    raw: str
    answers: dict[str, Any]
    not_sure: dict[str, list[str]]
    teacher_model: str
    prompt_version: str


class _Invalid(Exception):
    """One annotation can't become a label; the message is safe to print."""


# --- pull -----------------------------------------------------------------------------------------------


def pull(store: Store, client: Client, scheme: Scheme, now: datetime) -> SyncResult:
    """Write a label for every new, valid annotation in the three projects (see the module docstring).

    ``now`` is kept for the call shape the CLI and expiry share; labels take the
    annotation's own ``created_at``.
    """
    written = skipped = 0
    invalid: list[tuple[int | None, str]] = []
    for kind in KINDS:
        project_id = client.project_id(kind)
        if project_id is None:
            continue
        exported = client.export_annotations(project_id)  # never under the store lock
        with store.transaction() as conn:
            for task in exported if isinstance(exported, list) else []:
                for ann in _annotations(task):
                    ann_id = ann.get("id")
                    if not _is_int(ann_id):
                        invalid.append((None, "an annotation has no ID"))
                        continue
                    if ann.get("was_cancelled") is True:
                        skipped += 1
                        continue
                    if conn.execute("SELECT 1 FROM labels WHERE ls_annotation_id = ?", (ann_id,)).fetchone():
                        continue
                    try:
                        conn.execute(*_label_insert(conn, scheme, kind, task, ann))
                    except _Invalid as exc:
                        invalid.append((ann_id, str(exc)))
                        continue
                    written += 1
    return SyncResult(written, invalid, skipped)


def _annotations(task: Any) -> list[dict]:
    if not isinstance(task, dict) or not isinstance(task.get("annotations"), list):
        return []
    return [a for a in task["annotations"] if isinstance(a, dict)]


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _label_insert(conn: sqlite3.Connection, scheme: Scheme, kind: str, task: dict, ann: dict) -> tuple[str, tuple]:
    window_id = _check_window(conn, kind, task)
    answers, not_sure, note, source = _parse_result(scheme, ann.get("result"))
    try:
        created_at = iso(parse_dt(ann.get("created_at"), "created_at"))
    except ExportRejected:
        raise _Invalid("created_at: must be a time with an explicit offset") from None
    lead = ann.get("lead_time")
    if lead is not None and (not isinstance(lead, int | float) or isinstance(lead, bool) or not math.isfinite(lead)
                             or lead < 0):
        raise _Invalid("lead_time: must be a number of seconds >= 0")

    suggestion_json = teacher_model = prompt_version = same = None
    if kind == "suggested":
        s = latest_suggestion(conn, scheme, window_id)
        if s is None:
            provenance = "unsuggested"
        else:
            same = int(_filled(scheme, s.answers, s.not_sure) == answers and s.not_sure == not_sure)
            provenance = "accepted" if same else "corrected"
            suggestion_json, teacher_model, prompt_version = s.raw, s.teacher_model, s.prompt_version
    else:
        provenance = kind  # blind or repeat

    return (
        "INSERT INTO labels (window_id, project, origin, provenance, labels_version, guide_version, answers_json,"
        " not_sure_json, cause_note, cause_source, suggestion_json, same_as_suggestion, teacher_model,"
        " prompt_version, lead_time_s, ls_annotation_id, created_at)"
        " VALUES (?, ?, 'aron', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            window_id,
            kind,
            provenance,
            scheme.version,
            scheme.guide_version,
            json.dumps(answers),
            json.dumps(not_sure),
            note,
            source,
            suggestion_json,
            same,
            teacher_model,
            prompt_version,
            None if lead is None else float(lead),
            ann["id"],
            created_at,
        ),
    )


def _check_window(conn: sqlite3.Connection, kind: str, task: dict) -> str:
    """The task's window ID once the window and the project fit (B12)."""
    data = task.get("data")
    window_id = data.get("window_id") if isinstance(data, dict) else None
    if not isinstance(window_id, str) or not conn.execute(
        "SELECT 1 FROM windows WHERE window_id = ?", (window_id,)
    ).fetchone():
        raise _Invalid("the task's window is not in the store")
    task_id = task.get("id")
    if _is_int(task_id):
        row = conn.execute("SELECT window_id, project_kind FROM ls_tasks WHERE task_id = ?", (task_id,)).fetchone()
        if row is not None and tuple(row) != (window_id, kind):
            raise _Invalid("the task is tracked for another window or project")
    if kind == "repeat":
        if not conn.execute("SELECT 1 FROM repeats WHERE window_id = ?", (window_id,)).fetchone():
            raise _Invalid("the window has no repeat scheduled")
    else:
        row = conn.execute("SELECT exposure FROM assignments WHERE window_id = ?", (window_id,)).fetchone()
        if row is None:
            raise _Invalid("the window has no assignment")
        if row[0] != kind:
            raise _Invalid(f"the window is assigned to {row[0]}, not {kind}")
    return window_id


def _parse_result(scheme: Scheme, result: Any) -> tuple[dict, dict, str | None, str | None]:
    """``(answers, not_sure, cause_note, cause_source)`` from a Label Studio result list."""
    if not isinstance(result, list):
        raise _Invalid("the result is not a list")
    questions = [q.key for q in scheme.questions]
    choice_fields = {*questions, *(config.not_sure_name(k) for k in questions), config.CAUSE_SOURCE}
    fields: dict[str, list[str]] = {}
    for item in result:
        if not isinstance(item, dict):
            raise _Invalid("a result item is not an object")
        name = item.get("from_name")
        value = item.get("value")
        if not isinstance(name, str) or (name not in choice_fields and name != config.CAUSE_NOTE):
            raise _Invalid("the result has a field the labeling view doesn't have")
        if item.get("to_name") != config.CHAT:
            raise _Invalid(f"{name}: not attached to the chat")
        key, kind = ("text", "textarea") if name == config.CAUSE_NOTE else ("choices", "choices")
        values = value.get(key) if isinstance(value, dict) else None
        if item.get("type") != kind or not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise _Invalid(f"{name}: not a list of {key}")
        if name in fields:
            raise _Invalid(f"{name}: given twice")
        fields[name] = values

    answers: dict[str, Any] = {}
    not_sure: dict[str, list[str]] = {}
    for q in scheme.questions:
        values = fields.get(q.key) or []
        reasons = fields.get(config.not_sure_name(q.key)) or []
        if values and reasons:
            raise _Invalid(f"{q.key}: both a value and a not-sure reason; give one or the other")
        if values:
            allowed = config.choice_values(q)
            if any(v not in allowed for v in values):
                raise _Invalid(f"{q.key}: a value is not in the scheme")
            if len(set(values)) != len(values):
                raise _Invalid(f"{q.key}: repeats a value")
            if q.kind == "multi":
                answers[q.key] = sorted(values, key=allowed.index)
            elif len(values) != 1:
                raise _Invalid(f"{q.key}: takes one value")
            else:
                answers[q.key] = values[0]
        elif reasons:
            if any(r not in scheme.not_sure_reasons for r in reasons):
                raise _Invalid(f"{q.key}: a not-sure reason is not in the scheme")
            if len(set(reasons)) != len(reasons):
                raise _Invalid(f"{q.key}: repeats a not-sure reason")
            not_sure[q.key] = sorted(reasons, key=scheme.not_sure_reasons.index)
        elif q.blank_value:
            answers[q.key] = q.blank_value  # an optional question left blank (tone)
        else:
            raise _Invalid(f"{q.key}: no value and no not-sure reason")

    note = "\n".join(t.strip() for t in fields.get(config.CAUSE_NOTE, []) if t.strip()) or None
    sources = fields.get(config.CAUSE_SOURCE) or []
    if len(sources) > 1:
        raise _Invalid(f"{config.CAUSE_SOURCE}: takes one value")
    source = sources[0] if sources else None
    if source is not None and source not in scheme.cause_sources:
        raise _Invalid(f"{config.CAUSE_SOURCE}: a value is not in the scheme")
    if (note is None) != (source is None):
        raise _Invalid("cause: a note needs a source, and a source needs a note")
    try:
        validate_answers(scheme, answers, not_sure)
        validate_cause_source(scheme, source)
    except ValueError as exc:  # its messages never echo a value
        raise _Invalid(str(exc)) from None
    return answers, not_sure, note, source


def _filled(scheme: Scheme, answers: dict[str, Any], not_sure: dict[str, list[str]]) -> dict[str, Any]:
    """``answers`` with each optional question that neither dict names set to its blank value."""
    filled = dict(answers)
    for q in scheme.questions:
        if q.blank_value and q.key not in filled and q.key not in not_sure:
            filled[q.key] = q.blank_value
    return filled


def latest_suggestion(conn: sqlite3.Connection, scheme: Scheme, window_id: str) -> _Suggestion | None:
    """The window's newest suggestion when it holds usable answers; None for none, an error or bad JSON."""
    row = conn.execute(
        "SELECT answers_json, teacher_model, prompt_version FROM suggestions WHERE window_id = ?"
        " ORDER BY id DESC LIMIT 1",
        (window_id,),
    ).fetchone()
    if row is None or row[0] is None:
        return None
    try:
        body = json.loads(row[0])
        answers, not_sure = body["answers"], body["not_sure"]
        validate_answers(scheme, answers, not_sure)
    except (ValueError, KeyError, TypeError, RecursionError):
        return None
    return _Suggestion(row[0], answers, not_sure, row[1], row[2])


# --- adopt --------------------------------------------------------------------------------------------


def adopt_untracked(store: Store, client: Client, kinds: tuple[str, ...] = KINDS) -> Adopted:
    """Record in ``ls_tasks`` every task of ``kinds`` that Label Studio holds for a known window."""
    recorded = unknown = 0
    for kind in kinds:
        project_id = client.project_id(kind)
        if project_id is None:
            continue
        exported = client.export_annotations(project_id, all_tasks=True)
        with store.transaction() as conn:
            for task in exported if isinstance(exported, list) else []:
                task_id = task.get("id") if isinstance(task, dict) else None
                data = task.get("data") if isinstance(task, dict) else None
                window_id = data.get("window_id") if isinstance(data, dict) else None
                if not _is_int(task_id):
                    unknown += 1
                    continue
                if conn.execute("SELECT 1 FROM ls_tasks WHERE task_id = ?", (task_id,)).fetchone():
                    continue
                if not isinstance(window_id, str) or not conn.execute(
                    "SELECT 1 FROM windows WHERE window_id = ?", (window_id,)
                ).fetchone():
                    unknown += 1
                    continue
                conn.execute(
                    "INSERT INTO ls_tasks (window_id, project_kind, task_id) VALUES (?, ?, ?)",
                    (window_id, kind, task_id),
                )
                recorded += 1
    return Adopted(recorded, unknown)


# --- push -------------------------------------------------------------------------------------------------


def push(
    store: Store,
    client: Client,
    scheme: Scheme,
    kind: str,
    limit: int,
    now: datetime,
    rng: random.Random | None = None,
) -> PushResult:
    """Import up to ``limit`` windows into the ``kind`` project (see the module docstring)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError("limit must be a whole number >= 0")
    project_id = client.ensure_project(kind, config.label_config(scheme))
    adopted = adopt_untracked(store, client, (kind,))
    candidates = _candidates(store, kind, now)
    if kind == "repeat":
        (rng or random.Random()).shuffle(candidates)

    pushed = with_prediction = 0
    error = None
    model_version = None
    for window_id in candidates[:limit]:
        w = store.as_window(window_id)
        suggestion = None
        if kind == "suggested":
            with store.lock:
                s = latest_suggestion(store.conn, scheme, window_id)
            if s is not None:
                suggestion = {"answers": s.answers, "not_sure": s.not_sure, "teacher_model": s.teacher_model,
                              "prompt_version": s.prompt_version}
        task = tasks.task_json(w, spread.compute(w), suggestion)
        try:
            (task_id,) = client.import_tasks(project_id, kind, [task])
        except Exception as exc:  # unreachable or refused: stop; a task it made anyway is adopted below
            error = type(exc).__name__
            try:
                late = adopt_untracked(store, client, (kind,))
                adopted = Adopted(adopted.recorded + late.recorded, late.unknown)
            except Exception:  # still unreachable: the next push adopts it
                pass
            break
        with store.transaction() as conn:
            conn.execute(
                "INSERT INTO ls_tasks (window_id, project_kind, task_id) VALUES (?, ?, ?)", (window_id, kind, task_id)
            )
        pushed += 1
        if suggestion is not None:
            with_prediction += 1
            model_version = tasks.model_version(suggestion["teacher_model"], suggestion["prompt_version"])
    if model_version is not None:
        try:
            client.use_model_version(project_id, model_version)
        except Exception as exc:
            error = error or type(exc).__name__
    return PushResult(pushed, with_prediction, adopted.recorded, adopted.unknown, error)


def _candidates(store: Store, kind: str, now: datetime) -> list[str]:
    now_s = iso(now)
    joins = "JOIN assignments a ON a.window_id = w.window_id"
    where = [
        "w.text_state = 'present'",
        "w.text_expires_at > :now",
        "NOT EXISTS (SELECT 1 FROM ls_tasks t WHERE t.window_id = w.window_id AND t.project_kind = :kind)",
        "NOT EXISTS (SELECT 1 FROM labels l WHERE l.window_id = w.window_id AND l.project = :kind"
        " AND l.origin = 'aron')",
    ]
    if kind == "repeat":
        joins += " JOIN repeats r ON r.window_id = w.window_id"
        where.append("r.due_at <= :now")
        order = "w.window_id"  # shuffled after
    else:
        where.append("a.exposure = :kind")
        order = "w.text_expires_at, w.window_id"
        if kind == "suggested":
            where.append("EXISTS (SELECT 1 FROM suggestions s WHERE s.window_id = w.window_id)")
    sql = f"SELECT w.window_id FROM windows w {joins} WHERE {' AND '.join(where)} ORDER BY {order}"
    with store.lock:
        return [r[0] for r in store.conn.execute(sql, {"now": now_s, "kind": kind}).fetchall()]
