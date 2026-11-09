"""What the Moment page shows for each window and question (ruling B31).

Per window and question, the first that applies:

1. the owner's latest label that names the question (origin ``aron``, the scheme's
   ``labels_version``; rows that don't name it are skipped, ruling C1);
2. else the active model's latest saved prediction (``manifest.active_per_question``),
   when its confidence is at least that model's own threshold from its ``manifest.json``;
3. else, when the active model is rung >= 1 and has no row for the window and question,
   the newest usable rung-0 model's prediction (same origin), with rung 0's threshold;
4. else not sure, with the reasons the label gives or "under threshold .52 / .65".

A model answer is also not sure, whatever its confidence, for a window with 0 or 1
messages ("no chat in this window"; rung 1 stores an intercept-only answer for it) and for
a window whose ``features.emote_counts`` holds an emote ID on the review list ("needs
review"). The owner's own labels always stand.

``cte`` builds this as one SQL ``WITH`` prefix ending in a ``shown`` table, so a later
read-only query can reuse it; ``shown`` runs it and turns the rows into ``Shown`` values.
``backfill_rung0`` saves rung-0 predictions, read from counts, for windows that have none
(C19): the page runs it at start and hourly, after expiry. It is the page's only write
besides the label rows of ``/fix`` and ``/cause``.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .. import manifest
from ..export_format import iso
from ..scheme import Scheme
from ..store import Store

LABEL_SOURCE = "your label"
MADE_UP_SOURCE = "made-up labels"
NO_MODEL_SOURCE = "no model"

# Reason codes for a model answer that isn't shown, and the words the page uses for them.
REASON_TEXT = {
    "no_chat": "no chat in this window",
    "no_threshold": "no threshold",
    "no_prediction": "no prediction yet",
    "no_model": "no model yet",
}  # "under_threshold" is worded with its numbers; "needs_review" uses the scheme's display text


@dataclass(frozen=True)
class ModelUse:
    """One model as the page uses it for one question."""

    model_id: str
    rung: int
    origin: str
    threshold: float | None


@dataclass(frozen=True)
class Plan:
    """Which model answers each question, read from ``manifests`` and each ``manifest.json``."""

    origin: str | None  # "aron", "synthetic", or None with no usable model
    questions: tuple[str, ...] = ()  # the scheme's question keys, in its order
    active: dict[str, ModelUse] = field(default_factory=dict)  # question -> active model
    fallback: dict[str, ModelUse] = field(default_factory=dict)  # question -> rung 0, where a rung >= 1 is active
    rung0: str | None = None  # the newest usable rung-0 model of ``origin``

    @property
    def made_up(self) -> bool:
        """True when only synthetic-origin models answer (the page then says "made-up labels")."""
        return self.origin == "synthetic"


@dataclass(frozen=True)
class Shown:
    """One question's shown answer for one window; ``values`` is empty when not sure."""

    question: str
    values: tuple[str, ...]
    kind: str  # "label", "model" (a prediction row was read) or "none"
    source: str  # "your label", "model rung N", "made-up labels" or "no model"
    confidence: float | None = None  # the model's, for kind "model"
    threshold: float | None = None
    model_id: str | None = None
    rung: int | None = None
    reasons: tuple[str, ...] = ()  # codes: the label's not-sure reasons, or one model reason
    not_sure: tuple[str, ...] = ()  # the same, in the page's words
    model_values: tuple[str, ...] = ()  # the model's raw answer, shown or not (Fix hints)
    provenance: str | None = None  # the label row's, for kind "label"

    @property
    def sure(self) -> bool:
        return bool(self.values)


def source_of(origin: str | None, rung: int | None) -> str:
    if origin is None or rung is None:
        return NO_MODEL_SOURCE
    return MADE_UP_SOURCE if origin == "synthetic" else f"model rung {rung}"


def conf_text(x: float) -> str:
    """``0.52`` -> ``.52``; ``1.0`` -> ``1.00``."""
    s = f"{x:.2f}"
    return s[1:] if s.startswith("0") else s


# --- the plan --------------------------------------------------------------------------------------


def _threshold(doc: dict[str, Any] | None, question: str) -> float | None:
    found = (doc or {}).get("thresholds")
    value = found.get(question) if isinstance(found, dict) else None
    if isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 <= value <= 1.0:
        return float(value)
    return None


def _usable(store: Store, models_dir: Path) -> list[tuple[str, int, str, dict[str, Any]]]:
    """``(model_id, rung, origin, manifest)`` of every model ``active_per_question`` could pick, newest first."""
    with store.lock:
        rows = store.conn.execute(
            "SELECT model_id, rung, origin FROM manifests WHERE retired_at IS NULL AND file_deleted_at IS NULL"
            " ORDER BY created_at DESC, model_id"
        ).fetchall()
    out = []
    for model_id, rung, origin in rows:
        folder = manifest.model_folder(models_dir, model_id)
        if folder is None or not (folder / manifest.MODEL_NAME).is_file():
            continue
        doc = manifest.read(folder)
        if manifest._scores(doc) is not None:  # the same test active_per_question applies
            out.append((model_id, rung, origin, doc or {}))
    return out


def plan(store: Store, scheme: Scheme, models_dir: Path) -> Plan:
    models_dir = Path(models_dir)
    active_ids = manifest.active_per_question(store, models_dir=models_dir)
    usable = {m: (rung, origin, doc) for m, rung, origin, doc in _usable(store, models_dir)}
    active: dict[str, ModelUse] = {}
    for q, model_id in active_ids.items():
        if model_id in usable:
            rung, origin, doc = usable[model_id]
            active[q] = ModelUse(model_id, rung, origin, _threshold(doc, q))
    questions = tuple(q.key for q in scheme.questions)
    if not active:
        return Plan(origin=None, questions=questions)
    origin = next(iter(active.values())).origin
    rung0 = next((m for m, (rung, o, _) in usable.items() if rung == 0 and o == origin), None)
    fallback: dict[str, ModelUse] = {}
    if rung0 is not None:
        doc0 = usable[rung0][2]
        for q in questions:
            if q in active and active[q].rung >= 1:
                fallback[q] = ModelUse(rung0, 0, origin, _threshold(doc0, q))
    return Plan(origin=origin, questions=questions, active=active, fallback=fallback, rung0=rung0)


# --- the SQL ---------------------------------------------------------------------------------------


def cte(
    p: Plan,
    *,
    scheme_version: str,
    review_ids: Iterable[str] = (),
    window_ids: Iterable[str] | None = None,
    questions: Iterable[str] | None = None,
) -> tuple[str, list[Any]]:
    """A ``WITH ...`` prefix whose last table is ``shown``; append ``SELECT ... FROM shown``.

    ``shown`` has one row per window (all of them, or ``window_ids``) and question:
    ``window_id, question, q_order, kind, value_json`` (NULL when not sure),
    ``confidence, threshold, model_id, rung, origin, reason`` (a code, NULL when shown or a
    label), ``reasons_json`` (a not-sure label's reasons), ``model_value_json`` (the
    model's raw answer), ``provenance`` (the label's), ``n_messages`` and ``review_hit``.
    ``questions`` defaults to the plan's: every question in the scheme.
    """
    qs = list(questions) if questions is not None else list(p.questions)
    params: list[Any] = []
    plan_rows = []
    for order, q in enumerate(qs):
        a = p.active.get(q)
        fb = p.fallback.get(q)
        plan_rows.append("(?, ?, ?, ?, ?, ?, ?, ?)")
        params += [
            q, order, a and a.model_id, a and a.rung, p.origin, a and a.threshold, fb and fb.model_id,
            fb and fb.threshold,
        ]  # fmt: skip
    review = sorted(set(review_ids))
    if review:
        review_sql = "VALUES " + ", ".join("(?)" for _ in review)
        params += review
    else:
        review_sql = "SELECT NULL WHERE 0"
    if window_ids is not None:
        win_filter = " WHERE w.window_id IN (SELECT value FROM json_each(?))"
        params.append(json.dumps(sorted(set(window_ids))))
    else:
        win_filter = ""
    params.append(scheme_version)
    sql = f"""
WITH
plan(question, q_order, model_id, rung, origin, threshold, fb_model_id, fb_threshold) AS (
    VALUES {", ".join(plan_rows)}
),
review(emote_id) AS ({review_sql}),
win AS (
    SELECT w.window_id, f.json AS counts
    FROM windows w LEFT JOIN features f ON f.window_id = w.window_id{win_filter}
),
lab AS (
    SELECT l.window_id, p.question, l.answers_json, l.not_sure_json, l.provenance,
           ROW_NUMBER() OVER (PARTITION BY l.window_id, p.question ORDER BY l.created_at DESC, l.id DESC) AS rn
    FROM labels l JOIN plan p
    WHERE l.origin = 'aron' AND l.labels_version = ?
      AND l.window_id IN (SELECT window_id FROM win)
      AND (json_type(l.answers_json, '$.' || p.question) IS NOT NULL
           OR json_type(l.not_sure_json, '$.' || p.question) IS NOT NULL)
),
pred AS (
    SELECT pr.window_id, pr.model_id, pr.question, pr.value_json, pr.confidence,
           ROW_NUMBER() OVER (PARTITION BY pr.window_id, pr.model_id, pr.question
                              ORDER BY pr.created_at DESC, pr.id DESC) AS rn
    FROM predictions pr
    WHERE pr.model_id IN (SELECT model_id FROM plan UNION SELECT fb_model_id FROM plan)
      AND pr.window_id IN (SELECT window_id FROM win)
),
pick AS (
    SELECT w.window_id, p.question, p.q_order, p.origin,
           lab.window_id AS lab_window, lab.answers_json, lab.not_sure_json, lab.provenance,
           CASE WHEN a.window_id IS NOT NULL THEN p.model_id
                WHEN fb.window_id IS NOT NULL THEN p.fb_model_id ELSE p.model_id END AS model_id,
           CASE WHEN a.window_id IS NOT NULL THEN p.rung
                WHEN fb.window_id IS NOT NULL THEN 0 ELSE p.rung END AS rung,
           CASE WHEN a.window_id IS NOT NULL THEN p.threshold
                WHEN fb.window_id IS NOT NULL THEN p.fb_threshold ELSE p.threshold END AS threshold,
           COALESCE(a.value_json, fb.value_json) AS m_value,
           CASE WHEN a.window_id IS NOT NULL THEN a.confidence ELSE fb.confidence END AS m_conf,
           json_extract(w.counts, '$.n_messages') AS n_messages,
           EXISTS (SELECT 1 FROM json_each(w.counts, '$.emote_counts') e JOIN review r ON r.emote_id = e.key)
               AS review_hit
    FROM win w CROSS JOIN plan p
    LEFT JOIN lab ON lab.window_id = w.window_id AND lab.question = p.question AND lab.rn = 1
    LEFT JOIN pred a ON a.window_id = w.window_id AND a.model_id = p.model_id AND a.question = p.question
                     AND a.rn = 1
    LEFT JOIN pred fb ON a.window_id IS NULL AND fb.window_id = w.window_id AND fb.model_id = p.fb_model_id
                      AND fb.question = p.question AND fb.rn = 1
),
shown AS (
    SELECT window_id, question, q_order,
           CASE WHEN lab_window IS NOT NULL THEN 'label' WHEN m_value IS NOT NULL THEN 'model' ELSE 'none' END
               AS kind,
           CASE WHEN lab_window IS NOT NULL THEN answers_json -> ('$.' || question)
                WHEN n_messages <= 1 OR review_hit OR m_value IS NULL OR threshold IS NULL
                     OR m_conf IS NULL OR m_conf < threshold THEN NULL
                ELSE m_value END AS value_json,
           CASE WHEN lab_window IS NULL THEN m_conf END AS confidence,
           CASE WHEN lab_window IS NULL THEN threshold END AS threshold,
           CASE WHEN lab_window IS NULL THEN model_id END AS model_id,
           CASE WHEN lab_window IS NULL THEN rung END AS rung,
           origin,
           CASE WHEN lab_window IS NOT NULL THEN NULL
                WHEN n_messages <= 1 THEN 'no_chat'
                WHEN review_hit THEN 'needs_review'
                WHEN model_id IS NULL THEN 'no_model'
                WHEN m_value IS NULL THEN 'no_prediction'
                WHEN threshold IS NULL THEN 'no_threshold'
                WHEN m_conf IS NULL OR m_conf < threshold THEN 'under_threshold'
           END AS reason,
           CASE WHEN lab_window IS NOT NULL THEN not_sure_json -> ('$.' || question) END AS reasons_json,
           m_value AS model_value_json,
           CASE WHEN lab_window IS NOT NULL THEN provenance END AS provenance,
           n_messages,
           review_hit
    FROM pick
)"""
    return sql, params


# --- reading ---------------------------------------------------------------------------------------


_COLUMNS = (
    "window_id", "question", "kind", "value_json", "confidence", "threshold", "model_id", "rung", "origin",
    "reason", "reasons_json", "model_value_json", "provenance",
)  # fmt: skip


def shown(
    store: Store,
    scheme: Scheme,
    models_dir: Path,
    window_ids: Iterable[str] | None = None,
    *,
    review_ids: Iterable[str] = frozenset(),
) -> dict[str, dict[str, Shown]]:
    """``{window_id: {question: Shown}}`` for every window (or ``window_ids`` that exist)."""
    p = plan(store, scheme, models_dir)
    return read(store, scheme, p, window_ids, review_ids=review_ids)


def read(
    store: Store,
    scheme: Scheme,
    p: Plan,
    window_ids: Iterable[str] | None = None,
    *,
    review_ids: Iterable[str] = frozenset(),
) -> dict[str, dict[str, Shown]]:
    """``shown`` with a plan already read (the page reads the plan once per request)."""
    sql, params = cte(
        p,
        scheme_version=scheme.version,
        review_ids=review_ids,
        window_ids=window_ids,
    )
    with store.lock:
        rows = store.conn.execute(
            sql + f" SELECT {', '.join(_COLUMNS)} FROM shown ORDER BY window_id, q_order", params
        ).fetchall()
    out: dict[str, dict[str, Shown]] = {}
    for row in rows:
        r = dict(zip(_COLUMNS, row))
        out.setdefault(r["window_id"], {})[r["question"]] = _shown(scheme, r)
    return out


def _values(scheme: Scheme, question: str, raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    items = raw if isinstance(raw, list) else [raw]
    q = next(x for x in scheme.questions if x.key == question)
    order = {v: i for i, v in enumerate(q.values + ((q.blank_value,) if q.blank_value else ()))}
    return tuple(sorted({v for v in items if isinstance(v, str)}, key=lambda v: (order.get(v, len(order)), v)))


def _shown(scheme: Scheme, r: dict[str, Any]) -> Shown:
    q = r["question"]
    if r["kind"] == "label":
        values = _values(scheme, q, json.loads(r["value_json"]) if r["value_json"] is not None else None)
        reasons: tuple[str, ...] = ()
        if not values and r["reasons_json"] is not None:
            found = json.loads(r["reasons_json"])
            order = {v: i for i, v in enumerate(scheme.not_sure_reasons)}
            reasons = tuple(sorted({x for x in found if isinstance(x, str)}, key=lambda x: order.get(x, len(order))))
        return Shown(
            question=q,
            values=values,
            kind="label",
            source=LABEL_SOURCE,
            reasons=reasons,
            not_sure=tuple(scheme.display.get(x, x) for x in reasons),
            provenance=r["provenance"],
        )
    values = _values(scheme, q, json.loads(r["value_json"]) if r["value_json"] is not None else None)
    model_values = _values(scheme, q, json.loads(r["model_value_json"]) if r["model_value_json"] else None)
    confidence = r["confidence"]
    threshold = r["threshold"]
    reason = r["reason"]
    if reason is None:
        codes: tuple[str, ...] = ()
        words: tuple[str, ...] = ()
    elif reason == "under_threshold":
        codes = (reason,)
        conf = conf_text(confidence) if confidence is not None else "none"
        words = (f"under threshold {conf} / {conf_text(threshold)}",)
    elif reason == "needs_review":
        codes, words = (reason,), (scheme.display.get("needs_review", "needs review"),)
    else:
        codes, words = (reason,), (REASON_TEXT[reason],)
    return Shown(
        question=q,
        values=values,
        kind=r["kind"],
        source=source_of(r["origin"], r["rung"]) if r["model_id"] is not None else NO_MODEL_SOURCE,
        confidence=confidence,
        threshold=threshold,
        model_id=r["model_id"],
        rung=r["rung"],
        reasons=codes,
        not_sure=words,
        model_values=model_values,
    )


# --- model files -----------------------------------------------------------------------------------


class ModelFileError(ValueError):
    """A model file that is missing, changed since its manifest was written, or not a model."""


def load_model(models_dir: Path, model_id: str) -> Any:
    """Load ``<models_dir>/<model_id>/model.joblib`` as a ``train.Model``.

    The file is one ``manifest.write`` made; when its manifest records ``model_sha256``,
    the file must still match it. Raises ``ModelFileError`` otherwise.
    """
    from .. import train  # scikit-learn and joblib load only when there is a model to run

    folder = manifest.model_folder(Path(models_dir), model_id)
    if folder is None:
        raise ModelFileError("not a plain model ID")
    path = folder / manifest.MODEL_NAME
    try:
        raw = path.read_bytes()
    except OSError as e:
        raise ModelFileError(f"unreadable ({type(e).__name__})") from None
    want = (manifest.read(folder) or {}).get("model_sha256")
    if want is not None and hashlib.sha256(raw).hexdigest() != want:
        raise ModelFileError("changed since its manifest was written")
    try:
        import joblib

        model = joblib.load(io.BytesIO(raw))
    except Exception as e:  # a model file is a boundary: any failure means "no model"
        raise ModelFileError(f"won't load ({type(e).__name__})") from None
    if not isinstance(model, train.Model):
        raise ModelFileError("not a model")
    return model


# --- rung-0 predictions saved by the page ---------------------------------------------------------


def backfill_rung0(store: Store, scheme: Scheme, models_dir: Path, now: datetime) -> int:
    """Save the rung-0 model's answers, from counts, for windows it has no row for; return how many windows.

    Uses the newest usable rung-0 model of the origin the page shows (``plan().rung0``).
    A model file that won't load is reported on stderr and nothing is saved.
    """
    from .. import train
    from ..spread import Counts

    p = plan(store, scheme, models_dir)
    if p.rung0 is None:
        return 0
    model_id = p.rung0
    sql = (
        "SELECT f.window_id, f.json FROM features f WHERE NOT EXISTS"
        " (SELECT 1 FROM predictions p WHERE p.window_id = f.window_id AND p.model_id = ?) ORDER BY f.window_id"
    )
    with store.lock:
        todo = store.conn.execute(sql, (model_id,)).fetchall()
    if not todo:
        return 0
    try:
        model = load_model(models_dir, model_id)
        if model.rung != 0:
            raise ModelFileError("not a rung-0 model")
        answers = train.predict(model, [(Counts.from_json(raw), None) for _, raw in todo])
    except Exception as e:  # report it and keep the page up; the next hourly run tries again
        why = str(e) if isinstance(e, ModelFileError) else type(e).__name__
        print(f"page: rung-0 model {model_id} could not run ({why}); no predictions saved", file=sys.stderr)
        return 0
    now_s = iso(now)
    windows = 0
    with store.transaction() as conn:
        have = {r[0] for r in conn.execute("SELECT DISTINCT window_id FROM predictions WHERE model_id = ?",
                                           (model_id,))}  # fmt: skip
        rows = []
        for (window_id, _), per in zip(todo, answers):
            if window_id in have:
                continue  # written meanwhile
            got = [(q, a) for q, a in per.items() if a is not None]
            if not got:
                continue
            windows += 1
            for q, a in got:
                value = list(a.value) if isinstance(a.value, tuple) else a.value
                rows.append((window_id, model_id, q, json.dumps(value), a.confidence, now_s))
        conn.executemany(
            "INSERT INTO predictions (window_id, model_id, question, value_json, confidence, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
    return windows
