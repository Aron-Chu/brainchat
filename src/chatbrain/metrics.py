"""Per-question metrics, Wilson intervals, label reading and the owner's own repeat consistency.

Spec section 8 reports every question on its own, never one blended score:

- **agreement** is answered agreement: of the windows the model answered (confidence
  at or above the question's threshold), the share whose answer equals the owner's label.
  ``doing`` agrees only on an exact set match (ruling C21);
- **per-value precision and recall** give partial credit (for ``doing``, each value is
  scored on its own). Precision is over the answered windows that hold the value;
  recall is over every graded window whose label holds it, so an abstention is a miss.
  ``macro_f1`` is the mean F1 over the values that are in a label or an answer;
- **abstain rate** is the share of graded windows left "not sure";
- **Wilson 95% intervals** go with every agreement;
- ``aron_not_sure_also_abstained`` is ``(abstained, aron_not_sure)``: of the graded
  windows the owner marked not sure for that question, how many the model also left unanswered.

``self_consistency`` compares each window's first owner label (blind or suggested
project) with their repeat label, question by question, counting only pairs where both
labels answer. Spec section 8 treats it as the practical ceiling.

``label_rows`` reads label rows for one scheme version, oldest first, each checked
again with ``scheme.validate_answers``. Errors name a label's row ID and the rule,
never a window ID or a value.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .scheme import Scheme, validate_answers
from .store import Store

Z95 = 1.96
ORIGINS: tuple[str, ...] = ("aron", "synthetic")
FIRST_PROJECTS: tuple[str, ...] = ("blind", "suggested")
REPEAT_PROJECT = "repeat"


@dataclass(frozen=True)
class QuestionReport:
    """One question's numbers on the graded test windows (see the module docstring)."""

    n_graded: int
    answered: int
    agreement: float | None  # None when nothing was answered
    agreement_ci: tuple[float, float]
    macro_f1: float
    per_value: dict[str, tuple[float, float, int]]  # value -> (precision, recall, support)
    abstain_rate: float | None  # None when nothing was graded
    aron_not_sure_also_abstained: tuple[int, int]  # (model abstained, owner not sure)

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-ready values (tuples become lists when dumped)."""
        return asdict(self)


@dataclass(frozen=True)
class LabelRow:
    """One ``labels`` row as answers: ``doing`` is a tuple in scheme order, reasons are tuples."""

    id: int
    window_id: str
    project: str
    origin: str
    answers: dict[str, Any]
    not_sure: dict[str, tuple[str, ...]]
    created_at: str


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """The Wilson score interval for ``k`` successes in ``n`` trials; ``(0.0, 1.0)`` when ``n == 0``."""
    for name, v in (("k", k), ("n", n)):
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"wilson: {name} must be a whole number")
    if n < 0 or not 0 <= k <= n:
        raise ValueError("wilson: needs 0 <= k <= n")
    if isinstance(z, bool) or not isinstance(z, (int, float)) or not math.isfinite(z) or z <= 0:
        raise ValueError("wilson: z must be a positive number")
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    lo = 0.0 if k == 0 else max(0.0, center - half)
    hi = 1.0 if k == n else min(1.0, center + half)
    return (lo, hi)


def agrees(target: Any, answer: Any, multi: bool) -> bool:
    """Whether an answer equals a label: an exact set match for a multi question."""
    if multi:
        return set(target) == set(answer)
    return target == answer


def question_report(
    values: Sequence[str],
    multi: bool,
    graded: Sequence[tuple[Any, Any | None]],
    not_sure: tuple[int, int],
) -> QuestionReport:
    """Score one question.

    ``graded`` holds ``(label, shown answer)`` per graded window, the answer ``None``
    when the model abstained; ``values`` are the question's values in scheme order;
    ``not_sure`` is ``(abstained, aron_not_sure)``.
    """
    abstained, aron_not_sure = not_sure
    if not 0 <= abstained <= aron_not_sure:
        raise ValueError("question_report: not_sure must be (abstained, total) with abstained <= total")
    n = len(graded)
    shown = [(t, a) for t, a in graded if a is not None]
    answered = len(shown)
    agree = sum(agrees(t, a, multi) for t, a in shown)

    per_value: dict[str, tuple[float, float, int]] = {}
    f1s: list[float] = []
    for v in values:
        support = sum(_holds(t, v, multi) for t, _ in graded)
        predicted = sum(_holds(a, v, multi) for _, a in shown)
        hits = sum(_holds(t, v, multi) and _holds(a, v, multi) for t, a in shown)
        precision = hits / predicted if predicted else 0.0
        recall = hits / support if support else 0.0
        per_value[v] = (precision, recall, support)
        if support or predicted:
            f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)

    return QuestionReport(
        n_graded=n,
        answered=answered,
        agreement=agree / answered if answered else None,
        agreement_ci=wilson(agree, answered),
        macro_f1=sum(f1s) / len(f1s) if f1s else 0.0,
        per_value=per_value,
        abstain_rate=(n - answered) / n if n else None,
        aron_not_sure_also_abstained=(abstained, aron_not_sure),
    )


def label_rows(store: Store, scheme: Scheme, origin: str | None = None) -> list[LabelRow]:
    """Label rows of ``scheme.version`` (one origin, or all), oldest first by ``created_at`` then ID.

    Each row is checked with ``validate_answers``; a bad row raises ``ValueError``
    naming its row ID and the rule.
    """
    if origin is not None and origin not in ORIGINS:
        raise ValueError(f"origin must be one of {', '.join(ORIGINS)}, or None")
    sql = (
        "SELECT id, window_id, project, origin, answers_json, not_sure_json, created_at FROM labels"
        " WHERE labels_version = ?"
    )
    params: list[Any] = [scheme.version]
    if origin is not None:
        sql += " AND origin = ?"
        params.append(origin)
    with store.lock:
        raw = store.conn.execute(sql + " ORDER BY created_at, id", params).fetchall()
    rows: list[LabelRow] = []
    for label_id, window_id, project, row_origin, answers_json, not_sure_json, created_at in raw:
        try:
            answers = json.loads(answers_json)
            not_sure = json.loads(not_sure_json)
        except json.JSONDecodeError:
            raise ValueError(f"label {label_id}: answers or not_sure is not JSON") from None
        try:
            validate_answers(scheme, answers, not_sure)
        except ValueError as exc:
            raise ValueError(f"label {label_id}: {exc}") from None
        rows.append(
            LabelRow(
                id=label_id,
                window_id=window_id,
                project=project,
                origin=row_origin,
                answers={q: tuple(v) if isinstance(v, list) else v for q, v in answers.items()},
                not_sure={q: tuple(r) for q, r in not_sure.items()},
                created_at=created_at,
            )
        )
    return rows


_NOT_SURE = object()


def self_consistency(store: Store, scheme: Scheme) -> dict[str, tuple[int, int, float, float]]:
    """Per question: ``(agreeing, compared, Wilson low, Wilson high)`` between the owner's first and repeat labels.

    The first label is their earliest blind- or suggested-project label that names the
    question; the repeat label is their earliest repeat-project label that names it. A
    pair counts only when both answer (a not-sure on either side is left out);
    ``doing`` agrees on an exact set match. Page fixes and synthetic labels are not used.
    """
    first: dict[tuple[str, str], Any] = {}
    repeat: dict[tuple[str, str], Any] = {}
    for r in label_rows(store, scheme, "aron"):
        if r.project == REPEAT_PROJECT:
            into = repeat
        elif r.project in FIRST_PROJECTS:
            into = first
        else:
            continue
        for q in scheme.questions:
            if q.key in r.answers:
                into.setdefault((r.window_id, q.key), r.answers[q.key])
            elif q.key in r.not_sure:
                into.setdefault((r.window_id, q.key), _NOT_SURE)

    out: dict[str, tuple[int, int, float, float]] = {}
    for q in scheme.questions:
        agree = compared = 0
        for (window_id, key), a in sorted(first.items(), key=lambda kv: kv[0]):
            b = repeat.get((window_id, key))
            if key != q.key or b is None or a is _NOT_SURE or b is _NOT_SURE:
                continue
            compared += 1
            agree += agrees(a, b, q.kind == "multi")
        out[q.key] = (agree, compared, *wilson(agree, compared))
    return out


def _holds(answer: Any, value: str, multi: bool) -> bool:
    if multi:
        return value in answer
    return answer == value
