"""Rungs 0 and 1 (spec section 8): datasets, grouped splits, one head per question,
thresholds, reports and evidence.

Dataset (``load_dataset``): the labeled windows of one origin, since synthetic labels
never mix with the owner's (``MixedOrigins`` when both exist and none is chosen; with one
present, that one is used). Only labels of the scheme's version are read.

- ``train_targets``: per window and question, the latest label that names the question
  (``created_at``, then row ID), from any project, page fixes included. A not-sure
  answer leaves the question out of the targets and puts it in ``not_sure``.
- ``grade_targets`` / ``grade_not_sure`` / ``graded`` (ruling C6): the same, from the owner's
  blind- and repeat-project labels only. For synthetic labels every window is graded
  and the grade maps equal the train maps.
- ``counts`` come from the ``features`` table (``Counts.from_json``) and outlive the
  text; ``windows`` holds only windows whose text is still present.
- Windows with no ``assignments`` row (text gone at assign time, ruling B12) simply
  have no owner label; nothing here reads ``assignments``.

Split (``split``): grouped by channel, so no channel (and so no stream) is in two
splits. Channels are shuffled with ``seed`` and taken in that order. Test channels are
added until test holds at least ``min(200, 20%)`` of the dataset's windows and both
Moment and ordinary windows; ``picked_rare`` windows on test channels leave every split
(ruling C21). Validation channels are then added until validation holds at least 15%
of the remaining windows and both kinds. The rest is train. At least one channel is
always left for validation and one for train, so a dataset needs 3 channels.

Heads (``fit``): one per question, trained on train windows that have a target.

- Rung 0: ``DictVectorizer`` over ``features.counts_row``, then ``MaxAbsScaler`` (raw
  counts keep lbfgs from converging; the scaler keeps the matrix sparse and puts
  counts and shares on one scale), + ``LogisticRegression``.
- Rung 1: ``TfidfVectorizer(ngram_range=(1, 2), min_df=2)`` over ``features.text_doc``
  + the same classifier; its token pattern keeps ``emote_<id>`` and ``@someone`` whole.
  A window without text can't be scored at rung 1 and is left out of its fit,
  thresholds and report.
- The classifier is ``LogisticRegression(max_iter=1000, class_weight="balanced")``.
  ``doing`` is one-vs-rest: one binary classifier per value; its answer is every value
  with p >= 0.5, or the top value when none is, and its confidence is the lowest p
  among the chosen values. A value that is always (or never) present in training is a
  constant p of 1 (or 0). A single-value question takes the most probable class.
- A head that saw one class predicts it with confidence 1.0; a head with no training
  windows (or, at rung 1, no word in two windows) never answers. ``config`` records both.

Thresholds (``calibrate``): per question, the smallest of 0.30, 0.35, ..., 0.95 at which
answered agreement on the graded validation windows (against ``grade_targets``) is at
least 0.80 with at least one answer; ``None`` (always "not sure") when none is.

Evidence (``evidence``): up to 5 message IDs that support one value, scored line by
line (``line_score``) and kept only when the score is above 0, best first:

- rung 1: the line alone through the head's vectorizer; ``doing`` uses that value's
  decision function (w.x + b), and a single-value question uses the class's
  coefficient row (w.x, ruling C21). A line with no word the model knows has no score;
- rung 0 (ruling C20): the largest coefficient, for the value, among the line's
  locatable features: each emote ID in the line (``emote:<id>``) and ``?``
  (``question_share``) when the line has one. Coefficients are on the scaled features.

Nothing here prints chat text, and errors never carry window IDs.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

from . import features, metrics
from .export_format import Message, Window
from .scheme import Question, Scheme
from .spread import Counts
from .store import Store, WindowRow

Origin = Literal["aron", "synthetic"]
ORIGINS: tuple[str, ...] = metrics.ORIGINS
GRADED_PROJECTS: tuple[str, ...] = ("blind", "repeat")

TEST_MAX = 200
TEST_SHARE = 0.20
VAL_SHARE = 0.15
THRESHOLDS: tuple[float, ...] = tuple(round(0.30 + 0.05 * i, 2) for i in range(14))  # 0.30 .. 0.95
TARGET_AGREEMENT = 0.80
DOING_MIN_P = 0.5
MAX_EVIDENCE = 5

MAX_ITER = 1000
CLASS_WEIGHT = "balanced"
NGRAM_RANGE = (1, 2)
MIN_DF = 2
# Words of 2+ characters (the scikit-learn default), plus emote_<id> and @someone kept whole.
TOKEN_PATTERN = r"(?u)emote_\S+|@someone|\b\w\w+\b"
QUESTION_FEATURE = "question_share"  # rung-0 evidence locates it by a "?" in the line


class MixedOrigins(ValueError):
    """The store holds the owner's (``aron``) and synthetic labels and no origin was chosen."""


@dataclass(frozen=True)
class Answer:
    """One head's answer: a value (a tuple in scheme order for ``doing``) and its confidence."""

    value: str | tuple[str, ...]
    confidence: float


@dataclass(frozen=True)
class Dataset:
    rows: dict[str, WindowRow]
    counts: dict[str, Counts]
    windows: dict[str, Window]  # only windows whose text is still present
    train_targets: dict[str, dict[str, Any]]
    not_sure: dict[str, set[str]]  # window -> questions whose latest label is not sure
    graded: set[str]
    grade_targets: dict[str, dict[str, Any]]  # blind and repeat labels only (C6)
    grade_not_sure: dict[str, set[str]]
    origin: str | None
    scheme: Scheme


@dataclass(frozen=True)
class Split:
    train: list[str]
    val: list[str]
    test: list[str]


@dataclass(frozen=True)
class Model:
    rung: int
    emote_mode: str
    heads: dict[str, object]
    config: dict


class Head(Protocol):
    """What ``predict``, ``calibrate`` and ``report`` need from a head."""

    def predict(self, inputs: list[Any]) -> list[Answer | None]: ...


# --- dataset -------------------------------------------------------------------------------------


def load_dataset(store: Store, scheme: Scheme, origin: Origin | None = None) -> Dataset:
    """Every window with a label of one origin, its counts, its text if present, and its targets."""
    if origin is not None and origin not in ORIGINS:
        raise ValueError(f"origin must be one of {', '.join(ORIGINS)}, or None")
    with store.lock:
        present = [
            r[0]
            for r in store.conn.execute(
                # Rows that name no question (the page's cause-only note) give no targets, so no origin (C1).
                "SELECT DISTINCT origin FROM labels WHERE labels_version = ?"
                " AND (answers_json <> '{}' OR not_sure_json <> '{}') ORDER BY origin",
                (scheme.version,),
            )
        ]
        if origin is None:
            if len(present) > 1:
                raise MixedOrigins("the store holds both aron and synthetic labels; choose one origin, they never mix")
            origin = present[0] if present else None  # type: ignore[assignment]
        label_rows = metrics.label_rows(store, scheme, origin) if origin is not None else []

        train_targets, not_sure = _latest(label_rows)
        if origin == "aron":
            grade_targets, grade_not_sure = _latest([r for r in label_rows if r.project in GRADED_PROJECTS])
        else:
            grade_targets = {w: dict(t) for w, t in train_targets.items()}
            grade_not_sure = {w: set(q) for w, q in not_sure.items()}

        ids = sorted(train_targets)
        rows = {w: store.window(w) for w in ids}
        counts: dict[str, Counts] = {}
        missing = 0
        for w in ids:
            found = store.conn.execute("SELECT json FROM features WHERE window_id = ?", (w,)).fetchone()
            if found is None:
                missing += 1
            else:
                counts[w] = Counts.from_json(found[0])
        if missing:
            raise ValueError(f"{missing} labeled window(s) have no counted features")
        windows = {w: store.as_window(w) for w in ids if rows[w].text_state == "present"}

    return Dataset(
        rows=rows,
        counts=counts,
        windows=windows,
        train_targets=train_targets,
        not_sure=not_sure,
        graded=set(grade_targets),
        grade_targets=grade_targets,
        grade_not_sure=grade_not_sure,
        origin=origin,
        scheme=scheme,
    )


def _latest(rows: Sequence[metrics.LabelRow]) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]]]:
    """Latest answer per window and question; rows that name no question are skipped (C1)."""
    targets: dict[str, dict[str, Any]] = {}
    not_sure: dict[str, set[str]] = {}
    for r in rows:  # oldest first, so a later row wins
        if not r.answers and not r.not_sure:
            continue
        t = targets.setdefault(r.window_id, {})
        ns = not_sure.setdefault(r.window_id, set())
        for q, v in r.answers.items():
            t[q] = v
            ns.discard(q)
        for q in r.not_sure:
            t.pop(q, None)
            ns.add(q)
    return targets, not_sure


# --- split ---------------------------------------------------------------------------------------


def split(ds: Dataset, seed: int) -> Split:
    """Train, validation and test window IDs, grouped by channel (see the module docstring)."""
    by_channel: dict[str, list[str]] = {}
    for w in sorted(ds.rows):
        by_channel.setdefault(ds.rows[w].channel, []).append(w)
    channels = sorted(by_channel)
    if len(channels) < 3:
        raise ValueError(f"a split needs windows from at least 3 channels; the dataset has {len(channels)}")
    random.Random(seed).shuffle(channels)

    def test_windows(c: str) -> list[str]:
        return [w for w in by_channel[c] if not ds.rows[w].picked_rare]

    test_kinds = {ds.rows[w].anchor_kind for c in channels for w in test_windows(c)}
    test_need = min(TEST_MAX, TEST_SHARE * len(ds.rows))
    test_ch, rest = _take(ds, channels, test_windows, test_need, test_kinds, keep=2)
    remaining = [w for c in rest for w in by_channel[c]]
    val_kinds = {ds.rows[w].anchor_kind for w in remaining}
    val_ch, train_ch = _take(ds, rest, by_channel.__getitem__, VAL_SHARE * len(remaining), val_kinds, keep=1)
    return Split(
        train=sorted(w for c in train_ch for w in by_channel[c]),
        val=sorted(w for c in val_ch for w in by_channel[c]),
        test=sorted(w for c in test_ch for w in test_windows(c)),
    )


def _take(
    ds: Dataset,
    channels: list[str],
    windows_of: Callable[[str], list[str]],
    need: float,
    kinds: set[str],
    keep: int,
) -> tuple[list[str], list[str]]:
    """Take channels in order until they hold ``need`` windows and every kind, leaving ``keep``."""
    taken: list[str] = []
    got: list[str] = []
    for c in channels:
        if len(got) >= need and kinds <= {ds.rows[w].anchor_kind for w in got}:
            break
        if len(channels) - len(taken) <= keep:
            break
        taken.append(c)
        got.extend(windows_of(c))
    return taken, [c for c in channels if c not in taken]


# --- heads ---------------------------------------------------------------------------------------


def _classifier(seed: int) -> LogisticRegression:
    return LogisticRegression(max_iter=MAX_ITER, class_weight=CLASS_WEIGHT, random_state=seed)


def _vectorizer(rung: int) -> Pipeline | TfidfVectorizer:
    if rung == 0:
        # Raw counts (a busy window has hundreds of messages) keep lbfgs from converging in
        # 1000 iterations; MaxAbsScaler keeps the matrix sparse and puts every feature on one scale.
        return Pipeline([("dict", DictVectorizer()), ("scale", MaxAbsScaler())])
    return TfidfVectorizer(ngram_range=NGRAM_RANGE, min_df=MIN_DF, token_pattern=TOKEN_PATTERN)


def _vocabulary(vec: Pipeline | TfidfVectorizer) -> dict[str, int]:
    """Feature name -> column of a fitted vectorizer."""
    return vec.named_steps["dict"].vocabulary_ if isinstance(vec, Pipeline) else vec.vocabulary_


def _fit_vectorizer(rung: int, inputs: list[Any]) -> tuple[Any, Any] | None:
    """A fitted vectorizer and the training matrix, or ``None`` when no feature is left."""
    vec = _vectorizer(rung)
    try:
        matrix = vec.fit_transform(inputs)
    except ValueError:  # TfidfVectorizer: no word is in 2 or more training windows
        return None
    # Older scikit-learn keeps the words min_df dropped (one-off words) in stop_words_; never pickle them.
    if hasattr(vec, "stop_words_"):
        delattr(vec, "stop_words_")
    return vec, matrix


class _SingleHead:
    """A single-value question: the most probable class of one LogisticRegression."""

    def __init__(self, values: tuple[str, ...]) -> None:
        self.values = values  # every value the question allows, in scheme order
        self.kind = "untrained"
        self.reason: str | None = "no_training_windows"
        self.constant: str | None = None
        self.vectorizer: Any = None
        self.clf: LogisticRegression | None = None

    def fit(self, rung: int, inputs: list[Any], y: list[str], seed: int) -> _SingleHead:
        if not inputs:
            return self
        classes = sorted(set(y), key=self.values.index)
        if len(classes) == 1:
            self.kind, self.reason, self.constant = "constant", None, classes[0]
            return self
        fitted = _fit_vectorizer(rung, inputs)
        if fitted is None:
            self.reason = "no_vocabulary"
            return self
        self.vectorizer, matrix = fitted
        self.clf = _classifier(seed).fit(matrix, y)
        self.kind, self.reason = ("binary" if len(classes) == 2 else "multinomial"), None
        return self

    def predict(self, inputs: list[Any]) -> list[Answer | None]:
        if self.kind == "untrained":
            return [None] * len(inputs)
        if self.kind == "constant":
            return [Answer(self.constant, 1.0) for _ in inputs]  # type: ignore[arg-type]
        if not inputs:
            return []
        assert self.clf is not None
        proba = self.clf.predict_proba(self.vectorizer.transform(inputs))
        best = np.argmax(proba, axis=1)
        return [Answer(str(self.clf.classes_[i]), float(p[i])) for i, p in zip(best, proba)]

    def coef_row(self, value: str) -> tuple[np.ndarray, float | None] | None:
        """The class's coefficient row and no intercept (C21); ``None`` without one."""
        if self.clf is None or value not in self.clf.classes_:
            return None
        classes = [str(c) for c in self.clf.classes_]
        if len(classes) == 2:  # scikit-learn keeps one row, for classes_[1]
            row = self.clf.coef_[0] if value == classes[1] else -self.clf.coef_[0]
        else:
            row = self.clf.coef_[classes.index(value)]
        return row, None

    def describe(self) -> dict[str, Any]:
        if self.kind == "constant":
            return {"kind": "constant", "value": self.constant}
        if self.kind == "untrained":
            return {"kind": "untrained", "reason": self.reason}
        assert self.clf is not None
        return {"kind": self.kind, "classes": [str(c) for c in self.clf.classes_]}


class _DoingHead:
    """The multi-value question: one binary LogisticRegression per value (one-vs-rest)."""

    def __init__(self, values: tuple[str, ...]) -> None:
        self.values = values
        self.kind = "untrained"
        self.reason: str | None = "no_training_windows"
        self.vectorizer: Any = None
        self.per_value: dict[str, LogisticRegression | float] = {}  # a float is a constant p (1.0 or 0.0)

    def fit(self, rung: int, inputs: list[Any], y: list[tuple[str, ...]], seed: int) -> _DoingHead:
        if not inputs:
            return self
        present = {v: [int(v in t) for t in y] for v in self.values}
        if all(len(set(p)) == 1 for p in present.values()):
            self.per_value = {v: float(p[0]) for v, p in present.items()}
            self.kind, self.reason = "constant", None
            return self
        fitted = _fit_vectorizer(rung, inputs)
        if fitted is None:
            self.reason = "no_vocabulary"
            return self
        self.vectorizer, matrix = fitted
        for v, p in present.items():
            self.per_value[v] = float(p[0]) if len(set(p)) == 1 else _classifier(seed).fit(matrix, p)
        self.kind, self.reason = "one_vs_rest", None
        return self

    def proba(self, inputs: list[Any]) -> np.ndarray:
        """p(value present) per input, one column per value in scheme order."""
        matrix = self.vectorizer.transform(inputs) if self.vectorizer is not None else None
        cols = []
        for v in self.values:
            est = self.per_value[v]
            if isinstance(est, LogisticRegression):
                cols.append(est.predict_proba(matrix)[:, list(est.classes_).index(1)])
            else:
                cols.append(np.full(len(inputs), est))
        return np.column_stack(cols)

    def predict(self, inputs: list[Any]) -> list[Answer | None]:
        if self.kind == "untrained":
            return [None] * len(inputs)
        if not inputs:
            return []
        out: list[Answer | None] = []
        for p in self.proba(inputs):
            chosen = [i for i, pi in enumerate(p) if pi >= DOING_MIN_P] or [int(np.argmax(p))]
            out.append(Answer(tuple(self.values[i] for i in chosen), float(min(p[i] for i in chosen))))
        return out

    def coef_row(self, value: str) -> tuple[np.ndarray, float | None] | None:
        """That value's binary decision function: its coefficients and intercept."""
        est = self.per_value.get(value)
        if not isinstance(est, LogisticRegression):
            return None
        return est.coef_[0], float(est.intercept_[0])

    def describe(self) -> dict[str, Any]:
        if self.kind == "untrained":
            return {"kind": "untrained", "reason": self.reason}
        if self.kind == "constant":
            return {"kind": "constant", "value": [v for v in self.values if self.per_value[v] == 1.0]}
        return {
            "kind": "one_vs_rest",
            "values": {
                v: ("model" if isinstance(e, LogisticRegression) else "always" if e == 1.0 else "never")
                for v, e in self.per_value.items()
            },
        }


def _question_values(q: Question) -> tuple[str, ...]:
    return q.values + ((q.blank_value,) if q.blank_value else ())


# --- fit, predict --------------------------------------------------------------------------------


def fit(
    rung: Literal[0, 1],
    ds: Dataset,
    split: Split,
    seed: int,
    emote_mode: Literal["name", "id"] = "name",
) -> Model:
    """One head per question on the train split (see the module docstring)."""
    if isinstance(rung, bool) or rung not in (0, 1):
        raise ValueError("rung must be 0 or 1")
    if emote_mode not in features.EMOTE_MODES:
        raise ValueError(f"emote_mode must be one of {', '.join(features.EMOTE_MODES)}")
    heads: dict[str, object] = {}
    described: dict[str, dict[str, Any]] = {}
    for q in ds.scheme.questions:
        wids = [w for w in split.train if q.key in ds.train_targets.get(w, {})]
        usable = [w for w in wids if rung == 0 or w in ds.windows]
        inputs = _inputs(rung, emote_mode, ds, usable)
        y = [ds.train_targets[w][q.key] for w in usable]
        head: _SingleHead | _DoingHead
        if q.kind == "multi":
            head = _DoingHead(_question_values(q)).fit(rung, inputs, y, seed)
        else:
            head = _SingleHead(_question_values(q)).fit(rung, inputs, y, seed)
        heads[q.key] = head
        described[q.key] = {**head.describe(), "n_train": len(usable), "n_without_text": len(wids) - len(usable)}

    config: dict[str, Any] = {
        "rung": rung,
        "emote_mode": emote_mode,
        "seed": seed,
        "origin": ds.origin,
        "labels_version": ds.scheme.version,
        "guide_version": ds.scheme.guide_version,
        "features": "counts_row" if rung == 0 else "text_doc",
        "vectorizer": {"kind": "DictVectorizer", "scaler": "MaxAbsScaler"}
        if rung == 0
        else {
            "kind": "TfidfVectorizer",
            "ngram_range": list(NGRAM_RANGE),
            "min_df": MIN_DF,
            "token_pattern": TOKEN_PATTERN,
        },
        "classifier": {"kind": "LogisticRegression", "max_iter": MAX_ITER, "class_weight": CLASS_WEIGHT},
        "doing": {"one_vs_rest": True, "min_p": DOING_MIN_P, "fallback": "top_value", "confidence": "min_p_chosen"},
        "heads": described,
    }
    return Model(rung=rung, emote_mode=emote_mode, heads=heads, config=config)


def _inputs(rung: int, emote_mode: str, ds: Dataset, wids: Sequence[str]) -> list[Any]:
    if rung == 0:
        return [features.counts_row(ds.counts[w]) for w in wids]
    return [features.text_doc(ds.windows[w], emote_mode) for w in wids]  # type: ignore[arg-type]


def predict(model: Model, items: Sequence[tuple[Counts, Window | None]]) -> list[dict[str, Answer | None]]:
    """Every head's answer for each ``(counts, window)``; ``None`` where a head can't answer.

    Rung 0 needs only the counts. Rung 1 needs the window's text, so a ``None`` window
    gets no answers.
    """
    if model.rung == 0:
        inputs: list[Any] = [features.counts_row(c) for c, _ in items]
    else:
        mode = model.emote_mode
        inputs = [None if w is None else features.text_doc(w, mode) for _, w in items]  # type: ignore[arg-type]
    usable = [i for i, x in enumerate(inputs) if x is not None]
    out: list[dict[str, Answer | None]] = [dict.fromkeys(model.heads) for _ in items]
    for q, head in model.heads.items():
        answers = head.predict([inputs[i] for i in usable]) if usable else []  # type: ignore[attr-defined]
        for i, a in zip(usable, answers):
            out[i][q] = a
    return out


def _scorable(model: Model, ds: Dataset, wids: Sequence[str]) -> dict[str, dict[str, Answer | None]]:
    """Answers for the windows the model can score (rung 1: windows with text)."""
    usable = [w for w in wids if model.rung == 0 or w in ds.windows]
    answers = predict(model, [(ds.counts[w], ds.windows.get(w)) for w in usable])
    return dict(zip(usable, answers))


# --- thresholds and reports ----------------------------------------------------------------------


def calibrate(
    model: Model, ds: Dataset, split: Split, target: float = TARGET_AGREEMENT
) -> dict[str, float | None]:
    """Per question, the smallest threshold whose answered validation agreement reaches ``target``."""
    if isinstance(target, bool) or not isinstance(target, (int, float)) or not 0 < target <= 1:
        raise ValueError("target must be a number above 0 and at most 1")
    answers = _scorable(model, ds, [w for w in split.val if w in ds.graded])
    out: dict[str, float | None] = {}
    for q in ds.scheme.questions:
        multi = q.kind == "multi"
        pairs: list[tuple[Any, Answer]] = []
        for w, a in answers.items():
            answer = a.get(q.key)
            if answer is not None and q.key in ds.grade_targets.get(w, {}):
                pairs.append((ds.grade_targets[w][q.key], answer))
        out[q.key] = _threshold(pairs, multi, target)
    return out


def _threshold(pairs: Sequence[tuple[Any, Answer]], multi: bool, target: float) -> float | None:
    """The smallest grid threshold with at least one answer and answered agreement >= ``target``."""
    for t in THRESHOLDS:
        shown = [(y, a) for y, a in pairs if a.confidence >= t]
        if shown and sum(metrics.agrees(y, a.value, multi) for y, a in shown) / len(shown) >= target:
            return t
    return None


def report(
    model: Model, ds: Dataset, split: Split, thresholds: dict[str, float | None]
) -> dict[str, metrics.QuestionReport]:
    """Per question, metrics on the graded test windows against ``grade_targets``.

    An answer is shown when the question has a threshold and the confidence reaches
    it; otherwise the window counts as abstained.
    """
    answers = _scorable(model, ds, [w for w in split.test if w in ds.graded])
    out: dict[str, metrics.QuestionReport] = {}
    for q in ds.scheme.questions:
        t = thresholds.get(q.key)

        def shown(a: Answer | None) -> Any:
            return a.value if a is not None and t is not None and a.confidence >= t else None

        graded = [
            (ds.grade_targets[w][q.key], shown(a.get(q.key)))
            for w, a in answers.items()
            if q.key in ds.grade_targets.get(w, {})
        ]
        aron_not_sure = [w for w in answers if q.key in ds.grade_not_sure.get(w, set())]
        abstained = sum(shown(answers[w].get(q.key)) is None for w in aron_not_sure)
        out[q.key] = metrics.question_report(
            _question_values(q), q.kind == "multi", graded, (abstained, len(aron_not_sure))
        )
    return out


# --- evidence ------------------------------------------------------------------------------------


def evidence(model: Model, w: Window, question: str, value: str, k: int = MAX_EVIDENCE) -> list[str]:
    """Up to ``k`` (at most 5) message IDs of ``w`` whose lines support ``value``, best first.

    Lines are scored alone by ``line_score``; only scores above 0 are kept, and ties
    keep the window's order.
    """
    if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k <= MAX_EVIDENCE:
        raise ValueError(f"k must be a whole number from 0 to {MAX_EVIDENCE}")
    _check_value(model, question, value)
    scored = []
    for m in w.messages:
        s = line_score(model, m, question, value)
        if s is not None and s > 0:
            scored.append((-s, m.rel_ms, m.message_id))
    scored.sort()
    return [mid for _, _, mid in scored[:k]]


def line_score(model: Model, m: Message, question: str, value: str) -> float | None:
    """How much one line alone supports ``value`` (see the module docstring); ``None`` when it can't be scored."""
    _check_value(model, question, value)
    head = model.heads[question]
    coef_row = getattr(head, "coef_row", None)
    row = coef_row(value) if coef_row is not None else None
    if row is None:
        return None
    coef, intercept = row
    vec = head.vectorizer  # type: ignore[attr-defined]
    if model.rung == 0:
        names = {features.EMOTE_PREFIX + e.id for e in m.emotes}
        if "?" in m.text:
            names.add(QUESTION_FEATURE)
        vocabulary = _vocabulary(vec)
        scores = [float(coef[vocabulary[n]]) for n in names if n in vocabulary]
        return max(scores) if scores else None
    x = vec.transform([features.message_doc(m, model.emote_mode)])  # type: ignore[arg-type]
    if x.nnz == 0:
        return None
    return float((x @ coef)[0]) + (intercept or 0.0)


def _check_value(model: Model, question: str, value: str) -> None:
    head = model.heads.get(question)
    if head is None:
        raise ValueError("question: not a question this model answers")
    values = getattr(head, "values", ())
    if value not in values:
        raise ValueError(f"value: not a value of {question}")
