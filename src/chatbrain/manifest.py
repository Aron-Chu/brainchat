"""Model manifests (spec section 8 lineage), retiring, the active model per question, saved predictions.

``write`` saves ``<models_dir>/<model_id>/model.joblib`` (``joblib.dump`` of the
``train.Model``) and ``manifest.json`` beside it. ``model_id`` is the first 12 hex
characters of the model file's sha256. The manifest holds:

- ``model_id``, ``model_sha256``, ``rung``, ``origin`` (C16), ``base_checkpoint`` (null for
  rungs 0-1), ``trained_at``;
- ``input_sha256``: sha256 of the sorted ``window_id|label_id|labels_version`` lines, one
  per label row the dataset was built from (one origin, the scheme's version, rows that
  name a question), each line ending in ``\\n``;
- ``eligible_as_of``: the newest deletion check, where an export counts as one for the
  windows it carries (B7), so the gate's own "newest";
- ``splits`` (window IDs per split) and ``split_channels`` (channels per split);
- ``text_message_counts``: for a rung that holds words (rung >= 1), each split window whose
  text the model could see and its message count then; ``{}`` for rung 0;
- ``labels_version``, ``guide_version``, ``teacher`` (the sorted ``model/prompt_version``
  of the suggested labels among the inputs), ``config``, ``seed``, ``thresholds``,
  ``report``, ``self_consistency`` (C28: the owner's first vs repeat label per question, as
  ``{agree, compared, ci}``) and ``versions`` (Python and library versions).

The manifest holds window IDs, channels, counts and scheme values, never chat text.

Before anything is written, ``write`` reads the label rows again and refuses
(``LabelsChanged``) unless they still give the dataset's targets, so ``input_sha256``
names the labels the model was trained on. Then the model file and the manifest are
written, and one transaction inserts the ``manifests`` row (or re-activates it when the
model ID already exists, C16) and retires every other unretired model of the same rung and
origin (``retired_at = now``). If that transaction fails, a model file it would have
created is deleted again, so no file holding words is left without a row.

A model's files are always found from the models folder in use, as
``<models_dir>/<model_id>/``, never from a stored absolute path, so a moved, restored or
copied data folder works on its own files only. The row's ``path`` is relative to that
folder (``<model_id>/model.joblib``). ``models_dir`` defaults to the ``models`` folder
beside the store's database, which is where the CLI keeps it.

``active_per_question`` maps each question to a model ID. It looks only at unretired
models whose file still exists and whose manifest can be read; owner-origin (``aron``) models when any
exist, otherwise synthetic ones (C16); the newest model of each rung. Going up the ladder,
a higher rung replaces the current one for a question only when its test-agreement Wilson
lower bound is above the current one's point estimate (spec section 8: it must win by more
than the interval). A model that answered nothing on test has a point estimate of 0.

``save_predictions`` stores one row per window that still has text and per question the
model answers (the raw answer and its confidence; thresholds are applied when shown),
replacing that model's earlier rows.

Word-model files are deleted by ``expiry.run`` (rung >= 1: once every split window's text
is gone, or once a deletion check removed a message it saw, C5), which also deletes model
files no row accounts for (``.writing-*`` dumps and ``<model_id>/model.joblib`` leftovers
of an interrupted write). This module never imports
scikit-learn or joblib at import time, so ``expiry`` and the CLI stay quick to load.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import metrics
from .export_format import iso
from .ingest import remove_file
from .store import Store

if TYPE_CHECKING:
    from .train import Dataset, Model, Split

MODEL_NAME = "model.joblib"
MANIFEST_NAME = "manifest.json"
MODELS_DIR_NAME = "models"  # the data folder's models/ (paths.SUBDIRS), beside cb.db
WRITING_PREFIX = ".writing-"  # temp dumps in models/ while a model is written
ID_LENGTH = 12
WRITTEN_ID = re.compile(rf"[0-9a-f]{{{ID_LENGTH}}}")  # the folder names write makes
_PLAIN_ID = re.compile(r"[0-9A-Za-z_-]{1,64}")  # one plain folder name: no separator, dot or drive
WORD_RUNG = 1  # rungs at or above this hold word features (rung 1: the TF-IDF vocabulary)
RUNGS: tuple[int, ...] = (0, 1)  # what P0 writes
SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")
LIBRARIES: tuple[str, ...] = ("chatbrain", "scikit-learn", "numpy", "scipy", "joblib")


class LabelsChanged(ValueError):
    """The store's labels no longer give the dataset the model was trained on."""


@dataclass(frozen=True)
class PredictionsSaved:
    windows: int
    rows: int


# --- write ---------------------------------------------------------------------------------------


def write(
    model: Model,
    ds: Dataset,
    split: Split,
    thresholds: Mapping[str, float | None],
    report: Mapping[str, metrics.QuestionReport],
    *,
    seed: int,
    store: Store,
    models_dir: Path,
    versions: Mapping[str, str],
    now: datetime | None = None,
) -> Path:
    """Save the model and its manifest, record it, retire older models of its rung; return the manifest path.

    ``now`` defaults to the current UTC time.
    """
    import joblib  # loaded only when a model is written

    questions = [q.key for q in ds.scheme.questions]
    if isinstance(model.rung, bool) or model.rung not in RUNGS:
        raise ValueError(f"rung must be one of {', '.join(map(str, RUNGS))} in P0")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be a whole number")
    if model.config.get("seed", seed) != seed:
        raise ValueError("seed is not the seed the model was fit with")
    if ds.origin not in metrics.ORIGINS or not ds.rows:
        raise ValueError("the dataset holds no labels; nothing to write")
    if model.config.get("origin", ds.origin) != ds.origin:
        raise ValueError("the model was fit on another origin's labels")
    if set(thresholds) != set(questions) or set(report) != set(questions):
        raise ValueError("thresholds and report must name every question of the scheme")
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in versions.items()):
        raise ValueError("versions must map names to version strings")
    now = datetime.now(timezone.utc) if now is None else now
    now_s = iso(now)

    input_sha256, teacher = _inputs(store, ds)
    eligible_as_of = _eligible_as_of(store)
    self_consistency = metrics.self_consistency(store, ds.scheme)

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    tmp = models_dir / f"{WRITING_PREFIX}{uuid.uuid4().hex}.joblib"
    try:
        joblib.dump(model, tmp)
        digest = _sha256(tmp)
        model_id = digest[:ID_LENGTH]
        folder = models_dir / model_id
        folder.mkdir(exist_ok=True)
        model_file = folder / MODEL_NAME
        file_existed = model_file.exists()
        os.replace(tmp, model_file)
    finally:
        remove_file(tmp)  # gone already after the replace; a leftover would hold the model's words

    doc: dict[str, Any] = {
        "model_id": model_id,
        "model_sha256": digest,
        "rung": model.rung,
        "origin": ds.origin,
        "base_checkpoint": None,
        "input_sha256": input_sha256,
        "eligible_as_of": eligible_as_of,
        "trained_at": now_s,
        "splits": {name: list(getattr(split, name)) for name in SPLIT_NAMES},
        "split_channels": {
            name: sorted({ds.rows[w].channel for w in getattr(split, name)}) for name in SPLIT_NAMES
        },
        "text_message_counts": _text_message_counts(model, ds, split),
        "labels_version": ds.scheme.version,
        "guide_version": ds.scheme.guide_version,
        "teacher": teacher,
        "config": model.config,
        "seed": seed,
        "thresholds": {q: thresholds[q] for q in questions},
        "report": {q: report[q].to_dict() for q in questions},
        "self_consistency": {
            q: {"agree": agree, "compared": compared, "ci": [lo, hi]}
            for q, (agree, compared, lo, hi) in self_consistency.items()
        },
        "versions": dict(versions),
    }
    manifest_path = folder / MANIFEST_NAME
    try:
        _write_json(manifest_path, doc)
        _record(store, model_id, model.rung, ds.origin, now_s)
    except BaseException:
        if not file_existed:
            remove_file(model_file)
        raise
    return manifest_path


def _inputs(store: Store, ds: Dataset) -> tuple[str, list[str]]:
    """``input_sha256`` and ``teacher`` from the label rows the dataset was built from.

    Raises ``LabelsChanged`` when those rows no longer give the dataset's targets.
    """
    from . import train

    with store.lock:
        rows = [r for r in metrics.label_rows(store, ds.scheme, ds.origin) if r.answers or r.not_sure]
        teachers = store.conn.execute(
            "SELECT id, teacher_model, prompt_version FROM labels"
            " WHERE origin = ? AND labels_version = ? AND teacher_model IS NOT NULL",
            (ds.origin, ds.scheme.version),
        ).fetchall()

    targets, not_sure = train._latest(rows)
    if ds.origin == "aron":
        grade_targets, _ = train._latest([r for r in rows if r.project in train.GRADED_PROJECTS])
    else:
        grade_targets = targets
    if targets != ds.train_targets or not_sure != ds.not_sure or grade_targets != ds.grade_targets:
        raise LabelsChanged("labels changed while the model was training; run train again")

    lines = sorted(f"{r.window_id}|{r.id}|{ds.scheme.version}" for r in rows)
    input_sha256 = hashlib.sha256("".join(f"{line}\n" for line in lines).encode("utf-8")).hexdigest()
    used = {r.id for r in rows}
    teacher = sorted({f"{m}/{p or 'unknown'}" for label_id, m, p in teachers if label_id in used})
    return input_sha256, teacher


def _eligible_as_of(store: Store) -> str | None:
    """The newest deletion check or export coverage (B7), as stored; None when there is none."""
    with store.lock:
        newest = [
            store.conn.execute("SELECT MAX(checked_at) FROM deletion_checks").fetchone()[0],
            store.conn.execute("SELECT MAX(checked_at) FROM checked_windows").fetchone()[0],
        ]
    found = [t for t in newest if t is not None]
    return max(found) if found else None


def _text_message_counts(model: Model, ds: Dataset, split: Split) -> dict[str, int]:
    """Per split window whose text a word rung saw, its message count then; ``{}`` for rung 0 (C5)."""
    if model.rung < WORD_RUNG:
        return {}
    ids = sorted({w for name in SPLIT_NAMES for w in getattr(split, name)})
    return {w: len(ds.windows[w].messages) for w in ids if w in ds.windows}


def _record(store: Store, model_id: str, rung: int, origin: str, now_s: str) -> None:
    """Insert or re-activate the ``manifests`` row and retire the rest of its rung and origin."""
    path = f"{model_id}/{MODEL_NAME}"  # relative to the models folder; never an absolute path
    with store.transaction() as conn:
        if conn.execute("SELECT 1 FROM manifests WHERE model_id = ?", (model_id,)).fetchone():
            conn.execute(
                "UPDATE manifests SET rung = ?, origin = ?, path = ?, retired_at = NULL, file_deleted_at = NULL"
                " WHERE model_id = ?",
                (rung, origin, path, model_id),
            )
        else:
            conn.execute(
                "INSERT INTO manifests (model_id, rung, origin, path, created_at) VALUES (?, ?, ?, ?, ?)",
                (model_id, rung, origin, path, now_s),
            )
        conn.execute(
            "UPDATE manifests SET retired_at = ? WHERE rung = ? AND origin = ? AND model_id != ? AND retired_at IS NULL",
            (now_s, rung, origin, model_id),
        )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, doc: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def default_models_dir(store: Store) -> Path:
    """The ``models`` folder beside the store's database (``<data>/models``)."""
    return Path(store.path).parent / MODELS_DIR_NAME


def model_folder(models_dir: Path, model_id: str) -> Path | None:
    """``<models_dir>/<model_id>``, or None when ``model_id`` isn't one plain folder name."""
    return Path(models_dir) / model_id if _PLAIN_ID.fullmatch(model_id) else None


def read(folder: Path) -> dict[str, Any] | None:
    """The manifest in ``folder``, or None when it is missing or isn't a JSON object."""
    try:
        doc = json.loads((Path(folder) / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError, RecursionError):
        return None
    return doc if isinstance(doc, dict) else None


def library_versions() -> dict[str, str]:
    """Python's version and each library's installed version (``unknown`` when not installed)."""
    out = {"python": platform.python_version()}
    for name in LIBRARIES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "unknown"
    return out


# --- the active model per question -----------------------------------------------------------------


@dataclass(frozen=True)
class _Candidate:
    model_id: str
    rung: int
    origin: str
    scores: dict[str, tuple[float, float]]  # question -> (point estimate, Wilson lower bound)


def active_per_question(store: Store, *, models_dir: Path | None = None) -> dict[str, str]:
    """Each question's model ID (see the module docstring); ``{}`` when no model is usable.

    Model files are looked up in ``models_dir`` (default: ``default_models_dir(store)``).
    """
    models_dir = default_models_dir(store) if models_dir is None else Path(models_dir)
    with store.lock:
        rows = store.conn.execute(
            "SELECT model_id, rung, origin FROM manifests WHERE retired_at IS NULL AND file_deleted_at IS NULL"
            " ORDER BY created_at DESC, model_id"
        ).fetchall()
    usable: list[_Candidate] = []
    for model_id, rung, origin in rows:
        folder = model_folder(models_dir, model_id)
        if folder is None or not (folder / MODEL_NAME).is_file():
            continue
        scores = _scores(read(folder))
        if scores is not None:
            usable.append(_Candidate(model_id, rung, origin, scores))
    if not usable:
        return {}
    origin = "aron" if any(c.origin == "aron" for c in usable) else "synthetic"
    newest: dict[int, _Candidate] = {}
    for c in usable:  # newest first
        if c.origin == origin:
            newest.setdefault(c.rung, c)

    out: dict[str, str] = {}
    current: dict[str, _Candidate] = {}
    for rung in sorted(newest):
        c = newest[rung]
        for q, (_, lower) in c.scores.items():
            held = current.get(q)
            if held is None or lower > held.scores[q][0]:
                current[q] = c
    for q, c in current.items():
        out[q] = c.model_id
    return out


def _scores(doc: dict[str, Any] | None) -> dict[str, tuple[float, float]] | None:
    """Per question ``(agreement or 0, Wilson lower bound or 0)`` from a manifest's report; None if unusable."""
    report = doc.get("report") if doc is not None else None
    if not isinstance(report, dict):
        return None
    out: dict[str, tuple[float, float]] = {}
    for q, r in report.items():
        if not isinstance(r, dict):
            return None
        agreement, ci = r.get("agreement"), r.get("agreement_ci")
        if agreement is None:
            point = lower = 0.0
        elif _number(agreement) and isinstance(ci, list) and len(ci) == 2 and _number(ci[0]):
            point, lower = float(agreement), float(ci[0])
        else:
            return None
        out[q] = (point, lower)
    return out


def _number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# --- predictions -----------------------------------------------------------------------------------


def save_predictions(store: Store, model: Model, model_id: str, *, now: datetime) -> PredictionsSaved:
    """Store ``model``'s answers for every window that still has text, replacing its earlier rows."""
    from . import train
    from .spread import Counts

    now_s = iso(now)
    with store.lock:
        if not store.conn.execute("SELECT 1 FROM manifests WHERE model_id = ?", (model_id,)).fetchone():
            raise KeyError("model_id: no such model in the store")
        kept: list[str] = []
        items: list[tuple[Counts, Any]] = []
        for w in store.window_ids(text_state="present"):
            found = store.conn.execute("SELECT json FROM features WHERE window_id = ?", (w,)).fetchone()
            if found is None:
                continue
            kept.append(w)
            # rung 0 reads the counts only; a word rung needs the text
            items.append((Counts.from_json(found[0]), store.as_window(w) if model.rung >= WORD_RUNG else None))

    rows = []
    for w, answers in zip(kept, train.predict(model, items)):
        for q, a in answers.items():
            if a is None:
                continue
            value = list(a.value) if isinstance(a.value, tuple) else a.value
            rows.append((w, model_id, q, json.dumps(value), a.confidence, now_s))
    with store.transaction() as conn:
        conn.execute("DELETE FROM predictions WHERE model_id = ?", (model_id,))
        conn.executemany(
            "INSERT INTO predictions (window_id, model_id, question, value_json, confidence, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
    return PredictionsSaved(windows=len(kept), rows=len(rows))
