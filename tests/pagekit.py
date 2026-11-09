"""Shared helpers for the Moment page tests: a made-up world in tmp_path, stub models, rows.

Every store here is built from ``fake`` in a temporary folder; nothing reads the real data folder.
"""

from __future__ import annotations

import dataclasses
import html as html_mod
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from chatbrain import fake, ingest, manifest
from chatbrain import scheme as scheme_mod
from chatbrain.export_format import StreamActivity, iso, parse_export
from chatbrain.page import app as app_mod
from chatbrain.store import Store

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)
SCHEME = scheme_mod.load()
QUESTIONS = [q.key for q in SCHEME.questions]
RUNG0_ID = "a00000000000"
RUNG1_ID = "b11111111111"
ARON0_ID = "c22222222222"


def build_world(tmp: Path, *, activity: bool = True, raw: bytes | None = None) -> SimpleNamespace:
    """Two made-up streams on one channel: ``old`` (text gone by NOW + 5 days) and ``new`` (the newest).

    ``activity=False`` imports P0's ``chatbrain-export/1`` file, with the same windows and no activity lines.
    ``raw`` imports that export (made from the same made-up one) in its place.
    """
    if raw is None:
        raw = fake.make_export(3, 12, 16, NOW, channels=1, activity=activity)
    (tmp / "e.json").write_bytes(raw)
    store = Store(tmp / "cb.db")
    ingest.import_export(store, tmp / "e.json", tmp / "imported", NOW)
    export = parse_export(raw)
    by_stream: dict[str, list] = {}
    for w in export.windows:
        by_stream.setdefault(w.stream_id, []).append(w)
    ordered = sorted(by_stream, key=lambda s: max(w.end_at for w in by_stream[s]))
    old, new = ordered[0], ordered[-1]
    assert min(w.text_expires_at for w in by_stream[old]) <= NOW + timedelta(days=5)
    assert min(w.text_expires_at for w in by_stream[new]) > NOW + timedelta(days=5)

    def moments(stream: str) -> list[str]:
        return [w.window_id for w in sorted(by_stream[stream], key=lambda w: w.anchor_at) if w.anchor_kind == "moment"]

    return SimpleNamespace(
        store=store,
        tmp=tmp,
        raw=raw,
        export=export,
        windows={w.window_id: w for w in export.windows},
        old=old,
        new=new,
        old_moments=moments(old),
        new_moments=moments(new),
        models=tmp / "models",
    )


def activity_line(
    world: SimpleNamespace,
    stream_id: str,
    *,
    base: int | None = 9,
    peaks: dict[str, int] | None = None,
    gaps: tuple[tuple[int, int], ...] = (),
    shift: timedelta = timedelta(0),
) -> StreamActivity:
    """Made-up people-chatting counts over ``stream_id``'s exported span: ``base`` in every 10 s bin,
    ``peaks`` (window ID -> count) in those Moments' anchor bins, and None over ``gaps`` ((first bin,
    bins) pairs). ``shift`` moves the whole line (and its 30-day deadline) earlier. Numbers only."""
    src = next(s for s in world.export.streams if s.stream_id == stream_id)
    chatters: list[int | None] = [base] * len(src.chatters)
    for window_id, count in (peaks or {}).items():
        chatters[int((world.windows[window_id].anchor_at - src.started_at).total_seconds()) // src.bin_s] = count
    for first, run in gaps:
        chatters[first : first + run] = [None] * run
    return dataclasses.replace(
        src, started_at=src.started_at - shift, ended_at=src.ended_at - shift, chatters=tuple(chatters)
    )


def store_activity(store: Store, a: StreamActivity, *, now: datetime = NOW) -> str:
    """Store ``a`` as its stream's activity line in place of the imported one; returns the put's result.

    The stream's Moment tallies (F1) take the new line's deadline, as if it were the only one;
    the put deletes them when the line is already expired."""
    with store.transaction() as conn:
        conn.execute("DELETE FROM stream_activity WHERE stream_id = ?", (a.stream_id,))
        put = store.put_stream_activity(a, NOW, now)
        conn.execute(
            "UPDATE moment_emotes SET expires_at = (SELECT expires_at FROM stream_activity WHERE stream_id = ?)"
            " WHERE stream_id = ? AND EXISTS (SELECT 1 FROM stream_activity WHERE stream_id = ?)",
            (a.stream_id, a.stream_id, a.stream_id),
        )
        return put


def clear_activity(store: Store) -> None:
    """Drop every stored activity line (not the expired markers), so a test's own lines are the only ones.

    The Moment tallies (F1) go too, so a counts-missing test world never shows a face."""
    with store.transaction() as conn:
        conn.execute("DELETE FROM stream_activity")
        conn.execute("DELETE FROM moment_emotes")


def counts(store: Store, window_id: str) -> dict:
    return json.loads(store.conn.execute("SELECT json FROM features WHERE window_id = ?", (window_id,)).fetchone()[0])


def report(point: float = 0.8, lower: float = 0.7, questions=None) -> dict:
    return {
        q: {"agreement": point, "agreement_ci": [lower, min(1.0, point + 0.1)], "answered": 100}
        for q in (questions or QUESTIONS)
    }


def stub_model(
    store: Store,
    models: Path,
    model_id: str,
    rung: int,
    *,
    origin: str = "synthetic",
    thresholds: dict | None = None,
    rep: dict | None = None,
    at: int = 0,
) -> Path:
    """A usable model row and folder; ``model.joblib`` is a stub that can't be loaded."""
    folder = models / model_id
    folder.mkdir(parents=True)
    (folder / manifest.MODEL_NAME).write_bytes(b"stub")
    windows = store.window_ids()
    doc = {
        "model_id": model_id,
        "rung": rung,
        "origin": origin,
        "report": rep if rep is not None else report(),
        "thresholds": {q: 0.5 for q in QUESTIONS} | (thresholds or {}),
        # a word model's files outlive page start's expiry only while a split window keeps its text
        "splits": {"train": windows, "val": [], "test": []},
        "text_message_counts": {},
    }
    (folder / manifest.MANIFEST_NAME).write_text(json.dumps(doc), encoding="utf-8")
    with store.transaction() as conn:
        conn.execute(
            "INSERT INTO manifests (model_id, rung, origin, path, created_at) VALUES (?, ?, ?, ?, ?)",
            (model_id, rung, origin, f"{model_id}/{manifest.MODEL_NAME}", iso(NOW + timedelta(minutes=at))),
        )
    return folder


def predict(store: Store, window_id: str, model_id: str, question: str, value, confidence: float, at: int = 0) -> None:
    with store.transaction() as conn:
        conn.execute(
            "INSERT INTO predictions (window_id, model_id, question, value_json, confidence, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (window_id, model_id, question, json.dumps(value), confidence, iso(NOW + timedelta(minutes=at))),
        )


def label(
    store: Store,
    window_id: str,
    answers: dict | None = None,
    not_sure: dict | None = None,
    *,
    origin: str = "aron",
    provenance: str = "blind",
    project: str = "blind",
    at: int = 0,
    version: str | None = None,
    cause_note: str | None = None,
    cause_source: str | None = None,
) -> None:
    with store.transaction() as conn:
        conn.execute(
            "INSERT INTO labels (window_id, project, origin, provenance, labels_version, guide_version,"
            " answers_json, not_sure_json, cause_note, cause_source, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                window_id, project, origin, provenance, version or SCHEME.version, SCHEME.guide_version,
                json.dumps(answers or {}), json.dumps(not_sure or {}), cause_note, cause_source,
                iso(NOW + timedelta(minutes=at)),
            ),
        )  # fmt: skip


def client(
    world: SimpleNamespace, *, now: datetime = NOW, clock=None, review=(), narrator=None, cpu_captions: bool = False
) -> TestClient:
    """The page over ``world``; page start (expiry, rung-0 backfill) runs here, the hourly thread does not.

    ``narrator`` is None (no local model) or one built on a made-up Ollama (``ollamakit``); the
    client is used without ``with``, so the lifespan (and with it the model preload) never runs.
    """
    path = world.tmp / "review_emotes.json"
    path.write_text(json.dumps({"ids": list(review)}), encoding="utf-8")
    app = app_mod.create_app(
        world.store,
        SCHEME,
        world.models,
        imported_dir=world.tmp / "imported",
        clock=clock or (lambda: now),
        review_emotes=path,
        hourly=False,
        narrator=narrator,
        cpu_captions=cpu_captions,
    )
    return TestClient(app, base_url=f"http://{app_mod.HOST}:{app_mod.PORT}")  # the page answers to its own Host only


def cards(page: str) -> list[str]:
    """The window IDs of the Moment cards on a list page, in order."""
    return re.findall(r'<li class="mc[^"]*" data-window="([^"]+)"', page)


def text(fragment: str) -> str:
    """Visible text: tags dropped, entities decoded, whitespace folded."""
    return " ".join(html_mod.unescape(re.sub(r"<[^>]+>", " ", fragment)).split())


def label_rows(store: Store) -> list[dict]:
    cols = (
        "window_id", "project", "origin", "provenance", "labels_version", "guide_version", "answers_json",
        "not_sure_json", "cause_note", "cause_source",
    )  # fmt: skip
    rows = store.conn.execute(f"SELECT {', '.join(cols)} FROM labels ORDER BY id").fetchall()
    return [dict(zip(cols, r)) for r in rows]


def truth_labels(world: SimpleNamespace, *, skip=(), at: int = 0) -> dict:
    """Label every window (but ``skip``) with its made-up truth as the owner; returns the truth."""
    t = fake.truth(world.raw)
    for wid, answers in t.items():
        if wid not in skip:
            label(world.store, wid, answers, at=at)
    return t
