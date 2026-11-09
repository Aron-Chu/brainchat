import hashlib
import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import joblib
import pytest

from chatbrain import assign, cli, expiry, fake, ingest, manifest, metrics, paths, train
from chatbrain import scheme as scheme_mod
from chatbrain.export_format import iso, parse_deletion_check, parse_export
from chatbrain.store import Store

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)
SCHEME = scheme_mod.load()
QUESTIONS = [q.key for q in SCHEME.questions]
VERSIONS = {"python": "3.12.0", "scikit-learn": "1.9.1", "chatbrain": "0.1.0"}
TEACHER = ("qwen3:8b", "cb_teacher_2026_11_06_v1")
FIELDS = {
    "model_id", "model_sha256", "rung", "origin", "base_checkpoint", "input_sha256", "eligible_as_of",
    "trained_at", "splits", "split_channels", "text_message_counts", "labels_version", "guide_version",
    "teacher", "config", "seed", "thresholds", "report", "self_consistency", "versions",
}  # fmt: skip


@pytest.fixture(autouse=True)
def _no_real_data_folder(monkeypatch):
    # A CLI test that forgot --data-dir must fail, never fall back to the real data folder.
    monkeypatch.delenv(paths.ENV_VAR, raising=False)


def _labels(store: Store, truth: dict[str, dict], origin: str = "synthetic") -> None:
    """One label per window from ``fake.truth``; project = the window's exposure (ruling C1).

    The owner's suggested-project labels carry the teacher, as ``sync.pull`` writes them.
    """
    with store.transaction() as conn:
        exposure = dict(conn.execute("SELECT window_id, exposure FROM assignments").fetchall())
        rows = []
        for wid, answers in sorted(truth.items()):
            project = exposure[wid]
            model = prompt = None
            if origin == "synthetic":
                provenance = "synthetic"
            elif project == "suggested":
                provenance, (model, prompt) = "accepted", TEACHER
            else:
                provenance = project
            rows.append((wid, project, origin, provenance, SCHEME.version, SCHEME.guide_version, json.dumps(answers),
                         model, prompt, iso(NOW)))
        conn.executemany(
            "INSERT INTO labels (window_id, project, origin, provenance, labels_version, guide_version,"
            " answers_json, not_sure_json, teacher_model, prompt_version, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, '{}', ?, ?, ?)",
            rows,
        )


def _trained(ds, sp, rung: int, seed: int):
    m = train.fit(rung, ds, sp, seed)
    th = train.calibrate(m, ds, sp)
    return SimpleNamespace(model=m, thresholds=th, report=train.report(m, ds, sp, th), ds=ds, split=sp, seed=seed)


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    """A made-up world with synthetic labels, and rung 0 and 1 trained on it once."""
    tmp = tmp_path_factory.mktemp("base")
    raw = fake.make_export(7, 160, 40, NOW)
    (tmp / "e.json").write_bytes(raw)
    with Store(tmp / "cb.db") as store:
        ingest.import_export(store, tmp / "e.json", tmp / "imported", NOW)
        assign.assign_new(store, "7", NOW)
        _labels(store, fake.truth(raw))
        ds = train.load_dataset(store, SCHEME, "synthetic")
    sp = train.split(ds, 7)
    return SimpleNamespace(db=tmp / "cb.db", raw=raw, ds=ds, sp=sp, rungs={r: _trained(ds, sp, r, 7) for r in (0, 1)})


@pytest.fixture
def store(base, tmp_path):
    shutil.copyfile(base.db, tmp_path / "cb.db")
    with Store(tmp_path / "cb.db") as s:
        yield s


@pytest.fixture
def empty(tmp_path):
    with Store(tmp_path / "cb.db") as s:
        yield s


def _write(store: Store, models: Path, t, now: datetime = NOW) -> Path:
    return manifest.write(t.model, t.ds, t.split, t.thresholds, t.report, seed=t.seed, store=store,
                          models_dir=models, versions=VERSIONS, now=now)


def _doc(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _row(store: Store, model_id: str) -> dict:
    cols = ("model_id", "rung", "origin", "path", "created_at", "retired_at", "file_deleted_at")
    found = store.conn.execute(f"SELECT {', '.join(cols)} FROM manifests WHERE model_id = ?", (model_id,)).fetchone()
    assert found is not None
    return dict(zip(cols, found))


def _json(value):
    return json.loads(json.dumps(value))


# --- write -----------------------------------------------------------------------------------------


def test_manifest_fields_complete(base, store, tmp_path):
    # The owner's labels beside the synthetic ones: only the owner's are read, and the teacher comes from them.
    _labels(store, fake.truth(base.raw), origin="aron")
    checked = NOW + timedelta(hours=1)
    ingest.apply_deletion_check(store, parse_deletion_check(fake.make_deletion_check(base.raw, set(), checked)))
    ds = train.load_dataset(store, SCHEME, "aron")
    t = _trained(ds, train.split(ds, 7), 0, 7)
    written = NOW + timedelta(hours=2)

    path = _write(store, tmp_path / "models", t, written)
    doc = _doc(path)

    assert set(doc) == FIELDS
    assert doc["rung"] == 0 and doc["origin"] == "aron" and doc["base_checkpoint"] is None
    assert doc["seed"] == 7 and doc["trained_at"] == iso(written)
    assert doc["eligible_as_of"] == iso(checked)  # the newest deletion check
    assert doc["splits"] == {"train": t.split.train, "val": t.split.val, "test": t.split.test}
    assert doc["split_channels"] == {
        name: sorted({ds.rows[w].channel for w in ids})
        for name, ids in (("train", t.split.train), ("val", t.split.val), ("test", t.split.test))
    }
    assert not set(doc["split_channels"]["train"]) & set(doc["split_channels"]["test"])
    assert doc["labels_version"] == SCHEME.version and doc["guide_version"] == SCHEME.guide_version
    assert doc["teacher"] == ["/".join(TEACHER)]
    assert doc["config"] == _json(t.model.config)
    assert doc["thresholds"] == t.thresholds
    assert doc["report"] == _json({q: r.to_dict() for q, r in t.report.items()})
    assert doc["self_consistency"] == {
        q: {"agree": a, "compared": n, "ci": [lo, hi]} for q, (a, n, lo, hi) in metrics.self_consistency(
            store, SCHEME).items()
    }
    assert doc["versions"] == VERSIONS
    assert doc["text_message_counts"] == {}  # rung 0 holds no words

    lines = sorted(f"{w}|{i}|{v}" for i, w, v in store.conn.execute(
        "SELECT id, window_id, labels_version FROM labels WHERE origin = 'aron'"))
    assert doc["input_sha256"] == hashlib.sha256("".join(f"{line}\n" for line in lines).encode()).hexdigest()

    synthetic = _doc(_write(store, tmp_path / "models", base.rungs[1]))
    assert synthetic["teacher"] == [] and synthetic["origin"] == "synthetic"
    assert synthetic["text_message_counts"] == {
        w: len(base.ds.windows[w].messages) for ids in synthetic["splits"].values() for w in ids
    }


def test_model_id_is_file_hash_prefix(base, store, tmp_path):
    models = tmp_path / "models"
    path = _write(store, models, base.rungs[0])
    doc = _doc(path)
    model_file = path.parent / manifest.MODEL_NAME
    digest = hashlib.sha256(model_file.read_bytes()).hexdigest()

    assert path.name == manifest.MANIFEST_NAME
    assert doc["model_sha256"] == digest and doc["model_id"] == digest[:12] == path.parent.name
    assert path.parent.parent == models
    row = _row(store, doc["model_id"])
    assert row["rung"] == 0 and row["origin"] == "synthetic" and row["created_at"] == iso(NOW)
    assert row["path"] == f"{doc['model_id']}/{manifest.MODEL_NAME}"  # relative to models/, never absolute
    assert row["retired_at"] is None and row["file_deleted_at"] is None

    loaded = joblib.load(model_file)
    assert isinstance(loaded, train.Model) and loaded.rung == 0
    items = [(base.ds.counts[w], None) for w in base.sp.test[:5]]
    assert train.predict(loaded, items) == train.predict(base.rungs[0].model, items)

    # the same model again: the same ID, one row
    assert _write(store, models, base.rungs[0], NOW + timedelta(minutes=1)) == path
    assert store.conn.execute("SELECT COUNT(*) FROM manifests").fetchone()[0] == 1


def test_new_model_retires_same_rung_only(base, store, tmp_path):
    models = tmp_path / "models"
    at = [NOW + timedelta(minutes=i) for i in range(5)]
    a = _doc(_write(store, models, base.rungs[0], at[0]))["model_id"]
    b = _doc(_write(store, models, base.rungs[1], at[1]))["model_id"]
    c = _doc(_write(store, models, _trained(base.ds, train.split(base.ds, 8), 0, 8), at[2]))["model_id"]
    assert len({a, b, c}) == 3

    assert _row(store, a)["retired_at"] == iso(at[2])  # the older rung 0
    assert _row(store, b)["retired_at"] is None  # rung 1 is another rung
    assert _row(store, c)["retired_at"] is None

    # retiring is per (rung, origin): the owner's rung 0 leaves the synthetic rung 0 alone (C16)
    _labels(store, fake.truth(base.raw), origin="aron")
    ds_aron = train.load_dataset(store, SCHEME, "aron")
    d = _doc(_write(store, models, _trained(ds_aron, train.split(ds_aron, 7), 0, 7), at[3]))["model_id"]
    assert _row(store, c)["retired_at"] is None and _row(store, d)["retired_at"] is None

    # a model ID that already exists is re-activated, not inserted again (C16)
    assert _doc(_write(store, models, base.rungs[0], at[4]))["model_id"] == a
    assert _row(store, a)["retired_at"] is None
    assert _row(store, c)["retired_at"] == iso(at[4])
    assert _row(store, b)["retired_at"] is None and _row(store, d)["retired_at"] is None
    assert store.conn.execute("SELECT COUNT(*) FROM manifests").fetchone()[0] == 4


def _strings(value):
    if isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)
    elif isinstance(value, str):
        yield value


def test_manifest_contains_no_message_text(base, store, tmp_path):
    export = parse_export(base.raw)
    texts = {m.text for w in export.windows for m in w.messages}
    markers = {fake.marker(w.window_id) for w in export.windows}
    assert len(markers) > 100
    for rung in (0, 1):
        raw = _write(store, tmp_path / "models", base.rungs[rung]).read_text(encoding="utf-8")
        assert not set(_strings(json.loads(raw))) & texts  # no stored string is a message
        assert not [m for m in markers if m in raw]  # no marker line anywhere
        assert not [t for t in texts if " " in t and t in raw]  # no multi-word message anywhere


def test_write_refuses_labels_changed_since_loading(base, store, tmp_path):
    w = base.sp.train[0]
    (tone_q,) = [q for q in SCHEME.questions if q.key == "tone"]
    tone = next(v for v in tone_q.values if v != base.ds.train_targets[w].get("tone"))  # always a real change
    with store.transaction() as conn:
        conn.execute(
            "INSERT INTO labels (window_id, project, origin, provenance, labels_version, guide_version,"
            " answers_json, not_sure_json, created_at) VALUES (?, 'page', 'synthetic', 'synthetic', ?, ?, ?, '{}', ?)",
            (w, SCHEME.version, SCHEME.guide_version, json.dumps({"tone": tone}), iso(NOW + timedelta(hours=1))),
        )
    with pytest.raises(manifest.LabelsChanged):
        _write(store, tmp_path / "models", base.rungs[0])
    assert store.conn.execute("SELECT COUNT(*) FROM manifests").fetchone()[0] == 0
    assert not (tmp_path / "models").exists() or not any((tmp_path / "models").iterdir())


def test_save_predictions_covers_windows_with_text(base, store, tmp_path):
    model_id = _doc(_write(store, tmp_path / "models", base.rungs[0]))["model_id"]
    expiry.run(store, NOW + timedelta(days=5), tmp_path / "imported")  # some windows lose their text
    present = set(store.window_ids(text_state="present"))
    assert present and len(present) < len(base.ds.rows)

    saved = manifest.save_predictions(store, base.rungs[0].model, model_id, now=NOW + timedelta(days=5))
    rows = store.conn.execute(
        "SELECT window_id, question, value_json, confidence FROM predictions WHERE model_id = ?", (model_id,)
    ).fetchall()
    assert saved.windows == len(present) and saved.rows == len(rows) == len(present) * len(QUESTIONS)
    assert {r[0] for r in rows} == present
    values = {q.key: q.values + ((q.blank_value,) if q.blank_value else ()) for q in SCHEME.questions}
    for _, q, value_json, confidence in rows:
        value = json.loads(value_json)
        if q == "doing":
            assert isinstance(value, list) and value and set(value) <= set(values[q])
        else:
            assert value in values[q]
        assert 0.0 <= confidence <= 1.0

    again = manifest.save_predictions(store, base.rungs[0].model, model_id, now=NOW + timedelta(days=5))
    assert again == saved  # replaced, never doubled
    assert store.conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == len(rows)


# --- active_per_question ------------------------------------------------------------------------------


def _q(agree: int, answered: int) -> dict:
    lo, hi = metrics.wilson(agree, answered)
    return {"agreement": agree / answered if answered else None, "agreement_ci": [lo, hi], "answered": answered}


def _fake_model(store, models, model_id, rung, origin, report, *, at=0, retired=False, file=True):
    folder = models / model_id
    folder.mkdir(parents=True)
    if file:
        (folder / manifest.MODEL_NAME).write_bytes(b"stub")
    (folder / manifest.MANIFEST_NAME).write_text(
        json.dumps({"model_id": model_id, "rung": rung, "origin": origin, "report": report}), encoding="utf-8")
    with store.transaction() as conn:
        conn.execute(
            "INSERT INTO manifests (model_id, rung, origin, path, created_at, retired_at) VALUES (?, ?, ?, ?, ?, ?)",
            (model_id, rung, origin, str(folder / manifest.MODEL_NAME), iso(NOW + timedelta(minutes=at)),
             iso(NOW) if retired else None),
        )
    return folder


RUNG0 = {"doing": _q(70, 100), "aimed_at": _q(80, 100), "feeling": _q(0, 0), "tone": _q(50, 100)}
RUNG1 = {"doing": _q(95, 100), "aimed_at": _q(9, 10), "feeling": _q(3, 3), "tone": _q(0, 0)}


def test_active_per_question_rung1_must_beat_interval(empty, tmp_path):
    models = tmp_path / "models"
    assert manifest.active_per_question(empty) == {}
    zero = _fake_model(empty, models, "000000000000", 0, "synthetic", RUNG0)
    _fake_model(empty, models, "111111111111", 1, "synthetic", RUNG1, at=1)

    assert manifest.active_per_question(empty) == {
        "doing": "111111111111",  # lower bound 0.89 beats 0.70
        "aimed_at": "000000000000",  # 0.90 beats 0.80, but its lower bound (0.60) does not
        "feeling": "111111111111",  # rung 0 never answered: any agreement beats it
        "tone": "000000000000",  # rung 1 never answered
    }

    # a lower bound equal to rung 0's point estimate does not win
    tied = {**RUNG0, "doing": {**RUNG0["doing"], "agreement": RUNG1["doing"]["agreement_ci"][0]}}
    (zero / manifest.MANIFEST_NAME).write_text(json.dumps({"report": tied}), encoding="utf-8")
    assert manifest.active_per_question(empty)["doing"] == "000000000000"


def test_active_per_question_skips_retired_and_missing_files_and_prefers_aron(empty, tmp_path):
    models = tmp_path / "models"
    _fake_model(empty, models, "s0old0000000", 0, "synthetic", RUNG1)  # an older, unretired rung 0
    _fake_model(empty, models, "s00000000000", 0, "synthetic", RUNG0, at=1)
    s1 = _fake_model(empty, models, "s11111111111", 1, "synthetic", RUNG1, at=2)
    mixed = {"doing": "s11111111111", "aimed_at": "s00000000000", "feeling": "s11111111111",
             "tone": "s00000000000"}
    assert manifest.active_per_question(empty) == mixed

    _fake_model(empty, models, "a10000000000", 1, "aron", RUNG1, at=3, file=False)  # no file: never used
    assert manifest.active_per_question(empty) == mixed

    _fake_model(empty, models, "a00000000000", 0, "aron", RUNG0, at=4)
    assert manifest.active_per_question(empty) == dict.fromkeys(QUESTIONS, "a00000000000")  # the owner's win

    with empty.transaction() as conn:
        conn.execute("UPDATE manifests SET retired_at = ? WHERE model_id = 'a00000000000'", (iso(NOW),))
    assert manifest.active_per_question(empty) == mixed

    (s1 / manifest.MODEL_NAME).unlink()
    assert manifest.active_per_question(empty) == dict.fromkeys(QUESTIONS, "s00000000000")


# --- expiry and deletion checks reach word models -----------------------------------------------------


def test_expiry_deletes_word_model_file_after_all_windows_expire(base, store, tmp_path):
    models = tmp_path / "models"
    m0 = _write(store, models, base.rungs[0])
    m1 = _write(store, models, base.rungs[1], NOW + timedelta(minutes=1))
    f0, f1 = m0.parent / manifest.MODEL_NAME, m1.parent / manifest.MODEL_NAME
    id0, id1 = m0.parent.name, m1.parent.name
    rung0_bytes = f0.read_bytes()

    r = expiry.run(store, NOW + timedelta(days=5), tmp_path / "imported")
    assert r.windows_expired and store.window_ids(text_state="present")  # some windows, not all
    assert r.models_deleted == 0 and f1.is_file() and _row(store, id1)["retired_at"] is None

    gone = NOW + timedelta(days=15)
    r = expiry.run(store, gone, tmp_path / "imported")
    assert store.window_ids(text_state="present") == []
    assert r.models_deleted == 1 and r.models_pending == 0
    assert not f1.exists() and m1.is_file()  # the manifest stays
    assert _row(store, id1)["file_deleted_at"] == iso(gone) and _row(store, id1)["retired_at"] == iso(gone)
    assert f0.read_bytes() == rung0_bytes  # rung 0 holds no words: untouched
    assert _row(store, id0)["retired_at"] is None and _row(store, id0)["file_deleted_at"] is None
    assert manifest.active_per_question(store) == dict.fromkeys(QUESTIONS, id0)

    assert expiry.run(store, gone + timedelta(days=1), tmp_path / "imported").models_deleted == 0


def test_deletion_check_deletes_word_model_file(base, store, tmp_path):
    models = tmp_path / "models"
    m0 = _write(store, models, base.rungs[0])
    m1 = _write(store, models, base.rungs[1], NOW + timedelta(minutes=1))
    f0, f1 = m0.parent / manifest.MODEL_NAME, m1.parent / manifest.MODEL_NAME
    at = NOW + timedelta(hours=1)

    # a check that drops nothing, and one that drops a message from a window outside the splits
    ingest.apply_deletion_check(store, parse_deletion_check(fake.make_deletion_check(base.raw, set(), at)))
    other = fake.make_export(8, 3, 1, NOW, channels=2)
    (tmp_path / "other.json").write_bytes(other)
    ingest.import_export(store, tmp_path / "other.json", tmp_path / "imported", NOW)
    outside = parse_export(other).windows[0].messages[0].message_id
    applied = ingest.apply_deletion_check(store, parse_deletion_check(fake.make_deletion_check(other, {outside}, at)))
    assert applied.messages_dropped == 1
    assert expiry.run(store, at, tmp_path / "imported").models_deleted == 0
    assert f1.is_file()

    # a message the rung 1 model saw is deleted: its file goes at the next expiry (C5)
    w = base.ds.windows[base.sp.train[0]]
    applied = ingest.apply_deletion_check(
        store, parse_deletion_check(fake.make_deletion_check(base.raw, {w.messages[0].message_id}, at)))
    assert applied.windows == frozenset({w.window_id})
    r = expiry.run(store, at, tmp_path / "imported")
    assert r.models_deleted == 1 and not f1.exists() and m1.is_file() and f0.is_file()
    assert _row(store, m1.parent.name)["retired_at"] == iso(at)
    assert _row(store, m0.parent.name)["retired_at"] is None


def test_unreadable_word_manifest_deletes_its_file(base, store, tmp_path):
    m1 = _write(store, tmp_path / "models", base.rungs[1])
    m1.write_text("{not json", encoding="utf-8")
    r = expiry.run(store, NOW, tmp_path / "imported")
    assert r.models_deleted == 1 and not (m1.parent / manifest.MODEL_NAME).exists()


def test_word_model_file_that_cannot_be_deleted_is_retired_and_retried(base, store, tmp_path, monkeypatch):
    m1 = _write(store, tmp_path / "models", base.rungs[1])
    f1, id1 = m1.parent / manifest.MODEL_NAME, m1.parent.name
    gone = NOW + timedelta(days=15)
    monkeypatch.setattr(expiry, "remove_file", lambda path: False)
    r = expiry.run(store, gone, tmp_path / "imported")
    assert r.models_deleted == 0 and r.models_pending == 1 and f1.is_file()
    assert _row(store, id1)["retired_at"] == iso(gone) and _row(store, id1)["file_deleted_at"] is None
    assert manifest.active_per_question(store) == {}  # retired: never used while it waits

    monkeypatch.undo()
    r = expiry.run(store, gone + timedelta(hours=1), tmp_path / "imported")
    assert r.models_deleted == 1 and not f1.exists()
    assert _row(store, id1)["retired_at"] == iso(gone)  # the first retirement stands


def test_deletion_check_reaches_word_model_after_the_window_expires(base, store, tmp_path):
    models = tmp_path / "models"
    m0 = _write(store, models, base.rungs[0])
    m1 = _write(store, models, base.rungs[1], NOW + timedelta(minutes=1))
    f0, f1 = m0.parent / manifest.MODEL_NAME, m1.parent / manifest.MODEL_NAME
    at = NOW + timedelta(hours=1)
    w = next(base.ds.windows[x] for x in base.sp.train if base.ds.windows[x].messages)

    applied = ingest.apply_deletion_check(
        store, parse_deletion_check(fake.make_deletion_check(base.raw, {w.messages[0].message_id}, at)))
    assert applied.windows == frozenset({w.window_id})
    assert store.conn.execute("SELECT messages_left FROM text_drops WHERE window_id = ?",
                              (w.window_id,)).fetchone()[0] == len(w.messages) - 1
    # no expiry ran after the check (a crash, or a command stopped by a refused file), and the window's
    # text expires before the next one: the check is still remembered (C5)
    with store.transaction() as conn:
        conn.execute("UPDATE windows SET text_expires_at = ? WHERE window_id = ?", (iso(at), w.window_id))

    r = expiry.run(store, at, tmp_path / "imported")
    present = store.window_ids(text_state="present")
    assert w.window_id not in present and len(present) > 100  # the other split windows still have text
    assert r.models_deleted == 1 and not f1.exists() and m1.is_file() and f0.is_file()
    assert _row(store, m1.parent.name)["retired_at"] == iso(at)
    assert _row(store, m0.parent.name)["retired_at"] is None


def test_moved_or_copied_data_folder_uses_its_own_model_files(base, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    shutil.copyfile(base.db, data / "cb.db")
    with Store(data / "cb.db") as store:
        id0 = _write(store, data / "models", base.rungs[0]).parent.name
        id1 = _write(store, data / "models", base.rungs[1], NOW + timedelta(minutes=1)).parent.name
    moved = tmp_path / "moved"
    shutil.move(str(data), str(moved))
    files = {i: moved / "models" / i / manifest.MODEL_NAME for i in (id0, id1)}

    with Store(moved / "cb.db") as store:
        r = expiry.run(store, NOW + timedelta(minutes=2), moved / "imported")  # every window still has text
        assert (r.models_deleted, r.models_pending, r.models_missing, r.model_leftovers_deleted) == (0, 0, 0, 0)
        assert _row(store, id1)["retired_at"] is None and _row(store, id1)["file_deleted_at"] is None
        assert set(manifest.active_per_question(store).values()) <= {id0, id1}
        assert manifest.active_per_question(store)  # the moved files are found and used
    assert all(f.is_file() for f in files.values())

    # expiry on a copy deletes the copy's word file only, never the original folder's
    copy = tmp_path / "copy"
    shutil.copytree(moved, copy)
    gone = NOW + timedelta(days=15)
    with Store(copy / "cb.db") as store:
        r = expiry.run(store, gone, copy / "imported")
        assert (r.models_deleted, r.models_missing) == (1, 0)
    assert not (copy / "models" / id1 / manifest.MODEL_NAME).exists()
    assert (copy / "models" / id0 / manifest.MODEL_NAME).is_file()
    assert all(f.is_file() for f in files.values())


def test_word_model_missing_from_models_folder_is_not_counted_as_deleted(base, store, tmp_path):
    m1 = _write(store, tmp_path / "models", base.rungs[1])
    folder, id1 = m1.parent, m1.parent.name
    doc = m1.read_text(encoding="utf-8")
    shutil.rmtree(folder)
    gone = NOW + timedelta(days=15)

    r = expiry.run(store, gone, tmp_path / "imported")
    assert (r.models_deleted, r.models_pending, r.models_missing) == (0, 0, 1)
    assert _row(store, id1)["retired_at"] is None and _row(store, id1)["file_deleted_at"] is None  # left as it is
    assert expiry.run(store, gone, tmp_path / "imported").models_missing == 1  # looked for again on every run

    # its manifest is back but the file isn't: recorded as gone, but not counted as a file this run deleted
    folder.mkdir()
    (folder / manifest.MANIFEST_NAME).write_text(doc, encoding="utf-8")
    r = expiry.run(store, gone, tmp_path / "imported")
    assert (r.models_deleted, r.models_pending, r.models_missing) == (0, 0, 0)
    assert _row(store, id1)["file_deleted_at"] == iso(gone) and _row(store, id1)["retired_at"] == iso(gone)


def test_expiry_deletes_model_leftovers_of_unfinished_writes(base, store, tmp_path):
    models = tmp_path / "models"
    kept0 = _write(store, models, base.rungs[0]).parent / manifest.MODEL_NAME  # has a live row
    m1 = _write(store, models, base.rungs[1])
    with store.transaction() as conn:  # its row says the file was deleted, but the file is back (a restored copy)
        conn.execute("UPDATE manifests SET retired_at = ?, file_deleted_at = ? WHERE model_id = ?",
                     (iso(NOW), iso(NOW), m1.parent.name))
    back = m1.parent / manifest.MODEL_NAME
    writing = models / f"{manifest.WRITING_PREFIX}0123.joblib"  # a dump an interrupted write left
    orphan = models / "abcdefabcdef" / manifest.MODEL_NAME  # moved into place, but no row was recorded
    fresh = models / "fedcbafedcba" / manifest.MODEL_NAME  # no row yet: maybe a write still running
    fresh_writing = models / f"{manifest.WRITING_PREFIX}4567.joblib"
    other = models / "notes" / manifest.MODEL_NAME  # not a folder write makes
    for f in (writing, orphan, fresh, fresh_writing, other):
        f.parent.mkdir(exist_ok=True)
        f.write_bytes(b"made-up words")
    old = (NOW - expiry.LEFTOVER_MIN_AGE).timestamp()
    recent = old + 1
    for f, t in ((kept0, old), (back, old), (writing, old), (orphan, old), (other, old),
                 (fresh, recent), (fresh_writing, recent)):
        os.utime(f, (t, t))

    r = expiry.run(store, NOW, tmp_path / "imported")
    assert r.model_leftovers_deleted == 3 and r.models_deleted == 0
    assert not writing.exists() and not orphan.exists() and not back.exists()
    assert kept0.is_file() and fresh.is_file() and fresh_writing.is_file() and other.is_file()
    assert m1.is_file() and (kept0.parent / manifest.MANIFEST_NAME).is_file()  # manifests stay
    # one second later both are old enough
    assert expiry.run(store, NOW + timedelta(seconds=1), tmp_path / "imported").model_leftovers_deleted == 2
    assert not fresh.exists() and not fresh_writing.exists() and other.is_file() and kept0.is_file()


# --- the train command ---------------------------------------------------------------------------------


def _cli(capsys, data: Path, *args: str) -> tuple[int, str]:
    code = cli.main(["--data-dir", str(data), *args])
    out = capsys.readouterr()
    return code, out.out + out.err


def _cli_world(tmp_path, capsys):
    data = tmp_path / "data"
    now = datetime.now(timezone.utc).replace(microsecond=0)
    raw = fake.make_export(5, 40, 10, now, channels=6)
    (data / "inbox").mkdir(parents=True)
    (data / "inbox" / "e.json").write_bytes(raw)
    assert _cli(capsys, data, "import")[0] == 0
    assert _cli(capsys, data, "assign", "--seed", "5")[0] == 0
    with Store(data / "cb.db") as store:
        _labels(store, fake.truth(raw))
    return data, raw, now


def _counts(data: Path) -> tuple[int, int]:
    with Store(data / "cb.db") as store:
        return (store.conn.execute("SELECT COUNT(*) FROM manifests").fetchone()[0],
                store.conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])


def _model_files(data: Path) -> dict[int, Path]:
    """Each rung's unretired model file, found in the data folder's ``models/``."""
    with Store(data / "cb.db") as store:
        rows = store.conn.execute(
            "SELECT rung, model_id FROM manifests WHERE retired_at IS NULL ORDER BY created_at").fetchall()
    return {rung: data / "models" / model_id / manifest.MODEL_NAME for rung, model_id in rows}


def _assert_no_ids_or_text(out: str, raw: bytes) -> None:
    for w in parse_export(raw).windows:
        assert w.window_id not in out
        assert fake.marker(w.window_id) not in out


def test_train_command_refuses_stale_deletion_check(tmp_path, capsys):
    data, raw, now = _cli_world(tmp_path, capsys)
    with Store(data / "cb.db") as store, store.transaction() as conn:
        conn.execute("UPDATE checked_windows SET checked_at = ?", (iso(now - timedelta(days=8)),))

    code, out = _cli(capsys, data, "train", "--rung", "0", "--origin", "synthetic", "--seed", "7")
    assert code == 1 and "deletion check: stale" in out
    assert _counts(data) == (0, 0)
    assert not any((data / "models").iterdir())

    # a fresh check waiting in the inbox: train applies it first, then the gate passes (C36)
    (data / "inbox" / "check.json").write_bytes(fake.make_deletion_check(raw, set(), now))
    code, out = _cli(capsys, data, "train", "--rung", "0", "--origin", "synthetic", "--seed", "7")
    assert code == 0, out
    assert not (data / "inbox" / "check.json").exists()
    n_manifests, n_predictions = _counts(data)
    assert n_manifests == 1 and n_predictions == len(parse_export(raw).windows) * len(QUESTIONS)
    assert "self-consistency" in out and "predictions:" in out
    _assert_no_ids_or_text(out, raw)


def test_train_command_refusals(tmp_path, capsys):
    data, raw, now = _cli_world(tmp_path, capsys)
    code, out = _cli(capsys, data, "train", "--rung", "0", "--origin", "aron", "--seed", "7")
    assert code == 1 and "no labels" in out
    with Store(data / "cb.db") as store:
        _labels(store, fake.truth(raw), origin="aron")
    code, out = _cli(capsys, data, "train", "--rung", "0", "--seed", "7")
    assert code == 1 and "choose one origin" in out
    assert _counts(data) == (0, 0)
    with pytest.raises(SystemExit):
        cli.main(["--data-dir", str(data), "train", "--rung", "2", "--seed", "7"])
    with pytest.raises(SystemExit):
        cli.main(["--data-dir", str(data), "train", "--rung", "0", "--seed", "x"])


def test_deletion_check_after_train_deletes_the_rung1_file(tmp_path, capsys):
    data, raw, now = _cli_world(tmp_path, capsys)
    for rung in ("0", "1"):
        code, out = _cli(capsys, data, "train", "--rung", rung, "--origin", "synthetic", "--seed", "7")
        assert code == 0, out
    files = _model_files(data)
    f0, f1 = files[0], files[1]
    assert f0.is_file() and f1.is_file()

    dropped = parse_export(raw).windows[0].messages[0].message_id
    (data / "inbox" / "check.json").write_bytes(fake.make_deletion_check(raw, {dropped}, now))
    code, out = _cli(capsys, data, "deletions")
    assert code == 0, out
    assert "models: 1 word model file(s) deleted" in out
    assert not f1.exists() and (f1.parent / manifest.MANIFEST_NAME).is_file() and f0.is_file()
    _assert_no_ids_or_text(out, raw)


def test_refused_check_file_still_reaches_word_models(tmp_path, capsys):
    data, raw, now = _cli_world(tmp_path, capsys)
    for rung in ("0", "1"):
        code, out = _cli(capsys, data, "train", "--rung", rung, "--origin", "synthetic", "--seed", "7")
        assert code == 0, out
    files = _model_files(data)
    f0, f1 = files[0], files[1]
    seen = manifest.read(f1.parent)["text_message_counts"]
    w = next(x for x in parse_export(raw).windows if x.window_id in seen and x.messages)
    inbox = data / "inbox"
    (inbox / "a.json").write_bytes(fake.make_deletion_check(raw, {w.messages[0].message_id}, now))
    (inbox / "z.json").write_bytes(fake.make_deletion_check(raw, set(), now + timedelta(days=1)))  # refused

    # the good check is applied, the refused one stops train: expiry still runs after the good one (C5)
    code, out = _cli(capsys, data, "train", "--rung", "0", "--origin", "synthetic", "--seed", "7")
    assert code == 1 and "refused, left in place" in out
    assert "models: 1 word model file(s) deleted" in out
    assert not f1.exists() and f0.is_file()
    assert not (inbox / "a.json").exists() and (inbox / "z.json").exists()
    _assert_no_ids_or_text(out, raw)

    # a rung 1 model trained after that check never saw the dropped message: the drop leaves its file alone
    (inbox / "z.json").unlink()
    code, out = _cli(capsys, data, "train", "--rung", "1", "--origin", "synthetic", "--seed", "7")
    assert code == 0, out
    again = _model_files(data)[1]
    code, out = _cli(capsys, data, "expire")
    assert code == 0 and "models:" not in out and again.is_file()
