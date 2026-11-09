import json
from collections import Counter
from datetime import timedelta
from types import SimpleNamespace

import joblib
import pytest

from chatbrain import fake, ingest, manifest, paths, train
from chatbrain.export_format import iso
from chatbrain.page import shown as shown_mod
from chatbrain.store import Store
from pagekit import (
    ARON0_ID, NOW, QUESTIONS, RUNG0_ID, RUNG1_ID, SCHEME, build_world, client, counts, label, predict, report,
    stub_model,
)  # fmt: skip


@pytest.fixture(autouse=True)
def _no_real_data_folder(monkeypatch):
    monkeypatch.delenv(paths.ENV_VAR, raising=False)


@pytest.fixture
def world(tmp_path):
    w = build_world(tmp_path)
    yield w
    w.store.close()


# rung 1 beats rung 0 on doing and aimed_at only (its lower bound .9 > rung 0's .8); rung 0 keeps feeling and tone
RUNG1_REPORT = report(0.95, 0.9, ["doing", "aimed_at"]) | {
    q: {"agreement": None, "agreement_ci": [0.0, 1.0], "answered": 0} for q in ("feeling", "tone")
}


def _two_rungs(world):
    stub_model(world.store, world.models, RUNG0_ID, 0, thresholds={"doing": 0.5})
    stub_model(world.store, world.models, RUNG1_ID, 1, thresholds={"doing": 0.6, "aimed_at": 0.6}, rep=RUNG1_REPORT,
               at=1)  # fmt: skip
    assert manifest.active_per_question(world.store, models_dir=world.models) == {
        "doing": RUNG1_ID, "aimed_at": RUNG1_ID, "feeling": RUNG0_ID, "tone": RUNG0_ID,
    }  # fmt: skip


def test_shown_order(world):
    _two_rungs(world)
    a, b, c = world.new_moments[:3]
    s = world.store

    # 1. the owner's latest label naming the question beats any prediction
    predict(s, a, RUNG1_ID, "doing", ["laughing"], 0.95)
    label(s, a, {"doing": ["cheering"]}, at=1)
    label(s, a, {"doing": ["sad"]}, at=2)  # the latest label naming doing wins
    label(s, a, {"tone": "playful"}, at=3)  # names tone only: doing stays sad (C1)
    label(s, a, {"doing": ["shocked"]}, origin="synthetic", provenance="synthetic", at=4)  # never "your label"
    label(s, a, {"doing": ["tense"]}, version="cb_labels_old", at=5)  # another labels_version is skipped
    label(s, a, not_sure={"feeling": ["conflicting", "unknown_in_joke"]}, at=6)

    # 2. the active model's prediction counts only when it clears that model's own threshold
    predict(s, a, RUNG1_ID, "aimed_at", "guest", 0.55)  # under rung 1's .60
    predict(s, b, RUNG1_ID, "aimed_at", "streamer", 0.61)
    predict(s, b, RUNG1_ID, "aimed_at", "guest", 0.75, at=1)  # the latest prediction row wins
    predict(s, b, RUNG0_ID, "feeling", "critical", 0.5)  # exactly the threshold clears it

    # 3. rung 1 has no row for c: rung 0's prediction with rung 0's threshold
    predict(s, c, RUNG0_ID, "doing", ["shocked", "laughing"], 0.52)  # .52 clears rung 0's .50, not rung 1's .60
    predict(s, c, RUNG0_ID, "aimed_at", "guest", 0.4)  # under rung 0's .50: not sure

    got = shown_mod.shown(s, SCHEME, world.models)
    assert set(got) == set(s.window_ids())
    assert all(set(per) == set(QUESTIONS) for per in got.values())

    d = got[a]["doing"]
    assert (d.kind, d.values, d.source, d.confidence) == ("label", ("sad",), "your label", None)
    assert got[a]["tone"].values == ("playful",) and got[a]["tone"].source == "your label"
    f = got[a]["feeling"]
    assert f.kind == "label" and f.values == () and f.reasons == ("conflicting", "unknown_in_joke")
    assert f.not_sure == ("conflicting", "unknown in-joke")

    under = got[a]["aimed_at"]
    assert under.kind == "model" and under.values == () and under.model_id == RUNG1_ID
    assert under.not_sure == ("under threshold .55 / .60",)
    assert (under.confidence, under.threshold) == (0.55, 0.6)

    assert got[b]["aimed_at"].values == ("guest",) and got[b]["aimed_at"].confidence == 0.75
    assert got[b]["feeling"].values == ("critical",) and got[b]["feeling"].source == "made-up labels"
    assert got[b]["doing"].values == () and got[b]["doing"].not_sure == ("no prediction yet",)

    fb = got[c]["doing"]
    assert fb.values == ("laughing", "shocked")  # scheme order
    assert (fb.model_id, fb.rung, fb.threshold, fb.confidence) == (RUNG0_ID, 0, 0.5, 0.52)
    assert got[c]["aimed_at"].values == () and got[c]["aimed_at"].not_sure == ("under threshold .40 / .50",)

    # window_ids narrows the read
    assert set(shown_mod.shown(s, SCHEME, world.models, [a, c])) == {a, c}


def test_shown_without_models_is_not_sure(world):
    a = world.new_moments[0]
    got = shown_mod.shown(world.store, SCHEME, world.models, [a])[a]
    assert all(v.values == () and v.kind == "none" and v.not_sure == ("no model yet",) for v in got.values())
    assert shown_mod.plan(world.store, SCHEME, world.models).made_up is False


def test_shown_quiet_window_and_review_list(world):
    stub_model(world.store, world.models, RUNG0_ID, 0)
    quiet = next(w for w in world.store.window_ids() if counts(world.store, w)["n_messages"] <= 1)
    loud = world.new_moments[0]
    emote = sorted(counts(world.store, loud)["emote_counts"])[0]
    for w in (quiet, loud):
        predict(world.store, w, RUNG0_ID, "doing", ["laughing"], 0.9)
        predict(world.store, w, RUNG0_ID, "aimed_at", "chat_itself", 0.9)
    label(world.store, loud, {"aimed_at": "guest"})

    got = shown_mod.shown(world.store, SCHEME, world.models, review_ids=frozenset({emote}))
    assert got[quiet]["doing"].values == () and got[quiet]["doing"].not_sure == ("no chat in this window",)
    assert got[loud]["doing"].values == () and got[loud]["doing"].not_sure == ("needs review",)
    assert got[loud]["doing"].reasons == ("needs_review",)
    assert got[loud]["aimed_at"].values == ("guest",)  # the owner's own label stands

    plain = shown_mod.shown(world.store, SCHEME, world.models)
    assert plain[loud]["doing"].values == ("laughing",)


def test_cte_is_reusable_sql(world):
    stub_model(world.store, world.models, RUNG0_ID, 0)
    a = world.new_moments[0]
    predict(world.store, a, RUNG0_ID, "doing", ["cheering"], 0.8)
    p = shown_mod.plan(world.store, SCHEME, world.models)
    sql, params = shown_mod.cte(p, scheme_version=SCHEME.version, review_ids=frozenset())
    assert sql.lstrip().upper().startswith("WITH")
    rows = world.store.conn.execute(
        sql + " SELECT window_id, question, value_json, kind FROM shown WHERE window_id = ? ORDER BY question",
        [*params, a],
    ).fetchall()
    assert {r[1] for r in rows} == set(QUESTIONS)
    assert ("doing", json.dumps(["cheering"]), "model") in {(r[1], r[2], r[3]) for r in rows}


def test_made_up_labels_notice(world):
    stub_model(world.store, world.models, RUNG0_ID, 0)
    a = world.new_moments[0]
    predict(world.store, a, RUNG0_ID, "doing", ["laughing"], 0.9)
    assert shown_mod.plan(world.store, SCHEME, world.models).made_up is True
    page = client(world).get("/").text
    # the glance has no recap heading (Task 3): the notice is the top bar's badge alone
    assert '<span class="bd" data-state="review">made-up labels</span>' in page.split("<header", 1)[1].split("</header>")[0]
    assert page.count("made-up labels") == 1
    assert "made-up labels" in client(world).get(f"/m/{a}").text

    # an owner-origin model takes over: no notice, and the source names its rung
    stub_model(world.store, world.models, ARON0_ID, 0, origin="aron", at=1)
    predict(world.store, a, ARON0_ID, "doing", ["cheering"], 0.9)
    assert shown_mod.plan(world.store, SCHEME, world.models).made_up is False
    got = shown_mod.shown(world.store, SCHEME, world.models, [a])[a]["doing"]
    assert got.values == ("cheering",) and got.source == "model rung 0"
    assert "made-up labels" not in client(world).get("/").text


# --- rung-0 predictions saved at page start (C19, B31) ----------------------------------------------


@pytest.fixture(scope="module")
def rung0(tmp_path_factory):
    """A real rung-0 model trained on another made-up world."""
    tmp = tmp_path_factory.mktemp("train")
    raw = fake.make_export(11, 160, 40, NOW)
    (tmp / "e.json").write_bytes(raw)
    with Store(tmp / "cb.db") as store:
        ingest.import_export(store, tmp / "e.json", tmp / "imported", NOW)
        rows = [
            (w, "blind", "synthetic", "synthetic", SCHEME.version, SCHEME.guide_version, json.dumps(a), iso(NOW))
            for w, a in sorted(fake.truth(raw).items())
        ]
        with store.transaction() as conn:
            conn.executemany(
                "INSERT INTO labels (window_id, project, origin, provenance, labels_version, guide_version,"
                " answers_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        ds = train.load_dataset(store, SCHEME, "synthetic")
    sp = train.split(ds, 7)
    m = train.fit(0, ds, sp, 7)
    th = train.calibrate(m, ds, sp)
    return SimpleNamespace(model=m, thresholds=th, report={q: r.to_dict() for q, r in train.report(m, ds, sp, th).items()})


def _real_rung0(world, rung0, model_id=RUNG0_ID):
    folder = stub_model(world.store, world.models, model_id, 0, rep=rung0.report)
    joblib.dump(rung0.model, folder / manifest.MODEL_NAME)
    doc = json.loads((folder / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))
    doc["thresholds"] = rung0.thresholds
    (folder / manifest.MANIFEST_NAME).write_text(json.dumps(doc), encoding="utf-8")


def test_page_start_saves_missing_rung0_predictions(world, rung0):
    _real_rung0(world, rung0)
    a = world.new_moments[0]
    predict(world.store, a, RUNG0_ID, "doing", ["sad"], 0.99)  # already has rows: left alone
    with_features = {r[0] for r in world.store.conn.execute("SELECT window_id FROM features")}

    client(world)  # page start

    rows = world.store.conn.execute(
        "SELECT window_id, question, value_json, created_at FROM predictions WHERE model_id = ?", (RUNG0_ID,)
    ).fetchall()
    per_window = Counter(r[0] for r in rows)
    assert set(per_window) == with_features
    assert per_window[a] == 1  # never re-predicted
    assert all(n == len(QUESTIONS) for w, n in per_window.items() if w != a)
    assert {r[3] for r in rows if r[0] != a} == {iso(NOW)}

    before = len(rows)
    assert shown_mod.backfill_rung0(world.store, SCHEME, world.models, NOW + timedelta(hours=1)) == 0
    assert world.store.conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == before


def test_backfill_skips_a_model_file_that_will_not_load(world, capsys):
    stub_model(world.store, world.models, RUNG0_ID, 0)  # b"stub" is no joblib file
    assert shown_mod.backfill_rung0(world.store, SCHEME, world.models, NOW) == 0
    assert world.store.conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 0
    assert "rung-0" in capsys.readouterr().err
