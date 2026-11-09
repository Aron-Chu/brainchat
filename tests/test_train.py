import json
import pickle
import re
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from chatbrain import assign, fake, features, ingest, metrics, scheme as scheme_mod, train
from chatbrain.export_format import Emote, Message, Window, iso
from chatbrain.spread import Counts, compute
from chatbrain.store import Store, WindowRow

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)
SCHEME = scheme_mod.load()
QUESTIONS = [q.key for q in SCHEME.questions]
PROVENANCE = {"blind": "blind", "suggested": "accepted", "repeat": "repeat", "page": "corrected-on-page"}
FULL = {"doing": ["laughing"], "aimed_at": "streamer", "feeling": "supportive", "tone": "playful"}


# --- the made-up world: fake.make_export(7, 480, 120, NOW), synthetic labels = fake.truth ---------


def _synthetic_labels(store: Store, truth: dict[str, dict]) -> None:
    """Synthetic labels use project = the window's exposure (ruling C1)."""
    with store.transaction() as conn:
        exposure = dict(conn.execute("SELECT window_id, exposure FROM assignments").fetchall())
        conn.executemany(
            "INSERT INTO labels (window_id, project, origin, provenance, labels_version, guide_version,"
            " answers_json, not_sure_json, created_at) VALUES (?, ?, 'synthetic', 'synthetic', ?, ?, ?, '{}', ?)",
            [(wid, exposure[wid], SCHEME.version, SCHEME.guide_version, json.dumps(ans), iso(NOW))
             for wid, ans in sorted(truth.items())],
        )


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("world")
    raw = fake.make_export(7, 480, 120, NOW)
    (tmp / "inbox").mkdir()
    (tmp / "inbox" / "e.json").write_bytes(raw)
    store = Store(tmp / "cb.db")
    ingest.import_export(store, tmp / "inbox" / "e.json", tmp / "imported", NOW)
    assign.assign_new(store, "7", NOW)
    _synthetic_labels(store, fake.truth(raw))
    ds = train.load_dataset(store, SCHEME, "synthetic")
    sp = train.split(ds, seed=7)
    m0 = train.fit(0, ds, sp, seed=7)
    m1 = train.fit(1, ds, sp, seed=7)
    yield SimpleNamespace(store=store, raw=raw, truth=fake.truth(raw), ds=ds, sp=sp, m0=m0, m1=m1)
    store.close()


# --- a small hand-made store -------------------------------------------------------------------


def _window(wid: str, channel: str = "fakechan01", *, kind: str = "moment", lines=(("s1", "KEKW"),)) -> Window:
    start = NOW - timedelta(days=1)
    msgs = tuple(
        Message(f"{wid}-m{i}", 30_000 + i, spk, text,
                tuple(Emote(f"fake-{t.lower()}", t) for t in text.split() if t in fake.EMOTE_NAMES))
        for i, (spk, text) in enumerate(lines, start=1)
    )
    return Window(wid, channel, f"{channel}-s1", kind, start + timedelta(seconds=30), start,  # type: ignore[arg-type]
                  start + timedelta(seconds=120), "en", NOW + timedelta(days=10), msgs)


@pytest.fixture
def small(tmp_path):
    with Store(tmp_path / "cb.db") as s:
        with s.transaction() as conn:
            for i in range(1, 7):
                w = _window(f"w{i}", f"fakechan0{i}")
                assert s.insert_window(w, "vod", NOW)
                conn.execute("INSERT INTO features (window_id, json) VALUES (?, ?)",
                             (w.window_id, compute(w).to_json()))
        yield s


def _label(store, wid, project="blind", answers=None, not_sure=None, *, origin="aron", at=0):
    with store.transaction() as conn:
        conn.execute(
            "INSERT INTO labels (window_id, project, origin, provenance, labels_version, guide_version,"
            " answers_json, not_sure_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (wid, project, origin, "synthetic" if origin == "synthetic" else PROVENANCE[project], SCHEME.version,
             SCHEME.guide_version, json.dumps(FULL if answers is None else answers), json.dumps(not_sure or {}),
             iso(NOW + timedelta(hours=at))),
        )


# --- features ----------------------------------------------------------------------------------


def test_counts_row_keys():
    w = _window("w1", lines=(("s1", "KEKW KEKW"), ("s2", "@s1 what?"), ("s3", "LUL")))
    row = features.counts_row(compute(w))
    base = {"n_messages", "distinct_posters", "msg_per_min", "top_emote_share", "chant_share", "mention_share",
            "question_share"}
    assert set(row) == base | {"emote:fake-kekw", "emote:fake-lul"}
    assert row["emote:fake-kekw"] == 2.0 and row["n_messages"] == 3.0 and row["question_share"] == pytest.approx(1 / 3)
    assert all(isinstance(v, float) for v in row.values())


def test_text_doc_emote_modes():
    w = _window("w1", lines=(("s1", "KEKW  KEKW"), ("s2", "@s1 LUL that was so bad"), ("s3", "hi @user kekw")))
    assert features.text_doc(w) == "KEKW KEKW\n@someone LUL that was so bad\nhi @user kekw"
    assert features.text_doc(w, "id") == (
        "emote_fake-kekw emote_fake-kekw\n@someone emote_fake-lul that was so bad\nhi @user kekw"
    )
    with pytest.raises(ValueError):
        features.text_doc(w, "words")  # type: ignore[arg-type]
    empty = _window("w0", lines=())
    assert features.text_doc(empty) == "" and features.text_doc(empty, "id") == ""


# --- load_dataset ------------------------------------------------------------------------------


def test_not_sure_excluded_from_targets(small):
    # w1: blind label, tone not sure; a later page fix answers tone and names nothing else
    _label(small, "w1", "blind", {k: v for k, v in FULL.items() if k != "tone"}, {"tone": ["unclear"]}, at=0)
    _label(small, "w1", "page", {"tone": "serious"}, at=5)
    # w2: suggested label, then a later not-sure on doing replaces the doing answer
    _label(small, "w2", "suggested", at=0)
    _label(small, "w2", "page", {}, {"doing": ["conflicting"]}, at=5)
    # w3: repeat label only; w4: a label that names no question
    _label(small, "w3", "repeat", {**FULL, "doing": ["laughing", "cheering"]}, at=0)
    _label(small, "w4", "suggested", {}, at=0)

    ds = train.load_dataset(small, SCHEME)
    assert ds.origin == "aron" and set(ds.rows) == {"w1", "w2", "w3"}
    # train targets: the latest owner label per question from any project, page fixes included
    assert ds.train_targets["w1"] == {"doing": ("laughing",), "aimed_at": "streamer", "feeling": "supportive",
                                      "tone": "serious"}
    assert "doing" not in ds.train_targets["w2"] and ds.not_sure["w2"] == {"doing"}
    assert ds.not_sure["w1"] == set()
    assert ds.train_targets["w3"]["doing"] == ("laughing", "cheering")
    # grade targets (C6): blind and repeat projects only, so the page fix is not a grading target
    assert set(ds.graded) == {"w1", "w3"}
    assert "tone" not in ds.grade_targets["w1"] and ds.grade_not_sure["w1"] == {"tone"}
    assert "w2" not in ds.grade_targets
    assert set(ds.windows) == {"w1", "w2", "w3"} and ds.counts["w1"].n_messages == 1


def test_windows_hold_only_present_text(small):
    for wid in ("w1", "w2"):
        _label(small, wid)
    with small.transaction() as conn:
        conn.execute("DELETE FROM messages WHERE window_id = 'w2'")
        conn.execute("UPDATE windows SET text_state = 'expired' WHERE window_id = 'w2'")
    assign.pick_rare(small, ["w1"])
    ds = train.load_dataset(small, SCHEME, "aron")
    assert set(ds.rows) == {"w1", "w2"} and set(ds.windows) == {"w1"}
    assert set(ds.counts) == {"w1", "w2"}  # counts outlive the text
    assert ds.rows["w1"].picked_rare and not ds.rows["w2"].picked_rare


def test_mixed_origins_refused(small):
    _label(small, "w1", origin="aron")
    _label(small, "w2", "suggested", origin="synthetic")
    with pytest.raises(train.MixedOrigins):
        train.load_dataset(small, SCHEME)
    aron = train.load_dataset(small, SCHEME, "aron")
    synth = train.load_dataset(small, SCHEME, "synthetic")
    assert set(aron.rows) == {"w1"} and set(synth.rows) == {"w2"}
    assert synth.graded == {"w2"} and synth.grade_targets == synth.train_targets  # synthetic: every window counts
    with pytest.raises(ValueError):
        train.load_dataset(small, SCHEME, "teacher")  # type: ignore[arg-type]


def test_one_origin_is_used_when_none_is_chosen(small):
    _label(small, "w1", "suggested", origin="synthetic")
    assert train.load_dataset(small, SCHEME).origin == "synthetic"


def test_cause_only_aron_row_does_not_mix_origins(small):
    """A page cause note names no question (C1), so it is not one of the owner's labels for training."""
    _label(small, "w1", "suggested", origin="synthetic")
    _label(small, "w2", "page", {}, {})  # a cause-only row: no answers, no not-sure reasons
    ds = train.load_dataset(small, SCHEME)
    assert ds.origin == "synthetic" and set(ds.rows) == {"w1"}
    _label(small, "w3", "page", {"tone": "playful"})  # one answered question is an owner label
    with pytest.raises(train.MixedOrigins):
        train.load_dataset(small, SCHEME)


# --- split -------------------------------------------------------------------------------------


def test_split_never_shares_a_channel(world):
    ds, sp = world.ds, world.sp
    parts = (sp.train, sp.val, sp.test)
    channels = [{ds.rows[w].channel for w in p} for p in parts]
    streams = [{ds.rows[w].stream_id for w in p} for p in parts]
    for a in range(3):
        for b in range(a + 1, 3):
            assert not channels[a] & channels[b] and not streams[a] & streams[b]
    for p in parts:
        assert {ds.rows[w].anchor_kind for w in p} == {"moment", "ordinary"}  # Moment + ordinary everywhere
    assert sorted(sp.train + sp.val + sp.test) == sorted(ds.rows)  # nothing picked: every window placed once
    assert len(sp.test) >= min(200, 0.2 * len(ds.rows))
    assert len(sp.val) >= 0.15 * (len(sp.train) + len(sp.val))
    assert train.split(ds, seed=7) == sp  # same seed, same split
    assert train.split(ds, seed=8) != sp


def test_picked_rare_not_in_test(world):
    ds = world.ds
    first_per_channel: dict[str, str] = {}
    for wid in sorted(ds.rows):
        first_per_channel.setdefault(ds.rows[wid].channel, wid)
    picked = set(first_per_channel.values())
    ds2 = replace(ds, rows={w: replace(r, picked_rare=w in picked) for w, r in ds.rows.items()})
    sp = train.split(ds2, seed=7)
    assert not picked & set(sp.test)
    test_channels = {ds.rows[w].channel for w in sp.test}
    placed = set(sp.train) | set(sp.val) | set(sp.test)
    for wid in picked:
        # on a test channel it leaves every split (C21); elsewhere it stays with its channel
        assert (wid in placed) == (ds.rows[wid].channel not in test_channels)


def test_split_needs_three_channels(small):
    for wid in ("w1", "w2"):
        _label(small, wid)
    with pytest.raises(ValueError):
        train.split(train.load_dataset(small, SCHEME), seed=1)


# --- fit, calibrate, report ----------------------------------------------------------------------


def test_near_empty_windows_train_fine(world):
    ds, sp = world.ds, world.sp
    zero = [w for w in sp.train if ds.counts[w].n_messages == 0]
    one = [w for w in sp.train if ds.counts[w].n_messages == 1]
    assert zero and one  # Review Focus 4: 0- and 1-message windows are in the training set
    c0, c1 = ds.counts[zero[0]], ds.counts[one[0]]
    assert c0 == Counts(0, 0, 0.0, None, 0.0, 0.0, 0.0, 0.0, {}, "few_posters", 0, 0)
    assert (c1.n_messages, c1.distinct_posters, c1.msg_per_min, c1.spread) == (1, 1, 0.5, "few_posters")  # C32
    for model in (world.m0, world.m1):
        for wid in (zero[0], one[0]):
            answers = train.predict(model, [(ds.counts[wid], ds.windows[wid])])[0]
            assert set(answers) == set(QUESTIONS) and all(a is not None for a in answers.values())
        assert train.evidence(model, ds.windows[zero[0]], "doing", "chatting_as_usual") == []


def test_rung1_learns_made_up_truth(world):
    ds, sp, m1 = world.ds, world.sp, world.m1
    thresholds = train.calibrate(m1, ds, sp)
    assert set(thresholds) == set(QUESTIONS)
    rep = train.report(m1, ds, sp, thresholds)
    assert rep["doing"].agreement >= 0.6
    assert rep["doing"].n_graded == len([w for w in sp.test if w in ds.windows])
    for q in QUESTIONS:
        t = thresholds[q]
        assert t is None or (0.30 <= t <= 0.95 and round(t / 0.05, 6) == round(t / 0.05))
    rep0 = train.report(world.m0, ds, sp, train.calibrate(world.m0, ds, sp))
    assert rep0["doing"].n_graded == len(sp.test)  # rung 0 needs counts only


def test_fit_records_config_and_is_deterministic(world):
    cfg = world.m1.config
    assert cfg["rung"] == 1 and cfg["emote_mode"] == "name" and cfg["origin"] == "synthetic"
    assert cfg["labels_version"] == SCHEME.version and set(cfg["heads"]) == set(QUESTIONS)
    json.dumps(cfg)  # the manifest stores it
    again = train.fit(1, world.ds, world.sp, seed=7)
    assert pickle.dumps(again) == pickle.dumps(world.m1)  # identical retrains, identical bytes (C16)


def test_rung1_model_holds_no_marker_or_message_text(world):
    blob = pickle.dumps(world.m1)
    markers = {fake.marker(w) for w in world.ds.rows}
    assert not [m for m in markers if m.encode() in blob]  # min_df=2 drops one-off tokens, and none is kept


def test_one_class_head_predicts_constant(world):
    ds, sp = world.ds, world.sp
    one_tone = {w: {**t, "tone": "serious"} for w, t in ds.train_targets.items()}
    ds2 = replace(ds, train_targets=one_tone, grade_targets=one_tone)
    for rung in (0, 1):
        m = train.fit(rung, ds2, sp, seed=7)
        assert m.config["heads"]["tone"]["kind"] == "constant" and m.config["heads"]["tone"]["value"] == "serious"
        wid = sp.test[0]
        answer = train.predict(m, [(ds.counts[wid], ds.windows[wid])])[0]["tone"]
        assert answer == train.Answer("serious", 1.0)
        assert train.evidence(m, ds.windows[wid], "tone", "serious") == []


class _Stub:
    """A head whose answer depends only on the window's message count (rung 0 input)."""

    def __init__(self, by_n: dict[int, train.Answer]):
        self.by_n = by_n

    def predict(self, rows):
        return [self.by_n.get(int(r["n_messages"])) for r in rows]


def _row(wid: str, channel: str) -> WindowRow:
    t = NOW - timedelta(days=1)
    return WindowRow(wid, channel, f"{channel}-s1", "moment", t, t, t + timedelta(seconds=120), "en",
                     NOW + timedelta(days=10), "vod", NOW, "present", False)


def _counts(n: int) -> Counts:
    return Counts(n, 1, n / 2, None, 0.0, 0.0, 0.0, 0.0, {}, "few_posters", 0, 0)


def _stub_world(answers: dict[int, train.Answer], targets: dict[int, str], *, graded=True):
    ids = {n: f"v{n:02d}" for n in answers}
    ds = train.Dataset(
        rows={w: _row(w, "fakechan01") for w in ids.values()},
        counts={w: _counts(n) for n, w in ids.items()},
        windows={},
        # train targets always match the stub's answers, so a threshold read from them would be 0.30
        train_targets={ids[n]: {"aimed_at": answers[n].value} for n in targets},
        not_sure={w: set() for w in ids.values()},
        graded=set(ids.values()) if graded else set(),
        grade_targets={ids[n]: {"aimed_at": v} for n, v in targets.items()} if graded else {},
        grade_not_sure={w: set() for w in ids.values()} if graded else {},
        origin="aron",
        scheme=SCHEME,
    )
    heads = {q: _Stub({}) for q in QUESTIONS}
    heads["aimed_at"] = _Stub(answers)
    model = train.Model(rung=0, emote_mode="name", heads=heads, config={})
    return ds, train.Split(train=[], val=sorted(ids.values()), test=sorted(ids.values())), model


def test_threshold_none_when_80_never_reached():
    A = train.Answer
    answers = {1: A("streamer", 0.9), 2: A("streamer", 0.9), 3: A("streamer", 0.9), 4: A("guest", 0.9),
               5: A("guest", 0.9), 6: A("guest", 0.5), 7: A("guest", 0.5), 8: A("guest", 0.5), 9: A("guest", 0.5),
               10: A("guest", 0.5)}
    # graded against blind labels (all streamer), never against the train targets (C6)
    ds, sp, model = _stub_world(answers, {n: "streamer" for n in answers})
    # answered agreement: 3/10 at 0.30-0.50, 3/5 at 0.55-0.90, nothing answered at 0.95
    assert train.calibrate(model, ds, sp) == {q: None for q in QUESTIONS}
    assert train.calibrate(model, replace(ds, grade_targets=ds.train_targets), sp)["aimed_at"] == 0.30


def test_threshold_is_the_smallest_reaching_80():
    A = train.Answer
    answers = {n: A("guest", 0.6) for n in (1, 2, 3)} | {n: A("streamer", 0.7) for n in (4, 5, 6, 7)} | \
              {n: A("streamer", 0.8) for n in (8, 9, 10, 11)}
    ds, sp, model = _stub_world(answers, {n: "streamer" for n in answers})
    # 8/11 = 0.727 up to 0.60; 8/8 from 0.65
    assert train.calibrate(model, ds, sp)["aimed_at"] == pytest.approx(0.65)
    ungraded, sp2, model2 = _stub_world(answers, {n: "streamer" for n in answers}, graded=False)
    assert train.calibrate(model2, ungraded, sp2)["aimed_at"] is None  # no graded windows, no threshold (C6)


def test_report_uses_graded_test_windows_and_threshold():
    A = train.Answer
    answers = {1: A("streamer", 0.9), 2: A("guest", 0.9), 3: A("streamer", 0.4), 4: A("streamer", 0.9)}
    ds, sp, model = _stub_world(answers, {1: "streamer", 2: "streamer", 3: "streamer"})
    ds = replace(ds, grade_not_sure={**ds.grade_not_sure, "v04": {"aimed_at"}})
    rep = train.report(model, ds, sp, {"aimed_at": 0.5, "doing": None, "feeling": None, "tone": None})["aimed_at"]
    assert (rep.n_graded, rep.answered) == (3, 2) and rep.agreement == pytest.approx(0.5)
    assert rep.abstain_rate == pytest.approx(1 / 3)
    assert rep.aron_not_sure_also_abstained == (0, 1)  # v04: the owner not sure, the model answered
    none = train.report(model, ds, sp, {q: None for q in QUESTIONS})["aimed_at"]
    assert none.answered == 0 and none.aron_not_sure_also_abstained == (1, 1)


# --- evidence ----------------------------------------------------------------------------------


def _busy_test_moment(world) -> str:
    ds = world.ds
    return max((w for w in world.sp.test if ds.rows[w].anchor_kind == "moment"), key=lambda w: ds.counts[w].n_messages)


def test_evidence_at_most_five_ids(world):
    ds, m1, m0 = world.ds, world.m1, world.m0
    wid = _busy_test_moment(world)
    w = ds.windows[wid]
    ids = {m.message_id for m in w.messages}
    marker_ids = {m.message_id for m in w.messages if m.text == fake.marker(wid)}
    answers = train.predict(m1, [(ds.counts[wid], w)])[0]
    for q in QUESTIONS:
        value = answers[q].value
        for v in value if isinstance(value, tuple) else (value,):
            ev = train.evidence(m1, w, q, v)
            assert len(ev) <= 5 and len(set(ev)) == len(ev) and set(ev) <= ids and not set(ev) & marker_ids
    doing = answers["doing"].value[0]
    assert train.evidence(m1, w, "doing", doing)  # a busy Moment has lines for its main reaction
    assert len(train.evidence(m1, w, "doing", doing, k=2)) <= 2 and train.evidence(m1, w, "doing", doing, k=0) == []
    with pytest.raises(ValueError):
        train.evidence(m1, w, "doing", doing, k=6)
    with pytest.raises(ValueError):
        train.evidence(m1, w, "doing", "dancing")
    # rung 0 (C20): lines holding a positively weighted emote ID or a "?"
    by_id = {m.message_id: m for m in w.messages}
    ev0 = train.evidence(m0, w, "doing", doing)
    assert len(ev0) <= 5 and set(ev0) <= ids
    assert all(by_id[i].emotes or "?" in by_id[i].text for i in ev0)


def test_rung0_evidence_finds_emote_lines(world):
    ds = world.ds
    wid = next(w for w in world.sp.test
               if world.truth[w]["doing"] == ["laughing"] and ds.counts[w].emote_counts.get("fake-kekw", 0) >= 3)
    ev = train.evidence(world.m0, ds.windows[wid], "doing", "laughing")
    by_id = {m.message_id: m for m in ds.windows[wid].messages}
    assert ev and all(by_id[i].emotes for i in ev)


@pytest.mark.parametrize("rung", [0, 1])
def test_evidence_keeps_the_top_scoring_lines(world, rung):
    ds = world.ds
    model = world.m1 if rung else world.m0
    wid = _busy_test_moment(world)
    w = ds.windows[wid]
    by_id = {m.message_id: m for m in w.messages}
    answers = train.predict(model, [(ds.counts[wid], w)])[0]
    for q, value in (("doing", answers["doing"].value[0]), ("aimed_at", answers["aimed_at"].value)):
        ev = train.evidence(model, w, q, value)
        scores = [train.line_score(model, by_id[i], q, value) for i in ev]
        assert all(s is not None and s > 0 for s in scores) and scores == sorted(scores, reverse=True)
        if len(ev) == 5:  # every line left out scores no higher than the last one kept
            rest = [train.line_score(model, m, q, value) for m in w.messages if m.message_id not in ev]
            assert all(s is None or s <= scores[-1] for s in rest)


def test_text_doc_words_never_hold_speaker_ids(world):
    ds = world.ds
    for wid in list(ds.windows)[:50]:
        assert not re.search(r"@s\d+", features.text_doc(ds.windows[wid]))
