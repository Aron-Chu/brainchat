import json
from datetime import datetime, timedelta, timezone

import pytest

from chatbrain import metrics, scheme as scheme_mod
from chatbrain.export_format import Emote, Message, Window, iso
from chatbrain.spread import compute
from chatbrain.store import Store

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)
SCHEME = scheme_mod.load()
PROVENANCE = {"blind": "blind", "suggested": "accepted", "repeat": "repeat", "page": "corrected-on-page"}


def test_wilson_matches_research_notes():
    lo, hi = metrics.wilson(200, 400); assert abs(lo - 0.4512) < 5e-4 and abs(hi - 0.5488) < 5e-4
    lo, hi = metrics.wilson(360, 400); assert abs(lo - 0.8667) < 5e-4 and abs(hi - 0.9257) < 5e-4


def test_wilson_edges():
    assert metrics.wilson(0, 0) == (0.0, 1.0)  # no trials: anything is possible
    lo, hi = metrics.wilson(0, 10)
    assert lo == 0.0 and abs(hi - 0.2775) < 5e-4
    lo, hi = metrics.wilson(10, 10)
    assert hi == 1.0 and abs(lo - 0.7225) < 5e-4
    for k, n in ((5, 4), (-1, 4), (0, -1), (True, 4), (1.5, 4)):
        with pytest.raises(ValueError):
            metrics.wilson(k, n)


def test_question_report_single_value_counts_abstentions():
    # (target, shown answer or None for an abstention)
    graded = [("streamer", "streamer"), ("streamer", "guest"), ("guest", "guest"), ("guest", None), ("streamer", None)]
    rep = metrics.question_report(("streamer", "guest", "unclear"), False, graded, not_sure=(1, 3))
    assert (rep.n_graded, rep.answered) == (5, 3)
    assert rep.agreement == pytest.approx(2 / 3)
    assert rep.agreement_ci == metrics.wilson(2, 3)
    assert rep.abstain_rate == pytest.approx(2 / 5)
    # precision over answered windows; recall over every graded window (an abstention is a miss)
    assert rep.per_value["streamer"] == (pytest.approx(1.0), pytest.approx(1 / 3), 3)
    assert rep.per_value["guest"] == (pytest.approx(0.5), pytest.approx(0.5), 2)
    assert rep.per_value["unclear"] == (0.0, 0.0, 0)
    f1_streamer, f1_guest = 2 * 1 * (1 / 3) / (1 + 1 / 3), 0.5
    assert rep.macro_f1 == pytest.approx((f1_streamer + f1_guest) / 2)  # unclear: no support, never predicted
    assert rep.aron_not_sure_also_abstained == (1, 3)


def test_question_report_doing_is_exact_set_with_partial_credit():
    values = ("laughing", "cheering", "shocked")
    graded = [
        (("laughing",), ("laughing",)),
        (("laughing", "cheering"), ("laughing",)),  # partial: wrong for agreement, a hit for laughing
        (("shocked",), ("laughing", "shocked")),
    ]
    rep = metrics.question_report(values, True, graded, not_sure=(0, 0))
    assert rep.answered == 3 and rep.agreement == pytest.approx(1 / 3)
    assert rep.per_value["laughing"] == (pytest.approx(2 / 3), pytest.approx(1.0), 2)
    assert rep.per_value["cheering"] == (0.0, 0.0, 1)
    assert rep.per_value["shocked"] == (pytest.approx(1.0), pytest.approx(1.0), 1)


def test_question_report_with_nothing_answered():
    rep = metrics.question_report(("a", "b"), False, [("a", None)], not_sure=(0, 0))
    assert rep.answered == 0 and rep.agreement is None and rep.agreement_ci == (0.0, 1.0)
    assert rep.abstain_rate == 1.0 and rep.macro_f1 == 0.0
    empty = metrics.question_report(("a", "b"), False, [], not_sure=(0, 0))
    assert empty.n_graded == 0 and empty.abstain_rate is None and empty.agreement is None


def test_question_report_serialises():
    rep = metrics.question_report(("a", "b"), False, [("a", "a"), ("b", None)], not_sure=(0, 1))
    doc = json.loads(json.dumps(rep.to_dict()))
    assert doc["n_graded"] == 2 and doc["per_value"]["a"] == [1.0, 1.0, 1]


# --- self-consistency ----------------------------------------------------------------------


def _window(wid: str) -> Window:
    start = NOW - timedelta(days=1)
    msgs = (Message(f"{wid}-m1", 31_000, "s1", "KEKW", (Emote("fake-kekw", "KEKW"),)),)
    return Window(wid, "fakechan01", "fakechan01-s1", "moment", start + timedelta(seconds=30), start,
                  start + timedelta(seconds=120), "en", NOW + timedelta(days=10), msgs)


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "cb.db") as s:
        with s.transaction() as conn:
            for wid in ("w1", "w2", "w3", "w4", "w5"):
                assert s.insert_window(_window(wid), "vod", NOW)
                conn.execute("INSERT INTO features (window_id, json) VALUES (?, ?)",
                             (wid, compute(_window(wid)).to_json()))
        yield s


def _label(store, wid, project, answers, not_sure=None, *, origin="aron", at=0):
    with store.transaction() as conn:
        conn.execute(
            "INSERT INTO labels (window_id, project, origin, provenance, labels_version, guide_version,"
            " answers_json, not_sure_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (wid, project, origin, "synthetic" if origin == "synthetic" else PROVENANCE[project], SCHEME.version,
             SCHEME.guide_version, json.dumps(answers), json.dumps(not_sure or {}), iso(NOW + timedelta(hours=at))),
        )


def test_self_consistency_first_vs_repeat(store):
    full = {"doing": ["laughing"], "aimed_at": "streamer", "feeling": "supportive", "tone": "playful"}
    # w1: the repeat agrees everywhere
    _label(store, "w1", "blind", full, at=0)
    _label(store, "w1", "repeat", full, at=100)
    # w2: doing differs (exact set), tone not sure in the repeat (not compared), and a later
    # page fix doesn't replace the first label
    _label(store, "w2", "suggested", {**full, "doing": ["laughing", "cheering"]}, at=0)
    _label(store, "w2", "page", full, at=1)
    _label(store, "w2", "repeat", {k: v for k, v in full.items() if k != "tone"}, {"tone": ["unclear"]}, at=100)
    # w3: first label only; w4: repeat only (a demo task with no earlier label); w5: synthetic pair
    _label(store, "w3", "blind", full, at=0)
    _label(store, "w4", "repeat", full, at=100)
    _label(store, "w5", "blind", full, origin="synthetic", at=0)
    _label(store, "w5", "repeat", {**full, "aimed_at": "guest"}, origin="synthetic", at=100)

    sc = metrics.self_consistency(store, SCHEME)
    assert set(sc) == {"doing", "aimed_at", "feeling", "tone"}
    assert sc["doing"][:2] == (1, 2)
    assert sc["aimed_at"][:2] == (2, 2)
    assert sc["feeling"][:2] == (2, 2)
    assert sc["tone"][:2] == (1, 1)
    for agree, compared, lo, hi in sc.values():
        assert (lo, hi) == metrics.wilson(agree, compared)


def test_self_consistency_with_no_repeats(store):
    _label(store, "w1", "blind", {"aimed_at": "streamer"})
    assert metrics.self_consistency(store, SCHEME) == {q.key: (0, 0, 0.0, 1.0) for q in SCHEME.questions}
