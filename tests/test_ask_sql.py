"""The read-only ask connection and one SQL template per intent, on made-up data (Task 10b, B14)."""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import timedelta

import pytest

import pagekit
from chatbrain import ask
from chatbrain.ask import query as q_mod
from chatbrain.ask import sql as sql_mod


@pytest.fixture
def world(tmp_path):
    w = pagekit.build_world(tmp_path)
    yield w
    w.store.close()


def page_of(world):
    return pagekit.client(world).app.state.page


def q(**kw):
    return q_mod.blank() | kw


def local_hm(world, window_id):
    return world.windows[window_id].anchor_at.astimezone().strftime("%H:%M")


# --- the connection ------------------------------------------------------------------------------


def test_readonly_reads_approved_columns(world):
    with sql_mod.readonly(world.store.path) as ro:
        rows = ro.rows("SELECT window_id, anchor_at FROM windows ORDER BY window_id LIMIT 2", [])
    assert len(rows) == 2


def test_readonly_denies_attach(world, tmp_path):
    other = tmp_path / "other.db"
    sqlite3.connect(other).close()
    with sql_mod.readonly(world.store.path) as ro, pytest.raises(sql_mod.Denied):
        ro.rows("ATTACH DATABASE ? AS x", [str(other)])


def test_readonly_denies_write(world):
    with sql_mod.readonly(world.store.path) as ro:
        for stmt, params in (
            ("DELETE FROM windows", []),
            ("UPDATE windows SET channel = 'x'", []),
            ("INSERT INTO emote_names (channel, emote_id, name) VALUES ('a', 'b', 'c')", []),
            ("CREATE TABLE t (x)", []),
            ("PRAGMA query_only = OFF", []),
            ("PRAGMA writable_schema = ON", []),
        ):
            with pytest.raises(sql_mod.Denied):
                ro.rows(stmt, params)
    assert world.store.window_ids()  # nothing changed


def test_readonly_denies_unlisted_table(world):
    with sql_mod.readonly(world.store.path) as ro:
        for stmt in (
            "SELECT speaker FROM messages",  # a listed table, but not an approved column
            "SELECT * FROM assignments",
            "SELECT * FROM asks",
            "SELECT * FROM sqlite_master",
            "SELECT load_extension('x')",
            "SELECT readfile('cb.db')",
        ):
            with pytest.raises(sql_mod.Denied):
                ro.rows(stmt, [])


def test_slow_query_stops_at_deadline(world):
    big = json.dumps(list(range(3000)))
    with sql_mod.readonly(world.store.path, deadline_s=0.2) as ro, pytest.raises(sql_mod.TooSlow):
        ro.rows("SELECT count(*) FROM json_each(?) a, json_each(?) b, json_each(?) c", [big, big, big])


def test_row_cap(world):
    with sql_mod.readonly(world.store.path) as ro:
        rows = ro.rows("SELECT value FROM json_each(?)", [json.dumps(list(range(50)))], cap=10)
    assert len(rows) == 10 and ro.capped


# --- one template per intent ---------------------------------------------------------------------


def test_each_intent_on_fake_data(world):
    t = pagekit.truth_labels(world, skip={"fake3-w0021"})
    page = page_of(world)
    new = world.new_moments

    found = ask.run(page, q(intent="find_moments", doing=["laughing"]))
    assert found.ids == ["fake3-w0018", "fake3-w0027"]  # posters joining, biggest first
    assert any(b["kind"] == "cards" for b in found.blocks)

    top = ask.run(page, q(intent="top_moments"))
    assert top.ids[0] == "fake3-w0018" and len(top.ids) == 5
    assert any(b["kind"] == "bars" for b in top.blocks)

    detail = ask.run(page, q(intent="moment_detail", at=local_hm(world, "fake3-w0020")))
    assert detail.ids == ["fake3-w0020"]
    assert any(b["kind"] == "moment" for b in detail.blocks)

    explain = ask.run(page, q(intent="explain_answer", doing=["cheering"]), window_id="fake3-w0022")
    assert explain.ids == ["fake3-w0022"]
    assert any(b["kind"] == "evidence" for b in explain.blocks)
    report = ask.run(page, q(intent="explain_answer", tone=["sarcastic"]))
    assert any(b["kind"] == "notice" for b in report.blocks)  # no model yet: says so plainly

    count = ask.run(page, q(intent="count", about="doing", by="value"))
    rows = {r["label"]: r for r in next(b for b in count.blocks if b["kind"] == "count")["rows"]}
    assert rows["laughing"]["k"] == 2 and rows["laughing"]["n"] == len(new)
    assert rows["not sure"]["k"] == 1  # w0021 has no label and no model

    trend = ask.run(page, q(intent="trend", doing=["laughing"], time="all", by="stream"))
    groups = next(b for b in trend.blocks if b["kind"] == "bars")["rows"]
    assert [g["k"] for g in groups] == [3, 2]  # old stream, then the last stream
    assert all(g["n"] >= g["k"] for g in groups)

    emote = ask.run(page, q(intent="emote_meaning", emote=["KEKW"], time="all"))
    mix = next(b for b in emote.blocks if b["kind"] == "mix")
    kekw = [wid for wid in world.windows if "fake-kekw" in pagekit.counts(world.store, wid)["emote_counts"]]
    assert mix["n"] == len(kekw)
    assert mix["moments"] + mix["ordinary"] == mix["n"]
    assert len(emote.ids) <= 3 and emote.ids[0] == "fake3-w0018"  # most uses first

    cant = ask.run(page, q(intent="cant_answer", reason="on_stream", asks_cause=True))
    assert cant.line_text.startswith("Chat alone can't show what happened on stream")
    assert cant.ids  # the closest Moments
    assert t  # the truth labels were written


def test_not_sure_never_matches_a_value(world):
    pagekit.stub_model(world.store, world.models, pagekit.RUNG1_ID, 1)
    under, over = "fake3-w0018", "fake3-w0027"
    pagekit.predict(world.store, under, pagekit.RUNG1_ID, "doing", ["laughing"], 0.3)  # under the .5 threshold
    pagekit.predict(world.store, over, pagekit.RUNG1_ID, "doing", ["laughing"], 0.9)
    page = page_of(world)
    found = ask.run(page, q(intent="find_moments", doing=["laughing"]))
    assert found.ids == [over]
    assert found.maybe >= 1  # w0018 and every other unanswered Moment are counted apart
    assert found.more == f"+{found.maybe} more where BrainChat isn't sure."
    assert found.line_text.endswith(found.more)
    assert under not in found.ids
    # The owner's own "not sure" wins over a model that is sure, and still never matches
    pagekit.label(page.store, over, {}, {"doing": ["conflicting"]}, at=5)
    again = ask.run(page, q(intent="find_moments", doing=["laughing"]))
    assert over not in again.ids and again.maybe == found.maybe + 1


def test_top_sorts_by_count_not_confidence(world):
    pagekit.stub_model(world.store, world.models, pagekit.RUNG1_ID, 1)
    big, small = "fake3-w0018", "fake3-w0027"  # 11 and 3 posters joining
    pagekit.predict(world.store, big, pagekit.RUNG1_ID, "doing", ["laughing"], 0.55)
    pagekit.predict(world.store, small, pagekit.RUNG1_ID, "doing", ["laughing"], 0.99)
    page = page_of(world)
    top = ask.run(page, q(intent="top_moments", doing=["laughing"]))
    assert top.ids == [big, small]
    lines = ask.run(page, q(intent="top_moments", doing=["laughing"], sort="lines"))
    assert lines.ids == [big, small]


def test_phrase_reaches_only_windows_with_text(world):
    pagekit.truth_labels(world)
    # one page with a moving clock: page start ran expiry at NOW, and nothing runs it again here
    t = {"now": pagekit.NOW}
    page = pagekit.client(world, clock=lambda: t["now"]).app.state.page
    word = next(m.text.split()[0] for m in world.windows["fake3-w0018"].messages if m.text.split())
    asked = q(intent="find_moments", phrase=word, time="all")
    hit = ask.run(page, asked)
    assert hit.ids
    chip = next(c for c in hit.chips if c.token.startswith("phrase:"))
    assert "text" in chip.text  # says it only reaches windows whose text still exists
    last = max(world.windows[i].text_expires_at for i in hit.ids)
    t["now"] = last - timedelta(seconds=1)
    assert ask.run(page, asked).ids
    t["now"] = last  # every hit's text is due, but the hourly run hasn't deleted it yet
    marks = ",".join("?" * len(hit.ids))
    stored = world.store.conn.execute(
        f"SELECT COUNT(*) FROM messages WHERE window_id IN ({marks})", hit.ids
    ).fetchone()[0]
    states = {r[0] for r in world.store.conn.execute(
        f"SELECT text_state FROM windows WHERE window_id IN ({marks})", hit.ids)}  # fmt: skip
    assert stored > 0 and states == {"present"}
    assert ask.run(page, asked).ids == []
    later = pagekit.client(world, now=pagekit.NOW + timedelta(days=40)).app.state.page  # after the deletion
    gone = ask.run(later, asked)
    assert gone.ids == []


def test_min_posters_and_channel_scope(world):
    pagekit.truth_labels(world)
    page = page_of(world)
    many = ask.run(page, q(intent="find_moments", min_posters=10))
    assert set(many.ids) == {"fake3-w0018", "fake3-w0022"}
    other = ask.run(page, q(intent="find_moments", channel=["fakechan01"], time="all"))
    assert len(other.ids) == 5  # the limit; all 12 Moments are on fakechan01
    assert other.total == 12


def test_ask_sql_never_holds_question_text(world, monkeypatch):
    pagekit.truth_labels(world)
    seen: list[tuple[str, list]] = []
    real = sql_mod.ReadOnly.rows

    def spy(self, sql, params, cap=None):
        seen.append((sql, list(params)))
        return real(self, sql, params, cap)

    monkeypatch.setattr(sql_mod.ReadOnly, "rows", spy)
    ask.run(page_of(world), q(intent="find_moments", doing=["laughing"], phrase="drop table x", time="all"))
    assert seen
    assert all("drop table" not in sql.lower() for sql, _ in seen)
    assert any("%drop table x%" in str(p) for _, params in seen for p in params)  # bound, never spliced


def test_count_sides_show_their_not_sure(world):
    pagekit.truth_labels(world, skip={"fake3-w0021"})
    page = page_of(world)
    a = ask.run(page, q(intent="count", doing=["laughing"], by="stream", time="all"))
    rows = next(b for b in a.blocks if b["kind"] == "count")["rows"]
    assert [r["maybe"] for r in rows] == [0, 1]  # w0021 (last stream) has no label and no model
    assert "1 not sure" in a.line_text
    period = ask.run(page, q(intent="count", doing=["laughing"], by="period"))
    rows = next(b for b in period.blocks if b["kind"] == "count")["rows"]
    assert rows[0]["label"] == "last stream" and rows[0]["maybe"] == 1


def test_trend_says_how_much_history_it_covers(world):
    pagekit.truth_labels(world)
    page = page_of(world)
    two = ask.run(page, q(intent="trend", doing=["laughing"], time="all", by="stream"))
    assert "Covers 2 streams." in two.line_text
    one = ask.run(page, q(intent="trend", doing=["laughing"], time="last_stream", by="stream"))  # run, not check
    assert one.line_text.startswith("Not enough history: only 1 stream in reach.")


def test_explain_without_a_moment_reads_the_model_report(world):
    rep = pagekit.report()
    rep["tone"] |= {"n_graded": 120, "per_value": {"sarcastic": [0.5, 0.25, 8]}}
    pagekit.stub_model(world.store, world.models, pagekit.RUNG1_ID, 1, rep=rep, thresholds={"tone": 0.65})
    a = ask.run(page_of(world), q(intent="explain_answer", tone=["sarcastic"]))
    text = a.line_text
    assert "Tone: agreement .80 [.70, .90] on 100 answered of 120 graded" in text
    assert "threshold .65 (made-up labels)" in text
    assert "sarcastic: precision .50, recall .25, 8 in the test set" in text
    # a synthetic model: the suffix stays on the settled line (F3); a bare run times nothing, so no seconds
    assert re.fullmatch(r"\d+ steps? · made-up labels", a.step_text), a.step_text


def test_emote_mix_shows_doing_feeling_and_tone(world):
    pagekit.truth_labels(world)
    a = ask.run(page_of(world), q(intent="emote_meaning", emote=["KEKW"], time="all"))
    mix = next(b for b in a.blocks if b["kind"] == "mix")
    assert [g for g, _ in mix["groups"]] == ["Doing", "Feels", "Tone"]
    for _, rows in mix["groups"]:
        assert rows and all(r["n"] == mix["n"] for r in rows)


def test_moment_detail_has_every_answer_evidence_and_cause(world):
    pagekit.truth_labels(world)
    page = page_of(world)
    a = ask.run(page, q(intent="moment_detail", at=local_hm(world, "fake3-w0018")))
    b = next(b for b in a.blocks if b["kind"] == "moment")
    assert b["questions"] == list(page.questions)
    assert b["m"].lines  # chat lines (and evidence, with a model) are filled while text exists
    assert "Cause unknown · no cause note." in a.line_text
    pagekit.label(page.store, "fake3-w0018", {}, cause_note="raid came in", cause_source="clip", at=9)
    noted = ask.run(page, q(intent="moment_detail", at=local_hm(world, "fake3-w0018")))
    assert "Cause note: raid came in." in noted.line_text


def test_no_match_offers_to_drop_the_narrowest_filter(world):
    pagekit.truth_labels(world)
    a = ask.run(page_of(world), q(intent="find_moments", doing=["laughing"], aimed=["tech_problem"], time="all"))
    assert a.ids == [] and a.line_text.startswith("0 Moments match.")
    label, toks = a.follow_ups[0]
    assert label == "without aimed at: tech problem"
    assert "aimed:tech_problem" not in toks and "doing:laughing" in toks
