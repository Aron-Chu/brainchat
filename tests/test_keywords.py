"""Asking with no model: the keyword parser, the refusal pre-check and the default reading (Task 10b)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import pagekit
from chatbrain import ask, fake, ingest
from chatbrain.ask import keywords
from chatbrain.ask import query as q_mod
from chatbrain.store import Store

DATA = Path(__file__).parent / "data" / "ask_questions.json"
LIST_KEYS = ("doing", "aimed", "feels", "tone", "spread", "is", "emote", "channel")
SCALAR_KEYS = ("phrase", "at", "min_posters", "reason")
OPTIONAL_KEYS = ("time", "sort", "about", "by", "limit")

FIVE = [
    (
        "What were the biggest moments last stream?",
        {"intent": "top_moments", "sort": "posters", "time": "last_stream"},
    ),
    (
        "What made chat laugh the most?",
        {"intent": "top_moments", "doing": ["laughing"], "sort": "posters", "time": "last_stream"},
    ),
    (
        "When did chat turn on the streamer?",
        {"intent": "find_moments", "aimed": ["streamer"], "feels": ["critical"], "time": "last_stream"},
    ),
    (
        "What does KEKW mean in fakechan01?",
        {"intent": "emote_meaning", "emote": ["KEKW"], "channel": ["fakechan01"], "time": "all"},
    ),
    (
        "Which Moments is BrainChat least sure about?",
        {"intent": "find_moments", "is": ["not_sure"], "sort": "least_sure", "time": "all"},
    ),
]


@pytest.fixture
def world(tmp_path):
    w = pagekit.build_world(tmp_path)
    yield w
    w.store.close()


def page_of(world):
    return pagekit.client(world).app.state.page


def held(world):
    return q_mod.held(world.store)


def parse(world, question, **kw):
    return keywords.parse(question, pagekit.SCHEME, held(world), **kw)


def same(query: dict, expect: dict) -> list[str]:
    """The keys where ``query`` differs from ``expect`` (lists default to empty, scalars to None)."""
    bad = [k for k in LIST_KEYS if query[k] != expect.get(k, [])]
    bad += [k for k in SCALAR_KEYS if query[k] != expect.get(k)]
    bad += [k for k in OPTIONAL_KEYS if k in expect and query[k] != expect[k]]
    return bad


def test_five_home_suggestions_parse(world):
    assert [s for s, _ in FIVE] == q_mod.suggestions(held(world))
    t = pagekit.truth_labels(world, skip={"fake3-w0021"})  # one Moment stays not sure
    late = world.new_moments[-2]  # w0027: laughing at the streamer; the owner later calls it critical
    pagekit.label(world.store, late, t[late] | {"feeling": "critical"}, at=5)
    page = page_of(world)
    for question, expect in FIVE:
        got = parse(world, question)
        assert got.refused is None and got.leftover == [], question
        assert got.query["intent"] == expect["intent"], question
        assert same(got.query, expect) == [], (question, got.query)
        answer = ask.run(page, got.query)
        if expect["intent"] == "emote_meaning":
            mix = next(b for b in answer.blocks if b["kind"] == "mix")
            assert mix["n"] > 0
        else:
            assert answer.ids, question
    turned = ask.run(page, parse(world, FIVE[2][0]).query)
    assert turned.ids == [late]


def test_walkthrough_questions_parse(world):
    liked = parse(world, "what did chat like most")
    assert same(liked.query, {"intent": "top_moments", "feels": ["supportive"], "sort": "posters",
                              "time": "last_stream"}) == []  # fmt: skip
    assert liked.query["intent"] == "top_moments"

    crowd = parse(world, "more than 20 people")
    assert crowd.query["intent"] == "count"
    assert same(crowd.query, {"min_posters": 21, "by": "value", "about": "doing", "time": "last_stream"}) == []
    assert crowd.default is True

    bits = parse(world, "funniest bits from last night")
    assert bits.query["intent"] == "top_moments"
    assert same(bits.query, {"doing": ["laughing"], "time": "last_stream", "sort": "posters"}) == []
    assert bits.leftover == ["bits"]
    assert bits.query["reason"] is None  # "bits" is not an on-stream word


def test_default_reading_for_vague_question(world):
    pagekit.truth_labels(world)
    got = parse(world, "was it good?")
    assert got.default is True
    assert got.query["intent"] == "count"
    assert same(got.query, {"by": "value", "about": "doing", "time": "last_stream"}) == []
    answer = ask.run(page_of(world), got.query, notes={"default_reading": True})
    assert [b["kind"] for b in answer.blocks][:2] == ["count", "cards"]
    assert len(answer.ids) == 3
    assert [label for label, _ in answer.follow_ups] == ["funniest", "most critical"]
    assert answer.line_text.startswith("Took the plainest reading: what chat did on ")


def test_gibberish_still_answers(world):
    got = parse(world, "zxqv plorb wibble!!")
    assert got.default is True and got.refused is None
    assert got.leftover == ["zxqv", "plorb", "wibble"]
    answer = ask.run(page_of(world), got.query, notes={"default_reading": True, "didnt_understand": got.leftover})
    assert any(b["kind"] == "count" for b in answer.blocks)
    struck = [c.text for c in answer.chips if c.kind == "didnt"]
    assert struck == ["didn't understand: zxqv", "didn't understand: plorb", "didn't understand: wibble"]


@pytest.mark.parametrize(
    ("question", "reason"),
    [
        ("who started it", "identity"),
        ("who was chat dunking on", "identity"),
        ("is s4 the same guy from yesterday", "identity"),
        ("what is the username of s3?", "identity"),
        ("whats s2's real name", "identity"),
        ("did chat say anything about me", "identity"),
        ("did anyone say something racist", "traits"),
        ("which messages were homophobic", "traits"),
        ("is this message toxic?", "single_message"),
        ("say it like the streamer", "as_streamer"),
        ("send this to discord", "outbound"),
    ],
)
def test_refusal_pre_check(question, reason):
    assert keywords.refusal(question) == reason


@pytest.mark.parametrize(
    "question",
    ["what made chat laugh the most", "when did chat turn on the streamer", "funniest bits from last night",
     "what were they chanting at 3:12", "export the timestamps", "how did chat feel about my guest"],
)  # fmt: skip
def test_ordinary_questions_are_not_refused(question):
    assert keywords.refusal(question) is None


def test_refused_question_gets_a_cant_answer_query(world):
    got = parse(world, "who started it at 22:03?")
    assert got.refused == "identity"
    assert got.query["intent"] == "cant_answer" and got.query["reason"] == "identity"
    assert got.query["at"] == "22:03"
    assert got.leftover == []  # nothing from the text is kept


def test_word_map_is_wide():
    words = keywords.load()
    for q in pagekit.SCHEME.questions:
        key = q_mod.QUESTION_KEY[q.key]
        for v in q.values + ((q.blank_value,) if q.blank_value else ()):
            assert len(words.values[key][v]) >= 5, (key, v)
    for v in pagekit.SCHEME.spread_values:
        assert len(words.values["spread"][v]) >= 5, v


def test_emote_name_or_slang(world):
    meaning = parse(world, "what does kekw mean?")
    assert meaning.query["intent"] == "emote_meaning" and meaning.query["emote"] == ["KEKW"]
    slang = parse(world, "when was chat KEKW last stream")
    assert slang.query["doing"] == ["laughing"] and slang.query["emote"] == []
    spam = parse(world, "where did chat spam OMEGALUL")
    assert spam.query["emote"] == ["OMEGALUL"] and spam.query["spread"] == []
    assert spam.query["intent"] == "find_moments"


def test_times_numbers_and_refs(world):
    at = parse(world, "what was chat doing at 10:03pm?")
    assert at.query["intent"] == "moment_detail" and at.query["at"] == "22:03"
    assert parse(world, "why was everyone spamming KEKW at like 1am").query["at"] == "01:00"
    top = parse(world, "best 10 this week ranked")
    assert top.query["intent"] == "top_moments" and top.query["limit"] == 10 and top.query["time"] == "7d"
    assert parse(world, "hype moments with 50+ people in").query["min_posters"] == 50
    phrase = parse(world, 'where did chat say "gg go next"')
    assert phrase.query["phrase"] == "gg go next"


def test_on_stream_questions_cant_answer(world):
    for question in ("was chat mad about the sponsor", "wheres the boss fight", "how many people were watching"):
        got = parse(world, question)
        assert got.query["intent"] == "cant_answer" and got.query["reason"] == "on_stream", question


def test_follow_ups_carry_over_with_kept_marks(world):
    first = parse(world, "what made chat laugh the most")
    only = parse(world, "only the guest ones", previous=first.query)
    assert only.query["follow_up"] == "refine"
    assert only.query["doing"] == ["laughing"] and only.query["aimed"] == ["guest"]
    assert "doing:laughing" in only.kept and "aimed:guest" not in only.kept
    week = parse(world, "same for last week", previous=first.query)
    assert week.query["time"] == "7d" and week.query["doing"] == ["laughing"]
    second = parse(world, "the 2nd one", previous=first.query)
    assert second.query["follow_up"] == "about_result"
    assert second.query["intent"] == "moment_detail" and second.query["ref"] == 2


def test_per_moment_questions(world):
    why = parse(world, "why cheering?", window_id="fake3-w0022")
    assert why.query["intent"] == "explain_answer" and why.query["doing"] == ["cheering"]
    what = parse(world, "what happened here", window_id="fake3-w0022")
    assert what.query["intent"] == "moment_detail"


def _world3(tmp: Path) -> Store:
    raw = fake.make_export(3, 12, 16, pagekit.NOW, channels=3)
    (tmp / "e3.json").write_bytes(raw)
    store = Store(tmp / "cb3.db")
    ingest.import_export(store, tmp / "e3.json", tmp / "imported3", pagekit.NOW)
    return store


def test_ask_questions_file_scores(tmp_path, capsys):
    entries = json.loads(DATA.read_text(encoding="utf-8"))
    assert len(entries) == 60
    store = _world3(tmp_path)
    try:
        h = q_mod.held(store)
        assert len(h.channels) >= 2
        intents = filters = 0
        misses = []
        for e in entries:
            text = e["q"].lower()
            assert "demo" not in text and "aware" not in text, e["q"]  # made-up names only
            expect = e["expect"]
            assert q_mod.check(q_mod.blank() | expect, pagekit.SCHEME, h).ignored == [], e["q"]
            got = keywords.parse(e["q"], pagekit.SCHEME, h)
            ok_intent = got.query["intent"] == expect["intent"]
            ok_filters = ok_intent and same(got.query, expect) == []
            intents += ok_intent
            filters += ok_filters
            if not ok_filters:
                misses.append(e["q"])
        with capsys.disabled():
            print(f"\nask_questions.json (keyword path): exact intent {intents}/60 = {intents / 60:.0%},"
                  f" exact filters {filters}/60 = {filters / 60:.0%}")  # fmt: skip
            print("misses: " + " | ".join(misses))
    finally:
        store.close()


def test_on_stream_keeps_labels_for_the_closest_moments(world):
    got = parse(world, "was chat mad about the sponsor")
    assert got.query["intent"] == "cant_answer" and got.query["reason"] == "on_stream"
    assert got.query["doing"] == ["complaining"]  # ENGINE §2: the closest Moments by labels


def test_count_about_a_question_reads_by_value(world):
    for question in ("how did chat feel about my guest", "does chat even like me", "did chat like the game tonight"):
        got = parse(world, question)
        assert got.query["intent"] == "count" and got.query["about"] == "feeling", question
        assert got.query["by"] == "value", question


def test_more_word_map_cases(world):
    assert parse(world, "export the timestamps").query["sort"] == "oldest"
    lose = parse(world, "which windows lose their text")
    assert lose.query["is"] == ["has_text"] and lose.query["sort"] == "oldest"
    assert all("who" not in w.split() for w in keywords.load().values["aimed"]["unclear"])  # never refused


def test_viewer_questions_say_posters_not_viewers(world):
    pagekit.truth_labels(world)
    got = parse(world, "how many people were watching")
    assert got.viewers is True and got.query["reason"] == "on_stream"
    answer = ask.run(page_of(world), got.query, notes={"viewers": True})
    assert "BrainChat counts people who posted, not viewers." in answer.line_text
    assert answer.ids  # then the posters joining, on the closest Moments
    assert parse(world, "what made chat laugh the most").viewers is False


def test_per_moment_on_stream_question_answers_that_moment(world):
    pagekit.truth_labels(world)
    got = parse(world, "was chat mad about the sponsor", window_id="fake3-w0022")
    assert got.query["intent"] == "moment_detail" and got.query["asks_cause"] is True
    answer = ask.run(page_of(world), got.query, window_id="fake3-w0022")
    assert answer.ids == ["fake3-w0022"]
    assert answer.line[0] == "Chat alone can't show what happened on stream, so the cause is unknown."
