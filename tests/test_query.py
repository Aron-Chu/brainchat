"""The ask query object: keys, the per-request schema, the validator and tokens (Task 10b)."""

from __future__ import annotations

import pytest

import pagekit
from chatbrain.ask import query as q_mod


@pytest.fixture
def world(tmp_path):
    w = pagekit.build_world(tmp_path)
    yield w
    w.store.close()


def held(world):
    return q_mod.held(world.store)


def test_blank_has_exactly_the_engine_keys():
    q = q_mod.blank()
    assert tuple(q) == q_mod.KEYS
    assert "min_posters" in q and "window_id" not in q
    assert q["limit"] == 5 and q["sort"] == "posters" and q["time"] == "last_stream"


def test_schema_enums_come_from_scheme_and_store(world):
    s = q_mod.schema(world.store, pagekit.SCHEME)
    props = s["properties"]
    assert s["type"] == "object" and s["additionalProperties"] is False
    assert sorted(s["required"]) == sorted(q_mod.KEYS)
    by_key = {q.key: q for q in pagekit.SCHEME.questions}
    assert props["doing"]["items"]["enum"] == list(by_key["doing"].values)
    assert props["aimed"]["items"]["enum"] == list(by_key["aimed_at"].values)
    assert props["feels"]["items"]["enum"] == list(by_key["feeling"].values)
    assert props["tone"]["items"]["enum"] == [*by_key["tone"].values, by_key["tone"].blank_value]
    assert props["spread"]["items"]["enum"] == list(pagekit.SCHEME.spread_values)
    assert props["channel"]["items"]["enum"] == ["fakechan01"]  # channels held, nothing else
    emotes = props["emote"]["items"]["enum"]
    assert "KEKW" in emotes and "OMEGALUL" in emotes
    assert all(isinstance(e, str) and e for e in emotes)
    assert "Aware" not in emotes  # not in the fake data, so the model can't name it
    assert props["intent"]["enum"] == list(q_mod.INTENTS)
    assert props["limit"]["minimum"] == 1 and props["limit"]["maximum"] == 20
    assert props["min_posters"]["anyOf"][0]["maximum"] == 10000


def test_schema_with_no_emotes_or_channels_holds_null(tmp_path):
    """B34: an empty enum isn't valid JSON schema for Ollama ``format``; the key is null instead,
    and the validator reads a null emote or channel list as empty."""
    store = pagekit.Store(tmp_path / "empty.db")
    try:
        h = q_mod.held(store)
        props = q_mod.schema(store, pagekit.SCHEME, h)["properties"]
        assert props["emote"] == {"type": "null"} and props["channel"] == {"type": "null"}
        assert props["doing"]["items"]["enum"]  # the scheme's keys keep their enums
        got = q_mod.check(q_mod.blank() | {"emote": None, "channel": None}, pagekit.SCHEME, h)
        assert got.query["emote"] == [] and got.query["channel"] == [] and got.ignored == []
        held_one = q_mod.Held(("fakechan01",), {})
        props = q_mod.schema(store, pagekit.SCHEME, held_one)["properties"]
        assert props["channel"]["items"]["enum"] == ["fakechan01"] and props["emote"] == {"type": "null"}
        assert q_mod.check(q_mod.blank() | {"doing": None}, pagekit.SCHEME, h).ignored == ["doing"]
    finally:
        store.close()


def test_check_passes_a_good_query(world):
    raw = q_mod.blank() | {"intent": "top_moments", "doing": ["laughing"], "aimed": ["guest"]}
    got = q_mod.check(raw, pagekit.SCHEME, held(world))
    assert got.query == raw
    assert got.ignored == [] and got.suggestions == []


@pytest.mark.parametrize(("limit", "want"), [(0, 1), (-3, 1), (1, 1), (7, 7), (20, 20), (99, 20)])
def test_check_clamps_limit(world, limit, want):
    got = q_mod.check(q_mod.blank() | {"limit": limit}, pagekit.SCHEME, held(world))
    assert got.query["limit"] == want
    if limit != want:
        assert "limit_clamped" in got.codes


def test_unknown_values_become_ignored_chips(world):
    raw = q_mod.blank() | {
        "doing": ["laughing", "dancing"],
        "channel": ["fakechan01", "fakechan77"],
        "emote": ["KEKW", "Aware"],
        "time": "last_year",
        "sort": "confidence",
        "is": ["toxic"],
    }
    got = q_mod.check(raw, pagekit.SCHEME, held(world))
    assert got.query["doing"] == ["laughing"]
    assert got.query["channel"] == ["fakechan01"]
    assert got.query["emote"] == ["KEKW"]
    assert got.query["time"] == "last_stream"
    assert got.query["sort"] == "posters"
    assert got.query["is"] == []
    for token in ("doing:dancing", "channel:fakechan77", "emote:Aware", "time:last_year", "sort:confidence", "is:toxic"):
        assert token in got.ignored


def test_exact_keys_drop_extras_and_never_take_a_window_id(world):
    raw = q_mod.blank() | {"window_id": "fake3-w0018", "sql": "DROP TABLE windows"}
    del raw["about"]
    got = q_mod.check(raw, pagekit.SCHEME, held(world))
    assert tuple(got.query) == q_mod.KEYS
    assert "window_id" not in got.query and got.query["about"] is None
    assert "window_id" in got.ignored and "sql" in got.ignored
    assert "missing:about" in got.codes


def test_unusable_query_becomes_cant_answer_with_three_runnable_suggestions(world):
    for raw in (None, "top", [], q_mod.blank() | {"intent": "summarise_everything"}):
        got = q_mod.check(raw, pagekit.SCHEME, held(world))
        assert got.query["intent"] == "cant_answer"
        assert got.query["reason"] is None
        assert len(got.suggestions) == 3
        assert all(s in q_mod.suggestions(held(world)) for s in got.suggestions)
        assert "unusable" in got.codes


def test_emote_meaning_needs_exactly_one_emote(world):
    two = q_mod.check(
        q_mod.blank() | {"intent": "emote_meaning", "emote": ["KEKW", "LUL"], "time": "all"},
        pagekit.SCHEME,
        held(world),
    )
    assert two.query["emote"] == ["KEKW"] and "emote:LUL" in two.ignored
    none = q_mod.check(q_mod.blank() | {"intent": "emote_meaning"}, pagekit.SCHEME, held(world))
    assert none.query["intent"] == "cant_answer" and len(none.suggestions) == 3


def test_count_by_channel_needs_two_channels(world):
    got = q_mod.check(
        q_mod.blank() | {"intent": "count", "by": "channel", "channel": ["fakechan01"]}, pagekit.SCHEME, held(world)
    )
    assert got.query["by"] is None and "by:channel" in got.ignored


def test_top_sorts_only_by_a_count(world):
    got = q_mod.check(
        q_mod.blank() | {"intent": "top_moments", "sort": "least_sure"}, pagekit.SCHEME, held(world)
    )
    assert got.query["sort"] == "posters" and "sort:least_sure" in got.ignored


def test_phrase_must_be_word_for_word_in_the_question(world):
    raw = q_mod.blank() | {"phrase": "boss fight"}
    kept = q_mod.check(raw, pagekit.SCHEME, held(world), question="where did chat say boss fight?")
    assert kept.query["phrase"] == "boss fight"
    dropped = q_mod.check(raw, pagekit.SCHEME, held(world), question="where did chat say the boss died?")
    assert dropped.query["phrase"] is None and "phrase:boss fight" in dropped.ignored


@pytest.mark.parametrize(("at", "ok"), [("21:14", True), ("01:00", True), ("24:00", False), ("9:5", False), (930, False)])
def test_at_is_a_local_clock_time(world, at, ok):
    got = q_mod.check(q_mod.blank() | {"at": at}, pagekit.SCHEME, held(world))
    assert (got.query["at"] == at) is ok


def test_min_posters_range(world):
    assert q_mod.check(q_mod.blank() | {"min_posters": 21}, pagekit.SCHEME, held(world)).query["min_posters"] == 21
    assert q_mod.check(q_mod.blank() | {"min_posters": 0}, pagekit.SCHEME, held(world)).query["min_posters"] == 1
    assert (
        q_mod.check(q_mod.blank() | {"min_posters": 10**6}, pagekit.SCHEME, held(world)).query["min_posters"] == 10000
    )
    assert q_mod.check(q_mod.blank() | {"min_posters": True}, pagekit.SCHEME, held(world)).query["min_posters"] is None


def test_tokens_round_trip(world):
    raw = q_mod.blank() | {
        "intent": "find_moments", "doing": ["laughing", "cheering"], "aimed": ["guest"], "is": ["not_sure"],
        "emote": ["KEKW"], "channel": ["fakechan01"], "time": "7d", "at": "21:14", "sort": "newest", "limit": 9,
        "min_posters": 21, "about": "feeling", "phrase": "gg",
    }  # fmt: skip
    tokens = q_mod.tokens(raw)
    assert "doing:laughing" in tokens and "at:21:14" in tokens and "min_posters:21" in tokens
    back = q_mod.from_tokens(tokens, pagekit.SCHEME, held(world))
    assert back.query == raw
    assert back.ignored == []


def test_from_tokens_strikes_what_it_cant_read(world):
    back = q_mod.from_tokens(["doing:laughing", "nonsense", "limit:many", "doing:flying"], pagekit.SCHEME, held(world))
    assert back.query["doing"] == ["laughing"]
    assert back.query["limit"] == 5
    assert {"nonsense", "limit:many", "doing:flying"} <= set(back.ignored)


def test_chips_read_as_words(world):
    raw = q_mod.blank() | {"intent": "top_moments", "doing": ["laughing"], "aimed": ["game_content"]}
    texts = [c.text for c in q_mod.chips(raw, pagekit.SCHEME)]
    assert "doing: laughing" in texts
    assert "aimed at: game/content" in texts
    assert "sort: posters joining ↓" in texts
    assert "time: last stream" in texts


def test_five_suggestions_use_held_names(world):
    s = q_mod.suggestions(held(world))
    assert len(s) == 5
    assert s[3] == "What does KEKW mean in fakechan01?"
    assert s[4] == "Which Moments is BrainChat least sure about?"
