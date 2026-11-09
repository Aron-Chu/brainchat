import json
from datetime import datetime, timedelta, timezone

import pytest

from chatbrain import fake, scheme, spread
from chatbrain.export_format import Emote, Message, Window, parse_export
from chatbrain.spread import Counts, compute

START = datetime(2026, 11, 1, 20, 0, tzinfo=timezone.utc)
NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)
# Made-up emotes: a token with one of these names becomes an emote, like fake.py does.
EMOTES = {"KEKW": "fake-kekw", "LUL": "fake-lul", "Pog": "fake-pog", "Wubble": "fake-wubble"}


def window(*lines: tuple[str, str]) -> Window:
    """A hand-built made-up window; each line is ``(speaker, text)``."""
    messages = tuple(
        Message(f"m{i:03d}", i * 1000, speaker, text, tuple(Emote(EMOTES[t], t) for t in text.split() if t in EMOTES))
        for i, (speaker, text) in enumerate(lines, start=1)
    )
    return Window(
        window_id="fake0-w0001",
        channel="fakechan01",
        stream_id="fake0-c01-s1",
        anchor_kind="moment",
        anchor_at=START + timedelta(seconds=30),
        start_at=START,
        end_at=START + timedelta(seconds=120),
        language="en",
        text_expires_at=START + timedelta(days=14),
        messages=messages,
    )


def test_empty_window_is_few_posters_with_zero_counts():
    assert compute(window()) == Counts(
        n_messages=0, distinct_posters=0, msg_per_min=0.0, top_emote_id=None, top_emote_share=0.0,
        chant_share=0.0, mention_share=0.0, question_share=0.0, emote_counts={}, spread="few_posters",
        joining_lines=0, joining_posters=0,
    )  # fmt: skip


def test_one_message_window_is_few_posters_with_true_counts():
    # C32: zero counts are for the empty window only. One line said once is not a chant,
    # and an emote in one message is nobody joining (B2).
    assert compute(window(("s1", "KEKW KEKW what?"))) == Counts(
        n_messages=1, distinct_posters=1, msg_per_min=0.5, top_emote_id="fake-kekw", top_emote_share=1.0,
        chant_share=0.0, mention_share=0.0, question_share=1.0, emote_counts={"fake-kekw": 2}, spread="few_posters",
        joining_lines=0, joining_posters=0,
    )  # fmt: skip


def test_four_posters_is_few_posters_even_if_chanting():
    lines = [(f"s{i % 4 + 1}", "lets go") for i in range(8)]
    c = compute(window(*lines))
    assert (c.n_messages, c.distinct_posters, c.chant_share) == (8, 4, 1.0)
    assert c.spread == "few_posters"
    assert (c.joining_lines, c.joining_posters) == (8, 4)
    assert compute(window(*lines, ("s5", "lets go"))).spread == "chant"  # a fifth poster


def test_chant_at_exactly_30_percent():
    # Normalised: lowercase, whitespace collapsed, emotes as emote_<id>, so all three match.
    chant = [("s1", "lets go Pog"), ("s2", "LETS  GO Pog"), ("s3", " Lets go\tPog ")]
    others = [("s4", "what a run"), ("s5", "hi chat"), ("s6", "lol"), ("s1", "nice"), ("s2", "ok then"),
              ("s3", "brb"), ("s4", "hmm")]  # fmt: skip
    c = compute(window(*chant, *others))
    assert (c.n_messages, c.distinct_posters) == (10, 6)
    assert c.chant_share == 0.3 and c.top_emote_share == 0.3
    assert c.spread == "chant"
    assert compute(window(*chant, *others, ("s5", "anyway"))).spread == "separate_messages"  # 3 of 11


def test_repeated_emotes_at_exactly_40_percent():
    emote_lines = [("s1", "KEKW"), ("s2", "KEKW"), ("s3", "KEKW KEKW KEKW"), ("s4", "that was KEKW")]
    others = [("s5", "what a run"), ("s6", "hi chat"), ("s1", "lol"), ("s2", "nice"), ("s3", "LUL"), ("s4", "hmm")]
    c = compute(window(*emote_lines, *others))
    assert c.top_emote_id == "fake-kekw" and c.top_emote_share == 0.4  # share of messages, not of uses
    assert c.emote_counts == {"fake-kekw": 6, "fake-lul": 1}  # uses
    assert c.chant_share == 0.0  # one-token repeats are not a chant, and no longer line repeats
    assert c.spread == "repeated_emotes"
    assert compute(window(*emote_lines, *others, ("s5", "anyway"))).spread == "separate_messages"  # 4 of 11


def test_back_and_forth_at_20_percent():
    lines = [("s1", "hi chat"), ("s2", "@s1 hey there"), ("s3", "what a run"), ("s1", "@s3 true"), ("s4", "lol"),
             ("s5", "nice"), ("s6", "@user welcome"), ("s2", "brb"), ("s3", "hmm"), ("s4", "ok then")]  # fmt: skip
    c = compute(window(*lines))
    assert c.mention_share == 0.2  # "@user" names nobody who posted here, so it is not a reply
    assert c.spread == "back_and_forth"
    assert compute(window(*lines, ("s5", "anyway"))).spread == "separate_messages"  # 2 of 11


def test_otherwise_separate_messages():
    lines = [("s1", "what game is this?"), ("s2", "hi chat"), ("s3", "KEKW"), ("s4", "nice weather today"),
             ("s5", "who is that?"), ("s6", "lurking"), ("s1", "LUL"), ("s2", "back from work")]  # fmt: skip
    assert compute(window(*lines)) == Counts(
        n_messages=8, distinct_posters=6, msg_per_min=4.0,
        top_emote_id="fake-kekw",  # tied with fake-lul at one message each: the smaller ID wins
        top_emote_share=0.125, chant_share=0.0, mention_share=0.0, question_share=0.25,
        emote_counts={"fake-kekw": 1, "fake-lul": 1}, spread="separate_messages",
        joining_lines=0, joining_posters=0,  # the top emote is in one message: nobody joined it (B2)
    )  # fmt: skip


def test_joining_counts():
    lines = [
        ("s1", "lets go Pog"), ("s2", "LETS GO Pog"), ("s3", "lets go Pog"), ("s1", "lets   go Pog"),  # chant + Pog
        ("s4", "Pog"), ("s5", "Pog Pog"),  # the top emote only
        ("s6", "lets go"),  # neither: not the chant text, no Pog
        ("s6", "what a run"), ("s2", "KEKW"), ("s7", "hi chat"), ("s3", "ok"), ("s5", "nice"),
    ]  # fmt: skip
    c = compute(window(*lines))
    assert c.spread == "chant"  # the chant rule comes before Pog's 6 of 12
    assert c.chant_share == 4 / 12 and c.top_emote_id == "fake-pog" and c.top_emote_share == 0.5
    assert c.joining_lines == 6  # each line once, though the chant lines hold Pog too
    assert c.joining_posters == 5  # s1 s2 s3 s4 s5


def test_the_top_emote_joins_only_from_two_messages():
    # B2: like the chant, the top emote counts as joining only when it is in 2 or more messages.
    chant = [("s1", "lets go"), ("s2", "LETS GO"), ("s3", "zz KEKW")]
    c = compute(window(*chant))
    assert c.top_emote_id == "fake-kekw" and c.top_emote_share == 1 / 3
    assert (c.joining_lines, c.joining_posters) == (2, 2)  # the chant only: KEKW's one message is not joining
    c = compute(window(*chant, ("s4", "KEKW")))
    assert (c.joining_lines, c.joining_posters) == (4, 4)  # KEKW in 2 messages now joins, beside the chant


def test_ties_break_the_same_way_in_any_order():
    # Two chants of 2 lines each and two emotes of 2 lines each: the smaller text and ID win.
    lines = [("s1", "zz top"), ("s2", "aa bb"), ("s3", "zz top"), ("s4", "aa bb"), ("s5", "LUL"), ("s6", "KEKW"),
             ("s7", "LUL"), ("s1", "KEKW")]  # fmt: skip
    c = compute(window(*lines))
    assert c.top_emote_id == "fake-kekw" and c.chant_share == 0.25
    assert (c.joining_lines, c.joining_posters) == (4, 4)  # "aa bb" (s2 s4) and KEKW (s6 s1)
    assert compute(window(*reversed(lines))) == c


def test_counts_json_round_trip():
    c = compute(window(("s1", "lets go Pog"), ("s2", "lets go Pog"), ("s3", "@s1 what?"), ("s4", "KEKW"),
                       ("s5", "hi chat"), ("s6", "Pog")))  # fmt: skip
    s = c.to_json()
    assert Counts.from_json(s) == c
    assert set(json.loads(s)) == {
        "n_messages", "distinct_posters", "msg_per_min", "top_emote_id", "top_emote_share", "chant_share",
        "mention_share", "question_share", "emote_counts", "spread", "joining_lines", "joining_posters",
    }  # fmt: skip
    empty = compute(window())
    assert Counts.from_json(empty.to_json()) == empty
    assert json.loads(empty.to_json())["top_emote_id"] is None


def test_counts_from_json_is_strict():
    good = json.loads(compute(window(("s1", "KEKW"), ("s2", "hi chat"))).to_json())
    bad = [
        {**good, "chant_text": "lets go"},  # an unknown key, which could carry words
        {k: v for k, v in good.items() if k != "spread"},
        {**good, "spread": "riot"},
        {**good, "n_messages": True},
        {**good, "n_messages": 1.0},
        {**good, "chant_share": "0.5"},
        {**good, "chant_share": 1.5},
        {**good, "emote_counts": {"fake-kekw": -1}},
        {**good, "emote_counts": ["fake-kekw"]},
        {**good, "top_emote_id": 7},
    ]
    for doc in bad:
        with pytest.raises(ValueError):
            Counts.from_json(json.dumps(doc))
    for raw in ("[]", "not json", '{"n_messages": NaN}'):
        with pytest.raises(ValueError):
            Counts.from_json(raw)


def test_counts_json_holds_no_words():
    # C33: built on a chanting window, so a stored chant text would show up here.
    lines = [("s1", "zorbly plimf Wubble"), ("s2", "ZORBLY  plimf Wubble"), ("s3", "zorbly plimf Wubble"),
             ("s4", "grenkel snoff?"), ("s5", "@s1 quaddle"), ("s6", "vexmoor Wubble"), ("s2", "trillop"),
             ("s3", "blenwick drosk")]  # fmt: skip
    w = window(*lines)
    c = compute(w)
    assert c.spread == "chant" and c.chant_share == 3 / 8
    s = c.to_json()
    for m in w.messages:
        for token in m.text.split():
            if token in EMOTES:
                assert token not in s  # emote names never, only their IDs
            else:
                assert token.casefold() not in s.casefold()
    assert "emote_fake" not in s  # no normalised text either
    assert "fake-wubble" in s


def test_spread_values_match_the_scheme():
    assert spread.SPREAD_VALUES == scheme.load().spread_values


def test_every_made_up_window_computes():
    e = parse_export(fake.make_export(7, 60, 20, NOW))
    for w in e.windows:
        c = compute(w)
        assert c.spread in spread.SPREAD_VALUES
        assert c.n_messages == len(w.messages) and sum(c.emote_counts.values()) == sum(len(m.emotes) for m in w.messages)
        if len(w.messages) <= 1:
            assert c.spread == "few_posters"
        assert Counts.from_json(c.to_json()) == c
