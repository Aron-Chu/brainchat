import dataclasses
import json
import shutil

import pytest

from chatbrain import scheme


def test_scheme_values_match_spec():
    s = scheme.load()
    q = {x.key: x for x in s.questions}
    assert s.version == "cb_labels_2026_11_06_v1"
    assert q["doing"].kind == "multi" and q["doing"].values == ("laughing","cheering","shocked","tense","sad","asking","advising","complaining","chatting_as_usual","other")
    assert q["aimed_at"].values == ("streamer","game_content","guest","chat_itself","tech_problem","unclear")
    assert q["feeling"].values == ("supportive","critical","mixed","neutral")
    assert q["tone"].optional and q["tone"].blank_value == "no_clear_tone" and q["tone"].values == ("playful","serious","sarcastic","hostile")
    assert s.not_sure_reasons == ("unclear","conflicting","cause_missing","unknown_in_joke","needs_review")
    assert s.spread_values == ("repeated_emotes","chant","back_and_forth","separate_messages","few_posters")
    assert s.cause_sources == ("clip","game","stream_title")
    assert s.display["advising"] == "advising (backseating)"


def test_question_order_kinds_and_blanks():
    s = scheme.load()
    assert [x.key for x in s.questions] == ["doing", "aimed_at", "feeling", "tone"]
    assert [x.kind for x in s.questions] == ["multi", "single", "single", "single"]
    for x in s.questions:
        if x.key != "tone":
            assert not x.optional and x.blank_value is None


def test_guide_version_is_pinned():
    assert scheme.load().guide_version == "cb_guide_2026_11_06_v1"


def test_scheme_is_frozen():
    s = scheme.load()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.version = "other"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.questions[0].key = "other"  # type: ignore[misc]


def test_display_and_help_cover_every_value():
    s = scheme.load()
    for x in s.questions:
        assert x.key in s.display and x.key in s.help
        for v in x.values + ((x.blank_value,) if x.blank_value else ()):
            assert s.display[v]
            assert s.help[f"{x.key}.{v}"]
    for group, values in (("not_sure", s.not_sure_reasons), ("spread", s.spread_values), ("cause_source", s.cause_sources)):
        for v in values:
            assert s.display[v]
            assert s.help[f"{group}.{v}"]
    # "unclear" is both an aimed_at value and a not-sure reason: help keeps them apart.
    assert s.help["aimed_at.unclear"] != s.help["not_sure.unclear"]
    assert all("\n" not in line for line in s.help.values())


def test_load_explicit_path_matches_default(tmp_path):
    copy = tmp_path / "labels.json"
    shutil.copyfile(scheme.DEFAULT_PATH, copy)
    assert scheme.load(copy) == scheme.load()


def _raw():
    return json.loads(scheme.DEFAULT_PATH.read_text(encoding="utf-8"))


def _write(tmp_path, raw):
    p = tmp_path / "labels.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    return p


def test_load_rejects_unknown_kind(tmp_path):
    raw = _raw()
    raw["questions"][0]["kind"] = "ranked"
    with pytest.raises(scheme.SchemeError):
        scheme.load(_write(tmp_path, raw))


def test_load_rejects_duplicate_value(tmp_path):
    raw = _raw()
    raw["questions"][2]["values"].append("neutral")
    with pytest.raises(scheme.SchemeError):
        scheme.load(_write(tmp_path, raw))


def test_load_rejects_blank_value_on_required_question(tmp_path):
    raw = _raw()
    raw["questions"][1]["blank_value"] = "no_target"
    # Display and help cover the made-up value, so only the blank rule can fail.
    raw["display"]["no_target"] = "no target"
    raw["help"]["aimed_at.no_target"] = "No target."
    with pytest.raises(scheme.SchemeError, match="only an optional question may have a blank_value"):
        scheme.load(_write(tmp_path, raw))


def test_load_rejects_name_with_trailing_newline(tmp_path):
    raw = _raw()
    raw["questions"][0]["values"][0] = "laughing\n"
    raw["display"]["laughing\n"] = "laughing"
    raw["help"]["doing.laughing\n"] = "Laughing."
    with pytest.raises(scheme.SchemeError, match="snake_case"):
        scheme.load(_write(tmp_path, raw))


@pytest.mark.parametrize("brk", ["\r", "\n", "\r\n", " "])
def test_load_rejects_display_or_help_with_a_line_break(tmp_path, brk):
    for section, key in (("display", "advising"), ("help", "doing.advising")):
        raw = _raw()
        raw[section][key] = f"advising{brk}(backseating)"
        with pytest.raises(scheme.SchemeError, match="one line"):
            scheme.load(_write(tmp_path, raw))


def test_load_rejects_missing_help(tmp_path):
    raw = _raw()
    del raw["help"]["doing.laughing"]
    with pytest.raises(scheme.SchemeError):
        scheme.load(_write(tmp_path, raw))


def test_load_rejects_missing_display(tmp_path):
    raw = _raw()
    del raw["display"]["few_posters"]
    with pytest.raises(scheme.SchemeError):
        scheme.load(_write(tmp_path, raw))


def test_load_rejects_unknown_top_level_key(tmp_path):
    raw = _raw()
    raw["extra"] = 1
    with pytest.raises(scheme.SchemeError):
        scheme.load(_write(tmp_path, raw))


# --- validate_answers / validate_cause_source (ruling C1) -----------------------------

FULL = {"doing": ["laughing", "cheering"], "aimed_at": "streamer", "feeling": "supportive", "tone": "playful"}


def _check(answers, not_sure=None):
    scheme.validate_answers(scheme.load(), answers, {} if not_sure is None else not_sure)


@pytest.mark.parametrize(
    "answers, not_sure",
    [
        (FULL, {}),
        ({**FULL, "tone": "no_clear_tone"}, {}),  # the optional question's blank value
        ({"doing": ["chatting_as_usual"]}, {}),  # partial: one question
        ({}, {}),  # partial: nothing named yet
        ({}, {"aimed_at": ["unclear", "conflicting"]}),
        ({"doing": ["laughing"], "feeling": "mixed"}, {"aimed_at": ["cause_missing"], "tone": ["needs_review"]}),
        ({"aimed_at": "unclear"}, {"doing": ["unknown_in_joke"]}),  # "unclear" the value vs the reason
        ({"doing": ["laughing", "cheering", "shocked", "tense", "sad", "asking", "advising", "complaining",
                    "chatting_as_usual", "other"]}, {}),
    ],
)
def test_validate_answers_accepts_full_and_partial(answers, not_sure):
    _check(answers, not_sure)


@pytest.mark.parametrize(
    "answers, not_sure, rule",
    [
        # every key is a question key
        ({"spread": "chant"}, {}, "not a question"),
        ({}, {"cause": ["unclear"]}, "not a question"),
        # a question sits in at most one of the two dicts
        ({"tone": "playful"}, {"tone": ["unclear"]}, "both"),
        # doing: a non-empty list of valid values, in scheme order, no duplicates
        ({"doing": "laughing"}, {}, "non-empty list"),
        ({"doing": []}, {}, "non-empty list"),
        ({"doing": ["laughing", 1]}, {}, "doing values"),
        ({"doing": ["streamer"]}, {}, "doing values"),
        ({"doing": ["chatting_as_usual", "laughing"]}, {}, "scheme order"),
        ({"doing": ["laughing", "laughing"]}, {}, "repeats"),
        # each single question: one valid value string
        ({"aimed_at": ["streamer"]}, {}, "one aimed_at value"),
        ({"aimed_at": "laughing"}, {}, "one aimed_at value"),
        ({"feeling": None}, {}, "one feeling value"),
        ({"tone": ""}, {}, "one tone value"),
        # only an optional question may hold a blank value
        ({"aimed_at": "no_clear_tone"}, {}, "one aimed_at value"),
        # not_sure: a non-empty list of valid reasons, in scheme order, no duplicates
        ({}, {"tone": "unclear"}, "non-empty list"),
        ({}, {"tone": []}, "non-empty list"),
        ({}, {"tone": ["streamer"]}, "not-sure reasons"),
        ({}, {"tone": [None]}, "not-sure reasons"),
        ({}, {"tone": ["conflicting", "unclear"]}, "scheme order"),
        ({}, {"tone": ["unclear", "unclear"]}, "repeats"),
    ],
)
def test_validate_answers_rejects_each_rule(answers, not_sure, rule):
    with pytest.raises(ValueError, match=rule):
        _check(answers, not_sure)


def test_validate_answers_errors_name_the_question():
    with pytest.raises(ValueError, match="answers.doing"):
        _check({"doing": ["chatting_as_usual", "laughing"]})
    with pytest.raises(ValueError, match="not_sure.feeling"):
        _check({}, {"feeling": ["needs_review", "unclear"]})
    with pytest.raises(ValueError, match="tone"):
        _check({"tone": "playful"}, {"tone": ["unclear"]})


def test_validate_answers_rejects_non_dicts():
    with pytest.raises(ValueError, match="answers"):
        scheme.validate_answers(scheme.load(), [("doing", ["laughing"])], {})
    with pytest.raises(ValueError, match="not_sure"):
        scheme.validate_answers(scheme.load(), {}, None)


def test_validate_answers_never_echoes_free_text():
    secret = "zz secret chat line"
    cases = [
        ({secret: "x"}, {}),
        ({}, {secret: ["unclear"]}),
        ({"doing": [secret]}, {}),
        ({"aimed_at": secret}, {}),
        ({}, {"tone": [secret]}),
    ]
    for answers, not_sure in cases:
        with pytest.raises(ValueError) as info:
            _check(answers, not_sure)
        assert secret not in str(info.value)


def test_validate_answers_accepts_fake_truth_for_every_window():
    from datetime import datetime, timezone

    from chatbrain import fake

    s = scheme.load()
    t = fake.truth(fake.make_export(7, 200, 50, datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)))
    assert len(t) == 250
    for answers in t.values():
        scheme.validate_answers(s, answers, {})


def test_validate_cause_source():
    s = scheme.load()
    scheme.validate_cause_source(s, None)
    for v in s.cause_sources:
        scheme.validate_cause_source(s, v)
    for bad in ("", "chat", "Clip", "clip ", 1, ["clip"]):
        with pytest.raises(ValueError, match="cause_source"):
            scheme.validate_cause_source(s, bad)  # type: ignore[arg-type]
