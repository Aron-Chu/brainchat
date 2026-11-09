import copy
import json
from datetime import datetime, timedelta, timezone

import pytest

from chatbrain.export_format import (
    DeletionCheck,
    Emote,
    Export,
    ExportRejected,
    Message,
    StreamActivity,
    Window,
    dump_deletion_check,
    dump_export,
    parse_deletion_check,
    parse_export,
    window_list,
)

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)
START = NOW - timedelta(days=3)


def _z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _export_dict() -> dict:
    """A small valid export, written by hand: one Moment window, made-up chat."""
    return {
        "format": "chatbrain-export/1",
        "exported_at": _z(NOW),
        "source": "vod",
        "windows": [
            {
                "window_id": "w-0001",
                "channel": "fakechan01",
                "stream_id": "fs-01",
                "anchor_kind": "moment",
                "anchor_at": _z(START + timedelta(seconds=30)),
                "start_at": _z(START),
                "end_at": _z(START + timedelta(seconds=120)),
                "language": "en",
                "text_expires_at": _z(NOW + timedelta(days=10)),
                "messages": [
                    {"message_id": "m1", "rel_ms": 31000, "speaker": "s1", "text": "KEKW", "emotes": [{"id": "fake-kekw", "name": "KEKW"}]},
                    {"message_id": "m2", "rel_ms": 32500, "speaker": "s2", "text": "LUL that was so bad", "emotes": [{"id": "fake-lul", "name": "LUL"}]},
                    {"message_id": "m3", "rel_ms": 40000, "speaker": "s3", "text": "@s1 same", "emotes": []},
                ],
            }
        ],
    }


def _raw(d: dict) -> bytes:
    return json.dumps(d).encode("utf-8")


def _msg(d: dict, i: int = 0) -> dict:
    return d["windows"][0]["messages"][i]


# --- the brief's tests -------------------------------------------------------


@pytest.mark.parametrize("field", ["anchor_at", "start_at", "end_at", "text_expires_at"])
def test_parse_rejects_naive_datetime(field):
    d = _export_dict()
    d["windows"][0][field] = d["windows"][0][field].removesuffix("Z")
    with pytest.raises(ExportRejected):
        parse_export(_raw(d))


def test_parse_rejects_naive_exported_at():
    d = _export_dict()
    d["exported_at"] = "2026-11-06T12:00:00"
    with pytest.raises(ExportRejected):
        parse_export(_raw(d))


IDENTITY_KEYS = ["login", "display_name", "commenter_login", "sender_hash", "user_id"]


@pytest.mark.parametrize("key", IDENTITY_KEYS)
@pytest.mark.parametrize("where", ["top", "window", "message", "emote"])
def test_parse_rejects_identity_fields(key, where):
    d = _export_dict()
    target = {
        "top": d,
        "window": d["windows"][0],
        "message": _msg(d),
        "emote": _msg(d)["emotes"][0],
    }[where]
    target[key] = "fake-value"
    with pytest.raises(ExportRejected, match="identity"):
        parse_export(_raw(d))


@pytest.mark.parametrize("key", ["displayName", "userId", "Sender-Hash", "LOGIN"])
def test_parse_rejects_identity_fields_in_any_spelling(key):
    d = _export_dict()
    _msg(d)[key] = "x"
    with pytest.raises(ExportRejected, match="identity"):
        parse_export(_raw(d))


def test_parse_rejects_wrong_window_shape():
    d = _export_dict()
    d["windows"][0]["end_at"] = _z(START + timedelta(seconds=121))
    with pytest.raises(ExportRejected):
        parse_export(_raw(d))
    d = _export_dict()
    d["windows"][0]["anchor_at"] = _z(START + timedelta(seconds=29))
    with pytest.raises(ExportRejected):
        parse_export(_raw(d))


def test_parse_rejects_bad_speaker_and_mentions():
    d = _export_dict()
    _msg(d)["speaker"] = "bob"
    with pytest.raises(ExportRejected):
        parse_export(_raw(d))

    d = _export_dict()
    _msg(d, 2)["text"] = "@bob hi"
    with pytest.raises(ExportRejected):
        parse_export(_raw(d))

    d = _export_dict()
    _msg(d, 2)["text"] = "@s3 and @user agree"
    e = parse_export(_raw(d))
    assert e.windows[0].messages[2].text == "@s3 and @user agree"


@pytest.mark.parametrize("speaker", ["s0", "s01", "S1", "s", "s1x", "user", ""])
def test_parse_rejects_malformed_speaker_numbers(speaker):
    d = _export_dict()
    _msg(d)["speaker"] = speaker
    with pytest.raises(ExportRejected):
        parse_export(_raw(d))


@pytest.mark.parametrize(
    "text",
    [
        "@s9 who never posted here",  # a speaker number must belong to someone who posted in the window
        "@S1 caps",
        "@users plural",
        "@s1x glued",
        "mail me at someone@example.com",  # every @<name> is a mention; the server rewrites it
        "@user_1 underscore",
    ],
)
def test_parse_rejects_unrewritten_mentions(text):
    d = _export_dict()
    _msg(d, 2)["text"] = text
    with pytest.raises(ExportRejected):
        parse_export(_raw(d))


@pytest.mark.parametrize("text", ["@s2 lol", "@user lol", "@s1, @s2!", "a @ sign alone", "@ö", "no mention"])
def test_parse_accepts_rewritten_mentions(text):
    d = _export_dict()
    _msg(d, 2)["text"] = text
    parse_export(_raw(d))


def test_parse_rejects_messages_under_one_hour_old():
    # The last message sits exactly 1 h before export: accepted.
    d = _export_dict()
    w = d["windows"][0]
    start = NOW - timedelta(hours=1) - timedelta(seconds=40)
    w["start_at"] = _z(start)
    w["anchor_at"] = _z(start + timedelta(seconds=30))
    w["end_at"] = _z(start + timedelta(seconds=120))
    parse_export(_raw(d))
    # One millisecond later: rejected.
    _msg(d, 2)["rel_ms"] = 40001
    with pytest.raises(ExportRejected):
        parse_export(_raw(d))


def test_window_list_has_no_text():
    e = parse_export(_raw(_export_dict()))
    rows = [(w.window_id, w.stream_id, w.start_at, w.end_at) for w in e.windows]
    out = json.loads(window_list(rows))
    assert out["format"] == "chatbrain-window-list/1"
    assert set(out) == {"format", "windows"}
    assert len(out["windows"]) == 1
    for item in out["windows"]:
        assert set(item) == {"window_id", "stream_id", "start_at", "end_at"}
    raw = window_list(rows).decode("utf-8")
    for m in e.windows[0].messages:
        assert m.text not in raw and m.message_id not in raw


# --- parse: shapes, types and contents ---------------------------------------


def test_parse_valid_export_builds_utc_objects():
    d = _export_dict()
    # A non-UTC offset is accepted and normalised to UTC.
    d["exported_at"] = NOW.astimezone(timezone(timedelta(hours=-7))).isoformat()
    e = parse_export(_raw(d))
    assert isinstance(e, Export)
    assert e.format == "chatbrain-export/1" and e.source == "vod"
    assert e.exported_at == NOW and e.exported_at.utcoffset() == timedelta(0)
    assert e.exported_at.tzinfo is timezone.utc
    w = e.windows[0]
    assert isinstance(w, Window)
    assert w.anchor_kind == "moment" and w.channel == "fakechan01"
    assert w.end_at - w.start_at == timedelta(seconds=120)
    assert w.anchor_at - w.start_at == timedelta(seconds=30)
    for dt in (w.anchor_at, w.start_at, w.end_at, w.text_expires_at):
        assert dt.tzinfo is timezone.utc
    assert w.messages[0] == Message("m1", 31000, "s1", "KEKW", (Emote("fake-kekw", "KEKW"),))
    assert w.messages[2].emotes == ()


def test_parse_sorts_messages_by_time_then_id():
    d = _export_dict()
    msgs = d["windows"][0]["messages"]
    msgs.reverse()
    msgs[0]["rel_ms"] = 31000  # m3 now ties with m1
    msgs[0]["text"] = "@s1 same"
    e = parse_export(_raw(d))
    assert [m.message_id for m in e.windows[0].messages] == ["m1", "m3", "m2"]


def test_parse_accepts_empty_window_and_empty_export():
    d = _export_dict()
    d["windows"][0]["messages"] = []
    assert parse_export(_raw(d)).windows[0].messages == ()
    d["windows"] = []
    assert parse_export(_raw(d)).windows == ()


def test_parse_accepts_ordinary_and_live():
    d = _export_dict()
    d["source"] = "live"
    d["windows"][0]["anchor_kind"] = "ordinary"
    e = parse_export(_raw(d))
    assert e.source == "live" and e.windows[0].anchor_kind == "ordinary"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(format="chatbrain-export/3"),
        lambda d: d.update(source="clip"),
        lambda d: d.update(extra=1),
        lambda d: d.pop("source"),
        lambda d: d.update(windows={}),
        lambda d: d["windows"][0].update(anchor_kind="spike"),
        lambda d: d["windows"][0].update(extra=1),
        lambda d: d["windows"][0].pop("language"),
        lambda d: d["windows"][0].update(language=""),
        lambda d: d["windows"][0].update(channel="Fake Chan"),
        lambda d: d["windows"][0].update(window_id=""),
        lambda d: d["windows"][0].update(window_id="w 1"),
        lambda d: d["windows"][0].update(stream_id=7),
        lambda d: d["windows"][0].update(anchor_at=1762430400),
        lambda d: d["windows"][0].update(anchor_at="2026-11-03"),
        lambda d: d["windows"][0].update(anchor_at="2026-11-03 12:00:30Z"),
        lambda d: d["windows"][0].update(messages=None),
        lambda d: _msg(d).update(rel_ms=-1),
        lambda d: _msg(d).update(rel_ms=120000),
        lambda d: _msg(d).update(rel_ms=31000.0),
        lambda d: _msg(d).update(rel_ms=True),
        lambda d: _msg(d).update(text=None),
        lambda d: _msg(d).update(extra=1),
        lambda d: _msg(d).pop("emotes"),
        lambda d: _msg(d).update(message_id="m2"),  # duplicate in the window
        lambda d: _msg(d).update(emotes=[{"id": "fake-kekw"}]),
        lambda d: _msg(d).update(emotes=[{"id": "fake kekw", "name": "KEKW"}]),
        lambda d: _msg(d).update(emotes=[{"id": "fake-kekw", "name": ""}]),
        lambda d: _msg(d).update(emotes=[{"id": "fake-kekw", "name": "KEKW", "extra": 1}]),
        lambda d: _msg(d).update(emotes="KEKW"),
        lambda d: d["windows"].append(copy.deepcopy(d["windows"][0])),  # duplicate window_id
    ],
)
def test_parse_rejects_bad_shapes(mutate):
    d = _export_dict()
    mutate(d)
    with pytest.raises(ExportRejected):
        parse_export(_raw(d))


def test_parse_rejects_text_kept_past_14_days_after_export():
    d = _export_dict()
    d["windows"][0]["text_expires_at"] = _z(NOW + timedelta(days=14))
    parse_export(_raw(d))
    d["windows"][0]["text_expires_at"] = _z(NOW + timedelta(days=14, seconds=1))
    with pytest.raises(ExportRejected):
        parse_export(_raw(d))


def test_parse_rejects_expiry_before_window_end():
    d = _export_dict()
    d["windows"][0]["text_expires_at"] = d["windows"][0]["end_at"]
    with pytest.raises(ExportRejected):
        parse_export(_raw(d))


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"not json",
        b"[]",
        b'{"format": "chatbrain-export/1", "format": "chatbrain-export/1"}',
        b"\xff\xfe",
        b'{"format": NaN}',
    ],
)
def test_parse_rejects_bad_json(raw):
    with pytest.raises(ExportRejected):
        parse_export(raw)


def test_parse_rejects_deep_nesting():
    with pytest.raises(ExportRejected):
        parse_export(b"[" * 100_000 + b"]" * 100_000)


def test_parse_rejects_times_out_of_range():
    d = _export_dict()
    d["exported_at"] = "9999-12-31T00:00:00Z"
    with pytest.raises(ExportRejected):
        parse_export(_raw(d))
    d = _export_dict()
    d["windows"][0]["anchor_at"] = "9999-12-31T23:59:59-01:00"
    with pytest.raises(ExportRejected):
        parse_export(_raw(d))
    c = {"format": "chatbrain-deletion-check/1", "checked_at": "9999-12-31T23:59:59-01:00", "windows": []}
    with pytest.raises(ExportRejected):
        parse_deletion_check(_raw(c))


def test_parse_rejects_overlong_integers():
    # Python 3.12 refuses to convert integers over 4300 digits with a plain ValueError;
    # both parsers must still raise ExportRejected, at the top level and nested.
    big = b"1" * 5000
    with pytest.raises(ExportRejected):
        parse_export(b'{"format": ' + big + b"}")
    raw = _raw(_export_dict()).replace(b'"rel_ms": 31000', b'"rel_ms": ' + big)
    with pytest.raises(ExportRejected):
        parse_export(raw)
    with pytest.raises(ExportRejected):
        parse_deletion_check(b'{"format": ' + big + b"}")
    raw = _raw(_check_dict()).replace(b'["m1", "m3"]', b'["m1", ' + big + b"]")
    with pytest.raises(ExportRejected):
        parse_deletion_check(raw)
    # A bare integer document and a negative literal hit the same limit.
    for doc in (big, b"-" + big, b'{"format": -' + big + b"}"):
        for parse in (parse_export, parse_deletion_check):
            with pytest.raises(ExportRejected) as info:
                parse(doc)
            assert type(info.value) is ExportRejected
            assert "4300" not in str(info.value) and "digits" not in str(info.value)


def test_parse_deletion_check_rejects_a_window_listed_twice():
    # Ruling C2: the PC applies a check only to the windows it lists, so each may appear once.
    d = _check_dict()
    d["windows"].append({"window_id": "w-0001", "present": ["m1", "m3"]})
    with pytest.raises(ExportRejected, match=r"windows\[2\]\.window_id: repeats an earlier window"):
        parse_deletion_check(_raw(d))


def test_parse_rejects_lone_surrogates():
    d = _export_dict()
    raw = _raw(d).replace(b'"LUL that was so bad"', b'"LUL \\ud800"')
    with pytest.raises(ExportRejected):
        parse_export(raw)


def test_parse_rejects_duplicate_keys_anywhere():
    raw = _raw(_export_dict()).replace(b'"speaker": "s1"', b'"speaker": "s1", "speaker": "s2"')
    with pytest.raises(ExportRejected):
        parse_export(raw)


def test_rejections_never_echo_chat_or_identity_values():
    secret = "zz_secret_chatter_name"
    cases = []
    d = _export_dict(); _msg(d)["speaker"] = secret; cases.append(d)
    d = _export_dict(); _msg(d, 2)["text"] = f"@{secret} hi"; cases.append(d)
    d = _export_dict(); _msg(d)["commenter_login"] = secret; cases.append(d)
    d = _export_dict(); d["windows"][0]["channel"] = secret.upper(); cases.append(d)
    d = _export_dict(); _msg(d)["emotes"] = [{"id": secret + " x", "name": "KEKW"}]; cases.append(d)
    # A map keyed by chatter login: unknown keys are never echoed in the path.
    d = _export_dict(); d["chatters"] = {secret: {"display_name": "Alice"}}; cases.append(d)
    d = _export_dict(); d["by_login"] = {secret: "\ud800"}; cases.append(d)
    d = _export_dict(); _msg(d)[secret] = {"user_id": 1}; cases.append(d)
    d = _export_dict(); _msg(d)[secret] = {"note": "\ud800"}; cases.append(d)
    for d in cases:
        with pytest.raises(ExportRejected) as info:
            parse_export(_raw(d))
        assert secret not in str(info.value)
        assert secret.upper() not in str(info.value)
    checks = []
    c = _check_dict(); c["windows"][0][secret] = {"sender_hash": "x"}; checks.append(c)
    c = _check_dict(); c["by_login"] = {secret: "\ud800"}; checks.append(c)
    for c in checks:
        with pytest.raises(ExportRejected) as info:
            parse_deletion_check(_raw(c))
        assert secret not in str(info.value)


def test_dump_export_round_trips():
    e = parse_export(_raw(_export_dict()))
    raw = dump_export(e)
    assert isinstance(raw, bytes)
    assert parse_export(raw) == e
    assert dump_export(parse_export(raw)) == raw


def test_dump_export_refuses_invalid_objects():
    e = parse_export(_raw(_export_dict()))
    w = e.windows[0]
    bad = Message("m9", 50000, "bob", "hi", ())
    broken = Export(e.format, e.exported_at, e.source, (Window(**{**w.__dict__, "messages": w.messages + (bad,)}),))
    with pytest.raises(ExportRejected):
        dump_export(broken)


def test_parse_accepts_utf8_bom():
    parse_export(b"\xef\xbb\xbf" + _raw(_export_dict()))


# --- deletion check ------------------------------------------------------------


def _check_dict() -> dict:
    return {
        "format": "chatbrain-deletion-check/1",
        "checked_at": _z(NOW),
        "windows": [
            {"window_id": "w-0001", "present": ["m1", "m3"]},
            {"window_id": "w-0002", "present": []},
        ],
    }


def test_parse_deletion_check():
    c = parse_deletion_check(_raw(_check_dict()))
    assert isinstance(c, DeletionCheck)
    assert c.checked_at == NOW and c.checked_at.tzinfo is timezone.utc
    assert c.present == {"w-0001": frozenset({"m1", "m3"}), "w-0002": frozenset()}


def test_deletion_check_round_trips():
    c = parse_deletion_check(_raw(_check_dict()))
    assert parse_deletion_check(dump_deletion_check(c)) == c


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(format="chatbrain-export/1"),
        lambda d: d.update(checked_at="2026-11-06T12:00:00"),
        lambda d: d.update(extra=1),
        lambda d: d.pop("windows"),
        lambda d: d["windows"][0].update(sender_hash="x"),
        lambda d: d["windows"][0].update(present="m1"),
        lambda d: d["windows"][0].update(present=["m1", "m1"]),
        lambda d: d["windows"][0].update(present=[1]),
        lambda d: d["windows"][0].update(text="hi"),
        lambda d: d["windows"].append({"window_id": "w-0001", "present": []}),
    ],
)
def test_parse_deletion_check_rejects_bad_input(mutate):
    d = _check_dict()
    mutate(d)
    with pytest.raises(ExportRejected):
        parse_deletion_check(_raw(d))


# --- window list ------------------------------------------------------------------


def test_window_list_format_and_utc():
    start = START.astimezone(timezone(timedelta(hours=2)))
    out = json.loads(window_list([("w-0001", "fs-01", start, start + timedelta(seconds=120))]))
    assert out["windows"] == [
        {"window_id": "w-0001", "stream_id": "fs-01", "start_at": "2026-11-03T12:00:00.000000Z", "end_at": "2026-11-03T12:02:00.000000Z"}
    ]


def test_window_list_rejects_naive_or_duplicate_rows():
    naive = datetime(2026, 11, 3, 12, 0)
    with pytest.raises(ExportRejected):
        window_list([("w-0001", "fs-01", naive, naive + timedelta(seconds=120))])
    row = ("w-0001", "fs-01", START, START + timedelta(seconds=120))
    with pytest.raises(ExportRejected):
        window_list([row, row])
    with pytest.raises(ExportRejected):
        window_list([("w-0001", "fs-01", START, START + timedelta(seconds=60))])


def test_window_list_empty():
    assert json.loads(window_list([])) == {"format": "chatbrain-window-list/1", "windows": []}


# --- format /2: stream activity -----------------------------------------------------


def _v2_dict() -> dict:
    d = _export_dict()
    d["format"] = "chatbrain-export/2"
    d["streams"] = [
        {
            "stream_id": "fs-01",
            "channel": "fakechan01",
            "started_at": _z(START),
            "ended_at": _z(START + timedelta(seconds=45)),  # 4.5 bins -> 5
            "title": "fake stream one",
            "bin_s": 10,
            "chatters": [3, 5, None, None, 0],
        },
        {
            "stream_id": "fs-02",
            "channel": "fakechan02",
            "started_at": _z(START),
            "ended_at": _z(START + timedelta(seconds=30)),
            "title": None,
            "bin_s": 10,
            "chatters": [1, 2, 100_000],
        },
    ]
    return d


def test_v1_parses_with_no_streams():
    e = parse_export(_raw(_export_dict()))
    assert e.format == "chatbrain-export/1"
    assert e.streams == ()


def test_v2_parses_streams():
    e = parse_export(_raw(_v2_dict()))
    assert e.format == "chatbrain-export/2"
    assert e.streams[0] == StreamActivity(
        stream_id="fs-01",
        channel="fakechan01",
        started_at=START,
        ended_at=START + timedelta(seconds=45),
        title="fake stream one",
        bin_s=10,
        chatters=(3, 5, None, None, 0),
    )
    assert e.streams[1].title is None


def test_v2_round_trip():
    e = parse_export(_raw(_v2_dict()))
    assert len(e.streams) == 2
    assert parse_export(dump_export(e)) == e


def test_v2_empty_streams_ok():
    d = _v2_dict()
    d["streams"] = []
    assert parse_export(_raw(d)).streams == ()


def test_v2_needs_streams_key():
    d = _v2_dict()
    del d["streams"]
    with pytest.raises(ExportRejected, match=r"export: missing keys \['streams'\]"):
        parse_export(_raw(d))


def test_v1_rejects_streams_key():
    d = _v2_dict()
    d["format"] = "chatbrain-export/1"
    with pytest.raises(ExportRejected, match=r"export: missing keys \[\], 1 unknown key"):
        parse_export(_raw(d))


def test_v1_dump_refuses_streams():
    e = parse_export(_raw(_v2_dict()))
    v1 = Export(format="chatbrain-export/1", exported_at=e.exported_at, source=e.source, windows=e.windows, streams=e.streams)
    with pytest.raises(ExportRejected, match="export.streams"):
        dump_export(v1)


def _set(update):
    def apply(d):
        update(d["streams"][0])
        return d

    return apply


def _dup_id(d):
    d["streams"][1]["stream_id"] = "fs-01"
    return d


def _long(d):
    s = d["streams"][0]
    s["ended_at"] = _z(START + timedelta(seconds=172_810))
    s["chatters"] = [1] * 17_281
    return d


STREAM_CASES = [
    ("bin_s", _set(lambda s: s.update(bin_s=5)), "streams[0].bin_s"),
    ("short", _set(lambda s: s.update(chatters=[1, 2, 3, 4])), "streams[0].chatters"),
    ("long_by_one", _set(lambda s: s.update(chatters=[1] * 6)), "streams[0].chatters"),
    ("negative", _set(lambda s: s.update(chatters=[-1, 2, 3, 4, 5])), "streams[0].chatters[0]"),
    ("over_cap", _set(lambda s: s.update(chatters=[1, 2, 100_001, 4, 5])), "streams[0].chatters[2]"),
    ("bool", _set(lambda s: s.update(chatters=[1, 2, 3, 4, True])), "streams[0].chatters[4]"),
    ("float", _set(lambda s: s.update(chatters=[1, 1.5, 3, 4, 5])), "streams[0].chatters[1]"),
    ("ends_before_start", _set(lambda s: s.update(ended_at=_z(START - timedelta(seconds=10)))), "streams[0].ended_at"),
    ("ends_after_export", _set(lambda s: s.update(ended_at=_z(NOW + timedelta(seconds=10)))), "streams[0].ended_at"),
    ("repeat_id", _dup_id, "streams[1].stream_id"),
    ("title_newline", _set(lambda s: s.update(title="a\nb")), "streams[0].title"),
    ("title_141", _set(lambda s: s.update(title="x" * 141)), "streams[0].title"),
    ("title_cr", _set(lambda s: s.update(title="a\rb")), "streams[0].title"),
    ("title_line_sep", _set(lambda s: s.update(title="a\u2028b")), "streams[0].title"),
    ("title_para_sep", _set(lambda s: s.update(title="a\u2029b")), "streams[0].title"),
    ("title_nel", _set(lambda s: s.update(title="a\u0085b")), "streams[0].title"),
    ("title_escape", _set(lambda s: s.update(title="a\x1bb")), "streams[0].title"),
    ("title_nul", _set(lambda s: s.update(title="a\x00b")), "streams[0].title"),
    ("title_blank", _set(lambda s: s.update(title="   ")), "streams[0].title"),
    ("extra_key", _set(lambda s: s.update(vod_id="v1")), "streams[0]"),
    ("17281_bins", _long, "streams[0].chatters"),
]


@pytest.mark.parametrize("mutate,field", [(c[1], c[2]) for c in STREAM_CASES], ids=[c[0] for c in STREAM_CASES])
def test_stream_activity_rejects(mutate, field):
    d = mutate(_v2_dict())
    with pytest.raises(ExportRejected) as err:
        parse_export(_raw(d))
    assert f"export.{field}" in str(err.value)


def test_stream_identity_refused_still():
    d = _v2_dict()
    d["streams"][0]["login"] = "x"
    with pytest.raises(ExportRejected, match="identity field"):
        parse_export(_raw(d))


def test_17280_bins_ok():
    d = _v2_dict()
    s = d["streams"][0]
    s["ended_at"] = _z(START + timedelta(seconds=172_800))
    s["chatters"] = [1] * 17_280
    assert len(parse_export(_raw(d)).streams[0].chatters) == 17_280


def test_title_of_140_characters_ok():
    d = _v2_dict()
    d["streams"][0]["title"] = "x" * 140
    assert parse_export(_raw(d)).streams[0].title == "x" * 140


def test_ended_at_equal_to_started_at_is_refused():
    d = _v2_dict()
    d["streams"][0]["ended_at"] = d["streams"][0]["started_at"]
    with pytest.raises(ExportRejected, match=r"export\.streams\[0\]\.ended_at: must be after started_at"):
        parse_export(_raw(d))


def test_ended_at_equal_to_exported_at_ok():
    d = _v2_dict()
    s = d["streams"][0]
    s["started_at"] = _z(NOW - timedelta(seconds=100))
    s["ended_at"] = d["exported_at"]
    s["chatters"] = [1] * 10
    assert parse_export(_raw(d)).streams[0].ended_at == NOW


def test_v1_file_with_a_streams_key_keeps_its_v1_messages():
    # Expected text is the v1 parser's message for these files (/2 key names were not schema keys).
    d = _export_dict()
    d["streams"] = [{"login": "x"}]
    with pytest.raises(ExportRejected) as err:
        parse_export(_raw(d))
    assert str(err.value) == "export.?[0]: identity field login is never exported"
    d = _export_dict()
    d["streams"] = ["\ud800"]
    with pytest.raises(ExportRejected) as err:
        parse_export(_raw(d))
    assert str(err.value) == "export.?[0]: a string is not valid Unicode"
