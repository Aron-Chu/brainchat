import hashlib
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import pytest

from chatbrain import fake, scheme
from chatbrain.export_format import EXPORT_FORMAT, EXPORT_FORMAT_V2, parse_deletion_check, parse_export

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)


def test_fake_is_made_up_and_deterministic():
    a = fake.make_export(7, 40, 10, NOW); assert a == fake.make_export(7, 40, 10, NOW)
    e = parse_export(a); assert {w.channel for w in e.windows} <= {f"fakechan{i:02d}" for i in range(1, 13)}
    assert sum(w.anchor_kind == "ordinary" for w in e.windows) == 10


def test_fake_counts_and_seed_change():
    e = parse_export(fake.make_export(7, 40, 10, NOW))
    assert len(e.windows) == 50
    assert sum(w.anchor_kind == "moment" for w in e.windows) == 40
    assert e.exported_at == NOW and e.source == "vod"
    assert fake.make_export(8, 40, 10, NOW) != fake.make_export(7, 40, 10, NOW)
    ids_7 = {w.window_id for w in e.windows}
    ids_8 = {w.window_id for w in parse_export(fake.make_export(8, 40, 10, NOW)).windows}
    assert not ids_7 & ids_8  # two seeds can share one store


def test_fake_uses_made_up_names_only():
    e = parse_export(fake.make_export(11, 120, 30, NOW))
    assert all(w.language == "en" for w in e.windows)
    for w in e.windows:
        for m in w.messages:
            assert re.fullmatch(r"s[1-9]\d*", m.speaker)
            for em in m.emotes:
                assert em.id == "fake-" + em.name.lower()
                assert em.name in m.text.split()
            for name in re.findall(r"@(\w+)", m.text):
                assert re.fullmatch(r"s[1-9]\d*|user", name)


def test_fake_speakers_numbered_by_first_post():
    e = parse_export(fake.make_export(5, 60, 15, NOW))
    for w in e.windows:
        order = list(dict.fromkeys(m.speaker for m in w.messages))
        assert order == [f"s{i}" for i in range(1, len(order) + 1)]


def test_fake_ordinary_windows_five_minutes_from_moments():
    e = parse_export(fake.make_export(3, 300, 75, NOW))
    moments = defaultdict(list)
    for w in e.windows:
        if w.anchor_kind == "moment":
            moments[w.stream_id].append(w.anchor_at)
    ordinary = [w for w in e.windows if w.anchor_kind == "ordinary"]
    assert len(ordinary) == 75
    for w in ordinary:
        assert moments[w.stream_id], "ordinary windows come from streams that have Moments"
        for m in moments[w.stream_id]:
            assert abs((w.anchor_at - m).total_seconds()) >= 300


def test_fake_streams_end_1_to_13_days_before_now():
    e = parse_export(fake.make_export(7, 480, 120, NOW))
    ends = {}
    for w in e.windows:
        stream_end = w.text_expires_at - timedelta(days=14)
        assert ends.setdefault(w.stream_id, stream_end) == stream_end  # one end per stream
        assert NOW - timedelta(days=13) <= stream_end <= NOW - timedelta(days=1)
        assert w.end_at <= stream_end
    assert len(ends) >= 2


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 1234])
def test_fake_later_clock_expires_some_windows_not_all(seed):
    e = parse_export(fake.make_export(seed, 40, 10, NOW))
    later = NOW + timedelta(days=5)
    expired = [w for w in e.windows if w.text_expires_at <= later]
    assert 0 < len(expired) < len(e.windows)


def test_fake_windows_have_moment_shape_and_minute_anchors():
    e = parse_export(fake.make_export(9, 50, 12, NOW))
    for w in e.windows:
        assert w.anchor_at.second == 0 and w.anchor_at.microsecond == 0
        assert w.anchor_at - w.start_at == timedelta(seconds=30)
        assert w.end_at - w.start_at == timedelta(seconds=120)


def test_fake_includes_near_empty_windows():
    e = parse_export(fake.make_export(7, 480, 120, NOW))
    sizes = [len(w.messages) for w in e.windows]
    assert any(n == 0 for n in sizes) and any(n == 1 for n in sizes)
    assert max(sizes) >= 20


def test_fake_moment_lines_follow_the_truth():
    raw = fake.make_export(7, 200, 50, NOW)
    t = fake.truth(raw)
    e = parse_export(raw)
    laughing = [w for w in e.windows if t[w.window_id]["doing"] == ["laughing"] and w.anchor_kind == "moment"]
    assert laughing
    for w in laughing:
        assert sum("fake-kekw" in {em.id for em in m.emotes} or "LUL" in m.text for m in w.messages) >= len(w.messages) // 3


def test_fake_templates_cover_the_scheme_in_order():
    s = scheme.load()
    q = {x.key: x for x in s.questions}
    assert tuple(fake.DOING_LINES) == q["doing"].values
    assert set(fake.AIMED_LINES) == set(q["aimed_at"].values)
    assert set(fake.FEELING_LINES) == set(q["feeling"].values)
    assert set(fake.TONE_LINES) == set(q["tone"].values) | {q["tone"].blank_value}


def test_truth_shape_matches_scheme():
    s = scheme.load()
    q = {x.key: x for x in s.questions}
    raw = fake.make_export(7, 200, 50, NOW)
    t = fake.truth(raw)
    e = parse_export(raw)
    assert set(t) == {w.window_id for w in e.windows}
    seen = defaultdict(set)
    for wid, answers in t.items():
        assert set(answers) == {"doing", "aimed_at", "feeling", "tone"}
        assert isinstance(answers["doing"], list) and 1 <= len(answers["doing"]) <= 2
        assert len(set(answers["doing"])) == len(answers["doing"])
        assert set(answers["doing"]) <= set(q["doing"].values)
        assert answers["aimed_at"] in q["aimed_at"].values
        assert answers["feeling"] in q["feeling"].values
        assert answers["tone"] in q["tone"].values + (q["tone"].blank_value,)
        for key, v in answers.items():
            seen[key].update(v if isinstance(v, list) else [v])
    # The made-up data reaches every doing value and the rare tones.
    assert seen["doing"] == set(q["doing"].values)
    assert {"sarcastic", "hostile", "no_clear_tone"} <= seen["tone"]
    assert {"tech_problem", "guest"} <= seen["aimed_at"]
    assert t == fake.truth(raw)


def test_truth_is_json_ready():
    raw = fake.make_export(7, 20, 5, NOW)
    t = fake.truth(raw)
    assert json.loads(json.dumps(t)) == t


def test_truth_refuses_a_real_looking_export():
    raw = fake.make_export(7, 4, 1, NOW).replace(b"fakechan", b"realchan")
    with pytest.raises(ValueError):
        fake.truth(raw)


def test_make_deletion_check():
    raw = fake.make_export(7, 20, 5, NOW)
    e = parse_export(raw)
    w = next(w for w in e.windows if len(w.messages) >= 3)
    drop = {w.messages[0].message_id, w.messages[2].message_id}
    checked = NOW + timedelta(days=1)
    c = parse_deletion_check(fake.make_deletion_check(raw, drop, checked))
    assert c.checked_at == checked
    assert set(c.present) == {x.window_id for x in e.windows}
    for x in e.windows:
        want = {m.message_id for m in x.messages} - drop
        assert c.present[x.window_id] == want
    assert fake.make_deletion_check(raw, drop, checked) == fake.make_deletion_check(raw, drop, checked)


def test_make_deletion_check_refuses_unknown_ids_and_naive_time():
    raw = fake.make_export(7, 4, 1, NOW)
    with pytest.raises(ValueError):
        fake.make_deletion_check(raw, {"no-such-message"}, NOW)
    with pytest.raises(ValueError):
        fake.make_deletion_check(raw, set(), datetime(2026, 11, 6, 12, 0))


def test_make_export_argument_checks():
    with pytest.raises(ValueError):
        fake.make_export(1, 4, 1, datetime(2026, 11, 6, 12, 0))
    with pytest.raises(ValueError):
        fake.make_export(1, -1, 1, NOW)
    with pytest.raises(ValueError):
        fake.make_export(1, 4, 1, NOW, channels=0)
    with pytest.raises(ValueError):
        fake.make_export(1, 4, 1, NOW, channels=100)
    with pytest.raises(ValueError):
        fake.make_export(1, 0, 3, NOW)  # ordinary windows need a stream with Moments


def test_make_export_small_and_empty():
    assert parse_export(fake.make_export(1, 0, 0, NOW)).windows == ()
    e = parse_export(fake.make_export(1, 1, 0, NOW, channels=1))
    assert len(e.windows) == 1 and e.windows[0].channel == "fakechan01"


@pytest.mark.parametrize("seed, n_moment, n_ordinary, channels", [(3, 300, 75, 12), (7, 40, 10, 12), (1, 6, 9, 3), (5, 2, 5, 1)])
def test_fake_ordinary_windows_share_a_stream_with_a_moment(seed, n_moment, n_ordinary, channels):
    # Spec section 5: ordinary windows come "from the same streams" as the Moments.
    e = parse_export(fake.make_export(seed, n_moment, n_ordinary, NOW, channels=channels))
    moment_streams = {w.stream_id for w in e.windows if w.anchor_kind == "moment"}
    ordinary = [w for w in e.windows if w.anchor_kind == "ordinary"]
    assert len(ordinary) == n_ordinary
    for w in ordinary:
        assert w.stream_id in moment_streams


# --- markers (ruling C9) ----------------------------------------------------------------


def test_marker_is_a_made_up_token():
    m = fake.marker("fake7-w0001")
    assert re.fullmatch(r"zq[0-9a-f]{12}", m)
    assert m == fake.marker("fake7-w0001")
    assert m != fake.marker("fake7-w0002")
    assert m not in fake.EMOTE_NAMES and "@" not in m and " " not in m


def test_markers_are_unique_and_sit_in_every_window_with_two_or_more_messages():
    raw = fake.make_export(7, 480, 120, NOW)
    e = parse_export(raw)
    markers = [fake.marker(w.window_id) for w in e.windows]
    assert len(set(markers)) == len(markers)
    all_markers = set(markers)
    for w in e.windows:
        texts = [m.text for m in w.messages]
        own = fake.marker(w.window_id)
        # A window has 0 or 1 messages (quiet, no marker) or its own lines plus the
        # marker, so never exactly 2.
        assert len(w.messages) != 2
        if len(w.messages) >= 2:
            assert texts.count(own) == 1
        else:
            assert own not in texts  # quiet windows stay near-empty
        # No other window's marker leaks in, and the marker line is plain text.
        assert not (all_markers - {own}) & set(texts)
        for m in w.messages:
            if m.text == own:
                assert m.emotes == ()
    sizes = [len(w.messages) for w in e.windows]
    assert 0 in sizes and 1 in sizes and any(n >= 3 for n in sizes)


def test_marker_does_not_change_truth_or_posters():
    raw = fake.make_export(7, 200, 50, NOW)
    e = parse_export(raw)
    t = fake.truth(raw)
    assert set(t) == {w.window_id for w in e.windows}
    for w in e.windows:
        own = fake.marker(w.window_id)
        marked = [m for m in w.messages if m.text == own]
        if marked:
            # The marker comes from someone who also posts an ordinary line in the window.
            others = {m.speaker for m in w.messages if m.text != own}
            assert marked[0].speaker in others


# --- per-stream activity (Slice 1, Task 2) ---------------------------------------------------------


def _bin(s, at) -> int:
    return int((at - s.started_at) // timedelta(seconds=10))


@pytest.mark.parametrize("seed, n_moment, n_ordinary, channels", [(7, 40, 10, 12), (3, 12, 16, 1), (11, 120, 30, 12)])
def test_fake_activity_covers_every_moment(seed, n_moment, n_ordinary, channels):
    e = parse_export(fake.make_export(seed, n_moment, n_ordinary, NOW, channels=channels))
    streams = {s.stream_id: s for s in e.streams}
    assert set(streams) == {w.stream_id for w in e.windows}  # one activity line per made-up stream
    for w in e.windows:
        if w.anchor_kind != "moment":
            continue
        s = streams[w.stream_id]
        anchor = s.chatters[_bin(s, w.anchor_at)]
        recorded = [c for c in s.chatters if c is not None]
        assert anchor is not None
        assert anchor > statistics.median(recorded)


def test_fake_is_deterministic_v2():
    raw = fake.make_export(7, 40, 10, NOW)
    assert raw == fake.make_export(7, 40, 10, NOW)
    assert raw != fake.make_export(8, 40, 10, NOW)
    e = parse_export(raw)
    assert e.format == EXPORT_FORMAT_V2
    by_stream = defaultdict(list)
    for w in e.windows:
        by_stream[w.stream_id].append(w)
    assert [s.stream_id for s in e.streams] == sorted(by_stream)
    for s in e.streams:
        ws = by_stream[s.stream_id]
        k = int(s.stream_id.rsplit("-s", 1)[1])
        assert s.title == f"Made-up stream {k}"
        assert {w.channel for w in ws} == {s.channel}
        assert s.bin_s == 10
        assert s.started_at.second == 0 and s.started_at.microsecond == 0  # bins sit on 10 s
        assert s.started_at <= min(w.start_at for w in ws)
        assert s.ended_at >= max(w.end_at for w in ws)
        assert s.ended_at == ws[0].text_expires_at - fake.TEXT_LIFETIME  # the made-up stream's own span
        assert s.ended_at <= e.exported_at
        assert len(s.chatters) == (s.ended_at - s.started_at) // timedelta(seconds=10)
        assert all(c is None or (type(c) is int and c >= 0) for c in s.chatters)
    assert fake.truth(raw) == fake.truth(fake.make_export(7, 40, 10, NOW, activity=False))


# Bytes of P0's /1 generator: activity=False must give exactly these.
P0_SHA256 = {
    (7, 40, 10, 12): "00a304599c928134c60a1c6ff1cc0223e9ca4cf35ce81906c685e9d8add334eb",
    (3, 12, 16, 12): "978502f6141dab4f37c5ca0fc55ff3b1a0ad6a425f31daebaff17e8224b9f99b",
    (3, 12, 16, 1): "074b54388bebec6bda8fa4ca518914361d8efee3c97de07b7e12dfb70906a56a",
}


@pytest.mark.parametrize("seed, n_moment, n_ordinary, channels", sorted(P0_SHA256))
def test_fake_without_activity_is_p0s_v1(seed, n_moment, n_ordinary, channels):
    v1 = fake.make_export(seed, n_moment, n_ordinary, NOW, channels=channels, activity=False)
    assert hashlib.sha256(v1).hexdigest() == P0_SHA256[(seed, n_moment, n_ordinary, channels)]
    e1 = parse_export(v1)
    assert e1.format == EXPORT_FORMAT and e1.streams == ()
    e2 = parse_export(fake.make_export(seed, n_moment, n_ordinary, NOW, channels=channels))
    assert e2.windows == e1.windows  # activity changes no window


def test_fake_activity_gaps():
    e = parse_export(fake.make_export(11, 120, 30, NOW))
    anchors = defaultdict(set)
    for w in e.windows:
        if w.anchor_kind == "moment":
            anchors[w.stream_id].add(w.anchor_at)
    with_gap = []
    for s in e.streams:
        runs, start = [], None
        for i, c in enumerate((*s.chatters, 0)):
            if c is None and start is None:
                start = i
            elif c is not None and start is not None:
                runs.append((start, i))
                start = None
        assert len(runs) <= 1
        for a, b in runs:
            assert 12 <= b - a <= 36 and (b - a) % 6 == 0  # one run of 2-6 whole minutes
            assert not any(a <= _bin(s, t) < b for t in anchors[s.stream_id])  # never over a Moment anchor
            with_gap.append(s.stream_id)
    assert with_gap == [s.stream_id for i, s in enumerate(e.streams) if i % 4 == 0]  # 1 in every 4 streams
