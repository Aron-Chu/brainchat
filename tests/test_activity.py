"""Pure activity-line functions: line, gaps, peaks and chip numbering. Made-up data only."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pytest

from chatbrain.export_format import StreamActivity
from chatbrain.page.activity import (
    Chip,
    chips,
    coverage,
    line,
    peak_near,
    ratio_text,
    usual_pace,
)

T0 = datetime(2026, 11, 1, 12, 0, 0, tzinfo=timezone.utc)
TIMES = "×"


def act(chatters, *, bin_s: int = 10, stream_id: str = "fakestream01") -> StreamActivity:
    return StreamActivity(
        stream_id=stream_id,
        channel="fakechan01",
        started_at=T0,
        ended_at=T0 + timedelta(seconds=bin_s * len(chatters)),
        title=None,
        bin_s=bin_s,
        chatters=tuple(chatters),
    )


def points(d: str) -> list[tuple[float, float]]:
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", d.replace(" h0", ""))
    return [(float(nums[i]), float(nums[i + 1])) for i in range(0, len(nums), 2)]


def test_line_sqrt_height():
    ln = line(act([0, 1, 4, 16]))
    assert ln.width_s == 40
    assert ln.height == 48
    assert ln.peak == 16
    assert [y for _, y in points(ln.d_line)] == [48, 37, 26, 4]
    assert [x for x, _ in points(ln.d_line)] == [5, 15, 25, 35]
    assert ln.d_line.startswith("M5 48 L15 37")
    assert ln.gaps == ()


def test_line_area_closes_to_baseline():
    ln = line(act([1, 4]))
    assert ln.d_area.startswith("M5 48 L5 ")
    assert ln.d_area.endswith("L15 48 Z")


def test_line_breaks_on_null():
    ln = line(act([1, None, None, 1]))
    assert ln.d_line.count("M") == 2
    assert ln.gaps == ((10, 30),)
    assert ln.d_area.count("M") == 2
    assert ln.d_area.count("Z") == 2


def test_downsample_keeps_peak():
    bins = [1] * 10_000
    bins[7777] = 400
    ln = line(act(bins))
    assert ln.peak == 400
    pts = points(ln.d_line)
    assert len(pts) == 715
    assert len(pts) <= 720
    assert min(y for _, y in pts) == 4
    assert ln.width_s == 100_000


def test_all_null_stream():
    ln = line(act([None] * 6))
    assert ln.d_line == ""
    assert ln.d_area == ""
    assert ln.gaps == ((0, 60),)
    assert ln.peak == 0


def test_all_zero_stream_sits_on_baseline():
    ln = line(act([0, 0, 0]))
    assert ln.peak == 0
    assert [y for _, y in points(ln.d_line)] == [48, 48, 48]


def test_group_is_null_only_if_all_bins_are():
    # 4 bins, max_points 2: groups (1, None) and (None, None)
    ln = line(act([1, None, None, None]), max_points=2)
    assert len(points(ln.d_line)) == 1
    assert ln.gaps == ((10, 40),)


def test_usual_pace_median_and_minimum():
    assert usual_pace([act([3] * 29 + [0, None])]) is None
    assert usual_pace([act([3] * 30)]) == 3
    # the median spans all the streams given; zeros and nulls are ignored
    a = act([1] * 15 + [0] * 5)
    b = act([5] * 15 + [None] * 5, stream_id="fakestream02")
    assert usual_pace([a, b]) == 3.0
    assert usual_pace([]) is None


def test_peak_near_window():
    a = act([1, 2, 3, 9, 4, 5, 6, 7, 8, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    # anchor at 60 s: window [30, 150) covers bins 3..14
    assert peak_near(a, T0 + timedelta(seconds=60)) == 9
    # anchor at 100 s: window [70, 190) covers bins 7..18; the 9 at bin 3 is out
    assert peak_near(a, T0 + timedelta(seconds=100)) == 8
    # a window that only partly overlaps the stream still counts the overlap
    assert peak_near(a, T0 + timedelta(seconds=-85)) == 1


def test_peak_near_in_gap_is_none():
    a = act([5, 5, 5] + [None] * 12 + [5] * 5)  # bins 3..14 are a gap
    assert peak_near(a, T0 + timedelta(seconds=60)) is None
    # a gap that covers the anchor but not the whole span: still not recorded
    b = act([5, 5, 5, 9, 9, 9] + [None] * 4 + [7] * 10)  # anchor 80 s = bin 8 (null); span bins 5..16
    assert peak_near(b, T0 + timedelta(seconds=80)) is None
    # anchor bin recorded, a later part of the span null: the max of the recorded bins
    c = act([2, 2, 2, 2, 2, 2, 4, 3, None, None, None, None, None, None, None, None, None, None])
    assert peak_near(c, T0 + timedelta(seconds=60)) == 4  # anchor bin 6 is recorded; bins 8.. null


def test_peak_near_outside_stream_is_none():
    a = act([1, 2, 3])
    assert peak_near(a, T0 + timedelta(hours=2)) is None
    assert peak_near(a, T0 - timedelta(hours=2)) is None


def test_coverage_kinds():
    a = act([5, 5, None, None, 5, 5])  # bins 2..3 are null: not recorded
    assert coverage(a, T0 + timedelta(seconds=5)) == "known"
    assert coverage(a, T0 + timedelta(seconds=25)) == "gap"
    assert coverage(a, T0 + timedelta(seconds=39.9)) == "gap"
    assert coverage(a, T0 + timedelta(seconds=40)) == "known"
    assert coverage(a, T0 + timedelta(seconds=59.9)) == "known"
    # outside the counts' span: the anchor bin is not 0..n-1, so nobody knows
    assert coverage(a, T0 + timedelta(seconds=60)) == "unknown"
    assert coverage(a, T0 - timedelta(seconds=0.1)) == "unknown"
    assert coverage(a, T0 + timedelta(hours=2)) == "unknown"
    # a partial overlap keeps its number (peak_near unchanged) but its anchor is still unknown
    b = act([1, 2, 3])
    assert peak_near(b, T0 - timedelta(seconds=20)) == 3  # span [-50 s, 70 s) covers all three bins
    assert coverage(b, T0 - timedelta(seconds=20)) == "unknown"
    # zero is a count, not a gap
    assert coverage(act([0, 0]), T0 + timedelta(seconds=10)) == "known"


def test_ratio_text():
    assert ratio_text(2.44) == "2.4" + TIMES
    assert ratio_text(3.0) == "3" + TIMES
    assert ratio_text(9.96) == "10" + TIMES
    assert ratio_text(12.4) == "12" + TIMES


def test_chip_shapes():
    moments = [
        ("w1", T0 + timedelta(seconds=10), True, "cheering"),
        ("w2", T0 + timedelta(seconds=20), False, None),
        ("w3", T0 + timedelta(seconds=30), False, "cheering"),
        ("w4", T0 + timedelta(seconds=40), True, None),
    ]
    out = chips(moments, T0, T0 + timedelta(seconds=100))
    assert [c.shape for c in out] == ["solid", "unsure", "ring", "solid"]
    assert [c.n for c in out] == [1, 2, 3, 4]
    assert [c.x_pct for c in out] == pytest.approx([10, 20, 30, 40])
    assert out[0] == Chip(n=1, window_id="w1", x_pct=pytest.approx(10), shape="solid", doing="cheering")
    assert out[1].doing is None


def test_chip_x_pct_is_clamped():
    end = T0 + timedelta(seconds=100)
    out = chips(
        [("w1", T0 - timedelta(seconds=5), False, "a"), ("w2", end + timedelta(seconds=5), False, "a")],
        T0,
        end,
    )
    assert [c.x_pct for c in out] == [0, 100]


def test_chip_numbers_stable_under_filter():
    anchor = T0 + timedelta(seconds=500)
    moments = [
        ("w-late", anchor + timedelta(seconds=2), False, "cheering"),
        ("w-early", anchor, False, "arguing"),
        ("w-first", T0 + timedelta(seconds=5), True, None),
    ]
    out = chips(moments, T0, T0 + timedelta(seconds=1000))
    assert [(c.n, c.window_id) for c in out] == [(1, "w-first"), (2, "w-early"), (3, "w-late")]
    # numbering belongs to the stream's whole list, not to any shown subset
    shown = [c for c in out if c.doing == "cheering"]
    assert [c.n for c in shown] == [3]
    # two Moments 2 s apart get n 1 and 2, in anchor order
    pair = chips(
        [("wb", T0 + timedelta(seconds=102), False, "a"), ("wa", T0 + timedelta(seconds=100), False, "b")],
        T0,
        T0 + timedelta(seconds=1000),
    )
    assert [(c.n, c.window_id) for c in pair] == [(1, "wa"), (2, "wb")]


def test_chip_ties_break_on_window_id():
    out = chips([("wb", T0, False, "a"), ("wa", T0, False, "a")], T0, T0 + timedelta(seconds=10))
    assert [(c.n, c.window_id) for c in out] == [(1, "wa"), (2, "wb")]


def test_chip_zero_length_stream():
    out = chips([("w1", T0, False, "a")], T0, T0)
    assert out[0].x_pct == 0


def test_one_point_stretch_draws_a_dot():
    ln = line(act([None, 5, None]))
    assert ln.d_line == "M15 4 h0"
