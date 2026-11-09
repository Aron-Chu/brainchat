"""The activity line under a stream: people chatting per bin, drawn as a path.

Pure functions over ``StreamActivity`` counts, standard library only (no FastAPI, no
Jinja), so they stay fast to test. Counts are numbers: no chat text, no speakers, no IDs.

Bin ``i`` covers ``[started_at + bin_s * i, started_at + bin_s * (i + 1))``. A point sits
at its group's centre, in seconds from ``started_at``. ``None`` bins were not recorded;
they break the line and become gaps, never zeros.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from ..export_format import StreamActivity

TIMES = "×"
MIN_PACE_BINS = 30


@dataclass(frozen=True)
class Line:
    width_s: int
    height: int
    d_line: str
    d_area: str
    gaps: tuple[tuple[int, int], ...]  # (start_s, end_s) from started_at, from the raw bins
    peak: int


@dataclass(frozen=True)
class Chip:
    n: int
    window_id: str
    x_pct: float
    shape: str  # "solid", "ring" or "unsure"
    doing: str | None


def _num(v: float) -> str:
    return f"{v:g}"


def _gaps(chatters: tuple[int | None, ...], bin_s: int) -> tuple[tuple[int, int], ...]:
    out: list[tuple[int, int]] = []
    start: int | None = None
    for i, c in enumerate(chatters):
        if c is None:
            if start is None:
                start = i
        elif start is not None:
            out.append((start * bin_s, i * bin_s))
            start = None
    if start is not None:
        out.append((start * bin_s, len(chatters) * bin_s))
    return tuple(out)


def line(a: StreamActivity, *, height: int = 48, max_points: int = 720) -> Line:
    chatters = a.chatters
    n = len(chatters)
    bin_s = a.bin_s
    known = [c for c in chatters if c is not None]
    peak = max(known) if known else 0
    gaps = _gaps(chatters, bin_s)

    # Each group keeps its maximum, so the tallest spike survives downsampling.
    group = max(1, math.ceil(n / max_points)) if n else 1
    pts: list[tuple[float, int | None]] = []
    for lo in range(0, n, group):
        hi = min(lo + group, n)
        vals = [c for c in chatters[lo:hi] if c is not None]
        pts.append(((lo + hi) / 2 * bin_s, max(vals) if vals else None))

    def y_of(c: int) -> float:
        if peak == 0:
            return float(height)
        return round(height - math.sqrt(c) / math.sqrt(peak) * (height - 4), 1)

    runs: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] | None = None
    for x, c in pts:
        if c is None:
            current = None
            continue
        if current is None:
            current = []
            runs.append(current)
        current.append((x, y_of(c)))

    d_line_parts: list[str] = []
    d_area_parts: list[str] = []
    for run in runs:
        body = " ".join(f"L{_num(x)} {_num(y)}" for x, y in run)
        rest = "".join(f" L{_num(x)} {_num(y)}" for x, y in run[1:])
        # A lone point is "M x y h0": a zero-length segment a round line cap shows as a dot.
        tail = rest if len(run) > 1 else " h0"
        d_line_parts.append(f"M{_num(run[0][0])} {_num(run[0][1])}{tail}")
        d_area_parts.append(
            f"M{_num(run[0][0])} {_num(height)} {body} L{_num(run[-1][0])} {_num(height)} Z"
        )
    return Line(
        width_s=n * bin_s,
        height=height,
        d_line=" ".join(d_line_parts),
        d_area=" ".join(d_area_parts),
        gaps=gaps,
        peak=peak,
    )


def usual_pace(streams: list[StreamActivity]) -> float | None:
    """Median of the bins above zero across the streams, or None under 30 such bins."""
    vals = [c for s in streams for c in s.chatters if c is not None and c > 0]
    if len(vals) < MIN_PACE_BINS:
        return None
    return float(statistics.median(vals))


def peak_near(a: StreamActivity, at: datetime, before_s: int = 30, after_s: int = 90) -> int | None:
    """The largest recorded bin overlapping the half-open span ``[at - before_s, at + after_s)``.

    Returns None when the bin holding ``at`` was not recorded (a gap over the anchor), when
    every bin in the span is null, or when the span lies wholly outside the stream.
    """
    off = (at - a.started_at).total_seconds()
    lo, hi = off - before_s, off + after_s
    n = len(a.chatters)
    anchor_bin = math.floor(off / a.bin_s)
    if 0 <= anchor_bin < n and a.chatters[anchor_bin] is None:
        return None
    first = max(0, math.floor(lo / a.bin_s))
    last = min(n - 1, math.ceil(hi / a.bin_s) - 1)
    best: int | None = None
    for i in range(first, last + 1):
        c = a.chatters[i]
        if c is not None and (best is None or c > best):
            best = c
    return best


Coverage = Literal["known", "gap", "unknown"]


def coverage(a: StreamActivity, at: datetime) -> Coverage:
    """What the counts know about the bin holding ``at``.

    ``"unknown"`` when that bin lies outside the counts (index not in ``0..n-1``): nothing says
    whether chat was recorded there. ``"gap"`` when the bin is ``None``: the export says it was not
    recorded. ``"known"`` otherwise, zero included. A partial overlap of ``peak_near``'s span keeps
    its number there; this only names the anchor's own bin."""
    i = math.floor((at - a.started_at).total_seconds() / a.bin_s)
    if not 0 <= i < len(a.chatters):
        return "unknown"
    return "gap" if a.chatters[i] is None else "known"


def ratio_text(r: float) -> str:
    """Under 10: one decimal, a trailing .0 dropped; otherwise a whole number."""
    tenth = round(r, 1)
    if tenth < 10:
        text = f"{tenth:.1f}"
        if text.endswith(".0"):
            text = text[:-2]
        return text + TIMES
    return f"{math.floor(r + 0.5)}{TIMES}"


def chips(
    moments: list[tuple[str, datetime, bool, str | None]], start: datetime, end: datetime
) -> list[Chip]:
    """Numbered chips for all of a stream's Moments, in (anchor, window_id) order."""
    span = (end - start).total_seconds()
    out: list[Chip] = []
    for n, (window_id, anchor, has_cause, doing) in enumerate(
        sorted(moments, key=lambda m: (m[1], m[0])), start=1
    ):
        pct = (anchor - start).total_seconds() / span * 100 if span > 0 else 0.0
        if has_cause:
            shape = "solid"
        elif doing is None:
            shape = "unsure"
        else:
            shape = "ring"
        out.append(Chip(n, window_id, min(100.0, max(0.0, pct)), shape, doing))
    return out
