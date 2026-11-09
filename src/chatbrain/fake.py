"""Made-up chat for P0 and Slice 1: exports, ground truth and deletion checks with no real data.

Everything here is invented: channels ``fakechan01``..., per-window speakers
``s1``..., emote IDs ``fake-<name>`` (``Emote("fake-kekw", "KEKW")``), and lines
built from small templates per ``doing`` value. Output is a pure function of the
arguments, so the same call gives the same bytes.

Streams: each channel has 2-3 made-up streams. Each stream ends 1-13 days before
``now`` (seeded) and its text expires 14 days after it ends, so a later clock
expires only some windows. The first stream that holds windows ends 10-13 days
before ``now`` and the second 1-8 days before, so ``now + 5 days`` always expires
some windows but not all once two streams hold windows.

Windows: anchors fall on whole minutes. Ordinary windows come from streams that
have Moments and sit at least 5 minutes from every Moment anchor in their stream.
About 1 in 7 ordinary windows is quiet (0 or 1 messages) so near-empty windows
reach every later step.

Activity (Slice 1, Task 2): ``make_export`` emits ``chatbrain-export/2`` with one made-up
activity line per stream that holds windows, binned every 10 s over the stream's own
span. Each stream gets one baseline of 3-25 people chatting, with +-2 of noise per bin;
each Moment adds a bump of 8-80 that peaks at its anchor bin and falls off over +-30 s,
so every Moment anchor sits above the stream's median. In 1 of every 4 streams (in
stream ID order) one run of 2-6 whole minutes is ``null`` (not recorded), never over a
Moment anchor. The title is ``Made-up stream <k>``. Activity is drawn after every
window, so the windows, and ``truth``, are the same as P0's. ``activity=False`` gives
P0's ``chatbrain-export/1`` bytes exactly.

Markers: template lines repeat across windows, so a check that an expired window's
text is gone can't rest on them. Every window with at least 2 lines therefore gets
one more ordinary message whose text is ``marker(window_id)``, a made-up token found
in no other window (ruling C9). It comes from someone who already posts in the
window, carries no emotes or mentions, and doesn't change the truth labels. Quiet
windows get no marker, so they stay at 0 or 1 messages.

Ground truth: ``truth`` reads it back from the export alone, because each
window's answers are a seeded function of its ``window_id`` and ``anchor_kind``,
and ``make_export`` writes the lines from those same answers.

Showcase (Task 11, rulings B3/B25): ``make_export`` gives a realistic mix, so nearly
every Moment is ``separate_messages`` and a stream may hold no chant at all.
``make_showcase`` adds one short stream on ``fakechan13`` that ends 2 hours before
``now``, so it is the newest stream and the page's home shows it: a chant, a
repeated-emotes Moment, a back-and-forth, two critical turns, a shocked Moment and an
unclear one, plus two ordinary windows. Its lines come from the same templates, in
fixed amounts per window (``SHOWCASE``), and its answers are fixed per window, so
``truth`` reads them back from the window IDs (``fake<seed>-show<NN>``) too.
"""

from __future__ import annotations

import hashlib
import random
import re
from bisect import bisect_left
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from .export_format import (
    ACTIVITY_BIN,
    ACTIVITY_BIN_S,
    EXPORT_FORMAT,
    EXPORT_FORMAT_V2,
    DeletionCheck,
    Emote,
    Export,
    Message,
    StreamActivity,
    Window,
    dump_deletion_check,
    dump_export,
    iso,
    parse_export,
)

TEXT_LIFETIME = timedelta(days=14)
_DAY_MIN = 24 * 60
_FAKE_CHANNEL = re.compile(r"fakechan[0-9]{2}")

EMOTE_NAMES = (
    "KEKW", "LUL", "OMEGALUL", "4Head", "PogChamp", "Pog", "EZ", "Clap", "WutFace", "monkaW", "monkaS",
    "BibleThump", "Sadge", "PepeHands", "ResidentSleeper", "NotLikeThis", "DansGame", "HeyGuys", "Kappa",
    "catJAM", "Kreygasm",
)  # fmt: skip
_EMOTES = {name: Emote(f"fake-{name.lower()}", name) for name in EMOTE_NAMES}

# "{m}" becomes "@s<N>" (someone who already posted in the window) or "@user".
DOING_LINES: dict[str, tuple[str, ...]] = {
    "laughing": (
        "KEKW", "KEKW KEKW", "KEKW KEKW KEKW", "LUL that was so bad", "LUL", "OMEGALUL", "OMEGALUL OMEGALUL",
        "lmao KEKW", "KEKW he really did that", "LUL LUL", "4Head", "hahaha KEKW",
    ),
    "cheering": (
        "LETS GO", "LETS GO", "LETS GO PogChamp", "GG", "gg wp", "Pog", "PogChamp", "EZ Clap", "Clap Clap",
        "what a play Pog", "HYPE", "LETS GO",
    ),
    "shocked": (
        "WHAT", "no way", "WutFace", "omg", "did that just happen", "WHAT WAS THAT", "monkaW", "how",
        "NO WAY", "WutFace WutFace", "excuse me",
    ),
    "tense": (
        "monkaS", "careful", "oh no oh no", "dont die", "monkaS monkaS", "please please", "this is it monkaS",
        "hold it", "so close", "breathe",
    ),
    "sad": (
        "BibleThump", "rip", "nooo", "Sadge", "F", "PepeHands", "so close BibleThump", "unlucky", "F F F",
        "Sadge rip",
    ),
    "asking": (
        "what game is this?", "what rank is he?", "{m} what happened?", "how long has he been live?",
        "is this the new patch?", "what did he say?", "who is that?", "{m} did you see that?",
        "when is the next stream?",
    ),
    "advising": (
        "go left", "use the shield", "you should reload", "{m} he needs to heal first", "check the map",
        "craft the potion first", "dont go there", "turn around", "{m} tell him to save", "drink a potion",
    ),
    "complaining": (
        "this is boring ResidentSleeper", "so much lag", "why does he always do this", "NotLikeThis",
        "unwatchable", "the audio is so quiet", "DansGame", "this patch is terrible", "ResidentSleeper",
        "NotLikeThis again",
    ),
    "chatting_as_usual": (
        "hi chat", "HeyGuys", "good morning", "what is everyone eating", "{m} how are you", "back from work",
        "lurking", "anyone watch the match yesterday", "{m} hi", "nice weather today", "just got here",
    ),
    "other": (
        "catJAM", "the music Kreygasm", "first time here", "nice overlay", "catJAM catJAM", "vibing catJAM",
        "this song Kreygasm", "song name",
    ),
}  # fmt: skip

AIMED_LINES: dict[str, tuple[str, ...]] = {
    "streamer": ("he is so good at this", "streamer carried", "classic streamer move", "this guy"),
    "game_content": ("that boss is huge", "the map looks great", "this level is hard", "the final boss"),
    "guest": ("the guest is so funny", "guest carried", "the guest is cracked", "hi to the guest"),
    "chat_itself": ("chat is wild today", "{m} lol", "chat calm down", "{m} true"),
    "tech_problem": ("stream is lagging", "audio cut out", "frozen for me", "buffering again", "no sound"),
    "unclear": ("anyway", "ok", "hmm", "lol"),
}

TONE_LINES: dict[str, tuple[str, ...]] = {
    "playful": ("Kappa", "4Head", "haha", "lol"),
    "serious": ("for real", "honestly", "that matters"),
    "sarcastic": ("wow great play genius Kappa", "nice one Kappa", "sure that will work Kappa", "amazing aim as always Kappa"),
    "hostile": ("this is garbage", "worst play ever", "just uninstall", "so bad it hurts"),
    "no_clear_tone": (),
}

FEELING_LINES: dict[str, tuple[str, ...]] = {
    "supportive": ("you got this", "well played", "proud of him"),
    "critical": ("that was bad", "terrible call", "what a mistake"),
    "mixed": ("you got this", "that was bad", "well played", "terrible call"),
    "neutral": (),
}

# Weighted choices: (value, weight).
MOMENT_DOING = (
    ("laughing", 22), ("cheering", 18), ("shocked", 12), ("tense", 8), ("sad", 7), ("asking", 6),
    ("advising", 7), ("complaining", 10), ("other", 5), ("chatting_as_usual", 5),
)  # fmt: skip
ORDINARY_DOING = (("chatting_as_usual", 70), ("asking", 12), ("laughing", 8), ("complaining", 6), ("other", 4))
SECOND_DOING: dict[str, tuple[str, ...]] = {
    "laughing": ("cheering", "shocked"),
    "cheering": ("laughing", "shocked"),
    "shocked": ("laughing", "tense"),
    "tense": ("advising", "shocked"),
    "sad": ("complaining",),
    "asking": ("advising",),
    "advising": ("asking", "complaining"),
    "complaining": ("advising", "sad"),
    "chatting_as_usual": ("asking",),
}
AIMED_BY_DOING = {
    "laughing": (("streamer", 4), ("chat_itself", 3), ("game_content", 2), ("guest", 1)),
    "cheering": (("streamer", 5), ("game_content", 3), ("guest", 1)),
    "shocked": (("game_content", 5), ("streamer", 3), ("tech_problem", 1)),
    "tense": (("game_content", 6), ("streamer", 2)),
    "sad": (("game_content", 4), ("streamer", 3), ("guest", 1)),
    "asking": (("streamer", 4), ("game_content", 4), ("guest", 1), ("tech_problem", 1)),
    "advising": (("streamer", 6), ("game_content", 2)),
    "complaining": (("tech_problem", 4), ("streamer", 3), ("game_content", 3)),
    "chatting_as_usual": (("chat_itself", 6), ("unclear", 4)),
    "other": (("unclear", 4), ("chat_itself", 3), ("game_content", 1)),
}
FEELING_BY_DOING = {
    "laughing": (("supportive", 5), ("neutral", 2), ("critical", 1), ("mixed", 1)),
    "cheering": (("supportive", 9), ("mixed", 1)),
    "shocked": (("neutral", 4), ("supportive", 3), ("mixed", 2)),
    "tense": (("supportive", 6), ("neutral", 3)),
    "sad": (("supportive", 6), ("neutral", 2)),
    "asking": (("neutral", 8), ("mixed", 1)),
    "advising": (("critical", 4), ("neutral", 4), ("mixed", 2)),
    "complaining": (("critical", 8), ("mixed", 2)),
    "chatting_as_usual": (("neutral", 9), ("supportive", 1)),
    "other": (("neutral", 5), ("mixed", 2), ("supportive", 2)),
}
TONE_BY_DOING = {
    "laughing": (("playful", 7), ("sarcastic", 2), ("no_clear_tone", 1)),
    "cheering": (("playful", 5), ("serious", 2), ("no_clear_tone", 3)),
    "shocked": (("no_clear_tone", 5), ("serious", 3), ("playful", 2)),
    "tense": (("serious", 6), ("no_clear_tone", 3)),
    "sad": (("serious", 6), ("no_clear_tone", 3)),
    "asking": (("no_clear_tone", 6), ("serious", 4)),
    "advising": (("serious", 6), ("sarcastic", 2), ("hostile", 1)),
    "complaining": (("serious", 3), ("sarcastic", 3), ("hostile", 3)),
    "chatting_as_usual": (("no_clear_tone", 6), ("playful", 4)),
    "other": (("no_clear_tone", 5), ("playful", 4)),
}
QUIET_SHARE = 0.15  # of ordinary windows: 0 or 1 messages
_DOING_ORDER = tuple(DOING_LINES)  # scheme order (config/labels_v1.json)


@dataclass(frozen=True)
class _Plan:
    doing: tuple[str, ...]  # primary first
    aimed_at: str
    feeling: str
    tone: str
    quiet_size: int | None  # 0 or 1 for a quiet window, else None

    def answers(self) -> dict[str, object]:
        return {
            "doing": [v for v in _DOING_ORDER if v in self.doing],
            "aimed_at": self.aimed_at,
            "feeling": self.feeling,
            "tone": self.tone,
        }


@dataclass(frozen=True)
class _Show:
    """One showcase window: its anchor minute in the stream, its answers and its lines."""

    minute: int  # the anchor, in whole minutes after the stream starts
    kind: str  # "moment" or "ordinary"
    plan: _Plan
    posters: int
    lines: tuple[tuple[str, int], ...]  # (template, how many times); "{m}" is a reply to an earlier poster


def _show(doing: tuple[str, ...], aimed_at: str, feeling: str, tone: str) -> _Plan:
    return _Plan(doing, aimed_at, feeling, tone, quiet_size=None)


SHOWCASE_CHANNEL = "fakechan13"  # make_export's default channels are fakechan01..fakechan12
SHOWCASE_MINUTES = 120  # the showcase stream's length
SHOWCASE_END_BEFORE_NOW = timedelta(hours=2)  # newer than every make_export stream (they end >= 1 day back)
SHOWCASE: tuple[_Show, ...] = (
    _Show(10, "moment", _show(("cheering",), "streamer", "supportive", "playful"), 14, (  # chant
        ("LETS GO", 12), ("LETS GO PogChamp", 3), ("Pog", 2), ("GG", 2), ("what a play Pog", 2), ("EZ Clap", 2),
        ("HYPE", 1), ("streamer carried", 2), ("he is so good at this", 1), ("well played", 2), ("you got this", 1),
        ("4Head", 1),
    )),
    _Show(25, "moment", _show(("laughing",), "streamer", "supportive", "playful"), 16, (  # repeated_emotes
        ("KEKW", 9), ("KEKW KEKW", 3), ("lmao KEKW", 3), ("KEKW he really did that", 2), ("hahaha KEKW", 2),
        ("KEKW KEKW KEKW", 1), ("LUL that was so bad", 2), ("OMEGALUL", 2), ("classic streamer move", 2),
        ("this guy", 1), ("haha", 1), ("Kappa", 1),
    )),
    _Show(40, "moment", _show(("complaining",), "tech_problem", "critical", "serious"), 10, (
        ("so much lag", 3), ("stream is lagging", 3), ("audio cut out", 2), ("buffering again", 2),
        ("frozen for me", 2), ("the audio is so quiet", 1), ("NotLikeThis", 6), ("NotLikeThis again", 2),
        ("DansGame", 3), ("ResidentSleeper", 1), ("unwatchable", 2), ("that was bad", 2), ("terrible call", 1), ("for real", 1),
        ("no sound", 1),
    )),
    _Show(55, "moment", _show(("complaining",), "streamer", "critical", "hostile"), 9, (
        ("why does he always do this", 3), ("this is garbage", 2), ("worst play ever", 2), ("what a mistake", 2),
        ("terrible call", 2), ("that was bad", 2), ("DansGame", 5), ("this is boring ResidentSleeper", 2),
        ("ResidentSleeper", 3), ("NotLikeThis", 2), ("just uninstall", 1), ("so bad it hurts", 1), ("this guy", 1),
        ("classic streamer move", 1),
    )),
    _Show(70, "moment", _show(("shocked",), "game_content", "neutral", "no_clear_tone"), 11, (
        ("WHAT", 4), ("no way", 3), ("NO WAY", 2), ("WutFace", 2), ("did that just happen", 2), ("WHAT WAS THAT", 2),
        ("monkaW", 1), ("that boss is huge", 2), ("the final boss", 2), ("this level is hard", 1), ("how", 1),
        ("excuse me", 1), ("omg", 2),
    )),
    _Show(85, "moment", _show(("asking", "advising"), "streamer", "neutral", "no_clear_tone"), 9, (  # back_and_forth
        ("what game is this?", 2), ("{m} what happened?", 3), ("{m} did you see that?", 3), ("what rank is he?", 2),
        ("who is that?", 1), ("{m} he needs to heal first", 2), ("{m} tell him to save", 2), ("check the map", 1),
        ("go left", 1), ("is this the new patch?", 1), ("he is so good at this", 1),
    )),
    _Show(100, "moment", _show(("other",), "unclear", "mixed", "no_clear_tone"), 8, (
        ("catJAM", 3), ("the music Kreygasm", 2), ("song name", 2), ("first time here", 1), ("nice overlay", 1),
        ("you got this", 1), ("that was bad", 1), ("anyway", 1), ("ok", 1), ("hmm", 1), ("lol", 1),
        ("vibing catJAM", 2),
    )),
    _Show(108, "ordinary", _show(("chatting_as_usual",), "chat_itself", "neutral", "no_clear_tone"), 6, (
        ("hi chat", 2), ("HeyGuys", 1), ("good morning", 1), ("back from work", 1), ("lurking", 1), ("{m} hi", 1),
        ("chat is wild today", 1), ("just got here", 1),
    )),
    _Show(116, "ordinary", _show(("chatting_as_usual",), "chat_itself", "neutral", "no_clear_tone"), 5, (
        ("anyone watch the match yesterday", 1), ("{m} how are you", 2), ("nice weather today", 1),
        ("what is everyone eating", 1), ("{m} true", 1), ("chat calm down", 1), ("hi chat", 1),
    )),
)  # fmt: skip
_SHOW_ID = re.compile(r"fake[0-9]+-show([0-9]{2})")


def make_export(
    seed: int, n_moment: int, n_ordinary: int, now: datetime, channels: int = 12, *, activity: bool = True
) -> bytes:
    """A made-up ``chatbrain-export/2`` file with ``exported_at = now``: windows plus one
    activity line per stream that holds them. ``activity=False`` gives P0's
    ``chatbrain-export/1`` file, byte for byte, with the same windows.
    """
    iso(now)  # refuses a naive datetime
    if n_moment < 0 or n_ordinary < 0:
        raise ValueError("window counts must be >= 0")
    if not 1 <= channels <= 99:
        raise ValueError("channels must be 1..99 (fakechan01..fakechan99)")
    if n_ordinary and not n_moment:
        raise ValueError("ordinary windows come from streams that have Moments; ask for at least one Moment")

    rng = random.Random(seed)
    streams = [(c, k) for c in range(1, channels + 1) for k in range(1, rng.randint(2, 3) + 1)]
    moments = Counter(rng.randrange(len(streams)) for _ in range(n_moment))
    with_moments = sorted(moments)
    ordinary = Counter(rng.choice(with_moments) for _ in range(n_ordinary))
    used = sorted(set(moments) | set(ordinary))

    minute_now = now.replace(second=0, microsecond=0)
    slots: list[tuple[int, int, int, str, datetime, datetime]] = []  # channel, k, minute, kind, start, end
    spans: list[tuple[int, int, datetime, datetime, list[int]]] = []  # channel, k, start, end, Moment minutes
    for order, index in enumerate(used):
        c, k = streams[index]
        if order == 0:
            back = rng.randint(10 * _DAY_MIN, 13 * _DAY_MIN - 1)  # expired by now + 5 days
        elif order == 1:
            back = rng.randint(_DAY_MIN + 1, 8 * _DAY_MIN)  # still present at now + 5 days
        else:
            back = rng.randint(_DAY_MIN + 1, 13 * _DAY_MIN - 1)
        n_m, n_o = moments[index], ordinary[index]
        length = max(rng.randint(180, 480), 9 * n_m + n_o + 4)  # minutes; leaves room for every window
        stream_end = minute_now - timedelta(minutes=back)
        stream_start = stream_end - timedelta(minutes=length)
        # An anchor on minute m keeps [anchor - 30 s, anchor + 90 s] inside the stream for 1 <= m <= length - 2.
        moment_minutes = rng.sample(range(1, length - 1), n_m)
        blocked = {m + d for m in moment_minutes for d in range(-4, 5)}  # < 5 minutes from a Moment
        free = [m for m in range(1, length - 1) if m not in blocked]
        ordinary_minutes = rng.sample(free, n_o)
        spans.append((c, k, stream_start, stream_end, moment_minutes))
        for kind, minutes in (("moment", moment_minutes), ("ordinary", ordinary_minutes)):
            for m in minutes:
                slots.append((c, k, m, kind, stream_start, stream_end))
    slots.sort(key=lambda s: (s[0], s[1], s[2]))

    windows = []
    for i, (c, k, m, kind, stream_start, stream_end) in enumerate(slots, start=1):
        anchor = stream_start + timedelta(minutes=m)
        window_id = f"fake{seed}-w{i:04d}"
        windows.append(
            Window(
                window_id=window_id,
                channel=f"fakechan{c:02d}",
                stream_id=f"fake{seed}-c{c:02d}-s{k}",
                anchor_kind=kind,  # type: ignore[arg-type]
                anchor_at=anchor,
                start_at=anchor - timedelta(seconds=30),
                end_at=anchor + timedelta(seconds=90),
                language="en",
                text_expires_at=stream_end + TEXT_LIFETIME,
                messages=_messages(window_id, kind, _plan(window_id, kind)),
            )
        )
    if not activity:
        return dump_export(Export(format=EXPORT_FORMAT, exported_at=now, source="vod", windows=tuple(windows)))
    # Drawn after every window, so the windows stay P0's. Sorted (c, k) is stream ID order.
    lines = tuple(
        _activity_line(rng, seed, c, k, start, end, minutes, gap=order % 4 == 0)
        for order, (c, k, start, end, minutes) in enumerate(sorted(spans, key=lambda s: (s[0], s[1])))
    )
    return dump_export(
        Export(format=EXPORT_FORMAT_V2, exported_at=now, source="vod", windows=tuple(windows), streams=lines)
    )


def _activity_line(
    rng: random.Random,
    seed: int,
    c: int,
    k: int,
    start: datetime,
    end: datetime,
    moment_minutes: Sequence[int],
    *,
    gap: bool,
) -> StreamActivity:
    """One made-up stream's people chatting per 10 s bin (see the module docstring)."""
    n = (end - start) // ACTIVITY_BIN
    per_minute = 60 // ACTIVITY_BIN_S
    baseline = rng.randint(3, 25)
    chatters: list[int | None] = [baseline + rng.randint(-2, 2) for _ in range(n)]
    anchors = sorted(per_minute * m for m in moment_minutes)  # an anchor on minute m sits in bin 6m
    for a in anchors:
        h = rng.randint(8, 80)
        for d in range(-2, 3):  # peaks at the anchor, gone by +-30 s
            chatters[a + d] += h * (3 - abs(d)) // 3  # type: ignore[operator]
    if gap:
        run = per_minute * rng.randint(2, 6)
        # A run starting at i covers no anchor when the first anchor at or after i lies past the run.
        starts = [
            i for i in range(n - run + 1) if (j := bisect_left(anchors, i)) == len(anchors) or anchors[j] >= i + run
        ]
        if starts:
            first = rng.choice(starts)
            chatters[first : first + run] = [None] * run
    return StreamActivity(
        stream_id=f"fake{seed}-c{c:02d}-s{k}",
        channel=f"fakechan{c:02d}",
        started_at=start,
        ended_at=end,
        title=f"Made-up stream {k}",
        bin_s=ACTIVITY_BIN_S,
        chatters=tuple(chatters),
    )


def truth(export_raw: bytes) -> dict[str, dict]:
    """The made-up answers per window, in label-answer shape.

    ``{window_id: {"doing": [values in scheme order], "aimed_at": v, "feeling": v, "tone": v}}``,
    with ``tone`` = ``no_clear_tone`` for a blank tone. Every question is answered (no
    not-sure). Refuses an export with any channel that isn't ``fakechanNN``, so made-up
    answers never land on real windows.
    """
    export = parse_export(export_raw)
    for w in export.windows:
        if not _FAKE_CHANNEL.fullmatch(w.channel):
            raise ValueError("truth() only reads made-up exports (fakechanNN channels)")
    return {w.window_id: _plan(w.window_id, w.anchor_kind).answers() for w in export.windows}


def marker(window_id: str) -> str:
    """The made-up marker line for ``window_id``: ``zq`` + 12 hex characters of a hash.

    Never a real word, an emote name or a mention, and one token, so a vectorizer with
    ``min_df=2`` drops it. ``make_export`` adds it to every window with at least 2 messages.
    """
    return "zq" + hashlib.sha256(f"chatbrain-marker:{window_id}".encode("utf-8")).hexdigest()[:12]


def make_deletion_check(export_raw: bytes, drop_ids: set[str], checked_at: datetime) -> bytes:
    """A made-up ``chatbrain-deletion-check/1`` for every window in the export.

    Each window lists its message IDs except ``drop_ids``. Raises ``ValueError`` for an
    ID that isn't in the export, so a typo can't pass for a deletion.
    """
    iso(checked_at)
    export = parse_export(export_raw)
    drop = set(drop_ids)
    known = {m.message_id for w in export.windows for m in w.messages}
    if drop - known:
        raise ValueError(f"{len(drop - known)} message ID(s) to drop are not in the export")
    present = {w.window_id: frozenset(m.message_id for m in w.messages) - drop for w in export.windows}
    return dump_deletion_check(DeletionCheck(checked_at=checked_at, present=present))


def make_showcase(seed: int, now: datetime) -> bytes:
    """A made-up ``chatbrain-export/1`` file with one short showcase stream (``SHOWCASE``).

    The stream is on ``fakechan13`` and ends ``SHOWCASE_END_BEFORE_NOW`` before ``now``
    (on a whole minute), so it is newer than every ``make_export`` stream for the same
    ``now``; its text expires 14 days after it ends. Window IDs are ``fake<seed>-show<NN>``,
    so they never meet ``make_export``'s, and ``truth`` reads their answers back. Every
    window gets its marker line, like ``make_export``'s. The same arguments give the same bytes.
    """
    iso(now)  # refuses a naive datetime
    stream_end = now.replace(second=0, microsecond=0) - SHOWCASE_END_BEFORE_NOW
    stream_start = stream_end - timedelta(minutes=SHOWCASE_MINUTES)
    windows = []
    for i, show in enumerate(SHOWCASE, start=1):
        window_id = f"fake{seed}-show{i:02d}"
        anchor = stream_start + timedelta(minutes=show.minute)
        windows.append(
            Window(
                window_id=window_id,
                channel=SHOWCASE_CHANNEL,
                stream_id=f"fake{seed}-c13-show",
                anchor_kind=show.kind,  # type: ignore[arg-type]
                anchor_at=anchor,
                start_at=anchor - timedelta(seconds=30),
                end_at=anchor + timedelta(seconds=90),
                language="en",
                text_expires_at=stream_end + TEXT_LIFETIME,
                messages=_show_messages(window_id, show),
            )
        )
    return dump_export(Export(format=EXPORT_FORMAT, exported_at=now, source="vod", windows=tuple(windows)))


# --- internals ---------------------------------------------------------------------------


def _rng(*parts: object) -> random.Random:
    digest = hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _pick(r: random.Random, weighted: Sequence[tuple[str, int]]) -> str:
    values, weights = zip(*weighted)
    return r.choices(values, weights=weights, k=1)[0]


def _plan(window_id: str, anchor_kind: str) -> _Plan:
    show = _showcase_entry(window_id, anchor_kind)
    if show is not None:
        return show.plan
    r = _rng("truth", window_id, anchor_kind)
    if anchor_kind == "ordinary" and r.random() < QUIET_SHARE:
        return _Plan(("chatting_as_usual",), "unclear", "neutral", "no_clear_tone", quiet_size=r.randint(0, 1))
    primary = _pick(r, MOMENT_DOING if anchor_kind == "moment" else ORDINARY_DOING)
    doing: tuple[str, ...] = (primary,)
    if primary in SECOND_DOING and r.random() < 0.2:
        doing += (r.choice(SECOND_DOING[primary]),)
    return _Plan(
        doing=doing,
        aimed_at=_pick(r, AIMED_BY_DOING[primary]),
        feeling=_pick(r, FEELING_BY_DOING[primary]),
        tone=_pick(r, TONE_BY_DOING[primary]),
        quiet_size=None,
    )


def _messages(window_id: str, kind: str, plan: _Plan) -> tuple[Message, ...]:
    r = _rng("text", window_id)
    if plan.quiet_size is not None:
        n, posters = plan.quiet_size, 1
    elif kind == "moment":
        n = r.randint(20, 60)
        posters = r.randint(6, min(25, n))
    else:
        n = r.randint(6, 18)
        posters = r.randint(3, min(8, n))

    poster_weights = [1 / (i + 1) ** 0.8 for i in range(posters)]
    drafts = []
    for j in range(n):
        if kind == "moment" and r.random() < 0.75:
            rel_ms = r.randrange(30_000, 120_000)  # most of a Moment's chat comes after the anchor
        else:
            rel_ms = r.randrange(0, 120_000)
        poster = r.choices(range(posters), weights=poster_weights, k=1)[0]
        drafts.append((rel_ms, j, poster, _template(r, plan)))
    if n >= 2:
        # The marker line (ruling C9) draws from its own generator, so every other line
        # keeps its text, and comes from someone who already posts here.
        rm = _rng("marker", window_id)
        drafts.append((rm.randrange(0, 120_000), n, rm.choice(sorted({d[2] for d in drafts})), marker(window_id)))
    drafts.sort()

    numbers: dict[int, str] = {}  # poster -> "s<N>" by first post
    messages = []
    for rel_ms, j, poster, template in drafts:
        speaker = numbers.setdefault(poster, f"s{len(numbers) + 1}")
        if j == n:  # the marker: plain text, and no draw from r, so the other lines don't shift
            text = template
        else:
            text = template.replace("{m}", _mention(r, speaker, list(numbers.values())))
        emotes = tuple(_EMOTES[t] for t in text.split(" ") if t in _EMOTES)
        messages.append(Message(f"{window_id}-m{len(messages) + 1:03d}", rel_ms, speaker, text, emotes))
    return tuple(messages)


def _showcase_entry(window_id: str, anchor_kind: str) -> _Show | None:
    found = _SHOW_ID.fullmatch(window_id)
    if found is None:
        return None
    index = int(found.group(1)) - 1
    if not 0 <= index < len(SHOWCASE) or SHOWCASE[index].kind != anchor_kind:
        return None
    return SHOWCASE[index]


def _show_messages(window_id: str, show: _Show) -> tuple[Message, ...]:
    """The showcase window's lines in a seeded order and timing; every poster posts at least once."""
    r = _rng("showcase", window_id)
    texts = [t for t, n in show.lines for _ in range(n)]
    r.shuffle(texts)
    first = next(i for i, t in enumerate(texts) if "{m}" not in t)  # a reply needs someone before it
    texts.insert(0, texts.pop(first))
    n = len(texts)
    if show.kind == "moment":  # most of a Moment's chat comes after the anchor
        times = sorted(r.randrange(0, 30_000) if r.random() < 0.25 else r.randrange(30_000, 120_000) for _ in range(n))
    else:
        times = sorted(r.randrange(0, 120_000) for _ in range(n))
    posters = [j % show.posters if j < show.posters else r.randrange(show.posters) for j in range(n)]
    drafts = [(rel_ms, j, posters[j], texts[j]) for j, rel_ms in enumerate(times)]
    rm = _rng("marker", window_id)
    drafts.append((rm.randrange(0, 120_000), n, rm.choice(sorted(set(posters))), marker(window_id)))
    drafts.sort()

    numbers: dict[int, str] = {}  # poster -> "s<N>" by first post
    messages = []
    for rel_ms, _j, poster, template in drafts:
        speaker = numbers.setdefault(poster, f"s{len(numbers) + 1}")
        others = [s for s in numbers.values() if s != speaker]
        text = template
        if "{m}" in template:
            text = template.replace("{m}", "@" + (r.choice(others) if others else "user"))
        emotes = tuple(_EMOTES[t] for t in text.split(" ") if t in _EMOTES)
        messages.append(Message(f"{window_id}-m{len(messages) + 1:03d}", rel_ms, speaker, text, emotes))
    return tuple(messages)


def _template(r: random.Random, plan: _Plan) -> str:
    x = r.random()
    primary, *rest = plan.doing
    if rest:
        if x < 0.45:
            return r.choice(DOING_LINES[primary])
        if x < 0.65:
            return r.choice(DOING_LINES[rest[0]])
    elif x < 0.62:
        return r.choice(DOING_LINES[primary])
    if x < 0.74:
        return r.choice(AIMED_LINES[plan.aimed_at])
    if x < 0.82 and TONE_LINES[plan.tone]:
        return r.choice(TONE_LINES[plan.tone])
    if x < 0.88 and FEELING_LINES[plan.feeling]:
        return r.choice(FEELING_LINES[plan.feeling])
    return r.choice(DOING_LINES["chatting_as_usual"])


def _mention(r: random.Random, speaker: str, posted: Iterable[str]) -> str:
    others = [s for s in posted if s != speaker]
    if others and r.random() < 0.8:
        return "@" + r.choice(others)
    return "@user"
