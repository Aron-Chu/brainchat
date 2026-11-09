"""Asking with no model: the refusal pre-check and the keyword parser (engine design note §2, Task 10b).

``refusal`` runs on every question before anything else. A refused question never reaches
a parser or a model, and only ``[refused: <reason>]`` is stored for it.

``parse`` turns a question into a query object with no model, from the word map in
``config/ask_words.json`` plus the emote names and channels the store holds:

1. a quoted phrase becomes ``phrase`` (matched only in text that is still kept);
2. clock times (``10:03pm``, ``at like 1am``) become ``at``; ``more than 20`` becomes
   ``min_posters`` 21; ``top 10`` becomes ``limit``; ``#2`` and ``the 2nd one`` become ``ref``;
3. the rest is scanned left to right, longest phrase first, against the word map;
4. an emote name filters by that emote when the question is about the emote (``mean``,
   ``spam``, ``emote``) or the name is not chat slang; otherwise its slang value counts
   (``KEKW`` alone reads as laughing);
5. the intent follows a fixed order: refusal, on-stream questions, emote meaning, explain,
   one Moment, trend, count, top, find, and else the plainest reading (what chat did,
   counted by value). Words it couldn't place come back as ``leftover`` and show as struck
   "didn't understand" chips.

An on-stream question (a sponsor, a death, a raid) keeps the labels it named, so the closest
Moments match them; asked on one Moment it becomes that Moment's detail with the cause line.
A question about viewers sets ``viewers``, so the answer says it counts posters, not viewers.

With a ``previous`` query, a question with refine words (``only``, ``same for``) or with
filters and no intent words narrows it; ``the 2nd one`` points at its results.
"""

from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..scheme import Scheme
from .query import KEY_QUESTION, KEYS, LIST_KEYS, VALUE_KEYS, Held, blank, tokens

WORDS_PATH = Path(__file__).resolve().parents[3] / "config" / "ask_words.json"
WORDS_FORMAT = "chatbrain-ask-words/1"
_SECTIONS = ("is", "time", "sort", "intent", "about", "by")
_FLAGS = ("vague", "refine", "result_ref", "emote_context", "why", "on_stream", "viewers", "account", "traits")
YES_NO = ("was", "were", "did", "does", "do", "is", "are")
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
            "ninth": 9, "tenth": 10, "last": None}  # fmt: skip
STRONG = ("top_moments", "count", "trend", "emote_meaning", "explain_answer", "moment_detail")


class WordsError(ValueError):
    """The word map file is malformed."""


@dataclass(frozen=True, eq=False)  # hashed by identity, so the loaded map can key caches
class Words:
    values: dict[str, dict[str, tuple[str, ...]]]
    sections: dict[str, dict[str, tuple[str, ...]]]
    flags: dict[str, tuple[str, ...]]
    stop: frozenset[str]


@functools.lru_cache(maxsize=4)
def load(path: Path | None = None) -> Words:
    """Read ``config/ask_words.json`` (or ``path``) and check its shape."""
    p = WORDS_PATH if path is None else Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WordsError(f"cannot read the ask word map {p}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("format") != WORDS_FORMAT:
        raise WordsError(f"the ask word map must have format {WORDS_FORMAT!r}")

    def words(x: Any, where: str) -> tuple[str, ...]:
        if not isinstance(x, list) or not all(isinstance(w, str) and w == w.lower().strip() and w for w in x):
            raise WordsError(f"{where}: a list of lowercase words")
        return tuple(x)

    def table(x: Any, where: str) -> dict[str, tuple[str, ...]]:
        if not isinstance(x, dict):
            raise WordsError(f"{where}: an object")
        return {k: words(v, f"{where}.{k}") for k, v in x.items()}

    values = raw.get("values")
    if not isinstance(values, dict) or set(values) != {*VALUE_KEYS, "spread"}:
        raise WordsError("values: one table per value key")
    return Words(
        values={k: table(v, f"values.{k}") for k, v in values.items()},
        sections={s: table(raw.get(s), s) for s in _SECTIONS},
        flags={f: words(raw.get(f), f) for f in _FLAGS},
        stop=frozenset(words(raw.get("stop"), "stop")),
    )


# --- normalising ---------------------------------------------------------------------------------

_APOSTROPHES = re.compile(r"['’‘`]")
_QUOTED = re.compile(r"[\"“”]([^\"“”]{1,100})[\"“”]")
_WORD = re.compile(r"[a-z0-9_]+")


def _norm(text: str) -> str:
    return _APOSTROPHES.sub("", text.lower())


def _phrase_re(words: tuple[str, ...]) -> re.Pattern[str]:
    alts = sorted({re.escape(w) for w in words}, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(alts) + r")\b")


# --- the refusal pre-check -----------------------------------------------------------------------

_WHO = re.compile(r"\b(?:who|whos|whom|whose)\b")
_SPEAKER = re.compile(r"\bs\d+s?\b")
_ABOUT_ME = re.compile(r"\b(?:say|says|said|saying|talk|talks|talked|talking|mention|mentioned|mentioning)\b.*\babout me\b")
_ONE_MESSAGE = re.compile(r"\b(?:this|that)\s+(?:message|msg|line|comment|chatter)\b")
_WHICH_MESSAGE = re.compile(r"\bwhich\s+(?:message|messages|msg|msgs|line|lines|comment|comments)\b")
_JUDGED = re.compile(r"\b(?:toxic|mean|rude|bad|offensive|hateful|worst|problem|problematic|bannable|reportable)\b")
_AS_STREAMER = re.compile(
    r"\b(?:say|speak|talk|write|reply|respond|answer)\b.*\b(?:like|as)\s+(?:the\s+)?(?:streamer|me|him|her)\b"
    r"|\bpretend\b|\broleplay\b|\brole play\b|\bimpersonat"
)
_OUTBOUND = re.compile(
    r"\b(?:send|post|share|upload|tweet|email)\s+(?:this|it|that|these|them|the)\b"
    r"|\b(?:discord|twitter|reddit|slack|webhook|tiktok)\b"
)


@functools.lru_cache(maxsize=4)
def _refusal_words(words: Words) -> tuple[re.Pattern[str], re.Pattern[str]]:
    return _phrase_re(words.flags["account"]), _phrase_re(words.flags["traits"])


def refusal(question: str) -> str | None:
    """The refusal reason for a question, or None: identity, traits, single message, as the
    streamer, or sending something out (engine design note §5)."""
    t = _norm(question)
    account, traits = _refusal_words(load())
    if _WHO.search(t) or _SPEAKER.search(t) or account.search(t) or _ABOUT_ME.search(t):
        return "identity"
    if traits.search(t):
        return "traits"
    if _ONE_MESSAGE.search(t) or (_WHICH_MESSAGE.search(t) and _JUDGED.search(t)):
        return "single_message"
    if _AS_STREAMER.search(t):
        return "as_streamer"
    if _OUTBOUND.search(t):
        return "outbound"
    return None


# --- times and numbers ---------------------------------------------------------------------------

_CLOCK = re.compile(r"\b(?:at\s+)?(?:like\s+|around\s+|about\s+)?([01]?\d|2[0-3]):([0-5]\d)\s*(am|pm)?\b")
_HOUR = re.compile(r"\b(?:at\s+)?(?:like\s+|around\s+|about\s+)?(1[0-2]|0?[1-9])\s*(am|pm)\b")
_MORE = re.compile(r"\b(?:more than|over|above)\s+(\d{1,6})\b")
_LEAST = re.compile(r"\bat least\s+(\d{1,6})\b|\b(\d{1,6})\s*(?:\+|or more\b)")
_LIMIT = re.compile(r"\b(top|best|first|biggest)\s+(\d{1,3})\b|\b(\d{1,3})\s+(best|biggest|top)\b")
_REF = re.compile(r"#(\d{1,2})\b|\b(\d{1,2})(?:st|nd|rd|th)\b")
_REF_WORD = re.compile(r"\b(?:the\s+)?(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+(?:one|moment|result)\b")


def _clock(h: int, m: int, ampm: str | None) -> str | None:
    if ampm == "pm" and h < 12:
        h += 12
    elif ampm == "am" and h == 12:
        h = 0
    return f"{h:02d}:{m:02d}" if h < 24 and m < 60 else None


def _times(t: str) -> tuple[str | None, str]:
    at = None
    m = _CLOCK.search(t)
    if m:
        at = _clock(int(m.group(1)), int(m.group(2)), m.group(3))
        t = t[: m.start()] + " " + t[m.end() :]
    else:
        m = _HOUR.search(t)
        if m:
            at = _clock(int(m.group(1)), 0, m.group(2))
            t = t[: m.start()] + " " + t[m.end() :]
    return at, t


def _numbers(t: str) -> tuple[dict[str, int], str]:
    got: dict[str, int] = {}

    def cut(m: re.Match[str], keep: str = " ") -> str:
        return t[: m.start()] + keep + t[m.end() :]

    m = _MORE.search(t)
    if m:
        got["min_posters"] = int(m.group(1)) + 1
        t = cut(m)
    else:
        m = _LEAST.search(t)
        if m:
            got["min_posters"] = int(m.group(1) or m.group(2))
            t = cut(m)
    m = _LIMIT.search(t)
    if m:
        got["limit"] = int(m.group(2) or m.group(3))
        t = cut(m, f" {m.group(1) or m.group(4)} ")  # keep "top"/"best": it names the intent
    m = _REF.search(t)
    if m:
        got["ref"] = int(m.group(1) or m.group(2))
        t = cut(m)
    else:
        m = _REF_WORD.search(t)
        if m and ORDINALS.get(m.group(1)):
            got["ref"] = int(ORDINALS[m.group(1)] or 0)
            t = cut(m)
    return got, t


# --- the phrase scan -----------------------------------------------------------------------------

Tag = tuple[str, ...]


@functools.lru_cache(maxsize=4)
def _base_index(words: Words) -> dict[tuple[str, ...], tuple[Tag, ...]]:
    index: dict[tuple[str, ...], list[Tag]] = {}

    def put(phrase: str, tag: Tag) -> None:
        key = tuple(_WORD.findall(phrase))
        if key and tag not in index.setdefault(key, []):
            index[key].append(tag)

    for key, table in words.values.items():
        for value, phrases in table.items():
            for ph in phrases:
                put(ph, ("value", key, value))
    for section, table in words.sections.items():
        for value, phrases in table.items():
            for ph in phrases:
                put(ph, (section, value))
    for flag, phrases in words.flags.items():
        if flag in ("account", "traits"):
            continue
        for ph in phrases:
            put(ph, ("flag", flag))
    for ph in words.stop:
        if " " in ph:
            put(ph, ("stop",))
    return {k: tuple(v) for k, v in index.items()}


@dataclass(frozen=True)
class Match:
    pos: int
    words: tuple[str, ...]
    tags: tuple[Tag, ...]


def _scan(toks: list[str], index: dict[tuple[str, ...], tuple[Tag, ...]], stop: frozenset[str]) -> tuple[list[Match], list[str]]:
    longest = max((len(k) for k in index), default=1)
    out: list[Match] = []
    unknown: list[str] = []
    i = 0
    while i < len(toks):
        for n in range(min(longest, len(toks) - i), 0, -1):
            key = tuple(toks[i : i + n])
            if key in index:
                out.append(Match(i, key, index[key]))
                i += n
                break
        else:
            if toks[i] not in stop and toks[i] not in unknown:
                unknown.append(toks[i])
            i += 1
    return out, unknown


# --- the parser ----------------------------------------------------------------------------------


@dataclass
class Parsed:
    query: dict[str, Any]
    refused: str | None = None
    leftover: list[str] = field(default_factory=list)  # words it couldn't place
    default: bool = False  # the plainest reading, for a vague question
    kept: list[str] = field(default_factory=list)  # tokens carried over from ``previous``
    viewers: bool = False  # asks about viewers: the answer says BrainChat counts posters, not viewers


_MEAN_QUESTION = re.compile(r"\b(?:what|whats)\b.*\bmean|\bmeaning of\b|\bstands? for\b")


def parse(
    question: str,
    scheme: Scheme,
    held: Held,
    previous: dict[str, Any] | None = None,
    window_id: str | None = None,
) -> Parsed:
    """The query object for ``question`` with no model (see the module docstring)."""
    words = load()
    if previous is not None and previous.get("intent") == "cant_answer":
        previous = None  # a refusal or a can't-answer has no scope to narrow
    reason = refusal(question)
    if reason is not None:
        at, _ = _times(_norm(question))
        return Parsed(blank() | {"intent": "cant_answer", "reason": reason, "at": at}, refused=reason)

    qm = _QUOTED.search(question)
    phrase = qm.group(1).strip() if qm else None
    rest = question[: qm.start()] + " " + question[qm.end() :] if qm else question
    t = _norm(rest)
    at, t = _times(t)
    nums, t = _numbers(t)
    toks = _WORD.findall(t)

    index = dict(_base_index(words))
    for name in held.emotes:
        key = tuple(_WORD.findall(name.lower()))
        if key:
            index[key] = (*index.get(key, ()), ("emote", name))
    for ch in held.channels:
        key = tuple(_WORD.findall(ch.lower()))
        if key:
            index[key] = (*index.get(key, ()), ("channel", ch))
    matches, leftover = _scan(toks, index, words.stop)

    vals: dict[str, list[str]] = {k: [] for k in (*VALUE_KEYS, "spread")}
    val_pos: dict[str, int] = {}  # question key -> first position, for "about"
    spread_from_context: set[str] = set()
    is_: list[str] = []
    channels: list[str] = []
    emotes: list[str] = []
    time = sort = about = by = None
    intents: dict[str, int] = {}
    flags: set[str] = set()
    emote_hits: list[Match] = []
    context = set(words.flags["emote_context"])

    for m in matches:
        if any(tag[0] == "flag" for tag in m.tags):
            flags |= {tag[1] for tag in m.tags if tag[0] == "flag"}
    for m in matches:
        if any(tag[0] == "emote" for tag in m.tags):
            emote_hits.append(m)
            continue
        for tag in m.tags:
            kind = tag[0]
            if kind == "value":
                key, value = tag[1], tag[2]
                if value not in vals[key]:
                    vals[key].append(value)
                val_pos.setdefault(key, m.pos)
                if key == "spread" and " ".join(m.words) in context:
                    spread_from_context.add(value)
            elif kind == "is" and tag[1] not in is_:
                is_.append(tag[1])
            elif kind == "time":
                time = time or tag[1]
            elif kind == "sort":
                sort = sort or tag[1]
            elif kind == "intent":
                intents.setdefault(tag[1], m.pos)
            elif kind == "about":
                about = about or tag[1]
            elif kind == "by":
                by = by or tag[1]
            elif kind == "channel" and tag[1] not in channels:
                channels.append(tag[1])
    # an emote name: the emote itself when the question is about the emote, else its slang
    about_emote = bool(flags & {"emote_context"}) or "emote_meaning" in intents
    for m in emote_hits:
        name = next(tag[1] for tag in m.tags if tag[0] == "emote")
        slang = [tag for tag in m.tags if tag[0] == "value"]
        if about_emote or not slang:
            if name not in emotes:
                emotes.append(name)
        else:
            for _, key, value in slang:
                if value not in vals[key]:
                    vals[key].append(value)
                val_pos.setdefault(key, m.pos)
    if emotes:
        vals["spread"] = [v for v in vals["spread"] if v not in spread_from_context]
    if "emote_meaning" in intents and not emotes and not _MEAN_QUESTION.search(t):
        del intents["emote_meaning"]  # "was chat mean": not about an emote
        if "critical" not in vals["feels"]:
            vals["feels"].append("critical")
        val_pos.setdefault("feels", 0)
    if by == "channel" and len(channels) < 2:
        by = None  # "vs" compares channels only when two are named

    q = blank()
    for k in (*VALUE_KEYS, "spread"):
        q[k] = vals[k]
    q["is"], q["emote"], q["channel"] = is_, emotes, channels
    q["phrase"], q["at"] = phrase, at
    for k in ("min_posters", "limit", "ref"):
        if k in nums:
            q[k] = nums[k]
    if sort:
        q["sort"] = sort
    q["about"], q["by"] = about, by
    q["asks_cause"] = "why" in flags
    has_filters = any(q[k] for k in (*VALUE_KEYS, "spread", "is", "emote")) or phrase is not None
    names_result = "ref" in nums or "result_ref" in flags or window_id is not None
    pointing = at is not None or names_result  # a clock time points at a Moment, not at an answer
    refine = previous is not None and (
        "refine" in flags or (not intents and bool(has_filters or time or at or "min_posters" in nums))
    )
    default = False

    def first_about() -> str | None:
        found = sorted((p, k) for k, p in val_pos.items() if k in KEY_QUESTION)
        return KEY_QUESTION[found[0][1]] if found else None

    def keep_scope() -> None:
        for k in (*VALUE_KEYS, "spread", "is"):
            q[k] = []

    per_result = set(intents) <= {"explain_answer", "moment_detail"}  # "explain that", "what happened in that one"
    if previous is not None and ("ref" in nums or ("result_ref" in flags and per_result)):
        out = dict(previous)
        out.update(follow_up="about_result", ref=nums.get("ref", 1),
                   intent="explain_answer" if "why" in flags or "explain_answer" in intents
                   else "moment_detail")  # fmt: skip
        out["asks_cause"] = "why" in flags
        return Parsed(_tidy(out), leftover=leftover, kept=[t for t in tokens(previous) if t in tokens(out)])

    viewers = "viewers" in flags
    if window_id is not None:
        if "why" in flags or "explain_answer" in intents:
            q["intent"] = "explain_answer"
        elif "on_stream" in flags:  # what chat did in this Moment; the cause stays unknown
            q["about"] = about or first_about()
            keep_scope()
            q.update(intent="moment_detail", asks_cause=True)
        else:
            q["about"] = about or first_about()
            keep_scope()
            q["intent"] = "moment_detail"
        q["time"] = "all"
        return Parsed(_tidy(q), leftover=leftover, viewers=viewers)

    if "on_stream" in flags or ("why" in flags and "explain_answer" not in intents and not names_result):
        q.update(intent="cant_answer", reason="on_stream")  # the labels stay: the closest Moments match them
    elif "emote_meaning" in intents:
        q["intent"] = "emote_meaning"
        if not time:
            q["time"] = "all"
    elif "explain_answer" in intents or ("why" in flags and names_result):
        q["intent"] = "explain_answer"
    elif pointing and not ({"top_moments", "find_moments", "trend"} & set(intents)):
        q["about"] = about or first_about()
        keep_scope()
        q["intent"] = "moment_detail"
    elif "trend" in intents:
        q["intent"] = "trend"
    elif "count" in intents or (toks[:1] and toks[0] in YES_NO and has_filters):
        q["intent"] = "count"
        if q["about"] and q["by"] is None:
            q["by"] = "value"  # "how did chat feel about my guest": the feelings, counted
    elif "top_moments" in intents:
        q["intent"] = "top_moments"
    elif "find_moments" in intents or has_filters:
        q["intent"] = "find_moments"
    elif refine:
        q["intent"] = previous["intent"] if previous else "find_moments"  # "same for last week"
    else:
        q.update(intent="count", by="value", about=q["about"] or "doing")
        default = True

    if time:
        q["time"] = time
    elif q["intent"] == "trend":
        q["time"] = "14d"
    elif q["is"] and q["intent"] in ("find_moments", "count", "top_moments"):
        q["time"] = "all"

    if previous is not None and refine:
        merged = _merge(previous, q, time_given=bool(time), strong=set(intents) & set(STRONG))
        kept = [t for t in tokens(previous) if t in tokens(merged) and not t.startswith("intent:")]
        return Parsed(_tidy(merged), leftover=leftover, kept=kept, viewers=viewers)
    return Parsed(_tidy(q), leftover=leftover, default=default, viewers=viewers)


def _merge(prev: dict[str, Any], new: dict[str, Any], *, time_given: bool, strong: set[str]) -> dict[str, Any]:
    """``prev`` narrowed by ``new``: lists joined (``prev`` first), the scalars ``new`` sets replace."""
    out = dict(prev)
    for k in LIST_KEYS:
        out[k] = list(prev.get(k) or []) + [v for v in new[k] if v not in (prev.get(k) or [])]
    b = blank()
    for k in ("phrase", "at", "min_posters", "about", "by", "ref"):
        if new[k] is not None:
            out[k] = new[k]
    if new["limit"] != b["limit"]:
        out["limit"] = new["limit"]
    if new["sort"] != b["sort"]:
        out["sort"] = new["sort"]
    if time_given:
        out["time"] = new["time"]
    if strong:
        out["intent"] = new["intent"]
    out["follow_up"] = "refine"
    out["asks_cause"] = bool(new["asks_cause"])
    return out


def _tidy(q: dict[str, Any]) -> dict[str, Any]:
    """Exactly the query keys, in order."""
    b = blank()
    return {k: q.get(k, b[k]) for k in KEYS}
