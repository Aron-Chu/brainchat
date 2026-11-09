"""The ask query object: its keys, the per-request JSON schema, the validator, tokens and chips.

A question becomes one query object (engine design note §1). The keyword parser (``keywords``) builds it
with no model; Task 10c's model step returns one through Ollama ``format`` = ``schema(...)``.
Either way ``check`` re-checks every key and value in Python and fails closed:

- the keys are exactly ``KEYS``; an extra key is dropped (an "ignored" chip) and a missing one
  takes its blank value (code ``missing:<key>``);
- every value comes from the label scheme, the store (channels held, emote names held) or a
  fixed list here; anything else is dropped as an "ignored: …" chip;
- ``limit`` is clamped to 1–20 and ``min_posters`` to 1–10000;
- per-intent rules: ``emote_meaning`` needs exactly one emote, ``count`` by channel needs two
  channels, ``top_moments`` sorts only by a count, ``trend`` groups by day or stream;
- a query it can't use becomes ``cant_answer`` with no reason and three runnable suggestions.
  There is no retry.

With no emote names or channels held, ``schema`` makes that key ``{"type": "null"}`` (an empty
enum is not a valid schema, B34) and ``check`` reads a null emote or channel list as empty.

``window_id`` is never a query key: it comes from the page (a per-Moment Ask) or from ``ref``.

Tokens (``doing:laughing``, ``time:7d``) are the query written as a list; chips are tokens in
words. A chip edit posts tokens back, so tokens, chips and the stored query are the same thing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..scheme import Scheme
from ..store import Store

KEYS = (
    "intent", "follow_up", "doing", "aimed", "feels", "tone", "spread", "is", "emote", "phrase", "channel",
    "time", "at", "min_posters", "sort", "limit", "ref", "about", "by", "asks_cause", "reason",
)  # fmt: skip
INTENTS = (
    "find_moments", "top_moments", "moment_detail", "explain_answer", "count", "trend", "emote_meaning",
    "cant_answer",
)  # fmt: skip
FOLLOW_UPS = ("new", "refine", "about_result")
QUESTION_KEY = {"doing": "doing", "aimed_at": "aimed", "feeling": "feels", "tone": "tone"}  # scheme -> query
KEY_QUESTION = {v: k for k, v in QUESTION_KEY.items()}
VALUE_KEYS = tuple(QUESTION_KEY.values())
LIST_KEYS = (*VALUE_KEYS, "spread", "is", "emote", "channel")
IS_VALUES = ("not_sure", "needs_review", "labeled", "has_cause", "has_text")
TIMES = ("last_stream", "24h", "7d", "14d", "all")
SORTS = ("posters", "lines", "rate", "newest", "oldest", "least_sure")
COUNT_SORTS = ("posters", "lines", "rate")
ABOUT = tuple(QUESTION_KEY)
BY = ("value", "channel", "stream", "period", "day")
TREND_BY = ("day", "stream")
REASONS = (
    "identity", "traits", "single_message", "as_streamer", "outbound", "off_topic", "on_stream", "unresolved_ref",
)  # fmt: skip
LIMIT_DEFAULT, LIMIT_MIN, LIMIT_MAX = 5, 1, 20
POSTERS_MAX = 10000
PHRASE_MAX = 100
AT_PATTERN = r"^([01][0-9]|2[0-3]):[0-5][0-9]$"
_AT = re.compile(AT_PATTERN)

IS_TEXT = {
    "not_sure": "not sure",
    "needs_review": "needs review",
    "labeled": "has your label",
    "has_cause": "has a cause note",
    "has_text": "text still kept",
}
TIME_TEXT = {"last_stream": "last stream", "24h": "last 24 h", "7d": "last 7 days", "14d": "last 14 days", "all": "all time"}
SORT_TEXT = {
    "posters": "posters joining ↓",
    "lines": "lines joining ↓",
    "rate": "messages per minute ↓",
    "newest": "newest first",
    "oldest": "oldest first",
    "least_sure": "least sure first",
}
KEY_TEXT = {"doing": "doing", "aimed": "aimed at", "feels": "feels", "tone": "tone", "spread": "spread"}
ABOUT_TEXT = {"doing": "doing", "aimed_at": "aimed at", "feeling": "feeling", "tone": "tone"}
BY_TEXT = {"value": "value", "channel": "channel", "stream": "stream", "period": "this time vs the rest", "day": "day"}


def blank() -> dict[str, Any]:
    """The widest query: every Moment of the last stream, biggest first."""
    return {
        "intent": "find_moments", "follow_up": "new", "doing": [], "aimed": [], "feels": [], "tone": [],
        "spread": [], "is": [], "emote": [], "phrase": None, "channel": [], "time": "last_stream", "at": None,
        "min_posters": None, "sort": "posters", "limit": LIMIT_DEFAULT, "ref": None, "about": None, "by": None,
        "asks_cause": False, "reason": None,
    }  # fmt: skip


# --- what the store holds ------------------------------------------------------------------------


@dataclass(frozen=True)
class Held:
    """Channels with windows, and public emote names with their ``(channel, emote_id)`` pairs."""

    channels: tuple[str, ...] = ()
    emotes: dict[str, tuple[tuple[str, str], ...]] = field(default_factory=dict)

    @property
    def emote_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.emotes, key=lambda n: (n.lower(), n)))

    def emote_name(self, word: str) -> str | None:
        """The held name for ``word``: exact first, else the one name that matches ignoring case."""
        if word in self.emotes:
            return word
        found = [n for n in self.emotes if n.lower() == word.lower()]
        return found[0] if len(found) == 1 else None


def held(store: Store) -> Held:
    with store.lock:
        channels = tuple(r[0] for r in store.conn.execute("SELECT DISTINCT channel FROM windows ORDER BY channel"))
        rows = store.conn.execute("SELECT name, channel, emote_id FROM emote_names ORDER BY name, channel").fetchall()
    emotes: dict[str, list[tuple[str, str]]] = {}
    for name, channel, emote_id in rows:
        emotes.setdefault(name, []).append((channel, emote_id))
    return Held(channels, {k: tuple(v) for k, v in emotes.items()})


def allowed(scheme: Scheme, key: str) -> tuple[str, ...]:
    """The values a value key takes, from the scheme (tone includes its blank value)."""
    if key == "spread":
        return tuple(scheme.spread_values)
    q = next(x for x in scheme.questions if x.key == KEY_QUESTION[key])
    return tuple(q.values) + ((q.blank_value,) if q.blank_value else ())


# --- the JSON schema 10c sends as Ollama ``format`` --------------------------------------------


def schema(store: Store, scheme: Scheme, h: Held | None = None) -> dict[str, Any]:
    """The query object's JSON schema for this store: enums from ``labels_v1.json`` and the store."""
    h = h if h is not None else held(store)

    def arr(enum: tuple[str, ...]) -> dict[str, Any]:
        if not enum:  # B34: an empty enum is no valid schema; nothing held means the key is null
            return {"type": "null"}
        return {"type": "array", "items": {"type": "string", "enum": list(enum)}, "uniqueItems": True}

    def opt(enum: tuple[str, ...]) -> dict[str, Any]:
        return {"anyOf": [{"type": "string", "enum": list(enum)}, {"type": "null"}]}

    props: dict[str, Any] = {
        "intent": {"type": "string", "enum": list(INTENTS)},
        "follow_up": {"type": "string", "enum": list(FOLLOW_UPS)},
        **{k: arr(allowed(scheme, k)) for k in (*VALUE_KEYS, "spread")},
        "is": arr(IS_VALUES),
        "emote": arr(h.emote_names),
        "phrase": {"anyOf": [{"type": "string", "minLength": 1, "maxLength": PHRASE_MAX}, {"type": "null"}]},
        "channel": arr(h.channels),
        "time": {"type": "string", "enum": list(TIMES)},
        "at": {"anyOf": [{"type": "string", "pattern": AT_PATTERN}, {"type": "null"}]},
        "min_posters": {"anyOf": [{"type": "integer", "minimum": 1, "maximum": POSTERS_MAX}, {"type": "null"}]},
        "sort": {"type": "string", "enum": list(SORTS)},
        "limit": {"type": "integer", "minimum": LIMIT_MIN, "maximum": LIMIT_MAX},
        "ref": {"anyOf": [{"type": "integer", "minimum": 1, "maximum": LIMIT_MAX}, {"type": "null"}]},
        "about": opt(ABOUT),
        "by": opt(BY),
        "asks_cause": {"type": "boolean"},
        "reason": opt(REASONS),
    }
    return {"type": "object", "properties": {k: props[k] for k in KEYS}, "required": list(KEYS),
            "additionalProperties": False}  # fmt: skip


# --- the validator ---------------------------------------------------------------------------------


@dataclass
class Checked:
    query: dict[str, Any]
    ignored: list[str] = field(default_factory=list)  # tokens or keys dropped, shown struck
    codes: list[str] = field(default_factory=list)  # fixed strings only: safe to log
    suggestions: list[str] = field(default_factory=list)  # three runnable questions when unusable


def suggestions(h: Held) -> list[str]:
    """The five home suggestions (questions design note, home page), with a held channel and emote."""
    channel = h.channels[0] if h.channels else None
    emote = next((n for n in ("KEKW", "LUL", "Pog", "Sadge") if n in h.emotes), h.emote_names[0] if h.emotes else None)
    if emote and channel:
        fourth = f"What does {emote} mean in {channel}?"
    elif emote:
        fourth = f"What does {emote} mean?"
    else:
        fourth = "How did chat feel about the guest?"
    return [
        "What were the biggest moments last stream?",
        "What made chat laugh the most?",
        "When did chat turn on the streamer?",
        fourth,
        "Which Moments is BrainChat least sure about?",
    ]


def _int(x: Any) -> int | None:
    return x if isinstance(x, int) and not isinstance(x, bool) else None


def check(
    raw: Any, scheme: Scheme, h: Held, *, question: str | None = None, carried_phrase: str | None = None
) -> Checked:
    """Validate ``raw`` into a query (exact keys, known values, per-intent rules); fail closed.

    ``question`` is the typed text when there is one: ``phrase`` must appear in it word for word,
    unless it is ``carried_phrase``, the phrase of the earlier ask in the thread that this one
    refines (that ask's own question was checked when it was asked).
    """
    if not isinstance(raw, dict) or raw.get("intent") not in INTENTS:
        return _unusable(h, [])
    q = blank()
    out = Checked(q)
    ign, codes = out.ignored, out.codes
    for k in raw:
        if k not in KEYS:
            ign.append(str(k)[:40])
    if any(k not in KEYS for k in raw):
        codes.append("extra_keys")
    for k in KEYS:
        if k not in raw:
            codes.append(f"missing:{k}")
    q["intent"] = raw["intent"]
    fu = raw.get("follow_up", "new")
    if fu in FOLLOW_UPS:
        q["follow_up"] = fu
    else:
        ign.append(f"follow_up:{fu}")

    for k in LIST_KEYS:
        items = raw.get(k, [])
        if items is None and k in ("emote", "channel"):
            items = []  # B34: the schema asks for null when the store holds no emote or channel
        if not isinstance(items, list):
            ign.append(k)
            continue
        if k in (*VALUE_KEYS, "spread"):
            ok: tuple[str, ...] = allowed(scheme, k)
        elif k == "is":
            ok = IS_VALUES
        elif k == "channel":
            ok = h.channels
        else:
            ok = ()
        for item in items:
            if k == "emote" and isinstance(item, str):
                name = h.emote_name(item)
                if name is not None and name not in q[k]:
                    q[k].append(name)
                elif name is None:
                    ign.append(f"emote:{item}")
            elif isinstance(item, str) and item in ok:
                if item not in q[k]:
                    q[k].append(item)
            else:
                ign.append(f"{k}:{item}")

    phrase = raw.get("phrase")
    if phrase is not None:
        text = phrase.strip() if isinstance(phrase, str) else ""
        carried = carried_phrase is not None and text == carried_phrase
        inside = question is None or carried or text.lower() in " ".join(question.lower().split())
        if text and len(text) <= PHRASE_MAX and inside:
            q["phrase"] = text
        else:
            ign.append(f"phrase:{phrase}")

    t = raw.get("time", "last_stream")
    if t in TIMES:
        q["time"] = t
    else:
        ign.append(f"time:{t}")
    at = raw.get("at")
    if at is not None:
        if isinstance(at, str) and _AT.match(at):
            q["at"] = at
        else:
            ign.append(f"at:{at}")
    mp = raw.get("min_posters")
    if mp is not None:
        n = _int(mp)
        if n is None:
            ign.append(f"min_posters:{mp}")
        else:
            q["min_posters"] = min(POSTERS_MAX, max(1, n))
            if q["min_posters"] != n:
                codes.append("min_posters_clamped")
    s = raw.get("sort", "posters")
    if s in SORTS:
        q["sort"] = s
    else:
        ign.append(f"sort:{s}")
    lim = raw.get("limit", LIMIT_DEFAULT)
    n = _int(lim)
    if n is None:
        ign.append(f"limit:{lim}")
    else:
        q["limit"] = min(LIMIT_MAX, max(LIMIT_MIN, n))
        if q["limit"] != n:
            codes.append("limit_clamped")
    ref = raw.get("ref")
    if ref is not None:
        n = _int(ref)
        if n is not None and 1 <= n <= LIMIT_MAX:
            q["ref"] = n
        else:
            ign.append(f"ref:{ref}")
    for k, ok in (("about", ABOUT), ("by", BY), ("reason", REASONS)):
        v = raw.get(k)
        if v is None:
            continue
        if v in ok:
            q[k] = v
        else:
            ign.append(f"{k}:{v}")
    cause = raw.get("asks_cause", False)
    if isinstance(cause, bool):
        q["asks_cause"] = cause
    else:
        ign.append("asks_cause")

    # per-intent rules
    intent = q["intent"]
    if intent == "emote_meaning":
        if not q["emote"]:
            return _unusable(h, ign, codes)
        ign += [f"emote:{e}" for e in q["emote"][1:]]
        q["emote"] = q["emote"][:1]
    if intent == "count" and q["by"] == "channel" and len(q["channel"] or h.channels) < 2:
        ign.append("by:channel")
        q["by"] = None
    if intent == "top_moments" and q["sort"] not in COUNT_SORTS:
        ign.append(f"sort:{q['sort']}")
        q["sort"] = "posters"
    if intent == "trend":
        if q["by"] not in (None, *TREND_BY):
            ign.append(f"by:{q['by']}")
            q["by"] = None
        if q["time"] == "last_stream":
            ign.append("time:last_stream")
            q["time"] = "14d"
    if intent == "cant_answer" and q["reason"] is None:
        return _unusable(h, ign, codes)
    return out


def _unusable(h: Held, ignored: list[str], codes: list[str] | None = None) -> Checked:
    q = blank() | {"intent": "cant_answer"}
    s = suggestions(h)
    return Checked(q, list(ignored), [*(codes or []), "unusable"], [s[0], s[1], s[4]])


# --- tokens and chips ------------------------------------------------------------------------------


def tokens(query: dict[str, Any]) -> list[str]:
    """The query as ``key:value`` tokens: every list item, and every scalar that isn't blank.

    ``intent``, ``time`` and ``sort`` are always written.
    """
    b = blank()
    out = [f"intent:{query['intent']}"]
    for k in KEYS:
        v = query.get(k)
        if k == "intent":
            continue
        if k in LIST_KEYS:
            out += [f"{k}:{x}" for x in v or []]
        elif k in ("time", "sort"):
            out.append(f"{k}:{v}")
        elif k == "asks_cause":
            if v:
                out.append("asks_cause:1")
        elif v is not None and v != b[k]:
            out.append(f"{k}:{v}")
    return out


_INT_KEYS = ("min_posters", "limit", "ref")


def from_tokens(toks: list[Any], scheme: Scheme, h: Held) -> Checked:
    """Rebuild and check a query from tokens; a token it can't read is ignored (struck)."""
    raw = blank()
    bad: list[str] = []
    for t in toks:
        if not isinstance(t, str) or ":" not in t:
            bad.append(str(t)[:60])
            continue
        k, v = t.split(":", 1)
        if k not in KEYS:
            bad.append(t)
        elif k in LIST_KEYS:
            raw[k].append(v)
        elif k in _INT_KEYS:
            if v.isdigit():
                raw[k] = int(v)
            else:
                bad.append(t)
        elif k == "asks_cause":
            raw[k] = v in ("1", "true")
        else:
            raw[k] = v
    got = check(raw, scheme, h)
    got.ignored = bad + got.ignored
    return got


@dataclass(frozen=True)
class Chip:
    token: str
    text: str
    kind: str = "filter"  # "filter", "ignored" (struck) or "didnt" (struck: a word not understood)
    kept: bool = False  # carried over from the previous ask


def chip_text(token: str, scheme: Scheme) -> str:
    k, _, v = token.partition(":")
    d = scheme.display
    if k in KEY_TEXT:
        return f"{KEY_TEXT[k]}: {d.get(v, v)}"
    if k == "is":
        return f"is: {IS_TEXT.get(v, v)}"
    if k == "time":
        return f"time: {TIME_TEXT.get(v, v)}"
    if k == "sort":
        return f"sort: {SORT_TEXT.get(v, v)}"
    if k == "phrase":
        return f"said: “{v}” · only windows whose text is still kept"
    if k == "min_posters":
        return f"posters joining: {v}+"
    if k == "limit":
        return f"show: {v}"
    if k == "ref":
        return f"result: #{v}"
    if k == "about":
        return f"about: {ABOUT_TEXT.get(v, v)}"
    if k == "by":
        return f"by: {BY_TEXT.get(v, v)}"
    if k == "asks_cause":
        return "asks why"
    return f"{k}: {v}" if v else k


_NO_CHIP = ("intent", "follow_up", "reason")


def chips(
    query: dict[str, Any],
    scheme: Scheme,
    *,
    kept: list[str] | tuple[str, ...] = (),
    ignored: list[str] | tuple[str, ...] = (),
    didnt: list[str] | tuple[str, ...] = (),
) -> list[Chip]:
    """The "Looked for" row: the query's tokens in words, then struck ignored tokens and words."""
    out = [
        Chip(t, chip_text(t, scheme), "filter", t in kept)
        for t in tokens(query)
        if t.split(":", 1)[0] not in _NO_CHIP and not (t == "limit:5")
    ]
    out += [Chip(t, f"ignored: {t}", "ignored") for t in ignored]
    out += [Chip("", f"didn't understand: {w}", "didnt") for w in didnt]
    return out
