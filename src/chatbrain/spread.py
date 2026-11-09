"""Counted spread and participation for one window.

Spread is counted, never labeled (spec section 4): a deterministic function of the
window's messages that is shown to help and is never a training target. ``Counts``
holds only numbers, emote IDs and the spread value, so it outlives the text (spec
section 7). The chant text is used while counting and then dropped, because it is
words.

Rules, applied in this order (n = the window's messages; every share is a share of
messages, compared exactly as a fraction):

1. ``few_posters`` if distinct posters < 5;
2. ``chant`` if chant_share >= 0.30;
3. ``repeated_emotes`` if top_emote_share >= 0.40;
4. ``back_and_forth`` if mention_share >= 0.20;
5. ``separate_messages`` otherwise.

Definitions:

- A message is normalised by splitting its text on whitespace (which collapses and
  trims it), writing each token that is one of the message's emote names as
  ``emote_<id>`` and lowercasing every other token.
- The chant text is the most common normalised message with at least 2 tokens, and
  it must occur in at least 2 messages: a line said once is not a chant. Ties go to
  the smaller text. chant_share is the share of messages equal to it (0 when there
  is none).
- The top emote is the emote ID found in the most messages; ties go to the smaller
  ID. top_emote_share is the share of messages that contain it. ``emote_counts``
  counts uses (one per entry in a message's ``emotes``, which has one entry per
  occurrence), so ``KEKW KEKW`` adds 2 to the count but 1 message to the share.
- mention_share is the share of messages containing ``@s<N>``, a reply to someone who
  posted in the window. ``@user`` names someone who didn't, so it is not a reply.
- question_share is the share of messages containing ``?``.
- msg_per_min is n / 2.0 (the window is 2 minutes).
- Joining lines are the messages that contain the top emote or equal the chant text,
  each counted once. Like the chant, the top emote counts only when it is in at least
  2 messages: an emote one person used once is nobody joining (ruling B2). Joining
  posters are the distinct speakers among them.

The empty window gets ``few_posters`` and zero counts. A window with one message
gets ``few_posters`` with its true counts (ruling C32).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field, fields
from fractions import Fraction
from typing import Any

from .export_format import Message, Window

# The five spread values, in the label scheme's order (config/labels_v1.json).
SPREAD_VALUES: tuple[str, ...] = ("repeated_emotes", "chant", "back_and_forth", "separate_messages", "few_posters")

MIN_POSTERS = 5
CHANT_MIN_SHARE = Fraction(3, 10)
REPEATED_EMOTES_MIN_SHARE = Fraction(2, 5)
BACK_AND_FORTH_MIN_SHARE = Fraction(1, 5)
CHANT_MIN_TOKENS = 2
CHANT_MIN_LINES = 2
EMOTE_JOIN_MIN_LINES = 2  # the top emote counts as joining only from this many messages (B2)
WINDOW_MINUTES = 2.0

_REPLY = re.compile(r"@s\d+", re.ASCII)
_INT_FIELDS = ("n_messages", "distinct_posters", "joining_lines", "joining_posters")
_SHARE_FIELDS = ("top_emote_share", "chant_share", "mention_share", "question_share")


@dataclass(frozen=True)
class Counts:
    """A window's counted features. Never holds words: numbers, emote IDs and the spread value only."""

    n_messages: int
    distinct_posters: int
    msg_per_min: float
    top_emote_id: str | None
    top_emote_share: float
    chant_share: float
    mention_share: float
    question_share: float
    emote_counts: dict[str, int] = field(hash=False)  # emote ID -> uses in the window
    spread: str
    joining_lines: int
    joining_posters: int

    def to_json(self) -> str:
        """Compact JSON with sorted keys, so the same counts give the same text."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_json(cls, s: str) -> Counts:
        """Read ``to_json`` output back. Raises ``ValueError`` for anything else.

        Messages name the field and the rule, never the value: a bad value could be words.
        """
        try:
            doc = json.loads(s, object_pairs_hook=_no_duplicate_keys, parse_constant=_no_constants)
        except (TypeError, json.JSONDecodeError):
            raise ValueError("counts: not valid JSON text") from None
        if not isinstance(doc, dict):
            raise ValueError("counts: must be a JSON object")
        names = {f.name for f in fields(cls)}
        missing = sorted(names - doc.keys())
        extra = len(doc.keys() - names)
        if missing or extra:
            raise ValueError(f"counts: missing keys {missing}, {extra} unknown key(s)")
        for name in _INT_FIELDS:
            if type(doc[name]) is not int or doc[name] < 0:
                raise ValueError(f"counts.{name}: must be a whole number >= 0")
        if not _real(doc["msg_per_min"]) or doc["msg_per_min"] < 0:
            raise ValueError("counts.msg_per_min: must be a number >= 0")
        for name in _SHARE_FIELDS:
            if not _real(doc[name]) or not 0.0 <= doc[name] <= 1.0:
                raise ValueError(f"counts.{name}: must be a number from 0 to 1")
        top = doc["top_emote_id"]
        if top is not None and (not isinstance(top, str) or not top):
            raise ValueError("counts.top_emote_id: must be an emote ID or null")
        emote_counts = doc["emote_counts"]
        if not isinstance(emote_counts, dict) or any(type(v) is not int or v < 1 for v in emote_counts.values()):
            raise ValueError("counts.emote_counts: must map emote IDs to whole numbers >= 1")
        if doc["spread"] not in SPREAD_VALUES:
            raise ValueError(f"counts.spread: must be one of {', '.join(SPREAD_VALUES)}")
        return cls(**doc)


def compute(w: Window) -> Counts:
    """Count a window's spread and participation (rules in the module docstring)."""
    msgs = w.messages
    n = len(msgs)
    posters = {m.speaker for m in msgs}

    emote_counts = Counter(e.id for m in msgs for e in m.emotes)
    emote_lines = Counter(eid for m in msgs for eid in {e.id for e in m.emotes})
    top_emote_id = min(emote_lines, key=lambda eid: (-emote_lines[eid], eid)) if emote_lines else None
    top_lines = emote_lines[top_emote_id] if top_emote_id is not None else 0

    normalised = [_normalise(m) for m in msgs]
    phrases = Counter(" ".join(tokens) for tokens in normalised if len(tokens) >= CHANT_MIN_TOKENS)
    chant_text = min(phrases, key=lambda t: (-phrases[t], t)) if phrases else None
    if chant_text is not None and phrases[chant_text] < CHANT_MIN_LINES:
        chant_text = None
    chant_lines = phrases[chant_text] if chant_text is not None else 0

    mention_lines = sum(1 for m in msgs if _REPLY.search(m.text))
    question_lines = sum(1 for m in msgs if "?" in m.text)

    joining_emote = top_emote_id if top_lines >= EMOTE_JOIN_MIN_LINES else None
    joining = [
        m
        for m, tokens in zip(msgs, normalised)
        if (joining_emote is not None and any(e.id == joining_emote for e in m.emotes))
        or (chant_text is not None and " ".join(tokens) == chant_text)
    ]

    if len(posters) < MIN_POSTERS:
        spread = "few_posters"
    elif _at_least(chant_lines, n, CHANT_MIN_SHARE):
        spread = "chant"
    elif _at_least(top_lines, n, REPEATED_EMOTES_MIN_SHARE):
        spread = "repeated_emotes"
    elif _at_least(mention_lines, n, BACK_AND_FORTH_MIN_SHARE):
        spread = "back_and_forth"
    else:
        spread = "separate_messages"

    return Counts(
        n_messages=n,
        distinct_posters=len(posters),
        msg_per_min=n / WINDOW_MINUTES,
        top_emote_id=top_emote_id,
        top_emote_share=_share(top_lines, n),
        chant_share=_share(chant_lines, n),
        mention_share=_share(mention_lines, n),
        question_share=_share(question_lines, n),
        emote_counts=dict(sorted(emote_counts.items())),
        spread=spread,
        joining_lines=len(joining),
        joining_posters=len({m.speaker for m in joining}),
    )


def _normalise(m: Message) -> tuple[str, ...]:
    """Tokens of a message: whitespace collapsed, emotes as ``emote_<id>``, other tokens lowercased."""
    ids: dict[str, str] = {}
    for e in m.emotes:
        ids.setdefault(e.name, e.id)
    return tuple(f"emote_{ids[t]}" if t in ids else t.lower() for t in m.text.split())


def _at_least(k: int, n: int, share: Fraction) -> bool:
    """``k / n >= share``, exactly (no float rounding at the boundary)."""
    return n > 0 and k * share.denominator >= share.numerator * n


def _share(k: int, n: int) -> float:
    return k / n if n else 0.0


def _real(v: Any) -> bool:
    return type(v) is float and math.isfinite(v)


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("counts: a JSON object repeats a key")
        out[key] = value
    return out


def _no_constants(name: str) -> Any:
    raise ValueError("counts: NaN and Infinity are not allowed")
