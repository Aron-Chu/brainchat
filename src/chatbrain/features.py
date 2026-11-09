"""Model inputs for rungs 0 and 1 (spec section 8).

- ``counts_row`` (rung 0) turns a window's ``Counts`` into named numbers: the message
  count, distinct posters, rate and shares, plus one ``emote:<id>`` count per emote ID.
  It holds no words, so it outlives the text like ``Counts`` does.
- ``text_doc`` (rung 1) turns a window's messages into one document, one line per
  message with whitespace collapsed. ``@s<N>`` (a reply to someone who posted in the
  window) becomes ``@someone``, so a per-window speaker number never becomes a word.
  With ``emote_mode="name"`` (the spec section 8 baseline) emote names stay as words;
  with ``"id"`` each emote token becomes ``emote_<id>``, matched the way ``spread``
  matches it (a token equal to one of that message's emote names).
- ``message_doc`` is the same transform for one message (rung-1 evidence scores lines alone).
"""

from __future__ import annotations

import re
from typing import Literal

from .export_format import Message, Window
from .spread import Counts

EmoteMode = Literal["name", "id"]
EMOTE_MODES: tuple[str, ...] = ("name", "id")

COUNT_FEATURES: tuple[str, ...] = (
    "n_messages",
    "distinct_posters",
    "msg_per_min",
    "top_emote_share",
    "chant_share",
    "mention_share",
    "question_share",
)
EMOTE_PREFIX = "emote:"
SOMEONE = "@someone"

_REPLY = re.compile(r"@s\d+", re.ASCII)  # the same pattern spread counts as a reply


def counts_row(c: Counts) -> dict[str, float]:
    """The rung-0 features of one window: the seven counted numbers and ``emote:<id>`` uses."""
    row = {name: float(getattr(c, name)) for name in COUNT_FEATURES}
    for emote_id, uses in c.emote_counts.items():
        row[EMOTE_PREFIX + emote_id] = float(uses)
    return row


def text_doc(w: Window, emote_mode: EmoteMode = "name") -> str:
    """The rung-1 document of one window: one line per message (``""`` for no messages)."""
    _check_mode(emote_mode)
    return "\n".join(message_doc(m, emote_mode) for m in w.messages)


def message_doc(m: Message, emote_mode: EmoteMode = "name") -> str:
    """One message as a line of ``text_doc``."""
    _check_mode(emote_mode)
    ids: dict[str, str] = {}
    if emote_mode == "id":
        for e in m.emotes:
            ids.setdefault(e.name, e.id)
    return " ".join(f"emote_{ids[t]}" if t in ids else _REPLY.sub(SOMEONE, t) for t in m.text.split())


def _check_mode(emote_mode: str) -> None:
    if emote_mode not in EMOTE_MODES:
        raise ValueError(f"emote_mode must be one of {', '.join(EMOTE_MODES)}")
