"""F1 "top emotes typed": each Moment's top emote, counted by exact emote id (glance calls 2 and 3).

Pure and stdlib-only, like ``spread`` and ``page.activity``. ``top_emote`` reads one window's
messages while their text is present; the store keeps the result as a count in
``moment_emotes`` (no text, no speaker), which lives as long as the stream's activity counts.

Counting:

- A line counts once per emote id it carries (the set of its ``emotes[].id``), so one person
  repeating an emote in a line, or over many lines, can't win alone.
- People are the distinct speakers who sent that id in this window only (speakers are per-window
  numbers and never link across windows).
- The winner has the most lines, then the most people, then the earliest first line (messages in
  the store's order, ``(rel_ms, message_id)``), then the smaller id as a last resort.
- ``name`` is the name at the winner's first line. Names are never matched back to ids, and a text
  token with no ``emotes[]`` entry is never counted.

``shows`` is the bar a face must pass (call 3): the winner is in at least 10% of the window's
lines and was sent by at least 5 people. Only the winner is ever kept, so a weaker emote never
shows in its place. ``face_picture`` is the later picture stage's seam; it gives nothing yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .export_format import Window

MIN_SHARE = Fraction(1, 10)
MIN_PEOPLE = 5


@dataclass(frozen=True)
class Tally:
    """A Moment's top emote: in ``lines`` of the window's ``of_lines`` lines, sent by ``people`` people."""

    emote_id: str
    name: str
    lines: int
    of_lines: int
    people: int


def top_emote(w: Window) -> Tally | None:
    """The window's top emote by lines, or None when no line carries an emote id."""
    messages = sorted(w.messages, key=lambda m: (m.rel_ms, m.message_id))
    lines: dict[str, int] = {}
    people: dict[str, set[str]] = {}
    first: dict[str, int] = {}
    names: dict[str, str] = {}
    for i, m in enumerate(messages):
        seen: set[str] = set()
        for e in m.emotes:
            if e.id in seen:
                continue
            seen.add(e.id)
            lines[e.id] = lines.get(e.id, 0) + 1
            people.setdefault(e.id, set()).add(m.speaker)
            if e.id not in first:
                first[e.id] = i
                names[e.id] = e.name
    if not lines:
        return None
    best = min(lines, key=lambda k: (-lines[k], -len(people[k]), first[k], k))
    return Tally(best, names[best], lines[best], len(w.messages), len(people[best]))


def shows(t: Tally) -> bool:
    """True when the tally passes the face bar: >= 10% of the lines and >= 5 people (exact integers)."""
    return t.lines * MIN_SHARE.denominator >= t.of_lines * MIN_SHARE.numerator and t.people >= MIN_PEOPLE


def face_picture(provider: str | None, emote_id: str) -> None:
    """The seam for the later picture stage (call 1 ★c, deferred by G-13): a picture of the exact
    emote ``emote_id`` from ``provider``, fetched on this PC and kept 24 h. That stage is blocked on
    exact provider ids (the export has no provider, and its ids are StreamPulse's own), so this
    milestone has no picture: it always returns None, the page draws every face as a text chip,
    and no template calls it."""
    return None
