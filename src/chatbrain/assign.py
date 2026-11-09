"""Blind assignment, delayed repeats and rare-value picks (spec section 6).

Exposure: every window is assigned ``blind`` (1 in 4) or ``suggested`` before any
suggestion exists, and the assignment is recorded in ``assignments`` with its seed.
The split is a hash, not a random draw, so the same seed always gives the same
split: a window is blind iff
``int(sha256(f"{seed}:{window_id}").hexdigest(), 16) % 10000 < blind_rate * 10000``
(the text is hashed as UTF-8). ``assign_new`` writes a row only for windows that
still have text and have no row yet. A row never changes afterwards, so windows a
pilot run assigned with ``all_blind=True`` stay blind when later windows are
assigned by hash.

Repeats: 1 in 10 windows the owner labeled in the blind or suggested project goes to the
Repeat project, due 3 to 7 days after their first such label and before its text
expires. With ``h = int(sha256(f"{seed}:repeat:{window_id}").hexdigest(), 16)``, a
window is picked iff ``h % 10000 < rate * 10000`` and is due at
``first label's created_at + (3 + h % 5) days``; ``created_at`` is the Label Studio
annotation time (ruling C35). A window is skipped while its due date is at or after
its ``text_expires_at``, and when its text is already gone at ``now``. Synthetic
labels, page fixes and repeat-project labels never schedule a repeat, and a window
that already has a repeat row or one of the owner's repeat-project labels gets no other.
A repeat row never changes afterwards.

Rare picks: ``pick_rare`` marks windows picked on purpose for rare values
(``windows.picked_rare``); metrics keep them out of headline numbers (ruling C21).
The mark outlives the text.

Every write runs in one ``Store.transaction()``. Nothing here reads chat text, and
errors never carry window IDs.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Literal

from .export_format import iso, parse_dt
from .store import Store

Exposure = Literal["blind", "suggested"]

BLIND_RATE = 0.25
REPEAT_RATE = 0.10
REPEAT_MIN_DAYS = 3
REPEAT_SPREAD_DAYS = 5  # due 3 + h % 5 days after the first label: 3..7
_BUCKETS = 10_000


class UnknownWindows(ValueError):
    """``pick_rare`` was given window IDs the store doesn't hold; nothing was marked."""


def exposure(window_id: str, seed: str, blind_rate: float = BLIND_RATE) -> Exposure:
    """``blind`` or ``suggested`` for ``window_id`` under ``seed``: the same answer every time."""
    _check_rate(blind_rate, "blind_rate")
    return "blind" if _hash(f"{seed}:{window_id}") % _BUCKETS < blind_rate * _BUCKETS else "suggested"


def assign_new(store: Store, seed: str, now: datetime, all_blind: bool = False) -> int:
    """Assign every window that has text at ``now`` and no assignment yet; return how many.

    ``all_blind`` (pilot mode) makes each new assignment blind. Existing rows never
    change. Windows whose text has expired (``text_expires_at <= now``) are left
    unassigned, since nobody can label them. A naive ``now`` raises ``ValueError``.
    """
    now_s = iso(now)
    with store.transaction() as conn:
        window_ids = [
            r[0]
            for r in conn.execute(
                "SELECT w.window_id FROM windows w"
                " WHERE w.text_state = 'present' AND w.text_expires_at > ?"
                " AND NOT EXISTS (SELECT 1 FROM assignments a WHERE a.window_id = w.window_id)"
                " ORDER BY w.window_id",
                (now_s,),
            )
        ]
        conn.executemany(
            "INSERT INTO assignments (window_id, exposure, assigned_at, seed) VALUES (?, ?, ?, ?)",
            [(wid, "blind" if all_blind else exposure(wid, seed), now_s, seed) for wid in window_ids],
        )
    return len(window_ids)


def schedule_repeats(store: Store, seed: str, now: datetime, rate: float = REPEAT_RATE) -> int:
    """Schedule repeats for windows the owner labeled in the blind or suggested project; return how many.

    See the module docstring for the pick and the due date. Windows with no pick,
    or whose due date falls at or after their ``text_expires_at``, get no row and
    are looked at again (with the same result) on the next run. A naive ``now`` or a
    ``rate`` outside 0..1 raises ``ValueError``.
    """
    now_s = iso(now)
    _check_rate(rate, "rate")
    with store.transaction() as conn:
        candidates = conn.execute(
            "SELECT l.window_id, MIN(l.created_at), w.text_expires_at FROM labels l"
            " JOIN windows w ON w.window_id = l.window_id"
            " WHERE l.origin = 'aron' AND l.project IN ('blind', 'suggested')"
            " AND w.text_state = 'present' AND w.text_expires_at > ?"
            " AND NOT EXISTS (SELECT 1 FROM repeats r WHERE r.window_id = l.window_id)"
            " AND NOT EXISTS (SELECT 1 FROM labels p WHERE p.window_id = l.window_id"
            " AND p.origin = 'aron' AND p.project = 'repeat')"
            " GROUP BY l.window_id ORDER BY l.window_id",
            (now_s,),
        ).fetchall()
        rows: list[tuple[str, str]] = []
        for window_id, first_s, expires_s in candidates:
            h = _hash(f"{seed}:repeat:{window_id}")
            if h % _BUCKETS >= rate * _BUCKETS:
                continue
            days = REPEAT_MIN_DAYS + h % REPEAT_SPREAD_DAYS
            due_at = parse_dt(first_s, "labels.created_at") + timedelta(days=days)
            if due_at >= parse_dt(expires_s, "windows.text_expires_at"):
                continue
            rows.append((window_id, iso(due_at)))
        conn.executemany("INSERT INTO repeats (window_id, due_at) VALUES (?, ?)", rows)
    return len(rows)


def pick_rare(store: Store, window_ids: Iterable[str]) -> int:
    """Mark windows picked on purpose for rare values; return how many were newly marked.

    All or nothing: if any ID is not in the store, ``UnknownWindows`` is raised (with
    counts only) and nothing is marked. Repeated IDs count once, and a window that is
    already marked counts 0.
    """
    if isinstance(window_ids, (str, bytes)):
        raise TypeError("pick_rare takes a list of window IDs, not one string")
    wanted = sorted(set(window_ids))
    with store.transaction() as conn:
        missing = sum(
            conn.execute("SELECT 1 FROM windows WHERE window_id = ?", (wid,)).fetchone() is None for wid in wanted
        )
        if missing:
            raise UnknownWindows(
                f"{missing} of {len(wanted)} window ID(s) are not in the store; nothing was marked"
            )
        return sum(
            conn.execute("UPDATE windows SET picked_rare = 1 WHERE window_id = ? AND picked_rare = 0", (wid,)).rowcount
            for wid in wanted
        )


def totals(store: Store) -> dict[str, int]:
    """Counts for the terminal: ``blind``, ``suggested``, ``repeats`` and ``picked_rare``."""
    with store.lock:
        c = store.conn
        out = {"blind": 0, "suggested": 0}
        out.update(dict(c.execute("SELECT exposure, COUNT(*) FROM assignments GROUP BY exposure").fetchall()))
        out["repeats"] = c.execute("SELECT COUNT(*) FROM repeats").fetchone()[0]
        out["picked_rare"] = c.execute("SELECT COUNT(*) FROM windows WHERE picked_rare = 1").fetchone()[0]
    return out


def _hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


def _check_rate(rate: float, name: str) -> None:
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or math.isnan(rate) or not 0 <= rate <= 1:
        raise ValueError(f"{name} must be a number from 0 to 1")
