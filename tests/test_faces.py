"""F1 data: each Moment's top emote, counted by exact emote id while the text is present.

Made-up windows only (``fake-<name>`` ids, speakers ``s<N>``).
"""

import dataclasses
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pagekit
from chatbrain import faces, ingest
from chatbrain.export_format import EXPORT_FORMAT, Emote, Export, Message, Window, dump_export, iso, parse_dt
from chatbrain.faces import Tally
from chatbrain.store import ACTIVITY_LIFETIME, Store

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)
START = NOW - timedelta(days=3)
T = NOW + timedelta(days=5)
K = Emote("fake-kekw", "KEKW")
L = Emote("fake-lul", "LUL")
P = Emote("fake-pog", "Pog")


def _msg(i: int, speaker: str, *emotes: Emote, rel_ms: int | None = None, text: str | None = None) -> Message:
    return Message(
        f"m{i:03d}",
        31_000 + i * 100 if rel_ms is None else rel_ms,
        speaker,
        text if text is not None else (" ".join(e.name for e in emotes) or "hi chat"),
        tuple(emotes),
    )


def _win(window_id: str = "w-1", messages=(), *, kind: str = "moment", start: datetime = START) -> Window:
    return Window(
        window_id=window_id,
        channel="fakechan01",
        stream_id="fs-01",
        anchor_kind=kind,  # type: ignore[arg-type]
        anchor_at=start + timedelta(seconds=30),
        start_at=start,
        end_at=start + timedelta(seconds=120),
        language="en",
        text_expires_at=T,
        messages=tuple(messages),
    )


def _import(store: Store, tmp_path: Path, *windows: Window, name: str = "e.json", now: datetime = NOW):
    src = tmp_path / "inbox" / name
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(dump_export(Export(EXPORT_FORMAT, NOW, "vod", tuple(windows))))
    return ingest.import_export(store, src, tmp_path / "imported", now)


def _rows(store: Store) -> list[tuple]:
    return store.conn.execute(
        "SELECT window_id, stream_id, emote_id, name, lines, of_lines, people, expires_at FROM moment_emotes"
        " ORDER BY window_id"
    ).fetchall()


# --- top_emote -----------------------------------------------------------------------------------


def test_line_counts_once_per_emote():
    # s1 types KEKW three times in one line, then once more: 2 lines, 1 person.
    # LUL is in 3 lines, all from s2: the most lines wins, and one speaker is one person.
    w = _win(messages=(
        _msg(1, "s1", K, K, K),
        _msg(2, "s1", K),
        _msg(3, "s2", L),
        _msg(4, "s2", L, L),
        _msg(5, "s2", L),
        _msg(6, "s3"),
    ))  # fmt: skip
    assert faces.top_emote(w) == Tally("fake-lul", "LUL", lines=3, of_lines=6, people=1)
    only_k = _win(messages=(_msg(1, "s1", K, K, K), _msg(2, "s2")))
    assert faces.top_emote(only_k) == Tally("fake-kekw", "KEKW", lines=1, of_lines=2, people=1)


def test_tie_more_people_then_first():
    # Equal lines: more people wins, even when the other came first.
    w = _win(messages=(_msg(1, "s1", K), _msg(2, "s1", K), _msg(3, "s2", L), _msg(4, "s3", L)))
    assert faces.top_emote(w) == Tally("fake-lul", "LUL", lines=2, of_lines=4, people=2)

    # Equal lines and people: the emote whose first line came first wins, by (rel_ms, message_id),
    # whatever order the messages are listed in.
    tied = (_msg(1, "s1", L, rel_ms=40_000), _msg(2, "s2", L, rel_ms=41_000),
            _msg(3, "s3", P, rel_ms=35_000), _msg(4, "s4", P, rel_ms=50_000))  # fmt: skip
    assert faces.top_emote(_win(messages=tied)).emote_id == "fake-pog"
    assert faces.top_emote(_win(messages=tuple(reversed(tied)))).emote_id == "fake-pog"
    same_ms = (_msg(9, "s1", L, rel_ms=40_000), _msg(8, "s2", P, rel_ms=40_000))  # m008 sorts before m009
    assert faces.top_emote(_win(messages=same_ms)).emote_id == "fake-pog"


def test_counts_by_id_not_name():
    a = Emote("fake-kekw-a", "KEKW")
    b = Emote("fake-kekw-b", "KEKW")
    renamed = Emote("fake-kekw-a", "KEKWide")  # same id, a later name
    w = _win(messages=(
        _msg(1, "s1", a),
        _msg(2, "s2", b),
        _msg(3, "s3", renamed),
        # A text token that matches a name but has no emotes[] entry is never counted.
        _msg(4, "s4", text="LUL LUL"),
        _msg(5, "s5", text="LUL"),
        _msg(6, "s6", text="KEKW LUL"),
    ))  # fmt: skip
    # Two lines carry id fake-kekw-a, one carries fake-kekw-b; the name is the one at its first line.
    assert faces.top_emote(w) == Tally("fake-kekw-a", "KEKW", lines=2, of_lines=6, people=2)
    no_ids = _win(messages=(_msg(1, "s1", text="KEKW"), _msg(2, "s2", text="LUL KEKW")))
    assert faces.top_emote(no_ids) is None


def test_below_share_or_people_not_shown():
    assert faces.MIN_PEOPLE == 5
    assert faces.shows(Tally("fake-lul", "LUL", lines=3, of_lines=30, people=5))  # exactly 10% and 5 people
    assert not faces.shows(Tally("fake-lul", "LUL", lines=2, of_lines=30, people=5))  # A2: below share
    assert not faces.shows(Tally("fake-lul", "LUL", lines=29, of_lines=300, people=29))
    assert not faces.shows(Tally("fake-lul", "LUL", lines=10, of_lines=30, people=4))  # A3: below people
    assert faces.shows(Tally("fake-lul", "LUL", lines=10, of_lines=10, people=5))


def test_runner_up_never_shown(tmp_path):
    # KEKW wins on lines (10 lines from 4 people) and is below the people bar; LUL (9 lines from
    # 9 people) would pass it, but only the winner is kept, so no face is ever a weaker one.
    msgs = [_msg(i, f"s{i % 4 + 1}", K) for i in range(10)]
    msgs += [_msg(10 + i, f"s{10 + i}", L) for i in range(9)]
    w = _win(messages=msgs)
    t = faces.top_emote(w)
    assert t == Tally("fake-kekw", "KEKW", lines=10, of_lines=19, people=4)
    assert not faces.shows(t)

    store = Store(tmp_path / "cb.db")
    _import(store, tmp_path, w)
    assert [r[2] for r in _rows(store)] == ["fake-kekw"]  # one row per Moment, the winner only
    assert store.moment_faces("fs-01", NOW) == {"w-1": t}
    assert not any(faces.shows(x) for x in store.moment_faces("fs-01", NOW).values())


def test_empty_window_no_row(tmp_path):
    empty = _win("w-1", ())
    no_emote = _win("w-2", (_msg(1, "s1"),))
    one_line = _win("w-3", (_msg(1, "s1", K),))
    ordinary = _win("w-4", [_msg(i, f"s{i + 1}", K) for i in range(8)], kind="ordinary")
    assert faces.top_emote(empty) is None and faces.top_emote(no_emote) is None
    assert faces.top_emote(one_line) == Tally("fake-kekw", "KEKW", lines=1, of_lines=1, people=1)
    assert not faces.shows(faces.top_emote(one_line))

    store = Store(tmp_path / "cb.db")
    _import(store, tmp_path, empty, no_emote, one_line, ordinary)
    # No emote, no row; an ordinary window never gets one; a one-line Moment keeps its count but shows nothing.
    assert [r[0] for r in _rows(store)] == ["w-3"]
    assert {k: faces.shows(v) for k, v in store.moment_faces("fs-01", NOW).items()} == {"w-3": False}


# --- store: write, read, backfill ------------------------------------------------------------------


def test_import_writes_tally_with_proxy_deadline_and_reads_live_only(tmp_path):
    store = Store(tmp_path / "cb.db")
    busy = _win("w-1", [_msg(i, f"s{i + 1}", K) for i in range(6)] + [_msg(9, "s9", L)])
    later = _win("w-2", [_msg(i, f"s{i + 1}", L) for i in range(5)], start=START + timedelta(hours=1))
    _import(store, tmp_path, busy, later)
    proxy = iso(START + ACTIVITY_LIFETIME)  # no activity counts: the earliest window start + 30 days
    assert _rows(store) == [
        ("w-1", "fs-01", "fake-kekw", "KEKW", 6, 7, 6, proxy),
        ("w-2", "fs-01", "fake-lul", "LUL", 5, 5, 5, proxy),  # one deadline per stream
    ]
    due = START + ACTIVITY_LIFETIME
    assert set(store.moment_faces("fs-01", due - timedelta(microseconds=1))) == {"w-1", "w-2"}
    assert store.moment_faces("fs-01", due) == {}  # read filter, before any sweep
    assert store.moment_faces("fs-77", NOW) == {}
    with store.transaction() as conn:
        conn.execute("INSERT INTO stream_activity_expired (stream_id) VALUES ('fs-01')")
    assert store.moment_faces("fs-01", NOW) == {}  # a marker hides them whatever the clock says
    assert len(_rows(store)) == 2


def test_import_skips_a_stream_whose_deadline_passed(tmp_path):
    store = Store(tmp_path / "cb.db")
    w = _win("w-1", [_msg(i, f"s{i + 1}", K) for i in range(6)])
    # Text still present (a made-up long text lifetime), but the window started more than 30 days ago.
    old = dataclasses.replace(w, start_at=NOW - timedelta(days=31), anchor_at=NOW - timedelta(days=31, seconds=-30),
                              end_at=NOW - timedelta(days=31, seconds=-120))  # fmt: skip
    _import(store, tmp_path, old)
    assert store.window_ids() == ["w-1"]
    assert _rows(store) == []


def test_backfill_fills_present_text_only_and_is_idempotent(tmp_path):
    store = Store(tmp_path / "cb.db")
    a = _win("w-1", [_msg(i, f"s{i + 1}", K) for i in range(6)])
    b = _win("w-2", [_msg(i, f"s{i + 1}", L) for i in range(6)], start=START + timedelta(hours=1))
    _import(store, tmp_path, a, b)
    with store.transaction() as conn:  # a store made before the table existed
        conn.execute("DELETE FROM moment_emotes")
        conn.execute("UPDATE windows SET text_state = 'expired' WHERE window_id = 'w-2'")
        conn.execute("DELETE FROM messages WHERE window_id = 'w-2'")
    assert store.backfill_moment_emotes(NOW) == 1  # w-2's text is gone: nothing is made up
    assert [r[0] for r in _rows(store)] == ["w-1"]
    first = _rows(store)
    assert store.backfill_moment_emotes(NOW) == 0
    assert _rows(store) == first
    assert store.backfill_moment_emotes(START + ACTIVITY_LIFETIME) == 0  # nothing written past the deadline


def test_page_start_backfills(tmp_path):
    world = pagekit.build_world(tmp_path)
    expected = {
        wid: faces.top_emote(world.windows[wid])
        for wid in world.new_moments + world.old_moments
        if wid in world.store.window_ids() and faces.top_emote(world.windows[wid]) is not None
    }
    assert expected, "the made-up world has Moments with emotes"
    with world.store.transaction() as conn:
        conn.execute("DELETE FROM moment_emotes")
    pagekit.client(world)  # page start runs maintain()
    got = {**world.store.moment_faces(world.new, pagekit.NOW), **world.store.moment_faces(world.old, pagekit.NOW)}
    assert got == expected


def test_showcase_repeated_emotes_passes_the_bar(tmp_path):
    world = pagekit.build_world(tmp_path)
    shown = {
        wid: t
        for s in (world.new, world.old)
        for wid, t in world.store.moment_faces(s, pagekit.NOW).items()
        if faces.shows(t)
    }
    assert shown, "some made-up Moment passes the face bar"
    for wid, t in shown.items():  # the stored numbers equal a recount from the messages
        assert t == faces.top_emote(world.windows[wid])


def test_pagekit_helpers_keep_faces_in_step(tmp_path):
    world = pagekit.build_world(tmp_path)
    assert world.store.moment_faces(world.new, pagekit.NOW)
    shifted = pagekit.activity_line(world, world.new, shift=timedelta(days=10))
    assert pagekit.store_activity(world.store, shifted) == "written"
    deadline = iso(shifted.started_at + ACTIVITY_LIFETIME)
    stored = {r[0] for r in world.store.conn.execute(
        "SELECT expires_at FROM moment_emotes WHERE stream_id = ?", (world.new,)
    )}  # fmt: skip
    assert stored == {deadline}
    pagekit.clear_activity(world.store)
    assert world.store.moment_faces(world.new, pagekit.NOW) == {}
    assert world.store.moment_faces(world.old, pagekit.NOW) == {}
    assert world.store.conn.execute("SELECT COUNT(*) FROM moment_emotes").fetchone()[0] == 0
    # Page start counts them again as a counts-missing stream would be (A10): the earliest window
    # start + 30 days, never a deadline left over from the cleared line.
    pagekit.client(world)
    first = world.store.conn.execute("SELECT MIN(start_at) FROM windows WHERE stream_id = ?", (world.new,))
    proxy = iso(parse_dt(first.fetchone()[0], "start_at") + ACTIVITY_LIFETIME)
    stored = {r[0] for r in world.store.conn.execute(
        "SELECT expires_at FROM moment_emotes WHERE stream_id = ?", (world.new,)
    )}  # fmt: skip
    assert stored == {proxy}
