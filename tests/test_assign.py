import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from chatbrain import assign, expiry
from chatbrain.export_format import Emote, Message, Window, iso, parse_dt
from chatbrain.store import Store

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)
START = NOW - timedelta(days=1)
LATER = NOW + timedelta(days=13)  # default text_expires_at: long after any repeat is due
LABELS_VERSION = "cb_labels_2026_11_06_v1"
GUIDE_VERSION = "cb_guide_2026_11_06_v1"
PROVENANCE = {"blind": "blind", "suggested": "accepted", "repeat": "repeat", "page": "corrected-on-page"}


def _window(window_id: str, expires: datetime = LATER) -> Window:
    msgs = (
        Message(f"{window_id}-m1", 31_000, "s1", "KEKW", (Emote("fake-kekw", "KEKW"),)),
        Message(f"{window_id}-m2", 32_000, "s2", "LUL that was so bad", (Emote("fake-lul", "LUL"),)),
    )
    return Window(
        window_id=window_id,
        channel="fakechan01",
        stream_id="fs-01",
        anchor_kind="moment",
        anchor_at=START + timedelta(seconds=30),
        start_at=START,
        end_at=START + timedelta(seconds=120),
        language="en",
        text_expires_at=expires,
        messages=msgs,
    )


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "cb.db") as s:
        yield s


def _ids(n: int, prefix: str = "fake-w") -> list[str]:
    return [f"{prefix}{i:04d}" for i in range(1, n + 1)]


def _add(store: Store, *window_ids: str, expires: datetime = LATER, imported: datetime = NOW) -> None:
    with store.transaction():
        for wid in window_ids:
            assert store.insert_window(_window(wid, expires), "vod", imported)


def _labels(store: Store, rows: list[tuple[str, str, str, datetime]]) -> None:
    """Insert ``(window_id, project, origin, created_at)`` label rows in one transaction."""
    with store.transaction() as conn:
        conn.executemany(
            "INSERT INTO labels (window_id, project, origin, provenance, labels_version, guide_version,"
            " answers_json, created_at) VALUES (?, ?, ?, ?, ?, ?, '{}', ?)",
            [
                (wid, project, origin, "synthetic" if origin == "synthetic" else PROVENANCE[project],
                 LABELS_VERSION, GUIDE_VERSION, iso(created))
                for wid, project, origin, created in rows
            ],
        )


def _label(store: Store, window_id: str, *, project: str = "blind", origin: str = "aron", created: datetime) -> None:
    _labels(store, [(window_id, project, origin, created)])


def _assignments(store: Store) -> dict[str, tuple[str, str, str]]:
    rows = store.conn.execute("SELECT window_id, exposure, assigned_at, seed FROM assignments").fetchall()
    return {wid: (exp, at, seed) for wid, exp, at, seed in rows}


def _repeats(store: Store) -> dict[str, datetime]:
    return {wid: parse_dt(due) for wid, due in store.conn.execute("SELECT window_id, due_at FROM repeats")}


def _h(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


def _repeat_h(seed: str, window_id: str) -> int:
    return _h(f"{seed}:repeat:{window_id}")


# --- exposure ---------------------------------------------------------------------------------------


def test_blind_rate_near_quarter():
    ids = _ids(4000)
    blind = sum(assign.exposure(wid, "pilot-2026-11") == "blind" for wid in ids)
    assert 0.23 <= blind / len(ids) <= 0.27


def test_exposure_stable():
    ids = _ids(300)
    for wid in ids:
        expected = "blind" if _h(f"seed-a:{wid}") % 10000 < 2500 else "suggested"
        assert assign.exposure(wid, "seed-a") == expected
        assert assign.exposure(wid, "seed-a") == expected  # same input, same answer
    # The seed is part of the hash: another seed gives another split.
    assert [assign.exposure(w, "seed-a") for w in ids] != [assign.exposure(w, "seed-b") for w in ids]
    assert {assign.exposure(w, "seed-a", blind_rate=0.0) for w in ids} == {"suggested"}
    assert {assign.exposure(w, "seed-a", blind_rate=1.0) for w in ids} == {"blind"}


@pytest.mark.parametrize("rate", [-0.01, 1.01, float("nan"), True])
def test_exposure_refuses_a_rate_outside_0_to_1(rate):
    with pytest.raises(ValueError):
        assign.exposure("fake-w0001", "seed-a", blind_rate=rate)


# --- assign_new -----------------------------------------------------------------------------------


def test_assign_new_records_the_hash_exposure(store):
    ids = _ids(60)
    _add(store, *ids)
    assert assign.assign_new(store, "seed-a", NOW) == 60
    rows = _assignments(store)
    assert rows == {wid: (assign.exposure(wid, "seed-a"), iso(NOW), "seed-a") for wid in ids}
    assert {exp for exp, _, _ in rows.values()} == {"blind", "suggested"}


def test_all_blind_mode(store):
    ids = _ids(40)
    _add(store, *ids)
    assert assign.assign_new(store, "pilot", NOW, all_blind=True) == 40
    assert _assignments(store) == {wid: ("blind", iso(NOW), "pilot") for wid in ids}


def test_existing_assignment_never_changes(store):
    first, second = _ids(30, "fake-a"), _ids(10, "fake-b")
    _add(store, *first)
    assert assign.assign_new(store, "seed-a", NOW) == 30
    before = _assignments(store)
    assert any(exp == "suggested" for exp, _, _ in before.values())

    later = NOW + timedelta(days=1)
    assert assign.assign_new(store, "seed-b", later, all_blind=True) == 0
    assert assign.assign_new(store, "seed-b", later) == 0
    assert _assignments(store) == before

    _add(store, *second)
    assert assign.assign_new(store, "seed-b", later, all_blind=True) == 10
    after = _assignments(store)
    assert {wid: after[wid] for wid in first} == before
    assert {wid: after[wid] for wid in second} == {wid: ("blind", iso(later), "seed-b") for wid in second}


def test_assign_new_skips_windows_without_text(store, tmp_path):
    _add(store, "fake-old", expires=NOW + timedelta(hours=1))
    _add(store, "fake-gone")
    _add(store, "fake-live")
    # Expired by an earlier run whose clock was ahead (`expire --now`), so text_state
    # says expired while text_expires_at is still in the future.
    with store.transaction() as conn:
        conn.execute("DELETE FROM messages WHERE window_id = 'fake-gone'")
        conn.execute("UPDATE windows SET text_state = 'expired' WHERE window_id = 'fake-gone'")
    # `fake-old` expires exactly at this `now`: now == text_expires_at is expired.
    assert assign.assign_new(store, "seed-a", NOW + timedelta(hours=1)) == 1
    assert list(_assignments(store)) == ["fake-live"]


def test_assign_new_refuses_a_naive_now(store):
    _add(store, "fake-w0001")
    with pytest.raises(ValueError):
        assign.assign_new(store, "seed-a", datetime(2026, 11, 6, 12, 0))
    assert _assignments(store) == {}


# --- schedule_repeats -------------------------------------------------------------------------------


def test_repeat_due_between_3_and_7_days(store):
    seed = "seed-r"
    ids = _ids(400)
    _add(store, *ids)
    first = NOW - timedelta(hours=2)
    rows = [(wid, "blind" if i % 2 else "suggested", "aron", first) for i, wid in enumerate(ids)]
    # A later label on the same window doesn't move the due date.
    rows += [(wid, "suggested", "aron", first + timedelta(days=1)) for wid in ids[::3]]
    _labels(store, rows)

    n = assign.schedule_repeats(store, seed, NOW)

    picked = [wid for wid in ids if _repeat_h(seed, wid) % 10000 < 1000]
    expected = {wid: first + timedelta(days=3 + _repeat_h(seed, wid) % 5) for wid in picked}
    assert n == len(expected)
    assert _repeats(store) == expected
    assert {(due - first) for due in expected.values()} == {timedelta(days=d) for d in range(3, 8)}
    assert 0.06 <= n / len(ids) <= 0.14


def test_repeat_skipped_when_text_expires_first(store):
    seed = "seed-x"
    first = NOW - timedelta(hours=1)
    a, b = [wid for wid in _ids(500) if _repeat_h(seed, wid) % 10000 < 1000][:2]
    due_a = first + timedelta(days=3 + _repeat_h(seed, a) % 5)
    due_b = first + timedelta(days=3 + _repeat_h(seed, b) % 5)
    _add(store, a, expires=due_a)  # due exactly when the text expires: skipped
    _add(store, b, expires=due_b + timedelta(seconds=1))  # due a second before: scheduled
    _label(store, a, created=first)
    _label(store, b, created=first)

    assert assign.schedule_repeats(store, seed, NOW) == 1
    assert _repeats(store) == {b: due_b}


def test_repeat_skipped_when_text_already_expired(store):
    seed = "seed-x"
    old, gone = [w for w in _ids(500) if _repeat_h(seed, w) % 10000 < 1000][:2]
    first = NOW - timedelta(days=10)
    # Labeled ten days ago and due 3-7 days later, so the due date alone would pass;
    # but the text expired a day ago.
    _add(store, old, expires=NOW - timedelta(days=1), imported=first - timedelta(hours=1))
    _label(store, old, created=first)
    # Expired by an earlier run whose clock was ahead (`expire --now`): text_state says
    # expired while text_expires_at is still in the future.
    _add(store, gone)
    _label(store, gone, created=NOW - timedelta(hours=1))
    with store.transaction() as conn:
        conn.execute("DELETE FROM messages WHERE window_id = ?", (gone,))
        conn.execute("UPDATE windows SET text_state = 'expired' WHERE window_id = ?", (gone,))
    assert assign.schedule_repeats(store, seed, NOW) == 0
    assert _repeats(store) == {}


def test_synthetic_labels_never_scheduled(store):
    _add(store, "fake-syn", "fake-page", "fake-rep", "fake-aron")
    first = NOW - timedelta(hours=1)
    _label(store, "fake-syn", project="blind", origin="synthetic", created=first)
    _label(store, "fake-syn", project="suggested", origin="synthetic", created=first)
    _label(store, "fake-page", project="page", created=first)
    _label(store, "fake-rep", project="repeat", created=first)
    _label(store, "fake-aron", project="blind", created=first)

    # rate 1.0 picks every window that qualifies at all
    assert assign.schedule_repeats(store, "seed-r", NOW, rate=1.0) == 1
    assert list(_repeats(store)) == ["fake-aron"]


def test_window_already_repeated_not_scheduled_again(store):
    _add(store, "fake-w0001")
    _label(store, "fake-w0001", project="suggested", created=NOW - timedelta(days=5))
    _label(store, "fake-w0001", project="repeat", created=NOW - timedelta(hours=1))
    assert assign.schedule_repeats(store, "seed-r", NOW, rate=1.0) == 0
    assert _repeats(store) == {}


def test_repeat_rows_never_change(store):
    _add(store, "fake-w0001")
    first = NOW - timedelta(hours=3)
    _label(store, "fake-w0001", created=first)
    assert assign.schedule_repeats(store, "seed-r", NOW, rate=1.0) == 1
    before = _repeats(store)
    # An earlier annotation pulled late, another seed, another clock: the row stays.
    _label(store, "fake-w0001", project="suggested", created=first - timedelta(hours=5))
    assert assign.schedule_repeats(store, "seed-other", NOW + timedelta(days=1), rate=1.0) == 0
    assert _repeats(store) == before


def test_schedule_repeats_refuses_bad_input(store):
    with pytest.raises(ValueError):
        assign.schedule_repeats(store, "seed-r", datetime(2026, 11, 6, 12, 0))
    with pytest.raises(ValueError):
        assign.schedule_repeats(store, "seed-r", NOW, rate=1.5)


# --- pick_rare ------------------------------------------------------------------------------------


def test_pick_rare_marks_window(store, tmp_path):
    _add(store, "fake-w0001", "fake-w0002", "fake-w0003")
    assert assign.pick_rare(store, ["fake-w0002", "fake-w0002"]) == 1
    assert store.window("fake-w0002").picked_rare
    assert not store.window("fake-w0001").picked_rare
    assert not store.window("fake-w0003").picked_rare

    # Already marked counts 0; the mark outlives the text.
    assert assign.pick_rare(store, ["fake-w0002", "fake-w0003"]) == 1
    expiry.run(store, LATER, tmp_path / "imported")
    assert store.window("fake-w0002").text_state == "expired"
    assert store.window_ids(picked_rare=True) == ["fake-w0002", "fake-w0003"]
    assert assign.pick_rare(store, []) == 0


def test_pick_rare_unknown_id_marks_nothing(store):
    _add(store, "fake-w0001")
    with pytest.raises(assign.UnknownWindows) as refused:
        assign.pick_rare(store, ["fake-w0001", "no-such-window"])
    assert "1 of 2" in str(refused.value)
    assert "no-such-window" not in str(refused.value)
    assert not store.window("fake-w0001").picked_rare


def test_pick_rare_refuses_a_bare_string(store):
    _add(store, "fake-w0001")
    with pytest.raises(TypeError):
        assign.pick_rare(store, "fake-w0001")  # type: ignore[arg-type]


# --- totals ---------------------------------------------------------------------------------------


def test_totals(store):
    assert assign.totals(store) == {"blind": 0, "suggested": 0, "repeats": 0, "picked_rare": 0}
    ids = _ids(20)
    _add(store, *ids)
    assign.assign_new(store, "seed-a", NOW)
    _label(store, ids[0], created=NOW)
    assign.schedule_repeats(store, "seed-a", NOW, rate=1.0)
    assign.pick_rare(store, ids[:2])
    blind = sum(assign.exposure(wid, "seed-a") == "blind" for wid in ids)
    assert assign.totals(store) == {"blind": blind, "suggested": 20 - blind, "repeats": 1, "picked_rare": 2}
