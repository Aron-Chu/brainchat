import dataclasses
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from chatbrain import expiry, fake, ingest, spread
from chatbrain.export_format import (
    EXPORT_FORMAT,
    EXPORT_FORMAT_V2,
    WINDOW_LIST_FORMAT,
    DeletionCheck,
    Emote,
    Export,
    ExportRejected,
    Message,
    StreamActivity,
    Window,
    dump_deletion_check,
    dump_export,
    iso,
    parse_export,
    window_list,
)
from chatbrain.spread import Counts
from chatbrain.store import Store

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)
START = NOW - timedelta(days=3)
T = NOW + timedelta(days=5)  # text_expires_at of the made-up windows
LONG = NOW + timedelta(days=13)  # for gate checks that run days after NOW
EMOTES = {"KEKW": Emote("fake-kekw", "KEKW"), "LUL": Emote("fake-lul", "LUL")}


def _win(window_id, texts=("KEKW", "LUL that was so bad"), *, expires=T, language="en", start=START) -> Window:
    msgs = tuple(
        Message(f"{window_id}-m{i + 1}", 31_000 + i * 1000, f"s{i + 1}", text,
                tuple(EMOTES[t] for t in text.split() if t in EMOTES))
        for i, text in enumerate(texts)
    )
    return Window(
        window_id=window_id,
        channel="fakechan01",
        stream_id="fs-01",
        anchor_kind="moment",
        anchor_at=start + timedelta(seconds=30),
        start_at=start,
        end_at=start + timedelta(seconds=120),
        language=language,
        text_expires_at=expires,
        messages=msgs,
    )


def _write_export(path: Path, *windows: Window, exported_at=NOW) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(dump_export(Export(EXPORT_FORMAT, exported_at, "vod", tuple(windows))))
    return path


def _import(store, tmp_path, *windows, name="e.json", now=NOW, exported_at=NOW):
    src = _write_export(tmp_path / "inbox" / name, *windows, exported_at=exported_at)
    return ingest.import_export(store, src, tmp_path / "imported", now)


def _features(store, window_id) -> str:
    return store.conn.execute("SELECT json FROM features WHERE window_id = ?", (window_id,)).fetchone()[0]


def _log(store) -> list[tuple]:
    return store.conn.execute(
        "SELECT path, window_ids_json, imported_at, deleted_at FROM imported_files ORDER BY rowid"
    ).fetchall()


def _db_bytes(tmp_path) -> bytes:
    return b"".join(p.read_bytes() for p in tmp_path.glob("cb.db*"))


# --- import ------------------------------------------------------------------------------


def test_import_adds_windows_and_their_counted_features(tmp_path):
    store = Store(tmp_path / "cb.db")
    w1, w2 = _win("w-1"), _win("w-2", texts=())
    r = _import(store, tmp_path, w1, w2)
    assert (r.windows_added, r.windows_duplicate, r.skipped_language, r.skipped_expired) == (2, 0, 0, 0)
    assert r.source_deleted is True
    assert store.window_ids() == ["w-1", "w-2"]
    assert store.window("w-1").source == "vod"
    assert store.as_window("w-1") == w1
    for w in (w1, w2):
        # Task 3's Counts.to_json is what goes into features, and it reads back strictly.
        assert _features(store, w.window_id) == spread.compute(w).to_json()
        assert Counts.from_json(_features(store, w.window_id)) == spread.compute(w)
    empty = Counts.from_json(_features(store, "w-2"))
    assert (empty.n_messages, empty.spread, empty.emote_counts) == (0, "few_posters", {})


def test_import_twice_counts_duplicates(tmp_path):
    store = Store(tmp_path / "cb.db")
    w1, w2 = _win("w-1"), _win("w-2")
    first = _import(store, tmp_path, w1, w2)
    assert (first.windows_added, first.windows_duplicate) == (2, 0)
    n_messages = store.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]

    # The same export again (the first copy was deleted on import), a day later.
    again = _import(store, tmp_path, w1, w2, now=NOW + timedelta(days=1))
    assert (again.windows_added, again.windows_duplicate, again.skipped_language, again.skipped_expired) == (0, 2, 0, 0)
    assert store.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == n_messages
    assert store.window("w-1").imported_at == NOW

    # An overlapping export: one new window, one already held.
    third = _import(store, tmp_path, w2, _win("w-3"), now=NOW + timedelta(days=1))
    assert (third.windows_added, third.windows_duplicate) == (1, 1)
    assert store.window_ids() == ["w-1", "w-2", "w-3"]
    assert len(_log(store)) == 3  # imported_files is a log: one row per import


def test_import_skips_non_english_and_already_expired(tmp_path):
    store = Store(tmp_path / "cb.db")
    exported_at = NOW - timedelta(days=2)
    windows = [
        _win("w-en", texts=("KEKW", "fox-lantern-77")),
        _win("w-de", texts=("KEKW", "heron-violin-58"), language="de"),
        _win("w-en-gb", texts=("KEKW", "heron-violin-59"), language="en-GB"),
        # now == text_expires_at counts as expired (Review Focus 5).
        _win("w-at-boundary", texts=("KEKW", "walrus-teapot-23"), expires=NOW),
        _win("w-past", texts=("KEKW", "walrus-teapot-24"), expires=NOW - timedelta(hours=1)),
    ]
    r = _import(store, tmp_path, *windows, exported_at=exported_at)
    assert (r.windows_added, r.windows_duplicate, r.skipped_language, r.skipped_expired) == (1, 0, 2, 2)
    assert store.window_ids() == ["w-en"]
    assert store.conn.execute("SELECT COUNT(*) FROM features").fetchone()[0] == 1
    store.close()
    blob = _db_bytes(tmp_path)
    assert b"fox-lantern-77" in blob
    for skipped in (b"heron-violin-58", b"heron-violin-59", b"walrus-teapot-23", b"walrus-teapot-24"):
        assert skipped not in blob


def test_import_leaves_no_text_file(tmp_path):
    # Ruling C3: the store is the only copy of text. The source file is deleted as soon
    # as its import commits, whether its windows were added, skipped or duplicates.
    store = Store(tmp_path / "cb.db")
    first = _import(store, tmp_path, _win("w-dup", texts=("KEKW", "badger-mitten-31")), name="first.json")
    assert first.windows_added == 1

    sentences = {
        "w-new": "otter-kettle-12",
        "w-de": "heron-violin-58",
        "w-old": "walrus-teapot-23",
        "w-dup": "badger-mitten-31",
    }
    windows = [
        _win("w-new", texts=("KEKW", sentences["w-new"])),
        _win("w-de", texts=("KEKW", sentences["w-de"]), language="de"),
        _win("w-old", texts=("KEKW", sentences["w-old"]), expires=NOW),
        _win("w-dup", texts=("KEKW", sentences["w-dup"])),
    ]
    src = _write_export(tmp_path / "inbox" / "Export.JSON", *windows)
    r = ingest.import_export(store, src, tmp_path / "imported", NOW)
    assert (r.windows_added, r.windows_duplicate, r.skipped_language, r.skipped_expired) == (1, 1, 1, 1)
    assert r.source_deleted is True

    assert not src.exists()
    imported = tmp_path / "imported"
    assert not imported.exists() or not any(p.is_file() for p in imported.rglob("*"))
    store.close()
    others = [p for p in tmp_path.rglob("*") if p.is_file() and not p.name.startswith("cb.db")]
    assert others == []  # no export, copy or leftover anywhere outside the store
    assert not any(s.encode() in _db_bytes(tmp_path) for s in (sentences["w-de"], sentences["w-old"]))

    store = Store(tmp_path / "cb.db")
    rows = _log(store)
    assert len(rows) == 2
    path, ids_json, imported_at, deleted_at = rows[1]
    assert path == os.path.normcase(os.path.realpath(src))  # normalised before logging (T1)
    assert json.loads(ids_json) == ["w-new", "w-de", "w-old", "w-dup"]
    assert imported_at == deleted_at == iso(NOW)
    assert rows[0][3] == iso(NOW)


def test_import_logs_a_normalised_path(tmp_path):
    store = Store(tmp_path / "cb.db")
    (tmp_path / "Inbox" / "Sub").mkdir(parents=True)
    src = _write_export(tmp_path / "Inbox" / "Data.json", _win("w-1"))
    winding = tmp_path / "Inbox" / "Sub" / ".." / "Data.json"
    ingest.import_export(store, winding, tmp_path / "imported", NOW)
    logged = _log(store)[0][0]
    assert logged == os.path.normcase(os.path.realpath(src))
    assert ".." not in Path(logged).parts
    if os.name == "nt":
        assert logged == logged.lower()


def test_import_is_all_or_nothing(tmp_path, monkeypatch):
    store = Store(tmp_path / "cb.db")
    real = spread.compute
    calls = []

    def flaky(w):
        calls.append(w.window_id)
        if len(calls) == 2:
            raise RuntimeError("boom")
        return real(w)

    monkeypatch.setattr(spread, "compute", flaky)
    src = _write_export(tmp_path / "inbox" / "e.json", _win("w-1"), _win("w-2"), _win("w-3"))
    with pytest.raises(RuntimeError):
        ingest.import_export(store, src, tmp_path / "imported", NOW)
    assert store.window_ids() == []
    for table in ("messages", "features", "imported_files"):
        assert store.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    assert src.exists()  # nothing committed, so the file stays where it was put
    assert not store.conn.in_transaction


def test_import_leaves_a_refused_file_where_it_is(tmp_path):
    store = Store(tmp_path / "cb.db")
    src = tmp_path / "inbox" / "bad.json"
    src.parent.mkdir(parents=True)
    raw = json.loads(dump_export(Export(EXPORT_FORMAT, NOW, "vod", (_win("w-1"),))))
    raw["windows"][0]["messages"][0]["display_name"] = "someone"
    src.write_bytes(json.dumps(raw).encode())
    with pytest.raises(ExportRejected):
        ingest.import_export(store, src, tmp_path / "imported", NOW)
    assert src.exists()
    assert store.window_ids() == []
    assert _log(store) == []


def test_import_refuses_a_naive_now(tmp_path):
    store = Store(tmp_path / "cb.db")
    src = _write_export(tmp_path / "inbox" / "e.json", _win("w-1"))
    with pytest.raises(ValueError):
        ingest.import_export(store, src, tmp_path / "imported", NOW.replace(tzinfo=None))
    assert src.exists()
    assert store.window_ids() == []


def test_import_reports_a_source_it_could_not_delete(tmp_path, monkeypatch):
    store = Store(tmp_path / "cb.db")
    src = _write_export(tmp_path / "inbox" / "e.json", _win("w-1"))
    real_unlink = Path.unlink

    def locked(self, *args, **kwargs):
        if Path(self) == src:
            raise PermissionError("in use by another program")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", locked)
    r = ingest.import_export(store, src, tmp_path / "imported", NOW)
    assert r.windows_added == 1  # the import itself committed
    assert r.source_deleted is False
    assert src.exists()
    assert _log(store)[0][3] is None  # deleted_at stays empty, so expiry retries it


def test_import_deletes_a_read_only_source(tmp_path):
    store = Store(tmp_path / "cb.db")
    src = _write_export(tmp_path / "inbox" / "e.json", _win("w-1"))
    os.chmod(src, 0o444)
    r = ingest.import_export(store, src, tmp_path / "imported", NOW)
    assert r.source_deleted is True
    assert not src.exists()


# --- deletion checks --------------------------------------------------------------------------


def _check(checked_at, **present) -> DeletionCheck:
    return DeletionCheck(checked_at=checked_at, present={k.replace("_", "-"): frozenset(v) for k, v in present.items()})


def _ids(store, window_id) -> list[str]:
    return [m.message_id for m in store.messages(window_id)]


def _add_label(store, window_id):
    with store.lock, store.conn:
        store.conn.execute(
            "INSERT INTO labels (window_id, project, origin, provenance, labels_version, guide_version, answers_json,"
            " created_at) VALUES (?, 'blind', 'aron', 'blind', 'v1', 'g1', '{\"feeling\": \"neutral\"}', ?)",
            (window_id, iso(NOW)),
        )


def test_deletion_check_drops_missing_ids_and_recomputes(tmp_path):
    store = Store(tmp_path / "cb.db")
    busy = _win("w-1", texts=("KEKW", "KEKW", "KEKW", "KEKW", "hi chat", "good morning"))
    calm = _win("w-2")
    _import(store, tmp_path, busy, calm)
    before = Counts.from_json(_features(store, "w-1"))
    assert before.spread == "repeated_emotes"
    _add_label(store, "w-1")

    keep = {"w-1-m4", "w-1-m5", "w-1-m6"}
    check = _check(NOW + timedelta(days=1), w_1=keep | {"never-held"}, w_2=set(_ids(store, "w-2")))
    applied = ingest.apply_deletion_check(store, check)

    assert applied.messages_dropped == 3
    assert applied.windows == frozenset({"w-1"})  # w-2 lost nothing
    assert set(_ids(store, "w-1")) == keep  # IDs the PC never held are ignored
    after = Counts.from_json(_features(store, "w-1"))
    assert after == spread.compute(store.as_window("w-1"))
    assert (after.n_messages, after.spread, after.emote_counts) == (3, "few_posters", {"fake-kekw": 1})
    assert _features(store, "w-2") == spread.compute(calm).to_json()
    assert store.conn.execute("SELECT COUNT(*) FROM labels WHERE window_id = 'w-1'").fetchone()[0] == 1
    row = store.conn.execute("SELECT checked_at, windows_checked, messages_dropped FROM deletion_checks").fetchall()
    assert row == [(iso(NOW + timedelta(days=1)), 2, 3)]


def test_window_missing_from_check_is_left_alone(tmp_path):
    # Ruling C2: the check lists every window the PC asked about. A window it leaves
    # out (say, one imported after the window list was written) is not touched.
    store = Store(tmp_path / "cb.db")
    _import(store, tmp_path, _win("w-1"), _win("w-2"))
    w2_features = _features(store, "w-2")
    applied = ingest.apply_deletion_check(store, _check(NOW, w_1=set(_ids(store, "w-1"))))
    assert applied.messages_dropped == 0
    assert applied.windows == frozenset()
    assert _ids(store, "w-2") == ["w-2-m1", "w-2-m2"]
    assert _features(store, "w-2") == w2_features


def test_window_listed_with_empty_present_loses_all_messages(tmp_path):
    store = Store(tmp_path / "cb.db")
    _import(store, tmp_path, _win("w-1"), _win("w-2"))
    _add_label(store, "w-2")
    applied = ingest.apply_deletion_check(store, _check(NOW, w_1=set(_ids(store, "w-1")), w_2=set()))
    assert applied.messages_dropped == 2
    assert applied.windows == frozenset({"w-2"})
    assert _ids(store, "w-2") == []
    counts = Counts.from_json(_features(store, "w-2"))
    assert (counts.n_messages, counts.distinct_posters, counts.spread, counts.emote_counts) == (0, 0, "few_posters", {})
    assert store.window("w-2").text_state == "present"
    assert store.conn.execute("SELECT COUNT(*) FROM labels WHERE window_id = 'w-2'").fetchone()[0] == 1


def test_deletion_check_keeps_features_of_an_expired_window(tmp_path):
    store = Store(tmp_path / "cb.db")
    _import(store, tmp_path, _win("w-1"))
    features = _features(store, "w-1")
    expiry.run(store, T, tmp_path / "imported")
    applied = ingest.apply_deletion_check(store, _check(T, w_1=set()))
    assert (applied.messages_dropped, applied.windows) == (0, frozenset())
    assert _features(store, "w-1") == features  # counts outlive the text


def test_deletion_check_ignores_windows_the_pc_does_not_hold(tmp_path):
    store = Store(tmp_path / "cb.db")
    _import(store, tmp_path, _win("w-1"))
    applied = ingest.apply_deletion_check(store, _check(NOW, w_ghost={"x"}, w_1=set(_ids(store, "w-1"))))
    assert applied.windows == frozenset()
    assert store.conn.execute("SELECT windows_checked FROM deletion_checks").fetchone()[0] == 1


def test_deletion_check_queues_label_studio_deletions(tmp_path):
    # Ruling C5: the affected windows' pushed tasks hold the deleted text, so they are
    # deleted from Label Studio (and their ls_tasks rows dropped) on the next run with a
    # client, even if Label Studio is down when the check is applied.
    store = Store(tmp_path / "cb.db")
    _import(store, tmp_path, _win("w-1"), _win("w-2"))
    with store.lock, store.conn:
        store.conn.executemany(
            "INSERT INTO ls_tasks (window_id, project_kind, task_id) VALUES (?, ?, ?)",
            [("w-1", "blind", 11), ("w-1", "repeat", 12), ("w-2", "suggested", 21)],
        )
    applied = ingest.apply_deletion_check(store, _check(NOW, w_1={"w-1-m1"}, w_2=set(_ids(store, "w-2"))))
    assert applied.windows == frozenset({"w-1"})

    pending = expiry.run(store, NOW, tmp_path / "imported")
    assert (pending.ls_tasks_deleted, pending.ls_tasks_pending) == (0, 2)

    ls = _FakeLS()
    done = expiry.run(store, NOW, tmp_path / "imported", ls=ls)
    assert sorted(ls.deleted) == [11, 12]
    assert ls.compacts == 1
    assert (done.ls_tasks_deleted, done.ls_tasks_pending) == (2, 0)
    rows = store.conn.execute("SELECT window_id, task_id FROM ls_tasks ORDER BY task_id").fetchall()
    assert rows == [("w-2", 21)]  # w-1 can be pushed again, without the deleted message


class _FakeLS:
    def __init__(self):
        self.deleted: list[int] = []
        self.compacts = 0

    def delete_tasks(self, task_ids: list[int]) -> int:
        self.deleted.extend(task_ids)
        return len(task_ids)

    def compact(self) -> None:
        self.compacts += 1


# --- freshness gate ---------------------------------------------------------------------------------


def test_freshness_gate(tmp_path):
    store = Store(tmp_path / "cb.db")
    with pytest.raises(ingest.StaleDeletionCheck, match="no deletion check"):
        ingest.require_fresh_deletion_check(store, NOW)
    # The import covers w-1 as of its exported_at (B7), two days before the check below.
    _import(store, tmp_path, _win("w-1", expires=NOW + timedelta(days=12)), exported_at=NOW - timedelta(days=2))
    ingest.require_fresh_deletion_check(store, NOW)

    checked = NOW
    ingest.apply_deletion_check(store, _check(checked, w_1=set(_ids(store, "w-1"))))
    ingest.require_fresh_deletion_check(store, checked + timedelta(days=6))
    ingest.require_fresh_deletion_check(store, checked + timedelta(days=7))  # "more than 7 days" refuses
    with pytest.raises(ingest.StaleDeletionCheck):
        ingest.require_fresh_deletion_check(store, checked + timedelta(days=7, seconds=1))
    ingest.require_fresh_deletion_check(store, checked + timedelta(days=2), max_age=timedelta(days=3))
    with pytest.raises(ingest.StaleDeletionCheck):
        ingest.require_fresh_deletion_check(store, checked + timedelta(days=4), max_age=timedelta(days=3))

    # A check dated more than 5 minutes after this PC's clock is refused too (T1).
    ingest.require_fresh_deletion_check(store, checked - timedelta(minutes=5))
    with pytest.raises(ingest.StaleDeletionCheck, match="future"):
        ingest.require_fresh_deletion_check(store, checked - timedelta(minutes=5, seconds=1))


def test_freshness_gate_with_no_windows_needs_a_check(tmp_path):
    store = Store(tmp_path / "cb.db")
    with pytest.raises(ingest.StaleDeletionCheck):
        ingest.require_fresh_deletion_check(store, NOW)
    ingest.apply_deletion_check(store, _check(NOW))
    ingest.require_fresh_deletion_check(store, NOW)
    with pytest.raises(ingest.StaleDeletionCheck, match="more than 7 days old"):
        ingest.require_fresh_deletion_check(store, NOW + timedelta(days=7, seconds=1))


# An export older than 7 days covers nothing at NOW (B7), so these tests see only the checks.
OLD_EXPORT = NOW - timedelta(days=7, seconds=1)
OLD_START = NOW - timedelta(days=8)


def test_freshness_gate_needs_every_window_with_text_listed(tmp_path):
    # Ruling B6: under C2 an unlisted window keeps its text, so the gate checks that
    # every window with text is listed by a check from the last 7 days.
    store = Store(tmp_path / "cb.db")
    expires = OLD_EXPORT + timedelta(days=14)
    _import(store, tmp_path, _win("w-1", expires=expires, start=OLD_START),
            _win("w-2", expires=expires, start=OLD_START), exported_at=OLD_EXPORT)
    ingest.apply_deletion_check(store, _check(NOW - timedelta(days=1), w_1=set(_ids(store, "w-1"))))
    with pytest.raises(ingest.StaleDeletionCheck) as refused:
        ingest.require_fresh_deletion_check(store, NOW)
    message = str(refused.value)
    assert "1 window" in message
    assert "w-2" not in message and "w-1" not in message  # counts only, never IDs or text

    ingest.apply_deletion_check(store, _check(NOW, w_2=set(_ids(store, "w-2"))))
    ingest.require_fresh_deletion_check(store, NOW)

    # Listing by an old check doesn't count: 6 days and 1 s on, the newest check (w-2's)
    # is still fresh, but w-1 was last listed 7 days and 1 s before.
    later = NOW + timedelta(days=6, seconds=1)
    with pytest.raises(ingest.StaleDeletionCheck, match="1 window"):
        ingest.require_fresh_deletion_check(store, later)
    ingest.apply_deletion_check(store, _check(later, w_1=set(_ids(store, "w-1")), w_2=set(_ids(store, "w-2"))))
    ingest.require_fresh_deletion_check(store, later)


def test_freshness_gate_ignores_windows_whose_text_expired(tmp_path):
    store = Store(tmp_path / "cb.db")
    _import(store, tmp_path, _win("w-1", start=OLD_START),
            _win("w-short", expires=NOW + timedelta(days=1), start=OLD_START), exported_at=OLD_EXPORT)
    ingest.apply_deletion_check(store, _check(NOW, w_1=set(_ids(store, "w-1"))))
    with pytest.raises(ingest.StaleDeletionCheck, match="1 window"):
        ingest.require_fresh_deletion_check(store, NOW + timedelta(days=1))
    expiry.run(store, NOW + timedelta(days=1), tmp_path / "imported")
    ingest.require_fresh_deletion_check(store, NOW + timedelta(days=1))


# --- an export counts as deletion-check coverage (B7) ------------------------------------------------


def _covered(store, window_id):
    with store.lock:
        row = store.conn.execute("SELECT checked_at FROM checked_windows WHERE window_id = ?", (window_id,)).fetchone()
    return row[0] if row else None


def test_fresh_import_passes_the_gate_without_a_check(tmp_path):
    store = Store(tmp_path / "cb.db")
    _import(store, tmp_path, _win("w-1", expires=LONG))
    ingest.require_fresh_deletion_check(store, NOW)
    ingest.require_fresh_deletion_check(store, NOW + timedelta(days=7))
    with pytest.raises(ingest.StaleDeletionCheck) as refused:
        ingest.require_fresh_deletion_check(store, NOW + timedelta(days=8))
    message = str(refused.value)
    assert message.startswith("1 window(s) with text are not listed")
    assert "w-1" not in message and "2026" not in message  # the count only: no ID, no date


def test_import_coverage_is_the_earlier_of_exported_at_and_now(tmp_path):
    store = Store(tmp_path / "cb.db")
    _import(store, tmp_path, _win("w-1"), exported_at=NOW - timedelta(days=2))
    assert _covered(store, "w-1") == iso(NOW - timedelta(days=2))
    # An export dated a little after this PC's clock covers only up to now.
    _import(store, tmp_path, _win("w-2"), name="f.json", exported_at=NOW + timedelta(minutes=3))
    assert _covered(store, "w-2") == iso(NOW)


def test_import_coverage_keeps_the_newer_date(tmp_path):
    store = Store(tmp_path / "cb.db")
    w = _win("w-1", expires=NOW + timedelta(days=12))
    _import(store, tmp_path, w, exported_at=NOW - timedelta(days=2))
    ingest.apply_deletion_check(store, _check(NOW + timedelta(days=1), w_1=set(_ids(store, "w-1"))))
    _import(store, tmp_path, w, name="again.json", now=NOW + timedelta(days=2))  # exported_at NOW: older
    assert _covered(store, "w-1") == iso(NOW + timedelta(days=1))
    _import(store, tmp_path, w, name="third.json", now=NOW + timedelta(days=3), exported_at=NOW + timedelta(days=3))
    assert _covered(store, "w-1") == iso(NOW + timedelta(days=3))


def test_duplicate_import_missing_a_message_gives_no_coverage(tmp_path):
    store = Store(tmp_path / "cb.db")
    expires = OLD_EXPORT + timedelta(days=14)
    full = _win("w-1", expires=expires, start=OLD_START)
    _import(store, tmp_path, full, exported_at=OLD_EXPORT)
    assert _covered(store, "w-1") == iso(OLD_EXPORT)

    # The server no longer has one of the stored messages: this export can't vouch for the window.
    short = _win("w-1", texts=("KEKW",), expires=expires, start=OLD_START)
    r = _import(store, tmp_path, short, name="short.json")
    assert r.windows_duplicate == 1
    assert _covered(store, "w-1") == iso(OLD_EXPORT)
    assert _ids(store, "w-1") == ["w-1-m1", "w-1-m2"]  # an export never deletes; a deletion check does
    with pytest.raises(ingest.StaleDeletionCheck, match="1 window"):
        ingest.require_fresh_deletion_check(store, NOW)

    # The same window with every stored message (and one more) does.
    more = _win("w-1", texts=("KEKW", "LUL that was so bad", "gg"), expires=expires, start=OLD_START)
    _import(store, tmp_path, more, name="more.json")
    assert _covered(store, "w-1") == iso(NOW)
    ingest.require_fresh_deletion_check(store, NOW)


# --- pending check files (C36) ----------------------------------------------------------------------


def _write_check(path: Path, check: DeletionCheck) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(dump_deletion_check(check))
    return path


def test_apply_pending_checks(tmp_path):
    store = Store(tmp_path / "cb.db")
    _import(store, tmp_path, _win("w-1"), _win("w-2"), _win("w-3"))
    inbox = tmp_path / "inbox"
    all_ids = {w: set(_ids(store, w)) for w in ("w-1", "w-2", "w-3")}
    # Named so that name order is the reverse of checked_at order.
    newer = _write_check(inbox / "a-check.json", _check(NOW, w_1=all_ids["w-1"] - {"w-1-m1"}, w_2=set(),
                                                         w_3=all_ids["w-3"]))
    older = _write_check(inbox / "b-check.json", _check(NOW - timedelta(days=1), w_1=all_ids["w-1"],
                                                         w_2=all_ids["w-2"] - {"w-2-m2"}, w_3=all_ids["w-3"]))
    export = _write_export(inbox / "c-export.json", _win("w-4"))
    listing = inbox / "d-window-list.json"
    listing.write_bytes(window_list([("w-1", "fs-01", START, START + timedelta(seconds=120))]))
    (inbox / "notes.txt").write_text("not json, not ours", encoding="utf-8")

    affected = ingest.apply_pending_checks(store, inbox, now=NOW)
    assert affected == frozenset({"w-1", "w-2"})
    assert _ids(store, "w-1") == ["w-1-m2"]
    assert _ids(store, "w-2") == []
    assert not newer.exists() and not older.exists()  # applied checks leave the inbox
    assert export.exists() and listing.exists() and (inbox / "notes.txt").exists()
    order = [r[0] for r in store.conn.execute("SELECT checked_at FROM deletion_checks ORDER BY id")]
    assert order == [iso(NOW - timedelta(days=1)), iso(NOW)]  # oldest first
    ingest.require_fresh_deletion_check(store, NOW)

    assert ingest.apply_pending_checks(store, inbox, now=NOW) == frozenset()


def test_apply_pending_checks_refuses_bad_files_after_applying_good_ones(tmp_path):
    store = Store(tmp_path / "cb.db")
    _import(store, tmp_path, _win("w-1"))
    inbox = tmp_path / "inbox"
    good = _write_check(inbox / "good.json", _check(NOW, w_1={"w-1-m1"}))
    future = _write_check(inbox / "future.json", _check(NOW + timedelta(minutes=6), w_1=set()))
    broken = inbox / "broken.json"
    broken.write_bytes(b'{"format": "chatbrain-deletion-check/1", "checked_at": "2026-11-06T12:00:00Z"}')
    junk = inbox / "junk.json"
    junk.write_bytes(b"{not json")
    with pytest.raises(ExportRejected) as refused:
        ingest.apply_pending_checks(store, inbox, now=NOW)
    message = str(refused.value)
    for name in ("future.json", "broken.json", "junk.json"):
        assert name in message
    assert not good.exists()
    assert _ids(store, "w-1") == ["w-1-m1"]  # the good check was applied anyway
    assert future.exists() and broken.exists() and junk.exists()
    assert store.conn.execute("SELECT COUNT(*) FROM deletion_checks").fetchone()[0] == 1


def test_apply_pending_checks_refuses_a_naive_now(tmp_path):
    store = Store(tmp_path / "cb.db")
    with pytest.raises(ValueError):
        ingest.apply_pending_checks(store, tmp_path / "inbox", now=NOW.replace(tzinfo=None))


# --- window list --------------------------------------------------------------------------------------


def test_write_window_list_lists_windows_with_text_only(tmp_path):
    store = Store(tmp_path / "cb.db")
    _import(store, tmp_path, _win("w-1", texts=("KEKW", "otter-kettle-12")),
            _win("w-short", texts=("KEKW", "walrus-teapot-23"), expires=NOW + timedelta(days=1)))
    expiry.run(store, NOW + timedelta(days=1), tmp_path / "imported")
    out = tmp_path / "out" / "window-list.json"
    assert ingest.write_window_list(store, out) == 1
    raw = out.read_bytes()
    assert b"otter-kettle-12" not in raw and b"KEKW" not in raw and b"w-1-m1" not in raw
    doc = json.loads(raw)
    assert doc == {
        "format": WINDOW_LIST_FORMAT,
        "windows": [{"window_id": "w-1", "stream_id": "fs-01", "start_at": iso(START),
                     "end_at": iso(START + timedelta(seconds=120))}],
    }


def test_store_transaction_rolls_back_and_does_not_nest(tmp_path):
    store = Store(tmp_path / "cb.db")
    with pytest.raises(RuntimeError, match="boom"):
        with store.transaction() as conn:
            conn.execute("INSERT INTO deletion_checks (checked_at, windows_checked, messages_dropped) VALUES ('x', 0, 0)")
            raise RuntimeError("boom")
    assert store.conn.execute("SELECT COUNT(*) FROM deletion_checks").fetchone()[0] == 0
    with store.transaction():
        with pytest.raises(RuntimeError):
            with store.transaction():
                pass
    with pytest.raises(RuntimeError):
        store.insert_window(_win("w-1"), "vod", NOW)  # only inside a transaction
    assert store.window_ids() == []


# --- per-stream activity (Slice 1, Task 2; Q10) -----------------------------------------------------


def _stream(stream_id="fs-01", *, channel="fakechan01", started_at=START - timedelta(minutes=10), minutes=20,
            title="Made-up stream 1", base=5) -> StreamActivity:
    return StreamActivity(
        stream_id=stream_id,
        channel=channel,
        started_at=started_at,
        ended_at=started_at + timedelta(minutes=minutes),
        title=title,
        bin_s=10,
        chatters=tuple(None if i == 7 else base + i % 3 for i in range(minutes * 6)),
    )  # fmt: skip


def _write_v2(path: Path, windows=(), streams=(), *, exported_at=NOW) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(dump_export(Export(EXPORT_FORMAT_V2, exported_at, "vod", tuple(windows), tuple(streams))))
    return path


def _import_v2(store, tmp_path, windows=(), streams=(), *, name="e2.json", now=NOW, exported_at=NOW):
    src = _write_v2(tmp_path / "inbox" / name, windows, streams, exported_at=exported_at)
    return ingest.import_export(store, src, tmp_path / "imported", now)


def _activity_rows(store) -> dict[str, int]:
    return {
        t: store.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        for t in ("stream_activity", "stream_activity_expired", "imported_streams")
    }


NO_ACTIVITY = {"stream_activity": 0, "stream_activity_expired": 0, "imported_streams": 0}


def test_import_stores_activity_with_windows(tmp_path):
    store = Store(tmp_path / "cb.db")
    a = _stream("fs-01")
    b = _stream("fs-02", title=None, base=9)
    w2 = dataclasses.replace(_win("w-2"), stream_id="fs-02")
    r = _import_v2(store, tmp_path, (_win("w-1"), w2), (a, b))
    assert (r.windows_added, r.streams_written, r.streams_expired_skipped) == (2, 2, 0)
    assert r.source_deleted is True
    assert store.stream_activity("fs-01", now=NOW) == a
    assert store.stream_activity("fs-02", now=NOW) == b
    assert store.channel_activity("fakechan01", now=NOW) == [a, b]  # equal ended_at: by stream ID
    assert store.activity_state("fs-01", now=NOW) == "active"
    assert _activity_rows(store) == {"stream_activity": 2, "stream_activity_expired": 0, "imported_streams": 0}

    # The same file again (an equal exported_at) keeps the stored payload: counted in neither.
    r = _import_v2(store, tmp_path, (_win("w-1"), w2), (a, _stream("fs-02", title=None, base=1)))
    assert (r.windows_duplicate, r.streams_written, r.streams_expired_skipped) == (2, 0, 0)
    assert store.stream_activity("fs-02", now=NOW) == b
    # A newer export of the same stream replaces it, with no window in the file: w-2 is already stored.
    newer = _stream("fs-02", title="Made-up stream 2", base=1)
    r = _import_v2(store, tmp_path, (), (newer,), exported_at=NOW + timedelta(minutes=1), now=NOW + timedelta(minutes=1))
    assert (r.streams_written, r.streams_expired_skipped) == (1, 0)
    assert store.stream_activity("fs-02", now=NOW) == newer


def test_import_stores_no_activity_without_a_window(tmp_path):
    # Only a stream a stored window names gets a line, so its expired marker always has a window
    # keeping it (E2) and swept counts can never come back with a re-export (Q10).
    store = Store(tmp_path / "cb.db")
    fr = dataclasses.replace(_win("w-4", language="fr"), stream_id="fs-04")
    text_gone = dataclasses.replace(_win("w-5", expires=NOW), stream_id="fs-05")  # text expired at import
    streams = (
        _stream("fs-01"),
        _stream("fs-03"),  # no window in any file
        _stream("fs-04"),  # only a window that isn't English
        _stream("fs-05"),  # only a window whose text already expired
        _stream("fs-06", started_at=NOW - timedelta(days=31)),  # no window, counts already expired
    )
    r = _import_v2(store, tmp_path, (_win("w-1"), fr, text_gone), streams)
    assert (r.windows_added, r.skipped_language, r.skipped_expired) == (1, 1, 1)
    assert (r.streams_written, r.streams_expired_skipped) == (1, 1)  # fs-06 is counted, but gets no marker
    assert _activity_rows(store) == {"stream_activity": 1, "stream_activity_expired": 0, "imported_streams": 0}
    assert store.channel_activity("fakechan01", now=NOW) == [streams[0]]
    for s in ("fs-03", "fs-04", "fs-05", "fs-06"):
        assert store.stream_activity(s, now=NOW) is None
        assert store.activity_state(s, now=NOW) == "missing"

    # A later file that brings fs-03 a window stores its line.
    w3 = dataclasses.replace(_win("w-3"), stream_id="fs-03")
    r = _import_v2(store, tmp_path, (w3,), (streams[1],), name="later.json")
    assert (r.windows_added, r.streams_written, r.streams_expired_skipped) == (1, 1, 0)
    assert store.stream_activity("fs-03", now=NOW) == streams[1]


def test_import_logs_every_stream_while_its_source_stays(tmp_path, monkeypatch):
    store = Store(tmp_path / "cb.db")
    src = _write_v2(tmp_path / "inbox" / "e2.json", (_win("w-1"),), (_stream("fs-01"), _stream("fs-02")))
    real_unlink = Path.unlink

    def locked(self, *args, **kwargs):
        if Path(self) == src:
            raise PermissionError("in use by another program")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", locked)
    r = ingest.import_export(store, src, tmp_path / "imported", NOW)
    assert r.source_deleted is False
    logged = os.path.normcase(os.path.realpath(src))
    rows = store.conn.execute("SELECT path, stream_id, exported_at FROM imported_streams ORDER BY stream_id").fetchall()
    assert rows == [(logged, "fs-01", iso(NOW)), (logged, "fs-02", iso(NOW))]


def test_refused_export_stores_no_activity(tmp_path, monkeypatch):
    store = Store(tmp_path / "cb.db")
    live, gone = _stream("fs-01"), _stream("fs-02", started_at=NOW - timedelta(days=31))  # gone: expired at import
    windows = (_win("w-1"), dataclasses.replace(_win("w-2"), stream_id="fs-02"))  # so both streams get a line

    # A file the parser refuses.
    src = _write_v2(tmp_path / "inbox" / "bad.json", windows, (live, gone))
    raw = json.loads(src.read_bytes())
    raw["streams"][0]["login"] = "someone"
    src.write_bytes(json.dumps(raw).encode())
    with pytest.raises(ExportRejected):
        ingest.import_export(store, src, tmp_path / "imported", NOW)
    assert src.exists()

    # A file whose import fails after its first stream was written: all or nothing.
    src = _write_v2(tmp_path / "inbox" / "flaky.json", windows, (live, gone))
    real_put = Store.put_stream_activity
    calls = []

    def flaky(self, *args, **kwargs):
        calls.append(real_put(self, *args, **kwargs))
        if len(calls) == 2:
            raise RuntimeError("boom")
        return calls[-1]

    with monkeypatch.context() as m:
        m.setattr(Store, "put_stream_activity", flaky)
        with pytest.raises(RuntimeError):
            ingest.import_export(store, src, tmp_path / "imported", NOW)
    assert calls == ["written", "expired"]  # both were written, then rolled back
    assert src.exists()
    assert not store.conn.in_transaction

    assert store.window_ids() == []
    assert _log(store) == []
    assert _activity_rows(store) == NO_ACTIVITY
    assert store.activity_state("fs-02", now=NOW) == "missing"


def test_import_refuses_a_stream_filed_under_another_channel(tmp_path):
    # C22: a stream's activity never lands under another channel than its windows'.
    store = Store(tmp_path / "cb.db")
    src = _write_v2(tmp_path / "inbox" / "a.json", (_win("w-1"),), (_stream("fs-01", channel="fakechan02"),))
    with pytest.raises(ExportRejected) as refused:
        ingest.import_export(store, src, tmp_path / "imported", NOW)
    assert "fs-01" not in str(refused.value) and "w-1" not in str(refused.value)
    assert src.exists()
    assert store.window_ids() == []
    assert _activity_rows(store) == NO_ACTIVITY

    # The same when the window is already stored and the new file carries only the stream.
    _import(store, tmp_path, _win("w-1"))
    src = _write_v2(tmp_path / "inbox" / "b.json", (), (_stream("fs-01", channel="fakechan02"),))
    with pytest.raises(ExportRejected):
        ingest.import_export(store, src, tmp_path / "imported", NOW)
    assert src.exists()
    assert len(_log(store)) == 1
    assert _activity_rows(store) == NO_ACTIVITY

    # A matching channel imports.
    r = _import_v2(store, tmp_path, (), (_stream("fs-01"),), name="c.json")
    assert r.streams_written == 1


def test_v1_import_stores_none(tmp_path):
    store = Store(tmp_path / "cb.db")
    r = _import(store, tmp_path, _win("w-1"))
    assert (r.windows_added, r.streams_written, r.streams_expired_skipped) == (1, 0, 0)
    raw = fake.make_export(3, 4, 2, NOW, channels=1, activity=False)
    (tmp_path / "inbox" / "fake.json").write_bytes(raw)
    r = ingest.import_export(store, tmp_path / "inbox" / "fake.json", tmp_path / "imported", NOW)
    assert (r.windows_added, r.streams_written, r.streams_expired_skipped) == (6, 0, 0)
    assert _activity_rows(store) == NO_ACTIVITY
    for stream_id in {"fs-01"} | {w.stream_id for w in parse_export(raw).windows}:
        assert store.activity_state(stream_id, now=NOW) == "missing"
        assert store.stream_activity(stream_id, now=NOW) is None


def test_import_skips_and_counts_expired_activity(tmp_path):
    started = NOW - timedelta(days=25)  # an export made on day 25 of the stream
    a = _stream("fs-03", started_at=started)
    w3 = dataclasses.replace(_win("w-3", expires=NOW + timedelta(days=10)), stream_id="fs-03")  # text outlives day 30
    store = Store(tmp_path / "cb.db")
    r = _import_v2(store, tmp_path, (w3,), (a,))
    assert (r.streams_written, r.streams_expired_skipped) == (1, 0)
    expires = store.conn.execute("SELECT expires_at FROM stream_activity").fetchone()[0]
    assert expires == iso(started + timedelta(days=30))  # 5 days left, never 30 from the export or import

    day30 = started + timedelta(days=30)
    r = _import_v2(store, tmp_path, (), (a,), name="again.json", now=day30, exported_at=day30)
    assert (r.streams_written, r.streams_expired_skipped) == (0, 1)
    assert _activity_rows(store) == {"stream_activity": 0, "stream_activity_expired": 1, "imported_streams": 0}

    # The day-25 export first imported on day 30 is skipped too.
    other = Store(tmp_path / "other.db")
    r = _import_v2(other, tmp_path, (w3,), (a,), name="late.json", now=day30)
    assert (r.windows_added, r.streams_written, r.streams_expired_skipped) == (1, 0, 1)
    assert other.stream_activity("fs-03", now=day30 - timedelta(days=1)) is None
    assert other.activity_state("fs-03", now=day30) == "expired"


def test_import_skip_records_expired_state(tmp_path, monkeypatch):
    # E1: counts already past 30 days at import read as expired, never as "never imported".
    store = Store(tmp_path / "cb.db")
    started = NOW - timedelta(days=31)
    w = _win("w-1", start=started + timedelta(minutes=5))  # its text is still live
    src = _write_v2(tmp_path / "inbox" / "e2.json", (w,), (_stream("fs-01", started_at=started),))
    real_unlink = Path.unlink

    def locked(self, *args, **kwargs):
        if Path(self) == src:
            raise PermissionError("in use by another program")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", locked)
    r = ingest.import_export(store, src, tmp_path / "imported", NOW)
    assert (r.windows_added, r.streams_written, r.streams_expired_skipped) == (1, 0, 1)
    for page_now in (started, NOW):
        assert store.activity_state("fs-01", now=page_now) == "expired"
        assert store.stream_activity("fs-01", now=page_now) is None
    assert store.conn.execute("SELECT COUNT(*) FROM stream_activity").fetchone()[0] == 0
    assert store.conn.execute("SELECT stream_id FROM stream_activity_expired").fetchall() == [("fs-01",)]
    # The stream is logged for its source whatever happened to its counts (E9).
    logged = os.path.normcase(os.path.realpath(src))
    assert store.conn.execute("SELECT path, stream_id, exported_at FROM imported_streams").fetchall() == [
        (logged, "fs-01", iso(NOW))
    ]


def test_inbox_routes_a_v2_export_as_an_export(tmp_path):
    store = Store(tmp_path / "cb.db")
    inbox = tmp_path / "inbox"
    v2 = _write_v2(inbox / "a.json", (_win("w-1"),), (_stream(),))
    v1 = _write_export(inbox / "b.json", _win("w-2"))
    kinds = ingest.inbox_files(inbox)
    assert kinds["export"] == [v2, v1]
    assert kinds["unknown"] == []
    result = ingest.apply_inbox_checks(store, inbox, now=NOW)
    assert result.refused == ()
    assert v2.exists() and v1.exists()  # exports wait for `chatbrain import`


# --- F1 tally (glance call 2): kept in step at import, with a /2 and by deletion checks ---------------------


def _face_rows(store) -> list[tuple]:
    return store.conn.execute(
        "SELECT window_id, emote_id, lines, of_lines, people, expires_at FROM moment_emotes ORDER BY window_id"
    ).fetchall()


KEKW6 = ("KEKW",) * 6


def test_missing_counts_proxy_deadline_lowered_by_later_v2(tmp_path):
    store = Store(tmp_path / "cb.db")
    _import(store, tmp_path, _win("w-1", KEKW6), name="a.json")  # a /1 file: no activity counts
    proxy = START + timedelta(days=30)  # the stream's earliest window start + 30 days
    assert _face_rows(store) == [("w-1", "fake-kekw", 6, 6, 6, iso(proxy))]
    assert set(store.moment_faces("fs-01", proxy - timedelta(microseconds=1))) == {"w-1"}
    assert store.moment_faces("fs-01", proxy) == {}

    # An earlier window of the same stream pulls every face of the stream in.
    early = START - timedelta(hours=1)
    _import(store, tmp_path, _win("w-0", KEKW6[:5], start=early), name="b.json")
    lowered = iso(early + timedelta(days=30))
    assert [r[5] for r in _face_rows(store)] == [lowered, lowered]

    # A /2 whose counts would live longer never raises it ...
    late = _stream(started_at=START + timedelta(hours=2))
    assert _import_v2(store, tmp_path, (), (late,), name="c.json").streams_written == 1
    assert [r[5] for r in _face_rows(store)] == [lowered, lowered]
    # ... and an earlier stream start lowers it to the counts' own deadline.
    earlier = _stream(started_at=START - timedelta(hours=3))
    _import_v2(store, tmp_path, (), (earlier,), name="d.json", exported_at=NOW - timedelta(hours=1))
    counts_due = iso(START - timedelta(hours=3) + timedelta(days=30))
    assert store.conn.execute("SELECT expires_at FROM stream_activity").fetchone()[0] == counts_due
    assert [r[5] for r in _face_rows(store)] == [counts_due, counts_due]
    # A Moment imported later shares the stream's one deadline.
    _import(store, tmp_path, _win("w-2", KEKW6, start=START + timedelta(hours=1)), name="e.json")
    assert [r[5] for r in _face_rows(store)] == [counts_due] * 3


def test_expired_v2_deletes_rows(tmp_path):
    store = Store(tmp_path / "cb.db")
    _import(store, tmp_path, _win("w-1", KEKW6), name="a.json")
    assert len(_face_rows(store)) == 1
    old = _stream(started_at=NOW - timedelta(days=31))
    r = _import_v2(store, tmp_path, (), (old,), name="b.json")
    assert r.streams_expired_skipped == 1
    assert _face_rows(store) == []
    assert store.activity_state("fs-01", now=NOW) == "expired"
    # The stream's counts are gone for good, and so are its faces: a later Moment gets no row.
    _import(store, tmp_path, _win("w-2", KEKW6, start=START + timedelta(hours=1)), name="c.json")
    assert store.window_ids() == ["w-1", "w-2"]
    assert _face_rows(store) == []
    assert store.backfill_moment_emotes(NOW) == 0
    assert _face_rows(store) == []


def test_deletion_check_recomputes(tmp_path):
    store = Store(tmp_path / "cb.db")
    busy = _win("w-1", ("KEKW", "KEKW", "KEKW", "LUL", "LUL that was so bad", "hi chat"))
    gone = _win("w-3", ("KEKW", "KEKW"), expires=NOW + timedelta(days=1))
    _import(store, tmp_path, busy, gone)
    due = iso(START + timedelta(days=30))
    assert _face_rows(store) == [("w-1", "fake-kekw", 3, 6, 3, due), ("w-3", "fake-kekw", 2, 2, 2, due)]
    expiry.run(store, NOW + timedelta(days=1), tmp_path / "imported")  # w-3's text expires
    at = NOW + timedelta(days=1)

    # Two KEKW lines dropped: LUL now leads (2 lines to 1), counted again from what is left.
    ingest.apply_deletion_check(store, _check(at, w_1={"w-1-m3", "w-1-m4", "w-1-m5", "w-1-m6"}, w_3=set()))
    assert _face_rows(store) == [("w-1", "fake-lul", 2, 4, 2, due), ("w-3", "fake-kekw", 2, 2, 2, due)]
    # The LUL lines go too: KEKW's one line wins back.
    ingest.apply_deletion_check(store, _check(at, w_1={"w-1-m3", "w-1-m6"}))
    assert _face_rows(store)[0] == ("w-1", "fake-kekw", 1, 2, 1, due)
    # No emote left: the row goes. The expired-text window keeps its count.
    ingest.apply_deletion_check(store, _check(at, w_1={"w-1-m6"}))
    assert _face_rows(store) == [("w-3", "fake-kekw", 2, 2, 2, due)]
    assert store.backfill_moment_emotes(at) == 0  # nothing to count in w-1 any more
