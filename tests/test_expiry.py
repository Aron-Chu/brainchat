import dataclasses
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from chatbrain import expiry, ingest
from chatbrain.export_format import (
    EXPORT_FORMAT,
    EXPORT_FORMAT_V2,
    Emote,
    Export,
    Message,
    StreamActivity,
    Window,
    dump_export,
    iso,
)
from chatbrain.store import Store

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)
START = NOW - timedelta(days=3)
T = NOW + timedelta(days=5)  # text_expires_at of the window that expires first
KEKW = Emote("fake-kekw", "KEKW")


def _win(window_id, texts=("KEKW", "LUL that was so bad"), *, expires=T) -> Window:
    msgs = tuple(
        Message(f"{window_id}-m{i + 1}", 31_000 + i * 1000, f"s{i + 1}", text, (KEKW,) if "KEKW" in text.split() else ())
        for i, text in enumerate(texts)
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


def _export_file(path: Path, *windows: Window) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(dump_export(Export(EXPORT_FORMAT, NOW, "vod", tuple(windows))))
    return path


def _import(store, tmp_path, *windows, name="e.json"):
    return ingest.import_export(store, _export_file(tmp_path / "inbox" / name, *windows), tmp_path / "imported", NOW)


def _db_bytes(tmp_path) -> bytes:
    return b"".join(p.read_bytes() for p in tmp_path.glob("cb.db*"))


def _count(store, table, where="1=1", *args) -> int:
    return store.conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}", args).fetchone()[0]


def _ls_tasks(store, rows):
    with store.lock, store.conn:
        store.conn.executemany("INSERT INTO ls_tasks (window_id, project_kind, task_id) VALUES (?, ?, ?)", rows)


class FakeLS:
    """A made-up Label Studio: records deletes, and can fail like an unreachable one.

    ``not_deleted`` IDs are answered with a count of 0 and no exception, like a client
    whose DELETE got a non-2xx answer it doesn't raise for.
    """

    def __init__(self, fail_on=(), compact_failures=0, not_deleted=()):
        self.calls: list[list[int]] = []
        self.deleted: list[int] = []
        self.compacts = 0
        self.fail_on = set(fail_on)
        self.not_deleted = set(not_deleted)
        self.compact_failures = compact_failures

    def delete_tasks(self, task_ids: list[int]) -> int:
        self.calls.append(list(task_ids))
        if self.fail_on & set(task_ids):
            raise ConnectionError("label studio is down")
        gone = [t for t in task_ids if t not in self.not_deleted]
        self.deleted.extend(gone)
        return len(gone)

    def compact(self) -> None:
        if self.compact_failures:
            self.compact_failures -= 1
            raise RuntimeError("docker is stopped")
        self.compacts += 1


# --- text ---------------------------------------------------------------------------------------


def test_expired_text_gone_from_db_file(tmp_path):
    # A made-up window containing the sentence "zebra-umbrella-41", text_expires_at = T.
    store = Store(tmp_path / "cb.db")
    _import(store, tmp_path, _win("w-zebra", texts=("KEKW", "zebra-umbrella-41", "KEKW zebra-umbrella-41")),
            _win("w-otter", texts=("KEKW", "otter-kettle-12"), expires=T + timedelta(days=1)))
    assert b"zebra-umbrella-41" in _db_bytes(tmp_path)  # so the check below can fail

    r = expiry.run(store, now=T, imported_dir=tmp_path / "imported")
    assert (r.windows_expired, r.messages_deleted) == (1, 3)
    store.conn.close()
    blob = b"".join(p.read_bytes() for p in tmp_path.glob("cb.db*"))
    assert b"zebra-umbrella-41" not in blob
    assert b"otter-kettle-12" in blob  # text that hasn't expired stays
    assert [p.name for p in tmp_path.glob("cb.db*")] == ["cb.db"]  # no journal left behind


def test_expiry_keeps_labels_features_predictions(tmp_path):
    store = Store(tmp_path / "cb.db")
    _import(store, tmp_path, _win("w-1"))
    features = store.conn.execute("SELECT json FROM features WHERE window_id = 'w-1'").fetchone()[0]
    with store.lock, store.conn:
        store.conn.execute(
            "INSERT INTO labels (window_id, project, origin, provenance, labels_version, guide_version, answers_json,"
            " created_at) VALUES ('w-1', 'blind', 'aron', 'blind', 'v1', 'g1', '{}', ?)", (iso(NOW),))
        store.conn.execute(
            "INSERT INTO predictions (window_id, model_id, question, value_json, confidence, created_at)"
            " VALUES ('w-1', 'm-0', 'feeling', '\"neutral\"', 0.9, ?)", (iso(NOW),))
        store.conn.execute(
            "INSERT INTO assignments (window_id, exposure, assigned_at, seed) VALUES ('w-1', 'blind', ?, 's')",
            (iso(NOW),))
        store.conn.execute(
            "INSERT INTO manifests (model_id, rung, origin, path, created_at) VALUES ('m-0', 0, 'synthetic', 'm', ?)",
            (iso(NOW),))

    r = expiry.run(store, T, tmp_path / "imported")
    assert r.windows_expired == 1
    w = store.window("w-1")
    assert (w.text_state, w.channel, w.stream_id, w.anchor_kind, w.start_at) == (
        "expired", "fakechan01", "fs-01", "moment", START)
    assert store.as_window("w-1").messages == ()
    assert _count(store, "messages") == 0
    assert store.conn.execute("SELECT json FROM features WHERE window_id = 'w-1'").fetchone()[0] == features
    for table in ("labels", "predictions", "assignments", "manifests"):
        assert _count(store, table) == 1, table


def test_expiry_before_boundary_keeps_text(tmp_path):
    store = Store(tmp_path / "cb.db")
    _import(store, tmp_path, _win("w-1"))
    r = expiry.run(store, T - timedelta(seconds=1), tmp_path / "imported")
    assert (r.windows_expired, r.messages_deleted) == (0, 0)
    assert store.window("w-1").text_state == "present"
    assert len(store.messages("w-1")) == 2
    # now == text_expires_at is expired (Review Focus 5), even given in another offset.
    r = expiry.run(store, T.astimezone(timezone(timedelta(hours=-7))), tmp_path / "imported")
    assert (r.windows_expired, r.messages_deleted) == (1, 2)
    assert store.window("w-1").text_state == "expired"


def test_expiry_is_idempotent(tmp_path):
    store = Store(tmp_path / "cb.db")
    _import(store, tmp_path, _win("w-1"))
    expiry.run(store, T, tmp_path / "imported")
    again = expiry.run(store, T + timedelta(days=1), tmp_path / "imported")
    assert again == expiry.ExpiryResult(0, 0, 0, 0)


def test_expiry_deletes_stray_text_under_an_expired_window(tmp_path):
    store = Store(tmp_path / "cb.db")
    _import(store, tmp_path, _win("w-1"))
    expiry.run(store, T, tmp_path / "imported")
    with store.lock, store.conn:  # a row that should not exist, say from an interrupted run
        store.conn.execute("INSERT INTO messages VALUES ('w-1', 'm-stray', 0, 's1', 'quokka-anvil-90', '[]')")
    r = expiry.run(store, T, tmp_path / "imported")
    assert (r.windows_expired, r.messages_deleted) == (0, 1)
    store.close()
    assert b"quokka-anvil-90" not in _db_bytes(tmp_path)


def test_expiry_refuses_a_naive_now(tmp_path):
    store = Store(tmp_path / "cb.db")
    _import(store, tmp_path, _win("w-1"))
    with pytest.raises(ValueError):
        expiry.run(store, T.replace(tzinfo=None), tmp_path / "imported")
    assert store.window("w-1").text_state == "present"


# --- files (ruling C3) --------------------------------------------------------------------------------


def _fail_unlink(monkeypatch, target: Path):
    real = Path.unlink

    def locked(self, *args, **kwargs):
        if Path(self) == target:
            raise PermissionError("in use by another program")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", locked)


def test_expiry_retries_a_source_file_the_import_could_not_delete(tmp_path, monkeypatch):
    store = Store(tmp_path / "cb.db")
    src = _export_file(tmp_path / "inbox" / "e.json", _win("w-1"), _win("w-2"))
    with monkeypatch.context() as m:
        _fail_unlink(m, src)
        r = ingest.import_export(store, src, tmp_path / "imported", NOW)
    assert r.source_deleted is False and src.exists()

    done = expiry.run(store, NOW, tmp_path / "imported")
    assert done.files_deleted == 1
    assert not src.exists()
    assert store.conn.execute("SELECT deleted_at FROM imported_files").fetchone()[0] == iso(NOW)
    assert expiry.run(store, NOW, tmp_path / "imported").files_deleted == 0


def test_expiry_leaves_a_different_file_at_a_logged_path(tmp_path, monkeypatch):
    store = Store(tmp_path / "cb.db")
    src = _export_file(tmp_path / "inbox" / "e.json", _win("w-1"))
    with monkeypatch.context() as m:
        _fail_unlink(m, src)
        ingest.import_export(store, src, tmp_path / "imported", NOW)
    # Before any retry, a new export with a window this PC hasn't imported lands at the
    # same path. It isn't the logged file, so expiry must not delete it unimported.
    _export_file(src, _win("w-9"))
    r = expiry.run(store, NOW, tmp_path / "imported")
    assert r.files_deleted == 0
    assert src.exists()
    assert store.conn.execute("SELECT deleted_at FROM imported_files").fetchone()[0] is None

    # Once it is gone (imported, or removed by hand), the log row is closed.
    ingest.import_export(store, src, tmp_path / "imported", NOW)
    assert not src.exists()
    expiry.run(store, NOW, tmp_path / "imported")
    assert [r[0] for r in store.conn.execute("SELECT deleted_at FROM imported_files")] == [iso(NOW), iso(NOW)]

    # /2 (E9): a file at a logged path also stays until every stream it carries is logged for that path.
    logged = os.path.normcase(os.path.realpath(src))

    def streams_logged() -> list[tuple]:
        return store.conn.execute("SELECT path, stream_id, exported_at FROM imported_streams ORDER BY 2").fetchall()

    def open_rows() -> int:
        return _count(store, "imported_files", "deleted_at IS NULL")

    older = NOW - timedelta(hours=2)
    _v2_file(src, (_win("w-1"),), (_stream(),), exported_at=older)
    with monkeypatch.context() as m:
        _fail_unlink(m, src)
        assert not ingest.import_export(store, src, tmp_path / "imported", NOW).source_deleted
    assert streams_logged() == [(logged, "fs-01", iso(older))]

    _v2_file(src, (), (_stream("fs-02"),))  # streams only: no window to compare, and fs-02 isn't logged here
    assert expiry.run(store, NOW, tmp_path / "imported").files_deleted == 0
    assert src.exists()
    _v2_file(src, (_win("w-1"),), (_stream(),))  # a newer export of the logged stream
    assert expiry.run(store, NOW, tmp_path / "imported").files_deleted == 0
    assert src.exists()
    assert open_rows() == 1

    # Importing that newer file at the same path logs fs-01 again: the upsert keeps the newer exported_at.
    ingest.import_export(store, src, tmp_path / "imported", NOW)
    assert not src.exists()
    assert streams_logged() == [(logged, "fs-01", iso(NOW))]  # the first /2 import's row is still open

    # The same file imported twice while it can't be deleted: both commit, and the next run deletes it
    # (an equal exported_at counts as logged), closes every row for the path and drops its streams.
    _v2_file(src, (_win("w-1"),), (_stream(),))
    with monkeypatch.context() as m:
        _fail_unlink(m, src)
        for _ in range(2):
            assert not ingest.import_export(store, src, tmp_path / "imported", NOW).source_deleted
    assert src.exists()
    assert open_rows() == 3
    assert expiry.run(store, NOW, tmp_path / "imported").files_deleted == 1
    assert not src.exists()
    assert open_rows() == 0
    assert streams_logged() == []


def test_expiry_empties_the_imported_folder(tmp_path):
    # Ruling C3 keeps imported/ empty: anything there is a copy of chat that must not outlive its import.
    imported = tmp_path / "imported"
    (imported / "old").mkdir(parents=True)
    _export_file(imported / "copy.json", _win("w-1", texts=("KEKW", "lynx-ladder-64")))
    (imported / "old" / "notes.txt").write_text("lynx-ladder-64", encoding="utf-8")
    store = Store(tmp_path / "cb.db")
    r = expiry.run(store, NOW, imported)
    assert r.files_deleted == 2
    assert not any(p.is_file() for p in imported.rglob("*"))


def test_expiry_without_an_imported_folder(tmp_path):
    store = Store(tmp_path / "cb.db")
    assert expiry.run(store, NOW, tmp_path / "missing").files_deleted == 0


def _folder_link(link: Path, target: Path, kind: str) -> None:
    """Make ``link`` a junction or a directory symlink pointing at ``target``."""
    if kind == "junction":
        if os.name != "nt":
            pytest.skip("junctions are Windows only")
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)
        assert os.path.isjunction(link) and not link.is_symlink()  # os.walk would descend into it
    else:
        try:
            os.symlink(target, link, target_is_directory=True)
        except OSError as exc:  # Windows without Developer Mode or admin rights
            pytest.skip(f"no directory symlinks here: {exc}")
        assert link.is_symlink()


def _outside_folder(tmp_path) -> Path:
    outside = tmp_path / "outside"
    (outside / "deep").mkdir(parents=True)
    (outside / "precious.txt").write_text("keep me", encoding="utf-8")
    (outside / "deep" / "also.txt").write_text("keep me too", encoding="utf-8")
    return outside


def _outside_intact(outside: Path) -> bool:
    return (outside / "precious.txt").read_text(encoding="utf-8") == "keep me" and (
        outside / "deep" / "also.txt"
    ).read_text(encoding="utf-8") == "keep me too"


@pytest.mark.parametrize("kind", ["junction", "symlink"])
def test_expiry_never_follows_a_folder_link_out_of_imported(tmp_path, kind):
    # A link in imported/ to a folder outside it: expiry removes the link, never what it points at.
    outside = _outside_folder(tmp_path)
    imported = tmp_path / "imported"
    (imported / "sub").mkdir(parents=True)
    _folder_link(imported / "link", outside, kind)
    _folder_link(imported / "sub" / "nested", outside, kind)  # one level down as well
    (imported / "sub" / "copy.txt").write_text("lynx-ladder-64", encoding="utf-8")

    store = Store(tmp_path / "cb.db")
    r = expiry.run(store, NOW, imported)
    assert _outside_intact(outside)
    assert r.files_deleted == 1  # copy.txt only; removing a link deletes no file
    assert os.listdir(imported) == []  # both links and sub/ are gone
    assert expiry.run(store, NOW, imported).files_deleted == 0
    assert _outside_intact(outside)


@pytest.mark.parametrize("kind", ["junction", "symlink"])
def test_expiry_leaves_an_imported_folder_that_is_itself_a_link(tmp_path, kind):
    outside = _outside_folder(tmp_path)
    _folder_link(tmp_path / "imported", outside, kind)
    store = Store(tmp_path / "cb.db")
    assert expiry.run(store, NOW, tmp_path / "imported").files_deleted == 0
    assert _outside_intact(outside)


# --- Label Studio (rulings C4, C5) --------------------------------------------------------------------


def test_task_deleter_protocol():
    assert isinstance(FakeLS(), expiry.TaskDeleter)
    assert not isinstance(object(), expiry.TaskDeleter)


def test_expiry_deletes_label_studio_tasks_of_every_expired_window(tmp_path):
    store = Store(tmp_path / "cb.db")
    _import(store, tmp_path, _win("w-1"), _win("w-2", expires=T + timedelta(days=1)))
    _ls_tasks(store, [("w-1", "blind", 11), ("w-1", "repeat", 12), ("w-2", "suggested", 21)])

    # A run without a client expires the text and reports the tasks as pending.
    first = expiry.run(store, T, tmp_path / "imported")
    assert (first.windows_expired, first.ls_tasks_deleted, first.ls_tasks_pending) == (1, 0, 2)
    assert _count(store, "ls_tasks") == 3

    # The next run with a client deletes them, although the window is already expired (C4).
    ls = FakeLS()
    second = expiry.run(store, T, tmp_path / "imported", ls=ls)
    assert sorted(ls.deleted) == [11, 12]
    assert ls.compacts == 1
    assert (second.windows_expired, second.ls_tasks_deleted, second.ls_tasks_pending) == (0, 2, 0)
    assert second.ls_compaction_pending is False and second.ls_error is None
    assert [r[0] for r in store.conn.execute("SELECT task_id FROM ls_tasks")] == [21]

    # Nothing left to delete: no delete and no compact().
    third = expiry.run(store, T, tmp_path / "imported", ls=ls)
    assert (third.ls_tasks_deleted, ls.compacts) == (0, 1)


def test_expiry_keeps_rows_when_label_studio_is_unreachable(tmp_path):
    store = Store(tmp_path / "cb.db")
    _import(store, tmp_path, _win("w-1"))
    _ls_tasks(store, [("w-1", "blind", 11), ("w-1", "repeat", 12)])
    down = FakeLS(fail_on={12})
    r = expiry.run(store, T, tmp_path / "imported", ls=down)
    assert down.deleted == [11]
    assert (r.ls_tasks_deleted, r.ls_tasks_pending, r.ls_error) == (1, 1, "ConnectionError")
    assert [row[0] for row in store.conn.execute("SELECT task_id FROM ls_tasks")] == [12]
    assert r.windows_expired == 1  # the store's own expiry is not held up by Label Studio

    up = FakeLS()
    later = expiry.run(store, T, tmp_path / "imported", ls=up)
    assert up.deleted == [12]
    assert (later.ls_tasks_deleted, later.ls_tasks_pending, later.ls_error) == (1, 0, None)
    assert _count(store, "ls_tasks") == 0


def test_expiry_keeps_a_row_the_client_did_not_delete(tmp_path):
    # delete_tasks returns how many it deleted; a 0 without an exception means the task,
    # and its text, are still in Label Studio, so the ls_tasks row must stay (C4).
    store = Store(tmp_path / "cb.db")
    _import(store, tmp_path, _win("w-1"))
    _ls_tasks(store, [("w-1", "blind", 11), ("w-1", "repeat", 12)])
    ls = FakeLS(not_deleted={11})
    r = expiry.run(store, T, tmp_path / "imported", ls=ls)
    assert ls.calls == [[11]]  # stops at the first task that wasn't deleted, like on an exception
    assert (r.ls_tasks_deleted, r.ls_tasks_pending, r.ls_error) == (0, 2, "NotDeleted")
    assert (r.ls_compaction_pending, ls.compacts) == (False, 0)
    assert sorted(row[0] for row in store.conn.execute("SELECT task_id FROM ls_tasks")) == [11, 12]

    again = expiry.run(store, T, tmp_path / "imported", ls=ls)  # still pending, so asked again
    assert ls.calls == [[11], [11]]
    assert (again.ls_tasks_pending, again.ls_error) == (2, "NotDeleted")

    up = FakeLS()
    later = expiry.run(store, T, tmp_path / "imported", ls=up)
    assert sorted(up.deleted) == [11, 12]
    assert (later.ls_tasks_deleted, later.ls_tasks_pending, later.ls_error) == (2, 0, None)
    assert _count(store, "ls_tasks") == 0


def test_expiry_keeps_a_queued_deletion_the_client_did_not_delete(tmp_path):
    # The same for a task a deletion check queued (C5): its queue row stays too.
    store = Store(tmp_path / "cb.db")
    _import(store, tmp_path, _win("w-1"))
    _ls_tasks(store, [("w-1", "blind", 11)])
    with store.lock, store.conn:
        store.conn.execute("INSERT INTO ls_deletions (task_id) VALUES (11)")
    ls = FakeLS(not_deleted={11})
    r = expiry.run(store, NOW, tmp_path / "imported", ls=ls)  # w-1's text hasn't expired
    assert (r.ls_tasks_deleted, r.ls_tasks_pending, r.ls_error) == (0, 1, "NotDeleted")
    assert (_count(store, "ls_tasks"), _count(store, "ls_deletions")) == (1, 1)


def test_expiry_retries_compaction_until_it_succeeds(tmp_path):
    store = Store(tmp_path / "cb.db")
    _import(store, tmp_path, _win("w-1"))
    _ls_tasks(store, [("w-1", "blind", 11)])
    ls = FakeLS(compact_failures=1)
    r = expiry.run(store, T, tmp_path / "imported", ls=ls)
    assert (r.ls_tasks_deleted, r.ls_compaction_pending, r.ls_error) == (1, True, "RuntimeError")
    assert _count(store, "ls_tasks") == 0

    without = expiry.run(store, T, tmp_path / "imported")
    assert (without.ls_tasks_pending, without.ls_compaction_pending) == (0, True)

    again = expiry.run(store, T, tmp_path / "imported", ls=ls)
    assert ls.compacts == 1
    assert (again.ls_tasks_deleted, again.ls_compaction_pending, again.ls_error) == (0, False, None)
    assert expiry.run(store, T, tmp_path / "imported", ls=ls).ls_compaction_pending is False
    assert ls.compacts == 1


def test_expiry_result_defaults_to_nothing_pending():
    r = expiry.ExpiryResult(1, 2, 3, 4)
    assert (r.windows_expired, r.messages_deleted, r.files_deleted, r.ls_tasks_deleted) == (1, 2, 3, 4)
    assert (r.ls_tasks_pending, r.ls_compaction_pending, r.ls_error) == (0, False, None)


# --- per-stream activity (Slice 1, Task 2; Q10) -----------------------------------------------------

S1 = START - timedelta(minutes=10)  # fs-01's made-up stream start
A_DUE = S1 + timedelta(days=30)  # its activity counts expire here (UTC started_at + 30 days)
TITLE = "Made-up stream heron-31"
COUNTS = b"40001,40002,40003"  # how the stored counts start (compact JSON)


def _stream(stream_id="fs-01", *, started_at=S1, minutes=20, title=TITLE) -> StreamActivity:
    return StreamActivity(
        stream_id=stream_id,
        channel="fakechan01",
        started_at=started_at,
        ended_at=started_at + timedelta(minutes=minutes),
        title=title,
        bin_s=10,
        chatters=tuple(None if i == 5 else 40001 + i for i in range(minutes * 6)),
    )


def _win_at(window_id: str, start: datetime) -> Window:
    return dataclasses.replace(
        _win(window_id), start_at=start, anchor_at=start + timedelta(seconds=30), end_at=start + timedelta(seconds=120)
    )


def _v2_file(path: Path, windows=(), streams=(), *, exported_at=NOW) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(dump_export(Export(EXPORT_FORMAT_V2, exported_at, "vod", tuple(windows), tuple(streams))))
    return path


def _import_v2(store, tmp_path, windows=(), streams=(), *, name="e2.json"):
    src = _v2_file(tmp_path / "inbox" / name, windows, streams)
    return ingest.import_export(store, src, tmp_path / "imported", NOW)


def _markers(store) -> list[str]:
    return [r[0] for r in store.conn.execute("SELECT stream_id FROM stream_activity_expired ORDER BY 1")]


def _activity_table(store) -> list[tuple]:
    return store.conn.execute("SELECT * FROM stream_activity ORDER BY stream_id").fetchall()


def test_activity_survives_text_expiry_before_30_days(tmp_path):
    store = Store(tmp_path / "cb.db")
    _import_v2(store, tmp_path, (_win("w-1"),), (_stream(),))
    before = _activity_table(store)
    assert len(before) == 1

    r = expiry.run(store, T, tmp_path / "imported")
    assert r.windows_expired == 1  # P0 text expiry ran
    assert _activity_table(store) == before
    assert store.stream_activity("fs-01", now=T) == _stream()
    assert store.activity_state("fs-01", now=T) == "active"

    expiry.run(store, A_DUE - timedelta(microseconds=1), tmp_path / "imported")
    assert _activity_table(store) == before
    assert _markers(store) == []


def test_activity_expires_exactly_30_days_after_stream_start(tmp_path):
    store = Store(tmp_path / "cb.db")
    _import_v2(store, tmp_path, (_win("w-1"),), (_stream(),))
    assert TITLE.encode() in _db_bytes(tmp_path) and COUNTS in _db_bytes(tmp_path)  # so the checks below can fail

    just_before = A_DUE - timedelta(microseconds=1)
    expiry.run(store, just_before, tmp_path / "imported")
    assert store.activity_state("fs-01", now=just_before) == "active"
    assert _count(store, "stream_activity") == 1

    # now == started_at + 30 days is expired, even given in another offset.
    expiry.run(store, A_DUE.astimezone(timezone(timedelta(hours=-7))), tmp_path / "imported")
    assert _count(store, "stream_activity") == 0
    assert _markers(store) == ["fs-01"]  # w-1 still names the stream
    assert store.activity_state("fs-01", now=A_DUE) == "expired"

    expiry.run(store, A_DUE + timedelta(days=1), tmp_path / "imported")
    assert _count(store, "stream_activity") == 0
    assert _markers(store) == ["fs-01"]
    store.close()
    blob = _db_bytes(tmp_path)
    assert TITLE.encode() not in blob
    assert COUNTS not in blob


def test_marker_goes_when_no_window_needs_it(tmp_path):
    store = Store(tmp_path / "cb.db")
    old = NOW - timedelta(days=31)
    r = _import_v2(
        store, tmp_path, (_win_at("w-1", old + timedelta(minutes=5)),), (_stream("fs-01", started_at=old),
                                                                          _stream("fs-55", started_at=old))
    )  # fmt: skip
    assert (r.streams_written, r.streams_expired_skipped) == (0, 2)
    assert _markers(store) == ["fs-01"]  # E1: both already expired; import gives fs-55 (no window) no marker
    # Only a direct store call can leave a marker no window names. The sweep is the safety net for it.
    with store.transaction():
        assert store.put_stream_activity(_stream("fs-55", started_at=old), NOW, NOW) == "expired"
    assert _markers(store) == ["fs-01", "fs-55"]

    expiry.run(store, NOW, tmp_path / "imported")
    assert _markers(store) == ["fs-01"]  # fs-55 has no window that needs it
    assert store.activity_state("fs-01", now=NOW) == "expired"
    assert store.activity_state("fs-55", now=NOW) == "missing"


def test_expired_activity_state_survives_payload_removal(tmp_path):
    store = Store(tmp_path / "cb.db")
    _import_v2(store, tmp_path, (_win("w-1"),), (_stream(),))
    expiry.run(store, A_DUE + timedelta(days=2), tmp_path / "imported")  # the sweep's clock is later than the page's

    for page_now in (NOW, A_DUE - timedelta(days=1), A_DUE, A_DUE + timedelta(days=5)):
        assert store.activity_state("fs-01", now=page_now) == "expired"
        assert store.stream_activity("fs-01", now=page_now) is None
        assert store.channel_activity("fakechan01", now=page_now) == []
    assert _count(store, "stream_activity") == 0
    assert store.conn.execute("SELECT * FROM stream_activity_expired").fetchall() == [("fs-01",)]
    store.close()
    blob = _db_bytes(tmp_path)
    assert TITLE.encode() not in blob
    assert COUNTS not in blob


@pytest.mark.parametrize(
    ("older_start", "sweep_at", "again_at"),
    [
        # An older export with an earlier source clock pulls the deadline in a day (E3); the sweep runs past it.
        (S1 - timedelta(days=1), A_DUE - timedelta(hours=23), A_DUE - timedelta(hours=22)),
        # The sweep ran with a later clock than the re-import's (E2's demo case).
        (None, A_DUE + timedelta(days=2), NOW),
    ],
    ids=["earlier-source-clock", "later-sweep-clock"],
)
def test_swept_activity_never_comes_back(tmp_path, older_start, sweep_at, again_at):
    # Q10 / Review Focus 4: counts that expired and were swept never come back with a re-import.
    # fs-01 has a window, so its marker stays. fs-02 has no window and fs-03's only window is not
    # English: nothing could keep a marker for them (E2), so they never get an activity line at all.
    store = Store(tmp_path / "cb.db")
    ids = ("fs-01", "fs-02", "fs-03")
    v1 = tuple(_stream(s) for s in ids)
    not_en = dataclasses.replace(_win("w-3"), stream_id="fs-03", language="fr")

    def imp(streams, name, *, now, windows=(), exported_at=NOW):
        src = _v2_file(tmp_path / "inbox" / name, windows, streams, exported_at=exported_at)
        return ingest.import_export(store, src, tmp_path / "imported", now)

    def seen(now) -> list[str]:
        return [a.stream_id for a in store.channel_activity("fakechan01", now=now)]

    r = imp(v1, "v1.json", now=NOW, windows=(_win("w-1"), not_en))
    assert (r.windows_added, r.skipped_language, r.streams_written, r.streams_expired_skipped) == (1, 1, 1, 0)
    assert seen(NOW) == ["fs-01"]
    assert _count(store, "stream_activity") == 1
    assert [store.activity_state(s, now=NOW) for s in ids] == ["active", "missing", "missing"]
    if older_start is not None:
        older = tuple(_stream(s, started_at=older_start) for s in ids)
        r = imp(older, "older.json", now=NOW, exported_at=NOW - timedelta(hours=1))
        assert (r.streams_written, r.streams_expired_skipped) == (0, 0)  # fs-01 kept, its deadline pulled in

    expiry.run(store, sweep_at, tmp_path / "imported")
    assert _count(store, "stream_activity") == 0
    assert _markers(store) == ["fs-01"]  # w-1 keeps it

    r = imp(v1, "again.json", now=again_at)
    assert (r.streams_written, r.streams_expired_skipped) == (0, 1)  # fs-01 stays expired; fs-02/03 get no line
    for now in (NOW, again_at):
        assert seen(now) == []
        assert [store.stream_activity(s, now=now) for s in ids] == [None, None, None]
    assert store.activity_state("fs-01", now=again_at) == "expired"
    assert _count(store, "stream_activity") == 0


def test_imported_streams_go_with_their_source(tmp_path, monkeypatch):
    store = Store(tmp_path / "cb.db")
    inbox = tmp_path / "inbox"

    def rows(path: Path) -> list[tuple]:
        logged = os.path.normcase(os.path.realpath(path))
        return store.conn.execute(
            "SELECT stream_id, exported_at FROM imported_streams WHERE path = ? ORDER BY 1", (logged,)
        ).fetchall()

    # A source deleted at import leaves nothing for its path.
    a = _v2_file(inbox / "a.json", (_win("w-1"),), (_stream(),))
    assert ingest.import_export(store, a, tmp_path / "imported", NOW).source_deleted
    assert _count(store, "imported_streams") == 0

    # A streams-only source the import could not delete is logged until expiry deletes it. fs-01's
    # line (w-1 names it) is swept at 30 days; fs-88 has no window, so it never gets a line.
    b = _v2_file(inbox / "b.json", (), (_stream(), _stream("fs-88", title="Made-up stream osprey-52")))
    with monkeypatch.context() as m:
        _fail_unlink(m, b)
        assert not ingest.import_export(store, b, tmp_path / "imported", NOW).source_deleted
    assert rows(b) == [("fs-01", iso(NOW)), ("fs-88", iso(NOW))]
    assert _count(store, "stream_activity") == 1

    # A source that still can't be deleted keeps its rows.
    c = _v2_file(inbox / "c.json", (), (_stream("fs-89"),))
    with monkeypatch.context() as m:
        _fail_unlink(m, c)
        assert not ingest.import_export(store, c, tmp_path / "imported", NOW).source_deleted
        r = expiry.run(store, A_DUE, tmp_path / "imported")  # every stream's counts are past 30 days here
    assert r.files_deleted == 1
    assert not b.exists() and c.exists()
    assert rows(b) == []
    assert rows(c) == [("fs-89", iso(NOW))]
    assert _count(store, "stream_activity") == 0
    assert _markers(store) == ["fs-01"]  # only w-1 still needs one

    # A row whose path has no open import left (a crash between two writes, say) goes on the next run.
    with store.transaction() as conn:
        conn.execute("UPDATE imported_files SET deleted_at = ? WHERE deleted_at IS NULL", (iso(NOW),))
    assert rows(c) == [("fs-89", iso(NOW))]
    expiry.run(store, A_DUE, tmp_path / "imported")
    assert _count(store, "imported_streams") == 0

    store.close()
    blob = _db_bytes(tmp_path)
    assert b"fs-88" not in blob  # no window, no marker, no log: the stream ID is gone
    assert b"osprey-52" not in blob


# --- F1 tally (glance call 2): lives as long as the stream's activity counts -----------------------------

FACE_TEXTS = ("KEKW",) * 6 + ("hi chat",)  # six people type KEKW, one line has no emote


def _faces_count(store, stream_id="fs-01") -> int:
    return _count(store, "moment_emotes", "stream_id = ?", stream_id)


def test_tally_survives_text_expiry(tmp_path):
    from chatbrain.faces import Tally

    store = Store(tmp_path / "cb.db")
    _import_v2(store, tmp_path, (_win("w-1", FACE_TEXTS),), (_stream(),))
    want = {"w-1": Tally("fake-kekw", "KEKW", lines=6, of_lines=7, people=6)}
    assert store.moment_faces("fs-01", NOW) == want
    assert store.conn.execute("SELECT expires_at FROM moment_emotes").fetchone()[0] == iso(A_DUE)

    r = expiry.run(store, T, tmp_path / "imported")
    assert (r.windows_expired, r.messages_deleted) == (1, 7)  # the text is gone
    assert store.window("w-1").text_state == "expired"
    assert store.moment_faces("fs-01", T) == want  # the count stays
    assert store.backfill_moment_emotes(T) == 0  # and nothing tries to rebuild it


def test_tally_dies_with_counts_at_30d(tmp_path):
    store = Store(tmp_path / "cb.db")
    w2 = dataclasses.replace(_win("w-2", FACE_TEXTS), stream_id="fs-02")
    _import_v2(store, tmp_path, (_win("w-1", FACE_TEXTS), w2), (_stream(),))
    assert _faces_count(store) == 1 and _faces_count(store, "fs-02") == 1

    just_before = A_DUE - timedelta(microseconds=1)
    expiry.run(store, just_before, tmp_path / "imported")
    assert set(store.moment_faces("fs-01", just_before)) == {"w-1"}
    assert _faces_count(store) == 1

    # now == started_at + 30 days is expired: the read hides the face before any sweep...
    assert store.moment_faces("fs-01", A_DUE) == {}
    assert _faces_count(store) == 1
    # ...and the sweep deletes the row; the marker stays.
    expiry.run(store, A_DUE, tmp_path / "imported")
    assert _faces_count(store) == 0
    assert _markers(store) == ["fs-01"]
    assert store.moment_faces("fs-01", NOW) == {}

    # A marked stream's rows go too, whatever their own deadline says.
    fs02_deadline = store.conn.execute("SELECT expires_at FROM moment_emotes WHERE stream_id = 'fs-02'").fetchone()[0]
    assert fs02_deadline > iso(A_DUE)  # no counts for fs-02: its window start + 30 days
    with store.transaction() as conn:
        conn.execute("INSERT INTO stream_activity_expired (stream_id) VALUES ('fs-02')")
    assert store.moment_faces("fs-02", NOW) == {}
    expiry.run(store, A_DUE, tmp_path / "imported")
    assert _faces_count(store, "fs-02") == 0
    assert store.backfill_moment_emotes(NOW) == 0  # a marked or expired stream is never filled again
    assert _count(store, "moment_emotes") == 0
