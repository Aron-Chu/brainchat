import dataclasses
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from chatbrain.export_format import Emote, Message, StreamActivity, Window, iso
from chatbrain.store import Store, WindowRow

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)
START = NOW - timedelta(days=3)


def _window(window_id="w-0001", kind="moment", channel="fakechan01", n=3, expires=NOW + timedelta(days=10)) -> Window:
    msgs = tuple(
        Message(f"{window_id}-m{i}", 31000 + i * 1000, f"s{i + 1}", "KEKW" if i % 2 == 0 else "LUL that was so bad",
                (Emote("fake-kekw", "KEKW"),) if i % 2 == 0 else (Emote("fake-lul", "LUL"),))
        for i in range(n)
    )
    return Window(
        window_id=window_id,
        channel=channel,
        stream_id="fs-01",
        anchor_kind=kind,
        anchor_at=START + timedelta(seconds=30),
        start_at=START,
        end_at=START + timedelta(seconds=120),
        language="en",
        text_expires_at=expires,
        messages=msgs,
    )


def test_store_pragmas(tmp_path):
    path = tmp_path / "cb.db"
    # A db left in WAL mode by someone else: WAL is persistent, so only the Store's
    # own pragma can turn it back to DELETE (a new file would be DELETE anyway).
    plain = sqlite3.connect(path)
    assert plain.execute("PRAGMA journal_mode = WAL").fetchone()[0] == "wal"
    with plain:
        plain.execute("CREATE TABLE leftover (x)")
        plain.execute("INSERT INTO leftover VALUES (1)")
    plain.close()
    probe = sqlite3.connect(path)
    assert probe.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    probe.close()

    s = Store(path)
    assert s.conn.execute("PRAGMA secure_delete").fetchone()[0] == 1
    assert s.conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    s.add_window(_window(), "vod", NOW)
    s.close()
    assert not (tmp_path / "cb.db-wal").exists()
    assert not (tmp_path / "cb.db-shm").exists()


def test_store_extra_pragmas(tmp_path):
    s = Store(tmp_path / "cb.db")
    assert s.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert s.conn.execute("PRAGMA temp_store").fetchone()[0] == 2  # MEMORY


def test_add_window_is_idempotent(tmp_path):
    s = Store(tmp_path / "cb.db")
    w = _window()
    assert s.add_window(w, "vod", NOW) is True
    count = s.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    assert count == 3
    assert s.add_window(w, "vod", NOW + timedelta(hours=1)) is False
    assert s.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == count
    assert s.window(w.window_id).imported_at == NOW


def test_add_window_refuses_already_expired_text(tmp_path):
    s = Store(tmp_path / "cb.db")
    w = _window(expires=NOW)
    with pytest.raises(ValueError):
        s.add_window(w, "vod", NOW)
    assert s.window_ids() == []


def test_add_window_rejects_naive_now_and_bad_source(tmp_path):
    s = Store(tmp_path / "cb.db")
    with pytest.raises(ValueError):
        s.add_window(_window(), "vod", datetime(2026, 11, 6, 12, 0))
    with pytest.raises(ValueError):
        s.add_window(_window(), "clip", NOW)
    assert s.window_ids() == []


def test_window_row(tmp_path):
    s = Store(tmp_path / "cb.db")
    w = _window()
    s.add_window(w, "live", NOW)
    row = s.window(w.window_id)
    assert isinstance(row, WindowRow)
    assert row == WindowRow(
        window_id=w.window_id,
        channel=w.channel,
        stream_id=w.stream_id,
        anchor_kind=w.anchor_kind,
        anchor_at=w.anchor_at,
        start_at=w.start_at,
        end_at=w.end_at,
        language=w.language,
        text_expires_at=w.text_expires_at,
        source="live",
        imported_at=NOW,
        text_state="present",
        picked_rare=False,
    )
    assert row.anchor_at.tzinfo is not None and row.anchor_at.utcoffset() == timedelta(0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.text_state = "expired"  # type: ignore[misc]
    assert {f.name for f in dataclasses.fields(WindowRow)} == (
        {f.name for f in dataclasses.fields(Window)} - {"messages"}
    ) | {"source", "imported_at", "text_state", "picked_rare"}


def test_unknown_window_raises_key_error(tmp_path):
    s = Store(tmp_path / "cb.db")
    with pytest.raises(KeyError):
        s.window("nope")
    with pytest.raises(KeyError):
        s.as_window("nope")
    with pytest.raises(KeyError):
        s.messages("nope")


def test_as_window_round_trips_and_messages_are_ordered(tmp_path):
    s = Store(tmp_path / "cb.db")
    # Inserted out of (rel_ms, message_id) order, and with message_id order unlike
    # time order, so neither insertion order nor the primary-key index gives the
    # right answer: only ORDER BY rel_ms, message_id does. m-b and m-c tie on rel_ms.
    inserted = (
        Message("m-c", 5000, "s1", "KEKW", (Emote("fake-kekw", "KEKW"),)),
        Message("m-a", 9000, "s2", "LUL", (Emote("fake-lul", "LUL"),)),
        Message("m-e", 1000, "s3", "hi chat", ()),
        Message("m-b", 5000, "s2", "@s1 same", ()),
        Message("m-d", 3000, "s1", "what", ()),
    )
    w = dataclasses.replace(_window(n=0), messages=inserted)
    s.add_window(w, "vod", NOW)
    ordered = tuple(sorted(inserted, key=lambda m: (m.rel_ms, m.message_id)))
    assert [m.message_id for m in ordered] == ["m-e", "m-d", "m-b", "m-c", "m-a"]
    assert ordered != inserted
    assert s.as_window(w.window_id) == dataclasses.replace(w, messages=ordered)
    assert s.messages(w.window_id) == list(ordered)
    # The usual case: a window already in order round-trips unchanged.
    w5 = _window("w-0005", n=5)
    s.add_window(w5, "vod", NOW)
    assert s.as_window(w5.window_id) == w5


def test_as_window_has_no_messages_once_expired(tmp_path):
    s = Store(tmp_path / "cb.db")
    w = _window()
    s.add_window(w, "vod", NOW)
    with s.conn:
        s.conn.execute("DELETE FROM messages WHERE window_id = ?", (w.window_id,))
        s.conn.execute("UPDATE windows SET text_state = 'expired' WHERE window_id = ?", (w.window_id,))
    got = s.as_window(w.window_id)
    assert got.messages == ()
    assert got == dataclasses.replace(w, messages=())
    assert s.window(w.window_id).text_state == "expired"


def test_as_window_hides_text_of_an_expired_window_even_if_rows_remain(tmp_path):
    s = Store(tmp_path / "cb.db")
    w = _window()
    s.add_window(w, "vod", NOW)
    with s.conn:
        s.conn.execute("UPDATE windows SET text_state = 'expired' WHERE window_id = ?", (w.window_id,))
    assert s.as_window(w.window_id).messages == ()


def test_window_ids_filters(tmp_path):
    s = Store(tmp_path / "cb.db")
    s.add_window(_window("w-b", "moment", "fakechan01"), "vod", NOW)
    s.add_window(_window("w-a", "ordinary", "fakechan02"), "vod", NOW)
    s.add_window(_window("w-c", "ordinary", "fakechan01"), "live", NOW)
    assert s.window_ids() == ["w-a", "w-b", "w-c"]
    assert s.window_ids(anchor_kind="ordinary") == ["w-a", "w-c"]
    assert s.window_ids(anchor_kind="ordinary", channel="fakechan01") == ["w-c"]
    assert s.window_ids(source="live") == ["w-c"]
    assert s.window_ids(text_state="present") == ["w-a", "w-b", "w-c"]
    with s.conn:
        s.conn.execute("UPDATE windows SET picked_rare = 1 WHERE window_id = 'w-b'")
    assert s.window_ids(picked_rare=True) == ["w-b"]
    assert s.window_ids(picked_rare=False) == ["w-a", "w-c"]
    assert s.window("w-b").picked_rare is True
    assert s.window_ids(text_expires_at=NOW + timedelta(days=10)) == ["w-a", "w-b", "w-c"]
    with pytest.raises(ValueError):
        s.window_ids(messages="x")
    with pytest.raises(ValueError):
        s.window_ids(**{"1=1; DROP TABLE windows; --": 1})


def test_empty_window_is_stored(tmp_path):
    s = Store(tmp_path / "cb.db")
    w = _window(n=0)
    assert s.add_window(w, "vod", NOW)
    assert s.as_window(w.window_id) == w
    assert s.messages(w.window_id) == []


EXPECTED_COLUMNS = {
    "windows": {"window_id", "channel", "stream_id", "anchor_kind", "anchor_at", "start_at", "end_at", "language",
                "text_expires_at", "source", "imported_at", "text_state", "picked_rare"},
    "messages": {"window_id", "message_id", "rel_ms", "speaker", "text", "emotes_json"},
    "features": {"window_id", "json"},
    "assignments": {"window_id", "exposure", "assigned_at", "seed"},
    "repeats": {"window_id", "due_at"},
    "labels": {"id", "window_id", "project", "origin", "provenance", "labels_version", "guide_version", "answers_json",
               "not_sure_json", "cause_note", "cause_source", "suggestion_json", "same_as_suggestion", "teacher_model",
               "prompt_version", "lead_time_s", "ls_annotation_id", "created_at"},
    "suggestions": {"window_id", "teacher_model", "prompt_version", "labels_version", "answers_json", "error", "created_at"},
    "predictions": {"window_id", "model_id", "question", "value_json", "confidence", "created_at"},
    "deletion_checks": {"checked_at", "windows_checked", "messages_dropped"},
    "imported_files": {"path", "window_ids_json", "imported_at", "deleted_at"},
    "ls_tasks": {"window_id", "project_kind", "task_id"},
    "manifests": {"model_id", "rung", "origin", "path", "created_at", "retired_at", "file_deleted_at"},
    "text_drops": {"window_id", "messages_left"},
}


def test_schema_has_every_table_and_column(tmp_path):
    s = Store(tmp_path / "cb.db")
    for table, cols in EXPECTED_COLUMNS.items():
        got = {r[1] for r in s.conn.execute(f"PRAGMA table_info({table})")}
        assert cols <= got, (table, cols - got)
    # The messages primary key is window_id + message_id.
    pk = [r[1] for r in sorted(s.conn.execute("PRAGMA table_info(messages)"), key=lambda r: r[5]) if r[5]]
    assert pk == ["window_id", "message_id"]


def test_reopen_keeps_data(tmp_path):
    path = tmp_path / "cb.db"
    s = Store(path)
    s.add_window(_window(), "vod", NOW)
    s.close()
    s2 = Store(path)
    assert s2.window_ids() == ["w-0001"]
    assert len(s2.messages("w-0001")) == 3


def _label(window_id="w-0001", **over):
    row = {
        "window_id": window_id, "project": "blind", "origin": "aron", "provenance": "blind",
        "labels_version": "cb_labels_2026_11_06_v1", "guide_version": "cb_guide_2026_11_06_v1",
        "answers_json": "{}", "not_sure_json": "{}", "ls_annotation_id": 1, "created_at": "2026-11-06T12:00:00.000000Z",
    }
    row.update(over)
    cols = ", ".join(row)
    marks = ", ".join("?" for _ in row)
    return f"INSERT INTO labels ({cols}) VALUES ({marks})", tuple(row.values())


def test_label_constraints(tmp_path):
    s = Store(tmp_path / "cb.db")
    s.add_window(_window(), "vod", NOW)
    with s.conn:
        s.conn.execute(*_label())
    # ls_annotation_id is unique.
    with pytest.raises(sqlite3.IntegrityError), s.conn:
        s.conn.execute(*_label())
    # Values outside the brief's sets are refused.
    for bad in ({"project": "other"}, {"origin": "teacher"}, {"provenance": "guessed"}):
        with pytest.raises(sqlite3.IntegrityError), s.conn:
            s.conn.execute(*_label(ls_annotation_id=None, **bad))
    # Labels must point at a stored window.
    with pytest.raises(sqlite3.IntegrityError), s.conn:
        s.conn.execute(*_label("no-such-window", ls_annotation_id=None))
    # Two labels without a Label Studio annotation are fine.
    with s.conn:
        s.conn.execute(*_label(ls_annotation_id=None, project="page", provenance="corrected-on-page"))
        s.conn.execute(*_label(ls_annotation_id=None, origin="synthetic", provenance="synthetic"))
    assert s.conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == 3


def test_label_provenance_values(tmp_path):
    s = Store(tmp_path / "cb.db")
    s.add_window(_window(), "vod", NOW)
    allowed = ("blind", "accepted", "corrected", "repeat", "corrected-on-page", "synthetic", "unsuggested")
    with s.conn:
        for p in allowed:
            s.conn.execute(*_label(ls_annotation_id=None, provenance=p))
    got = {r[0] for r in s.conn.execute("SELECT provenance FROM labels")}
    assert got == set(allowed)
    # A suggested-project label whose suggestion only held an error (ruling C15).
    with s.conn:
        s.conn.execute(*_label(ls_annotation_id=None, project="suggested", provenance="unsuggested",
                               same_as_suggestion=None))


def test_label_cause_source_is_checked_by_the_scheme_not_sql(tmp_path):
    # The scheme is the only source of value lists (ruling C1): SQL takes any text
    # and scheme.validate_cause_source refuses what isn't in scheme.cause_sources.
    from chatbrain import scheme

    s = Store(tmp_path / "cb.db")
    s.add_window(_window(), "vod", NOW)
    with s.conn:
        for v in (None, "clip", "game", "stream_title", "a_future_source"):
            s.conn.execute(*_label(ls_annotation_id=None, cause_source=v))
    with pytest.raises(ValueError):
        scheme.validate_cause_source(scheme.load(), "a_future_source")


def test_imported_files_is_a_log(tmp_path):
    s = Store(tmp_path / "cb.db")
    cols = [(r[1], r[2], r[3], r[5]) for r in s.conn.execute("PRAGMA table_info(imported_files)")]
    # name, type, notnull, pk
    assert cols == [
        ("path", "TEXT", 1, 0),
        ("window_ids_json", "TEXT", 1, 0),
        ("imported_at", "TEXT", 1, 0),
        ("deleted_at", "TEXT", 0, 0),
    ]
    sql = "INSERT INTO imported_files (path, window_ids_json, imported_at, deleted_at) VALUES (?, ?, ?, ?)"
    with s.conn:
        # The same path twice is fine: a re-export with the same name is a new log row.
        s.conn.execute(sql, ("inbox/a.json", '["w-0001"]', "2026-11-06T12:00:00.000000Z", None))
        s.conn.execute(sql, ("inbox/a.json", "[]", "2026-11-07T12:00:00.000000Z", "2026-11-07T12:00:01.000000Z"))
    assert s.conn.execute("SELECT COUNT(*) FROM imported_files").fetchone()[0] == 2
    with pytest.raises(sqlite3.IntegrityError), s.conn:
        s.conn.execute(sql, ("inbox/b.json", "[]", None, None))


def test_manifest_origin(tmp_path):
    s = Store(tmp_path / "cb.db")
    sql = "INSERT INTO manifests (model_id, rung, origin, path, created_at) VALUES (?, 1, ?, 'm', '2026-11-06T12:00:00.000000Z')"
    with s.conn:
        s.conn.execute(sql, ("m-aron", "aron"))
        s.conn.execute(sql, ("m-syn", "synthetic"))
    for bad in ("teacher", None):
        with pytest.raises(sqlite3.IntegrityError), s.conn:
            s.conn.execute(sql, (f"m-{bad}", bad))
    with pytest.raises(sqlite3.IntegrityError), s.conn:
        s.conn.execute("INSERT INTO manifests (model_id, rung, path, created_at) VALUES ('m-none', 1, 'm', 'x')")


# --- thread safety (T3) -----------------------------------------------------------------


def test_store_is_shared_safely_across_threads(tmp_path):
    import threading

    s = Store(tmp_path / "cb.db")
    windows = [_window(f"w-{i:04d}", n=4) for i in range(1, 6)]
    for w in windows:
        s.add_window(w, "vod", NOW)
    want_ids = [w.window_id for w in windows]
    errors: list[BaseException] = []
    start = threading.Barrier(8)

    def reader(k: int) -> None:
        try:
            start.wait()
            for j in range(50):
                assert s.window_ids() == want_ids
                w = windows[(k + j) % len(windows)]
                assert s.as_window(w.window_id) == w
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=reader, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in threads)
    assert errors == []


@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.window_ids(),
        lambda s: s.window("w-0001"),
        lambda s: s.as_window("w-0001"),
        lambda s: s.messages("w-0001"),
        lambda s: s.add_window(_window("w-0002"), "vod", NOW),
        lambda s: s.close(),
    ],
    ids=["window_ids", "window", "as_window", "messages", "add_window", "close"],
)
def test_store_methods_wait_for_the_lock(tmp_path, call):
    import threading

    s = Store(tmp_path / "cb.db")
    s.add_window(_window(), "vod", NOW)
    done = threading.Event()

    def run() -> None:
        call(s)
        done.set()

    with s.lock:
        t = threading.Thread(target=run)
        t.start()
        assert not done.wait(0.3), "the call ran while another thread held the store's lock"
    assert done.wait(10)
    t.join(10)


def test_window_constraints(tmp_path):
    s = Store(tmp_path / "cb.db")
    s.add_window(_window(), "vod", NOW)
    with pytest.raises(sqlite3.IntegrityError), s.conn:
        s.conn.execute("UPDATE windows SET text_state = 'gone'")
    with pytest.raises(sqlite3.IntegrityError), s.conn:
        s.conn.execute("UPDATE windows SET picked_rare = 2")


def test_suggestion_holds_answers_or_error(tmp_path):
    s = Store(tmp_path / "cb.db")
    s.add_window(_window(), "vod", NOW)
    sql = ("INSERT INTO suggestions (window_id, teacher_model, prompt_version, labels_version, answers_json, error, created_at)"
           " VALUES ('w-0001', 'qwen3:8b', 'p1', 'l1', ?, ?, '2026-11-06T12:00:00.000000Z')")
    with s.conn:
        s.conn.execute(sql, ('{"doing": {"value": ["laughing"], "not_sure": []}}', None))
        s.conn.execute(sql, (None, "timeout"))
    for bad in ((None, None), ("{}", "timeout")):
        with pytest.raises(sqlite3.IntegrityError), s.conn:
            s.conn.execute(sql, bad)


# --- per-stream activity (Slice 1, Task 2; Q10) ------------------------------------------

S0 = datetime(2026, 10, 20, 18, 0, tzinfo=timezone.utc)  # a made-up stream start


def _activity(stream_id="fs-01", channel="fakechan01", started_at=S0, minutes=10, title="Made-up stream 1", base=5):
    return StreamActivity(
        stream_id=stream_id,
        channel=channel,
        started_at=started_at,
        ended_at=started_at + timedelta(minutes=minutes),
        title=title,
        bin_s=10,
        chatters=tuple(None if i == 3 else base + i % 4 for i in range(minutes * 6)),
    )


def _put(s: Store, a: StreamActivity, exported_at: datetime, now: datetime) -> str:
    with s.transaction():
        return s.put_stream_activity(a, exported_at, now)


def _expires_at(s: Store, stream_id: str = "fs-01") -> str | None:
    row = s.conn.execute("SELECT expires_at FROM stream_activity WHERE stream_id = ?", (stream_id,)).fetchone()
    return row[0] if row else None


def _marked(s: Store, stream_id: str = "fs-01") -> bool:
    return s.conn.execute("SELECT 1 FROM stream_activity_expired WHERE stream_id = ?", (stream_id,)).fetchone() is not None


def test_put_stream_activity_runs_inside_a_transaction(tmp_path):
    s = Store(tmp_path / "cb.db")
    with pytest.raises(RuntimeError):
        s.put_stream_activity(_activity(), S0 + timedelta(days=1), S0 + timedelta(days=2))
    assert s.conn.execute("SELECT COUNT(*) FROM stream_activity").fetchone()[0] == 0


def test_put_stream_activity_newer_wins(tmp_path):
    s = Store(tmp_path / "cb.db")
    now = S0 + timedelta(days=2)
    first = _activity(title="Made-up stream 1", base=5)
    exported = S0 + timedelta(days=1)
    assert _put(s, first, exported, now) == "written"
    assert s.stream_activity("fs-01", now=now) == first
    row = s.conn.execute("SELECT exported_at, imported_at, expires_at, bin_s FROM stream_activity").fetchone()
    assert row == (iso(exported), iso(now), iso(S0 + timedelta(days=30)), 10)

    newer = _activity(title="Made-up stream 2", base=9)
    assert _put(s, newer, exported + timedelta(hours=1), now) == "written"
    assert s.stream_activity("fs-01", now=now) == newer

    for at in (exported + timedelta(hours=1), exported):  # equal, then older: the payload stays
        assert _put(s, _activity(title="Made-up stream 3", base=1), at, now) == "kept"
        assert s.stream_activity("fs-01", now=now) == newer
    assert s.conn.execute("SELECT COUNT(*) FROM stream_activity").fetchone()[0] == 1
    assert not _marked(s)


def test_activity_reimport_never_extends_expiry(tmp_path):
    s = Store(tmp_path / "cb.db")
    now = S0 + timedelta(days=2)
    deadline = iso(S0 + timedelta(days=30))
    assert _put(s, _activity(), S0 + timedelta(days=1), now) == "written"
    assert _expires_at(s) == deadline

    later_start = _activity(started_at=S0 + timedelta(hours=1))  # a later source clock
    assert _put(s, later_start, S0 + timedelta(days=1, hours=1), now) == "written"  # newer
    assert _expires_at(s) == deadline
    assert _put(s, later_start, S0 + timedelta(days=1, hours=1), now) == "kept"  # equal
    assert _expires_at(s) == deadline
    assert _put(s, later_start, S0 + timedelta(hours=12), now) == "kept"  # older
    assert _expires_at(s) == deadline

    earlier_start = _activity(started_at=S0 - timedelta(days=1))  # an earlier source clock pulls it in
    assert _put(s, earlier_start, S0 + timedelta(hours=12), now) == "kept"
    assert _expires_at(s) == iso(S0 + timedelta(days=29))
    assert s.stream_activity("fs-01", now=now).started_at == S0 + timedelta(hours=1)  # the payload stayed

    # An older version already expired by its own clock lowers and expires the stored row.
    stale = _activity(started_at=S0 - timedelta(days=29))  # its own deadline: S0 + 1 day
    assert _put(s, stale, S0 + timedelta(hours=6), now) == "expired"
    assert _expires_at(s) is None
    assert _marked(s)
    assert s.activity_state("fs-01", now=now) == "expired"

    # Nothing comes back after deletion, not even a newer export with a later clock.
    assert _put(s, _activity(), S0 + timedelta(days=3), now) == "expired"
    assert s.stream_activity("fs-01", now=now) is None
    assert s.conn.execute("SELECT COUNT(*) FROM stream_activity").fetchone()[0] == 0

    # A newer version arriving after the stored deadline passed is "expired", never "written".
    assert _put(s, _activity("fs-02"), S0 + timedelta(days=1), now) == "written"
    late = S0 + timedelta(days=30)
    assert _put(s, _activity("fs-02", base=20), S0 + timedelta(days=29), late) == "expired"
    assert _expires_at(s, "fs-02") is None
    assert _marked(s, "fs-02")


def test_activity_expiry_boundary_at_put(tmp_path):
    s = Store(tmp_path / "cb.db")
    boundary = S0 + timedelta(days=30)
    assert _put(s, _activity("fs-01"), S0 + timedelta(days=1), boundary - timedelta(microseconds=1)) == "written"
    assert _put(s, _activity("fs-02"), S0 + timedelta(days=1), boundary) == "expired"  # now == expires_at
    assert _put(s, _activity("fs-03"), S0 + timedelta(days=1), boundary + timedelta(seconds=1)) == "expired"
    marked = [r[0] for r in s.conn.execute("SELECT stream_id FROM stream_activity_expired ORDER BY 1")]
    assert marked == ["fs-02", "fs-03"]


def test_marker_holds_only_stream_id(tmp_path):
    s = Store(tmp_path / "cb.db")
    assert [r[1] for r in s.conn.execute("PRAGMA table_info(stream_activity_expired)")] == ["stream_id"]
    assert {r[1] for r in s.conn.execute("PRAGMA table_info(imported_streams)")} == {"path", "stream_id", "exported_at"}
    assert {r[1] for r in s.conn.execute("PRAGMA table_info(stream_activity)")} == {
        "stream_id", "channel", "start_at", "end_at", "title", "bin_s", "counts_json", "exported_at",
        "imported_at", "expires_at",
    }  # fmt: skip
    # The schema text is in the database file, and P0's ask tests check those bytes for a refused
    # question's words, "started" and "chatter" among them.
    schema = " ".join(r[0] for r in s.conn.execute("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL"))
    assert "started" not in schema and "chatter" not in schema
    with pytest.raises(sqlite3.IntegrityError), s.conn:
        s.conn.execute(
            "INSERT INTO stream_activity VALUES ('fs-09', 'fakechan01', 'a', 'b', NULL, 60, '[]', 'c', 'd', 'e')"
        )


def test_schema_has_no_text_or_speaker(tmp_path):
    # F1 (glance call 2): each Moment's top emote is a count, with no text, speaker or per-user ID.
    s = Store(tmp_path / "cb.db")
    assert [r[1] for r in s.conn.execute("PRAGMA table_info(moment_emotes)")] == [
        "window_id", "stream_id", "emote_id", "name", "lines", "of_lines", "people", "expires_at",
    ]  # fmt: skip
    assert [r[1] for r in s.conn.execute("PRAGMA index_list(moment_emotes)") if not r[1].startswith("sqlite_")] == [
        "moment_emotes_expires"
    ]
    schema = " ".join(r[0] for r in s.conn.execute("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL"))
    for word in ("started", "chatter", "copypasta"):
        assert word not in schema
    table = s.conn.execute("SELECT sql FROM sqlite_master WHERE name = 'moment_emotes'").fetchone()[0]
    for word in ("speaker", "sender", "login", "display", "user", "message"):  # (column types say TEXT)
        assert word not in table.lower()
    s.add_window(_window(), "vod", NOW)
    ins = "INSERT INTO moment_emotes VALUES ('w-0001', 'fs-01', 'fake-kekw', 'KEKW', ?, ?, ?, 'x')"
    with s.conn:
        s.conn.execute(ins, (2, 3, 2))
        s.conn.execute("DELETE FROM moment_emotes")
    for bad in ((0, 3, 1), (4, 3, 1), (2, 3, 3), (2, 3, 0)):  # lines >= 1, of_lines >= lines, 1 <= people <= lines
        with pytest.raises(sqlite3.IntegrityError), s.conn:
            s.conn.execute(ins, bad)
    from chatbrain.ask.sql import APPROVED

    assert "moment_emotes" not in APPROVED


def test_activity_reads_exclude_expired_before_sweep(tmp_path):
    s = Store(tmp_path / "cb.db")
    put_at = S0 + timedelta(days=3)
    a = _activity("fs-01", started_at=S0)
    b = _activity("fs-02", started_at=S0 + timedelta(days=2))
    c = _activity("fs-03", channel="fakechan02", started_at=S0 + timedelta(days=1))
    for x in (a, b, c):
        assert _put(s, x, put_at, put_at) == "written"

    before = S0 + timedelta(days=30) - timedelta(seconds=1)
    assert s.stream_activity("fs-01", now=before) == a
    assert s.channel_activity("fakechan01", now=before) == [b, a]  # newest first by ended_at
    assert s.activity_state("fs-01", now=before) == "active"

    at = S0 + timedelta(days=30)  # fs-01's deadline; no sweep has run
    assert s.stream_activity("fs-01", now=at) is None
    assert s.channel_activity("fakechan01", now=at) == [b]
    assert s.channel_activity("fakechan02", now=at) == [c]
    assert s.activity_state("fs-01", now=at) == "expired"
    assert s.activity_state("fs-02", now=at) == "active"
    assert s.activity_state("fs-77", now=at) == "missing"
    assert s.conn.execute("SELECT COUNT(*) FROM stream_activity").fetchone()[0] == 3  # only filtered, not deleted

    # A marker reads as expired whatever the clock says.
    with s.transaction() as conn:
        conn.execute("INSERT INTO stream_activity_expired (stream_id) VALUES ('fs-02')")
    assert s.stream_activity("fs-02", now=before) is None
    assert s.channel_activity("fakechan01", now=before) == [a]
    assert s.activity_state("fs-02", now=before) == "expired"


@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.stream_activity("fs-01", now=NOW),
        lambda s: s.channel_activity("fakechan01", now=NOW),
        lambda s: s.activity_state("fs-01", now=NOW),
    ],
    ids=["stream_activity", "channel_activity", "activity_state"],
)
def test_activity_reads_wait_for_the_lock(tmp_path, call):
    import threading

    s = Store(tmp_path / "cb.db")
    done = threading.Event()

    def run() -> None:
        call(s)
        done.set()

    with s.lock:
        t = threading.Thread(target=run)
        t.start()
        assert not done.wait(0.3), "the call ran while another thread held the store's lock"
    assert done.wait(10)
    t.join(10)
