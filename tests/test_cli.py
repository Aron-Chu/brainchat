import hashlib
import json
import tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from chatbrain import cli, fake, ingest, paths
from chatbrain.export_format import WINDOW_LIST_FORMAT, iso, parse_dt, parse_export
from chatbrain.store import Store

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _no_real_data_folder(monkeypatch):
    # A test that forgot --data-dir must fail, never fall back to the real data folder.
    monkeypatch.delenv(paths.ENV_VAR, raising=False)


def _run(capsys, data: Path, *args: str) -> tuple[int, str]:
    code = cli.main(["--data-dir", str(data), *args])
    out = capsys.readouterr()
    return code, out.out + out.err


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def test_entry_point_is_declared():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["scripts"]["chatbrain"] == "chatbrain.cli:main"


def test_import_window_list_deletions_expire(tmp_path, capsys):
    data = tmp_path / "data"
    now = _now()
    raw = fake.make_export(3, 4, 2, now, channels=2)
    export = parse_export(raw)
    window_ids = [w.window_id for w in export.windows]
    (data / "inbox").mkdir(parents=True)
    (data / "inbox" / "export.json").write_bytes(raw)

    code, out = _run(capsys, data, "import")
    assert code == 0, out
    assert "export.json: 6 added, 0 duplicate" in out
    assert not (data / "inbox" / "export.json").exists()
    printed = out

    code, out = _run(capsys, data, "window-list")
    assert code == 0, out
    assert "wrote 6 window(s)" in out
    doc = json.loads((data / "window-list.json").read_bytes())
    assert doc["format"] == WINDOW_LIST_FORMAT
    assert sorted(w["window_id"] for w in doc["windows"]) == sorted(window_ids)
    printed += out

    dropped = export.windows[0].messages[0].message_id
    (data / "inbox" / "check.json").write_bytes(fake.make_deletion_check(raw, {dropped}, now))
    code, out = _run(capsys, data, "deletions")
    assert code == 0, out
    assert "applied 1 check(s), dropped 1 message(s) in 1 window(s)" in out
    assert "deletion check: fresh" in out
    assert not (data / "inbox" / "check.json").exists()
    printed += out

    code, out = _run(capsys, data, "expire", "--now", iso(now + timedelta(days=30)))
    assert code == 0, out
    assert "6 window(s) expired" in out
    printed += out

    # The terminal only ever sees counts and status: no window IDs, no text.
    for w in export.windows:
        assert w.window_id not in printed
        assert fake.marker(w.window_id) not in printed
    with Store(data / "cb.db") as store:
        assert store.window_ids(text_state="present") == []
        assert store.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0


def test_every_command_runs_expiry_first(tmp_path, capsys):
    data = tmp_path / "data"
    paths.data_dir({paths.ENV_VAR: str(data)})
    now = _now()
    raw = fake.make_export(5, 1, 0, now - timedelta(days=13), channels=1)
    w = parse_export(raw).windows[0]
    with Store(data / "cb.db") as store:
        assert store.add_window(w, "vod", now - timedelta(days=13))
    assert w.text_expires_at <= now  # the window's text is already due to go

    code, out = _run(capsys, data, "window-list")
    assert code == 0, out
    assert "1 window(s) expired" in out
    assert "wrote 0 window(s)" in out
    with Store(data / "cb.db") as store:
        assert store.window(w.window_id).text_state == "expired"
        assert store.messages(w.window_id) == []


def test_import_reports_a_refused_file_and_keeps_going(tmp_path, capsys):
    data = tmp_path / "data"
    inbox = data / "inbox"
    inbox.mkdir(parents=True)
    now = _now()
    good = fake.make_export(1, 1, 0, now, channels=1)
    bad = json.loads(fake.make_export(2, 1, 0, now, channels=1))
    bad["windows"][0]["messages"][0]["sender_hash"] = "abc"
    (inbox / "a-bad.json").write_text(json.dumps(bad), encoding="utf-8")
    (inbox / "b-good.json").write_bytes(good)

    code, out = _run(capsys, data, "import")
    assert code == 1
    assert "a-bad.json: refused" in out
    assert "b-good.json: 1 added" in out
    assert (inbox / "a-bad.json").exists()
    assert not (inbox / "b-good.json").exists()


def test_import_named_files(tmp_path, capsys):
    data = tmp_path / "data"
    src = tmp_path / "elsewhere" / "export.json"
    src.parent.mkdir()
    src.write_bytes(fake.make_export(1, 2, 0, _now(), channels=1))
    code, out = _run(capsys, data, "import", str(src))
    assert code == 0, out
    assert "export.json: 2 added" in out
    assert not src.exists()


def test_deletions_with_nothing_waiting_reports_the_gate(tmp_path, capsys):
    code, out = _run(capsys, tmp_path / "data", "deletions")
    assert code == 0, out
    assert "no deletion checks waiting" in out
    assert "deletion check: stale" in out


def test_expire_prints_pending_label_studio_deletions(tmp_path, capsys):
    data = tmp_path / "data"
    paths.data_dir({paths.ENV_VAR: str(data)})
    now = _now()
    raw = fake.make_export(4, 1, 0, now, channels=1)
    (data / "inbox" / "e.json").write_bytes(raw)
    wid = parse_export(raw).windows[0].window_id
    with Store(data / "cb.db") as store:
        ingest.import_export(store, data / "inbox" / "e.json", data / "imported", now)
        with store.lock, store.conn:
            store.conn.executemany("INSERT INTO ls_tasks (window_id, project_kind, task_id) VALUES (?, 'blind', ?)",
                                   [(wid, 1), (wid, 2)])
    code, out = _run(capsys, data, "expire", "--now", iso(now + timedelta(days=30)))
    assert code == 1, out  # text that should be gone is still in Label Studio: the scheduled run fails
    assert "Label Studio: 2 task deletion(s) pending" in out


def test_expire_refuses_a_naive_now(tmp_path, capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["--data-dir", str(tmp_path / "data"), "expire", "--now", "2026-11-06T12:00:00"])
    assert stopped.value.code == 2
    assert "explicit offset" in capsys.readouterr().err


def test_refuses_an_unsafe_data_dir(tmp_path, capsys):
    target = REPO_ROOT / "cli-test-data"
    code, out = _run(capsys, target, "expire")
    assert code == 1
    assert "inside the chat-brain repo" in out
    assert not target.exists()


def test_needs_a_data_dir(capsys):
    assert cli.main(["expire"]) == 1
    assert paths.ENV_VAR in capsys.readouterr().err


def test_needs_a_subcommand(tmp_path):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["--data-dir", str(tmp_path / "data")])
    assert stopped.value.code == 2


# --- assign, repeats, pick-rare (Task 5) -----------------------------------------------------------


def _import_fake(data: Path, n_moment: int, n_ordinary: int, now: datetime) -> list[str]:
    paths.data_dir({paths.ENV_VAR: str(data)})
    raw = fake.make_export(7, n_moment, n_ordinary, now, channels=2)
    (data / "inbox" / "e.json").write_bytes(raw)
    with Store(data / "cb.db") as store:
        ingest.import_export(store, data / "inbox" / "e.json", data / "imported", now)
    return [w.window_id for w in parse_export(raw).windows]


def test_assign_prints_counts_only(tmp_path, capsys):
    data = tmp_path / "data"
    ids = _import_fake(data, 16, 4, _now())

    code, out = _run(capsys, data, "assign", "--all-blind", "--seed", "pilot")
    assert code == 0, out
    assert "assigned 20 window(s) (all blind); assignments now: 20 blind, 0 suggested" in out
    printed = out

    code, out = _run(capsys, data, "assign", "--seed", "seed-a")
    assert code == 0, out
    assert "assigned 0 window(s); assignments now: 20 blind, 0 suggested" in out
    printed += out
    for wid in ids:
        assert wid not in printed
    with Store(data / "cb.db") as store:
        rows = store.conn.execute("SELECT exposure, seed FROM assignments").fetchall()
    assert rows == [("blind", "pilot")] * 20


def test_assign_and_repeats_need_a_seed(tmp_path, capsys):
    for command in (["assign"], ["assign", "--seed", "  "], ["repeats"]):
        with pytest.raises(SystemExit) as stopped:
            cli.main(["--data-dir", str(tmp_path / "data"), *command])
        assert stopped.value.code == 2
    assert not (tmp_path / "data").exists()  # refused before the data folder was made


def test_repeats_schedules_from_aron_labels(tmp_path, capsys):
    data = tmp_path / "data"
    now = _now()
    # 200 windows: some picked windows are due before their text expires and some after, so both the
    # scheduled and the skipped path run. (30 windows picked 3, and all 3 expired first: expected was 0.)
    ids = _import_fake(data, 200, 0, now)
    with Store(data / "cb.db") as store, store.transaction() as conn:
        conn.executemany(
            "INSERT INTO labels (window_id, project, origin, provenance, labels_version, guide_version,"
            " answers_json, created_at) VALUES (?, 'blind', 'aron', 'blind', 'v', 'g', '{}', ?)",
            [(wid, iso(now - timedelta(minutes=5))) for wid in ids],
        )
        text_expires = dict(store.conn.execute("SELECT window_id, text_expires_at FROM windows").fetchall())
    picked = expected = 0
    for wid in ids:
        h = int(hashlib.sha256(f"seed-r:repeat:{wid}".encode()).hexdigest(), 16)
        due = now - timedelta(minutes=5) + timedelta(days=3 + h % 5)
        picked += h % 10000 < 1000
        expected += h % 10000 < 1000 and iso(due) < text_expires[wid]
    # Never vacuous: a `repeats` that schedules nothing, or ignores the expiry skip, must fail.
    assert 1 <= expected < picked

    code, out = _run(capsys, data, "repeats", "--seed", "seed-r")
    assert code == 0, out
    assert f"scheduled {expected} repeat(s); repeats now: {expected}" in out
    code, out = _run(capsys, data, "repeats", "--seed", "seed-r")
    assert f"scheduled 0 repeat(s); repeats now: {expected}" in out
    for wid in ids:
        assert wid not in out


def test_pick_rare_marks_and_refuses_unknown_ids(tmp_path, capsys):
    data = tmp_path / "data"
    ids = _import_fake(data, 4, 0, _now())

    code, out = _run(capsys, data, "pick-rare", ids[0], ids[1], ids[1])
    assert code == 0, out
    assert "marked 2 window(s) picked for rare values (0 already marked); picked now: 2" in out
    printed = out

    code, out = _run(capsys, data, "pick-rare", ids[1], ids[2], "no-such-window")
    assert code == 1
    assert "pick-rare: refused: 1 of 3 window ID(s) are not in the store; nothing was marked" in out
    printed += out

    code, out = _run(capsys, data, "pick-rare", ids[1], ids[2])
    assert code == 0, out
    assert "marked 1 window(s) picked for rare values (1 already marked); picked now: 3" in out
    printed += out

    for wid in ids:
        assert wid not in printed
    assert "no-such-window" not in printed
    with Store(data / "cb.db") as store:
        assert store.window_ids(picked_rare=True) == sorted(ids[:3])


def test_pick_rare_needs_an_id(tmp_path):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["--data-dir", str(tmp_path / "data"), "pick-rare"])
    assert stopped.value.code == 2


def test_a_data_dir_run_creates_nothing_in_the_real_data_folder(tmp_path, capsys, monkeypatch):
    real = tmp_path / "real-canary"  # stands in for the folder CHATBRAIN_DATA names
    monkeypatch.setenv(paths.ENV_VAR, str(real))
    code, out = _run(capsys, tmp_path / "data", "expire")
    assert code == 0, out
    assert not real.exists()  # --data-dir runs never touch CHATBRAIN_DATA's folder


def test_expire_flags_inbox_exports_that_still_hold_text(tmp_path, capsys):
    data = tmp_path / "data"
    inbox = data / "inbox"
    inbox.mkdir(parents=True)
    now = _now()
    bad = json.loads(fake.make_export(2, 1, 0, now, channels=1))
    bad["windows"][0]["messages"][0]["login"] = "zebra-umbrella-41"  # an identity field: refused
    (inbox / "a-bad.json").write_text(json.dumps(bad), encoding="utf-8")
    never = fake.make_export(3, 1, 0, now, channels=1)  # a valid export nobody imported
    (inbox / "b-never.json").write_bytes(never)
    (inbox / "c-garbled.json").write_text("{not json", encoding="utf-8")  # unknown: dated by its last change
    bad_due = min(parse_dt(w["text_expires_at"]) for w in bad["windows"])
    due = max(bad_due, min(w.text_expires_at for w in parse_export(never).windows))  # both files past by then

    code, out = _run(capsys, data, "import", str(inbox / "a-bad.json"))
    assert code == 1 and "a-bad.json: refused, left in place with its text" in out

    code, out = _run(capsys, data, "expire", "--now", iso(now))
    assert code == 0, out  # nothing past its expiry yet
    assert "inbox: 3 file(s) not imported may still hold chat text, 0 past its text expiry" in out

    code, out = _run(capsys, data, "deletions")
    assert "inbox: 3 file(s) not imported may still hold chat text" in out  # every command says so

    code, out = _run(capsys, data, "expire", "--now", iso(due))
    assert code == 1, out
    assert "inbox: 3 file(s) not imported may still hold chat text, 2 past its text expiry" in out

    code, out = _run(capsys, data, "expire", "--now", iso(now + timedelta(days=30)))
    assert code == 1, out
    assert "3 past its text expiry; import or delete them" in out
    assert sorted(p.name for p in inbox.iterdir()) == ["a-bad.json", "b-never.json", "c-garbled.json"]  # kept (B10)
    assert "zebra-umbrella-41" not in out and "a-bad.json" not in out.split("inbox:")[-1]  # counts only

    for name in ("a-bad.json", "b-never.json", "c-garbled.json"):
        (inbox / name).unlink()
    code, out = _run(capsys, data, "expire", "--now", iso(now + timedelta(days=30)))
    assert code == 0, out
    assert "inbox:" not in out


def test_expire_flags_an_import_source_it_still_cannot_delete(tmp_path, capsys, monkeypatch):
    data = tmp_path / "data"
    paths.data_dir({paths.ENV_VAR: str(data)})
    now = _now()
    raw = fake.make_export(4, 1, 0, now, channels=1)
    src = data / "inbox" / "e.json"
    src.write_bytes(raw)
    due = min(w.text_expires_at for w in parse_export(raw).windows)
    real = Path.unlink

    def locked(self, *args, **kwargs):
        if Path(self) == src:
            raise PermissionError("in use by another program")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", locked)
    code, out = _run(capsys, data, "import")
    assert code == 0 and "could not be deleted yet" in out
    assert "imported: 1 source file(s) the import could not delete yet, 0 past its text expiry" in out
    assert "inbox:" not in out  # an imported source is counted once, as a source

    code, out = _run(capsys, data, "expire", "--now", iso(due))
    assert code == 1, out
    assert "imported: 1 source file(s) the import could not delete yet, 1 past its text expiry" in out

    monkeypatch.setattr(Path, "unlink", real)
    code, out = _run(capsys, data, "expire", "--now", iso(due))
    assert code == 0, out
    assert not src.exists() and "imported:" not in out


@pytest.mark.parametrize(
    "held",
    [{"ls_error": "ConnectError"}, {"ls_tasks_pending": 1}, {"ls_compaction_pending": True}, {"models_pending": 1},
     {"models_missing": 1}, {"inbox_past": 1}, {"sources_left_past": 1}],
)  # fmt: skip
def test_expire_fails_while_text_it_should_delete_is_held(tmp_path, capsys, monkeypatch, held):
    # C27: the daily scheduled `expire` is the safety net, so it must not report success
    # while text that is due to go is still held somewhere.
    monkeypatch.setattr(cli.expiry, "run", lambda *a, **k: cli.expiry.ExpiryResult(0, 0, 0, 0, **held))
    code, out = _run(capsys, tmp_path / "data", "expire")
    assert code == 1, out
    assert "expire: not finished" in out
    monkeypatch.setattr(cli.expiry, "run", lambda *a, **k: cli.expiry.ExpiryResult(0, 0, 0, 0))
    code, out = _run(capsys, tmp_path / "data", "expire")
    assert code == 0 and "not finished" not in out


def test_import_prints_activity_counts_only(tmp_path, capsys):
    data = tmp_path / "data"
    inbox = data / "inbox"
    inbox.mkdir(parents=True)
    now = _now()
    raw = fake.make_export(3, 4, 2, now, channels=1)
    export = parse_export(raw)
    n = len(export.streams)
    assert n > 1
    (inbox / "a.json").write_bytes(raw)
    (inbox / "b.json").write_bytes(fake.make_export(4, 1, 0, now, channels=1))  # one stream
    (inbox / "c.json").write_bytes(fake.make_export(5, 1, 0, now, channels=1, activity=False))  # /1
    (inbox / "d.json").write_bytes(fake.make_export(6, 1, 0, now - timedelta(days=31), channels=1))  # counts expired

    code, out = _run(capsys, data, "import")
    assert code == 0, out
    lines = out.splitlines()
    assert lines[lines.index("a.json: 6 added, 0 duplicate, 0 skipped (language), 0 skipped (expired)") + 1] == (
        f"streams: {n} activity lines stored, 0 skipped (expired)"
    )
    assert "streams: 1 activity line stored, 0 skipped (expired)" in lines
    assert "streams: 0 activity lines stored, 0 skipped (expired)" in lines
    assert "streams: 0 activity lines stored, 1 skipped (expired)" in lines
    # Counts only: no stream ID, title or channel reaches the terminal.
    for s in export.streams:
        assert s.stream_id not in out
        assert s.title not in out
        assert s.channel not in out
