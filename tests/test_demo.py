"""The end-to-end demo on made-up chat (Task 11).

Everything runs offline in ``tmp_path``: ``demo`` writes only under ``<data>/demo``,
Label Studio and Ollama are fakes, and nothing opens a socket or runs Docker (ruling B35).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from chatbrain import cli, fake, ingest, manifest, ollama, paths, spread
from chatbrain import scheme as scheme_mod
from chatbrain.export_format import parse_export
from chatbrain.lstudio import client as lsclient
from chatbrain.lstudio import config, sync
from chatbrain.page import app as app_mod
from chatbrain.store import Store

SCHEME = scheme_mod.load()
NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)
GOOD_REPLY = {
    "doing": {"value": ["laughing", "cheering"], "not_sure": []},
    "aimed_at": {"value": "streamer", "not_sure": []},
    "feeling": {"value": None, "not_sure": ["unclear", "conflicting"]},
    "tone": {"value": None, "not_sure": []},
}


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    # Never the real data folder, its .env, a real Ollama or Ollama's real log folder.
    monkeypatch.delenv(paths.ENV_VAR, raising=False)

    def no_ollama():
        raise AssertionError("a test reached for the real Ollama client")

    monkeypatch.setattr(cli, "_ollama_client", no_ollama)
    monkeypatch.setattr(cli, "_ollama_log_dir", lambda: None, raising=False)


def _demo(capsys, data: Path, *args: str) -> tuple[int, str]:
    code = cli.main(["--data-dir", str(data), "demo", *args])
    out = capsys.readouterr()
    return code, out.out + out.err


def _counts(store: Store) -> dict[str, spread.Counts]:
    rows = store.conn.execute("SELECT window_id, json FROM features").fetchall()
    return {wid: spread.Counts.from_json(j) for wid, j in rows}


def _files_bytes(folder: Path) -> bytes:
    return b"".join(p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file())


def _page(store: Store, demo: Path, tmp_path: Path) -> str:
    review = tmp_path / "review_emotes.json"
    review.write_text(json.dumps({"ids": []}), encoding="utf-8")
    app = app_mod.create_app(
        store, SCHEME, demo / "models", imported_dir=demo / "imported",
        clock=lambda: datetime.now(timezone.utc), review_emotes=review, hourly=False, narrator=None,
    )  # fmt: skip
    return TestClient(app, base_url=f"http://{app_mod.HOST}:{app_mod.PORT}").get("/").text


def test_demo_without_ollama_or_label_studio(tmp_path, capsys):
    data = tmp_path / "data"
    code, out = _demo(capsys, data, "--seed", "7")
    assert code == 0, out
    demo = data / "demo"
    assert (data / cli.DB_NAME).is_file()  # the command's own expiry ran on the data folder's store first

    with Store(demo / cli.DB_NAME) as store:
        conn = store.conn
        # two models, and rung 1 did not retire rung 0 (retiring is per rung and origin)
        rows = conn.execute("SELECT rung, origin, retired_at, file_deleted_at FROM manifests ORDER BY rung").fetchall()
        assert rows == [(0, "synthetic", None, None), (1, "synthetic", None, None)]
        for (model_id,) in conn.execute("SELECT model_id FROM manifests"):
            assert (demo / "models" / model_id / "manifest.json").is_file()
            assert (demo / "models" / model_id / "model.joblib").is_file()
        active = manifest.active_per_question(store, models_dir=demo / "models")
        assert set(active) == {q.key for q in SCHEME.questions}

        expired = [r[0] for r in conn.execute("SELECT window_id FROM windows WHERE text_state = 'expired'")]
        live = [r[0] for r in conn.execute("SELECT window_id FROM windows WHERE text_state = 'present'")]
        assert expired and live  # now + 5 days expires some windows, not all
        for wid in expired:
            assert conn.execute("SELECT COUNT(*) FROM labels WHERE window_id = ?", (wid,)).fetchone()[0] >= 1
            assert conn.execute("SELECT 1 FROM features WHERE window_id = ?", (wid,)).fetchone()
            assert conn.execute("SELECT COUNT(*) FROM messages WHERE window_id = ?", (wid,)).fetchone()[0] == 0
        counts = _counts(store)
        assert conn.execute("SELECT COUNT(*) FROM deletion_checks").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM labels WHERE origin = 'synthetic'").fetchone()[0] == len(counts)
        assert conn.execute("SELECT COUNT(*) FROM suggestions").fetchone()[0] == 0  # no --with-ollama
        assert conn.execute("SELECT COUNT(*) FROM ls_tasks").fetchone()[0] == 0  # no --with-label-studio

        # the newest stream is the showcase: a chant and a repeated-emotes Moment (B3)
        newest = conn.execute(
            "SELECT stream_id FROM windows GROUP BY stream_id ORDER BY MAX(end_at) DESC, stream_id LIMIT 1"
        ).fetchone()[0]
        moments = [
            r[0] for r in conn.execute(
                "SELECT window_id FROM windows WHERE stream_id = ? AND anchor_kind = 'moment'", (newest,)
            )
        ]  # fmt: skip
        spreads = {counts[w].spread for w in moments}
        assert {"chant", "repeated_emotes"} <= spreads
        assert all(w in live for w in moments)

        # the page's glance over the demo: the eyebrow counts the newest stream's Moments, and the
        # busiest (or first) three are cards (Task 3; the five-sentence recap is gone)
        page = _page(store, demo, tmp_path)
    eyebrow = re.search(r'<h1 class="gl-eyebrow" id="gl-eb">(Busiest|First) (\d) of (\d+) Moments</h1>', page)
    assert eyebrow and int(eyebrow.group(3)) == len(moments), eyebrow
    k = min(3, len(moments))
    assert int(eyebrow.group(2)) == k
    shown = re.findall(r'<li class="gl-card" data-window="([^"]+)"', page)
    assert len(shown) == k and set(shown) <= set(moments)

    # C9: no expired window's marker survives anywhere in the demo folder; a live one does in the store
    marked_expired = [w for w in expired if counts[w].n_messages >= 2]
    assert marked_expired  # at least one expired window had a marker line
    blob = _files_bytes(demo)
    for wid in expired:
        assert fake.marker(wid).encode() not in blob
    db = (demo / cli.DB_NAME).read_bytes()
    assert any(fake.marker(w).encode() in db for w in live if counts[w].n_messages >= 2)

    # the summary: counts only, n_graded beside every agreement, the next command
    for wid in expired[:20] + live[:20]:
        assert wid not in out and fake.marker(wid) not in out
    for secret in ("KEKW", "LETS GO", "fakechan"):
        assert secret not in out
    agreement_lines = [line for line in out.splitlines() if "agreement" in line]
    assert agreement_lines and all("graded" in line for line in agreement_lines)
    assert "sentinel search: expired windows' marker lines found in 0 of" in out
    assert "suggest: skipped (no --with-ollama)" in out
    assert "Label Studio: skipped (no --with-label-studio)" in out
    assert f'chatbrain --data-dir "{demo}" page' in out


def test_showcase_is_the_newest_stream_and_reads_back_as_truth():
    main = fake.make_export(7, 480, 120, NOW)
    show = fake.make_showcase(7, NOW)
    assert fake.make_showcase(7, NOW) == show  # a pure function of its arguments
    export = parse_export(show)
    newest_main = max(w.end_at for w in parse_export(main).windows)
    assert min(w.start_at for w in export.windows) > newest_main
    assert max(w.end_at for w in export.windows) <= NOW - timedelta(hours=1)
    assert {w.window_id for w in export.windows}.isdisjoint({w.window_id for w in parse_export(main).windows})

    counts = {w.window_id: spread.compute(w) for w in export.windows}
    moments = [w for w in export.windows if w.anchor_kind == "moment"]
    spreads = {counts[w.window_id].spread for w in moments}
    assert {"chant", "repeated_emotes", "back_and_forth", "separate_messages"} <= spreads
    assert all(counts[w.window_id].distinct_posters >= 5 for w in moments)
    for w in export.windows:
        assert sum(m.text == fake.marker(w.window_id) for m in w.messages) == 1

    truth = fake.truth(show)
    assert set(truth) == {w.window_id for w in export.windows}
    for answers in truth.values():
        scheme_mod.validate_answers(SCHEME, answers, {})
    feelings = {truth[w.window_id]["feeling"] for w in moments}
    assert "critical" in feelings


class DemoLS:
    """A made-up Label Studio holding the demo's projects: no network, no Docker.

    Like the real one it keeps its tasks' data in a file under ``<data_dir>/labelstudio``,
    so the demo's sentinel search has a file to read. ``forgets`` = False keeps deleted
    tasks' data in that file, like a database that was never compacted.
    """

    def __init__(self, data_dir: Path, forgets: bool = True):
        self.data_dir = data_dir
        self.db = data_dir / "labelstudio" / "label_studio.sqlite3"
        self.db.parent.mkdir(parents=True)
        self.forgets = forgets
        self.ids = {"suggested": 71, "blind": 72, "repeat": 73}
        self.configs: dict[str, str | None] = {}
        self.imports: list[tuple[str, dict, int]] = []
        self.deleted: list[int] = []
        self.compacts = 0
        self.closed = False
        self.next_id = 900
        self.order: list[str] = []
        self._save()

    def _save(self):
        kept = [t for _, t, tid in self.imports if not (self.forgets and tid in self.deleted)]
        self.db.write_text(json.dumps(kept), encoding="utf-8")

    def project_id(self, kind):
        return self.ids[kind] if kind in self.configs else None

    def ensure_project(self, kind, label_config=None):
        self.configs[kind] = label_config
        return self.ids[kind]

    def import_tasks(self, project_id, kind, task_list):
        lsclient.check_import(kind, task_list)
        ids = []
        for t in task_list:
            self.next_id += 1
            self.imports.append((kind, t, self.next_id))
            self.order.append("import")
            ids.append(self.next_id)
        self._save()
        return ids

    def export_annotations(self, project_id, *, all_tasks=False):
        kind = {v: k for k, v in self.ids.items()}[project_id]
        self.order.append(f"export:{kind}")
        if not all_tasks:
            return []
        return [{"id": tid, "data": t["data"], "annotations": []} for k, t, tid in self.imports
                if k == kind and tid not in self.deleted]  # fmt: skip

    def use_model_version(self, project_id, model_version):
        pass

    def delete_tasks(self, task_ids):
        self.order.append("delete")
        self.deleted.extend(task_ids)
        self._save()
        return len(task_ids)

    def compact(self):
        self.order.append("compact")
        self.compacts += 1

    def close(self):
        self.closed = True


class FakeOllama:
    """A made-up Ollama behind ``httpx.MockTransport`` that answers every window the same way."""

    def __init__(self):
        self.paths: list[str] = []
        self.chats = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.31.2"})
        if request.url.path == "/api/chat":
            self.chats += 1
            body = json.loads(request.content)
            content = json.dumps(GOOD_REPLY)
            return httpx.Response(
                200, json={"model": body["model"], "message": {"role": "assistant", "content": content}, "done": True}
            )
        return httpx.Response(404)

    def client(self) -> httpx.Client:
        return ollama.make_client(transport=httpx.MockTransport(self.handler))


def test_demo_with_fake_ollama_and_label_studio_starts_from_a_clean_folder(tmp_path, capsys, monkeypatch):
    data = tmp_path / "data"
    code, out = _demo(capsys, data, "--seed", "3")
    assert code == 0, out
    demo = data / "demo"

    # a demo task the store still tracks blocks a fresh run that has no Label Studio to delete it
    with Store(demo / cli.DB_NAME) as store, store.transaction() as conn:
        wid = conn.execute("SELECT window_id FROM windows WHERE text_state = 'present' LIMIT 1").fetchone()[0]
        conn.execute("INSERT INTO ls_tasks (window_id, project_kind, task_id) VALUES (?, 'blind', 4242)", (wid,))
    code, out = _demo(capsys, data, "--seed", "7")
    assert code == 1
    assert "1 Label Studio task(s) from the last demo" in out and "--with-label-studio" in out
    with Store(demo / cli.DB_NAME) as store:
        assert store.conn.execute("SELECT COUNT(*) FROM ls_tasks").fetchone()[0] == 1  # left as it was

    ls = DemoLS(tmp_path / "ls-data")
    fo = FakeOllama()
    logs = tmp_path / "ollama-logs"
    logs.mkdir()
    (logs / "server.log").write_text("made-up log line\n", encoding="utf-8")
    monkeypatch.setattr(cli, "_demo_label_studio", lambda d: ls)
    monkeypatch.setattr(cli, "_ollama_client", fo.client)
    monkeypatch.setattr(cli, "_ollama_log_dir", lambda: logs)
    code, out = _demo(capsys, data, "--seed", "7", "--with-ollama", "--with-label-studio")
    assert code == 0, out
    assert 4242 in ls.deleted and ls.order.index("delete") < ls.order.index("compact")  # the leftover went first

    with Store(demo / cli.DB_NAME) as store:
        conn = store.conn
        ids = [r[0] for r in conn.execute("SELECT window_id FROM windows")]
        assert ids and all(i.startswith("fake7-") for i in ids)  # the seed-3 demo is gone
        assert conn.execute("SELECT COUNT(*) FROM manifests WHERE retired_at IS NULL").fetchone()[0] == 2
        # the teacher went through teacher.run: the version check first, then 3 windows
        assert fo.paths[0] == "/api/version" and fo.chats == 3
        assert conn.execute("SELECT COUNT(*) FROM suggestions").fetchone()[0] == 3
        # every task the demo made is deleted at the end, after a pull, then Label Studio compacts
        assert conn.execute("SELECT COUNT(*) FROM ls_tasks").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM ls_deletions").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM ls_compaction").fetchone()[0] == 0
    kinds = [k for k, _, _ in ls.imports]
    assert (kinds.count("suggested"), kinds.count("blind"), kinds.count("repeat")) == (3, 10, 10)
    assert all("predictions" not in t for k, t, _ in ls.imports if k == "repeat")
    assert all("predictions" in t for k, t, _ in ls.imports if k == "suggested")
    assert {tid for _, _, tid in ls.imports} <= set(ls.deleted)
    last_import = max(i for i, o in enumerate(ls.order) if o == "import")
    first_delete = ls.order.index("delete", last_import)
    assert any(o.startswith("export:") for o in ls.order[last_import:first_delete])  # pulled before deleting (B11)
    assert ls.order[-1] == "compact" and ls.closed
    assert set(ls.configs) == {"suggested", "blind", "repeat"}

    assert "pushed 10 task(s) to repeat" in out
    assert "no predictions and no earlier labels" in out
    assert "sentinel search: expired windows' marker lines found in 0 of" in out
    assert "sentinel search: deleted Label Studio tasks' marker lines found in 0 of 1 file(s)" in out
    assert "sentinel search: marker lines of windows sent to Ollama found in 0 of 1 log file(s)" in out
    assert "3 suggested, 10 blind, 10 repeat task(s) pushed, all deleted again and compacted" in out
    assert "KEKW" not in out and "fakechan" not in out


def test_demo_flags_markers_left_in_ollama_logs_or_label_studio(tmp_path, capsys, monkeypatch):
    fo = FakeOllama()
    logs = tmp_path / "ollama-logs"
    logs.mkdir()
    sent: list[str] = []

    def client():
        return ollama.make_client(transport=httpx.MockTransport(spy))

    def spy(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            body = request.content.decode("utf-8")
            sent.extend(m for m in _markers_in(body))
            (logs / "server.log").write_text(body, encoding="utf-8")  # a debug log would hold the prompt
        return fo.handler(request)

    ls = DemoLS(tmp_path / "ls-data", forgets=False)  # deleted tasks' text stays in its file
    monkeypatch.setattr(cli, "_demo_label_studio", lambda d: ls)
    monkeypatch.setattr(cli, "_ollama_client", client)
    monkeypatch.setattr(cli, "_ollama_log_dir", lambda: logs)
    code, out = _demo(capsys, tmp_path / "data", "--seed", "5", "--with-ollama", "--with-label-studio")
    assert code == 1, out
    assert sent and "sentinel search: marker lines of windows sent to Ollama found in 1 of 1 log file(s)" in out
    assert "sentinel search: deleted Label Studio tasks' marker lines found in 1 of 1 file(s)" in out
    assert "a marker line that should be gone was found" in out
    assert not any(m in out for m in sent)
    assert not any(m in out for m in _markers_in(ls.db.read_text(encoding="utf-8")))


def test_demo_reset_waits_for_a_pending_compaction(tmp_path, capsys):
    """The last demo's store is the only record of a delete Label Studio hasn't compacted yet."""
    demo = paths.data_dir({paths.ENV_VAR: str(tmp_path / "data" / "demo")})
    with Store(demo / cli.DB_NAME) as store, store.transaction() as conn:
        conn.execute("INSERT INTO ls_compaction (id, due_at) VALUES (1, '2026-11-06T12:00:00.000000Z')")
    assert cli._demo_reset(demo, NOW, None) is False
    assert "not compacted yet" in capsys.readouterr().out
    assert (demo / cli.DB_NAME).is_file()  # kept

    ls = DemoLS(tmp_path / "ls-data")
    assert cli._demo_reset(demo, NOW, ls) is True
    assert ls.compacts == 1 and ls.deleted == []
    assert not (demo / cli.DB_NAME).exists()


def _markers_in(text: str) -> list[str]:
    import re

    return re.findall(r"zq[0-9a-f]{12}", text)


def test_demo_label_studio_needs_the_real_data_folder(tmp_path, capsys):
    data = tmp_path / "data"
    code, out = _demo(capsys, data, "--seed", "7", "--with-label-studio")
    assert code == 1
    assert "Label Studio is not configured" in out
    assert not (data / "demo" / cli.DB_NAME).exists()  # refused before anything was made


def test_demo_projects_are_apart_from_the_real_ones(tmp_path):
    """The real client never finds a ``BrainChat demo`` project, so ``adopt_untracked`` ignores them (B20)."""
    store = Store(tmp_path / "cb.db")
    raw = fake.make_export(1, 2, 0, NOW, channels=1)
    (tmp_path / "e.json").write_bytes(raw)
    ingest.import_export(store, tmp_path / "e.json", tmp_path / "imported", NOW)
    known = sorted(fake.truth(raw))
    projects = [
        {"id": 71 + i, "title": lsclient.DEMO_TITLES[k]} for i, k in enumerate(("suggested", "blind", "repeat"))
    ]
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.method == "GET" and request.url.path == "/api/projects/":
            title = request.url.params["title"].lower()
            found = [p for p in projects if title in p["title"].lower()]  # Label Studio filters on substrings
            return httpx.Response(200, json={"count": len(found), "next": None, "results": found})
        if request.method == "POST" and request.url.path == "/api/projects/":
            body = json.loads(request.content)
            projects.append({"id": 90 + len(projects), **body})
            return httpx.Response(201, json=projects[-1])
        if request.url.path.endswith("/export"):
            tasks = [{"id": 5000 + i, "data": {"window_id": w}, "annotations": []} for i, w in enumerate(known)]
            return httpx.Response(200, json=tasks)
        return httpx.Response(404)

    def make(titles=None):
        return lsclient.LabelStudio(
            token="fake-token-123", auth="legacy", data_dir=tmp_path / "data", compose_file=tmp_path / "compose.yaml",
            transport=httpx.MockTransport(handler), titles=titles,
        )  # fmt: skip

    real, demo = make(), make(lsclient.DEMO_TITLES)
    assert all(t.startswith("BrainChat demo · ") for t in lsclient.DEMO_TITLES.values())
    assert set(lsclient.DEMO_TITLES) == set(lsclient.TITLES)
    assert sync.adopt_untracked(store, real) == sync.Adopted(0, 0)
    assert not any(path.endswith("/export") for _, path in calls)  # no demo project was even read
    assert store.conn.execute("SELECT COUNT(*) FROM ls_tasks").fetchone()[0] == 0
    assert real.project_id("blind") is None and demo.project_id("blind") == 72

    assert real.ensure_project("blind", config.label_config(SCHEME)) not in (71, 72, 73)
    assert projects[-1]["title"] == "Chat Brain · Blind"
    with pytest.raises(ValueError):
        make({"blind": "x"})  # every kind needs its own title
    real.close()
    demo.close()
    store.close()
