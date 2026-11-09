"""Label Studio: the label config, task JSON, the loopback client, sync, push and the CLI.

Everything here is made up and offline: the client talks to ``httpx.MockTransport``
and ``docker compose`` is a fake runner; sync and push use ``FakeLS``.
"""

import json
import random
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from chatbrain import cli, expiry, ingest, paths, scheme, spread
from chatbrain.export_format import EXPORT_FORMAT, Emote, Export, Message, Window, dump_export, iso
from chatbrain.lstudio import client as lsclient
from chatbrain.lstudio import config, sync, tasks
from chatbrain.store import Store

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)
START = NOW - timedelta(days=1)
EXPIRES = NOW + timedelta(days=5)
SCHEME = scheme.load()
QUESTION_KEYS = [q.key for q in SCHEME.questions]
KEKW = Emote("fake-kekw", "KEKW")
TOKEN = "fake-token-123"
TEACHER = ("qwen3:8b", "cb_teacher_2026_11_06_v1")


@pytest.fixture(autouse=True)
def _no_real_data_folder(monkeypatch):
    # A test must never reach the real data folder or its .env.
    monkeypatch.delenv(paths.ENV_VAR, raising=False)


# --- made-up windows and store ----------------------------------------------------------------------


def _win(window_id, texts=("KEKW KEKW", "LETS GO"), *, expires=EXPIRES, start=START, speakers=None) -> Window:
    speakers = speakers or [f"s{i + 1}" for i in range(len(texts))]
    msgs = tuple(
        Message(f"{window_id}-m{i + 1}", 31_000 + i * 1000, speakers[i], text,
                tuple(KEKW for t in text.split() if t == "KEKW"))
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
        language="en",
        text_expires_at=expires,
        messages=msgs,
    )


def _import(store, folder: Path, *windows, now=NOW, exported_at=NOW) -> None:
    src = folder / "inbox" / "e.json"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(dump_export(Export(EXPORT_FORMAT, exported_at, "vod", tuple(windows))))
    ingest.import_export(store, src, folder / "imported", now)


def _store(tmp_path, *windows) -> Store:
    tmp_path.mkdir(parents=True, exist_ok=True)
    store = Store(tmp_path / "cb.db")
    _import(store, tmp_path, *windows)
    return store


def _assign(store, exposure, *window_ids, now=NOW):
    with store.transaction() as conn:
        conn.executemany(
            "INSERT INTO assignments (window_id, exposure, assigned_at, seed) VALUES (?, ?, ?, 'fake-seed')",
            [(w, exposure, iso(now)) for w in window_ids],
        )


def _suggest(store, window_id, answers=None, not_sure=None, *, error=None):
    body = None if error else json.dumps({"answers": answers, "not_sure": not_sure or {}})
    with store.transaction() as conn:
        conn.execute(
            "INSERT INTO suggestions (window_id, teacher_model, prompt_version, labels_version, answers_json, error,"
            " created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (window_id, *TEACHER, SCHEME.version, body, error, iso(NOW)),
        )


def _label(store, window_id, project, *, origin="aron"):
    with store.transaction() as conn:
        conn.execute(
            "INSERT INTO labels (window_id, project, origin, provenance, labels_version, guide_version, answers_json,"
            " created_at) VALUES (?, ?, ?, ?, ?, ?, '{\"feeling\": \"neutral\"}', ?)",
            (window_id, project, origin, "blind" if project == "blind" else "repeat", SCHEME.version,
             SCHEME.guide_version, iso(NOW)),
        )


def _repeat(store, window_id, due_at):
    with store.transaction() as conn:
        conn.execute("INSERT INTO repeats (window_id, due_at) VALUES (?, ?)", (window_id, iso(due_at)))


def _ls_tasks(store) -> list[tuple]:
    with store.lock:
        return store.conn.execute(
            "SELECT window_id, project_kind, task_id FROM ls_tasks ORDER BY task_id"
        ).fetchall()


def _labels(store) -> dict[str, dict]:
    with store.lock:
        cur = store.conn.execute("SELECT * FROM labels ORDER BY id")
        names = [d[0] for d in cur.description]
        return {row[names.index("window_id")]: dict(zip(names, row)) for row in cur.fetchall()}


def _annotation(task_id, window_id, ann_id, result, *, cancelled=False, created="2026-11-06T15:00:00Z",
                lead_time=3.5, predictions=()):
    return {
        "id": task_id,
        "data": {"window_id": window_id, "messages": [], "counts_text": ""},
        "annotations": [{"id": ann_id, "result": result, "was_cancelled": cancelled, "created_at": created,
                         "lead_time": lead_time}],
        "predictions": list(predictions),
    }


def _choices(from_name, *values):
    return {"from_name": from_name, "to_name": "chat", "type": "choices", "value": {"choices": list(values)}}


FULL_ANSWER = [_choices("doing", "laughing"), _choices("aimed_at", "streamer"), _choices("feeling", "supportive")]

# Made-up Label Studio exports, built here so no export file ships with the tests.
# Blind project: two answered tasks and one cancelled (skipped) task.
BLIND_EXPORT = [
    # tone left blank; doing in another order than the scheme's; fractional-seconds time
    _annotation(1001, "fake1-w0001", 2001,
                [_choices("doing", "cheering", "laughing"), _choices("aimed_at", "streamer"),
                 _choices("feeling", "supportive")],
                created="2026-11-06T12:30:00.123456Z", lead_time=42.5),
    # not-sure reasons out of scheme order, a cause note with its source, a whole-number lead time
    _annotation(1004, "fake1-w0004", 2004,
                [_choices("doing", "asking"), _choices("aimed_at_not_sure", "cause_missing"),
                 _choices("feeling_not_sure", "conflicting", "unclear"), _choices("tone", "serious"),
                 {"from_name": "cause_note", "to_name": "chat", "type": "textarea",
                  "value": {"text": ["the made-up boss fight ended"]}},
                 _choices("cause_source", "clip")],
                created="2026-11-06T13:00:00Z", lead_time=12),
    _annotation(1005, "fake1-w0005", 2005, FULL_ANSWER, cancelled=True),
]

# Blind project: four broken annotations, none of which may become a label.
INVALID_EXPORT = [
    _annotation(1002, "fake1-w0002", 2002,
                [*FULL_ANSWER, _choices("doing_not_sure", "unclear")]),  # doing: a value and a reason
    _annotation(1003, "fake1-w0003", 2003,
                [_choices("doing", "laughing"), _choices("feeling", "supportive")]),  # aimed_at unanswered
    _annotation(1006, "fake1-w0006", 2006,
                [*FULL_ANSWER, _choices("tone", "playful"), _choices("tone_not_sure", "unclear")]),  # tone: both
    _annotation(1007, "fake1-w0007", 2007,
                [_choices("doing", "not-a-value"), _choices("aimed_at", "streamer"),
                 _choices("feeling", "supportive")]),  # doing: a value outside the scheme
]

# Suggested project: w0011 matches its stored suggestion; w0012 matches only the Label Studio
# prediction (the stored suggestion differs); w0013's suggestion is an error.
_W0012_RESULT = [_choices("doing", "laughing"), _choices("aimed_at", "streamer"), _choices("feeling", "critical"),
                 _choices("tone", "sarcastic")]
SUGGESTED_EXPORT = [
    _annotation(1011, "fake1-w0011", 2011,
                [_choices("doing", "laughing"), _choices("aimed_at", "streamer"),
                 _choices("feeling_not_sure", "unclear"), _choices("tone", "playful")]),
    _annotation(1012, "fake1-w0012", 2012, _W0012_RESULT,
                predictions=[{"model_version": "/".join(TEACHER), "result": _W0012_RESULT}]),
    _annotation(1013, "fake1-w0013", 2013, FULL_ANSWER),
]


class FakeLS:
    """A made-up Label Studio for sync, push and CLI tests: no network, no Docker."""

    def __init__(self, exports=None, fail_import_at=None):
        self.ids = {"suggested": 1, "blind": 2, "repeat": 3}
        self.exports = exports or {}
        self.configs: dict[str, str | None] = {}
        self.imports: list[tuple[str, dict, int]] = []
        self.model_versions: list[tuple[int, str]] = []
        self.deleted: list[int] = []
        self.compacts = 0
        self.closed = False
        self.next_id = 500
        self.fail_import_at = fail_import_at
        self.order: list[str] = []

    def project_id(self, kind):
        return self.ids[kind]

    def ensure_project(self, kind, label_config=None):
        self.configs[kind] = label_config
        return self.ids[kind]

    def import_tasks(self, project_id, kind, task_list):
        lsclient.check_import(kind, task_list)  # the same refusal the real client makes
        if self.fail_import_at is not None and len(self.imports) == self.fail_import_at:
            raise httpx.ConnectError("label studio went away")
        ids = []
        for t in task_list:
            self.next_id += 1
            self.imports.append((kind, t, self.next_id))
            ids.append(self.next_id)
        return ids

    def export_annotations(self, project_id, *, all_tasks=False):
        kind = {v: k for k, v in self.ids.items()}[project_id]
        self.order.append(f"export:{kind}")
        if not all_tasks:
            return self.exports.get(kind, [])
        mine = [{"id": tid, "data": t["data"], "annotations": []} for k, t, tid in self.imports if k == kind]
        return mine + self.exports.get(kind, [])

    def use_model_version(self, project_id, model_version):
        self.model_versions.append((project_id, model_version))

    def delete_tasks(self, task_ids):
        self.order.append("delete")
        self.deleted.extend(task_ids)
        return len(task_ids)

    def compact(self):
        self.compacts += 1

    def close(self):
        self.closed = True


# --- label config -------------------------------------------------------------------------------


def _config_root():
    return ET.fromstring(config.label_config(SCHEME))


def test_label_config_hotkeys_unique():
    root = _config_root()
    keys = [c.get("hotkey") for c in root.iter("Choice") if c.get("hotkey") is not None]
    expected = sum(len(q.values) + (1 if q.blank_value else 0) for q in SCHEME.questions)
    assert expected == 25
    assert len(keys) == expected
    assert len(set(keys)) == expected
    assert keys == list(config.HOTKEYS[:expected])  # assigned in order
    for choices in root.iter("Choices"):
        has = [c.get("hotkey") is not None for c in choices.iter("Choice")]
        if choices.get("name") in QUESTION_KEYS:
            assert all(has), choices.get("name")
        else:
            assert not any(has), choices.get("name")  # not-sure reasons and cause_source get none


def test_doing_is_multiple_others_single():
    root = _config_root()
    by_name = {c.get("name"): c for c in root.iter("Choices")}
    assert by_name["doing"].get("choice") == "multiple"
    for key in ("aimed_at", "feeling", "tone", "cause_source"):
        assert by_name[key].get("choice") == "single", key
    assert set(by_name) == {*QUESTION_KEYS, *(f"{k}_not_sure" for k in QUESTION_KEYS), "cause_source"}
    for q in SCHEME.questions:
        values = list(q.values) + ([q.blank_value] if q.blank_value else [])
        choices = list(by_name[q.key].iter("Choice"))
        assert [c.get("alias") for c in choices] == values
        assert [c.get("value") for c in choices] == [SCHEME.display[v] for v in values]
        assert [c.get("hint") for c in choices] == [SCHEME.help[f"{q.key}.{v}"] for v in values]
        reasons = list(by_name[f"{q.key}_not_sure"].iter("Choice"))
        assert by_name[f"{q.key}_not_sure"].get("choice") == "multiple"
        assert [c.get("alias") for c in reasons] == list(SCHEME.not_sure_reasons)
        assert [c.get("hint") for c in reasons] == [SCHEME.help[f"not_sure.{r}"] for r in SCHEME.not_sure_reasons]
    assert [c.get("alias") for c in by_name["tone"].iter("Choice")][-1] == "no_clear_tone"
    assert [c.get("alias") for c in by_name["cause_source"].iter("Choice")] == list(SCHEME.cause_sources)
    for choices in by_name.values():
        assert choices.get("toName") == "chat"
        assert choices.get("hint") is None  # one hint per <Choice>, none on <Choices>

    (paragraphs,) = root.iter("Paragraphs")
    assert paragraphs.attrib == {"name": "chat", "value": "$messages", "layout": "dialogue",
                                 "nameKey": "speaker", "textKey": "text"}
    (text,) = root.iter("Text")
    assert text.attrib == {"name": "counts", "value": "$counts_text"}
    (note,) = root.iter("TextArea")
    assert (note.get("name"), note.get("toName")) == ("cause_note", "chat")
    assert any(SCHEME.guide_version in (h.get("value") or "") for h in root.iter("Header"))


# --- task JSON -----------------------------------------------------------------------------------------


def test_task_json_holds_speaker_and_text_only_and_predictions_only_with_a_suggestion():
    w = _win("fake1-w0001")
    c = spread.compute(w)
    plain = tasks.task_json(w, c, None)
    assert set(plain) == {"data"}
    assert set(plain["data"]) == {"window_id", "messages", "counts_text"}
    assert plain["data"]["window_id"] == "fake1-w0001"
    assert plain["data"]["messages"] == [{"speaker": "s1", "text": "KEKW KEKW"}, {"speaker": "s2", "text": "LETS GO"}]

    suggestion = {"answers": {"doing": ["laughing", "cheering"], "aimed_at": "streamer", "tone": "no_clear_tone"},
                  "not_sure": {"feeling": ["unclear", "conflicting"]}, "teacher_model": TEACHER[0],
                  "prompt_version": TEACHER[1]}
    task = tasks.task_json(w, c, suggestion)
    (prediction,) = task["predictions"]
    assert prediction["model_version"] == f"{TEACHER[0]}/{TEACHER[1]}"
    assert prediction["result"] == [
        _choices("doing", "laughing", "cheering"),
        _choices("aimed_at", "streamer"),
        _choices("tone", "no_clear_tone"),
        _choices("feeling_not_sure", "unclear", "conflicting"),
    ]
    assert task["data"] == plain["data"]


def test_counts_text_has_no_words():
    # C33: a chanting window, so the chant text exists while counting and must not reach the task.
    texts = ["zebra umbrella forever"] * 6 + ["KEKW", "quokka marmalade"]
    w = _win("fake1-w0009", texts=texts, speakers=[f"s{i + 1}" for i in range(len(texts))])
    c = spread.compute(w)
    assert c.spread == "chant"
    line = tasks.counts_text(c)
    assert line == tasks.task_json(w, c, None)["data"]["counts_text"]
    for word in ("zebra", "umbrella", "forever", "quokka", "marmalade"):
        assert word not in line.lower()
    assert "KEKW" not in line  # an emote shows as its ID, never its name
    assert "8 messages" in line and "chant" in line and "fake-kekw" in line


# --- client ------------------------------------------------------------------------------------------


def _client(tmp_path, handler, *, auth="legacy", runner=None, clock=None, base_url="http://127.0.0.1:14610"):
    return lsclient.LabelStudio(
        base_url,
        token=TOKEN,
        auth=auth,
        data_dir=tmp_path / "data",
        compose_file=tmp_path / "compose.yaml",
        transport=httpx.MockTransport(handler),
        runner=runner,
        clock=clock,
    )


def test_client_refuses_non_loopback(tmp_path):
    for url in (
        "http://192.168.1.5:14610",
        "http://example.com:14610",
        "http://127.0.0.1.example.com:14610",
        "http://fake-user:fake-pass@127.0.0.1:14610",
        "ftp://127.0.0.1:14610",
        "http://0.0.0.0:14610",
        "http:///api",
    ):
        with pytest.raises(ValueError):
            _client(tmp_path, lambda r: httpx.Response(200), base_url=url)
    for url in ("http://127.0.0.1:14610", "http://localhost:14610", "http://[::1]:14610", "http://127.0.0.2:8080/"):
        _client(tmp_path, lambda r: httpx.Response(200), base_url=url).close()
    with pytest.raises(ValueError):
        lsclient.LabelStudio(token=TOKEN, auth="basic", data_dir=tmp_path, compose_file=tmp_path / "c.yaml")
    with pytest.raises(ValueError):
        lsclient.LabelStudio(token="", auth="pat", data_dir=tmp_path, compose_file=tmp_path / "c.yaml")
    with pytest.raises(TypeError):  # every parameter after base_url is keyword-only
        lsclient.LabelStudio("http://127.0.0.1:14610", TOKEN, "pat")


def test_client_satisfies_task_deleter(tmp_path):
    assert isinstance(_client(tmp_path, lambda r: httpx.Response(200)), expiry.TaskDeleter)


def test_legacy_token_header(tmp_path):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(204)

    c = _client(tmp_path, handler, auth="legacy")
    assert c.delete_tasks([5]) == 1
    assert seen[0].headers["Authorization"] == f"Token {TOKEN}"
    assert seen[0].method == "DELETE" and seen[0].url.path == "/api/tasks/5/"


def test_pat_refreshes_an_access_token_every_four_minutes(tmp_path):
    t = [0.0]
    refreshes = []
    seen = []

    def handler(request):
        if request.url.path == "/api/token/refresh/":
            refreshes.append(json.loads(request.content))
            return httpx.Response(200, json={"access": f"access-{len(refreshes)}"})
        seen.append(request.headers["Authorization"])
        return httpx.Response(204)

    c = _client(tmp_path, handler, auth="pat", clock=lambda: t[0])
    c.delete_tasks([1])
    c.delete_tasks([2])
    t[0] = 239.0
    c.delete_tasks([3])
    t[0] = 241.0
    c.delete_tasks([4])
    assert refreshes == [{"refresh": TOKEN}, {"refresh": TOKEN}]
    assert seen == ["Bearer access-1"] * 3 + ["Bearer access-2"]


def test_pat_retries_once_after_a_401(tmp_path):
    refreshes = []
    answers = iter([401, 204])

    def handler(request):
        if request.url.path == "/api/token/refresh/":
            refreshes.append(1)
            return httpx.Response(200, json={"access": f"access-{len(refreshes)}"})
        return httpx.Response(next(answers))

    c = _client(tmp_path, handler, auth="pat")
    assert c.delete_tasks([9]) == 1
    assert len(refreshes) == 2


def test_errors_never_carry_the_token(tmp_path):
    def handler(request):
        return httpx.Response(401, json={"detail": f"bad token {TOKEN}"})

    c = _client(tmp_path, handler, auth="pat")
    with pytest.raises(lsclient.LabelStudioError) as refused:
        c.ensure_project("blind", "<View/>")
    assert TOKEN not in str(refused.value)
    assert "401" in str(refused.value)


def test_delete_tasks_counts_a_missing_task_as_deleted(tmp_path):
    status = {1: 204, 2: 404, 3: 500, 4: 403, 5: 200}

    def handler(request):
        return httpx.Response(status[int(request.url.path.split("/")[3])])

    c = _client(tmp_path, handler)
    assert c.delete_tasks([1, 2]) == 2
    assert c.delete_tasks([3]) == 0
    assert c.delete_tasks([4]) == 0
    assert c.delete_tasks([5]) == 1

    def down(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(httpx.ConnectError):
        _client(tmp_path, down).delete_tasks([1])


def test_ensure_project_creates_once_and_updates_a_changed_config(tmp_path):
    projects: list[dict] = []
    calls = []

    def handler(request):
        calls.append((request.method, request.url.path, dict(request.url.params)))
        if request.method == "GET" and request.url.path == "/api/projects/":
            title = request.url.params["title"]
            found = [p for p in projects if title.lower() in p["title"].lower()]
            return httpx.Response(200, json={"count": len(found), "next": None, "previous": None, "results": found})
        if request.method == "POST" and request.url.path == "/api/projects/":
            body = json.loads(request.content)
            projects.append({"id": 40 + len(projects), **body})
            return httpx.Response(201, json=projects[-1])
        if request.method == "PATCH":
            pid = int(request.url.path.split("/")[3])
            p = next(p for p in projects if p["id"] == pid)
            p.update(json.loads(request.content))
            return httpx.Response(200, json=p)
        return httpx.Response(404)

    c = _client(tmp_path, handler)
    xml = config.label_config(SCHEME)
    assert c.ensure_project("blind", xml) == 40
    assert c.ensure_project("suggested", xml) == 41
    assert [p["title"] for p in projects] == ["Chat Brain · Blind", "Chat Brain · Suggested"]
    assert projects[0]["label_config"] == xml
    assert projects[0]["show_collab_predictions"] is False
    assert projects[1]["show_collab_predictions"] is True
    assert projects[0]["maximum_annotations"] == 1
    assert projects[0]["enable_empty_annotation"] is False

    calls.clear()
    assert c.ensure_project("blind", xml) == 40
    assert [m for m, _, _ in calls] == ["GET"]  # exists, same config: nothing written
    assert c.ensure_project("blind", xml + " ") == 40
    assert [m for m, _, _ in calls][-1] == "PATCH"
    assert c.project_id("repeat") is None


def test_blind_and_repeat_imports_refuse_predictions(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(201, json={"task_count": 1, "task_ids": [7]})

    c = _client(tmp_path, handler)
    w = _win("fake1-w0001")
    suggestion = {"answers": {"doing": ["laughing"]}, "not_sure": {}, "teacher_model": TEACHER[0],
                  "prompt_version": TEACHER[1]}
    with_prediction = tasks.task_json(w, spread.compute(w), suggestion)
    with_annotation = {**tasks.task_json(w, spread.compute(w), None), "annotations": [{"result": []}]}
    for kind in ("blind", "repeat"):
        for task in (with_prediction, with_annotation):
            with pytest.raises(ValueError, match="prediction|annotation"):
                c.import_tasks(5, kind, [task])
    assert calls == []  # refused before anything was sent

    assert c.import_tasks(5, "suggested", [with_prediction]) == [7]
    assert c.import_tasks(5, "blind", [tasks.task_json(w, spread.compute(w), None)]) == [7]
    (first, second) = calls
    assert first.method == "POST" and first.url.path == "/api/projects/5/import"
    assert first.url.params["return_task_ids"] == "true"
    assert json.loads(first.content) == [with_prediction]
    with pytest.raises(ValueError):
        c.import_tasks(5, "page", [with_prediction])


def test_import_refuses_a_count_mismatch(tmp_path):
    c = _client(tmp_path, lambda r: httpx.Response(201, json={"task_count": 0, "task_ids": []}))
    w = _win("fake1-w0001")
    with pytest.raises(lsclient.LabelStudioError):
        c.import_tasks(5, "blind", [tasks.task_json(w, spread.compute(w), None)])


def test_export_annotations(tmp_path):
    seen = []

    def handler(request):
        seen.append(dict(request.url.params))
        return httpx.Response(200, json=BLIND_EXPORT)

    c = _client(tmp_path, handler)
    assert c.export_annotations(2) == BLIND_EXPORT
    assert seen[0] == {"exportType": "JSON"}
    c.export_annotations(2, all_tasks=True)
    assert seen[1] == {"exportType": "JSON", "download_all_tasks": "true"}


@pytest.mark.parametrize("all_tasks", [False, True])
@pytest.mark.parametrize("answer", ["ok", "http_error", "not_json", "unreachable"])
def test_export_deletes_label_studio_copies(tmp_path, all_tasks, answer):
    # Ruling B13: the copy Label Studio writes for each export holds live text, so it
    # goes as soon as the export has been read, whatever the answer was.
    ls_dir = tmp_path / "data" / "labelstudio"
    folders = (ls_dir / "export", ls_dir / "media" / "export")
    ls_dir.mkdir(parents=True)
    (ls_dir / "keep.txt").write_text("not an export", encoding="utf-8")

    def handler(request):
        for folder in folders:  # what Label Studio does while it answers an export call
            (folder / "deep").mkdir(parents=True, exist_ok=True)
            (folder / "project-2-at-fake.json").write_text("zebra-umbrella-41", encoding="utf-8")
            (folder / "deep" / "x.json").write_text("zebra-umbrella-41", encoding="utf-8")
        if answer == "http_error":
            return httpx.Response(500)
        if answer == "not_json":
            return httpx.Response(200, content=b"zebra-umbrella-41")
        if answer == "unreachable":
            raise httpx.ConnectError("made-up refusal", request=request)
        return httpx.Response(200, json=BLIND_EXPORT)

    c = _client(tmp_path, handler)
    if answer == "ok":
        assert c.export_annotations(2, all_tasks=all_tasks) == BLIND_EXPORT
    else:
        with pytest.raises((lsclient.LabelStudioError, httpx.HTTPError)):
            c.export_annotations(2, all_tasks=all_tasks)
    for folder in folders:
        assert list(folder.iterdir()) == []
    assert (ls_dir / "keep.txt").read_text(encoding="utf-8") == "not an export"


def _lock_file(monkeypatch, target: Path):
    """Make ``target`` impossible to delete, like a file another program holds open (WinError 32)."""
    real = Path.unlink

    def locked(self, *args, **kwargs):
        if Path(self) == target:
            raise PermissionError("in use by another program")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", locked)


def _export_copies(tmp_path) -> Path:
    """A made-up export copy in each export folder; returns the one in ``export``."""
    ls_dir = tmp_path / "data" / "labelstudio"
    for folder in (ls_dir / "export", ls_dir / "media" / "export"):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "project-2-at-fake.json").write_text("zebra-umbrella-41", encoding="utf-8")
    return ls_dir / "export" / "project-2-at-fake.json"


def test_compact_fails_while_an_export_copy_cannot_be_deleted(tmp_path, monkeypatch):
    runs = []

    def runner(args, **kwargs):
        runs.append(args)
        return subprocess.CompletedProcess(args, 0, b"", b"")

    c = _client(tmp_path, lambda r: httpx.Response(200), runner=runner)
    held = _export_copies(tmp_path)
    with monkeypatch.context() as m:
        _lock_file(m, held)
        with pytest.raises(lsclient.CompactionFailed, match="export copy"):
            c.compact()
    assert held.exists() and c.export_copies_left == 1
    assert list((tmp_path / "data" / "labelstudio" / "media" / "export").iterdir()) == []  # the rest went
    assert len(runs) == 1  # the database was still vacuumed
    c.compact()  # unlocked: the copy goes and compaction succeeds
    assert not held.exists() and c.export_copies_left == 0


def test_export_counts_an_export_copy_it_could_not_delete(tmp_path, monkeypatch):
    def handler(request):
        _export_copies(tmp_path)  # what Label Studio does while it answers an export call
        return httpx.Response(200, json=[])

    c = _client(tmp_path, handler)
    held = tmp_path / "data" / "labelstudio" / "export" / "project-2-at-fake.json"
    with monkeypatch.context() as m:
        _lock_file(m, held)
        assert c.export_annotations(2) == []
    assert held.exists() and c.export_copies_left == 1
    assert c.export_annotations(2) == [] and c.export_copies_left == 0


def test_expiry_keeps_compaction_pending_while_an_export_copy_is_held(tmp_path, monkeypatch):
    soon = NOW + timedelta(days=1)
    store = _store(tmp_path, _win("fake1-w0120", expires=soon))
    with store.transaction() as conn:
        conn.execute("INSERT INTO ls_tasks (window_id, project_kind, task_id) VALUES ('fake1-w0120', 'blind', 41)")

    def runner(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, b"", b"")

    c = _client(tmp_path, lambda r: httpx.Response(204), runner=runner)
    held = _export_copies(tmp_path)
    with monkeypatch.context() as m:
        _lock_file(m, held)
        r = expiry.run(store, soon, tmp_path / "imported", ls=c)
    assert (r.ls_tasks_deleted, r.ls_compaction_pending, r.ls_error) == (1, True, "CompactionFailed")
    again = expiry.run(store, soon, tmp_path / "imported", ls=c)
    assert (again.ls_compaction_pending, again.ls_error) == (False, None) and not held.exists()


def test_compact_empties_export_folders_and_vacuums(tmp_path):
    runs = []

    def runner(args, **kwargs):
        runs.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, b"", b"")

    c = _client(tmp_path, lambda r: httpx.Response(200), runner=runner)
    ls_dir = tmp_path / "data" / "labelstudio"
    for folder in (ls_dir / "export", ls_dir / "media" / "export"):
        (folder / "deep").mkdir(parents=True)
        (folder / "project-2-at-fake.json").write_text("made-up export", encoding="utf-8")
        (folder / "deep" / "x.json").write_text("made-up", encoding="utf-8")
    (ls_dir / "keep.txt").write_text("not an export", encoding="utf-8")

    c.compact()
    assert list((ls_dir / "export").iterdir()) == []
    assert list((ls_dir / "media" / "export").iterdir()) == []
    assert (ls_dir / "keep.txt").exists()
    ((args, kwargs),) = runs
    assert args[:8] == ["docker", "compose", "-f", str(tmp_path / "compose.yaml"), "exec", "-T", "chatbrain-ls",
                        "python"]
    assert args[8] == "-c" and "VACUUM" in args[9] and lsclient.LS_DATABASE in args[9]
    assert kwargs["env"][paths.ENV_VAR] == str(tmp_path / "data")

    def failing(args, **kwargs):
        return subprocess.CompletedProcess(args, 1, b"", b"made-up failure")

    with pytest.raises(lsclient.CompactionFailed):
        _client(tmp_path, lambda r: httpx.Response(200), runner=failing).compact()


def test_from_env(tmp_path):
    env = tmp_path / ".env"
    kwargs = {"data_dir": tmp_path, "compose_file": tmp_path / "compose.yaml"}
    assert lsclient.from_env(env, **kwargs) is None  # no file: Label Studio isn't configured
    env.write_text("# made up\nLS_AUTH=legacy\n", encoding="utf-8")
    assert lsclient.from_env(env, **kwargs) is None  # no token
    env.write_text(f'export LS_TOKEN="{TOKEN}"\nLS_AUTH=legacy\n\n', encoding="utf-8")
    c = lsclient.from_env(env, **kwargs)
    assert isinstance(c, lsclient.LabelStudio)
    assert (c.base_url, c.auth) == ("http://127.0.0.1:14610", "legacy")
    c.close()
    env.write_text(f"LS_TOKEN={TOKEN}\nLS_AUTH=basic\n", encoding="utf-8")
    with pytest.raises(lsclient.ConfigError) as refused:
        lsclient.from_env(env, **kwargs)
    assert "LS_AUTH" in str(refused.value) and TOKEN not in str(refused.value)
    env.write_text(f"LS_TOKEN={TOKEN}\nLS_URL=http://192.168.1.9:14610\n", encoding="utf-8")
    with pytest.raises(lsclient.ConfigError, match="LS_URL"):
        lsclient.from_env(env, **kwargs)


# --- sync: pull ------------------------------------------------------------------------------------------


def _blind_store(tmp_path, *ids):
    store = _store(tmp_path, *(_win(i) for i in ids))
    _assign(store, "blind", *ids)
    return store


def test_tone_blank_becomes_no_clear_tone(tmp_path):
    store = _blind_store(tmp_path, "fake1-w0001", "fake1-w0004", "fake1-w0005")
    ls = FakeLS(exports={"blind": BLIND_EXPORT})
    r = sync.pull(store, ls, SCHEME, NOW)
    assert (r.written, r.invalid) == (2, [])
    labels = _labels(store)
    assert set(labels) == {"fake1-w0001", "fake1-w0004"}  # the cancelled (skipped) task is not a label

    first = labels["fake1-w0001"]
    assert json.loads(first["answers_json"]) == {
        "doing": ["laughing", "cheering"],  # scheme order, whatever order Label Studio sent
        "aimed_at": "streamer",
        "feeling": "supportive",
        "tone": "no_clear_tone",
    }
    assert json.loads(first["not_sure_json"]) == {}
    assert (first["project"], first["origin"], first["provenance"]) == ("blind", "aron", "blind")
    assert (first["labels_version"], first["guide_version"]) == (SCHEME.version, SCHEME.guide_version)
    assert first["same_as_suggestion"] is None and first["suggestion_json"] is None
    assert first["lead_time_s"] == 42.5
    assert first["ls_annotation_id"] == 2001
    assert first["created_at"] == "2026-11-06T12:30:00.123456Z"  # the annotation's time, fixed width (C35)
    assert (first["cause_note"], first["cause_source"]) == (None, None)

    second = labels["fake1-w0004"]
    assert json.loads(second["answers_json"]) == {"doing": ["asking"], "tone": "serious"}
    assert json.loads(second["not_sure_json"]) == {"aimed_at": ["cause_missing"], "feeling": ["unclear", "conflicting"]}
    assert (second["cause_note"], second["cause_source"]) == ("the made-up boss fight ended", "clip")
    assert second["created_at"] == "2026-11-06T13:00:00.000000Z"
    assert second["lead_time_s"] == 12.0


def test_sync_idempotent(tmp_path):
    store = _blind_store(tmp_path, "fake1-w0001", "fake1-w0004", "fake1-w0005")
    ls = FakeLS(exports={"blind": BLIND_EXPORT})
    assert sync.pull(store, ls, SCHEME, NOW).written == 2
    before = _labels(store)
    again = sync.pull(store, ls, SCHEME, NOW + timedelta(hours=1))
    assert (again.written, again.invalid) == (0, [])
    assert _labels(store) == before


def test_sync_rejects_value_and_not_sure_together(tmp_path):
    store = _blind_store(tmp_path, "fake1-w0002", "fake1-w0003", "fake1-w0006", "fake1-w0007")
    r = sync.pull(store, FakeLS(exports={"blind": INVALID_EXPORT}), SCHEME, NOW)
    reasons = dict(r.invalid)
    assert r.written == 0
    assert "doing" in reasons[2002] and "both" in reasons[2002]
    assert "tone" in reasons[2006] and "both" in reasons[2006]
    assert _labels(store) == {}


def test_sync_rejects_unanswered_question(tmp_path):
    store = _blind_store(tmp_path, "fake1-w0002", "fake1-w0003", "fake1-w0006", "fake1-w0007")
    r = sync.pull(store, FakeLS(exports={"blind": INVALID_EXPORT}), SCHEME, NOW)
    reasons = dict(r.invalid)
    assert sorted(reasons) == [2002, 2003, 2006, 2007]
    assert "aimed_at" in reasons[2003] and "no value" in reasons[2003]
    assert "doing" in reasons[2007]
    assert "not-a-value" not in reasons[2007]  # a value from Label Studio is never echoed
    assert _labels(store) == {}


def test_sync_rejects_other_broken_annotations(tmp_path):
    ids = [f"fake1-w00{n}" for n in range(20, 29)]
    store = _blind_store(tmp_path, *ids)
    note = {"from_name": "cause_note", "to_name": "chat", "type": "textarea", "value": {"text": ["made-up note"]}}
    broken = {
        3101: [*FULL_ANSWER, note],  # a cause note with no source
        3102: [*FULL_ANSWER, _choices("cause_source", "clip")],  # a source with no note
        3103: [*FULL_ANSWER, _choices("mystery", "x")],  # a field the config doesn't have
        3104: [_choices("doing", "laughing"), _choices("aimed_at", "streamer", "guest"), _choices("feeling", "mixed")],
        3105: [*FULL_ANSWER, _choices("doing", "sad")],  # the same field twice
        3106: [*FULL_ANSWER, _choices("feeling_not_sure", "made-up-reason")],
    }
    export = [_annotation(900 + i, ids[i], ann, result) for i, (ann, result) in enumerate(broken.items())]
    export.append(_annotation(990, ids[7], 3107, FULL_ANSWER, created="2026-11-06T15:00:00"))  # naive time
    export.append(_annotation(991, "fake1-w9999", 3108, FULL_ANSWER))  # a window the store doesn't hold
    r = sync.pull(store, FakeLS(exports={"blind": export}), SCHEME, NOW)
    assert r.written == 0
    assert sorted(dict(r.invalid)) == [3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108]
    reasons = dict(r.invalid)
    assert "fake1-w9999" not in reasons[3108] and "made-up" not in " ".join(reasons.values())


def test_sync_checks_the_window_belongs_to_the_project(tmp_path):
    store = _store(tmp_path, _win("fake1-w0030"), _win("fake1-w0031"), _win("fake1-w0032"))
    _assign(store, "suggested", "fake1-w0030")
    # fake1-w0031 has no assignment (B12); fake1-w0032 is blind but answered in the repeat project unscheduled.
    _assign(store, "blind", "fake1-w0032")
    ls = FakeLS(exports={
        "blind": [_annotation(1, "fake1-w0030", 4001, FULL_ANSWER), _annotation(2, "fake1-w0031", 4002, FULL_ANSWER)],
        "repeat": [_annotation(3, "fake1-w0032", 4003, FULL_ANSWER)],
    })
    r = sync.pull(store, ls, SCHEME, NOW)
    assert r.written == 0
    assert sorted(dict(r.invalid)) == [4001, 4002, 4003]


def test_sync_accepted_vs_corrected(tmp_path):
    ids = ("fake1-w0011", "fake1-w0012", "fake1-w0013")
    store = _store(tmp_path, *(_win(i) for i in ids))
    _assign(store, "suggested", *ids)
    _suggest(store, "fake1-w0011", {"doing": ["laughing"], "aimed_at": "streamer", "tone": "playful"},
             {"feeling": ["unclear"]})
    # The Label Studio prediction for w0012 matches the owner's answer, but the stored suggestion doesn't:
    # the comparison must use the suggestions table.
    _suggest(store, "fake1-w0012", {"doing": ["laughing"], "aimed_at": "streamer", "feeling": "supportive",
                                    "tone": "no_clear_tone"})
    _suggest(store, "fake1-w0013", error="timeout")
    r = sync.pull(store, FakeLS(exports={"suggested": SUGGESTED_EXPORT}), SCHEME, NOW)
    assert (r.written, r.invalid) == (3, [])
    labels = _labels(store)

    accepted = labels["fake1-w0011"]
    assert (accepted["project"], accepted["provenance"], accepted["same_as_suggestion"]) == ("suggested", "accepted", 1)
    assert json.loads(accepted["suggestion_json"]) == {
        "answers": {"doing": ["laughing"], "aimed_at": "streamer", "tone": "playful"},
        "not_sure": {"feeling": ["unclear"]},
    }
    assert (accepted["teacher_model"], accepted["prompt_version"]) == TEACHER

    corrected = labels["fake1-w0012"]
    assert (corrected["provenance"], corrected["same_as_suggestion"]) == ("corrected", 0)
    assert (corrected["teacher_model"], corrected["prompt_version"]) == TEACHER

    unsuggested = labels["fake1-w0013"]
    assert (unsuggested["provenance"], unsuggested["same_as_suggestion"]) == ("unsuggested", None)
    assert unsuggested["suggestion_json"] is None and unsuggested["teacher_model"] is None


def test_sync_repeat_project(tmp_path):
    store = _blind_store(tmp_path, "fake1-w0040")
    _repeat(store, "fake1-w0040", NOW - timedelta(days=1))
    r = sync.pull(store, FakeLS(exports={"repeat": [_annotation(5, "fake1-w0040", 4100, FULL_ANSWER)]}), SCHEME, NOW)
    assert (r.written, r.invalid) == (1, [])
    row = _labels(store)["fake1-w0040"]
    assert (row["project"], row["provenance"], row["same_as_suggestion"]) == ("repeat", "repeat", None)


# --- sync: push ---------------------------------------------------------------------------------------------


def test_push_blind_skips_unassigned_labeled_pushed_and_expired_windows(tmp_path):
    ids = ["fake1-w0050", "fake1-w0051", "fake1-w0052", "fake1-w0053", "fake1-w0054", "fake1-w0055", "fake1-w0056"]
    windows = [_win(i) for i in ids[:6]] + [_win(ids[6], expires=NOW + timedelta(hours=1))]
    store = _store(tmp_path, *windows)
    _assign(store, "blind", ids[0], ids[3], ids[4], ids[5], ids[6])
    _assign(store, "suggested", ids[1])  # ids[2] has no assignment (B12)
    _label(store, ids[3], "blind")
    with store.transaction() as conn:
        conn.execute("INSERT INTO ls_tasks (window_id, project_kind, task_id) VALUES (?, 'blind', 77)", (ids[4],))
    ls = FakeLS()
    later = NOW + timedelta(hours=2)  # ids[6]'s text is due to go
    r = sync.push(store, ls, SCHEME, "blind", 10, later)
    assert r.pushed == 2 and r.with_prediction == 0
    assert [t["data"]["window_id"] for _, t, _ in ls.imports] == [ids[0], ids[5]]
    assert all("predictions" not in t for _, t, _ in ls.imports)
    assert ls.configs["blind"] == config.label_config(SCHEME)
    assert (ids[0], "blind", 501) in _ls_tasks(store) and (ids[5], "blind", 502) in _ls_tasks(store)

    assert sync.push(store, ls, SCHEME, "blind", 10, later).pushed == 0  # each window once
    store2 = _store(tmp_path / "again", *(_win(i) for i in ids[:1]))
    _assign(store2, "blind", ids[0])
    assert sync.push(store2, FakeLS(), SCHEME, "blind", 0, NOW).pushed == 0  # a limit of 0 pushes nothing


def test_push_suggested_needs_a_suggestion_row(tmp_path):
    ids = ["fake1-w0060", "fake1-w0061", "fake1-w0062"]
    store = _store(tmp_path, *(_win(i) for i in ids))
    _assign(store, "suggested", *ids)
    _suggest(store, ids[0], {"doing": ["laughing"], "aimed_at": "streamer", "feeling": "supportive"})
    _suggest(store, ids[1], error="timeout")  # ids[2] has no suggestion yet
    ls = FakeLS()
    r = sync.push(store, ls, SCHEME, "suggested", 10, NOW)
    assert (r.pushed, r.with_prediction) == (2, 1)
    by_window = {t["data"]["window_id"]: t for _, t, _ in ls.imports}
    assert set(by_window) == {ids[0], ids[1]}
    assert by_window[ids[0]]["predictions"][0]["model_version"] == "/".join(TEACHER)
    assert "predictions" not in by_window[ids[1]]  # an error-only suggestion is pushed with no prediction (C15)
    assert ls.model_versions == [(1, "/".join(TEACHER))]


def test_push_repeat_sends_only_due_repeats_shuffled_without_predictions(tmp_path):
    ids = [f"fake1-w007{n}" for n in range(6)]
    store = _store(tmp_path, *(_win(i) for i in ids))
    _assign(store, "suggested", *ids)
    for i in ids:
        _suggest(store, i, {"doing": ["laughing"]})
    for i in ids[:4]:
        _repeat(store, i, NOW - timedelta(hours=1))
    _repeat(store, ids[4], NOW + timedelta(hours=1))  # not due yet
    _label(store, ids[3], "repeat")  # already repeated
    ls = FakeLS()
    rng = random.Random(7)
    r = sync.push(store, ls, SCHEME, "repeat", 10, NOW, rng=rng)
    pushed = [t["data"]["window_id"] for _, t, _ in ls.imports]
    assert r.pushed == 3 and sorted(pushed) == ids[:3]
    expected = list(ids[:3])
    random.Random(7).shuffle(expected)
    assert pushed == expected
    assert all("predictions" not in t and "annotations" not in t for _, t, _ in ls.imports)
    assert {k for k, _, _ in ls.imports} == {"repeat"}


def test_push_records_each_task_before_the_next(tmp_path):
    ids = ["fake1-w0080", "fake1-w0081", "fake1-w0082"]
    store = _store(tmp_path, *(_win(i) for i in ids))
    _assign(store, "blind", *ids)
    ls = FakeLS(fail_import_at=1)
    r = sync.push(store, ls, SCHEME, "blind", 10, NOW)
    assert r.pushed == 1 and r.error == "ConnectError"
    assert _ls_tasks(store) == [(ids[0], "blind", 501)]


def test_adopt_untracked_records_tasks_the_store_lost_track_of(tmp_path):
    store = _blind_store(tmp_path, "fake1-w0090")
    ls = FakeLS(exports={"blind": [_annotation(812, "fake1-w0090", 1, []), _annotation(813, "fake9-w0001", 2, [])]})
    adopted = sync.adopt_untracked(store, ls)
    assert (adopted.recorded, adopted.unknown) == (1, 1)
    assert _ls_tasks(store) == [("fake1-w0090", "blind", 812)]
    assert sync.adopt_untracked(store, ls).recorded == 0


# --- expiry through the real client -------------------------------------------------------------------------------


def test_expiry_deletes_ls_tasks(tmp_path):
    soon = NOW + timedelta(days=1)
    store = _store(tmp_path, _win("fake1-w0100", expires=soon), _win("fake1-w0101", expires=soon),
                   _win("fake1-w0102"))
    with store.transaction() as conn:
        conn.executemany(
            "INSERT INTO ls_tasks (window_id, project_kind, task_id) VALUES (?, ?, ?)",
            [("fake1-w0100", "blind", 11), ("fake1-w0100", "repeat", 12), ("fake1-w0101", "suggested", 13),
             ("fake1-w0102", "blind", 14)],
        )
    deletes = []

    def handler(request):
        assert request.method == "DELETE"
        task_id = int(request.url.path.split("/")[3])
        deletes.append(task_id)
        return httpx.Response(404 if task_id == 12 else 204)  # 12 is already gone: counts as deleted

    runs = []

    def runner(args, **kwargs):
        runs.append(args)
        return subprocess.CompletedProcess(args, 0, b"", b"")

    c = _client(tmp_path, handler, runner=runner)
    r = expiry.run(store, soon, tmp_path / "imported", ls=c)
    assert sorted(deletes) == [11, 12, 13]  # every task of each expired window, nothing else
    assert (r.ls_tasks_deleted, r.ls_tasks_pending, r.ls_error) == (3, 0, None)
    assert len(runs) == 1 and "VACUUM" in runs[0][-1]
    assert _ls_tasks(store) == [("fake1-w0102", "blind", 14)]
    again = expiry.run(store, soon, tmp_path / "imported", ls=c)
    assert (again.ls_tasks_deleted, len(runs)) == (0, 1)  # nothing deleted: no second compaction


def test_expiry_pulls_labels_before_deleting(tmp_path):
    soon = NOW + timedelta(days=1)
    store = _store(tmp_path, _win("fake1-w0110", expires=soon), _win("fake1-w0111", expires=soon))
    with store.transaction() as conn:
        conn.executemany("INSERT INTO ls_tasks (window_id, project_kind, task_id) VALUES (?, 'blind', ?)",
                         [("fake1-w0110", 21), ("fake1-w0111", 22)])
    _label(store, "fake1-w0110", "blind")
    ls = FakeLS()
    calls = []

    def pull():
        calls.append(list(ls.order))

    r = expiry.run(store, soon, tmp_path / "imported", ls=ls, before_delete=pull)
    assert calls == [[]]  # pulled before any delete
    assert (r.ls_tasks_deleted, r.ls_pull_error, r.ls_unsynced_deleted) == (2, None, 0)

    # When the pull fails the deletes still run, and the tasks without a stored label are counted.
    store2 = _store(tmp_path / "b", _win("fake1-w0110", expires=soon), _win("fake1-w0111", expires=soon))
    with store2.transaction() as conn:
        conn.executemany("INSERT INTO ls_tasks (window_id, project_kind, task_id) VALUES (?, 'blind', ?)",
                         [("fake1-w0110", 21), ("fake1-w0111", 22)])
    _label(store2, "fake1-w0110", "blind")

    def broken():
        raise httpx.ConnectError("made-up outage")

    ls2 = FakeLS()
    r2 = expiry.run(store2, soon, tmp_path / "b" / "imported", ls=ls2, before_delete=broken)
    assert sorted(ls2.deleted) == [21, 22]
    assert (r2.ls_tasks_deleted, r2.ls_pull_error, r2.ls_unsynced_deleted) == (2, "ConnectError", 1)

    # Nothing due: no pull at all.
    assert expiry.run(store2, soon, tmp_path / "b" / "imported", ls=ls2, before_delete=broken).ls_pull_error is None


# --- CLI ---------------------------------------------------------------------------------------------


def _run(capsys, data: Path, *args: str) -> tuple[int, str]:
    code = cli.main(["--data-dir", str(data), *args])
    out = capsys.readouterr()
    return code, out.out + out.err


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _cli_store(tmp_path, *, exported_days_ago: float = 0) -> tuple[Path, list[str], datetime]:
    data = tmp_path / "data"
    paths.data_dir({paths.ENV_VAR: str(data)})
    now = _now()
    exported = now - timedelta(days=exported_days_ago, hours=2)
    start = exported - timedelta(days=1)
    ids = ["fake1-w0200", "fake1-w0201"]
    windows = [_win(i, start=start, expires=exported + timedelta(days=14)) for i in ids]
    with Store(data / cli.DB_NAME) as store:
        _import(store, tmp_path, *windows, now=now, exported_at=exported)
        _assign(store, "blind", *ids, now=now)
    return data, ids, now


def test_ls_push_refuses_stale_deletion_check(tmp_path, capsys, monkeypatch):
    data, _, _ = _cli_store(tmp_path, exported_days_ago=8)  # B7 coverage is 8 days old, no check since
    ls = FakeLS()
    monkeypatch.setattr(cli, "_label_studio", lambda d: ls)
    code, out = _run(capsys, data, "ls-push", "--kind", "blind", "--limit", "5")
    assert code == 1
    assert "deletion check: stale" in out
    assert ls.imports == []
    assert ls.closed


def test_ls_push_refuses_a_refused_deletion_check_file(tmp_path, capsys, monkeypatch):
    data, _, _ = _cli_store(tmp_path)
    (data / "inbox" / "stray.json").write_text('{"format": "something-else"}', encoding="utf-8")
    ls = FakeLS()
    monkeypatch.setattr(cli, "_label_studio", lambda d: ls)
    code, out = _run(capsys, data, "ls-push", "--kind", "blind", "--limit", "5")
    assert code == 1 and "stray.json" in out and ls.imports == []


def test_ls_push_and_pull_print_counts_only(tmp_path, capsys, monkeypatch):
    data, ids, _ = _cli_store(tmp_path)
    ls = FakeLS()
    monkeypatch.setattr(cli, "_label_studio", lambda d: ls)
    code, out = _run(capsys, data, "ls-setup")
    assert code == 0, out
    assert "3 project(s) ready" in out and set(ls.configs) == {"suggested", "blind", "repeat"}

    code, out = _run(capsys, data, "ls-push", "--kind", "blind", "--limit", "1")
    assert code == 0, out
    assert "pushed 1 task(s) to blind" in out
    assert not any(i in out for i in ids) and "KEKW" not in out

    (_, task, task_id) = ls.imports[0]
    ls.exports["blind"] = [_annotation(task_id, task["data"]["window_id"], 6001, FULL_ANSWER)]
    code, out = _run(capsys, data, "ls-pull")
    assert code == 0, out
    assert "written: 1, invalid: []" in out

    ls.exports["blind"].append(_annotation(task_id + 1, ids[1], 6002, FULL_ANSWER[:1]))
    code, out = _run(capsys, data, "ls-pull")
    assert code == 1
    assert "written: 0, invalid: [(6002, " in out


def test_ls_commands_need_label_studio(tmp_path, capsys):
    data, _, _ = _cli_store(tmp_path)
    for args in (["ls-setup"], ["ls-push", "--kind", "blind", "--limit", "1"], ["ls-pull"]):
        code, out = _run(capsys, data, *args)
        assert code == 1
        assert "Label Studio is not configured" in out


def test_ls_push_needs_a_kind_and_a_limit(tmp_path):
    for args in (["ls-push", "--limit", "1"], ["ls-push", "--kind", "page", "--limit", "1"],
                 ["ls-push", "--kind", "blind", "--limit", "-1"]):
        with pytest.raises(SystemExit):
            cli.main(["--data-dir", str(tmp_path / "data"), *args])


def test_label_studio_client_comes_from_the_real_data_folder(tmp_path, monkeypatch):
    real = tmp_path / "real"
    other = tmp_path / "other"
    paths.data_dir({paths.ENV_VAR: str(real)})
    paths.data_dir({paths.ENV_VAR: str(other)})
    (other / cli.ENV_FILE).write_text(f"LS_TOKEN={TOKEN}\nLS_AUTH=legacy\n", encoding="utf-8")
    assert cli._label_studio(other) is None  # CHATBRAIN_DATA unset: no client, whatever --data-dir holds

    monkeypatch.setenv(paths.ENV_VAR, str(real))
    assert cli._label_studio(real) is None  # the real folder has no .env yet
    (real / cli.ENV_FILE).write_text(f"LS_TOKEN={TOKEN}\nLS_AUTH=legacy\n", encoding="utf-8")
    c = cli._label_studio(real)
    assert isinstance(c, lsclient.LabelStudio)
    assert c.data_dir == real.resolve() and c.compose_file == paths.REPO_ROOT / "compose.yaml"
    c.close()
    assert cli._label_studio(other) is None  # a --data-dir run never reaches the real Label Studio


def test_expire_says_whether_label_studio_was_unreachable_or_did_not_delete(tmp_path, capsys, monkeypatch):
    data, ids, now = _cli_store(tmp_path)
    with Store(data / cli.DB_NAME) as store, store.transaction() as conn:
        conn.execute("INSERT INTO ls_tasks (window_id, project_kind, task_id) VALUES (?, 'blind', 31)", (ids[0],))
    later = iso(now + timedelta(days=20))

    class Refusing(FakeLS):
        def delete_tasks(self, task_ids):
            return 0

    class Down(FakeLS):
        def delete_tasks(self, task_ids):
            raise httpx.ConnectError("made-up outage")

        def export_annotations(self, project_id, *, all_tasks=False):
            raise httpx.ConnectError("made-up outage")

    monkeypatch.setattr(cli, "_label_studio", lambda d: Refusing())
    code, out = _run(capsys, data, "expire", "--now", later)
    assert code == 1, out  # the task's text is still in Label Studio
    assert "Label Studio: not deleted" in out and "unreachable" not in out
    assert "1 task deletion(s) pending" in out

    monkeypatch.setattr(cli, "_label_studio", lambda d: Down())
    code, out = _run(capsys, data, "expire", "--now", later)
    assert code == 1, out  # unreachable with a task due: the scheduled run must fail, not report success
    assert "Label Studio: unreachable (ConnectError)" in out and "not deleted" not in out
    assert "labels were not pulled first (ConnectError)" in out
    assert "1 task deletion(s) pending" in out

    ls = FakeLS()
    monkeypatch.setattr(cli, "_label_studio", lambda d: ls)
    code, out = _run(capsys, data, "expire", "--now", later)
    assert code == 0, out
    assert "Label Studio: 1 task(s) deleted" in out and ls.deleted == [31] and ls.compacts == 1


def test_expire_reports_an_export_copy_it_could_not_delete(tmp_path, capsys, monkeypatch):
    data, ids, now = _cli_store(tmp_path)
    with Store(data / cli.DB_NAME) as store, store.transaction() as conn:
        conn.execute("INSERT INTO ls_tasks (window_id, project_kind, task_id) VALUES (?, 'blind', 32)", (ids[0],))

    class HeldCopy(FakeLS):
        export_copies_left = 0

        def compact(self):
            self.export_copies_left = 1
            raise lsclient.CompactionFailed("export copy still present")

    monkeypatch.setattr(cli, "_label_studio", lambda d: HeldCopy())
    code, out = _run(capsys, data, "expire", "--now", iso(now + timedelta(days=20)))
    assert code == 1, out
    assert "Label Studio: 1 task(s) deleted" in out
    assert "Label Studio: 1 export copy file(s) could not be deleted yet" in out
    assert "Label Studio: compaction pending" in out
    with Store(data / cli.DB_NAME) as store:
        assert store.conn.execute("SELECT COUNT(*) FROM ls_compaction").fetchone()[0] == 1  # retried next run


def test_an_export_copy_left_by_a_pull_makes_compaction_due(tmp_path, capsys, monkeypatch):
    data, _, _ = _cli_store(tmp_path)

    class HeldCopy(FakeLS):
        export_copies_left = 0

        def export_annotations(self, project_id, *, all_tasks=False):
            self.export_copies_left = 1
            return super().export_annotations(project_id, all_tasks=all_tasks)

    ls = HeldCopy()
    monkeypatch.setattr(cli, "_label_studio", lambda d: ls)
    code, out = _run(capsys, data, "ls-pull")
    assert "Label Studio: 1 export copy file(s) could not be deleted yet" in out
    with Store(data / cli.DB_NAME) as store:
        assert store.conn.execute("SELECT COUNT(*) FROM ls_compaction").fetchone()[0] == 1
    ls.export_copies_left = 0
    ls2 = FakeLS()
    monkeypatch.setattr(cli, "_label_studio", lambda d: ls2)
    code, out = _run(capsys, data, "expire")
    assert code == 0, out
    assert ls2.compacts == 1  # the due compaction ran and emptied the export folders again
    with Store(data / cli.DB_NAME) as store:
        assert store.conn.execute("SELECT COUNT(*) FROM ls_compaction").fetchone()[0] == 0


def test_expire_adopts_and_deletes_a_task_the_store_lost_track_of(tmp_path, capsys, monkeypatch):
    # A push stopped between Label Studio's import and the ls_tasks row leaves a task with chat
    # text that no row names; expiry must still find it and delete it once the text is due.
    data, ids, now = _cli_store(tmp_path)
    ls = FakeLS()
    ls.imports.append(("blind", {"data": {"window_id": ids[0]}}, 77))
    monkeypatch.setattr(cli, "_label_studio", lambda d: ls)
    code, out = _run(capsys, data, "expire", "--now", iso(now + timedelta(days=20)))
    assert code == 0, out
    assert "Label Studio: recorded 1 task(s) the store had lost track of" in out
    assert "Label Studio: 1 task(s) deleted" in out and ls.deleted == [77] and ls.compacts == 1
    with Store(data / cli.DB_NAME) as store:
        assert _ls_tasks(store) == []


def test_expire_counts_a_failed_adoption_as_a_label_studio_error(tmp_path, capsys, monkeypatch):
    data, _, now = _cli_store(tmp_path)

    class NoExport(FakeLS):
        def export_annotations(self, project_id, *, all_tasks=False):
            raise httpx.ConnectError("made-up outage")

    monkeypatch.setattr(cli, "_label_studio", lambda d: NoExport())
    code, out = _run(capsys, data, "expire", "--now", iso(now))
    assert code == 1, out  # it could not check Label Studio for tasks it lost track of
    assert "Label Studio: unreachable (ConnectError)" in out


def test_guide_definitions_equal_scheme_help():
    guide = (paths.REPO_ROOT / "docs" / "LABELING_GUIDE.md").read_text(encoding="utf-8")
    assert SCHEME.guide_version in guide
    found = dict(config.guide_definitions(guide))
    assert found == SCHEME.help  # every help key, word for word, and nothing else
