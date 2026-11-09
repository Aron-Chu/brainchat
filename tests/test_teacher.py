"""The local teacher: Ollama checks, the prompt, the JSON schema, suggestions and ``run``.

Everything here is made up and offline: Ollama is ``httpx.MockTransport`` (a fake
teacher), never the network, and Label Studio is a small fake for ``sync.push``.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from chatbrain import cli, fake, ingest, ollama, paths, scheme, spread, teacher
from chatbrain.export_format import EXPORT_FORMAT, Emote, Export, Message, Window, dump_export, iso
from chatbrain.lstudio import sync, tasks
from chatbrain.store import Store

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)
START = NOW - timedelta(days=1)
EXPIRES = NOW + timedelta(days=5)
SCHEME = scheme.load()
KEKW = Emote("fake-kekw", "KEKW")
MODEL_SENTINEL = "zqmodelreply7f3a"  # made up: only ever in a fake teacher reply, never in chat

GOOD_REPLY = {
    "doing": {"value": ["laughing", "cheering"], "not_sure": []},
    "aimed_at": {"value": "streamer", "not_sure": []},
    "feeling": {"value": None, "not_sure": ["unclear", "conflicting"]},
    "tone": {"value": None, "not_sure": []},
}
GOOD_ANSWERS = {"doing": ["laughing", "cheering"], "aimed_at": "streamer", "tone": "no_clear_tone"}
GOOD_NOT_SURE = {"feeling": ["unclear", "conflicting"]}


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    # Never the real data folder, its .env or a real Ollama.
    monkeypatch.delenv(paths.ENV_VAR, raising=False)

    def no_network():
        raise AssertionError("a test reached for the real Ollama client")

    monkeypatch.setattr(cli, "_ollama_client", no_network)


# --- the fake teacher -------------------------------------------------------------------------------


class FakeOllama:
    """A made-up Ollama behind ``httpx.MockTransport``.

    Each ``/api/chat`` call takes the next reply: a dict (sent as the JSON content), a
    str (sent as raw content), an exception (raised), an ``httpx.Response``, or a
    callable that gets the request body and returns one of those.
    """

    def __init__(self, *replies, version: str = "0.31.2"):
        self.replies = list(replies)
        self.version = version
        self.requests: list[httpx.Request] = []
        self.chats: list[dict] = []

    @property
    def paths(self) -> list[str]:
        return [r.url.path for r in self.requests]

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": self.version})
        if request.url.path == "/api/chat" and request.method == "POST":
            body = json.loads(request.content)
            self.chats.append(body)
            reply = self.replies.pop(0) if self.replies else GOOD_REPLY
            if callable(reply):
                reply = reply(body)
            if isinstance(reply, Exception):
                raise reply
            if isinstance(reply, httpx.Response):
                return reply
            content = reply if isinstance(reply, str) else json.dumps(reply)
            return httpx.Response(
                200,
                json={"model": body["model"], "message": {"role": "assistant", "content": content}, "done": True},
            )
        return httpx.Response(404, json={"error": "not found"})

    def client(self, base_url: str = ollama.DEFAULT_URL) -> httpx.Client:
        return ollama.make_client(base_url, transport=httpx.MockTransport(self.handler))

    def sent_windows(self) -> list[str]:
        """The made-up window IDs whose marker line reached the teacher, in order."""
        return [_marker_owner(body) for body in self.chats]


_MARKERS: dict[str, str] = {}


def _marker_owner(body: dict) -> str:
    user = body["messages"][-1]["content"]
    owners = [wid for marker, wid in _MARKERS.items() if marker in user]
    assert len(owners) == 1, "each prompt holds exactly one window's marker"
    return owners[0]


class MiniLS:
    """Just enough Label Studio for ``sync.push``."""

    def __init__(self):
        self.imports: list[tuple[str, dict]] = []
        self.versions: list[str] = []

    def project_id(self, kind):
        return 1

    def ensure_project(self, kind, label_config=None):
        return 1

    def import_tasks(self, project_id, kind, tasks):
        self.imports.append((kind, tasks[0]))
        return [100 + len(self.imports)]

    def export_annotations(self, project_id, *, all_tasks=False):
        return []

    def use_model_version(self, project_id, model_version):
        self.versions.append(model_version)


# --- made-up windows and store ---------------------------------------------------------------------


def _win(window_id, texts=None, *, expires=EXPIRES, start=START) -> Window:
    marker = fake.marker(window_id)
    _MARKERS[marker] = window_id
    texts = texts if texts is not None else ("KEKW KEKW", f"LETS GO {marker}", "@s1 what was that?")
    msgs = tuple(
        Message(f"{window_id}-m{i + 1}", 31_000 + i * 1000, f"s{i + 1}", text,
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


def _store(tmp_path, *windows, exported_at=NOW) -> Store:
    tmp_path.mkdir(parents=True, exist_ok=True)
    store = Store(tmp_path / "cb.db")
    _import(store, tmp_path, *windows, now=exported_at, exported_at=exported_at)
    return store


def _assign(store, exposure, *window_ids):
    with store.transaction() as conn:
        conn.executemany(
            "INSERT INTO assignments (window_id, exposure, assigned_at, seed) VALUES (?, ?, ?, 'fake-seed')",
            [(w, exposure, iso(NOW)) for w in window_ids],
        )


def _suggested_store(tmp_path, *ids) -> Store:
    store = _store(tmp_path, *(_win(i) for i in ids))
    _assign(store, "suggested", *ids)
    return store


def _rows(store) -> list[dict]:
    with store.lock:
        cur = store.conn.execute("SELECT * FROM suggestions ORDER BY id")
        names = [d[0] for d in cur.description]
        return [dict(zip(names, row)) for row in cur.fetchall()]


def _run(store, fake_ollama, *, limit=10, **kwargs) -> int:
    with fake_ollama.client() as client:
        return teacher.run(store, SCHEME, NOW, limit, client, **kwargs)


# --- the request -------------------------------------------------------------------------------------


def test_request_sets_format_num_ctx_think_temperature():
    w = _win("fake1-w0001")
    c = spread.compute(w)
    fo = FakeOllama()
    with fo.client() as client:
        s = teacher.suggest(w, c, SCHEME, client=client)
    assert fo.paths == ["/api/chat"]  # suggest itself never asks for the version
    req = fo.requests[0]
    assert req.method == "POST" and str(req.url) == "http://127.0.0.1:11434/api/chat"
    assert req.extensions["timeout"]["read"] == 120
    body = fo.chats[0]
    assert set(body) == {"model", "messages", "format", "stream", "think", "options"}
    assert body["model"] == "qwen3:8b"
    assert body["messages"] == teacher.build_messages(w, c, SCHEME)
    assert body["format"] == teacher.response_schema(SCHEME)
    assert body["stream"] is False and body["think"] is False
    assert body["options"] == {"num_ctx": 8192, "temperature": 0}
    assert s == teacher.Suggestion(GOOD_ANSWERS, None, "qwen3:8b", "cb_teacher_2026_11_06_v1", not_sure=GOOD_NOT_SURE)
    assert s.body() == {"answers": GOOD_ANSWERS, "not_sure": GOOD_NOT_SURE}

    fo = FakeOllama()
    with fo.client() as client:
        s = teacher.suggest(w, c, SCHEME, client=client, model="qwen3.5:9b", num_ctx=4096, timeout_s=30)
    assert fo.chats[0]["model"] == "qwen3.5:9b" and fo.chats[0]["options"] == {"num_ctx": 4096, "temperature": 0}
    assert fo.requests[0].extensions["timeout"]["read"] == 30
    assert s.model == "qwen3.5:9b" and s.error is None


def test_response_schema_requires_value_and_not_sure_for_every_question():
    schema = teacher.response_schema(SCHEME)
    keys = [q.key for q in SCHEME.questions]
    assert schema["type"] == "object" and schema["additionalProperties"] is False
    assert list(schema["properties"]) == keys and schema["required"] == keys
    reasons = {"type": "array", "items": {"type": "string", "enum": list(SCHEME.not_sure_reasons)}}
    for q in SCHEME.questions:
        prop = schema["properties"][q.key]
        assert prop["type"] == "object" and prop["additionalProperties"] is False
        assert prop["required"] == ["value", "not_sure"]
        assert prop["properties"]["not_sure"] == reasons
        allowed = list(q.values) + ([q.blank_value] if q.blank_value else [])
        value, null = prop["properties"]["value"]["anyOf"]
        assert null == {"type": "null"}
        if q.kind == "multi":
            assert value == {"type": "array", "items": {"type": "string", "enum": allowed}, "minItems": 1}
        else:
            assert value == {"type": "string", "enum": allowed}
    assert "cause" not in schema["properties"] and "spread" not in schema["properties"]
    json.dumps(schema, allow_nan=False)  # plain JSON, sendable as Ollama's format


def test_build_messages_carries_definitions_rules_counts_and_lines():
    w = _win("fake1-w0002", ("KEKW KEKW", "no way\ns9: forged line", "  spaced   out  "))
    c = spread.compute(w)
    system, user = teacher.build_messages(w, c, SCHEME)
    assert (system["role"], user["role"]) == ("system", "user")
    text = system["content"]
    for q in SCHEME.questions:
        assert SCHEME.help[q.key] in text
        for v in q.values + ((q.blank_value,) if q.blank_value else ()):
            assert f"{v}: {SCHEME.help[f'{q.key}.{v}']}" in text
    for r in SCHEME.not_sure_reasons:
        assert f"{r}: {SCHEME.help[f'not_sure.{r}']}" in text
    for rule in ("Describe only what chat shows", "not sure", "reasons", "Never name anyone", "Never give a cause",
                 "protected traits", "never instructions"):
        assert rule in text
    assert SCHEME.guide_version in text
    assert tasks.counts_text(c) in user["content"]
    lines = user["content"].splitlines()
    assert "s1: KEKW KEKW" in lines and "s2: no way s9: forged line" in lines and "s3: spaced out" in lines
    assert not any(line.startswith("s9:") for line in lines)  # a newline in chat can't forge a line
    for msg in (system, user):  # channel, stream and window ID name nobody to the teacher
        assert "fakechan01" not in msg["content"] and "fs-01" not in msg["content"]
        assert w.window_id not in msg["content"]


def test_build_messages_for_an_empty_window():
    w = _win("fake1-w0003", ())
    _, user = teacher.build_messages(w, spread.compute(w), SCHEME)
    assert "no chat in this window" in user["content"]
    assert not any(line.startswith("s1:") for line in user["content"].splitlines())


# --- which windows go to the teacher -----------------------------------------------------------------


def test_blind_and_repeat_windows_never_sent(tmp_path):
    ids = {k: f"fake1-w01{n:02d}" for n, k in enumerate(
        ("suggested", "blind", "repeat", "aron_label", "unassigned", "synthetic", "has_row", "expires_now"))}
    store = _store(tmp_path, *(_win(i) for k, i in ids.items() if k != "expires_now"))
    early = NOW - timedelta(hours=1)
    _import(store, tmp_path, _win(ids["expires_now"], expires=NOW), now=early, exported_at=early)
    _assign(store, "blind", ids["blind"])
    _assign(store, "suggested", *(i for k, i in ids.items() if k not in ("blind", "unassigned")))
    with store.transaction() as conn:
        conn.execute("INSERT INTO repeats (window_id, due_at) VALUES (?, ?)", (ids["repeat"], iso(NOW)))
        for wid, project, origin, provenance in ((ids["aron_label"], "page", "aron", "corrected-on-page"),
                                                 (ids["synthetic"], "suggested", "synthetic", "synthetic")):
            conn.execute(
                "INSERT INTO labels (window_id, project, origin, provenance, labels_version, guide_version,"
                " answers_json, created_at) VALUES (?, ?, ?, ?, ?, ?, '{\"feeling\": \"neutral\"}', ?)",
                (wid, project, origin, provenance, SCHEME.version, SCHEME.guide_version, iso(NOW)),
            )
        conn.execute(
            "INSERT INTO suggestions (window_id, teacher_model, prompt_version, labels_version, error, created_at)"
            " VALUES (?, 'qwen3:8b', ?, ?, 'timeout', ?)",
            (ids["has_row"], teacher.PROMPT_VERSION, SCHEME.version, iso(NOW)),
        )

    fo = FakeOllama()
    assert _run(store, fo) == 2
    assert fo.sent_windows() == [ids["suggested"], ids["synthetic"]]
    assert fo.paths == ["/api/version", "/api/chat", "/api/chat"]
    rows = _rows(store)
    assert [r["window_id"] for r in rows] == [ids["has_row"], ids["suggested"], ids["synthetic"]]

    # A second run finds nothing new to send.
    fo = FakeOllama()
    assert _run(store, fo) == 0 and fo.chats == []


def test_cause_only_aron_row_leaves_a_suggested_window_eligible(tmp_path):
    """A page cause note names no question (C1), so the window still gets a suggestion and reaches Label Studio."""
    ids = ("fake1-w0150", "fake1-w0151")
    store = _suggested_store(tmp_path, *ids)
    with store.transaction() as conn:
        for wid, answers in ((ids[0], "{}"), (ids[1], '{"feeling": "neutral"}')):
            conn.execute(
                "INSERT INTO labels (window_id, project, origin, provenance, labels_version, guide_version,"
                " answers_json, not_sure_json, cause_note, cause_source, created_at)"
                " VALUES (?, 'page', 'aron', 'corrected-on-page', ?, ?, ?, '{}', 'made-up raid', 'clip', ?)",
                (wid, SCHEME.version, SCHEME.guide_version, answers, iso(NOW)),
            )

    fo = FakeOllama(GOOD_REPLY)
    assert _run(store, fo) == 1
    assert fo.sent_windows() == [ids[0]]  # the row that names a question still keeps the teacher away (C14)
    assert [(r["window_id"], r["error"]) for r in _rows(store)] == [(ids[0], None)]
    ls = MiniLS()
    assert sync.push(store, ls, SCHEME, "suggested", 10, NOW).with_prediction == 1


def test_run_takes_the_soonest_expiring_windows_up_to_the_limit(tmp_path):
    late, soon, mid = (_win("fake1-w0200", expires=NOW + timedelta(days=9)),
                       _win("fake1-w0201", expires=NOW + timedelta(days=2)),
                       _win("fake1-w0202", expires=NOW + timedelta(days=4)))
    store = _store(tmp_path, late, soon, mid)
    _assign(store, "suggested", late.window_id, soon.window_id, mid.window_id)
    fo = FakeOllama()
    assert _run(store, fo, limit=2) == 2
    assert fo.sent_windows() == [soon.window_id, mid.window_id]
    for bad in (-1, True, 1.5, "3"):
        with pytest.raises(ValueError):
            _run(store, FakeOllama(), limit=bad)


def test_a_window_that_changes_during_the_call_gets_no_row(tmp_path):
    ids = ("fake1-w0250", "fake1-w0251")
    store = _suggested_store(tmp_path, *ids)

    def labeled_meanwhile(body):  # the owner fixes the window on the page while the teacher thinks
        with store.transaction() as conn:
            conn.execute(
                "INSERT INTO labels (window_id, project, origin, provenance, labels_version, guide_version,"
                " answers_json, created_at) VALUES (?, 'page', 'aron', 'corrected-on-page', ?, ?,"
                " '{\"feeling\": \"neutral\"}', ?)",
                (ids[0], SCHEME.version, SCHEME.guide_version, iso(NOW)),
            )
        return GOOD_REPLY

    fo = FakeOllama(labeled_meanwhile, GOOD_REPLY)
    assert _run(store, fo) == 1
    assert fo.sent_windows() == list(ids)
    assert [r["window_id"] for r in _rows(store)] == [ids[1]]


def test_text_that_expires_during_the_batch_is_not_sent_or_recorded(tmp_path):
    # The batch starts at NOW and each made-up call takes 3 s on a fake clock.
    own_call, earlier_call, alive = (
        _win("fake1-w0260", expires=NOW + timedelta(seconds=2)),  # expires during its own call
        _win("fake1-w0261", expires=NOW + timedelta(seconds=3)),  # expires as the first call ends (== is expired)
        _win("fake1-w0262", expires=NOW + timedelta(minutes=1)),
    )
    store = _store(tmp_path, own_call, earlier_call, alive)
    _assign(store, "suggested", own_call.window_id, earlier_call.window_id, alive.window_id)
    elapsed = [100.0]  # a monotonic clock starts anywhere

    def slow_reply(body):
        elapsed[0] += 3
        return GOOD_REPLY

    fo = FakeOllama(slow_reply, slow_reply, slow_reply)
    with fo.client() as client:
        assert teacher.run(store, SCHEME, NOW, 10, client, clock=lambda: elapsed[0]) == 1
    assert fo.sent_windows() == [own_call.window_id, alive.window_id]  # never the one that expired meanwhile
    rows = _rows(store)
    assert [r["window_id"] for r in rows] == [alive.window_id]
    assert rows[0]["created_at"] == iso(NOW)  # rows still carry the batch's start time


# --- what gets stored ----------------------------------------------------------------------------------


def test_answers_stored_in_the_sync_shape_and_pushed_as_a_prediction(tmp_path):
    ids = ("fake1-w0300", "fake1-w0301")
    store = _suggested_store(tmp_path, *ids)
    assert _run(store, FakeOllama(GOOD_REPLY, {**GOOD_REPLY, "doing": {"value": ["cheering", "laughing"],
                                                                       "not_sure": []}}), model="qwen3.5:9b") == 2
    for row in _rows(store):
        assert row["teacher_model"] == "qwen3.5:9b"
        assert row["prompt_version"] == "cb_teacher_2026_11_06_v1"
        assert row["labels_version"] == SCHEME.version and row["created_at"] == iso(NOW)
        assert row["error"] is None
        # Values come back in the scheme's order, whatever order the teacher used.
        assert json.loads(row["answers_json"]) == {"answers": GOOD_ANSWERS, "not_sure": GOOD_NOT_SURE}
        with store.lock:
            s = sync.latest_suggestion(store.conn, SCHEME, row["window_id"])
        assert (s.answers, s.not_sure) == (GOOD_ANSWERS, GOOD_NOT_SURE)

    ls = MiniLS()
    assert sync.push(store, ls, SCHEME, "suggested", 10, NOW).with_prediction == 2
    assert all("predictions" in task for _, task in ls.imports)
    assert ls.versions[-1] == "qwen3.5:9b/cb_teacher_2026_11_06_v1"


def test_malformed_json_recorded_as_error(tmp_path):
    malformed = (
        "not json {" + MODEL_SENTINEL,  # content isn't JSON
        httpx.Response(200, content=b"<html>" + MODEL_SENTINEL.encode()),  # the body isn't JSON
        httpx.Response(200, json={"message": {"role": "assistant"}, "done": True}),  # no content
        '{"doing": {"value": ["laughing"], "not_sure": []}, "doing": {"value": null, "not_sure": []}}',
        '{"feeling": {"value": NaN, "not_sure": []}}',  # not strict JSON
    )
    replies = (*malformed, GOOD_REPLY)  # five bad replies in a row, then a good one
    ids = tuple(f"fake1-w04{n:02d}" for n in range(len(replies)))
    store = _suggested_store(tmp_path, *ids)
    assert _run(store, FakeOllama(*replies)) == len(ids)  # the batch keeps going
    rows = _rows(store)
    expected = [None if r is GOOD_REPLY else "bad_json" for r in replies]
    assert [r["error"] for r in rows] == expected
    assert [r["answers_json"] is None for r in rows] == [e is not None for e in expected]

    # Each window stays labelable: pushed to the suggested project, with no prediction.
    ls = MiniLS()
    assert sync.push(store, ls, SCHEME, "suggested", 10, NOW).pushed == len(ids)
    with_prediction = [task["data"]["window_id"] for _, task in ls.imports if "predictions" in task]
    assert with_prediction == [i for i, e in zip(ids, expected) if e is None]

    store.close()  # the model's words never reach the store (ruling T2)
    assert MODEL_SENTINEL.encode() not in (tmp_path / "cb.db").read_bytes()


@pytest.mark.parametrize(
    "change",
    [
        {"doing": {"value": ["dancing"], "not_sure": []}},  # out of the scheme
        {"aimed_at": {"value": "the_mods", "not_sure": []}},
        {"feeling": {"value": None, "not_sure": ["bored"]}},
        {"tone": {"value": "no_clear_tone_at_all", "not_sure": []}},
        {"doing": {"value": "laughing", "not_sure": []}},  # a multi question takes a list
        {"aimed_at": {"value": ["streamer"], "not_sure": []}},  # a single question takes one value
        {"doing": {"value": ["laughing", "laughing"], "not_sure": []}},  # repeats a value
        {"feeling": {"value": None, "not_sure": ["unclear", "unclear"]}},  # repeats a reason
        {"aimed_at": {"value": "streamer", "not_sure": ["unclear"]}},  # both a value and a reason
        {"aimed_at": {"value": None, "not_sure": []}},  # neither
        {"doing": {"value": [], "not_sure": []}},  # neither
        {"feeling": {"value": None}},  # a field missing
        {"feeling": {"value": "neutral", "not_sure": [], "why": MODEL_SENTINEL}},  # an extra field
        {"cause": {"value": MODEL_SENTINEL, "not_sure": []}},  # not a question
        {"feeling": "neutral"},
    ],
)
def test_out_of_enum_value_recorded_as_error(tmp_path, change):
    reply = {**GOOD_REPLY, **change}
    w = _win("fake1-w0500")
    with FakeOllama(reply).client() as client:
        s = teacher.suggest(w, spread.compute(w), SCHEME, client=client)
    assert (s.answers, s.not_sure, s.error) == (None, None, "schema")
    assert s.body() is None

    store = _suggested_store(tmp_path, "fake1-w0501", "fake1-w0502")
    assert _run(store, FakeOllama(reply, GOOD_REPLY)) == 2  # recorded, and the batch keeps going
    assert [(r["answers_json"], r["error"]) for r in _rows(store)][0] == (None, "schema")
    assert _rows(store)[1]["error"] is None


def test_missing_question_or_a_non_object_reply_recorded_as_error():
    w = _win("fake1-w0510")
    for reply in ({k: v for k, v in GOOD_REPLY.items() if k != "tone"}, "[]", '"laughing"', "null"):
        with FakeOllama(reply).client() as client:
            assert teacher.suggest(w, spread.compute(w), SCHEME, client=client).error == "schema"


def test_explicit_blank_tone_and_all_not_sure_are_valid():
    w = _win("fake1-w0520")
    reply = {q.key: {"value": None, "not_sure": ["needs_review"]} for q in SCHEME.questions}
    reply["tone"] = {"value": "no_clear_tone", "not_sure": []}
    with FakeOllama(reply).client() as client:
        s = teacher.suggest(w, spread.compute(w), SCHEME, client=client)
    assert s.error is None
    assert s.answers == {"tone": "no_clear_tone"}
    assert s.not_sure == {"doing": ["needs_review"], "aimed_at": ["needs_review"], "feeling": ["needs_review"]}


# --- failures ------------------------------------------------------------------------------------------


def test_timeout_recorded_and_batch_continues(tmp_path):
    ids = ("fake1-w0600", "fake1-w0601", "fake1-w0602")
    store = _suggested_store(tmp_path, *ids)
    timeout = httpx.ReadTimeout("made-up timeout")
    fo = FakeOllama(timeout, GOOD_REPLY, httpx.Response(500, json={"error": MODEL_SENTINEL}))
    assert _run(store, fo) == 3
    assert fo.sent_windows() == list(ids)
    rows = _rows(store)
    assert [(r["window_id"], r["error"]) for r in rows] == [(ids[0], "timeout"), (ids[1], None), (ids[2], "http_500")]
    assert rows[1]["answers_json"] is not None


def test_errors_in_a_row_keep_the_batch_going(tmp_path):
    # Review Focus 3: every picked window gets a row, however many errors come in a row.
    ids = tuple(f"fake1-w07{n:02d}" for n in range(6))
    store = _suggested_store(tmp_path, *ids)
    timeout = httpx.ReadTimeout("made-up")
    fo = FakeOllama("x", timeout, "{}", timeout, httpx.Response(500, json={"error": MODEL_SENTINEL}), GOOD_REPLY)
    assert _run(store, fo) == len(ids)
    assert fo.sent_windows() == list(ids)
    assert [r["error"] for r in _rows(store)] == ["bad_json", "timeout", "schema", "timeout", "http_500", None]


def test_unreachable_or_missing_model_stops_the_batch_without_a_row(tmp_path):
    ids = ("fake1-w0800", "fake1-w0801", "fake1-w0802")
    store = _suggested_store(tmp_path, *ids)
    with pytest.raises(ollama.OllamaError, match="not reachable"):
        _run(store, FakeOllama(GOOD_REPLY, httpx.ConnectError("made-up refusal")))
    assert [r["window_id"] for r in _rows(store)] == [ids[0]]  # the second window is tried again next run

    with pytest.raises(ollama.OllamaError, match="ollama pull"):
        _run(store, FakeOllama(httpx.Response(404, json={"error": "model 'qwen3:8b' not found"})))
    assert len(_rows(store)) == 1

    w = _win("fake1-w0803")
    with FakeOllama(httpx.ConnectError("made-up")).client() as client, pytest.raises(ollama.OllamaError):
        teacher.suggest(w, spread.compute(w), SCHEME, client=client)

    assert _run(store, FakeOllama()) == 2  # once Ollama is back, the rest get their suggestions


# --- refusals ------------------------------------------------------------------------------------------


def test_refuses_non_loopback_ollama(tmp_path):
    for url in ("http://192.168.1.5:11434", "http://example.com:11434", "http://127.0.0.1.example.com:11434",
                "http://0.0.0.0:11434", "http://fake-user:fake-pass@127.0.0.1:11434", "ftp://127.0.0.1:11434",
                "http://127.0.0.1:11434/?x=1", "http:///api"):
        with pytest.raises(ollama.OllamaError):
            ollama.make_client(url)
    for url in ("http://127.0.0.1:11434", "http://localhost:11434", "http://[::1]:11434"):
        ollama.make_client(url).close()

    store = _suggested_store(tmp_path, "fake1-w0900")
    fo = FakeOllama()
    transport = httpx.MockTransport(fo.handler)
    w = _win("fake1-w0901")
    for client in (
        httpx.Client(base_url="http://192.168.1.5:11434", transport=transport, trust_env=False),
        httpx.Client(base_url="http://example.com:11434", transport=transport, trust_env=False),
        httpx.Client(base_url="http://127.0.0.1:11434", transport=transport),  # trusts proxy settings
        httpx.Client(base_url="http://127.0.0.1:11434", transport=transport, trust_env=False, follow_redirects=True),
    ):
        with client, pytest.raises(ollama.OllamaError):
            teacher.run(store, SCHEME, NOW, 10, client)
        with pytest.raises(ollama.OllamaError):
            teacher.suggest(w, spread.compute(w), SCHEME, client=client)
    assert fo.requests == [] and _rows(store) == []


@pytest.mark.parametrize(
    "version", ["0.31.1", "0.30.9", "0.31.2-rc1", "0.0.0", "garbage", "", None, 31, "0.31"]
)
def test_refuses_old_ollama(tmp_path, version):
    store = _suggested_store(tmp_path, "fake1-w1000")
    fo = FakeOllama(version=version)
    with fo.client() as client:
        with pytest.raises(ollama.OllamaError, match="0.31.2"):
            ollama.check(client)
        with pytest.raises(ollama.OllamaError, match="0.31.2"):
            teacher.run(store, SCHEME, NOW, 10, client)
    assert "/api/chat" not in fo.paths and _rows(store) == []


@pytest.mark.parametrize("version", ["0.31.2", "0.31.10", "0.32.0", "1.0.0", "0.31.3-rc0"])
def test_accepts_new_enough_ollama(version):
    with FakeOllama(version=version).client() as client:
        ollama.check(client)


def test_version_endpoint_failures_refuse():
    for handler in (lambda r: httpx.Response(500), lambda r: httpx.Response(200, content=b"<html>"),
                    lambda r: httpx.Response(200, json=["0.31.2"])):
        with ollama.make_client(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(ollama.OllamaError):
                ollama.check(client)

    def down(request):
        raise httpx.ConnectError("made-up refusal")

    with ollama.make_client(transport=httpx.MockTransport(down)) as client:
        with pytest.raises(ollama.OllamaError, match="not reachable"):
            ollama.check(client)


def test_stale_deletion_check_blocks_run(tmp_path):
    old = NOW - timedelta(days=8)
    w = _win("fake1-w1100", start=old - timedelta(days=1), expires=NOW + timedelta(days=1))
    store = _store(tmp_path, w, exported_at=old)  # export coverage is 8 days old, no check since
    _assign(store, "suggested", w.window_id)
    fo = FakeOllama()
    with pytest.raises(ingest.StaleDeletionCheck):
        _run(store, fo)
    assert fo.requests == [] and _rows(store) == []  # refused before Ollama is even asked its version


# --- totals ---------------------------------------------------------------------------------------------


def test_totals_count_answers_and_errors_by_code(tmp_path):
    store = _suggested_store(tmp_path, *(f"fake1-w12{n:02d}" for n in range(4)))
    _run(store, FakeOllama(GOOD_REPLY, "x", httpx.ReadTimeout("made-up"), GOOD_REPLY))
    assert teacher.totals(store) == {"answers": 2, "bad_json": 1, "timeout": 1}


# --- CLI ------------------------------------------------------------------------------------------------


def _cli(capsys, data: Path, *args: str) -> tuple[int, str]:
    code = cli.main(["--data-dir", str(data), *args])
    out = capsys.readouterr()
    return code, out.out + out.err


def _cli_store(tmp_path, *, exported_days_ago: float = 0) -> tuple[Path, list[str]]:
    data = tmp_path / "data"
    paths.data_dir({paths.ENV_VAR: str(data)})
    now = datetime.now(timezone.utc).replace(microsecond=0)
    exported = now - timedelta(days=exported_days_ago, hours=2)
    start = exported - timedelta(days=1)
    ids = ["fake1-w1300", "fake1-w1301", "fake1-w1302"]
    windows = [_win(i, start=start, expires=exported + timedelta(days=14)) for i in ids]
    with Store(data / cli.DB_NAME) as store:
        _import(store, tmp_path, *windows, now=now, exported_at=exported)
        _assign(store, "suggested", *ids[:2])
        _assign(store, "blind", ids[2])
    return data, ids


def test_suggest_command_prints_counts_only(tmp_path, capsys, monkeypatch):
    data, ids = _cli_store(tmp_path)
    fo = FakeOllama(GOOD_REPLY, "x")
    clients = []
    monkeypatch.setattr(cli, "_ollama_client", lambda: clients.append(fo.client()) or clients[-1])
    code, out = _cli(capsys, data, "suggest", "--limit", "5")
    assert code == 0, out
    assert "recorded 2 suggestion(s)" in out and "1 with answers" in out and "bad_json 1" in out
    assert fo.sent_windows() == ids[:2] and fo.chats[0]["model"] == "qwen3:8b"
    for secret in (*ids, "KEKW", "LETS GO", fake.marker(ids[0])):
        assert secret not in out
    assert clients and clients[0].is_closed

    fo2 = FakeOllama()
    monkeypatch.setattr(cli, "_ollama_client", fo2.client)
    code, out = _cli(capsys, data, "suggest", "--limit", "5", "--model", "qwen3.5:9b")
    assert code == 0 and "recorded 0 suggestion(s)" in out and fo2.chats == []


def test_suggest_command_refusals(tmp_path, capsys, monkeypatch):
    data, _ = _cli_store(tmp_path, exported_days_ago=8)
    fo = FakeOllama()
    monkeypatch.setattr(cli, "_ollama_client", fo.client)
    code, out = _cli(capsys, data, "suggest", "--limit", "5")
    assert code == 1 and "deletion check: stale" in out and fo.requests == []

    data, _ = _cli_store(tmp_path / "old")
    fo = FakeOllama(version="0.31.1")
    monkeypatch.setattr(cli, "_ollama_client", fo.client)
    code, out = _cli(capsys, data, "suggest", "--limit", "5")
    assert code == 1 and "0.31.2" in out and "/api/chat" not in fo.paths

    (data / "inbox" / "stray.json").write_text('{"format": "something-else"}', encoding="utf-8")
    fo = FakeOllama()
    monkeypatch.setattr(cli, "_ollama_client", fo.client)
    code, out = _cli(capsys, data, "suggest", "--limit", "5")
    assert code == 1 and "stray.json" in out and fo.requests == []


def test_suggest_command_needs_a_limit_and_a_plain_model_name(tmp_path):
    for args in (["suggest"], ["suggest", "--limit", "-1"], ["suggest", "--limit", "1", "--model", ""],
                 ["suggest", "--limit", "1", "--model", "qwen3 8b"], ["suggest", "--limit", "1", "--model", "-x"]):
        with pytest.raises(SystemExit):
            cli.main(["--data-dir", str(tmp_path / "data"), *args])
