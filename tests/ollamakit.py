"""A made-up local Ollama for the asking tests (Task 10c), behind ``httpx.MockTransport``.

Nothing here opens a socket: every client comes from ``ollama.make_client(transport=...)``, so no
test reaches 127.0.0.1:11434 (ruling B35).

``/api/chat`` streams NDJSON like Ollama: the reply text in three parts, then a last line with
``done`` and ``eval_count``. A step-1 call (it has ``format``) takes its reply from ``read``, a
step-2 call from ``caption``; each is a callable that gets the request body and returns a dict
(sent as JSON text), a str, an exception (raised) or an ``httpx.Response``. An empty ``messages``
list is the preload.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx

from chatbrain import narrator, ollama
from chatbrain.ask import query as q_mod

Reply = Callable[[dict], Any]


def keep(body: dict) -> dict:
    """A step-1 reply that reads every question as the blank query (the tests swap it)."""
    return q_mod.blank()


def plain_caption(body: dict) -> str:
    return "Chat was laughing at {F1}."


def question_of(body: dict) -> str:
    """The question in a step-1 request (its user message's last line)."""
    return body["messages"][-1]["content"].rsplit("Question: ", 1)[-1]


class FakeOllama:
    def __init__(
        self,
        *,
        version: str = "0.40.15",
        models: tuple[str, ...] = ("qwen3:8b",),
        ps: list[dict] | None = None,
        down: bool = False,
        read: Reply = keep,
        caption: Reply = plain_caption,
    ) -> None:
        self.version = version
        self.models = models
        self.ps = ps if ps is not None else []
        self.down = down
        self.read = read
        self.caption = caption
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict] = []  # every /api/chat body, in order

    @property
    def paths(self) -> list[str]:
        return [r.url.path for r in self.requests]

    @property
    def reads(self) -> list[dict]:
        return [b for b in self.bodies if b.get("messages") and "format" in b]

    @property
    def captions(self) -> list[dict]:
        return [b for b in self.bodies if b.get("messages") and "format" not in b]

    @property
    def preloads(self) -> list[dict]:
        return [b for b in self.bodies if b.get("messages") == []]

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.down:
            raise httpx.ConnectError("connection refused", request=request)
        path = request.url.path
        if path == "/api/version":
            return httpx.Response(200, json={"version": self.version})
        if path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": n, "model": n} for n in self.models]})
        if path == "/api/ps":
            return httpx.Response(200, json={"models": self.ps})
        if path == "/api/chat" and request.method == "POST":
            body = json.loads(request.content)
            self.bodies.append(body)
            if body["model"] not in self.models:
                return httpx.Response(404, json={"error": f"model '{body['model']}' not found"})
            if body.get("messages") == []:
                return httpx.Response(200, json={"model": body["model"], "done": True, "done_reason": "load"})
            reply = (self.read if "format" in body else self.caption)(body)
            if isinstance(reply, Exception):
                raise reply
            if isinstance(reply, httpx.Response):
                return reply
            text = reply if isinstance(reply, str) else json.dumps(reply)
            return httpx.Response(200, content=stream(body["model"], text))
        return httpx.Response(404, json={"error": "not found"})

    def client(self, base_url: str = ollama.DEFAULT_URL) -> httpx.Client:
        return ollama.make_client(base_url, transport=httpx.MockTransport(self.handler))

    def narrator(self, **kw: Any) -> narrator.Narrator:
        return narrator.Narrator(self.client(), **kw)


def stream(model: str, text: str, parts: int = 3) -> bytes:
    """``text`` as Ollama's NDJSON chat stream: ``parts`` chunks, then ``done`` with ``eval_count``."""
    step = max(1, -(-len(text) // parts))
    chunks = [text[i : i + step] for i in range(0, len(text), step)] or [""]
    lines = [{"model": model, "message": {"role": "assistant", "content": c}, "done": False} for c in chunks]
    lines.append({"model": model, "message": {"role": "assistant", "content": ""}, "done": True,
                  "done_reason": "stop", "eval_count": 42, "eval_duration": 1_000_000})  # fmt: skip
    return b"".join(json.dumps(x).encode() + b"\n" for x in lines)
