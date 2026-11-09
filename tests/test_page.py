import dataclasses
import hashlib
import json
import re
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from chatbrain import fake, paths
from chatbrain.export_format import StreamActivity, dump_export, parse_export
from chatbrain.page import activity as activity_mod
from chatbrain.page import app as app_mod
from pagekit import (
    NOW, RUNG0_ID, RUNG1_ID, SCHEME, activity_line, build_world, clear_activity, client, counts, label,
    label_rows, predict, report, store_activity, stub_model, text,
)  # fmt: skip

STATIC = Path(app_mod.__file__).parent / "static"
TEMPLATES = Path(app_mod.__file__).parent / "templates"


@pytest.fixture(autouse=True)
def _no_real_data_folder(monkeypatch):
    monkeypatch.delenv(paths.ENV_VAR, raising=False)


@pytest.fixture
def world(tmp_path):
    w = build_world(tmp_path)
    yield w
    w.store.close()


def _message_texts(world, window_id: str) -> list[str]:
    return [m.text for m in world.windows[window_id].messages]


SECTION_TITLE = {"Answers": "What chat did"}  # Task 4 (G1) renamed the answers section inside Details


def _section(page: str, title: str) -> str:
    """The HTML of the Moment's Details section whose heading starts with ``title``."""
    title = SECTION_TITLE.get(title, title)
    found = re.search(rf'<section class="sd-sec">\s*<h3 class="sd-h">{re.escape(title)}.*?</section>', page, re.S)
    assert found, f"no section {title!r}"
    return found.group(0)


def _chat(page: str) -> str:
    """The "What chat said" block of an opened Moment."""
    return page.split('<section class="sd-chat"')[1].split("</section>")[0]


def _rows(fragment: str) -> list[tuple[str, str]]:
    """``(value, confidence or reason)`` for each answer row in ``fragment``."""
    found = re.findall(r'<div class="vr[^"]*"><span class="vn">(.*?)</span>(?:<span class="cv">(.*?)</span>)?', fragment)
    return [(text(v), text(c)) for v, c in found]


def _answer(page: str, i: int) -> str:
    """The i-th question block (0 doing, 1 aimed at, 2 feeling, 3 tone) of the Answers section."""
    return _section(page, "Answers").split('<div class="aq">')[i + 1]


def _header(page: str) -> str:
    return page.split('<header class="sd-head">')[1].split("</header>")[0]


# --- serving ---------------------------------------------------------------------------------------


def test_serve_refuses_0_0_0_0(world, monkeypatch):
    app = client(world).app
    calls = []
    monkeypatch.setattr(app_mod.uvicorn, "run", lambda *a, **k: calls.append((a, k)))
    for host in ("0.0.0.0", "::", "", "192.168.1.20", "127.0.0.2", "localhost.example"):
        with pytest.raises(ValueError):
            app_mod.serve(app, host=host)
    assert calls == []

    app_mod.serve(app)
    app_mod.serve(app, host="localhost", port=14699)
    assert [(k["host"], k["port"], k["access_log"]) for _, k in calls] == [
        ("127.0.0.1", 14612, False),
        ("localhost", 14699, False),
    ]


def test_page_makes_no_outbound_requests(world):
    stub_model(world.store, world.models, RUNG0_ID, 0)
    a = world.new_moments[0]
    predict(world.store, a, RUNG0_ID, "doing", ["laughing"], 0.9)
    c = client(world)
    pages = [c.get("/").text, c.get(f"/m/{a}").text, c.get(f"/m/{a}?panel=1").text, c.get(f"/?stream={world.old}").text]
    refs = [r for p in pages for r in re.findall(r'\b(?:src|href|action)\s*=\s*"([^"]*)"', p)]
    assert refs
    bad = [r for r in refs if not (r.startswith("/") or r.startswith("#")) or r.startswith("//")]  # "#" stays on the page
    assert bad == []
    for p in pages:
        assert "http://" not in p and "https://" not in p
        assert "@import" not in p
    files = [f for f in STATIC.rglob("*") if f.suffix in {".css", ".js"}] + list(TEMPLATES.rglob("*.html"))
    assert {f.suffix for f in files} >= {".css", ".js", ".html"}
    for f in files:
        body = f.read_text(encoding="utf-8")
        assert "http://" not in body and "https://" not in body, f.name
    for ref in sorted({r.split("?")[0] for r in refs if r.startswith("/static/")}):
        assert c.get(ref).status_code == 200, ref
    css = (STATIC / "page.css").read_text(encoding="utf-8")
    for font in re.findall(r"url\((/static/fonts/[^)]+)\)", css):
        assert c.get(font).status_code == 200, font
    assert "Sentry" not in css and "sentry" not in css


def test_post_from_another_site_is_refused(world):
    a = world.new_moments[0]
    c = client(world)
    fix = {"question": "doing", "value": ["sad"]}
    r = c.post(f"/m/{a}/fix", data=fix, headers={"Origin": "http://evil.example"}, follow_redirects=False)
    assert r.status_code == 403
    r = c.post(f"/m/{a}/cause", data={"cause_note": "x", "cause_source": "clip"},
               headers={"Sec-Fetch-Site": "cross-site"}, follow_redirects=False)  # fmt: skip
    assert r.status_code == 403
    assert label_rows(world.store) == []
    assert TestClient(c.app, base_url="http://evil.example").get("/").status_code == 400  # DNS rebinding
    page = c.get("/")
    assert page.headers["x-frame-options"] == "DENY" and "default-src 'self'" in page.headers["content-security-policy"]
    ok = c.post(f"/m/{a}/fix", data=fix, headers={"Origin": f"http://{app_mod.HOST}:{app_mod.PORT}"},
                follow_redirects=False)  # fmt: skip
    assert ok.status_code == 303 and len(label_rows(world.store)) == 1


def test_form_post_with_null_origin_from_the_page_is_saved(world):
    """A native form post (full Moment page, or script off) may carry ``Origin: null``; Sec-Fetch-Site decides."""
    a = world.new_moments[0]
    c = client(world)
    page = c.get(f"/m/{a}")
    assert page.headers["referrer-policy"] == "same-origin"  # same-origin posts carry the real Origin
    assert '<meta name="referrer" content="same-origin">' in page.text
    fix = {"question": "doing", "value": ["sad"]}
    cause = {"cause_note": "raid", "cause_source": "clip"}

    for site in (None, "cross-site", "same-site"):
        headers = {"Origin": "null"} | ({"Sec-Fetch-Site": site} if site else {})
        assert c.post(f"/m/{a}/fix", data=fix, headers=headers, follow_redirects=False).status_code == 403, site
        assert c.post(f"/m/{a}/cause", data=cause, headers=headers, follow_redirects=False).status_code == 403, site
    assert label_rows(world.store) == []

    same = {"Origin": "null", "Sec-Fetch-Site": "same-origin"}
    r = c.post(f"/m/{a}/fix", data=fix, headers=same, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == f"/m/{a}"
    assert len(label_rows(world.store)) == 1
    r = c.post(f"/m/{a}/cause", data=cause, headers=same, follow_redirects=False)
    assert r.status_code == 303 and len(label_rows(world.store)) == 2


def test_page_command_serves_on_loopback(tmp_path, monkeypatch, capsys):
    from chatbrain import cli

    from ollamakit import FakeOllama

    calls = []
    monkeypatch.setattr(app_mod.uvicorn, "run", lambda app, **k: calls.append(k))
    fake = FakeOllama()  # the page builds its narrator on this made-up Ollama, never on 127.0.0.1:11434 (B35)
    monkeypatch.setattr(cli, "_ollama_client", fake.client)
    data = tmp_path / "data"
    assert cli.main(["--data-dir", str(data), "page"]) == 0
    assert cli.main(["--data-dir", str(data), "page", "--host", "localhost", "--port", "14699"]) == 0
    assert [(k["host"], k["port"]) for k in calls] == [("127.0.0.1", 14612), ("localhost", 14699)]
    assert "127.0.0.1:14612" in capsys.readouterr().out
    for bad in (["--host", "0.0.0.0"], ["--port", "0"]):
        with pytest.raises(SystemExit):
            cli.main(["--data-dir", str(data), "page", *bad])
    assert len(calls) == 2
    assert fake.requests == []  # building the page asks Ollama nothing


# --- the list --------------------------------------------------------------------------------------


def test_doing_param_checked_not_filtering(world):
    """The glance (Task 3) has no doing filter: ``?doing=`` is still checked, but the lane keeps every Moment."""
    stub_model(world.store, world.models, RUNG0_ID, 0)
    m = world.new_moments
    doing = {m[0]: ["laughing"], m[1]: ["cheering"], m[2]: ["laughing", "shocked"]}
    for w in m:
        predict(world.store, w, RUNG0_ID, "doing", doing.get(w, ["tense"]), 0.9)

    c = client(world)
    assert [x.window_id for x in _lane(c.get("/").text)] == m  # the newest stream by default, in time order
    assert [x.window_id for x in _lane(c.get("/?doing=laughing").text)] == m
    assert [x.window_id for x in _lane(c.get(f"/?stream={world.old}").text)] == world.old_moments
    assert "Moments match." not in c.get("/?doing=sad").text
    assert c.get("/?doing=bogus").status_code == 400
    assert c.get("/?stream=nope").status_code == 404


# --- a Moment --------------------------------------------------------------------------------------


def test_below_threshold_shows_not_sure(world):
    stub_model(world.store, world.models, RUNG0_ID, 0, thresholds={"doing": 0.65})
    a = world.new_moments[0]
    predict(world.store, a, RUNG0_ID, "doing", ["laughing"], 0.52)
    predict(world.store, a, RUNG0_ID, "aimed_at", "guest", 0.88)

    page = client(world).get(f"/m/{a}").text
    assert _rows(_answer(page, 0)) == [("not sure", "under threshold .52 / .65")]
    assert _rows(_answer(page, 1)) == [("guest", ".88")]
    assert "1 of 4 answered" in _section(page, "Answers")


def test_expired_window_shows_expired_text_and_no_lines(world):
    stub_model(world.store, world.models, RUNG0_ID, 0)
    w = world.old_moments[0]
    predict(world.store, w, RUNG0_ID, "doing", ["laughing"], 0.9)
    c = client(world, now=NOW + timedelta(days=5))  # page start runs expiry
    assert world.store.window(w).text_state == "expired"

    page = c.get(f"/m/{w}").text
    assert "Chat text expired" in _chat(page) and "Labels and counts remain." in _chat(page)
    assert '<li class="sd-ln' not in page and 'class="sd-k"' not in page
    assert '<summary class="evt">' not in page  # no evidence lines either
    assert "text expired" in _section(page, "Answers")
    # Task 5 (F1, A8): the face line outlives the text by design. It holds a counted emote name (not a
    # chat line), which can equal an emote-only line's text, so it is set aside before the text check.
    no_face = re.sub(r'<p class="fc-src">.*?</p>', "", page, flags=re.S)
    for t in _message_texts(world, w):
        assert t not in no_face
    assert _rows(_answer(page, 0)) == [("laughing", ".90")]  # the answer outlives the text
    assert "Chat text kept until" not in _chat(page)  # Task 4 dropped the header's "text expired" badge; the chat block says it


def test_empty_window_shows_no_chat(world):
    stub_model(world.store, world.models, RUNG0_ID, 0)
    stub_model(world.store, world.models, RUNG1_ID, 1, rep=report(0.95, 0.9), at=1)
    quiet = [w for w in world.store.window_ids() if counts(world.store, w)["n_messages"] <= 1]
    assert {counts(world.store, w)["n_messages"] for w in quiet} == {0, 1}
    c = client(world)
    for w in quiet:
        for q, v in (("doing", ["chatting_as_usual"]), ("aimed_at", "chat_itself"), ("feeling", "neutral"),
                     ("tone", "no_clear_tone")):  # fmt: skip
            predict(world.store, w, RUNG1_ID, q, v, 0.97)  # rung 1's intercept-only answer
        page = c.get(f"/m/{w}").text
        assert "No chat in this window." in _chat(page)
        assert '<li class="sd-ln' not in page
        assert "0 of 4 answered" in page
        assert [_rows(_answer(page, i)) for i in range(4)] == [[("not sure", "no chat in this window")]] * 4


def test_needs_review_fires_on_listed_emote(world):
    stub_model(world.store, world.models, RUNG0_ID, 0)
    m = world.new_moments
    use = {w: set(counts(world.store, w)["emote_counts"]) for w in m}
    emote = min({e for s in use.values() for e in s}, key=lambda e: (sum(e in s for s in use.values()), e))
    hit = [w for w in m if emote in use[w]]
    miss = [w for w in m if emote not in use[w]]
    assert hit and miss
    for w in m:
        predict(world.store, w, RUNG0_ID, "doing", ["laughing"], 0.9)

    c = client(world, review=[emote])
    for w in hit:  # Task 4 dropped the header badge: the answer row says it, and the number is a ring (G-4)
        page = c.get(f"/m/{w}").text
        assert 'data-state="review">needs review' not in page
        assert _rows(_answer(page, 0)) == [("not sure", "needs review")]
        assert _badge(page)[0] == "ns"
    page = c.get(f"/m/{miss[0]}").text
    assert _badge(page)[0] == "sure" and _rows(_answer(page, 0)) == [("laughing", ".90")]
    listing = c.get("/").text  # the glance counts a needs-review Moment as not sure, and its mark is a ring
    assert f"Not sure about {len(hit)} of {len(m)} Moments." in text(listing.split('<p class="gl-status">')[1].split("</p>")[0])
    lane = {x.window_id: x for x in _lane(listing)}
    assert [w for w in m if not lane[w].sure] == hit

    plain = client(world)  # nothing listed: nothing needs review
    assert _rows(_answer(plain.get(f"/m/{hit[0]}").text, 0)) == [("laughing", ".90")]


# --- fix -------------------------------------------------------------------------------------------


def test_fix_writes_corrected_on_page_label(world):
    a = world.new_moments[0]
    c = client(world)

    r = c.post(f"/m/{a}/fix", data={"question": "doing", "value": ["shocked", "laughing"]}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == f"/m/{a}"
    r = c.post(f"/m/{a}/fix", data={"question": "tone", "not_sure": ["unknown_in_joke", "conflicting"]},
               follow_redirects=False)  # fmt: skip
    assert r.status_code == 303
    rows = label_rows(world.store)
    assert [(x["window_id"], x["project"], x["origin"], x["provenance"]) for x in rows] == [
        (a, "page", "aron", "corrected-on-page")
    ] * 2
    assert {x["labels_version"] for x in rows} == {SCHEME.version}
    assert {x["guide_version"] for x in rows} == {SCHEME.guide_version}
    assert json.loads(rows[0]["answers_json"]) == {"doing": ["laughing", "shocked"]}  # scheme order
    assert json.loads(rows[0]["not_sure_json"]) == {}
    assert json.loads(rows[1]["answers_json"]) == {}
    assert json.loads(rows[1]["not_sure_json"]) == {"tone": ["conflicting", "unknown_in_joke"]}

    page = c.get(f"/m/{a}").text
    assert _rows(_answer(page, 0)) == [("laughing", ""), ("shocked", "")]
    assert _rows(_answer(page, 3)) == [("not sure · conflicting, unknown in-joke", "")]
    assert '<span class="src">your label</span>' in _answer(page, 0)
    assert '<span class="sd-tag sd-you">your label</span>' in _header(page)  # Task 4's tag replaces the "fixed" badge

    # refusals write nothing
    for bad in ({"question": "doing", "value": ["bogus"]}, {"question": "aimed_at", "value": ["guest", "streamer"]},
                {"question": "nope", "value": ["x"]}, {"question": "doing"},
                {"question": "doing", "value": ["sad"], "not_sure": ["unclear"]}):  # fmt: skip
        assert c.post(f"/m/{a}/fix", data=bad, follow_redirects=False).status_code == 400, bad
    assert c.post("/m/nope/fix", data={"question": "doing", "value": ["sad"]}).status_code == 404
    stale = client(world, now=NOW + timedelta(days=8))  # the deletion check is more than 7 days old
    r = stale.post(f"/m/{world.new_moments[1]}/fix", data={"question": "doing", "value": ["sad"]},
                   follow_redirects=False)  # fmt: skip
    assert r.status_code == 409 and "deletion check" in r.text
    assert len(label_rows(world.store)) == 2


def test_fix_same_value_saves_aron_label(world):
    stub_model(world.store, world.models, RUNG0_ID, 0)
    a = world.new_moments[0]
    predict(world.store, a, RUNG0_ID, "aimed_at", "guest", 0.9)
    c = client(world)
    assert "made-up labels" in c.get(f"/m/{a}").text

    r = c.post(f"/m/{a}/fix", data={"question": "aimed_at", "value": "guest"}, follow_redirects=False)
    assert r.status_code == 303
    (row,) = label_rows(world.store)
    assert (row["origin"], row["provenance"], row["project"]) == ("aron", "corrected-on-page", "page")
    assert json.loads(row["answers_json"]) == {"aimed_at": "guest"}
    aimed = _answer(c.get(f"/m/{a}").text, 1)
    assert '<span class="src">your label</span>' in aimed and _rows(aimed) == [("guest", "")]


# --- cause -----------------------------------------------------------------------------------------


def test_cause_note_written_and_shown(world):
    a = world.new_moments[0]
    label(world.store, a, {"doing": ["cheering"]})
    c = client(world)
    note = "The guest arrived on camera <b>"
    r = c.post(f"/m/{a}/cause", data={"cause_note": f"  {note}  ", "cause_source": "clip"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == f"/m/{a}"

    row = label_rows(world.store)[-1]
    assert (row["project"], row["origin"], row["provenance"]) == ("page", "aron", "corrected-on-page")
    assert (row["cause_note"], row["cause_source"]) == (note, "clip")
    assert json.loads(row["answers_json"]) == {} and json.loads(row["not_sure_json"]) == {}

    page = c.get(f"/m/{a}").text
    cause = _section(page, "Cause")
    assert "The guest arrived on camera &lt;b&gt;" in cause and "clip note" in cause
    assert "Cause unknown" not in cause
    assert "cheering" in _section(page, "Answers")  # a cause-only row leaves the answers alone

    for bad in ({"cause_note": "x" * 301, "cause_source": "clip"}, {"cause_note": "   ", "cause_source": "clip"},
                {"cause_note": "ok", "cause_source": "chat"}, {"cause_note": "ok"}):  # fmt: skip
        assert c.post(f"/m/{a}/cause", data=bad, follow_redirects=False).status_code == 400, bad
    assert c.post(f"/m/{a}/cause", data={"cause_note": "x" * 300, "cause_source": "game"},
                  follow_redirects=False).status_code == 303  # fmt: skip


def test_cause_unknown_without_note(world):
    a = world.new_moments[0]
    page = client(world).get(f"/m/{a}").text
    cause = _section(page, "Cause")
    assert "Cause unknown · no cause note" in cause
    assert f'action="/m/{a}/cause"' in cause and "Add cause note" in cause
    for source in SCHEME.cause_sources:
        assert f'value="{source}"' in cause


# --- expiry between runs ---------------------------------------------------------------------------


def test_no_text_past_expiry(world):
    now = [NOW]
    c = client(world, clock=lambda: now[0])
    w = world.old_moments[0]
    expires = world.store.window(w).text_expires_at
    texts = _message_texts(world, w)
    page = c.get(f"/m/{w}").text
    assert sum(t in page for t in texts) > 0 and '<li class="sd-ln' in page  # text before the time is up

    now[0] = expires  # no hourly run since page start
    assert world.store.messages(w)  # the rows are still in the store
    for url in (f"/m/{w}", f"/m/{w}?panel=1", f"/?stream={world.old}"):
        page = c.get(url).text
        assert not [t for t in texts if t in page], url
        assert '<li class="sd-ln' not in page
    assert "Chat text expired" in c.get(f"/m/{w}").text
    assert c.post(f"/m/{w}/fix", data={"question": "doing", "value": ["sad"]}, follow_redirects=False).status_code == 303


# --- the glance (Task 3) ------------------------------------------------------------------------------

NO_COUNTS = "This stream's export has no activity counts."
EXPIRED_COUNTS = "Activity counts expired 30 days after this stream started."


def _lane(page: str) -> list[SimpleNamespace]:
    """The glance lane's marks, in order: window ID, number, chip (top three) or mark, sure, aria-label."""
    lane = page.split('<div class="gl-lane"')[1].split("</div>")[0]
    found = re.findall(
        r'<a class="gl-(c3|mk)( gl-ns)?" href="/m/([^"]+)" data-open data-n="(\d+)" data-x="[\d.]+" style="[^"]*" aria-label="([^"]+)"', lane
    )
    assert len(found) == lane.count("<a "), "a mark the pattern missed"
    return [SimpleNamespace(top=k == "c3", sure=not ns, window_id=w, n=n, label=lb) for k, ns, w, n, lb in found]


def _cards(page: str) -> list[str]:
    return re.findall(r'<li class="gl-card" data-window="([^"]+)"', page)


def _card(page: str, window_id: str) -> str:
    return page.split(f'<li class="gl-card" data-window="{window_id}"')[1].split("</li>")[0]


def _stat(page: str, window_id: str) -> str | None:
    found = re.findall(r'<span class="gl-size">(.*?)</span>\s*(?=<span class="gl-chat|</a>)', _card(page, window_id), re.S)
    assert len(found) <= 1
    return text(found[0]) if found else None


def _plot(page: str) -> str:
    return page.split('<figure class="gl-graph">')[1].split("</figure>")[0]


def _gone(page: str) -> str | None:
    found = re.findall(r'<p class="gl-gone" id="gone">(.*?)</p>', page, re.S)
    return text(found[0]) if found else None


def test_lane_marks_every_moment_in_time_order(world):
    m = world.new_moments
    s = world.store
    label(s, m[0], {"doing": ["laughing"]}, cause_note="A clip shows the guest arriving", cause_source="clip")
    label(s, m[1], {"aimed_at": "streamer"}, not_sure={"doing": ["conflicting"]})
    label(s, m[2], {"doing": ["cheering"]})
    page = client(world).get("/").text
    found = _lane(page)
    assert [x.window_id for x in found] == m
    assert [x.n for x in found] == [str(i) for i in range(1, len(m) + 1)]
    assert [x.sure for x in found[:4]] == [True, False, True, False]  # m[3]: no label, no model
    assert sum(x.top for x in found) == 3
    for x in found:  # each mark says its number and time, and "not sure" when the doing answer is
        when = app_mod.clock_text(world.windows[x.window_id].anchor_at)
        assert x.label == f"Moment {x.n}, {when}" + ("" if x.sure else ", not sure")
    by_id = {x.window_id: x for x in found}
    for w in _cards(page):  # a card carries its mark's number, and a ring when not sure
        num = re.search(r'<span class="gl-num( gl-ns)?">(\d+)</span>', _card(page, w))
        assert num and num.group(2) == by_id[w].n and (not num.group(1)) == by_id[w].sure


def test_home_draws_activity_line(world):
    m = world.new_moments
    clear_activity(world.store)
    a = activity_line(world, world.new, peaks={m[0]: 60, m[-1]: 30}, gaps=((0, 6), (20, 9)))
    assert store_activity(world.store, a) == "written"
    drawn = activity_mod.line(world.store.stream_activity(world.new, now=NOW), height=app_mod.GL_H)
    assert len(drawn.gaps) == 2

    page = client(world).get(f"/?stream={world.new}").text
    plot = _plot(page)
    assert re.search(rf'<svg class="gl-cv" viewBox="[-\d.]+ 0 [\d.]+ {app_mod.GL_H}" preserveAspectRatio="none"', plot)
    assert "xmlns" not in page  # an inline svg needs none; a URL in it would break the no-outbound check
    assert re.findall(r'<path class="gl-area" d="([^"]+)"', plot) == [drawn.d_area]
    assert re.findall(r'<path class="gl-ln" d="([^"]+)" vector-effect="non-scaling-stroke"', plot) == [drawn.d_line]
    assert _gap_titles(plot) == ["No chat recorded"] * len(drawn.gaps)
    assert "Height: people chatting per 10 s" in text(plot)
    assert NO_COUNTS not in text(page) and EXPIRED_COUNTS not in text(page) and _gone(page) is None


def test_home_without_activity_falls_back(tmp_path):
    world = build_world(tmp_path, activity=False)
    try:
        page = client(world).get("/").text
        assert 'class="gl-graph"' not in page and 'class="gl-ln"' not in page and 'class="gl-lane"' not in page
        assert _gone(page) == NO_COUNTS and EXPIRED_COUNTS not in page
        assert "Height: people chatting" not in page
        assert _cards(page) == world.new_moments[:3]  # the first three by time
        for w in _cards(page):
            assert _stat(page, w) is None and f'href="/m/{w}"' in _card(page, w)
    finally:
        world.store.close()


def _assert_expired_glance(page: str, world) -> None:
    """No curve, gap, lane or card stat from expired counts; the approved line; the first three kept."""
    assert 'class="gl-ln"' not in page and 'class="gl-gap"' not in page and 'class="gl-lane"' not in page
    assert _gone(page) == EXPIRED_COUNTS and NO_COUNTS not in text(page)
    assert "Height: people chatting" not in page
    assert _cards(page) == world.new_moments[:3]
    for w in _cards(page):
        assert _stat(page, w) is None and f'href="/m/{w}"' in _card(page, w)


def _activity_rows(store) -> tuple[int, int]:
    rows = store.conn.execute("SELECT COUNT(*) FROM stream_activity").fetchone()[0]
    marks = store.conn.execute("SELECT COUNT(*) FROM stream_activity_expired").fetchone()[0]
    return rows, marks


def test_home_with_expired_activity_uses_q10_copy(tmp_path):
    # 1. before the sweep: the page clock passes the deadline with no hourly run since page start
    (tmp_path / "q10").mkdir()
    world = build_world(tmp_path / "q10")
    clear_activity(world.store)
    src = next(s for s in world.export.streams if s.stream_id == world.new)
    early = src.started_at - (NOW - timedelta(days=29, hours=12))  # the counts expire at NOW + 12 h
    a = activity_line(world, world.new, peaks={world.new_moments[0]: 81}, shift=early)
    assert store_activity(world.store, a) == "written"
    now = [NOW]
    c = client(world, clock=lambda: now[0])
    assert 'class="gl-ln"' in c.get(f"/?stream={world.new}").text  # live before its deadline
    now[0] = NOW + timedelta(days=1)
    assert _activity_rows(world.store) == (1, 0)  # the row is still stored and no marker is written
    _assert_expired_glance(c.get(f"/?stream={world.new}").text, world)

    # 2. after the sweep: page start at the later clock deletes the counts and writes the marker
    c = client(world, now=NOW + timedelta(days=1))
    assert _activity_rows(world.store) == (0, 1)
    _assert_expired_glance(c.get(f"/?stream={world.new}").text, world)

    # 4. a page clock earlier than the sweep's: the marker still wins, so it never reads as "missing" (E2)
    c = client(world, now=NOW)
    _assert_expired_glance(c.get(f"/?stream={world.new}").text, world)
    world.store.close()

    # 3. an import that met counts already past their 30 days records the expired state (E1)
    raw = fake.make_export(3, 12, 16, NOW, channels=1)
    export = parse_export(raw)
    new = max(export.streams, key=lambda s: s.ended_at).stream_id
    late = timedelta(days=30)
    streams = tuple(
        dataclasses.replace(s, started_at=s.started_at - late, ended_at=s.ended_at - late) if s.stream_id == new else s
        for s in export.streams
    )
    (tmp_path / "e1").mkdir()
    world = build_world(tmp_path / "e1", raw=dump_export(dataclasses.replace(export, streams=streams)))
    try:
        assert world.new == new and world.store.activity_state(new, now=NOW) == "expired"
        _assert_expired_glance(client(world).get(f"/?stream={new}").text, world)
    finally:
        world.store.close()


def test_usual_excludes_expired_siblings(world):
    m = world.new_moments
    clear_activity(world.store)
    new = activity_line(world, world.new, peaks={m[0]: 81})
    assert store_activity(world.store, new) == "written"
    # the old stream: three times as many bins of 50, so counting it would move the median to 50;
    # its counts started 29.5 days before NOW, so they expire at NOW + 12 h
    start = NOW - timedelta(days=29, hours=12)
    bins = 3 * len(new.chatters)
    old = StreamActivity(
        stream_id=world.old, channel=new.channel, started_at=start, ended_at=start + timedelta(seconds=10 * bins),
        title=None, bin_s=10, chatters=(50,) * bins,
    )  # fmt: skip
    assert store_activity(world.store, old) == "written"
    now = [NOW]
    c = client(world, clock=lambda: now[0])
    assert _stat(c.get(f"/?stream={world.new}").text, m[0]) == "81 people in 10 s · 1.6× usual pace"  # both live: 81 / 50

    now[0] = NOW + timedelta(days=1)  # the old counts are past their deadline, not yet swept
    assert _activity_rows(world.store) == (2, 0)
    alone = activity_mod.ratio_text(81 / activity_mod.usual_pace([new]))
    assert alone == "9×"
    assert _stat(c.get(f"/?stream={world.new}").text, m[0]) == f"81 people in 10 s · {alone} usual pace"

    c = client(world, now=NOW + timedelta(days=1))  # swept: the marker keeps them out too
    assert _activity_rows(world.store) == (1, 1)
    assert _stat(c.get(f"/?stream={world.new}").text, m[0]) == f"81 people in 10 s · {alone} usual pace"


def test_cards_show_peak_and_usual(world):
    m = world.new_moments
    clear_activity(world.store)
    # bins of 9 with an 81 on the first Moment: the usual pace is the median, 9 (C4)
    a = activity_line(world, world.new, peaks={m[0]: 81})
    assert store_activity(world.store, a) == "written"
    page = client(world).get(f"/?stream={world.new}").text
    assert _cards(page)[0] == m[0]
    assert _stat(page, m[0]) == "81 people in 10 s · 9× usual pace"
    assert "posters joining" not in page
    for w in _cards(page)[1:]:
        assert re.fullmatch(r"\d+ people in 10 s · \d+(\.\d)?× usual pace", _stat(page, w) or ""), w

    # under 30 bins above zero: no usual pace, so the card shows the peak alone, and zero is a count
    quiet = activity_line(world, world.new, base=0, peaks={m[0]: 81})
    assert store_activity(world.store, quiet) == "written"
    page = client(world).get(f"/?stream={world.new}").text
    assert _cards(page) == m[:3]
    assert _stat(page, m[0]) == "81 people in 10 s" and _stat(page, m[1]) == "0 people in 10 s"


def test_card_in_gap_says_not_recorded(world):
    m = world.new_moments
    clear_activity(world.store)
    src = next(s for s in world.export.streams if s.stream_id == world.new)
    at = {w: int((world.windows[w].anchor_at - src.started_at).total_seconds()) // 10 for w in m}
    keep = m[3]
    # a gap over every other anchor that leaves part of [anchor - 30 s, anchor + 90 s) recorded (C1)
    gaps = tuple((at[w] - 1, 4) for w in m if w != keep)
    assert all(first >= 0 and not first <= at[keep] < first + 4 for first, _ in gaps)
    a = activity_line(world, world.new, peaks={keep: 40}, gaps=gaps)
    assert store_activity(world.store, a) == "written"
    page = client(world).get(f"/?stream={world.new}").text
    assert _cards(page) == [keep, m[0], m[1]]  # a known peak first, then the unknown ones by time (G-3)
    assert _stat(page, m[0]) == "not recorded" and _stat(page, m[1]) == "not recorded"
    assert (_stat(page, keep) or "").startswith("40 people in 10 s")
    assert m[0] in [x.window_id for x in _lane(page)]  # its mark still shows
    assert "coverage unknown" not in page


# --- the peek ---------------------------------------------------------------------------------------

def _badge(page: str) -> tuple[str, str] | None:
    """(shape, number) of the number that starts the Moment's header, or None without one.

    Task 4 (G-4): the shape is "sure" (a solid number, Doing is sure) or "ns" (a ring, not sure)."""
    found = re.match(r'\s*<p class="sd-top"><span class="sd-num( sd-ns)?" data-lift>(\d+)</span>', _header(page))
    return ("ns" if found.group(1) else "sure", found.group(2)) if found else None


def _slot(page: str) -> str:
    found = re.search(r'<div class="sd-slot">(.*)</div>\s*$', _header(page), re.S)
    assert found, "no stat slot"
    return found.group(1)


def _peak_line(page: str) -> str | None:
    """The visible text of the header's stat slot (the stats joined with " · "), or None when it is empty."""
    slot = _slot(page)
    stats = re.findall(r'<p class="sd-stat">(.*?)</p>', slot, re.S)
    if stats:
        return " · ".join(text(s) for s in stats)
    found = re.findall(r"<p[^>]*>(.*?)</p>", slot, re.S)
    assert len(found) <= 1, found
    return text(found[0]) if found else None


def test_peek_number_matches_row(world):
    m = world.new_moments
    s = world.store
    label(s, m[0], {"doing": ["laughing"]}, cause_note="A clip shows the guest arriving", cause_source="clip")
    label(s, m[1], {"doing": ["complaining"]})
    label(s, m[2], {"doing": ["complaining", "laughing"]})
    c = client(world)
    rows = {x.window_id: x.n for x in _lane(c.get(f"/?stream={world.new}").text)}  # the glance lane's numbers
    assert sorted(int(n) for n in rows.values()) == list(range(1, len(m) + 1))
    shapes = [_badge(c.get(f"/m/{w}").text)[0] for w in m[:4]]
    assert shapes == ["sure", "sure", "sure", "ns"]  # G-4: solid when Doing is sure, a ring when not
    for w in m:  # a doing filter never changes a number, even for a Moment the filter leaves out (C7)
        for url in (f"/m/{w}?doing=laughing", f"/m/{w}?doing=laughing&panel=1", f"/m/{w}"):
            assert _badge(c.get(url).text)[1] == rows[w], url
    old = {x.window_id: x.n for x in _lane(c.get(f"/?stream={world.old}").text)}  # each stream numbers from 1
    for w in world.old_moments:
        assert _badge(c.get(f"/m/{w}?panel=1").text)[1] == old[w], w
    # an ordinary window is no Moment, so it has no number
    ordinary = next(w.window_id for w in world.export.windows if w.anchor_kind == "ordinary")
    page = c.get(f"/m/{ordinary}").text
    assert _badge(page) is None and "sd-num" not in _header(page)
    assert _header(page).lstrip().startswith('<p class="sd-top"><span class="sd-time">')


def test_peek_peak_line(world, tmp_path):
    m = world.new_moments
    clear_activity(world.store)
    # bins of 9 with an 81 on the first Moment: the usual pace is the median, 9 (C4)
    a = activity_line(world, world.new, peaks={m[0]: 81})
    assert store_activity(world.store, a) == "written"
    c = client(world)
    for url in (f"/m/{m[0]}", f"/m/{m[0]}?panel=1"):
        assert _peak_line(c.get(url).text) == "81 people chatting in the busiest 10 s · 9× the usual pace", url
    home = c.get(f"/?stream={world.new}").text
    assert _stat(home, m[0]) == "81 people in 10 s · 9× usual pace"  # the card says the same
    on_cards = _cards(home)
    for w in m[1:]:
        line = _peak_line(c.get(f"/m/{w}").text)
        found = re.fullmatch(r"(\d+) people chatting in the busiest 10 s · (\d+(?:\.\d)?×) the usual pace", line or "")
        assert found, (w, line)
        if w in on_cards:  # the card's peak and ratio
            assert _stat(home, w) == f"{found.group(1)} people in 10 s · {found.group(2)} usual pace"
    assert "not recorded" not in _header(c.get(f"/m/{m[0]}").text)

    # under 30 bins above zero: no usual pace, so the line has no ratio part
    quiet = activity_line(world, world.new, base=0, peaks={m[0]: 81})
    assert store_activity(world.store, quiet) == "written"
    assert _peak_line(c.get(f"/m/{m[0]}").text) == "81 people chatting in the busiest 10 s"
    assert _peak_line(c.get(f"/m/{m[-1]}?panel=1").text) == "0 people chatting in the busiest 10 s"  # zero is a count

    # an export with no activity counts: the Q10 line in the stat slot and in Counted (Task 4), and the number stays
    (tmp_path / "v1").mkdir()
    v1 = build_world(tmp_path / "v1", activity=False)
    try:
        page = client(v1).get(f"/m/{v1.new_moments[0]}").text
        assert _peak_line(page) == NO_COUNTS and NO_COUNTS in text(_section(page, "Counted"))
        assert "people chatting" not in page and "not recorded" not in page
        assert _badge(page) is not None and _badge(page)[1] == "1"
        assert EXPIRED_COUNTS not in page
    finally:
        v1.store.close()


def test_peek_in_gap_says_not_recorded(world):
    m = world.new_moments
    clear_activity(world.store)
    src = next(s for s in world.export.streams if s.stream_id == world.new)
    at = int((world.windows[m[0]].anchor_at - src.started_at).total_seconds()) // 10
    # a gap over the anchor that leaves part of [anchor - 30 s, anchor + 90 s) recorded (C1)
    a = activity_line(world, world.new, peaks={m[1]: 40}, gaps=((at - 1, 4),))
    assert store_activity(world.store, a) == "written"
    c = client(world)
    for url in (f"/m/{m[0]}", f"/m/{m[0]}?panel=1"):
        page = c.get(url).text
        assert _peak_line(page) == "Chat not recorded at this Moment.", url
        assert "people chatting" not in page
        assert _badge(page) == ("ns", "1")
    assert _peak_line(c.get(f"/m/{m[1]}").text).startswith("40 people chatting in the busiest 10 s · ")


def test_peek_expired_stream_shows_no_peak(world):
    m = world.new_moments
    clear_activity(world.store)
    a = activity_line(world, world.new, peaks={m[0]: 81})
    assert store_activity(world.store, a) == "written"
    deadline = a.started_at + timedelta(days=30)
    before = deadline - timedelta(minutes=1)
    now = [before]
    c = client(world, clock=lambda: now[0])
    n = {x.window_id: x.n for x in _lane(c.get(f"/?stream={world.new}").text)}[m[0]]
    live = c.get(f"/m/{m[0]}?panel=1").text
    badge = _badge(live)
    assert badge[1] == n
    assert _peak_line(live) == "81 people chatting in the busiest 10 s · 9× the usual pace"  # live before its deadline
    assert EXPIRED_COUNTS not in live

    def expired(page: str) -> None:
        """The Q10 expired line in the stat slot and in Counted, never the no-peak copy; the number stays (Task 4)."""
        assert _peak_line(page) == EXPIRED_COUNTS and EXPIRED_COUNTS in _section(page, "Counted")
        assert 'class="sd-stat"' not in page and "people chatting" not in page and "not recorded" not in page
        assert NO_COUNTS not in text(page)
        assert "Labels and counts remain." not in _chat(page)  # the text-expired line's "counts gone" variant
        assert _badge(page) == badge

    # before the sweep: the page clock reaches the deadline with no hourly run since page start
    now[0] = deadline
    assert _activity_rows(world.store) == (1, 0)
    for url in (f"/m/{m[0]}?panel=1", f"/m/{m[0]}"):
        expired(c.get(url).text)
    unswept = c.get(f"/m/{m[0]}?panel=1").text

    # after the sweep: page start at that clock deletes the counts and writes the marker
    c = client(world, now=deadline)
    assert _activity_rows(world.store) == (0, 1)
    for url in (f"/m/{m[0]}?panel=1", f"/m/{m[0]}"):
        expired(c.get(url).text)
    assert c.get(f"/m/{m[0]}?panel=1").text == unswept  # the sweep changes nothing the panel shows

    # a page clock earlier than the sweep's: the marker still wins (E2)
    c = client(world, now=before)
    expired(c.get(f"/m/{m[0]}").text)


# --- look and shell --------------------------------------------------------------------------------

GEIST = {
    "geist-latin-wght-normal.woff2": "19f9c92546aa300c312235e3125af1b81394d8db9a4bc4a425cd5b641d2d54e1",
    "geist-mono-latin-wght-normal.woff2": "684ad5b531f81d43c1e8c7038262d5db7cdc1f68006e04d6c7769efa8d33c8cc",
}


def test_fonts_are_geist_and_local():
    css = (STATIC / "page.css").read_text(encoding="utf-8")
    urls = re.findall(r"url\((/static/fonts/[^)]+)\)", css)
    assert sorted(set(urls)) == sorted(f"/static/fonts/{name}" for name in GEIST)
    fonts = STATIC / "fonts"
    assert sorted(f.name for f in fonts.glob("*.woff2")) == sorted(GEIST)
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    decisions = (root / "docs" / "decisions.md").read_text(encoding="utf-8")
    for name, pinned in GEIST.items():
        digest = hashlib.sha256((fonts / name).read_bytes()).hexdigest()
        assert digest == pinned, name
        assert f"`{name}` {digest}" in decisions, name
    for licence in ("OFL-geist.txt", "OFL-geist-mono.txt"):
        assert "SIL OPEN FONT LICENSE" in (fonts / licence).read_text(encoding="utf-8").upper(), licence
    families = re.findall(r'font-family:\s*"([^"]+)"', css)
    assert sorted(set(families)) == ["Geist", "Geist Mono"]
    assert "Rubik" not in css and "Roboto" not in css
    for doc in (readme, decisions):
        assert "Rubik" not in doc and "Roboto" not in doc


# --- final review fixes -------------------------------------------------------------------------------


def _glance_data(world, stream_id: str, *, now=NOW) -> tuple[dict, dict]:
    """The home handler's data for ``stream_id`` at ``now``, without its markup: ``({window_id:
    Moment}, the activity line's dict)``. Built like ``home`` does, on a fresh ``Page`` over the store."""
    page = app_mod.Page(
        world.store, SCHEME, world.models, imported_dir=world.tmp / "imported", clock=lambda: now,
        review_ids=frozenset(),
    )  # fmt: skip
    stream = next(s for s in page.streams(now) if s.stream_id == stream_id)
    p = page.plan()
    ms = page.moments(page.stream_moments(stream_id), now, p)
    state, act = page.counts(stream_id, now)
    app_mod._number(ms, [m.win for m in ms], act)
    return {m.win.window_id: m for m in ms}, app_mod._activity_line(stream, ms, act, activity_state=state)


def _gap_titles(block: str) -> list[str]:
    return re.findall(r'<rect class="gl-gap"[^>]*><title>([^<]+)</title></rect>', block)


def _gap_labels(page: str) -> list[str]:
    """The words printed over the hatched stretches (the glance has no legend: each gap says itself)."""
    return re.findall(r'<span class="gl-gapl(?: gl-[lr])?"[^>]*>([^<]+)</span>', page)


def test_outside_counts_reads_coverage_unknown(world):
    """Axis the counts don't cover is unknown, never a "No chat recorded" gap (R1, G-1)."""
    m = world.new_moments
    clear_activity(world.store)
    # the counts end 5 h early, so the later Moments are anchored after their last bin
    a = activity_line(world, world.new, shift=timedelta(hours=5))
    assert store_activity(world.store, a) == "written"
    assert any(world.windows[w].anchor_at > a.ended_at for w in m) and any(world.windows[w].anchor_at < a.ended_at for w in m)
    _, strip = _glance_data(world, world.new)
    assert [kind for _, _, kind in strip["gaps_pct"]] == ["unknown"]
    page = client(world).get(f"/?stream={world.new}").text
    block = _plot(page)
    lane = page.split('<div class="gl-lane"')[1].split("</div>")[0]
    xs = [float(x) for x in re.findall(r'style="left:([\d.]+)%', lane)]
    assert len(xs) == len(m)
    assert xs == sorted(xs) and len(set(xs)) == len(xs), xs  # none piled on one edge
    assert xs[-1] == 100.0 and xs[0] > 0
    axis_s = (max(world.windows[w].anchor_at for w in m) - a.started_at).total_seconds()
    width_s = len(a.chatters) * a.bin_s
    assert re.search(rf'viewBox="0 0 {axis_s:g} [\d.]+"', block)
    # the part the counts don't cover is the hatched "coverage unknown" stretch, to the axis end
    gaps = re.findall(r'<rect class="gl-gap" x="([\d.]+)%" y="0" width="([\d.]+)%"', block)
    assert len(gaps) == 1
    left, width = map(float, gaps[0])
    assert abs(left - width_s / axis_s * 100) < 0.01 and abs(left + width - 100) < 0.01
    assert _gap_titles(block) == ["Coverage unknown"] and _gap_labels(page) == ["Coverage unknown"]
    assert "No chat recorded" not in block
    # a Moment inside the counts sits where the counts put it: its chip is left of the stretch
    assert xs[0] < left

    # anchored before the counts began: the unknown stretch leads, and the axis starts at the first anchor
    early = activity_line(world, world.new, shift=timedelta(hours=-5))
    assert store_activity(world.store, early) == "written"
    _, strip = _glance_data(world, world.new)
    assert [kind for _, _, kind in strip["gaps_pct"]] == ["unknown"]
    page = client(world).get(f"/?stream={world.new}").text
    block = _plot(page)
    lead_s = (early.started_at - min(world.windows[w].anchor_at for w in m)).total_seconds()
    assert lead_s > 0
    vx = re.search(r'viewBox="(-[\d.]+) 0 ', block)
    assert vx and abs(float(vx.group(1)) + lead_s) < 0.1
    gaps = re.findall(r'<rect class="gl-gap" x="([\d.]+)%" y="0" width="([\d.]+)%"', block)
    assert gaps and float(gaps[0][0]) == 0.0
    assert _gap_titles(block) == ["Coverage unknown"]


def test_null_bins_read_no_chat_recorded(world):
    """A None run inside the counts' span is a known gap (G-1): "No chat recorded", never unknown."""
    clear_activity(world.store)
    a = activity_line(world, world.new, gaps=((0, 6), (20, 9)))
    assert store_activity(world.store, a) == "written"
    _, strip = _glance_data(world, world.new)
    assert [kind for _, _, kind in strip["gaps_pct"]] == ["gap", "gap"]
    page = client(world).get(f"/?stream={world.new}").text
    assert _gap_titles(_plot(page)) == ["No chat recorded", "No chat recorded"]
    assert _gap_labels(page) == ["No chat recorded", "No chat recorded"]
    assert "Coverage unknown" not in page

    # a null run at the counts' end that meets the uncovered stretch: two kinds side by side, never merged
    n = len(a.chatters)
    both = activity_line(world, world.new, gaps=((n - 6, 6),), shift=timedelta(hours=5))
    assert store_activity(world.store, both) == "written"
    _, strip = _glance_data(world, world.new)
    kinds = [kind for _, _, kind in strip["gaps_pct"]]
    assert kinds == ["gap", "unknown"], kinds
    (gl, gw, _), (ul, _, _) = strip["gaps_pct"]
    assert abs(gl + gw - ul) < 0.01  # they touch
    page = client(world).get(f"/?stream={world.new}").text
    assert _gap_titles(_plot(page)) == ["No chat recorded", "Coverage unknown"]
    assert _gap_labels(page) == ["No chat recorded", "Coverage unknown"]


def test_card_and_panel_unknown_copy(world):
    """A Moment outside the counts reads "coverage unknown"; one on a null bin keeps "not recorded"."""
    m = world.new_moments
    clear_activity(world.store)
    shift = timedelta(hours=5)
    base = activity_line(world, world.new, shift=shift)
    end = base.ended_at
    # wholly outside: nothing in [anchor - 30 s, anchor + 90 s) is covered, so no number either
    outside = next(w for w in m if world.windows[w].anchor_at >= end + timedelta(seconds=30))
    # inside, with room for its peak span: a null run over its anchor bin
    inside = next(w for w in m if world.windows[w].anchor_at < end - timedelta(seconds=120))
    at = int((world.windows[inside].anchor_at - base.started_at).total_seconds()) // base.bin_s
    a = activity_line(world, world.new, gaps=((at - 1, 4),), shift=shift)
    assert store_activity(world.store, a) == "written"
    assert activity_mod.coverage(a, world.windows[outside].anchor_at) == "unknown"
    assert activity_mod.coverage(a, world.windows[inside].anchor_at) == "gap"

    ms, _ = _glance_data(world, world.new)
    assert (ms[outside].peak, ms[outside].peak_known, ms[outside].peak_why) == (None, False, "unknown")
    assert (ms[inside].peak, ms[inside].peak_known, ms[inside].peak_why) == (None, False, "gap")
    known = [w for w, x in ms.items() if x.peak is not None]
    assert known and all(ms[w].peak_why == "" and ms[w].peak_known for w in known)

    c = client(world)
    for url in (f"/m/{outside}", f"/m/{outside}?panel=1"):
        page = c.get(url).text
        assert _peak_line(page) == "Coverage unknown at this Moment.", url
        assert "not recorded" not in _header(page)
    for url in (f"/m/{inside}", f"/m/{inside}?panel=1"):
        page = c.get(url).text
        assert _peak_line(page) == "Chat not recorded at this Moment.", url
        assert "Coverage unknown" not in _header(page)

    # the cards say the same. Counts that start between the first two Moments with no chat recorded
    # at all: no Moment has a peak, so the cards are the first three by time (G-3), the first one
    # outside the counts and the second inside a null run.
    src = next(s for s in world.export.streams if s.stream_id == world.new)
    start = world.windows[m[0]].anchor_at + timedelta(seconds=100)
    assert world.windows[m[1]].anchor_at - timedelta(seconds=30) > start
    late = activity_line(world, world.new, base=None, shift=src.started_at - start)
    assert store_activity(world.store, late) == "written"
    assert activity_mod.coverage(late, world.windows[m[0]].anchor_at) == "unknown"
    assert activity_mod.coverage(late, world.windows[m[1]].anchor_at) == "gap"
    home = client(world).get(f"/?stream={world.new}").text
    assert _cards(home) == m[:3]
    assert _stat(home, m[0]) == "coverage unknown" and _stat(home, m[1]) == "not recorded"
    assert "people" not in (_stat(home, m[0]) or "") + (_stat(home, m[1]) or "")


def test_no_counts_peak_why_empty(world, tmp_path):
    """With counts missing, no Moment is a gap or unknown: ``peak_why`` is "" and neither phrase shows."""
    phrases = ("not recorded", "No chat recorded", "Coverage unknown", "coverage unknown")

    def check(w, stream_id, moments) -> None:
        ms, strip = _glance_data(w, stream_id)
        assert all(x.peak is None and x.peak_known and x.peak_why == "" for x in ms.values())
        assert strip["gaps_pct"] == []
        c = client(w)
        pages = [c.get(f"/?stream={stream_id}").text] + [c.get(f"/m/{x}?panel=1").text for x in moments]
        for page in pages:
            assert not any(p in page for p in phrases)

    clear_activity(world.store)  # counts missing (never imported for this test's purpose)
    check(world, world.new, world.new_moments)
    (tmp_path / "v1").mkdir()
    v1 = build_world(tmp_path / "v1", activity=False)  # an export with no activity counts
    try:
        check(v1, v1.new, v1.new_moments)
    finally:
        v1.store.close()


def test_lane_names_say_not_sure(world):
    m = world.new_moments
    label(world.store, m[0], {"doing": ["laughing"]}, cause_note="A clip shows the guest arriving", cause_source="clip")
    label(world.store, m[2], {"doing": ["cheering"]})
    page = client(world).get("/").text
    marks = {x.window_id: x for x in _lane(page)}
    assert marks[m[0]].sure and "not sure" not in marks[m[0]].label
    assert marks[m[1]].label.endswith(", not sure") and marks[m[1]].label.count("not sure") == 1
    assert not marks[m[1]].sure and marks[m[2]].sure and "not sure" not in marks[m[2]].label
    assert "data-tip" not in page.split('<div class="gl-lane"')[1].split("</div>")[0]  # the name is the label


def test_skip_link_is_first_and_targets_the_page(world):
    c = client(world)
    for url in ("/", f"/m/{world.new_moments[0]}", "/m/nonexistent-window"):
        page = c.get(url).text
        body = page.split("<body", 1)[1]
        assert re.search(r'>\s*<a class="skip" href="#main">Skip to content</a>\s*<header class="tb gl-top"', body), url
        assert page.count('id="main"') == 1 and '<main class="main" id="main"' in page
    css = (STATIC / "page.css").read_text(encoding="utf-8")
    assert ".skip:not(:focus)" in css and ".skip:focus{" in css


def test_lane_focus_and_selection_rise_above_neighbours():
    css = (STATIC / "page.css").read_text(encoding="utf-8")
    assert re.search(r'\.gl-mk:is\(:hover,:focus-visible,\[aria-current="true"\]\)\{z-index:3;', css)
    assert re.search(r"\.gl-c3\{position:absolute;top:18px;z-index:2;", css)


def test_focus_clears_the_sticky_bar_and_dock(world):
    css = (STATIC / "page.css").read_text(encoding="utf-8")
    rule = re.search(r"html\{(scroll-padding-top:calc\(56px[^}]*scroll-padding-bottom:calc\(var\(--cb-dock-h[^}]*)\}", css)
    assert rule
    # The --cb-* tokens are declared on .cb (body); on html every var() needs a
    # fallback, or the computed value is invalid and scroll-padding drops to auto.
    assert re.search(r"^\.cb\s*\{", css, re.M)
    for name, fallback in re.findall(r"var\((--cb-[\w-]+)(,[^)]+)?\)", rule.group(1)):
        assert fallback, f"{name} has no fallback on html"
    js = (STATIC / "page.js").read_text(encoding="utf-8")
    assert "--cb-dock-h" in js and "ResizeObserver(clearDock)" in js


def test_media_queries_leave_no_fractional_gap():
    css = (STATIC / "page.css").read_text(encoding="utf-8")
    js = (STATIC / "page.js").read_text(encoding="utf-8")
    for source in (css, js):
        assert not re.search(r"max-width:\s*(759|1279)px", source)
        assert re.search(r"max-width:\s*759\.98px", source)
    assert "max-width:1279.98px" in css  # the side panel narrows below 1280 px
    # the glance is one centred column at every width (Task 3): no wide-screen lane breakpoint is left
    assert "min-width:1280px" not in css


def test_phone_sheet_makes_the_page_behind_inert():
    js = (STATIC / "page.js").read_text(encoding="utf-8")
    assert "function streamMenu()" in js
    sheet = js.split("function sheetBehind()", 1)[1].split("function streamMenu()", 1)[0]
    # L1: its own 1024 px query for a Moment or list; the answer keeps the 760 px one (G-8)
    assert "? narrow : sheetWidth;" in sheet and "q.matches" in sheet and "!pane.hidden" in sheet
    assert 'setAttribute("inert", "")' in sheet and 'removeAttribute("inert")' in sheet
    # it follows the peek open, close and the page widening, and it frees the row before focus returns to it
    load = js.split("function load(", 1)[1].split("function close()", 1)[0]
    assert "sheetBehind();" in load
    closing = js.split("function close()", 1)[1].split("function problem(", 1)[0]
    assert closing.index("sheetBehind();") < closing.index("back.focus()")
    assert "sheetBehind(); // the peek is a sheet only" in js


def test_favicon_is_no_content(world):
    c = client(world)
    r = c.get("/favicon.ico")
    assert r.status_code == 204 and r.content == b""
    assert "default-src 'self'" in r.headers["content-security-policy"] and r.headers["cache-control"] == "no-store"
    assert c.get("/favicon.ico", headers={"host": "example.com"}).status_code == 400  # the Host rule still holds
    assert c.post("/favicon.ico").status_code == 405


def test_long_suggestion_shortens_with_an_ellipsis_and_keeps_its_words(world):
    page = client(world).get("/").text
    found = re.findall(r'<button class="sg" type="submit" title="([^"]+)"><span>([^<]+)</span></button>', page)
    assert found and all(title == label for title, label in found)
    css = (STATIC / "page.css").read_text(encoding="utf-8")
    rule = re.search(r"\.sg>span\{([^}]*)\}", css)
    assert rule and all(p in rule.group(1) for p in ("min-width:0", "overflow:hidden", "text-overflow:ellipsis", "white-space:nowrap"))


def _ordinary_in(world, stream_id: str, far_from: str) -> str:
    near = world.windows[far_from].anchor_at
    return next(
        w.window_id for w in world.export.windows
        if w.anchor_kind == "ordinary" and w.stream_id == stream_id and abs((w.anchor_at - near).total_seconds()) > 600
    )  # fmt: skip


def test_peek_of_an_ordinary_window_shows_its_peak(world):
    m = world.new_moments
    ordinary = _ordinary_in(world, world.new, m[0])
    clear_activity(world.store)
    a = activity_line(world, world.new, peaks={m[0]: 81})
    assert store_activity(world.store, a) == "written"
    expected = activity_mod.peak_near(world.store.stream_activity(world.new, now=NOW), world.windows[ordinary].anchor_at)
    assert expected == 9
    c = client(world)
    for url in (f"/m/{ordinary}", f"/m/{ordinary}?panel=1"):
        page = c.get(url).text
        assert _peak_line(page) == "9 people chatting in the busiest 10 s · 1× the usual pace", url
        assert _badge(page) is None  # an ordinary window is no Moment


def test_peek_of_an_ordinary_window_in_a_gap_says_not_recorded(world):
    m = world.new_moments
    ordinary = _ordinary_in(world, world.new, m[0])
    clear_activity(world.store)
    src = next(s for s in world.export.streams if s.stream_id == world.new)
    at = int((world.windows[ordinary].anchor_at - src.started_at).total_seconds()) // 10
    a = activity_line(world, world.new, peaks={m[1]: 40}, gaps=((at - 1, 4),))
    assert store_activity(world.store, a) == "written"
    c = client(world)
    for url in (f"/m/{ordinary}", f"/m/{ordinary}?panel=1"):
        page = c.get(url).text
        assert _peak_line(page) == "Chat not recorded at this Moment.", url
        assert "people chatting" not in page and _badge(page) is None
