"""Asking on the page: POST /ask, threads, chips, the asks table and its expiry (Task 10b)."""

from __future__ import annotations

import json
import logging
import re
from datetime import timedelta
from urllib.parse import unquote

import pytest

import pagekit
from chatbrain import expiry
from chatbrain.ask import query as q_mod
from chatbrain.export_format import iso, parse_dt

ORIGIN = {"origin": f"http://{pagekit.app_mod.HOST}:{pagekit.app_mod.PORT}"}
THREAD = re.compile(r"^/\?thread=([0-9a-f]{16})#a(\d+)$")


@pytest.fixture
def world(tmp_path):
    w = pagekit.build_world(tmp_path)
    pagekit.truth_labels(w, skip={"fake3-w0021"})
    yield w
    w.store.close()


def asks(world) -> list[dict]:
    cols = ("id", "thread_id", "created_at", "expires_at", "question", "query_json", "follow_up",
            "result_window_ids_json", "summary")  # fmt: skip
    rows = world.store.conn.execute(f"SELECT {', '.join(cols)} FROM asks ORDER BY id").fetchall()
    return [dict(zip(cols, r)) for r in rows]


def post_ask(c, question, **extra):
    r = c.post("/ask", data={"q": question, **extra}, headers=ORIGIN, follow_redirects=False)
    assert r.status_code == 303, r.text
    m = THREAD.match(r.headers["location"])
    assert m, r.headers["location"]
    return m.group(1), int(m.group(2)), r.headers["location"]


def test_home_shows_ask_line_banner_and_three_suggestions(world):
    # F3 (Task 6, G-8): one ask line with a "/" key; on focus "Try one" and the first three suggestions
    page = pagekit.client(world).get("/").text
    assert 'class="ask"' in page and 'name="q"' in page and 'maxlength="300"' in page
    assert 'placeholder="Ask about this stream"' in page and "Ask a follow-up…" not in page
    assert '<kbd class="sx-k" aria-hidden="true">/</kbd>' in page
    assert (
        "Asking is limited: the local model isn't running. Words were matched to filters; check the chips."
        in pagekit.text(page)
    )
    dock = page[page.index('<div class="dock">'):].split('<p class="loc">')[0]
    sug = re.search(r'<div class="sug"><p class="sm mu">Try one</p>(.*)', dock, re.S)
    assert sug
    shown = [pagekit.text(b) for b in re.findall(r'<button class="sg"[^>]*>(.*?)</button>', sug.group(1), re.S)]
    assert shown == q_mod.suggestions(q_mod.held(world.store))[:3]
    assert "Runs on this PC. Numbers come from stored labels and counts." in pagekit.text(page)


def test_url_never_holds_question(world):
    c = pagekit.client(world)
    question = "what made chat laugh the most"
    thread, ask_id, location = post_ask(c, question)
    assert "laugh" not in unquote(location)
    page = c.get(location.split("#")[0]).text
    assert question in pagekit.text(page)  # echoed in the answer, from the asks row
    for href in re.findall(r'(?:href|action)="([^"]+)"', page):
        assert "laugh the most" not in unquote(href)
    assert f'id="a{ask_id}"' in page


ANATOMY = [
    # (question, window_id, stub model, expected intent); "{at}" is fake3-w0018's local clock time
    ("what made chat laugh the most", None, False, "top_moments"),
    ("show every moment chat was laughing", None, False, "find_moments"),
    ("show every moment chat was laughing at a tech problem all time", None, False, "find_moments"),  # 0 match
    ("how many moments were hostile all time", None, False, "count"),
    ("was it good?", None, False, "count"),  # the default reading
    ("is chat getting more critical all time", None, False, "trend"),
    ("What does KEKW mean in fakechan01?", None, False, "emote_meaning"),  # all time
    ("what does KEKW mean last stream", None, False, "emote_meaning"),
    ("what was chat doing at {at}", None, False, "moment_detail"),
    ("what happened here", "fake3-w0021", False, "moment_detail"),  # a Moment with no "doing" answer
    ("why cheering?", "fake3-w0022", False, "explain_answer"),
    ("how sure is it", None, False, "explain_answer"),  # no model
    ("how sure is it", None, True, "explain_answer"),  # the model's report
    ("why did chat go off at {at}", None, False, "cant_answer"),  # on stream
    ("why did chat go off all time", None, False, "cant_answer"),  # on stream, already all time
]


@pytest.mark.parametrize(("question", "window_id", "model", "intent"), ANATOMY)
def test_answer_anatomy(world, question, window_id, model, intent):
    if model:
        pagekit.stub_model(world.store, world.models, pagekit.RUNG1_ID, 1)
    c = pagekit.client(world)
    at = world.windows["fake3-w0018"].anchor_at.astimezone().strftime("%H:%M")
    extra = {"window_id": window_id} if window_id else {}
    _, ask_id, location = post_ask(c, question.format(at=at), **extra)
    q = json.loads(asks(world)[-1]["query_json"])["query"]
    assert q["intent"] == intent and (intent != "cant_answer" or q["reason"] == "on_stream")
    block = block_of(c, location, ask_id)
    parts = ['class="q"', 'class="stp"', 'class="ans"', 'class="lf"', *(['class="mcs"'] if "mcs" in block else []),
             'class="fus"']  # fmt: skip
    order = [block.index(x) for x in parts]
    assert order == sorted(order)
    assert len(re.findall(r'<button class="fu"', block)) == 2  # every answer: two follow-ups
    held = q_mod.held(world.store)
    for form in re.findall(r'<form class="fuf"[^>]*>(.*?)</form>', block, re.S):
        checked = q_mod.from_tokens(re.findall(r'name="tok" value="([^"]+)"', form), pagekit.SCHEME, held)
        assert not checked.ignored and "unusable" not in checked.codes  # each one runs as it is


def test_find_answer_shows_scope_chips_and_cards(world):
    c = pagekit.client(world)
    _, ask_id, location = post_ask(c, "what made chat laugh the most")
    block = block_of(c, location, ask_id)
    assert "Read the shown answers of 7 Moments" in pagekit.text(block)  # the step that read them (F3)
    assert "doing: laughing" in pagekit.text(block)
    assert pagekit.cards(block)[:2] == ["fake3-w0018", "fake3-w0027"]


def test_thread_reruns_its_stored_query(world):
    c = pagekit.client(world)
    thread, ask_id, location = post_ask(c, "what made chat laugh the most")
    (row,) = asks(world)
    stored = json.loads(row["query_json"])
    assert stored["query"]["doing"] == ["laughing"]
    assert "7 Moments" not in row["query_json"] and row["summary"] is None  # numbers are never stored
    # a later fix changes the answer; the thread shows the new numbers
    pagekit.label(world.store, "fake3-w0022", {"doing": ["laughing", "cheering"]}, at=9)
    page = c.get(f"/?thread={thread}").text
    block = page[page.index(f'id="a{ask_id}"'):].split("</article>")[0]
    assert "fake3-w0022" in pagekit.cards(block)


def test_follow_up_in_a_thread_keeps_chips(world):
    c = pagekit.client(world)
    thread, _, _ = post_ask(c, "what made chat laugh the most")
    thread2, ask2, location = post_ask(c, "only the guest ones", thread_id=thread)
    assert thread2 == thread
    page = c.get(location.split("#")[0]).text
    block = page[page.index(f'id="a{ask2}"'):].split("</article>")[0]
    assert re.search(r"doing: laughing\s*<em>kept</em>", block)
    assert "aimed at: guest" in pagekit.text(block)
    assert pagekit.cards(block) == ["fake3-w0018"]
    assert [r["follow_up"] for r in asks(world)] == ["new", "refine"]


@pytest.mark.parametrize("follow", ["only the guest ones", "same for last week", "what happened in the 1st one"])
def test_typed_follow_up_keeps_a_quoted_phrase(world, follow):
    c = pagekit.client(world)
    word = next(m.text.split()[0] for m in world.windows["fake3-w0018"].messages if m.text.split())
    thread, _, _ = post_ask(c, f'find "{word}" all time')
    assert json.loads(asks(world)[0]["query_json"])["query"]["phrase"] == word
    _, ask2, location = post_ask(c, follow, thread_id=thread)
    stored = json.loads(asks(world)[-1]["query_json"])
    assert stored["query"]["phrase"] == word  # checked when first asked; not dropped by this question's text
    assert f"phrase:{word}" in stored["notes"]["kept"]
    assert not any(t.startswith("phrase:") for t in stored["notes"]["ignored"])
    block = block_of(c, location, ask2)
    assert re.search(rf"said: “{word}” · only windows whose text is still kept\s*<em>kept</em>", block)
    assert "ignored: phrase" not in pagekit.text(block)


def test_a_new_phrase_must_be_in_the_question(world):
    c = pagekit.client(world)
    thread, _, _ = post_ask(c, 'find "zzqq" all time')
    page = c.app.state.page
    held = q_mod.held(world.store)
    raw = json.loads(asks(world)[0]["query_json"])["query"] | {"phrase": "made up"}
    checked = q_mod.check(raw, page.scheme, held, question="only the guest ones", carried_phrase="zzqq")
    assert checked.query["phrase"] is None and "phrase:made up" in checked.ignored
    kept = q_mod.check(raw | {"phrase": "zzqq"}, page.scheme, held, question="only the guest ones",
                       carried_phrase="zzqq")  # fmt: skip
    assert kept.query["phrase"] == "zzqq" and not kept.ignored


def test_chip_edit_adds_a_refine_ask(world):
    c = pagekit.client(world)
    thread, ask_id, _ = post_ask(c, "what made chat laugh the most")
    stored = json.loads(asks(world)[0]["query_json"])["query"]
    toks = q_mod.tokens(stored)
    r = c.post(f"/ask/{ask_id}/edit", data={"tok": toks, "drop": "doing:laughing"}, headers=ORIGIN,
               follow_redirects=False)  # fmt: skip
    assert r.status_code == 303
    m = THREAD.match(r.headers["location"])
    assert m and m.group(1) == thread
    new = asks(world)[-1]
    assert new["follow_up"] == "refine" and new["question"] is None
    assert json.loads(new["query_json"])["query"]["doing"] == []
    r = c.post(f"/ask/{ask_id}/edit", data={"tok": toks, "add": "aimed:guest"}, headers=ORIGIN,
               follow_redirects=False)  # fmt: skip
    page = c.get(r.headers["location"].split("#")[0]).text
    last = int(THREAD.match(r.headers["location"]).group(2))
    block = page[page.index(f'id="a{last}"'):].split("</article>")[0]
    assert "Edited the chips" in pagekit.text(block)
    assert re.search(r"doing: laughing\s*<em>kept</em>", block)
    assert pagekit.cards(block) == ["fake3-w0018"]


def test_refusals_never_store_text(world, caplog):
    c = pagekit.client(world)
    cases = [
        ("who started it at 22:03?", "identity", "started"),
        ("did anyone say something racist", "traits", "racist"),
        ("say it like the streamer would", "as_streamer", "would"),
    ]
    with caplog.at_level(logging.DEBUG):
        for question, reason, word in cases:
            _, ask_id, location = post_ask(c, question)
            page = c.get(location.split("#")[0]).text
            block = page[page.index(f'id="a{ask_id}"'):].split("</article>")[0]
            assert f"[refused: {reason}]" in pagekit.text(block)
    rows = asks(world)
    assert [r["question"] for r in rows] == [f"[refused: {r}]" for _, r, _ in cases]
    for row, (_, _, word) in zip(rows, cases):
        assert word not in row["query_json"]
        assert json.loads(row["result_window_ids_json"]) == []
    raw = world.store.path.read_bytes()
    for _, _, word in cases:
        assert word.encode() not in raw
    assert "racist" not in caplog.text and "started" not in caplog.text


def test_identity_refusal_offers_the_nearest_moment(world):
    c = pagekit.client(world)
    at = world.windows["fake3-w0018"].anchor_at.astimezone().strftime("%H:%M")
    _, ask_id, location = post_ask(c, f"who started it at {at}")
    page = c.get(location.split("#")[0]).text
    block = page[page.index(f'id="a{ask_id}"'):].split("</article>")[0]
    assert "BrainChat doesn't track who people are." in pagekit.text(block)
    assert 'class="chip rch"' in block and "fake3-w0018" in block
    assert "11 posters joining" in pagekit.text(block)


def test_app_never_logs_the_question(world, caplog):
    c = pagekit.client(world)
    with caplog.at_level(logging.DEBUG):
        post_ask(c, "what made chat laugh the most wibblefrog")
        c.get("/")
    assert "wibblefrog" not in caplog.text
    assert any(r.name == "chatbrain.ask" for r in caplog.records)


def test_asks_expire_with_earliest_window_text(world):
    c = pagekit.client(world)
    post_ask(c, "what made chat laugh the most")
    (row,) = asks(world)
    ids = json.loads(row["result_window_ids_json"])
    assert ids == ["fake3-w0018", "fake3-w0027"]
    earliest = min(world.windows[i].text_expires_at for i in ids)
    assert earliest < pagekit.NOW + timedelta(days=14)
    assert row["expires_at"] == iso(earliest)
    before = expiry.run(world.store, earliest - timedelta(seconds=1), world.tmp / "imported", None,
                        models_dir=world.models)  # fmt: skip
    assert before.asks_deleted == 0 and len(asks(world)) == 1
    after = expiry.run(world.store, earliest, world.tmp / "imported", None, models_dir=world.models)
    assert after.asks_deleted == 1 and asks(world) == []


def test_asks_expire_at_14_days(world):
    c = pagekit.client(world)
    post_ask(c, "who started it")
    (row,) = asks(world)
    assert parse_dt(row["expires_at"], "x") == pagekit.NOW + timedelta(days=14)
    expiry.run(world.store, pagekit.NOW + timedelta(days=14) - timedelta(seconds=1), world.tmp / "imported", None,
               models_dir=world.models)  # fmt: skip
    assert len(asks(world)) == 1
    expiry.run(world.store, pagekit.NOW + timedelta(days=14), world.tmp / "imported", None, models_dir=world.models)
    assert asks(world) == []


def test_expired_ask_does_not_show_before_the_hourly_run(world):
    # one page with a moving clock: page start ran expiry at NOW, and nothing runs it again here
    t = {"now": pagekit.NOW}
    c = pagekit.client(world, clock=lambda: t["now"])
    thread, ask_id, _ = post_ask(c, "what made chat laugh the most")
    (row,) = asks(world)
    late = parse_dt(row["expires_at"], "x")
    echo = '<p class="q">what made chat laugh the most</p>'
    t["now"] = late - timedelta(seconds=1)
    page = c.get(f"/?thread={thread}").text
    assert f'id="a{ask_id}"' in page and echo in page and f'href="/?thread={thread}"' in page
    t["now"] = late
    assert len(asks(world)) == 1  # the row is still stored: only the hourly run deletes it
    page = c.get(f"/?thread={thread}").text
    assert f'id="a{ask_id}"' not in page and echo not in page
    assert f'href="/?thread={thread}"' not in page  # nor listed under Asked before
    toks = q_mod.tokens(json.loads(row["query_json"])["query"])
    r = c.post(f"/ask/{ask_id}/edit", data={"tok": toks}, headers=ORIGIN, follow_redirects=False)
    assert r.status_code == 404  # an expired ask can't be edited either
    assert len(asks(world)) == 1


def test_emote_names_outlive_text(world):
    names = world.store.conn.execute("SELECT channel, emote_id, name FROM emote_names ORDER BY emote_id").fetchall()
    assert ("fakechan01", "fake-kekw", "KEKW") in names
    cols = [r[1] for r in world.store.conn.execute("PRAGMA table_info(emote_names)").fetchall()]
    assert cols == ["channel", "emote_id", "name"]  # public names only: no text, no speaker
    late = pagekit.NOW + timedelta(days=40)
    expiry.run(world.store, late, world.tmp / "imported", None, models_dir=world.models)
    assert world.store.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
    assert world.store.conn.execute("SELECT COUNT(*) FROM emote_names").fetchone()[0] == len(names)
    c = pagekit.client(world, now=late)
    _, ask_id, location = post_ask(c, "What does KEKW mean in fakechan01?")
    page = c.get(location.split("#")[0]).text
    block = page[page.index(f'id="a{ask_id}"'):].split("</article>")[0]
    assert "KEKW showed up in 6 windows" in pagekit.text(block)


def test_history_rail_lists_threads_by_first_question(world):
    c = pagekit.client(world)
    t1, _, _ = post_ask(c, "what made chat laugh the most")
    post_ask(c, "only the guest ones", thread_id=t1)
    post_ask(c, "who started it")
    page = c.get("/").text
    rail = re.search(r'<nav class="hist"[^>]*>(.*?)</nav>', page, re.S)
    assert rail
    items = [pagekit.text(x) for x in re.findall(r"<li>(.*?)</li>", rail.group(1), re.S)]
    assert items == ["[refused: identity]", "what made chat laugh the most"]
    assert f'href="/?thread={t1}"' in rail.group(1)


def test_asked_before_is_reachable_at_every_width(world):
    c = pagekit.client(world)
    t1, _, _ = post_ask(c, "what made chat laugh the most")
    post_ask(c, "who started it")
    page = c.get(f"/?thread={t1}").text
    # the glance has no right lane (Task 3, G-8): "Asked before" is one fold under the dock
    assert '<aside class="lane"' not in page
    fold = re.search(r'<details class="hsd"><summary class="fold">Asked before · 2</summary>(.*?)</details>', page,
                     re.S)  # fmt: skip
    assert fold and page.index('class="dock"') < page.index('class="hsd"')
    assert len(re.findall(r'<nav class="hist"', page)) == 1
    links = re.findall(r'<a href="/\?thread=([0-9a-f]{16})"[^>]*>', fold.group(1))
    assert len(links) == 2 and t1 in links
    assert f'<a href="/?thread={t1}" aria-current="page">' in fold.group(1)
    assert "this one stays under Asked before" in pagekit.text(page)
    # nothing hides the fold at any width
    css = (pagekit.app_mod.PAGE_DIR / "static" / "page.css").read_text(encoding="utf-8")
    assert not re.search(r"\.hsd\{display:none", css) and ".lane{" not in css


def test_moment_part_7_asks_about_that_moment(world):
    c = pagekit.client(world)
    page = c.get("/m/fake3-w0022").text
    form = re.search(r'<form class="ask pask"[^>]*data-ask[^>]*>(.*?)</form>', page, re.S)
    assert form and 'name="window_id" value="fake3-w0022"' in form.group(1)
    assert "Ask about this Moment…" in form.group(1)
    _, ask_id, location = post_ask(c, "why cheering?", window_id="fake3-w0022")
    row = asks(world)[-1]
    assert json.loads(row["query_json"])["query"]["intent"] == "explain_answer"
    assert json.loads(row["result_window_ids_json"]) == ["fake3-w0022"]
    page = c.get(location.split("#")[0]).text
    assert f'id="a{ask_id}"' in page
    _, _, _ = post_ask(c, "what happened here", window_id="fake3-w0022")
    assert json.loads(asks(world)[-1]["query_json"])["query"]["intent"] == "moment_detail"


def test_ask_refuses_bad_input(world):
    c = pagekit.client(world)
    assert c.post("/ask", data={"q": "x" * 301}, headers=ORIGIN).status_code == 400
    assert c.post("/ask", data={"q": "   "}, headers=ORIGIN).status_code == 400
    assert c.post("/ask", data={"q": "hi", "thread_id": "../etc"}, headers=ORIGIN).status_code == 400
    assert c.post("/ask", data={"q": "hi", "window_id": "nope"}, headers=ORIGIN).status_code == 404
    assert c.post("/ask", data={"q": "hi"}, headers={"origin": "http://evil.example"}).status_code == 403
    assert c.get("/?thread=nothex").status_code == 400
    assert asks(world) == []


def test_new_question_starts_a_new_thread(world):
    c = pagekit.client(world)
    t1, _, _ = post_ask(c, "what made chat laugh the most")
    t2, _, _ = post_ask(c, "what were the biggest moments last stream")
    assert t1 != t2


def block_of(c, location, ask_id):
    page = c.get(location.split("#")[0]).text
    return page[page.index(f'id="a{ask_id}"'):].split("</article>")[0]


@pytest.mark.parametrize(
    ("question", "reason", "runs"),
    [
        ("who started it", "identity", None),  # the nearest Moment opens
        ("did anyone say something racist", "traits", "hostile"),
        ("is this message toxic?", "single_message", "hostile"),
        ("say it like the streamer would", "as_streamer", "count"),
        ("send this to discord", "outbound", "find_moments"),
    ],
)
def test_each_refusal_offers_one_runnable_chip(world, question, reason, runs):
    c = pagekit.client(world)
    _, ask_id, location = post_ask(c, question)
    block = block_of(c, location, ask_id)
    assert f"[refused: {reason}]" in pagekit.text(block)
    assert block.count('class="chip rch"') == 1
    assert 'class="fus"' not in block  # one chip, no follow-up row
    if runs is None:
        assert re.search(r'<a class="chip rch" href="/m/fake3-w\d+"', block)
        return
    form = re.search(r'<form class="rcf" method="post" action="/ask/(\d+)/edit">(.*?)</form>', block, re.S)
    assert form and int(form.group(1)) == ask_id
    toks = re.findall(r'name="tok" value="([^"]+)"', form.group(2))
    r = c.post(f"/ask/{ask_id}/edit", data={"tok": toks}, headers=ORIGIN, follow_redirects=False)
    assert r.status_code == 303
    q = json.loads(asks(world)[-1]["query_json"])["query"]
    if runs == "hostile":
        assert q["intent"] == "find_moments" and q["tone"] == ["hostile"]
    elif runs == "count":
        assert q["intent"] == "count" and q["about"] == "doing" and q["by"] == "value"
    else:
        assert q["intent"] == "find_moments" and q["sort"] == "oldest"
    later = block_of(c, r.headers["location"], int(THREAD.match(r.headers["location"]).group(2)))
    assert "reason" not in pagekit.text(later)  # the echo lists only the chips that changed


def test_full_moment_page_asks_and_the_panel_does_not(world):
    # G-14: the side-lane panel drops the per-Moment ask; the full page keeps it
    c = pagekit.client(world)
    page = c.get("/m/fake3-w0022").text
    form = re.search(r'<form class="ask pask"[^>]*action="/ask"[^>]*>(.*?)</form>', page, re.S)
    assert form and 'name="window_id" value="fake3-w0022"' in form.group(1)
    assert 'class="ask pask"' not in c.get("/m/fake3-w0022?panel=1").text
    _, ask_id, location = post_ask(c, "was chat mad about the sponsor", window_id="fake3-w0022")
    block = block_of(c, location, ask_id)
    assert "Chat alone can't show what happened on stream, so the cause is unknown." in pagekit.text(block)
    assert json.loads(asks(world)[-1]["result_window_ids_json"]) == ["fake3-w0022"]


def test_viewer_question_says_posters_not_viewers(world):
    c = pagekit.client(world)
    _, ask_id, location = post_ask(c, "how many people were watching")
    block = block_of(c, location, ask_id)
    assert "BrainChat counts people who posted, not viewers." in pagekit.text(block)
    assert pagekit.cards(block)


def test_moment_answer_shows_answers_cause_and_lines(world):
    c = pagekit.client(world)
    at = world.windows["fake3-w0018"].anchor_at.astimezone().strftime("%H:%M")
    _, ask_id, location = post_ask(c, f"what was chat doing at {at}")
    block = block_of(c, location, ask_id)
    assert block.count('class="aq"') == 4  # doing, aimed at, feels, tone
    assert "Cause unknown · no cause note" in pagekit.text(block)
    assert 'class="cl"' in block or 'class="exp"' in block  # the chat lines, or why there are none


def test_open_thread_offers_a_new_question(world):
    c = pagekit.client(world)
    assert "New question" not in pagekit.text(c.get("/").text)
    thread, _, _ = post_ask(c, "what made chat laugh the most")
    page = c.get(f"/?thread={thread}").text
    assert re.search(r'<a class="nq" href="/">New question</a>', page)


def test_a_phrase_ask_expires_with_every_window_the_phrase_reaches(world):
    # B26: the stored question holds the quoted phrase, and that phrase is chat text of every
    # window that says it, not only of the (capped) result windows.
    c = pagekit.client(world)
    post_ask(c, 'top 1 moments where chat said "lurking"')
    (row,) = asks(world)
    ids = json.loads(row["result_window_ids_json"])
    assert json.loads(row["query_json"])["query"]["phrase"] == "lurking" and len(ids) == 1
    said = {w for (w,) in world.store.conn.execute(
        "SELECT DISTINCT window_id FROM messages WHERE lower(text) LIKE '%lurking%'")}  # fmt: skip
    earliest = min(world.windows[w].text_expires_at for w in said)
    assert earliest < min(world.windows[w].text_expires_at for w in ids)  # a match past the cap expires first
    assert row["expires_at"] == iso(earliest)
    after = expiry.run(world.store, earliest, world.tmp / "imported", None, models_dir=world.models)
    assert after.asks_deleted == 1 and asks(world) == []


# --- F3: the answer in the side lane, with the steps that really ran (Task 6) ----------------------

STEPS = re.compile(r'<ol class="stl sx-ol" aria-label="Search steps">(.*?)</ol>', re.S)
ROW = re.compile(r'<li class="sx-row" data-s="(\w+)"><i class="sx-i" aria-hidden="true"></i><span class="sx-l">(.*?)</span></li>',
                 re.S)  # fmt: skip
SETTLED = re.compile(r"<summary>✓ (\d+) steps? · (\d+\.\d) s(?: · made-up labels)?(?: · keywords)?</summary>")


def steps_of(block: str) -> list[tuple[str, str]] | None:
    """The settled step rows of one answer as (state, text), or None when it shows no steps."""
    m = STEPS.search(block)
    return [(s, pagekit.text(t)) for s, t in ROW.findall(m.group(1))] if m else None


def test_steps_are_real_keyword_only(world):
    # G-9: with the model off only the keyword steps ran, so only they show; no read or caption row
    c = pagekit.client(world)
    _, ask_id, location = post_ask(c, "what made chat laugh the most")
    block = block_of(c, location, ask_id)
    rows = steps_of(block)
    assert rows is not None
    assert rows[:2] == [("done", "Matched words to filters (no model)"), ("done", "Read the shown answers of 7 Moments")]
    assert all(s == "done" for s, _ in rows)
    assert all(re.fullmatch(r"Ran \d+ read-only quer(y|ies)", t) for _, t in rows[2:]) and len(rows) <= 3
    assert not [t for _, t in rows if "local model" in t or "caption" in t]
    settled = SETTLED.search(block)
    assert settled and int(settled.group(1)) == len(rows)


def test_step_ms_not_remeasured_on_rerun(world):
    c = pagekit.client(world)
    thread, ask_id, _ = post_ask(c, "what made chat laugh the most")
    (row,) = asks(world)
    stored = json.loads(row["query_json"])
    ms = stored["notes"]["ms"]
    assert set(ms) == {"words", "run"} and all(isinstance(v, int) and v >= 0 for v in ms.values())
    # a re-run shows the stored times, never new ones
    stored["notes"]["ms"] = {"words": 1200, "run": 2300}
    world.store.conn.execute("UPDATE asks SET query_json = ? WHERE id = ?", (json.dumps(stored), ask_id))
    world.store.conn.commit()
    lines = []
    for _ in range(2):
        block = block_of(c, f"/?thread={thread}", ask_id)
        settled = SETTLED.search(block)
        assert settled
        lines.append(settled.group(0))
        assert settled.group(2) == "3.5"
    assert lines[0] == lines[1]
    # an ask stored before the steps had times shows no time at all
    del stored["notes"]["ms"]
    world.store.conn.execute("UPDATE asks SET query_json = ? WHERE id = ?", (json.dumps(stored), ask_id))
    world.store.conn.commit()
    block = block_of(c, f"/?thread={thread}", ask_id)
    assert re.search(r"<summary>✓ \d+ steps(?: · made-up labels)?</summary>", block)


def test_refusal_has_no_steps(world):
    c = pagekit.client(world)
    _, ask_id, location = post_ask(c, "who started it")
    block = block_of(c, location, ask_id)
    assert "[refused: identity]" in pagekit.text(block)
    assert 'class="stp"' not in block and "Search steps" not in block and 'class="stl' not in block
    assert json.loads(asks(world)[-1]["query_json"])["notes"] == {}  # nothing kept, not even times


def test_thread_opens_lane_without_js(world):
    c = pagekit.client(world)
    thread, ask_id, _ = post_ask(c, "what made chat laugh the most")
    page = c.get(f"/?thread={thread}").text
    head = '<div class="pn" role="region" aria-label="Answer">'
    assert page.count(head) == 1
    lane = page.split(head, 1)[1]
    assert f'<div class="sd-in sd-a" data-thread="{thread}">' in lane
    assert re.search(r'<a class="sd-x" href="/\?stream=[^"]+" data-close aria-label="Close the answer">×</a>', lane)
    assert f'id="a{ask_id}"' in lane and '<div class="thr">' in lane
    # the glance keeps the main column; the dock stays under it; the answer is in the lane
    assert page.index('class="gl-recap"') < page.index('<div class="dock">') < page.index(head)
    assert not re.search(r'id="a\d+"', page.split(head, 1)[0])
    home = c.get("/").text
    assert '<div class="pn" role="region" aria-label="Moment" hidden></div>' in home
    assert not re.search(r'id="a\d+"', home) and 'aria-label="Answer"' not in home


def test_answer_back_from_moment(world):
    c = pagekit.client(world)
    thread, ask_id, location = post_ask(c, "what made chat laugh the most")
    block = block_of(c, location, ask_id)
    assert 'href="/m/fake3-w0018" data-open' in block  # a cited Moment opens in the lane
    panel = c.get(f"/m/fake3-w0018?panel=1&answer={thread}").text
    back = re.search(r'<a class="sd-back" href="/\?stream=[^"&]+&amp;thread=([0-9a-f]{16})" data-answer>'
                     r'<span aria-hidden="true">‹</span> Answer</a>', panel)  # fmt: skip
    assert back and back.group(1) == thread
    assert 'data-open><span aria-hidden="true">‹</span> All' not in panel  # one way back: to the answer
    plain = c.get("/m/fake3-w0018?panel=1").text
    assert "data-answer" not in plain and '<span aria-hidden="true">‹</span> All ' in plain
    assert c.get("/m/fake3-w0018?panel=1&answer=nothex").status_code == 400
    js = (pagekit.app_mod.PAGE_DIR / "static" / "page.js").read_text(encoding="utf-8")
    assert 'searchParams.set("answer"' in js and "[data-answer]" in js
    # G-11 in the script: a Moment opened from the answer goes back to it; leaving the answer for
    # anything else stops a running step; closing it puts focus on the ask line, suggestions shut
    leave = js.split("function leaveAnswer()", 1)[1].split("\n  function ", 1)[0]
    assert "stopAsk(" in leave
    closing = js.split("function close()", 1)[1].split("function problem(", 1)[0]
    assert "leaveAnswer();" in closing and "askLine(" in closing
    keys = js.split("if (esc) {", 1)[1].split("return;", 1)[0]
    assert re.search(r"if \(busy\) \{\s*stopAsk\(busy\);\s*\} else if \(fixMenu\)", keys)  # a running step first
    assert keys.index("listHref") < keys.index("showAnswer(") < keys.index("close();")


def test_thread_sheet_takes_focus_on_load():
    """F3 (G-11): on a phone a thread opens on its answer as a modal sheet; focus goes into it (the
    answer the address names, else the newest, else the close link), never left behind the sheet."""
    js = (pagekit.app_mod.PAGE_DIR / "static" / "page.js").read_text(encoding="utf-8")
    load = js.split("  if (answerNode) {\n    sheetBehind();", 1)
    assert len(load) == 2
    body = load[1].split("\n  }\n", 1)[0]
    assert 'pane.getAttribute("aria-modal") === "true"' in body
    assert "/^#a\\d+$/.test(location.hash)" in body and "answerNode.contains(named)" in body
    assert 'arts[arts.length - 1] || answerNode.querySelector(".sd-x")' in body and "into.focus()" in body


def test_dock_try_one_opens_on_focus():
    """F3 (G-8): the ask line is 46 px; "Try one" shows only while focus is in the dock, and stays
    shut after the answer closes (sx-shut) until a key or focus leaves the dock."""
    static = pagekit.app_mod.PAGE_DIR / "static"
    css = (static / "page.css").read_text(encoding="utf-8")
    rules = (".dock .sug{display:none;", ".dock:focus-within .sug{display:grid}", ".dock.sx-shut .sug{display:none}")
    order = [css.index(x) for x in rules]
    assert order == sorted(order)
    assert ".dock .ask{min-height:46px}" in css and ".ask:focus-within .sx-k{display:none}" in css
    # "Try one" is a popover above the ask line (G1/F3 mocks), so the dock never grows on focus
    pop = css.split(".dock .sug{display:none;", 1)[1].split("}", 1)[0]
    assert "position:absolute" in pop and "bottom:calc(100% +" in pop
    assert ".sx-al{position:relative" in css
    # the "/" key is a desk hint: the phone block hides it
    phone = [b for b in css.split("@media (max-width:759.98px){")[1:] if ".dock{width:auto" in b.split("\n}", 1)[0]]
    assert phone and ".sx-k{display:none}" in phone[0].split("\n}", 1)[0]
    # dead rules from the history's old place under the dock are gone
    assert ".main>.hsd" not in css and ".dock,.sug{" not in css
    js = (static / "page.js").read_text(encoding="utf-8")
    line = js.split("function askLine(shut)", 1)[1].split("\n  function ", 1)[0]
    assert 'classList.toggle("sx-shut", !!shut)' in line
    assert 'classList.remove("sx-shut")' in js and "askLine(false)" in js


def test_dock_clears_answer_lane_on_desk():
    """F3 (G-8): on desk the lane is fixed at the right (408 px); while it is open the dock keeps clear
    of it, so the ask line, its "/" key and Ask stay usable beside the answer at 760 to 1575 px."""
    css = (pagekit.app_mod.PAGE_DIR / "static" / "page.css").read_text(encoding="utf-8")
    block = css.split("@media (min-width:760px){\n.main:has(>.pn:not([hidden]))>.dock{", 1)
    assert len(block) == 2, "the dock has no rule for an open lane on desk"
    rule = block[1].split("}", 1)[0]
    assert "max-width:min(760px,calc(100% - 408px))" in rule
    assert "margin-left:max(0px,calc((100% - 408px - 760px) / 2))" in rule and "margin-right:auto" in rule
    assert ".pn{position:absolute;top:0;right:0;bottom:0;z-index:10;width:408px;" in css  # the width it clears


def test_dock_suggestions_continue_the_thread(world):
    """F3 (G-8): in a thread, "Try one" continues the thread, as a typed question does."""
    c = pagekit.client(world)
    thread, _ask_id, location = post_ask(c, "what made chat laugh the most")
    page = c.get(location.split("#")[0]).text
    sug = page[page.index('<div class="sug">'):].split('<p class="loc">')[0]
    forms = re.findall(r'<form class="sgf".*?</form>', sug, re.S)
    assert forms and all(f'<input type="hidden" name="thread_id" value="{thread}">' in f for f in forms)
    home = c.get("/").text  # no thread: a suggestion starts one
    sug0 = home[home.index('<div class="sug">'):].split('<p class="loc">')[0]
    assert 'name="thread_id"' not in sug0
