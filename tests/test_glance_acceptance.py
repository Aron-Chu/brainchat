"""Synthetic acceptance (Task 7): one test per page state, each asserting the copy table's strings on
the glance, the All N list and the opened Moment together. Made-up worlds only (``fake`` / ``pagekit``);
nothing here reads the real data folder, starts a model or opens a socket."""

import re
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from chatbrain import fake
from chatbrain.page import app as app_mod
from chatbrain.store import Store
from pagekit import (
    NOW, RUNG0_ID, activity_line, build_world, clear_activity, client, label, predict, store_activity, stub_model,
    text,
)  # fmt: skip

STATIC = Path(app_mod.__file__).parent / "static"
ORIGIN = {"origin": f"http://{app_mod.HOST}:{app_mod.PORT}"}

# the copy table, verbatim
CAPTION = "Height: people chatting per 10 s"
NO_COUNTS = "This stream's export has no activity counts."
EXPIRED_COUNTS = "Activity counts expired 30 days after this stream started."
NOT_SURE = "Not sure what chat was doing"
NOT_SURE_LINE = "BrainChat isn't sure here, so it makes no guess."
WHEN = "When chat reacted. The screen moment is usually 3–8 s earlier."
FOOT = "No preview pictures yet. Times are when chat reacted; the screen moment is usually 3–8 s earlier."
EMPTY_STREAM = "No Moments in this stream."
FACE_CAPTION = " · top emotes typed"
FACE = re.compile(r"(.+) · in (\d+) of (\d+) lines · sent by (\d+) people · the 120 s around \d\d:\d\d")


@pytest.fixture
def world(tmp_path):
    w = build_world(tmp_path)
    yield w
    w.store.close()


# --- reading the three surfaces ----------------------------------------------------------------------


def _one(pattern: str, page: str) -> str | None:
    found = re.findall(pattern, page, re.S)
    assert len(found) <= 1, found
    return text(found[0]) if found else None


def _eyebrow(page: str) -> str | None:
    return _one(r'<h1 class="gl-eyebrow"[^>]*>(.*?)</h1>', page)


def _heading(page: str) -> str | None:
    return _one(r'<h2 class="gl-h"[^>]*>(.*?)</h2>', page)


def _status(page: str) -> str | None:
    return _one(r'<p class="gl-status">(.*?)</p>', page)


def _gone(page: str) -> str | None:
    return _one(r'<p class="gl-gone" id="gone">(.*?)</p>', page)


def _kept(page: str) -> str | None:
    return _one(r'<span class="gl-kept">(.*?)</span>', page)


def _menu_kept(page: str) -> str | None:
    return _one(r'<p class="gl-pk">(.*?)</p>', page)


def _cards(page: str) -> list[str]:
    return re.findall(r'<li class="gl-card" data-window="([^"]+)"', page)


def _card(page: str, window_id: str) -> str:
    return page.split(f'<li class="gl-card" data-window="{window_id}"')[1].split("</li>")[0]


def _say(card: str) -> str | None:
    return _one(r'<span class="gl-say">(.*?)</span>', card)


def _stat(card: str) -> str | None:
    return _one(r'<span class="gl-size">(.*?)</span>\s*(?=<span class="gl-chat|</a>)', card)


def _quote(card: str) -> str | None:
    return _one(r'<span class="gl-chat[^"]*">(.*?)</span>\s*(?:<span class="fc-sr"|</a>)', card)


def _plot_label(page: str) -> str | None:
    found = re.findall(r'<div class="gl-plot" role="img" aria-label="([^"]+)"', page)
    assert len(found) <= 1
    return text(found[0]) if found else None


def _gap_titles(page: str) -> list[str]:
    return re.findall(r'<rect class="gl-gap"[^>]*><title>([^<]+)</title></rect>', page)


def _head(panel: str) -> str:
    return panel.split('<header class="sd-head">')[1].split("</header>")[0]


def _title(panel: str) -> str:
    return text(re.search(r'<h2 class="sd-say" tabindex="-1">(.*?)</h2>', _head(panel), re.S).group(1))


def _slot(panel: str) -> str:
    """The stat slot's visible text: both stats joined with " · ", a no-peak line, or the Q10 line."""
    found = re.search(r'<div class="sd-slot">(.*)</div>', _head(panel), re.S)
    assert found, "no stat slot"
    stats = re.findall(r'<p class="sd-stat">(.*?)</p>', found.group(1), re.S)
    return " · ".join(text(s) for s in stats) if stats else text(found.group(1))


def _said(panel: str) -> str:
    found = re.search(r'<section class="sd-chat" aria-label="What chat said">(.*?)</section>', panel, re.S)
    assert found, "no What chat said"
    return found.group(1)


def _fix(panel: str) -> str:
    return text(re.search(r'<summary class="sd-fixt">(.*?)</summary>', panel, re.S).group(1))


def _counted(panel: str) -> str:
    found = re.search(r'<h3 class="sd-h">Counted .*?</section>', panel, re.S)
    assert found, "no Counted"
    return found.group(0)


def _sub(listed: str) -> str | None:
    return _one(r'<p class="sd-sub">(.*?)</p>', listed)


def _rows(listed: str) -> list[str]:
    return re.findall(r'<li class="sd-row" data-window="([^"]+)"', listed)


def _row(listed: str, window_id: str) -> str:
    return listed.split(f'<li class="sd-row" data-window="{window_id}"')[1].split("</li>")[0]


def _row_stat(row: str) -> str | None:
    return _one(r'<span class="sd-sz">(.*?)</span>', row)


def _settle(world, peaks: dict, *, stream: str | None = None, **kw):
    """Only this stream's made-up counts, with ``peaks`` in its Moments' anchor bins."""
    clear_activity(world.store)
    a = activity_line(world, stream or world.new, peaks=peaks, **kw)
    assert store_activity(world.store, a) == "written"
    return a


def _views(c, stream: str, window_id: str, *, order: str = "") -> tuple[str, str, str]:
    """The glance, the All N list (in the lane) and one opened Moment (in the lane)."""
    home = c.get(f"/?stream={stream}").text
    listed = c.get(f"/list?stream={stream}&panel=1" + (f"&order={order}" if order else "")).text
    panel = c.get(f"/m/{window_id}?panel=1").text
    return home, listed, panel


def _assert_counts_off(world, home: str, listed: str, panels: list[str], q10: str, n: int, stream: str) -> None:
    """With the counts missing or expired: the First headings, the Q10 line in place of the line and in
    the stat slot and Counted, no stat, no face and no gap phrase anywhere."""
    firsts = [w for w in sorted(_rows(listed), key=lambda w: world.windows[w].anchor_at)][:3]
    assert _eyebrow(home) == f"First 3 of {n} Moments" and _heading(home) == "First three"
    assert _gone(home) == q10 and "gl-plot" not in home and 'class="gl-lane"' not in home and CAPTION not in home
    assert _cards(home) == firsts
    for w in firsts:
        assert _stat(_card(home, w)) is None
    assert _sub(listed) == "In time order" and 'class="sd-ord"' not in listed and 'class="sd-sz"' not in listed
    assert text(listed).count(FOOT) == 1
    for page in (home, listed, *panels):
        for never in ("people in 10 s", "usual pace", "people chatting", "Coverage unknown", "coverage unknown",
                      "not recorded", "fc-chip", "fc-src", FACE_CAPTION):  # fmt: skip
            assert never not in page, never
    for panel in panels:
        assert _slot(panel) == q10
        assert q10 in text(_counted(panel))
    other = EXPIRED_COUNTS if q10 == NO_COUNTS else NO_COUNTS
    assert all(other not in text(p) for p in (home, listed, *panels))


# --- the states ---------------------------------------------------------------------------------------


def test_accept_kept(world):
    m = world.new_moments
    n = len(m)
    _settle(world, {m[2]: 81, m[0]: 60, m[5]: 40})
    c = client(world)
    home, listed, panel = _views(c, world.new, m[2])
    # the glance: eyebrow, status, the line and its caption, the busiest three by peak
    assert _eyebrow(home) == f"Busiest 3 of {n} Moments" and _heading(home) == "Busiest three"
    assert _status(home) == f"Not sure about {n} of {n} Moments. No model yet."
    assert CAPTION in text(home) and _gone(home) is None
    assert _cards(home) == [m[2], m[0], m[5]]
    assert "<b>81</b> people in 10 s · <b>9×</b> usual pace" in _card(home, m[2])
    assert _stat(_card(home, m[0])) == "60 people in 10 s · 6.7× usual pace"
    day = app_mod.day(world.store.window(m[0]).text_expires_at)
    assert _kept(home) == f"chat text kept until {day}"
    assert _menu_kept(home) == f"Chat text kept until {day}"
    assert re.search(rf'<a class="gl-all" href="[^"]+" data-open>All {n} Moments</a>', home)
    for w in _cards(home):
        quote = _quote(_card(home, w))
        assert quote and not quote.startswith("Chat text expired")  # one line of chat, kept
    assert 'placeholder="Ask about this stream"' in home and "Try one" in text(home)
    # the list: busiest first, the toggle, the stat, the footer
    assert f'<h2 class="sd-lh" tabindex="-1">All {n} Moments</h2>' in listed
    assert _sub(listed) == "Busiest first" and _rows(listed)[:3] == [m[2], m[0], m[5]]
    assert _row_stat(_row(listed, m[2])) == "81 people in 10 s · 9× usual pace"
    assert ">In time order</a>" in listed
    assert text(listed).count(FOOT) == 1
    # the Moment: timing line, not sure title (no model), both stats, three kept lines
    assert WHEN in text(_head(panel))
    assert _title(panel).startswith(NOT_SURE) and "sd-tag" not in _head(panel)
    assert _slot(panel) == "81 people chatting in the busiest 10 s · 9× the usual pace"
    said = _said(panel)
    assert f"Chat text kept until {day}" in text(said)
    assert re.search(rf'<p class="sd-k" id="sd-k">3 of {len(world.windows[m[2]].messages)} lines</p>', said)
    assert NOT_SURE_LINE in text(panel) and _fix(panel) == "Know what chat did? Fix"
    assert "Labels and counts remain." not in panel and "Chat text expired" not in panel


def test_accept_text_expired(world):
    old = world.old_moments
    later = NOW + timedelta(days=5)
    _settle(world, {old[0]: 30}, stream=world.old)
    c = client(world, now=later)  # page start sweeps the old stream's text
    assert world.store.window(old[0]).text_state == "expired"
    assert world.store.activity_state(world.old, now=later) == "active"
    day = app_mod.day(world.store.window(old[0]).text_expires_at)
    home, listed, panel = _views(c, world.old, old[0])
    assert _eyebrow(home) == f"Busiest 3 of {len(old)} Moments" and _gone(home) is None
    assert _kept(home) == f"chat text expired {day}"
    assert _menu_kept(home) == f"Chat text expired {day}"
    for w in _cards(home):
        assert _quote(_card(home, w)) == f"Chat text expired {day}"
    assert _stat(_card(home, old[0])).startswith("30 people in 10 s")  # counts remain
    assert _sub(listed) == "Busiest first" and _row_stat(_row(listed, old[0])).startswith("30 people in 10 s")
    assert text(_said(panel)) == f"What chat said Chat text expired {day}. Labels and counts remain."
    assert _slot(panel).startswith("30 people chatting in the busiest 10 s")
    assert "Chat text kept until" not in panel and 'class="sd-ln"' not in panel
    for page in (home, listed, panel):
        for t in (x.text for w in old for x in world.windows[w].messages):
            assert f">{t}<" not in page  # no chat line survives its text
    assert "chat text kept until" not in home


def test_accept_text_and_counts_expired(world):
    old = world.old_moments
    n = len(old)
    later = NOW + timedelta(days=5)
    src = next(s for s in world.export.streams if s.stream_id == world.old)
    due = later + timedelta(hours=1)  # the counts' 30 days end an hour after the text has gone
    _settle(world, {old[0]: 30}, stream=world.old, shift=src.started_at + timedelta(days=30) - due)
    now = [later]
    c = client(world, clock=lambda: now[0])
    day = app_mod.day(world.store.window(old[0]).text_expires_at)
    home, _, panel = _views(c, world.old, old[0])
    assert _eyebrow(home).startswith("Busiest") and _gone(home) is None
    assert f"Chat text expired {day}. Labels and counts remain." in text(_said(panel))

    def gone(c) -> None:
        home, listed, panel = _views(c, world.old, old[0])
        _assert_counts_off(world, home, listed, [panel], EXPIRED_COUNTS, n, world.old)
        assert _kept(home) == f"chat text expired {day}" and _menu_kept(home) == f"Chat text expired {day}"
        for w in _cards(home):
            assert _quote(_card(home, w)) == f"Chat text expired {day}"
        # the short line: nothing says counts remain
        assert f'<p class="sd-exp">Chat text expired {day}.</p>' in _said(panel)
        assert "Labels and counts remain." not in panel

    now[0] = due  # before the sweep: the counts are past their 30 days, the row is still stored
    assert world.store.conn.execute("SELECT COUNT(*) FROM stream_activity WHERE stream_id = ?", (world.old,)).fetchone()[0] == 1
    gone(c)
    after = client(world, now=due + timedelta(hours=1))  # page start sweeps the counts
    assert world.store.activity_state(world.old, now=due + timedelta(hours=1)) == "expired"
    gone(after)


def test_accept_counts_expired_before_and_after_the_sweep(world):
    m = world.new_moments
    n = len(m)
    src = next(s for s in world.export.streams if s.stream_id == world.new)
    due = NOW + timedelta(hours=1)  # the text is kept well past it
    shift = src.started_at + timedelta(days=30) - due
    a = _settle(world, {m[3]: 81}, shift=shift)
    assert a.started_at + timedelta(days=30) == due
    truth = fake.truth(world.raw)
    stub_model(world.store, world.models, RUNG0_ID, 0)
    for w in m:
        predict(world.store, w, RUNG0_ID, "doing", truth[w]["doing"], 0.9)  # sure: faces could show
    now = [due - timedelta(minutes=1)]
    c = client(world, clock=lambda: now[0])
    live = c.get(f"/?stream={world.new}").text
    assert _eyebrow(live) == f"Busiest 3 of {n} Moments" and _gone(live) is None
    assert _status(live) == f"Sure about all {n} Moments. Guessed on this PC."

    def expired(c) -> None:
        home, listed, panel = _views(c, world.new, m[3])
        _assert_counts_off(world, home, listed, [panel, c.get(f"/m/{m[0]}?panel=1").text], EXPIRED_COUNTS, n, world.new)
        day = app_mod.day(world.store.window(m[0]).text_expires_at)
        assert _kept(home) == f"chat text kept until {day}"
        assert f"Chat text kept until {day}" in text(_said(panel))
        assert _status(home) == f"Sure about all {n} Moments. Guessed on this PC."

    now[0] = due  # before the sweep
    assert world.store.conn.execute("SELECT COUNT(*) FROM stream_activity WHERE stream_id = ?", (world.new,)).fetchone()[0] == 1
    expired(c)
    after = client(world, now=due + timedelta(hours=1))  # page start sweeps the counts
    assert world.store.conn.execute("SELECT COUNT(*) FROM stream_activity WHERE stream_id = ?", (world.new,)).fetchone()[0] == 0
    expired(after)


def test_accept_counts_missing(world):
    m = world.new_moments
    n = len(m)
    clear_activity(world.store)
    c = client(world)
    home, listed, panel = _views(c, world.new, m[0])
    _assert_counts_off(world, home, listed, [panel, c.get(f"/m/{m[1]}?panel=1").text], NO_COUNTS, n, world.new)
    assert _rows(listed) == m and _rows(c.get(f"/list?stream={world.new}&order=busy&panel=1").text) == m
    assert _kept(home).startswith("chat text kept until ")


def test_accept_untrained(world):
    m = world.new_moments
    n = len(m)
    _settle(world, {m[0]: 81, m[1]: 60, m[2]: 40})
    c = client(world)
    home, listed, panel = _views(c, world.new, m[0])
    assert _status(home) == f"Not sure about {n} of {n} Moments. No model yet."
    assert '<i class="gl-gns" aria-hidden="true"></i>' in home
    for w in _cards(home):
        card = _card(home, w)
        assert _say(card).startswith(NOT_SURE) and 'class="gl-num gl-ns"' in card
        assert _stat(card)  # a not-sure card keeps its time and stat (call 7a)
    for w in _rows(listed):
        row = _row(listed, w)
        assert _one(r'<span class="sd-lsay">(.*?)</span>', row).startswith(NOT_SURE) and "sd-num sd-ns" in row
    assert _title(panel).startswith(NOT_SURE) and NOT_SURE_LINE in text(panel)
    assert _fix(panel) == "Know what chat did? Fix"
    for page in (home, listed, panel):
        for never in ("model guess", "sd-you", "gl-you", "Guessed on this PC.", "made-up labels", "fc-chip", "fc-src",
                      FACE_CAPTION):  # fmt: skip
            assert never not in page, never
    assert all(", not sure" in x for x in re.findall(r'<a class="gl-(?:c3|mk)[^"]*"[^>]*aria-label="([^"]+)"', home))


def test_accept_made_up_model(world):
    m = world.new_moments
    n = len(m)
    _settle(world, {m[0]: 81, m[1]: 60, m[4]: 40})
    truth = fake.truth(world.raw)
    stub_model(world.store, world.models, RUNG0_ID, 0)  # a synthetic model: made-up labels
    for w in m:
        predict(world.store, w, RUNG0_ID, "doing", truth[w]["doing"], 0.2 if w == m[1] else 0.9)
    c = client(world)
    home, listed, panel = _views(c, world.new, m[0])
    assert _status(home) == f"Not sure about 1 of {n} Moments. Guessed on this PC."
    header = home.split("<header", 1)[1].split("</header>")[0]
    assert '<span class="bd" data-state="review">made-up labels</span>' in header and home.count("made-up labels") == 1
    # a sure guess carries "model guess" in the lane; the card has no tag; the unsure one keeps "Not sure"
    assert '<span class="sd-tag">model guess</span>' in _head(panel) and "sd-you" not in panel
    assert _fix(panel) == "Not right? Fix" and NOT_SURE_LINE not in panel
    assert "model guess" not in home and "gl-you" not in home
    assert _say(_card(home, m[1])).startswith(NOT_SURE) and not _say(_card(home, m[0])).startswith(NOT_SURE)
    unsure = c.get(f"/m/{m[1]}?panel=1").text
    assert _title(unsure).startswith(NOT_SURE) and "sd-tag" not in _head(unsure) and NOT_SURE_LINE in text(unsure)
    # F1: the caption shows only with a face on a shown card, and each face reads the copy table's sentence
    faced = [w for w in _cards(home) if "fc-chip" in _card(home, w)]
    assert (FACE_CAPTION in home) == bool(faced)
    for w in faced:
        sr = _one(r'<span class="fc-sr">\. (.*?)\.</span>', _card(home, w))
        assert sr and FACE.fullmatch(sr), sr
    for w in _rows(listed):
        if "fc-chip" in _row(listed, w):
            assert FACE.fullmatch(_one(r'<span class="fc-sr">\. (.*?)\.</span>', _row(listed, w)))
    # every Moment sure (the owner labels the unsure one): the "Sure about all" status
    label(world.store, m[1], {"doing": truth[m[1]]["doing"]})
    home = c.get(f"/?stream={world.new}").text
    assert _status(home) == f"Sure about all {n} Moments. Guessed on this PC."
    assert '<i class="gl-gns"' not in home
    # an answer's settled line keeps the made-up suffix (F3)
    r = c.post("/ask", data={"q": "what made chat laugh the most"}, headers=ORIGIN, follow_redirects=False)
    assert r.status_code == 303, r.text
    answer = c.get(r.headers["location"].split("#")[0]).text
    settled = re.findall(r'<details class="stp"[^>]*><summary>(.*?)</summary>', answer, re.S)
    assert settled and all(re.fullmatch(r"✓ \d+ steps? · \d+\.\d s · made-up labels( · .+)?", text(s)) for s in settled), settled


def test_accept_your_label(world):
    m = world.new_moments
    _settle(world, {m[0]: 81, m[1]: 60, m[2]: 40})
    label(world.store, m[0], {"doing": ["laughing"]})
    c = client(world)
    home, listed, panel = _views(c, world.new, m[0])
    card = _card(home, m[0])
    assert '<span class="gl-tag gl-you">your label</span>' in card and _say(card).startswith("Laughing")
    assert 'class="gl-num"' in card
    assert "your label" not in _card(home, m[1]) and "your label" not in _card(home, m[2])
    assert home.count("your label") == 1
    assert '<span class="sd-tag sd-you">your label</span>' in _row(listed, m[0]) and listed.count("sd-you") == 1
    assert _title(panel).endswith(" · your label") and "model guess" not in panel
    assert _fix(panel) == "Not right? Fix" and NOT_SURE_LINE not in panel
    n = len(m)
    assert _status(home) == f"Not sure about {n - 1} of {n} Moments. No model yet."


def test_accept_unknown_coverage(world):
    """A Moment outside the counts reads "coverage unknown" everywhere, never "not recorded" (R1)."""
    m = world.new_moments
    src = next(s for s in world.export.streams if s.stream_id == world.new)
    start = world.windows[m[0]].anchor_at + timedelta(seconds=100)  # the counts begin after the first Moment
    _settle(world, {m[1]: 60, m[2]: 40}, shift=src.started_at - start)
    c = client(world)
    home, listed, panel = _views(c, world.new, m[0])
    assert _cards(home)[:2] == [m[1], m[2]] and m[0] not in _cards(home)  # an unknown peak sorts last (G-3)
    rows = _rows(listed)
    assert rows[-1] == m[0] and _row_stat(_row(listed, m[0])) == "coverage unknown"
    assert _slot(panel) == "Coverage unknown at this Moment."
    said = _plot_label(home)
    assert re.search(r"Coverage unknown from \d\d:\d\d(:\d\d)? to \d\d:\d\d(:\d\d)?\.", said), said
    assert _gap_titles(home) == ["Coverage unknown"]
    for page in (home, listed, panel):
        assert "not recorded" not in page and "No chat recorded" not in page
    # no chat recorded at all in the counts: no Moment has a peak, so the cards are the first three by
    # time; the first, outside the counts, reads "coverage unknown", the next (a null bin) "not recorded"
    _settle(world, {}, base=None, shift=src.started_at - start)
    home = client(world).get(f"/?stream={world.new}").text
    assert _cards(home) == m[:3]
    assert _stat(_card(home, m[0])) == "coverage unknown" and _stat(_card(home, m[1])) == "not recorded"


def test_accept_known_gap(world):
    """A null run inside the counts is a known gap: "No chat recorded" / "not recorded", never unknown."""
    m = world.new_moments
    src = next(s for s in world.export.streams if s.stream_id == world.new)
    at = int((world.windows[m[0]].anchor_at - src.started_at).total_seconds()) // src.bin_s
    _settle(world, {m[1]: 60, m[2]: 40}, gaps=((at - 1, 4),))
    c = client(world)
    home, listed, panel = _views(c, world.new, m[0])
    assert m[0] not in _cards(home)[:2] and _stat(_card(home, _cards(home)[2])) != "coverage unknown"
    assert _row_stat(_row(listed, m[0])) == "not recorded"
    assert _slot(panel) == "Chat not recorded at this Moment."
    said = _plot_label(home)
    assert re.search(r"No chat recorded from \d\d:\d\d(:\d\d)? to \d\d:\d\d(:\d\d)?\.", said), said
    assert _gap_titles(home) == ["No chat recorded"]
    for page in (home, listed, panel):
        assert "Coverage unknown" not in page and "coverage unknown" not in page
    # no chat recorded anywhere in the counts: the first three by time, each "not recorded" on its card
    _settle(world, {}, base=None)
    home = client(world).get(f"/?stream={world.new}").text
    assert _cards(home) == m[:3] and all(_stat(_card(home, w)) == "not recorded" for w in m[:3])
    assert "Coverage unknown" not in home and "coverage unknown" not in home


def test_accept_empty_stream(world):
    with world.store.transaction() as conn:
        conn.execute("UPDATE windows SET anchor_kind = 'ordinary' WHERE stream_id = ?", (world.new,))
    home = client(world).get(f"/?stream={world.new}").text
    assert '<p class="gl-empty">No Moments in this stream.</p>' in home
    for gone in ("gl-eyebrow", "gl-status", "gl-plot", "gl-lane", "gl-card", "gl-h", 'id="gone"', "gl-all", CAPTION):
        assert gone not in home, gone
    assert '<details class="gl-sm"' in home and "Other streams" in text(home)  # other streams stay reachable
    assert 'placeholder="Ask about this stream"' in home


def test_accept_empty_store(tmp_path):
    store = Store(tmp_path / "cb.db")
    try:
        c = client(SimpleNamespace(store=store, tmp=tmp_path, models=tmp_path / "models"))
        home = c.get("/").text
        # no copy-table string covers it: Slice 1's approved line stays (Task 3 ruling)
        assert '<p class="sl">No windows yet. Import an export with <code>uv run chatbrain import</code>.</p>' in home
        for gone in ("gl-eyebrow", "gl-status", "gl-card", "gl-sm", EMPTY_STREAM, 'class="dock"', "made-up labels"):
            assert gone not in home, gone
        assert c.get("/list?stream=nope").status_code == 404
    finally:
        store.close()


def test_accept_no_usual_pace(world):
    m = world.new_moments
    _settle(world, {m[0]: 81}, base=0)  # under 30 bins above zero: no usual pace
    c = client(world)
    home, listed, panel = _views(c, world.new, m[0])
    assert _stat(_card(home, m[0])) == "81 people in 10 s"
    assert _row_stat(_row(listed, m[0])) == "81 people in 10 s"
    assert _slot(panel) == "81 people chatting in the busiest 10 s"
    glance = home.split("<main", 1)[1].split('<div class="dock">')[0]
    for page in (glance, listed, panel):
        assert "usual pace" not in page and "×" not in text(page.replace(">×</a>", "></a>"))  # the close link aside


def test_accept_long_label(world):
    m = world.new_moments
    _settle(world, {m[0]: 81})
    label(world.store, m[0], {"doing": ["advising", "laughing", "complaining"], "aimed_at": "tech_problem"})
    c = client(world)
    home, listed, panel = _views(c, world.new, m[0])
    title = _say(_card(home, m[0]))
    assert "advising (backseating)" in title and " · about a tech problem" in title and len(title) > 40
    assert _one(r'<span class="sd-lsay">(.*?)<span class="sr">', _row(listed, m[0])) == title  # whole, never cut
    assert _title(panel) == f"{title} · your label"
    css = (STATIC / "page.css").read_text(encoding="utf-8")
    rules = re.findall(r"([^{}]*)\{([^{}]*)\}", css)
    for cls in (".gl-say", ".sd-say", ".sd-lsay", ".gl-eyebrow", ".gl-status", ".sd-lh"):
        styled = 0
        for selector, body in rules:
            # rules on the title element itself (not on a tag chip inside it)
            if any(re.search(re.escape(cls) + r"(?![\w-])(?:\[[^\]]*\]|::?[\w-]+(?:\([^)]*\))?)*$", s.strip())
                   for s in selector.split(",")):  # fmt: skip
                styled += 1
                assert "nowrap" not in body and "ellipsis" not in body, selector
                assert "height:" not in body.replace("line-height:", "").replace("max-height:none", ""), selector
        assert styled, cls  # the check read real rules


def test_accept_stream_button_stays_on_one_line():
    """Browser check finding: at 1280 x 800 the stream button's text wrapped ("7 h 36" / "m") inside its
    fixed 32 px height, so the last word spilled out below the bar. A fixed-height button keeps one line."""
    css = (STATIC / "page.css").read_text(encoding="utf-8")
    rules = [(s.strip(), b) for s, b in re.findall(r"([^{}]*)\{([^{}]*)\}", css) if s.strip() == ".gl-sb"]
    fixed = [b for _, b in rules if re.search(r"(?<![\w-])height:\d", b)]
    assert fixed, "the stream button rule with its fixed height"
    for body in fixed:
        assert "white-space:nowrap" in body, body
    # the cause was a negative margin on the summary, which shrank its flex item 6 px below the
    # button's own width: the pull-left sits on the item now, and the button keeps its full width
    for _, body in rules:
        assert "margin-left:-" not in body, body
    sm = [b for sel, b in re.findall(r"([^{}]*)\{([^{}]*)\}", css) if sel.strip() == ".gl-sm"]
    assert [b for b in sm if "margin-left:-6px" in b] and [b for b in sm if "margin-left:-8px" in b], sm


def test_accept_glance_sits_beside_an_open_lane():
    """G1-desk (.app.open): an open lane sits beside the glance, not over it. From 1024 px the column
    keeps clear of the 408 px lane, centred in what is left, as the dock does."""
    css = (STATIC / "page.css").read_text(encoding="utf-8")
    m = re.search(r"@media \(min-width:1024px\)\{\s*\.main:has\(>\.pn:not\(\[hidden\]\)\)>\.col\{([^{}]*)\}", css)
    assert m, "the column rule for an open lane"
    assert "max-width:min(1040px,calc(100% - 408px))" in m.group(1)
    assert "margin-left:max(0px,calc((100% - 408px - 1040px) / 2))" in m.group(1)


def test_accept_motion_leaves_no_dom_behind():
    """The motion DOM matches the reduced-motion DOM once motion ends: the guide's drop class comes off
    on animationend, and the lift's name goes without leaving an empty style attribute."""
    js = (STATIC / "page.js").read_text(encoding="utf-8")
    assert "viewTransitionName = \"\"" not in js
    assert 'removeProperty("view-transition-name")' in js and 'removeAttribute("style")' in js
    assert re.search(r'"animationend".*?animationName === "gl-drop".*?classList\.remove\("gl-drop"\)', js, re.S)
