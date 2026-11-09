"""F1 faces on the page (Task 5): each busy Moment's top emote typed, as a text chip.

The chip shows on the card (in the number row, after the time), in the All N list's face column
and, as a sentence, in the card's sr-only text, the busiest lane chip's name, a desk note and a
muted line under the opened Moment's time. It shows only for a live tally that passes the bar
(``faces.shows``) on a Moment whose Doing is sure. No pictures, no network. Made-up worlds only
(``fake`` emote ids ``fake-<name>``); the data design note (section 6) names the states (A1-A13).
"""

import html
import re
import socket
from datetime import timedelta
from pathlib import Path

import pytest

from chatbrain import faces
from chatbrain.page import app as app_mod
from chatbrain.store import ACTIVITY_LIFETIME
from pagekit import (
    NOW, SCHEME, activity_line, build_world, clear_activity, client, label, store_activity, text, truth_labels,
)  # fmt: skip

STATIC = Path(app_mod.__file__).parent / "static"
TEMPLATES = Path(app_mod.__file__).parent / "templates"
CAPTION = '<span class="fc-cap"> · top emotes typed</span>'
NO_COUNTS = "This stream's export has no activity counts."


@pytest.fixture
def world(tmp_path):
    w = build_world(tmp_path)
    yield w
    w.store.close()


def _peaked(world, peaks: dict):
    """The new stream's counts with ``peaks``, and its Moment tallies counted again at NOW (so a
    later clock past the text's expiry still finds them, as an import at NOW would have left them)."""
    clear_activity(world.store)
    a = activity_line(world, world.new, peaks=peaks)
    assert store_activity(world.store, a) == "written"
    world.store.backfill_moment_emotes(NOW)
    return a


def _faced(world):
    """Every window labelled sure by the owner; the busiest three are m1 (KEKW, passes the bar), m4
    (Clap, passes) and m0 (its top emote has 4 people: below the bar)."""
    m = world.new_moments
    truth_labels(world)
    a = _peaked(world, {m[1]: 81, m[4]: 60, m[0]: 40})
    live = world.store.moment_faces(world.new, NOW)
    assert faces.shows(live[m[1]]) and faces.shows(live[m[4]]) and not faces.shows(live[m[0]])
    return m, a


def _sentence(world, wid: str) -> tuple[faces.Tally, str]:
    """The tally recounted from the made-up export's messages, and its sentence (exact copy)."""
    t = faces.top_emote(world.windows[wid])
    at = app_mod.hm(world.windows[wid].anchor_at)
    return t, f"{t.name} · in {t.lines} of {t.of_lines} lines · sent by {t.people} people · the 120 s around {at}"


def _cards(page: str) -> list[str]:
    return re.findall(r'<li class="gl-card" data-window="([^"]+)"', page)


def _card(page: str, wid: str) -> str:
    return page.split(f'<li class="gl-card" data-window="{wid}"')[1].split("</li>")[0]


def _row(page: str, wid: str) -> str:
    return page.split(f'<li class="sd-row" data-window="{wid}">')[1].split("</li>")[0]


def _chips(fragment: str) -> list[str]:
    """The chips' visible names (the note inside each chip is dropped)."""
    out = []
    for inner in re.findall(r'<span class="fc-chip" aria-hidden="true">(.*?)<span class="fc-note', fragment, re.S):
        out.append(html.unescape(inner))
    return out


def _notes(fragment: str) -> list[str]:
    return [text(x) for x in re.findall(r'<span class="fc-note[^"]*"[^>]*>(.*?)</span>', fragment, re.S)]


def _sr(fragment: str) -> list[str]:
    return [html.unescape(x) for x in re.findall(r'<span class="fc-sr">(.*?)</span>', fragment, re.S)]


def _src(fragment: str) -> list[str]:
    return [text(x) for x in re.findall(r'<p class="fc-src">(.*?)</p>', fragment, re.S)]


def _lane(page: str) -> dict[str, tuple[str, str, str]]:
    """href -> (class, aria-label, inner HTML) for every mark on the lane."""
    lane = page.split('<div class="gl-lane"')[1].split("</div>")[0]
    found = re.findall(r'<a class="([^"]+)" href="([^"]+)"[^>]*aria-label="([^"]*)">(.*?)</a>', lane, re.S)
    return {href: (cls, html.unescape(lab), inner) for cls, href, lab, inner in found}


def _no_face(page: str) -> None:
    for gone in ("fc-chip", "fc-note", "fc-sr", "fc-src", "fc-cap", "top emotes typed", " lines · sent by "):
        assert gone not in page, gone


# --- A1: live counts, sure Doing, the tally passes ---------------------------------------------------


def test_face_on_card_row_lane_and_panel(world):
    m, _ = _faced(world)
    c = client(world)
    page = c.get("/").text
    assert _cards(page) == [m[1], m[4], m[0]]
    assert CAPTION in page  # after the h2, before the kept line
    assert page.index('<h2 class="gl-h"') < page.index(CAPTION) < page.index('<span class="gl-kept">')
    lane = _lane(page)
    for wid in (m[1], m[4]):
        t, said = _sentence(world, wid)
        assert world.store.moment_faces(world.new, NOW)[wid] == t  # the stored numbers equal the recount
        card = _card(page, wid)
        assert _chips(card) == [t.name]
        # in the number row, after the time
        assert card.index('class="gl-time"') < card.index('class="fc-chip"') < card.index('class="gl-say"')
        assert _sr(card) == [f". {said}."]
        cls, lab, _inner = lane[app_mod.moment_url(wid)]
        assert cls.startswith("gl-c3")
        assert lab == f"Moment {_n(page, wid)}, {app_mod.clock_text(world.windows[wid].anchor_at)}. {said}"
    below = _card(page, m[0])
    assert _chips(below) == [] and _sr(below) == [] and "fc-" not in below
    for href, (cls, lab, _inner) in lane.items():
        if href not in (app_mod.moment_url(m[1]), app_mod.moment_url(m[4])):
            assert " lines · sent by " not in lab, href  # small marks and the faceless chip keep their names
    # the All N list: a face column, the chip in it, an empty cell where there is none
    lst = c.get("/list", params={"stream": world.new, "panel": 1}).text
    assert '<ol class="sd-rows fc-rows">' in lst
    for wid in world.new_moments:
        row = _row(lst, wid)
        assert row.count('<span class="fc-cell">') == 1, wid
        if wid in (m[1], m[4]):
            t, said = _sentence(world, wid)
            assert _chips(row) == [t.name] and _sr(row) == [f". {said}."]
            assert row.index('class="sd-num') < row.index('class="fc-cell"') < row.index('class="sd-time"')
        else:
            assert '<span class="fc-cell"></span>' in row and _sr(row) == []
    # the opened Moment: a muted line under the time, in the lane and on the full page
    for wid in (m[1], m[4]):
        _t, said = _sentence(world, wid)
        for url in (f"/m/{wid}?panel=1", f"/m/{wid}"):
            p = c.get(url).text
            assert _src(p) == [said], url
            assert p.index('class="sd-top"') < p.index('class="fc-src"') < p.index('class="sd-when"')
    _no_face(c.get(f"/m/{m[0]}?panel=1").text)
    # the glance data: one live tally per Moment, shown only past the bar with a sure Doing
    g = _glance(world)
    assert g.faced is True
    assert [x.face_shown for x in g.top] == [True, True, False]


def _n(page: str, wid: str) -> str:
    return re.search(rf'<li class="gl-card" data-window="{re.escape(wid)}" data-n="(\d+)"', page).group(1)


def _glance(world, now=NOW) -> app_mod.Glance:
    page = app_mod.Page(
        world.store, SCHEME, world.models, imported_dir=world.tmp / "imported", clock=lambda: now,
        review_ids=frozenset(),
    )  # fmt: skip
    stream = next(s for s in page.streams(now) if s.stream_id == world.new)
    return app_mod.glance(page, stream, now)


# --- A2 / A3: below the bar --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lines,of_lines,people",
    [(5, 60, 5), (10, 30, 4)],  # A2: under 10% of the lines (5 people); A3: 4 people (a third of the lines)
)
def test_below_bar_no_chip_anywhere(world, lines, of_lines, people):
    m, _ = _faced(world)
    with world.store.transaction() as conn:
        conn.execute(
            "UPDATE moment_emotes SET lines = ?, of_lines = ?, people = ? WHERE stream_id = ?",
            (lines, of_lines, people, world.new),
        )
    c = client(world)
    page = c.get("/").text
    _no_face(page)
    assert all(" lines · sent by " not in lab for _c, lab, _i in _lane(page).values())
    _no_face(c.get("/list", params={"stream": world.new, "panel": 1}).text)
    for wid in m:
        _no_face(c.get(f"/m/{wid}?panel=1").text)
    assert _glance(world).faced is False


# --- A6: untrained (no model, no labels) ---------------------------------------------------------------


def test_untrained_no_face(world):
    m = world.new_moments
    _peaked(world, {m[1]: 81, m[4]: 60, m[0]: 40})
    assert faces.shows(world.store.moment_faces(world.new, NOW)[m[1]])  # the count is there, the face is not
    c = client(world)
    page = c.get("/").text
    says = [text(x) for x in re.findall(r'<span class="gl-say">(.*?)</span>', page, re.S)]
    assert len(says) == 3 and all(s.startswith("Not sure what chat was doing") for s in says)
    assert "No model yet." in page and "model guess" not in page
    _no_face(page)
    _no_face(c.get("/list", params={"stream": world.new, "panel": 1}).text)
    _no_face(c.get(f"/m/{m[1]}?panel=1").text)
    _no_face(c.get(f"/m/{m[1]}").text)


# --- A7: one Moment not sure, the others keep their faces -----------------------------------------------


def test_not_sure_moment_loses_its_face_only(world):
    m, _ = _faced(world)
    label(world.store, m[1], {"aimed_at": "streamer"}, not_sure={"doing": ["conflicting"]}, at=5)
    c = client(world)
    page = c.get("/").text
    gone = _card(page, m[1])
    assert text(re.search(r'<span class="gl-say">(.*?)</span>', gone, re.S).group(1)).startswith(
        "Not sure what chat was doing"
    )
    assert _chips(gone) == [] and _sr(gone) == []
    t4, said4 = _sentence(world, m[4])
    assert _chips(_card(page, m[4])) == [t4.name] and _sr(_card(page, m[4])) == [f". {said4}."]
    assert CAPTION in page  # a shown card still has one
    assert " lines · sent by " not in _lane(page)[app_mod.moment_url(m[1])][1]
    lst = c.get("/list", params={"stream": world.new, "panel": 1}).text
    assert '<span class="fc-cell"></span>' in _row(lst, m[1]) and _chips(_row(lst, m[4])) == [t4.name]
    _no_face(c.get(f"/m/{m[1]}?panel=1").text)
    assert _src(c.get(f"/m/{m[4]}?panel=1").text) == [said4]


# --- A8: chat text expired, counts live --------------------------------------------------------------------


def test_text_expired_keeps_the_face(world):
    m, a = _faced(world)
    news = [w for w in world.export.windows if w.stream_id == world.new]
    later = max(w.text_expires_at for w in news) + timedelta(days=1)
    assert later < a.started_at + ACTIVITY_LIFETIME  # the counts (and the tallies) still live
    c = client(world, now=later)  # page start sweeps the text away
    assert world.store.conn.execute(
        "SELECT COUNT(*) FROM messages WHERE window_id IN (?, ?)", (m[1], m[4])
    ).fetchone()[0] == 0  # fmt: skip
    page = c.get("/").text
    t, said = _sentence(world, m[1])
    card = _card(page, m[1])
    assert _chips(card) == [t.name] and _sr(card) == [f". {said}."]
    assert "Chat text expired" in text(card)
    assert CAPTION in page
    p = c.get(f"/m/{m[1]}?panel=1").text
    assert _src(p) == [said] and "Chat text expired" in text(p)
    assert world.store.conn.execute(
        "SELECT COUNT(*) FROM moment_emotes WHERE window_id = ?", (m[1],)
    ).fetchone()[0] == 1  # fmt: skip


# --- A9: counts expired at started_at + 30 d, before and after the sweep -----------------------------------


def test_counts_expired_hide_faces_before_and_after_the_sweep(world):
    m, a = _faced(world)
    due = a.started_at + ACTIVITY_LIFETIME  # the boundary is expired
    clock = [NOW]
    c = client(world, clock=lambda: clock[0])  # page start at NOW: no sweep at the deadline yet
    assert CAPTION in c.get("/").text
    clock[0] = due
    rows = world.store.conn.execute("SELECT COUNT(*) FROM moment_emotes WHERE stream_id = ?", (world.new,))
    assert rows.fetchone()[0] > 0  # still stored: only the read filter hides them
    for url in ("/", f"/m/{m[1]}?panel=1", f"/m/{m[1]}"):
        _no_face(c.get(url).text)
    _no_face(c.get("/list", params={"stream": world.new, "panel": 1}).text)
    assert "First three" in c.get("/").text
    g = _glance(world, now=due)
    assert g.faced is False and not any(x.face_shown for x in g.top)
    # after the sweep: the rows are gone, the marker stays, and still no face
    after = client(world, now=due)
    rows = world.store.conn.execute("SELECT COUNT(*) FROM moment_emotes WHERE stream_id = ?", (world.new,))
    assert rows.fetchone()[0] == 0
    marked = world.store.conn.execute("SELECT 1 FROM stream_activity_expired WHERE stream_id = ?", (world.new,))
    assert marked.fetchone() is not None
    _no_face(after.get("/").text)
    _no_face(after.get(f"/m/{m[1]}?panel=1").text)


# --- A10: counts missing (a /1 export): faces with the proxy deadline -------------------------------------


def test_missing_counts_show_faces_in_the_first_three(tmp_path):
    world = build_world(tmp_path, activity=False)
    try:
        m = world.new_moments
        truth_labels(world)
        c = client(world)
        page = c.get("/").text
        assert NO_COUNTS in html.unescape(page)
        assert _cards(page) == m[:3]  # the first three by time
        assert "First three" in page and CAPTION in page
        t, said = _sentence(world, m[1])
        card = _card(page, m[1])
        assert _chips(card) == [t.name] and _sr(card) == [f". {said}."]
        assert 'class="gl-size"' not in card  # no stat with the counts off
        lst = c.get("/list", params={"stream": world.new, "panel": 1}).text
        t4, _ = _sentence(world, m[4])
        assert _chips(_row(lst, m[1])) == [t.name] and _chips(_row(lst, m[4])) == [t4.name]
        assert _src(c.get(f"/m/{m[1]}?panel=1").text) == [said]
        # the proxy deadline: the earliest window start + 30 days
        first = min(w.start_at for w in world.export.windows if w.stream_id == world.new)
        at_proxy = client(world, now=first + ACTIVITY_LIFETIME)
        _no_face(at_proxy.get(f"/?stream={world.new}").text)
    finally:
        world.store.close()


# --- A13: no network, no picture ----------------------------------------------------------------------------


def test_faces_open_no_socket_and_draw_no_picture(world, monkeypatch):
    m, _ = _faced(world)

    def refuse(*_a, **_k):
        raise AssertionError("the page opened a socket")

    real_connect = socket.socket.connect

    def loopback_only(self, address):
        # The test client's event loop pairs two loopback sockets for its own wake-ups (Windows has
        # no native socketpair); anything that leaves this PC is refused.
        if not (isinstance(address, tuple) and address[0] in ("127.0.0.1", "::1")):
            refuse()
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", loopback_only)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    c = client(world)
    csp = app_mod.SECURITY_HEADERS["Content-Security-Policy"]
    assert "img-src 'self'" in csp and "default-src 'self'" in csp
    for url in ("/", f"/list?stream={world.new}&panel=1", f"/m/{m[1]}?panel=1", f"/m/{m[1]}"):
        r = c.get(url)
        assert r.status_code == 200 and r.headers["content-security-policy"] == csp
        assert "<img" not in r.text
        assert "fc-" in r.text, url
        for part in re.findall(r'<(?:span|p) class="fc-[^"]*"[^>]*>.*?</(?:span|p)>', r.text, re.S):
            assert "http" not in part and "src=" not in part and "url(" not in part, part
        lane = r.text.split('<div class="gl-lane"')[1].split("</div>")[0] if 'class="gl-lane"' in r.text else ""
        assert "http" not in lane
    # the picture stage's seam exists and gives nothing; no template draws a picture
    assert faces.face_picture("twitch", "fake-kekw") is None
    assert faces.face_picture(None, "fake-kekw") is None
    for tpl in TEMPLATES.glob("*.html"):
        body = tpl.read_text(encoding="utf-8")
        assert "<img" not in body and "face_picture" not in body, tpl.name


# --- escaping -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["<b>x</b>", '"onload=alert(1)'])
def test_face_name_escaped(world, name):
    m, _ = _faced(world)
    with world.store.transaction() as conn:
        conn.execute("UPDATE moment_emotes SET name = ? WHERE window_id = ?", (name, m[1]))
    c = client(world)
    esc = html.escape(name, quote=True).replace("&quot;", "&#34;").replace("&#x27;", "&#39;")
    page = c.get("/").text
    lst = c.get("/list", params={"stream": world.new, "panel": 1}).text
    panel = c.get(f"/m/{m[1]}?panel=1").text
    full = c.get(f"/m/{m[1]}").text
    for p in (page, lst, panel, full):
        assert name not in p  # never raw, in text or in an attribute
    card, row = _card(page, m[1]), _row(lst, m[1])
    for where, frag in (("card", card), ("row", row)):
        assert _chips(frag) == [name], where  # the chip (unescaped back, so it was escaped once)
        assert f'<span class="fc-chip" aria-hidden="true">{esc}<span class="fc-note' in frag, where
        assert _notes(frag)[0].startswith(name), where  # the desk note
        assert _sr(frag)[0].startswith(f". {name} · in "), where  # the sr text
        assert f"<b>{esc}</b>" in frag, where
    lab = _lane(page)[app_mod.moment_url(m[1])][1]
    assert f". {name} · in " in lab  # the lane chip's name, escaped in its attribute
    assert re.search(r'aria-label="[^"]*' + re.escape(esc) + r' · in ', page)
    for p in (panel, full):
        assert _src(p)[0].startswith(f"{name} · in ")
        assert f'<p class="fc-src"><b>{esc}</b>' in p
    assert 'style="' not in "".join(re.findall(r'<span class="fc-chip[^>]*>', page + lst))  # never in a style


# --- the desk note -------------------------------------------------------------------------------------------


def _rules(css: str, selector_part: str) -> list[str]:
    return [b for s, b in re.findall(r"([^{}]+)\{([^{}]*)\}", css) if selector_part in s]


def test_face_note_markup(world):
    m, _ = _faced(world)
    c = client(world)
    page = c.get("/").text
    lst = c.get("/list", params={"stream": world.new, "panel": 1}).text
    for wid in (m[1], m[4]):
        _t, said = _sentence(world, wid)
        # the note sits inside the chip (hidden from the reader, which has the sr text) and names the source
        assert _notes(_card(page, wid)) == [said]
        assert _notes(_row(lst, wid)) == [said]
        cls, _lab, inner = _lane(page)[app_mod.moment_url(wid)]
        assert cls.startswith("gl-c3") and _notes(inner) == [said]
        assert re.search(r'<span class="fc-note(?: fc-[lr])?" aria-hidden="true">', inner)
    # no note on a faceless chip or a small mark
    for href, (cls, _lab, inner) in _lane(page).items():
        if href not in (app_mod.moment_url(m[1]), app_mod.moment_url(m[4])):
            assert "fc-note" not in inner, href
    css = (STATIC / "page.css").read_text(encoding="utf-8")
    note = " ".join(_rules(css, ".fc-note"))
    assert "pointer-events:none" in note
    base = re.search(r"\n\.fc-note\{([^}]*)\}", css)
    assert base and "display:none" in base.group(1)  # hidden until hover or focus
    shown = [s.strip() for s, b in re.findall(r"([^{}]+)\{([^{}]*)\}", css) if ".fc-note" in s and "display:block" in b]
    joined = " ".join(shown)
    assert ".fc-chip:hover" in joined and ".gl-c3:focus-visible" in joined
    # it unfolds in 120 ms only where motion is allowed; under reduced motion it appears at once
    motion = css.split("@media (prefers-reduced-motion:no-preference){")
    assert any(".fc-note" in blk.split("\n}\n")[0] and "var(--cb-d1)" in blk.split("\n}\n")[0] for blk in motion[1:])
    unmoved = css
    for blk in motion[1:]:
        unmoved = unmoved.replace(blk.split("\n}\n")[0], "")
    assert not re.search(r"\.fc-note[^{]*\{[^}]*animation", unmoved)
    assert "@keyframes fc-unfold" in css
    # the desk only: a phone has no hover, so the note never shows there
    phone = css.split("@media (max-width:759.98px){")
    assert any(".fc-note" in blk.split("\n}\n")[0] and "display:none" in blk.split("\n}\n")[0] for blk in phone[1:])
    # the chip: mono, a sunken background, a neutral ring, wraps, no tint
    chip = re.search(r"\n\.fc-chip\{([^}]*)\}", css)
    assert chip
    body = chip.group(1)
    for want in ("var(--cb-font-mono)", "var(--cb-bg-sunken", "inset 0 0 0 1px var(--cb-border)", "overflow-wrap:anywhere"):
        assert want in body, want
    for tint in ("accent", "--c:", "doing", "nowrap", "ellipsis"):
        assert tint not in body, tint
