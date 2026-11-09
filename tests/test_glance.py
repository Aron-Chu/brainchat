"""The glance home (Task 3): the top bar's stream menu, the eyebrow, the status line, the activity
line with its lane, and the busiest (or first) three Moments as cards. Made-up worlds only."""

import re
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from chatbrain import fake, ingest
from chatbrain.export_format import parse_export
from chatbrain.page import app as app_mod
from chatbrain.store import Store
from pagekit import (
    NOW, RUNG0_ID, SCHEME, activity_line, build_world, clear_activity, client, label, predict, store_activity,
    stub_model, text,
)  # fmt: skip

STATIC = Path(app_mod.__file__).parent / "static"
NO_COUNTS = "This stream's export has no activity counts."
EXPIRED_COUNTS = "Activity counts expired 30 days after this stream started."


@pytest.fixture
def world(tmp_path):
    w = build_world(tmp_path)
    yield w
    w.store.close()


def _g(world, stream_id: str, *, now=NOW) -> app_mod.Glance:
    page = app_mod.Page(
        world.store, SCHEME, world.models, imported_dir=world.tmp / "imported", clock=lambda: now,
        review_ids=frozenset(),
    )  # fmt: skip
    stream = next(s for s in page.streams(now) if s.stream_id == stream_id)
    return app_mod.glance(page, stream, now)


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


def _cards(page: str) -> list[str]:
    return re.findall(r'<li class="gl-card" data-window="([^"]+)"', page)


def _card(page: str, window_id: str) -> str:
    return page.split(f'<li class="gl-card" data-window="{window_id}"')[1].split("</li>")[0]


def _stat(page: str, window_id: str) -> str | None:
    return _one(r'<span class="gl-size">(.*?)</span>\s*(?=<span class="gl-chat|</a>)', _card(page, window_id))


def _plot_label(page: str) -> str | None:
    found = re.findall(r'<div class="gl-plot" role="img" aria-label="([^"]+)"', page)
    assert len(found) <= 1
    return found[0] if found else None


def _peaked(world, peaks: dict, **kw):
    clear_activity(world.store)
    a = activity_line(world, world.new, peaks=peaks, **kw)
    assert store_activity(world.store, a) == "written"
    return a


def test_glance_kept_state(world):
    m = world.new_moments
    n = len(m)
    assert n >= 6
    _peaked(world, {m[2]: 81, m[0]: 60, m[5]: 40})
    page = client(world).get("/").text  # the newest stream by default
    assert _eyebrow(page) == f"Busiest 3 of {n} Moments"
    assert _heading(page) == "Busiest three"
    assert _status(page) == f"Not sure about {n} of {n} Moments. No model yet."
    assert re.search(r'<p class="gl-status">.*?<b>Not sure about \d+ of \d+ Moments\.</b>', page, re.S)
    assert _cards(page) == [m[2], m[0], m[5]]  # by peak people in 10 s
    # bins of 9 with the three peaks: the usual pace is the median, 9 (C4)
    assert _stat(page, m[2]) == "81 people in 10 s · 9× usual pace"
    assert "<b>81</b> people in 10 s · <b>9×</b> usual pace" in _card(page, m[2])
    assert _stat(page, m[0]) == "60 people in 10 s · 6.7× usual pace"
    kept = _one(r'<span class="gl-kept">(.*?)</span>', page)
    assert kept and re.fullmatch(r"chat text kept until \w{3} \d{1,2}", kept)
    assert "Height: people chatting per 10 s" in page
    assert 'id="gone"' not in page and NO_COUNTS not in text(page) and EXPIRED_COUNTS not in text(page)
    assert f"All {n} Moments" in text(page)
    # the lane: chips for the busiest three, small marks for the rest
    lane = page.split('<div class="gl-lane"')[1].split("</div>")[0]
    assert 'role="group" aria-label="Moments on the line. Arrow keys move between them."' in lane
    assert lane.count('class="gl-c3') == 3 and lane.count('class="gl-mk') == n - 3
    # every mark is reachable without the script; page.js keeps one tab stop (roving)
    assert "tabindex" not in lane and lane.count(" data-x=") == n
    g = _g(world, world.new)
    assert (g.n, g.counts_ok, g.gone, len(g.top)) == (n, True, None, 3)
    assert [x.win.window_id for x in g.top] == [m[2], m[0], m[5]]
    assert [x.top for x in g.lane].count(True) == 3 and [x.n for x in g.lane] == list(range(1, n + 1))
    assert g.kept == kept and g.kept_menu == kept[:1].upper() + kept[1:]
    assert all(x.label.startswith(f"Moment {x.n}, ") and x.label.endswith(", not sure") for x in g.lane)
    # the old recap, list, legend and lane hint are gone
    for gone in ('class="rcl"', 'class="mcs"', 'class="act-key"', 'class="trk"', 'class="lane"', 'class="sl"',
                 "Moments open here", "act-tint"):  # fmt: skip
        assert gone not in page, gone


def test_glance_usual_none_drops_ratio(world):
    m = world.new_moments
    # under 30 bins above zero: no usual pace, so the card shows the peak alone
    _peaked(world, {m[0]: 81}, base=0)
    page = client(world).get("/").text
    assert _cards(page)[0] == m[0]
    assert _stat(page, m[0]) == "81 people in 10 s"
    assert "usual pace" not in page and "×" not in text(page.split('class="gl-t3"')[1])
    assert _stat(page, _cards(page)[1]) == "0 people in 10 s"  # zero is a count


def test_glance_counts_missing_first_three(world):
    m = world.new_moments
    n = len(m)
    clear_activity(world.store)
    page = client(world).get("/").text
    assert _eyebrow(page) == f"First 3 of {n} Moments"
    assert _heading(page) == "First three"
    assert _one(r'<p class="gl-gone" id="gone">(.*?)</p>', page) == NO_COUNTS
    assert _cards(page) == m[:3]
    for w in m[:3]:
        assert _stat(page, w) is None and "people" not in text(_card(page, w))
    assert "gl-plot" not in page and "Height: people chatting" not in page and 'class="gl-lane"' not in page
    assert EXPIRED_COUNTS not in page
    g = _g(world, world.new)
    assert (g.counts_ok, g.gone, g.line, g.peaks, g.gaps) == (False, NO_COUNTS, None, [], [])


def test_glance_counts_expired_first_three(world):
    m = world.new_moments
    n = len(m)
    a = _peaked(world, {m[3]: 81})
    due = a.started_at + timedelta(days=30)  # the counts expire 30 days after the stream started
    assert due > NOW
    now = [due - timedelta(microseconds=1)]
    c = client(world, clock=lambda: now[0])
    live = c.get("/").text
    assert _eyebrow(live) == f"Busiest 3 of {n} Moments" and _cards(live)[0] == m[3]

    def expired(page: str) -> None:
        assert _eyebrow(page) == f"First 3 of {n} Moments" and _heading(page) == "First three"
        assert _one(r'<p class="gl-gone" id="gone">(.*?)</p>', page) == EXPIRED_COUNTS
        assert NO_COUNTS not in text(page) and "gl-plot" not in page
        assert _cards(page) == m[:3]
        assert all(_stat(page, w) is None for w in m[:3])
        assert "people in 10 s" not in page and "usual pace" not in page

    now[0] = due  # before the sweep: the page clock has reached the deadline, the row is still stored
    assert world.store.conn.execute("SELECT COUNT(*) FROM stream_activity WHERE stream_id = ?", (world.new,)).fetchone()[0] == 1
    expired(c.get("/").text)
    after = client(world, now=due + timedelta(hours=1))  # page start sweeps the counts
    assert world.store.conn.execute("SELECT COUNT(*) FROM stream_activity WHERE stream_id = ?", (world.new,)).fetchone()[0] == 0
    expired(after.get("/").text)


def test_glance_untrained(world):
    n = len(world.new_moments)
    page = client(world).get("/").text
    assert _status(page) == f"Not sure about {n} of {n} Moments. No model yet."
    cards = _cards(page)
    assert len(cards) == 3
    for w in cards:
        card = _card(page, w)
        assert "Not sure what chat was doing" in _one(r'<span class="gl-say">(.*?)</span>', card)
        assert 'class="gl-num gl-ns"' in card  # a ring: not sure
    assert "model guess" not in page and "your label" not in page
    assert "Guessed on this PC." not in page


def test_glance_your_label_tag(world):
    m = world.new_moments
    _peaked(world, {m[0]: 81, m[1]: 60, m[2]: 40})
    label(world.store, m[0], {"doing": ["laughing"]})
    stub_model(world.store, world.models, RUNG0_ID, 0)
    predict(world.store, m[1], RUNG0_ID, "doing", ["cheering"], 0.9)
    page = client(world).get("/").text
    assert _cards(page) == [m[0], m[1], m[2]]
    mine = _card(page, m[0])
    assert '<span class="gl-tag gl-you">your label</span>' in mine
    assert _one(r'<span class="gl-say">(.*?)</span>', mine).startswith("Laughing · ")
    assert 'class="gl-num"' in mine  # solid: Doing is sure
    guessed = _card(page, m[1])
    assert "your label" not in guessed and "model guess" not in guessed
    assert _one(r'<span class="gl-say">(.*?)</span>', guessed).startswith("Cheering · ")
    assert "your label" not in _card(page, m[2])
    assert _status(page).endswith(" Guessed on this PC.")


def test_glance_empty_stream(world):
    with world.store.transaction() as conn:
        conn.execute("UPDATE windows SET anchor_kind = 'ordinary' WHERE stream_id = ?", (world.new,))
    page = client(world).get(f"/?stream={world.new}").text
    assert "No Moments in this stream." in text(page)
    assert _eyebrow(page) is None and _status(page) is None and _heading(page) is None
    assert "gl-plot" not in page and 'class="gl-lane"' not in page and not _cards(page) and 'id="gone"' not in page
    g = _g(world, world.new)
    assert (g.n, g.eyebrow, g.top, g.lane, g.cards) == (0, "", [], [], [])
    assert '<details class="gl-sm"' in page  # the stream menu still reaches the other streams


def test_glance_long_label_wraps(world):
    m = world.new_moments
    _peaked(world, {m[0]: 81})
    label(world.store, m[0], {"doing": ["laughing", "complaining", "advising"], "aimed_at": "tech_problem"})
    page = client(world).get("/").text
    title = _g(world, world.new).cards[0]["m"].title
    assert "advising (backseating)" in title and len(title) > 40
    assert _one(r'<span class="gl-say">(.*?)</span>', _card(page, m[0])) == title  # whole, never cut
    css = (STATIC / "page.css").read_text(encoding="utf-8")
    rules = re.findall(r"([^{}]*\.gl-[^{}]*)\{([^{}]*)\}", css)
    assert rules
    allowed = (".gl-tx", ".gl-sb")  # the one-line card quote; the stream button's chrome on a phone
    for selector, body in rules:
        if "nowrap" in body or "ellipsis" in body:
            assert all(s.strip().startswith(allowed) for s in selector.split(",")), selector
    for title_class in (".gl-say", ".gl-eyebrow", ".gl-h", ".gl-status", ".gl-gone", ".gl-pt"):
        for selector, body in rules:
            if title_class in selector:
                assert "nowrap" not in body and "ellipsis" not in body and "height:" not in body.replace("line-height:", ""), selector


def test_rail_gone_streams_in_menu(tmp_path):
    raw = fake.make_export(4, 6, 4, NOW, channels=2)
    (tmp_path / "e.json").write_bytes(raw)
    store = Store(tmp_path / "cb.db")
    try:
        ingest.import_export(store, tmp_path / "e.json", tmp_path / "imported", NOW)
        info: dict[str, dict] = {}
        for w in parse_export(raw).windows:
            s = info.setdefault(w.stream_id, {"channel": w.channel, "start": w.start_at, "end": w.end_at, "n": 0})
            s["start"], s["end"] = min(s["start"], w.start_at), max(s["end"], w.end_at)
            s["n"] += w.anchor_kind == "moment"
        channels = sorted({s["channel"] for s in info.values()})
        assert len(channels) == 2 and len(info) > len(channels)
        newest = sorted(info, key=lambda i: (-info[i]["end"].timestamp(), i))
        c = client(SimpleNamespace(store=store, tmp=tmp_path, models=tmp_path / "models"))
        other = newest[-1]
        some_moment = store.conn.execute(
            "SELECT window_id FROM windows WHERE stream_id = ? AND anchor_kind = 'moment' LIMIT 1", (other,)
        ).fetchone()[0]
        pages = {newest[0]: c.get("/").text, other: c.get(f"/?stream={other}").text}
        pages_m = {other: c.get(f"/m/{some_moment}").text}
        for current, page in [*pages.items(), *pages_m.items()]:
            for old in ('<aside class="rail"', "rail-b", "rail-scrim", "data-theme-toggle", 'id="rail"', "☰", "◐"):
                assert old not in page, old
            s = info[current]
            menu = page.split('<details class="gl-sm">')[1].split("</details>")[0]
            summary = _one(r'<summary class="gl-sb"[^>]*>(.*?)</summary>', menu)
            dur = app_mod.duration((s["end"] - s["start"]).total_seconds())
            assert summary == (
                f"{s['channel']} · {app_mod.day(s['start'])} · {app_mod.hm(s['start'])}–{app_mod.hm(s['end'])} · {dur}"
            )
            assert re.search(rf"<b>{s['channel']}</b>", menu)
            assert re.search(r"Chat text (kept until|expired) \w{3} \d{1,2}", text(menu))
            assert "Other streams" in text(menu)
            got = []
            for group in menu.split('<p class="gl-pc">')[1:]:
                name, body = group.split("</p>", 1)
                link = r'<a class="gl-ps" href="([^"]*)"[^>]*><span class="gl-pt">(.*?)</span><span class="gl-pn">(.*?)</span></a>'
                got.append((text(name), [(h, text(t), text(n)) for h, t, n in re.findall(link, body)]))
            expected = [
                (ch, [(f"/?stream={i}", f"{app_mod.day(info[i]['start'])} · {app_mod.hm(info[i]['start'])}–"
                       f"{app_mod.hm(info[i]['end'])}", app_mod.plural(info[i]["n"], "Moment"))
                      for i in newest if info[i]["channel"] == ch])
                for ch in channels
            ]  # fmt: skip
            assert got == expected
            assert re.findall(r'<a class="gl-ps" href="([^"]*)" aria-current="page">', menu) == [f"/?stream={current}"]
            seg = re.search(r'<div class="gl-seg" role="group" aria-label="Theme">(.*?)</div>', menu, re.S)
            assert seg
            buttons = re.findall(r'<button type="button" data-theme-set="(\w+)" aria-pressed="(\w+)">(\w+)</button>', seg.group(1))
            assert [(v, w) for v, _, w in buttons] == [("system", "System"), ("dark", "Dark"), ("light", "Light")]
            assert [p for _, p, _ in buttons] == ["true", "false", "false"]  # the script marks the stored one
            assert text(_one(r'<p class="gl-keys">(.*?)</p>', menu)) == "/ ask · J K next, previous · Esc close"
    finally:
        store.close()
    js = (STATIC / "page.js").read_text(encoding="utf-8")
    assert "data-theme-set" in js and '"chatbrain-theme"' in js and '"system"' in js
    assert "rail-open" not in js and "railButton" not in js


def test_glance_classes_prefixed(world):
    m = world.new_moments
    _peaked(world, {m[0]: 81}, gaps=((20, 9),))
    page = client(world).get("/").text
    header = page.split("<header", 1)[1].split("</header>")[0]
    glance = page.split('<section class="gl-recap"', 1)[1].split('<div class="dock">')[0]  # the thread is in the lane (F3)
    shared = {"tb", "bd"}  # the bar and its made-up badge stay Slice 1's
    for part in (header, glance):
        for attr in re.findall(r'\bclass="([^"]*)"', part):
            for token in attr.split():
                assert token in shared or token.startswith(("gl-", "sd-", "fc-", "sx-")), token
    for taken in ("ln", "st", "sl", "lane"):
        assert not re.search(rf'class="(?:[^"]* )?{taken}(?: [^"]*)?"', header + glance), taken
    css = (STATIC / "page.css").read_text(encoding="utf-8")
    for token in {t for a in re.findall(r'\bclass="([^"]*)"', glance) for t in a.split()}:
        assert f".{token}" in css, token  # every glance class is styled


def test_plot_label_names_gaps(world):
    hm = app_mod.hm
    a = _peaked(world, {world.new_moments[1]: 81}, gaps=((0, 6), (20, 9)))
    page = client(world).get("/").text
    said = _plot_label(page)
    t0 = a.started_at
    assert said.startswith("Activity line of people chatting per 10 s from ")
    assert "The three busiest peaks are at " in said
    assert f"No chat recorded from {hm(t0)} to {hm(t0 + timedelta(seconds=60))}." in said
    assert f"No chat recorded from {hm(t0 + timedelta(seconds=200))} to {hm(t0 + timedelta(seconds=290))}." in said
    assert "Coverage unknown" not in said
    assert re.findall(r'<rect class="gl-gap"[^>]*><title>([^<]+)</title></rect>', page) == ["No chat recorded"] * 2

    # the counts end 5 h early: the rest of the axis is unknown, never "No chat recorded" (R1)
    m = world.new_moments
    b = _peaked(world, {}, shift=timedelta(hours=5))
    page = client(world).get("/").text
    said = _plot_label(page)
    end = b.started_at + timedelta(seconds=len(b.chatters) * b.bin_s)
    last = max(world.windows[w].anchor_at for w in m)
    assert f"Coverage unknown from {hm(end)} to {hm(last)}." in said
    assert "No chat recorded" not in said and "No chat recorded" not in page.split('class="gl-t3"')[0]
    assert re.findall(r'<rect class="gl-gap"[^>]*><title>([^<]+)</title></rect>', page) == ["Coverage unknown"]


def test_plot_label_short_gap_keeps_seconds(world):
    a = _peaked(world, {world.new_moments[1]: 81}, gaps=((20, 1),))  # one 10 s bin
    said = _plot_label(client(world).get("/").text)
    t0 = a.started_at
    clock = app_mod.clock_text
    assert f"No chat recorded from {clock(t0 + timedelta(seconds=200))} to {clock(t0 + timedelta(seconds=210))}." in said


def _gap_labels(page: str) -> list[tuple[str, float]]:
    return [(cls, float(left)) for cls, left in re.findall(r'<span class="(gl-gapl[^"]*)" style="left:([\d.]+)%"', page)]


def test_gap_label_at_axis_end_hangs_inward(world):
    src = next(s for s in world.export.streams if s.stream_id == world.new)
    bins = len(src.chatters)
    # a known gap at each end of the counts and one in the middle
    _peaked(world, {world.new_moments[1]: 81}, gaps=((0, 3), (bins // 2, 3), (bins - 3, 3)))
    page = client(world).get("/").text
    g = _g(world, world.new)
    assert len(g.gaps) == 3
    for x in g.gaps:
        assert x["edge"] == ("l" if x["mid"] < app_mod.GAP_EDGE else "r" if x["mid"] > 100 - app_mod.GAP_EDGE else "")
    labels = _gap_labels(page)
    assert len(labels) == 3
    by_left = sorted(labels, key=lambda c: c[1])
    assert by_left[-1][1] > 100 - app_mod.GAP_EDGE and by_left[-1][0] == "gl-gapl gl-r"  # right-aligned
    assert by_left[1][0] == "gl-gapl"  # the middle one stays centred
    css = (STATIC / "page.css").read_text(encoding="utf-8")
    assert ".gl-gapl.gl-r{translate:-100% 0" in css and ".gl-gapl.gl-l{translate:0 0" in css
    base = re.search(r"\n\.gl-gapl\{([^}]*)\}", css)
    assert base and "width:max-content" in base.group(1)  # no shrink-wrap to one word near an edge

    # coverage unknown after the counts end sits at the axis end too
    _peaked(world, {}, shift=timedelta(hours=5))
    labels = _gap_labels(client(world).get("/").text)
    assert len(labels) == 1 and labels[0][0] == "gl-gapl gl-r"


def test_status_flows_as_one_sentence(world):
    page = client(world).get("/").text
    n = len(world.new_moments)
    assert re.search(
        rf'<p class="gl-status"><i class="gl-gns" aria-hidden="true"></i><span><b>Not sure about {n} of {n} Moments\.</b>'
        r" No model yet\.</span></p>",
        page,
    )


def test_close_returns_focus_to_the_opener_or_lane_mark():
    js = (STATIC / "page.js").read_text(encoding="utf-8")
    closing = js.split("function close()", 1)[1].split("function problem(", 1)[0]
    assert "backTo(id)" in closing
    back = closing.split("function backTo(id)", 1)[1]
    # the link that opened the Moment first, then its card, then its lane mark (most Moments have no card)
    assert back.index("opener") < back.index(".gl-card[data-window=") < back.index('".gl-lane a"')
    assert "windowOf(marks[i].getAttribute(\"href\")) === id" in back
    clicks = js.split('var open = t.closest("a[data-open]");', 1)[1].split("load(", 1)[0]
    assert "opener = open" in clicks and "!pane.contains(open)" in clicks


def test_phone_lane_follows_g1_placement():
    js = (STATIC / "page.js").read_text(encoding="utf-8")
    place = js.split("function placeLane()", 1)[1].split('document.addEventListener("click"', 1)[0]
    assert "var ROW = 22;" in js and "var C3_CLEAR = 19;" in js and "var TAP = 22;" in js
    # two rows: row 1 when ROW px past its last mark, else the row with more room; each row keeps its last x
    assert "g1 >= ROW || g1 >= g2 ? 0 : 1" in place and "last[row] = x;" in place
    # a chip fills row 2's band, and a row-2 mark is kept C3_CLEAR px from every chip's centre
    assert "last[1] = at(a) + 8" in place and "c - C3_CLEAR : c + C3_CLEAR" in place
    assert 'a.style.left = x.toFixed(1) + "px";' in place and 'a.getAttribute("data-x") + "%"' in place
    assert "new ResizeObserver(function () { placeLane(); })" in js
    css = (STATIC / "page.css").read_text(encoding="utf-8")
    assert ".gl-mk.gl-r2{top:25px}" in css
    # the lane keeps one tab stop once the script runs
    roving = js.split("function rove()", 1)[1].split("function roveTo(", 1)[0]
    assert '".gl-lane"' in roving and "roveLane(pick)" in roving


# --- the side lane (Task 4): an opened Moment and the All N list ------------------------------------

WHEN = "When chat reacted. The screen moment is usually 3–8 s earlier."
FOOT = "No preview pictures yet. Times are when chat reacted; the screen moment is usually 3–8 s earlier."
NOT_SURE_LINE = "BrainChat isn't sure here, so it makes no guess."


def _head(page: str) -> str:
    """The opened Moment's head: number, time, timing line, title and the stat slot."""
    return page.split('<header class="sd-head">')[1].split("</header>")[0]


def _slot(page: str) -> str | None:
    """The stat slot's visible text: both stats joined with " · ", a no-peak line, or the Q10 line."""
    found = re.search(r'<div class="sd-slot">(.*)</div>', _head(page), re.S)
    if not found:
        return None
    stats = re.findall(r'<p class="sd-stat">(.*?)</p>', found.group(1), re.S)
    return " · ".join(text(s) for s in stats) if stats else text(found.group(1))


def _sec(page: str, title: str) -> str:
    """A section of the Details fold, by its heading."""
    found = re.search(rf'<section class="sd-sec">\s*<h3 class="sd-h">{re.escape(title)}.*?</section>', page, re.S)
    assert found, f"no Details section {title!r}"
    return found.group(0)


def _said(page: str) -> str:
    found = re.search(r'<section class="sd-chat" aria-label="What chat said">(.*?)</section>', page, re.S)
    assert found, "no What chat said"
    return found.group(1)


def _summary(fragment: str) -> str:
    return text(re.search(r"<summary[^>]*>(.*?)</summary>", fragment, re.S).group(1))


def _rows_of(page: str) -> list[str]:
    return re.findall(r'<li class="sd-row" data-window="([^"]+)"', page)


def _row(page: str, window_id: str) -> str:
    return page.split(f'<li class="sd-row" data-window="{window_id}"')[1].split("</li>")[0]


def test_panel_g1_sections(world):
    m = world.new_moments
    n = len(m)
    _peaked(world, {m[0]: 81})
    label(world.store, m[0], {"doing": ["laughing"]})
    c = client(world)
    page = c.get(f"/m/{m[0]}?panel=1").text
    num = {x.window_id: x.n for x in _g(world, world.new).lane}[m[0]]
    # the bar: back to the list, and close
    bar = page.split('<div class="sd-bar">')[1].split("</div>")[0]
    assert re.search(r'<a class="sd-x" href="/\?stream=[^"]+" data-close aria-label="Close the Moment">×</a>', bar)
    back = re.search(r'<a class="sd-back" href="([^"]+)" data-open>(.*?)</a>', bar, re.S)
    assert back and back.group(1) == f"/list?stream={world.new}" and text(back.group(2)) == f"‹ All {n}"
    assert back.group(2) == f'<span aria-hidden="true">‹</span> All {n}'  # the name is "All N"
    # the head: number and time, the timing line, the title with its tag, the stat
    head = _head(page)
    clock = app_mod.clock_text(world.windows[m[0]].anchor_at)
    assert re.search(rf'<span class="sd-num" data-lift>{num}</span><span class="sd-time">{re.escape(clock)}</span>', head)
    assert f'<p class="sd-when">{WHEN}</p>' in head
    h2 = re.search(r'<h2 class="sd-say" tabindex="-1">(.*?)</h2>', head, re.S)
    assert h2 and text(h2.group(1)).startswith("Laughing · ") and text(h2.group(1)).endswith("your label")
    assert '<span class="sr"> · </span><span class="sd-tag sd-you">your label</span>' in h2.group(1)
    assert text(h2.group(1)).endswith(" · your label")  # the name reads with a break before the tag
    assert _slot(page) == "81 people chatting in the busiest 10 s · 9× the usual pace"
    assert "<b>81</b>" in head and "<b>9×</b>" in head
    # what chat said: the kept line and three lines, labelled k of m
    said = _said(page)
    expires = world.store.window(m[0]).text_expires_at
    assert f"Chat text kept until {app_mod.day(expires)}" in text(said)
    total = len(world.windows[m[0]].messages)
    assert total > 3
    assert said.count('<li class="sd-ln">') == 3
    assert re.search(rf'<p class="sd-k" id="sd-k">3 of {total} lines</p>', said)
    assert 'aria-labelledby="sd-k"' in said
    assert NOT_SURE_LINE not in page
    # fix: G1's trigger over Slice 1's four-question menu with its Not sure flow (G-5)
    fix = page.split('<details class="sd-fx">')[1].split('<details class="sd-more">')[0]
    assert _summary(fix) == "Not right? Fix"
    assert fix.count('<details class="fx">') == 4 and fix.count("Not sure ▸") == 4
    assert "Saves your label. It goes into the next training run." in fix
    # details: the answers, the cause and Counted, folded
    more = page.split('<details class="sd-more">')[1]
    assert _summary(more) == "Details cause unknown"
    assert _sec(more, "What chat did").count('<div class="aq">') == 4
    cause = _sec(more, "Cause")
    assert "Cause unknown · no cause note" in text(cause) and "Add cause note" in cause
    counted = _sec(more, "Counted")
    assert [text(x) for x in re.findall(r"<dt>(.*?)</dt>", counted)] == ["Spread", "Lines joining", "Posters joining", "Text"]
    # in G1's order
    order = [page.index(x) for x in ('class="sd-bar"', 'class="sd-head"', 'class="sd-chat"', 'class="sd-fx"', 'class="sd-more"')]
    assert order == sorted(order)
    # with a cause note the summary is plain "Details"
    label(world.store, m[0], {"doing": ["laughing"]}, cause_note="The guest arrived", cause_source="clip", at=1)
    page = c.get(f"/m/{m[0]}?panel=1").text
    assert _summary(page.split('<details class="sd-more">')[1]) == "Details"
    # a model's sure answer reads "model guess"
    stub_model(world.store, world.models, RUNG0_ID, 0)
    predict(world.store, m[1], RUNG0_ID, "doing", ["cheering"], 0.9)
    head = _head(client(world).get(f"/m/{m[1]}?panel=1").text)
    assert '<span class="sd-tag">model guess</span>' in head and "your label" not in head


def test_panel_not_sure(world):
    m = world.new_moments
    _peaked(world, {m[0]: 81})
    page = client(world).get(f"/m/{m[1]}?panel=1").text
    head = _head(page)
    assert re.search(r'<span class="sd-num sd-ns" data-lift>\d+</span>', head)
    say = text(re.search(r'<h2 class="sd-say" tabindex="-1">(.*?)</h2>', head, re.S).group(1))
    assert say.startswith("Not sure what chat was doing")
    assert "your label" not in head and "model guess" not in head and "sd-tag" not in head
    assert f'<p class="sd-nsl">{NOT_SURE_LINE}</p>' in page
    fix = page.split('<details class="sd-fx">')[1]
    assert _summary(fix) == "Know what chat did? Fix"
    assert page.index('class="sd-nsl"') < page.index('class="sd-fx"')


def test_panel_text_expired(world):
    old = world.old_moments[0]
    later = NOW + timedelta(days=5)
    # counts live: the approved Slice 1 line
    clear_activity(world.store)
    assert store_activity(world.store, activity_line(world, world.old, peaks={old: 30})) == "written"
    c = client(world, now=later)
    assert world.store.window(old).text_state == "expired"
    assert world.store.activity_state(world.old, now=later) == "active"
    day = app_mod.day(world.store.window(old).text_expires_at)
    page = c.get(f"/m/{old}?panel=1").text
    said = _said(page)
    assert f'<p class="sd-exp">Chat text expired {day}. Labels and counts remain.</p>' in said
    assert "sd-ln" not in said and "Chat text kept until" not in said and " lines</p>" not in said
    for t in (x.text for x in world.windows[old].messages):
        assert t not in page
    # counts gone: the short line
    clear_activity(world.store)
    page = client(world, now=later).get(f"/m/{old}?panel=1").text
    assert f'<p class="sd-exp">Chat text expired {day}.</p>' in _said(page)
    assert "Labels and counts remain." not in page


def test_panel_no_chart_no_window_id(world):
    m = world.new_moments
    _peaked(world, {m[0]: 81})
    label(world.store, m[0], {"doing": ["laughing"]})
    c = client(world)
    page = c.get(f"/m/{m[0]}?panel=1").text
    for gone in ('class="chart"', "Messages per 5 s", "<code>", 'class="pm"', 'class="bds"', 'class="ask pask"',
                 "Ask about this Moment", "evidence line", 'class="pbar"', 'class="ph"', 'class="bd" data-state'):  # fmt: skip
        assert gone not in page, gone
    assert m[0] not in text(page)  # the window id shows nowhere (it stays in links and form actions)
    assert '<h3 class="sd-h">Chat lines' not in page and '<ol class="cl">' not in page
    # the full page keeps the per-Moment ask (G-14), the same Moment, and Slice 1's window meta and
    # id, the Messages per 5 s chart and the whole chat line list
    full = c.get(f"/m/{m[0]}").text
    form = re.search(r'<form class="ask pask"[^>]*data-ask[^>]*>(.*?)</form>', full, re.S)
    assert form and f'name="window_id" value="{m[0]}"' in form.group(1)
    assert _slot(full) == _slot(page) and '<details class="sd-more">' in full
    meta = re.search(r'<p class="pm">(.*?)</p>', _head(full), re.S)
    assert meta and f"<code>{m[0]}</code>" in meta.group(1) and " · window " in text(meta.group(1))
    chart = re.search(r'<section class="sd-full">\s*<h3 class="sd-h">Messages per 5 s</h3>(.*?)</section>', full, re.S)
    assert chart and '<div class="chart">' in chart.group(1) and 'aria-label="Messages per 5 seconds' in chart.group(1)
    lines = re.search(r'<section class="sd-full">\s*<h3 class="sd-h">Chat lines <span>time from the anchor</span></h3>(.*?)</section>', full, re.S)  # fmt: skip
    total = len(world.windows[m[0]].messages)
    assert lines and '<ol class="cl">' in lines.group(1)
    assert lines.group(1).count('<li class="ln') == total  # every line, the folded ones inside their fold
    # in order: G1's lane, then Slice 1's extras, then the ask
    order = [full.index(x) for x in ('class="sd-more"', "Messages per 5 s", "Chat lines", 'class="sd-ask"')]
    assert order == sorted(order)
    # an expired window on the full page: the chart's own expired copy, no folded list, and the one
    # text-expired line keeps the counts-gone variant
    old = world.old_moments[0]
    later = client(world, now=NOW + timedelta(days=5))
    clear_activity(world.store)
    gone = later.get(f"/m/{old}").text
    day = app_mod.day(world.store.window(old).text_expires_at)
    assert f'<p class="exp">The chart left with the chat text on {day}.</p>' in gone
    assert "Chat lines" not in gone and "Labels and counts remain." not in gone


def test_panel_counts_off_shows_q10(world):
    m = world.new_moments
    old = world.old_moments[0]
    gone_missing = NO_COUNTS.replace("'", "&#39;")
    # missing: the Q10 line in the stat slot and in Counted, never the no-peak copy
    clear_activity(world.store)
    c = client(world)
    for w in (m[0], m[1]):
        page = c.get(f"/m/{w}?panel=1").text
        assert _slot(page) == NO_COUNTS
        assert f'<p class="sd-gone">{gone_missing}</p>' in _sec(page, "Counted")
        for never in ("Coverage unknown", "coverage unknown", "not recorded", "people chatting", EXPIRED_COUNTS):
            assert never not in page, never
    later = client(world, now=NOW + timedelta(days=5)).get(f"/m/{old}?panel=1").text
    assert _slot(later) == NO_COUNTS and "Labels and counts remain." not in later
    assert re.search(r'<p class="sd-exp">Chat text expired \w{3} \d{1,2}\.</p>', later)

    # expired: the old stream's counts reach their 30 days, before and after the sweep
    a = activity_line(world, world.old, peaks={old: 30})
    assert store_activity(world.store, a) == "written"
    due = a.started_at + timedelta(days=30)
    now = [due - timedelta(minutes=1)]
    c = client(world, clock=lambda: now[0])
    assert _slot(c.get(f"/m/{old}?panel=1").text).startswith("30 people chatting in the busiest 10 s")
    now[0] = due
    for page in (c.get(f"/m/{old}?panel=1").text, client(world, now=due).get(f"/m/{old}?panel=1").text):
        assert _slot(page) == EXPIRED_COUNTS
        assert f'<p class="sd-gone">{EXPIRED_COUNTS}</p>' in _sec(page, "Counted")
        assert "Coverage unknown" not in page and "Labels and counts remain." not in page
        assert "people chatting" not in page and "not recorded" not in page and gone_missing not in page
        assert re.search(r'<p class="sd-exp">Chat text expired \w{3} \d{1,2}\.</p>', page)


def test_list_orders(world):
    m = world.new_moments
    n = len(m)
    _peaked(world, {m[2]: 81, m[0]: 60, m[5]: 40})
    c = client(world)
    page = c.get(f"/list?stream={world.new}&panel=1").text
    assert f'<h2 class="sd-lh" tabindex="-1">All {n} Moments</h2>' in page
    assert re.search(r'<a class="sd-x" href="[^"]+" data-close aria-label="Close the list">×</a>', page)
    assert _one(r'<p class="sd-sub">(.*?)</p>', page) == "Busiest first"
    rows = _rows_of(page)
    assert sorted(rows) == sorted(m) and rows[:3] == [m[2], m[0], m[5]]
    peaks = [int(re.search(r"<b>(\d+)</b> people in 10 s", _row(page, w)).group(1)) for w in rows]
    at = {w: i for i, w in enumerate(m)}
    keys = [(-p, at[w]) for p, w in zip(peaks, rows)]
    assert keys == sorted(keys)  # G-3: by peak, then by time
    g = {x.window_id: x for x in _g(world, world.new).lane}
    first = _row(page, m[2])
    assert f'<a class="sd-ro" href="/m/{m[2]}" data-open data-n="{g[m[2]].n}">' in first
    assert re.search(rf'<span class="sd-num sd-ns">{g[m[2]].n}</span><span class="sd-time">', first)
    assert _one(r'<span class="sd-lsay">(.*?)</span>', first).startswith("Not sure what chat was doing")
    assert _one(r'<span class="sd-sz">(.*?)</span>', first) == "81 people in 10 s · 9× usual pace"
    toggle = re.search(r'<div class="sd-ord" role="group" aria-label="Order">(.*?)</div>', page, re.S)
    assert toggle
    links = re.findall(r'<a class="sd-o" href="([^"]+)" data-open( aria-current="true")?>([^<]+)</a>', toggle.group(1))
    assert [(h.replace("&amp;", "&"), bool(cur), t) for h, cur, t in links] == [
        (f"/list?stream={world.new}&order=busy", True, "Busiest first"),
        (f"/list?stream={world.new}&order=time", False, "In time order"),
    ]
    timed = c.get(f"/list?stream={world.new}&order=time&panel=1").text
    assert _one(r'<p class="sd-sub">(.*?)</p>', timed) == "In time order" and _rows_of(timed) == m
    # counts off: time order only, no stat and no toggle
    clear_activity(world.store)
    for url in (f"/list?stream={world.new}&panel=1", f"/list?stream={world.new}&order=busy&panel=1"):
        off = c.get(url).text
        assert _one(r'<p class="sd-sub">(.*?)</p>', off) == "In time order" and _rows_of(off) == m
        assert 'class="sd-sz"' not in off and 'class="sd-ord"' not in off
    # without the script: a full page; bad input refused
    full = c.get(f"/list?stream={world.new}").text
    assert '<main class="main" id="main"' in full and '<details class="gl-sm"' in full and _rows_of(full) == m
    assert c.get("/list?stream=nope").status_code == 404
    assert c.get(f"/list?stream={world.new}&order=loud").status_code == 400
    assert c.get("/list").status_code in (400, 422)
    # the glance's All button opens this list, not a Moment
    home = c.get("/").text
    found = re.search(r'<a class="gl-all" href="([^"]+)" data-open>All (\d+) Moments</a>', home)
    assert found and found.group(1) == f"/list?stream={world.new}" and int(found.group(2)) == n
    assert 'class="gl-all" href="/m/' not in home


def test_list_footer_verbatim(world):
    m = world.new_moments
    _peaked(world, {m[0]: 81})
    c = client(world)
    for url in (f"/list?stream={world.new}&panel=1", f"/list?stream={world.new}", f"/list?stream={world.old}&panel=1"):
        page = c.get(url).text
        assert page.count(f'<p class="sd-foot">{FOOT}</p>') == 1, url
        assert page.index('class="sd-rows"') < page.index('class="sd-foot"')
    clear_activity(world.store)
    assert f'<p class="sd-foot">{FOOT}</p>' in c.get(f"/list?stream={world.new}&panel=1").text


def test_side_lane_script_and_motion():
    """The lane's script and motion: the phone sheet is a modal dialog over an inert page; the number
    lifts into the lane only where view transitions run and motion is on; Esc closes in layers."""
    js = (STATIC / "page.js").read_text(encoding="utf-8")
    sheet = js.split("function sheetBehind()", 1)[1].split("function streamMenu()", 1)[0]
    assert 'setAttribute("role", "dialog")' in sheet and 'setAttribute("aria-modal", "true")' in sheet
    assert 'removeAttribute("aria-modal")' in sheet and 'setAttribute("role", "region")' in sheet
    assert '"sd-lock"' in sheet
    load = js.split("function load(", 1)[1].split("function close()", 1)[0]
    assert "document.startViewTransition" in load and "moving()" in load and '"lift"' in load
    assert "[data-lift]" in load and "sheetBehind();" in load
    keys = js.split("if (esc) {", 1)[1].split("return;", 1)[0]
    # the Fix menu, then Fix, then the stream menu, then a Moment opened from the list goes back to it
    assert keys.index("details.fx[open]") < keys.index("details.sd-fx[open]") < keys.index("closeMenu(true)")
    assert keys.index("closeMenu(true)") < keys.index("listHref") < keys.index("close();")
    css = (STATIC / "page.css").read_text(encoding="utf-8")
    parts = css.split("@media (prefers-reduced-motion:no-preference){\n.sd-", 1)
    assert len(parts) == 2
    block = ".sd-" + parts[1].split("\n}", 1)[0]
    assert "::view-transition-group(lift){animation-duration:200ms" in block
    assert "animation:sd-unfold var(--cb-d2)" in block and "animation:sd-lift var(--cb-d3)" in block
    assert "@keyframes sd-unfold{" in css and "@keyframes sd-lift{" in css
    assert "body.sd-lock{overflow:hidden}" in css
    assert "@keyframes cb-in{" not in css and "cb-sheet" not in css  # the old slide-in made way for G1's lift
    # the list marks the row of the Moment last opened from it, like the cards and lane marks
    mark = js.split("function mark(", 1)[1].split("function hot(", 1)[0]
    assert ".sd-row[aria-current]" in mark and ".sd-row[data-window]" in mark
    assert ".sd-row[aria-current] .sd-ro{" in css
    assert "mark(marked(href), true)" in load and "lastMoment = null;" in js.split("function close()", 1)[1]


def test_list_and_panel_share_the_tag_rule(world):
    """One tag rule for the glance card, the list row and the opened Moment: "your label" only on
    the owner's sure Doing label; their Not sure label (no values) carries no tag in any of them."""
    m = world.new_moments
    _peaked(world, {m[0]: 81})
    label(world.store, m[0], {"doing": ["laughing"]})
    label(world.store, m[1], {"aimed_at": "streamer"}, not_sure={"doing": ["conflicting"]})
    c = client(world)
    page = c.get(f"/list?stream={world.new}&order=time&panel=1").text
    mine = _row(page, m[0])
    assert '<span class="sr"> · </span><span class="sd-tag sd-you">your label</span>' in mine
    assert text(_one(r'<span class="sd-lsay">(.*)</span>', mine.split('<span class="sd-sz">')[0])).endswith(" · your label")
    unsure = _row(page, m[1])
    assert "sd-tag" not in unsure and "your label" not in unsure
    opened = c.get(f"/m/{m[1]}?panel=1").text
    head = _head(opened)
    assert "sd-tag" not in head and "your label" not in head
    # it is the owner's label all the same: Details names the source
    doing = _sec(opened, "What chat did").split('<div class="aq">')[1]
    assert '<span class="src">your label</span>' in doing
    home = c.get("/").text  # the glance card follows the same rule
    assert _cards(home)[:2] == [m[0], m[1]]
    assert '<span class="gl-tag gl-you">your label</span>' in _card(home, m[0])
    assert "gl-you" not in _card(home, m[1]) and "your label" not in _card(home, m[1])
