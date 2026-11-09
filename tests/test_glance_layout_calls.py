"""Glance layout calls L1 + D1 (the owner's picks, Nov 8).

L1: below 1024 px the opened Moment / All N list lane is the phone's full-screen sheet (a modal over
an inert page; Esc and close give focus back to the card or link that opened it). From 1024 px the
glance keeps its reflow beside the lane. The phone's two-row lane marks keep their own 760 px query,
and so does a thread's answer in the lane (G-8: it stays beside a live ask line down to 760 px).

D1: with the model off, the "Asking is limited" banner is one quiet line inside the dock: a small
amber dot, then both approved sentences word for word. No page bottom padding (D1-pad was left out).
Made-up worlds only; nothing here reads the real data folder, starts a model or opens a socket.
"""

import re
from pathlib import Path

import pytest

import pagekit
from chatbrain.page import app as app_mod

STATIC = Path(app_mod.__file__).parent / "static"
HEAD = "Asking is limited: the local model isn't running."
BODY = "Words were matched to filters; check the chips."


def css() -> str:
    return (STATIC / "page.css").read_text(encoding="utf-8")


def js() -> str:
    return (STATIC / "page.js").read_text(encoding="utf-8")


def blocks(source: str, query: str) -> list[str]:
    """The bodies of every top-level ``@media <query>{ ... }`` block (one rule per line, as page.css)."""
    out = []
    for part in source.split("@media " + query + "{\n")[1:]:
        out.append(part.split("\n}\n", 1)[0])
    return out


def rules(body: str) -> dict[str, str]:
    return {sel.strip(): decl for sel, decl in re.findall(r"([^{}\n]+)\{([^{}]*)\}", body)}


@pytest.fixture
def world(tmp_path):
    w = pagekit.build_world(tmp_path)
    yield w
    w.store.close()


# --- L1: the sheet below 1024 px ------------------------------------------------------------------


def test_l1_sheet_has_its_own_1024_query():
    source = js()
    m = re.search(r'var (\w+) = window\.matchMedia \? window\.matchMedia\("\(max-width: 1023\.98px\)"\) : null;', source)
    assert m, "the sheet's own (max-width: 1023.98px) query"
    sheet_q = m.group(1)
    assert sheet_q != "narrow"
    sheet = source.split("function sheetBehind()", 1)[1].split("function streamMenu()", 1)[0]
    # a Moment or list follows the 1024 px query; a thread's answer in the lane keeps 760 px (G-8)
    assert f"var q = answerNode && pane && pane.contains(answerNode) ? narrow : {sheet_q};" in sheet
    assert "var on = !!pane && !pane.hidden && !!q && q.matches;" in sheet
    # the sheet follows its own query: crossing 1024 px turns it on or off, as 760 px did
    assert re.search(sheet_q + r"\.addEventListener\(\"change\", sheetChange\)", source)
    # and crossing 760 px does too, for the answer
    wide = source.split("var wide = function () {", 1)[1].split("};", 1)[0]
    assert "placeLane();" in wide and "sheetChange();" in wide
    # no MutationObserver hack: load(), showAnswer() and close() call sheetBehind() themselves
    assert "MutationObserver" not in source


def test_l1_sheet_css_is_the_phone_sheet_up_to_1023():
    source = css()
    assert not re.search(r"max-width:\s*1023px", source)  # no fractional gap at 1023.5 px
    wide = blocks(source, "(max-width:1023.98px)")
    lane = {}
    for b in wide:
        lane.update(rules(b))
    sheet_lane = ".pn:not(:has(>.sd-a))"  # a Moment or list, not a thread's answer (G-8)
    # the lane fills the screen over the top bar, like the phone sheet (G1-phone)
    assert lane.get(sheet_lane) == "top:0;left:0;z-index:40;width:auto;border-left:0"
    assert ".pn" not in lane  # the answer lane keeps its 408 px side lane down to 760 px
    # it comes after the 1279.98 px rule that narrows the lane to 408 px, so it wins
    assert source.index("@media (max-width:1023.98px){\n" + sheet_lane + "{") > source.index("@media (max-width:1279.98px){\n.pn{")
    # the sheet's own look, for the Moment and list lane only (the answer lane and the full Moment
    # and list pages keep theirs from 760 px)
    w = f":where({sheet_lane})"
    assert lane[w + " .sd-in"] == "padding:0 16px 40px"
    assert lane[w + " .sd-bar"] == "height:48px;margin:0 -16px 8px;padding:0 8px;border-bottom:1px solid var(--cb-border-subtle)"
    assert lane[w + " .sd-time"] == "font-size:20px"
    assert lane[w + " :is(.sd-say,.sd-lh)"] == "font-size:20px"
    assert not any(":where(.pn)" in k for k in lane)
    # the sheet lifts up from below, only with motion on
    lift = blocks(source, "(prefers-reduced-motion:no-preference) and (max-width:1023.98px)")
    assert lift and ".sd-in.sd-enter{animation-name:sd-lift-up}" in lift[0]
    assert "(prefers-reduced-motion:no-preference) and (max-width:759.98px)" not in source
    # from 1024 px the glance still reflows beside the open lane (G1-desk), unchanged
    assert re.search(r"@media \(min-width:1024px\)\{\s*\.main:has\(>\.pn:not\(\[hidden\]\)\)>\.col\{", source)


def test_l1_full_pages_and_answer_keep_their_phone_look_below_760():
    phone = {}
    for b in blocks(css(), "(max-width:759.98px)"):
        phone.update(rules(b))
    assert phone[":where(.pn,.pv) .sd-in"] == "padding:0 16px 40px"
    assert phone[":where(.pn,.pv) .sd-bar"] == "height:48px;margin:0 -16px 8px;padding:0 8px;border-bottom:1px solid var(--cb-border-subtle)"
    assert phone[":where(.pn,.pv) .sd-time"] == "font-size:20px"
    assert phone[":where(.pn,.pv) :is(.sd-say,.sd-lh)"] == "font-size:20px"


def test_l1_answer_lane_stays_beside_the_ask_line_from_760():
    """G-8 is not part of L1: between 760 and 1023 px a thread's answer stays a side lane beside a live
    dock (ask line, "/", Ask, New question); only a phone makes it the full-screen sheet. A Moment
    opened from the answer is a sheet there, and "‹ Answer" / Esc bring back the side-lane answer."""
    source = css()
    phone = {}
    for b in blocks(source, "(max-width:759.98px)"):
        phone.update(rules(b))
    assert phone[".pn"] == "top:0;left:0;z-index:40;width:auto;border-left:0"
    assert source.index("@media (max-width:759.98px){\n.main{padding:0 16px}\n.tb{gap:6px;padding:0 8px 0 12px}\n.pn{") > source.index(
        "@media (max-width:1279.98px){\n.pn{"
    )
    # the dock keeps clear of the open lane from 760 px, as approved
    assert "@media (min-width:760px){\n.main:has(>.pn:not([hidden]))>.dock{" in source
    # showAnswer() puts the answer back in the lane before it re-reads the sheet query
    shown = js().split("function showAnswer()", 1)[1].split("function leaveAnswer()", 1)[0]
    assert shown.index("pane.appendChild(answerNode);") < shown.index("sheetBehind();")


def test_l1_two_row_lane_marks_keep_760():
    source = js()
    assert 'var narrow = window.matchMedia ? window.matchMedia("(max-width: 759.98px)") : null;' in source
    place = source.split("function placeLane()", 1)[1].split("\n  }\n", 1)[0]
    assert "var two = !!(narrow && narrow.matches);" in place
    tap = source.split('var lane = e.target.closest && e.target.closest(".gl-lane");', 1)[1].split("\n", 2)[1]
    assert "narrow.matches" in tap  # the near-miss tap is a phone aid for the two rows
    # widening across 760 px still re-places the marks
    assert re.search(r"narrow\.addEventListener\(\"change\", \w+\)", source)
    phone = {}
    for b in blocks(css(), "(max-width:759.98px)"):
        phone.update(rules(b))
    assert phone[".gl-mk.gl-r2"] == "top:25px" and phone[".gl-lane"] == "height:44px"


def test_l1_close_gives_focus_back_to_the_opener_not_the_page():
    source = js()
    closing = source.split("function close()", 1)[1].split("function backTo(", 1)[0]
    # the page behind comes back (inert off) before focus goes to the card or link that opened it
    assert closing.index("sheetBehind();") < closing.index("back.focus()")
    assert 'document.querySelector(".gl-all")' in closing
    back = source.split("function backTo(id)", 1)[1].split("function problem(", 1)[0]
    assert ".gl-co" in back and ".mc-o" in back and ".gl-lane a" in back
    for target in (closing, back):
        assert ".cb" not in target and "document.body.focus" not in target
    # Esc closes through the same close() (or goes back to the list / the answer first)
    keys = source.split("if (esc) {", 1)[1].split("return;", 1)[0]
    assert keys.index("listHref") < keys.index("close();")


def test_l1_sheet_turning_on_moves_focus_into_it():
    """Narrowing past 1024 px with the lane open makes it modal: focus left behind (now inert) moves
    into the sheet, to its heading, so it never falls to the page body."""
    source = js()
    handler = source.split("function sheetChange()", 1)
    assert len(handler) == 2
    body = handler[1].split("\n  }\n", 1)[0]
    assert "sheetBehind();" in body
    assert 'pane.getAttribute("aria-modal") === "true"' in body and "pane.contains(document.activeElement)" in body
    assert 'pane.querySelector("h2")' in body and "preventScroll" in body


# --- D1: the banner as one quiet line ---------------------------------------------------------------


def test_d1_banner_markup_keeps_both_strings(world):
    page = pagekit.client(world).get("/").text  # no local model: the banner shows
    dock = page[page.index('<div class="dock">'):].split('<form class="ask"')[0]
    # one element, both sentences verbatim, a real space between them (not CSS content)
    assert '<div class="ban" role="status"><b>Asking is limited: the local model isn&#39;t running.</b> <span>Words were matched to filters; check the chips.</span></div>' in dock
    assert pagekit.text(dock).startswith(f"{HEAD} {BODY}")


def test_d1_banner_is_one_quiet_line_in_the_dock():
    source = css()
    all_rules = re.findall(r"(?m)^([^{}\n@]+)\{([^{}]*)\}", source)
    ban = [decl for sel, decl in all_rules if sel.strip() == ".ban"]
    assert ban == ["display:block;padding:0 2px;border:0;border-radius:0;background:none;font-size:12.5px;line-height:18px;color:var(--cb-text-muted);text-wrap:pretty"]
    assert "--cb-warning-soft" not in ban[0]  # no amber box
    dot = [decl for sel, decl in all_rules if sel.strip() == ".ban::before"]
    assert dot == ['content:"";display:inline-block;width:7px;height:7px;margin-right:7px;border-radius:50%;background:var(--cb-warning-border);vertical-align:1px']
    head = [decl for sel, decl in all_rules if sel.strip() == ".ban b"]
    assert head == ["font-weight:540;color:var(--cb-text)"]
    assert ".ban span::before" not in source  # the space between the sentences is in the markup
    # D1-pad is not part of the pick: no page bottom padding from the dock height
    assert not re.search(r"\.main\{[^}]*padding-bottom:var\(--cb-dock-h", source)
