// The Moment page's small script: the theme (System / Dark / Light), the stream menu, the Moment
// sheet under 1024 px, the right panel, menus, keys, the glance's motion and its lane, saving from
// the panel, the ask box and the local model steps. Every request goes to this page's own address;
// nothing loads from elsewhere. Asking posts a plain form (the page answers with a redirect to the
// thread), so it works without this script too; the question never goes into a URL. Without the
// script an answer that waits on the local model keeps its keyword answer.
(function () {
  "use strict";

  var root = document.documentElement;
  var KEY = "chatbrain-theme";

  function stored() {
    try { return window.localStorage.getItem(KEY); } catch (e) { return null; }
  }
  function store(value) {
    try { window.localStorage.setItem(KEY, value); } catch (e) { /* the switch still works for this visit */ }
  }

  // The theme segment (in the stream menu, or in the bar when there is no stream): "system"
  // follows the device; dark and light pin the page. The stored one is pressed.
  function theme(value) {
    if (value === "dark" || value === "light") root.dataset.theme = value;
    else {
      value = "system";
      delete root.dataset.theme;
    }
    document.querySelectorAll("[data-theme-set]").forEach(function (b) {
      b.setAttribute("aria-pressed", b.getAttribute("data-theme-set") === value ? "true" : "false");
    });
    return value;
  }
  theme(stored());

  var pane = document.querySelector(".pn");
  var current = null; // the Moment URL the panel shows
  var opener = null; // the link outside the panel that opened it (a card, chip or lane mark)
  var listHref = null; // the All N list a Moment was opened from: Esc goes back to it (G-11)
  var lastMoment = null; // the window last opened in the lane: its row stays current in the All N list
  // F3 (Task 6): the answer in the lane. On a thread (?thread=) the server renders it there;
  // ``answerNode`` keeps it while a Moment cited from it is open, so "‹ Answer" and Esc bring it
  // back without a reload. ``answerFrom`` holds while the lane shows a Moment opened from it.
  var answerNode = pane && pane.querySelector(".sd-a");
  var answerFrom = false;
  var answerOpener = null; // the citation that opened that Moment: focus comes back to it
  var still = window.matchMedia ? window.matchMedia("(prefers-reduced-motion: reduce)") : null;
  function moving() {
    return !(still && still.matches);
  }
  // Motion leaves nothing behind: the lift's name goes with its empty style attribute, and the
  // guide's drop class comes off when the drop ends, so the DOM matches the reduced-motion one.
  function unlift(el) {
    el.style.removeProperty("view-transition-name");
    if (!el.getAttribute("style")) el.removeAttribute("style");
  }
  document.addEventListener("animationend", function (e) {
    if (e.animationName === "gl-drop" && e.target.classList) e.target.classList.remove("gl-drop");
  });
  // Two widths: under 760 px the glance's lane marks take two rows (a phone); under 1024 px an opened
  // Moment or All N list is the phone's full-screen sheet (L1), so between 760 and 1023 px the glance
  // keeps its desk look and the lane never lies over it. A thread's answer in the lane stays a side
  // lane down to 760 px, beside a live ask line (G-8): it is a sheet only on a phone.
  var narrow = window.matchMedia ? window.matchMedia("(max-width: 759.98px)") : null;
  var sheetWidth = window.matchMedia ? window.matchMedia("(max-width: 1023.98px)") : null;

  // Below 1024 px the Moment peek is a sheet over the page (below 760 px for the answer): while it
  // is open everything behind it (the bar, the glance and the rest of the page, the skip link) is
  // inert, and comes back when it closes or the page widens. ``sheeted`` holds only what this set
  // inert. The open sheet is a modal dialog (G1-phone) and the page under it does not scroll; wider
  // than that the lane is a plain region beside the glance.
  var sheeted = [];
  function sheetBehind() {
    var q = answerNode && pane && pane.contains(answerNode) ? narrow : sheetWidth;
    var on = !!pane && !pane.hidden && !!q && q.matches;
    sheeted.forEach(function (el) { el.removeAttribute("inert"); });
    sheeted = [];
    if (pane) {
      if (on) {
        pane.setAttribute("role", "dialog");
        pane.setAttribute("aria-modal", "true");
      } else {
        pane.setAttribute("role", "region");
        pane.removeAttribute("aria-modal");
      }
    }
    document.body.classList.toggle("sd-lock", on);
    if (!on) return;
    var host = pane.parentNode;
    Array.prototype.forEach.call(document.body.children, function (el) {
      if (el === host) {
        Array.prototype.forEach.call(host.children, function (kid) {
          if (kid !== pane) sheeted.push(kid);
        });
      } else {
        sheeted.push(el);
      }
    });
    sheeted.forEach(function (el) { el.setAttribute("inert", ""); });
  }

  // The stream menu is a plain disclosure (it works without this script). Esc closes it and puts
  // focus back on the stream button; a click outside closes it.
  function streamMenu() {
    return document.querySelector("details.gl-sm[open]");
  }
  function closeMenu(refocus) {
    var d = streamMenu();
    if (!d) return false;
    d.removeAttribute("open");
    var s = d.querySelector("summary");
    if (refocus && s) s.focus();
    return true;
  }
  if (narrow) {
    var wide = function () {
      placeLane(); // two rows of marks only on a phone
      sheetChange(); // the answer in the lane is a sheet only on a phone
    };
    if (narrow.addEventListener) narrow.addEventListener("change", wide);
    else if (narrow.addListener) narrow.addListener(wide);
  }
  // Crossing 1024 px with a Moment or list open (760 px with the answer): the sheet turns on or off.
  // Focus left behind it (now inert) moves into the sheet, to its heading, so it never falls to the
  // page body.
  function sheetChange() {
    sheetBehind(); // the peek is a sheet only while the page is under 1024 px (760 px for the answer)
    if (pane && pane.getAttribute("aria-modal") === "true" && !pane.contains(document.activeElement)) {
      var h = pane.querySelector("h2") || pane.querySelector(".sd-x");
      if (h) h.focus({ preventScroll: true });
    }
  }
  if (sheetWidth) {
    if (sheetWidth.addEventListener) sheetWidth.addEventListener("change", sheetChange);
    else if (sheetWidth.addListener) sheetWidth.addListener(sheetChange);
  }

  function panelUrl(href) {
    var u = new URL(href, document.baseURI);
    u.searchParams.set("panel", "1");
    // a Moment opened from the answer gets "‹ Answer" as its one way back (G-11)
    if (answerFrom && answerNode && windowOf(href)) u.searchParams.set("answer", answerNode.dataset.thread);
    return u.pathname + u.search;
  }

  // The window a Moment link names, or null for any other link (the All N list).
  function windowOf(href) {
    var path = new URL(href, document.baseURI).pathname;
    if (!/^\/m\//.test(path)) return null;
    return decodeURIComponent(path.replace(/^\/m\//, ""));
  }
  function isList(href) {
    return new URL(href, document.baseURI).pathname === "/list";
  }
  // The window to mark for what the lane shows: the open Moment, or on the All N list the Moment
  // last opened from it (G1's list marks that row).
  function marked(href) {
    if (!href) return null;
    return windowOf(href) || (isList(href) ? lastMoment : null);
  }

  // The open Moment: its card, its lane mark and its All N list row are current; a top-three peak's
  // dot and guide turn with it, and the guide drops in only when the pick changes (drop), never on a
  // redraw after a save.
  var guided = null; // the window the guide last showed for
  function mark(id, drop) {
    document.querySelectorAll(".mc[aria-current], .gl-card[aria-current], .gl-lane a[aria-current], .sd-row[aria-current]").forEach(function (el) {
      el.removeAttribute("aria-current");
    });
    document.querySelectorAll(".gl-cur").forEach(function (el) { el.classList.remove("gl-cur"); });
    var n = null;
    document.querySelectorAll(".mc[data-window], .gl-card[data-window], .gl-lane a, .sd-row[data-window]").forEach(function (el) {
      var own = el.dataset.window || windowOf(el.getAttribute("href"));
      if (!id || own !== id) return;
      el.setAttribute("aria-current", "true");
      if (el.dataset.n) n = el.dataset.n;
    });
    if (n) {
      document.querySelectorAll('.gl-plot [data-n="' + n + '"]').forEach(function (el) {
        el.classList.add("gl-cur");
        if (el.classList.contains("gl-guide") && drop && id !== guided && moving()) {
          el.classList.remove("gl-drop");
          void el.offsetWidth; // restart the drop
          el.classList.add("gl-drop");
        }
      });
    }
    var here = document.querySelector('.gl-lane a[aria-current="true"]');
    if (here) roveLane(here);
    guided = id || null;
    rove();
  }

  // Hot pairing: pointing at a card, or its chip, lights the card, the chip, its peak dot and guide.
  function hot(n) {
    document.querySelectorAll(".gl-hot").forEach(function (el) { el.classList.remove("gl-hot"); });
    if (!n) return;
    document.querySelectorAll('.gl-card[data-n="' + n + '"], .gl-c3[data-n="' + n + '"], .gl-plot [data-n="' + n + '"]')
      .forEach(function (el) { el.classList.add("gl-hot"); });
  }
  function hotOf(t) {
    var el = t && t.closest && t.closest(".gl-card[data-n], .gl-c3[data-n]");
    return el ? el.dataset.n : null;
  }
  document.addEventListener("pointerover", function (e) { hot(hotOf(e.target)); });
  document.addEventListener("focusin", function (e) { hot(hotOf(e.target)); });
  document.addEventListener("focusout", function (e) {
    if (!hotOf(e.relatedTarget)) hot(null);
  });

  // The lane keeps one mark in the tab order (the open Moment's, else the one last focused, else
  // the first): ArrowLeft and ArrowRight (and Up and Down) move along it in time order; Home and
  // End go to either end.
  function laneMarks(lane) {
    return Array.prototype.slice.call(lane.querySelectorAll("a"));
  }
  function roveLane(a) {
    laneMarks(a.parentNode).forEach(function (b) { b.tabIndex = b === a ? 0 : -1; });
  }
  function isMark(el) {
    return !!el && !!el.matches && el.matches(".gl-lane > a");
  }

  // Under 760 px marks closer than a finger apart would sit on each other, so the lane takes two
  // rows (G1-phone.html). Left to right, a mark goes to row 1 when it is ROW px or more past row
  // 1's last mark, otherwise to whichever row has more room. The busiest three sit in row 2's band,
  // so a row-2 mark (.gl-r2) next to one is kept C3_CLEAR px from its centre. Wider than that,
  // every mark sits at its own time (data-x, in %). A tap on the lane that misses every mark opens
  // the nearest one within TAP px.
  var ROW = 22;
  var C3_CLEAR = 19;
  var TAP = 22;
  function placeLane() {
    var two = !!(narrow && narrow.matches);
    document.querySelectorAll(".gl-lane").forEach(function (lane) {
      var w = lane.clientWidth;
      var marks = laneMarks(lane);
      var at = function (a) { return parseFloat(a.getAttribute("data-x")) / 100 * w; };
      var c3s = marks.filter(function (a) { return a.classList.contains("gl-c3"); }).map(at);
      var last = [-Infinity, -Infinity];
      marks.forEach(function (a) {
        if (!a.classList.contains("gl-mk")) {
          if (two && w) last[1] = at(a) + 8; // a chip fills row 2's band at its time
          return;
        }
        a.classList.remove("gl-r2");
        a.style.left = a.getAttribute("data-x") + "%";
        if (!two || !w) return;
        var x = at(a);
        var g1 = x - last[0], g2 = x - last[1];
        var row = g1 >= ROW || g1 >= g2 ? 0 : 1;
        if (row === 1) {
          c3s.forEach(function (c) {
            if (Math.abs(x - c) < C3_CLEAR) x = x < c ? c - C3_CLEAR : c + C3_CLEAR;
          });
          a.classList.add("gl-r2");
        }
        last[row] = x;
        a.style.left = x.toFixed(1) + "px";
      });
    });
  }
  document.addEventListener("click", function (e) {
    var lane = e.target.closest && e.target.closest(".gl-lane");
    if (!lane || isMark(e.target.closest("a")) || !narrow || !narrow.matches) return;
    var box = lane.getBoundingClientRect();
    var best = null, gap = TAP;
    laneMarks(lane).forEach(function (a) {
      var r = a.getBoundingClientRect();
      var d = Math.abs(e.clientX - (r.left + r.width / 2));
      if (d <= gap) {
        gap = d;
        best = a;
      }
    });
    if (best && e.clientY >= box.top && e.clientY <= box.bottom) {
      e.preventDefault();
      best.click();
    }
  });

  // First paint of the home page only: the line draws left to right (360 ms), then the marks,
  // chips, dots and guides rise in time order, 45 ms apart from 120 ms (none later than 900 ms, so
  // all have risen before the class goes at 1400 ms). The class goes on once; a
  // redraw after a save doesn't run it again. With reduced motion nothing animates.
  function firstPaint() {
    if (!moving() || location.pathname !== "/") return;
    document.querySelectorAll(".gl-graph").forEach(function (g) {
      g.classList.add("gl-drawn");
      window.setTimeout(function () { g.classList.remove("gl-drawn"); }, 1400);
    });
  }

  // A Moment list keeps one row in the tab order: the open Moment's, else the one last focused,
  // else the first. ArrowDown and ArrowUp move between its rows.
  function rowsOf(list) {
    var out = [];
    Array.prototype.forEach.call(list.children, function (li) {
      var a = li.classList.contains("mc") ? li.querySelector(":scope > .mc-o") : null;
      if (a) out.push(a);
    });
    return out;
  }
  function rove() {
    document.querySelectorAll("ol.mcs").forEach(function (list) {
      var all = rowsOf(list);
      var pick = all.filter(function (a) { return a.parentNode.hasAttribute("aria-current"); })[0] ||
        all.filter(function (a) { return a.getAttribute("tabindex") === "0"; })[0] || all[0];
      all.forEach(function (a) { a.tabIndex = a === pick ? 0 : -1; });
    });
    // The lane comes from the server with every mark reachable (no script, no arrow keys); here it
    // keeps one tab stop: the open Moment's mark, else the one last focused, else the first.
    document.querySelectorAll(".gl-lane").forEach(function (lane) {
      var marks = laneMarks(lane);
      var pick = marks.filter(function (a) { return a.getAttribute("aria-current") === "true"; })[0] ||
        marks.filter(function (a) { return a.getAttribute("tabindex") === "0"; })[0] || marks[0];
      if (pick) roveLane(pick);
    });
  }
  function roveTo(a) {
    rowsOf(a.parentNode.parentNode).forEach(function (b) { b.tabIndex = b === a ? 0 : -1; });
  }
  function isRow(el) {
    return !!el && !!el.matches && el.matches("ol.mcs > li.mc > a.mc-o");
  }
  document.addEventListener("focusin", function (e) {
    if (isRow(e.target)) roveTo(e.target);
    else if (isMark(e.target)) roveLane(e.target);
  });

  // Opens ``href`` (a Moment or the All N list) in the lane. A new view lifts in; a Moment opened
  // from a numbered chip (``chip``) has that number lift into the lane instead (view transition
  // "lift", 200 ms), only where the browser has view transitions and motion is allowed.
  function load(href, focus, chip) {
    return fetch(panelUrl(href), { credentials: "same-origin" }).then(function (r) {
      return r.text().then(function (html) {
        if (!r.ok) throw new Error(problem(html));
        var fresh = pane.hidden || href !== current;
        function apply() {
          pane.innerHTML = html;
          current = href;
          pane.setAttribute("aria-label", isList(href) ? "All Moments" : "Moment");
          pane.hidden = false;
          sheetBehind();
          if (windowOf(href)) lastMoment = windowOf(href);
          mark(marked(href), true);
        }
        function after() {
          if (!focus) return;
          var h = pane.querySelector("h2");
          if (h) h.focus({ preventScroll: true });
          pane.scrollTop = 0;
        }
        if (chip && chip.isConnected && fresh && windowOf(href) && moving() && document.startViewTransition) {
          chip.style.viewTransitionName = "lift";
          var vt = document.startViewTransition(function () {
            unlift(chip);
            apply();
            var to = pane.querySelector("[data-lift]");
            if (to) to.style.viewTransitionName = "lift";
          });
          vt.finished.finally(function () {
            var to = pane.querySelector("[data-lift]");
            if (to) unlift(to);
          });
          return vt.updateCallbackDone.then(after);
        }
        apply();
        var body = pane.querySelector(".sd-in");
        if (fresh && body && moving()) body.classList.add("sd-enter");
        after();
      });
    });
  }

  // "‹ Answer" or Esc from a Moment cited in the answer: the answer comes back into the lane, with
  // focus on the citation that opened the Moment.
  function showAnswer() {
    if (!pane || !answerNode) return;
    pane.innerHTML = "";
    pane.appendChild(answerNode);
    current = null;
    listHref = null;
    answerFrom = false;
    pane.setAttribute("aria-label", "Answer");
    pane.hidden = false;
    sheetBehind();
    mark(null);
    var to = answerOpener && answerNode.contains(answerOpener) ? answerOpener : answerNode.querySelector(".sd-x");
    answerOpener = null;
    if (to) to.focus();
  }

  // Leaving the answer for anything but a Moment it cites (G-11): its running step stops, and the
  // address drops the thread, as the close link would.
  function leaveAnswer() {
    if (!answerNode) return;
    var busy = running ? answerNode.querySelector("#" + CSS.escape(running.id)) : null;
    if (busy && busy.hasAttribute("data-pending")) stopAsk(busy);
    var x = answerNode.querySelector(".sd-x");
    try {
      if (x) history.replaceState(null, "", x.getAttribute("href"));
    } catch (err) { /* the address keeps the thread; the page still works */ }
    answerNode = null;
    answerFrom = false;
    answerOpener = null;
  }

  // The ask line; ``shut`` keeps its "Try one" list closed until the next key or until focus leaves
  // the dock (closing the answer lands here without opening the list).
  function askLine(shut) {
    var d = document.querySelector(".dock");
    if (!d) return null;
    d.classList.toggle("sx-shut", !!shut);
    return d.querySelector("form.ask textarea");
  }

  function close() {
    if (!pane || pane.hidden) return;
    var id = current && windowOf(current);
    var answer = !!answerNode;
    leaveAnswer();
    pane.hidden = true;
    pane.innerHTML = "";
    pane.setAttribute("aria-label", "Moment");
    current = null;
    listHref = null;
    lastMoment = null;
    sheetBehind(); // before focus goes back to its row, which sits behind the sheet
    mark(null);
    // the answer goes back to the ask line, suggestions shut; the list to the link that opened it,
    // else to the All N button
    var back = answer ? askLine(true) : id ? backTo(id) : opener && opener.isConnected && !pane.contains(opener) ? opener : document.querySelector(".gl-all");
    opener = null;
    if (back) back.focus();
  }

  // Where focus goes when the Moment closes (G-11): the link that opened it, while it is still on
  // the page and still names this Moment; else its card; else its mark on the lane (most Moments
  // have no card); else its row.
  function backTo(id) {
    if (opener && opener.isConnected && !pane.contains(opener) && windowOf(opener.getAttribute("href")) === id) return opener;
    var card = document.querySelector('.gl-card[data-window="' + CSS.escape(id) + '"] .gl-co, .mc[data-window="' + CSS.escape(id) + '"] .mc-o');
    if (card) return card;
    var marks = document.querySelectorAll(".gl-lane a");
    for (var i = 0; i < marks.length; i++) {
      if (windowOf(marks[i].getAttribute("href")) === id) return marks[i];
    }
    return null;
  }

  function problem(html) {
    var doc = new DOMParser().parseFromString(html, "text/html");
    var msg = doc.querySelector(".msg");
    return msg ? msg.textContent.trim() : "Something went wrong. Reload the page and try again.";
  }

  function refreshList() {
    return fetch(location.pathname + location.search, { credentials: "same-origin" })
      .then(function (r) { return r.ok ? r.text() : null; })
      .then(function (html) {
        if (!html) return;
        var fresh = new DOMParser().parseFromString(html, "text/html").querySelector(".col");
        var col = document.querySelector(".col");
        if (fresh && col) col.innerHTML = fresh.innerHTML;
        placeLane(); // the new lane: second row for close marks on a phone
        mark(marked(current), false); // the current mark comes back, without a drop
        scan();
      });
  }

  function say(where, text) {
    var old = where.querySelector(".msg[role=alert]");
    if (old) old.remove();
    var p = document.createElement("p");
    p.className = "msg";
    p.setAttribute("role", "alert");
    p.textContent = text;
    where.prepend(p);
  }

  document.addEventListener("click", function (e) {
    var t = e.target;
    document.querySelectorAll("details.fx[open]").forEach(function (d) {
      if (!d.contains(t)) d.removeAttribute("open");
    });

    var menu = streamMenu();
    if (menu && !menu.contains(t)) menu.removeAttribute("open");
    var pick = t.closest("[data-theme-set]");
    if (pick) {
      store(theme(pick.getAttribute("data-theme-set")));
      return;
    }
    if (!pane || e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;

    var close_ = t.closest("[data-close]");
    if (close_ && pane.contains(close_)) {
      e.preventDefault();
      close();
      return;
    }
    var back_ = t.closest("a[data-answer]");
    if (back_ && pane.contains(back_) && answerNode) {
      e.preventDefault();
      showAnswer();
      return;
    }
    var open = t.closest("a[data-open]");
    if (open) {
      e.preventDefault();
      var href = open.getAttribute("href");
      if (answerNode && pane.contains(answerNode) && answerNode.contains(open)) {
        // a Moment the answer cites keeps the answer (and its running step); anything else leaves it
        if (windowOf(href)) {
          answerFrom = true;
          answerOpener = open;
        } else {
          leaveAnswer();
        }
      } else if (answerNode && !pane.contains(open)) {
        leaveAnswer();
      }
      if (!pane.contains(open)) {
        opener = open; // focus comes back here when the Moment closes
        listHref = null;
      } else if (!windowOf(href)) {
        listHref = null; // the list itself (‹ All N, or its order)
      } else if (current && isList(current)) {
        listHref = current; // a row of the list: Esc comes back to the list
      }
      // the number that lifts into the lane: a card's or row's number, or a numbered chip on the line
      var chip = open.querySelector(".gl-num, .sd-num") || (open.matches(".gl-c3") ? open : null);
      load(href, true, chip).catch(function (err) {
        location.href = open.getAttribute("href");
        return err;
      });
    }
  });

  document.addEventListener("submit", function (e) {
    var form = e.target;
    // asking leaves for the thread; so do the answer's own forms (chips, follow-ups, suggestions)
    if (!pane || !pane.contains(form) || form.hasAttribute("data-ask") || (answerNode && answerNode.contains(form))) return;
    e.preventDefault();
    var body = new URLSearchParams(new FormData(form));
    if (e.submitter && e.submitter.name) body.append(e.submitter.name, e.submitter.value);
    fetch(form.getAttribute("action"), {
      method: "POST",
      body: body,
      credentials: "same-origin",
    }).then(function (r) {
      if (r.ok) return load(current, false).then(refreshList).then(function () {
        var h = pane.querySelector("h2");
        if (h) h.focus();
      });
      return r.text().then(function (html) { say(form.closest(".menu, .sd-sec") || pane, problem(html)); });
    }).catch(function () {
      say(form.closest(".menu, .sd-sec") || pane, "Not saved: the page did not answer. Is chatbrain page still running?");
    });
  });

  // The ask box: Enter asks, Shift+Enter starts a new line; the box grows with the question.
  document.addEventListener("keydown", function (e) {
    var t = e.target;
    if (e.key !== "Enter" || e.shiftKey || e.isComposing || !t.matches || !t.matches("form[data-ask] textarea")) return;
    e.preventDefault();
    if (t.value.trim()) t.form.requestSubmit();
  });
  document.addEventListener("input", function (e) {
    var t = e.target;
    if (!t.matches || !t.matches("form[data-ask] textarea")) return;
    var d = t.closest(".dock");
    if (d) d.classList.remove("sx-shut"); // typing opens "Try one" again
    t.style.height = "auto";
    t.style.height = t.scrollHeight + "px";
  });
  // A chip picked from the menu runs at once (the + button does the same without this script).
  document.addEventListener("change", function (e) {
    var t = e.target;
    if (t.matches && t.matches("form.lf select[name=add]") && t.value) t.form.requestSubmit();
  });

  document.addEventListener("keydown", function (e) {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    var t = e.target;
    var esc = e.key === "Escape";
    // Esc in the dock with "Try one" open shuts it and keeps focus on the ask line
    var sug = esc && t.closest && t.closest(".dock") ? document.querySelector(".dock .sug") : null;
    if (sug && window.getComputedStyle(sug).display !== "none") {
      var line_ = askLine(true);
      if (line_ && t !== line_) line_.focus();
      return;
    }
    if (t.closest && t.closest("textarea, select, [contenteditable], input:not([type=radio]):not([type=checkbox])")) return;
    if (!esc && t.closest && t.closest("input")) return; // a choice in a Fix menu: only Esc acts there
    if (esc) {
      // One layer at a time (G-11): a running local model step on screen stops first, then an open
      // Fix menu, then Fix, then the stream menu, then a Moment opened from the list or the answer
      // goes back to it, then the lane closes.
      var busy = running ? blockOf(running.id) : null;
      if (busy && (!busy.isConnected || !busy.hasAttribute("data-pending"))) busy = null;
      var fixMenu = document.querySelector("details.fx[open]");
      var fixOpen = pane && pane.querySelector("details.sd-fx[open]");
      if (busy) {
        stopAsk(busy);
      } else if (fixMenu) {
        fixMenu.removeAttribute("open");
        var s = fixMenu.querySelector("summary");
        if (s) s.focus();
      } else if (fixOpen) {
        fixOpen.removeAttribute("open");
        var fs = fixOpen.querySelector("summary");
        if (fs) fs.focus();
      } else if (closeMenu(true)) {
        // focus goes back to the stream button
      } else if (pane && !pane.hidden) {
        var from = current && windowOf(current);
        if (listHref && from) {
          var list = listHref;
          listHref = null;
          load(list, false).then(function () {
            var row = pane.querySelector('.sd-row[data-window="' + CSS.escape(from) + '"] .sd-ro') || pane.querySelector("h2");
            if (row) row.focus();
          });
        } else if (answerNode && from) {
          showAnswer();
        } else {
          close();
        }
      }
      return;
    }
    if (isMark(t) && /^(ArrowLeft|ArrowRight|ArrowUp|ArrowDown|Home|End)$/.test(e.key)) {
      var marks = laneMarks(t.parentNode);
      var at = marks.indexOf(t);
      var go = e.key === "Home" ? 0 : e.key === "End" ? marks.length - 1
        : at + (e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : -1);
      e.preventDefault();
      if (marks[go]) {
        roveLane(marks[go]);
        marks[go].focus();
      }
      return;
    }
    if (e.key === "/") {
      var box = askLine(false);
      if (box) {
        e.preventDefault();
        box.focus();
      }
      return;
    }
    if ((e.key === "ArrowDown" || e.key === "ArrowUp") && isRow(t)) {
      var rows = rowsOf(t.parentNode.parentNode);
      var to = rows[rows.indexOf(t) + (e.key === "ArrowDown" ? 1 : -1)];
      e.preventDefault();
      if (to) {
        roveTo(to);
        to.focus();
      }
      return;
    }
    if (!pane) return;
    var key = e.key.toLowerCase();
    if (key === "j" || key === "k") {
      if (pane.hidden) {
        var first = document.querySelector(".gl-card .gl-co") || document.querySelector(".mc .mc-o");
        if (first && key === "j") {
          e.preventDefault();
          first.click();
        }
        return;
      }
      var step = pane.querySelector(key === "j" ? "[data-next]" : "[data-prev]");
      if (step) {
        e.preventDefault();
        step.click();
      }
    } else if (key === "f" && !pane.hidden) {
      var wrap = pane.querySelector("details.sd-fx");
      if (wrap) wrap.setAttribute("open", "");
      var fix = pane.querySelector("details.fx");
      if (fix) {
        e.preventDefault();
        fix.setAttribute("open", "");
        var item = fix.querySelector(".mi");
        if (item) item.focus();
      }
    }
  });
  // The local model (Task 10c). An answer that waits on the model carries data-model (its next
  // step) and data-stop. One step runs at a time, in page order: the script posts the step, marks
  // its row active and shows the step list with its seconds only once the step has run past
  // REVEAL_MS (F3, G-10: a fast answer shows only its settled line), then lands the answer block
  // the server sends back the moment it comes, which may wait on a second step (the caption).
  // Stop posts data-stop and keeps the keyword answer.
  var SLOW_AFTER_S = 10; // the "Still working" line shows after 10 s
  var REVEAL_MS = 150; // client side only: the server never holds an answer back
  var FAILED = "The local model step did not finish: the page did not answer. The keyword answer stays.";
  var queue = []; // ids of waiting answers, in page order
  var running = null; // {id, ctrl, started, shown}
  var ticker = null;
  var revealing = null; // the REVEAL_MS timer of the running step

  // An answer is on the page, or in the answer kept aside while a Moment it cites fills the lane.
  function blockOf(id) {
    return document.getElementById(id) || (answerNode ? answerNode.querySelector("#" + CSS.escape(id)) : null);
  }
  function secsOf(r) {
    return Math.floor((Date.now() - r.started) / 1000);
  }
  function tick(el, secs) {
    var out = el.querySelector("[data-secs]");
    if (out) out.textContent = String(secs);
    if (secs >= SLOW_AFTER_S) el.querySelectorAll(".slow[data-wait]").forEach(function (w) { w.hidden = false; });
  }
  function showWaiting(el, secs) {
    el.querySelectorAll("[data-wait]").forEach(function (w) {
      if (!w.classList.contains("slow")) w.hidden = false;
    });
    var lost = false; // focus on the settled line, which hides now: it moves to Stop
    el.querySelectorAll("[data-done]").forEach(function (d) {
      if (d.contains(document.activeElement)) lost = true;
      d.hidden = true;
    });
    el.setAttribute("aria-busy", "true");
    tick(el, secs);
    var stop = lost ? el.querySelector("button.stop") : null;
    if (stop) stop.focus();
  }
  // The running step: its row turns active at once; the list shows only past REVEAL_MS.
  function revealSteps(el, id) {
    var line = el.querySelector("li.wk");
    if (line) line.setAttribute("data-s", "active");
    window.clearTimeout(revealing);
    revealing = window.setTimeout(function () {
      revealing = null;
      var now = blockOf(id);
      if (!now || !running || running.id !== id || !now.hasAttribute("data-pending")) return;
      running.shown = true;
      showWaiting(now, secsOf(running));
    }, REVEAL_MS);
  }
  function settle(el) {
    el.querySelectorAll("[data-wait]").forEach(function (w) { w.hidden = true; });
    el.querySelectorAll("[data-done]").forEach(function (d) { d.hidden = false; });
    el.removeAttribute("aria-busy");
    el.removeAttribute("data-pending");
  }
  function finished(id) {
    if (!running || running.id !== id) return;
    window.clearInterval(ticker);
    ticker = null;
    window.clearTimeout(revealing);
    revealing = null;
    running = null;
  }
  // The answer the server sent back takes the old one's place at once, settled; one that waits on
  // the caption step shows its steps again only when that step runs past REVEAL_MS.
  function landAnswer(id, html) {
    var fresh = new DOMParser().parseFromString(html, "text/html").querySelector("article.blk");
    var old = blockOf(id);
    if (!fresh || !old) return null;
    var focused = old.contains(document.activeElement);
    var el = document.importNode(fresh, true);
    old.replaceWith(el);
    rove();
    if (focused) {
      var f = el.querySelector(".stp > summary");
      if (!f) {
        el.tabIndex = -1; // an answer with no steps (a refusal): focus stays on the answer
        f = el;
      }
      f.focus();
    }
    return el;
  }
  function failed(id, text) {
    var el = blockOf(id);
    if (!el) return;
    settle(el);
    say(el.querySelector(".ab") || el, text);
  }

  function scan() {
    document.querySelectorAll("article.blk[data-pending]").forEach(function (el) {
      if (running && running.id === el.id) {
        if (running.shown) showWaiting(el, secsOf(running));
        return;
      }
      // a queued answer keeps its settled keyword line until its step runs
      if (queue.indexOf(el.id) < 0) queue.push(el.id);
    });
    next();
  }

  function next() {
    if (running) return;
    var id;
    while ((id = queue.shift())) {
      var el = blockOf(id);
      if (el && el.hasAttribute("data-pending")) {
        step(el);
        return;
      }
    }
  }

  function step(el) {
    var id = el.id;
    var ctrl = window.AbortController ? new AbortController() : null;
    running = { id: id, ctrl: ctrl, started: Date.now(), shown: false };
    revealSteps(el, id);
    ticker = window.setInterval(function () {
      var now = blockOf(id);
      if (now && running && running.id === id) tick(now, secsOf(running));
    }, 1000);
    fetch(el.getAttribute("data-model"), {
      method: "POST",
      credentials: "same-origin",
      signal: ctrl ? ctrl.signal : undefined,
    }).then(function (r) {
      return r.text().then(function (html) {
        finished(id);
        if (!r.ok) return failed(id, problem(html));
        var now = landAnswer(id, html);
        if (now && now.hasAttribute("data-pending")) queue.unshift(id); // the caption step comes next
      });
    }).catch(function (err) {
      finished(id);
      if (!err || err.name !== "AbortError") failed(id, FAILED); // Stop aborts; it settles the block itself
    }).then(next);
  }

  function stopAsk(el) {
    var id = el.id;
    var line = el.querySelector("li.wk");
    var focused = el.contains(document.activeElement);
    if (running && running.id === id) {
      window.clearTimeout(revealing); // a step stopped before 150 ms never shows its list
      revealing = null;
    }
    if (line) {
      var icon = line.querySelector(".sx-i");
      line.setAttribute("data-s", "skipped");
      line.textContent = "";
      if (icon) line.appendChild(icon);
      line.appendChild(document.createTextNode(line.getAttribute("data-stopped")));
      line.tabIndex = -1;
      if (focused && !line.closest("[hidden]")) line.focus(); // the Stop button is gone; keep focus in this answer
    }
    el.querySelectorAll(".cap[data-wait], .slow[data-wait]").forEach(function (w) { w.hidden = true; });
    el.removeAttribute("data-pending");
    el.removeAttribute("aria-busy");
    var at = queue.indexOf(id);
    if (at >= 0) queue.splice(at, 1);
    var model = el.getAttribute("data-model");
    var wasRunning = running && running.id === id ? running : null;
    fetch(el.getAttribute("data-stop"), { method: "POST", credentials: "same-origin" }).then(function (r) {
      if (wasRunning && wasRunning.ctrl) wasRunning.ctrl.abort();
      if (!r.ok) return r.text().then(function (html) { failed(id, problem(html)); });
      // the step is stopped, so this only fetches the settled answer
      return fetch(model, { method: "POST", credentials: "same-origin" }).then(function (r2) {
        return r2.text().then(function (html) {
          if (r2.ok) landAnswer(id, html);
          else failed(id, problem(html));
        });
      });
    }).catch(function () {
      if (wasRunning && wasRunning.ctrl) wasRunning.ctrl.abort();
      failed(id, "Not stopped on the page: it did not answer. Is chatbrain page still running?");
    });
  }

  document.addEventListener("click", function (e) {
    var b = e.target.closest && e.target.closest("article.blk[data-pending] button.stop");
    if (!b) return;
    e.preventDefault();
    stopAsk(b.closest("article.blk"));
  });

  // The sticky dock can grow with a long question; its height keeps a focused row clear of it
  // (page.css reads --cb-dock-h in scroll-padding-bottom).
  var dock = document.querySelector(".dock");
  function clearDock() {
    root.style.setProperty("--cb-dock-h", (dock ? dock.offsetHeight : 0) + "px");
  }
  if (dock) {
    if (window.ResizeObserver) new ResizeObserver(clearDock).observe(dock);
    else window.addEventListener("resize", clearDock);
    clearDock();
    // "Try one" opens while focus is in the dock (CSS :focus-within). A press on a suggestion keeps
    // focus where it is, so the list stays open until the click lands (Safari does not focus
    // buttons on click). Leaving the dock lifts the shut that closing the answer set.
    dock.addEventListener("mousedown", function (e) {
      if (e.target.closest && e.target.closest(".sug")) e.preventDefault();
    });
    dock.addEventListener("focusout", function (e) {
      if (!e.relatedTarget || !dock.contains(e.relatedTarget)) dock.classList.remove("sx-shut");
    });
  }
  if (answerNode) {
    sheetBehind(); // a thread opens on its answer: a sheet on a phone, a side lane from 760 px
    // The sheet is a modal dialog, so focus goes into it: the answer the address names, else the
    // newest one, else the close link.
    if (pane.getAttribute("aria-modal") === "true") {
      var named = /^#a\d+$/.test(location.hash) ? document.getElementById(location.hash.slice(1)) : null;
      if (named && !answerNode.contains(named)) named = null;
      var arts = answerNode.querySelectorAll("article.blk");
      var into = named || arts[arts.length - 1] || answerNode.querySelector(".sd-x");
      if (into && into.tagName === "ARTICLE") into.tabIndex = -1;
      if (into) into.focus();
    }
  }

  placeLane();
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(placeLane);
  window.addEventListener("resize", placeLane);
  var column = document.querySelector(".col"); // also catches a scrollbar coming or going
  if (column && window.ResizeObserver) new ResizeObserver(function () { placeLane(); }).observe(column);
  firstPaint();
  rove();
  scan();
})();
