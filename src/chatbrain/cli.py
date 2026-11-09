"""The ``chatbrain`` command: ``import``, ``deletions``, ``window-list``, ``expire``,
``assign``, ``repeats``, ``pick-rare``, ``ls-setup``, ``ls-push``, ``ls-pull``, ``suggest``,
``train``, ``page``, ``ask-eval`` and ``demo``.

Run as ``uv run chatbrain [--data-dir PATH] <command>``. ``--data-dir`` overrides
``CHATBRAIN_DATA`` for one run and is checked by ``paths.data_dir`` the same way
(ruling C17). The store is ``<data>/cb.db``.

Label Studio: the client comes from ``<data>/.env`` (``LS_TOKEN``, ``LS_AUTH``,
``LS_URL``) and only when the data folder is the real ``CHATBRAIN_DATA`` (C17), so a
``--data-dir`` run on made-up data never reaches the real Label Studio.

Every command but ``ask-eval`` runs ``expiry.run`` first, on ``<data>/imported`` and
``<data>/models``; when Label Studio tasks are due for deletion, expiry pulls every project's
annotations before it deletes them (B11). With a Label Studio client it first records every
task the store lost track of (``sync.adopt_untracked``, a push stopped between Label Studio's
import and its ``ls_tasks`` row), so expiry deletes that task's text too; that is best effort,
and a failure counts as a Label Studio error. A command that applies deletion checks runs expiry again after them,
also when a refused check file stops it, since the good ones were applied. ``expire --now``
sets the clock for that run only, for checks on made-up data; the other commands use
the real UTC time.

Every command also prints, counts only, how many files in ``<data>/inbox`` were not imported
(export and unknown ``.json`` files: refused or never imported) and how many imported sources
could not be deleted yet, and how many of each are past their text expiry; they may still hold
chat text and nothing deletes them for the owner (B10). An export copy Label Studio's client could
not delete is counted too, and keeps a compaction due so the next expiry retries it.

``expire`` exits 1 while text that is due to go may still be held: a Label Studio error,
task deletions or compaction pending, a word-model file not deleted yet or missing, or an
inbox file or import source past its text expiry. The daily scheduled run (C27) then shows a
failed result instead of success.

``ls-push`` refuses while Label Studio deletions or compaction are pending, applies the
waiting deletion checks (and refuses on a refused check file), then needs the freshness
gate (C36) before it imports anything.

``suggest`` asks the local teacher (Ollama on ``127.0.0.1:11434``, ``teacher.run``) about
windows assigned ``suggested``. Like ``ls-push`` it applies the waiting deletion checks
first (a refused file stops it); ``teacher.run`` then needs the freshness gate and
``ollama.check`` (loopback, version 0.31.2 or newer) before any window is sent. It prints
how many rows it recorded, the time per window, and the totals by error code.

``train --rung {0,1} [--origin {aron,synthetic}] --seed S [--emote-mode {name,id}]``
retrains one rung from scratch on the labels still held. Like ``suggest`` it applies the
waiting deletion checks first (a refused file stops it; expiry then deletes the files of
word models those checks reached, C5), then needs the freshness gate. Then it fits,
calibrates, reports and writes the model and its manifest (``manifest.write``, which
retires the older models of the same rung and origin), and stores predictions for every
window that still has text. It prints, per question, the threshold and the test numbers
beside how many windows were graded, the owner's own repeat consistency (C28) and which model
now answers each question. scikit-learn loads only for this command.

``page [--host 127.0.0.1|localhost] [--port 14612] [--model NAME] [--cpu-captions]
[--no-model]`` serves the local Moment page (``page.app``) on this PC only, until Ctrl+C.
The page itself shows chat text while it lives, so it is for your own browser; the command
prints only its address. Asking uses the local model (``--model``, default qwen3:8b, through
Ollama on 127.0.0.1:11434) to read questions and caption answers, and falls back to the keyword
path when it can't run; ``--no-model`` keeps the keyword path alone, and ``--cpu-captions``
keeps captions on while the model runs on the CPU. Building the page asks Ollama nothing; the
model is preloaded once when the server starts.

``ask-eval [--model NAME] [--questions PATH]`` reads the 60 test questions with the local
model and with the keyword path, on a made-up world in a temporary folder, and prints the
exact-intent and exact-filter rates, the bar (90% intent, 75% filters) and the failure codes.
It never prints a question and reads no stored data, so it runs no expiry.

``demo --seed S [--with-ollama] [--with-label-studio]`` runs the whole loop on made-up chat
in ``<data>/demo``, its own data folder with its own store (checked by ``paths.data_dir``),
after this command's own expiry on ``<data>``. It starts clean: the last demo's store and
files are deleted first, once the Label Studio tasks that store still tracks are deleted
and compacted (without ``--with-label-studio`` it refuses while any are left). The clock
is the real UTC time (C17). Then, each step through the same code as its command:

1. import ``fake.make_export(S, 480, 120, now)`` and the showcase stream
   (``fake.make_showcase``, the newest stream: a chant, repeated emotes, critical turns);
2. apply a made-up deletion check that drops 3 messages (the import already counts as
   coverage, B7);
3. ``assign``; 4. a made-up label for every window from ``fake.truth`` (origin and
   provenance ``synthetic``, project = the window's exposure);
5. with ``--with-ollama``, ``suggest --limit 3`` (``teacher.run``: version check and gate);
6. ``train --rung 0`` and ``--rung 1 --origin synthetic --seed S``;
7. with ``--with-label-studio``, push up to 10 windows to each demo project (titled
   ``BrainChat demo · ...``, B20): suggested windows need a suggestion, and the 10 repeat
   windows get a ``repeats`` row due now, so they go in with no prediction and no earlier
   label (C18);
8. ``expire --now <now + 5 days>``, which expires the windows of streams that ended 9 or
   more days ago (some, never all);
9. delete every demo Label Studio task still there (pull first, then delete, then
   compact: B11), search for marker lines that should be gone (expired windows' in the
   whole data folder, deleted tasks' under Label Studio's folder, sent windows' in
   Ollama's ``*.log`` files), and print a summary and the command that opens the page.

``--with-label-studio`` needs the real data folder, like every Label Studio command (C17).

Output is counts and status only: never chat text and never window IDs, so an agent
may run these commands (see ``AGENTS.md``).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import re
import sys
import time
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from . import assign, expiry, fake, ingest, narrator, ollama, paths, teacher
from . import scheme as scheme_module
from .export_format import ExportRejected, iso, parse_dt, parse_export
from .lstudio import client as lsclient
from .lstudio import config as lsconfig
from .lstudio import sync
from .store import Store

DB_NAME = "cb.db"
WINDOW_LIST_NAME = "window-list.json"
ENV_FILE = ".env"
COMPOSE_FILE = paths.REPO_ROOT / "compose.yaml"
NOT_CONFIGURED = "Label Studio is not configured: set LS_TOKEN in the data folder's .env (see README)"
PAGE_HOST = "127.0.0.1"  # = page.app.HOST; the page module (FastAPI, uvicorn) loads only for ``page``
PAGE_HOSTS = ("127.0.0.1", "localhost")
PAGE_PORT = 14612
ASK_QUESTIONS = paths.REPO_ROOT / "tests" / "data" / "ask_questions.json"
DEMO_DIR = "demo"  # under the data folder
DEMO_MOMENTS, DEMO_ORDINARY = 480, 120
DEMO_DROPS = 3  # messages the made-up deletion check drops
DEMO_SUGGEST = 3  # windows the teacher is asked about
DEMO_PUSH = 10  # tasks per Label Studio project
DEMO_EXPIRE_AFTER = timedelta(days=5)

Command = Callable[[Store, Path, datetime, argparse.Namespace], int]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        data = _data_dir(args.data_dir)
    except paths.UnsafeDataDir as exc:
        print(f"chatbrain: {exc}", file=sys.stderr)
        return 1
    now = args.now if getattr(args, "now", None) is not None else datetime.now(timezone.utc)
    args.ls = _label_studio(data)
    try:
        with Store(data / DB_NAME) as store:
            return args.command(store, data, now, args)
    finally:
        close = getattr(args.ls, "close", None)
        if close is not None:
            close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chatbrain",
        description=(
            "BrainChat: import, deletion checks, expiry, assignments, Label Studio, teacher suggestions,"
            " training and the local Moment page."
        ),
    )
    parser.add_argument("--data-dir", metavar="PATH", help="data folder for this run (default: CHATBRAIN_DATA)")
    sub = parser.add_subparsers(dest="name", required=True, metavar="COMMAND")

    p = sub.add_parser("import", help="import export files (default: every export in <data>/inbox), then delete them")
    p.add_argument("paths", nargs="*", metavar="PATH")
    p.set_defaults(command=_import)

    p = sub.add_parser("deletions", help="apply the deletion checks in <data>/inbox and show the freshness gate")
    p.set_defaults(command=_deletions)

    p = sub.add_parser("window-list", help="write the window list a deletion check answers")
    p.add_argument("--out", type=Path, metavar="PATH", help=f"where to write it (default: <data>/{WINDOW_LIST_NAME})")
    p.set_defaults(command=_window_list)

    p = sub.add_parser("expire", help="delete text whose time is up")
    p.add_argument("--now", type=_aware_datetime, metavar="ISO", help="clock for this run, with an explicit offset")
    p.set_defaults(command=_expire)

    p = sub.add_parser("assign", help="assign windows with text to blind (1 in 4) or suggested, once each")
    p.add_argument("--seed", required=True, type=_seed, help="the assignment seed, recorded with each row")
    p.add_argument("--all-blind", action="store_true", help="pilot mode: every window assigned now is blind")
    p.set_defaults(command=_assign)

    p = sub.add_parser("repeats", help="schedule 1 in 10 labeled windows for a repeat 3-7 days after the label")
    p.add_argument("--seed", required=True, type=_seed, help="the repeat seed")
    p.set_defaults(command=_repeats)

    p = sub.add_parser("pick-rare", help="mark windows picked on purpose for rare values")
    p.add_argument("window_ids", nargs="+", metavar="WINDOW_ID")
    p.set_defaults(command=_pick_rare)

    p = sub.add_parser("ls-setup", help="create or update the three Label Studio projects")
    p.set_defaults(command=_ls_setup)

    p = sub.add_parser("ls-push", help="import windows into one Label Studio project, one task each")
    p.add_argument("--kind", required=True, choices=lsclient.KINDS, help="the project")
    p.add_argument("--limit", required=True, type=_limit, metavar="N", help="at most this many windows")
    p.set_defaults(command=_ls_push)

    p = sub.add_parser("ls-pull", help="store your Label Studio annotations as labels")
    p.set_defaults(command=_ls_pull)

    p = sub.add_parser("suggest", help="ask the local teacher (Ollama) to suggest answers for suggested windows")
    p.add_argument("--limit", required=True, type=_limit, metavar="N", help="at most this many windows")
    p.add_argument(
        "--model", default=teacher.DEFAULT_MODEL, type=_model, metavar="NAME",
        help=f"the Ollama model (default: {teacher.DEFAULT_MODEL})",
    )
    p.set_defaults(command=_suggest)

    p = sub.add_parser("train", help="retrain one rung from scratch, write its manifest and store predictions")
    p.add_argument("--rung", required=True, type=int, choices=(0, 1), help="0: counts, 1: TF-IDF over the words")
    p.add_argument("--origin", choices=("aron", "synthetic"), help="whose labels (needed when both exist)")
    p.add_argument("--seed", required=True, type=_int_seed, metavar="S", help="the split and classifier seed")
    p.add_argument(
        "--emote-mode", default="name", choices=("name", "id"), help="rung 1: emotes as names (default) or IDs"
    )
    p.set_defaults(command=_train)

    p = sub.add_parser("page", help="serve the local Moment page on 127.0.0.1:14612 until Ctrl+C")
    p.add_argument(
        "--host", default=PAGE_HOST, choices=PAGE_HOSTS, help=f"the address to listen on (default: {PAGE_HOST})"
    )
    p.add_argument("--port", default=PAGE_PORT, type=_port, metavar="N", help=f"default: {PAGE_PORT}")
    p.add_argument(
        "--model", default=narrator.DEFAULT_MODEL, type=_model, metavar="NAME",
        help=f"the local model that reads questions and writes captions (default: {narrator.DEFAULT_MODEL})",
    )  # fmt: skip
    p.add_argument("--cpu-captions", action="store_true", help="write captions even while the model runs on the CPU")
    p.add_argument("--no-model", action="store_true", help="ask with the keyword path alone; never call Ollama")
    p.set_defaults(command=_page)

    p = sub.add_parser("ask-eval", help="score the local model's reading of the 60 test questions (rates only)")
    p.add_argument(
        "--model", default=narrator.DEFAULT_MODEL, type=_model, metavar="NAME",
        help=f"the Ollama model (default: {narrator.DEFAULT_MODEL})",
    )  # fmt: skip
    p.add_argument("--questions", type=Path, default=ASK_QUESTIONS, metavar="PATH", help="the question file")
    p.set_defaults(command=_ask_eval)

    p = sub.add_parser("demo", help="run the whole loop on made-up chat in <data>/demo")
    p.add_argument("--seed", required=True, type=_int_seed, metavar="S", help="the made-up chat and every other seed")
    p.add_argument("--with-ollama", action="store_true", help="also ask the local teacher about 3 suggested windows")
    p.add_argument(
        "--with-label-studio", action="store_true",
        help="also push 10 windows to each demo Label Studio project, then delete them (real data folder only)",
    )  # fmt: skip
    p.set_defaults(command=_demo)
    return parser


def _aware_datetime(value: str) -> datetime:
    try:
        return parse_dt(value, "--now")
    except ExportRejected:
        raise argparse.ArgumentTypeError(
            "must be RFC 3339 text with an explicit offset, like 2026-11-06T12:00:00Z"
        ) from None


def _limit(value: str) -> int:
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a whole number >= 0") from None
    if n < 0:
        raise argparse.ArgumentTypeError("must be a whole number >= 0")
    return n


_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/+-]{0,199}")


def _model(value: str) -> str:
    if not _MODEL.fullmatch(value):
        raise argparse.ArgumentTypeError("must be an Ollama model name such as qwen3:8b")
    return value


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError:
        port = 0
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("a port is a whole number from 1 to 65535")
    return port


def _seed(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("must not be empty")
    return value


def _int_seed(value: str) -> int:
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a whole number >= 0") from None
    if n < 0:
        raise argparse.ArgumentTypeError("must be a whole number >= 0")
    return n


def _data_dir(override: str | None) -> Path:
    if override is None:
        return paths.data_dir(os.environ)
    return paths.data_dir({**os.environ, paths.ENV_VAR: override})


def _label_studio(data: Path) -> lsclient.LabelStudio | None:
    """The Label Studio client from ``<data>/.env``, only when ``data`` is the real data folder (C17).

    None when ``CHATBRAIN_DATA`` is unset or unsafe, when ``data`` is another folder
    (a ``--data-dir`` run), when ``.env`` or its ``LS_TOKEN`` is missing, or when its
    settings are unusable (the key is named on stderr, never its value). Without a
    client, expiry keeps Label Studio deletions pending and every command prints how
    many there are.
    """
    return _ls_client(data, lsclient.TITLES)


def _demo_label_studio(data: Path) -> lsclient.LabelStudio | None:
    """The demo's client: the same Label Studio and rules as ``_label_studio``, its own projects (B20).

    Tests replace this.
    """
    return _ls_client(data, lsclient.DEMO_TITLES)


def _ls_client(data: Path, titles: dict[str, str]) -> lsclient.LabelStudio | None:
    try:
        real = paths.resolve(os.environ)  # checked only: a --data-dir run creates nothing in the real folder
    except paths.UnsafeDataDir:
        return None
    if os.path.normcase(str(real)) != os.path.normcase(os.path.realpath(data)):
        return None
    try:
        return lsclient.from_env(real / ENV_FILE, data_dir=real, compose_file=COMPOSE_FILE, titles=titles)
    except lsclient.ConfigError as exc:
        print(f"chatbrain: Label Studio is not configured: {exc}", file=sys.stderr)
        return None


_LS_ERRORS = {
    "NotDeleted": "Label Studio: not deleted (it answered, but the task is still there)",
    "CompactionFailed": "Label Studio: compaction failed",
    "LabelStudioError": "Label Studio: refused the request",
}


def _run_expiry(
    store: Store,
    data: Path,
    now: datetime,
    ls: lsclient.LabelStudio | None,
    *,
    always: bool = False,
    show_held: bool = True,
) -> expiry.ExpiryResult:
    pulled: list[sync.SyncResult] = []

    def pull_first() -> None:  # B11: annotations are stored before their tasks go
        pulled.append(sync.pull(store, ls, scheme_module.load(), now))

    adopt_error = None
    if ls is not None:
        try:  # a task a stopped push never recorded must still go when its window's text does
            adopted = sync.adopt_untracked(store, ls)
        except Exception as exc:  # unreachable or refused: reported below like any Label Studio error
            adopt_error = type(exc).__name__
        else:
            _print_adopted(adopted.recorded, adopted.unknown)
        _export_copies(store, ls, now)  # a copy that export left keeps a compaction due: this run retries it
    r = expiry.run(
        store, now, data / "imported", ls, before_delete=pull_first if ls is not None else None,
        models_dir=data / "models", inbox=data / "inbox",
    )
    copies = _export_copies(store, ls, now) if ls is not None else 0
    if adopt_error is not None and r.ls_error is None:
        r = dataclasses.replace(r, ls_error=adopt_error)
    if copies:
        r = dataclasses.replace(r, ls_compaction_pending=True)
    if always or r.windows_expired or r.messages_deleted or r.files_deleted:
        print(
            f"expiry: {r.windows_expired} window(s) expired, {r.messages_deleted} message(s) deleted,"
            f" {r.files_deleted} file(s) deleted"
        )
    for p in pulled:
        if p.written or p.invalid:
            print(f"Label Studio: pulled {p.written} label(s) before deleting tasks, {len(p.invalid)} invalid")
    if r.ls_pull_error:
        line = f"Label Studio: labels were not pulled first ({r.ls_pull_error})"
        if r.ls_unsynced_deleted:
            line += (
                f"; {r.ls_unsynced_deleted} deleted task(s) had no stored label, so their annotations may be lost"
            )
        print(line)
    if r.ls_tasks_deleted:
        print(f"Label Studio: {r.ls_tasks_deleted} task(s) deleted")
    if r.ls_error:
        print(_LS_ERRORS.get(r.ls_error, f"Label Studio: unreachable ({r.ls_error})"))
    if copies:
        _print_export_copies(copies)
    if r.ls_tasks_pending:
        print(f"Label Studio: {r.ls_tasks_pending} task deletion(s) pending")
    if r.ls_compaction_pending:
        print("Label Studio: compaction pending")
    if r.models_deleted:
        print(f"models: {r.models_deleted} word model file(s) deleted")
    if r.models_pending:
        print(f"models: {r.models_pending} word model file(s) retired but not deleted yet; expiry retries them")
    if r.models_missing:
        print(
            f"models: {r.models_missing} word model(s) due for deletion have no file or manifest in this data"
            " folder's models/; nothing deleted for them (was the folder copied without models/?)"
        )
    if r.model_leftovers_deleted:
        print(f"models: {r.model_leftovers_deleted} leftover model file(s) of an unfinished write deleted")
    if show_held:
        _print_held(r)
    return r


def _print_held(h: expiry.ExpiryResult | expiry.HeldFiles) -> None:
    """Counts only: files that may still hold chat text, and how many are past their text expiry."""
    if h.inbox_files:
        print(
            f"inbox: {h.inbox_files} file(s) not imported may still hold chat text, {h.inbox_past} past its text"
            " expiry; import or delete them"
        )
    if h.sources_left:
        print(
            f"imported: {h.sources_left} source file(s) the import could not delete yet, {h.sources_left_past} past"
            " its text expiry; expiry retries them"
        )


def _export_copies(store: Store, ls: object, now: datetime) -> int:
    """Export copies the client could not delete (B13); while any is left a compaction stays due,
    so the next expiry empties the export folders again."""
    left = getattr(ls, "export_copies_left", 0)
    if left:
        with store.transaction() as conn:
            conn.execute("INSERT OR IGNORE INTO ls_compaction (id, due_at) VALUES (1, ?)", (iso(now),))
    return left


def _print_export_copies(n: int) -> None:
    print(f"Label Studio: {n} export copy file(s) could not be deleted yet; expiry retries them")


def _still_held(r: expiry.ExpiryResult) -> bool:
    """True while text that is due to go may still be held somewhere expiry could not reach."""
    return bool(
        r.ls_error or r.ls_tasks_pending or r.ls_compaction_pending or r.models_pending or r.models_missing
        or r.inbox_past or r.sources_left_past
    )  # fmt: skip


# --- commands ---------------------------------------------------------------------------------------


def _import(store: Store, data: Path, now: datetime, args: argparse.Namespace) -> int:
    _run_expiry(store, data, now, args.ls, show_held=False)  # said once, after the files below
    code = 0
    if args.paths:
        files = [Path(p) for p in args.paths]
        unknown: list[Path] = []
        waiting_checks = 0
    else:
        kinds = ingest.inbox_files(data / "inbox")
        files, unknown, waiting_checks = kinds["export"], kinds["unknown"], len(kinds["deletion-check"])
        if not files:
            print("no export files waiting")
    for path in files:
        try:
            r = ingest.import_export(store, path, data / "imported", now)
        except ExportRejected as exc:
            print(f"{path.name}: refused, left in place with its text: {exc}")
            code = 1
            continue
        except OSError as exc:
            print(f"{path.name}: could not be read ({type(exc).__name__})")
            code = 1
            continue
        line = (
            f"{path.name}: {r.windows_added} added, {r.windows_duplicate} duplicate,"
            f" {r.skipped_language} skipped (language), {r.skipped_expired} skipped (expired)"
        )
        if not r.source_deleted:
            line += "; the file could not be deleted yet, expiry retries it"
        print(line)
        n = r.streams_written  # counts only: never a stream ID, title or channel
        print(
            f"streams: {n} activity {'line' if n == 1 else 'lines'} stored,"
            f" {r.streams_expired_skipped} skipped (expired)"
        )
    for path in unknown:
        print(f"{path.name}: refused, not a BrainChat file")
        code = 1
    if waiting_checks:
        print(f"{waiting_checks} deletion check(s) waiting; run `chatbrain deletions`")
    _print_held(expiry.held_files(store, data / "inbox", now))
    return code


def _deletions(store: Store, data: Path, now: datetime, args: argparse.Namespace) -> int:
    _run_expiry(store, data, now, args.ls)
    r = ingest.apply_inbox_checks(store, data / "inbox", now=now)
    if r.checks_applied:
        print(
            f"applied {r.checks_applied} check(s), dropped {r.messages_dropped} message(s)"
            f" in {len(r.windows)} window(s)"
        )
        _run_expiry(store, data, now, args.ls, show_held=False)  # deletes or queues the changed windows' Label Studio tasks, then VACUUM
    else:
        print("no deletion checks waiting")
    for line in r.refused:
        print(f"refused, left in place: {line}")
    try:
        ingest.require_fresh_deletion_check(store, now)
        print("deletion check: fresh")
    except ingest.StaleDeletionCheck as exc:
        print(f"deletion check: stale ({exc})")
    return 1 if r.refused else 0


def _window_list(store: Store, data: Path, now: datetime, args: argparse.Namespace) -> int:
    _run_expiry(store, data, now, args.ls)
    out = args.out if args.out is not None else data / WINDOW_LIST_NAME
    n = ingest.write_window_list(store, out)
    print(f"wrote {n} window(s) to {out}")
    return 0


def _expire(store: Store, data: Path, now: datetime, args: argparse.Namespace) -> int:
    r = _run_expiry(store, data, now, args.ls, always=True)
    if _still_held(r):  # C27: the scheduled run must show a failure, not success
        print("expire: not finished; text that is due to go may still be held (see above)")
        return 1
    return 0


def _assign(store: Store, data: Path, now: datetime, args: argparse.Namespace) -> int:
    _run_expiry(store, data, now, args.ls)
    n = assign.assign_new(store, args.seed, now, all_blind=args.all_blind)
    t = assign.totals(store)
    mode = " (all blind)" if args.all_blind else ""
    print(f"assigned {n} window(s){mode}; assignments now: {t['blind']} blind, {t['suggested']} suggested")
    return 0


def _repeats(store: Store, data: Path, now: datetime, args: argparse.Namespace) -> int:
    _run_expiry(store, data, now, args.ls)
    n = assign.schedule_repeats(store, args.seed, now)
    print(f"scheduled {n} repeat(s); repeats now: {assign.totals(store)['repeats']}")
    return 0


def _pick_rare(store: Store, data: Path, now: datetime, args: argparse.Namespace) -> int:
    _run_expiry(store, data, now, args.ls)
    try:
        n = assign.pick_rare(store, args.window_ids)
    except assign.UnknownWindows as exc:
        print(f"pick-rare: refused: {exc}")  # counts only: the IDs are never echoed
        return 1
    already = len(set(args.window_ids)) - n
    print(
        f"marked {n} window(s) picked for rare values ({already} already marked);"
        f" picked now: {assign.totals(store)['picked_rare']}"
    )
    return 0


def _ls_setup(store: Store, data: Path, now: datetime, args: argparse.Namespace) -> int:
    _run_expiry(store, data, now, args.ls)
    if args.ls is None:
        print(NOT_CONFIGURED)
        return 1
    view = lsconfig.label_config(scheme_module.load())
    try:
        for kind in lsclient.KINDS:
            args.ls.ensure_project(kind, view)
    except (httpx.HTTPError, lsclient.LabelStudioError) as exc:
        print(f"Label Studio: setup failed ({type(exc).__name__})")
        return 1
    print(f"Label Studio: {len(lsclient.KINDS)} project(s) ready")
    return 0


def _ls_blocked(r: expiry.ExpiryResult) -> bool:
    """True (and says why) while Label Studio still holds text that should be gone."""
    if r.ls_tasks_pending or r.ls_compaction_pending or r.ls_error:
        print("Label Studio: nothing pushed while deletions or compaction are pending; run `chatbrain expire`")
        return True
    return False


def _ls_push(store: Store, data: Path, now: datetime, args: argparse.Namespace) -> int:
    r = _run_expiry(store, data, now, args.ls)
    if args.ls is None:
        print(NOT_CONFIGURED)
        return 1
    if _ls_blocked(r):
        return 1
    try:
        changed = ingest.apply_pending_checks(store, data / "inbox", now=now)
    except ExportRejected as exc:
        print(f"refused, left in place: {exc}")
        _run_expiry(store, data, now, args.ls, show_held=False)  # the good checks were applied: act on them now (C5)
        return 1
    if changed:
        print(f"applied deletion checks: {len(changed)} window(s) changed")
        if _ls_blocked(_run_expiry(store, data, now, args.ls, show_held=False)):
            return 1
    try:
        ingest.require_fresh_deletion_check(store, now)
    except ingest.StaleDeletionCheck as exc:
        print(f"deletion check: stale ({exc})")
        return 1
    try:
        p = sync.push(store, args.ls, scheme_module.load(), args.kind, args.limit, now)
    except (httpx.HTTPError, lsclient.LabelStudioError) as exc:
        print(f"Label Studio: push failed ({type(exc).__name__})")
        return 1
    line = f"pushed {p.pushed} task(s) to {args.kind}"
    if args.kind == "suggested":
        line += f" ({p.with_prediction} with a suggestion)"
    print(line)
    _print_adopted(p.adopted, p.unknown)
    copies = _export_copies(store, args.ls, now)
    if copies:
        _print_export_copies(copies)
    if p.error:
        print(f"Label Studio: push stopped ({p.error})")
        return 1
    return 0


def _ls_pull(store: Store, data: Path, now: datetime, args: argparse.Namespace) -> int:
    _run_expiry(store, data, now, args.ls)
    if args.ls is None:
        print(NOT_CONFIGURED)
        return 1
    try:
        adopted = sync.adopt_untracked(store, args.ls)
        r = sync.pull(store, args.ls, scheme_module.load(), now)
    except (httpx.HTTPError, lsclient.LabelStudioError) as exc:
        print(f"Label Studio: pull failed ({type(exc).__name__})")
        copies = _export_copies(store, args.ls, now)
        if copies:
            _print_export_copies(copies)
        return 1
    _print_adopted(adopted.recorded, adopted.unknown)
    copies = _export_copies(store, args.ls, now)
    if copies:
        _print_export_copies(copies)
    # Reasons name fields and rules only, never values, window IDs or chat text.
    print(f"written: {r.written}, invalid: {r.invalid}, cancelled: {r.skipped}")
    return 1 if r.invalid else 0


def _ollama_client() -> httpx.Client:
    """The loopback Ollama client (tests replace this)."""
    return ollama.make_client()


def _suggest(store: Store, data: Path, now: datetime, args: argparse.Namespace) -> int:
    _run_expiry(store, data, now, args.ls)
    try:
        changed = ingest.apply_pending_checks(store, data / "inbox", now=now)
    except ExportRejected as exc:
        print(f"refused, left in place: {exc}")
        _run_expiry(store, data, now, args.ls, show_held=False)  # the good checks were applied: act on them now (C5)
        return 1
    if changed:
        print(f"applied deletion checks: {len(changed)} window(s) changed")
        _run_expiry(store, data, now, args.ls, show_held=False)  # deletes or queues the changed windows' Label Studio tasks
    code = 0
    started = time.monotonic()
    try:
        with _ollama_client() as client:
            recorded = teacher.run(store, scheme_module.load(), now, args.limit, client, args.model)
    except ingest.StaleDeletionCheck as exc:
        print(f"deletion check: stale ({exc})")
        return 1
    except ollama.OllamaError as exc:
        print(f"Ollama: {exc}")
        code = 1
    else:
        line = f"recorded {recorded} suggestion(s)"
        if recorded:
            elapsed = time.monotonic() - started
            line += f" in {elapsed:.0f} s ({elapsed / recorded:.1f} s per window)"
        print(line)
    t = teacher.totals(store)
    answers = t.pop("answers", 0)
    errors = ", ".join(f"{name} {n}" for name, n in sorted(t.items())) or "none"
    print(f"suggestions now: {answers} with answers; errors: {errors}")
    return code


def _train(store: Store, data: Path, now: datetime, args: argparse.Namespace) -> int:
    from . import manifest, metrics, train  # scikit-learn loads only for this command

    _run_expiry(store, data, now, args.ls)
    try:
        changed = ingest.apply_pending_checks(store, data / "inbox", now=now)
    except ExportRejected as exc:
        print(f"refused, left in place: {exc}")
        _run_expiry(store, data, now, args.ls, show_held=False)  # the good checks were applied: act on them now (C5)
        return 1
    if changed:
        print(f"applied deletion checks: {len(changed)} window(s) changed")
        _run_expiry(store, data, now, args.ls, show_held=False)  # deletes word-model files and Label Studio tasks they reach (C5)
    try:
        ingest.require_fresh_deletion_check(store, now)
    except ingest.StaleDeletionCheck as exc:
        print(f"deletion check: stale ({exc})")
        return 1

    scheme = scheme_module.load()
    try:
        ds = train.load_dataset(store, scheme, args.origin)
        if not ds.rows:
            print("train: refused: no labels to train on")
            return 1
        sp = train.split(ds, args.seed)
    except ValueError as exc:  # MixedOrigins, too few channels, a bad label row: rules and counts only
        print(f"train: refused: {exc}")
        return 1
    model = train.fit(args.rung, ds, sp, args.seed, args.emote_mode)
    thresholds = train.calibrate(model, ds, sp)
    report = train.report(model, ds, sp, thresholds)
    with store.lock:
        before = {
            r[0]
            for r in store.conn.execute(
                "SELECT model_id FROM manifests WHERE rung = ? AND origin = ? AND retired_at IS NULL",
                (args.rung, ds.origin),
            )
        }
    try:
        path = manifest.write(
            model, ds, sp, thresholds, report, seed=args.seed, store=store, models_dir=data / "models",
            versions=manifest.library_versions(), now=now,
        )
    except manifest.LabelsChanged as exc:
        print(f"train: {exc}")
        return 1
    model_id = path.parent.name
    saved = manifest.save_predictions(store, model, model_id, now=now)

    channels = {name: len({ds.rows[w].channel for w in getattr(sp, name)}) for name in ("train", "val", "test")}
    words = f", emotes as {args.emote_mode}s" if args.rung >= manifest.WORD_RUNG else ""
    print(f"train: rung {args.rung}, {ds.origin} labels, seed {args.seed}{words}")
    print(
        f"windows: {len(ds.rows)} labeled; {len(sp.train)} train, {len(sp.val)} validation, {len(sp.test)} test"
        f" (channels {channels['train']} / {channels['val']} / {channels['test']})"
    )
    print(f"model {model_id} written; {len(before - {model_id})} older rung-{args.rung} {ds.origin} model(s) retired")
    for q in scheme.questions:
        r = report[q.key]
        t = thresholds[q.key]
        if t is None:
            print(f"{q.key}: always not sure (no threshold reaches 80% on validation); {r.n_graded} graded on test")
            continue
        lo, hi = r.agreement_ci
        agreement = "none answered" if r.agreement is None else f"agreement {r.agreement:.2f} [{lo:.2f}, {hi:.2f}]"
        abstain = "" if r.abstain_rate is None else f", abstain {r.abstain_rate:.0%}"
        print(f"{q.key}: threshold {t:.2f}; test {agreement} on {r.answered} answered of {r.n_graded} graded{abstain}")
    consistency = [
        f"{q} {agree}/{compared}" + (f" [{lo:.2f}, {hi:.2f}]" if compared else "")
        for q, (agree, compared, lo, hi) in metrics.self_consistency(store, scheme).items()
    ]
    print("self-consistency (first vs repeat label): " + ", ".join(consistency))
    print(f"predictions: {saved.windows} window(s) with text, {saved.rows} row(s)")
    with store.lock:
        rungs = dict(store.conn.execute("SELECT model_id, rung FROM manifests").fetchall())
    active = manifest.active_per_question(store, models_dir=data / "models")
    print("answering now: " + (", ".join(f"{q} rung {rungs[m]}" for q, m in active.items()) or "no model"))
    return 0


def _print_adopted(recorded: int, unknown: int) -> None:
    if recorded:
        print(f"Label Studio: recorded {recorded} task(s) the store had lost track of")
    if unknown:
        print(f"Label Studio: {unknown} task(s) hold windows the store doesn't know; delete them in Label Studio")


def _page(store: Store, data: Path, now: datetime, args: argparse.Namespace) -> int:
    from .page import app as page_app  # FastAPI, Jinja and uvicorn load only for this command

    _run_expiry(store, data, now, args.ls)  # with Label Studio, so its annotations are pulled first (B11)
    nr = None if args.no_model else narrator.Narrator(_ollama_client(), model=args.model)  # asks nothing yet
    app = page_app.create_app(
        store, scheme_module.load(), data / "models", imported_dir=data / "imported", narrator=nr,
        cpu_captions=args.cpu_captions,
    )  # fmt: skip
    print(f"page: http://{args.host}:{args.port}/ (Ctrl+C stops it)")
    page_app.serve(app, host=args.host, port=args.port)
    return 0


def _ask_eval(store: Store, data: Path, now: datetime, args: argparse.Namespace) -> int:
    from .ask import evaluate  # the ask package loads the page module (FastAPI, Jinja)

    client = _ollama_client()
    try:
        return evaluate.run(narrator.Narrator(client, model=args.model), scheme_module.load(), args.questions, now, print)
    finally:
        client.close()


# --- demo -------------------------------------------------------------------------------------------


def _ollama_log_dir() -> Path | None:
    """Ollama for Windows' log folder, ``%LOCALAPPDATA%\\Ollama`` (tests replace this)."""
    base = os.environ.get("LOCALAPPDATA")
    return Path(base) / "Ollama" if base else None


def _demo(store: Store, data: Path, now: datetime, args: argparse.Namespace) -> int:
    _run_expiry(store, data, now, args.ls)  # like every command: the data folder's own store first
    ls = None
    if args.with_label_studio:
        ls = _demo_label_studio(data)
        if ls is None:
            print(NOT_CONFIGURED)
            print("demo: --with-label-studio needs the real data folder (CHATBRAIN_DATA), not another --data-dir")
            return 1
    try:
        return _demo_run(data, now, args, ls)
    finally:
        if ls is not None:
            ls.close()


def _demo_run(data: Path, now: datetime, args: argparse.Namespace, ls: lsclient.LabelStudio | None) -> int:
    try:
        demo = paths.data_dir({**os.environ, paths.ENV_VAR: str(data / DEMO_DIR)})
    except paths.UnsafeDataDir as exc:
        print(f"demo: {exc}")
        return 1
    if not _demo_reset(demo, now, ls):
        return 1
    seed, later = args.seed, now + DEMO_EXPIRE_AFTER
    scheme = scheme_module.load()
    code = 0
    with Store(demo / DB_NAME) as ds:
        print(f"demo 1/9: import made-up chat (seed {seed}) into {demo}")
        main_raw = fake.make_export(seed, DEMO_MOMENTS, DEMO_ORDINARY, now)
        show_raw = fake.make_showcase(seed, now)
        (demo / "inbox" / "demo-export.json").write_bytes(main_raw)
        (demo / "inbox" / "demo-showcase.json").write_bytes(show_raw)
        if _import(ds, demo, now, _ns(ls, paths=[])):
            return 1

        print(f"demo 2/9: a made-up deletion check that drops {DEMO_DROPS} messages")
        check = fake.make_deletion_check(main_raw, _demo_drops(main_raw, seed), now)
        (demo / "inbox" / "demo-deletion-check.json").write_bytes(check)
        if _deletions(ds, demo, now, _ns(ls)):
            return 1

        print("demo 3/9: assign")
        _assign(ds, demo, now, _ns(ls, seed=f"demo-{seed}", all_blind=False))

        print("demo 4/9: made-up labels from the generator's own answers")
        truth = {**fake.truth(main_raw), **fake.truth(show_raw)}
        print(f"labeled {_demo_labels(ds, scheme, truth, now)} window(s) (origin synthetic, project = exposure)")

        print(f"demo 5/9: suggest --limit {DEMO_SUGGEST}")
        if args.with_ollama:
            if _suggest(ds, demo, now, _ns(ls, limit=DEMO_SUGGEST, model=teacher.DEFAULT_MODEL)):
                code = 1  # the rest of the loop doesn't need suggestions
        else:
            print("suggest: skipped (no --with-ollama)")

        for rung in (0, 1):
            print(f"demo 6/9: train --rung {rung} --origin synthetic --seed {seed}")
            if _train(ds, demo, now, _ns(ls, rung=rung, origin="synthetic", seed=seed, emote_mode="name")):
                return 1

        print(f"demo 7/9: push up to {DEMO_PUSH} windows to each demo Label Studio project")
        if ls is not None:
            print(f"repeats: {_demo_repeats(ds, seed, now)} window(s) made due now for the repeat project")
            for kind in lsclient.KINDS:
                if _ls_push(ds, demo, now, _ns(ls, kind=kind, limit=DEMO_PUSH)):
                    code = 1
                    break
            else:
                print("Label Studio: the repeat tasks went in directly, with no predictions and no earlier labels")
        else:
            print("Label Studio: skipped (no --with-label-studio)")
        with ds.lock:
            pushed = Counter(dict(ds.conn.execute("SELECT project_kind, COUNT(*) FROM ls_tasks GROUP BY project_kind")))
            pushed_ids = {r[0] for r in ds.conn.execute("SELECT window_id FROM ls_tasks")}

        print(f"demo 8/9: expire --now {iso(later)}")
        _expire(ds, demo, later, _ns(ls))

        print("demo 9/9: clean up, check and sum up")
        ls_clean = ls is None or _demo_delete_tasks(ds, demo, later, ls)
        if not ls_clean:
            code = 1
        with ds.lock:
            conn = ds.conn
            states = dict(conn.execute("SELECT text_state, COUNT(*) FROM windows GROUP BY text_state").fetchall())
            kinds = dict(conn.execute("SELECT anchor_kind, COUNT(*) FROM windows GROUP BY anchor_kind").fetchall())
            expired_ids = [r[0] for r in conn.execute("SELECT window_id FROM windows WHERE text_state = 'expired'")]
            sent_ids = [r[0] for r in conn.execute("SELECT DISTINCT window_id FROM suggestions")]
            labels = conn.execute("SELECT COUNT(*) FROM labels WHERE origin = 'synthetic'").fetchone()[0]
            models = conn.execute(
                "SELECT rung FROM manifests WHERE retired_at IS NULL AND file_deleted_at IS NULL ORDER BY rung"
            ).fetchall()
            newest = conn.execute(
                "SELECT stream_id FROM windows GROUP BY stream_id ORDER BY MAX(end_at) DESC, stream_id LIMIT 1"
            ).fetchone()
            spreads = Counter(
                json.loads(r[0]).get("spread") for r in conn.execute(
                    "SELECT f.json FROM windows w JOIN features f ON f.window_id = w.window_id"
                    " WHERE w.stream_id = ? AND w.anchor_kind = 'moment'", (newest[0] if newest else "",),
                )
            )  # fmt: skip

    # the stores' files are closed or flushed: search them
    if not _demo_sentinels(data, ls, expired_ids, pushed_ids, sent_ids if args.with_ollama else None):
        code = 1

    print("demo summary:")
    print(
        f"  windows: {sum(states.values())} ({kinds.get('moment', 0)} Moments, {kinds.get('ordinary', 0)} ordinary);"
        f" {states.get('expired', 0)} expired at now + 5 days, {states.get('present', 0)} still have text"
    )
    print(f"  labels: {labels} made-up (synthetic); suggestions: {len(sent_ids)}")
    print(
        f"  models: {len(models)} active (rungs {', '.join(str(r[0]) for r in models) or 'none'}),"
        " trained on made-up labels; the page says so"
    )
    if ls is not None:
        print(
            "  Label Studio: "
            + ", ".join(f"{pushed.get(k, 0)} {k}" for k in lsclient.KINDS)
            + " task(s) pushed"
            + (", all deleted again and compacted" if ls_clean else ", NOT all deleted yet (run the demo again)")
            + "; the repeat tasks had no predictions and no earlier labels"
        )
    shown = ", ".join(f"{n} {v}" for v, n in sorted(spreads.items(), key=lambda kv: (-kv[1], str(kv[0]))))
    print(f"  the newest stream (the showcase): {sum(spreads.values())} Moments: {shown}")
    print(f'  next: uv run chatbrain --data-dir "{demo}" page, then open http://{PAGE_HOST}:{PAGE_PORT}/')
    return code


def _ns(ls: lsclient.LabelStudio | None, **values: object) -> argparse.Namespace:
    """The arguments one command step of the demo gets."""
    return argparse.Namespace(ls=ls, **values)


def _demo_reset(demo: Path, now: datetime, ls: lsclient.LabelStudio | None) -> bool:
    """Delete the last demo's store and files, once Label Studio holds none of its tasks.

    A task the last demo's store still lists, or a delete not compacted yet, would lose its
    only record with that store, so it is finished first (or the demo refuses without
    ``--with-label-studio``).
    """
    db = demo / DB_NAME
    if db.exists():
        with Store(db) as old:
            with old.lock:
                left = old.conn.execute("SELECT COUNT(*) FROM ls_tasks").fetchone()[0]
                compaction = old.conn.execute("SELECT 1 FROM ls_compaction").fetchone() is not None
            if (left or compaction) and ls is None:
                what = f"{left} Label Studio task(s) from the last demo are still listed" if left else (
                    "the last demo's Label Studio deletes are not compacted yet"
                )
                print(f"demo: {what}; run the demo with --with-label-studio so it finishes them first")
                return False
            if (left or compaction) and not _demo_delete_tasks(old, demo, now, ls):
                return False
    for name in (DB_NAME, f"{DB_NAME}-journal", f"{DB_NAME}-wal", f"{DB_NAME}-shm"):
        if not ingest.remove_file(demo / name):
            print(f"demo: the last demo's {name} can't be deleted (is `chatbrain page` running on it?); stop it first")
            return False
    for sub in paths.SUBDIRS:
        expiry.empty_folder(demo / sub)
    return True


def _demo_drops(raw: bytes, seed: int) -> set[str]:
    """``DEMO_DROPS`` message IDs, each from another window, never a marker line."""
    windows = [w for w in parse_export(raw).windows if len(w.messages) >= 3]
    rng = random.Random(f"demo-drops:{seed}")
    picked = rng.sample(windows, DEMO_DROPS)
    return {rng.choice([m for m in w.messages if m.text != fake.marker(w.window_id)]).message_id for w in picked}


def _demo_labels(store: Store, scheme: scheme_module.Scheme, truth: dict[str, dict], now: datetime) -> int:
    """One synthetic label per assigned window: the made-up answers, project = its exposure."""
    for answers in truth.values():
        scheme_module.validate_answers(scheme, answers, {})
    with store.transaction() as conn:
        exposure = dict(conn.execute("SELECT window_id, exposure FROM assignments").fetchall())
        rows = [
            (wid, exposure[wid], scheme.version, scheme.guide_version, json.dumps(answers), iso(now))
            for wid, answers in sorted(truth.items())
            if wid in exposure
        ]
        conn.executemany(
            "INSERT INTO labels (window_id, project, origin, provenance, labels_version, guide_version,"
            " answers_json, not_sure_json, created_at) VALUES (?, ?, 'synthetic', 'synthetic', ?, ?, ?, '{}', ?)",
            rows,
        )
    return len(rows)


def _demo_repeats(store: Store, seed: int, now: datetime) -> int:
    """Make ``DEMO_PUSH`` windows with chat due for a repeat now (C18: ``schedule_repeats`` is unchanged)."""
    now_s = iso(now)
    with store.transaction() as conn:
        ids = [
            r[0] for r in conn.execute(
                "SELECT w.window_id FROM windows w WHERE w.text_state = 'present' AND w.text_expires_at > ?"
                " AND NOT EXISTS (SELECT 1 FROM repeats r WHERE r.window_id = w.window_id)"
                " AND (SELECT COUNT(*) FROM messages m WHERE m.window_id = w.window_id) >= 2 ORDER BY w.window_id",
                (now_s,),
            )
        ]  # fmt: skip
        picked = sorted(random.Random(f"demo-repeats:{seed}").sample(ids, min(DEMO_PUSH, len(ids))))
        conn.executemany("INSERT INTO repeats (window_id, due_at) VALUES (?, ?)", [(w, now_s) for w in picked])
    return len(picked)


def _demo_delete_tasks(store: Store, demo: Path, now: datetime, ls: lsclient.LabelStudio) -> bool:
    """Delete every Label Studio task ``store`` tracks, the way expiry does (B11): pull, delete, compact."""
    with store.transaction() as conn:
        queued = conn.execute("INSERT OR IGNORE INTO ls_deletions (task_id) SELECT task_id FROM ls_tasks").rowcount
    r = _run_expiry(store, demo, now, ls)
    if r.ls_tasks_pending or r.ls_compaction_pending or r.ls_error:
        print("demo: Label Studio still holds demo tasks; run the demo again with --with-label-studio")
        return False
    print(f"Label Studio: the demo's {queued} remaining task(s) deleted, then compacted")
    return True


def _demo_sentinels(
    data: Path, ls: lsclient.LabelStudio | None, expired: list[str], pushed: set[str], sent: list[str] | None
) -> bool:
    """Search for marker lines that should be gone (B15/B16); print counts only. True when none is found.

    Expired windows' markers are searched in every file under the data folder, deleted
    Label Studio tasks' under Label Studio's folder, and the markers of windows sent to
    Ollama in its ``*.log`` files. A folder that isn't there is named, never guessed.
    """
    clean = True
    searches = [("expired windows' marker lines", data, expired, "file(s) under the data folder", "")]
    if ls is not None:
        searches.append(
            ("deleted Label Studio tasks' marker lines", Path(ls.data_dir) / "labelstudio", sorted(pushed),
             "file(s) under Label Studio's folder", "")
        )  # fmt: skip
    if sent is not None:
        searches.append(("marker lines of windows sent to Ollama", _ollama_log_dir(), sent, "log file(s)", ".log"))
    for what, root, ids, where, suffix in searches:
        if root is None or not root.is_dir():
            print(f"sentinel search: {what}: the folder was not found, so nothing was searched; search by hand (README)")
            continue
        scanned, hits, unreadable = _sentinel_search(root, {fake.marker(w) for w in ids}, suffix)
        line = f"sentinel search: {what} found in {hits} of {scanned} {where}"
        if unreadable:
            line += f"; {unreadable} could not be read"
        print(line)
        clean = clean and hits == 0 and unreadable == 0
    if not clean:
        print("demo: a marker line that should be gone was found, or a file could not be searched (no names printed)")
    return clean


_MARKER_SHAPE = re.compile(rb"zq[0-9a-f]{12}")  # fake.marker's lines
_MARKER_TAIL = 13  # bytes kept between reads, so a marker split across two reads is still seen


def _sentinel_search(root: Path, markers: set[str], suffix: str = "") -> tuple[int, int, int]:
    """(files read, files holding one of ``markers``, files that couldn't be read) under ``root``.

    Links are never followed. Files are read in 1 MiB pieces and searched for anything shaped
    like a marker, so a large file costs one pass whatever the number of markers.
    """
    wanted = {m.encode("ascii") for m in markers}
    scanned = hits = unreadable = 0
    stack = [root]
    while stack:
        folder = stack.pop()
        try:
            with os.scandir(folder) as it:
                entries = list(it)
        except OSError:
            unreadable += 1
            continue
        for entry in entries:
            if entry.is_symlink() or entry.is_junction():
                continue
            if entry.is_dir(follow_symlinks=False):
                stack.append(Path(entry.path))
            elif entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(suffix):
                try:
                    found = _holds_marker(Path(entry.path), wanted)
                except OSError:
                    unreadable += 1
                    continue
                scanned += 1
                hits += found
    return scanned, hits, unreadable


def _holds_marker(path: Path, wanted: set[bytes]) -> bool:
    if not wanted:
        return False
    tail = b""
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            buf = tail + chunk
            if any(m in wanted for m in _MARKER_SHAPE.findall(buf)):
                return True
            tail = buf[-_MARKER_TAIL:]
    return False


if __name__ == "__main__":
    sys.exit(main())
