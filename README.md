# BrainChat

BrainChat is a small local model that says what Twitch chat is doing around a
StreamPulse Moment: what chat is doing, who it is aimed at, how chat feels and the
tone, or "not sure" with a reason.

It runs locally only, on 127.0.0.1. This repo ships made-up data only. The code is under the Apache License 2.0 (see License below).

- Made-up data only: the tests and the demo build their own fake chat (channels
  `fakechanNN`, speakers `s<N>`, emote IDs `fake-<name>`). No real chat, labels,
  exports or models are included.
- Local only: nothing talks to a hosted AI service, and every service binds to
  `127.0.0.1`.
- Quick check: `uv sync`, then `uv run pytest -q` (CPU only; Ollama and Label Studio are
  faked in the tests).
- Tested on Windows.

Codes such as `C12`, `B7`, `R4` or `T1`, "Task n", "spec §n", "Slice 1", "P0", "Q10" or
"G-14", "DESIGN-BRIEF §n", design boards (such as "the dir-c board"), and mentions of the
engine, questions or data design notes, in code comments and in `docs/decisions.md` refer
to private design notes that are not part of this repo.

## Run

1. Install Windows Python 3.12 and `uv`, then from this folder:

   ```
   uv sync
   uv run pytest -q
   ```

2. Pick a data folder outside this repo, outside OneDrive and outside any git
   work tree, and set it as an absolute path:

   ```
   setx CHATBRAIN_DATA C:\brainchat-data
   ```

   `chatbrain.paths.data_dir()` refuses an unsafe folder and creates `inbox/`,
   `imported/`, `models/` and `labelstudio/` inside a safe one.

3. Put export files and deletion-check files in `<data>/inbox/`, then:

   ```
   uv run chatbrain import        # imports every export in the inbox, then deletes it
   uv run chatbrain window-list   # writes <data>/window-list.json for the server's check script
   uv run chatbrain deletions     # applies the check files in the inbox; shows the freshness gate
   uv run chatbrain expire        # deletes text whose time is up
   uv run chatbrain assign --seed S [--all-blind]   # blind 1 in 4 or suggested, once per window
   uv run chatbrain repeats --seed S                # 1 in 10 labeled windows, due 3-7 days later
   uv run chatbrain pick-rare WINDOW_ID...          # marks windows picked on purpose for rare values
   ```

   Every command runs expiry first and prints counts and status only. `--data-dir PATH`
   overrides `CHATBRAIN_DATA` for one run; `expire --now ISO` sets the clock for checks
   on made-up data.

4. Label Studio runs in Docker on `127.0.0.1:14610` only, with its files in
   `<data>/labelstudio`. Start Docker Desktop, then from this folder (with
   `CHATBRAIN_DATA` set):

   ```
   docker compose up -d
   ```

   Open http://127.0.0.1:14610, create the local account, enable a token type under
   Organization → Settings → Access Token Settings, and put the token in
   `<data>/.env`:

   ```
   LS_TOKEN=...
   LS_AUTH=pat        # pat (personal access token, the default) or legacy
   # LS_URL=http://127.0.0.1:14610   (loopback only; this is the default)
   ```

   Then:

   ```
   uv run chatbrain ls-setup                                   # the three projects: suggested, blind, repeat
   uv run chatbrain ls-push --kind blind --limit 20            # suggested | blind | repeat
   uv run chatbrain ls-pull                                    # your annotations become labels
   ```

   Label with `docs/LABELING_GUIDE.md` open; the same one-line definitions show as
   hints on every choice.

5. Teacher suggestions come from Ollama for Windows (0.31.2 or newer) on
   `127.0.0.1:11434`. Install it from ollama.com, run `ollama pull qwen3:8b`, and run it
   with `OLLAMA_NUM_PARALLEL=1`, never with `OLLAMA_DEBUG` (debug logging can write
   whole prompts, which hold chat, to Ollama's logs). Leave `OLLAMA_HOST` unset or on
   `127.0.0.1`. Then:

   ```
   uv run chatbrain assign --seed S                        # suggestions need suggested windows
   uv run chatbrain suggest --limit 20 [--model qwen3:8b]  # one row per window: answers or an error code
   uv run chatbrain ls-push --kind suggested --limit 20    # suggested windows go out with their suggestion
   ```

   `suggest` prints how many rows it recorded, the time per window and the totals by
   error code. Time per window on this PC: not measured yet (first live run).

6. Train a rung from scratch on the labels still held:

   ```
   uv run chatbrain train --rung 0|1 [--origin aron|synthetic] --seed S [--emote-mode name|id]
   ```

   Labels carry an origin: `aron` is the owner's own labels (Label Studio and page fixes),
   `synthetic` the demo's made-up ones. `--origin` is needed only when both exist. The command
   applies waiting deletion checks, needs the freshness gate, writes
   `models/<model_id>/model.joblib` and `manifest.json`, retires the older model of the
   same rung and origin, and stores predictions for every window that still has text.
   It prints each question's threshold and test agreement beside how many windows were
   graded, the owner's repeat consistency, and which model now answers each question.

7. Look at the Moments on the local page:

   ```
   uv run chatbrain page          # serves http://127.0.0.1:14612 until Ctrl+C
   ```

   The page opens on the newest stream: a status line, the activity line with a numbered
   chip per Moment in the lane under it, and the three busiest Moments as cards (the first
   three when the counts are off); "All N Moments" opens the whole list. A Moment opens in the right panel, or
   on its own at `/m/<window_id>`, with the counts, the four answers and where each came
   from, the cause note, messages per 5 s and the chat lines while the text lives
   (`?doing=<value>` on a Moment limits its previous and next arrows to Moments with that
   Doing value). Fix
   saves your label (`corrected-on-page`) for the next training run; picking the value
   already shown saves it too. Both writes refuse while the deletion check is stale.
   The page runs expiry and saves missing rung-0 predictions at start and then hourly.
   `config/review_emotes.json` lists the emote IDs whose windows show "needs review".

   The activity line shows people chatting per 10 s, counts only, from export `/2`. A
   hatched band marks time where the export recorded no chat. The numbered chips under
   the line match the Moment row numbers. Opening a Moment drops a guide at its time, and
   the Moment shows "N people chatting at the peak · X× the usual pace". The line draws,
   then the chips rise; with reduced motion everything shows at once. An export without
   counts (such as `/1`) shows "This stream's export has no activity counts." Activity
   counts expire 30 days after the stream started (UTC): the expiry run deletes the counts
   and keeps a marker, so the line stays gone and shows "Activity counts expired 30 days
   after this stream started." Without counts, the rows show posters joining instead.

   Keys: J / K open the next and previous Moment; the arrow keys move between rows (one
   tab stop); Enter opens a row; F fixes the first answer; Esc closes Fix, then the
   Moment; / focuses Ask.

   Asking: the box under the list turns a question into chips, counts in SQL and shows
   the Moments at once. When Ollama runs on this PC with `qwen3:8b`, the local model then
   reads the question again (the chips may swap, marked "refined by the local model") and
   writes a short caption for one Moment, an emote's meaning or a vague question. Stop
   keeps the keyword answer. When the model isn't usable, an amber banner says how to fix it.

   ```
   uv run chatbrain page --no-model       # keyword answers only, no local model steps
   uv run chatbrain page --model qwen3:14b --cpu-captions
   uv run chatbrain ask-eval              # how well the local model reads the 60 test questions
   ```

8. Run the whole loop on made-up chat:

   ```
   uv run chatbrain demo --seed 7 [--with-ollama] [--with-label-studio]
   uv run chatbrain --data-dir "$env:CHATBRAIN_DATA\demo" page
   ```

   The demo works in `<data>/demo`, a data folder of its own that it empties first, on
   the real clock. It imports 480 made-up Moments and 120 ordinary windows plus one short
   showcase stream (the newest, so the page opens on it: a chant, repeated emotes, two
   critical turns), applies a made-up deletion check that drops 3 messages, assigns,
   labels every window from the generator's own answers (origin `synthetic`), asks the
   teacher about 3 windows (`--with-ollama`), trains rungs 0 and 1, pushes up to 10 tasks
   to each `BrainChat demo · ...` project (`--with-label-studio`, real data folder only),
   expires at now + 5 days, deletes its Label Studio tasks again and prints a summary.
   The page then says its answers come from made-up labels.

   It ends with a sentinel search: the marker lines of expired windows in every file
   under the data folder, of deleted tasks under `labelstudio/`, and of windows sent to
   Ollama in its `*.log` files (`%LOCALAPPDATA%\Ollama`). Each line prints counts only and
   should say `found in 0`. To search by hand (or when it says a folder was not found),
   write the expired windows' markers for that seed to a file, search, then delete it:

   ```
   uv run python -c "from datetime import datetime, timedelta, timezone as tz; from chatbrain import fake; from chatbrain.export_format import parse_export as p; n = datetime.now(tz.utc); print(*[fake.marker(w.window_id) for w in p(fake.make_export(7, 480, 120, n)).windows if w.text_expires_at <= n + timedelta(days=5) and len(w.messages) > 1], sep=chr(10))" > $env:TEMP\cb-sentinels.txt
   $m = Get-Content $env:TEMP\cb-sentinels.txt
   (Get-ChildItem -Recurse -File $env:CHATBRAIN_DATA | Select-String -SimpleMatch -Pattern $m -List).Count            # expect 0
   (Get-ChildItem -Recurse -File "$env:LOCALAPPDATA\Ollama" -Filter *.log | Select-String -SimpleMatch -Pattern $m -List).Count   # expect 0
   Remove-Item $env:TEMP\cb-sentinels.txt
   ```

   Before any real chat: BitLocker is on for the drive that holds the data folder, Ollama
   runs with `OLLAMA_NUM_PARALLEL=1` and never `OLLAMA_DEBUG`, and the daily expiry job
   (see `docs/decisions.md`) is set up.

Coding agents never open the data folder; see `AGENTS.md`.

## Layout

- `config/labels_v1.json`: label scheme v1 (spec section 4) and guide version.
- `src/chatbrain/paths.py`: the data-folder guard.
- `src/chatbrain/scheme.py`: loads and checks the label scheme.
- `src/chatbrain/export_format.py`: the export, window-list and deletion-check formats; see `docs/export-format.md`.
- `src/chatbrain/store.py`: the SQLite store (windows, text while it lives, and everything that outlives it).
- `src/chatbrain/fake.py`: made-up exports, ground truth and deletion checks for tests and the demo.
- `src/chatbrain/spread.py`: counted spread and participation (`Counts`), kept after the text expires.
- `src/chatbrain/ingest.py`: import, deletion checks, the window list and the freshness gate.
- `src/chatbrain/expiry.py`: text expiry in the store, the files and Label Studio (`TaskDeleter`).
- `src/chatbrain/assign.py`: blind or suggested assignment, delayed repeats and rare-value picks.
- `src/chatbrain/lstudio/config.py`: the Label Studio labeling view built from the scheme, and the guide's definitions.
- `src/chatbrain/lstudio/tasks.py`: one task's JSON (speaker and text per line, the counts line, a prelabel only in the suggested project).
- `src/chatbrain/lstudio/client.py`: the loopback-only Label Studio client (projects, imports, exports, deletes, compaction) and `.env` loading.
- `src/chatbrain/lstudio/sync.py`: push, pull with provenance, and adoption of tasks the store lost track of.
- `src/chatbrain/ollama.py`: the one gate in front of Ollama: a loopback-only client with no proxies or redirects, and the 0.31.2 version check.
- `src/chatbrain/teacher.py`: teacher suggestions: the JSON schema, the prompt, reading the reply, and the batch over suggested windows.
- `src/chatbrain/features.py`: model inputs: counted numbers and emote-ID counts (rung 0), one document per window (rung 1).
- `src/chatbrain/metrics.py`: Wilson intervals, per-question reports, label reading and the owner's own repeat consistency.
- `src/chatbrain/train.py`: rungs 0-1: the dataset, grouped splits, one head per question, thresholds, reports and evidence.
- `src/chatbrain/manifest.py`: model files and manifests, retiring, which model answers each question, and stored predictions.
- `src/chatbrain/page/shown.py`: what the page shows per question: the owner's latest label, else the answering model's prediction at or over its threshold, else "not sure" with a reason; the rung-0 backfill.
- `src/chatbrain/page/app.py`: the local Moment page (FastAPI, Jinja), its two label writes, and `serve` (loopback only).
- `src/chatbrain/page/templates/`, `src/chatbrain/page/static/`: the page's HTML, CSS, script and self-hosted fonts.
- `config/review_emotes.json`: emote IDs whose windows show every answer as "not sure, needs review" (contested readings).
- `src/chatbrain/cli.py`: the `chatbrain` command (`import`, `deletions`, `window-list`, `expire`, `assign`, `repeats`, `pick-rare`, `ls-setup`, `ls-push`, `ls-pull`, `suggest`, `train`, `page`, `ask-eval`, `demo`).
- `compose.yaml`: Label Studio 1.23.2 on `127.0.0.1:14610`, telemetry off.
- `docs/LABELING_GUIDE.md`: the labeling guide (`cb_guide_2026_11_06_v1`); its definitions equal `labels_v1.json` `help`.
- `docs/decisions.md`: the builder's calls, one line per choice the spec leaves open.

## Builder calls

The decision log, one line per choice the spec leaves to the builder or the plan rules on, is in
[`docs/decisions.md`](docs/decisions.md).

## License

- Code: [Apache License 2.0](LICENSE).
- The Geist and Geist Mono fonts in `src/chatbrain/page/static/fonts/` keep their own SIL Open Font License 1.1 (`OFL-geist.txt`, `OFL-geist-mono.txt`).
- No chat datasets, labels or model weights ship in this repo. If any are released later, they get their own terms.
