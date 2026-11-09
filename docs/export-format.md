# Export file contract

This is the contract between the read-only export script on the StreamPulse server
(not part of this repo) and this PC. The PC side lives in
`src/chatbrain/export_format.py`. If the two disagree, the PC refuses the whole
file and imports nothing.

Three formats cross the line:

| Format | Direction | What it carries |
|---|---|---|
| `chatbrain-export/1` | server → PC | windows of chat around Moments and ordinary minutes |
| `chatbrain-window-list/1` | PC → server | the windows the PC holds: IDs and bounds, no text |
| `chatbrain-deletion-check/1` | server → PC | which of those windows' messages still exist |

## Rules for every file

- One UTF-8 JSON object. A UTF-8 byte-order mark is tolerated.
- Every object has exactly the keys listed below. A missing key, an unknown key or a
  key repeated inside one object refuses the file. `NaN` and `Infinity` refuse it too.
- **Times** are RFC 3339 text with an explicit offset:
  `YYYY-MM-DDTHH:MM:SS[.ffffff](Z|+HH:MM|-HH:MM)`, with at most 6 fractional digits.
  A time without an offset (naive) refuses the file. The PC converts every time to
  UTC. It writes its own times as `YYYY-MM-DDTHH:MM:SS.ffffffZ`.
- **IDs** (`window_id`, `stream_id`, `message_id`, emote `id`) are 1–128 characters
  of `A-Z a-z 0-9 . _ : -`, starting with a letter or digit.
- **Never exported, anywhere in any file:** `login`, `display_name`,
  `commenter_login`, `sender_hash`, `user_id`, any other per-user ID that is stable
  across windows, and message hashes. The PC refuses a key with any of those names at
  any depth, in any spelling: case, `_` and `-` are ignored, so `displayName` and
  `Sender-Hash` are refused too. It also refuses `user_login`, `username`,
  `user_name`, `message_hash` and `text_hash`. The check reads keys only, never
  message text, so a chat line that says "login" is fine.
- **Rejection messages** name the place and the rule (`export.windows[3].messages[2].speaker:
  must be a per-window speaker number s1, s2, ...`), never the value, because a bad
  value may be chat text or a chatter's name. A key that isn't in this contract shows
  as `?` in the place, because a key may be a chatter's name too.

## `chatbrain-export/1` (server → PC)

```json
{
  "format": "chatbrain-export/1",
  "exported_at": "2026-11-06T12:00:00Z",
  "source": "vod",
  "windows": [
    {
      "window_id": "w-0001",
      "channel": "fakechan01",
      "stream_id": "fs-01",
      "anchor_kind": "moment",
      "anchor_at": "2026-11-03T12:00:30Z",
      "start_at": "2026-11-03T12:00:00Z",
      "end_at": "2026-11-03T12:02:00Z",
      "language": "en",
      "text_expires_at": "2026-11-16T08:00:00Z",
      "messages": [
        {"message_id": "m1", "rel_ms": 31000, "speaker": "s1", "text": "KEKW",
         "emotes": [{"id": "fake-kekw", "name": "KEKW"}]},
        {"message_id": "m2", "rel_ms": 40000, "speaker": "s2", "text": "@s1 same", "emotes": []}
      ]
    }
  ]
}
```

### Top level

| Field | Type | Rule |
|---|---|---|
| `format` | string | exactly `chatbrain-export/1` |
| `exported_at` | time | when the script ran |
| `source` | string | `vod` (VOD chat) or `live` (live text) |
| `windows` | array | zero or more windows; no `window_id` twice |

### Window

| Field | Type | Rule |
|---|---|---|
| `window_id` | ID | chosen by the server and **stable**: the same stream, anchor and kind always get the same ID, so a re-export is skipped as a duplicate |
| `channel` | string | the streamer's channel name, 1–25 of `a-z 0-9 _` |
| `stream_id` | ID | the stream the window comes from |
| `anchor_kind` | string | `moment` or `ordinary` |
| `anchor_at` | time | the Moment time, or the ordinary minute |
| `start_at` | time | exactly `anchor_at − 30 s` |
| `end_at` | time | exactly `start_at + 120 s` (so `anchor_at + 90 s`) |
| `language` | string | the stream's language code, like `en`. The PC imports only `en` and counts the rest as skipped |
| `text_expires_at` | time | when the PC must delete this window's text: the stream's end + 14 days (`vod`), or the earliest `synced_at` in the window + 14 days (`live`). Must be after `end_at` and no later than `exported_at + 14 days` |
| `messages` | array | the window's messages; may be empty. No `message_id` twice in one window |

Ordinary windows are a random minute from the same streams as the Moments, at
least 5 minutes from every Moment anchor in their stream (spec §5). Windows may
overlap; a message in two windows appears in both with the same `message_id`.

### Message

| Field | Type | Rule |
|---|---|---|
| `message_id` | ID | the platform's message ID |
| `rel_ms` | integer | milliseconds after `start_at`, `0 ≤ rel_ms < 120000` (the window is `[start_at, end_at)`). A float or `true` is refused |
| `speaker` | string | the per-window speaker number `s1`, `s2`, … (no leading zeros). Made from `sender_hash` with a per-window salt that is then discarded, so posters can be counted but not linked across windows. Numbering by order of first post is recommended |
| `text` | string | the message text with every mention rewritten (below) |
| `emotes` | array | one `{"id", "name"}` entry per emote occurrence, in text order; `name` is the emote's text, 1–100 characters with no spaces |

**Mentions.** Every `@` followed by one or more of `A-Z a-z 0-9 _` is a mention,
wherever it appears (so `someone@example.com` counts). The server replaces the
whole token with `@s<N>`, the mentioned person's speaker number, if they have at
least one exported message in this window, and with `@user` otherwise. The PC
refuses any other mention token, including `@S1`, `@users` and `@user_1`, and
refuses `@s<N>` when no message in the window has speaker `s<N>`.

**Speaker numbers and deleted messages.** The server drops deleted messages first
and numbers speakers afterwards, from the messages it actually exports. So every
`s<N>` in a window, as a `speaker` or in a mention, belongs to someone with at least
one message in that window's `messages`, and a person whose only message was
deleted is mentioned as `@user`.

**Age.** Only messages at least 1 hour old are exported, so moderation has landed:
`start_at + rel_ms ≤ exported_at − 1 hour`. A newer message refuses the file.
Messages already deleted on the server are never exported.

The PC sorts each window's messages by `rel_ms`, then `message_id`.

## Stream activity (format /2)

`chatbrain-export/2` is `/1` plus one required top-level key, `streams`: a list, empty
if the server has nothing to count. Everything else is exactly as in `/1`. A `/1` file
with a `streams` key is refused, and a `/2` file without one is refused.

```json
"streams": [
  {
    "stream_id": "fs-01",
    "channel": "fakechan01",
    "started_at": "2026-11-03T12:00:00Z",
    "ended_at": "2026-11-03T12:00:45Z",
    "title": "fake stream one",
    "bin_s": 10,
    "chatters": [3, 5, null, null, 0]
  }
]
```

| Field | Type | Rule |
|---|---|---|
| `stream_id` | string | An ID; appears once in `streams`. |
| `channel` | string | A lowercase channel name. |
| `started_at`, `ended_at` | RFC 3339 with offset | `started_at < ended_at <= exported_at`. |
| `title` | string or `null` | After stripping, 1 to 140 characters. Refused when it holds any control character (Unicode category Cc, which covers NUL, ESC, tab, line feed, vertical tab, form feed, carriage return and U+0085) or U+2028 or U+2029. |
| `bin_s` | integer | Exactly `10`. |
| `chatters` | array | One entry per 10 s bin: `ceil((ended_at − started_at) / 10 s)` of them, between 1 and 17,280 (48 h). Each is `null` or a whole number from 0 to 100,000. |

What a count means: the number of distinct people with at least one message in that
10 s bin. The server counts them; no IDs, names or chat text leave the server, so a
count is only a number. `null` means that bin was not recorded (the page shows a
hatched gap, never a zero).

The PC keeps a stream's activity counts only when a stored window, or a window in the
same file, names that stream, so the server must send each stream's counts in a file
that carries one of its windows, or send them again after one is stored, within 30
days of the stream's start.

The owner's export script moves to `/2` once it can count. Until then `/1` keeps
working, and the PC reads it exactly as before (with no streams).

## `chatbrain-window-list/1` (PC → server)

The request for a deletion check, written by the PC from the windows it holds.
It carries no text and no message IDs.

```json
{
  "format": "chatbrain-window-list/1",
  "windows": [
    {"window_id": "w-0001", "stream_id": "fs-01",
     "start_at": "2026-11-03T12:00:00.000000Z", "end_at": "2026-11-03T12:02:00.000000Z"}
  ]
}
```

| Field | Type | Rule |
|---|---|---|
| `format` | string | exactly `chatbrain-window-list/1` |
| `windows[].window_id` | ID | as exported; no `window_id` twice |
| `windows[].stream_id` | ID | as exported |
| `windows[].start_at`, `end_at` | time | as exported; `end_at − start_at` is exactly 120 s |

## `chatbrain-deletion-check/1` (server → PC)

The server's answer to a window list: for each window the PC asked about, the
message IDs that still exist on the server inside that window's bounds. The check
lists **every** window ID from the request, each exactly once, with `present: []`
when none of its messages remain.

```json
{
  "format": "chatbrain-deletion-check/1",
  "checked_at": "2026-11-07T09:00:00Z",
  "windows": [
    {"window_id": "w-0001", "present": ["m1"]},
    {"window_id": "w-0002", "present": []}
  ]
}
```

| Field | Type | Rule |
|---|---|---|
| `format` | string | exactly `chatbrain-deletion-check/1` |
| `checked_at` | time | when the script ran |
| `windows` | array | **every** window from the request, in any order; no `window_id` twice (the PC refuses the file) |
| `windows[].window_id` | ID | a window from the request |
| `windows[].present` | array of IDs | message IDs that still exist; no ID twice; `[]` when none remain (for example, after the server's own retention removed them) |

How the PC applies it (`ingest.apply_deletion_check`):

- The PC applies a check **only to the windows it lists**. A listed window keeps
  only the messages whose IDs are in its `present` list. IDs the PC never held are
  ignored.
- A window the check doesn't list is left as it is. That covers windows imported
  after the window list was written, which the server was never asked about. It is
  also why the server must list every requested window, with `present: []` when
  nothing is left: leaving a window out does not delete its messages, it only skips
  it until a later check lists it.
- A listed window's counts are recomputed; its labels stay. Its Label Studio tasks
  are deleted (or queued for deletion while Label Studio is unreachable).
- The PC records, per window, the newest `checked_at` of a check that listed it.
- Labeling, suggesting and training refuse to run (`ingest.require_fresh_deletion_check`)
  when there is no deletion check, when the newest is more than 7 days old or more
  than 5 minutes ahead of the PC's clock, or when any window that still has text was
  not listed by a check from the last 7 days. The refusal gives counts and dates, never
  window IDs. A window imported after the last window list therefore blocks labeling
  until a new window list is answered.

On the PC, check files go in `<data>/inbox/`. `chatbrain deletions` (and
`ingest.apply_pending_checks`) applies every check file there, oldest `checked_at`
first, and deletes each one once it is applied. A check that breaks this contract,
one dated more than 5 minutes ahead of the PC's clock, and a `.json` file that isn't a
BrainChat file are refused and left in place. Import an export before applying the
check that answers its window list; the order of a round is `chatbrain import`,
`chatbrain window-list`, run the check script on the server, then `chatbrain deletions`.

## Rejection rules, in one list

The PC refuses the whole file when:

1. it is not UTF-8 JSON, is not one object, repeats a key inside an object, holds `NaN`/`Infinity`,
   or holds an integer literal longer than Python converts (over 4300 digits);
2. any identity key (listed above) appears anywhere, in any spelling;
3. `format` is not the expected string (`/1` or `/2`), or any key is missing or unknown;
4. a time is not RFC 3339 with an explicit offset (a naive time);
5. `source`, `anchor_kind`, `channel`, `language` or an ID breaks its rule;
6. `end_at − start_at` is not exactly 120 s, or `anchor_at − start_at` is not exactly 30 s;
7. `text_expires_at` is not after `end_at`, or is more than 14 days after `exported_at`;
8. a `speaker` is not `s<N>`;
9. a mention is not `@s<N>` for a speaker with a message in the window, or `@user`;
10. `rel_ms` is not an integer in `[0, 120000)`;
11. a message is under 1 hour old at `exported_at`;
12. a `window_id` repeats in the file (in an export, or in a deletion check that
    lists the same window twice), or a `message_id` repeats in a window (in a
    deletion check: a `message_id` repeats in one `present` list).
13. in a `/2` stream: `bin_s` is not 10, `chatters` is not one entry per 10 s (1 to 17,280),
    an entry is not `null` or an integer from 0 to 100,000, `ended_at` is not after
    `started_at` or is after `exported_at`, the `title` breaks its rule, or a
    `stream_id` repeats.
