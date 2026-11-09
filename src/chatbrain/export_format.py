"""The export file contract between the StreamPulse server script and this PC.

Three JSON formats cross the line, all described field by field in
``docs/export-format.md``:

- ``chatbrain-export/1`` (server -> PC): windows of made-up or real chat;
  ``chatbrain-export/2`` adds ``streams``: per-stream activity counts (numbers only);
- ``chatbrain-window-list/1`` (PC -> server): the windows the PC holds, no text;
- ``chatbrain-deletion-check/1`` (server -> PC): which of those messages still exist.

Parsing is strict and fails closed: unknown or missing keys, duplicate keys, any
identity field (``login``, ``display_name``, ``commenter_login``, ``sender_hash``,
``user_id``) anywhere, naive datetimes, a window that isn't anchor -30 s to +90 s,
speakers that aren't per-window numbers, un-rewritten ``@mentions`` and messages
under one hour old all raise ``ExportRejected``.

Rejection messages name the place and the rule, never the offending value: a bad
value may be chat text or a chatter's name, and the terminal is not a safe place
for either.
"""

from __future__ import annotations

import functools
import json
import re
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, ParamSpec, TypeVar, get_args

P = ParamSpec("P")
R = TypeVar("R")

EXPORT_FORMAT = "chatbrain-export/1"
EXPORT_FORMAT_V2 = "chatbrain-export/2"
EXPORT_FORMATS = (EXPORT_FORMAT, EXPORT_FORMAT_V2)
DELETION_CHECK_FORMAT = "chatbrain-deletion-check/1"
WINDOW_LIST_FORMAT = "chatbrain-window-list/1"

WINDOW_BEFORE = timedelta(seconds=30)
WINDOW_LENGTH = timedelta(seconds=120)
WINDOW_MS = 120_000
MIN_MESSAGE_AGE = timedelta(hours=1)
MAX_TEXT_LIFETIME = timedelta(days=14)
ACTIVITY_BIN = timedelta(seconds=10)
ACTIVITY_BIN_S = 10
MAX_ACTIVITY_BINS = 17_280  # 48 h of 10 s bins
MAX_CHATTERS = 100_000
MAX_TITLE_CHARS = 140

AnchorKind = Literal["moment", "ordinary"]
Source = Literal["vod", "live"]
ANCHOR_KINDS: tuple[str, ...] = get_args(AnchorKind)
SOURCES: tuple[str, ...] = get_args(Source)

# Field names that must never appear anywhere, compared case-insensitively with
# "_" and "-" removed, so "displayName" and "Sender-Hash" are caught too.
IDENTITY_FIELDS = ("login", "display_name", "commenter_login", "sender_hash", "user_id")
_IDENTITY_EXTRA = ("user_login", "username", "user_name", "message_hash", "text_hash")
_IDENTITY_KEYS = frozenset(k.replace("_", "") for k in IDENTITY_FIELDS + _IDENTITY_EXTRA)

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_CHANNEL = re.compile(r"[a-z0-9_]{1,25}")
_LANGUAGE = re.compile(r"[A-Za-z]{2,8}(-[A-Za-z0-9]{1,8})*")
_SPEAKER = re.compile(r"s[1-9][0-9]*")
_MENTION = re.compile(r"@([A-Za-z0-9_]+)")
_EMOTE_NAME = re.compile(r"\S{1,100}")
_DATETIME = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]{1,6})?(Z|[+-][0-9]{2}:[0-9]{2})")

_EXPORT_KEYS = {"format", "exported_at", "source", "windows"}
_EXPORT_KEYS_V2 = _EXPORT_KEYS | {"streams"}
_STREAM_KEYS = {"stream_id", "channel", "started_at", "ended_at", "title", "bin_s", "chatters"}
_WINDOW_KEYS = {
    "window_id",
    "channel",
    "stream_id",
    "anchor_kind",
    "anchor_at",
    "start_at",
    "end_at",
    "language",
    "text_expires_at",
    "messages",
}
_MESSAGE_KEYS = {"message_id", "rel_ms", "speaker", "text", "emotes"}
_EMOTE_KEYS = {"id", "name"}
_CHECK_KEYS = {"format", "checked_at", "windows"}
_CHECK_WINDOW_KEYS = {"window_id", "present"}


class ExportRejected(ValueError):
    """An export, deletion check or window list broke the contract."""


@dataclass(frozen=True)
class Emote:
    id: str
    name: str


@dataclass(frozen=True)
class Message:
    message_id: str
    rel_ms: int  # milliseconds after the window's start_at, 0 <= rel_ms < 120000
    speaker: str  # per-window speaker number: "s1", "s2", ...
    text: str
    emotes: tuple[Emote, ...]


@dataclass(frozen=True)
class Window:
    window_id: str
    channel: str
    stream_id: str
    anchor_kind: AnchorKind
    anchor_at: datetime
    start_at: datetime
    end_at: datetime
    language: str
    text_expires_at: datetime
    messages: tuple[Message, ...]


@dataclass(frozen=True)
class StreamActivity:
    """People chatting per 10 s bin across one stream: counts only, no chat text or IDs."""

    stream_id: str
    channel: str
    started_at: datetime
    ended_at: datetime
    title: str | None
    bin_s: int
    chatters: tuple[int | None, ...]  # None = that bin was not recorded


@dataclass(frozen=True)
class Export:
    format: str
    exported_at: datetime
    source: Source
    windows: tuple[Window, ...]
    streams: tuple[StreamActivity, ...] = ()


@dataclass(frozen=True)
class DeletionCheck:
    checked_at: datetime
    # window_id -> the message IDs that still exist on the server.
    present: dict[str, frozenset[str]] = field(hash=False)


# --- datetimes -----------------------------------------------------------------


def iso(dt: datetime) -> str:
    """Canonical UTC text for a timezone-aware datetime: ``YYYY-MM-DDTHH:MM:SS.ffffffZ``.

    Fixed width, so the text sorts the same way the times do (the store relies on it).
    """
    if not isinstance(dt, datetime) or dt.utcoffset() is None:
        raise ExportRejected("datetime must be timezone-aware")
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_dt(value: Any, where: str = "datetime") -> datetime:
    """Parse RFC 3339 text with an explicit offset (``Z`` or ``+HH:MM``) into a UTC datetime."""
    if not isinstance(value, str) or not _DATETIME.fullmatch(value):
        raise ExportRejected(f"{where}: must be RFC 3339 text with an explicit offset, like 2026-11-06T12:00:00Z")
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        raise ExportRejected(f"{where}: not a valid date and time") from None
    if dt.utcoffset() is None:  # unreachable after the pattern, kept as a guard
        raise ExportRejected(f"{where}: naive datetime (no UTC offset)")
    return dt.astimezone(timezone.utc)


# --- export ---------------------------------------------------------------------


def _fail_closed(fn: Callable[P, R]) -> Callable[P, R]:
    """Turn errors that hostile input can trigger deep inside the checks into ``ExportRejected``."""

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return fn(*args, **kwargs)
        except OverflowError:
            raise ExportRejected("a time is out of range") from None
        except RecursionError:
            raise ExportRejected("input is nested too deeply") from None

    return wrapper


@_fail_closed
def parse_export(raw: bytes) -> Export:
    """Parse and check a ``chatbrain-export/1`` or ``/2`` file. Raises ``ExportRejected``."""
    doc = _load(raw)
    # The format string picks the key set; anything but /2 is held to /1's keys and messages.
    v2 = isinstance(doc, dict) and doc.get("format") == EXPORT_FORMAT_V2
    _refuse_identity(doc, "export", v2)
    obj = _object(doc, _EXPORT_KEYS_V2 if v2 else _EXPORT_KEYS, "export")
    if obj["format"] != (EXPORT_FORMAT_V2 if v2 else EXPORT_FORMAT):
        raise ExportRejected(f"export.format: must be {EXPORT_FORMAT!r}")
    exported_at = parse_dt(obj["exported_at"], "export.exported_at")
    source = obj["source"]
    if source not in SOURCES:
        raise ExportRejected(f"export.source: must be one of {', '.join(SOURCES)}")
    windows = _list(obj["windows"], "export.windows")

    parsed: list[Window] = []
    seen: set[str] = set()
    for i, raw_window in enumerate(windows):
        w = _window(raw_window, exported_at, f"export.windows[{i}]")
        if w.window_id in seen:
            raise ExportRejected(f"export.windows[{i}].window_id: repeats an earlier window")
        seen.add(w.window_id)
        parsed.append(w)
    streams = _streams(obj["streams"], exported_at) if v2 else ()
    return Export(
        format=EXPORT_FORMAT_V2 if v2 else EXPORT_FORMAT,
        exported_at=exported_at,
        source=source,
        windows=tuple(parsed),
        streams=streams,
    )


def dump_export(export: Export) -> bytes:
    """Write an ``Export`` as ``chatbrain-export/1`` or ``/2`` bytes, checked by ``parse_export`` first."""
    if export.format != EXPORT_FORMAT_V2 and export.streams:
        raise ExportRejected(f"export.streams: only {EXPORT_FORMAT_V2!r} carries streams")
    doc: dict[str, Any] = {
        "format": export.format,
        "exported_at": iso(export.exported_at),
        "source": export.source,
        "windows": [
            {
                "window_id": w.window_id,
                "channel": w.channel,
                "stream_id": w.stream_id,
                "anchor_kind": w.anchor_kind,
                "anchor_at": iso(w.anchor_at),
                "start_at": iso(w.start_at),
                "end_at": iso(w.end_at),
                "language": w.language,
                "text_expires_at": iso(w.text_expires_at),
                "messages": [
                    {
                        "message_id": m.message_id,
                        "rel_ms": m.rel_ms,
                        "speaker": m.speaker,
                        "text": m.text,
                        "emotes": [{"id": e.id, "name": e.name} for e in m.emotes],
                    }
                    for m in w.messages
                ],
            }
            for w in export.windows
        ],
    }
    if export.format == EXPORT_FORMAT_V2:
        doc["streams"] = [
            {
                "stream_id": s.stream_id,
                "channel": s.channel,
                "started_at": iso(s.started_at),
                "ended_at": iso(s.ended_at),
                "title": s.title,
                "bin_s": s.bin_s,
                "chatters": list(s.chatters),
            }
            for s in export.streams
        ]
    raw = _dump(doc)
    parse_export(raw)  # never write a file this PC would refuse to read
    return raw


def _streams(raw: Any, exported_at: datetime) -> tuple[StreamActivity, ...]:
    out: list[StreamActivity] = []
    seen: set[str] = set()
    for i, raw_stream in enumerate(_list(raw, "export.streams")):
        s = _stream(raw_stream, exported_at, f"export.streams[{i}]")
        if s.stream_id in seen:
            raise ExportRejected(f"export.streams[{i}].stream_id: repeats an earlier stream")
        seen.add(s.stream_id)
        out.append(s)
    return tuple(out)


def _stream(raw: Any, exported_at: datetime, where: str) -> StreamActivity:
    obj = _object(raw, _STREAM_KEYS, where)
    stream_id = _match(obj["stream_id"], _ID, f"{where}.stream_id", "an ID of letters, digits and . _ : -")
    channel = _match(obj["channel"], _CHANNEL, f"{where}.channel", "a lowercase channel name")
    started_at = parse_dt(obj["started_at"], f"{where}.started_at")
    ended_at = parse_dt(obj["ended_at"], f"{where}.ended_at")
    if ended_at <= started_at:
        raise ExportRejected(f"{where}.ended_at: must be after started_at")
    if ended_at > exported_at:
        raise ExportRejected(f"{where}.ended_at: must not be after exported_at")
    title = obj["title"]
    if title is not None:
        bad = (
            not isinstance(title, str)
            or any(unicodedata.category(ch) == "Cc" or ch in "\u2028\u2029" for ch in title)
            or not 1 <= len(title.strip()) <= MAX_TITLE_CHARS
        )
        if bad:
            raise ExportRejected(f"{where}.title: must be null or 1-{MAX_TITLE_CHARS} characters with no control character or line separator")
        title = title.strip()
    bin_s = obj["bin_s"]
    if type(bin_s) is not int or bin_s != ACTIVITY_BIN_S:
        raise ExportRejected(f"{where}.bin_s: must be {ACTIVITY_BIN_S}")
    values = _list(obj["chatters"], f"{where}.chatters")
    if not 1 <= len(values) <= MAX_ACTIVITY_BINS:
        raise ExportRejected(f"{where}.chatters: must hold 1 to {MAX_ACTIVITY_BINS} values")
    expected = -((started_at - ended_at) // ACTIVITY_BIN)  # ceil((ended_at - started_at) / 10 s)
    if len(values) != expected:
        raise ExportRejected(f"{where}.chatters: must hold one value per 10 s of the stream")
    for j, v in enumerate(values):
        if v is not None and (type(v) is not int or not 0 <= v <= MAX_CHATTERS):
            raise ExportRejected(f"{where}.chatters[{j}]: must be null or a whole number 0 to {MAX_CHATTERS}")
    return StreamActivity(
        stream_id=stream_id,
        channel=channel,
        started_at=started_at,
        ended_at=ended_at,
        title=title,
        bin_s=bin_s,
        chatters=tuple(values),
    )


def _window(raw: Any, exported_at: datetime, where: str) -> Window:
    obj = _object(raw, _WINDOW_KEYS, where)
    window_id = _match(obj["window_id"], _ID, f"{where}.window_id", "an ID of letters, digits and . _ : -")
    channel = _match(obj["channel"], _CHANNEL, f"{where}.channel", "a lowercase channel name")
    stream_id = _match(obj["stream_id"], _ID, f"{where}.stream_id", "an ID of letters, digits and . _ : -")
    anchor_kind = obj["anchor_kind"]
    if anchor_kind not in ANCHOR_KINDS:
        raise ExportRejected(f"{where}.anchor_kind: must be one of {', '.join(ANCHOR_KINDS)}")
    anchor_at = parse_dt(obj["anchor_at"], f"{where}.anchor_at")
    start_at = parse_dt(obj["start_at"], f"{where}.start_at")
    end_at = parse_dt(obj["end_at"], f"{where}.end_at")
    if end_at - start_at != WINDOW_LENGTH:
        raise ExportRejected(f"{where}: end_at - start_at must be exactly 120 s")
    if anchor_at - start_at != WINDOW_BEFORE:
        raise ExportRejected(f"{where}: anchor_at - start_at must be exactly 30 s")
    language = _match(obj["language"], _LANGUAGE, f"{where}.language", "a language code such as en")
    text_expires_at = parse_dt(obj["text_expires_at"], f"{where}.text_expires_at")
    if text_expires_at <= end_at:
        raise ExportRejected(f"{where}.text_expires_at: must be after end_at")
    if text_expires_at > exported_at + MAX_TEXT_LIFETIME:
        raise ExportRejected(f"{where}.text_expires_at: more than 14 days after exported_at")

    newest_allowed = exported_at - MIN_MESSAGE_AGE
    messages: list[Message] = []
    seen: set[str] = set()
    for j, raw_msg in enumerate(_list(obj["messages"], f"{where}.messages")):
        m = _message(raw_msg, f"{where}.messages[{j}]")
        if m.message_id in seen:
            raise ExportRejected(f"{where}.messages[{j}].message_id: repeats an earlier message in this window")
        seen.add(m.message_id)
        if start_at + timedelta(milliseconds=m.rel_ms) > newest_allowed:
            raise ExportRejected(f"{where}.messages[{j}]: under one hour old at exported_at")
        messages.append(m)

    speakers = {m.speaker for m in messages}
    for j, m in enumerate(messages):
        for name in _MENTION.findall(m.text):
            if name == "user":
                continue
            if not _SPEAKER.fullmatch(name):
                raise ExportRejected(f"{where}.messages[{j}].text: @mention not rewritten to @s<N> or @user")
            if name not in speakers:
                raise ExportRejected(f"{where}.messages[{j}].text: @s<N> names a speaker who did not post in this window")

    messages.sort(key=lambda m: (m.rel_ms, m.message_id))
    return Window(
        window_id=window_id,
        channel=channel,
        stream_id=stream_id,
        anchor_kind=anchor_kind,
        anchor_at=anchor_at,
        start_at=start_at,
        end_at=end_at,
        language=language,
        text_expires_at=text_expires_at,
        messages=tuple(messages),
    )


def _message(raw: Any, where: str) -> Message:
    obj = _object(raw, _MESSAGE_KEYS, where)
    message_id = _match(obj["message_id"], _ID, f"{where}.message_id", "an ID of letters, digits and . _ : -")
    rel_ms = obj["rel_ms"]
    if type(rel_ms) is not int or not 0 <= rel_ms < WINDOW_MS:
        raise ExportRejected(f"{where}.rel_ms: must be a whole number of ms, 0 <= rel_ms < 120000")
    speaker = _match(obj["speaker"], _SPEAKER, f"{where}.speaker", "a per-window speaker number s1, s2, ...")
    text = obj["text"]
    if not isinstance(text, str):
        raise ExportRejected(f"{where}.text: must be a string")
    emotes = []
    for k, raw_emote in enumerate(_list(obj["emotes"], f"{where}.emotes")):
        e = _object(raw_emote, _EMOTE_KEYS, f"{where}.emotes[{k}]")
        emotes.append(
            Emote(
                id=_match(e["id"], _ID, f"{where}.emotes[{k}].id", "an ID of letters, digits and . _ : -"),
                name=_match(e["name"], _EMOTE_NAME, f"{where}.emotes[{k}].name", "1-100 characters, no spaces"),
            )
        )
    return Message(message_id=message_id, rel_ms=rel_ms, speaker=speaker, text=text, emotes=tuple(emotes))


# --- deletion check -------------------------------------------------------------------


@_fail_closed
def parse_deletion_check(raw: bytes) -> DeletionCheck:
    """Parse and check a ``chatbrain-deletion-check/1`` file. Raises ``ExportRejected``."""
    doc = _load(raw)
    _refuse_identity(doc, "deletion_check")
    obj = _object(doc, _CHECK_KEYS, "deletion_check")
    if obj["format"] != DELETION_CHECK_FORMAT:
        raise ExportRejected(f"deletion_check.format: must be {DELETION_CHECK_FORMAT!r}")
    checked_at = parse_dt(obj["checked_at"], "deletion_check.checked_at")
    present: dict[str, frozenset[str]] = {}
    for i, raw_window in enumerate(_list(obj["windows"], "deletion_check.windows")):
        where = f"deletion_check.windows[{i}]"
        w = _object(raw_window, _CHECK_WINDOW_KEYS, where)
        window_id = _match(w["window_id"], _ID, f"{where}.window_id", "an ID of letters, digits and . _ : -")
        if window_id in present:
            raise ExportRejected(f"{where}.window_id: repeats an earlier window")
        ids = [
            _match(x, _ID, f"{where}.present[{j}]", "a message ID")
            for j, x in enumerate(_list(w["present"], f"{where}.present"))
        ]
        if len(set(ids)) != len(ids):
            raise ExportRejected(f"{where}.present: repeats a message ID")
        present[window_id] = frozenset(ids)
    return DeletionCheck(checked_at=checked_at, present=present)


def dump_deletion_check(check: DeletionCheck) -> bytes:
    """Write a ``DeletionCheck`` as ``chatbrain-deletion-check/1`` bytes (sorted, so stable)."""
    doc = {
        "format": DELETION_CHECK_FORMAT,
        "checked_at": iso(check.checked_at),
        "windows": [{"window_id": wid, "present": sorted(ids)} for wid, ids in sorted(check.present.items())],
    }
    raw = _dump(doc)
    parse_deletion_check(raw)
    return raw


# --- window list ---------------------------------------------------------------------------


@_fail_closed
def window_list(rows: Iterable[tuple[str, str, datetime, datetime]]) -> bytes:
    """The PC -> server request for a deletion check: window bounds only, never text.

    Each row is ``(window_id, stream_id, start_at, end_at)``; the output is
    ``{"format": "chatbrain-window-list/1", "windows": [{window_id, stream_id, start_at, end_at}]}``.
    """
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for i, row in enumerate(rows):
        where = f"window_list[{i}]"
        if not isinstance(row, tuple) or len(row) != 4:
            raise ExportRejected(f"{where}: must be (window_id, stream_id, start_at, end_at)")
        window_id, stream_id, start_at, end_at = row
        _match(window_id, _ID, f"{where}.window_id", "an ID of letters, digits and . _ : -")
        _match(stream_id, _ID, f"{where}.stream_id", "an ID of letters, digits and . _ : -")
        for name, dt in (("start_at", start_at), ("end_at", end_at)):
            if not isinstance(dt, datetime) or dt.utcoffset() is None:
                raise ExportRejected(f"{where}.{name}: must be a timezone-aware datetime")
        if end_at - start_at != WINDOW_LENGTH:
            raise ExportRejected(f"{where}: end_at - start_at must be exactly 120 s")
        if window_id in seen:
            raise ExportRejected(f"{where}.window_id: repeats an earlier window")
        seen.add(window_id)
        out.append({"window_id": window_id, "stream_id": stream_id, "start_at": iso(start_at), "end_at": iso(end_at)})
    return _dump({"format": WINDOW_LIST_FORMAT, "windows": out})


# --- helpers -----------------------------------------------------------------------------------


def _load(raw: bytes) -> Any:
    if not isinstance(raw, (bytes, bytearray)):
        raise ExportRejected("input must be bytes")
    try:
        text = bytes(raw).decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ExportRejected("input is not UTF-8") from None
    try:
        return json.loads(text, object_pairs_hook=_no_duplicate_keys, parse_constant=_no_constants)
    except json.JSONDecodeError as exc:
        raise ExportRejected(f"input is not valid JSON (line {exc.lineno}, column {exc.colno})") from None
    except ExportRejected:
        raise  # from the hooks below: already a clear rejection
    except ValueError:
        # e.g. an integer over Python's 4300-digit conversion limit
        raise ExportRejected("input is not valid JSON") from None


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            # The key comes from the file, so it is not echoed.
            raise ExportRejected("a JSON object repeats a key")
        out[key] = value
    return out


def _no_constants(name: str) -> Any:
    raise ExportRejected("NaN and Infinity are not allowed")


def _refuse_identity(node: Any, where: str, v2: bool = False) -> None:
    """Refuse any identity field at any depth, before the schema check, for a clear message.

    The same walk refuses strings that aren't valid Unicode (a lone surrogate from a
    ``\\ud800`` escape), which SQLite and UTF-8 output could not store later.
    """
    if isinstance(node, str):
        _unicode(node, where)
    elif isinstance(node, dict):
        for key, value in node.items():
            _unicode(key, where)
            norm = key.casefold().replace("_", "").replace("-", "")
            if norm in _IDENTITY_KEYS:
                label = key if key.casefold() in {k.casefold() for k in IDENTITY_FIELDS + _IDENTITY_EXTRA} else "a field"
                raise ExportRejected(f"{where}: identity field {label} is never exported")
            _refuse_identity(value, f"{where}.{_safe_key(key, v2)}", v2)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            _refuse_identity(item, f"{where}[{i}]", v2)


def _unicode(text: str, where: str) -> None:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        raise ExportRejected(f"{where}: a string is not valid Unicode") from None


_SCHEMA_KEYS = frozenset(_EXPORT_KEYS | _WINDOW_KEYS | _MESSAGE_KEYS | _EMOTE_KEYS | _CHECK_KEYS | _CHECK_WINDOW_KEYS)
# A /2 file also knows the stream keys; a /1 file keeps exactly its /1 key names (and messages).
_SCHEMA_KEYS_V2 = frozenset(_SCHEMA_KEYS | _EXPORT_KEYS_V2 | _STREAM_KEYS)


def _safe_key(key: str, v2: bool = False) -> str:
    # Only this contract's own key names are echoed: any other key comes from the
    # file and may be a chatter's name (a map keyed by login, say).
    return key if key in (_SCHEMA_KEYS_V2 if v2 else _SCHEMA_KEYS) else "?"


def _object(value: Any, keys: set[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExportRejected(f"{where}: must be a JSON object")
    missing = sorted(keys - value.keys())
    extra = len(value.keys() - keys)
    if missing or extra:
        # Unknown keys come from the file, so they are counted, not echoed.
        raise ExportRejected(f"{where}: missing keys {missing}, {extra} unknown key(s)")
    return value


def _list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise ExportRejected(f"{where}: must be a JSON array")
    return value


def _match(value: Any, pattern: re.Pattern[str], where: str, rule: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ExportRejected(f"{where}: must be {rule}")
    return value


def _dump(doc: dict[str, Any]) -> bytes:
    return json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
