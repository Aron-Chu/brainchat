"""Label scheme v1 (spec section 4), loaded from ``config/labels_v1.json``.

The JSON file is the single source of the value lists. ``load`` checks its shape
so a typo fails loudly instead of reaching Label Studio, the teacher or a model.

``display`` maps question keys, section keys and bare values to the words a
person reads. ``help`` holds one-line definitions keyed by question or section
key (``"doing"``, ``"not_sure"``) and by ``"<group>.<value>"`` for values
(``"doing.laughing"``, ``"not_sure.unclear"``, ``"spread.chant"``,
``"cause_source.clip"``), because ``unclear`` is both an ``aimed_at`` value and
a not-sure reason with different meanings.

``validate_answers`` is the one check for a label's answers (ruling C1): sync, the
teacher and the page fix all call it. ``validate_cause_source`` checks a cause note's
source, since the store keeps no value list of its own.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "labels_v1.json"

_NAME = re.compile(r"[a-z][a-z0-9_]*")  # always used with fullmatch
_TOP_KEYS = {
    "version",
    "guide_version",
    "questions",
    "not_sure_reasons",
    "spread_values",
    "cause_sources",
    "display",
    "help",
}
_QUESTION_KEYS = {"key", "kind", "optional", "blank_value", "values"}


class SchemeError(ValueError):
    """The label scheme file is malformed or incomplete."""


@dataclass(frozen=True)
class Question:
    key: str
    kind: Literal["multi", "single"]
    values: tuple[str, ...]
    optional: bool
    blank_value: str | None


@dataclass(frozen=True)
class Scheme:
    version: str
    questions: tuple[Question, ...]
    not_sure_reasons: tuple[str, ...]
    spread_values: tuple[str, ...]
    cause_sources: tuple[str, ...]
    display: dict[str, str] = field(hash=False)
    help: dict[str, str] = field(hash=False)
    guide_version: str = "cb_guide_2026_11_06_v1"


def load(path: Path | None = None) -> Scheme:
    """Read and check a label scheme file (default: ``config/labels_v1.json``)."""
    path = DEFAULT_PATH if path is None else Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemeError(f"cannot read label scheme {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise SchemeError("label scheme must be a JSON object")
    _exact_keys(raw, _TOP_KEYS, "label scheme")

    version = _text(raw["version"], "version")
    guide_version = _text(raw["guide_version"], "guide_version")

    if not isinstance(raw["questions"], list) or not raw["questions"]:
        raise SchemeError("questions must be a non-empty list")
    questions = tuple(_question(q) for q in raw["questions"])
    _unique([q.key for q in questions], "question keys")

    reasons = _names(raw["not_sure_reasons"], "not_sure_reasons")
    spread = _names(raw["spread_values"], "spread_values")
    causes = _names(raw["cause_sources"], "cause_sources")

    display = _text_map(raw["display"], "display")
    help_ = _text_map(raw["help"], "help")

    groups: list[tuple[str, tuple[str, ...]]] = [
        (q.key, q.values + ((q.blank_value,) if q.blank_value else ())) for q in questions
    ]
    groups += [("not_sure", reasons), ("spread", spread), ("cause_source", causes)]

    missing_display = [q.key for q in questions if q.key not in display]
    missing_display += [v for _, values in groups for v in values if v not in display]
    if missing_display:
        raise SchemeError(f"display is missing: {', '.join(missing_display)}")

    missing_help = [q.key for q in questions if q.key not in help_]
    missing_help += [f"{g}.{v}" for g, values in groups for v in values if f"{g}.{v}" not in help_]
    if missing_help:
        raise SchemeError(f"help is missing: {', '.join(missing_help)}")

    return Scheme(
        version=version,
        questions=questions,
        not_sure_reasons=reasons,
        spread_values=spread,
        cause_sources=causes,
        display=display,
        help=help_,
        guide_version=guide_version,
    )


def validate_answers(scheme: Scheme, answers: dict, not_sure: dict) -> None:
    """Check one label's ``answers`` and ``not_sure`` dicts against the scheme (ruling C1).

    - ``answers = {question: value}``: a multi question (``doing``) maps to a non-empty
      list of its values in the scheme's order with no repeats; a single question maps
      to one value string, and an optional question may hold its ``blank_value``.
    - ``not_sure = {question: [reasons]}``: a non-empty list of ``not_sure_reasons`` in
      the scheme's order with no repeats.
    - Every key is a question key, and a question sits in at most one of the two dicts.
      Either dict may be partial: a label row covers only the questions it names.

    Raises ``ValueError`` naming the question and the rule. Messages never echo a value
    or an unknown key, because those can come from chat (a teacher reply, a Label
    Studio field).
    """
    if not isinstance(answers, dict):
        raise ValueError("answers: must be a dict of question -> value")
    if not isinstance(not_sure, dict):
        raise ValueError("not_sure: must be a dict of question -> [reasons]")
    questions = {q.key: q for q in scheme.questions}
    for where, given in (("answers", answers), ("not_sure", not_sure)):
        if any(k not in questions for k in given):
            raise ValueError(f"{where}: a key is not a question in the scheme ({', '.join(questions)})")
    for key, q in questions.items():  # scheme order, so the first error reported is stable
        if key in answers and key in not_sure:
            raise ValueError(f"{key}: in both answers and not_sure; a question is answered or not sure, never both")
        if key in answers:
            _check_answer(q, answers[key])
        if key in not_sure:
            _check_list(not_sure[key], scheme.not_sure_reasons, f"not_sure.{key}", "not-sure reasons", "reason")


def validate_cause_source(scheme: Scheme, value: str | None) -> None:
    """Check a label's ``cause_source``: ``None`` (no cause note) or one of ``scheme.cause_sources``."""
    if value is None:
        return
    if not isinstance(value, str) or value not in scheme.cause_sources:
        raise ValueError(f"cause_source: must be one of {', '.join(scheme.cause_sources)}, or None")


def _check_answer(q: Question, value: Any) -> None:
    where = f"answers.{q.key}"
    if q.kind == "multi":
        _check_list(value, q.values, where, f"{q.key} values", "value")
        return
    allowed = q.values + ((q.blank_value,) if q.optional and q.blank_value else ())
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{where}: must be one {q.key} value ({', '.join(allowed)})")


def _check_list(value: Any, allowed: tuple[str, ...], where: str, what: str, item: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{where}: must be a non-empty list of {what}")
    if not all(isinstance(v, str) and v in allowed for v in value):
        raise ValueError(f"{where}: must hold only {what} ({', '.join(allowed)})")
    if len(set(value)) != len(value):
        raise ValueError(f"{where}: repeats a {item}")
    order = [allowed.index(v) for v in value]
    if order != sorted(order):
        raise ValueError(f"{where}: must be in scheme order ({', '.join(allowed)})")


def _question(raw: Any) -> Question:
    if not isinstance(raw, dict):
        raise SchemeError("each question must be a JSON object")
    _exact_keys(raw, _QUESTION_KEYS, "question")
    key = _name(raw["key"], "question key")
    kind = raw["kind"]
    if kind not in ("multi", "single"):
        raise SchemeError(f"question {key}: kind must be 'multi' or 'single', not {kind!r}")
    values = _names(raw["values"], f"question {key} values")
    optional = raw["optional"]
    if not isinstance(optional, bool):
        raise SchemeError(f"question {key}: optional must be true or false")
    blank = raw["blank_value"]
    if blank is not None:
        blank = _name(blank, f"question {key} blank_value")
        if not optional:
            raise SchemeError(f"question {key}: only an optional question may have a blank_value")
        if blank in values:
            raise SchemeError(f"question {key}: blank_value {blank!r} repeats a value")
    return Question(key=key, kind=kind, values=values, optional=optional, blank_value=blank)


def _exact_keys(raw: dict, expected: set[str], what: str) -> None:
    missing = expected - raw.keys()
    extra = raw.keys() - expected
    if missing or extra:
        raise SchemeError(f"{what}: missing keys {sorted(missing)}, unknown keys {sorted(extra)}")


def _text(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemeError(f"{what} must be a non-empty string")
    return value


def _name(value: Any, what: str) -> str:
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise SchemeError(f"{what} must be a snake_case name, not {value!r}")
    return value


def _names(value: Any, what: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise SchemeError(f"{what} must be a non-empty list")
    names = tuple(_name(v, what) for v in value)
    _unique(names, what)
    return names


def _unique(items: list[str] | tuple[str, ...], what: str) -> None:
    dupes = sorted(x for x, n in Counter(items).items() if n > 1)
    if dupes:
        raise SchemeError(f"{what} repeat: {', '.join(dupes)}")


def _text_map(value: Any, what: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise SchemeError(f"{what} must be a JSON object")
    out: dict[str, str] = {}
    for k, v in value.items():
        out[k] = _text(v, f"{what}[{k!r}]")
        if v.splitlines() != [v]:  # \n, \r and every other line break
            raise SchemeError(f"{what}[{k!r}] must be one line")
    return out
