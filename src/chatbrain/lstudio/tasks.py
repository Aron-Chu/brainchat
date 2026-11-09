"""The JSON of one Label Studio task: a window's lines, its counts, and maybe a prelabel.

``task_json`` sends only what labeling needs: the window ID, each message's per-window
speaker (``s1``, ``s2``, ...) and text, and ``counts_text``. Message IDs, channel and
stream never go to Label Studio. ``counts_text`` holds numbers, emote IDs and the
spread value only, never words (the chant text is counted, then dropped).

A prediction (prelabel) is added only when a suggestion is given, which ``sync.push``
does for the suggested project alone; blind and repeat tasks never carry one.
"""

from __future__ import annotations

from typing import Any

from ..export_format import Window
from ..spread import Counts
from . import config


def counts_text(c: Counts) -> str:
    """One line of counted features: numbers, emote IDs and the spread value, never words."""
    spread = c.spread.replace("_", " ")
    if c.n_messages == 0:
        return f"no chat in this window · spread: {spread}"
    parts = [
        f"{c.n_messages} messages from {c.distinct_posters} posters ({c.msg_per_min:g} per minute)",
        f"spread: {spread}",
    ]
    if c.top_emote_id is not None:
        parts.append(f"top emote {c.top_emote_id} in {c.top_emote_share:.0%} of messages")
    parts += [
        f"chant {c.chant_share:.0%}",
        f"replies {c.mention_share:.0%}",
        f"questions {c.question_share:.0%}",
        f"joining: {c.joining_lines} messages from {c.joining_posters} posters",
    ]
    return " · ".join(parts)


def prediction_result(answers: dict[str, Any], not_sure: dict[str, list[str]]) -> list[dict[str, Any]]:
    """Label Studio ``choices`` results for an ``answers`` / ``not_sure`` pair (ruling C1 shape)."""
    result = []
    for key, value in answers.items():
        values = list(value) if isinstance(value, list) else [value]
        result.append(_choices(key, values))
    for key, reasons in not_sure.items():
        result.append(_choices(config.not_sure_name(key), list(reasons)))
    return result


def _choices(from_name: str, values: list[str]) -> dict[str, Any]:
    return {"from_name": from_name, "to_name": config.CHAT, "type": "choices", "value": {"choices": values}}


def model_version(teacher_model: str, prompt_version: str) -> str:
    return f"{teacher_model}/{prompt_version}"


def task_json(w: Window, c: Counts, suggestion: dict | None) -> dict:
    """The task for window ``w`` with counts ``c``.

    ``suggestion`` is ``{"answers", "not_sure", "teacher_model", "prompt_version"}`` or
    None. Only when it is given does the task carry ``predictions``, with
    ``model_version`` = ``f"{teacher_model}/{prompt_version}"``.
    """
    task: dict[str, Any] = {
        "data": {
            "window_id": w.window_id,
            "messages": [{"speaker": m.speaker, "text": m.text} for m in w.messages],
            "counts_text": counts_text(c),
        }
    }
    if suggestion is not None:
        task["predictions"] = [
            {
                "model_version": model_version(suggestion["teacher_model"], suggestion["prompt_version"]),
                "result": prediction_result(suggestion["answers"], suggestion["not_sure"]),
            }
        ]
    return task
