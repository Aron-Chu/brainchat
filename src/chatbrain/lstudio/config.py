"""The Label Studio labeling view, built from the label scheme, and the guide's definitions.

``label_config(scheme)`` returns the XML for all three projects (suggested, blind,
repeat):

- ``<Paragraphs name="chat">`` shows the window's lines as ``speaker: text``, and
  ``<Text name="counts">`` shows the counted features (numbers and emote IDs only).
- Per question, ``<Choices name="{key}">`` (``doing`` is ``multiple``, the others
  ``single``) whose ``<Choice>`` shows the scheme's display text and stores the
  scheme value through ``alias``, with the value's help as its ``hint`` (ruling C31).
  Tone adds its ``blank_value`` (``no_clear_tone``) as a choice.
- Per question, ``<Choices name="{key}_not_sure" choice="multiple">`` holding the
  not-sure reasons. A question is answered with values or with reasons, never both;
  ``sync.pull`` refuses an annotation that breaks this.
- ``<TextArea name="cause_note">`` and ``<Choices name="cause_source">`` for the
  optional cause note.

Hotkeys go only on the four questions' values, in order from ``HOTKEYS``, so they
are unique across the view (25 with the current scheme).

``guide_definitions(text)`` reads the one-line definitions out of
``docs/LABELING_GUIDE.md``; a test checks they equal ``labels_v1.json`` ``help``.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from ..scheme import Question, Scheme

HOTKEYS = "1234567890qwertyuiopasdfghjklzxcvbnm"
CHAT = "chat"
NOT_SURE_SUFFIX = "_not_sure"
CAUSE_NOTE = "cause_note"
CAUSE_SOURCE = "cause_source"

_DEFINITION = re.compile(r"^Definition \(`([a-z_]+(?:\.[a-z_]+)?)`\): (.+?)\s*$", re.MULTILINE)


def choice_values(q: Question) -> tuple[str, ...]:
    """The values a question's ``<Choices>`` offers: its values, then its blank value if it has one."""
    return q.values + ((q.blank_value,) if q.blank_value else ())


def not_sure_name(key: str) -> str:
    return key + NOT_SURE_SUFFIX


def label_config(scheme: Scheme) -> str:
    """The Label Studio XML for ``scheme`` (see the module docstring)."""
    needed = sum(len(choice_values(q)) for q in scheme.questions)
    if needed > len(HOTKEYS):
        raise ValueError(f"the scheme needs {needed} hotkeys; only {len(HOTKEYS)} are available")
    hotkeys = iter(HOTKEYS)

    view = ET.Element("View")
    ET.SubElement(view, "Header", value=f"BrainChat · labels {scheme.version} · guide {scheme.guide_version}")
    ET.SubElement(
        view, "Paragraphs", name=CHAT, value="$messages", layout="dialogue", nameKey="speaker", textKey="text"
    )
    ET.SubElement(view, "Text", name="counts", value="$counts_text")

    for q in scheme.questions:
        group = ET.SubElement(view, "View")
        ET.SubElement(group, "Header", value=scheme.display[q.key])
        ET.SubElement(group, "Header", value=scheme.help[q.key], size="6")
        choices = ET.SubElement(
            group,
            "Choices",
            name=q.key,
            toName=CHAT,
            choice="multiple" if q.kind == "multi" else "single",
            showInline="true",
        )
        for v in choice_values(q):
            ET.SubElement(
                choices,
                "Choice",
                value=scheme.display[v],
                alias=v,
                hint=scheme.help[f"{q.key}.{v}"],
                hotkey=next(hotkeys),
            )
        ET.SubElement(group, "Header", value=f"{scheme.display['not_sure']}: {scheme.help['not_sure']}", size="6")
        reasons = ET.SubElement(
            group, "Choices", name=not_sure_name(q.key), toName=CHAT, choice="multiple", showInline="true"
        )
        for r in scheme.not_sure_reasons:
            ET.SubElement(reasons, "Choice", value=scheme.display[r], alias=r, hint=scheme.help[f"not_sure.{r}"])

    cause = ET.SubElement(view, "View")
    ET.SubElement(cause, "Header", value=scheme.display["cause"])
    ET.SubElement(cause, "Header", value=scheme.help["cause"], size="6")
    ET.SubElement(cause, "TextArea", name=CAUSE_NOTE, toName=CHAT, rows="2", maxSubmissions="1")
    sources = ET.SubElement(cause, "Choices", name=CAUSE_SOURCE, toName=CHAT, choice="single", showInline="true")
    for src in scheme.cause_sources:
        ET.SubElement(sources, "Choice", value=scheme.display[src], alias=src, hint=scheme.help[f"cause_source.{src}"])
    return ET.tostring(view, encoding="unicode")


def guide_definitions(text: str) -> list[tuple[str, str]]:
    """``(help key, definition)`` for every ``Definition (`key`): ...`` line of the guide."""
    return [(m.group(1), m.group(2)) for m in _DEFINITION.finditer(text)]
