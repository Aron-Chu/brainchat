"""``chatbrain ask-eval``: how well the local model reads the 60 test questions (Task 10c).

Step 1 (``Narrator.interpret``, then ``query.check``) reads every question in
``tests/data/ask_questions.json`` against a made-up three-channel world built in a temporary
folder, and so does the keyword path. A question counts for "exact intent" when the intent
matches, and for "exact filters" when the intent and every filter the entry names match
(``same``). The bar is 90% intent and 75% filters for the model.

The output is rates and failure codes only: never a question, a prompt or model output. It asks
Ollama nothing before ``ollama.check`` passes and never preloads the model (each question loads
it as needed, with keep_alive 30 minutes).
"""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from .. import fake, ingest, narrator, ollama
from ..scheme import Scheme
from ..store import Store
from . import keywords
from . import query as q_mod

BAR_INTENT = 0.9
BAR_FILTERS = 0.75
LIST_KEYS = ("doing", "aimed", "feels", "tone", "spread", "is", "emote", "channel")
SCALAR_KEYS = ("phrase", "at", "min_posters", "reason")
OPTIONAL_KEYS = ("time", "sort", "about", "by", "limit")
PROBLEM = {
    "off": "Ollama is not running; start it with ollama serve",
    "old": "Ollama 0.31.2 or newer is needed",
    "missing": "the model is missing; get it with ollama pull",
    "not_loopback": "BrainChat uses Ollama only on this PC (127.0.0.1)",
}


def same(query: dict[str, Any], expect: dict[str, Any]) -> list[str]:
    """The keys where ``query`` differs from ``expect`` (lists default to empty, scalars to None)."""
    bad = [k for k in LIST_KEYS if query[k] != expect.get(k, [])]
    bad += [k for k in SCALAR_KEYS if query[k] != expect.get(k)]
    bad += [k for k in OPTIONAL_KEYS if k in expect and query[k] != expect[k]]
    return bad


def _rate(k: int, n: int) -> str:
    return f"{k}/{n} = {k / n:.0%}" if n else "0/0"


def run(nr: narrator.Narrator, scheme: Scheme, questions: Path, now: datetime, out: Callable[[str], None]) -> int:
    entries = json.loads(Path(questions).read_text(encoding="utf-8"))
    try:
        nr.check()
    except ollama.OllamaError as exc:
        out(f"Ollama: {exc}")
        return 1
    st = nr.state(fresh=True)
    if not st.ok:
        out(f"Ollama: {PROBLEM.get(st.problem or '', st.problem)}")
        return 1
    n = len(entries)
    m_intent = m_filters = k_intent = k_filters = 0
    failed: Counter[str] = Counter()
    with tempfile.TemporaryDirectory(prefix="chatbrain-ask-eval-") as tmp:
        folder = Path(tmp)
        (folder / "e3.json").write_bytes(fake.make_export(3, 12, 16, now, channels=3))
        with Store(folder / "cb.db") as store:
            ingest.import_export(store, folder / "e3.json", folder / "imported", now)
            held = q_mod.held(store)
            schema = q_mod.schema(store, scheme, held)
            for e in entries:
                expect = e["expect"]
                got = keywords.parse(e["q"], scheme, held).query
                if got["intent"] == expect["intent"]:
                    k_intent += 1
                    k_filters += same(got, expect) == []
                try:
                    raw = nr.interpret(e["q"], schema=schema, previous=None, previous_rows=[], scheme=scheme)
                except narrator.StepTimeout:
                    failed["timeout"] += 1
                    continue
                except narrator.StepFailed as exc:
                    failed[exc.code] += 1
                    continue
                except ollama.OllamaError:
                    failed["ollama"] += 1
                    continue
                checked = q_mod.check(raw, scheme, held, question=e["q"])
                if "unusable" in checked.codes:
                    failed["unusable"] += 1
                    continue
                q = checked.query
                if q["intent"] == expect["intent"]:
                    m_intent += 1
                    m_filters += same(q, expect) == []
    met = n and m_intent / n >= BAR_INTENT and m_filters / n >= BAR_FILTERS
    out(f"local model {nr.model}: exact intent {_rate(m_intent, n)}, exact filters {_rate(m_filters, n)}"
        f" (prompt {narrator.PROMPT_VERSION})")  # fmt: skip
    out(f"keyword path: exact intent {_rate(k_intent, n)}, exact filters {_rate(k_filters, n)}")
    out(f"bar: {BAR_INTENT:.0%} intent, {BAR_FILTERS:.0%} filters: {'met' if met else 'not met'}")
    out("failed: " + (", ".join(f"{code} {k}" for code, k in sorted(failed.items())) or "none"))
    return 0
