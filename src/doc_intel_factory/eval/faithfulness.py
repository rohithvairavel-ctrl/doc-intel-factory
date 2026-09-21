"""Faithfulness / groundedness metrics on synthetic gold."""

from __future__ import annotations

import re


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def faithfulness(answer: str, supporting_spans: list[str]) -> float:
    """Fraction of answer content tokens that appear in supporting spans.

    Higher = more grounded. Template boilerplate is lightly down-weighted by
    stripping common prefixes.
    """
    ans = answer
    for prefix in (
        "based on retrieved document spans:",
        "i cannot find supporting evidence in the indexed documents.",
    ):
        if ans.lower().startswith(prefix):
            ans = ans[len(prefix) :]
    # drop trailing (query: ...)
    if "(query:" in ans.lower():
        ans = ans[: ans.lower().rfind("(query:")]
    at = _tokens(ans)
    if not at:
        return 1.0 if not supporting_spans else 0.0
    ctx = _tokens(" ".join(supporting_spans))
    if not ctx:
        return 0.0
    return len(at & ctx) / len(at)


def groundedness(answer: str, supporting_spans: list[str], gold_answer: str) -> float:
    """Gold value appears in answer AND in at least one supporting span."""
    g = gold_answer.lower().strip()
    if not g:
        return 0.0
    in_ans = g in answer.lower()
    in_span = any(g in s.lower() for s in supporting_spans)
    if in_ans and in_span:
        return 1.0
    if in_ans or in_span:
        return 0.5
    # token overlap fallback
    gt = _tokens(g)
    at = _tokens(answer)
    if gt and len(gt & at) / len(gt) >= 0.8:
        return 0.5
    return 0.0
