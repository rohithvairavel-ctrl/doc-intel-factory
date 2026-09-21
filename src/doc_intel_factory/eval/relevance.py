"""Retrieval relevance metrics."""

from __future__ import annotations

from typing import Sequence


def hit_at_k(retrieved_doc_ids: Sequence[str], gold_doc_id: str, k: int = 5) -> float:
    return 1.0 if gold_doc_id in list(retrieved_doc_ids)[:k] else 0.0


def mrr(retrieved_doc_ids: Sequence[str], gold_doc_id: str) -> float:
    for i, did in enumerate(retrieved_doc_ids):
        if did == gold_doc_id:
            return 1.0 / (i + 1)
    return 0.0


def answer_relevance(answer: str, gold_answer: str) -> float:
    """Binary-ish relevance: gold value present in answer."""
    g = gold_answer.lower().strip()
    if not g:
        return 0.0
    if g in answer.lower():
        return 1.0
    # partial
    parts = g.replace("-", " ").split()
    if parts and sum(1 for p in parts if p.lower() in answer.lower()) / len(parts) >= 0.6:
        return 0.6
    return 0.0
