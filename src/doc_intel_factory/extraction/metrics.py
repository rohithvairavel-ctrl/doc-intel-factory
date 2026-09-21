"""Precision / recall / F1 for field extraction."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

from ..synthetic.ocr_noise import normalize_for_match


def _match(pred: str | None, gold: str | None, *, fuzzy: bool = True) -> bool:
    if pred is None or gold is None:
        return False
    p, g = normalize_for_match(str(pred)), normalize_for_match(str(gold))
    if p == g:
        return True
    if not fuzzy:
        return False
    # tolerate OCR noise: containment or high char overlap
    if p in g or g in p:
        return True
    # simple Jaccard on character bigrams
    def bigrams(s: str) -> set[str]:
        return {s[i : i + 2] for i in range(len(s) - 1)} if len(s) > 1 else {s}

    bp, bg = bigrams(p), bigrams(g)
    if not bp or not bg:
        return False
    j = len(bp & bg) / len(bp | bg)
    return j >= 0.7


def field_f1(
    predictions: list[Mapping[str, Any]],
    golds: list[Mapping[str, Any]],
    fields: list[str],
) -> dict[str, float]:
    """Micro-averaged P/R/F1 over listed fields."""
    tp = fp = fn = 0
    for pred, gold in zip(predictions, golds):
        for name in fields:
            p = pred.get(name)
            g = gold.get(name)
            if p is None and g is None:
                continue
            if p is None:
                fn += 1
            elif g is None:
                fp += 1
            elif _match(str(p), str(g)):
                tp += 1
            else:
                fp += 1
                fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def extraction_report(
    predictions: list[Mapping[str, Any]],
    golds: list[Mapping[str, Any]],
    fields: list[str],
) -> dict[str, Any]:
    """Per-field and micro metrics."""
    per_field = {}
    for name in fields:
        per_field[name] = field_f1(
            [{name: p.get(name)} for p in predictions],
            [{name: g.get(name)} for g in golds],
            [name],
        )
    micro = field_f1(predictions, golds, fields)
    return {"micro": micro, "per_field": per_field}
