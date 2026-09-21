"""Lightweight features for field value candidate scoring."""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

# Patterns hinting at field types
PATTERNS = {
    "invoice_id": re.compile(r"\bINV[- ]?\d{4,6}\b", re.I),
    "contract_id": re.compile(r"\bCTR[- ]?\d{3,5}\b", re.I),
    "ticket_id": re.compile(r"\bTKT[- ]?\d{5,7}\b", re.I),
    "po_number": re.compile(r"\bPO[- ]?\d{3,5}\b", re.I),
    "date": re.compile(r"\b20\d{2}-\d{2}-\d{2}\b"),
    "amount": re.compile(r"\b\d{1,6}\.\d{2}\b"),
    "term_months": re.compile(r"\b(?:6|12|24|36)\b"),
    "contract_value": re.compile(r"\b\d{4,7}(?:\.\d{2})?\b"),
    "priority": re.compile(r"\b(?:low|medium|high|critical)\b", re.I),
    "category": re.compile(r"\b(?:billing|access|bug|feature|infra|security)\b", re.I),
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9._\-]*", text)


def candidate_windows(text: str, window: int = 5) -> list[tuple[str, int, int]]:
    """Return (snippet, start_char, end_char) windows around tokens."""
    spans: list[tuple[str, int, int]] = []
    for m in re.finditer(r"\S+", text):
        start = max(0, m.start() - 40)
        end = min(len(text), m.end() + 40)
        spans.append((text[start:end], start, end))
    # dedupe roughly
    seen = set()
    out = []
    for s, a, b in spans:
        key = (a // 20, b // 20)
        if key in seen:
            continue
        seen.add(key)
        out.append((s, a, b))
    return out


_hasher = HashingVectorizer(
    n_features=2**10,
    alternate_sign=False,
    norm="l2",
    lowercase=True,
    ngram_range=(1, 2),
    analyzer="char_wb",
)


def hash_embed(texts: Iterable[str]) -> np.ndarray:
    return _hasher.transform(list(texts)).toarray().astype(np.float64)
