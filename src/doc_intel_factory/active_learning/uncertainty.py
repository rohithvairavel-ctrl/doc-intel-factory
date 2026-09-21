"""Uncertainty sampling utilities."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..extraction.hybrid import HybridExtractor
from ..synthetic.corpus import Document


def uncertainty_scores(extractor: HybridExtractor, docs: Sequence[Document]) -> np.ndarray:
    return np.array([extractor.predict_proba_doc(d) for d in docs], dtype=float)


def select_uncertain(
    extractor: HybridExtractor,
    pool: Sequence[Document],
    *,
    batch_size: int,
    labeled_ids: set[str],
) -> list[Document]:
    remaining = [d for d in pool if d.doc_id not in labeled_ids]
    if not remaining:
        return []
    scores = uncertainty_scores(extractor, remaining)
    order = np.argsort(-scores)
    chosen = [remaining[int(i)] for i in order[:batch_size]]
    return chosen


def select_random(
    pool: Sequence[Document],
    *,
    batch_size: int,
    labeled_ids: set[str],
    rng: np.random.Generator,
) -> list[Document]:
    remaining = [d for d in pool if d.doc_id not in labeled_ids]
    if not remaining:
        return []
    k = min(batch_size, len(remaining))
    idx = rng.choice(len(remaining), size=k, replace=False)
    return [remaining[int(i)] for i in idx]
