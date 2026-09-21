"""Active learning loop: seed → uncertainty sample → retrain → evaluate."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from ..extraction.hybrid import HybridExtractor
from ..extraction.metrics import field_f1
from ..synthetic.corpus import PRIMARY_FIELDS, Document
from .uncertainty import select_random, select_uncertain


@dataclass
class ALCurvePoint:
    labeled: int
    f1_uncertainty: float
    f1_random: float
    precision_u: float
    recall_u: float


class ActiveLearningLoop:
    def __init__(
        self,
        train_pool: list[Document],
        test_docs: list[Document],
        *,
        seed: int = 42,
        seed_size: int = 20,
        batch_size: int = 10,
        budget: int = 80,
    ):
        self.train_pool = train_pool
        self.test_docs = test_docs
        self.seed = seed
        self.seed_size = seed_size
        self.batch_size = batch_size
        self.budget = budget

    def _fields(self) -> list[str]:
        return sorted({k for d in self.test_docs for k in PRIMARY_FIELDS.get(d.doc_type, [])})

    def _eval(self, extractor: HybridExtractor) -> dict[str, float]:
        preds = extractor.predict(self.test_docs)
        # strip meta
        clean = [{k: v for k, v in p.items() if not k.startswith("_")} for p in preds]
        golds = [d.fields for d in self.test_docs]
        return field_f1(clean, golds, self._fields())

    def run(self) -> list[ALCurvePoint]:
        rng = np.random.default_rng(self.seed)
        # shared seed set
        seed_idx = rng.choice(len(self.train_pool), size=min(self.seed_size, len(self.train_pool)), replace=False)
        seed_docs = [self.train_pool[int(i)] for i in seed_idx]
        seed_ids = {d.doc_id for d in seed_docs}

        labeled_u = list(seed_docs)
        labeled_r = list(seed_docs)
        ids_u = set(seed_ids)
        ids_r = set(seed_ids)

        curve: list[ALCurvePoint] = []
        labeled_count = len(seed_docs)

        while labeled_count <= self.budget:
            ext_u = HybridExtractor(seed=self.seed).fit(labeled_u)
            ext_r = HybridExtractor(seed=self.seed + 1).fit(labeled_r)
            m_u = self._eval(ext_u)
            m_r = self._eval(ext_r)
            curve.append(
                ALCurvePoint(
                    labeled=labeled_count,
                    f1_uncertainty=m_u["f1"],
                    f1_random=m_r["f1"],
                    precision_u=m_u["precision"],
                    recall_u=m_u["recall"],
                )
            )
            if labeled_count >= self.budget:
                break
            batch = min(self.batch_size, self.budget - labeled_count)
            new_u = select_uncertain(ext_u, self.train_pool, batch_size=batch, labeled_ids=ids_u)
            new_r = select_random(self.train_pool, batch_size=batch, labeled_ids=ids_r, rng=rng)
            if not new_u and not new_r:
                break
            for d in new_u:
                labeled_u.append(d)
                ids_u.add(d.doc_id)
            for d in new_r:
                labeled_r.append(d)
                ids_r.add(d.doc_id)
            labeled_count = max(len(labeled_u), len(labeled_r))
            # align count reporting to uncertainty path length
            labeled_count = len(labeled_u)

        return curve

    @staticmethod
    def curve_to_dicts(curve: list[ALCurvePoint]) -> list[dict[str, Any]]:
        return [asdict(p) for p in curve]
