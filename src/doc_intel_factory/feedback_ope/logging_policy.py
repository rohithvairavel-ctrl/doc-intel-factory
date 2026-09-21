"""Logging policy: which documents get human review."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..extraction.hybrid import HybridExtractor
from ..synthetic.corpus import Document
from ..extraction.metrics import field_f1
from ..synthetic.corpus import PRIMARY_FIELDS


@dataclass
class ReviewDecision:
    doc_id: str
    action: int  # 1 = send to human review, 0 = auto-accept
    propensity: float
    reward: float  # observed only if action==1 in logging; we store oracle for OPE sim
    uncertainty: float


class ReviewLogger:
    """Simulate a logging policy that routes uncertain docs to humans.

    Reward = extraction quality improvement from review (oracle: 1 if auto would err).
    """

    def __init__(self, extractor: HybridExtractor, *, temperature: float = 1.5, seed: int = 0):
        self.extractor = extractor
        self.temperature = temperature
        self.seed = seed

    def _propensity(self, uncertainty: float) -> float:
        # logistic in uncertainty
        z = (uncertainty - 0.35) / max(self.temperature, 1e-6)
        p = 1.0 / (1.0 + np.exp(-4.0 * z))
        return float(np.clip(p, 0.05, 0.95))

    def _oracle_reward(self, doc: Document, pred: dict) -> float:
        fields = PRIMARY_FIELDS.get(doc.doc_type, [])
        clean = {k: v for k, v in pred.items() if not k.startswith("_")}
        m = field_f1([clean], [doc.fields], fields)
        # reward for reviewing: higher when auto F1 is low (human fixes errors)
        return float(1.0 - m["f1"])

    def log(self, docs: Sequence[Document]) -> list[ReviewDecision]:
        rng = np.random.default_rng(self.seed)
        decisions = []
        for doc in docs:
            pred = self.extractor.predict_one(doc.text, doc.doc_type)
            unc = self.extractor.predict_proba_doc(doc)
            p = self._propensity(unc)
            action = 1 if rng.random() < p else 0
            reward = self._oracle_reward(doc, pred)
            decisions.append(
                ReviewDecision(
                    doc_id=doc.doc_id,
                    action=action,
                    propensity=p,
                    reward=reward,
                    uncertainty=unc,
                )
            )
        return decisions


def target_policy_propensity(uncertainty: float, *, threshold: float = 0.4) -> float:
    """Deterministic-ish target: review if uncertainty >= threshold (soft)."""
    return 0.9 if uncertainty >= threshold else 0.1
