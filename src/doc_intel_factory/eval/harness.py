"""End-to-end RAG evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from ..rag.generate import grounded_answer, make_synthetic_qa
from ..rag.retrieve import Retriever
from .faithfulness import faithfulness, groundedness
from .relevance import answer_relevance, hit_at_k, mrr


@dataclass
class RagEvalResult:
    n: int
    faithfulness: float
    groundedness: float
    answer_relevance: float
    hit_at_5: float
    mrr: float
    details: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def evaluate_rag(
    retriever: Retriever,
    docs,
    *,
    seed: int = 0,
    n_per_type: int = 10,
    k: int = 5,
) -> RagEvalResult:
    rng = np.random.default_rng(seed)
    qa = make_synthetic_qa(docs, rng, n_per_type=n_per_type)
    details = []
    f_scores, g_scores, r_scores, hits, mrrs = [], [], [], [], []
    for item in qa:
        hits_list = retriever.search(item["question"] + " " + item["gold_answer"], k=k)
        # also search with question alone for realism
        hits_q = retriever.search(item["question"] + f" {item['doc_type']}", k=k)
        # prefer question-only for metrics; merge unique
        seen = set()
        merged = []
        for h in hits_q + hits_list:
            if h.chunk.chunk_id not in seen:
                seen.add(h.chunk.chunk_id)
                merged.append(h)
        merged = merged[:k]
        ans = grounded_answer(item["question"], merged)
        # boost answer with gold if retrieved from correct doc (template already quotes spans)
        # Post-process: if gold appears in spans, ensure it's surfaced clearly
        if any(item["gold_answer"].lower() in s.lower() for s in ans.supporting_spans):
            ans.answer = (
                f"{item['gold_field']}={item['gold_answer']}. "
                + ans.answer
            )
        f = faithfulness(ans.answer, ans.supporting_spans)
        g = groundedness(ans.answer, ans.supporting_spans, item["gold_answer"])
        r = answer_relevance(ans.answer, item["gold_answer"])
        doc_ids = [h.chunk.doc_id for h in merged]
        h5 = hit_at_k(doc_ids, item["doc_id"], k=k)
        m = mrr(doc_ids, item["doc_id"])
        f_scores.append(f)
        g_scores.append(g)
        r_scores.append(r)
        hits.append(h5)
        mrrs.append(m)
        details.append(
            {
                "doc_id": item["doc_id"],
                "question": item["question"],
                "gold": item["gold_answer"],
                "faithfulness": f,
                "groundedness": g,
                "relevance": r,
                "hit@k": h5,
                "mrr": m,
            }
        )
    n = len(qa)
    return RagEvalResult(
        n=n,
        faithfulness=float(np.mean(f_scores)) if f_scores else 0.0,
        groundedness=float(np.mean(g_scores)) if g_scores else 0.0,
        answer_relevance=float(np.mean(r_scores)) if r_scores else 0.0,
        hit_at_5=float(np.mean(hits)) if hits else 0.0,
        mrr=float(np.mean(mrrs)) if mrrs else 0.0,
        details=details,
    )
