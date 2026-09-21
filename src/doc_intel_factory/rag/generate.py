"""Template-based grounded answer generation (no hosted LLM)."""

from __future__ import annotations

from dataclasses import dataclass

from .retrieve import Hit


@dataclass
class Answer:
    question: str
    answer: str
    supporting_spans: list[str]
    retrieved_scores: list[float]


def grounded_answer(question: str, hits: list[Hit], *, max_spans: int = 3) -> Answer:
    """Produce an answer that only quotes / lightly templates retrieved spans.

    This is intentionally not a free-form LLM — answers are grounded by construction.
    """
    if not hits:
        return Answer(
            question=question,
            answer="I cannot find supporting evidence in the indexed documents.",
            supporting_spans=[],
            retrieved_scores=[],
        )
    spans = [h.chunk.text.strip() for h in hits[:max_spans]]
    scores = [h.score for h in hits[:max_spans]]
    # Template: extract likely key=value or first sentence-ish
    evidence = " | ".join(s.replace("\n", " ")[:160] for s in spans)
    answer = (
        f"Based on retrieved document spans: {evidence}. "
        f"(query: {question})"
    )
    return Answer(
        question=question,
        answer=answer,
        supporting_spans=spans,
        retrieved_scores=scores,
    )


def make_synthetic_qa(docs, rng, *, n_per_type: int = 8) -> list[dict]:
    """Build gold QA pairs from document fields for RAG eval."""
    qa = []
    templates = {
        "invoice": [
            ("What is the invoice id?", "invoice_id"),
            ("Who is the vendor?", "vendor"),
            ("What is the total amount?", "amount"),
            ("What is the invoice date?", "date"),
        ],
        "contract": [
            ("What is the contract id?", "contract_id"),
            ("Who are the parties?", "parties"),
            ("What is the effective date?", "effective_date"),
            ("What is the contract value?", "contract_value"),
        ],
        "ticket": [
            ("What is the ticket id?", "ticket_id"),
            ("What is the priority?", "priority"),
            ("What is the category?", "category"),
            ("Who is the assignee?", "assignee"),
        ],
    }
    by_type: dict[str, list] = {}
    for d in docs:
        by_type.setdefault(d.doc_type, []).append(d)
    for dtype, pairs in templates.items():
        pool = by_type.get(dtype, [])
        if not pool:
            continue
        chosen = list(rng.choice(pool, size=min(n_per_type, len(pool)), replace=False))
        for doc in chosen:
            q_tmpl, field = pairs[int(rng.integers(0, len(pairs)))]
            gold = str(doc.fields.get(field, ""))
            qa.append(
                {
                    "doc_id": doc.doc_id,
                    "doc_type": dtype,
                    "question": q_tmpl,
                    "gold_answer": gold,
                    "gold_field": field,
                    "clean_context": doc.clean_text,
                }
            )
    return qa
