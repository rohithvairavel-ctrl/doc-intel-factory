"""Case B — RAG eval."""
from __future__ import annotations

import json
from pathlib import Path

from doc_intel_factory.eval.harness import evaluate_rag
from doc_intel_factory.rag.chunking import chunk_documents
from doc_intel_factory.rag.retrieve import Retriever
from doc_intel_factory.synthetic.corpus import generate_corpus


def main() -> None:
    docs = generate_corpus(200, seed=42)
    retriever = Retriever().index(chunk_documents(docs))
    result = evaluate_rag(retriever, docs, seed=42, n_per_type=12)
    out = result.to_dict()
    out.pop("details", None)
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/case_b.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Case B faithfulness={result.faithfulness:.3f} hit@5={result.hit_at_5:.3f}")


if __name__ == "__main__":
    main()
