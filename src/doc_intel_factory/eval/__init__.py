"""RAG evaluation: faithfulness, relevance, groundedness."""

from .harness import evaluate_rag, RagEvalResult

__all__ = ["evaluate_rag", "RagEvalResult"]
