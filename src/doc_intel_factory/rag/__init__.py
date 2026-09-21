"""Lightweight RAG: chunking, TF-IDF / hashing embeddings, retrieval, grounded answers."""

from .chunking import chunk_documents
from .embeddings import TfidfEmbedder
from .retrieve import Retriever
from .generate import grounded_answer

__all__ = ["chunk_documents", "TfidfEmbedder", "Retriever", "grounded_answer"]
