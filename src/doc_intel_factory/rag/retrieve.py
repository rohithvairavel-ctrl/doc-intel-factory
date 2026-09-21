"""Cosine retrieval over embedded chunks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chunking import Chunk
from .embeddings import TfidfEmbedder


@dataclass
class Hit:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(self, embedder: TfidfEmbedder | None = None):
        self.embedder = embedder or TfidfEmbedder()
        self.chunks: list[Chunk] = []
        self.matrix: np.ndarray | None = None

    def index(self, chunks: list[Chunk]) -> "Retriever":
        self.chunks = list(chunks)
        texts = [c.text for c in self.chunks]
        if not texts:
            self.matrix = np.zeros((0, 1))
            return self
        self.matrix = self.embedder.fit_transform(texts)
        # store dense for simple matmul (small corpora)
        if hasattr(self.matrix, "toarray"):
            self.matrix = self.matrix.toarray()
        return self

    def search(self, query: str, *, k: int = 5, doc_type: str | None = None) -> list[Hit]:
        if not self.chunks or self.matrix is None or self.matrix.shape[0] == 0:
            return []
        q = self.embedder.transform([query])
        if hasattr(q, "toarray"):
            q = q.toarray()
        scores = (self.matrix @ q.T).ravel()
        idxs = np.argsort(-scores)
        hits: list[Hit] = []
        for i in idxs:
            ch = self.chunks[int(i)]
            if doc_type and ch.doc_type != doc_type:
                continue
            hits.append(Hit(chunk=ch, score=float(scores[int(i)])))
            if len(hits) >= k:
                break
        return hits
