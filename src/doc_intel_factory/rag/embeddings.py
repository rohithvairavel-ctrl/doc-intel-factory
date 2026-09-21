"""TF-IDF embeddings for retrieval (lean, no heavy sentence-transformers)."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class TfidfEmbedder:
    def __init__(self, *, max_features: int = 4096, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=True,
            stop_words="english",
            sublinear_tf=True,
        )
        self._fitted = False

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        self.vectorizer.fit(texts)
        self._fitted = True
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder not fitted")
        X = self.vectorizer.transform(texts)
        return normalize(X, norm="l2", axis=1)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)
