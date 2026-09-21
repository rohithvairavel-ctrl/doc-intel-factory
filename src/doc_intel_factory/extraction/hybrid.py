"""Rule + ML hybrid extractor for key document fields."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from ..synthetic.corpus import PRIMARY_FIELDS, Document
from .features import PATTERNS, candidate_windows, hash_embed, tokenize


class HybridExtractor:
    """Combine regex/rule candidates with a logistic scorer over char n-grams.

    Training uses distant supervision from gold field values present in noisy text.
    At inference, candidates are proposed by rules + windows; ML ranks them.
    """

    def __init__(self, *, seed: int = 42):
        self.seed = seed
        self.models: dict[str, LogisticRegression] = {}
        self.label_encoders: dict[str, LabelEncoder] = {}
        self._fitted_fields: list[str] = []
        self._doc_type_priors: dict[str, float] = {}

    # ----- rules ----------------------------------------------------------
    def rule_extract(self, text: str, doc_type: str | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {}
        keys = PRIMARY_FIELDS.get(doc_type, []) if doc_type else list(PATTERNS)
        # also try type-agnostic patterns
        for name, pat in PATTERNS.items():
            if keys and name not in keys and name not in ("date", "amount", "priority", "category"):
                # still allow shared patterns
                if name not in PRIMARY_FIELDS.get(doc_type or "", []):
                    continue
            m = pat.search(text)
            if m:
                out[name] = m.group(0)
        # vendor / parties / assignee / summary via cue words
        lower = text.lower()
        for cue, field in [
            ("vendor:", "vendor"),
            ("bill from", "vendor"),
            ("parties:", "parties"),
            ("counterparty", "parties"),
            ("assignee", "assignee"),
            ("owner:", "assignee"),
            ("summary", "summary"),
            ("description:", "summary"),
        ]:
            if field in out:
                continue
            if keys and field not in keys:
                continue
            idx = lower.find(cue)
            if idx >= 0:
                snippet = text[idx + len(cue) : idx + len(cue) + 60].strip()
                # take until newline or pipe
                for sep in ["\n", "|", "  "]:
                    if sep in snippet:
                        snippet = snippet.split(sep)[0].strip()
                snippet = snippet.strip(" .:")
                if snippet:
                    out[field] = snippet[:80]
        return out

    # ----- ML training ----------------------------------------------------
    def fit(self, docs: list[Document], fields: list[str] | None = None) -> "HybridExtractor":
        by_field: dict[str, list[tuple[str, int]]] = {f: [] for f in (fields or [])}
        if not fields:
            # union of primary fields seen
            fields = sorted({k for d in docs for k in PRIMARY_FIELDS.get(d.doc_type, [])})
            by_field = {f: [] for f in fields}

        for doc in docs:
            gold = doc.fields
            windows = candidate_windows(doc.text)
            for fname in fields:
                gval = gold.get(fname)
                if gval is None:
                    continue
                gnorm = str(gval).lower()
                for snippet, _, _ in windows:
                    label = 1 if gnorm in snippet.lower() or snippet.lower() in gnorm else 0
                    # also positive if pattern matches gold shape
                    if label == 0 and fname in PATTERNS and PATTERNS[fname].search(snippet or ""):
                        # weak negative unless value-ish match
                        pass
                    by_field[fname].append((snippet, label))
                # ensure at least one positive from gold string itself
                by_field[fname].append((str(gval), 1))

        rng = np.random.default_rng(self.seed)
        self._fitted_fields = []
        for fname, pairs in by_field.items():
            if not pairs:
                continue
            texts = [p[0] for p in pairs]
            labels = np.array([p[1] for p in pairs], dtype=int)
            # subsample negatives if imbalanced
            pos = np.where(labels == 1)[0]
            neg = np.where(labels == 0)[0]
            if len(pos) == 0:
                continue
            if len(neg) > max(50, 3 * len(pos)):
                keep_neg = rng.choice(neg, size=max(50, 3 * len(pos)), replace=False)
                idx = np.concatenate([pos, keep_neg])
                texts = [texts[i] for i in idx]
                labels = labels[idx]
            X = hash_embed(texts)
            if labels.sum() == 0 or labels.sum() == len(labels):
                continue
            clf = LogisticRegression(
                max_iter=400,
                class_weight="balanced",
                random_state=self.seed,
                solver="lbfgs",
            )
            clf.fit(X, labels)
            self.models[fname] = clf
            self._fitted_fields.append(fname)
        return self

    def _ml_score_candidates(self, text: str, fname: str) -> list[tuple[str, float]]:
        if fname not in self.models:
            return []
        cands: list[str] = []
        if fname in PATTERNS:
            cands.extend(m.group(0) for m in PATTERNS[fname].finditer(text))
        for snippet, _, _ in candidate_windows(text):
            # extract token-ish center
            toks = tokenize(snippet)
            if toks:
                cands.append(max(toks, key=len) if fname.endswith("_id") else " ".join(toks[:6]))
        if not cands:
            return []
        # unique preserve order
        seen = set()
        uniq = []
        for c in cands:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        X = hash_embed(uniq)
        proba = self.models[fname].predict_proba(X)
        # positive class column
        pos_idx = list(self.models[fname].classes_).index(1) if 1 in self.models[fname].classes_ else -1
        scores = proba[:, pos_idx] if pos_idx >= 0 else proba[:, -1]
        return sorted(zip(uniq, scores.tolist()), key=lambda x: -x[1])

    def predict_one(self, text: str, doc_type: str | None = None) -> dict[str, Any]:
        rules = self.rule_extract(text, doc_type)
        fields = PRIMARY_FIELDS.get(doc_type, self._fitted_fields) if doc_type else self._fitted_fields
        out: dict[str, Any] = {}
        conf: dict[str, float] = {}
        for fname in fields:
            ranked = self._ml_score_candidates(text, fname)
            if ranked and ranked[0][1] >= 0.45:
                out[fname] = ranked[0][0]
                conf[fname] = float(ranked[0][1])
            elif fname in rules:
                out[fname] = rules[fname]
                conf[fname] = 0.55
            elif ranked:
                out[fname] = ranked[0][0]
                conf[fname] = float(ranked[0][1])
        out["_confidence"] = conf
        return out

    def predict(self, docs: list[Document]) -> list[dict[str, Any]]:
        return [self.predict_one(d.text, d.doc_type) for d in docs]

    def predict_proba_doc(self, doc: Document) -> float:
        """Document-level uncertainty proxy: mean (1 - max field confidence)."""
        pred = self.predict_one(doc.text, doc.doc_type)
        confs = pred.get("_confidence", {})
        if not confs:
            return 1.0
        return float(1.0 - np.mean(list(confs.values())))
