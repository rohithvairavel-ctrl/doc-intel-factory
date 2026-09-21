"""Synthetic multi-type document corpus with gold field annotations."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

from .layouts import render_layout
from .ocr_noise import apply_ocr_noise, normalize_for_match

VENDORS = [
    "Acme Supplies Ltd",
    "Northwind Traders",
    "Globex Corp",
    "Initech Services",
    "Umbrella Logistics",
    "Stark Industries",
    "Wayne Enterprises",
    "Oscorp Materials",
]
PARTIES = [
    "Acme Corp and Beta LLC",
    "Northwind and Contoso Inc",
    "Globex and Initech",
    "Stark and Wayne Enterprises",
    "Umbrella and Oscorp",
]
ASSIGNEES = ["alice", "bob", "carol", "dave", "eve", "frank"]
CATEGORIES = ["billing", "access", "bug", "feature", "infra", "security"]
PRIORITIES = ["low", "medium", "high", "critical"]
SUMMARIES = [
    "Cannot reset password after SSO migration",
    "Invoice PDF fails to download for Q3",
    "Dashboard latency spikes during peak hours",
    "Missing RBAC role for finance analysts",
    "API rate limit exceeded on batch export",
    "Duplicate charge on subscription renewal",
]


@dataclass
class FieldSpan:
    """Gold field with character offsets into clean text (pre-noise)."""

    name: str
    value: str
    start: int
    end: int


@dataclass
class Document:
    doc_id: str
    doc_type: str  # invoice | contract | ticket
    layout_variant: str
    clean_text: str
    text: str  # noisy / OCR-simulated
    fields: dict[str, Any]
    spans: list[FieldSpan] = field(default_factory=list)
    char_error_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def _date(rng: np.random.Generator) -> str:
    y = int(rng.integers(2022, 2027))
    m = int(rng.integers(1, 13))
    d = int(rng.integers(1, 28))
    return f"{y:04d}-{m:02d}-{d:02d}"


def _make_invoice_fields(rng: np.random.Generator) -> dict[str, Any]:
    return {
        "invoice_id": f"INV-{int(rng.integers(10000, 99999))}",
        "vendor": str(rng.choice(VENDORS)),
        "amount": round(float(rng.uniform(50, 25000)), 2),
        "date": _date(rng),
        "currency": str(rng.choice(["USD", "EUR", "GBP"])),
        "po_number": f"PO-{int(rng.integers(1000, 9999))}",
    }


def _make_contract_fields(rng: np.random.Generator) -> dict[str, Any]:
    return {
        "contract_id": f"CTR-{int(rng.integers(1000, 9999))}",
        "parties": str(rng.choice(PARTIES)),
        "effective_date": _date(rng),
        "term_months": int(rng.choice([6, 12, 24, 36])),
        "contract_value": round(float(rng.uniform(10_000, 500_000)), 2),
        "jurisdiction": str(rng.choice(["Delaware", "California", "New York", "Texas"])),
    }


def _make_ticket_fields(rng: np.random.Generator) -> dict[str, Any]:
    return {
        "ticket_id": f"TKT-{int(rng.integers(100000, 999999))}",
        "priority": str(rng.choice(PRIORITIES)),
        "category": str(rng.choice(CATEGORIES)),
        "assignee": str(rng.choice(ASSIGNEES)),
        "status": str(rng.choice(["open", "pending", "resolved"])),
        "summary": str(rng.choice(SUMMARIES)),
    }


_FIELD_BUILDERS = {
    "invoice": _make_invoice_fields,
    "contract": _make_contract_fields,
    "ticket": _make_ticket_fields,
}

# Primary extractable keys per type (used by IE + AL)
PRIMARY_FIELDS: dict[str, list[str]] = {
    "invoice": ["invoice_id", "vendor", "amount", "date", "po_number"],
    "contract": ["contract_id", "parties", "effective_date", "term_months", "contract_value"],
    "ticket": ["ticket_id", "priority", "category", "assignee", "summary"],
}


def _locate_spans(clean_text: str, fields: dict[str, Any], keys: list[str]) -> list[FieldSpan]:
    spans: list[FieldSpan] = []
    lower = clean_text.lower()
    for name in keys:
        val = fields.get(name)
        if val is None:
            continue
        sval = str(val)
        idx = lower.find(sval.lower())
        if idx < 0:
            # amount may be formatted differently — try without trailing zeros
            if isinstance(val, float):
                alt = f"{val:.0f}" if val == int(val) else sval
                idx = lower.find(alt.lower())
                sval = alt if idx >= 0 else sval
        if idx >= 0:
            spans.append(FieldSpan(name=name, value=sval, start=idx, end=idx + len(sval)))
    return spans


def generate_corpus(
    n: int = 200,
    *,
    seed: int = 42,
    doc_types: list[str] | None = None,
    char_error_rate: float = 0.04,
    type_weights: dict[str, float] | None = None,
) -> list[Document]:
    """Generate a synthetic labeled corpus.

    Layout variants: compact | table | narrative.
    OCR noise applied post-render.
    """
    rng = np.random.default_rng(seed)
    types = doc_types or ["invoice", "contract", "ticket"]
    if type_weights:
        weights = np.array([type_weights.get(t, 1.0) for t in types], dtype=float)
        weights /= weights.sum()
    else:
        weights = np.ones(len(types)) / len(types)

    variants = ["compact", "table", "narrative"]
    docs: list[Document] = []
    for i in range(n):
        dtype = str(rng.choice(types, p=weights))
        variant = str(rng.choice(variants))
        fields = _FIELD_BUILDERS[dtype](rng)
        # stringify numeric fields consistently for span finding
        render_fields = {k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in fields.items()}
        # keep original typed values in fields for metrics
        clean = render_layout(dtype, render_fields, variant, rng)
        spans = _locate_spans(clean, render_fields, PRIMARY_FIELDS[dtype])
        noisy = apply_ocr_noise(clean, rng, char_error_rate=char_error_rate)
        docs.append(
            Document(
                doc_id=f"{dtype[:3].upper()}-{i:05d}",
                doc_type=dtype,
                layout_variant=variant,
                clean_text=clean,
                text=noisy,
                fields={k: render_fields[k] for k in render_fields},
                spans=spans,
                char_error_rate=char_error_rate,
            )
        )
    return docs


def corpus_stats(docs: list[Document]) -> dict[str, Any]:
    from collections import Counter

    return {
        "n": len(docs),
        "by_type": dict(Counter(d.doc_type for d in docs)),
        "by_layout": dict(Counter(d.layout_variant for d in docs)),
        "mean_chars": float(np.mean([len(d.text) for d in docs])) if docs else 0.0,
        "mean_spans": float(np.mean([len(d.spans) for d in docs])) if docs else 0.0,
    }
