"""Case A — Extraction lift."""
from __future__ import annotations

import json
from pathlib import Path

from doc_intel_factory.extraction.hybrid import HybridExtractor
from doc_intel_factory.extraction.metrics import extraction_report
from doc_intel_factory.synthetic.corpus import PRIMARY_FIELDS, generate_corpus


def main() -> None:
    docs = generate_corpus(200, seed=42)
    train, test = docs[:140], docs[140:]
    fields = sorted({k for d in test for k in PRIMARY_FIELDS[d.doc_type]})
    golds = [d.fields for d in test]
    rules = [HybridExtractor().rule_extract(d.text, d.doc_type) for d in test]
    hybrid = HybridExtractor(seed=42).fit(train)
    preds = [{k: v for k, v in p.items() if not k.startswith("_")} for p in hybrid.predict(test)]
    out = {
        "rules": extraction_report(rules, golds, fields),
        "hybrid": extraction_report(preds, golds, fields),
    }
    out["f1_lift"] = out["hybrid"]["micro"]["f1"] - out["rules"]["micro"]["f1"]
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/case_a.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Case A F1 lift: {out['f1_lift']:.3f}")


if __name__ == "__main__":
    main()
