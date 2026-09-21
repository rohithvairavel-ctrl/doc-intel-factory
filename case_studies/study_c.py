"""Case C — AL budget curve + feedback OPE."""
from __future__ import annotations

import json
from pathlib import Path

from doc_intel_factory.active_learning.loop import ActiveLearningLoop
from doc_intel_factory.extraction.hybrid import HybridExtractor
from doc_intel_factory.feedback_ope.ips import ope_report
from doc_intel_factory.feedback_ope.logging_policy import ReviewLogger
from doc_intel_factory.synthetic.corpus import generate_corpus


def main() -> None:
    docs = generate_corpus(220, seed=42)
    train, test = docs[:160], docs[160:]
    curve = ActiveLearningLoop(train, test, seed=42, seed_size=20, batch_size=10, budget=70).run()
    ext = HybridExtractor(seed=42).fit(train)
    ope = ope_report(ReviewLogger(ext, seed=42).log(test))
    out = {"al_curve": ActiveLearningLoop.curve_to_dicts(curve), "ope": ope}
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/case_c.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(
        f"Case C final AL F1 u={curve[-1].f1_uncertainty:.3f} "
        f"OPE IPS err={ope['ips_abs_error']:.4f} naive err={ope['naive_abs_error']:.4f}"
    )


if __name__ == "__main__":
    main()
