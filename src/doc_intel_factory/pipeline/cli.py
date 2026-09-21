"""CLI entrypoint: doc-intel-factory run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .report import run_factory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="doc-intel-factory",
        description="Run the document-intelligence factory pipeline (corpus → IE → RAG → AL → OPE).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Execute full pipeline and write artifacts/factory_report.md")
    run_p.add_argument("--n-docs", type=int, default=300)
    run_p.add_argument("--seed", type=int, default=42)
    run_p.add_argument("--artifacts", type=str, default="artifacts")

    args = parser.parse_args(argv)
    if args.cmd == "run":
        results = run_factory(n_docs=args.n_docs, seed=args.seed, artifacts_dir=args.artifacts)
        out = Path(args.artifacts) / "factory_report.md"
        print(f"Wrote {out}")
        print(f"Case A F1 lift: {results['case_a_extraction']['f1_lift']:.3f}")
        print(f"Case B faithfulness: {results['case_b_rag']['faithfulness']:.3f}")
        print(
            f"Case C OPE IPS abs err: {results['case_c_ope']['ips_abs_error']:.4f} "
            f"vs naive {results['case_c_ope']['naive_abs_error']:.4f}"
        )
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
