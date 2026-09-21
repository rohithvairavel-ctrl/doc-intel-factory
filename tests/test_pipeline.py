from pathlib import Path
from doc_intel_factory.pipeline.report import run_factory


def test_run_factory_writes_report(tmp_path: Path):
    results = run_factory(n_docs=100, seed=21, artifacts_dir=tmp_path)
    assert (tmp_path / "factory_report.md").exists()
    assert (tmp_path / "factory_results.json").exists()
    assert "case_a_extraction" in results
    assert "case_b_rag" in results
    assert "case_c_ope" in results
    text = (tmp_path / "factory_report.md").read_text(encoding="utf-8")
    assert "Case A" in text and "Case B" in text and "Case C" in text
