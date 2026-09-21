from doc_intel_factory.synthetic.corpus import generate_corpus, corpus_stats, PRIMARY_FIELDS
from doc_intel_factory.synthetic.ocr_noise import apply_ocr_noise
import numpy as np


def test_corpus_shape_and_types():
    docs = generate_corpus(60, seed=1, char_error_rate=0.05)
    assert len(docs) == 60
    types = {d.doc_type for d in docs}
    assert types == {"invoice", "contract", "ticket"}
    for d in docs:
        assert d.doc_id
        assert d.text
        assert d.clean_text
        assert d.fields
        for key in PRIMARY_FIELDS[d.doc_type]:
            assert key in d.fields
    stats = corpus_stats(docs)
    assert stats["n"] == 60
    assert sum(stats["by_type"].values()) == 60


def test_ocr_noise_changes_text():
    rng = np.random.default_rng(0)
    clean = "Invoice INV-12345 from Acme Supplies Ltd dated 2024-01-15 amount 1234.56"
    noisy = apply_ocr_noise(clean, rng, char_error_rate=0.2)
    assert isinstance(noisy, str)
    assert len(noisy) > 0
    # high CER should usually alter text
    assert noisy != clean or apply_ocr_noise(clean, np.random.default_rng(99), char_error_rate=0.5) != clean
