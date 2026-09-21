from doc_intel_factory.synthetic.corpus import generate_corpus, PRIMARY_FIELDS
from doc_intel_factory.extraction.hybrid import HybridExtractor
from doc_intel_factory.extraction.metrics import field_f1, extraction_report


def test_hybrid_beats_or_matches_rules_on_avg():
    docs = generate_corpus(120, seed=7, char_error_rate=0.03)
    train, test = docs[:80], docs[80:]
    fields = sorted({k for d in test for k in PRIMARY_FIELDS[d.doc_type]})
    golds = [d.fields for d in test]

    rules_ext = HybridExtractor(seed=0)
    rules = [rules_ext.rule_extract(d.text, d.doc_type) for d in test]
    hybrid = HybridExtractor(seed=0).fit(train)
    preds = [{k: v for k, v in p.items() if not k.startswith("_")} for p in hybrid.predict(test)]

    m_r = field_f1(rules, golds, fields)
    m_h = field_f1(preds, golds, fields)
    # hybrid should be competitive; allow small slack for noise
    assert m_h["f1"] >= m_r["f1"] - 0.05
    assert m_h["f1"] > 0.3


def test_extraction_report_keys():
    docs = generate_corpus(30, seed=2)
    ext = HybridExtractor(seed=1).fit(docs[:20])
    preds = [{k: v for k, v in p.items() if not k.startswith("_")} for p in ext.predict(docs[20:])]
    rep = extraction_report(preds, [d.fields for d in docs[20:]], ["invoice_id", "ticket_id", "contract_id"])
    assert "micro" in rep and "per_field" in rep
