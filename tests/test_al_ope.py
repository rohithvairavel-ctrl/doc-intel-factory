from doc_intel_factory.synthetic.corpus import generate_corpus
from doc_intel_factory.extraction.hybrid import HybridExtractor
from doc_intel_factory.active_learning.loop import ActiveLearningLoop
from doc_intel_factory.feedback_ope.logging_policy import ReviewLogger
from doc_intel_factory.feedback_ope.ips import ope_report, ips_estimate, snips_estimate


def test_active_learning_curve_monotone_ish():
    docs = generate_corpus(150, seed=11, char_error_rate=0.04)
    train, test = docs[:100], docs[100:]
    loop = ActiveLearningLoop(train, test, seed=11, seed_size=15, batch_size=10, budget=45)
    curve = loop.run()
    assert len(curve) >= 2
    assert curve[0].labeled < curve[-1].labeled
    # final uncertainty F1 should be positive
    assert curve[-1].f1_uncertainty > 0.2


def test_ope_estimators():
    docs = generate_corpus(80, seed=13)
    ext = HybridExtractor(seed=13).fit(docs[:50])
    decisions = ReviewLogger(ext, seed=13).log(docs[50:])
    assert len(decisions) == 30
    report = ope_report(decisions, threshold=0.4)
    assert "ips" in report and "snips" in report and "naive" in report
    assert report["ips"]["n"] == 30
    ips = ips_estimate(decisions)
    snips = snips_estimate(decisions)
    assert isinstance(ips["estimate"], float)
    assert isinstance(snips["estimate"], float)
