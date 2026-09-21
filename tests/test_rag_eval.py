from doc_intel_factory.synthetic.corpus import generate_corpus
from doc_intel_factory.rag.chunking import chunk_documents
from doc_intel_factory.rag.retrieve import Retriever
from doc_intel_factory.rag.generate import grounded_answer
from doc_intel_factory.eval.harness import evaluate_rag
from doc_intel_factory.eval.faithfulness import faithfulness


def test_retrieve_and_grounded_answer():
    docs = generate_corpus(40, seed=3)
    chunks = chunk_documents(docs)
    retriever = Retriever().index(chunks)
    hits = retriever.search("invoice vendor amount", k=3)
    assert len(hits) <= 3
    ans = grounded_answer("What is the vendor?", hits)
    assert ans.answer
    assert faithfulness(ans.answer, ans.supporting_spans) >= 0.0


def test_rag_eval_harness():
    docs = generate_corpus(80, seed=5)
    chunks = chunk_documents(docs)
    retriever = Retriever().index(chunks)
    result = evaluate_rag(retriever, docs, seed=5, n_per_type=5, k=5)
    assert result.n > 0
    assert 0 <= result.faithfulness <= 1.01
    assert 0 <= result.hit_at_5 <= 1.01
