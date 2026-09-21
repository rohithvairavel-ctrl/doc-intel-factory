# Interview one-pager — doc-intel-factory

**30-second pitch:** An enterprise document intelligence *factory*: synthetic noisy docs → hybrid extraction → grounded RAG with faithfulness eval → active learning on a label budget → off-policy evaluation of human-review policies.

**Repo:** https://github.com/rohithvairavel-ctrl/doc-intel-factory

---

## Problem

Document AI portfolios often stop at “I called an LLM.” Staff-level work shows **OCR noise**, **IE metrics**, **grounded generation eval**, **label efficiency**, and **review-policy OPE** as one loop.

## System (what I built)

1. **Synthetic corpus** — invoices / contracts / tickets with layout variants and OCR-like noise (no Tesseract required)  
2. **Hybrid extraction** — regex/cues + HashingVectorizer logistic scoring; P/R/F1  
3. **RAG** — chunk → TF-IDF → cosine retrieve → **template-grounded** answers (no API keys)  
4. **Eval harness** — faithfulness / groundedness / relevance; hit@5 / MRR  
5. **Active learning** — uncertainty vs random label budget curves  
6. **Feedback OPE** — logging which docs get “human review”; IPS/SNIPS vs naive  
7. **CLI pipeline** → `artifacts/factory_report.md` (+ optional Streamlit)

## Proof points

- Case A: hybrid IE F1 beats rules under OCR noise (example lift ~+0.05 F1)  
- Case B: high faithfulness on grounded answers with retrieval hit@5/MRR in report  
- Case C: IPS error vs oracle much smaller than naive policy evaluation in the demo seed  
- Tests: 9 seeded tests at ship

## Metrics I own

| Layer | Metric | Talking point |
|-------|--------|----------------|
| IE | P/R/F1 by field | Noise robustness > demo on clean text |
| RAG | Faithfulness, groundedness, relevance | Don’t trust fluent lies |
| Retrieval | Hit@k, MRR | Generation quality follows retrieval |
| AL | F1 vs label budget | Uncertainty sampling ROI |
| OPE | IPS/SNIPS vs naive | Review routing is a policy |

## Threats to validity

- Synthetic OCR ≠ scanners / LayoutLM distributions  
- Template answers inflate faithfulness vs free-form LLMs  
- OPE rewards are oracle-simulated review quality  
- TF-IDF is a lean baseline, not multilingual dense retrieval  
- Fuzzy field matching can be optimistic under heavy noise  

## 5-minute talk track

1. **(30s)** Factory metaphor: docs in → decisions out, with eval at every stage  
2. **(60s)** Architecture walk: synthetic → IE → RAG → AL → OPE → report  
3. **(90s)** Deep dive: why faithfulness/groundedness, and why template-grounded answers are an honest teaching choice  
4. **(60s)** Active learning budget curve + when random wins  
5. **(60s)** OPE for human review routing; limitations and production next steps (real OCR, dense retrievers, human study)

## Likely questions → answers

- **Why not GPT extraction?** Cost, privacy, and you still need metrics; hybrid IE is controllable and testable offline.  
- **Is high faithfulness cheating?** Templates bound answers to spans on purpose — shows eval discipline, not SOTA generation.  
- **Why OPE on review?** Human time is the scarce resource; routing is a policy with offline estimators.

## What this is NOT

Not production Document AI, not a hosted LLM platform, not a SOTA benchmark claim.
