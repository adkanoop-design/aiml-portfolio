# Medical Assistant — Clinical Knowledge Assistant with Retrieval-Augmented Generation (Generative AI)

**Stack:** Python, llama-cpp-python (Mistral-7B-Instruct, GGUF), LangChain, ChromaDB, sentence-transformers, PyMuPDF, pandas — run on a Google Colab T4 GPU

## Goal
Give clinicians fast, verifiable answers from a trusted reference — the 4,114-page *Merck Manual of Diagnosis &
Therapy* — instead of relying on a general-purpose LLM's memory. The business problem is information overload in
time-critical care: an answer is only useful if it is complete, consistent from one clinician to the next, and
traceable to the page that justifies it.

## Approach
Five clinical questions (sepsis protocol, appendicitis, patchy hair loss, traumatic brain injury, leg fracture) were
answered in three progressively stronger ways, so that each improvement could be measured against the last:

1. **Raw LLM baseline** — Mistral-7B-Instruct-v0.2 (4-bit GGUF, fully GPU-offloaded) answering from parametric
   memory with greedy decoding.
2. **Prompt engineering and parameter tuning** — five combinations (clinical role prompt, high-temperature sampling,
   structured output template, chain-of-thought, constrained bullets with repeat penalty) applied to every question.
3. **Retrieval-augmented generation** — the manual loaded page by page, the per-page licence watermark stripped
   (713k characters that would otherwise sit in every chunk), split into 17,047 overlapping 1,000-character chunks,
   embedded with `all-MiniLM-L6-v2`, indexed in a persistent ChromaDB collection (cosine), and retrieved into a
   prompt that instructs the model to answer only from the manual and to say so when it cannot. **Six pipeline
   configurations** were then compared on all five questions — `k` = 3/4/5, maximum-marginal-relevance search, a
   second 500-character index (32,735 chunks), sampling temperature, and a longer token budget with a strict
   structured prompt — with the retrieved page numbers tabulated for every run.

An **LLM-as-a-judge** evaluation with separate groundedness and relevance prompts (1–5 scale) scored every answer of
the selected configuration.

## Results
Final configuration: 1,000/200 chunks, similarity search with `k = 4`, greedy decoding, 512-token budget, strict
structured system prompt.

| Question | Groundedness | Relevance | Pages used |
|---|---|---|---|
| Sepsis protocol | 5 | 5 | 2401, 2449, 2454 |
| Appendicitis | 5 | 5 | 173, 174, 3568 |
| Patchy hair loss | 5 | 5 | 856, 858, 859 |
| Traumatic brain injury | 5 | 5 | 1834, 1844, 3413, 3648 |
| Leg fracture | 5 | 5 | 3387, 3389, 3391, 3512 |

- **The raw LLM was fluent but unaccountable.** Four of five baseline answers were truncated and the appendicitis
  answer never reached the medicine-versus-surgery question. Nothing could be traced to a source
- **Prompting fixed form, not knowledge.** Structure and token budget made answers complete; a persona changed tone;
  chain-of-thought exposed the model's reasoning. But the "safe, constrained" prompt still recommended steroids for
  brain swelling — a treatment current trauma guidance does not support — and quoted a dose from memory, and a
  factual error (minoxidil described as an immunomodulator) survived four of five prompts
- **RAG closed the provenance gap.** Every answer became a structured paraphrase of identifiable pages, and retrieval
  — not the model — mapped the lay description "patchy hair loss" to the alopecia section of Chapter 86
- **Retrieval, not the model, was the limiting factor.** With `k = 3` the sepsis answer absorbed first-aid text from
  the neighbouring *Shock* chapter, and the brain-injury question missed Chapter 324 entirely because its lay wording
  sits closer to the rehabilitation chapters than to the chapter's clinical vocabulary. The finer 500-character
  index retrieved the actual sepsis treatment section and the TBI chapter; MMR helped the under-specified questions
  and hurt the well-covered ones
- **Self-grading is lenient.** The judge scored 5/5 throughout — including a brain-injury answer that omits acute
  management, because a context-based judge cannot detect that the wrong text was retrieved. The notebook says so,
  and recommends an independent judge model plus a retrieval-recall check and clinician review

## Business recommendations
Deploy as a **reference assistant that returns cited answers**, not an autonomous diagnostic tool; extend the index to
the hospital's own protocols and re-index on every revision; keep `temperature = 0`, the refusal rule and full
logging as governance controls; build a physician-written gold set and evaluate with an independent judge; serve the
open-weights model on-premise so no clinical query leaves the hospital; and move to hybrid (keyword + dense)
retrieval with a re-ranker as the next technical step.

## Files
- [`Medical_Assistant_RAG.ipynb`](Medical_Assistant_RAG.ipynb) — full notebook with outputs (48 cells, executed sequentially on a T4)
- [`Medical_Assistant_RAG_Full_Code.html`](Medical_Assistant_RAG_Full_Code.html) — rendered full-code export

*The Merck Manual PDF used as the knowledge base is a licensed reference distributed by Great Learning / UT Austin
McCombs for coursework and is not included in this repository.*
