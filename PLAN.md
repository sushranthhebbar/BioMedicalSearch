# Project Plan: Biomedical Hybrid Search & RAG Application

> **Reference:** [assignment.pdf](assignment.pdf)  
> **Dataset:** [`rag-datasets/rag-mini-bioasq`](https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq) on Hugging Face  
> **Repository:** https://github.com/sushranthhebbar/BioMedicalSearch

---

## 1. Objective & Scope

Build, evaluate, and document an end-to-end **Biomedical Hybrid Search and Question-Answering (RAG)** system over `rag-datasets/rag-mini-bioasq`. The system must:

1. Retrieve relevant biomedical passages over a single unified corpus using **Lexical (BM25)**, **Dense Vector**, and **Hybrid** modes with **Query Expansion** (plus a stronger reranked variant).
2. Present a clean **Search UI** showing the original query, generated query expansions, ranked evidence passages (IDs, scores, and PubMed source links), a grounded LLM answer with inline citations, and an explicit **"insufficient evidence"** state.
3. Run a reproducible **offline evaluation on a fixed, stratified 100-query set** comparing at least 5 retrieval configurations across both **traditional IR metrics** and **LLM-as-a-judge (RAGAS)** metrics, plus **deterministic citation validation** and **manual disagreement review**.
4. Select and configure the **empirically winning hybrid pipeline** as the production default, backed by a **system-design diagram**, a **technical README**, and a concise **evaluation report**.

---

## 2. Architecture & Technology Stack

| Component | Technology Choice | Rationale |
| :--- | :--- | :--- |
| **LLM (Expansion, Answer, Judge)** | Google Gemini API (`google-genai`, `gemini-2.5-flash`) | Free-tier compatible, fast, supports structured JSON output and exact token usage logging with built-in rate-limit pacing |
| **Dense Embeddings** | Local CPU ONNX (`fastembed`, `BAAI/bge-small-en-v1.5`) | \$0 cost, zero API calls/rate limits, fast ONNXRuntime inference on CPU (no PyTorch), deterministic & resumable checkpointing |
| **Lexical Index** | `bm25s` | Fast NumPy/SciPy BM25 implementation with native disk save/load and no C++ build requirements on Windows |
| **Dense Vector Index** | `faiss-cpu` (`IndexFlatIP` on L2-normalized vectors) | Exact cosine similarity search over ~40k passages in milliseconds with compact persistence |
| **Hybrid Fusion** | Reciprocal Rank Fusion (RRF) | Scale-invariant fusion of BM25 and cosine similarity scores; cleanly extends to multi-query fusion |
| **Strongest Variant Reranker** | `flashrank` (ONNX cross-encoder, `ms-marco-MiniLM-L-12-v2`) | CPU-friendly ONNX cross-encoder (~50–100 ms per query, no PyTorch bloat) to boost top-10 precision |
| **Backend API** | FastAPI + Uvicorn + Pydantic v2 | Typed request/response schemas, serves both REST endpoints and static UI assets |
| **Frontend UI** | Modern HTML5 / CSS3 / Vanilla JS | Zero-build, fast, clean evidence-first UI; runs locally via FastAPI and deployable to Netlify |
| **Offline Evaluation** | Custom IR metrics + `ragas` (`langchain-google-genai`) | Computes `Recall@5`, `Recall@10`, `MRR@10`, `nDCG@10`, latency, cost, and RAGAS correctness/faithfulness/context relevance |
| **Testing** | `pytest` + `httpx` | Unit & integration tests for retrieval, fusion, citation enforcement, insufficient-evidence gating, and API |

---

## 3. Target Repository Layout

```text
BioMedicalSearch/
├── PLAN.md                         # This implementation plan
├── README.md                       # Setup, architecture choices, usage & citation enforcement docs
├── REPORT.md                       # Final evaluation report (comparison table, analysis, cases)
├── requirements.txt                # Locked/pinned stack dependencies
├── .env.example                    # Environment variable template
├── .gitignore                      # Ignores .venv/, .env, large local indexes & caches
├── docs/
│   └── system_design.md            # System-design diagram (4 required paths) & component specs
├── data/                           # Local generated artifacts (gitignored)
│   ├── indexes/
│   │   ├── bm25/                   # Persisted bm25s index
│   │   ├── faiss/                  # Persisted FAISS index + ID mapping
│   │   └── passages.jsonl          # Cleaned canonical passage store (id, text, pubmed_url)
│   ├── cache/                      # Embedding checkpoints & eval query-expansion cache
│   └── logs/                       # Recorded search logs (query, expansions, passage IDs, scores)
├── eval/
│   ├── eval_queries_100.json       # Fixed, version-controlled 100-query benchmark set
│   └── results/                    # Reproducible run logs, summary CSV/JSON, disagreement report
├── src/
│   ├── config.py                   # Centralized settings, model names, thresholds & pricing
│   ├── data_loader.py              # Hugging Face loader, ID normalization & corpus cleaning
│   ├── indexing/
│   │   ├── bm25_index.py           # Build, save, load, and query BM25 index
│   │   └── dense_index.py          # Local ONNX fastembed embeddings + FAISS index
│   ├── retrieval/
│   │   ├── query_expansion.py      # Biomedical query expander (preserves original query)
│   │   ├── fusion.py               # RRF & multi-query rank fusion
│   │   ├── reranker.py             # FlashRank ONNX cross-encoder reranker
│   │   └── retriever.py            # Unified retriever supporting all 5 configurations
│   ├── generation/
│   │   ├── answerer.py             # Strictly grounded LLM answerer + insufficient-evidence gate
│   │   └── citations.py            # Deterministic inline citation extraction & validation
│   ├── pipeline.py                 # Single shared RAG pipeline (used by UI and evaluation)
│   └── api/
│       └── main.py                 # FastAPI endpoints (/api/search, /api/config, /health) + UI
├── frontend/
│   ├── index.html                  # Perplexity-style 3-pane Biomedical Search UI
│   ├── admin.html                  # Separate Admin & Metadata Dashboard UI
│   ├── styles.css                  # Shared sleek dark/light Perplexity-inspired styling
│   ├── app.js                      # 3-pane search flow, history, expandable references, citations
│   └── admin.js                    # Admin dashboard: corpus/index stats, search logs, latency/cost, eval table
├── scripts/
│   ├── build_indexes.py            # Offline ingestion & dual-index build script
│   ├── create_eval_set.py          # Stratified 100-query benchmark generator
│   ├── run_eval.py                 # Reproducible 5-config evaluation + RAGAS runner
│   └── inspect_disagreements.py    # Flags retrieval vs. LLM-judge disagreement cases
└── tests/
    ├── test_retrieval.py           # BM25, FAISS, RRF, and query expansion tests
    ├── test_citations.py           # Citation validator & insufficient-evidence tests
    └── test_api.py                 # FastAPI contract tests
```

---

## 4. Phase-by-Phase Execution Plan

### Phase 0 — Repository, Stack & Environment Setup *(Completed)*
- [x] Initialize Git repository and push to GitHub (`sushranthhebbar/BioMedicalSearch`).
- [x] Finalize stack and create `requirements.txt`, `.gitignore`, and `.env.example`.
- [x] Create Python 3.11 `.venv`, install dependencies, and verify all core imports.

---

### Phase 1 — Data Ingestion, Indexing & Retrieval Engine *(Maps to Step 1: 20 marks — Completed)*

**Deliverables:** `src/config.py`, `src/data_loader.py`, `src/indexing/`, `src/retrieval/`, `scripts/build_indexes.py`, `tests/test_retrieval.py`

1. **Dataset Loading & Normalization (`src/data_loader.py`)**
   - Load both subsets of `rag-datasets/rag-mini-bioasq` via Hugging Face `datasets`:
     - `text-corpus` (`passages` split): canonical `(id, passage)` records.
     - `question-answer-passages` (`test` split): `(id, question, answer, relevant_passage_ids)`.
   - Normalize `id` values to canonical string/int PMIDs and parse stringified `relevant_passage_ids` safely via `ast.literal_eval`.
   - Filter out empty/whitespace-only or `"nan"` passages while maintaining a **1-to-1 shared passage ordering** across BM25 and FAISS so both indexes query the exact same indexed passages.
   - Attach canonical PubMed source links: `https://pubmed.ncbi.nlm.nih.gov/{pmid}/` for valid numeric PMIDs.
   - Persist canonical passages to `data/indexes/passages.jsonl`.

2. **Lexical Index (`src/indexing/bm25_index.py`)**
   - Tokenize passages with biomedical-friendly normalization (preserving hyphenated gene/drug terms, alphanumeric tokens like `IL-6`, `TP53`, `COVID-19`, stopwords removal).
   - Build and persist a `bm25s` index under `data/indexes/bm25/`.
   - Return ranked `(passage_id, bm25_score)` pairs.

3. **Dense Vector Index (`src/indexing/dense_index.py`)**
   - Embed the exact same indexed passages locally on CPU with `fastembed` (`BAAI/bge-small-en-v1.5`, 384-dim L2-normalized ONNX embeddings — \$0 API cost, zero rate limits).
   - Implement **batched embedding with disk checkpointing** (`data/cache/embeddings_checkpoint.npz`) so indexing all ~40k passages is resumable and saved once to disk.
   - Store L2-normalized vectors in `faiss.IndexFlatIP` under `data/indexes/faiss/index.faiss` + `id_map.json`.
   - Return ranked `(passage_id, cosine_score)` pairs.

4. **Query Expansion (`src/retrieval/query_expansion.py`)**
   - Given a user biomedical question, prompt `gemini-2.5-flash` (structured JSON output, `temperature=0.2`) to generate **3 complementary biomedical alternate queries** (incorporating synonyms, MeSH terms, drug/gene aliases, and acronym expansions).
   - **Always retain the original query** as `queries[0]`, deduplicate case-insensitively, and gracefully fall back to `[original_query]` if expansion fails.
   - Record token usage and expansion latency.

5. **Hybrid Fusion & Strongest Variant (`src/retrieval/fusion.py`, `src/retrieval/reranker.py`, `src/retrieval/retriever.py`)**
   - **Hybrid Fusion (RRF):** Combine lexical and dense ranked lists using Reciprocal Rank Fusion:
     $$\text{score}(d) = \sum_{r} \frac{w_r}{k + \text{rank}_r(d)} \quad (k = 60)$$
   - **Multi-Query Fusion (for Query Expansion):** Run retrieval for `[original_query] + alternate_queries`, giving higher weight to `original_query` (e.g., `w_orig = 2.0`, `w_alt = 1.0`) so expansions improve recall without causing topic drift.
   - **Strongest Variant (Cross-Encoder Reranking):** Retrieve top-30 candidates via `Hybrid + Query Expansion`, then rerank using `FlashRank` (`ms-marco-MiniLM-L-12-v2`) against the original question to maximize top-5 / top-10 precision and MRR.
   - **Search Recording:** Log every search (timestamp, mode/config, original query, expansions, retrieved passage IDs, component & final scores, latency, tokens) to `data/logs/searches.jsonl`.

6. **Verification:** Build indexes over the full `rag-mini-bioasq` corpus and pass `pytest tests/test_retrieval.py`.

---

### Phase 2 — Grounded Generation, Citation Enforcement & User Interface *(Maps to Step 2: 20 marks)*

**Deliverables:** `src/generation/`, `src/pipeline.py`, `src/api/main.py`, `frontend/`, `tests/test_citations.py`, `tests/test_api.py`

1. **Strict Grounded Answer Generation (`src/generation/answerer.py`)**
   - Pass only the top-$K$ selected retrieved passages (default $K=5$, configurable) formatted with explicit ID tags (`[Passage ID: 12345678] ...`) into `gemini-2.5-flash` (`temperature=0.0` for faithful grounding).
   - System instructions strictly forbid using outside knowledge: every factual claim must be supported by the provided passages and cite the passage ID inline as `[12345678]`.
   - Structured output returns `{ "answer": "...", "sufficient_evidence": true/false, "insufficient_reason": "...", "cited_ids": [...] }`.

2. **Explicit "Insufficient Evidence" Gate (Two-Stage)**
   - **Pre-generation threshold check:** If no retrieved passages meet minimum relevance score thresholds (calibrated per retrieval mode, e.g., reranker score or lexical/dense minimum cutoff), short-circuit immediately to the explicit `"insufficient_evidence"` state without hallucinating.
   - **Model-level abstention:** If retrieved passages exist but do not actually answer the user's question (e.g., off-topic or missing key clinical fact), the LLM explicitly sets `sufficient_evidence: false` and explains what evidence is missing.

3. **Code-Enforced Citation Validation (`src/generation/citations.py`)**
   - Deterministically parse all inline `[ID]` citations in the generated answer using regex.
   - Verify every cited ID is a member of the **selected retrieved passage IDs** for that query.
   - Track `valid_citations`, `invalid_citations`, and `citation_validity_rate`.
   - Strip or flag any invalid/hallucinated citation IDs; if a non-abstaining answer has zero valid citations, run one repair/fallback check or downgrade to insufficient evidence.

4. **Shared End-to-End Pipeline (`src/pipeline.py`)**
   - Single `RAGPipeline.run(query, config_name, generate_answer=True)` entry point used identically by both the FastAPI UI and the offline 100-query evaluation harness.

5. **Backend API & Perplexity-Style UI + Admin UI (`src/api/main.py`, `frontend/`)**
   - **FastAPI Endpoints:**
     - `POST /api/search`: accepts `{ query, mode, top_k }`, returns search ID, original query, expansions, ranked evidence passages (ID, combined score, lexical/dense/rerank breakdown, text, PubMed URL, cited flag), final answer with inline citations, citation validation summary, `sufficient_evidence` flag, latency breakdown, token usage, and estimated cost.
     - `GET /api/history` & `GET /api/history/{search_id}`: returns recent search history items and full stored search payloads for the left sidebar.
     - `GET /api/config`: returns active default (winning) configuration and available retrieval modes.
     - `GET /api/health`: index readiness status and corpus stats.
     - `GET /api/admin/stats`: corpus size, BM25/FAISS index metadata, active models/thresholds, cumulative query count, average latency, total estimated cost, and citation validity rate.
     - `GET /api/admin/logs`: paginated/filterable search audit logs from `data/logs/searches.jsonl` (including full retrieved passage IDs, scores, expansions, timings, and tokens).
     - `GET /api/admin/eval-summary`: serves offline 100-query benchmark summary table (`eval/results/summary_table.csv` / JSON) once generated.
   - **Main Search UI (`/` — `frontend/index.html`, `styles.css`, `app.js`) — Perplexity-Style 3-Pane Layout:**
     - **Left Sidebar (Search History):**
       - "New Search" button, chronological list of previous searches (clickable to restore any past query, expansions, answer, and references), search filter/clear option, and a direct link to the **Admin UI (`/admin`)**.
     - **Center Main Pane (Query → Expansions → Answer → Bottom Citations):**
       - Perplexity-style prominent search bar (centered on landing with BioASQ example prompts; docked cleanly during active threads) plus a compact retrieval mode selector defaulting to the winning configuration.
       - **Query & Expansions Bar:** Compact Perplexity-style query-steps view showing the original query alongside generated biomedical query expansions.
       - **Grounded Final Answer:** Cleanly formatted answer with interactive inline citation badges (`[12345678]`) OR a prominent, explicit **"Insufficient Evidence"** callout state explaining why no grounded answer could be generated.
       - **Citations Footer (Bottom of Answer):** Structured citations strip/list below the answer showing cited passage IDs, snippet previews, and direct PubMed links; clicking any citation badge or bottom citation jumps to and expands that passage in the right panel.
     - **Right Side Panel (Expandable References / Evidence Drawer):**
       - Displays all ranked evidence passages with rank badge, **Passage ID (PMID)**, **score**, citation status badge (`Cited` / `Retrieved`), and PubMed link (`https://pubmed.ncbi.nlm.nih.gov/{id}/`).
       - **Accordion / Expandable Details:** Clicking any reference card expands it smoothly to reveal the full passage text, score breakdown (BM25 / Dense / RRF / Rerank), provenance (original vs. expanded query match), and direct source link.
   - **Separate Admin & Metadata UI (`/admin` — `frontend/admin.html`, `admin.js`):**
     - **System & Index Overview Cards:** Corpus passage count, BM25 + FAISS readiness, embedding/LLM/reranker models, active default mode, and sufficiency thresholds.
     - **Live Search Metadata & Audit Table:** Filterable log of every search showing timestamp, query, mode, expansion count, top score, `sufficient_evidence` status, citation validity %, latency breakdown (ms), token counts, and estimated USD cost — with click-to-expand raw metadata (full passage IDs, scores, and expansions).
     - **Offline Benchmark Tab:** Renders the 5-configuration comparison table once offline evaluation results are available.
   - Configurable `API_BASE_URL` so both the main UI and Admin UI run locally via FastAPI and remain deployable standalone (e.g., Netlify frontend + hosted API).

6. **Verification:** Pass `pytest tests/test_citations.py tests/test_api.py` and visually verify live searches (both answerable and unanswerable / insufficient-evidence queries) in the browser.

---

### Phase 3 — Fixed 100-Query Benchmark & Offline Experiments *(Maps to Step 3: 20 marks)*

**Deliverables:** `scripts/create_eval_set.py`, `eval/eval_queries_100.json`, `scripts/run_eval.py`, `eval/results/`

1. **Balanced 100-Query Evaluation Set (`scripts/create_eval_set.py`)**
   - Filter `rag-mini-bioasq` test questions to those with $\ge 1$ valid `relevant_passage_id` present in our indexed corpus.
   - Classify questions into the 6 assignment categories using deterministic linguistic/biomedical heuristics + manual spot-check:
     1. `yes_no` (~18 queries)
     2. `definition` (~16 queries)
     3. `list` (~16 queries)
     4. `treatment` (~17 queries)
     5. `mechanism` (~17 queries)
     6. `effect` (~16 queries)
   - Sample with a fixed random seed (`seed=42`) to produce **100 fixed query IDs** saved permanently to `eval/eval_queries_100.json` (committed to Git so every run uses the exact same 100 query IDs).

2. **Five Required Configurations**
   Run all 5 configurations through the exact same `RAGPipeline` over the exact same 100 query IDs:

   | # | Config ID | Retrieval Signal(s) | Query Expansion | Reranking |
   | :-: | :--- | :--- | :---: | :---: |
   | 1 | `lexical` | BM25 (`bm25s`) | No | No |
   | 2 | `dense` | `fastembed` (`BAAI/bge-small-en-v1.5`) + FAISS | No | No |
   | 3 | `hybrid` | BM25 + Dense (RRF, $k=60$) | No | No |
   | 4 | `hybrid_qe` | BM25 + Dense (Weighted Multi-Query RRF) | Yes (Orig + 3 alts) | No |
   | 5 | `hybrid_qe_rerank` *(Strongest Variant)* | BM25 + Dense (Weighted Multi-Query RRF, top-30) | Yes (Orig + 3 alts) | Yes (`FlashRank` ONNX cross-encoder) |

   - Cache generated expansions & dense query embeddings per query ID so Configs 4 & 5 are strictly apples-to-apples and API cost stays minimal.
   - Persist complete per-query outputs (retrieved top-10 IDs/scores, selected top-5 passages, generated answer, citations, latency breakdown, token usage) to `eval/results/<config_id>_runs.jsonl`.

---

### Phase 4 — Dual-Layer Scoring, Citation Checks & Manual Review *(Maps to Step 4: 20 marks)*

**Deliverables:** Evaluation metrics module (`src/eval/metrics.py`, `src/eval/ragas_judge.py`), `scripts/inspect_disagreements.py`, `eval/results/summary_table.csv`

1. **Layer 1 — Traditional IR Metrics (`src/eval/metrics.py`)**
   Compute macro-averaged metrics across all 100 queries against `relevant_passage_ids`:
   - `Recall@5`
   - `Recall@10`
   - `MRR@10`
   - `nDCG@10` (binary relevance against `relevant_passage_ids`)
   - `Avg Latency (ms)` and `P95 Latency (ms)` (end-to-end and retrieval-only)
   - `Estimated Cost (USD / 100 queries)` computed from actual logged Gemini prompt/completion/embedding tokens.

2. **Layer 2 — LLM-as-a-Judge Evaluation (`src/eval/ragas_judge.py`)**
   - Use **RAGAS** with a **strictly fixed judge configuration across all 5 runs**:
     - Model: `gemini-2.5-flash` via `langchain-google-genai`
     - `temperature = 0.0`, fixed prompts & metric settings
   - Evaluate 3 required dimensions:
     - **Answer Correctness** (comparing generated answer against BioASQ gold `answer`)
     - **Groundedness / Faithfulness** (checking answer claims against retrieved passages)
     - **Context Relevance** (assessing how relevant retrieved contexts are to the question)
   - Run **code-based citation validity check** across all 500 generated answers (valid citation rate, zero-hallucinated-ID compliance, abstention rate).

3. **Layer 3 — Disagreement & Manual Review (`scripts/inspect_disagreements.py`)**
   - Automatically surface examples where retrieval metrics and LLM-judge scores disagree, e.g.:
     - *High judge correctness despite low Recall@10* (e.g., relevant alternative passage not in incomplete gold `relevant_passage_ids`, or abstention behavior).
     - *High Recall@10 despite low judge correctness/groundedness* (e.g., gold passage ranked #6–10 outside top-5 context window, or partial/over-cautious synthesis).
   - Manually inspect representative successes, failure modes, and disagreement examples for inclusion in `REPORT.md`.

4. **Update Production Default**
   - Set the empirically winning configuration as the default mode in `src/config.py` and the UI.

---

### Phase 5 — System Design Diagram, README & Final Report *(Maps to Step 5: 20 marks + Submission Deliverables)*

**Deliverables:** `docs/system_design.md`, `README.md`, `REPORT.md`

1. **System-Design Diagram (`docs/system_design.md` + embedded in `README.md`)**
   - Clean, detailed Mermaid architecture diagram covering all 4 required paths:
     1. **Offline Data Path:** Hugging Face dataset loading → cleaning & PMID normalization → canonical passage store → parallel BM25 (`bm25s`) & Dense (`fastembed` `BAAI/bge-small-en-v1.5` + `faiss-cpu`) index building.
     2. **Online Query Path:** User query → Gemini query expansion → parallel BM25 + Dense retrieval → weighted RRF hybrid fusion → FlashRank cross-encoder reranking → evidence sufficiency gate → grounded Gemini answer generation → deterministic citation validator.
     3. **User-Facing Path:** Search UI input → query/expansions display → ranked evidence cards (IDs, scores, PubMed links) → grounded answer with clickable inline citations OR explicit insufficient-evidence state.
     4. **Experiment Path:** Fixed 100-query set → config registry → shared `RAGPipeline` runner → structured JSONL logging → IR metrics + fixed RAGAS judge + citation validator + disagreement inspector → selection of winning default config.

2. **Technical README (`README.md`)**
   - Setup & run instructions (local UI + reproducible offline evaluation).
   - Clear rationale for all 5 required design decisions:
     1. Storage / vector database (`bm25s` + `faiss-cpu` + JSONL canonical store)
     2. Embedding model (`fastembed` `BAAI/bge-small-en-v1.5`, local CPU ONNX — \$0 API cost)
     3. Fusion & reranking method (weighted RRF + `FlashRank` ONNX cross-encoder)
     4. Final-answer model (`gemini-2.5-flash` with strict grounding & two-stage abstention)
     5. Citation validation enforcement (code location, parsing logic, and unit test coverage)

3. **Final Evaluation Report (`REPORT.md`)**
   - **One clean master comparison table** covering all 5 configurations across `Recall@5`, `Recall@10`, `MRR@10`, `nDCG@10`, `Answer Correctness`, `Groundedness`, `Context Relevance`, `Citation Validity`, `Latency`, and `Cost`.
   - Clear verdict on **what won and why it won**, backed by numbers and per-category breakdown.
   - **Latency & cost trade-off analysis** (comparing Lexical vs. Dense vs. Hybrid vs. Hybrid+QE vs. Strongest Variant).
   - **Qualitative review:** 3–4 successful searches, 3–4 failure cases, and specific **retrieval-vs-judge disagreement cases** from manual inspection.

---

## 5. Risks & Mitigations

| Risk | Mitigation |
| :--- | :--- |
| **Gemini Free Tier RPM/RPD rate limits during evaluation** | Use local ONNX `fastembed` (0 API calls for embeddings), cache query expansions per query ID, apply configurable RPM pacing + exponential backoff on `429`, and checkpoint every evaluated query to disk so runs resume cleanly. |
| **Dirty/missing `relevant_passage_ids` or empty passages in `rag-mini-bioasq`** | Clean corpus upfront, parse stringified lists with `ast.literal_eval`, and validate 100% of eval queries have $\ge 1$ gold passage in the indexed corpus. |
| **Query expansion causing topic drift** | Always preserve original query with higher RRF weight (`2.0` vs `1.0` for alternates) and use cross-encoder reranking against the original query in the strongest variant. |
| **LLM citing non-existent passage IDs or using outside biomedical knowledge** | Use `temperature=0.0`, strict grounding prompt, explicit abstention threshold, and deterministic post-generation citation validator in Python. |
| **RAGAS judge drift or timeouts across 500 evaluations** | Fix judge model (`gemini-2.5-flash`), `temperature=0.0`, and prompts; checkpoint RAGAS results per query so partial runs never lose progress. |

---

## 6. Verification & Definition of Done

- [x] `scripts/build_indexes.py` builds BM25 and FAISS indexes over the exact same indexed `rag-mini-bioasq` passages.
- [ ] `pytest` passes all retrieval, citation-enforcement, insufficient-evidence, and API tests.
- [ ] Live UI clearly displays original query, expansions, ranked passages (IDs, scores, PubMed links), grounded inline-cited answer, and explicit insufficient-evidence state.
- [ ] `scripts/run_eval.py` reproducibly evaluates all 5 configurations on the exact same 100 fixed query IDs (`eval/eval_queries_100.json`) with both IR metrics and RAGAS judge scores.
- [ ] Winning configuration is set as the app default and backed by `REPORT.md`.
- [ ] `docs/system_design.md` and `README.md` document all 4 required system paths and all 5 required design choices.
- [ ] Final repo pushed cleanly to GitHub.
