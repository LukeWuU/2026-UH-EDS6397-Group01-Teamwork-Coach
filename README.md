# Teamwork & Leadership Coach (MVP)

AI-powered teamwork and leadership coach for engineering students (EDS 6397 — University of Houston).

Students submit a de-identified teamwork reflection. The system:

1. Redacts PII and checks for high-risk content
2. Diagnoses likely teamwork challenges
3. Retrieves evidence from a tagged research corpus (RAG)
4. Generates practical, evidence-grounded coaching
5. Validates safety / citations / scope before display
6. Finalizes coaching, abstains safely, or escalates to UH support resources

This repository also includes:

- **Knowledge Corpus Builder** — Docling + Ollama pipeline to expand the evidence corpus
- **Evaluation harness** — stratified golden set, gated-RAG vs no-RAG baselines, scorecard

## Stack

| Component | Choice |
| --- | --- |
| Orchestration | LangGraph |
| LLM (default) | Local Ollama `llama3.1:8b` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | ChromaDB (local) |
| UI | Streamlit |
| Observability | LangSmith (optional, sanitized nested traces) |

## Repository layout

```text
.
├── README.md
├── PRD.md
├── requirements.txt
├── .env.example
├── contract.py                 # Shared Pydantic state contract
├── main_system.py              # CLI / run_coach entry
├── agents/                     # LangGraph nodes
├── config/                     # Settings, taxonomy, safety, escalation
├── corpus/                     # Active evidence corpus (indexed for coach + eval)
│   ├── chunks/chunks.json
│   └── sources/sources.json
├── guardrails/
├── ingestion/                  # build_index (+ ingestion stubs)
├── interface/app.py            # Student coach UI
├── services/                   # LLM, embeddings, retrieval, tracing
├── evaluation/                 # Golden-set eval + reports
├── Knowledge_Corpus_Builder/   # Standalone corpus expansion app
├── tests/
└── report/                     # Course write-up placeholder
```

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) with `llama3.1:8b` (coach, eval, and corpus builder)

```bash
ollama pull llama3.1:8b
```

- Optional: LangSmith API key for tracing

## Setup

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
# Edit .env if your Ollama host/model differ from the defaults.

python -m ingestion.build_index
```

### Environment variables (`.env`)

```bash
# Coach / systems-under-test — local Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# Evaluation LLM-as-judge (rubric / pairwise)
EVAL_LLM_PROVIDER=gemini
GOOGLE_API_KEY=
GEMINI_MODEL=gemini-3.5-flash

# Retrieval / corpus
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=./.chroma
RETRIEVAL_TOP_K=4
RETRIEVAL_MIN_SCORE=0.05

# LangSmith (optional)
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=teamwork-leadership-coach
```

## Run the coach

```bash
# CLI smoke
python main_system.py

# Student UI
streamlit run interface/app.py
```

The coaching UI shows validated advice plus a **Supporting sources** list. Retrieved chunk text remains on the response object for evaluation, but is not shown to students.

## Knowledge Corpus Builder

Expand the evidence corpus from PDFs, slides, transcripts, and images:

```bash
pip install -r Knowledge_Corpus_Builder/requirements.txt
# Ollama required for markdown repair + tag suggestion
streamlit run Knowledge_Corpus_Builder/app.py
```

Pipeline: scan `Corpus_Inputs` → Docling → Ollama repair → structure chunk → hierarchical cluster → tag suggest → human review → export to `Knowledge_Corpus_Builder/Corpus_Output/`.

### Promote builder output into the coach corpus

The coach and evals read **`corpus/chunks/chunks.json`** and **`corpus/sources/sources.json`**. Builder export drops there (replace), then rebuild the index:

```bash
cp Knowledge_Corpus_Builder/Corpus_Output/sources/sources_mvp.json corpus/sources/sources.json
cp Knowledge_Corpus_Builder/Corpus_Output/chunks/chunks_mvp.json corpus/chunks/chunks.json
python -m ingestion.build_index
```

**Eval note:** the coach corpus is instructor-pluggable (builder → promote → `build_index`). Evaluation does **not** use fixed `gold_chunk_ids` / Recall@k. RAG quality is judged via citation gates, routing/safety/refusal suites, and gated-RAG vs no-RAG advice compare.

## Evaluation

Stratified golden set + gated RAG vs no-RAG baseline. See `evaluation/README.md`.

Eval runs use **Ollama** for systems under test (gated RAG / no-RAG). The optional LLM-as-judge rubric uses **Gemini** (`EVAL_LLM_PROVIDER=gemini`; needs `GOOGLE_API_KEY`).

```bash
python -m evaluation --dry-run
python -m evaluation                       # gated RAG (Ollama + Chroma)
python -m evaluation --system compare      # gated RAG vs no-RAG
python -m evaluation --system rubric       # Gemini judge on saved reports
python -m evaluation --system scorecard    # rebuild one-page scorecard
python -m evaluation --suites safety,privacy
```

Reports: `evaluation/reports/` (`latest.md`, `latest_compare.md`, `latest_scorecard.md`).

Offline metric tests (no Ollama / API key):

```bash
pytest -q tests/test_evaluation.py
```

## Tests

```bash
pytest -q
```

## LangSmith tracing (optional)

1. Create an API key at [smith.langchain.com](https://smith.langchain.com)
2. In `.env` set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY=...`
3. Each `run_coach` call appears as one nested trace: privacy → diagnosis/retrieval → advice → validation → finalize/repair/fallback/escalation

Raw reflections are omitted from telemetry; sensitive fields are redacted before upload.

## Shipping to GitHub

### What to commit

- Source code (`agents/`, `services/`, `interface/`, `evaluation/`, `Knowledge_Corpus_Builder/` Python modules, etc.)
- MVP `corpus/` JSON (hand-tagged)
- `PRD.md`, `README.md`, `requirements.txt`, `.env.example`, `pytest.ini`
- `Knowledge_Corpus_Builder/Corpus_Inputs/` only if license allows sharing those materials

### What NOT to commit (already gitignored)

- `.env` (secrets)
- `.venv/`, `Knowledge_Corpus_Builder/.venv/`
- `.chroma/` (rebuild with `python -m ingestion.build_index`)
- Generated `evaluation/reports/*` (keep `.gitkeep` plus committed `latest_scorecard.md`, `latest_compare.md`)
- Generated `Knowledge_Corpus_Builder/Corpus_Output/` workspace JSON/markdown
- `__pycache__/`, model caches
- `Latest Update XX/` snapshot folders (local merge artifacts)

### Suggested first push

```bash
git init   # if needed
git add .
git status   # confirm .env is NOT staged
git commit -m "Merge evaluation harness with corpus builder and Ollama coach path"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

## Team roles (PRD)

| Member | Focus |
| --- | --- |
| Francisco | Advice generation |
| Alex | Project management / evaluation |
| Luija | Reflection interface / workflow |
| Kashfin | Diagnosis / RAG / corpus builder |
| Roberto | Security / privacy / reliability |

## Notes

- Product requirements: `PRD.md`
- Decision log: `cursor_calls.md`
- The product is advisory only and does not replace instructors, counselors, or emergency services
