# KARL — Knowledge Assistant for Rights & Labor

KARL is an AI-powered RAG system designed to help workers and unions
understand, navigate, and enforce their collective bargaining agreements.

Currently serving: **UFCW Local 7 — Safeway Pueblo Clerks (2022-2025)**

The goal of KARL is to:
- Reduce information asymmetry between workers and employers
- Make labor contracts legible to the people bound by them
- Strengthen collective power through shared understanding

## Core Principles

- Union-first
- Worker-controlled
- Privacy-respecting
- Anti-surveillance
- Transparent by design

## Features

- **Citation-Focused Responses**: Every answer includes specific Article/Section citations
- **Deterministic Wage Lookups**: 100% accurate wage queries via structured JSON tables
- **Context-Aware Generation (CAG)**: 5-phase retrieval pipeline translates worker language into contract terminology
- **High-Stakes Detection**: Flags discipline/termination/harassment issues with escalation language
- **Hybrid Retrieval**: Vector search (ChromaDB + MiniLM-L6-v2) fused with BM25 keyword search
- **LLM Reranker**: Post-retrieval relevance scoring reorders chunks by semantic fit
- **Contract-Specific Routing**: Manifest-driven configuration — new contracts need only a JSON file, no code changes
- **Interactive Citation Navigation**: Clickable citations with popover previews and deep linking

## Current Performance

**55/55 (100%)** on the golden benchmark test set across all 19 categories.

| Metric | Result |
|--------|--------|
| Overall Retrieval Accuracy | **100%** (55/55) |
| Wage Lookup | 100% |
| Escalation Detection | 100% |
| All 19 categories | 100% |

## Project Structure

```
karl/
├── backend/
│   ├── ingest/
│   │   ├── smart_chunker.py       # Article/Section/LOU chunker
│   │   ├── enricher.py            # LLM metadata enrichment (Gemini)
│   │   ├── extract_wages.py       # Wage table extractor
│   │   ├── table_extractor.py     # JSON-based table extraction
│   │   ├── rebuild_index.py       # Vector index rebuild utility
│   │   ├── manifest.py            # Contract manifest generator
│   │   └── schema.py              # Chunk schema definitions
│   ├── retrieval/
│   │   ├── router.py              # Intent classification + multi-angle retrieval
│   │   ├── hybrid_search.py       # Vector + BM25 fusion with concept boost
│   │   ├── hypothesis.py          # LLM article title prediction (CAG Phase 2)
│   │   ├── query_interpreter.py   # Deep semantic query analysis (CAG Phase 4)
│   │   ├── reranker.py            # LLM relevance reranker (CAG Phase 5)
│   │   ├── vector_store.py        # ChromaDB wrapper (dual content fields)
│   │   └── query_expansion.py     # Slang-to-contract term expansion
│   ├── generation/
│   │   ├── prompts.py             # System prompts
│   │   ├── tools.py               # LLM tool definitions
│   │   ├── context.py             # Context assembly
│   │   └── verifier.py            # Citation verification
│   ├── user/
│   │   └── profile.py             # User profile management
│   ├── api.py                     # FastAPI backend
│   ├── evaluate.py                # Benchmark evaluation
│   └── config.py                  # Configuration & feature flags
├── data/
│   ├── chunks/                    # Parsed contract chunks (320 chunks)
│   ├── wages/                     # Structured wage tables (JSON)
│   ├── tables/                    # Extracted table data
│   ├── manifests/                 # Contract-specific routing config
│   ├── chroma_db/                 # Vector database (ChromaDB)
│   └── test_set/                  # Golden Q&A test set (55 cases)
├── frontend/
│   └── index.html                 # Web interface (dark mode, citation nav)
├── legal/
│   ├── ETHICAL-USE.md             # Ethical use policy
│   ├── COMMERCIAL-LICENSE.md      # Commercial licensing terms
│   └── LICENSE-EXCEPTION.md       # License exceptions for unions
├── requirements.txt
└── UPDATE_LOG.md                  # Detailed version history
```

## Architecture

KARL uses a 5-phase Context-Aware Generation (CAG) pipeline:

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Phase 1: Intent Router                         │
│  Wage query? High-stakes? Contract question?    │
│  Slang expansion ("break" → "rest period")      │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  Phase 2: Hypothesis Layer                      │
│  LLM predicts relevant article titles           │
│  Matching chunks get a similarity boost         │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  Phase 3: Hybrid Search + Article Expansion     │
│  Vector (MiniLM-L6-v2) + BM25 via RRF fusion   │
│  If 2+ top results from same article → expand   │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  Phase 4: Query Interpreter (Multi-Angle)       │
│  LLM generates hypothetical answers + alt       │
│  search queries for vocabulary bridging          │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  Phase 5: LLM Reranker                          │
│  Gemini scores each chunk for relevance (1-10)  │
│  Combined score: 30% original + 70% LLM         │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  Answer Generation (Gemini 2.5 Pro)             │
│  Grounded response with Article/Section cites   │
│  Citation verification before delivery          │
└─────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Key

Get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey), then create a `.env` file:

```bash
cp env.example .env
# Edit .env and add your key:
# GEMINI_API_KEY=your_actual_api_key_here
```

Without an API key, KARL still provides wage lookups and chunk retrieval, but cannot generate synthesized answers.

### 3. Process the Contract

```bash
# Parse contract into chunks
python -m backend.ingest.smart_chunker

# Enrich chunks with LLM-generated metadata
python -u -m backend.ingest.enricher --batch-size 15 --delay 1.0

# Build vector index
python -m backend.ingest.rebuild_index
```

If cloning from GitHub, processed data files may already be included.

### 4. Start the Server

```bash
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

### 5. Open the Frontend

Navigate to `http://127.0.0.1:8000` in your browser.

## API Endpoints

### Health Check
```
GET /api/health
```

### Query Contract
```
POST /api/query
{
  "question": "What is my overtime rate?",
  "user_classification": "all_purpose_clerk",
  "hours_worked": 0,
  "months_employed": 0
}
```

### Wage Lookup
```
POST /api/wage
{
  "classification": "courtesy_clerk",
  "months_employed": 48
}
```

## Evaluation

Run the benchmark against the golden test set:

```bash
python -m backend.evaluate
```

## Models

| Component | Model | Purpose |
|-----------|-------|---------|
| Answer Generation | Gemini 2.5 Pro | Final response synthesis |
| Hypothesis Layer | Gemini 2.5 Flash | Article title prediction |
| Query Interpreter | Gemini 2.5 Flash | Semantic query analysis |
| Reranker | Gemini 2.5 Flash | Chunk relevance scoring |
| Enricher | Gemini 2.5 Flash | Chunk metadata enrichment |
| Embeddings | all-MiniLM-L6-v2 | Local vector embeddings (no API) |

## Configuration

All features can be toggled via flags in `backend/config.py`:

```python
CAG_ENABLE_HYPOTHESIS_LAYER = True       # Phase 2
CAG_ENABLE_FULL_ARTICLE_EXPANSION = True # Phase 3
CAG_ENABLE_QUERY_INTERPRETER = True      # Phase 4
CAG_ENABLE_RERANKER = True               # Phase 5
```

### Environment Variables

Set via `.env` file or shell:

```bash
GEMINI_API_KEY=your-api-key
```

## Design Principles

1. **Citation Required**: Every claim must cite Article/Section
2. **No Hallucination**: Only answer from retrieved context
3. **Safe Refusal**: "I cannot find that in your contract" when uncertain
4. **Escalation**: High-stakes topics always recommend contacting your steward

## License

KARL is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

If you deploy KARL as a service, you must provide the source code to your users.

Unions and workers: use it freely.
Employers: see `legal/COMMERCIAL-LICENSE.md`.

## What KARL Will Not Do

- Track individual worker behavior
- Generate disciplinary recommendations
- Assist with union busting
- Serve as a productivity surveillance tool

See `legal/ETHICAL-USE.md` for the full ethical use policy.

## Status

This project is under active development (v0.8). Contributions are welcome — but must
align with the principles above.
