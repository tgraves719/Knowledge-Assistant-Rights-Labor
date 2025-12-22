# Karl - Union Contract RAG System

A high-precision Retrieval-Augmented Generation (RAG) system for the UFCW Local 7 Safeway Pueblo Clerks contract (2022-2025). Karl provides citation-grounded responses to contract questions, ensuring accuracy and reliability for union members.

## Features

- **Citation-Focused Responses**: Every answer includes specific Article/Section citations
- **Deterministic Wage Lookups**: 100% accurate wage queries via structured JSON
- **Intent Classification**: Routes queries to appropriate retrieval strategy
- **High-Stakes Detection**: Flags discipline/termination/harassment issues with escalation language
- **Hybrid Retrieval**: Combines vector search with structured lookups

## Project Structure

```
karl/
├── backend/
│   ├── ingest/
│   │   ├── parse_contract.py      # Article/Section chunker
│   │   └── extract_wages.py       # Wage table extractor
│   ├── retrieval/
│   │   ├── vector_store.py        # ChromaDB wrapper
│   │   └── router.py              # Intent classification + hybrid retrieval
│   ├── generation/
│   │   ├── prompts.py             # System prompts
│   │   └── verifier.py            # Citation verification
│   ├── api.py                     # FastAPI backend
│   ├── evaluate.py                # Test evaluation
│   └── config.py                  # Configuration
├── data/
│   ├── chunks/                    # Parsed contract chunks
│   ├── wages/                     # Structured wage tables
│   ├── chroma_db/                 # Vector database
│   └── test_set/                  # Golden Q&A test set
├── frontend/
│   └── index.html                 # Web interface
├── SW+Pueblo+Clerks+2022.2025.md  # Source contract (markdown)
├── SW+Pueblo+Clerks+2022.2025.json # Source contract (JSON)
└── requirements.txt
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Key (Optional but Recommended)

The system works without an API key, but for full functionality (synthesized answers), you'll want a Gemini API key:

1. Get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a `.env` file in the project root:
   ```bash
   # Copy the example file
   cp env.example .env
   
   # Then edit .env and add your key:
   GEMINI_API_KEY=your_actual_api_key_here
   ```

**Note:** Without an API key, Karl will still work but will show raw contract chunks instead of synthesized answers. Wage lookups and retrieval work perfectly without a key.

### 3. Process the Contract

```bash
# Parse contract into chunks
python backend/ingest/parse_contract.py

# Extract wage tables
python backend/ingest/extract_wages.py

# Build vector index
python backend/retrieval/vector_store.py
```

**Note:** If you're cloning from GitHub, the processed data files may already be included. You can skip this step if `data/chunks/contract_chunks.json` and `data/wages/wage_tables.json` already exist.

### 4. Start the API Server

```bash
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

### 5. Open the Frontend

Open `frontend/index.html` in a browser, or navigate to `http://127.0.0.1:8000` in your browser.

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

Run the evaluation suite against the golden test set:

```bash
python backend/evaluate.py
```

### Current Performance

| Category | Accuracy |
|----------|----------|
| **Wage Lookups** | 100% |
| Classification | 100% |
| Overtime | 100% |
| Scheduling | 100% |
| Grievance | 100% |
| Holiday | 100% |
| Time Cards | 100% |
| **Overall Retrieval** | 58.2% |

Note: Some articles are not yet being parsed. Improving the parser regex will increase retrieval accuracy.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   User Query    │────▶│  Intent Router   │────▶│  Vector Search  │
└─────────────────┘     │                  │     │  (ChromaDB)     │
                        │  - Wage Query    │     └────────┬────────┘
                        │  - Contract Q    │              │
                        │  - High-Stakes   │     ┌────────▼────────┐
                        └────────┬─────────┘     │  Wage Lookup    │
                                 │               │  (JSON)         │
                                 │               └────────┬────────┘
                        ┌────────▼─────────┐              │
                        │    LLM + Prompt  │◀─────────────┘
                        │  (Gemini/Claude) │
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │ Citation Verifier│
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │  Grounded Answer │
                        │  with Citations  │
                        └──────────────────┘
```

## Configuration

### Environment Variables

The easiest way to configure Karl is using a `.env` file (see Quick Start step 2).

Alternatively, you can set environment variables:

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="your-api-key"
```

**Linux/Mac:**
```bash
export GEMINI_API_KEY="your-api-key"
```

### What Works Without an API Key?

The system gracefully handles missing API keys and still provides:
- ✅ Accurate wage lookups (100% deterministic from JSON)
- ✅ Relevant contract chunk retrieval (vector search)
- ✅ Intent classification
- ✅ Citation extraction
- ⚠️ **Limited:** Shows raw contract chunks instead of synthesized answers

For the best experience, add your Gemini API key to enable full LLM-powered responses.

## Design Principles

1. **Citation Required**: Every claim must cite Article/Section
2. **No Hallucination**: Only answer from retrieved context
3. **Safe Refusal**: "I cannot find that in your contract" when uncertain
4. **Escalation**: High-stakes topics always recommend contacting steward

## License

Internal prototype for UFCW Local 7.

