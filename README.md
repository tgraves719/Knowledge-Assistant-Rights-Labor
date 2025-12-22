# KARL — The Union Steward Assistant

KARL is an AI assistant designed to help workers and unions
understand, navigate, and enforce their collective bargaining agreements.

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

### Current Performance

Based on evaluation against 55 golden test cases:

| Category | Accuracy | Notes |
|----------|----------|-------|
| **Wage Lookups** | 100% | Perfect accuracy on all wage queries |
| **Escalation Detection** | 100% | High-stakes topics correctly flagged |
| Classification | 100% | Job classification queries |
| Breaks | 100% | Rest period queries |
| Grievance | 100% | Grievance procedure queries |
| Holiday | 100% | Holiday pay and scheduling |
| Layoff | 100% | Layoff and bumping procedures |
| Refusal | 100% | Correctly refuses when context insufficient |
| Safety | 100% | Safety-related queries |
| Sick Leave | 100% | Sick leave provisions |
| Time Cards | 100% | Time card requirements |
| Vacation | 100% | Vacation accrual and scheduling |
| Benefits | 67% | Some benefit queries need improvement |
| Discipline | 75% | Most discipline queries work |
| Overtime | 50% | Some overtime scheduling queries fail |
| Union | 50% | Union-related queries need work |
| Seniority | 33% | Seniority calculation queries need improvement |
| Scheduling | 0% | Article 10 scheduling provisions not well parsed |
| Dress Code | 0% | LOU provisions not in test set |
| **Overall Retrieval** | **76.4%** | 42/55 test cases pass |

**Known Issues:**
- Article 10 (Scheduling) sections are not being parsed correctly
- Article 27 (Seniority) sections 63 and 66 need better parsing
- Some Letter of Understanding (LOU) provisions are missing from chunks

**Improvement Areas:**
- Parser regex needs updates for Article 10 and Article 27
- LOU provisions should be added to the chunking process

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

KARL is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

If you deploy KARL as a service, you must provide the source code to your users.

Unions and workers: use it freely.
Employers: see COMMERCIAL-LICENSE.md.

## What KARL Will Not Do

- Track individual worker behavior
- Generate disciplinary recommendations
- Assist with union busting
- Serve as a productivity surveillance tool

## Status

This project is under active development. Contributions are welcome—but must
align with the principles above.
