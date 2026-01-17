# Karl Update Log

## v0.7 - LLM Reranker (January 2025)

### Overview
Added an LLM-based reranker (CAG Phase 5) that scores retrieved chunks by semantic relevance before answer generation. Uses Gemini Flash to reorder chunks based on how well they actually answer the user's question.

---

### New Features

#### LLM Reranker (CAG Phase 5)
A post-retrieval relevance scoring layer that uses LLM reasoning to reorder chunks.

**Problem Solved**: Hybrid search (vector + BM25) returns chunks that match keywords or embeddings, but may not actually answer the question. The reranker asks: "Does this chunk help answer the user's question?"

**Solution**: Batch LLM scoring with weighted score combination

- **New File**: `backend/retrieval/reranker.py`
  - Sends all retrieved chunks to Gemini Flash in one call
  - Asks LLM to score each chunk 1-10 for relevance
  - Combines LLM score (70%) with original similarity (30%)
  - Graceful fallback: returns original order on any failure

- **Configuration** (`backend/config.py`):
  ```python
  CAG_ENABLE_RERANKER = True
  RERANKER_MODEL = "gemini-2.0-flash"
  RERANKER_TIMEOUT_MS = 10000
  RERANKER_ORIGINAL_WEIGHT = 0.3
  RERANKER_LLM_WEIGHT = 0.7
  RERANKER_MAX_CHUNKS = 15
  RERANKER_CONTENT_TRUNCATE = 500
  ```

- **Integration Point**: Runs after multi-angle retrieval merge, before full article expansion

---

### Technical Details

#### Reranker Flow
```
Retrieved Chunks (from multi-angle search)
    |
    v
[LLM Reranker]
    |
    +-- Build prompt with query + truncated chunk content
    +-- Gemini Flash scores each chunk 1-10
    +-- Parse JSON response: {"0": 8, "1": 5, "2": 9, ...}
    +-- Compute final score: (0.3 * original) + (0.7 * llm_score/10)
    +-- Re-sort by combined score
    |
    v
Reranked Chunks --> Full Article Expansion --> LLM Answer Generation
```

#### Scoring Prompt
The reranker uses a domain-specific prompt:
- 10: Directly and completely answers the question
- 8-9: Highly relevant, contains key information
- 6-7: Partially relevant, provides useful context
- 4-5: Tangentially related
- 1-3: Not relevant

---

### Benchmark Results

**Evaluation via `evaluate.py`** (uses `retrieve()` without reranker):
| Metric | Result |
|--------|--------|
| Overall | 39/55 (70.9%) |
| Retrieval Accuracy | 72.7% |
| Wage Lookup | 100% |
| Escalation Detection | 66.7% |

**API Testing** (uses `multi_angle_retrieve()` with reranker):
Queries that failed in benchmark but succeed via API:

| Query | Benchmark Result | API Result |
|-------|------------------|------------|
| "How long is my lunch break?" | Article 10 (wrong) | Article 24 (correct) |
| "What are the duties of a Courtesy Clerk?" | Article 2 (wrong) | Article 7, Section 14 (correct) |
| "Do I have to join the union?" | Article 4 (wrong) | Article 3, Section 5 (correct) |

---

### Files Modified/Added

| File | Changes |
|------|---------|
| `backend/retrieval/reranker.py` | NEW - LLM reranker module |
| `backend/config.py` | Added CAG Phase 5 configuration flags |
| `backend/retrieval/router.py` | Integrated reranker into `multi_angle_retrieve()` |

---

### Known Limitations

1. Reranker adds ~1-2s latency per query (LLM call)
2. Only runs in `multi_angle_retrieve()`, not basic `retrieve()`
3. Benchmark script uses `retrieve()` so doesn't test reranker

---

### Next Steps (Planned)

- [ ] Update `evaluate.py` to use `multi_angle_retrieve()` for accurate benchmarking
- [ ] Add reranker metrics to API response
- [ ] Consider caching reranker scores for repeated queries
- [ ] Tune score combination weights based on evaluation data

---
---

## v0.6 - Query Interpreter & UI Polish (January 2025)

### Overview
Major update introducing a systemic Query Interpreter for improved semantic search accuracy, plus significant UI/UX improvements including markdown rendering fixes and enhanced citation navigation.

---

### New Features

#### Query Interpreter System (CAG Phase 4)
A deep semantic analysis layer that runs before retrieval to bridge the vocabulary gap between worker slang and formal contract language.

**Problem Solved**: Questions like *"A vendor is doing a major reset of the snack aisle. How many per year?"* couldn't find Article 2's vendor work restrictions because "reset" doesn't appear in the contract text.

**Solution**: Multi-angle retrieval with HyDE (Hypothetical Document Embeddings)

- **New File**: `backend/retrieval/query_interpreter.py`
  - Extracts structured query understanding (intent, entities, concepts)
  - Generates hypothetical contract-like text for embedding matching
  - Creates multiple search queries from different vocabulary angles
  - Detects explicit article references (e.g., "check Article 2")

- **Configuration** (`backend/config.py`):
  ```python
  CAG_ENABLE_QUERY_INTERPRETER = True
  INTERPRETER_MODEL = "gemini-2.0-flash"
  MULTI_QUERY_MAX_SEARCHES = 3
  MULTI_QUERY_RESULTS_PER_SEARCH = 5
  MULTI_QUERY_TOTAL_RESULTS = 10
  ```

- **Key Innovation**: Direct vector search for hypothetical answers bypasses RRF fusion score distortion, preserving semantic similarity scores

#### Enhanced Citation System
Citations in chat responses are now fully interactive with deep linking support.

- **Clickable citation links** in response text (e.g., "Article 29, Section 77(a)")
- **Citation badges** at bottom of messages also clickable
- **Popover previews** show section content on hover/click
- **Deep navigation** to specific subsections in Contract tab

---

### UI/UX Improvements

#### Desktop Tab Navigation Fix
**Issue**: Tabs were stacking instead of switching on desktop
**Root Cause**: CSS `md:flex` classes on content containers overrode Tailwind's `hidden` class
**Fix**: Restructured HTML to put flex layouts inside wrapper divs

#### Markdown Rendering in Contract Viewer
**Issue**: Raw markdown showing (e.g., `## Section 84.` instead of formatted heading)
**Fix**: New `renderMarkdown()` function with placeholder approach:
- Processes `## headings` → `<h3>` tags
- Processes `**bold**` → `<strong>` tags
- Processes numbered lists and bullet points
- Escapes remaining content safely

#### Citation Parser Enhancements
Updated regex to handle multiple citation formats:
- `Article 29, Section 77` - basic
- `Article 29, Section 77(a)` - parenthetical subsection
- `Article 29, Section 77, Part a` - Part-style reference
- `**Article 29**` - bold markers preserved

#### Popover Error Handling
**Issue**: 404 errors when LLM generates non-existent section numbers
**Fix**: Graceful fallback to full article view with message:
> "Section X not found. Showing Article Y (Z sections)."

#### Dark Mode Improvements
- Added h3 heading styling for contract viewer
- Consistent gold accent colors for headings

---

### Backend API Changes

#### Subsection Filtering
`GET /api/section/{article_num}/{section_num}?subsection=a`

New optional query parameter for filtering to specific subsections:
- **With subsection**: Returns only that subsection's content
- **Without**: Returns all subsections combined (existing behavior)

---

### Files Modified

| File | Changes |
|------|---------|
| `backend/retrieval/query_interpreter.py` | NEW - Query interpretation module |
| `backend/retrieval/router.py` | Added `multi_angle_retrieve()` method |
| `backend/config.py` | Added CAG Phase 4 configuration |
| `backend/api.py` | Subsection filtering, use multi-angle retrieval |
| `frontend/index.html` | Tab fix, markdown rendering, citation links, popover improvements |

---

### Technical Details

#### Query Interpreter Flow
```
User Query
    ↓
Query Interpreter (LLM)
    ↓
┌─────────────────────────────────────┐
│ • Intent extraction                 │
│ • Key concepts identification       │
│ • Hypothetical answer generation    │
│ • Multiple search query generation  │
│ • Explicit article detection        │
└─────────────────────────────────────┘
    ↓
Multi-Angle Retrieval
    ↓
┌─────────────────────────────────────┐
│ 1. Original query → Hybrid search   │
│ 2. Hypothetical → Direct vector     │
│ 3. Alt queries → Hybrid search      │
└─────────────────────────────────────┘
    ↓
Merged & Deduplicated Results
    ↓
LLM Response Generation
```

#### Vocabulary Translation Examples
| Worker Term | Contract Term |
|-------------|---------------|
| vendor/vendor work | recognition, work jurisdiction |
| reset/major reset | vendor work, merchandising |
| fired/canned | discharge, termination |
| write up | discipline, warning |
| break | rest period, relief period |
| overtime/OT | overtime, premium pay |
| floater | personal holiday |

---

### Testing

**Golden Test Case** (now passing):
> "A vendor is seen doing a 'major reset' of the snack aisle. How many of these are they allowed per year?"

**Expected Answer**: Three (3) major resets per store per section per calendar year (Article 2, Section 3)

---

### Known Limitations

1. LLM occasionally generates section numbers that don't exist in the contract - handled gracefully with fallback
2. Query interpreter adds ~1-2 seconds latency for complex queries
3. Subsection format varies between `(a)` style and `Part a` style depending on contract section

---

### Next Steps (Planned)

- [ ] Cache query interpretations for repeated questions
- [ ] Add interpretation confidence scores
- [ ] Expand vocabulary translation dictionary
- [ ] Consider client-side caching for frequently accessed articles

---
---

## v0.5 - CAG "Rosetta Stone" Architecture (December 2025)

### Overview
Initial implementation of Context-Aware Generation (CAG) to solve the vocabulary mismatch problem between worker slang and formal contract language. Named "Rosetta Stone" for its role in translating between different terminologies.

**Core Problem**: Users search for "break" but the contract uses "relief periods" (Section 25). Pure semantic search struggled with these terminology gaps, requiring 3+ attempts with smaller models to find correct sections.

---

### New Features

#### Phase 1: Hybrid Search Tuning

Optimized the BM25 + Vector fusion for legal document terminology.

**Configuration Changes** (`backend/config.py`):
```python
# Rebalanced weights for better RRF fusion
HYBRID_VECTOR_WEIGHT = 1.0   # Was 1.2 - equalized
HYBRID_KEYWORD_WEIGHT = 1.0  # Was 0.8 - equalized

# BM25 parameter tuning
BM25_K1 = 1.8   # Was 1.5 - higher saturation for repeated legal terms
BM25_B = 0.75   # Document length normalization (default)
```

**Rationale**: Equal weights let RRF fusion work properly; increased k1 gives more weight to term frequency which helps with repeated contract terminology.

---

#### Phase 2: Hypothesis Layer

LLM-powered article title prediction to boost relevant sections before retrieval.

**How It Works**:
1. Before retrieval, ask fast LLM: "What article titles might answer this question?"
2. LLM generates 3 candidate titles (e.g., "Rest Periods", "Meal Breaks", "Work Hours")
3. During retrieval, chunks matching hypothesized titles get a score boost

**Configuration**:
```python
CAG_ENABLE_HYPOTHESIS_LAYER = True
HYPOTHESIS_MODEL = "gemini-2.0-flash"  # Better reasoning than flash-lite
HYPOTHESIS_MAX_TITLES = 3              # Number of title guesses
HYPOTHESIS_TIMEOUT_MS = 2000           # 2 second timeout
TITLE_BOOST_SCORE = 0.5                # Score boost for matching titles
```

**Example**:
- Query: "When do I get breaks?"
- Hypothesis: ["REST PERIODS", "MEAL PERIODS", "HOURS OF WORK"]
- Result: Article 24 (Meal) and Article 25 (Relief) get boosted

---

#### Phase 3: Full Article Expansion

When multiple top results come from the same article, fetch the entire article for complete context.

**Logic**:
1. After initial retrieval, check if 2+ of top-5 results are from same article
2. If yes, fetch ALL chunks from that article (up to limit)
3. Provides complete context for complex provisions

**Configuration**:
```python
CAG_ENABLE_FULL_ARTICLE_EXPANSION = True
FULL_ARTICLE_MAX_CHUNKS = 15           # Max chunks to fetch per article
FULL_ARTICLE_MIN_TOP_K_MATCH = 2       # Trigger: 2+ chunks in top-5
```

**Example**:
- Query about "vacation rollover"
- Initial results: 2 chunks from Article 20 (Vacations) in top-5
- Expansion: Fetch all 8 sections of Article 20
- Result: LLM has complete vacation policy context

---

### Architecture

```
┌─────────────────────┐
│   User Query        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  Hypothesis Layer (Phase 2)                 │
│  LLM predicts: "What articles might help?"  │
│  → ["REST PERIODS", "MEAL PERIODS", ...]    │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  Query Expansion (existing)                 │
│  SLANG_TO_CONTRACT dictionary               │
│  "break" → "rest period relief period"      │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  Hybrid Search (Phase 1 - Tuned)            │
│  ┌─────────────┐    ┌─────────────┐        │
│  │   Vector    │    │    BM25     │        │
│  │   1.0 wt    │    │   1.0 wt    │        │
│  └──────┬──────┘    └──────┬──────┘        │
│         └────────┬─────────┘                │
│                  ▼                          │
│         ┌────────────────┐                  │
│         │  RRF Fusion    │                  │
│         │  + Title Boost │ ← Hypothesis     │
│         └────────────────┘                  │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  Full Article Expansion (Phase 3)           │
│  If Article X has 2+ in top-5:              │
│  → Fetch all Article X chunks               │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────┐
│  Final Context      │
│  → LLM Generation   │
└─────────────────────┘
```

---

### Files Modified/Added

| File | Changes |
|------|---------|
| `backend/config.py` | CAG configuration flags and parameters |
| `backend/retrieval/router.py` | Hypothesis layer, title boosting, article expansion |
| `backend/retrieval/hybrid_search.py` | BM25 k1 parameter tuning |
| `data/manifests/` | Article title manifests for hypothesis matching |

---

### Test Results

| Query | Before CAG | After CAG |
|-------|------------|-----------|
| "When do I get breaks?" | ❌ Failed | ✅ Article 24, 25 |
| "15 minute breaks" | ❌ Failed | ✅ Article 25 |
| "relief breaks" | ❌ Failed | ✅ Article 25 |
| "lunch break policy" | ✅ Pass | ✅ Pass |
| "rest periods for 8 hour shift" | ❌ Failed | ✅ Article 25 |

**Retrieval Accuracy**: Improved from 58.2% → ~70%

---

### Feature Flags

All CAG features can be toggled independently for A/B testing:

```python
# Enable/disable each phase
CAG_ENABLE_HYPOTHESIS_LAYER = True      # Phase 2
CAG_ENABLE_TITLE_BOOSTING = True        # Part of Phase 2
CAG_ENABLE_FULL_ARTICLE_EXPANSION = True # Phase 3
```

---

### Performance Impact

| Metric | Before | After |
|--------|--------|-------|
| Average latency | ~800ms | ~1200ms |
| Retrieval accuracy | 58.2% | ~70% |
| High-stakes accuracy | 85% | 95% |

The ~400ms latency increase comes primarily from the hypothesis LLM call, but is acceptable given the accuracy improvement.

---

### Known Limitations

1. Hypothesis layer depends on LLM availability (has timeout fallback)
2. Title boosting requires article manifests to be pre-generated
3. Full article expansion can increase context size significantly

---

### Foundation for v0.6

This architecture laid the groundwork for v0.6's Query Interpreter:
- Proved LLM-in-the-loop retrieval is effective
- Established feature flag pattern for gradual rollout
- Identified need for deeper semantic understanding (→ Query Interpreter)
