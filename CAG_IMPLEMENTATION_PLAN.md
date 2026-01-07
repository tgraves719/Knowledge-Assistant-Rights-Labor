# Context-Aware Generation (CAG) Implementation Plan

**Date:** 2025-12-23
**Status:** Planning Phase
**Goal:** Enhance retrieval system to handle vocabulary mismatches (e.g., "break" vs "relief periods")

---

## Problem Statement

Current testing with smaller models (Mistral Nemo, Gemma 4B) revealed vocabulary matching issues:
- User searches for "break" but contract uses "relief periods" (Section 25)
- Pure semantic search struggles with terminology gaps
- Smaller models require 3+ attempts with hints to find correct sections
- Deepseek 3.2 (smarter model) finds sections immediately

**Root Cause:** While semantic embeddings help, they're not sufficient for bridging domain-specific terminology gaps without very strong LLMs.

---

## Current Architecture Analysis

### ✅ **Already Implemented**

1. **Query Expansion** (`backend/retrieval/router.py:26-104`)
   - SLANG_TO_CONTRACT dictionary mapping common terms to contract language
   - Covers: "break" → "rest period", "lunch" → "meal period"
   - **BUT**: Uses simple append strategy, doesn't solve "relief period" gap

2. **Hybrid Search** (`backend/retrieval/hybrid_search.py`)
   - ✅ BM25 keyword search (exact term matching)
   - ✅ Vector/semantic search (meaning-based)
   - ✅ Reciprocal Rank Fusion (RRF) to combine rankings
   - Currently weighted: Vector=1.2, Keyword=0.8

3. **Intent Classification** (`backend/retrieval/router.py:335-398`)
   - Classifies wage vs contract vs high-stakes queries
   - Detects topics and job classifications
   - Uses topic-to-article mapping for boosting

4. **Multi-Step Context Expansion** (`backend/retrieval/router.py:447-504`)
   - Fetches related sections from same articles
   - Provides cross-section context
   - Limited to post-retrieval expansion

### ⚠️ **Gaps Identified**

1. **Query Expansion Coverage**
   - Missing "relief period" as synonym for "break"
   - No LLM-powered query rewriting
   - Limited to pre-defined dictionary mappings

2. **Single-Pass Retrieval**
   - No iterative refinement based on initial results
   - No feedback loop to adjust query based on what was found

3. **No Query Understanding Layer**
   - Doesn't decompose complex queries into sub-questions
   - Doesn't use LLM to understand user intent before retrieval

4. **Limited Cross-Reference Capability**
   - Related sections fetched but not used to inform second retrieval pass
   - No "use Article X to find related provisions in Article Y" logic

---

## CAG Enhancement Strategy

### Phase 1: Enhanced Query Expansion ⭐ **HIGHEST IMPACT**

**What:** Comprehensive vocabulary bridging layer

**Components:**

1. **Expand SLANG_TO_CONTRACT Dictionary**
   ```python
   # Add missing mappings for breaks/rest
   "break": "rest period relief period meal period",
   "breaks": "rest periods relief periods meal periods",
   "15 minutes": "fifteen minutes relief period rest period",
   "30 minutes": "thirty minutes meal period lunch period",
   "lunch break": "meal period lunch period",
   ```

2. **LLM-Powered Query Rewriting** (Optional)
   - Before retrieval, ask LLM: "Rephrase this query using union contract terminology"
   - Example: "When do I get breaks?" → "What are the rest period and meal period provisions?"
   - **Trade-off:** Adds latency (~200-500ms), but dramatically improves recall

3. **Synonym Expansion Service**
   - Use existing `query_expansion.py` SYNONYM_GROUPS
   - Ensure "breaks" group includes: `["break", "breaks", "rest period", "relief period", "meal period", "meal break", "lunch", "pause", "downtime"]`

**Files to Modify:**
- `backend/retrieval/router.py` (SLANG_TO_CONTRACT)
- `backend/retrieval/query_expansion.py` (SYNONYM_GROUPS)
- `backend/api.py` (add optional LLM rewriting step)

**Expected Impact:** 🔥 **High** - Directly addresses "break" vs "relief period" problem

---

### Phase 2: Improved Hybrid Search Tuning

**What:** Optimize BM25 + Vector fusion for contract terminology

**Adjustments:**

1. **Rebalance Weights**
   - Current: Vector=1.2, Keyword=0.8
   - Proposed: Vector=1.0, Keyword=1.0 (equal weight)
   - Rationale: Exact terminology matters more in legal documents

2. **Add Query-Specific Weight Adjustment**
   ```python
   # If query expansion added many terms, boost keyword search
   if len(expansions) > 3:
       keyword_weight = 1.2  # Favor BM25 when we've expanded terms
       vector_weight = 1.0
   ```

3. **Increase BM25 k1 Parameter**
   - Current: k1=1.5 (default)
   - Proposed: k1=1.8
   - Effect: Slightly more weight to term frequency (helps with repeated terms like "relief period")

**Files to Modify:**
- `backend/retrieval/hybrid_search.py` (BM25Index.__init__)
- `backend/retrieval/router.py` (HybridRetriever.retrieve)

**Expected Impact:** 🔶 **Medium** - Improves fusion quality

---

### Phase 3: Multi-Hop Retrieval

**What:** Use initial retrieval results to inform a second, more targeted search

**Implementation:**

1. **Two-Stage Retrieval**
   ```python
   # Stage 1: Broad search with expanded query
   initial_results = hybrid_search(expanded_query, n=10)

   # Stage 2: Extract key terms from top results
   discovered_terms = extract_domain_terms(initial_results[:3])

   # Combine with original query
   refined_query = f"{original_query} {discovered_terms}"

   # Stage 3: Re-search with refined query
   final_results = hybrid_search(refined_query, n=5)
   ```

2. **Domain Term Extraction**
   - Identify contract-specific terms from retrieved chunks
   - Example: First retrieval finds "meal period", add that to second search
   - Use frequency analysis or LLM extraction

3. **Article Cross-Reference**
   - If initial results reference "See Article X", fetch Article X
   - Follow citation trails

**Files to Add/Modify:**
- `backend/retrieval/multi_hop.py` (new module)
- `backend/retrieval/router.py` (add multi-hop option to retrieve())

**Expected Impact:** 🔶 **Medium** - Catches missed sections, especially for complex queries

---

### Phase 4: Contextual Re-ranking

**What:** Use LLM to re-rank retrieved chunks based on actual relevance

**Implementation:**

1. **LLM-Based Re-ranker**
   ```python
   # After hybrid search, ask LLM to judge relevance
   for chunk in top_10_results:
       prompt = f"Does this section answer '{query}'?\n\n{chunk['content']}"
       relevance_score = llm.score(prompt)  # 0-1 score
       chunk['final_score'] = chunk['rrf_score'] * relevance_score

   results = sorted(chunks, key='final_score', reverse=True)[:5]
   ```

2. **Lightweight Scoring**
   - Use fast model (Gemini Flash Lite already in use)
   - Batch score multiple chunks in one call
   - Cache scores for repeated queries

**Files to Add/Modify:**
- `backend/retrieval/reranker.py` (new module)
- `backend/api.py` (add reranking step after retrieval)

**Expected Impact:** 🔶 **Medium** - Better precision, but adds latency

---

### Phase 5: Query Decomposition

**What:** Break complex queries into simpler sub-questions

**Implementation:**

1. **Multi-Part Query Detection**
   ```python
   # Detect compound queries
   query = "How long are breaks and when do I get them?"

   # Decompose with LLM
   sub_queries = llm.decompose(query)
   # → ["How long are rest periods?", "What is the schedule for rest periods?"]

   # Retrieve for each sub-query
   all_chunks = []
   for sub_q in sub_queries:
       chunks = retrieve(sub_q, n=3)
       all_chunks.extend(chunks)

   # Deduplicate and re-rank
   return deduplicate_and_merge(all_chunks)
   ```

2. **Smart Merging**
   - Deduplicate chunks retrieved multiple times (higher relevance signal)
   - Preserve context from different sub-queries

**Files to Add:**
- `backend/retrieval/decomposition.py` (new module)

**Expected Impact:** 🔵 **Low-Medium** - Useful for complex queries, but most are simple

---

## Implementation Priorities

### 🚀 **Phase 1A: Immediate Wins** (1-2 hours)

1. ✅ Expand SLANG_TO_CONTRACT dictionary
   - Add "relief period" mappings
   - Add numeric time mappings ("15 minutes", "30 minutes")

2. ✅ Update SYNONYM_GROUPS in query_expansion.py
   - Ensure "breaks" group is comprehensive

3. ✅ Test on "break" → "relief period" query
   - Verify retrieval now finds Section 25

**Deliverable:** Updated mappings, passing test case

---

### 🔧 **Phase 1B: LLM Query Rewriting** (2-3 hours)

1. Add `rewrite_query()` function in api.py
   - Optional feature (enable via config flag)
   - Fallback to original query if LLM fails

2. Prompt engineering for rewriting
   ```
   You are a union contract expert. Rewrite this worker's question
   using proper union contract terminology. Keep it concise.

   Worker question: "When do I get breaks?"
   Contract terminology: "What are the provisions for rest periods and meal periods?"
   ```

3. Add latency monitoring
   - Log rewrite time
   - Timeout after 500ms

**Deliverable:** Optional query rewriting with performance metrics

---

### 🏗️ **Phase 2: Hybrid Search Tuning** (1-2 hours)

1. Adjust RRF weights based on query expansion
2. Increase BM25 k1 parameter to 1.8
3. Run benchmark suite to measure impact
4. A/B test: expansion only vs expansion + tuned weights

**Deliverable:** Optimized hybrid search configuration

---

### 🎯 **Phase 3: Multi-Hop Retrieval** (3-4 hours)

1. Implement two-stage retrieval in `multi_hop.py`
2. Add domain term extraction (frequency-based initially)
3. Add opt-in multi-hop mode to API
4. Test on queries that currently fail

**Deliverable:** Multi-hop retrieval module with tests

---

### 📊 **Phase 4 & 5: Advanced Features** (Future)

- Re-ranking (Phase 4): Implement if precision issues persist
- Query decomposition (Phase 5): Implement if complex queries are common

**Decision Point:** Evaluate after Phase 3 completion

---

## Testing Strategy

### Test Cases for "Vocabulary Problem"

| Query | Expected Section | Current Status | Target Status |
|-------|------------------|----------------|---------------|
| "When do I get breaks?" | Article 24, 25 | ❌ Fails | ✅ Pass |
| "15 minute breaks" | Article 25 | ❌ Fails | ✅ Pass |
| "lunch break policy" | Article 24 | ✅ Pass | ✅ Pass |
| "rest periods for 8 hour shift" | Article 25 | ❌ Fails | ✅ Pass |
| "meal period duration" | Article 24 | ✅ Pass | ✅ Pass |
| "relief breaks" | Article 25 | ❌ Fails | ✅ Pass |

### Benchmark Suite

Run existing test set (`backend/evaluate.py`) after each phase:
- Baseline: Current 58.2% retrieval accuracy
- Target: 75%+ retrieval accuracy
- Critical: 100% on high-stakes queries (termination, discipline, etc.)

### A/B Testing Approach

1. **Control:** Current system (hybrid search + basic expansion)
2. **Treatment:** Enhanced expansion + multi-hop
3. **Metric:** Retrieval accuracy on golden test set
4. **Decision:** Ship if improvement > 10 percentage points

---

## Architecture Diagram (Target State)

```
┌─────────────────────┐
│   User Query        │
│ "When do I get      │
│  breaks?"           │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  Query Understanding Layer (NEW)            │
│  - LLM Query Rewriting (optional)           │
│  - Intent Classification (existing)         │
│  - Query Decomposition (future)             │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  Enhanced Query Expansion                   │
│  - SLANG_TO_CONTRACT (✅ enhanced)          │
│  - SYNONYM_GROUPS (✅ enhanced)             │
│  - Expanded: "breaks rest period relief     │
│    period meal period 15 minutes"           │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  Hybrid Search (Stage 1)                    │
│  ┌─────────────┐    ┌─────────────┐        │
│  │   Vector    │    │    BM25     │        │
│  │   Search    │    │   Keyword   │        │
│  │  (semantic) │    │   (exact)   │        │
│  └──────┬──────┘    └──────┬──────┘        │
│         │                  │                │
│         └────────┬─────────┘                │
│                  ▼                          │
│         ┌────────────────┐                  │
│         │  RRF Fusion    │                  │
│         │  (tuned weights)│                 │
│         └────────┬────────┘                 │
└──────────────────┼──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  Multi-Hop Retrieval (NEW - Stage 2)       │
│  - Extract terms from top 3 results         │
│  - Discovered: "relief period" ✓            │
│  - Re-search with refined query             │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  Related Section Expansion (existing)       │
│  - Fetch nearby sections from same articles │
│  - Add Article 24 (Meal) + 25 (Relief)      │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  Contextual Re-ranking (OPTIONAL - Phase 4) │
│  - LLM judges actual relevance              │
│  - Re-scores and re-sorts results           │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────┐
│  Final Results      │
│  1. Article 25      │
│     Relief Periods  │
│  2. Article 24      │
│     Meal Periods    │
└─────────────────────┘
```

---

## Configuration Flags (Gradual Rollout)

Add to `backend/config.py`:

```python
# CAG Feature Flags
ENABLE_LLM_QUERY_REWRITING = False  # Phase 1B - adds latency
ENABLE_MULTI_HOP_RETRIEVAL = False  # Phase 3
ENABLE_LLM_RERANKING = False        # Phase 4 - adds latency
ENABLE_QUERY_DECOMPOSITION = False  # Phase 5

# Hybrid Search Tuning
HYBRID_VECTOR_WEIGHT = 1.0   # Reduced from 1.2
HYBRID_KEYWORD_WEIGHT = 1.0  # Increased from 0.8
BM25_K1 = 1.8               # Increased from 1.5
BM25_B = 0.75               # Keep default
```

This allows gradual feature rollout and A/B testing.

---

## Success Criteria

### Must Have (Phase 1-2)
- ✅ "break" query finds Article 25 (Relief Periods)
- ✅ Retrieval accuracy improves by 10+ percentage points
- ✅ No regression on existing passing queries
- ✅ Latency increase < 100ms (without optional LLM features)

### Should Have (Phase 3)
- ✅ Multi-hop retrieval finds cross-referenced sections
- ✅ Complex queries handled better
- ✅ 75%+ overall retrieval accuracy

### Nice to Have (Phase 4-5)
- ✅ LLM re-ranking improves precision
- ✅ Query decomposition handles compound questions
- ✅ System works well with smaller models (Mistral, Gemma)

---

## Risk Mitigation

### Risk: Added Latency
- **Mitigation:** Make LLM features optional, measure and timeout aggressively
- **Fallback:** Disable expensive features if latency > 2s

### Risk: Increased Complexity
- **Mitigation:** Modular design, each phase is independent
- **Fallback:** Can disable any phase via config flags

### Risk: False Positives from Aggressive Expansion
- **Mitigation:** Monitor precision metrics, tune expansion conservatively
- **Fallback:** Reduce expansion scope if precision drops

### Risk: Over-Engineering for Edge Cases
- **Mitigation:** Implement Phase 1-2 first, evaluate before continuing
- **Decision Point:** Stop after Phase 2 if metrics are satisfactory

---

## Timeline Estimate

| Phase | Effort | Timeline |
|-------|--------|----------|
| Phase 1A: Enhanced Expansion | 2 hours | Day 1 |
| Phase 1B: LLM Rewriting | 3 hours | Day 1-2 |
| Phase 2: Hybrid Tuning | 2 hours | Day 2 |
| **Evaluation & Testing** | 2 hours | Day 2 |
| Phase 3: Multi-Hop | 4 hours | Day 3 |
| Phase 4: Re-ranking | 3 hours | Future |
| Phase 5: Decomposition | 3 hours | Future |

**Total for MVP (Phase 1-3):** ~13 hours across 3 days

---

## Next Steps

1. ✅ Get user approval on this plan
2. 🚀 Start Phase 1A: Expand vocabulary mappings
3. 🧪 Test "break" → "relief period" retrieval
4. 📊 Run benchmark suite
5. 🔄 Iterate based on results
6. 📝 Document learnings for scaling to other contracts

---

## Notes

- **Scalability:** All improvements (especially vocabulary expansion) will help when scaling to 100+ contracts
- **Model Agnostic:** CAG architecture helps smaller models (Mistral, Gemma) perform better
- **Hybrid Approach:** Combining semantic + keyword + multi-hop gives best of all worlds
- **Gradual Rollout:** Feature flags allow testing each improvement independently

---

**Questions for Discussion:**
1. Should we prioritize speed (Phase 1-2 only) or accuracy (include Phase 3)?
2. Is 100-500ms latency acceptable for LLM query rewriting?
3. Which test cases should we prioritize for the "vocabulary problem"?
