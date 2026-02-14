# Rigorous RAG Evaluation Strategy for Karl (v3 Plan)

## Overview

## Benchmark Taxonomy (Canonical Naming)

To avoid ambiguity across docs, KARL uses:

- **Benchmark v1**: legacy golden benchmark track (historical 100% on the original 55-case set)
- **Benchmark v2**: harder single-contract comprehensive benchmark track (the post-v1 drop into the mid/high-80s range happened here)
- **Benchmark v3**: scaled multi-contract evaluation framework (this document), partially implemented and not yet complete

In this file:

- **Phase A** corresponds to finalizing and hardening the **v2-style single-contract rigor**
- **Phase B** corresponds to the **v3 multi-contract scaling framework**

---

**Phase A (Days 1-7)**: Single-contract depth testing with rigorous ablation, entailment checking, and adversarial evaluation. This is the v2-hardening stage that validates the system genuinely works on one contract.

**Phase B (Days 8+)**: Multi-contract scaling via Universal Schema, Synthetic Benchmark Factory, and cross-contamination detection. This is the v3 scaling stage for moving from 1 contract toward 100+ without O(N) manual labor.

---

## Prototype -> 3-Contract Deployed Service (Execution Timeline)

### Phase 0 (Week 0-1): Program Baseline and Freeze

1. Freeze current baseline:
   - Tag repo
   - Snapshot corpus hashes
   - Record model/config versions
2. Finalize documentation truth:
   - `README.md`
   - `UPDATE_LOG.md`
   - `SETUP.md`
   - `Evaluation_Plan_v3.md`
3. Define release gates and ownership:
   - `legal/RELEASE-GATES.md`
   - Assign Mission/Model/Data council signers

Exit criteria:
- One canonical benchmark vocabulary (`v1`/`v2`/`v3`) across docs
- One canonical evaluation command for each benchmark track

Canonical commands:
- `python -m backend.evaluate_runner --track v1`
- `python -m backend.evaluate_runner --track v2 --ablation-mode normal`
- `python -m backend.evaluate_runner --track escalation`

### Phase 1 (Week 1-3): v2 Hardening (Single Contract, Production-Quality)

1. Fix evaluation consistency:
   - Primary eval path must match runtime retrieval path decisions
   - Ablation flags must toggle real runtime behavior
2. Implement missing v2 rigor pieces:
   - Entailment integrated into grader output
   - Precedence hard-fail integrated into final scoring
   - Unanswerable taxonomy scored separately
3. Add CI evaluation:
   - Core benchmark on every PR
   - Full v2 suite on protected-branch merges

Exit criteria:
- Reproducible v2 results with clear pass/fail thresholds
- No metric ambiguity between scripts

### Phase 1.5 (Immediate Pre-Scale): Escalation Precision Hardening

This phase is required before onboarding Contract 2/3.

1. Keep deterministic-first escalation policy:
   - No stochastic escalation behavior
   - Two-stage logic: `high_stakes_topic` vs `active_urgent_context`
2. Add conditional-language suppressors and tests:
   - "what are my rights if"
   - "hypothetically"
   - "in case"
   - "if I were"
3. Add dedicated escalation test slice:
   - Conditional/hypothetical rights questions
   - Active urgent situations
   - Neutral policy prompts containing trigger words
4. Produce decision-quality metrics:
   - Confusion matrix
   - Precision/recall
   - False-positive analysis and threshold tradeoffs

Exit criteria:
- High-stakes escalation precision meets approved threshold
- Escalation false-positive rate is at or below approved cap
- No escalation regressions versus approved baseline

### Phase 2 (Week 3-5): Multi-Tenant Architecture Foundation

1. Contract-agnostic refactor:
   - Remove hardcoded contract IDs/manifest paths
   - Require `union_local_id`, `contract_id`, `contract_version` in API/retrieval/eval paths
2. Retrieval isolation:
   - Enforce metadata filtering at query time
   - Add cross-contamination test suite
3. Environment hardening:
   - AuthN/AuthZ
   - Production CORS allowlist
   - Secret handling and rotation policy

Exit criteria:
- Single deployment can host multiple contracts with strict tenant isolation
- Cross-contamination test pass rate meets threshold (target 0 leakage)

### Phase 3 (Week 5-7): Contract 2 and Contract 3 Onboarding

1. Ingest and normalize two additional contracts
2. Build per-contract manifests and provenance records
3. Clone/adapt v2 tests per contract:
   - Shared templates + contract-specific expected answers/citations
4. Run side-by-side evaluation:
   - Compare Contract A/B/C performance deltas

Exit criteria:
- 3 contracts pass core release gates independently
- Delta spread within agreed limit

### Phase 4 (Week 7-9): Steward-Facing Pilot Deployment

1. Deploy staged pilot to selected stewards/reps
2. Enable structured telemetry:
   - Retrieval quality
   - Citation entailment failures
   - High-stakes escalation behavior
3. Incident process dry-run:
   - Rollback drills
   - Mis-citation escalation workflow

Exit criteria:
- Pilot stability window passed
- No unresolved Sev1/Sev2 issues

### Phase 5 (Week 9-12): Production Launch for 3 Contracts

1. Production rollout via canary
2. Weekly evaluation cadence:
   - Full v2 suite
   - Contract isolation tests
3. Governance cadence:
   - Weekly risk review for first month
   - Monthly thereafter

Exit criteria:
- Production SLOs met
- Signed release packet with benchmark, risk, and rollback evidence

---

## Design Principles

This evaluation separates four distinct concerns:
1. **Retrieval quality** - Does the right chunk appear in top-k?
2. **Generation quality** - Is the answer semantically correct and grounded?
3. **Safety/refusal behavior** - Does the system refuse appropriately?
4. **Evaluator reliability** - Can we trust the grading signal?

Each is tested independently to avoid conflation.

---

## Part 1: Leakage and Overfitting Checklist

### Critical Failure Modes

| Risk | Status | Evidence |
|------|--------|----------|
| **Question-Answer Co-development** | HIGH | Questions in `comprehensive_test.json` explicitly reference "Article X, Section Y" |
| **Single-Document Overfitting** | HIGH | All 55 questions derive from one 223KB document |
| **Lenient Retrieval Grading** | HIGH | `check_retrieval()` passes if ANY expected article in top-k |
| **Prompt Leakage** | MEDIUM | `SLANG_TO_CONTRACT` mappings likely tuned on benchmark failures |
| **Retriever Collapse** | UNKNOWN | Requires ablation to detect |
| **Implicit Memorization** | UNKNOWN | Could leak via caching, few-shot examples, or benchmark text in scaffolding - audit needed |
| **Evaluator Bias** | MEDIUM | Pattern matching only, no semantic grading |

### Non-Obvious Failure Modes

1. **Vocabulary Leakage**: `SLANG_TO_CONTRACT` mappings added after observing failures
2. **Topic-to-Article Hardcoding**: `get_topic_article_map()` makes retrieval deterministic for topic-matched questions
3. **Full Article Expansion Gaming**: `_expand_to_full_article()` inflates recall for clustered questions

### Required Fix: Delete SLANG_TO_CONTRACT

**Problem**: `SLANG_TO_CONTRACT` in `router.py` is hardcoding the test set into the compiler. This does not generalize to new contracts.

**Fix**: Replace with LLM-based Query Expansion layer:

```python
def expand_query_llm(query: str) -> str:
    """
    Use LLM to expand worker slang to contract terminology.
    Generalizes across contracts without hardcoding.
    """
    prompt = """
    Expand this worker's question to include formal contract terminology.
    
    Worker question: "{query}"
    
    Add synonyms and formal terms in parentheses. Keep the original question.
    Example: "break time" -> "break time (rest period, relief period, meal period)"
    """
    return llm.generate(prompt.format(query=query))
```

**Implementation Priority**: HIGH - Do this before running Phase A evaluation to get unbiased baseline.

---

## Part 2: Question Bucketing (Required for Valid Ablation)

Before running ablations, classify each question:

| Bucket | Definition | Example | Ablation Expectation |
|--------|------------|---------|---------------------|
| **World Knowledge** | Answerable from general labor law priors | "What is overtime?" | Small drop when retrieval off |
| **Contract-Only** | Requires specific contract text | "What is the grievance deadline?" | Large drop when retrieval off |
| **Multi-Hop** | Requires synthesizing 2+ sections | "Pharmacy Tech Sunday OT pay calculation" | Drops with top-1, no expansion |
| **Exact Numeric** | Requires precise number/date | "Starting All Purpose Clerk rate 1/21/2024" | Fails without exact chunk |

**Ablation metrics must be computed per-bucket.** A global "40% drop" threshold is meaningless without this stratification.

---

## Part 3: Benchmark Redesign

### 3.1 Section Masking (Synthetic Holdout)

**Target: Contract-specific weirdness, not generic concepts**

Mask Articles 35-42 and create 10 questions targeting:
- Specific deadlines ("How many days to notify about sick leave documentation?")
- Exact eligibility conditions ("Hours required for health benefit eligibility in a 4-week month?")
- Exception clauses ("When does the 90-day retroactive limit NOT apply?")
- Rare definitions ("What qualifies as 'specially trained' for Cake Decorator layoff protection?")

**Avoid**: Generic labor concepts like "what is overtime" that the base model can answer from priors.

### 3.2 Paraphrase Robustness (Correct Metric)

**Wrong metric**: Same article/chunk retrieved across paraphrases (optimizes for brittleness)

**Correct metrics**:
1. At least one supporting chunk in top-k for each paraphrase
2. Citation entailment passes (cited text supports the claim)
3. Final answer is semantically equivalent across paraphrases

Retrieval consistency is a **diagnostic** (reveals instability) but not the **goal** (a paraphrase might legitimately shift which chunk is most relevant).

### 3.3 Unanswerable Questions (Typed Taxonomy)

Separate into 4 types with different expected behaviors:

| Type | Example | Expected Behavior | Tests |
|------|---------|-------------------|-------|
| **Missing from Corpus** | "What is the 401k employer match?" | "Not specified in contract" | Retrieval grounding |
| **Contradictory Prompt** | "Find the 401k match" (none exists) | Clarify it doesn't exist | Hallucination resistance |
| **Wrong Scope (Close)** | "Meat department overtime rules" (excluded in Art 1) | Explain exclusion, refuse | Domain boundary |
| **Ambiguity Needs Clarification** | "When can I take leave?" (which type?) | Ask clarifying question | Disambiguation |

**Separate from policy alignment tests** (legal advice, comparative, speculation) - those test safety, not retrieval.

### 3.4 Needle-in-a-Haystack (Anti-Gaming Protocol)

**Problem**: `_expand_to_full_article()` can fetch needles without genuine retrieval.

**Protocol**:
1. Inject 5 synthetic facts with unique tokens: `KARL_NEEDLE_7Q2M: Employees with 10,000+ hours receive a $500 longevity bonus`
2. **Disable full-article expansion** for needle tests
3. Require system to cite the **exact sentence** containing the needle token
4. Automatic verification: grep for `KARL_NEEDLE_xxx` in cited text

### 3.4.1 Needle Position-Bias Testing

**Problem**: "Lost in the Middle" phenomenon - LLMs attend poorly to content in the middle of long contexts.

If you increase `n_results=20` to solve the "Specific Overrides General" problem, needles may get lost.

**Protocol**: For each needle, test at three positions in the context window:

| Position | Implementation | Expected | Failure Mode |
|----------|---------------|----------|--------------|
| **Top** | Needle in chunk #1 | 100% recall | Baseline |
| **Middle** | Needle in chunk #10 (of 20) | >90% recall | Lost in middle |
| **Bottom** | Needle in chunk #20 | >95% recall | Recency helps |

**Test Matrix**:

```python
def test_needle_position_bias():
    """Test if needle position affects retrieval/generation."""
    results = {}
    
    for needle_id in NEEDLE_IDS:
        for position in ["top", "middle", "bottom"]:
            # Reorder chunks to place needle at position
            chunks = get_chunks_with_needle_at(needle_id, position)
            
            # Run generation
            answer = generate(query, chunks)
            
            # Check if needle was used
            results[(needle_id, position)] = needle_in_answer(answer, needle_id)
    
    # Alert if middle/bottom significantly worse than top
    for needle_id in NEEDLE_IDS:
        top = results[(needle_id, "top")]
        mid = results[(needle_id, "middle")]
        if top and not mid:
            print(f"ALERT: Needle {needle_id} lost in middle")
```

### 3.5 Answer Extraction Questions

For 10 questions, require the system to:
1. Quote the exact clause (short span, 1-3 sentences)
2. Paraphrase what it means

**Purpose**: Forces retrieval to be correct, reveals chunk boundary problems, makes grading trivial.

Example: "Quote the exact contract language defining 'just cause' for discharge, then explain what it means in plain language."

### 3.6 Adversarial Near-Miss Questions

Target provisions that are **similar but have different exceptions**:

| Question | Trap | Correct Answer |
|----------|------|----------------|
| "Sunday premium for employee hired March 15, 2005" | March 26/27 cutoff | Check exact date in Art 13 |
| "Layoff bumping for employee who worked 4 months as Cake Decorator" | 6-month minimum | Cannot displace, insufficient time |
| "Recall rights for store 12 miles away" | 10-mile threshold | Must accept (within threshold) |

These expose retrieval systems that get "the vibe" but miss critical conditions.

### 3.7 Human-Written Questions

Have 3 non-developers write 10-15 questions each from a worker's perspective:
- "I just got written up, what do I do?"
- "My manager keeps changing my schedule last minute"
- "Am I getting paid right? Been here 2 years"

Developers write questions like lawyers; users write them like humans.

### 3.8 Escalation Precision Slice (Release-Blocking)

Goal: reduce false-positive steward-escalation prompts without reducing true-positive urgent detection.

Required constraints:
1. Do not make escalation stochastic.
2. Keep deterministic policy as the primary decision path.
3. Use two-stage classification:
   - `high_stakes_topic` (informational only)
   - `active_urgent_context` (actual escalation trigger)
4. Add conditional/hypothetical suppressors (examples: "what are my rights if", "hypothetically", "in case", "if I were")
5. Allow LLM only as a tie-breaker for low-confidence borderline cases, with deterministic fallback.

Required test slices:
- Conditional/hypothetical rights questions
- Active urgent situations
- Neutral policy questions containing escalation trigger words

Required reporting:
- Confusion matrix (TP/FP/TN/FN)
- Precision/recall
- False-positive rate
- Threshold tradeoff table for candidate policy cutoffs

---

## Part 4: Retrieval Ablation Matrix (Bucket-Aware)

| Condition | Implementation | Contract-Only Bucket | World-Knowledge Bucket |
|-----------|----------------|---------------------|----------------------|
| **Retrieval OFF** | `n_results=0` | Expect >50% drop | May survive |
| **Random Retrieval** | Random chunk selection | Expect >70% drop | May partially survive |
| **Top-1 Only** | `n_results=1`, no expansion | 10-30% drop acceptable | Minimal change |
| **No Hypothesis** | `CAG_ENABLE_HYPOTHESIS_LAYER=False` | ~5% drop on vocab-gap Qs | No change |
| **BM25 Only** | `use_hybrid=False` | ~15% drop on semantic Qs | No change |
| **Vector Only** | `HYBRID_KEYWORD_WEIGHT=0` | ~10% drop on exact-term Qs | No change |
| **No Expansion** | `CAG_ENABLE_FULL_ARTICLE_EXPANSION=False` | Reveals multi-section dependency | No change |

**Key diagnostic**: If Random Retrieval in **Contract-Only bucket** performs >50% accuracy, retriever is not contributing.

---

## Part 5: Citation Entailment Check (Critical Addition)

Valid citations are not enough. The cited text must **actually support** the claim.

### Entailment Grader Task

For each claim-citation pair in the answer, three-way classification:

```
CLAIM: "Overtime is paid at 1.5x the base rate"
CITED TEXT: "...time and one-half (1.5x) the employee's base hourly rate..."

Verdict: SUPPORTS | CONTRADICTS | IRRELEVANT
Answer: SUPPORTS
```

### Failure Modes Caught

- Model cites correct article but wrong section
- Model cites article that discusses the topic but states a different rule
- Model hallucinates a number/deadline not in the cited text

### Implementation Options

**Option A: Lightweight NLI (Low Token Cost)**
Use a fine-tuned NLI model (e.g., `roberta-large-mnli`, `deberta-v3-large-mnli`) for entailment classification. Fast, cheap, but may miss legal nuance.

**Option B: LLM Entailment (High Accuracy)**
Use GPT-4/Claude for entailment with structured output. Slower, expensive, but catches subtle failures.

**Recommended**: Use NLI for first-pass screening, escalate low-confidence or CONTRADICTS cases to LLM.

---

## Part 5.1: "Specific Overrides General" Check (CRITICAL)

### The Problem

Legal logic: **Specific language supersedes general language.**

Example:
- Article 10 (General): "All employees receive 1.5x for overtime."
- Article 22 (Specific - Pharmacy): "Pharmacy technicians receive 2.0x for overtime."

If user is a Pharmacy Tech and asks "What is my overtime?":
- KARL retrieves Article 10
- KARL answers "1.5x"
- KARL cites Article 10
- Entailment check **passes** (Article 10 does say 1.5x)
- **But the answer is legally wrong** because Article 22 exception applies

### The Fix: Precedence Failure Detection

Add `precedence_failure` as a **hard fail** condition (score = 0).

**Test Protocol**:

| Query | User Context | Expected | Negative Constraint | Pass Condition |
|-------|--------------|----------|---------------------|----------------|
| "Overtime rate?" | Pharmacy Tech | 2.0x (Art 22) | If answer = 1.5x → FAIL | Must cite Art 22 |
| "Sunday premium?" | Hired March 15, 2005 | Check Art 13 cutoff | If applies post-March 27 rule → FAIL | Correct date logic |
| "Probation period?" | Sanitation Clerk (1996) | Check Art 2 grandfathering | If ignores protection → FAIL | Must cite Art 2 |

**Implementation**:

```python
class PrecedenceCheck:
    """Check if a general rule was applied where a specific exception exists."""
    
    def check(self, query: str, user_context: dict, answer: str, chunks: list) -> bool:
        """
        Returns True if precedence failure detected.
        
        1. Identify if query has potential exceptions based on user_context
        2. Check if exception-bearing chunks were retrieved
        3. If exception exists but answer uses general rule → precedence_failure = True
        """
        # Identify applicable exceptions from user context
        exceptions = self.find_applicable_exceptions(user_context)
        
        if not exceptions:
            return False  # No exceptions apply
            
        # Check if exception was retrieved
        exception_retrieved = any(
            self.chunk_contains_exception(chunk, exceptions) 
            for chunk in chunks
        )
        
        # Check if answer used exception or general rule
        answer_uses_exception = self.answer_references_exception(answer, exceptions)
        
        # FAIL: Exception exists, was retrieved, but answer used general rule
        if exception_retrieved and not answer_uses_exception:
            return True
            
        # FAIL: Exception exists, wasn't retrieved (retrieval failure on precedence)
        if not exception_retrieved:
            return True  # Should have retrieved the exception
            
        return False
```

### Adversarial Near-Miss Questions (Extended)

Add 10 questions specifically targeting precedence:

| Question | Classification/Context | Trap | Correct Behavior |
|----------|----------------------|------|------------------|
| "Overtime rate?" | Pharmacy Tech | Art 10 general (1.5x) | Must find Art 22 (2.0x) |
| "Sunday premium?" | Hired March 15, 2005 | Post-March 27 rules | Check exact cutoff date |
| "Layoff bumping rights?" | Cake Decorator (4 months) | General bumping rules | Must check 6-month minimum |
| "Vacation accrual?" | Hired Jan 2004 | Post-2005 rules | Pre-2005 grandfathering applies |
| "Health eligibility?" | Part-time, 88 hrs/month | General eligibility | Check part-time specific threshold |

---

## Part 6: Grading Rubric (Tightened)

### 4-Point Scale

| Score | Label | Criteria |
|-------|-------|----------|
| **3** | Correct + Grounded | Semantically correct, all citations pass entailment, no fabrication |
| **2** | Partially Correct | Core correct but missing nuance, OR correct with 1+ entailment failures |
| **1** | Wrong but Honest | Wrong answer but hedged/uncertain, OR refuses when should answer |
| **0** | Hallucinated | Wrong + confident, OR citation doesn't support claim, OR fabricated citation |

### Anti-Gaming Rules

- **Hallucinated Confidence**: Wrong + confident without hedging = 0 (not 1)
- **Entailment Failure**: Any claim with NOT SUPPORT citation = max score 2
- **Citation Fabrication**: Non-existent Article/Section = automatic 0

### Grader Prompt Structure

```
You are evaluating a union contract Q&A system.

QUESTION: {question}
USER CONTEXT: {user_context}  # classification, hire date, etc.
EXPECTED ANSWER: {ground_truth}
SYSTEM ANSWER: {system_answer}
RETRIEVED CHUNKS: {chunks}
ENTAILMENT RESULTS: {entailment_checks}
APPLICABLE EXCEPTIONS: {exceptions}  # Specific rules that override general

Score on 4 dimensions (0-3 each):
1. Factual Accuracy: Does the answer match contract text?
2. Citation Entailment: Do cited sources actually support claims? (Use entailment results)
3. Completeness: Does answer address all parts of question?
4. Appropriate Uncertainty: Does system refuse/hedge when warranted?

CRITICAL CHECK - Precedence Failure:
If an exception in {exceptions} applies to this user but the answer used a general rule instead, this is a PRECEDENCE FAILURE (automatic score = 0).

For each dimension, provide score and 1-sentence justification.
```

### Evaluation Result Schema

```python
from pydantic import BaseModel, Field
from typing import List, Literal

class EntailmentResult(BaseModel):
    claim: str
    citation: str
    cited_text: str
    verdict: Literal["SUPPORTS", "CONTRADICTS", "IRRELEVANT"]
    confidence: float

class EvaluationResult(BaseModel):
    question_id: str
    contract_id: str
    
    # The 4 Dimensions (0-3 each)
    factual_accuracy: int = Field(..., ge=0, le=3)
    citation_entailment_score: float = Field(
        ..., 
        ge=0.0, le=1.0,
        description="Percentage of claims with SUPPORTS verdict"
    )
    completeness: int = Field(..., ge=0, le=3)
    uncertainty_calibrated: bool = Field(
        ..., 
        description="Did it refuse/hedge correctly when appropriate?"
    )
    
    # Hard Fail Conditions
    precedence_failure: bool = Field(
        False, 
        description="Applied general rule where specific exception exists"
    )
    cross_contamination_detected: bool = Field(
        False, 
        description="Retrieved/cited chunks from wrong contract"
    )
    citation_fabrication: bool = Field(
        False,
        description="Cited non-existent Article/Section"
    )
    
    # Entailment Details
    entailment_results: List[EntailmentResult] = []
    
    # Computed Final Score
    @property
    def final_score(self) -> int:
        """Final score with hard fail overrides."""
        if self.precedence_failure:
            return 0
        if self.cross_contamination_detected:
            return 0
        if self.citation_fabrication:
            return 0
        if self.citation_entailment_score < 0.5:
            return min(2, self.factual_accuracy)  # Cap at 2 if poor entailment
        return self.factual_accuracy
```

---

## Part 7: Implementation Plan (7 Days)

### Day 0: Freeze Artifacts

```bash
git tag eval-v2-$(date +%Y%m%d)
cp -r data/chroma_db data/chroma_db_eval_snapshot
sha256sum data/chunks/*.json > data/corpus_hashes.txt
```

Document exact versions of: embedding model, LLM, all config flags.

### Day 1-2: Question Set Construction

| Task | Count | File |
|------|-------|------|
| Bucket existing 55 questions | 55 | `comprehensive_test.json` (add `bucket` field) |
| Paraphrase families | 45 | `paraphrase_test.json` |
| Unanswerables (4 types) | 20 | `unanswerable_test.json` |
| Adversarial near-miss | 10 | `adversarial_test.json` |
| Needle injection | 5 | `needle_test.json` + modified chunks |
| Answer extraction | 10 | `extraction_test.json` |
| Human-written | 30-50 | `human_questions_test.json` |
| **Total new** | ~120-140 | |

### Day 3: Ablation Framework

Modify `evaluate_comprehensive.py`:
- Add `ablation_mode` parameter
- Add `--bucket-filter` to run on specific buckets
- Output metrics grouped by bucket

### Day 4: Entailment + LLM Grader

Create `backend/eval/entailment.py`:
- `extract_claim_citation_pairs(answer, chunks)` 
- `check_entailment(claim, cited_text) -> SUPPORTS | NOT_SUPPORT`

Create `backend/eval/llm_grader.py`:
- Integrate entailment results into grading prompt
- Return structured scores per dimension

### Day 5-6: Execute Evaluation Suite

| Test Set | Questions | Ablations | Total Evals |
|----------|-----------|-----------|-------------|
| Original (bucketed) | 55 | 7 | 385 |
| Paraphrase | 45 | Normal only | 45 |
| Unanswerable | 20 | Normal only | 20 |
| Adversarial | 10 | Normal only | 10 |
| Needle | 5 | No-expansion only | 5 |
| Extraction | 10 | Normal only | 10 |
| Human-written | 40 | Normal only | 40 |
| **Total** | ~185 | | ~515 |

Escalation policy hardening must complete before any multi-contract onboarding work:
- Run the escalation precision slice in Part 3.8
- Produce confusion matrix and threshold tradeoff analysis
- Verify deterministic-first policy and two-stage classification behavior in runtime and tests
- Treat any escalation precision regression as release-blocking

### Day 7: Analysis Report

Generate report answering:

1. **Per-bucket ablation analysis**: Does Contract-Only bucket drop >50% with retrieval off?
2. **Paraphrase semantic consistency**: Are answers equivalent across paraphrases (not just chunks)?
3. **Unanswerable by type**: Refusal rate per unanswerable category
4. **Entailment failure rate**: What % of citations actually support their claims?
5. **Adversarial accuracy**: Does system catch near-miss exceptions?
6. **Needle retrieval**: Can system find injected facts without expansion?
7. **Human vs dev question gap**: Performance difference on human-written questions?
8. **Escalation precision profile**: confusion matrix, precision/recall, and false-positive rate against policy cap

---

## Part 8: Success Thresholds (Bucket-Aware)

| Metric | Threshold | Notes |
|--------|-----------|-------|
| Contract-Only bucket (LLM-graded) | Mean >= 2.5/3.0 | Core capability |
| Retrieval-OFF drop (Contract-Only) | >50% absolute | Proves retrieval matters |
| Random-retrieval (Contract-Only) | <25% accuracy | Proves retrieval quality matters |
| Paraphrase semantic equivalence | >85% | Answers stay correct |
| Unanswerable (missing-corpus) refusal | >90% | |
| Unanswerable (wrong-scope) refusal | >80% | |
| Citation entailment pass rate | >90% | Critical grounding metric |
| Adversarial near-miss accuracy | >70% | Catches exception handling |
| Needle exact-citation rate | 100% (5/5) | With expansion off |
| High-stakes escalation precision | >= approved policy threshold | Release-blocking metric |
| High-stakes escalation false-positive rate | <= approved policy cap | Release-blocking metric |

---

## What Cannot Be Evaluated Without Additional Data

1. **Document-level generalization**: Requires 2+ contracts
2. **Temporal generalization**: Requires previous contract version
3. **Production query distribution**: Requires real user logs
4. **Cross-contract transfer**: Requires contracts from different locals/employers

---

## Audit Checklist Before Running

- [ ] Git commit tagged
- [ ] ChromaDB snapshot saved
- [ ] Corpus hashes documented
- [ ] No benchmark text in system prompts or test scaffolding
- [ ] No query/response caching enabled
- [ ] Config flags documented (hypothesis layer, expansion, etc.)
- [ ] LLM grader using different model than system (avoid self-eval bias)

---
---

# PHASE B: Multi-Contract Scaling (v3, Planned - Not Fully Implemented)

## The Scaling Problem

Phase A (v2 hardening) is O(N) labor - requires manual crafting of questions for every new contract. That works for 1, maybe 3, but fails at 100. To complete v3 scaling for KARL to 100+ contracts, we shift from **Writing Benchmarks** to **Generating Benchmarks**.

---

## Part 9: Universal Schema (The Abstraction Layer)

Before ingesting Contract #2, define a canonical JSON schema of standard labor variables that every contract likely contains:

```json
{
  "$schema": "https://karl.ai/schemas/labor-contract-v1.json",
  "contract_id": "string",
  "employer": "string",
  "union_local": "string",
  "term": {
    "start_date": "date",
    "end_date": "date"
  },
  "wages": {
    "classifications": [
      {
        "name": "string",
        "starting_rate": "float",
        "top_rate": "float",
        "progression_hours": "integer[]"
      }
    ],
    "premiums": {
      "overtime_multiplier": "float",
      "sunday_multiplier": "float | null",
      "night_premium_cents": "integer | null",
      "holiday_multiplier": "float"
    }
  },
  "scheduling": {
    "posting_deadline": "string",
    "minimum_shift_hours": "integer",
    "maximum_shift_hours": "integer",
    "bidding_by_seniority": "boolean"
  },
  "time_off": {
    "probation_days": "integer",
    "vacation_accrual": [
      {"years": "integer", "weeks": "integer"}
    ],
    "sick_leave_accrual_rate": "string",
    "personal_holidays": "integer",
    "bereavement_days": {
      "immediate_family": "integer",
      "extended_family": "integer"
    }
  },
  "grievance": {
    "filing_deadline_days": "integer",
    "discharge_deadline_days": "integer",
    "retroactive_pay_limit_days": "integer",
    "arbitration_request_days": "integer"
  },
  "seniority": {
    "calculation_basis": "string",
    "termination_triggers": "string[]",
    "recall_period_formula": "string"
  },
  "breaks": {
    "relief_period_minutes": "integer",
    "meal_period_minutes": "integer",
    "threshold_hours_for_meal": "float"
  },
  "benefits": {
    "health_eligibility_hours_per_month": "integer | null",
    "pension_contribution_rate": "float | null",
    "has_401k": "boolean"
  },
  "exclusions": {
    "excluded_departments": "string[]",
    "excluded_classifications": "string[]"
  },
  "special_provisions": {
    "grandfathering_dates": "date[]",
    "store_specific_rules": "object[]"
  }
}
```

### Why This Enables Scaling

This schema allows **Template Questions** that work across all contracts:

| Template | Contract A Truth | Contract B Truth |
|----------|-----------------|-----------------|
| "How many days to file a grievance?" | 20 days | 14 days |
| "What is the Sunday premium?" | 1.25x | No Sunday premium |
| "Probation period length?" | 60 days | 90 days |

Write the question template once. Ground truth is dynamically slotted per contract.

---

## Part 10: Synthetic Benchmark Factory (With Ouroboros Fix)

### The Ouroboros Problem

**Risk**: Using an LLM (Extraction Agent) to generate ground truth ("Teacher") and an LLM (KARL) to answer it ("Student") creates circular dependency. If both make the same misinterpretation of an ambiguous clause, KARL scores 100% but both are wrong.

**Fix**: Dual-Model Adversarial Extraction with contested-field routing.

### The Pipeline (Fixed)

```
Contract X (raw PDF/text)
        |
        +------------------+------------------+
        |                                     |
        v                                     v
+-------------------+               +-------------------+
| Extraction Agent  |               | Extraction Agent  |
| Model A (GPT-4o)  |               | Model B (Claude)  |
+-------------------+               +-------------------+
        |                                     |
        v                                     v
    Schema_A                              Schema_B
        |                                     |
        +------------------+------------------+
                           |
                           v
                  +-------------------+
                  |   Schema Differ   |
                  +-------------------+
                           |
            +--------------+--------------+
            |                             |
            v                             v
    Agreed Fields                  Contested Fields
    (Auto-trusted)                 (Route to Human)
            |                             |
            v                             v
+-------------------+           +-------------------+
| Q&A Generator     |           | Human Resolution  |
| (Template + Agreed)|          | (O(ambiguity))    |
+-------------------+           +-------------------+
            |                             |
            +------------------+----------+
                               |
                               v
                    gold_set_contract_X.json
```

### Dual-Model Extraction

```python
def extract_schema_adversarial(contract_text: str) -> tuple[dict, dict, list]:
    """
    Run extraction with two different model families.
    Returns: (merged_schema, contested_fields, confidence_scores)
    """
    # Model A: GPT-4o
    schema_a = extract_with_model(contract_text, model="gpt-4o")
    
    # Model B: Claude 3.5 Sonnet (different model family)
    schema_b = extract_with_model(contract_text, model="claude-3-5-sonnet")
    
    # Diff the schemas
    agreed = {}
    contested = []
    
    for field in SCHEMA_FIELDS:
        val_a = schema_a.get(field)
        val_b = schema_b.get(field)
        
        if values_match(val_a, val_b):
            agreed[field] = val_a
        else:
            contested.append({
                "field": field,
                "model_a_value": val_a,
                "model_b_value": val_b,
                "needs_human": True
            })
    
    return agreed, contested
```

### Human Verification (O(ambiguity), not O(N))

The human only reviews:
1. **Contested fields** where Model A and Model B disagreed
2. **Null fields** where one model found a value and the other didn't
3. **Edge cases** explicitly flagged (grandfathering dates, exceptions)

They do NOT verify:
- Agreed fields (dual-model consensus provides confidence)
- Individual QA pairs (generated deterministically from schema)

### Benchmark Generation Rule

**Only generate synthetic questions from fields where:**
- Model A and Model B agreed, OR
- A human resolved the conflict

This prevents model-family bias from polluting ground truth.

### Q&A Generator Code Structure

```python
def generate_gold_set(contract_id: str, schema: dict, templates: list) -> list:
    """Generate gold QA pairs from schema + templates."""
    gold_set = []
    
    for template in templates:
        # Check if this template applies to this contract
        if not template_applies(template, schema):
            continue
            
        question = template["question"]
        answer_template = template["answer_template"]
        
        # Slot in contract-specific values
        answer = answer_template.format(**flatten_schema(schema))
        citation = get_citation_for_field(schema, template["schema_field"])
        
        gold_set.append({
            "question": question,
            "ground_truth": answer,
            "expected_citation": citation,
            "schema_field": template["schema_field"],
            "contract_id": contract_id
        })
    
    return gold_set
```

---

## Part 11: Multi-Tenant Failure Modes

When moving to 2+ contracts, new catastrophic errors emerge that v2 doesn't catch.

### 11.1 Cross-Contamination Test

**Scenario**: KARL is loaded with Contract A (UFCW Local 7 Pueblo) and Contract B (UFCW Local 770 LA). User asks about Contract A.

**Risk**: KARL retrieves chunks from Contract B because vector similarity is high (both discuss "grievances" in similar legal-ese).

**The Fix**: RAG pipeline must enforce **Strict Metadata Filtering**.

**Benchmark Protocol**:

```python
def test_cross_contamination():
    # Load index with both contracts
    index = load_index(["contract_a", "contract_b"])
    
    # Query with explicit context
    query = "What is the Sunday premium?"
    context_id = "contract_a"
    
    result = karl.retrieve(query, contract_filter=context_id)
    
    # PASS: Zero chunks from wrong contract
    for chunk in result.chunks:
        assert chunk.contract_id == context_id, \
            f"CRITICAL: Retrieved chunk from {chunk.contract_id}"
```

**Test Cases**:

| Query | Context | Trap | Pass Condition |
|-------|---------|------|----------------|
| "Sunday premium rate?" | Contract A | Contract B has better-matching phrasing | 0 chunks from B |
| "Grievance deadline?" | Contract B | Contract A uses exact same phrase | 0 chunks from A |
| "Overtime calculation" | Contract A | Both have "time and one-half" | Only A chunks |

### 11.2 Hallucinated Standardization Test

**Scenario**: 90% of contracts have 30-day probation. Contract Z has 60-day probation.

**Risk**: LLM ignores retrieved text for Contract Z and answers "30 days" because its parametric memory (or few-shot examples from other contracts) biases toward the mean.

**Benchmark Protocol**:

1. Identify **Outlier Contracts** (statistically rare values in Universal Schema)
2. Weight outlier questions higher in evaluation
3. Require exact-match on outlier values

```python
def identify_outliers(all_schemas: list) -> dict:
    """Find values that are statistical outliers across corpus."""
    outliers = {}
    
    for field in NUMERIC_FIELDS:
        values = [s.get(field) for s in all_schemas if s.get(field)]
        mean, std = np.mean(values), np.std(values)
        
        for schema in all_schemas:
            if abs(schema.get(field, mean) - mean) > 2 * std:
                outliers.setdefault(schema["contract_id"], []).append({
                    "field": field,
                    "value": schema[field],
                    "corpus_mean": mean
                })
    
    return outliers
```

**Test Cases** (auto-generated from outlier detection):

| Contract | Field | Outlier Value | Corpus Mean | Risk |
|----------|-------|---------------|-------------|------|
| Contract Z | probation_days | 60 | 30 | LLM defaults to mean |
| Contract Q | grievance_deadline | 7 | 20 | Unusually short |
| Contract M | sunday_premium | null | 1.25x | No premium (rare) |

---

## Part 12: Context-Aware Evaluation Architecture

### The Evaluator Function

Instead of static JSON, the evaluator is now parameterized:

```
Score = Evaluate(Answer, GroundTruth(context_id))
```

### Implementation

```python
class MultiContractEvaluator:
    def __init__(self):
        self.gold_sets = {}  # contract_id -> gold_set
        
    def load_contract(self, contract_id: str):
        """Load or generate gold set for a contract."""
        gold_path = f"data/test_set/gold_{contract_id}.json"
        if os.path.exists(gold_path):
            self.gold_sets[contract_id] = load_json(gold_path)
        else:
            # Generate from schema + templates
            schema = load_schema(contract_id)
            self.gold_sets[contract_id] = generate_gold_set(
                contract_id, schema, TEMPLATE_QUESTIONS
            )
            
    def run_eval(self, contract_id: str):
        """Run evaluation for a specific contract."""
        gold_set = self.gold_sets[contract_id]
        karl = KarlSystem(contract_filter=contract_id)
        
        results = []
        for q in gold_set:
            response = karl.ask(q["question"])
            
            # Standard grading
            grade = self.grade(response, q["ground_truth"])
            
            # CROSS-CONTRACT LEAKAGE CHECK
            if self.detect_leakage(response.source_chunks, contract_id):
                grade.score = 0
                grade.notes = "CRITICAL: Retrieval Leakage"
                
            results.append(grade)
            
        return results
        
    def detect_leakage(self, chunks: list, expected_contract: str) -> bool:
        """Check if any chunk came from wrong contract."""
        for chunk in chunks:
            if chunk.get("contract_id") != expected_contract:
                return True
        return False
```

---

## Part 13: Day 8 - Multi-Contract Pilot

### Objective

Take `comprehensive_test.json` and clone it for a second, different contract.

**Keep questions identical. Change expected answers.**

### Prerequisite (Must Pass Before Day 8)

Escalation precision hardening from Phase A must pass before starting multi-contract pilot work:
- Deterministic-first two-stage policy is in place (`high_stakes_topic` vs `active_urgent_context`)
- Conditional/hypothetical suppressor tests pass
- Confusion matrix and threshold tradeoff report approved
- Escalation false-positive rate remains at or below approved cap

### Protocol

1. **Select Contract B**: Choose a contract that is:
   - Same industry (grocery/retail) for comparability
   - Different local/employer for independence
   - Has different values for key fields (probation, grievance deadlines, premiums)

2. **Extract Schema B**: Run Extraction Agent on Contract B

3. **Generate Gold Set B**: Apply template questions to Schema B

4. **Run KARL on Both**:
   ```bash
   python evaluate.py --contract=pueblo_clerks    # Contract A
   python evaluate.py --contract=denver_meat      # Contract B
   ```

5. **Delta Analysis**:
   - Compare accuracy A vs B
   - Identify questions where A passes but B fails
   - Hypothesis: Prompt engineering overfitted to Contract A's phrasing

### Expected Findings

| Finding | Interpretation | Action |
|---------|---------------|--------|
| B accuracy 10%+ lower than A | Overfitting to A's phrasing | Generalize prompts |
| Cross-contamination detected | Metadata filtering broken | Fix retrieval filter |
| Outlier values wrong | Parametric memory override | Add grounding verification |
| Template questions don't apply | Schema missing fields | Extend schema |

---

## Part 14: Generator Validation Protocol

Before trusting the Synthetic Benchmark Factory for 100+ contracts, validate it on the 3 manually-benchmarked contracts.

### Protocol

1. **Manual Gold Sets**: Hand-write 30 questions each for Contracts A, B, C (as in Phase A)

2. **Generated Gold Sets**: Run Extraction Agent + Q&A Generator on same 3 contracts

3. **Alignment Check**:
   ```python
   def compute_alignment(manual: list, generated: list) -> float:
       """Compare manual vs generated gold sets."""
       matches = 0
       for m in manual:
           for g in generated:
               if questions_match(m.question, g.question):
                   if answers_equivalent(m.ground_truth, g.ground_truth):
                       matches += 1
                   break
       return matches / len(manual)
   ```

4. **Threshold**: Generator is trusted when alignment > 90%

5. **Failure Analysis**: For misaligned pairs, categorize:
   - Schema extraction error
   - Template question didn't cover this case
   - Edge case requiring human judgment

---

## Part 15: Template Question Library (Starter Set)

30 template questions that apply across most labor contracts:

### Wages & Premiums

1. "What is the starting wage for {classification}?"
2. "What is the top pay rate for {classification}?"
3. "How many hours until {classification} reaches top rate?"
4. "What is the overtime pay rate?"
5. "Is there a Sunday premium? If so, what rate?"
6. "What is the night shift premium?"
7. "What is the holiday pay rate?"

### Scheduling

8. "When must the schedule be posted?"
9. "What is the minimum shift length?"
10. "What is the maximum shift length at straight time?"
11. "How does schedule bidding work?"
12. "Can I be forced to stay past my scheduled shift?"

### Time Off

13. "How long is the probationary period?"
14. "How much vacation do I get after {years} years?"
15. "How many personal holidays do I get?"
16. "How many bereavement days for immediate family?"
17. "How does sick leave accrue?"

### Grievance & Discipline

18. "How many days to file a grievance?"
19. "How many days to file a discharge grievance?"
20. "What are my Weingarten rights?"
21. "How far back can retroactive pay go?"
22. "What is the arbitration process?"

### Seniority & Layoff

23. "How is seniority calculated?"
24. "What causes loss of seniority?"
25. "How does bumping work during layoff?"
26. "What is the recall period?"

### Benefits

27. "How many hours for health benefit eligibility?"
28. "Is there a pension plan?"
29. "Is there a 401k plan?"

### Scope

30. "What departments/classifications are excluded from this contract?"

---

## Success Metrics for Phase B

| Metric | Threshold | Notes |
|--------|-----------|-------|
| Extraction Agent accuracy | >95% on spot-checked fields | Per-contract |
| Generator alignment with manual | >90% | On 3 validation contracts |
| Cross-contamination rate | 0% | Zero chunks from wrong contract |
| Outlier value accuracy | >90% | For statistically rare values |
| Delta (Contract A - Contract B) | <10% | Performance consistency |
| Time to onboard new contract | <2 hours | With schema extraction + validation |

---

## Scaling Roadmap

| Phase | Contracts | Effort | Eval Type |
|-------|-----------|--------|-----------|
| A (v2) | 1 | 7 days | Manual, rigorous |
| B Pilot | 2-3 | 3 days | Semi-automated |
| B Scale | 10 | 1 week | Automated + spot-check |
| Production | 100+ | Ongoing | Fully automated |

---

## Audit Checklist for Phase B

- [ ] Universal Schema covers all template question fields
- [ ] Extraction Agent tested on 3 diverse contracts
- [ ] Contract metadata (contract_id) attached to every chunk
- [ ] Retrieval enforces metadata filtering
- [ ] Cross-contamination test passes for all contract pairs
- [ ] Outlier detection identifies rare values
- [ ] Generator alignment validated >90%
