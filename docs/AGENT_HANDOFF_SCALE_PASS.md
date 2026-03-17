# Agent Handoff: Scale & Cleanup Pass

## What this project is

KARL is a worker-controlled AI contract retrieval and steward tool for union locals. It is NOT a chatbot. It retrieves and surfaces collective bargaining agreement language for workers and stewards. Currently deployed for UFCW Local 7 across 3 contracts. Target scale: 100+ contracts, multiple locals.

**Do not use the terms "interpret", "interpreter", or "interpretation" anywhere. KARL retrieves contract language. It is a contract helper and steward tool.**

## Read first (in this order)

1. `RELEASE_0_9_0_READINESS.md` — current state, what's green, what's pending
2. `legal/RELEASE-GATES.md` — what gates must pass before any push
3. `legal/GOVERNANCE-CHARTER.md` — three-council model, worker-first constraints
4. `legal/DEPLOYMENT-POLICY.md` — self-hosted per-local, contract-scoped tenancy, no cross-local data sharing
5. `UPDATE_LOG.md` (last 200 lines) — recent changes, what just shipped in v0.9.0
6. `backend/retrieval/router.py` (skim structure, do not read every line) — understand the 5-phase CAG pipeline
7. `backend/miss_records.py` — miss record schema
8. `backend/ingest/pack_acceptance.py` (skim gate names) — know what gates exist

## Current branch

`release/v0.9.0-rc2` — all canonical evals green (55/55, 18/18, 10/10). Do not break this.

## Runtime LLM stack

KARL itself uses Gemini (Google Cloud) for answer generation, hypothesis, and reranking. You are the dev assistant. Do not change LLM providers.

---

## Tasks (in priority order)

### Task 1 — router.py decomposition (HIGHEST PRIORITY)

**Problem:** `backend/retrieval/router.py` is 3,829 lines. One file. Everything in it. At 20+ contracts with different routing behaviors it becomes unmaintainable and a contributor safety hazard.

**Goal:** Decompose into logical submodules WITHOUT changing any behavior. This is a structural refactor only — no logic changes.

**Proposed split:**
- `backend/retrieval/intent.py` — intent classification, high-stakes detection, topic routing
- `backend/retrieval/wage_resolution.py` — wage lookup orchestration, role aliasing, canonical wage binding
- `backend/retrieval/retrieval_strategy.py` — multi-angle retrieval, hybrid search coordination, article expansion
- `backend/retrieval/escalation.py` — escalation logic, two-stage high-stakes routing, suppressor conditions
- `backend/retrieval/router.py` — thin orchestrator that imports and calls the above

**Constraints:**
- All existing tests and evals must pass after refactor
- Do not rename public methods that are called from `api.py` or `evaluate_*.py`
- Do not change routing logic, only move it
- Run `python backend/evaluate_runner.py --track all` after each module extraction to confirm nothing broke
- Prefer one module at a time — extract, test, confirm, then next

**Acceptance bar:** All canonical eval tracks green. `router.py` under 500 lines as a thin orchestrator.

---

### Task 2 — Miss record triage workflow (CLOSE THE LOOP)

**Problem:** Miss record infrastructure exists (schema, creation script, integrity eval) but the loop isn't closed. A miss record is currently a file that sits in `data/miss_records/`. At scale with multiple stewards generating misses, this becomes a graveyard without a triage-to-regression path.

**Goal:** Add a lightweight triage pipeline that turns miss records into governed regression inputs.

**What exists:**
- `backend/miss_records.py` — schema, normalization, regression stub generation
- `scripts/create_miss_record.py` — creation CLI
- `backend/evaluate_miss_record_integrity.py` — integrity eval
- `data/miss_records/` — the records themselves (gitignored, sensitive)

**What's needed:**
1. A triage script (`scripts/triage_miss_records.py`) that:
   - Lists all open miss records with their status
   - Marks a record as `triaged`, `promoted_to_regression`, or `closed`
   - Writes a regression stub to the appropriate eval test set when promoted
2. A pack acceptance gate that checks: if a miss record exists for a contract, there must be a regression test covering it (or the record must be explicitly closed with a reason)
3. Update `backend/evaluate_miss_record_integrity.py` to include the triage coverage check

**Constraints:**
- Miss records are real user data — never log them to stdout in a way that exposes question content
- Regression stubs should be anonymized (behavior pattern, not verbatim question)
- Do not change the miss record schema — extend it with a `triage_status` field only if needed

**Acceptance bar:** `python scripts/triage_miss_records.py --list` shows all open records. `python scripts/triage_miss_records.py --promote <id>` writes a regression stub. Miss record integrity eval passes.

---

### Task 3 — Pack onboarding runbook (DELEGATABILITY)

**Problem:** Adding a new contract requires deep system knowledge. At 20+ contracts this is a bottleneck. The tooling exists — `scripts/onboard_contract_packages.py`, pack acceptance gates, manifest schema — but there's no documented path a staff rep or coordinator could follow.

**Goal:** Write `docs/CONTRACT_ONBOARDING_RUNBOOK.md` — a step-by-step guide for onboarding a new contract pack.

**What to cover:**
1. Required directory structure (`CONTRACT_PACK_SPEC_v1.md` is the schema spec — reference it)
2. Required files per contract (manifest, wage tables, role catalog, ontology, chunks)
3. How to run pack acceptance and interpret failures (`scripts/onboard_contract_packages.py`)
4. How to handle review queue and classification overrides (`scripts/apply_review_overrides.py`)
5. How to materialize the effective contract once amendments exist
6. How to run the relevant eval tracks to confirm the contract is ready
7. Common failure modes and how to fix them (missing pdf_page in source_refs, role catalog gaps, etc.)

**Constraints:**
- Write for a technically competent person who doesn't know the codebase internals
- Reference actual commands, not pseudocode
- Do not document implementation details — document the workflow
- Cross-reference `legal/DEPLOYMENT-POLICY.md` for the self-hosted deployment model

**Acceptance bar:** A person who has never touched the repo could follow the runbook to onboard a new contract and have it pass pack acceptance.

---

### Task 4 — Append-only audit log (GOVERNANCE & VISIBILITY)

**Problem:** No record of what was queried, what was retrieved, or what was returned. At scale, steward coordinators need visibility into coverage gaps. Governance requires accountability. This is the opposite of surveillance — it's worker-controlled auditability.

**Goal:** Add a lightweight append-only audit log for KARL answer events.

**Schema (per event):**
```json
{
  "timestamp": "ISO8601",
  "contract_id": "local7_safeway_pueblo_clerks_2022",
  "question_hash": "sha256 of normalized question",
  "topic_category": "wages|scheduling|discipline|benefits|...",
  "retrieval_score": 0.87,
  "answer_quality": "answered|escalated|unavailable",
  "high_stakes_triggered": false,
  "session_id": "opaque random token, not linked to any user identity"
}
```

**What NOT to log:**
- The actual question text (hash only)
- Any worker identity
- Answer content
- IP addresses

**Implementation:**
1. Add `backend/audit_log.py` — append-only writer, JSONL format, rotated daily
2. Wire into `api.py` `/query` endpoint — fire-and-forget, never block the response
3. Add `/api/karl/audit-summary` endpoint returning aggregate stats (topic distribution, answer quality rates, escalation rate) — no raw events exposed via API
4. Gitignore `data/audit_logs/`

**Constraints:**
- Logging must never slow down query responses (async, non-blocking)
- No PII ever — question hash only, opaque session token only
- Summary endpoint is read-only, aggregate only
- Must be disableable via config flag for locals that don't want it

**Acceptance bar:** A query generates an audit log entry. `/api/karl/audit-summary` returns topic distribution. No question text or identity in any log file.

---

### Task 5 — Chunk provenance field consistency (MATERIALIZER HARDENING)

**Problem:** Effective contract chunks in `index_inputs/` have `source_type` set at the top level (e.g., `"moa"`) but `source_pdf`, `source_doc_id`, and `pdf_page` are `None`. The provenance array has the full data. This causes the PDF navigation to rely on the provenance array for URL construction — which works, but is fragile. Any caller using only top-level chunk fields gets the wrong PDF.

**Specific example:** `contract_chunks_enriched_local7_safeway_pueblo_clerks_2022.json` in the clerks effective index_inputs — Article 15 Section 34 (Night Premiums) has `source_type: "moa"`, `source_doc_id: None`.

**Goal:** In the materializer, when writing index_inputs chunks, populate `source_pdf`, `source_doc_id`, and `pdf_page` at the chunk top level from the best matching provenance ref when they are absent.

**Logic:**
- If chunk has `source_type: "moa"` and `source_doc_id` is None → find the MOA provenance ref → copy its `pdf`, `source_doc_id`, `pdf_page` to the chunk top level
- Apply only to effective contract index_inputs chunks, not base contract chunks
- Do not overwrite fields that are already populated

**Files to touch:**
- `backend/ingest/materializer.py` — where index_inputs chunks are written
- Possibly `backend/ingest/materialize_effective.py`

**After fixing materializer:** Re-materialize the clerks and meat effective contracts and confirm the chunk fields are populated. Run pack acceptance on both contracts.

**Acceptance bar:** Article 15 Section 34 in clerks index_inputs chunk has `source_doc_id: "albertsons_safeway_moa_2025_07_05"` and `pdf_page: 8`. Pack acceptance passes. Canonical evals pass.

---

## Hard constraints (never violate these)

- No weakening of contract-scoped tenant isolation — one local's data must never appear in another local's retrieval
- No cross-local raw data sharing of any kind
- Deterministic and schema-driven fixes over prompt patches
- Every behavioral change must have regression coverage before it's done
- Preserve current API surface — `api.py` endpoints must not change signatures
- Do not add dependencies without a clear reason — keep requirements/base.txt lean
- Run the narrowest relevant tests first, then canonical tracks if the change touches retrieval or generation

## Validation sequence

After each task:
1. `python scripts/karl.py smoke` — API health check
2. `python backend/evaluate_runner.py --track canonical` — must stay green
3. If retrieval or generation touched: `python backend/evaluate_runner.py --track all`
4. If pack structure touched: `python scripts/onboard_contract_packages.py --validate-existing`

## What success looks like

- router.py is a thin orchestrator under 500 lines with logical submodules
- A miss record can be promoted to a regression test in one command
- A staff rep can onboard a new contract following the runbook without help
- Every query generates a non-blocking audit log entry with no PII
- MOA chunk top-level fields match provenance array data
- All canonical eval tracks still green
- No new dependencies added unnecessarily
