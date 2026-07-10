# KARL — Knowledge Assistant for Rights and Labor

KARL helps union members and stewards understand, navigate, and enforce their collective
bargaining agreements. Ask a question in plain language — *"what's the Sunday premium for a
courtesy clerk?"* — and KARL answers from the contract itself, with citations to the article,
section, and page it came from.

KARL is two things in one repository:

1. **A contract-intelligence engine** — an amendment-aware retrieval and generation pipeline
   over CBA text, wage tables, MOAs, and side letters, guarded by a release-gated evaluation
   suite that blocks shipping when integrity checks fail.
2. **A production platform** — a multi-tenant deployment layer for unions: per-local isolation
   enforced by PostgreSQL row-level security, server-managed sessions, admin tooling, document
   ingestion, and privacy governance designed so that the people running KARL cannot quietly
   become the people surveilling its users.

Current deployment focus: **UFCW Local 7**, with three onboarded contracts — Safeway Pueblo
Clerks (2022–2025), Safeway Pueblo Meat (2022), and King Soopers Loveland Meat (2019–2022) —
including the July 2025 Safeway MOA as a materialized effective contract.

## Core Principles

- **Union-first.** Built by and for the labor side. Licensing, governance, and feature policy
  all encode this (see [Licensing](#license) and `legal/ETHICAL-USE.md`).
- **Worker-controlled.** Unions run their own instance, choose their own model provider, and
  control their own data. Governance councils (`legal/GOVERNANCE-CHARTER.md`) hold release and
  data-policy authority.
- **Privacy-respecting, verified — not just claimed.** Message retention is **off by default**.
  Telemetry defaults to anonymized with a real member opt-out that suppresses writes entirely.
  Members can delete their own data (`DELETE /api/member/me/data`, 24-hour SLA). Tenant
  isolation is enforced by Postgres row-level security and verified by an integration suite
  against live Postgres — not just application-layer filtering.
- **Anti-surveillance.** No employer-facing dashboards, no productivity scoring, no
  management-side analytics. Contributions that enable them are rejected
  (`legal/CONTRIBUTING.md`).
- **Transparent by design.** This README states plainly what the benchmarks do and don't
  measure, and where member questions travel.

## Where Your Questions Go (Data Flow)

Honesty about this matters more than the marketing value of omitting it:

- **Answering a question calls an external LLM API.** In the default single-tenant
  configuration, member questions and retrieved contract excerpts are sent to **Google's Gemini
  API** for answer generation, query interpretation, and reranking. Google's API data-handling
  terms apply to that traffic.
- **In platform mode, each union chooses its own provider** — any OpenAI-compatible endpoint
  (OpenRouter, a self-hosted model server, etc.) configured per-union with encrypted credentials.
  A union that wants zero third-party exposure can point KARL at infrastructure it controls.
- **KARL itself retains as little as possible.** Chat messages are not stored unless a union
  explicitly enables retention. Raw query storage is disabled by default and gated behind
  super-admin controls with audit and security events. Telemetry is anonymized by HMAC with no
  stored user id, and members can opt out entirely.
- **Deterministic paths stay local.** Wage lookups and entitlement lookups resolve from
  structured tables on the server without any LLM call.

## Evaluation: What the Numbers Mean

KARL's evaluation system is release infrastructure, not a scoreboard: CI runs the suites on
every pull request, and `backend.evaluate_gate_check` blocks release when any gated metric
falls below its floor. Read the numbers with these definitions:

- **Retrieval hit-rate** means the expected citation appeared in the retrieved context. It does
  **not** mean the generated answer was correct, complete, or safe.
- **Deterministic integrity checks** verify artifact and behavior invariants (isolation,
  contamination, abstention, provenance) with exact assertions — these are the strongest
  guarantees KARL makes.
- **Graded answer quality** (LLM-judged generation evaluation) exists in the suite but is the
  **least mature layer**. KARL does not currently claim a validated end-to-end answer-accuracy
  number.

### Benchmark tracks

1. **v1 (legacy, historical).** A 55-case golden set from early development. The historical
   "100% (55/55)" result is a **development-set retrieval hit-rate**: the set was used while
   building the system, several fixes were tuned against individual cases, and the metric is
   retrieval, not answer quality. It is kept for continuity, and should not be quoted as an
   accuracy claim. (`python -m backend.evaluate_runner --track v1`)
2. **v2 (comprehensive).** A harder retrieval benchmark where v1's ceiling drops to the
   mid/high-80% range — a more honest picture of retrieval quality under stress, same
   hit-rate caveat. (`python -m backend.evaluate_runner --track v2 --ablation-mode normal`)
3. **v3 (canonical, release-gated).** The multi-contract integrity suite that actually gates
   releases: contract isolation and cross-contamination checks, unanswerable/abstention slices,
   adversarial precedence, cross-contract mention abstention, false-unavailable recovery with
   negative controls, needle retrieval, wage/entitlement table evidence, MOA
   deleted-vs-updated regression, and role-catalog integrity — with per-slice thresholds
   enforced in CI. (`python -m backend.evaluate_runner --track v3`; plan:
   `Evaluation_Plan_v3.md`)

Real-user misses are captured as structured records (`backend/miss_records.py`) and promoted
into regression tests — the correction pathway is governed, not ad hoc.

## Architecture

### Contract-intelligence engine (single-tenant runtime)

A 5-phase CAG design currently running a **lean configuration** by default
(`backend/config.py`):

- Phase 1 (intent routing): enabled
- Phase 2 (hypothesis layer): disabled
- Phase 3 (full article expansion): disabled
- Phase 4 (query interpreter): enabled
- Phase 5 (LLM reranker): enabled
- BM25 fusion: disabled in lean mode (vector-first retrieval)

Feature highlights:

- Citation-focused responses with interactive citation navigation and PDF page anchors
- Amendment-aware retrieval: MOA patches materialize into effective contracts with full
  base+MOA provenance per section
- Deterministic wage and vacation-entitlement lookups from structured tables with
  `table_evidence` metadata
- Deterministic two-stage high-stakes routing (`high_stakes_topic` vs `active_urgent_context`)
  with conditional/hypothetical escalation suppressors
- Query interpreter for vocabulary bridging (member slang → contract language) and an LLM
  reranker for relevance ordering
- Manifest-driven routing metadata, contract-scoped concept indexes and language lexicons

### Production platform (multi-tenant)

`backend/platform/` adds the deployment layer, with schema managed by Alembic
(`alembic/versions/`):

- **Tenant isolation:** PostgreSQL row-level security on all tenant tables, including
  telemetry, raw-query, and session tables — verified end-to-end by
  `backend/test_platform_postgres_rls.py` against live Postgres (migrated from an empty schema,
  exercised as an unprivileged role)
- **Auth:** server-managed cookie sessions with role-tiered idle windows (member / union admin
  / super admin), revocation, login rate limits, and session-metadata retention purging
- **Privacy governance:** tracking policies with anonymized defaults and member preference
  tiers, message non-retention by default, complete user-data deletion (admin and member
  self-service), PII redaction guardrails at prompt and document boundaries, audit and
  security event trails
- **Per-union model config:** encrypted provider credentials, any OpenAI-compatible endpoint
- **Operations:** ingestion worker with job prioritization, document parsing/OCR retry,
  quotas and hard caps, ops dashboard, maintenance endpoints
- **Frontends:** member chat (`frontend/modular/`), embeddable member widget
  (`frontend/embed/karl-member.js`), union admin and super-admin consoles

Platform docs: `docs/PRODUCTION_FOUNDATION.md`, `docs/PRODUCTION_HANDOFF.md`,
`docs/DEMO_RUNBOOK.md`. Privacy/governance posture and its verification record:
`docs/PRIVACY_GOVERNANCE_REMEDIATION_PLAN.md`,
`legal/DATA-STEWARDSHIP-COUNCIL-SIGNOFF.md`.

## Model Stack

Default single-tenant pipeline:

| Component | Model |
|---|---|
| Answer generation | Gemini 2.5 Pro |
| Query interpreter | Gemini 2.5 Flash |
| Reranker | Gemini 2.5 Flash |
| Enricher (ingestion) | Gemini 2.5 Flash |
| Embeddings | all-MiniLM-L6-v2 (local) |

Platform mode: per-union provider configuration (OpenRouter / OpenAI-compatible / self-hosted),
with configurable embedding backends (`KARL_EMBEDDING_BACKEND`).

## Repository Structure

```text
karl/
├── backend/
│   ├── api.py                  # FastAPI app (engine + platform routers)
│   ├── config.py               # engine configuration
│   ├── ingest/                 # chunking, enrichment, indexing, MOA materialization
│   ├── retrieval/              # vector store, interpreter, reranker, hypothesis
│   ├── generation/             # prompts, context, citation verifier
│   ├── eval/                   # graders, entailment, precedence
│   ├── evaluate_*.py           # evaluation tracks (see Evaluation Commands)
│   └── platform/               # multi-tenant platform layer (auth, RLS, telemetry, ingestion)
├── alembic/                    # platform database migrations
├── data/
│   ├── contracts/              # per-contract packages (source, chunks, effective snapshots)
│   ├── chunks/ wages/ manifests/ ontologies/ tables/
│   └── test_set/               # evaluation datasets and result artifacts
├── frontend/
│   ├── modular/                # member app, admin and super-admin consoles
│   └── embed/                  # embeddable member widget
├── docs/                       # setup, demo runbook, production foundation/handoff
├── legal/                      # licenses, CLA, governance charter, policies
├── Evaluation_Plan_v3.md
├── CONTRACT_PACK_SPEC_v1.md
└── UPDATE_LOG.md
```

## Quick Start

Choose one setup path:

### Option A (Recommended): Docker / Dev Container

```bash
python scripts/karl.py docker-up --build
```

Direct Docker Compose also works:

```bash
docker compose -f docker-compose.dev.yml up --build
```

VS Code devcontainer is also available via `.devcontainer/devcontainer.json`.

### Option B: Windows Native (PowerShell)

Use the scripted setup path (recommended over manual `pip install`):

```powershell
python scripts/karl.py setup --profile backend
```

Start the local API:

```powershell
python scripts/karl.py start
```

Then validate runtime/API + Contract-tab endpoints:

```powershell
python scripts/karl.py smoke
```

Detailed Windows setup + troubleshooting: `docs/LOCAL_SETUP_WINDOWS.md`

### 1. Install dependencies (manual path)

For faster local installs, prefer dependency profiles instead of the full default:

```bash
# Runtime/API
pip install -r requirements/base.txt

# Eval tooling
pip install -r requirements/eval.txt
```

Preflight checks (recommended before/after setup):

```bash
python scripts/karl.py doctor --profile backend
```

`requirements.txt` still works, but profile-based installs are faster and easier to debug.

### 2. Configure API key (optional but recommended)

```bash
# Windows PowerShell
Set-Content -Path .env -Value "GEMINI_API_KEY=your_actual_api_key_here"

# Linux/Mac
printf "GEMINI_API_KEY=your_actual_api_key_here\n" > .env
```

Without an API key, KARL can still retrieve chunks and perform wage lookups, but it cannot
produce synthesized LLM answers. Never commit `.env`.

### 3. Build or refresh data (if needed)

```bash
python -m backend.ingest.smart_chunker
python -u -m backend.ingest.enricher --batch-size 15 --delay 1.0
python -m backend.ingest.rebuild_index
```

### 3b. Refresh MOA-effective artifacts after amendment changes (demo-safe)

If you updated MOA patch files or source docs and need current-effective demo behavior
(especially Appendix A wage changes), rematerialize the effective snapshot and validate
the effective wage artifact chronology:

```bash
# Optional (Pueblo Clerks / July 2025 Safeway MOA):
# rebuild full Appendix A wage row patch ops from data/source_docs/moa/.../output.json
python -m backend.ingest.sync_clerks_moa_appendix_from_output

python -m backend.ingest.materialize_effective --contract-id <contract_id>
python -m backend.evaluate_effective_wage_coverage --contract-id <contract_id>
```

This catches cases where an MOA wage patch was applied but no row was materialized at the patch
effective date. For the Pueblo Clerks July 2025 MOA, the sync command expands the approved
patch to all 52 Appendix A rows using the structured `output.json` Denver Metro Clerks tables
(`Current` column) before rematerialization.

### 4. Start server

```bash
python scripts/karl.py start
```

### 5. Open app

Go to `http://127.0.0.1:8000`.

### Platform mode (multi-tenant)

Platform features activate when a PostgreSQL URL is configured
(`KARL_POSTGRES_URL`; see `backend/platform/settings.py` for the full `KARL_*` variable set —
origins, rate limits, storage, embeddings, session windows, encryption key). Run migrations
with Alembic (`alembic upgrade head`). Setup, demo flow, and production checklists:
`docs/PRODUCTION_FOUNDATION.md` and `docs/DEMO_RUNBOOK.md`.

> Production note: set `KARL_ALLOWED_ORIGINS` explicitly and provide a real
> `KARL_SECRET_ENCRYPTION_KEY` — the development defaults are not production-safe.

## Query Contract Context

`/api/query` requires explicit contract context:
- `contract_id` (required)
- `union_local_id` (required, must match manifest `union_local`)
- `contract_version` (required, must match manifest version string)

Current manifest version format:
- `contract_version = "<term_start>__<term_end>"`
- Example: `January 23, 2022__January 18, 2025`

Chunk artifact resolution prefers per-contract files before shared fallback:
- `data/chunks/contract_chunks_enriched_<contract_id>.json`
- `data/chunks/contract_chunks_smart_<contract_id>.json`
- `data/chunks/contract_chunks_<contract_id>.json`
- fallback: shared `contract_chunks_enriched.json`, `contract_chunks_smart.json`, `contract_chunks.json`

Wage artifacts remain separate and deterministic:
- `data/wages/wage_tables_<contract_id>.json`
- fallback: shared `data/wages/wage_tables.json`

### Contract Catalog API

Frontend/runtime contract selection should use:

```bash
GET /api/contracts
```

Response includes:
- `default_contract_id`
- `contracts[]` with:
  - `contract_id`
  - `union_local_id`
  - `contract_version`
  - `employer`
  - `term_start`, `term_end`

Contract-scoped classification options are available via:

```bash
GET /api/classifications?contract_id=<contract_id>
```

When available, classification options are loaded from the contract-scoped
`role_catalog_<contract_id>.json` artifact, and default onboarding options are
restricted to wage-resolvable roles.

Contract-aware health/status can be queried via:

```bash
GET /api/health?contract_id=<contract_id>
```

Default contract resolution order in `backend/config.py`:
1. `KARL_CONTRACT_ID` or `CONTRACT_ID` env override
2. legacy benchmark default (`safeway_pueblo_clerks_2022`) when present
3. first manifest in `data/manifests/`

## Contract Onboarding

Process contract packages from `data/contracts/<package>/source/` into runtime artifacts:

```bash
python scripts/onboard_contract_packages.py --package local7_safeway_pueblo_meat_2022 --package local7_kingsoopers_loveland_meat_2019
```

The onboarding script generates and syncs:
- manifests (`data/manifests/<contract_id>.json`)
- chunks (`data/chunks/contract_chunks_*_<contract_id>.json`)
- structured tables (`data/tables/structured_tables_<contract_id>.json`, when JSON source exists)
- synthesized table chunks for unmatched structured tables (appendix/wage-table retrieval coverage)
- wage tables (`data/wages/wage_tables_<contract_id>.json`, deterministic table-registry first, markdown fallback)
- classification ontology (`data/ontologies/classification_ontology_<contract_id>.json`)
- role catalog (`data/ontologies/role_catalog_<contract_id>.json`)

Contract-pack scorecards run by default and are written per package:
- `data/contracts/<contract_id>/pack/pack_scorecard.json`
- Scorecard includes canonical wage-row schema checks (`canonical_wage_rows`), classification ontology checks, and role-catalog checks.

Package-local review loop artifacts:
- `data/contracts/<contract_id>/ontology/ingestion_review_queue.json` (auto-generated unresolved/ambiguous review queue)
- `data/contracts/<contract_id>/ontology/manual_classification_overrides.json` (human-edited deterministic alias overrides consumed on next onboarding run)

Review override workflow:

```bash
# Generate decision template from unresolved ontology mappings
python scripts/apply_review_overrides.py --contract-id <contract_id> --emit-template

# Preview coverage/diff impact of reviewed decisions
python scripts/apply_review_overrides.py --contract-id <contract_id> --decision-file data/contracts/<contract_id>/ontology/review_decisions_template.json

# Apply reviewed decisions into manual override file
python scripts/apply_review_overrides.py --contract-id <contract_id> --decision-file <reviewed_decisions.json> --apply
```

Useful onboarding flags:

```bash
# Block runtime sync when required pack gates fail
python scripts/onboard_contract_packages.py --package <contract_id> --enforce-pack-gates

# Treat advisory failures as blocking
python scripts/onboard_contract_packages.py --package <contract_id> --enforce-pack-gates --strict-pack-gates
```

Run pack acceptance independently:

```bash
python -m backend.ingest.pack_acceptance --package <contract_id>
python -m backend.ingest.pack_acceptance --package <contract_id> --strict
```

Deterministic ingestion outputs include:
- Contract-scoped `concept_index_<contract_id>.json` with non-empty concept/question mappings
- Contract-scoped `language_lexicon_<contract_id>.json` (frozen alias graph)
- Contract-scoped `role_catalog_<contract_id>.json` (wage-availability-aware onboarding role catalog)
- Contract-scoped canonical wage rows in `wage_tables_<contract_id>.json` (`canonical_wage_rows`) with table-backed source references
- Contract-scoped entitlement schedules in `entitlement_tables_<contract_id>.json` (`vacation_entitlements`)
- Contract-scoped manifest `query_routing` synthesized from ingestion artifacts (`topic_to_articles`, `topic_patterns`, `slang_to_contract`, `classification_to_articles`)
- `region_id` on manifests/chunks for hard tenancy filtering (`contract_id` + `region_id`)

Runtime wage lookup resolves canonical rows first and includes structured
`table_evidence` metadata in wage responses when table source references exist.

Runtime query expansion order is deterministic:
1. universal slang map
2. frozen contract language lexicon (`alias_to_canonical`)
3. manifest `query_routing.slang_to_contract` overrides

## Evaluation Commands

### Canonical runner (recommended)

```bash
# Benchmark v1 (legacy retrieval hit-rate; historical)
python -m backend.evaluate_runner --track v1

# Benchmark v2 (comprehensive retrieval)
python -m backend.evaluate_runner --track v2 --ablation-mode normal

# Escalation precision slice
python -m backend.evaluate_runner --track escalation

# Multi-contract deterministic slice
python -m backend.evaluate_runner --track v2_multi_contract

# Benchmark v3 (canonical multi-contract phase suite)
python -m backend.evaluate_runner --track v3

# Paraphrase/slang robustness slice
python -m backend.evaluate_runner --track paraphrase

# Adversarial formal-precedence slice (specific-over-general retrieval)
python -m backend.evaluate_runner --track adversarial

# Adversarial dataset integrity preflight
python -m backend.validate_adversarial_dataset

# Multi-contract unanswerable/abstention slice
python -m backend.evaluate_runner --track unanswerable

# Unanswerable dataset integrity preflight
python -m backend.validate_unanswerable_dataset

# Cross-contract entity mention abstention slice
python -m backend.evaluate_runner --track cross_contract_mentions

# Cross-contract mention dataset integrity preflight
python -m backend.validate_cross_contract_mentions_dataset

# Formal/non-slang routing drift checks (PF-01/PF-15-style phrasing)
python -m backend.test_topic_routing

# False-unavailable guard slice (forced unavailable first pass + deterministic recovery)
python -m backend.evaluate_runner --track false_unavailable

# MOA deleted-vs-updated regression slice (deleted clauses absent; updated MOA clauses retrievable)
python -m backend.evaluate_runner --track moa_deleted_vs_updated

# MOA deleted-vs-updated end-to-end answer regression slice
python -m backend.evaluate_runner --track moa_deleted_vs_updated_answer

# False-unavailable dataset integrity preflight
python -m backend.validate_false_unavailable_dataset

# Needle retrieval integrity slice
python -m backend.evaluate_runner --track needle

# Wage-table evidence integrity slice
python -m backend.evaluate_runner --track wage_table_evidence

# Entitlement-table evidence integrity slice
python -m backend.evaluate_runner --track entitlement_table_evidence

# Role-catalog integrity slice (contract-scoped role containment + default wage readiness)
python -m backend.evaluate_runner --track role_catalog_integrity

# Follow-up role-targeted wage integrity slice
python -m backend.evaluate_runner --track followup_role_wage

# Real-user regression slice
python -m backend.evaluate_runner --track real_user_regressions

# v0.9.0 readiness scorecard (aggregated must-have gates)
python -m backend.evaluate_runner --track release_090
```

False-unavailable evaluation includes both:
- evidence-present recovery cases
- evidence-absent negative controls that must preserve uncertainty

`backend.evaluate_runner` performs manifest validation preflight before running the selected track.

### Legacy v1 benchmark

```bash
python -m backend.evaluate
```

### v2 comprehensive benchmark

```bash
python -m backend.evaluate_comprehensive --ablation-mode normal
```

### Compare ablations

```bash
python -m backend.evaluate_comprehensive --compare baseline.json ablation.json
```

### Escalation precision slice

```bash
python -m backend.evaluate_escalation_precision
```

### Needle retrieval slice

```bash
# Synthetic needles are overlaid deterministically during this evaluator run.
python -m backend.evaluate_needle --bm25-only
```

`backend.evaluate_gate_check` treats needle thresholds as blocking when
`data/test_set/needle_results.json` is missing or below floor.

### Wage-table evidence slice

```bash
python -m backend.evaluate_wage_table_evidence --bm25-only
```

`backend.evaluate_gate_check` treats wage-table-evidence thresholds as blocking when
`data/test_set/wage_table_evidence_results.json` is missing or below floor.

### Effective wage snapshot coverage (MOA chronology integrity)

```bash
python -m backend.evaluate_effective_wage_coverage
python -m backend.evaluate_effective_wage_coverage --contract-id <contract_id>
```

Validates that approved MOA wage-table row patches materialize rows at the patch effective date
in the latest effective snapshot wage artifact.

### Entitlement-table evidence slice

```bash
python -m backend.evaluate_entitlement_table_evidence
```

`backend.evaluate_gate_check` treats entitlement-table-evidence thresholds as blocking when
`data/test_set/entitlement_table_evidence_results.json` is missing or below floor.

### Role-catalog integrity slice

```bash
python -m backend.evaluate_role_catalog_integrity
```

`backend.evaluate_gate_check` treats role-catalog-integrity thresholds as blocking when
`data/test_set/role_catalog_integrity_results.json` is missing or below floor.

### Follow-up role-targeted wage slice

```bash
python -m backend.evaluate_followup_role_wage --bm25-only
```

`backend.evaluate_gate_check` treats followup-role-wage thresholds as blocking when
`data/test_set/followup_role_wage_results.json` is missing or below floor.

### MOA deleted-vs-updated regression slice

```bash
python -m backend.evaluate_moa_deleted_vs_updated --bm25-only
```

Writes:
- `data/test_set/moa_deleted_vs_updated_results.json`

### MOA deleted-vs-updated answer regression slice

```bash
python -m backend.evaluate_moa_deleted_vs_updated_answer --bm25-only
```

Writes:
- `data/test_set/moa_deleted_vs_updated_answer_results.json`

### v0.9.0 readiness scorecard

```bash
python -m backend.evaluate_release_090
```

Writes deterministic release scorecard artifact:
- `data/test_set/release_0_9_0_scorecard.json`

### Release-gate check from artifacts

```bash
python -m backend.evaluate_gate_check \
  --v2-results data/test_set/comprehensive_results.json \
  --escalation-results data/test_set/escalation_precision_results.json \
  --cross-contamination-results data/test_set/cross_contamination_results.json \
  --v3-results data/test_set/v3_results.json \
  --paraphrase-results data/test_set/paraphrase_results.json \
  --adversarial-results data/test_set/adversarial_results.json \
  --unanswerable-results data/test_set/unanswerable_results.json \
  --cross-contract-mentions-results data/test_set/cross_contract_mentions_results.json \
  --false-unavailable-results data/test_set/false_unavailable_results.json \
  --wage-table-evidence-results data/test_set/wage_table_evidence_results.json \
  --entitlement-table-evidence-results data/test_set/entitlement_table_evidence_results.json \
  --role-catalog-integrity-results data/test_set/role_catalog_integrity_results.json \
  --followup-role-wage-results data/test_set/followup_role_wage_results.json \
  --moa-deleted-vs-updated-results data/test_set/moa_deleted_vs_updated_results.json \
  --release-090-results data/test_set/release_0_9_0_scorecard.json \
  --min-v3-components-pass-rate 1.00 \
  --min-paraphrase-formal-rewrite-pass-rate 0.90 \
  --required-adversarial-dataset-schema-version adversarial_precedence_test_v1 \
  --min-adversarial-total-cases 12 \
  --min-adversarial-cases-per-contract 3 \
  --min-adversarial-precedence-cases 4 \
  --min-adversarial-pass-rate 0.90 \
  --min-adversarial-per-contract 0.80 \
  --min-adversarial-precedence-pass-rate 0.90 \
  --required-unanswerable-dataset-schema-version unanswerable_multi_contract_test_v1 \
  --min-unanswerable-total-cases 12 \
  --min-unanswerable-cases-per-contract 3 \
  --min-unanswerable-scenario-types 3 \
  --min-unanswerable-pass-rate 0.90 \
  --min-unanswerable-per-contract 0.80 \
  --required-cross-contract-mentions-dataset-schema-version cross_contract_mentions_test_v1 \
  --min-cross-contract-mentions-total-cases 9 \
  --min-cross-contract-mentions-cases-per-contract 3 \
  --min-cross-contract-mentions-pass-rate 0.90 \
  --min-cross-contract-mentions-per-contract 0.80 \
  --min-cross-contract-mentions-no-citation-rate 0.90 \
  --required-false-unavailable-dataset-schema-version false_unavailable_test_v1 \
  --min-false-unavailable-total-cases 12 \
  --min-false-unavailable-recover-cases 9 \
  --min-false-unavailable-uncertain-cases 3 \
  --min-false-unavailable-cases-per-contract 3 \
  --min-false-unavailable-recovered-rate 0.90 \
  --min-false-unavailable-proper-uncertainty-rate 0.90 \
  --required-followup-role-wage-dataset-schema-version followup_role_wage_test_v1 \
  --min-followup-role-wage-total-cases 12 \
  --min-followup-role-wage-cases-per-contract 3 \
  --min-followup-role-wage-pass-rate 0.90 \
  --min-followup-role-wage-per-contract 0.80 \
  --min-followup-role-wage-target-resolution-rate 0.95 \
  --min-followup-role-wage-table-evidence-presence-rate 0.95 \
  --min-followup-role-wage-appendix-citation-rate 0.95 \
  --min-followup-role-wage-intent-wage-rate 0.95 \
  --min-followup-role-wage-no-unavailable-rate 0.95 \
  --min-followup-role-wage-explicit-override-rate 0.90 \
  --min-followup-role-wage-profile-fallback-rate 0.90 \
  --required-moa-deleted-vs-updated-schema-version moa_deleted_vs_updated_eval_v1 \
  --required-moa-deleted-vs-updated-dataset-schema-version moa_deleted_vs_updated_test_v1 \
  --min-moa-deleted-vs-updated-overall-pass-rate 1.00 \
  --min-moa-deleted-vs-updated-updated-pass-rate 1.00 \
  --min-moa-deleted-vs-updated-deleted-pass-rate 1.00 \
  --min-moa-deleted-vs-updated-updated-moa-source-type-match-rate 1.00 \
  --required-release-090-schema-version release_090_scorecard_v2 \
  --min-release-090-components-pass-rate 1.00 \
  --required-wage-table-evidence-dataset-schema-version wage_table_evidence_test_v1 \
  --min-wage-table-evidence-total-cases 12 \
  --min-wage-table-evidence-cases-per-contract 3 \
  --min-wage-table-evidence-pass-rate 0.90 \
  --min-wage-table-evidence-per-contract 0.80 \
  --min-wage-table-evidence-source-method-pass-rate 0.95 \
  --min-wage-table-evidence-presence-rate 0.95 \
  --min-wage-table-evidence-table-id-presence-rate 0.95 \
  --required-entitlement-table-evidence-dataset-schema-version entitlement_table_evidence_test_v1 \
  --min-entitlement-table-evidence-total-cases 12 \
  --min-entitlement-table-evidence-cases-per-contract 3 \
  --min-entitlement-table-evidence-pass-rate 0.90 \
  --min-entitlement-table-evidence-per-contract 0.80 \
  --min-entitlement-table-evidence-weeks-resolution-pass-rate 0.90 \
  --min-entitlement-table-evidence-source-method-pass-rate 0.95 \
  --min-entitlement-table-evidence-presence-rate 0.95 \
  --required-role-catalog-integrity-dataset-schema-version role_catalog_integrity_test_v1 \
  --min-role-catalog-integrity-total-cases 12 \
  --min-role-catalog-integrity-cases-per-contract 3 \
  --min-role-catalog-integrity-pass-rate 0.95 \
  --min-role-catalog-integrity-per-contract 0.90 \
  --min-role-catalog-integrity-dataset-case-pass-rate 0.95 \
  --min-role-catalog-integrity-default-wage-ready-rate 1.00 \
  --min-role-catalog-integrity-unresolved-not-default-rate 1.00 \
  --min-role-catalog-integrity-default-wage-key-unique-rate 1.00 \
  --needle-results data/test_set/needle_results.json
```

### Manifest schema validation

```bash
python -m backend.validate_manifests
```

### CI behavior

- Pull requests:
  - `pr-core-eval`: manifest validation + canonical slices (`v2`, `escalation`, `v2_multi_contract`, `paraphrase`, `adversarial`, `unanswerable`, `cross_contract_mentions`, `false_unavailable`, `moa_deleted_vs_updated`, `moa_deleted_vs_updated_answer`, `needle`, `wage_table_evidence`, `entitlement_table_evidence`, `role_catalog_integrity`, `followup_role_wage`, `v3`, `release_090`) + isolation/cross-contamination/topic-routing checks + gate-check thresholds
  - `pr-moa-release-gate`: dedicated MOA release gate (`moa_deep_suite`)
- Push to `main`: manifest validation + full v2 ablation suite + deterministic release slices (`v3`, `release_090`) + cross-contamination/topic-routing + gate-check job

`backend.evaluate_cross_contamination` skips automatically when only one manifest is present.

### Branch protection baseline (GitHub)

For branch `main`, set a protection rule with:

- Require pull request before merging
- Require branches to be up to date before merging
- Require status checks to pass before merging:
  - `KARL Evaluation CI / pr-core-eval`
  - `KARL Evaluation CI / pr-moa-release-gate`
- Include administrators (recommended)

### Contract isolation check

```bash
python -m backend.test_contract_isolation
```

### Query-routing ingestion check

```bash
python -m backend.test_query_routing_ingest
```

### Unavailable-answer recovery check

```bash
python -m backend.test_unavailability_recovery
```

### Cross-contamination evaluator (multi-contract scaffold)

```bash
python -m backend.evaluate_cross_contamination
```

### Platform test suites

```bash
# Default suite (SQLite in-memory; RLS is a no-op here)
python -m pytest backend/test_platform_*.py -q

# Live-Postgres RLS integration suite (the only place RLS is actually proven)
KARL_TEST_POSTGRES_ADMIN_URL=postgresql+psycopg://<admin>@<host>/postgres \
  python -m pytest backend/test_platform_postgres_rls.py -q
```

### Archive benchmark snapshots for git

```bash
python scripts/archive_eval_snapshot.py --label v0_9_step1
```

This creates timestamped copies under `data/test_set/history/` so you can review and commit/push them manually.

## Governance

KARL's governance is written down and binding on releases (`legal/GOVERNANCE-CHARTER.md`):

- **Mission Council** — union leadership, stewards, and the product owner hold authority over
  product direction and mission boundaries.
- **Model Risk Council** — owns release quality gates; can block release on benchmark
  integrity or regression risk.
- **Data Stewardship Council** — owns retention, deletion, and access policy; can block
  deployment for non-compliant data practices. Its sign-off record for the platform layer:
  `legal/DATA-STEWARDSHIP-COUNCIL-SIGNOFF.md`.

Related docs: `legal/DEPLOYMENT-POLICY.md`, `legal/RELEASE-GATES.md`,
`legal/MODEL-UPDATE-POLICY.md`, `legal/ETHICAL-USE.md`, `Evaluation_Plan_v3.md`,
`CONTRACT_PACK_SPEC_v1.md`.

## License

KARL is dual-licensed: **AGPL-3.0** for unions, workers, and non-commercial use, and a separate
**commercial license** for management-side use (`legal/COMMERCIAL-LICENSE.md`). Additional
project legal docs are in `legal/`.

Contributors license their work to the public under AGPL-3.0 and sign the Contributor License
Agreement (`legal/CLA.md`), which grants the Project Steward the right to offer contributions
under the commercial license as well. See `CONTRIBUTORS.md` and `legal/CONTRIBUTING.md`.

## Status

Active development (see `UPDATE_LOG.md` for the full history):

- **v0.9.0 shipped** — amendment-aware MOA handling end-to-end, real-user correction
  infrastructure, effective contract materialization; all release gates green at ship time
- **Production platform layer** merged lineage complete: multi-tenant isolation verified
  against live Postgres, privacy/governance remediation done, Data Stewardship Council
  sign-off awaiting signatures
- **Three contracts onboarded** under UFCW Local 7; contract-pack onboarding is spec-driven
  (`CONTRACT_PACK_SPEC_v1.md`)
- **Next:** graded answer-quality evaluation maturity, section-level citation verification,
  production hardening defaults
