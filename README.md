# KARL - Knowledge Assistant for Rights and Labor

KARL is an AI-powered RAG system designed to help workers and unions understand, navigate, and enforce collective bargaining agreements.

Current deployment focus: **UFCW Local 7 - Safeway Pueblo Clerks (2022-2025)**.

## Core Principles

- Union-first
- Worker-controlled
- Privacy-respecting
- Anti-surveillance
- Transparent by design

## Current Benchmark Status

KARL currently uses three benchmark labels in project planning:

1. **Benchmark v1 (legacy golden benchmark)**
- Legacy benchmark used earlier in development
- Historical result reached **100% (55/55)** on that benchmark
- Script path: `python -m backend.evaluate`
- Artifact: `data/test_set/evaluation_results.json`

2. **Benchmark v2 (harder comprehensive benchmark)**
- Newer and more difficult benchmark track used to stress retrieval quality
- This is where performance dropped from v1's 100% into the mid/high-80% range
- Current checked-in comprehensive artifact is in this track
- Script path: `python -m backend.evaluate_comprehensive --ablation-mode normal`
- Artifact: `data/test_set/comprehensive_results.json`

3. **Benchmark v3 (scaled multi-contract phase)**
- Canonical multi-contract evaluation suite with deterministic isolation/robustness slices
- Includes contamination checks and release-gated multi-contract integrity metrics
- Script path: `python -m backend.evaluate_runner --track v3`
- Artifact: `data/test_set/v3_results.json`
- Planning doc: `Evaluation_Plan_v3.md`

## Architecture (Current Runtime)

KARL has a 5-phase CAG design, but currently runs a **lean configuration** by default:

- Phase 1 (intent routing): enabled
- Phase 2 (hypothesis layer): disabled
- Phase 3 (full article expansion): disabled
- Phase 4 (query interpreter): enabled
- Phase 5 (LLM reranker): enabled
- BM25 fusion: disabled in lean mode (vector-first retrieval)

Key config file: `backend/config.py`.

## Features

- Citation-focused responses
- Deterministic wage lookups from structured wage tables
- Deterministic two-stage high-stakes routing (`high_stakes_topic` vs `active_urgent_context`)
- Conditional/hypothetical escalation suppressors to reduce false positives
- Query interpreter for vocabulary bridging
- LLM reranker for relevance ordering
- Manifest-driven routing metadata
- Interactive citation navigation in UI

## Repository Structure

```text
karl/
├── backend/
│   ├── api.py
│   ├── config.py
│   ├── evaluate.py
│   ├── evaluate_comprehensive.py
│   ├── ingest/
│   ├── retrieval/
│   ├── generation/
│   └── eval/
├── data/
│   ├── chunks/
│   ├── wages/
│   ├── manifests/
│   ├── chroma_db/
│   └── test_set/
├── frontend/
│   └── index.html
├── legal/
│   ├── GOVERNANCE-CHARTER.md
│   ├── DEPLOYMENT-POLICY.md
│   ├── RELEASE-GATES.md
│   ├── MODEL-UPDATE-POLICY.md
│   └── ETHICAL-USE.md
├── Evaluation_Plan_v3.md
└── UPDATE_LOG.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API key (optional but recommended)

```bash
# Windows PowerShell
Set-Content -Path .env -Value "GEMINI_API_KEY=your_actual_api_key_here"

# Linux/Mac
printf "GEMINI_API_KEY=your_actual_api_key_here\n" > .env
```

Without an API key, KARL can still retrieve chunks and perform wage lookups, but it cannot produce synthesized LLM answers.

### 3. Build or refresh data (if needed)

```bash
python -m backend.ingest.smart_chunker
python -u -m backend.ingest.enricher --batch-size 15 --delay 1.0
python -m backend.ingest.rebuild_index
```

### 4. Start server

```bash
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

### 5. Open app

Go to `http://127.0.0.1:8000`.

## Query Contract Context

`/api/query` requires explicit contract context:
- `contract_id` (required)
- `union_local_id` (required, must match manifest `union_local`)
- `contract_version` (required, must match manifest version string)

Current manifest version format:
- `contract_version = "<term_start>__<term_end>"`
- Example: `January 23, 2022__January 18, 2025`

Chunk artifact resolution now prefers per-contract files before shared fallback:
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

Deterministic ingestion outputs now include:
- Contract-scoped `concept_index_<contract_id>.json` with non-empty concept/question mappings
- Contract-scoped `language_lexicon_<contract_id>.json` (frozen alias graph)
- Contract-scoped `role_catalog_<contract_id>.json` (wage-availability-aware onboarding role catalog)
- Contract-scoped canonical wage rows in `wage_tables_<contract_id>.json` (`canonical_wage_rows`) with table-backed source references
- Contract-scoped entitlement schedules in `entitlement_tables_<contract_id>.json` (`vacation_entitlements`)
- Contract-scoped manifest `query_routing` synthesized from ingestion artifacts (`topic_to_articles`, `topic_patterns`, `slang_to_contract`, `classification_to_articles`)
- `region_id` on manifests/chunks for hard tenancy filtering (`contract_id` + `region_id`)

Runtime wage lookup now resolves canonical rows first and includes structured
`table_evidence` metadata in wage responses when table source references exist.

Runtime query expansion order is deterministic:
1. universal slang map
2. frozen contract language lexicon (`alias_to_canonical`)
3. manifest `query_routing.slang_to_contract` overrides

## Evaluation Commands

### Canonical runner (recommended)

```bash
# Benchmark v1
python -m backend.evaluate_runner --track v1

# Benchmark v2
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
python backend/test_topic_routing.py

# False-unavailable guard slice (forced unavailable first pass + deterministic recovery)
python -m backend.evaluate_runner --track false_unavailable

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
```

False-unavailable evaluation now includes both:
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

- Pull requests: manifest validation + canonical slices (`v2`, `escalation`, `v2_multi_contract`, `paraphrase`, `adversarial`, `unanswerable`, `cross_contract_mentions`, `false_unavailable`, `needle`, `wage_table_evidence`, `entitlement_table_evidence`, `role_catalog_integrity`, `v3`) + isolation/cross-contamination/topic-routing checks + gate-check thresholds
- Push to `main`: manifest validation + full v2 ablation suite + deterministic release slices + cross-contamination/topic-routing + gate-check job

`backend.evaluate_cross_contamination` skips automatically when only one manifest is present.

### Contract isolation check

```bash
python -m backend.test_contract_isolation
```

### Query-routing ingestion check

```bash
python backend/test_query_routing_ingest.py
```

### Unavailable-answer recovery check

```bash
python backend/test_unavailability_recovery.py
```

### Cross-contamination evaluator (multi-contract scaffold)

```bash
python -m backend.evaluate_cross_contamination
```

### Archive benchmark snapshots for git

```bash
python scripts/archive_eval_snapshot.py --label v0_9_step1
```

This creates timestamped copies under `data/test_set/history/` so you can review and commit/push them manually.

## Model Stack

| Component | Model |
|---|---|
| Answer generation | Gemini 2.5 Pro |
| Query interpreter | Gemini 2.5 Flash |
| Reranker | Gemini 2.5 Flash |
| Enricher | Gemini 2.5 Flash |
| Embeddings | all-MiniLM-L6-v2 |

## Governance and Scaling Docs

- `legal/GOVERNANCE-CHARTER.md`
- `legal/DEPLOYMENT-POLICY.md`
- `legal/RELEASE-GATES.md`
- `legal/MODEL-UPDATE-POLICY.md`
- `Evaluation_Plan_v3.md`
- `CONTRACT_PACK_SPEC_v1.md`

## License

KARL is AGPL-3.0 licensed with additional project legal docs in `legal/`.

## Status

Active development:
- Lean architecture is current baseline
- v2 benchmark hardening is active
- v3 multi-contract suite is implemented as canonical release-gated track
