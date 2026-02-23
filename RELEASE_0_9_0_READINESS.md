# KARL v0.9.0 Readiness Scorecard

Date: February 22, 2026

## Purpose

Provide an explicit current-vs-target snapshot for the v0.9.0 release decision.

## Current Snapshot

### Core Evaluation Health

- Status: PASS
- Evidence:
  - `python -m backend.evaluate_runner --track all` -> 20/20 pass
  - `python -m backend.evaluate_v3 --bm25-only` -> 16/16 pass
  - `python -m backend.evaluate_moa_deep_suite` -> 16/16 pass

### Manifest/Data Contract Hygiene

- Status: PASS
- Evidence:
  - `python -m backend.validate_manifests` -> 4/4 pass
  - `contract_version` + `query_routing` present for active manifests

### MOA Effective Retrieval Behavior

- Status: PASS (pilot scope)
- Evidence:
  - `python -m backend.evaluate_moa_effective --bm25-only` -> 4/4 pass
  - `python -m backend.evaluate_moa_readiness` -> gate pass true

### Side-Letter/LOA Retrieval Stability

- Status: PASS
- Evidence:
  - `python -m backend.evaluate_side_letter_retrieval --bm25-only` -> top-8 16/16, depth-20 16/16

## Remaining v0.9.0 Gaps

### Gap 1: Collision Ergonomics in Patch Materialization

- Current: apply-or-fail exists (hash mismatch fails), but mismatch reports should include compact diff + last-touch patch context.
- Target for v0.9.0:
  - structured collision report includes:
    - expected canonical text/row hash
    - current canonical text/row hash
    - compact before/after diff
    - last patch that touched target anchor/row

### Gap 2: MOA-at-Scale Instrument Modeling

- Current: pilot behavior is working for current contract flow.
- Target for v0.9.0:
  - model a single MOA instrument that can map to multiple contracts with deterministic contract-scoped derived patches.

### Gap 3: Release Scorecard Automation

- Current: gates are spread across multiple evaluator artifacts.
- Target for v0.9.0:
  - one machine-readable release scorecard artifact consolidating must-have gate state.

## Proposed v0.9.0 Must-Have Gates

1. Deterministic materializer: apply-or-fail, no silent skips, deterministic output bytes/hashes.
2. Manifest validation clean for active contracts.
3. Canonical eval green:
   - `evaluate_runner --track all`
   - `evaluate_v3`
   - `evaluate_moa_deep_suite`
4. Retrieval defaults to latest effective snapshot for MOA-enabled contracts.
5. Runtime responses include effective/provenance metadata:
   - `effective_version_id`
   - `amendments_applied`
   - citation `source_type` with base/MOA page refs.

## Should-Have (Strongly Recommended)

1. Collision mismatch report with compact diff and last-touch trace.
2. Deleted-clause and updated-clause regression dataset integrated in canonical runner.
3. Strict artifact integrity mode enabled in CI release lane.

## Next Evaluation Wants (Short Horizon)

1. Benchmark refresh from real misses:
   - convert recent production-like failures into deterministic test cases.
2. Drift monitoring:
   - weekly run comparison for top-k retrieval behavior on side-letter/MOA-sensitive queries.
3. 0.9.0 release dry-run:
   - run full gate suite and publish scorecard artifact before tagging.
