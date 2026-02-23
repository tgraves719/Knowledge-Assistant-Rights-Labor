# KARL v3.1 Execution Plan (Toward v0.9.0)

Date: February 22, 2026

## Objective

Move from a strong 3-contract deterministic benchmark stack to a scalable multi-contract + MOA operating model that is ready for a v0.9.0 release gate.

## Current Baseline (Verified)

- Canonical runner full track is green:
  - `python -m backend.evaluate_runner --track all` -> 20/20 commands pass
  - Artifact: `data/test_set/eval_run_metadata_all_20260222T011614Z.json`
- Canonical v3 suite is green:
  - `python -m backend.evaluate_v3 --bm25-only` -> 16/16 components pass
  - Artifact: `data/test_set/v3_results.json`
- MOA readiness is green:
  - `python -m backend.evaluate_moa_readiness` -> gate pass true
  - Artifact: `data/test_set/moa_readiness_results.json`

## Gap vs Evaluation_Plan_v3.md

Implemented well:
- Multi-contract deterministic evaluation, contamination checks, adversarial precedence, unanswerable, false-unavailable, needle, table evidence, follow-up role wage.

Not yet implemented from the Phase B scaling vision:
- Universal schema extraction layer (contract facts abstraction at scale).
- Dual-model synthetic benchmark factory with contested-field human resolution workflow.
- Outlier-contract detection and weighted evaluation for rare rule values.

## v3.1 Workstreams

### Workstream A: Benchmark Factory Foundation

Goal: remove O(N) manual benchmark authoring cost.

Deliverables:
- `backend/ingest/universal_schema.py`
  - deterministic contract fact schema export per contract.
- `backend/eval/synthetic_factory.py`
  - template-based QA generation from schema fields.
- `backend/eval/schema_diff_review.py`
  - contested-field diff artifact for human review.
- `data/test_set/generated/` contract-scoped generated benchmark artifacts.

Acceptance:
- Generated-vs-manual alignment >= 90% on 3-contract validation sample.
- Generation pipeline deterministic (stable bytes/hashes for fixed inputs).

### Workstream B: MOA-at-Scale Retrieval and Provenance

Goal: handle MOAs that amend multiple contracts without degrading retrieval precision.

Deliverables:
- MOA instrument registry model:
  - store MOA once as instrument-level source
  - explicit `applies_to_contract_ids` mapping
  - contract-scoped derived patch artifacts
- Effective-history retrieval routing:
  - default to latest effective snapshot
  - strict contract filter + amendment chain filter
- Citation provenance model in API response:
  - `source_type`: `base` | `moa`
  - source PDF/page refs per citation
  - amendment chain metadata (`effective_version_id`, `amendments_applied`)

Acceptance:
- Deleted-clause queries abstain cleanly and do not surface stale base clauses.
- Updated-clause queries resolve to effective text with correct MOA citation source.
- Side-letter + MOA suites maintain gate pass.

### Workstream C: v0.9.0 Readiness Program

Goal: define and hit explicit release criteria.

Must-have gates for v0.9.0:
- Deterministic materialization apply-or-fail with structured collision report.
- Manifest schema clean across active contracts.
- `evaluate_runner --track all` green.
- `evaluate_v3` green.
- `evaluate_moa_deep_suite` green.
- Contract history + provenance payload present in runtime answers.

Should-have gates:
- Collision ergonomics report includes last-touch patch and compact diff.
- Benchmark drift checks for retrieval misses seen in real user probes.
- Contract artifact integrity strict mode green in CI.

Stretch gates:
- Generated benchmark pilot (schema/template path) on at least 2 contracts.
- Outlier-value regression slice integrated into v3.

## Two-Week Execution Sequence

Week 1:
1. Land MOA instrument registry and contract mapping model.
2. Add retrieval router constraints for effective+amendment chain selection.
3. Add eval cases for deleted-vs-updated clause behavior.

Week 2:
1. Land universal schema extractor for existing 3 contracts.
2. Land template QA generator and generated benchmark artifact format.
3. Add v0.9.0 release scorecard doc + CI gate wiring.

## Immediate Next Actions

1. Add a release scorecard artifact (`data/test_set/release_0_9_0_scorecard.json`) that computes must-have gates from existing evaluator outputs.
2. Add MOA regression dataset for:
   - removed-clause abstention
   - updated-clause effective retrieval
   - citation source-type correctness (`base` vs `moa`)
3. Run benchmark refresh against recent real chat misses and feed failures into deterministic eval sets.
