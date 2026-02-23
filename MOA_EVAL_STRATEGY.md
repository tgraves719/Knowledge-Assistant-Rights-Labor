# MOA Eval And Testing Strategy

## Objective
Guarantee that KARL answers from the current effective contract state, preserves legal provenance, and does not regress as new MOAs are added across contracts.

## Gate Layers
1. Build Integrity Gate
- `python -m backend.test_moa_materializer`
- Validates deterministic bytes/hashes, apply-or-fail collisions, provenance persistence, and source-doc resolution.

2. Contract History/API Gate
- `python -m backend.test_contract_history_api`
- Validates Contract-tab lineage payloads and MOA/base PDF navigation behavior.

3. Routing Guard Gate
- `python -m backend.test_topic_routing`
- Prevents legal-clause prompts from drifting into wage-only retrieval paths.

4. Fallback Guard Gate
- `python -m backend.test_false_unavailable_fallback`
- Validates deterministic fallback ranking favors topic-defining and anchor articles when synthesis fails.

5. Effective Snapshot Coverage Gate
- `python -m backend.evaluate_effective_coverage`
- Validates latest effective index inputs preserve base chunk coverage/cardinality by `doc_type` (including LOUs) for replace-only patch flows.

6. Effective Retrieval Gate
- `python -m backend.evaluate_moa_effective --bm25-only`
- Measures amended-clause retrieval, removed-clause handling, source-type correctness, and wage-info suppression on non-wage prompts.

7. Readiness Aggregate Gate
- `python -m backend.evaluate_moa_readiness`
- Fails if any required command fails or MOA effective pass rate drops below threshold (default `0.90`).
- Optional deep gate:
  - `python -m backend.evaluate_moa_readiness --deep-input data/test_set/moa_effective_deep_test.json --deep-pass-rate-threshold 0.85`

8. Canonical Runner Integration
- `python -m backend.evaluate_runner --track moa_readiness --skip-manifest-validation`
- Use `--skip-manifest-validation` when unrelated manifest schema migrations are in progress.

9. Side-Letter Retrieval Gate
- `python -m backend.evaluate_side_letter_retrieval --bm25-only`
- Depth-aware gate:
  - Top-K window (`--n-results`, default `8`) tracks precision.
  - Depth window (`--depth-results`, default `20`) tracks recall.
  - Gate fails only if both degrade below thresholds (`--topk-pass-threshold` default `0.80`, `--depth-pass-threshold` default `0.95`).
- Validates LOA/LOU retrieval returns side-letter citations with matching `doc_type` buckets (`loa`/`lou`) on known side-letter-heavy contracts, including follow-up phrasing (for example, "that agreement" + "written notice").

10. Deep MOA Suite
- `python -m backend.evaluate_moa_deep_suite`
- Runs MOA baseline + deep dataset + readiness + contamination/guard benchmarks in one pass, including side-letter retrieval.

## Nightly CI Recommendation
1. `python -m backend.evaluate_runner --track moa_readiness`
2. `python -m backend.evaluate_runner --track cross_contract_mentions`
3. `python -m backend.evaluate_runner --track false_unavailable`
4. `python -m backend.evaluate_runner --track wage_table_evidence`
5. `python -m backend.evaluate_runner --track entitlement_table_evidence`
6. `python -m backend.evaluate_runner --track artifact_integrity`
7. `python -m backend.evaluate_runner --track side_letter_retrieval`

## Scale Artifact Audit
- `python -m backend.evaluate_contract_artifact_integrity`
- Audits every contract for:
  - LOU/LOA representation in chunks (`doc_type` mix + lexical hits)
  - Wage row linkage completeness (`source_reference.table_id` and `row_index`)
  - PDF and table navigation index presence
  - Effective snapshot doc-type coverage vs base when an effective snapshot exists

## Strict Side-Letter Policy Gate
- `python -m backend.evaluate_contract_artifact_integrity --strict-side-letter-buckets`
- Rule:
  - if `letter of agreement`/`letter of understanding` lexical hits exceed threshold (default `>=1`),
  - require at least one side-letter bucket (`doc_type=loa` or `doc_type=lou`).
- Optional threshold:
  - `--side-letter-hit-threshold <N>`
- Runner shortcut:
  - `python -m backend.evaluate_runner --track artifact_integrity_strict --skip-manifest-validation`
- Backfill utility for legacy packs:
  - dry run:
    - `python -m backend.ingest.backfill_side_letter_doc_types --contract-id <id>`
  - apply:
    - `python -m backend.ingest.backfill_side_letter_doc_types --apply --contract-id <id>`

## Dataset Growth Priorities
1. Removed-by-MOA clauses
- Query old language intentionally removed; expected behavior is abstain or cite replacement.

2. Amended-but-similar clauses
- Old and new text both mention same topic; expected behavior is MOA/effective citation priority.

3. Cross-contract contamination
- Same article/section numbers across different contracts; expected behavior is strict `contract_id` isolation.

4. Multi-source provenance
- Clause with both base and MOA provenance; ensure citation payload marks mixed lineage.

5. Table-row amendments
- Wage and entitlement row updates with stable `row_key`; ensure evidence points to amended row.

## Operational Workflow For New MOA
1. Register source doc:
- `python -m backend.ingest.register_source_doc ...`

2. Generate draft patch ops:
- Metadata-driven multi-contract targeting:
  - `python -m backend.ingest.generate_patch_drafts --source-doc-id <id>`
- Explicit override targeting:
  - `python -m backend.ingest.generate_patch_drafts --source-doc-id <id> --contract-id <id> [--contract-id ...]`
- Optional controls:
  - `--exclude-contract-id <id>` (repeatable)
  - `--strict` (fail when any target contract fails)
- Output targeting report:
  - `data/source_docs/<doc_type>/<source_doc_id>/patch_draft_generation_report.json`

3. Human review and approval:
- Move selected draft ops to contract amendment patch file and set `review_status: approved`.

4. Materialize:
- `python -m backend.ingest.materialize_effective --contract-id <id> --effective-version-id <id>`

5. Run gates:
- `python -m backend.evaluate_moa_readiness`
