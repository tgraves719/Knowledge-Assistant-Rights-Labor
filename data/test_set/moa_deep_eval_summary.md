# MOA Deep Eval Summary

Generated: 2026-07-17T20:08:04.805287+00:00

## Suite Status
- Commands passed: 19/19 (100.0%)
- All commands passed: True

## MOA Effective Scores
- Baseline dataset: 10/10 (100.0%)
- Deep dataset: 16/18 (88.9%)

## Track Scores
- `moa_effective_baseline`: rc=0, pass=10/10 (100.0%)
- `moa_effective_deep`: rc=0, pass=16/18 (88.9%)
- `moa_deleted_vs_updated`: rc=0, pass=4/4 (100.0%), gate=True
- `moa_deleted_vs_updated_answer`: rc=0, pass=4/4 (100.0%), gate=True
- `moa_readiness`: rc=0
- `effective_snapshot_coverage`: rc=0, pass=2/2 (100.0%)
- `effective_wage_snapshot_coverage`: rc=0, pass=2/2 (100.0%)
- `false_unavailable_fallback_test`: rc=0
- `cross_contract_mentions`: rc=0, pass=9/9 (100.0%)
- `false_unavailable`: rc=0, pass=15/15 (100.0%)
- `unanswerable`: rc=0, pass=12/12 (100.0%)
- `adversarial_precedence`: rc=0, pass=12/12 (100.0%)
- `wage_table_evidence`: rc=0, pass=12/12 (100.0%)
- `entitlement_table_evidence`: rc=0, pass=12/12 (100.0%)
- `side_letter_retrieval`: rc=0, pass=15/16 (93.8%), gate=True
- `role_catalog_integrity`: rc=0, pass=20/20 (100.0%)
- `followup_role_wage`: rc=0, pass=14/14 (100.0%)
- `needle`: rc=0, pass=5/5 (100.0%)
- `topic_routing_test`: rc=0

## Deep Dataset Failures
- `MOA-CL-NIGHT-02`: What is the overnight premium for courtesy clerks from midnight to 6 a.m.?
  - citation_hit=False keywords_ok=False source_type_ok=False intent=contract
  - top citations: Article 16, Section 35, Article 16, Section 37, Article 4, Section 8, Article 5, Section 11
- `MOA-CL-WAGE-02`: What is the courtesy clerk start rate effective January 21, 2024?
  - citation_hit=True keywords_ok=False source_type_ok=True intent=wage
  - top citations: Article 8, Section 18, Article 8, Section 19, Article 4, Section 8, Article 5, Section 12

## Priority Gaps
- `moa_effective_deep`: 16/18 (88.9%)
- `side_letter_retrieval`: 15/16 (93.8%)
