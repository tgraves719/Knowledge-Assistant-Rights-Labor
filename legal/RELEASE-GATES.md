# KARL Release Gates

Version: v0.1 (Draft)  
Effective Date: 2026-02-13  
Applies To: All production and pilot releases.

## Policy

A release is blocked unless all required gates pass and sign-off is recorded.

## Gate A: Evaluation Integrity

Required:
- Canonical evaluation runner used
- Dataset version and corpus hash recorded
- Reproducible run metadata captured (config snapshot + commit reference)
- No benchmark leakage findings

Owner: Model Risk Council

## Gate B: Accuracy and Grounding

Required:
- Contract-only accuracy threshold met
- Citation entailment threshold met
- Precedence failure threshold met
- Citation fabrication rate at or near zero per policy target
- Paraphrase robustness threshold met (family pass rate + worker-slang floor)
- Formal-rewrite drift threshold met (non-slang `formal_rewrite` variant pass-rate floor)
- Adversarial formal-precedence threshold met (overall + per-contract pass floors, precedence-order pass floor, canonical dataset schema)
- Adversarial dataset integrity checks passed (required schema version, minimum distribution, active-contract coverage)
- Unanswerable/abstention threshold met (overall + per-contract floors, canonical dataset schema)
- Unanswerable dataset integrity checks passed (required schema version, minimum distribution, scenario diversity, active-contract coverage)
- Cross-contract mention abstention threshold met (overall + per-contract floors, no-citation floor, canonical dataset schema)
- Cross-contract mention dataset integrity checks passed (required schema version, minimum distribution, active-contract coverage)
- False-unavailable guard check passed (no "not available" claim when strong in-context evidence exists)
- False-unavailable canonical evaluation thresholds met (recovery-rate floor on evidence-present cases, proper-uncertainty floor on evidence-absent cases, and overall/per-contract floors)
- False-unavailable dataset integrity checks passed (required schema version, minimum distribution, active-contract coverage)
- Needle retrieval threshold met (overall pass rate + top/middle/bottom position floors)
- Wage-table evidence threshold met (overall + per-contract floors, canonical dataset schema, source-method floor, table-evidence/table-id presence floors)
- Entitlement-table evidence threshold met (overall + per-contract floors, canonical dataset schema, weeks-resolution floor, source/evidence presence floors)
- Role-catalog integrity threshold met (overall + per-contract floors, canonical dataset schema, dataset-case pass floor, default-role wage-readiness floor, unresolved-role containment floor)

Owner: Model Risk Council

## Gate C: Safety and High-Stakes Behavior

Required:
- High-stakes detection threshold met
- Escalation behavior threshold met
- High-stakes escalation precision threshold met (confusion-matrix based)
- High-stakes escalation false-positive rate at or below approved cap
- Escalation deterministic-policy check passed (no stochastic escalation behavior in primary path)
- Unanswerable refusal threshold met
- Adversarial near-miss threshold met
- Escalation-focused slice pass:
  - Conditional/hypothetical rights prompts
  - Active urgent situations
  - Neutral policy prompts containing trigger words

Owner: Model Risk Council + Mission Council

## Gate D: Multi-Contract Safety

Required:
- Canonical v3 suite pass (`data/test_set/v3_results.json`)
- Canonical adversarial formal-precedence slice pass (`data/test_set/adversarial_results.json`)
- Canonical multi-contract unanswerable slice pass (`data/test_set/unanswerable_results.json`)
- Canonical cross-contract mention slice pass (`data/test_set/cross_contract_mentions_results.json`)
- Canonical wage-table evidence slice pass (`data/test_set/wage_table_evidence_results.json`)
- Canonical entitlement-table evidence slice pass (`data/test_set/entitlement_table_evidence_results.json`)
- Canonical role-catalog integrity slice pass (`data/test_set/role_catalog_integrity_results.json`)
- Cross-contamination test pass (zero wrong-tenant retrievals)
- Multi-contract benchmark slice pass (overall threshold + per-contract floor)
- Retrieval hard-filter enforcement verified (`contract_id` + `region_id`)
- Outlier-value test pass
- Contract-version routing pass

Owner: Model Risk Council

## Gate E: Security and Compliance

Required:
- Authentication/authorization checks verified
- Production CORS policy verified
- Secrets scan and config hygiene checks pass
- Data retention/deletion controls verified

Owner: Data Stewardship Council

## Gate F: Change Control and Documentation

Required:
- Model/provider/config changes documented
- Risk and rollback plan attached
- Known limitations updated
- Release note includes metrics, corpus version, and gate results

Owner: Mission Council + Model Risk Council

## Gate G: Contract Pack Quality (Ingestion-Owned)

Required:
- Pack scorecard generated for each active contract package
- Pack scorecard passes required checks (and advisory checks in strict releases)
- Language lexicon artifact gate passes (artifact present, alias graph non-empty, region metadata present)
- Manifest query-routing coverage gate passes (topic/article map + topic patterns + slang map + classification/article map)
- Classification ontology gate passes (artifact present, alias targets valid, manifest mapping decisions complete)
- Role catalog gate passes (artifact present, schema-valid, onboarding-default roles all wage-resolvable)
- Canonical wage-row schema gate passes (rows present when wages exist, row schema valid, no required integrity failures)
- Entitlement schedule artifact gate passes (artifact present, schema-valid, vacation schedules non-empty when vacation article language exists)
- Ingestion review-queue gate passes when unresolved/ambiguous ingestion issues exist
- Accepted pack hash recorded in `data/contracts/pack_registry.json`
- Benchmark metadata references active accepted pack hashes

Owner: Data Stewardship Council + Engineering release owner

## Sign-Off Matrix

Required signatures before production:
- Mission Council representative
- Model Risk Council representative
- Data Stewardship Council representative
- Engineering release owner

## Failure Handling

If any gate fails:
- Release status set to BLOCKED
- Blocking gate and owner documented
- Remediation plan required before re-run
- Any escalation precision regression vs approved baseline is release-blocking until resolved and re-verified
