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
- Needle retrieval threshold met (overall pass rate + top/middle/bottom position floors)

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
- Classification ontology gate passes (artifact present, alias targets valid, manifest mapping decisions complete)
- Canonical wage-row schema gate passes (rows present when wages exist, row schema valid, no required integrity failures)
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
