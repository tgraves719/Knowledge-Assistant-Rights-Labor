# KARL Model Update Policy

Version: v0.1 (Draft)  
Effective Date: 2026-02-13  
Applies To: Model, retrieval, reranker, and evaluation-model updates.

## Purpose

Control model and retrieval changes to prevent silent regressions in rights-critical outputs.

## Change Categories

1. Patch
- Prompt edits, minor retrieval tuning, non-architectural config changes

2. Minor
- Component replacement in same model family or retrieval strategy changes

3. Major
- Provider/model-family swap, architecture changes, multi-tenant routing changes

## Required Artifacts Per Change

- Change description and rationale
- Risk assessment
- Before/after evaluation report
- Rollback plan
- Known limitations update

## Evaluation Requirements

Patch:
- Core regression suite
- Safety checks

Minor:
- Core regression suite
- Adversarial suite
- Multi-contract safety checks

Major:
- Full v3 suite
- Ablation/diagnostic report
- Canary validation across at least one real tenant deployment path

## Rollout Process

1. Shadow evaluation in staging
2. Canary rollout to limited tenant/user slice
3. Monitor defined risk metrics
4. Full rollout only after stability window completion

## Automatic Rollback Triggers

Rollback is required if any of the following occurs:
- Citation entailment drops below policy threshold
- Precedence failures exceed policy threshold
- Cross-contract leakage is non-zero
- High-stakes escalation behavior breaches policy threshold
- Security/compliance incident linked to model change

## Model Registry Requirements

Each deployable model configuration must include:
- Model/provider identifier
- Prompt/version identifier
- Retrieval config version
- Evaluation model/version used for grading
- Effective deployment date

## Prohibited Practices

- Silent model/provider upgrades
- Production model changes without documented gate pass
- Using the same model as both primary system and sole evaluator for high-impact decisions

## Ownership

- Primary owner: Model Risk Council
- Co-approval: Data Stewardship Council for changes affecting data handling
- Mission Council informed for all Major updates

