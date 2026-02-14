# KARL Governance Charter

Version: v0.1 (Draft)  
Effective Date: 2026-02-13  
Applies To: All KARL deployments, data pipelines, models, and evaluation workflows.

## Purpose

KARL exists to provide accurate, contract-grounded rights information for union members and stewards, while minimizing legal, privacy, and operational risk.

## Scope

This charter governs:
- Product behavior and mission alignment
- Model and retrieval risk management
- Data stewardship and privacy controls
- Release approvals and incident response

## Governance Bodies

### 1. Mission Council

Composition:
- Union leadership representatives
- Steward representatives
- Product owner

Authority:
- Approve or reject product direction and feature categories
- Enforce union-first mission boundaries
- Prioritize rollout sequence across locals/chapters

### 2. Model Risk Council

Composition:
- Engineering lead
- Evaluation lead
- Retrieval/model maintainers

Authority:
- Define and enforce release quality gates
- Approve model/provider/config changes
- Block release for benchmark integrity or regression risk

### 3. Data Stewardship Council

Composition:
- Privacy/security lead
- Legal operations representative
- Platform operations representative

Authority:
- Set retention, deletion, and access policies
- Approve data handling and tenant isolation controls
- Block deployment for non-compliant data practices

## Authority Boundaries

- Any council may place a release hold within its scope.
- Mission Council may reject features that undermine worker interests.
- Model Risk Council may reject changes that fail evaluation or safety thresholds.
- Data Stewardship Council may reject deployments lacking policy compliance.
- Emergency rollback authority is delegated to Engineering Lead plus one council delegate.

## Decision Process

- Standard decisions: simple majority in relevant council.
- High-impact decisions (model provider swap, multi-local rollout, retention policy changes): approval required from all three councils.
- Release approval requires recorded sign-off against release gates.

## Mandatory Controls

1. No high-stakes release without passing approved evaluation gates.
2. No model/provider swap without documented regression analysis.
3. No local/chapter onboarding without data rights and provenance record.
4. No cross-tenant retrieval or storage access by default.

## Transparency and Audit

Each production release must publish:
- Code version/commit reference
- Model and retrieval configuration versions
- Contract corpus version/hash
- Evaluation summary and known limitations

Incident log requirements:
- Severity classification
- Root cause
- User/tenant impact
- Remediation and follow-up actions

## Review Cadence

- Quarterly charter review
- Immediate update after any Severity 1 or Severity 2 incident

