# KARL Deployment Policy

Version: v0.1 (Draft)  
Effective Date: 2026-02-13  
Applies To: All hosted, self-hosted, and pilot KARL deployments.

## Deployment Model Standard

KARL must use a multi-tenant architecture with:
- Shared control plane (codebase, CI/CD, evaluation framework, observability tooling)
- Isolated data planes per local/chapter (data, vector indexes, logs, secrets)

## Tenant Isolation Requirements

The following fields are mandatory in runtime paths:
- `union_local_id`
- `contract_id`
- `contract_version`
- `effective_date_range`

These identifiers must be enforced in:
- API request handling
- Retrieval filters
- Evaluation jobs
- Logging and analytics partitioning

Cross-tenant retrieval is prohibited by default.

## Environment Tiers

1. Development
- Local-only data
- Test keys only
- No production member data

2. Staging
- Production-like config
- Synthetic or approved non-sensitive data
- Full gate execution before promotion

3. Production
- Approved contracts only
- Full auth, audit, and retention controls enabled

## Security Baseline

Required controls:
- Authentication for non-health endpoints
- Role-based authorization (`member`, `steward`, `admin`, `evaluator`)
- CORS allowlist (no wildcard in production)
- TLS in transit
- Encryption at rest for persistent stores
- Secret management with rotation

## Data Governance Controls

- Data minimization by default
- Retention policy documented per deployment
- Deletion workflow with defined SLA
- Opt-in required for improvement/training use of user interactions
- No cross-local data reuse without explicit legal authorization

## Operational Reliability Standards

- Blue/green or canary release process
- Rollback objective: under 15 minutes
- Backup and restore tests at least quarterly
- SLOs defined and monitored for:
  - Availability
  - Retrieval latency
  - Citation validity
  - High-stakes escalation routing

## Contract Lifecycle Controls

For each onboarded contract:
- Provenance record (source, authorization, intake date)
- Versioning metadata (effective/expiration dates)
- Ingestion log and corpus hash
- Rollback-ready previous contract version

## Prohibited Deployment Patterns

- Single shared data namespace across multiple locals in production
- Production deployments without release gate sign-off
- Production deployments without incident response contact path

