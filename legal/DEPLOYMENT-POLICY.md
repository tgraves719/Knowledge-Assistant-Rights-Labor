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

### Secret Rotation Note — `secret_encryption_key` and analytics continuity

The anonymized telemetry identifier `anonymized_user_key` is an HMAC keyed on
`secret_encryption_key`. **Rotating `secret_encryption_key` changes every subsequently emitted
`anonymized_user_key`**, so analytics that group anonymized events by member will see a member
appear as a new anonymous identity after rotation.

- This is an **analytics-continuity** consideration only — it does **not** weaken privacy.
  Rotation never de-anonymizes prior data and never exposes a member's `user_id`.
- Rotation does not break data deletion: admin purges already match telemetry by `user_id`,
  by session id, and by reconstructed `anonymized_user_key` for the current key.
- If cross-rotation analytics continuity is required, provision a **separate, dedicated
  per-user analytics salt** (independent of `secret_encryption_key`) and rotate the two on
  independent schedules, so security-secret rotation does not perturb analytics identity.

## Data Governance Controls

- Data minimization by default
- Retention policy documented per deployment
- Deletion workflow with defined SLA
- Opt-in required for improvement/training use of user interactions
- No cross-local data reuse without explicit legal authorization

### Member Data Deletion SLA (required control)

A member-data deletion request must result in complete erasure of that member's personal
data within **24 hours of the request being received**.

- **Scope of erasure.** A purge removes the member's chats and messages, usage events,
  security events, notifications, authentication sessions, tracking preference, and all
  telemetry/raw-query records — matched by `user_id` (identified mode), by reconstructed
  `anonymized_user_key`, and by session id (anonymized mode). A full account purge also
  removes the user record, union memberships, and local-auth credentials.
- **Request paths.**
  - *Member self-service:* `DELETE /api/member/me/data` erases the requesting member's
    personal data within their current union immediately. The account login is retained;
    full cross-union account deletion is performed by an administrator.
  - *Administrator:* `DELETE /api/admin/unions/{union_id}/users/{user_id}?purge_user=true`
    performs a union-scoped or (for a member's last/only union, or by a super admin)
    global account purge.
- **Audit.** Every purge writes an `AuditEvent` (`union_user_purged` for the admin path,
  `member_self_service_data_deleted` for the self-service path).
- **Verification.** Deletion completeness is pinned by `backend/test_platform_data_deletion.py`,
  which asserts zero residual telemetry/raw-query/preference rows by both `user_id` and
  `anonymized_user_key` after a purge.

### Session Metadata Retention (required control)

Authentication sessions (`auth_sessions`) store `ip_address` and `user_agent`. Logout and
expiry only mark a session terminated (`revoked_at`/`expires_at`); the row — and its IP/UA — is
retained no longer than the **session retention window of 90 days** past termination.

- Terminated (expired or revoked) sessions older than the window are **fully deleted**, IP/UA
  included. Geolocation-for-abuse-detection retention of truncated IPs is not used; full
  deletion is the default.
- The window defaults to 90 days and is configurable per invocation.
- Enforcement: `SessionAuthService.purge_expired_sessions(db, older_than_days=90)`, invoked via
  the super-admin maintenance endpoint `POST /api/ops/maintenance/purge-sessions` (audited as
  `auth_sessions_purged`). Deployments must run this maintenance routine on a recurring
  schedule (e.g. daily) so the window is continuously enforced.
- Active sessions are never deleted. Verified by `backend/test_platform_auth.py`.

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

