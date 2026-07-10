# Data Stewardship Council — Platform Privacy/Governance Sign-Off Record

Status: **READY FOR COUNCIL SIGNATURE — all technical gates verified (sign-off not yet granted)**
Subject: Multi-tenant platform layer (`backend/platform/`) and Alembic migrations 0001–0005
Branch: KARL2 fork → production merge candidate
Prepared: 2026-06-13
References: [`docs/PRIVACY_GOVERNANCE_REMEDIATION_PLAN.md`](../docs/PRIVACY_GOVERNANCE_REMEDIATION_PLAN.md)

This record certifies the privacy/governance posture of the platform layer for merge to a
production branch. Per the remediation plan, sign-off may be recorded **only after every
merge-readiness checklist item is met and verified**. All technical gates, including the
live-Postgres RLS verification (§3), are now verified; the record awaits council signatures.

## 1. Scope reviewed

- Migration-0004 tracking schema (`tracking_policies`, `user_tracking_preferences`,
  `telemetry_events`, `raw_query_records`) and migration-0005 row-level security.
- The verified-good privacy posture in remediation-plan §2 (privacy-respecting defaults, real
  opt-out, HMAC anonymization, identified-mode access control, message non-retention by
  default, PII guardrails).

## 2. Controls implemented and verified (this remediation tranche)

| Control | Evidence |
|---|---|
| User-data deletion includes telemetry, raw-query (by `user_id` **and** `anonymized_user_key`), and tracking preferences | `backend/platform/routers/admin.py` `_purge_user_records`; `backend/test_platform_data_deletion.py` |
| Member self-service deletion request path | `DELETE /api/member/me/data`; `backend/test_platform_data_deletion.py` |
| Deletion SLA (24h) documented | `legal/DEPLOYMENT-POLICY.md` |
| Session metadata (IP/UA) retention: terminated sessions purged past a 90-day window | `SessionAuthService.purge_expired_sessions`; `POST /api/ops/maintenance/purge-sessions`; `backend/test_platform_auth.py` |
| Message non-retention by default is regression-pinned | `backend/test_platform_chat_history_retention.py` |
| Dedicated `SecurityEvent` on identified raw-query enable | `backend/platform/routers/admin.py`; `backend/test_platform_admin_routes.py` |
| HMAC key-rotation note (analytics continuity, not privacy) | `legal/DEPLOYMENT-POLICY.md` |
| RLS extended to tracking + session tables (code + migration) | `backend/platform/db.py`; `alembic/versions/20260613_0005_tracking_session_rls.py` |

## 3. Live-Postgres RLS verification (gate closed 2026-07-10)

- **Verified.** `backend/test_platform_postgres_rls.py` passed **6/6** against a live
  PostgreSQL 16.4 instance with pgvector 0.8.3 (conda-forge binaries), on a **fresh database
  migrated from scratch through revision 20260613_0005**, exercised as an unprivileged
  application role. This covers cross-tenant read isolation and null-union resolution for the
  tracking and session tables, super-admin global visibility, and cross-tenant insert blocking.

  Reproduce with:
  ```
  KARL_TEST_POSTGRES_ADMIN_URL=postgresql+psycopg://<admin>@<host>/postgres \
    python -m pytest backend/test_platform_postgres_rls.py -q
  ```

- **Defects found and fixed during verification** (all fixes included in this merge candidate):
  1. *Fresh-database migration failure.* Migration `20260320_0001` iterated the live
     `get_rls_statements()` aggregate, which the 0005 work had grown to include tracking/session
     tables that do not exist at revision 0001. Fresh upgrades failed; already-migrated databases
     were unaffected. The migration now pins `foundation_rls_statements()`, and the aggregate is
     reserved for the app-startup path where the schema is at head.
  2. *Latent test defect.* The Postgres-gated tests referenced `datetime`/`timedelta` without
     importing them — never executed in an environment without Postgres.
  3. *Test-seed gap.* The super-admin visibility test asserted both unions' tracking overrides
     were visible, but only one was seeded.

  Items 1–3 are precisely why this record required live-Postgres verification rather than
  certifying on SQLite results; the SQLite suite could not have surfaced any of them.

## 4. Known non-blocking notes

- Three pre-existing test failures are unrelated to privacy/governance and out of scope for
  this sign-off: two Windows-only `strftime("%-d")` failures in the ops dashboard
  (`backend/platform/routers/ops.py`) and one stale `PlatformSettings` constructor signature in
  `backend/test_platform_worker.py`.

## 5. Council decision

All technical gates are closed; the record is ready for signature. On approval, also emit an `AuditEvent`
(`event_type="data_stewardship_council_approval"`) referencing this record and the completed
remediation-plan checklist.

| Role | Name | Decision (approve / reject) | Date | Signature |
|---|---|---|---|---|
| Data Stewardship Council chair |  |  |  |  |
| Privacy lead |  |  |  |  |
| Engineering owner |  |  |  |  |
