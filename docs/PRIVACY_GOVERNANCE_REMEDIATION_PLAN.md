# Privacy & Governance Hardening — Handoff Plan

**Goal:** Close the remaining privacy/governance gaps in the platform layer so it can merge to the
production branch and receive Data Stewardship Council sign-off.

**Scope:** `backend/platform/` and Alembic migrations 0001–0004 (the multi-tenant platform layer).

**How to use this doc:** Each work item has a target, the files to touch, objective acceptance
criteria, and how to verify. Don't check a box until its acceptance criteria are met *and* verified
by the stated method. Run the [§5 checklist](#5-merge-readiness-checklist) before merging.

---

## 1. Test harness — read this first

There are two tiers, and they verify different things:

- **Default suite (SQLite in-memory).** Most `backend/test_platform_*.py` build a `ServiceContainer`
  over `sqlite+pysqlite:///:memory:` with `apply_rls_policies=False`. **RLS is a no-op here** —
  `apply_request_context()` only emits `set_config(...)` on the `postgresql` dialect
  (`backend/platform/db.py:138`). Good for deletion logic, session purge, message-retention guard.
- **Live-Postgres suite (opt-in).** `backend/test_platform_postgres_rls.py` is skipped unless
  `KARL_TEST_POSTGRES_ADMIN_URL` is set. It provisions a throwaway role + DB, runs Alembic to head,
  and exercises real RLS as an unprivileged role. **This is the only place RLS can be proven.**

> Do not mark any RLS work done on SQLite results alone.

---

## 2. Current posture — verified good, do not regress

These behaviors are correct today. Protect them with tests; don't change them.

- **Privacy-respecting defaults** — `TelemetryService.default_policy_values()` (`telemetry.py:99`):
  `tracking_mode=bug_and_journey`, `privacy_mode=anonymized`, `raw_query_storage_mode=disabled`,
  `default_member_preference=bug_only`.
- **Real opt-out** — `OFF` preference suppresses all event writes while the query still succeeds
  (`telemetry.py:61-69`).
- **HMAC anonymization** — `anonymized_user_key()` keyed on `secret_encryption_key`; no direct
  `user_id` stored in anonymized mode (`telemetry.py:235-245`).
- **Identified raw-query storage is access-controlled** — tracking-policy endpoints are
  `require_roles(SUPER_ADMIN)` + gated by `allow_union_override`, and each change writes an
  `AuditEvent` (`routers/admin.py:1250-1314`).
- **Message non-retention by default** — `persist_turn()` early-returns (writes nothing) when a
  union's `message_retention_enabled` is false (`chat_history.py:113-119`).
- **PII guardrails** — SSN/email/phone/financial/member-id/secret redaction at prompt and document
  boundaries (`guardrails.py:207`).

---

## 3. Work items (prioritized)

### P1 — Complete the user-data deletion path

**Now:** `_purge_user_records()` (`routers/admin.py:89-127`) deletes chats, messages, usage,
security events, notifications, and sessions, but **omits `TelemetryEvent`, `RawQueryRecord`, and
`UserTrackingPreference`**. No member self-service path; no documented SLA.

**Target:** A user purge removes *all* personal data, including telemetry/raw-query rows (by
`user_id` and `anonymized_user_key`) and tracking preferences; an SLA is documented; members have a
request path.

**Files:** `routers/admin.py` (extend `_purge_user_records`); `routers/member.py` (add
`DELETE /api/member/me/data`, or admin-only interim if documented); a deletion test
(`test_platform_data_deletion.py`); `legal/DEPLOYMENT-POLICY.md` (SLA).

**Acceptance:**
1. Post-purge: zero `TelemetryEvent` for the user by `user_id` **and** by `anonymized_user_key`.
2. Post-purge: zero `RawQueryRecord` by `user_id` **and** by `anonymized_user_key`.
3. Post-purge: zero `UserTrackingPreference` for the user.
4. Existing purge behavior unchanged (prior tests pass).
5. `DEPLOYMENT-POLICY.md` states a concrete SLA (proposed: **24h from request**).

**Verify:** `python -m pytest backend/test_platform_admin_routes.py backend/test_platform_data_deletion.py`

---

### P1 — Session metadata retention

**Now:** `AuthSession` stores `ip_address`/`user_agent` (`session_auth.py:76-86`). Expiry/logout set
`revoked_at` but never delete the row — IP/UA persist indefinitely. No purge job exists.

**Target:** Expired/revoked sessions older than a defined window are deleted; the window is
documented and enforced by a callable maintenance routine.

**Files:** `session_auth.py` (add `purge_expired_sessions(db, *, older_than_days=90)`); a maintenance
entry point (worker `__main__` hook or `POST /api/ops/maintenance/purge-sessions`,
`require_roles(SUPER_ADMIN)`); `test_platform_auth.py`; `legal/DEPLOYMENT-POLICY.md`.

**Decision to record here when made:** full deletion of expired rows (recommended) vs. IP truncation
to /24. Default to full deletion unless geolocation-for-abuse-detection is required.

**Acceptance:**
1. `purge_expired_sessions()` deletes rows past the window (by `expires_at` or `revoked_at`), keeps
   active sessions.
2. Window defaults to 90 days, configurable.
3. `DEPLOYMENT-POLICY.md` documents the window as a required control.

**Verify:** `python -m pytest backend/test_platform_auth.py`

---

### P1 — RLS for tracking + session tables  *(requires live Postgres)*

**Now:** `TENANT_RLS_TABLES` (`db.py:21-33`) omits `telemetry_events`, `raw_query_records`,
`tracking_policies`, `user_tracking_preferences`, and `auth_sessions`. Isolation for this data is
app-layer `union_id` filtering only, with no DB backstop.

**Target:** Those tables carry Postgres RLS consistent with the existing tenant model, with correct
handling of legitimately-null `union_id` rows.

**Files:** `db.py` (add the four tracking tables to `TENANT_RLS_TABLES`; tailored policy for
nullable-`union_id` tables); a new Alembic migration; extend `test_platform_postgres_rls.py`.

**Caveat — nullable `union_id` (handle this or you hide legitimate rows):** the global
`tracking_policies` row has `union_id IS NULL`; `user_tracking_preferences` and super-admin
`auth_sessions` can too. A naive `union_id::text = current_setting('app.current_union_id', true)`
hides them from everyone. Use
`current_setting('app.current_role', true) = 'super_admin' OR union_id::text = current_setting('app.current_union_id', true)`
so null-union rows resolve under the bootstrap/super-admin context (matching
`apply_service_bootstrap_context`).

**Acceptance:**
1. Union-A app role cannot read union-B `telemetry_events` / `raw_query_records`.
2. Global `tracking_policies` row still resolves under bootstrap/super-admin; effective-policy
   resolution still works.
3. Default SQLite suite still passes.

**Verify:** `KARL_TEST_POSTGRES_ADMIN_URL=... python -m pytest backend/test_platform_postgres_rls.py`
**plus** the full default suite.

---

### P2 — Regression test for message non-retention

**Now:** Correct in code (`chat_history.py:113-119`) but unpinned by any test.

**Target:** A test guarantees the guard can't silently regress.

**Acceptance:**
1. `persist_turn()` writes **zero** `Message` rows when `message_retention_enabled` is false.
2. The user+assistant pair **is** written when it is true.

**Verify:** `python -m pytest backend/test_platform_demo_flow.py` (or new `test_chat_history_retention.py`)

---

### P2 — Data Stewardship Council sign-off  *(unblocks merge)*

**Target:** Formal council approval of the migration-0004 tracking schema and the privacy posture in
§2, recorded before the platform layer merges to a production branch.

**Files:** a signed record under `legal/` and/or an `AuditEvent`
(`event_type="data_stewardship_council_approval"`).

**Acceptance:** The sign-off artifact exists and references this plan's completed checklist.

---

### P3 — Dedicated audit signal for identified mode

**Target:** When a policy update sets `raw_query_storage_mode = enabled_identified`, emit a
`SecurityEvent` (severity `warning`, `event_type="raw_query_identified_enabled"`) in addition to the
existing `AuditEvent`, so the most sensitive setting is unmistakable in the trail.

**Files:** `telemetry.py` / `routers/admin.py`; a test.

**Acceptance:** Enabling the mode creates exactly one such `SecurityEvent`.

---

### P3 — HMAC key-rotation note

**Target:** Document in `DEPLOYMENT-POLICY.md` that rotating `secret_encryption_key` invalidates
`anonymized_user_key` continuity (analytics only, not privacy), and note the option of a separate
per-user analytics salt if continuity is required.

**Acceptance:** The note exists in `DEPLOYMENT-POLICY.md`.

---

## 4. Suggested order

1. Deletion completion + session retention (both P1, SQLite-verifiable, no infra).
2. Message-retention regression test (cheap, locks in good behavior).
3. RLS for tracking/session tables (P1, needs a Postgres target — don't ship unverified).
4. Audit signal + HMAC doc.
5. Data Stewardship Council sign-off → merge.

---

## 5. Merge-readiness checklist

- [x] Purge removes `TelemetryEvent` (by `user_id` and `anonymized_user_key`) — *test green* (`test_platform_data_deletion.py`)
- [x] Purge removes `RawQueryRecord` (by `user_id` and `anonymized_user_key`) — *test green* (`test_platform_data_deletion.py`)
- [x] Purge removes `UserTrackingPreference` — *test green* (`test_platform_data_deletion.py`)
- [x] Existing purge behavior unchanged — *prior tests green* (`test_platform_admin_routes.py` purge/delete tests)
- [x] Deletion SLA documented in `DEPLOYMENT-POLICY.md` (24h; admin + member self-service paths)
- [x] `purge_expired_sessions()` deletes expired/revoked, keeps active — *test green* (`test_platform_auth.py`)
- [x] Session retention window documented + maintenance entry point exists (`POST /api/ops/maintenance/purge-sessions`, 90-day window)
- [x] Message non-retention regression test (off → 0 rows; on → pair) — *test green* (`test_platform_chat_history_retention.py`)
- [x] Tracking + session tables in RLS + migration (`db.py` + Alembic `20260613_0005`)
- [x] Nullable-`union_id` policy implemented (global `tracking_policies` resolvable under member context; super-admin sessions hidden from tenants) — SQLite suite green; **live-Postgres assertion below still required**
- [x] Cross-tenant read blocked on **live Postgres** — *`test_platform_postgres_rls.py` 6/6 green* against PostgreSQL 16.4 + pgvector 0.8.3 (conda-forge), fresh database migrated 0001→0005 (2026-07-10). Verification surfaced and fixed three defects: migration 0001 broke on fresh databases because it consumed the live `get_rls_statements()` aggregate (now pinned to `foundation_rls_statements()`); a missing `datetime` import in the never-executed Postgres-gated tests; and a test-seed gap (union-one tracking override asserted but never seeded).
- [x] Dedicated `SecurityEvent` on identified-mode enable — *test green* (`test_platform_admin_routes.py`)
- [x] HMAC rotation note in `DEPLOYMENT-POLICY.md`
- [~] Full default suite green: `python -m pytest backend -q` — all platform privacy/governance suites green (75 passed, 6 Postgres-gated skips). The Windows-only `ops.py` dashboard `strftime` crash is now fixed; one pre-existing, unrelated failure remains (stale `PlatformSettings` constructor in `test_platform_worker.py`, tracked separately)
- [ ] Data Stewardship Council sign-off recorded — technical gate closed (live-Postgres RLS verified 2026-07-10); readiness package at `legal/DATA-STEWARDSHIP-COUNCIL-SIGNOFF.md` awaits council signatures
