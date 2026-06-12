# Production Handoff

## Purpose

This branch is converting the current FastAPI RAG prototype into a production-oriented, multi-tenant union SaaS foundation with:

- PostgreSQL as system of record
- `pgvector` retrieval path
- tenant-aware data model
- role-gated admin and ops endpoints
- app-level auth plus PostgreSQL RLS
- quota, sentinel, and deterministic guardrail pipeline

Current branch: `codex/production-foundation`

## Recovery Checklist

If context is lost, re-establish state in this order:

1. Activate the project venv:
   - `. .venv/bin/activate`
2. Confirm Python/tooling:
   - `python --version`
   - `python -m pytest --version`
3. Confirm current branch and worktree:
   - `git status --short --branch`
4. Run focused production-foundation tests:
   - `python -m pytest -q backend/test_production_foundation.py backend/test_persistence_foundation.py backend/test_platform_db.py backend/test_platform_auth.py backend/test_platform_admin_routes.py`
5. Confirm Alembic sees the schema head:
   - `alembic heads`
6. Read:
   - `docs/PRODUCTION_FOUNDATION.md`
   - `docs/PRODUCTION_HANDOFF.md`

## Environment Notes

- Python 3.11 venv was rebuilt and dependencies were reinstalled from `requirements/base.txt`.
- `pytest` must be run through `python -m pytest` from the venv.
- Node tooling is now used for the local Tailwind build:
  - `npm run build:css`
  - output file: `frontend/modular/styles/generated.css`
- The repo was moved to a Mac OS Extended (Journaled) volume after prior external-drive metadata problems.
- Historical `._*` AppleDouble sidecar files caused Git and Alembic issues before the reformat.
- That should be safer now, but if sidecars ever reappear under `.git/` or `alembic/versions/`, they can still break local tooling.
- `.gitignore` already ignores `._*`, but ignored files can still interfere locally.

Recommended cleanup command when things look strange:

- `find . -path './.git' -prune -o -name '._*' -print`

If needed, remove non-`.git` sidecars:

- `find . -path './.git' -prune -o -name '._*' -delete`

## Completed Work

### 1. Environment stabilization

- Rebuilt `.venv` under Python 3.11.
- Reinstalled `requirements/base.txt` and `pytest`.
- Cleaned `.git` corruption caused by `._*` files.
- Added `pytest.ini`.

### 2. Production platform foundation

- Added `backend/platform/` package with:
  - auth adapter
  - DB wiring
  - tenant-aware models
  - storage abstraction
  - encrypted provider config support
  - chat history persistence
  - quotas
  - sentinel events
  - admin and ops routers
  - middleware for auth/governance/security headers

### 3. Guardrail hardening

- Guardrails are deterministic by default.
- `llm-guard` is now opt-in via `KARL_ENABLE_LLM_GUARD=true`.
- Model-backed scanners are separately opt-in via `KARL_ENABLE_MODEL_GUARDRAILS=true`.
- Startup/test runs no longer depend on live Hugging Face downloads.

### 4. Migration and schema baseline

- Added Alembic:
  - `alembic.ini`
  - `alembic/env.py`
  - `alembic/versions/20260320_0001_platform_foundation.py`
- Initial migration creates platform schema, `pgvector` extension, and RLS policies.
- Added `ingestion_jobs` to the platform model.
- `KARL_AUTO_CREATE_TABLES` now defaults to `false`.
- Production compose runs `alembic upgrade head` before API startup.

### 5. Auth and tenant context hardening

- `HeaderAuthAdapter` no longer trusts non-super-admin header roles when the DB is available.
- Tenant roles now come from `UnionMembership`.
- Bootstrap super-admin is supported via `KARL_BOOTSTRAP_SUPER_ADMIN_EMAILS`.
- Middleware sets PostgreSQL request context:
  - `app.current_role`
  - `app.current_union_id`
  - `app.current_user_id`
- Internal helper sessions now inherit request auth context where possible.

### 6. Endpoint-level authorization coverage

- Added tests proving:
  - forged `union_admin` header without membership is rejected
  - membership-backed union admin is allowed

### 7. Live PostgreSQL migration and RLS coverage

- Added `backend/test_platform_postgres_rls.py`.
- Integration fixture creates:
  - disposable test database
  - disposable non-superuser app role
  - Alembic-upgraded schema
- Verified against local PostgreSQL:
  - migration applies cleanly
  - `vector` extension is created
  - RLS is enabled and forced on tenant tables
  - tenant-scoped reads are isolated
  - `super_admin` app role sees global data
  - cross-tenant inserts are blocked

### 8. Postgres defects found and fixed during integration

- Fixed enum persistence mismatch:
  - ORM enums now persist enum `.value` strings, matching the migration enum values
- Fixed migration enum lifecycle:
  - removed redundant explicit enum creation in the initial migration
- Tightened RLS policy generation:
  - `FORCE ROW LEVEL SECURITY` added
  - explicit `WITH CHECK` clauses added for tenant-scoped writes

### 9. Live PostgreSQL route-level tenant isolation coverage

- Added `backend/test_platform_postgres_routes.py`.
- Verified admin and ops routes through the real middleware stack against local PostgreSQL.
- Fixed a bootstrap deadlock under forced RLS:
  - auth resolution now applies a temporary service bootstrap context before membership lookup
  - request auth context overwrites that bootstrap context immediately after resolution
- Verified:
  - union admin can read only same-union admin and ops data
  - cross-tenant admin reads are blocked
  - cross-tenant admin writes are blocked
  - bootstrap `super_admin` can access global admin and ops views
  - same-union provider and quota updates succeed through the live route stack
  - retained chat history is visible only within the owning union
  - unions with retention disabled do not persist chat turns
  - document uploads create tenant-scoped `documents` and `ingestion_jobs` rows together
  - query middleware records tenant-scoped usage events on successful `/api/query` requests
  - quota warnings create persisted sentinel security events and in-app notifications
  - quota-exceeded query blocks do not create usage rows or chats, but they now persist sentinel events correctly
  - text uploads ingest inline into tenant-scoped `chunk_embeddings`
  - tenant retrieval search returns only same-union chunks under the restricted app DB role
  - uploads now flow through a parser abstraction and ingestion service instead of parser logic living in the route
  - unsupported file types are deferred cleanly with explicit job metadata instead of failing implicitly
  - parse artifacts are now persisted to local storage for debugging and future reingest tooling
  - union admins can now list ingestion jobs, inspect a job, and enqueue retries without overwriting prior job history

### 10. Query middleware defect found and fixed during integration

- Fixed an early-return transaction bug in `QueryGovernanceMiddleware`.
- Before the fix, quota-blocked or prompt-blocked `/api/query` requests could return before committing the transient DB session opened by the middleware.
- Result: sentinel events for blocked requests could be silently dropped under live Postgres.
- The middleware now commits transient sessions before returning early on blocked prompt or quota decisions.

### 11. Retrieval execution foundation added

- Added `backend/platform/retrieval.py` with:
  - deterministic offline-safe text embeddings
  - inline text chunking
  - tenant-scoped chunk search
- `ServiceContainer` now exposes the retrieval service.
- Admin text uploads now:
  - create a document row
  - create an ingestion job row
  - inline-ingest plain text into `chunk_embeddings`
  - mark the ingestion job `succeeded` and the document `active` when inline ingestion finishes
- Non-inline-safe file types still remain available for deferred ingestion later.

### 12. Parser abstraction and deferred ingestion scaffolding added

- Added `backend/platform/parsing.py`:
  - parser interface
  - parsed-document data model
  - built-in plain-text parser
  - LiteParse adapter boundary
- Added `backend/platform/ingestion.py`:
  - upload registration
  - parse artifact persistence
  - deferred job processing
  - explicit failure-state updates for documents and ingestion jobs
- Added `backend/platform/worker.py`:
  - deferred pending-job processor with service bootstrap DB context
- Upload route now delegates to the ingestion service.
- Current behavior:
  - plain text files parse inline
  - PDFs and other rich documents remain pending until a parser backend is configured
  - parse artifacts are written under tenant storage paths and referenced from job metadata

### 13. Vendored LiteParse integration started

- Added vendored LiteParse source under `vendor/liteparse/`.
- Removed nested Git metadata from the vendored folder so it can live cleanly inside this repo.
- Added `scripts/build_liteparse.sh`.
- `PlatformSettings` now auto-detects a local LiteParse build at:
  - `vendor/liteparse/dist/src/index.js`
- The detected runtime command is:
  - `node vendor/liteparse/dist/src/index.js`
- Manual sanity check succeeded against:
  - `data/contracts/local7_safeway_pueblo_clerks_2022/source/SW+Pueblo+Clerks+2022.2025.pdf`
- Added guarded integration coverage in `backend/test_platform_liteparse_integration.py`.
- Current repo behavior:
  - if LiteParse is built locally, the parser adapter can use it
  - upload routing still defers rich documents by default
  - deferred worker/job execution is the intended path for PDFs and office docs
  - LiteParse currently runs with OCR disabled by default in our adapter for the fast/local text-PDF path
  - scanned/image-heavy PDFs now enter an OCR auto-retry path before final manual-review escalation, with room for stronger future policies

### 14. Quality gates and OCR retry controls added

- Ingestion now evaluates parse quality before making a document query-ready.
- Weak parses are marked with:
  - `quality_status: needs_review`
  - `quality_reason` describing why extraction is weak
  - `ocr_status` describing whether OCR is recommended, queued, or already attempted
  - `scan_likelihood` describing whether the document looks genuinely scan-like
  - `recommended_action: retry_with_ocr` or `manual_review_after_ocr`
- Review-required documents are held out of retrieval:
  - document stays non-ready
  - no `chunk_embeddings` are written for weak parses
- Retry jobs now accept `ocr_enabled=true`.
- LiteParse-backed low-confidence PDF parses now automatically enqueue a single OCR retry and leave the document in:
  - `quality_status: retrying_with_ocr`
  - `recommended_action: await_ocr_retry`
- If an OCR-enabled retry is still weak, the document falls back to:
  - `quality_status: needs_review`
  - `ocr_status: attempted_needs_review`
  - `recommended_action: manual_review_after_ocr`
- If parser warnings indicate the document is likely unrecoverable, ingestion now skips OCR and goes straight to:
  - `quality_reason: parser_reported_unparseable`
  - `ocr_status: not_recommended`
  - `recommended_action: manual_review_unparseable`
- Ingestion now creates in-app notifications for:
  - ready
  - ready with warnings
  - OCR retry queued
  - needs review
  - failed

### 15. Review escalation and ops surfacing added

- Admin document listing now surfaces:
  - `quality_status`
  - `quality_reason`
  - `ocr_status`
  - `scan_likelihood`
  - `recommended_action`
  - `ready_for_query`
  - latest ingestion job summary
- Added ingestion review escalation endpoint:
  - `POST /api/admin/unions/{union_id}/ingestion-jobs/{job_id}/escalate-review`
- Review-required ingestion outcomes now create sentinel-backed tenant security events.
- Manual escalation creates:
  - a union-scoped `SecurityEvent`
  - a super-admin-facing in-app notification inserted through a narrow elevated path that preserves RLS boundaries
- Live Postgres route coverage now proves this escalation path under the restricted app role.

### 16. Notification acknowledgement added

- Added tenant-scoped ops acknowledgement endpoint:
  - `POST /api/ops/notifications/{notification_id}/acknowledge`
- Union admins can acknowledge only notifications for their own union.
- Super admins retain visibility into global notifications.
- Live Postgres route coverage now proves acknowledgement stays tenant-scoped under RLS.

### 17. Retrieval readiness gating and review-state workflow added

- Tenant retrieval now filters out chunks for documents that are not explicitly query-ready.
- This closes a real gap where stale chunks from a previously active document could have remained searchable during reingest or manual review.
- Added review-state transition endpoint:
  - `POST /api/admin/unions/{union_id}/ingestion-jobs/{job_id}/review-state`
- Review workflow state is now surfaced on document listings and stored in document/job metadata, including states such as:
  - `needs_review`
  - `retrying_with_ocr`
  - `in_review`
  - `resolved`
- Live Postgres route coverage now proves:
  - non-ready documents are excluded from retrieval
  - review-state transitions remain tenant-scoped under RLS

### 18. Tenant-aware `/api/query` branch added for uploaded documents

- `backend/api.py` now routes `/api/query` for unions with ready uploaded documents through a tenant-aware platform retrieval path.
- The old manifest-based retriever is no longer used for those unions.
- Ready uploaded-document queries now flow through the prompt/response pipeline for synthesized answers, with deterministic excerpt fallback preserved as a last resort.
- If a union has uploaded documents but none are ready yet, `/api/query` still fails closed with an ingestion-readiness message.
- Added direct coverage in:
  - `backend/test_platform_query_guard.py`

### 19. Embedder abstraction added

- Added `backend/platform/embeddings.py` as the provider boundary for text embeddings.
- `backend/platform/retrieval.py` now depends on an injected embedder instead of hard-coding the local hashing implementation.
- Current runtime behavior is unchanged by default:
  - deterministic local embeddings remain the active backend
- Added reserved settings for later Google text embeddings:
  - `KARL_EMBEDDING_BACKEND`
  - `KARL_EMBEDDING_DIMENSIONS`
  - `KARL_GOOGLE_EMBEDDING_MODEL`
  - `KARL_GOOGLE_EMBEDDING_API_KEY`
- Added focused coverage in:
  - `backend/test_platform_embeddings.py`

### 20. Review-queue ops visibility added

- Added tenant-scoped ops triage endpoint:
  - `GET /api/ops/review-queue`
- The review queue returns unresolved document review/ingestion items with:
  - document state
  - OCR/extraction detail
  - scan-likelihood detail
  - latest job state
  - related review notifications
  - summary counts for pending vs acknowledged review notifications
- `GET /api/ops/notifications` now also supports review-focused filtering.
- Live Postgres route coverage now proves this queue remains tenant-scoped under RLS.

### 21. Page-aware uploaded-document citations added

- Ingestion now preserves parser page numbers in chunk metadata when page-aware parser output exists.
- Tenant uploaded-document answers can now cite sources as:
  - `Document Title, page N, chunk M`
  when that metadata is available.
- This improves auditability without requiring contract-style structure.

### 22. Worker priority and ETA heuristics added

- Added shared queue heuristics in:
  - `backend/platform/queueing.py`
- Admin ingestion-job ETA estimates now use the same heuristic model as the worker instead of simple FIFO count multiplication.
- Current ETA model now considers:
  - content type
  - file size
  - page count when known
  - OCR mode
  - scan likelihood
- The deferred worker now sorts the full pending job set by priority before applying its processing limit.
- Current priority behavior favors:
  - smaller plain-text jobs
  - lighter non-OCR work
  - retry jobs over equally weighted fresh jobs
  while deprioritizing large OCR-heavy scanned documents.
- Added focused coverage in:
  - `backend/test_platform_worker.py`
  - `backend/test_platform_admin_routes.py`

### 23. Upload-only platform query mode and local demo auth added

- Platform unions now default to upload-first query behavior:
  - if a platform union has ready uploaded documents, `/api/query` uses the tenant document path
  - if a platform union has no ready uploaded documents, legacy manifest-based query fallback is blocked by default
- New setting:
  - `KARL_LEGACY_CONTRACT_PIPELINE_ENABLED=false` by default
- Added local username/password auth for demo and fallback environments:
  - credentials stored in new `local_auth_credentials` table
  - login endpoint: `POST /api/auth/local/login`
  - auth introspection endpoint: `GET /api/auth/me`
- bearer tokens are signed locally and still resolve union role/scope from DB memberships on each request
- uploaded-document source links now prefer signed document access URLs, and the member document route accepts that signed access even if the request carries stale or invalid bearer auth
- Added provisioning helper:
  - `scripts/seed_demo_union_admin.py`
  - defaults to username `union_demo` and password `demo_password`
- Added end-to-end demo smoke coverage:
  - `backend/test_platform_demo_flow.py`
  - exercises local login, upload, and uploaded-document query through the real app routers
- Added operator runbook:
  - `docs/DEMO_RUNBOOK.md`
- Added migration:
  - `alembic/versions/20260328_0002_local_auth_credentials.py`

### 25. Server-managed browser sessions and UX simplification added

- Added DB-backed browser sessions in new table:
  - `auth_sessions`
- Added migration:
  - `alembic/versions/20260406_0003_auth_sessions.py`
- New browser auth endpoints:
  - `POST /api/auth/session/login`
  - `POST /api/auth/session/logout`
  - `GET /api/auth/session/me`
- Tenant member/admin/superadmin browser surfaces now use HTTP-only cookie sessions instead of browser-stored bearer tokens.
- Sliding idle timeout is now role-aware:
  - member: 7 days
  - union admin/steward admin: 3 days
  - super admin: 1 hour
- Middleware now resolves session cookies before compatibility bearer/header auth.
- Session expiry now clears the cookie and prompts re-login in member/admin UIs instead of surfacing raw auth failures.
- Member settings no longer duplicate sign-in controls; header session controls are the canonical entry point.
- Tenant admin page now prioritizes:
  - document upload/status
  - users
  - union settings
  - quota
  - review/alerts
  while moving lower-level diagnostics behind a debug section.

### 24. Tenant-derived route model and legacy member-flow quarantine added

- Added canonical v1 routes:
  - member app: `/u/{union_slug}/`
  - tenant admin app: `/u/{union_slug}/admin`
  - superadmin app: `/karl/`
- Added tenant bootstrap API:
  - `GET /api/tenant/{union_slug}/bootstrap`
- Member and admin frontends now derive tenant identity from the route slug instead of member-entered union discovery data.
- Local login now accepts route-derived tenant context via `X-Tenant-Slug` when `union_slug` is omitted from the payload.
- Middleware now validates route tenant vs authenticated membership and rejects mismatches for non-super-admin users.
- Tenant member flow is now login-first by default:
- unsigned users are blocked from chat
- sign-in UX is moving to header/session-triggered modals across member, tenant admin, and superadmin surfaces rather than Settings-only forms
  - tenant routes no longer use the old employer/location/city onboarding to establish union context
  - tenant-upload query mode is enforced from authenticated membership plus route context
- Tenant admin flow now loads union bootstrap/config from the route and no longer depends on manually picking a union after arriving on the tenant admin page.
- Added superadmin HTML surface:
  - `frontend/modular/superadmin.html`
- Added tenant settings management support for:
  - `member_login_required`
  - branding payload
  - optional custom-domain metadata
  - feature metadata
- Added demo member seed helper:
  - `scripts/seed_demo_union_member.py`
  - defaults to username `union_member` and password `demo_password`
- Added tenant-route smoke coverage in:
  - `backend/test_platform_demo_flow.py`
- Legacy static routes remain only as internal/reference entrypoints and are redirected away from the production demo path.

### 25. Answer-first uploaded-document evidence viewer added

- Uploaded-document query responses now return richer source metadata for UI evidence rendering, including:
  - `document_title`
  - `content_type`
  - `document_content_url`
  - `excerpt`
  - page-aware metadata when available
- Added authenticated member document-viewer endpoint:
  - `GET /api/member/documents/{document_id}/content`
- Member chat now renders:
  - primary answer first
  - supporting sources in a collapsible drawer
  - inline document preview below the answer when a source is opened
- PDF sources now open to the cited page when page metadata exists.
- Uploaded-document answers no longer append a synthetic trailing `Sources:` footer; evidence is rendered separately by the UI.
- Added focused coverage in:
  - `backend/test_platform_demo_flow.py`

### 26. Tailwind CDN removed from production demo pages

- Added root frontend build files:
  - `package.json`
  - `tailwind.config.js`
  - `frontend/modular/styles/tailwind.css`
- Built shared Tailwind output:
  - `frontend/modular/styles/generated.css`
- Member, tenant admin, and superadmin pages now load the local built stylesheet instead of `cdn.tailwindcss.com`.
- This removes the browser production warning and gives the demo a real local frontend asset path.

### 27. Superadmin union-operations surface improved

- Added superadmin platform summary endpoint:
  - `GET /api/admin/platform-summary`
- The summary returns cross-tenant totals for:
  - unions
  - active vs inactive unions
  - users
  - documents
  - pending review items
  - pending notifications
- Superadmin union cards now surface more operational detail directly in the UI:
  - active/inactive state
  - login-required policy
  - retention setting
  - optional custom domain
  - per-union user/document/review counts when available
- Union user creation now supports optional local-auth credential seeding:
  - `username`
  - `password`
- Union user listings now show whether a user has local login enabled and, when available, their local username.
- This makes `/karl/` usable for creating demo-ready tenant users without dropping back to seed scripts.

## Current Validation Status

Focused suite currently green:

- `backend/test_production_foundation.py`
- `backend/test_persistence_foundation.py`
- `backend/test_platform_db.py`
- `backend/test_platform_auth.py`
- `backend/test_platform_admin_routes.py`
- `backend/test_platform_ingestion.py`
- `backend/test_platform_liteparse_integration.py`
- `backend/test_platform_postgres_routes.py`
- `backend/test_platform_query_guard.py`
- `backend/test_platform_embeddings.py`
- `backend/test_platform_worker.py`
- `backend/test_platform_query_guard.py`
- `backend/test_platform_demo_flow.py`

Expected command:

```bash
python -m pytest -q backend/test_production_foundation.py backend/test_persistence_foundation.py backend/test_platform_db.py backend/test_platform_auth.py backend/test_platform_admin_routes.py backend/test_platform_ingestion.py backend/test_platform_liteparse_integration.py backend/test_platform_postgres_routes.py backend/test_platform_query_guard.py backend/test_platform_embeddings.py

# or include worker coverage explicitly
python -m pytest -q backend/test_production_foundation.py backend/test_persistence_foundation.py backend/test_platform_db.py backend/test_platform_auth.py backend/test_platform_admin_routes.py backend/test_platform_ingestion.py backend/test_platform_liteparse_integration.py backend/test_platform_postgres_routes.py backend/test_platform_query_guard.py backend/test_platform_embeddings.py backend/test_platform_worker.py

# or include the demo flow smoke coverage
python -m pytest -q backend/test_production_foundation.py backend/test_persistence_foundation.py backend/test_platform_db.py backend/test_platform_auth.py backend/test_platform_admin_routes.py backend/test_platform_ingestion.py backend/test_platform_liteparse_integration.py backend/test_platform_postgres_routes.py backend/test_platform_query_guard.py backend/test_platform_embeddings.py backend/test_platform_worker.py backend/test_platform_demo_flow.py
```

Expected result at handoff time:

- `21 passed` on the focused tenant-route/auth/admin/query/demo subset
- `36 passed, 17 skipped` on the larger production-foundation suite when including the existing broader set

Alembic status:

```bash
alembic heads
```

Expected result:

- `20260328_0002 (head)`

Live PostgreSQL integration test command:

```bash
find alembic -name '._*' -delete
export KARL_TEST_POSTGRES_ADMIN_URL='postgresql+psycopg://michaelm@localhost:5432/postgres'
python -m pytest -q backend/test_platform_postgres_rls.py
```

Expected result at handoff time:

- `4 passed`

Live PostgreSQL route integration command:

```bash
find alembic -name '._*' -delete
export KARL_TEST_POSTGRES_ADMIN_URL='postgresql+psycopg://michaelm@localhost:5432/postgres'
python -m pytest -q backend/test_platform_postgres_routes.py
```

Expected result at handoff time:

- `17 passed`

## Key Files

- `backend/platform/models.py`
- `backend/platform/db.py`
- `backend/platform/auth.py`
- `backend/platform/middleware.py`
- `backend/platform/service_container.py`
- `backend/platform/inference.py`
- `backend/platform/chat_history.py`
- `backend/test_platform_postgres_routes.py`
- `backend/platform/routers/admin.py`
- `backend/platform/routers/auth.py`
- `backend/platform/routers/ops.py`
- `backend/platform/ingestion.py`
- `backend/platform/sentinel.py`
- `backend/api.py`
- `frontend/modular/src/app.js`
- `frontend/modular/admin.js`
- `frontend/modular/admin.html`
- `frontend/modular/superadmin.html`
- `alembic/versions/20260320_0001_platform_foundation.py`
- `docs/PRODUCTION_FOUNDATION.md`

## Known Risks / Gaps

### 1. Some legacy code paths still need deeper production review

The branch is a production foundation layered onto a pre-existing prototype. More endpoint-by-endpoint review is still needed for:

- admin destructive actions
- query path persistence details
- document ingestion flow into tenant-scoped embeddings
- remaining legacy contract-manifest UI and frontend assets that should be removed instead of merely quarantined
- broader frontend build/test automation beyond the current Tailwind CSS build

### 2. External-drive metadata remains a local hazard

`._*` files may reappear and interfere with local tooling.

## Next Recommended Slice

### Immediate next target

1. Remove or isolate the remaining legacy member/admin frontend paths so `/u/{union_slug}/` and `/karl/` are the only production demo entrypoints.
2. Add superadmin union-operations polish:
   - create/edit/deactivate unions
   - tenant branding/auth policy management
   - seeded user/admin management
3. Add a true member self/profile API for optional preferences without affecting tenant derivation.
4. Add a repeatable frontend smoke/build check so CSS rebuilds are part of the normal demo workflow.

### After that

Continue into:

1. provider/auth adapter hardening
2. richer tenant bootstrap/domain mapping support
3. more admin/superadmin endpoint coverage
4. rate limits and abuse controls at middleware/reverse-proxy boundaries

## Important Decisions Already Made

- Production startup should prefer migrations, not `create_all()`.
- Deterministic guardrails are the default safe baseline.
- Model-backed guardrails are opt-in.
- Header roles are not trusted for tenant admins when DB-backed membership data exists.
- Security and tenant isolation take priority over convenience.
- Tenant identity for production flows is route-derived and membership-validated, not discovered from member-entered onboarding data.
