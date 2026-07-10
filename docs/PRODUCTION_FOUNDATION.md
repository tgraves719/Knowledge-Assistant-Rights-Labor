# Production Foundation

This project now includes a production-oriented platform layer under `backend/platform/` that adds:

- PostgreSQL-backed tenant models for unions, users, memberships, provider configs, quotas, documents, chats, audit events, security events, and notifications
- Alembic migrations for the platform schema, including `pgvector` and RLS bootstrap
- Optional PostgreSQL row-level security policy application on startup
- Header-based auth adapter that can be replaced by a managed IdP integration later
- Query governance middleware for guardrails, quota enforcement, and sentinel event creation
- Local-disk storage abstraction for union document uploads
- Admin and ops endpoints for union management, provider configuration, quotas, document uploads, and security visibility
- Canonical tenant-derived frontend routes for member, tenant admin, and superadmin entrypoints
- Session-bound chat persistence for unions that enable message retention
- PostgreSQL/`pgvector` retrieval backend with Chroma fallback

## Key environment variables

- `KARL_POSTGRES_URL`: SQLAlchemy PostgreSQL DSN, for example `postgresql+psycopg://karl:change-me@localhost:5432/karl`
- `KARL_AUTO_CREATE_TABLES`: `true` or `false`
- `KARL_APPLY_RLS_POLICIES`: `true` or `false`
- `KARL_SECRET_ENCRYPTION_KEY`: secret seed used to derive the Fernet encryption key for provider credentials
- `KARL_ALLOWED_ORIGINS`: comma-separated CORS allow list
- `KARL_STORAGE_ROOT`: filesystem path for uploaded union documents
- `KARL_BOOTSTRAP_SUPER_ADMIN_EMAILS`: reserved for future bootstrap flow
- `KARL_EMBEDDING_BACKEND`: embedding provider selector, currently `deterministic` by default
- `KARL_EMBEDDING_DIMENSIONS`: vector dimension size for the selected embedder
- `KARL_GOOGLE_EMBEDDING_MODEL`: reserved model name for future Google text embeddings
- `KARL_GOOGLE_EMBEDDING_API_KEY`: reserved API key for future Google text embeddings
- `KARL_LEGACY_CONTRACT_PIPELINE_ENABLED`: defaults to `false`; when disabled, platform unions must use uploaded documents instead of the legacy manifest contract pipeline
- `KARL_LOCAL_AUTH_TOKEN_TTL_SECONDS`: signed bearer token lifetime for local demo auth

## Canonical frontend routes

- Member app: `/u/{union_slug}/`
- Tenant admin app: `/u/{union_slug}/admin`
- Superadmin app: `/karl/`

In v1, tenant identity for production/demo flows comes from the route slug plus authenticated membership. Member-entered employer/location/city onboarding is no longer the source of truth for tenant resolution.

## Frontend asset build

- Tailwind is now built locally instead of loaded from the CDN warning path.
- Root frontend build files:
  - `package.json`
  - `tailwind.config.js`
  - `frontend/modular/styles/tailwind.css`
  - generated output: `frontend/modular/styles/generated.css`
- Build command:

```bash
npm run build:css
```

## Header-based auth bridge

Until a concrete IdP is chosen, the backend resolves request identity from headers:

- `X-Auth-User-Id`
- `X-Auth-Email`
- `X-Auth-Name`
- `X-Auth-Role`
- `X-Union-Slug` or `X-Union-Local-Id`
- `X-Tenant-Slug`

If a matching user and membership exist in PostgreSQL, the DB role wins over the header role.

## New endpoints

- `GET /api/admin/me`
- `GET /api/tenant/{union_slug}/bootstrap`
- `POST /api/auth/session/login`
- `POST /api/auth/session/logout`
- `GET /api/auth/session/me`
- `POST /api/auth/local/login`
- `GET /api/auth/me`
- `GET /api/admin/unions`
- `POST /api/admin/unions`
- `PUT /api/admin/unions/{union_id}`
- `GET /api/admin/unions/{union_id}/users`
- `POST /api/admin/unions/{union_id}/users`
- `GET /api/admin/unions/{union_id}/provider`
- `PUT /api/admin/unions/{union_id}/provider`
- `GET /api/admin/unions/{union_id}/quota`
- `PUT /api/admin/unions/{union_id}/quota`
- `GET /api/admin/unions/{union_id}/documents`
- `POST /api/admin/unions/{union_id}/documents`
- `GET /api/admin/unions/{union_id}/ingestion-jobs`
- `GET /api/admin/unions/{union_id}/ingestion-jobs/{job_id}`
- `POST /api/admin/unions/{union_id}/ingestion-jobs/{job_id}/retry`
- `POST /api/admin/unions/{union_id}/ingestion-jobs/{job_id}/escalate-review`
- `POST /api/admin/unions/{union_id}/ingestion-jobs/{job_id}/review-state`
- `GET /api/admin/unions/{union_id}/chats`
- `GET /api/admin/unions/{union_id}/chats/{chat_id}`
- `GET /api/ops/security-events`
- `GET /api/ops/notifications`
- `POST /api/ops/notifications/{notification_id}/acknowledge`
- `GET /api/ops/review-queue`
- `GET /api/ops/usage`

## Current scope

This foundation now attempts PostgreSQL/`pgvector` retrieval first and falls back to Chroma when the database or vector extension is unavailable. Full tenant-scoped ingestion and migration of existing artifacts into the new `chunk_embeddings` table still need a dedicated indexing path.

## Migration workflow

- Run `alembic upgrade head` against the configured `KARL_POSTGRES_URL`
- Production compose now applies migrations before starting the API
- `KARL_AUTO_CREATE_TABLES` defaults to `false` so production startup does not silently drift schema

## Integration testing notes

- Live PostgreSQL RLS coverage exists in `backend/test_platform_postgres_rls.py`
- Live PostgreSQL route coverage exists in `backend/test_platform_postgres_routes.py`
- The integration fixture uses a disposable non-superuser app role because owner/superuser sessions can bypass RLS and give false confidence
- Middleware now applies a temporary bootstrap DB context before membership lookup, then replaces it with the resolved request tenant context before tenant-scoped work continues
- Chat-history session binding now uses the same bootstrap-then-resolve pattern so retained chat persistence can safely resolve unions under forced RLS
- Middleware now also validates route-derived tenant slug against authenticated membership for non-super-admin users
- Admin document uploads now create a matching tenant-scoped `IngestionJob` row so ingestion state begins at upload time instead of being inferred later
- Query governance now persists quota/prompt block sentinel events even on early middleware returns by committing transient DB sessions before returning blocked responses
- Tenant retrieval now has a deterministic local baseline in `backend/platform/retrieval.py`, and plain-text uploads are ingested directly into tenant-scoped `chunk_embeddings`
- Embeddings now sit behind `backend/platform/embeddings.py`, so ingestion and retrieval are no longer hard-wired to the local deterministic embedder
- The current runtime backend is still deterministic by default, but the embedder boundary and settings are now in place for a later Google text-embedding implementation
- Upload parsing now sits behind `backend/platform/parsing.py` and `backend/platform/ingestion.py`, so rich-document parsers like LiteParse can be integrated without rewriting route or business logic
- Vendored LiteParse source now lives under `vendor/liteparse/`, with local builds enabled via `scripts/build_liteparse.sh` and auto-detected through `KARL_LITEPARSE_EXECUTABLE`
- The current LiteParse adapter still starts with OCR-disabled parsing for the fast text-PDF path
- Low-confidence LiteParse PDF parses now automatically enqueue one OCR retry before the document is escalated into manual review
- Ingestion applies parse-quality heuristics before activation, supports OCR-enabled retries, and keeps weak parses out of retrieval until reprocessed or reviewed
- Weak parses now also carry explicit `quality_reason` and `ocr_status` metadata so admin and ops views can distinguish:
  - OCR recommended but not yet attempted
  - OCR retry queued
  - OCR already attempted and still weak
- Weak parses now also carry `scan_likelihood`, and the OCR policy is more selective:
  - likely scanned documents can auto-queue one OCR retry
  - parser-reported encrypted/corrupt/unparseable documents skip OCR and go straight to manual review
  - low-confidence but not clearly scan-like documents can still require human review without automatic OCR
- Retrieval now enforces document readiness by filtering out chunks from non-ready documents, including documents under reingest or manual review
- Deferred ingestion now uses shared queue heuristics for both worker ordering and admin-facing ETA estimates, so the visible queue timing is aligned with how the worker will actually prioritize text, OCR, and large scanned jobs
- Admin document listings now surface document readiness, latest ingestion quality state, and review workflow state, and review-required jobs can be escalated or moved through explicit review-state transitions without weakening tenant RLS
- Ops notifications can now be acknowledged through a tenant-scoped endpoint so review queues do not remain a flat list of pending alerts
- Ops now includes a tenant-scoped unresolved review queue view so admins can triage ingestion/review issues without stitching together documents, jobs, and notifications manually
- `/api/query` now routes unions with ready uploaded documents through a tenant-aware platform retrieval path with synthesized answers grounded in ready uploaded chunks, and still fails closed when uploads exist but none are ready
- For platform unions, legacy contract-manifest `/api/query` fallback is now disabled by default so demo and production testing stay on the same upload-ingest-query path
- When parser page data exists, uploaded-document chunks now retain page metadata through ingestion so tenant-query citations can render as `document, page, chunk`
- Local username/password auth now issues server-managed browser sessions for the production tenant UIs:
  - browser login uses `POST /api/auth/session/login`
  - browser logout uses `POST /api/auth/session/logout`
  - browser signed-in state hydrates from `GET /api/auth/session/me`
  - session state is stored in an HTTP-only cookie, not browser-visible bearer storage
  - idle timeout is role-aware and sliding:
    - member: 7 days
    - union admin and steward admin: 3 days
    - super admin: 1 hour
- Local demo bearer auth still exists as a narrow compatibility path:
  - credentials are stored in `local_auth_credentials`
- `POST /api/auth/local/login` returns a signed bearer token
- uploaded-document source viewing can also use a signed per-source access URL, so member evidence preview does not depend exclusively on the current browser bearer state
  - bearer-token requests still resolve roles and union scope from `UnionMembership`, not token claims
- Union user creation now supports optional local-auth credential seeding for demo/local environments:
  - superadmin or union admin can provide `username` and `password` when creating a union user
  - user listings expose whether a local login exists and the configured username
- Superadmin now has a platform summary endpoint:
  - `GET /api/admin/platform-summary`
  - returns cross-tenant counts for unions, users, documents, pending reviews, and pending notifications
- Uploaded-document member viewing now has an authenticated file endpoint:
  - `GET /api/member/documents/{document_id}/content`
- The member UI now treats uploaded-document evidence as a first-class UI surface:
  - answers render first
  - supporting excerpts live in a collapsible drawer
  - opening a source loads the original uploaded file inline
  - PDFs jump to the cited page when page metadata exists
- Local login can now derive tenant context from the route via `X-Tenant-Slug`, so tenant member/admin pages do not need a separate union-discovery workflow
- A dedicated upload-first demo runbook now exists in `docs/DEMO_RUNBOOK.md`, and the repo includes a smoke test for the supported user flow:
  - local login
  - upload
  - ingestion
  - query against uploaded documents only
- Tenant frontends now treat login as the first step by default:
- unsigned tenant users are blocked from chat
- member/admin/superadmin browser auth now prefers session cookies over local bearer storage
- member/admin/superadmin auth UX is being normalized around explicit header sign-in/session controls instead of tenant discovery or settings-only auth forms
  - `/u/{union_slug}/` and `/u/{union_slug}/admin` are the canonical production demo entrypoints
  - `/karl/` is the superadmin entrypoint
  - legacy static member/admin entrypoints are now only internal reference paths and are redirected away from the production demo flow
- On this workspace filesystem, delete `alembic` `._*` sidecar files before running Alembic-driven tests if discovery errors appear
