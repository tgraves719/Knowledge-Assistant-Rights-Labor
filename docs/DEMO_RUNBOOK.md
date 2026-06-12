# Demo Runbook

This runbook exercises the production-facing upload-first path:

1. local login
2. union-admin upload
3. ingestion
4. query against uploaded documents only

Canonical browser routes:

- member app: `/u/demo-local/`
- tenant admin app: `/u/demo-local/admin`
- superadmin app: `/karl/`

## Preconditions

- `KARL_POSTGRES_URL` points at the target database.
- Frontend CSS is built locally:

```bash
npm run build:css
```

- Migrations are applied:

```bash
. .venv/bin/activate
alembic upgrade head
```

- Legacy contract fallback remains disabled:

```bash
export KARL_LEGACY_CONTRACT_PIPELINE_ENABLED=false
```

- v1 member flow is login-first:
  - tenant identity comes from the route slug
- unsigned users should not enter the old employer/location/city onboarding flow
- the member, tenant admin, and superadmin pages now open sign-in from header/session controls instead of relying on Settings-only forms
- uploaded-document answers should render a main answer first, with `Supporting Sources` available below the answer
- `Open In Document` uses a signed per-source access URL and should work even if the browser has stale or missing bearer auth for the content request
  - uploaded-document unions must not fall back to the legacy contract-manifest path

## Seed demo admin

Defaults:

- username: `union_demo`
- password: `demo_password`
- union slug: `demo-local`

Command:

```bash
. .venv/bin/activate
python scripts/seed_demo_union_admin.py
```

## Seed demo member

Defaults:

- username: `union_member`
- password: `demo_password`
- union slug: `demo-local`

Command:

```bash
. .venv/bin/activate
python scripts/seed_demo_union_member.py
```

## Login

Browser path:

- open `http://localhost:8000/u/demo-local/`
- sign in as `union_member` or `union_demo`

```bash
curl -s \
  -X POST http://localhost:8000/api/auth/session/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"union_demo","password":"demo_password","union_slug":"demo-local"}'
```

The browser UI now uses an HTTP-only session cookie instead of a browser-stored bearer token.

## Upload

Browser path:

- open `http://localhost:8000/u/demo-local/admin`
- sign in as `union_demo`
- upload documents there

```bash
curl -s \
  -X POST http://localhost:8000/api/admin/unions/<UNION_ID>/documents \
  -H "Authorization: Bearer <TOKEN>" \
  -F "file=@/absolute/path/to/document.pdf"
```

For small text files, ingestion may complete inline. For PDFs and richer formats, poll ingestion jobs or run the deferred worker.

## Process deferred jobs

```bash
. .venv/bin/activate
python -m backend.platform.worker
```

## Query

Browser path:

- return to `http://localhost:8000/u/demo-local/`
- sign in as `union_member` or `union_demo`
- ask a question in chat

```bash
curl -s \
  -X POST http://localhost:8000/api/query \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer <TOKEN>" \
  -d '{
    "question": "What does this document say?",
    "union_local_id": "demo-local",
    "contract_id": "tenant-upload",
    "contract_version": "current",
    "session_id": "demo-session"
  }'
```

Expected behavior:

- If no uploaded documents are ready, `/api/query` returns a block instead of falling back to the legacy contract pipeline.
- If uploaded documents are ready, `/api/query` answers from tenant-scoped uploaded chunks only.
- Tenant context comes from the `/u/{union_slug}/` route and authenticated membership, not from member-entered employer/location/city data.
- The member UI now treats the answer as primary and renders supporting evidence in a collapsible drawer.
- Clicking `Open In Document` on a supporting source loads the original uploaded file below the answer.
- For PDFs, the inline preview jumps to the cited page when page metadata is available.
