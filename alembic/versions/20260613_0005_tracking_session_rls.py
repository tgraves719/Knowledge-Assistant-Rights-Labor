"""Row level security for tracking + session tables

Adds Postgres RLS to telemetry_events, raw_query_records, tracking_policies,
user_tracking_preferences, and auth_sessions, consistent with the existing tenant model.

- The four tracking tables permit legitimately null-union rows (the global tracking policy,
  anonymous telemetry, super-admin preferences) to resolve under any tenant while isolating
  non-null rows to their union (or super-admin).
- auth_sessions excludes the null-union escape hatch so super-admin sessions stay invisible to
  union tenants.
- Append-only event/session tables use a permissive WITH CHECK (writes are app-controlled and
  may occur before a tenant context exists, e.g. failed-login telemetry or mid-login session
  creation); cross-tenant *read* isolation is the privacy-relevant control.

Revision ID: 20260613_0005
Revises: 20260510_0004
Create Date: 2026-06-13
"""

from __future__ import annotations

from alembic import op

from backend.platform.db import (
    NULLABLE_UNION_RLS_TABLES,
    SESSION_RLS_TABLES,
    tracking_and_session_rls_statements,
)


revision = "20260613_0005"
down_revision = "20260510_0004"
branch_labels = None
depends_on = None


_RLS_TABLES = (*NULLABLE_UNION_RLS_TABLES, *SESSION_RLS_TABLES)


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        # RLS is a Postgres-only control; nothing to do on other backends (e.g. SQLite tests).
        return
    for statement in tracking_and_session_rls_statements():
        op.execute(statement)


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    for table in _RLS_TABLES:
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation_{table} ON {table}")
        op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
