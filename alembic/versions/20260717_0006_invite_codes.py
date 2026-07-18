"""Invite codes for the QR enrollment flow.

Creates the tenant-scoped invite_codes table (one row per printed QR placement:
break-room poster, union board, steward hand-cards) plus an attribution column
on auth_sessions so member sessions can be traced back to the invite that
enrolled them.

RLS: invite codes are union-owned, but the anonymous join flow must resolve a
code before any tenant context exists — reads permit the no-context case while
established tenant contexts stay isolated to their own union (see
backend.platform.db.invite_rls_statements for the rationale).

Revision ID: 20260717_0006
Revises: 20260613_0005
Create Date: 2026-07-17
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from backend.platform.db import INVITE_RLS_TABLES, invite_rls_statements


revision = "20260717_0006"
down_revision = "20260613_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "invite_codes",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("union_id", sa.String(length=36), sa.ForeignKey("unions.id"), nullable=False),
        sa.Column("code", sa.String(length=64), nullable=False),
        sa.Column("label", sa.String(length=255), nullable=False, server_default=""),
        sa.Column("contract_id", sa.String(length=255), nullable=True),
        sa.Column("created_by", sa.String(length=36), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("max_uses", sa.Integer(), nullable=True),
        sa.Column("use_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("revoked_at", sa.DateTime(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_invite_codes_union_id", "invite_codes", ["union_id"])
    op.create_index("ix_invite_codes_code", "invite_codes", ["code"], unique=True)
    op.create_index("ix_invite_codes_expires_at", "invite_codes", ["expires_at"])
    op.create_index("ix_invite_codes_revoked_at", "invite_codes", ["revoked_at"])

    op.add_column(
        "auth_sessions",
        sa.Column("invite_code_id", sa.String(length=36), sa.ForeignKey("invite_codes.id"), nullable=True),
    )
    op.create_index("ix_auth_sessions_invite_code_id", "auth_sessions", ["invite_code_id"])

    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        for statement in invite_rls_statements():
            op.execute(statement)


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        for table in INVITE_RLS_TABLES:
            op.execute(f"DROP POLICY IF EXISTS tenant_isolation_{table} ON {table}")
            op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY")
            op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
    op.drop_index("ix_auth_sessions_invite_code_id", table_name="auth_sessions")
    op.drop_column("auth_sessions", "invite_code_id")
    op.drop_index("ix_invite_codes_revoked_at", table_name="invite_codes")
    op.drop_index("ix_invite_codes_expires_at", table_name="invite_codes")
    op.drop_index("ix_invite_codes_code", table_name="invite_codes")
    op.drop_index("ix_invite_codes_union_id", table_name="invite_codes")
    op.drop_table("invite_codes")
