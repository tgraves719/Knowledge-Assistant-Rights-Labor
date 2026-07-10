"""Add auth sessions for server-managed browser login.

Revision ID: 20260406_0003
Revises: 20260328_0002
Create Date: 2026-04-06 09:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260406_0003"
down_revision = "20260328_0002"
branch_labels = None
depends_on = None


session_type_enum = postgresql.ENUM("member", "union_admin", "super_admin", name="sessiontype", create_type=False)


def upgrade() -> None:
    bind = op.get_bind()
    session_type_enum.create(bind, checkfirst=True)
    op.create_table(
        "auth_sessions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("union_id", sa.String(length=36), nullable=True),
        sa.Column("session_secret_hash", sa.String(length=128), nullable=False),
        sa.Column("session_type", session_type_enum, nullable=False),
        sa.Column("ip_address", sa.String(length=120), nullable=True),
        sa.Column("user_agent", sa.String(length=500), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("last_seen_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("revoked_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["union_id"], ["unions.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("session_secret_hash"),
    )
    op.create_index(op.f("ix_auth_sessions_user_id"), "auth_sessions", ["user_id"], unique=False)
    op.create_index(op.f("ix_auth_sessions_union_id"), "auth_sessions", ["union_id"], unique=False)
    op.create_index(op.f("ix_auth_sessions_session_secret_hash"), "auth_sessions", ["session_secret_hash"], unique=True)
    op.create_index(op.f("ix_auth_sessions_session_type"), "auth_sessions", ["session_type"], unique=False)
    op.create_index(op.f("ix_auth_sessions_expires_at"), "auth_sessions", ["expires_at"], unique=False)
    op.create_index(op.f("ix_auth_sessions_revoked_at"), "auth_sessions", ["revoked_at"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_auth_sessions_revoked_at"), table_name="auth_sessions")
    op.drop_index(op.f("ix_auth_sessions_expires_at"), table_name="auth_sessions")
    op.drop_index(op.f("ix_auth_sessions_session_type"), table_name="auth_sessions")
    op.drop_index(op.f("ix_auth_sessions_session_secret_hash"), table_name="auth_sessions")
    op.drop_index(op.f("ix_auth_sessions_union_id"), table_name="auth_sessions")
    op.drop_index(op.f("ix_auth_sessions_user_id"), table_name="auth_sessions")
    op.drop_table("auth_sessions")
    session_type_enum.drop(op.get_bind(), checkfirst=True)
