"""Add local auth credentials for demo login.

Revision ID: 20260328_0002
Revises: 20260320_0001
Create Date: 2026-03-28 17:40:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260328_0002"
down_revision = "20260320_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "local_auth_credentials",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("username", sa.String(length=120), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("password_salt", sa.String(length=255), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id"),
        sa.UniqueConstraint("username"),
    )
    op.create_index(op.f("ix_local_auth_credentials_user_id"), "local_auth_credentials", ["user_id"], unique=True)
    op.create_index(op.f("ix_local_auth_credentials_username"), "local_auth_credentials", ["username"], unique=True)


def downgrade() -> None:
    op.drop_index(op.f("ix_local_auth_credentials_username"), table_name="local_auth_credentials")
    op.drop_index(op.f("ix_local_auth_credentials_user_id"), table_name="local_auth_credentials")
    op.drop_table("local_auth_credentials")
