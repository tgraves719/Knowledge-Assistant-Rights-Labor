"""Attribute usage events to the invite code that enrolled the member.

Adds usage_events.invite_code_id so token/request/cost usage can be metered
per printed QR placement, not just per union. Nullable: pre-existing usage and
any non-QR session have no originating code.

Revision ID: 20260721_0011
Revises: 20260720_0010
Create Date: 2026-07-21
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260721_0011"
down_revision = "20260720_0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "usage_events",
        sa.Column("invite_code_id", sa.String(length=36), sa.ForeignKey("invite_codes.id"), nullable=True),
    )
    op.create_index("ix_usage_events_invite_code_id", "usage_events", ["invite_code_id"])


def downgrade() -> None:
    op.drop_index("ix_usage_events_invite_code_id", table_name="usage_events")
    op.drop_column("usage_events", "invite_code_id")
