"""Invite-code audience (member vs steward) + usage timeline.

Splits printed QR placements into two audiences:
- member codes enroll rank-and-file members, pinned to a single contract;
- steward codes enroll stewards, who see every contract in their union and
  switch between them (no pin).

Also records first/last use so per-placement usage reads beyond the bare
`use_count` counter. Existing codes predate the split and are all member codes,
so the backfill defaults `audience` to 'member'.

Revision ID: 20260720_0010
Revises: 20260718_0009
Create Date: 2026-07-20
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260720_0010"
down_revision = "20260718_0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "invite_codes",
        sa.Column("audience", sa.String(length=16), nullable=False, server_default="member"),
    )
    op.add_column("invite_codes", sa.Column("first_used_at", sa.DateTime(), nullable=True))
    op.add_column("invite_codes", sa.Column("last_used_at", sa.DateTime(), nullable=True))
    op.create_index("ix_invite_codes_audience", "invite_codes", ["audience"])


def downgrade() -> None:
    op.drop_index("ix_invite_codes_audience", table_name="invite_codes")
    op.drop_column("invite_codes", "last_used_at")
    op.drop_column("invite_codes", "first_used_at")
    op.drop_column("invite_codes", "audience")
