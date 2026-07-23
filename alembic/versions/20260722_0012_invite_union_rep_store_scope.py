"""Three-tier invite audiences + steward store allowlist.

Splits QR placements into three access tiers:
- member (Tier 1): pinned to one contract (unchanged);
- steward (Tier 2): scoped to a store — an explicit set of contracts in the
  new ``contract_ids`` column;
- union_rep (Tier 3): every contract in the local (no pin, no allowlist).

The old two-tier model used ``steward`` to mean "all contracts", which is now
``union_rep``. Existing ``steward`` codes are therefore backfilled to
``union_rep`` so their scope is preserved. New per-store steward codes carry a
``contract_ids`` list; all existing codes default to an empty list.

Revision ID: 20260722_0012
Revises: 20260721_0011
Create Date: 2026-07-22
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260722_0012"
down_revision = "20260721_0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "invite_codes",
        sa.Column("contract_ids", sa.JSON(), nullable=False, server_default="[]"),
    )
    # Preserve behaviour: a pre-split steward code meant "all contracts", which
    # is now the union_rep tier.
    op.execute("UPDATE invite_codes SET audience = 'union_rep' WHERE audience = 'steward'")


def downgrade() -> None:
    # Collapse the new tier back onto the old two-tier meaning before dropping
    # the column so a downgrade leaves valid audiences.
    op.execute("UPDATE invite_codes SET audience = 'steward' WHERE audience IN ('steward', 'union_rep')")
    op.drop_column("invite_codes", "contract_ids")
