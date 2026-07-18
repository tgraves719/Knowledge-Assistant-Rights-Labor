"""Scope documents to a contract so retrieval can filter by it.

Without this, tenant retrieval filtered only on union_id, so every member
query searched every document in the union -- a clerks question could be
answered from the meat agreement, with a citation making it look verified.

Nullable and unbacked by a foreign key on purpose: contract identifiers are
pack-level strings (e.g. "local7_safeway_pueblo_meat_2022") owned by the
contract packs, not rows in this schema. NULL keeps existing documents
reachable by unscoped queries rather than silently hiding them.

Revision ID: 20260718_0008
Revises: 20260718_0007
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260718_0008"
down_revision = "20260718_0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("contract_id", sa.String(length=255), nullable=True))
    op.create_index(op.f("ix_documents_contract_id"), "documents", ["contract_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_documents_contract_id"), table_name="documents")
    op.drop_column("documents", "contract_id")
