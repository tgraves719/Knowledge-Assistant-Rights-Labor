"""Attach an optional source PDF to a document.

Members need to verify contract language against the printed book. The text
we index comes from markdown, so citations could name a section but never
show it. This stores the originating PDF alongside the document -- stored
only, never parsed, so it does not depend on the PDF parser being available
in the runtime image.

Revision ID: 20260718_0009
Revises: 20260718_0008
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260718_0009"
down_revision = "20260718_0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("source_pdf_key", sa.String(length=500), nullable=True))


def downgrade() -> None:
    op.drop_column("documents", "source_pdf_key")
