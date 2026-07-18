"""Widen chunk embeddings to 768 for gemini-embedding-001.

The column was created at 384 to suit the deterministic hashing stub. The real
provider (gemini-embedding-001) emits 768 at the documented output width, and
pgvector column widths are fixed at DDL time, so the column has to change.

DESTRUCTIVE: existing rows in chunk_embeddings are deleted. This is not
optional -- a 384-wide vector cannot be cast to 768, and vectors produced by a
different model are not comparable to new ones even if the widths matched.
The data is derived, not source: re-run ingestion for affected documents and
the embeddings are rebuilt. Uploaded document content itself is untouched.

Revision ID: 20260718_0007
Revises: 20260717_0006
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260718_0007"
down_revision = "20260717_0006"
branch_labels = None
depends_on = None


NEW_DIM = 768
OLD_DIM = 384


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def upgrade() -> None:
    # On SQLite (the test suite) the column is plain JSON and needs no change.
    if not _is_postgres():
        return

    op.execute("DELETE FROM chunk_embeddings")
    op.execute(f"ALTER TABLE chunk_embeddings ALTER COLUMN embedding TYPE vector({NEW_DIM})")


def downgrade() -> None:
    if not _is_postgres():
        return

    op.execute("DELETE FROM chunk_embeddings")
    op.execute(f"ALTER TABLE chunk_embeddings ALTER COLUMN embedding TYPE vector({OLD_DIM})")
