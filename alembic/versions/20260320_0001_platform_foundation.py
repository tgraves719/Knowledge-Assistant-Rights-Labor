"""Initial platform foundation schema.

Revision ID: 20260320_0001
Revises: None
Create Date: 2026-03-20 18:10:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from backend.platform.db import get_rls_statements


revision = "20260320_0001"
down_revision = None
branch_labels = None
depends_on = None


role_enum = sa.Enum("super_admin", "union_admin", "steward_admin", "user", name="role", create_type=False)
document_status_enum = sa.Enum(
    "active", "deleted", "processing", "failed", name="documentstatus", create_type=False
)
ingestion_job_status_enum = sa.Enum(
    "pending", "running", "succeeded", "failed", name="ingestionjobstatus", create_type=False
)
security_severity_enum = sa.Enum(
    "info", "warning", "critical", name="securityseverity", create_type=False
)
notification_status_enum = sa.Enum(
    "pending", "sent", "acknowledged", name="notificationstatus", create_type=False
)


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "unions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("slug", sa.String(length=120), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("union_local_id", sa.String(length=120), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("message_retention_enabled", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("metadata", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("slug"),
        sa.UniqueConstraint("union_local_id"),
    )
    op.create_index(op.f("ix_unions_slug"), "unions", ["slug"], unique=False)
    op.create_index(op.f("ix_unions_union_local_id"), "unions", ["union_local_id"], unique=False)

    op.create_table(
        "users",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("external_auth_id", sa.String(length=255), nullable=True),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("full_name", sa.String(length=255), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
        sa.UniqueConstraint("external_auth_id"),
    )
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=False)

    op.create_table(
        "union_memberships",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("union_id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("role", role_enum, nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["union_id"], ["unions.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("union_id", "user_id", name="uq_union_membership"),
    )
    op.create_index(op.f("ix_union_memberships_role"), "union_memberships", ["role"], unique=False)
    op.create_index(op.f("ix_union_memberships_union_id"), "union_memberships", ["union_id"], unique=False)
    op.create_index(op.f("ix_union_memberships_user_id"), "union_memberships", ["user_id"], unique=False)

    op.create_table(
        "provider_configs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("union_id", sa.String(length=36), nullable=False),
        sa.Column("provider_name", sa.String(length=120), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("encrypted_api_key", sa.Text(), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["union_id"], ["unions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_provider_configs_union_id"), "provider_configs", ["union_id"], unique=False)

    op.create_table(
        "quota_policies",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("union_id", sa.String(length=36), nullable=False),
        sa.Column("requests_per_day", sa.Integer(), nullable=False, server_default="500"),
        sa.Column("tokens_per_day", sa.Integer(), nullable=False, server_default="250000"),
        sa.Column("cost_usd_per_day", sa.Float(), nullable=False, server_default="25"),
        sa.Column("per_user_requests_per_hour", sa.Integer(), nullable=False, server_default="60"),
        sa.Column("warn_threshold_ratio", sa.Float(), nullable=False, server_default="0.8"),
        sa.Column("is_paused", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["union_id"], ["unions.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("union_id"),
    )
    op.create_index(op.f("ix_quota_policies_union_id"), "quota_policies", ["union_id"], unique=True)

    op.create_table(
        "documents",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("union_id", sa.String(length=36), nullable=False),
        sa.Column("uploaded_by_user_id", sa.String(length=36), nullable=True),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("storage_key", sa.String(length=500), nullable=False),
        sa.Column("content_type", sa.String(length=120), nullable=False),
        sa.Column("bytes_size", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", document_status_enum, nullable=False, server_default="processing"),
        sa.Column("metadata", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["union_id"], ["unions.id"]),
        sa.ForeignKeyConstraint(["uploaded_by_user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_documents_union_id"), "documents", ["union_id"], unique=False)

    op.create_table(
        "ingestion_jobs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("union_id", sa.String(length=36), nullable=False),
        sa.Column("document_id", sa.String(length=36), nullable=True),
        sa.Column("requested_by_user_id", sa.String(length=36), nullable=True),
        sa.Column("status", ingestion_job_status_enum, nullable=False, server_default="pending"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"]),
        sa.ForeignKeyConstraint(["requested_by_user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["union_id"], ["unions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_ingestion_jobs_document_id"), "ingestion_jobs", ["document_id"], unique=False)
    op.create_index(op.f("ix_ingestion_jobs_union_id"), "ingestion_jobs", ["union_id"], unique=False)

    op.create_table(
        "chunk_embeddings",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("union_id", sa.String(length=36), nullable=True),
        sa.Column("document_id", sa.String(length=36), nullable=True),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
        sa.Column("embedding", Vector(dim=384), nullable=True),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"]),
        sa.ForeignKeyConstraint(["union_id"], ["unions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_chunk_embeddings_document_id"), "chunk_embeddings", ["document_id"], unique=False)
    op.create_index(op.f("ix_chunk_embeddings_union_id"), "chunk_embeddings", ["union_id"], unique=False)

    op.create_table(
        "chats",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("union_id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=True),
        sa.Column("session_id", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["union_id"], ["unions.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_chats_session_id"), "chats", ["session_id"], unique=False)
    op.create_index(op.f("ix_chats_union_id"), "chats", ["union_id"], unique=False)

    op.create_table(
        "messages",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("union_id", sa.String(length=36), nullable=False),
        sa.Column("chat_id", sa.String(length=36), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["chat_id"], ["chats.id"]),
        sa.ForeignKeyConstraint(["union_id"], ["unions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_messages_chat_id"), "messages", ["chat_id"], unique=False)
    op.create_index(op.f("ix_messages_union_id"), "messages", ["union_id"], unique=False)

    op.create_table(
        "usage_events",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("union_id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=True),
        sa.Column("route", sa.String(length=255), nullable=False),
        sa.Column("request_count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("token_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("estimated_cost_usd", sa.Float(), nullable=False, server_default="0"),
        sa.Column("metadata", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["union_id"], ["unions.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_usage_events_created_at"), "usage_events", ["created_at"], unique=False)
    op.create_index(op.f("ix_usage_events_union_id"), "usage_events", ["union_id"], unique=False)

    op.create_table(
        "audit_events",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("union_id", sa.String(length=36), nullable=True),
        sa.Column("actor_user_id", sa.String(length=36), nullable=True),
        sa.Column("event_type", sa.String(length=120), nullable=False),
        sa.Column("event_payload", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["actor_user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["union_id"], ["unions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_audit_events_created_at"), "audit_events", ["created_at"], unique=False)
    op.create_index(op.f("ix_audit_events_event_type"), "audit_events", ["event_type"], unique=False)
    op.create_index(op.f("ix_audit_events_union_id"), "audit_events", ["union_id"], unique=False)

    op.create_table(
        "security_events",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("union_id", sa.String(length=36), nullable=True),
        sa.Column("user_id", sa.String(length=36), nullable=True),
        sa.Column("event_type", sa.String(length=120), nullable=False),
        sa.Column("severity", security_severity_enum, nullable=False, server_default="info"),
        sa.Column("response_action", sa.String(length=120), nullable=True),
        sa.Column("details", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["union_id"], ["unions.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_security_events_created_at"), "security_events", ["created_at"], unique=False)
    op.create_index(op.f("ix_security_events_event_type"), "security_events", ["event_type"], unique=False)
    op.create_index(op.f("ix_security_events_union_id"), "security_events", ["union_id"], unique=False)

    op.create_table(
        "notifications",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("union_id", sa.String(length=36), nullable=True),
        sa.Column("user_id", sa.String(length=36), nullable=True),
        sa.Column("channel", sa.String(length=32), nullable=False),
        sa.Column("subject", sa.String(length=255), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("status", notification_status_enum, nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["union_id"], ["unions.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_notifications_union_id"), "notifications", ["union_id"], unique=False)

    for statement in get_rls_statements():
        op.execute(statement)


def downgrade() -> None:
    op.drop_index(op.f("ix_notifications_union_id"), table_name="notifications")
    op.drop_table("notifications")
    op.drop_index(op.f("ix_security_events_union_id"), table_name="security_events")
    op.drop_index(op.f("ix_security_events_event_type"), table_name="security_events")
    op.drop_index(op.f("ix_security_events_created_at"), table_name="security_events")
    op.drop_table("security_events")
    op.drop_index(op.f("ix_audit_events_union_id"), table_name="audit_events")
    op.drop_index(op.f("ix_audit_events_event_type"), table_name="audit_events")
    op.drop_index(op.f("ix_audit_events_created_at"), table_name="audit_events")
    op.drop_table("audit_events")
    op.drop_index(op.f("ix_usage_events_union_id"), table_name="usage_events")
    op.drop_index(op.f("ix_usage_events_created_at"), table_name="usage_events")
    op.drop_table("usage_events")
    op.drop_index(op.f("ix_messages_union_id"), table_name="messages")
    op.drop_index(op.f("ix_messages_chat_id"), table_name="messages")
    op.drop_table("messages")
    op.drop_index(op.f("ix_chats_union_id"), table_name="chats")
    op.drop_index(op.f("ix_chats_session_id"), table_name="chats")
    op.drop_table("chats")
    op.drop_index(op.f("ix_chunk_embeddings_union_id"), table_name="chunk_embeddings")
    op.drop_index(op.f("ix_chunk_embeddings_document_id"), table_name="chunk_embeddings")
    op.drop_table("chunk_embeddings")
    op.drop_index(op.f("ix_ingestion_jobs_union_id"), table_name="ingestion_jobs")
    op.drop_index(op.f("ix_ingestion_jobs_document_id"), table_name="ingestion_jobs")
    op.drop_table("ingestion_jobs")
    op.drop_index(op.f("ix_documents_union_id"), table_name="documents")
    op.drop_table("documents")
    op.drop_index(op.f("ix_quota_policies_union_id"), table_name="quota_policies")
    op.drop_table("quota_policies")
    op.drop_index(op.f("ix_provider_configs_union_id"), table_name="provider_configs")
    op.drop_table("provider_configs")
    op.drop_index(op.f("ix_union_memberships_user_id"), table_name="union_memberships")
    op.drop_index(op.f("ix_union_memberships_union_id"), table_name="union_memberships")
    op.drop_index(op.f("ix_union_memberships_role"), table_name="union_memberships")
    op.drop_table("union_memberships")
    op.drop_index(op.f("ix_users_email"), table_name="users")
    op.drop_table("users")
    op.drop_index(op.f("ix_unions_union_local_id"), table_name="unions")
    op.drop_index(op.f("ix_unions_slug"), table_name="unions")
    op.drop_table("unions")

    notification_status_enum.drop(op.get_bind(), checkfirst=True)
    security_severity_enum.drop(op.get_bind(), checkfirst=True)
    ingestion_job_status_enum.drop(op.get_bind(), checkfirst=True)
    document_status_enum.drop(op.get_bind(), checkfirst=True)
    role_enum.drop(op.get_bind(), checkfirst=True)
