"""Tracking governance and telemetry foundation

Revision ID: 20260510_0004
Revises: 20260406_0003
Create Date: 2026-05-10
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260510_0004"
down_revision = "20260406_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    tracking_mode = postgresql.ENUM("none", "bug_and_journey", "usage_and_ux", "both", name="trackingmode", create_type=False)
    privacy_mode = postgresql.ENUM("anonymized", "identified", name="trackingprivacymode", create_type=False)
    member_choice_mode = postgresql.ENUM("none", "bug_only_or_full", "full_opt_out", name="membertrackingchoicemode", create_type=False)
    raw_query_storage_mode = postgresql.ENUM("disabled", "enabled_anonymized", "enabled_identified", name="rawquerystoragemode", create_type=False)
    tracking_preference = postgresql.ENUM("system_default", "bug_only", "full", "off", name="trackingpreference", create_type=False)

    bind = op.get_bind()
    postgresql.ENUM("none", "bug_and_journey", "usage_and_ux", "both", name="trackingmode").create(bind, checkfirst=True)
    postgresql.ENUM("anonymized", "identified", name="trackingprivacymode").create(bind, checkfirst=True)
    postgresql.ENUM("none", "bug_only_or_full", "full_opt_out", name="membertrackingchoicemode").create(bind, checkfirst=True)
    postgresql.ENUM("disabled", "enabled_anonymized", "enabled_identified", name="rawquerystoragemode").create(bind, checkfirst=True)
    postgresql.ENUM("system_default", "bug_only", "full", "off", name="trackingpreference").create(bind, checkfirst=True)

    op.create_table(
        "tracking_policies",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("union_id", sa.String(length=36), nullable=True),
        sa.Column("tracking_mode", tracking_mode, nullable=False),
        sa.Column("privacy_mode", privacy_mode, nullable=False),
        sa.Column("member_choice_mode", member_choice_mode, nullable=False),
        sa.Column("raw_query_storage_mode", raw_query_storage_mode, nullable=False),
        sa.Column("default_member_preference", tracking_preference, nullable=False),
        sa.Column("allow_union_override", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["union_id"], ["unions.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("union_id", name="uq_tracking_policy_union"),
    )
    op.create_index(op.f("ix_tracking_policies_union_id"), "tracking_policies", ["union_id"], unique=False)

    op.create_table(
        "user_tracking_preferences",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("union_id", sa.String(length=36), nullable=True),
        sa.Column("preference", tracking_preference, nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["union_id"], ["unions.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "union_id", name="uq_user_tracking_preference"),
    )
    op.create_index(op.f("ix_user_tracking_preferences_union_id"), "user_tracking_preferences", ["union_id"], unique=False)
    op.create_index(op.f("ix_user_tracking_preferences_user_id"), "user_tracking_preferences", ["user_id"], unique=False)

    op.create_table(
        "telemetry_events",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("union_id", sa.String(length=36), nullable=True),
        sa.Column("user_id", sa.String(length=36), nullable=True),
        sa.Column("session_id", sa.String(length=255), nullable=True),
        sa.Column("route", sa.String(length=255), nullable=True),
        sa.Column("category", sa.String(length=32), nullable=False),
        sa.Column("event_type", sa.String(length=120), nullable=False),
        sa.Column("anonymized_user_key", sa.String(length=128), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["union_id"], ["unions.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_telemetry_events_union_id"), "telemetry_events", ["union_id"], unique=False)
    op.create_index(op.f("ix_telemetry_events_user_id"), "telemetry_events", ["user_id"], unique=False)
    op.create_index(op.f("ix_telemetry_events_session_id"), "telemetry_events", ["session_id"], unique=False)
    op.create_index(op.f("ix_telemetry_events_route"), "telemetry_events", ["route"], unique=False)
    op.create_index(op.f("ix_telemetry_events_category"), "telemetry_events", ["category"], unique=False)
    op.create_index(op.f("ix_telemetry_events_event_type"), "telemetry_events", ["event_type"], unique=False)
    op.create_index(op.f("ix_telemetry_events_anonymized_user_key"), "telemetry_events", ["anonymized_user_key"], unique=False)
    op.create_index(op.f("ix_telemetry_events_created_at"), "telemetry_events", ["created_at"], unique=False)

    op.create_table(
        "raw_query_records",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("union_id", sa.String(length=36), nullable=True),
        sa.Column("user_id", sa.String(length=36), nullable=True),
        sa.Column("session_id", sa.String(length=255), nullable=True),
        sa.Column("route", sa.String(length=255), nullable=True),
        sa.Column("anonymized_user_key", sa.String(length=128), nullable=True),
        sa.Column("question_text", sa.Text(), nullable=True),
        sa.Column("answer_text", sa.Text(), nullable=True),
        sa.Column("provider_name", sa.String(length=120), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["union_id"], ["unions.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_raw_query_records_union_id"), "raw_query_records", ["union_id"], unique=False)
    op.create_index(op.f("ix_raw_query_records_user_id"), "raw_query_records", ["user_id"], unique=False)
    op.create_index(op.f("ix_raw_query_records_session_id"), "raw_query_records", ["session_id"], unique=False)
    op.create_index(op.f("ix_raw_query_records_route"), "raw_query_records", ["route"], unique=False)
    op.create_index(op.f("ix_raw_query_records_anonymized_user_key"), "raw_query_records", ["anonymized_user_key"], unique=False)
    op.create_index(op.f("ix_raw_query_records_created_at"), "raw_query_records", ["created_at"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_raw_query_records_created_at"), table_name="raw_query_records")
    op.drop_index(op.f("ix_raw_query_records_anonymized_user_key"), table_name="raw_query_records")
    op.drop_index(op.f("ix_raw_query_records_route"), table_name="raw_query_records")
    op.drop_index(op.f("ix_raw_query_records_session_id"), table_name="raw_query_records")
    op.drop_index(op.f("ix_raw_query_records_user_id"), table_name="raw_query_records")
    op.drop_index(op.f("ix_raw_query_records_union_id"), table_name="raw_query_records")
    op.drop_table("raw_query_records")

    op.drop_index(op.f("ix_telemetry_events_created_at"), table_name="telemetry_events")
    op.drop_index(op.f("ix_telemetry_events_anonymized_user_key"), table_name="telemetry_events")
    op.drop_index(op.f("ix_telemetry_events_event_type"), table_name="telemetry_events")
    op.drop_index(op.f("ix_telemetry_events_category"), table_name="telemetry_events")
    op.drop_index(op.f("ix_telemetry_events_route"), table_name="telemetry_events")
    op.drop_index(op.f("ix_telemetry_events_session_id"), table_name="telemetry_events")
    op.drop_index(op.f("ix_telemetry_events_user_id"), table_name="telemetry_events")
    op.drop_index(op.f("ix_telemetry_events_union_id"), table_name="telemetry_events")
    op.drop_table("telemetry_events")

    op.drop_index(op.f("ix_user_tracking_preferences_user_id"), table_name="user_tracking_preferences")
    op.drop_index(op.f("ix_user_tracking_preferences_union_id"), table_name="user_tracking_preferences")
    op.drop_table("user_tracking_preferences")

    op.drop_index(op.f("ix_tracking_policies_union_id"), table_name="tracking_policies")
    op.drop_table("tracking_policies")

    bind = op.get_bind()
    postgresql.ENUM(name="trackingpreference").drop(bind, checkfirst=True)
    postgresql.ENUM(name="rawquerystoragemode").drop(bind, checkfirst=True)
    postgresql.ENUM(name="membertrackingchoicemode").drop(bind, checkfirst=True)
    postgresql.ENUM(name="trackingprivacymode").drop(bind, checkfirst=True)
    postgresql.ENUM(name="trackingmode").drop(bind, checkfirst=True)
