"""SQLAlchemy models for tenant-aware production KARL state."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Enum, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.platform.db import Base

try:
    from pgvector.sqlalchemy import Vector
except Exception:  # pragma: no cover - pgvector optional in non-production tests
    Vector = None


def _uuid() -> str:
    return str(uuid.uuid4())


def _enum_values(enum_cls: type[enum.Enum]) -> list[str]:
    return [member.value for member in enum_cls]


class Role(str, enum.Enum):
    SUPER_ADMIN = "super_admin"
    UNION_ADMIN = "union_admin"
    STEWARD_ADMIN = "steward_admin"
    USER = "user"


class SessionType(str, enum.Enum):
    MEMBER = "member"
    UNION_ADMIN = "union_admin"
    SUPER_ADMIN = "super_admin"


class DocumentStatus(str, enum.Enum):
    ACTIVE = "active"
    DELETED = "deleted"
    PROCESSING = "processing"
    FAILED = "failed"


class IngestionJobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class SecuritySeverity(str, enum.Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class NotificationStatus(str, enum.Enum):
    PENDING = "pending"
    SENT = "sent"
    ACKNOWLEDGED = "acknowledged"


class TrackingMode(str, enum.Enum):
    NONE = "none"
    BUG_AND_JOURNEY = "bug_and_journey"
    USAGE_AND_UX = "usage_and_ux"
    BOTH = "both"


class TrackingPrivacyMode(str, enum.Enum):
    ANONYMIZED = "anonymized"
    IDENTIFIED = "identified"


class MemberTrackingChoiceMode(str, enum.Enum):
    NONE = "none"
    BUG_ONLY_OR_FULL = "bug_only_or_full"
    FULL_OPT_OUT = "full_opt_out"


class RawQueryStorageMode(str, enum.Enum):
    DISABLED = "disabled"
    ENABLED_ANONYMIZED = "enabled_anonymized"
    ENABLED_IDENTIFIED = "enabled_identified"


class TrackingPreference(str, enum.Enum):
    SYSTEM_DEFAULT = "system_default"
    BUG_ONLY = "bug_only"
    FULL = "full"
    OFF = "off"


class Union(Base):
    __tablename__ = "unions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    slug: Mapped[str] = mapped_column(String(120), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    union_local_id: Mapped[str] = mapped_column(String(120), unique=True, nullable=False, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    message_retention_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    external_auth_id: Mapped[str | None] = mapped_column(String(255), unique=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class LocalAuthCredential(Base):
    __tablename__ = "local_auth_credentials"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, unique=True, index=True)
    username: Mapped[str] = mapped_column(String(120), nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    password_salt: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class AuthSession(Base):
    __tablename__ = "auth_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    union_id: Mapped[str | None] = mapped_column(ForeignKey("unions.id"), index=True)
    session_secret_hash: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
    session_type: Mapped[SessionType] = mapped_column(
        Enum(SessionType, values_callable=_enum_values),
        nullable=False,
        index=True,
    )
    ip_address: Mapped[str | None] = mapped_column(String(120))
    user_agent: Mapped[str | None] = mapped_column(String(500))
    invite_code_id: Mapped[str | None] = mapped_column(ForeignKey("invite_codes.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, index=True)


class UnionMembership(Base):
    __tablename__ = "union_memberships"
    __table_args__ = (UniqueConstraint("union_id", "user_id", name="uq_union_membership"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    union_id: Mapped[str] = mapped_column(ForeignKey("unions.id"), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    role: Mapped[Role] = mapped_column(
        Enum(Role, values_callable=_enum_values),
        nullable=False,
        index=True,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class ProviderConfig(Base):
    __tablename__ = "provider_configs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    union_id: Mapped[str] = mapped_column(ForeignKey("unions.id"), nullable=False, index=True)
    provider_name: Mapped[str] = mapped_column(String(120), nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    encrypted_api_key: Mapped[str] = mapped_column(Text, nullable=False)
    config_json: Mapped[dict] = mapped_column("config", JSON, default=dict, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class QuotaPolicy(Base):
    __tablename__ = "quota_policies"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    union_id: Mapped[str] = mapped_column(ForeignKey("unions.id"), nullable=False, unique=True, index=True)
    requests_per_day: Mapped[int] = mapped_column(Integer, nullable=False, default=500)
    tokens_per_day: Mapped[int] = mapped_column(Integer, nullable=False, default=250000)
    cost_usd_per_day: Mapped[float] = mapped_column(Float, nullable=False, default=25.0)
    per_user_requests_per_hour: Mapped[int] = mapped_column(Integer, nullable=False, default=60)
    warn_threshold_ratio: Mapped[float] = mapped_column(Float, nullable=False, default=0.8)
    is_paused: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    union_id: Mapped[str] = mapped_column(ForeignKey("unions.id"), nullable=False, index=True)
    uploaded_by_user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"))
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    # Which contract this document belongs to, e.g.
    # "local7_safeway_pueblo_meat_2022". Retrieval filters on this so a meat
    # member is never answered from the clerks book and vice versa. NULL means
    # "not scoped to a contract" and the document is only reachable by queries
    # that ask for no particular contract.
    contract_id: Mapped[str | None] = mapped_column(String(255), index=True)
    storage_key: Mapped[str] = mapped_column(String(500), nullable=False)
    # Optional companion PDF: the original printed contract this document's
    # text was extracted from. Stored, never parsed, so citations can open the
    # real page a member can check against the book in the break room.
    source_pdf_key: Mapped[str | None] = mapped_column(String(500))
    content_type: Mapped[str] = mapped_column(String(120), nullable=False)
    bytes_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus, values_callable=_enum_values),
        nullable=False,
        default=DocumentStatus.PROCESSING,
    )
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    union_id: Mapped[str] = mapped_column(ForeignKey("unions.id"), nullable=False, index=True)
    document_id: Mapped[str | None] = mapped_column(ForeignKey("documents.id"), index=True)
    requested_by_user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"))
    status: Mapped[IngestionJobStatus] = mapped_column(
        Enum(IngestionJobStatus, values_callable=_enum_values),
        nullable=False,
        default=IngestionJobStatus.PENDING,
    )
    error_message: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class ChunkEmbedding(Base):
    __tablename__ = "chunk_embeddings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    union_id: Mapped[str | None] = mapped_column(ForeignKey("unions.id"), nullable=True, index=True)
    document_id: Mapped[str | None] = mapped_column(ForeignKey("documents.id"), nullable=True, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict, nullable=False)
    if Vector is not None:
        # Must equal KARL_EMBEDDING_DIMENSIONS. 768 is a documented output
        # width for gemini-embedding-001; changing one without the other, or
        # without a migration, breaks inserts at ingest time.
        embedding: Mapped[list[float] | None] = mapped_column(Vector(768))
    else:  # pragma: no cover - used only when pgvector unavailable
        embedding: Mapped[list[float] | None] = mapped_column(JSON)


class Chat(Base):
    __tablename__ = "chats"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    union_id: Mapped[str] = mapped_column(ForeignKey("unions.id"), nullable=False, index=True)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"))
    session_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    union_id: Mapped[str] = mapped_column(ForeignKey("unions.id"), nullable=False, index=True)
    chat_id: Mapped[str] = mapped_column(ForeignKey("chats.id"), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class UsageEvent(Base):
    __tablename__ = "usage_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    union_id: Mapped[str] = mapped_column(ForeignKey("unions.id"), nullable=False, index=True)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"))
    route: Mapped[str] = mapped_column(String(255), nullable=False)
    request_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False, index=True)


class TrackingPolicy(Base):
    __tablename__ = "tracking_policies"
    __table_args__ = (UniqueConstraint("union_id", name="uq_tracking_policy_union"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    union_id: Mapped[str | None] = mapped_column(ForeignKey("unions.id"), index=True)
    tracking_mode: Mapped[TrackingMode] = mapped_column(
        Enum(TrackingMode, values_callable=_enum_values),
        nullable=False,
        default=TrackingMode.BUG_AND_JOURNEY,
    )
    privacy_mode: Mapped[TrackingPrivacyMode] = mapped_column(
        Enum(TrackingPrivacyMode, values_callable=_enum_values),
        nullable=False,
        default=TrackingPrivacyMode.ANONYMIZED,
    )
    member_choice_mode: Mapped[MemberTrackingChoiceMode] = mapped_column(
        Enum(MemberTrackingChoiceMode, values_callable=_enum_values),
        nullable=False,
        default=MemberTrackingChoiceMode.BUG_ONLY_OR_FULL,
    )
    raw_query_storage_mode: Mapped[RawQueryStorageMode] = mapped_column(
        Enum(RawQueryStorageMode, values_callable=_enum_values),
        nullable=False,
        default=RawQueryStorageMode.DISABLED,
    )
    default_member_preference: Mapped[TrackingPreference] = mapped_column(
        Enum(TrackingPreference, values_callable=_enum_values),
        nullable=False,
        default=TrackingPreference.BUG_ONLY,
    )
    allow_union_override: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class UserTrackingPreference(Base):
    __tablename__ = "user_tracking_preferences"
    __table_args__ = (UniqueConstraint("user_id", "union_id", name="uq_user_tracking_preference"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    union_id: Mapped[str | None] = mapped_column(ForeignKey("unions.id"), index=True)
    preference: Mapped[TrackingPreference] = mapped_column(
        Enum(TrackingPreference, values_callable=_enum_values),
        nullable=False,
        default=TrackingPreference.SYSTEM_DEFAULT,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class TelemetryEvent(Base):
    __tablename__ = "telemetry_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    union_id: Mapped[str | None] = mapped_column(ForeignKey("unions.id"), index=True)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), index=True)
    session_id: Mapped[str | None] = mapped_column(String(255), index=True)
    route: Mapped[str | None] = mapped_column(String(255), index=True)
    category: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    anonymized_user_key: Mapped[str | None] = mapped_column(String(128), index=True)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False, index=True)


class RawQueryRecord(Base):
    __tablename__ = "raw_query_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    union_id: Mapped[str | None] = mapped_column(ForeignKey("unions.id"), index=True)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), index=True)
    session_id: Mapped[str | None] = mapped_column(String(255), index=True)
    route: Mapped[str | None] = mapped_column(String(255), index=True)
    anonymized_user_key: Mapped[str | None] = mapped_column(String(128), index=True)
    question_text: Mapped[str | None] = mapped_column(Text)
    answer_text: Mapped[str | None] = mapped_column(Text)
    provider_name: Mapped[str | None] = mapped_column(String(120))
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False, index=True)


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    union_id: Mapped[str | None] = mapped_column(ForeignKey("unions.id"), index=True)
    actor_user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"))
    event_type: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    event_payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False, index=True)


class SecurityEvent(Base):
    __tablename__ = "security_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    union_id: Mapped[str | None] = mapped_column(ForeignKey("unions.id"), index=True)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"))
    event_type: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    severity: Mapped[SecuritySeverity] = mapped_column(
        Enum(SecuritySeverity, values_callable=_enum_values),
        nullable=False,
        default=SecuritySeverity.INFO,
    )
    response_action: Mapped[str | None] = mapped_column(String(120))
    details_json: Mapped[dict] = mapped_column("details", JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False, index=True)


class Notification(Base):
    __tablename__ = "notifications"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    union_id: Mapped[str | None] = mapped_column(ForeignKey("unions.id"), index=True)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"))
    channel: Mapped[str] = mapped_column(String(32), nullable=False)
    subject: Mapped[str] = mapped_column(String(255), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[NotificationStatus] = mapped_column(
        Enum(NotificationStatus, values_callable=_enum_values),
        nullable=False,
        default=NotificationStatus.PENDING,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class InviteCode(Base):
    __tablename__ = "invite_codes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    union_id: Mapped[str] = mapped_column(ForeignKey("unions.id"), nullable=False, index=True)
    code: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    label: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    contract_id: Mapped[str | None] = mapped_column(String(255))
    created_by: Mapped[str | None] = mapped_column(ForeignKey("users.id"))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, index=True)
    max_uses: Mapped[int | None] = mapped_column(Integer)
    use_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, index=True)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
