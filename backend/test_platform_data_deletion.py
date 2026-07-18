"""Personal-data deletion coverage for the privacy/governance merge gate.

Verifies that purging a user removes *all* personal data, including telemetry events,
raw-query records (by direct ``user_id`` and by reconstructed ``anonymized_user_key``), and
tracking preferences — for both the admin purge path and the member self-service path.
"""

import base64
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.platform.auth import HeaderAuthAdapter
from backend.platform.chat_history import ChatHistoryStore
from backend.platform.crypto import SecretCipher
from backend.platform.db import Base
from backend.platform.guardrails import GuardrailService
from backend.platform.ingestion import IngestionService
from backend.platform.local_auth import LocalAuthService
from backend.platform.middleware import (
    PlatformContextMiddleware,
    QueryGovernanceMiddleware,
    SecurityHeadersMiddleware,
)
from backend.platform.models import (
    AuditEvent,
    AuthSession,
    Chat,
    Message,
    Notification,
    NotificationStatus,
    RawQueryRecord,
    Role,
    SessionType,
    TelemetryEvent,
    TrackingPreference,
    Union,
    UnionMembership,
    UsageEvent,
    User,
    UserTrackingPreference,
)
from backend.platform.parsing import LiteParseDocumentParser, ParserRegistry, PlainTextDocumentParser
from backend.platform.quotas import QuotaService
from backend.platform.retrieval import TenantRetrievalService
from backend.platform.routers import admin as admin_router
from backend.platform.routers import auth as auth_router
from backend.platform.routers import member as member_router
from backend.platform.routers import ops as ops_router
from backend.platform.routers import telemetry as telemetry_router
from backend.platform.sentinel import SentinelService
from backend.platform.service_container import ServiceContainer
from backend.platform.settings import PlatformSettings
from backend.platform.storage import LocalDiskStorage


def _settings(tmp_path: Path) -> PlatformSettings:
    return PlatformSettings(
        project_root=tmp_path,
        postgres_url="sqlite+pysqlite:///:memory:",
        auto_create_tables=False,
        apply_rls_policies=False,
        allowed_origins=["*"],
        request_rate_limit_per_minute=60,
        login_rate_limit_per_minute=10,
        query_token_limit=4000,
        hard_cap_default_requests_per_day=500,
        hard_cap_default_tokens_per_day=250000,
        hard_cap_default_cost_usd_per_day=25.0,
        local_storage_root=tmp_path / "storage",
        storage_backend="local",
        document_parser_backend="auto",
        liteparse_executable="",
        embedding_backend="deterministic",
        embedding_dimensions=768,
        google_embedding_model="text-embedding-004",
        google_embedding_api_key="",
        inference_request_timeout_seconds=8,
        legacy_contract_pipeline_enabled=False,
        inline_parse_max_bytes=1_000_000,
        ocr_auto_retry_enabled=True,
        ocr_auto_retry_max_attempts=1,
        local_auth_token_ttl_seconds=43_200,
        secret_encryption_key=base64.urlsafe_b64encode(b"0" * 32).decode("utf-8"),
        sentinel_email_from="sentinel@example.com",
        sentinel_email_enabled=False,
        bootstrap_super_admin_emails=[],
        session_cookie_name="karl_session",
        member_session_idle_seconds=604800,
        union_admin_session_idle_seconds=259200,
        super_admin_session_idle_seconds=3600,
    )


def _build_app(tmp_path: Path):
    settings = _settings(tmp_path)
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    Base.metadata.create_all(engine)
    storage = LocalDiskStorage(settings.local_storage_root)
    retrieval = TenantRetrievalService()
    parsers = ParserRegistry([PlainTextDocumentParser(), LiteParseDocumentParser(settings.liteparse_executable)])
    sentinel = SentinelService(settings)
    local_auth = LocalAuthService(secret_key=settings.secret_encryption_key, token_ttl_seconds=settings.local_auth_token_ttl_seconds)
    container = ServiceContainer(
        settings=settings,
        engine=engine,
        session_factory=SessionLocal,
        auth_adapter=HeaderAuthAdapter(settings, local_auth=local_auth),
        guardrails=GuardrailService(token_limit=settings.query_token_limit),
        quotas=QuotaService(settings),
        sentinel=sentinel,
        secret_cipher=SecretCipher(settings.secret_encryption_key),
        storage=storage,
        chat_history=ChatHistoryStore(SessionLocal),
        retrieval=retrieval,
        document_parsers=parsers,
        ingestion=IngestionService(
            storage=storage,
            retrieval=retrieval,
            parsers=parsers,
            sentinel=sentinel,
            inline_parse_max_bytes=settings.inline_parse_max_bytes,
        ),
        local_auth=local_auth,
    )

    app = FastAPI()
    app.state.platform = container
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(PlatformContextMiddleware)
    app.add_middleware(QueryGovernanceMiddleware)
    app.include_router(auth_router.router)
    app.include_router(admin_router.router)
    app.include_router(member_router.router)
    app.include_router(ops_router.router)
    app.include_router(telemetry_router.router)
    return app, SessionLocal, container


def _seed_member_with_telemetry(SessionLocal, container, *, union_slug="local-1"):
    """Create a member with identified + anonymized telemetry, raw queries, and a preference.

    Returns identifiers plus the member's reconstructed anonymized keys so tests can assert
    deletion both by user_id and by anonymized_user_key.
    """
    telemetry = container.telemetry
    with SessionLocal() as db:
        union = Union(slug=union_slug, name="Local 1", union_local_id=union_slug)
        admin_user = User(email="admin@example.com", full_name="Union Admin")
        member_user = User(email="member@example.com", full_name="Member User")
        other_user = User(email="other@example.com", full_name="Other Member")
        db.add_all([union, admin_user, member_user, other_user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=admin_user.id, role=Role.UNION_ADMIN))
        db.add(UnionMembership(union_id=union.id, user_id=member_user.id, role=Role.USER))
        db.add(UnionMembership(union_id=union.id, user_id=other_user.id, role=Role.USER))
        container.local_auth.create_or_update_credential(db, user=admin_user, username="union_admin", password="demo_password")
        container.local_auth.create_or_update_credential(db, user=member_user, username="member_login", password="demo_password")

        member_session = AuthSession(
            user_id=member_user.id,
            union_id=union.id,
            session_secret_hash="hash-member-data-session",
            session_type=SessionType.MEMBER,
            expires_at=datetime.utcnow() + timedelta(days=1),
        )
        db.add(member_session)
        db.flush()

        union_id = union.id
        member_id = member_user.id
        other_id = other_user.id
        member_session_id = member_session.id

        # Reconstruct the anonymized keys the system would emit for this member.
        anon_key_session = telemetry.anonymized_user_key(user_id=member_id, union_id=union_id, session_id=member_session_id)
        anon_key_no_session = telemetry.anonymized_user_key(user_id=member_id, union_id=union_id, session_id=None)
        other_anon_key = telemetry.anonymized_user_key(user_id=other_id, union_id=union_id, session_id=None)

        # Member chat + ancillary personal data (covers the pre-existing purge behavior).
        chat = Chat(union_id=union_id, user_id=member_id, session_id="member-chat")
        db.add(chat)
        db.flush()
        db.add(Message(union_id=union_id, chat_id=chat.id, role="user", content="private question", metadata_json={}))
        db.add(UsageEvent(union_id=union_id, user_id=member_id, route="/api/query", request_count=1, token_count=10))
        db.add(Notification(union_id=union_id, user_id=member_id, channel="ui", subject="t", body="t", status=NotificationStatus.PENDING))

        # Tracking preference for the member (must be deleted) and one for another user (must survive).
        db.add(UserTrackingPreference(user_id=member_id, union_id=union_id, preference=TrackingPreference.FULL))
        db.add(UserTrackingPreference(user_id=other_id, union_id=union_id, preference=TrackingPreference.FULL))

        # Identified telemetry/raw rows (user_id present).
        db.add(TelemetryEvent(union_id=union_id, user_id=member_id, session_id=member_session_id, category="usage_ux", event_type="query_completed", anonymized_user_key=anon_key_session, metadata_json={}))
        db.add(RawQueryRecord(union_id=union_id, user_id=member_id, session_id=member_session_id, route="/api/query", anonymized_user_key=anon_key_session, question_text="q", answer_text="a", metadata_json={}))

        # Anonymized telemetry/raw rows (no user_id) — matched by session_id and by anon key.
        db.add(TelemetryEvent(union_id=union_id, user_id=None, session_id=member_session_id, category="bug_journey", event_type="page_viewed", anonymized_user_key=anon_key_session, metadata_json={}))
        db.add(TelemetryEvent(union_id=union_id, user_id=None, session_id=None, category="bug_journey", event_type="page_viewed", anonymized_user_key=anon_key_no_session, metadata_json={}))
        db.add(RawQueryRecord(union_id=union_id, user_id=None, session_id=None, route="/api/query", anonymized_user_key=anon_key_no_session, question_text="q2", answer_text="a2", metadata_json={}))

        # Survivor rows belonging to another member.
        db.add(TelemetryEvent(union_id=union_id, user_id=other_id, session_id=None, category="usage_ux", event_type="query_completed", anonymized_user_key=other_anon_key, metadata_json={}))
        db.add(RawQueryRecord(union_id=union_id, user_id=None, session_id=None, route="/api/query", anonymized_user_key=other_anon_key, question_text="other", answer_text="other", metadata_json={}))

        db.commit()

    return {
        "union_id": union_id,
        "member_id": member_id,
        "other_id": other_id,
        "anon_keys": [anon_key_session, anon_key_no_session],
        "other_anon_key": other_anon_key,
    }


def _assert_member_telemetry_gone(db, seeded):
    member_id = seeded["member_id"]
    for model in (TelemetryEvent, RawQueryRecord):
        assert db.scalar(select(model).where(model.user_id == member_id)) is None
        for key in seeded["anon_keys"]:
            assert db.scalar(select(model).where(model.anonymized_user_key == key)) is None
    assert db.scalar(select(UserTrackingPreference).where(UserTrackingPreference.user_id == member_id)) is None


def _assert_other_member_survives(db, seeded):
    other_id = seeded["other_id"]
    assert db.scalar(select(TelemetryEvent).where(TelemetryEvent.user_id == other_id)) is not None
    assert db.scalar(select(RawQueryRecord).where(RawQueryRecord.anonymized_user_key == seeded["other_anon_key"])) is not None
    assert db.scalar(select(UserTrackingPreference).where(UserTrackingPreference.user_id == other_id)) is not None


def test_admin_purge_removes_telemetry_raw_query_and_preferences(tmp_path):
    app, SessionLocal, container = _build_app(tmp_path)
    seeded = _seed_member_with_telemetry(SessionLocal, container)

    client = TestClient(app)
    login = client.post("/api/auth/local/login", json={"username": "union_admin", "password": "demo_password", "union_slug": "local-1"})
    token = login.json()["access_token"]
    purge = client.delete(
        f"/api/admin/unions/{seeded['union_id']}/users/{seeded['member_id']}?purge_user=true",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert purge.status_code == 200
    assert purge.json()["purged"] is True

    with SessionLocal() as db:
        # P1 acceptance: telemetry/raw/preferences gone by user_id AND anonymized_user_key.
        _assert_member_telemetry_gone(db, seeded)
        # Existing purge behavior unchanged.
        assert db.get(User, seeded["member_id"]) is None
        assert db.scalar(select(Chat).where(Chat.user_id == seeded["member_id"])) is None
        assert db.scalar(select(AuthSession).where(AuthSession.user_id == seeded["member_id"])) is None
        # Other members are untouched.
        _assert_other_member_survives(db, seeded)


def test_member_self_service_deletes_personal_data(tmp_path):
    app, SessionLocal, container = _build_app(tmp_path)
    seeded = _seed_member_with_telemetry(SessionLocal, container)

    client = TestClient(app)
    login = client.post("/api/auth/local/login", json={"username": "member_login", "password": "demo_password", "union_slug": "local-1"})
    token = login.json()["access_token"]

    # Confirmation is required.
    unconfirmed = client.request(
        "DELETE",
        "/api/member/me/data",
        headers={"Authorization": f"Bearer {token}"},
        json={"confirm": False},
    )
    assert unconfirmed.status_code == 400

    deleted = client.request(
        "DELETE",
        "/api/member/me/data",
        headers={"Authorization": f"Bearer {token}"},
        json={"confirm": True},
    )
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True

    with SessionLocal() as db:
        _assert_member_telemetry_gone(db, seeded)
        assert db.scalar(select(Chat).where(Chat.user_id == seeded["member_id"])) is None
        # Self-service erasure scrubs personal data but does not hard-delete the account.
        assert db.get(User, seeded["member_id"]) is not None
        assert db.scalar(select(AuditEvent).where(AuditEvent.event_type == "member_self_service_data_deleted")) is not None
        _assert_other_member_survives(db, seeded)


def test_ops_purge_sessions_endpoint_requires_super_admin_and_audits(tmp_path):
    app, SessionLocal, container = _build_app(tmp_path)
    now = datetime.utcnow()

    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        super_user = User(email="super@example.com", full_name="Super Admin")
        member_user = User(email="member@example.com", full_name="Member User")
        db.add_all([union, super_user, member_user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=super_user.id, role=Role.SUPER_ADMIN))
        db.add(UnionMembership(union_id=union.id, user_id=member_user.id, role=Role.USER))
        container.local_auth.create_or_update_credential(db, user=super_user, username="superadmin", password="demo_password")
        container.local_auth.create_or_update_credential(db, user=member_user, username="member_login", password="demo_password")
        # One long-expired session that should be purged.
        db.add(AuthSession(user_id=member_user.id, union_id=union.id, session_secret_hash="old-expired", session_type=SessionType.MEMBER, created_at=now - timedelta(days=200), last_seen_at=now - timedelta(days=200), expires_at=now - timedelta(days=120)))
        db.commit()

    client = TestClient(app)

    # Members cannot invoke the maintenance route.
    member_login = client.post("/api/auth/local/login", json={"username": "member_login", "password": "demo_password", "union_slug": "local-1"})
    member_token = member_login.json()["access_token"]
    forbidden = client.post("/api/ops/maintenance/purge-sessions", headers={"Authorization": f"Bearer {member_token}"})
    assert forbidden.status_code == 403

    super_login = client.post("/api/auth/local/login", json={"username": "superadmin", "password": "demo_password", "union_slug": "local-1"})
    super_token = super_login.json()["access_token"]
    response = client.post("/api/ops/maintenance/purge-sessions", headers={"Authorization": f"Bearer {super_token}"})
    assert response.status_code == 200
    assert response.json()["deleted_count"] >= 1

    with SessionLocal() as db:
        assert db.scalar(select(AuthSession).where(AuthSession.session_secret_hash == "old-expired")) is None
        assert db.scalar(select(AuditEvent).where(AuditEvent.event_type == "auth_sessions_purged")) is not None
