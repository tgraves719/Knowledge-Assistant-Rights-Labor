import base64
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.platform.auth import HeaderAuthAdapter
from backend.platform.chat_history import ChatHistoryStore
from backend.platform.crypto import SecretCipher
from backend.platform.db import Base
from backend.platform.guardrails import GuardrailService
from backend.platform.ingestion import IngestionService
from backend.platform.local_auth import LocalAuthService
from backend.platform.middleware import PlatformContextMiddleware, QueryGovernanceMiddleware, SecurityHeadersMiddleware
from backend.platform.models import AuditEvent, AuthSession, Chat, ChunkEmbedding, Document, DocumentStatus, IngestionJob, IngestionJobStatus, LocalAuthCredential, Message, Notification, NotificationStatus, RawQueryStorageMode, Role, SecurityEvent, SecuritySeverity, SessionType, TelemetryEvent, TrackingMode, Union, UnionMembership, UsageEvent, User
from backend.platform.parsing import LiteParseDocumentParser, ParserRegistry, PlainTextDocumentParser
from backend.platform.quotas import QuotaService
from backend.platform.retrieval import TenantRetrievalService
from backend.platform.routers import auth as auth_router
from backend.platform.routers import admin as admin_router
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
    app.include_router(ops_router.router)
    app.include_router(telemetry_router.router)
    return app, SessionLocal


def test_admin_route_rejects_forged_union_admin_header(tmp_path):
    app, SessionLocal = _build_app(tmp_path)

    with SessionLocal() as db:
        db.add_all(
            [
                Union(slug="local-1", name="Local 1", union_local_id="local-1"),
                User(email="user@example.com", full_name="Regular User"),
            ]
        )
        db.commit()

    client = TestClient(app)
    response = client.get(
        "/api/admin/unions",
        headers={
            "X-Auth-Email": "user@example.com",
            "X-Auth-Role": "union_admin",
            "X-Union-Slug": "local-1",
        },
    )

    assert response.status_code == 403


def test_local_auth_login_can_access_admin_routes_with_bearer_token(tmp_path):
    app, SessionLocal = _build_app(tmp_path)

    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        user = User(email="union_demo@example.com", full_name="Union Demo")
        db.add_all([union, user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=user.id, role=Role.UNION_ADMIN))
        app.state.platform.local_auth.create_or_update_credential(
            db,
            user=user,
            username="union_demo",
            password="demo_password",
        )
        db.commit()

    client = TestClient(app)
    login = client.post(
        "/api/auth/local/login",
        json={"username": "union_demo", "password": "demo_password", "union_slug": "local-1"},
    )

    assert login.status_code == 200
    token = login.json()["access_token"]

    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    unions = client.get("/api/admin/unions", headers={"Authorization": f"Bearer {token}"})

    assert me.status_code == 200
    assert me.json()["source"] == "local_token"
    assert me.json()["role"] == Role.UNION_ADMIN.value
    assert unions.status_code == 200
    assert len(unions.json()["items"]) == 1


def test_super_admin_can_create_union_user_with_local_auth_credentials(tmp_path):
    app, SessionLocal = _build_app(tmp_path)

    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        super_user = User(email="super@example.com", full_name="Super Admin")
        db.add_all([union, super_user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=super_user.id, role=Role.SUPER_ADMIN))
        app.state.platform.local_auth.create_or_update_credential(
            db,
            user=super_user,
            username="superadmin",
            password="demo_password",
        )
        db.commit()
        union_id = union.id

    client = TestClient(app)
    login = client.post(
        "/api/auth/local/login",
        json={"username": "superadmin", "password": "demo_password", "union_slug": "local-1"},
    )
    assert login.status_code == 200
    token = login.json()["access_token"]

    create = client.post(
        f"/api/admin/unions/{union_id}/users",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "email": "member@example.com",
            "full_name": "Member User",
            "role": "user",
            "username": "member_login",
            "password": "demo_password",
        },
    )

    assert create.status_code == 200

    users = client.get(f"/api/admin/unions/{union_id}/users", headers={"Authorization": f"Bearer {token}"})
    assert users.status_code == 200
    payload = users.json()["items"]
    assert any(item["username"] == "member_login" and item["has_local_auth"] is True for item in payload)

    with SessionLocal() as db:
        created_user = db.scalar(select(User).where(User.email == "member@example.com"))
        credential = db.scalar(select(LocalAuthCredential).where(LocalAuthCredential.user_id == created_user.id))
        assert credential is not None
        assert credential.username == "member_login"


def test_super_admin_create_union_generates_internal_id_and_unique_slug(tmp_path):
    app, SessionLocal = _build_app(tmp_path)

    with SessionLocal() as db:
        existing_union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        super_user = User(email="super@example.com", full_name="Super Admin")
        db.add_all([existing_union, super_user])
        db.flush()
        db.add(UnionMembership(union_id=existing_union.id, user_id=super_user.id, role=Role.SUPER_ADMIN))
        app.state.platform.local_auth.create_or_update_credential(db, user=super_user, username="superadmin", password="demo_password")
        db.commit()

    client = TestClient(app)
    login = client.post("/api/auth/local/login", json={"username": "superadmin", "password": "demo_password", "union_slug": "local-1"})
    token = login.json()["access_token"]

    create = client.post(
        "/api/admin/unions",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": "Local 1", "slug": "local-1"},
    )

    assert create.status_code == 200
    created_slug = create.json()["slug"]
    assert created_slug != "local-1"

    with SessionLocal() as db:
        created = db.scalar(select(Union).where(Union.slug == created_slug))
        assert created is not None
        assert created.union_local_id.startswith("union-")
        assert created.union_local_id != "local-1"


def test_super_admin_security_events_support_union_scope_and_paging(tmp_path):
    app, SessionLocal = _build_app(tmp_path)

    with SessionLocal() as db:
        union_one = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        union_two = Union(slug="local-2", name="Local 2", union_local_id="local-2")
        super_user = User(email="super@example.com", full_name="Super Admin")
        db.add_all([union_one, union_two, super_user])
        db.flush()
        db.add(UnionMembership(union_id=union_one.id, user_id=super_user.id, role=Role.SUPER_ADMIN))
        app.state.platform.local_auth.create_or_update_credential(db, user=super_user, username="superadmin", password="demo_password")
        for index in range(7):
            db.add(SecurityEvent(union_id=union_one.id, event_type=f"union-one-{index}", severity=SecuritySeverity.WARNING))
        for index in range(3):
            db.add(SecurityEvent(union_id=union_two.id, event_type=f"union-two-{index}", severity=SecuritySeverity.INFO))
        db.commit()
        union_one_id = union_one.id

    client = TestClient(app)
    login = client.post("/api/auth/local/login", json={"username": "superadmin", "password": "demo_password", "union_slug": "local-1"})
    token = login.json()["access_token"]

    response = client.get(
        f"/api/ops/security-events?union_id={union_one_id}&page=2&page_size=5",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 7
    assert payload["page"] == 2
    assert payload["page_size"] == 5
    assert len(payload["items"]) == 2
    assert all(item["union_id"] == union_one_id for item in payload["items"])


def test_super_admin_notifications_support_union_scope_and_paging(tmp_path):
    app, SessionLocal = _build_app(tmp_path)

    with SessionLocal() as db:
        union_one = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        union_two = Union(slug="local-2", name="Local 2", union_local_id="local-2")
        super_user = User(email="super@example.com", full_name="Super Admin")
        db.add_all([union_one, union_two, super_user])
        db.flush()
        db.add(UnionMembership(union_id=union_one.id, user_id=super_user.id, role=Role.SUPER_ADMIN))
        app.state.platform.local_auth.create_or_update_credential(db, user=super_user, username="superadmin", password="demo_password")
        for index in range(6):
            db.add(Notification(union_id=union_one.id, channel="email", subject=f"Union one {index}", body="Notice", status=NotificationStatus.PENDING))
        for index in range(2):
            db.add(Notification(union_id=union_two.id, channel="email", subject=f"Union two {index}", body="Notice", status=NotificationStatus.PENDING))
        db.commit()
        union_one_id = union_one.id

    client = TestClient(app)
    login = client.post("/api/auth/local/login", json={"username": "superadmin", "password": "demo_password", "union_slug": "local-1"})
    token = login.json()["access_token"]

    response = client.get(
        f"/api/ops/notifications?union_id={union_one_id}&page=2&page_size=5",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 6
    assert payload["page"] == 2
    assert payload["page_size"] == 5
    assert len(payload["items"]) == 1
    assert all(item["union_id"] == union_one_id for item in payload["items"])


def test_union_admin_can_acknowledge_notification_for_their_union(tmp_path):
    app, SessionLocal = _build_app(tmp_path)

    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        admin_user = User(email="admin@example.com", full_name="Union Admin")
        db.add_all([union, admin_user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=admin_user.id, role=Role.UNION_ADMIN))
        app.state.platform.local_auth.create_or_update_credential(db, user=admin_user, username="union_admin", password="demo_password")
        notification = Notification(
            union_id=union.id,
            channel="email",
            subject="Union alert",
            body="Please review this issue.",
            status=NotificationStatus.PENDING,
        )
        db.add(notification)
        db.commit()
        notification_id = notification.id

    client = TestClient(app)
    login = client.post("/api/auth/local/login", json={"username": "union_admin", "password": "demo_password", "union_slug": "local-1"})
    token = login.json()["access_token"]

    response = client.post(
        f"/api/ops/notifications/{notification_id}/acknowledge",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json()["notification"]["status"] == NotificationStatus.ACKNOWLEDGED.value

    with SessionLocal() as db:
        notification = db.get(Notification, notification_id)
        assert notification.status == NotificationStatus.ACKNOWLEDGED

    list_response = client.get(
        "/api/ops/notifications",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert list_response.status_code == 200
    assert all(item["id"] != notification_id for item in list_response.json()["items"])


def test_super_admin_can_resolve_prompt_injection_safety_review_and_release_document(tmp_path):
    app, SessionLocal = _build_app(tmp_path)

    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        super_user = User(email="super@example.com", full_name="Super Admin")
        db.add_all([union, super_user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=super_user.id, role=Role.SUPER_ADMIN))
        app.state.platform.local_auth.create_or_update_credential(db, user=super_user, username="superadmin", password="demo_password")
        document = Document(
            union_id=union.id,
            uploaded_by_user_id=super_user.id,
            title="unsafe.txt",
            storage_key="local-1/unsafe.txt",
            content_type="text/plain",
            bytes_size=100,
            status=DocumentStatus.ACTIVE,
            metadata_json={
                "review_status": "escalated",
                "prompt_injection_risk": True,
                "member_visible": False,
                "ready_for_query": False,
                "safety_review_status": "blocked_pending_superadmin",
            },
        )
        db.add(document)
        db.flush()
        job = IngestionJob(
            union_id=union.id,
            document_id=document.id,
            requested_by_user_id=super_user.id,
            status=IngestionJobStatus.SUCCEEDED,
            metadata_json={"review_status": "escalated", "safety_review_status": "blocked_pending_superadmin"},
        )
        db.add(job)
        db.commit()
        union_id = union.id
        job_id = job.id

    client = TestClient(app)
    login = client.post("/api/auth/local/login", json={"username": "superadmin", "password": "demo_password", "union_slug": "local-1"})
    token = login.json()["access_token"]

    response = client.post(
        f"/api/admin/unions/{union_id}/ingestion-jobs/{job_id}/review-state",
        headers={"Authorization": f"Bearer {token}"},
        json={"review_status": "resolved", "note": "Reviewed and released."},
    )

    assert response.status_code == 200

    with SessionLocal() as db:
        document = db.get(Document, response.json()["job"]["document_id"])
        assert document.metadata_json["member_visible"] is True
        assert document.metadata_json["ready_for_query"] is True
        assert document.metadata_json["safety_review_status"] == "resolved"
        assert document.metadata_json["prompt_injection_risk"] is False


def test_union_admin_user_directory_hides_superadmins_and_supports_search(tmp_path):
    app, SessionLocal = _build_app(tmp_path)

    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        admin_user = User(email="admin@example.com", full_name="Union Admin")
        super_user = User(email="super@example.com", full_name="Super Admin")
        member_user = User(email="member@example.com", full_name="Vacation Member")
        db.add_all([union, admin_user, super_user, member_user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=admin_user.id, role=Role.UNION_ADMIN))
        db.add(UnionMembership(union_id=union.id, user_id=super_user.id, role=Role.SUPER_ADMIN))
        db.add(UnionMembership(union_id=union.id, user_id=member_user.id, role=Role.USER))
        app.state.platform.local_auth.create_or_update_credential(db, user=admin_user, username="union_admin", password="demo_password")
        app.state.platform.local_auth.create_or_update_credential(db, user=member_user, username="vacation_member", password="demo_password")
        db.commit()
        union_id = union.id

    client = TestClient(app)
    login = client.post("/api/auth/local/login", json={"username": "union_admin", "password": "demo_password", "union_slug": "local-1"})
    token = login.json()["access_token"]

    response = client.get(
        f"/api/admin/unions/{union_id}/users?q=vacation&page=1&page_size=10",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["union_total"] == 2
    assert payload["items"][0]["email"] == "member@example.com"
    assert all(item["role"] != Role.SUPER_ADMIN.value for item in payload["items"])


def test_union_admin_can_review_sensitive_document_and_restore_full_member_access(tmp_path):
    app, SessionLocal = _build_app(tmp_path)

    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        admin_user = User(email="admin@example.com", full_name="Union Admin")
        db.add_all([union, admin_user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=admin_user.id, role=Role.UNION_ADMIN))
        app.state.platform.local_auth.create_or_update_credential(db, user=admin_user, username="union_admin", password="demo_password")
        document = Document(
            union_id=union.id,
            uploaded_by_user_id=admin_user.id,
            title="members.txt",
            storage_key="local-1/members.txt",
            content_type="text/plain",
            bytes_size=100,
            status=DocumentStatus.ACTIVE,
            metadata_json={
                "review_status": "needs_review",
                "sensitive_data_risk": True,
                "member_visible": True,
                "ready_for_query": True,
                "safety_status": "flagged_sensitive_data",
                "safety_review_status": "needs_review",
                "safety_reasons": ["sensitive_ssn"],
            },
        )
        db.add(document)
        db.flush()
        db.add(
            ChunkEmbedding(
                union_id=union.id,
                document_id=document.id,
                chunk_index=0,
                chunk_text="Member SSN 123-45-6789",
                metadata_json={"sensitive_data_risk": True, "member_visible": True, "safety_status": "flagged_sensitive_data"},
            )
        )
        job = IngestionJob(
            union_id=union.id,
            document_id=document.id,
            requested_by_user_id=admin_user.id,
            status=IngestionJobStatus.SUCCEEDED,
            metadata_json={"review_status": "needs_review", "safety_review_status": "needs_review", "artifact_key": f"{union.slug}/{document.id}/parse/job.json"},
        )
        db.add(job)
        db.commit()
        union_id = union.id
        document_id = document.id

    artifact_path = app.state.platform.storage.open(f"local-1/{document_id}/parse/job.json")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        '{"text":"Member SSN 123-45-6789 can be used for payroll verification.","pages":[{"page_number":1,"text":"Member SSN 123-45-6789 can be used for payroll verification."}]}',
        encoding="utf-8",
    )

    client = TestClient(app)
    login = client.post("/api/auth/local/login", json={"username": "union_admin", "password": "demo_password", "union_slug": "local-1"})
    token = login.json()["access_token"]

    detail_response = client.get(
        f"/api/admin/unions/{union_id}/documents/{document_id}/review-detail",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert detail_response.status_code == 200
    detail_payload = detail_response.json()
    assert detail_payload["review_actions"]["can_approve_member_access"] is True
    assert detail_payload["review_preview"]["safety_findings"]
    assert "123-45-6789" in detail_payload["review_preview"]["text_excerpt"]

    approve_response = client.post(
        f"/api/admin/unions/{union_id}/documents/{document_id}/safety-review",
        headers={"Authorization": f"Bearer {token}"},
        json={"decision": "approve_member_access", "note": "Approved after union review."},
    )
    assert approve_response.status_code == 200

    with SessionLocal() as db:
        document = db.get(Document, document_id)
        chunk = db.scalar(select(ChunkEmbedding).where(ChunkEmbedding.document_id == document_id))
        assert document.metadata_json["sensitive_data_risk"] is False
        assert document.metadata_json["safety_review_status"] == "resolved"
        assert document.metadata_json["safety_status"] == "reviewed_safe"
        assert chunk.metadata_json["sensitive_data_risk"] is False
        assert chunk.metadata_json["safety_status"] == "reviewed_safe"


def test_union_admin_can_open_uploaded_document_file(tmp_path):
    app, SessionLocal = _build_app(tmp_path)

    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        admin_user = User(email="admin@example.com", full_name="Union Admin")
        db.add_all([union, admin_user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=admin_user.id, role=Role.UNION_ADMIN))
        app.state.platform.local_auth.create_or_update_credential(db, user=admin_user, username="union_admin", password="demo_password")
        stored = app.state.platform.storage.save_bytes(union.slug, "baddata.txt", b"Sensitive member test upload.")
        document = Document(
            union_id=union.id,
            uploaded_by_user_id=admin_user.id,
            title="baddata.txt",
            storage_key=stored.key,
            content_type="text/plain",
            bytes_size=stored.bytes_size,
            status=DocumentStatus.ACTIVE,
            metadata_json={"ready_for_query": True, "review_status": "not_required"},
        )
        db.add(document)
        db.commit()
        union_id = union.id
        document_id = document.id

    client = TestClient(app)
    login = client.post("/api/auth/local/login", json={"username": "union_admin", "password": "demo_password", "union_slug": "local-1"})
    token = login.json()["access_token"]

    response = client.get(
        f"/api/admin/unions/{union_id}/documents/{document_id}/content",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.text == "Sensitive member test upload."
    assert "inline" in response.headers["content-disposition"].lower()


def test_super_admin_can_manage_global_and_union_tracking_policy(tmp_path):
    app, SessionLocal = _build_app(tmp_path)

    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        super_user = User(email="super@example.com", full_name="Super Admin")
        db.add_all([union, super_user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=super_user.id, role=Role.SUPER_ADMIN))
        app.state.platform.local_auth.create_or_update_credential(db, user=super_user, username="superadmin", password="demo_password")
        db.commit()
        union_id = union.id

    client = TestClient(app)
    login = client.post("/api/auth/local/login", json={"username": "superadmin", "password": "demo_password", "union_slug": "local-1"})
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    global_get = client.get("/api/admin/tracking-policy/global", headers=headers)
    assert global_get.status_code == 200
    assert global_get.json()["policy"]["tracking_mode"] == "bug_and_journey"

    global_put = client.put(
        "/api/admin/tracking-policy/global",
        headers=headers,
        json={
            "tracking_mode": "both",
            "privacy_mode": "anonymized",
            "member_choice_mode": "bug_only_or_full",
            "raw_query_storage_mode": "disabled",
            "default_member_preference": "bug_only",
            "allow_union_override": True,
        },
    )
    assert global_put.status_code == 200
    assert global_put.json()["policy"]["tracking_mode"] == "both"

    union_put = client.put(
        f"/api/admin/unions/{union_id}/tracking-policy",
        headers=headers,
        json={
            "tracking_mode": "usage_and_ux",
            "privacy_mode": "identified",
            "member_choice_mode": "none",
            "raw_query_storage_mode": "enabled_identified",
            "default_member_preference": "full",
        },
    )
    assert union_put.status_code == 200
    assert union_put.json()["override_enabled"] is True
    assert union_put.json()["effective_policy"]["tracking_mode"] == "usage_and_ux"

    union_delete = client.delete(f"/api/admin/unions/{union_id}/tracking-policy", headers=headers)
    assert union_delete.status_code == 200
    assert union_delete.json()["override_enabled"] is False


def test_enabling_identified_raw_query_emits_single_security_event(tmp_path):
    app, SessionLocal = _build_app(tmp_path)

    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        super_user = User(email="super@example.com", full_name="Super Admin")
        db.add_all([union, super_user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=super_user.id, role=Role.SUPER_ADMIN))
        app.state.platform.local_auth.create_or_update_credential(db, user=super_user, username="superadmin", password="demo_password")
        db.commit()

    client = TestClient(app)
    login = client.post("/api/auth/local/login", json={"username": "superadmin", "password": "demo_password", "union_slug": "local-1"})
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    identified_payload = {
        "tracking_mode": "both",
        "privacy_mode": "identified",
        "member_choice_mode": "bug_only_or_full",
        "raw_query_storage_mode": "enabled_identified",
        "default_member_preference": "bug_only",
        "allow_union_override": True,
    }

    # First enable → exactly one warning-severity SecurityEvent.
    assert client.put("/api/admin/tracking-policy/global", headers=headers, json=identified_payload).status_code == 200
    with SessionLocal() as db:
        events = db.scalars(select(SecurityEvent).where(SecurityEvent.event_type == "raw_query_identified_enabled")).all()
        assert len(events) == 1
        assert events[0].severity == SecuritySeverity.WARNING

    # Re-saving while already identified → no additional event.
    assert client.put("/api/admin/tracking-policy/global", headers=headers, json=identified_payload).status_code == 200
    with SessionLocal() as db:
        events = db.scalars(select(SecurityEvent).where(SecurityEvent.event_type == "raw_query_identified_enabled")).all()
        assert len(events) == 1

    # Turning it back off then on again → a second event on the new enable transition.
    disabled_payload = {**identified_payload, "raw_query_storage_mode": "disabled"}
    assert client.put("/api/admin/tracking-policy/global", headers=headers, json=disabled_payload).status_code == 200
    assert client.put("/api/admin/tracking-policy/global", headers=headers, json=identified_payload).status_code == 200
    with SessionLocal() as db:
        events = db.scalars(select(SecurityEvent).where(SecurityEvent.event_type == "raw_query_identified_enabled")).all()
        assert len(events) == 2


def test_member_telemetry_endpoint_honors_preference_and_policy(tmp_path):
    app, SessionLocal = _build_app(tmp_path)

    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        member_user = User(email="member@example.com", full_name="Member User")
        db.add_all([union, member_user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=member_user.id, role=Role.USER))
        app.state.platform.local_auth.create_or_update_credential(db, user=member_user, username="member_login", password="demo_password")
        policy = app.state.platform.telemetry.get_or_create_global_policy(db)
        policy.tracking_mode = TrackingMode.BOTH
        policy.raw_query_storage_mode = RawQueryStorageMode.DISABLED
        db.commit()

    client = TestClient(app)
    login = client.post(
        "/api/auth/local/login",
        headers={"X-Tenant-Slug": "local-1"},
        json={"username": "member_login", "password": "demo_password", "union_slug": "local-1"},
    )
    assert login.status_code == 200
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}", "X-Tenant-Slug": "local-1"}

    update_pref = client.put("/api/auth/session/preferences", json={"tracking_preference": "bug_only"}, headers=headers)
    assert update_pref.status_code == 200

    usage_event = client.post(
        "/api/telemetry/event",
        headers=headers,
        json={"category": "usage_ux", "event_type": "source_opened", "surface": "member", "session_id": "session-a"},
    )
    journey_event = client.post(
        "/api/telemetry/event",
        headers=headers,
        json={"category": "bug_journey", "event_type": "query_failed", "surface": "member", "session_id": "session-a"},
    )

    assert usage_event.status_code == 200
    assert journey_event.status_code == 200

    with SessionLocal() as db:
        stored = db.scalars(select(TelemetryEvent).order_by(TelemetryEvent.created_at.asc())).all()
        journey_types = [item.event_type for item in stored]
        assert "query_failed" in journey_types
        assert "source_opened" not in journey_types


def test_union_admin_cannot_promote_user_above_own_role(tmp_path):
    app, SessionLocal = _build_app(tmp_path)

    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        admin_user = User(email="admin@example.com", full_name="Union Admin")
        member_user = User(email="member@example.com", full_name="Member User")
        db.add_all([union, admin_user, member_user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=admin_user.id, role=Role.UNION_ADMIN))
        db.add(UnionMembership(union_id=union.id, user_id=member_user.id, role=Role.USER))
        app.state.platform.local_auth.create_or_update_credential(db, user=admin_user, username="union_admin", password="demo_password")
        db.commit()
        union_id = union.id
        member_id = member_user.id

    client = TestClient(app)
    login = client.post("/api/auth/local/login", json={"username": "union_admin", "password": "demo_password", "union_slug": "local-1"})
    token = login.json()["access_token"]

    response = client.put(
        f"/api/admin/unions/{union_id}/users/{member_id}",
        headers={"Authorization": f"Bearer {token}"},
        json={"role": "super_admin"},
    )

    assert response.status_code == 403


def test_union_admin_can_update_reset_password_and_remove_user(tmp_path):
    app, SessionLocal = _build_app(tmp_path)

    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        admin_user = User(email="admin@example.com", full_name="Union Admin")
        member_user = User(email="member@example.com", full_name="Member User")
        db.add_all([union, admin_user, member_user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=admin_user.id, role=Role.UNION_ADMIN))
        db.add(UnionMembership(union_id=union.id, user_id=member_user.id, role=Role.USER))
        app.state.platform.local_auth.create_or_update_credential(db, user=admin_user, username="union_admin", password="demo_password")
        app.state.platform.local_auth.create_or_update_credential(db, user=member_user, username="member_login", password="demo_password")
        db.commit()
        union_id = union.id
        member_id = member_user.id

    client = TestClient(app)
    login = client.post("/api/auth/local/login", json={"username": "union_admin", "password": "demo_password", "union_slug": "local-1"})
    token = login.json()["access_token"]

    update = client.put(
        f"/api/admin/unions/{union_id}/users/{member_id}",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "full_name": "Updated Member",
            "email": "updated_member@example.com",
            "role": "steward_admin",
            "username": "updated_login",
            "is_active": True,
            "local_auth_enabled": True,
        },
    )
    assert update.status_code == 200
    assert update.json()["user"]["role"] == "steward_admin"
    assert update.json()["user"]["username"] == "updated_login"

    reset = client.post(
        f"/api/admin/unions/{union_id}/users/{member_id}/password",
        headers={"Authorization": f"Bearer {token}"},
        json={"password": "new_password"},
    )
    assert reset.status_code == 200

    relogin = client.post("/api/auth/local/login", json={"username": "updated_login", "password": "new_password", "union_slug": "local-1"})
    assert relogin.status_code == 200

    remove = client.delete(
        f"/api/admin/unions/{union_id}/users/{member_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert remove.status_code == 200

    users = client.get(f"/api/admin/unions/{union_id}/users", headers={"Authorization": f"Bearer {token}"})
    assert users.status_code == 200
    assert all(item["user_id"] != member_id for item in users.json()["items"])


def test_union_admin_can_purge_single_union_user_data(tmp_path):
    app, SessionLocal = _build_app(tmp_path)

    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        admin_user = User(email="admin@example.com", full_name="Union Admin")
        member_user = User(email="member@example.com", full_name="Member User")
        db.add_all([union, admin_user, member_user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=admin_user.id, role=Role.UNION_ADMIN))
        db.add(UnionMembership(union_id=union.id, user_id=member_user.id, role=Role.USER))
        app.state.platform.local_auth.create_or_update_credential(db, user=admin_user, username="union_admin", password="demo_password")
        app.state.platform.local_auth.create_or_update_credential(db, user=member_user, username="member_login", password="demo_password")
        chat = Chat(union_id=union.id, user_id=member_user.id, session_id="member-session")
        db.add(chat)
        db.flush()
        db.add(Message(union_id=union.id, chat_id=chat.id, role="user", content="private question", metadata_json={}))
        db.add(UsageEvent(union_id=union.id, user_id=member_user.id, route="/api/query", request_count=1, token_count=10))
        db.add(Notification(union_id=union.id, user_id=member_user.id, channel="ui", subject="Test", body="Test", status=NotificationStatus.PENDING))
        db.add(AuthSession(user_id=member_user.id, union_id=union.id, session_secret_hash="hash-purge-user", session_type=SessionType.MEMBER, expires_at=datetime.utcnow() + timedelta(days=1)))
        db.commit()
        union_id = union.id
        member_id = member_user.id

    client = TestClient(app)
    login = client.post("/api/auth/local/login", json={"username": "union_admin", "password": "demo_password", "union_slug": "local-1"})
    token = login.json()["access_token"]
    purge = client.delete(
        f"/api/admin/unions/{union_id}/users/{member_id}?purge_user=true",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert purge.status_code == 200
    assert purge.json()["purged"] is True

    with SessionLocal() as db:
        assert db.get(User, member_id) is None
        assert db.scalar(select(LocalAuthCredential).where(LocalAuthCredential.user_id == member_id)) is None
        assert db.scalar(select(UnionMembership).where(UnionMembership.user_id == member_id)) is None
        assert db.scalar(select(Chat).where(Chat.user_id == member_id)) is None
        assert db.scalar(select(AuthSession).where(AuthSession.user_id == member_id)) is None


def test_super_admin_can_delete_union_and_tenant_scoped_data(tmp_path):
    app, SessionLocal = _build_app(tmp_path)

    with SessionLocal() as db:
        union = Union(slug="local-delete", name="Delete Local", union_local_id="local-delete")
        super_user = User(email="super@example.com", full_name="Super Admin")
        member_user = User(email="member@example.com", full_name="Member User")
        db.add_all([union, super_user, member_user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=super_user.id, role=Role.SUPER_ADMIN))
        db.add(UnionMembership(union_id=union.id, user_id=member_user.id, role=Role.USER))
        db.add(
            Document(
                union_id=union.id,
                uploaded_by_user_id=member_user.id,
                title="notice.txt",
                storage_key="local-delete/notice.txt",
                content_type="text/plain",
                bytes_size=12,
                status=DocumentStatus.ACTIVE,
                metadata_json={},
            )
        )
        app.state.platform.local_auth.create_or_update_credential(db, user=super_user, username="superadmin", password="demo_password")
        db.commit()
        union_id = union.id

    client = TestClient(app)
    login = client.post("/api/auth/local/login", json={"username": "superadmin", "password": "demo_password", "union_slug": "local-delete"})
    token = login.json()["access_token"]

    response = client.delete(f"/api/admin/unions/{union_id}", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    assert response.json()["deleted_union_id"] == union_id

    with SessionLocal() as db:
        assert db.get(Union, union_id) is None
        assert db.scalar(select(UnionMembership).where(UnionMembership.union_id == union_id)) is None
        assert db.scalar(select(Document).where(Document.union_id == union_id)) is None
        audit = db.scalar(select(AuditEvent).where(AuditEvent.event_type == "union_deleted"))
        assert audit is not None


def test_super_admin_can_take_all_unions_offline(tmp_path):
    app, SessionLocal = _build_app(tmp_path)

    with SessionLocal() as db:
        union_one = Union(slug="local-1", name="Local 1", union_local_id="local-1", is_active=True)
        union_two = Union(slug="local-2", name="Local 2", union_local_id="local-2", is_active=True)
        super_user = User(email="super@example.com", full_name="Super Admin")
        db.add_all([union_one, union_two, super_user])
        db.flush()
        db.add(UnionMembership(union_id=union_one.id, user_id=super_user.id, role=Role.SUPER_ADMIN))
        db.add(UnionMembership(union_id=union_two.id, user_id=super_user.id, role=Role.SUPER_ADMIN))
        app.state.platform.local_auth.create_or_update_credential(db, user=super_user, username="superadmin", password="demo_password")
        db.commit()

    client = TestClient(app)
    login = client.post("/api/auth/local/login", json={"username": "superadmin", "password": "demo_password", "union_slug": "local-1"})
    token = login.json()["access_token"]
    response = client.post("/api/admin/unions/offline-all", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    assert response.json()["changed_unions"] == 2

    with SessionLocal() as db:
        assert db.scalar(select(Union).where(Union.slug == "local-1")).is_active is False
        assert db.scalar(select(Union).where(Union.slug == "local-2")).is_active is False


def test_super_admin_platform_ops_summary_includes_usage_provider_and_admins(tmp_path):
    app, SessionLocal = _build_app(tmp_path)

    with SessionLocal() as db:
        union = Union(slug="local-ops", name="Ops Local", union_local_id="local-ops")
        super_user = User(email="super@example.com", full_name="Super Admin")
        admin_user = User(email="admin@example.com", full_name="Union Admin")
        db.add_all([union, super_user, admin_user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=super_user.id, role=Role.SUPER_ADMIN))
        db.add(UnionMembership(union_id=union.id, user_id=admin_user.id, role=Role.UNION_ADMIN))
        app.state.platform.local_auth.create_or_update_credential(db, user=super_user, username="superadmin", password="demo_password")
        from backend.platform.models import ProviderConfig, QuotaPolicy, UsageEvent
        db.add(QuotaPolicy(union_id=union.id, requests_per_day=10, tokens_per_day=100, cost_usd_per_day=1.0, per_user_requests_per_hour=5, warn_threshold_ratio=0.8, is_paused=False))
        db.add(ProviderConfig(union_id=union.id, provider_name="openrouter", model_name="qwen/test", encrypted_api_key="encrypted", config_json={"base_url": "https://openrouter.ai/api/v1"}))
        db.add(UsageEvent(union_id=union.id, user_id=admin_user.id, route="/api/query", request_count=9, token_count=90, estimated_cost_usd=0.9))
        db.commit()

    client = TestClient(app)
    login = client.post("/api/auth/local/login", json={"username": "superadmin", "password": "demo_password", "union_slug": "local-ops"})
    token = login.json()["access_token"]

    response = client.get("/api/admin/platform-ops", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["warning_unions"] == 1
    assert len(payload["items"]) == 1
    item = payload["items"][0]
    assert item["provider_health"]["status"] == "configured"
    assert item["usage"]["warning_level"] == "warning"
    assert any(admin["role"] == "union_admin" for admin in item["admins"])


def test_super_admin_can_assign_self_takeover(tmp_path):
    app, SessionLocal = _build_app(tmp_path)

    with SessionLocal() as db:
        union = Union(slug="local-takeover", name="Takeover Local", union_local_id="local-takeover")
        super_user = User(email="super@example.com", full_name="Super Admin")
        db.add_all([union, super_user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=super_user.id, role=Role.SUPER_ADMIN))
        app.state.platform.local_auth.create_or_update_credential(db, user=super_user, username="superadmin", password="demo_password")
        db.commit()
        union_id = union.id
        super_user_id = super_user.id

    client = TestClient(app)
    login = client.post("/api/auth/local/login", json={"username": "superadmin", "password": "demo_password", "union_slug": "local-takeover"})
    token = login.json()["access_token"]

    response = client.post(f"/api/admin/unions/{union_id}/admin-takeover", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    with SessionLocal() as db:
        membership = db.scalar(select(UnionMembership).where(UnionMembership.union_id == union_id, UnionMembership.user_id == super_user_id, UnionMembership.role == Role.UNION_ADMIN))
        assert membership is not None


def test_super_admin_platform_summary_returns_union_totals(tmp_path):
    app, SessionLocal = _build_app(tmp_path)

    with SessionLocal() as db:
        union_one = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        union_two = Union(slug="local-2", name="Local 2", union_local_id="local-2", is_active=False)
        super_user = User(email="super@example.com", full_name="Super Admin")
        member = User(email="member@example.com", full_name="Member User")
        db.add_all([union_one, union_two, super_user, member])
        db.flush()
        db.add(UnionMembership(union_id=union_one.id, user_id=super_user.id, role=Role.SUPER_ADMIN))
        db.add(UnionMembership(union_id=union_one.id, user_id=member.id, role=Role.USER))
        db.add(
            Document(
                union_id=union_one.id,
                uploaded_by_user_id=member.id,
                title="notice.txt",
                storage_key="local-1/notice.txt",
                content_type="text/plain",
                bytes_size=12,
                status=DocumentStatus.PROCESSING,
                metadata_json={"quality_status": "needs_review", "review_status": "needs_review"},
            )
        )
        db.add(
            Notification(
                union_id=union_one.id,
                user_id=None,
                channel="in_app",
                subject="Review needed",
                body="Check the document",
            )
        )
        app.state.platform.local_auth.create_or_update_credential(
            db,
            user=super_user,
            username="superadmin",
            password="demo_password",
        )
        db.commit()

    client = TestClient(app)
    login = client.post(
        "/api/auth/local/login",
        json={"username": "superadmin", "password": "demo_password", "union_slug": "local-1"},
    )
    assert login.status_code == 200
    token = login.json()["access_token"]

    response = client.get("/api/admin/platform-summary", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["totals"]["unions"] == 2
    assert payload["totals"]["inactive_unions"] == 1
    assert payload["totals"]["pending_reviews"] == 1
    assert payload["totals"]["pending_notifications"] == 1
    assert any(item["slug"] == "local-1" and item["pending_review_count"] == 1 for item in payload["union_summaries"])


def test_union_admin_can_run_provider_diagnostics(tmp_path, monkeypatch):
    app, SessionLocal = _build_app(tmp_path)

    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        user = User(email="admin@example.com", full_name="Union Admin")
        db.add_all([union, user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=user.id, role=Role.UNION_ADMIN))
        app.state.platform.local_auth.create_or_update_credential(
            db,
            user=user,
            username="union_demo",
            password="demo_password",
        )
        db.commit()
        union_id = union.id

    async def _fake_test(config, *, timeout_seconds):
        return {
            "ok": True,
            "provider_name": config.provider_name,
            "model_name": config.model_name,
            "latency_ms": 123,
            "preview": "OK",
        }

    monkeypatch.setattr(admin_router, "test_inference_config", _fake_test)

    client = TestClient(app)
    login = client.post(
        "/api/auth/local/login",
        json={"username": "union_demo", "password": "demo_password", "union_slug": "local-1"},
    )
    assert login.status_code == 200
    token = login.json()["access_token"]

    response = client.post(
        f"/api/admin/unions/{union_id}/provider/test",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "provider_name": "openrouter",
            "model_name": "qwen/qwen3.5-flash-02-23",
            "api_key": "test-key",
            "config": {"base_url": "https://openrouter.ai/api/v1"},
        },
    )

    assert response.status_code == 200
    payload = response.json()["result"]
    assert payload["ok"] is True
    assert payload["provider_name"] == "openrouter"
    assert payload["model_name"] == "qwen/qwen3.5-flash-02-23"


def test_admin_route_allows_membership_backed_union_admin(tmp_path):
    app, SessionLocal = _build_app(tmp_path)

    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        user = User(email="admin@example.com", full_name="Union Admin")
        db.add_all([union, user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=user.id, role=Role.UNION_ADMIN))
        db.commit()

    client = TestClient(app)
    response = client.get(
        "/api/admin/unions",
        headers={
            "X-Auth-Email": "admin@example.com",
            "X-Auth-Role": "user",
            "X-Union-Slug": "local-1",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["items"]) == 1
    assert payload["items"][0]["slug"] == "local-1"


def test_document_upload_creates_ingestion_job(tmp_path):
    app, SessionLocal = _build_app(tmp_path)

    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        user = User(email="admin@example.com", full_name="Union Admin")
        db.add_all([union, user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=user.id, role=Role.UNION_ADMIN))
        db.commit()
        union_id = union.id

    client = TestClient(app)
    response = client.post(
        f"/api/admin/unions/{union_id}/documents",
        headers={
            "X-Auth-Email": "admin@example.com",
            "X-Auth-Role": "user",
            "X-Union-Slug": "local-1",
        },
        files={"file": ("contract.txt", b"test content", "text/plain")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"]
    assert payload["ingestion_job_id"]
    assert payload["ingestion_status"] == IngestionJobStatus.SUCCEEDED.value
    assert payload["artifact_key"]

    with SessionLocal() as db:
        document = db.get(Document, payload["id"])
        ingestion_job = db.get(IngestionJob, payload["ingestion_job_id"])
        chunks = db.query(ChunkEmbedding).filter(ChunkEmbedding.document_id == payload["id"]).order_by(ChunkEmbedding.chunk_index.asc()).all()

        assert document is not None
        assert document.title == "contract.txt"
        assert document.union_id == union_id
        assert document.status == DocumentStatus.ACTIVE

        assert ingestion_job is not None
        assert ingestion_job.document_id == document.id
        assert ingestion_job.union_id == union_id
        assert ingestion_job.status == IngestionJobStatus.SUCCEEDED
        assert ingestion_job.metadata_json["trigger"] == "upload"
        assert ingestion_job.metadata_json["mode"] == "inline"
        assert ingestion_job.metadata_json["chunk_count"] == len(chunks)
        assert ingestion_job.metadata_json["artifact_key"]
        assert len(chunks) >= 1
        assert "test content" in chunks[0].chunk_text


def test_document_delete_removes_file_jobs_and_chunks(tmp_path):
    app, SessionLocal = _build_app(tmp_path)

    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        user = User(email="admin@example.com", full_name="Union Admin")
        db.add_all([union, user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=user.id, role=Role.UNION_ADMIN))
        db.commit()
        union_id = union.id

    client = TestClient(app)
    upload = client.post(
        f"/api/admin/unions/{union_id}/documents",
        headers={
            "X-Auth-Email": "admin@example.com",
            "X-Auth-Role": "user",
            "X-Union-Slug": "local-1",
        },
        files={"file": ("contract.txt", b"test content", "text/plain")},
    )
    assert upload.status_code == 200
    payload = upload.json()

    deleted = client.delete(
        f"/api/admin/unions/{union_id}/documents/{payload['id']}",
        headers={
            "X-Auth-Email": "admin@example.com",
            "X-Auth-Role": "user",
            "X-Union-Slug": "local-1",
        },
    )
    assert deleted.status_code == 200
    deleted_payload = deleted.json()
    assert deleted_payload["deleted"] is True
    assert deleted_payload["deleted_chunks"] >= 1
    assert deleted_payload["deleted_jobs"] == 1

    with SessionLocal() as db:
        document = db.get(Document, payload["id"])
        ingestion_job = db.get(IngestionJob, payload["ingestion_job_id"])
        chunks = db.query(ChunkEmbedding).filter(ChunkEmbedding.document_id == payload["id"]).all()
        audit_events = db.query(AuditEvent).filter(AuditEvent.union_id == union_id, AuditEvent.event_type == "document_deleted").all()

        assert document is None
        assert ingestion_job is None
        assert chunks == []
        assert len(audit_events) == 1

    storage_root = tmp_path / "storage" / "local-1"
    assert not (storage_root / "contract.txt").exists()
    assert not (storage_root / payload["id"]).exists()


def test_pdf_upload_is_deferred_without_configured_parser(tmp_path):
    app, SessionLocal = _build_app(tmp_path)

    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        user = User(email="admin@example.com", full_name="Union Admin")
        db.add_all([union, user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=user.id, role=Role.UNION_ADMIN))
        db.commit()
        union_id = union.id

    client = TestClient(app)
    response = client.post(
        f"/api/admin/unions/{union_id}/documents",
        headers={
            "X-Auth-Email": "admin@example.com",
            "X-Auth-Role": "user",
            "X-Union-Slug": "local-1",
        },
        files={"file": ("contract.pdf", b"%PDF-1.4 fake pdf", "application/pdf")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ingestion_status"] == IngestionJobStatus.PENDING.value
    assert payload["artifact_key"] is None

    with SessionLocal() as db:
        document = db.get(Document, payload["id"])
        ingestion_job = db.get(IngestionJob, payload["ingestion_job_id"])
        chunks = db.query(ChunkEmbedding).filter(ChunkEmbedding.document_id == payload["id"]).all()

        assert document is not None
        assert document.status == DocumentStatus.PROCESSING
        assert ingestion_job is not None
        assert ingestion_job.status == IngestionJobStatus.PENDING
        assert ingestion_job.metadata_json["mode"] == "deferred"
        assert ingestion_job.metadata_json["parser"] is None
        assert ingestion_job.metadata_json["deferred_reason"] == "no_parser_available"
        assert chunks == []


def test_ingestion_job_routes_list_detail_and_retry(tmp_path):
    app, SessionLocal = _build_app(tmp_path)

    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        user = User(email="admin@example.com", full_name="Union Admin")
        db.add_all([union, user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=user.id, role=Role.UNION_ADMIN))
        db.commit()
        union_id = union.id

    client = TestClient(app)
    upload = client.post(
        f"/api/admin/unions/{union_id}/documents",
        headers={
            "X-Auth-Email": "admin@example.com",
            "X-Auth-Role": "user",
            "X-Union-Slug": "local-1",
        },
        files={"file": ("contract.pdf", b"%PDF-1.4 fake pdf", "application/pdf")},
    )
    assert upload.status_code == 200
    uploaded = upload.json()

    headers = {
        "X-Auth-Email": "admin@example.com",
        "X-Auth-Role": "user",
        "X-Union-Slug": "local-1",
    }

    listed = client.get(f"/api/admin/unions/{union_id}/ingestion-jobs", headers=headers)
    detail = client.get(
        f"/api/admin/unions/{union_id}/ingestion-jobs/{uploaded['ingestion_job_id']}",
        headers=headers,
    )
    retry = client.post(
        f"/api/admin/unions/{union_id}/ingestion-jobs/{uploaded['ingestion_job_id']}/retry",
        headers=headers,
        json={"ocr_enabled": True},
    )

    assert listed.status_code == 200
    assert detail.status_code == 200
    assert retry.status_code == 200

    list_items = listed.json()["items"]
    assert len(list_items) == 1
    assert list_items[0]["status"] == IngestionJobStatus.PENDING.value
    assert list_items[0]["estimated_ready_seconds"] is not None

    job = detail.json()["job"]
    assert job["id"] == uploaded["ingestion_job_id"]
    assert job["document_title"] == "contract.pdf"

    retry_job = retry.json()["job"]
    assert retry_job["status"] == IngestionJobStatus.PENDING.value
    assert retry_job["metadata"]["trigger"] == "retry"
    assert retry_job["metadata"]["source_job_id"] == uploaded["ingestion_job_id"]
    assert retry_job["metadata"]["ocr_enabled"] is True

    with SessionLocal() as db:
        jobs = (
            db.query(IngestionJob)
            .filter(IngestionJob.union_id == union_id)
            .order_by(IngestionJob.created_at.asc())
            .all()
        )
        assert len(jobs) == 2
        assert jobs[0].metadata_json["trigger"] == "upload"
        assert jobs[1].metadata_json["trigger"] == "retry"


def test_ingestion_job_estimates_reflect_queue_priority_and_ocr_weight(tmp_path):
    app, SessionLocal = _build_app(tmp_path)

    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        user = User(email="admin@example.com", full_name="Union Admin")
        db.add_all([union, user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=user.id, role=Role.UNION_ADMIN))
        db.flush()

        heavy_document = Document(
            union_id=union.id,
            uploaded_by_user_id=user.id,
            title="scan.pdf",
            storage_key="local-1/scan.pdf",
            content_type="application/pdf",
            bytes_size=12_000_000,
            status=DocumentStatus.PROCESSING,
            metadata_json={"page_count": 120, "scan_likelihood": "high", "review_status": "retry_pending"},
        )
        light_document = Document(
            union_id=union.id,
            uploaded_by_user_id=user.id,
            title="notice.txt",
            storage_key="local-1/notice.txt",
            content_type="text/plain",
            bytes_size=512,
            status=DocumentStatus.PROCESSING,
            metadata_json={"review_status": "pending_ingestion"},
        )
        db.add_all([heavy_document, light_document])
        db.flush()

        heavy_job = IngestionJob(
            union_id=union.id,
            document_id=heavy_document.id,
            requested_by_user_id=user.id,
            status=IngestionJobStatus.PENDING,
            metadata_json={"trigger": "retry", "ocr_enabled": True},
        )
        light_job = IngestionJob(
            union_id=union.id,
            document_id=light_document.id,
            requested_by_user_id=user.id,
            status=IngestionJobStatus.PENDING,
            metadata_json={"trigger": "upload"},
        )
        db.add_all([heavy_job, light_job])
        db.commit()
        union_id = union.id

    client = TestClient(app)
    response = client.get(
        f"/api/admin/unions/{union_id}/ingestion-jobs",
        headers={
            "X-Auth-Email": "admin@example.com",
            "X-Auth-Role": "user",
            "X-Union-Slug": "local-1",
        },
    )

    assert response.status_code == 200
    items_by_title = {item["document_title"]: item for item in response.json()["items"]}
    assert items_by_title["notice.txt"]["estimated_ready_seconds"] < items_by_title["scan.pdf"]["estimated_ready_seconds"]


def test_document_list_surfaces_quality_state_and_review_escalation(tmp_path):
    app, SessionLocal = _build_app(tmp_path)

    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        user = User(email="admin@example.com", full_name="Union Admin")
        db.add_all([union, user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=user.id, role=Role.UNION_ADMIN))
        document = Document(
            union_id=union.id,
            uploaded_by_user_id=user.id,
            title="scan.pdf",
            storage_key="local-1/scan.pdf",
            content_type="application/pdf",
            bytes_size=1234,
            status=DocumentStatus.FAILED,
            metadata_json={
                "quality_status": "needs_review",
                "recommended_action": "retry_with_ocr",
                "ready_for_query": False,
            },
        )
        db.add(document)
        db.flush()
        job = IngestionJob(
            union_id=union.id,
            document_id=document.id,
            requested_by_user_id=user.id,
            status=IngestionJobStatus.SUCCEEDED,
            metadata_json={
                "trigger": "upload",
                "quality_status": "needs_review",
                "recommended_action": "retry_with_ocr",
            },
        )
        db.add(job)
        db.commit()
        union_id = union.id
        job_id = job.id

    headers = {
        "X-Auth-Email": "admin@example.com",
        "X-Auth-Role": "user",
        "X-Union-Slug": "local-1",
    }
    client = TestClient(app)

    documents = client.get(f"/api/admin/unions/{union_id}/documents", headers=headers)
    escalate = client.post(
        f"/api/admin/unions/{union_id}/ingestion-jobs/{job_id}/escalate-review",
        headers=headers,
        json={"note": "Please review this scanned file."},
    )

    assert documents.status_code == 200
    assert escalate.status_code == 200

    item = documents.json()["items"][0]
    assert item["quality_status"] == "needs_review"
    assert item["ready_for_query"] is False
    assert item["recommended_action"] == "retry_with_ocr"
    assert item["review_status"] == "needs_review"
    assert item["latest_ingestion_job"]["id"] == job_id

    escalated = escalate.json()["job"]
    assert escalated["metadata"]["escalated_for_review"] is True
    assert escalated["metadata"]["escalation_note"] == "Please review this scanned file."

    with SessionLocal() as db:
        security_events = db.query(SecurityEvent).filter(SecurityEvent.union_id == union_id).all()
        notifications = db.query(Notification).filter(Notification.subject == "Security alert: ingestion_review_escalated").all()
        job = db.get(IngestionJob, job_id)

        assert any(event.event_type == "ingestion_review_escalated" for event in security_events)
        assert notifications
        assert job.metadata_json["escalated_for_review"] is True


def test_ingestion_review_state_can_be_updated(tmp_path):
    app, SessionLocal = _build_app(tmp_path)

    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        user = User(email="admin@example.com", full_name="Union Admin")
        db.add_all([union, user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=user.id, role=Role.UNION_ADMIN))
        document = Document(
            union_id=union.id,
            uploaded_by_user_id=user.id,
            title="scan.pdf",
            storage_key="local-1/scan.pdf",
            content_type="application/pdf",
            bytes_size=1234,
            status=DocumentStatus.FAILED,
            metadata_json={"quality_status": "needs_review", "review_status": "needs_review", "ready_for_query": False},
        )
        db.add(document)
        db.flush()
        job = IngestionJob(
            union_id=union.id,
            document_id=document.id,
            requested_by_user_id=user.id,
            status=IngestionJobStatus.SUCCEEDED,
            metadata_json={"quality_status": "needs_review", "review_status": "needs_review"},
        )
        db.add(job)
        db.commit()
        union_id = union.id
        job_id = job.id

    headers = {
        "X-Auth-Email": "admin@example.com",
        "X-Auth-Role": "user",
        "X-Union-Slug": "local-1",
    }
    client = TestClient(app)
    response = client.post(
        f"/api/admin/unions/{union_id}/ingestion-jobs/{job_id}/review-state",
        headers=headers,
        json={"review_status": "in_review", "note": "Assigned to steward"},
    )

    assert response.status_code == 200
    assert response.json()["review_status"] == "in_review"
    assert response.json()["job"]["metadata"]["review_status"] == "in_review"

    with SessionLocal() as db:
        job = db.get(IngestionJob, job_id)
        document = db.scalar(select(Document).where(Document.id == job.document_id))
        assert document is not None
        assert document.metadata_json["review_status"] == "in_review"
        assert document.metadata_json["review_note"] == "Assigned to steward"
        assert job.metadata_json["review_status"] == "in_review"


def test_superadmin_can_load_union_debug_config_and_export_bundle(tmp_path):
    app, SessionLocal = _build_app(tmp_path)

    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1", metadata_json={"branding": {"theme_color": "#123456"}})
        super_user = User(email="super@example.com", full_name="Super Admin")
        db.add_all([union, super_user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=super_user.id, role=Role.SUPER_ADMIN))
        document = Document(
            union_id=union.id,
            uploaded_by_user_id=super_user.id,
            title="contract.pdf",
            storage_key="local-1/contract.pdf",
            content_type="application/pdf",
            bytes_size=2048,
            status=DocumentStatus.ACTIVE,
            metadata_json={"ready_for_query": True},
        )
        db.add(document)
        db.add(
            AuditEvent(
                union_id=union.id,
                actor_user_id=super_user.id,
                event_type="seeded",
                event_payload={"source": "test"},
            )
        )
        app.state.platform.local_auth.create_or_update_credential(
            db,
            user=super_user,
            username="superadmin",
            password="demo_password",
        )
        db.commit()
        union_id = union.id

    client = TestClient(app)
    login = client.post("/api/auth/local/login", json={"username": "superadmin", "password": "demo_password", "union_slug": "local-1"})
    token = login.json()["access_token"]

    debug_response = client.get(f"/api/admin/unions/{union_id}/debug-config", headers={"Authorization": f"Bearer {token}"})
    export_response = client.get(f"/api/admin/unions/{union_id}/export", headers={"Authorization": f"Bearer {token}"})

    assert debug_response.status_code == 200
    assert debug_response.json()["union"]["slug"] == "local-1"
    assert debug_response.json()["union"]["metadata"]["branding"]["theme_color"] == "#123456"

    assert export_response.status_code == 200
    assert 'attachment; filename="local-1-export.json"' == export_response.headers["content-disposition"]
    export_payload = export_response.json()
    assert export_payload["union"]["slug"] == "local-1"
    assert export_payload["documents"][0]["title"] == "contract.pdf"
    assert export_payload["audit_events"][0]["event_type"] == "seeded"


def test_superadmin_review_queue_supports_union_and_query_filters(tmp_path):
    app, SessionLocal = _build_app(tmp_path)

    with SessionLocal() as db:
        union_one = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        union_two = Union(slug="local-2", name="Local 2", union_local_id="local-2")
        super_user = User(email="super@example.com", full_name="Super Admin")
        db.add_all([union_one, union_two, super_user])
        db.flush()
        db.add(UnionMembership(union_id=union_one.id, user_id=super_user.id, role=Role.SUPER_ADMIN))
        db.add(UnionMembership(union_id=union_two.id, user_id=super_user.id, role=Role.SUPER_ADMIN))
        db.add_all(
            [
                Document(
                    union_id=union_one.id,
                    uploaded_by_user_id=super_user.id,
                    title="Vacation Rules",
                    storage_key="local-1/vacation.pdf",
                    content_type="application/pdf",
                    bytes_size=1024,
                    status=DocumentStatus.FAILED,
                    metadata_json={
                        "review_status": "needs_review",
                        "quality_reason": "Vacation article needs manual verification",
                        "recommended_action": "review vacation section",
                    },
                ),
                Document(
                    union_id=union_two.id,
                    uploaded_by_user_id=super_user.id,
                    title="Seniority Rules",
                    storage_key="local-2/seniority.pdf",
                    content_type="application/pdf",
                    bytes_size=1024,
                    status=DocumentStatus.FAILED,
                    metadata_json={
                        "review_status": "needs_review",
                        "quality_reason": "Seniority language needs review",
                        "recommended_action": "review seniority section",
                    },
                ),
            ]
        )
        app.state.platform.local_auth.create_or_update_credential(
            db,
            user=super_user,
            username="superadmin",
            password="demo_password",
        )
        db.commit()
        union_one_id = union_one.id

    client = TestClient(app)
    login = client.post("/api/auth/local/login", json={"username": "superadmin", "password": "demo_password", "union_slug": "local-1"})
    token = login.json()["access_token"]

    response = client.get(
        f"/api/ops/review-queue?union_id={union_one_id}&q=vacation&review_status=needs_review&status=failed",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["unresolved_documents"] == 1
    assert payload["items"][0]["title"] == "Vacation Rules"
    assert payload["items"][0]["union_name"] == "Local 1"


def test_union_admin_dashboard_returns_union_scoped_metrics_and_trends(tmp_path):
    app, SessionLocal = _build_app(tmp_path)
    now = datetime.utcnow()

    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        admin_user = User(email="admin@example.com", full_name="Union Admin")
        db.add_all([union, admin_user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=admin_user.id, role=Role.UNION_ADMIN))
        app.state.platform.local_auth.create_or_update_credential(db, user=admin_user, username="union_admin", password="demo_password")
        db.add(
            Document(
                union_id=union.id,
                uploaded_by_user_id=admin_user.id,
                title="contract.pdf",
                storage_key="local-1/contract.pdf",
                content_type="application/pdf",
                bytes_size=1024,
                status=DocumentStatus.ACTIVE,
                metadata_json={"ready_for_query": True, "review_status": "ready"},
            )
        )
        db.add(
            Document(
                union_id=union.id,
                uploaded_by_user_id=admin_user.id,
                title="scan.pdf",
                storage_key="local-1/scan.pdf",
                content_type="application/pdf",
                bytes_size=1024,
                status=DocumentStatus.FAILED,
                metadata_json={"review_status": "needs_review"},
            )
        )
        db.add(
            UsageEvent(
                union_id=union.id,
                user_id=admin_user.id,
                route="/api/query",
                request_count=4,
                token_count=1200,
                estimated_cost_usd=0.18,
                created_at=now - timedelta(hours=2),
            )
        )
        db.add(
            Notification(
                union_id=union.id,
                channel="ui",
                subject="Union alert: ingestion_review_required",
                body="Needs review",
                status=NotificationStatus.PENDING,
                created_at=now - timedelta(hours=3),
            )
        )
        db.add(
            SecurityEvent(
                union_id=union.id,
                event_type="provider_timeout",
                severity=SecuritySeverity.WARNING,
                response_action="observe",
                details_json={"source": "test"},
                created_at=now - timedelta(days=1),
            )
        )
        db.add(
            AuthSession(
                user_id=admin_user.id,
                union_id=union.id,
                session_secret_hash="hash-dashboard-1",
                session_type=SessionType.UNION_ADMIN,
                created_at=now - timedelta(days=2),
                last_seen_at=now - timedelta(days=2),
                expires_at=now + timedelta(days=1),
            )
        )
        db.add_all(
            [
                TelemetryEvent(
                    union_id=union.id,
                    user_id=admin_user.id,
                    route="/u/local-1/",
                    category="bug_journey",
                    event_type="query_failed",
                    metadata_json={"reason": "timeout"},
                    created_at=now - timedelta(hours=2),
                ),
                TelemetryEvent(
                    union_id=union.id,
                    user_id=admin_user.id,
                    route="/u/local-1/",
                    category="usage_ux",
                    event_type="source_opened",
                    metadata_json={"document_id": "doc-1"},
                    created_at=now - timedelta(hours=1),
                ),
                TelemetryEvent(
                    union_id=union.id,
                    user_id=admin_user.id,
                    route="/u/local-1/",
                    category="bug_journey",
                    event_type="member_workspace_loaded",
                    metadata_json={},
                    created_at=now - timedelta(hours=1),
                ),
            ]
        )
        db.commit()

    client = TestClient(app)
    login = client.post("/api/auth/local/login", json={"username": "union_admin", "password": "demo_password", "union_slug": "local-1"})
    token = login.json()["access_token"]
    response = client.get("/api/ops/dashboard", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["scope_label"] == "Local 1"
    assert payload["summary"]["requests_last_24h"] == 4
    assert payload["summary"]["active_users_7d"] == 1
    assert payload["summary"]["open_review_items"] == 1
    assert payload["summary"]["pending_alerts"] == 1
    assert payload["summary"]["ready_documents"] == 1
    assert payload["summary"]["query_failures_7d"] == 1
    assert payload["summary"]["source_opens_7d"] == 1
    assert payload["summary"]["member_workspace_loads_7d"] == 1
    assert len(payload["trends"]["labels"]) == 7
    assert "query_failures" in payload["trends"]


def test_superadmin_telemetry_feed_supports_filters_and_paging(tmp_path):
    app, SessionLocal = _build_app(tmp_path)
    now = datetime.utcnow()

    with SessionLocal() as db:
        union_one = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        union_two = Union(slug="local-2", name="Local 2", union_local_id="local-2")
        super_user = User(email="super@example.com", full_name="Super Admin")
        db.add_all([union_one, union_two, super_user])
        db.flush()
        db.add(UnionMembership(union_id=union_one.id, user_id=super_user.id, role=Role.SUPER_ADMIN))
        db.add(UnionMembership(union_id=union_two.id, user_id=super_user.id, role=Role.SUPER_ADMIN))
        app.state.platform.local_auth.create_or_update_credential(db, user=super_user, username="superadmin", password="demo_password")
        db.add_all(
            [
                TelemetryEvent(
                    union_id=union_one.id,
                    user_id=super_user.id,
                    session_id="session-a",
                    route="/u/local-1/",
                    category="bug_journey",
                    event_type="query_failed",
                    metadata_json={"reason": "timeout"},
                    created_at=now - timedelta(minutes=5),
                ),
                TelemetryEvent(
                    union_id=union_one.id,
                    user_id=super_user.id,
                    session_id="session-a",
                    route="/u/local-1/",
                    category="usage_ux",
                    event_type="source_opened",
                    metadata_json={"document_id": "doc-1"},
                    created_at=now - timedelta(minutes=4),
                ),
                TelemetryEvent(
                    union_id=union_two.id,
                    user_id=super_user.id,
                    session_id="session-b",
                    route="/u/local-2/",
                    category="bug_journey",
                    event_type="member_workspace_loaded",
                    metadata_json={"signed_in": True},
                    created_at=now - timedelta(minutes=3),
                ),
            ]
        )
        db.commit()
        union_one_id = union_one.id

    client = TestClient(app)
    login = client.post("/api/auth/local/login", json={"username": "superadmin", "password": "demo_password", "union_slug": "local-1"})
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    filtered = client.get(
        f"/api/ops/telemetry-events?union_id={union_one_id}&category=bug_journey&event_type=query_failed&page=1&page_size=1",
        headers=headers,
    )

    assert filtered.status_code == 200
    payload = filtered.json()
    assert payload["total"] == 1
    assert payload["items"][0]["event_type"] == "query_failed"
    assert payload["items"][0]["union_name"] == "Local 1"

    session_timeline = client.get(
        f"/api/ops/telemetry-events/session/session-a?union_id={union_one_id}",
        headers=headers,
    )
    assert session_timeline.status_code == 200
    timeline_payload = session_timeline.json()
    assert timeline_payload["session_id"] == "session-a"
    assert timeline_payload["summary"]["total_events"] == 2
    assert [item["event_type"] for item in timeline_payload["items"]] == ["query_failed", "source_opened"]


def test_superadmin_dashboard_can_return_platform_or_selected_union_scope(tmp_path):
    app, SessionLocal = _build_app(tmp_path)
    now = datetime.utcnow()

    with SessionLocal() as db:
        union_one = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        union_two = Union(slug="local-2", name="Local 2", union_local_id="local-2")
        super_user = User(email="super@example.com", full_name="Super Admin")
        member_user = User(email="member@example.com", full_name="Member User")
        db.add_all([union_one, union_two, super_user, member_user])
        db.flush()
        db.add(UnionMembership(union_id=union_one.id, user_id=super_user.id, role=Role.SUPER_ADMIN))
        db.add(UnionMembership(union_id=union_two.id, user_id=super_user.id, role=Role.SUPER_ADMIN))
        db.add(UnionMembership(union_id=union_two.id, user_id=member_user.id, role=Role.USER))
        app.state.platform.local_auth.create_or_update_credential(db, user=super_user, username="superadmin", password="demo_password")
        db.add_all(
            [
                UsageEvent(
                    union_id=union_one.id,
                    user_id=super_user.id,
                    route="/api/query",
                    request_count=2,
                    token_count=500,
                    estimated_cost_usd=0.08,
                    created_at=now - timedelta(hours=1),
                ),
                UsageEvent(
                    union_id=union_two.id,
                    user_id=member_user.id,
                    route="/api/query",
                    request_count=3,
                    token_count=700,
                    estimated_cost_usd=0.11,
                    created_at=now - timedelta(hours=1),
                ),
            ]
        )
        db.commit()
        union_two_id = union_two.id

    client = TestClient(app)
    login = client.post("/api/auth/local/login", json={"username": "superadmin", "password": "demo_password", "union_slug": "local-1"})
    token = login.json()["access_token"]

    platform_response = client.get("/api/ops/dashboard", headers={"Authorization": f"Bearer {token}"})
    union_response = client.get(f"/api/ops/dashboard?union_id={union_two_id}", headers={"Authorization": f"Bearer {token}"})

    assert platform_response.status_code == 200
    assert platform_response.json()["scope_label"] == "All unions"
    assert platform_response.json()["summary"]["requests_last_24h"] == 5

    assert union_response.status_code == 200
    assert union_response.json()["scope_label"] == "Local 2"
    assert union_response.json()["summary"]["requests_last_24h"] == 3
