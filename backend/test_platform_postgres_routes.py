import base64
import os
import uuid
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from backend.platform.auth import HeaderAuthAdapter
from backend.platform.db import apply_request_context
from backend.platform.chat_history import ChatHistoryStore
from backend.platform.crypto import SecretCipher
from backend.platform.db import Base
from backend.platform.guardrails import GuardrailService
from backend.platform.ingestion import IngestionService
from backend.platform.local_auth import LocalAuthService
from backend.platform.middleware import PlatformContextMiddleware, QueryGovernanceMiddleware, SecurityHeadersMiddleware
from backend.platform.models import (
    AuditEvent,
    Chat,
    ChunkEmbedding,
    Document,
    DocumentStatus,
    IngestionJob,
    IngestionJobStatus,
    Message,
    Notification,
    NotificationStatus,
    ProviderConfig,
    QuotaPolicy,
    Role,
    SecurityEvent,
    SecuritySeverity,
    Union,
    UnionMembership,
    UsageEvent,
    User,
)
from backend.platform.parsing import LiteParseDocumentParser, ParserRegistry, PlainTextDocumentParser
from backend.platform.quotas import QuotaService
from backend.platform.retrieval import TenantRetrievalService
from backend.platform.routers import auth as auth_router
from backend.platform.routers import admin as admin_router
from backend.platform.routers import ops as ops_router
from backend.platform.sentinel import SentinelService
from backend.platform.service_container import ServiceContainer
from backend.platform.settings import PlatformSettings
from backend.platform.storage import LocalDiskStorage


ADMIN_URL_ENV = "KARL_TEST_POSTGRES_ADMIN_URL"
TEST_DB_ENV = "KARL_TEST_POSTGRES_DB"


pytestmark = pytest.mark.skipif(
    not os.getenv(ADMIN_URL_ENV),
    reason=f"Set {ADMIN_URL_ENV} to run live PostgreSQL route integration tests.",
)


@pytest.fixture()
def postgres_route_env(tmp_path):
    for sidecar in Path("alembic").rglob("._*"):
        sidecar.unlink(missing_ok=True)

    admin_url = make_url(os.environ[ADMIN_URL_ENV])
    db_name = os.getenv(TEST_DB_ENV, f"karl_routes_test_{uuid.uuid4().hex[:8]}")
    db_url = admin_url.set(database=db_name)
    app_role = f"karl_routes_app_{uuid.uuid4().hex[:8]}"
    app_password = f"pw_{uuid.uuid4().hex}"
    app_db_url = db_url.set(username=app_role, password=app_password)

    admin_engine = create_engine(admin_url, future=True, isolation_level="AUTOCOMMIT")
    with admin_engine.connect() as conn:
        conn.execute(text(f'DROP ROLE IF EXISTS "{app_role}"'))
        conn.execute(text(f"CREATE ROLE \"{app_role}\" LOGIN PASSWORD '{app_password}' NOSUPERUSER"))
        conn.execute(text(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)'))
        conn.execute(text(f'CREATE DATABASE "{db_name}"'))

    old_postgres_url = os.environ.get("KARL_POSTGRES_URL")
    os.environ["KARL_POSTGRES_URL"] = str(db_url)
    try:
        command.upgrade(Config("alembic.ini"), "head")

        owner_engine = create_engine(db_url, future=True, pool_pre_ping=True)
        with owner_engine.connect() as conn:
            conn.execute(text(f'GRANT CONNECT ON DATABASE "{db_name}" TO "{app_role}"'))
            conn.execute(text(f'GRANT USAGE ON SCHEMA public TO "{app_role}"'))
            conn.execute(text(f'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO "{app_role}"'))
            conn.execute(text(f'GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO "{app_role}"'))
            conn.commit()

        app_engine = create_engine(app_db_url, future=True, pool_pre_ping=True)
        OwnerSessionLocal = sessionmaker(bind=owner_engine, autoflush=False, autocommit=False, future=True)
        AppSessionLocal = sessionmaker(bind=app_engine, autoflush=False, autocommit=False, future=True)

        settings = PlatformSettings(
            project_root=tmp_path,
            postgres_url=str(app_db_url),
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
            legacy_contract_pipeline_enabled=False,
            inline_parse_max_bytes=1_000_000,
            ocr_auto_retry_enabled=True,
            ocr_auto_retry_max_attempts=1,
            local_auth_token_ttl_seconds=43_200,
            secret_encryption_key=base64.urlsafe_b64encode(b"0" * 32).decode("utf-8"),
            sentinel_email_from="sentinel@example.com",
            sentinel_email_enabled=False,
            bootstrap_super_admin_emails=["super@example.com"],
        )
        storage = LocalDiskStorage(settings.local_storage_root)
        retrieval = TenantRetrievalService()
        parsers = ParserRegistry([PlainTextDocumentParser(), LiteParseDocumentParser(settings.liteparse_executable)])

        sentinel = SentinelService(settings)
        local_auth = LocalAuthService(secret_key=settings.secret_encryption_key, token_ttl_seconds=settings.local_auth_token_ttl_seconds)
        container = ServiceContainer(
            settings=settings,
            engine=app_engine,
            session_factory=AppSessionLocal,
            auth_adapter=HeaderAuthAdapter(settings, local_auth=local_auth),
            guardrails=GuardrailService(token_limit=settings.query_token_limit),
            quotas=QuotaService(settings),
            sentinel=sentinel,
            secret_cipher=SecretCipher(settings.secret_encryption_key),
            storage=storage,
            chat_history=ChatHistoryStore(AppSessionLocal),
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

        @app.post("/api/query")
        async def query_stub(request: Request):
            payload = await request.json()
            session_id = str(payload.get("session_id") or "").strip()
            union_local_id = str(payload.get("union_local_id") or "").strip() or None
            if session_id:
                container.chat_history.bind_session(
                    session_id=session_id,
                    union_local_id=union_local_id,
                )
                container.chat_history.persist_turn(
                    session_id=session_id,
                    question=str(payload.get("question") or ""),
                    answer="Safe stub answer.",
                    metadata={"source": "query_stub"},
                )
            return {
                "answer": "Safe stub answer.",
                "session_id": session_id or None,
            }

        try:
            yield {
                "app": app,
                "owner_session_factory": OwnerSessionLocal,
                "app_session_factory": AppSessionLocal,
            }
        finally:
            app_engine.dispose()
            owner_engine.dispose()
    finally:
        if old_postgres_url is None:
            os.environ.pop("KARL_POSTGRES_URL", None)
        else:
            os.environ["KARL_POSTGRES_URL"] = old_postgres_url

        with admin_engine.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)'))
            conn.execute(text(f'DROP ROLE IF EXISTS "{app_role}"'))
        admin_engine.dispose()


def _seed_route_data(session_factory):
    with session_factory() as db:
        union_one = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        union_two = Union(slug="local-2", name="Local 2", union_local_id="local-2")
        user_one = User(email="admin1@example.com", full_name="Admin One")
        user_two = User(email="admin2@example.com", full_name="Admin Two")
        db.add_all([union_one, union_two, user_one, user_two])
        db.flush()

        db.add_all(
            [
                UnionMembership(union_id=union_one.id, user_id=user_one.id, role=Role.UNION_ADMIN),
                UnionMembership(union_id=union_two.id, user_id=user_two.id, role=Role.UNION_ADMIN),
                ProviderConfig(
                    union_id=union_one.id,
                    provider_name="openai",
                    model_name="gpt-test",
                    encrypted_api_key="enc-1",
                    config_json={"temperature": 0.1},
                ),
                ProviderConfig(
                    union_id=union_two.id,
                    provider_name="openai",
                    model_name="gpt-test",
                    encrypted_api_key="enc-2",
                    config_json={"temperature": 0.2},
                ),
                QuotaPolicy(union_id=union_one.id, requests_per_day=100),
                QuotaPolicy(union_id=union_two.id, requests_per_day=200),
                SecurityEvent(
                    union_id=union_one.id,
                    user_id=user_one.id,
                    event_type="quota_warning",
                    severity=SecuritySeverity.WARNING,
                    response_action="notify",
                    details_json={"source": "union_one"},
                ),
                SecurityEvent(
                    union_id=union_two.id,
                    user_id=user_two.id,
                    event_type="quota_warning",
                    severity=SecuritySeverity.WARNING,
                    response_action="notify",
                    details_json={"source": "union_two"},
                ),
                Notification(
                    union_id=union_one.id,
                    user_id=user_one.id,
                    channel="in_app",
                    subject="Union One Notice",
                    body="union one",
                    status=NotificationStatus.PENDING,
                ),
                Notification(
                    union_id=union_two.id,
                    user_id=user_two.id,
                    channel="in_app",
                    subject="Union Two Notice",
                    body="union two",
                    status=NotificationStatus.PENDING,
                ),
                UsageEvent(
                    union_id=union_one.id,
                    user_id=user_one.id,
                    route="/api/query",
                    token_count=10,
                    estimated_cost_usd=0.01,
                ),
                UsageEvent(
                    union_id=union_two.id,
                    user_id=user_two.id,
                    route="/api/query",
                    token_count=20,
                    estimated_cost_usd=0.02,
                ),
            ]
        )
        db.commit()
        return {
            "union_one_id": union_one.id,
            "union_two_id": union_two.id,
        }


def test_postgres_admin_and_ops_routes_are_tenant_scoped(postgres_route_env):
    seeded = _seed_route_data(postgres_route_env["owner_session_factory"])
    client = TestClient(postgres_route_env["app"])

    headers = {
        "X-Auth-Email": "admin1@example.com",
        "X-Auth-Role": "user",
        "X-Union-Slug": "local-1",
    }

    unions = client.get("/api/admin/unions", headers=headers)
    users = client.get(f"/api/admin/unions/{seeded['union_one_id']}/users", headers=headers)
    provider = client.get(f"/api/admin/unions/{seeded['union_one_id']}/provider", headers=headers)
    quota = client.get(f"/api/admin/unions/{seeded['union_one_id']}/quota", headers=headers)
    security_events = client.get("/api/ops/security-events", headers=headers)
    notifications = client.get("/api/ops/notifications", headers=headers)
    usage = client.get("/api/ops/usage", headers=headers)

    assert unions.status_code == 200
    assert [item["slug"] for item in unions.json()["items"]] == ["local-1"]

    assert users.status_code == 200
    assert [item["email"] for item in users.json()["items"]] == ["admin1@example.com"]

    assert provider.status_code == 200
    assert provider.json()["provider"]["config"] == {"temperature": 0.1}

    assert quota.status_code == 200
    assert quota.json()["quota"]["requests_per_day"] == 100

    assert security_events.status_code == 200
    assert [item["details"]["source"] for item in security_events.json()["items"]] == ["union_one"]

    assert notifications.status_code == 200
    assert [item["subject"] for item in notifications.json()["items"]] == ["Union One Notice"]

    assert usage.status_code == 200
    assert [item["token_count"] for item in usage.json()["items"]] == [10]


def test_postgres_admin_route_blocks_cross_tenant_access(postgres_route_env):
    seeded = _seed_route_data(postgres_route_env["owner_session_factory"])
    client = TestClient(postgres_route_env["app"])

    headers = {
        "X-Auth-Email": "admin1@example.com",
        "X-Auth-Role": "union_admin",
        "X-Union-Slug": "local-1",
    }

    response = client.get(f"/api/admin/unions/{seeded['union_two_id']}/users", headers=headers)
    assert response.status_code == 403


def test_postgres_super_admin_sees_global_ops_and_unions(postgres_route_env):
    _seed_route_data(postgres_route_env["owner_session_factory"])
    client = TestClient(postgres_route_env["app"])

    headers = {
        "X-Auth-Email": "super@example.com",
        "X-Auth-Role": "super_admin",
    }

    unions = client.get("/api/admin/unions", headers=headers)
    security_events = client.get("/api/ops/security-events", headers=headers)
    notifications = client.get("/api/ops/notifications", headers=headers)

    assert unions.status_code == 200
    assert [item["slug"] for item in unions.json()["items"]] == ["local-1", "local-2"]
    assert sorted(item["details"]["source"] for item in security_events.json()["items"]) == ["union_one", "union_two"]
    assert sorted(item["subject"] for item in notifications.json()["items"]) == ["Union One Notice", "Union Two Notice"]


def test_postgres_notification_acknowledgement_is_tenant_scoped(postgres_route_env):
    _seed_route_data(postgres_route_env["owner_session_factory"])
    client = TestClient(postgres_route_env["app"])

    with postgres_route_env["owner_session_factory"]() as db:
        union_notification = db.scalar(select(Notification).where(Notification.subject == "Union One Notice"))
        other_union_notification = db.scalar(select(Notification).where(Notification.subject == "Union Two Notice"))
        union_notification_id = union_notification.id
        other_union_notification_id = other_union_notification.id

    headers_one = {
        "X-Auth-Email": "admin1@example.com",
        "X-Auth-Role": "union_admin",
        "X-Union-Slug": "local-1",
    }

    ack = client.post(f"/api/ops/notifications/{union_notification_id}/acknowledge", headers=headers_one)
    blocked = client.post(f"/api/ops/notifications/{other_union_notification_id}/acknowledge", headers=headers_one)

    assert ack.status_code == 200
    assert ack.json()["notification"]["status"] == NotificationStatus.ACKNOWLEDGED.value
    assert blocked.status_code == 404

    with postgres_route_env["owner_session_factory"]() as db:
        acknowledged = db.get(Notification, union_notification_id)
        untouched = db.get(Notification, other_union_notification_id)

        assert acknowledged.status == NotificationStatus.ACKNOWLEDGED
        assert untouched.status == NotificationStatus.PENDING


def test_postgres_review_queue_and_filtered_notifications_are_tenant_scoped(postgres_route_env):
    seeded = _seed_route_data(postgres_route_env["owner_session_factory"])
    client = TestClient(postgres_route_env["app"])

    with postgres_route_env["owner_session_factory"]() as db:
        user = db.scalar(select(User).where(User.email == "admin1@example.com"))
        document = Document(
            union_id=seeded["union_one_id"],
            uploaded_by_user_id=user.id,
            title="scan.pdf",
            storage_key="local-1/scan.pdf",
            content_type="application/pdf",
            bytes_size=1234,
            status=DocumentStatus.FAILED,
            metadata_json={
                "quality_status": "needs_review",
                "quality_reason": "sparse_page_coverage_after_ocr",
                "ocr_status": "attempted_needs_review",
                "scan_likelihood": "high",
                "review_status": "in_review",
                "recommended_action": "manual_review_after_ocr",
                "ready_for_query": False,
            },
        )
        db.add(document)
        db.flush()
        db.add(
            IngestionJob(
                union_id=seeded["union_one_id"],
                document_id=document.id,
                requested_by_user_id=user.id,
                status=IngestionJobStatus.SUCCEEDED,
                metadata_json={
                    "quality_status": "needs_review",
                    "quality_reason": "sparse_page_coverage_after_ocr",
                    "ocr_status": "attempted_needs_review",
                    "scan_likelihood": "high",
                    "review_status": "in_review",
                },
            )
        )
        db.add(
            Notification(
                union_id=seeded["union_one_id"],
                user_id=user.id,
                channel="in_app",
                subject="Document needs review: scan.pdf",
                body="Review required.",
                status=NotificationStatus.PENDING,
            )
        )
        db.add(
            Notification(
                union_id=seeded["union_one_id"],
                user_id=user.id,
                channel="in_app",
                subject="Document ready: notice.txt",
                body="Done.",
                status=NotificationStatus.ACKNOWLEDGED,
            )
        )
        db.commit()

    headers_one = {
        "X-Auth-Email": "admin1@example.com",
        "X-Auth-Role": "union_admin",
        "X-Union-Slug": "local-1",
    }
    headers_two = {
        "X-Auth-Email": "admin2@example.com",
        "X-Auth-Role": "union_admin",
        "X-Union-Slug": "local-2",
    }

    queue = client.get("/api/ops/review-queue", headers=headers_one)
    review_notifications = client.get("/api/ops/notifications?review_only=true", headers=headers_one)
    blocked_queue = client.get("/api/ops/review-queue", headers=headers_two)

    assert queue.status_code == 200
    assert review_notifications.status_code == 200
    assert blocked_queue.status_code == 200

    queue_payload = queue.json()
    assert queue_payload["summary"]["unresolved_documents"] >= 1
    assert queue_payload["summary"]["pending_notifications"] >= 1
    assert any(item["title"] == "scan.pdf" and item["review_status"] == "in_review" for item in queue_payload["items"])
    queued_item = next(item for item in queue_payload["items"] if item["title"] == "scan.pdf")
    assert queued_item["quality_reason"] == "sparse_page_coverage_after_ocr"
    assert queued_item["ocr_status"] == "attempted_needs_review"
    assert queued_item["scan_likelihood"] == "high"

    review_items = review_notifications.json()["items"]
    assert any(item["subject"] == "Document needs review: scan.pdf" for item in review_items)
    assert all(item["subject"] != "Union Two Notice" for item in review_items)

    blocked_payload = blocked_queue.json()
    assert all(item["union_id"] == seeded["union_two_id"] for item in blocked_payload["items"])


def test_postgres_admin_write_is_blocked_for_other_union(postgres_route_env):
    seeded = _seed_route_data(postgres_route_env["owner_session_factory"])
    client = TestClient(postgres_route_env["app"])

    headers = {
        "X-Auth-Email": "admin1@example.com",
        "X-Auth-Role": "union_admin",
        "X-Union-Slug": "local-1",
    }

    response = client.put(
        f"/api/admin/unions/{seeded['union_two_id']}/quota",
        headers=headers,
        json={
            "requests_per_day": 999,
            "tokens_per_day": 99999,
            "cost_usd_per_day": 999.0,
            "per_user_requests_per_hour": 99,
            "warn_threshold_ratio": 0.9,
            "is_paused": False,
        },
    )

    assert response.status_code == 403


def test_postgres_union_admin_can_update_same_union_provider_and_quota(postgres_route_env):
    seeded = _seed_route_data(postgres_route_env["owner_session_factory"])
    client = TestClient(postgres_route_env["app"])

    headers = {
        "X-Auth-Email": "admin1@example.com",
        "X-Auth-Role": "union_admin",
        "X-Union-Slug": "local-1",
    }

    provider_response = client.put(
        f"/api/admin/unions/{seeded['union_one_id']}/provider",
        headers=headers,
        json={
            "provider_name": "anthropic",
            "model_name": "claude-test",
            "api_key": "updated-secret",
            "config": {"temperature": 0.25},
        },
    )
    quota_response = client.put(
        f"/api/admin/unions/{seeded['union_one_id']}/quota",
        headers=headers,
        json={
            "requests_per_day": 321,
            "tokens_per_day": 654321,
            "cost_usd_per_day": 12.5,
            "per_user_requests_per_hour": 45,
            "warn_threshold_ratio": 0.7,
            "is_paused": True,
        },
    )

    assert provider_response.status_code == 200
    assert quota_response.status_code == 200

    with postgres_route_env["owner_session_factory"]() as db:
        provider = db.scalar(select(ProviderConfig).where(ProviderConfig.union_id == seeded["union_one_id"]))
        quota = db.scalar(select(QuotaPolicy).where(QuotaPolicy.union_id == seeded["union_one_id"]))
        audit_types = db.scalars(
            select(AuditEvent.event_type).where(AuditEvent.union_id == seeded["union_one_id"]).order_by(AuditEvent.created_at.asc())
        ).all()

        assert provider is not None
        assert provider.provider_name == "anthropic"
        assert provider.model_name == "claude-test"
        assert provider.config_json == {"temperature": 0.25}
        assert provider.encrypted_api_key != "updated-secret"

        assert quota is not None
        assert quota.requests_per_day == 321
        assert quota.tokens_per_day == 654321
        assert quota.cost_usd_per_day == 12.5
        assert quota.per_user_requests_per_hour == 45
        assert quota.warn_threshold_ratio == 0.7
        assert quota.is_paused is True

        assert "provider_config_updated" in audit_types
        assert "quota_policy_updated" in audit_types


def test_postgres_chat_retention_is_union_scoped(postgres_route_env):
    seeded = _seed_route_data(postgres_route_env["owner_session_factory"])
    app = postgres_route_env["app"]
    client = TestClient(app)

    with postgres_route_env["owner_session_factory"]() as db:
        union_one = db.get(Union, seeded["union_one_id"])
        union_two = db.get(Union, seeded["union_two_id"])
        union_one.message_retention_enabled = True
        union_two.message_retention_enabled = False
        db.commit()

    chat_history = app.state.platform.chat_history

    retained_binding = chat_history.bind_session(session_id="sess-retained", union_local_id="local-1")
    dropped_binding = chat_history.bind_session(session_id="sess-dropped", union_local_id="local-2")

    assert retained_binding.message_retention_enabled is True
    assert dropped_binding.message_retention_enabled is False

    chat_history.persist_turn(
        session_id="sess-retained",
        question="What is the schedule?",
        answer="Here is the retained answer.",
        metadata={"source": "retained"},
    )
    chat_history.persist_turn(
        session_id="sess-dropped",
        question="Should not persist",
        answer="This should not be stored.",
        metadata={"source": "dropped"},
    )

    headers_one = {
        "X-Auth-Email": "admin1@example.com",
        "X-Auth-Role": "union_admin",
        "X-Union-Slug": "local-1",
    }
    headers_two = {
        "X-Auth-Email": "admin2@example.com",
        "X-Auth-Role": "union_admin",
        "X-Union-Slug": "local-2",
    }

    union_one_chats = client.get(f"/api/admin/unions/{seeded['union_one_id']}/chats", headers=headers_one)
    union_two_chats = client.get(f"/api/admin/unions/{seeded['union_two_id']}/chats", headers=headers_two)

    assert union_one_chats.status_code == 200
    assert [item["session_id"] for item in union_one_chats.json()["items"]] == ["sess-retained"]

    assert union_two_chats.status_code == 200
    assert union_two_chats.json()["items"] == []

    retained_chat_id = union_one_chats.json()["items"][0]["id"]
    retained_detail = client.get(
        f"/api/admin/unions/{seeded['union_one_id']}/chats/{retained_chat_id}",
        headers=headers_one,
    )
    blocked_cross_tenant = client.get(
        f"/api/admin/unions/{seeded['union_two_id']}/chats/{retained_chat_id}",
        headers=headers_two,
    )

    assert retained_detail.status_code == 200
    assert [item["role"] for item in retained_detail.json()["messages"]] == ["user", "assistant"]
    assert [item["content"] for item in retained_detail.json()["messages"]] == [
        "What is the schedule?",
        "Here is the retained answer.",
    ]

    assert blocked_cross_tenant.status_code == 404


def test_postgres_document_upload_creates_tenant_scoped_ingestion_job(postgres_route_env):
    seeded = _seed_route_data(postgres_route_env["owner_session_factory"])
    client = TestClient(postgres_route_env["app"])

    headers = {
        "X-Auth-Email": "admin1@example.com",
        "X-Auth-Role": "union_admin",
        "X-Union-Slug": "local-1",
    }

    response = client.post(
        f"/api/admin/unions/{seeded['union_one_id']}/documents",
        headers=headers,
        files={"file": ("notice.txt", b"tenant scoped upload", "text/plain")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"]
    assert payload["ingestion_job_id"]
    assert payload["ingestion_status"] == IngestionJobStatus.SUCCEEDED.value
    assert payload["artifact_key"]

    with postgres_route_env["owner_session_factory"]() as db:
        document = db.get(Document, payload["id"])
        ingestion_job = db.get(IngestionJob, payload["ingestion_job_id"])
        chunks = db.scalars(
            select(ChunkEmbedding).where(ChunkEmbedding.document_id == payload["id"]).order_by(ChunkEmbedding.chunk_index.asc())
        ).all()

        assert document is not None
        assert document.union_id == seeded["union_one_id"]
        assert document.title == "notice.txt"
        assert document.status == DocumentStatus.ACTIVE

        assert ingestion_job is not None
        assert ingestion_job.union_id == seeded["union_one_id"]
        assert ingestion_job.document_id == document.id
        assert ingestion_job.status == IngestionJobStatus.SUCCEEDED
        assert ingestion_job.metadata_json["filename"] == "notice.txt"
        assert ingestion_job.metadata_json["mode"] == "inline"
        assert ingestion_job.metadata_json["chunk_count"] == len(chunks)
        assert ingestion_job.metadata_json["artifact_key"]
        assert len(chunks) >= 1
        assert "tenant scoped upload" in chunks[0].chunk_text


def test_postgres_pdf_upload_is_deferred_without_parser(postgres_route_env):
    seeded = _seed_route_data(postgres_route_env["owner_session_factory"])
    client = TestClient(postgres_route_env["app"])

    headers = {
        "X-Auth-Email": "admin1@example.com",
        "X-Auth-Role": "union_admin",
        "X-Union-Slug": "local-1",
    }

    response = client.post(
        f"/api/admin/unions/{seeded['union_one_id']}/documents",
        headers=headers,
        files={"file": ("scan.pdf", b"%PDF-1.4 fake pdf", "application/pdf")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ingestion_status"] == IngestionJobStatus.PENDING.value
    assert payload["artifact_key"] is None

    with postgres_route_env["owner_session_factory"]() as db:
        document = db.get(Document, payload["id"])
        ingestion_job = db.get(IngestionJob, payload["ingestion_job_id"])
        chunks = db.scalars(select(ChunkEmbedding).where(ChunkEmbedding.document_id == payload["id"])).all()

        assert document is not None
        assert document.status == DocumentStatus.PROCESSING
        assert ingestion_job is not None
        assert ingestion_job.status == IngestionJobStatus.PENDING
        assert ingestion_job.metadata_json["mode"] == "deferred"
        assert ingestion_job.metadata_json["deferred_reason"] == "no_parser_available"
        assert chunks == []


def test_postgres_tenant_retrieval_search_returns_only_same_union_chunks(postgres_route_env):
    seeded = _seed_route_data(postgres_route_env["owner_session_factory"])
    client = TestClient(postgres_route_env["app"])

    headers_one = {
        "X-Auth-Email": "admin1@example.com",
        "X-Auth-Role": "union_admin",
        "X-Union-Slug": "local-1",
    }
    headers_two = {
        "X-Auth-Email": "admin2@example.com",
        "X-Auth-Role": "union_admin",
        "X-Union-Slug": "local-2",
    }

    upload_one = client.post(
        f"/api/admin/unions/{seeded['union_one_id']}/documents",
        headers=headers_one,
        files={
            "file": (
                "local1.txt",
                b"Seniority rights govern bidding and vacation preference for Local One members.",
                "text/plain",
            )
        },
    )
    upload_two = client.post(
        f"/api/admin/unions/{seeded['union_two_id']}/documents",
        headers=headers_two,
        files={
            "file": (
                "local2.txt",
                b"Uniform reimbursement rules apply to Local Two members and do not cover seniority bidding.",
                "text/plain",
            )
        },
    )

    assert upload_one.status_code == 200
    assert upload_two.status_code == 200

    with postgres_route_env["app_session_factory"]() as db:
        apply_request_context(
            db,
            type(
                "Auth",
                (),
                {
                    "role": Role.UNION_ADMIN.value,
                    "union_id": seeded["union_one_id"],
                    "user_id": None,
                },
            )(),
        )
        results = postgres_route_env["app"].state.platform.retrieval.search(
            db,
            union_id=seeded["union_one_id"],
            query="Which rules cover seniority bidding?",
            limit=5,
        )

        assert results
        assert all(result.document_id != upload_two.json()["id"] for result in results)
        assert any(result.document_id == upload_one.json()["id"] for result in results)
        assert "Local One" in results[0].content


def test_postgres_retrieval_search_excludes_non_ready_documents(postgres_route_env):
    seeded = _seed_route_data(postgres_route_env["owner_session_factory"])
    client = TestClient(postgres_route_env["app"])

    headers = {
        "X-Auth-Email": "admin1@example.com",
        "X-Auth-Role": "union_admin",
        "X-Union-Slug": "local-1",
    }

    upload = client.post(
        f"/api/admin/unions/{seeded['union_one_id']}/documents",
        headers=headers,
        files={"file": ("local1.txt", b"Seniority rights govern bidding and vacation preference.", "text/plain")},
    )
    assert upload.status_code == 200
    document_id = upload.json()["id"]

    with postgres_route_env["owner_session_factory"]() as db:
        document = db.get(Document, document_id)
        document.metadata_json = {**(document.metadata_json or {}), "ready_for_query": False, "review_status": "in_review"}
        db.commit()

    with postgres_route_env["app_session_factory"]() as db:
        apply_request_context(
            db,
            type("Auth", (), {"role": Role.UNION_ADMIN.value, "union_id": seeded["union_one_id"], "user_id": None})(),
        )
        results = postgres_route_env["app"].state.platform.retrieval.search(
            db,
            union_id=seeded["union_one_id"],
            query="seniority rights",
            limit=5,
        )

        assert all(result.document_id != document_id for result in results)


def test_postgres_query_route_records_usage_and_warning_events(postgres_route_env):
    seeded = _seed_route_data(postgres_route_env["owner_session_factory"])
    client = TestClient(postgres_route_env["app"])

    with postgres_route_env["owner_session_factory"]() as db:
        quota = db.scalar(select(QuotaPolicy).where(QuotaPolicy.union_id == seeded["union_one_id"]))
        quota.requests_per_day = 2
        quota.warn_threshold_ratio = 0.5
        union = db.get(Union, seeded["union_one_id"])
        union.message_retention_enabled = True
        db.commit()

    headers = {
        "X-Auth-Email": "admin1@example.com",
        "X-Auth-Role": "union_admin",
        "X-Union-Slug": "local-1",
    }

    response = client.post(
        "/api/query",
        headers=headers,
        json={
            "question": "Summarize the grievance procedure.",
            "union_local_id": "local-1",
            "session_id": "query-session-1",
        },
    )

    assert response.status_code == 200
    assert response.headers["X-KARL-Quota-Warning"] == "true"

    with postgres_route_env["owner_session_factory"]() as db:
        usage_events = db.scalars(
            select(UsageEvent).where(UsageEvent.union_id == seeded["union_one_id"]).order_by(UsageEvent.created_at.asc())
        ).all()
        quota_warnings = db.scalars(
            select(SecurityEvent).where(
                SecurityEvent.union_id == seeded["union_one_id"],
                SecurityEvent.event_type == "quota_warning",
            )
        ).all()
        quota_notifications = db.scalars(
            select(Notification).where(
                Notification.union_id == seeded["union_one_id"],
                Notification.subject == "Union alert: quota_warning",
            )
        ).all()
        retained_chat = db.scalar(select(Chat).where(Chat.session_id == "query-session-1"))
        retained_messages = db.scalars(
            select(Message).where(Message.union_id == seeded["union_one_id"]).order_by(Message.created_at.asc())
        ).all()

        assert len(usage_events) == 2
        latest_usage = usage_events[-1]
        assert latest_usage.route == "/api/query"
        assert latest_usage.request_count == 1
        assert latest_usage.token_count >= 3
        assert latest_usage.metadata_json["status_code"] == 200

        assert len(quota_warnings) == 2
        assert len(quota_notifications) == 1
        assert retained_chat is not None
        assert retained_chat.union_id == seeded["union_one_id"]
        assert [message.role for message in retained_messages] == ["user", "assistant"]
        assert [message.content for message in retained_messages] == [
            "Summarize the grievance procedure.",
            "Safe stub answer.",
        ]


def test_postgres_query_route_blocks_when_quota_exceeded(postgres_route_env):
    seeded = _seed_route_data(postgres_route_env["owner_session_factory"])
    client = TestClient(postgres_route_env["app"])

    with postgres_route_env["owner_session_factory"]() as db:
        quota = db.scalar(select(QuotaPolicy).where(QuotaPolicy.union_id == seeded["union_one_id"]))
        quota.requests_per_day = 1
        quota.warn_threshold_ratio = 0.5
        db.commit()

    headers = {
        "X-Auth-Email": "admin1@example.com",
        "X-Auth-Role": "union_admin",
        "X-Union-Slug": "local-1",
    }

    response = client.post(
        "/api/query",
        headers=headers,
        json={
            "question": "Any contract question?",
            "union_local_id": "local-1",
            "session_id": "query-session-blocked",
        },
    )

    assert response.status_code == 429
    assert response.json()["detail"] == "Union daily request cap exceeded."

    with postgres_route_env["owner_session_factory"]() as db:
        usage_events = db.scalars(select(UsageEvent).where(UsageEvent.union_id == seeded["union_one_id"])).all()
        quota_exceeded = db.scalars(
            select(SecurityEvent).where(
                SecurityEvent.union_id == seeded["union_one_id"],
                SecurityEvent.event_type == "quota_exceeded",
            )
        ).all()
        quota_notifications = db.scalars(
            select(Notification).where(
                Notification.union_id == seeded["union_one_id"],
                Notification.subject == "Union alert: quota_exceeded",
            )
        ).all()
        blocked_chat = db.scalar(select(Chat).where(Chat.session_id == "query-session-blocked"))

        assert len(usage_events) == 1
        assert len(quota_exceeded) == 1
        assert len(quota_notifications) == 1
        assert blocked_chat is None


def test_postgres_ingestion_job_routes_are_tenant_scoped_and_retryable(postgres_route_env):
    seeded = _seed_route_data(postgres_route_env["owner_session_factory"])
    client = TestClient(postgres_route_env["app"])

    headers_one = {
        "X-Auth-Email": "admin1@example.com",
        "X-Auth-Role": "union_admin",
        "X-Union-Slug": "local-1",
    }
    headers_two = {
        "X-Auth-Email": "admin2@example.com",
        "X-Auth-Role": "union_admin",
        "X-Union-Slug": "local-2",
    }

    upload = client.post(
        f"/api/admin/unions/{seeded['union_one_id']}/documents",
        headers=headers_one,
        files={"file": ("scan.pdf", b"%PDF-1.4 fake pdf", "application/pdf")},
    )
    assert upload.status_code == 200
    uploaded = upload.json()

    listed = client.get(f"/api/admin/unions/{seeded['union_one_id']}/ingestion-jobs", headers=headers_one)
    detail = client.get(
        f"/api/admin/unions/{seeded['union_one_id']}/ingestion-jobs/{uploaded['ingestion_job_id']}",
        headers=headers_one,
    )
    blocked = client.get(
        f"/api/admin/unions/{seeded['union_two_id']}/ingestion-jobs/{uploaded['ingestion_job_id']}",
        headers=headers_two,
    )
    retry = client.post(
        f"/api/admin/unions/{seeded['union_one_id']}/ingestion-jobs/{uploaded['ingestion_job_id']}/retry",
        headers=headers_one,
        json={"ocr_enabled": True},
    )

    assert listed.status_code == 200
    assert detail.status_code == 200
    assert blocked.status_code == 404
    assert retry.status_code == 200

    items = listed.json()["items"]
    assert len(items) == 1
    assert items[0]["status"] == IngestionJobStatus.PENDING.value
    assert items[0]["estimated_ready_seconds"] is not None

    retry_job = retry.json()["job"]
    assert retry_job["metadata"]["trigger"] == "retry"
    assert retry_job["metadata"]["source_job_id"] == uploaded["ingestion_job_id"]
    assert retry_job["metadata"]["ocr_enabled"] is True

    with postgres_route_env["owner_session_factory"]() as db:
        jobs = db.scalars(
            select(IngestionJob).where(IngestionJob.union_id == seeded["union_one_id"]).order_by(IngestionJob.created_at.asc())
        ).all()
        assert len(jobs) == 2
        assert jobs[0].metadata_json["trigger"] == "upload"
        assert jobs[1].metadata_json["trigger"] == "retry"


def test_postgres_document_quality_state_and_review_escalation_are_scoped(postgres_route_env):
    seeded = _seed_route_data(postgres_route_env["owner_session_factory"])
    client = TestClient(postgres_route_env["app"])

    with postgres_route_env["owner_session_factory"]() as db:
        user = db.scalar(select(User).where(User.email == "admin1@example.com"))
        document = Document(
            union_id=seeded["union_one_id"],
            uploaded_by_user_id=user.id,
            title="scan.pdf",
            storage_key="local-1/scan.pdf",
            content_type="application/pdf",
            bytes_size=1234,
            status=DocumentStatus.FAILED,
            metadata_json={
                "quality_status": "needs_review",
                "quality_reason": "no_text_after_ocr",
                "ocr_status": "attempted_needs_review",
                "scan_likelihood": "high",
                "recommended_action": "manual_review_after_ocr",
                "review_status": "needs_review",
                "ready_for_query": False,
            },
        )
        db.add(document)
        db.flush()
        job = IngestionJob(
            union_id=seeded["union_one_id"],
            document_id=document.id,
            requested_by_user_id=user.id,
            status=IngestionJobStatus.SUCCEEDED,
            metadata_json={
                "trigger": "upload",
                "quality_status": "needs_review",
                "quality_reason": "no_text_after_ocr",
                "ocr_status": "attempted_needs_review",
                "scan_likelihood": "high",
                "recommended_action": "manual_review_after_ocr",
                "review_status": "needs_review",
            },
        )
        db.add(job)
        db.commit()
        job_id = job.id
        document_id = document.id

    headers_one = {
        "X-Auth-Email": "admin1@example.com",
        "X-Auth-Role": "union_admin",
        "X-Union-Slug": "local-1",
    }
    headers_two = {
        "X-Auth-Email": "admin2@example.com",
        "X-Auth-Role": "union_admin",
        "X-Union-Slug": "local-2",
    }

    documents = client.get(f"/api/admin/unions/{seeded['union_one_id']}/documents", headers=headers_one)
    blocked = client.post(
        f"/api/admin/unions/{seeded['union_two_id']}/ingestion-jobs/{job_id}/escalate-review",
        headers=headers_two,
        json={"note": "cross-tenant"},
    )
    escalate = client.post(
        f"/api/admin/unions/{seeded['union_one_id']}/ingestion-jobs/{job_id}/escalate-review",
        headers=headers_one,
        json={"note": "Needs human review."},
    )

    assert documents.status_code == 200
    assert blocked.status_code == 404
    assert escalate.status_code == 200

    item = next(doc for doc in documents.json()["items"] if doc["id"] == document_id)
    assert item["quality_status"] == "needs_review"
    assert item["quality_reason"] == "no_text_after_ocr"
    assert item["ocr_status"] == "attempted_needs_review"
    assert item["scan_likelihood"] == "high"
    assert item["ready_for_query"] is False
    assert item["review_status"] == "needs_review"
    assert item["latest_ingestion_job"]["id"] == job_id

    with postgres_route_env["owner_session_factory"]() as db:
        security_events = db.scalars(
            select(SecurityEvent).where(
                SecurityEvent.union_id == seeded["union_one_id"],
                SecurityEvent.event_type == "ingestion_review_escalated",
            )
        ).all()
        notifications = db.scalars(
            select(Notification).where(Notification.subject == "Security alert: ingestion_review_escalated")
        ).all()
        job = db.get(IngestionJob, job_id)

        assert len(security_events) == 1
        assert len(notifications) == 1
        assert job.metadata_json["escalated_for_review"] is True


def test_postgres_ingestion_review_state_update_is_tenant_scoped(postgres_route_env):
    seeded = _seed_route_data(postgres_route_env["owner_session_factory"])
    client = TestClient(postgres_route_env["app"])

    with postgres_route_env["owner_session_factory"]() as db:
        user = db.scalar(select(User).where(User.email == "admin1@example.com"))
        document = Document(
            union_id=seeded["union_one_id"],
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
            union_id=seeded["union_one_id"],
            document_id=document.id,
            requested_by_user_id=user.id,
            status=IngestionJobStatus.SUCCEEDED,
            metadata_json={"quality_status": "needs_review", "review_status": "needs_review"},
        )
        db.add(job)
        db.commit()
        job_id = job.id
        document_id = document.id

    headers_one = {
        "X-Auth-Email": "admin1@example.com",
        "X-Auth-Role": "union_admin",
        "X-Union-Slug": "local-1",
    }
    headers_two = {
        "X-Auth-Email": "admin2@example.com",
        "X-Auth-Role": "union_admin",
        "X-Union-Slug": "local-2",
    }

    blocked = client.post(
        f"/api/admin/unions/{seeded['union_two_id']}/ingestion-jobs/{job_id}/review-state",
        headers=headers_two,
        json={"review_status": "in_review", "note": "cross-tenant"},
    )
    updated = client.post(
        f"/api/admin/unions/{seeded['union_one_id']}/ingestion-jobs/{job_id}/review-state",
        headers=headers_one,
        json={"review_status": "resolved", "note": "Manual review completed"},
    )

    assert blocked.status_code == 404
    assert updated.status_code == 200
    assert updated.json()["review_status"] == "resolved"

    with postgres_route_env["owner_session_factory"]() as db:
        document = db.get(Document, document_id)
        job = db.get(IngestionJob, job_id)
        assert document.metadata_json["review_status"] == "resolved"
        assert job.metadata_json["review_status"] == "resolved"
