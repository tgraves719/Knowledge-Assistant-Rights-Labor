import asyncio
import base64
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend import api
from backend.platform.auth import HeaderAuthAdapter
from backend.platform.chat_history import ChatHistoryStore
from backend.platform.crypto import SecretCipher
from backend.platform.db import Base
from backend.platform.guardrails import GuardrailService
from backend.platform.ingestion import IngestionService
from backend.platform.local_auth import LocalAuthService
from backend.platform.models import ChunkEmbedding, Document, DocumentStatus, IngestionJob, IngestionJobStatus, Role, Union, UnionMembership, User
from backend.platform.parsing import ParserRegistry, PlainTextDocumentParser
from backend.platform.quotas import QuotaService
from backend.platform.retrieval import TenantRetrievalService
from backend.platform.session_auth import SessionAuthService
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
    )


def _build_platform(tmp_path: Path) -> ServiceContainer:
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
    parsers = ParserRegistry([PlainTextDocumentParser()])
    sentinel = SentinelService(settings)
    local_auth = LocalAuthService(secret_key=settings.secret_encryption_key, token_ttl_seconds=settings.local_auth_token_ttl_seconds)
    session_auth = SessionAuthService(
        secret_key=settings.secret_encryption_key,
        cookie_name=settings.session_cookie_name,
        member_idle_seconds=settings.member_session_idle_seconds,
        union_admin_idle_seconds=settings.union_admin_session_idle_seconds,
        super_admin_idle_seconds=settings.super_admin_session_idle_seconds,
    )
    return ServiceContainer(
        settings=settings,
        engine=engine,
        session_factory=SessionLocal,
        auth_adapter=HeaderAuthAdapter(settings, local_auth=local_auth, session_auth=session_auth),
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
            ocr_auto_retry_enabled=settings.ocr_auto_retry_enabled,
            ocr_auto_retry_max_attempts=settings.ocr_auto_retry_max_attempts,
        ),
        local_auth=local_auth,
        session_auth=session_auth,
    )


def test_demo_union_admin_can_login_upload_and_query_uploaded_documents(tmp_path, monkeypatch):
    platform = _build_platform(tmp_path)
    prior_platform = getattr(api.app.state, "platform", None)
    prior_retriever = api.retriever
    api.app.state.platform = platform
    api.retriever = None
    monkeypatch.setattr(
        api,
        "generate_response",
        lambda question, system_prompt, chunks, union_local_id=None, **kwargs: asyncio.sleep(
            0,
            result=(
                "The uploaded notice says the bake sale starts at 9 AM in the union hall. Sources: bake_sale.txt, page 1, chunk 1",
                {"provider": "test"},
            ),
        ),
    )

    try:
        with platform.session_factory() as db:
            union = Union(slug="demo-local", name="Demo Local", union_local_id="demo-local")
            user = User(email="union_demo@example.com", full_name="Union Demo Admin")
            db.add_all([union, user])
            db.flush()
            db.add(UnionMembership(union_id=union.id, user_id=user.id, role=Role.UNION_ADMIN))
            platform.local_auth.create_or_update_credential(
                db,
                user=user,
                username="union_demo",
                password="demo_password",
            )
            db.commit()
            union_id = union.id

        client = TestClient(api.app)
        login = client.post(
            "/api/auth/local/login",
            json={"username": "union_demo", "password": "demo_password", "union_slug": "demo-local"},
        )
        assert login.status_code == 200
        token = login.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        upload = client.post(
            f"/api/admin/unions/{union_id}/documents",
            headers=headers,
            files={"file": ("bake_sale.txt", b"The bake sale starts at 9 AM in the union hall.", "text/plain")},
        )
        assert upload.status_code == 200
        upload_payload = upload.json()
        assert upload_payload["ingestion_status"] == "succeeded"

        query = client.post(
            "/api/query",
            headers=headers,
            json={
                "question": "When does the bake sale start?",
                "union_local_id": "demo-local",
                "contract_id": "tenant-upload",
                "contract_version": "current",
                "session_id": "demo-session",
            },
        )
        assert query.status_code == 200
        payload = query.json()
        assert payload["retrieval_strategy"] == "platform_tenant_documents"
        assert "bake sale starts at 9 am" in payload["answer"].lower()
        assert any("bake_sale.txt, page 1, chunk 1" in citation for citation in payload["citations"])
        assert payload["sources"]
        assert payload["sources"][0]["document_content_url"].endswith("/api/member/documents/" + payload["sources"][0]["document_id"] + "/content")
        assert payload["sources"][0]["document_access_url"]
        assert "/selection?" in payload["sources"][0]["document_selection_url"]
        assert payload["sources"][0]["excerpt"]

        signed_content = client.get(payload["sources"][0]["document_access_url"])
        assert signed_content.status_code == 200
        assert signed_content.text == "The bake sale starts at 9 AM in the union hall."

        selection = client.get(payload["sources"][0]["document_selection_url"])
        assert selection.status_code == 200
        assert "bake sale starts at 9 AM".lower() in selection.json()["excerpt"].lower()

        signed_content_with_bad_auth = client.get(
            payload["sources"][0]["document_access_url"],
            headers={"Authorization": "Bearer definitely-invalid"},
        )
        assert signed_content_with_bad_auth.status_code == 200
        assert signed_content_with_bad_auth.text == "The bake sale starts at 9 AM in the union hall."
    finally:
        api.app.state.platform = prior_platform
        api.retriever = prior_retriever


def test_approved_sensitive_document_remains_queryable_for_member_flow(tmp_path, monkeypatch):
    platform = _build_platform(tmp_path)
    prior_platform = getattr(api.app.state, "platform", None)
    prior_retriever = api.retriever
    api.app.state.platform = platform
    api.retriever = None
    monkeypatch.setattr(
        api,
        "generate_response",
        lambda question, system_prompt, chunks, union_local_id=None, **kwargs: asyncio.sleep(
            0,
            result=("Vacation sign-up happens through the posted scheduling process.", {"provider": "test"}),
        ),
    )

    try:
        with platform.session_factory() as db:
            union = Union(slug="demo-local", name="Demo Local", union_local_id="demo-local")
            user = User(email="union_demo@example.com", full_name="Union Demo Admin")
            db.add_all([union, user])
            db.flush()
            db.add(UnionMembership(union_id=union.id, user_id=user.id, role=Role.UNION_ADMIN))
            platform.local_auth.create_or_update_credential(
                db,
                user=user,
                username="union_demo",
                password="demo_password",
            )
            document = Document(
                union_id=union.id,
                uploaded_by_user_id=user.id,
                title="vacation_notice.txt",
                storage_key="demo-local/vacation_notice.txt",
                content_type="text/plain",
                bytes_size=120,
                status=DocumentStatus.ACTIVE,
                metadata_json={
                    "ready_for_query": True,
                    "review_status": "needs_review",
                    "safety_status": "flagged_sensitive_data",
                    "safety_review_status": "needs_review",
                    "sensitive_data_risk": True,
                    "member_visible": True,
                },
            )
            db.add(document)
            db.flush()
            db.add(
                ChunkEmbedding(
                    union_id=union.id,
                    document_id=document.id,
                    chunk_index=0,
                    chunk_text="Vacation sign-up happens through the posted scheduling process.",
                    metadata_json={
                        "document_title": document.title,
                        "page_number": 1,
                        "sensitive_data_risk": True,
                        "member_visible": True,
                        "safety_status": "flagged_sensitive_data",
                    },
                )
            )
            job = IngestionJob(
                union_id=union.id,
                document_id=document.id,
                requested_by_user_id=user.id,
                status=IngestionJobStatus.SUCCEEDED,
                metadata_json={
                    "review_status": "needs_review",
                    "safety_review_status": "needs_review",
                },
            )
            db.add(job)
            db.commit()
            union_id = union.id
            document_id = document.id

        client = TestClient(api.app)
        login = client.post(
            "/api/auth/local/login",
            json={"username": "union_demo", "password": "demo_password", "union_slug": "demo-local"},
        )
        assert login.status_code == 200
        token = login.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        approve = client.post(
            f"/api/admin/unions/{union_id}/documents/{document_id}/safety-review",
            headers=headers,
            json={"decision": "approve_member_access", "note": "Reviewed and approved."},
        )
        assert approve.status_code == 200

        query = client.post(
            "/api/query",
            headers=headers,
            json={
                "question": "How do I sign up for vacation?",
                "union_local_id": "demo-local",
                "contract_id": "tenant-upload",
                "contract_version": "current",
                "session_id": "demo-session-approved",
            },
        )
        assert query.status_code == 200
        payload = query.json()
        assert "vacation sign-up happens through the posted scheduling process" in payload["answer"].lower()
        assert payload["sources"]
    finally:
        api.app.state.platform = prior_platform
        api.retriever = prior_retriever


def test_query_redaction_response_recomputes_content_length(tmp_path, monkeypatch):
    platform = _build_platform(tmp_path)
    prior_platform = getattr(api.app.state, "platform", None)
    prior_retriever = api.retriever
    api.app.state.platform = platform
    api.retriever = None
    monkeypatch.setattr(
        api,
        "generate_response",
        lambda question, system_prompt, chunks, union_local_id=None, **kwargs: asyncio.sleep(
            0,
            result=("Member SSN 123-45-6789 and email jane@example.com.", {"provider": "test"}),
        ),
    )

    try:
        with platform.session_factory() as db:
            union = Union(slug="demo-local", name="Demo Local", union_local_id="demo-local")
            user = User(email="union_demo@example.com", full_name="Union Demo Admin")
            db.add_all([union, user])
            db.flush()
            db.add(UnionMembership(union_id=union.id, user_id=user.id, role=Role.UNION_ADMIN))
            platform.local_auth.create_or_update_credential(
                db,
                user=user,
                username="union_demo",
                password="demo_password",
            )
            document = Document(
                union_id=union.id,
                uploaded_by_user_id=user.id,
                title="member_notice.txt",
                storage_key="demo-local/member_notice.txt",
                content_type="text/plain",
                bytes_size=120,
                status=DocumentStatus.ACTIVE,
                metadata_json={"ready_for_query": True, "review_status": "not_required"},
            )
            db.add(document)
            db.flush()
            platform.retrieval.ingest_document(
                db,
                union_id=union.id,
                document_id=document.id,
                text="Member SSN 123-45-6789 and email jane@example.com.",
                pages=[{"page_number": 1, "text": "Member SSN 123-45-6789 and email jane@example.com."}],
                metadata={"document_title": document.title},
            )
            db.commit()

        client = TestClient(api.app)
        login = client.post(
            "/api/auth/local/login",
            json={"username": "union_demo", "password": "demo_password", "union_slug": "demo-local"},
        )
        assert login.status_code == 200
        token = login.json()["access_token"]

        response = client.post(
            "/api/query",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "question": "What contact details are in the notice?",
                "union_local_id": "demo-local",
                "contract_id": "tenant-upload",
                "contract_version": "current",
                "session_id": "demo-session-redaction",
            },
        )
        assert response.status_code == 200
        assert int(response.headers["content-length"]) == len(response.content)
        payload = response.json()
        assert "answer" in payload
    finally:
        api.app.state.platform = prior_platform
        api.retriever = prior_retriever


def test_demo_union_admin_session_login_sets_cookie_and_logout_clears_it(tmp_path):
    platform = _build_platform(tmp_path)
    prior_platform = getattr(api.app.state, "platform", None)
    api.app.state.platform = platform

    try:
        with platform.session_factory() as db:
            union = Union(slug="demo-local", name="Demo Local", union_local_id="demo-local")
            user = User(email="union_demo@example.com", full_name="Union Demo Admin")
            db.add_all([union, user])
            db.flush()
            db.add(UnionMembership(union_id=union.id, user_id=user.id, role=Role.UNION_ADMIN))
            platform.local_auth.create_or_update_credential(
                db,
                user=user,
                username="union_demo",
                password="demo_password",
            )
            db.commit()

        client = TestClient(api.app)
        login = client.post(
            "/api/auth/session/login",
            headers={"X-Tenant-Slug": "demo-local"},
            json={"username": "union_demo", "password": "demo_password"},
        )
        assert login.status_code == 200
        assert platform.session_auth.cookie_name in login.cookies

        me = client.get("/api/auth/session/me", headers={"X-Tenant-Slug": "demo-local"})
        assert me.status_code == 200
        assert me.json()["authenticated"] is True
        assert me.json()["role"] == Role.UNION_ADMIN.value

        logout = client.post("/api/auth/session/logout", headers={"X-Tenant-Slug": "demo-local"})
        assert logout.status_code == 200

        me_after = client.get("/api/auth/session/me", headers={"X-Tenant-Slug": "demo-local"})
        assert me_after.status_code == 200
        assert me_after.json()["authenticated"] is False
    finally:
        api.app.state.platform = prior_platform


def test_tenant_routes_and_bootstrap_resolve_demo_union(tmp_path):
    platform = _build_platform(tmp_path)
    prior_platform = getattr(api.app.state, "platform", None)
    api.app.state.platform = platform

    try:
        with platform.session_factory() as db:
            union = Union(
                slug="demo-local",
                name="Demo Local",
                union_local_id="demo-local",
                metadata_json={"auth_policy": {"member_login_required": True}, "branding": {"theme_color": "#123456"}},
            )
            db.add(union)
            db.commit()

        client = TestClient(api.app)
        bootstrap = client.get("/api/tenant/demo-local/bootstrap?page_mode=member")
        assert bootstrap.status_code == 200
        payload = bootstrap.json()
        assert payload["union"]["slug"] == "demo-local"
        assert payload["auth_policy"]["member_login_required"] is True
        assert payload["branding"]["theme_color"] == "#123456"

        member_page = client.get("/u/demo-local/")
        assert member_page.status_code == 200
        assert "karl-member-widget" in member_page.text
        assert "/static/embed/karl-member.js" in member_page.text

        member_frame = client.get("/embed/member-frame/demo-local")
        assert member_frame.status_code == 200
        assert "window.__KARL_ROUTE_CONTEXT__" in member_frame.text
        assert '"unionSlug": "demo-local"' in member_frame.text or '"unionSlug":"demo-local"' in member_frame.text
        assert "X-Frame-Options" not in member_frame.headers
        assert member_frame.headers.get("Content-Security-Policy") == "frame-ancestors *;"

        admin_page = client.get("/u/demo-local/admin")
        assert admin_page.status_code == 200
        assert admin_page.headers.get("X-Frame-Options") == "DENY"

        superadmin_page = client.get("/karl/")
        assert superadmin_page.status_code == 200

        embed_demo = client.get("/embed/member-demo/demo-local")
        assert embed_demo.status_code == 200
        assert "karl-member-widget" in embed_demo.text

        embed_script = client.get("/static/embed/karl-member.js")
        assert embed_script.status_code == 200
        assert "customElements.define('karl-member-widget'" in embed_script.text
    finally:
        api.app.state.platform = prior_platform


def test_tenant_query_without_ready_documents_returns_readiness_message(tmp_path):
    platform = _build_platform(tmp_path)
    prior_platform = getattr(api.app.state, "platform", None)
    prior_retriever = api.retriever
    api.app.state.platform = platform
    api.retriever = None

    try:
        with platform.session_factory() as db:
            union = Union(slug="demo-local", name="Demo Local", union_local_id="demo-local")
            user = User(email="union_member@example.com", full_name="Union Demo Member")
            db.add_all([union, user])
            db.flush()
            db.add(UnionMembership(union_id=union.id, user_id=user.id, role=Role.USER))
            platform.local_auth.create_or_update_credential(
                db,
                user=user,
                username="union_member",
                password="demo_password",
            )
            db.commit()

        client = TestClient(api.app)
        login = client.post(
            "/api/auth/local/login",
            headers={"X-Tenant-Slug": "demo-local"},
            json={"username": "union_member", "password": "demo_password"},
        )
        assert login.status_code == 200
        token = login.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}", "X-Tenant-Slug": "demo-local"}

        query = client.post(
            "/api/query",
            headers=headers,
            json={
                "question": "When does the bake sale start?",
                "union_local_id": "demo-local",
                "contract_id": "tenant-upload",
                "contract_version": "current",
                "session_id": "demo-session",
            },
        )
        assert query.status_code == 409
        assert "No uploaded union documents are ready yet" in query.json()["detail"]
    finally:
        api.app.state.platform = prior_platform
        api.retriever = prior_retriever


def test_member_can_register_account_for_tenant_session(tmp_path):
    platform = _build_platform(tmp_path)
    prior_platform = getattr(api.app.state, "platform", None)
    api.app.state.platform = platform

    try:
        with platform.session_factory() as db:
            union = Union(slug="demo-local", name="Demo Local", union_local_id="demo-local")
            db.add(union)
            db.commit()
            union_id = union.id

        client = TestClient(api.app)
        register = client.post(
            "/api/auth/session/register",
            headers={"X-Tenant-Slug": "demo-local"},
            json={
                "username": "fresh_member",
                "password": "demo_password",
                "email": "fresh_member@example.com",
                "full_name": "Fresh Member",
            },
        )
        assert register.status_code == 200
        assert register.json()["authenticated"] is True
        assert register.json()["user"]["role"] == Role.USER.value
        assert platform.session_auth.cookie_name in register.cookies

        me = client.get("/api/auth/session/me", headers={"X-Tenant-Slug": "demo-local"})
        assert me.status_code == 200
        assert me.json()["authenticated"] is True
        assert me.json()["role"] == Role.USER.value

        with platform.session_factory() as db:
            member = db.scalar(select(User).where(User.email == "fresh_member@example.com"))
            assert member is not None
            membership = db.scalar(
                select(UnionMembership).where(
                    UnionMembership.user_id == member.id,
                    UnionMembership.union_id == union_id,
                )
            )
            assert membership is not None
            assert membership.role == Role.USER
    finally:
        api.app.state.platform = prior_platform


def test_member_can_open_same_union_uploaded_document_content(tmp_path):
    platform = _build_platform(tmp_path)
    prior_platform = getattr(api.app.state, "platform", None)
    api.app.state.platform = platform

    try:
        with platform.session_factory() as db:
            union = Union(slug="demo-local", name="Demo Local", union_local_id="demo-local")
            admin = User(email="union_demo@example.com", full_name="Union Demo Admin")
            member = User(email="union_member@example.com", full_name="Union Demo Member")
            db.add_all([union, admin, member])
            db.flush()
            db.add(UnionMembership(union_id=union.id, user_id=admin.id, role=Role.UNION_ADMIN))
            db.add(UnionMembership(union_id=union.id, user_id=member.id, role=Role.USER))
            platform.local_auth.create_or_update_credential(db, user=member, username="union_member", password="demo_password")
            result = platform.ingestion.register_upload(
                db,
                union=union,
                uploaded_by_user_id=admin.id,
                filename="bake_sale.txt",
                content_type="text/plain",
                payload=b"The bake sale starts at 9 AM in the union hall.",
                storage_key=platform.storage.save_bytes(union.slug, "bake_sale.txt", b"The bake sale starts at 9 AM in the union hall.").key,
            )
            db.commit()
            document_id = result.document.id

        client = TestClient(api.app)
        login = client.post(
            "/api/auth/local/login",
            headers={"X-Tenant-Slug": "demo-local"},
            json={"username": "union_member", "password": "demo_password"},
        )
        assert login.status_code == 200
        token = login.json()["access_token"]

        document = client.get(
            f"/api/member/documents/{document_id}/content",
            headers={"Authorization": f"Bearer {token}", "X-Tenant-Slug": "demo-local"},
        )
        assert document.status_code == 200
        assert "text/plain" in str(document.headers.get("content-type") or "")
        assert b"bake sale starts at 9 AM" in document.content

        signed_document = client.get(
            f"/api/member/documents/{document_id}/content?access_token={token}",
            headers={"X-Tenant-Slug": "demo-local"},
        )
        assert signed_document.status_code == 200
        assert b"bake sale starts at 9 AM" in signed_document.content
    finally:
        api.app.state.platform = prior_platform


def test_member_selection_redacts_sensitive_content_and_blocks_full_document(tmp_path):
    platform = _build_platform(tmp_path)
    prior_platform = getattr(api.app.state, "platform", None)
    api.app.state.platform = platform

    try:
        with platform.session_factory() as db:
            union = Union(slug="demo-local", name="Demo Local", union_local_id="demo-local")
            admin = User(email="union_demo@example.com", full_name="Union Demo Admin")
            member = User(email="union_member@example.com", full_name="Union Demo Member")
            db.add_all([union, admin, member])
            db.flush()
            db.add(UnionMembership(union_id=union.id, user_id=admin.id, role=Role.UNION_ADMIN))
            db.add(UnionMembership(union_id=union.id, user_id=member.id, role=Role.USER))
            platform.local_auth.create_or_update_credential(db, user=member, username="union_member", password="demo_password")
            result = platform.ingestion.register_upload(
                db,
                union=union,
                uploaded_by_user_id=admin.id,
                filename="member_notice.txt",
                content_type="text/plain",
                payload=b"Member SSN 123-45-6789. Contact jane@example.com.",
                storage_key=platform.storage.save_bytes(union.slug, "member_notice.txt", b"Member SSN 123-45-6789. Contact jane@example.com.").key,
            )
            result.document.metadata_json = {
                **(result.document.metadata_json or {}),
                "ready_for_query": True,
                "review_status": "needs_review",
                "sensitive_data_risk": True,
                "member_visible": True,
            }
            platform.retrieval.delete_document(db, document_id=result.document.id)
            platform.retrieval.ingest_document(
                db,
                union_id=union.id,
                document_id=result.document.id,
                text="Member SSN 123-45-6789. Contact jane@example.com.",
                metadata={"document_title": result.document.title, "sensitive_data_risk": True, "member_visible": True},
            )
            db.commit()
            document_id = result.document.id

        client = TestClient(api.app)
        login = client.post(
            "/api/auth/local/login",
            headers={"X-Tenant-Slug": "demo-local"},
            json={"username": "union_member", "password": "demo_password"},
        )
        assert login.status_code == 200
        token = login.json()["access_token"]

        selection = client.get(
            f"/api/member/documents/{document_id}/selection?access_token={token}",
            headers={"X-Tenant-Slug": "demo-local"},
        )
        assert selection.status_code == 200
        assert "***-**-6789" in selection.json()["excerpt"]
        assert "123-45-6789" not in selection.json()["excerpt"]
        assert selection.json()["safety_redacted"] is True

        document = client.get(
            f"/api/member/documents/{document_id}/content?access_token={token}",
            headers={"X-Tenant-Slug": "demo-local"},
        )
        assert document.status_code == 409
        assert "sensitive data" in document.json()["detail"].lower()
    finally:
        api.app.state.platform = prior_platform
