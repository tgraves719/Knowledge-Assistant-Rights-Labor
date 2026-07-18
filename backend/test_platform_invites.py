"""End-to-end coverage for the QR invite-code enrollment flow (M2)."""

from __future__ import annotations

import base64
from datetime import datetime, timedelta
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
from backend.platform.models import (
    AuthSession,
    InviteCode,
    Role,
    TelemetryEvent,
    Union,
    UnionMembership,
    User,
)
from backend.platform.parsing import ParserRegistry, PlainTextDocumentParser
from backend.platform.quotas import QuotaService
from backend.platform.retrieval import TenantRetrievalService
from backend.platform.sentinel import SentinelService
from backend.platform.service_container import ServiceContainer
from backend.platform.session_auth import SessionAuthService
from backend.platform.settings import PlatformSettings


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
    storage_root = settings.local_storage_root
    from backend.platform.storage import LocalDiskStorage

    storage = LocalDiskStorage(storage_root)
    retrieval = TenantRetrievalService()
    parsers = ParserRegistry([PlainTextDocumentParser()])
    sentinel = SentinelService(settings)
    local_auth = LocalAuthService(
        secret_key=settings.secret_encryption_key,
        token_ttl_seconds=settings.local_auth_token_ttl_seconds,
    )
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


def _seed_union_with_admin(platform: ServiceContainer) -> tuple[str, str]:
    with platform.session_factory() as db:
        union = Union(slug="ufcw-7", name="UFCW Local 7", union_local_id="ufcw-7")
        admin = User(email="admin@ufcw7.example", full_name="Union Admin")
        db.add_all([union, admin])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=admin.id, role=Role.UNION_ADMIN))
        platform.local_auth.create_or_update_credential(
            db,
            user=admin,
            username="ufcw_admin",
            password="admin_password",
        )
        db.commit()
        return union.id, union.slug


def _with_app(platform: ServiceContainer):
    prior = getattr(api.app.state, "platform", None)
    api.app.state.platform = platform
    return prior


def _admin_client(platform: ServiceContainer) -> TestClient:
    client = TestClient(api.app)
    login = client.post(
        "/api/auth/session/login",
        json={"username": "ufcw_admin", "password": "admin_password", "union_slug": "ufcw-7"},
    )
    assert login.status_code == 200, login.text
    return client


def test_invite_create_join_and_attribution(tmp_path):
    platform = _build_platform(tmp_path)
    prior = _with_app(platform)
    try:
        union_id, union_slug = _seed_union_with_admin(platform)
        client = _admin_client(platform)

        created = client.post(
            f"/api/admin/unions/{union_id}/invites",
            json={"label": "Pueblo West break room poster", "max_uses": 5},
        )
        assert created.status_code == 200, created.text
        invite = created.json()
        assert invite["status"] == "active"
        assert invite["join_path"] == f"/j/{invite['code']}"
        code = invite["code"]

        # Public info endpoint works without auth.
        anon = TestClient(api.app)
        info = anon.get(f"/api/auth/join/{code}")
        assert info.status_code == 200, info.text
        assert info.json()["union"]["slug"] == union_slug

        # Landing page renders with union branding.
        page = anon.get(f"/j/{code}")
        assert page.status_code == 200
        assert "UFCW Local 7" in page.text
        assert code in page.text

        # Member joins through the code.
        join = anon.post(
            "/api/auth/session/join",
            json={
                "code": code,
                "full_name": "Pat Member",
                "email": "pat@example.com",
                "username": "pat_member",
                "password": "longenough1",
            },
        )
        assert join.status_code == 200, join.text
        body = join.json()
        assert body["authenticated"] is True
        assert body["user"]["union_slug"] == union_slug
        assert body["invite"]["code"] == code
        assert anon.cookies.get(platform.settings.session_cookie_name)

        with platform.session_factory() as db:
            stored = db.scalar(select(InviteCode).where(InviteCode.code == code))
            assert stored.use_count == 1
            session = db.scalar(
                select(AuthSession).where(AuthSession.invite_code_id == stored.id)
            )
            assert session is not None
            member = db.scalar(select(User).where(User.email == "pat@example.com"))
            assert member is not None
            membership = db.scalar(
                select(UnionMembership).where(
                    UnionMembership.user_id == member.id,
                    UnionMembership.union_id == union_id,
                )
            )
            assert membership is not None and membership.role == Role.USER
            event = db.scalar(
                select(TelemetryEvent).where(TelemetryEvent.event_type == "session_join_success")
            )
            assert event is not None
            assert (event.metadata_json or {}).get("invite_code") == code

        # Joined member can hit an authed endpoint with the cookie session.
        me = anon.get("/api/auth/session/me")
        assert me.status_code == 200
        assert me.json()["authenticated"] is True
        assert me.json()["email"] == "pat@example.com"
    finally:
        api.app.state.platform = prior


def test_invite_revocation_and_limits_close_the_door(tmp_path):
    platform = _build_platform(tmp_path)
    prior = _with_app(platform)
    try:
        union_id, _ = _seed_union_with_admin(platform)
        client = _admin_client(platform)

        # Capped invite: one use only.
        capped = client.post(
            f"/api/admin/unions/{union_id}/invites",
            json={"label": "steward hand-card", "max_uses": 1},
        ).json()
        anon = TestClient(api.app)
        first = anon.post(
            "/api/auth/session/join",
            json={
                "code": capped["code"],
                "full_name": "First Member",
                "email": "first@example.com",
                "username": "first_member",
                "password": "longenough1",
            },
        )
        assert first.status_code == 200, first.text
        second = TestClient(api.app).post(
            "/api/auth/session/join",
            json={
                "code": capped["code"],
                "full_name": "Second Member",
                "email": "second@example.com",
                "username": "second_member",
                "password": "longenough1",
            },
        )
        assert second.status_code == 410
        assert TestClient(api.app).get(f"/api/auth/join/{capped['code']}").status_code == 410

        # Expired invite.
        expired = client.post(
            f"/api/admin/unions/{union_id}/invites",
            json={"label": "old poster", "expires_at": (datetime.utcnow() - timedelta(days=1)).isoformat()},
        ).json()
        assert TestClient(api.app).get(f"/api/auth/join/{expired['code']}").status_code == 410

        # Revocation closes an active code and the landing page reflects it.
        active = client.post(
            f"/api/admin/unions/{union_id}/invites",
            json={"label": "union board"},
        ).json()
        revoked = client.post(f"/api/admin/unions/{union_id}/invites/{active['id']}/revoke")
        assert revoked.status_code == 200
        assert revoked.json()["status"] == "revoked"
        assert TestClient(api.app).get(f"/api/auth/join/{active['code']}").status_code == 410
        closed_page = TestClient(api.app).get(f"/j/{active['code']}")
        assert closed_page.status_code == 404
        assert "no longer open" in closed_page.text

        # Unknown code: 404 info, not-recognized page.
        assert TestClient(api.app).get("/api/auth/join/nosuchcode").status_code == 404
        missing_page = TestClient(api.app).get("/j/nosuchcode")
        assert missing_page.status_code == 404
        assert "not recognized" in missing_page.text

        # Admin list shows all codes with counts.
        listing = client.get(f"/api/admin/unions/{union_id}/invites")
        assert listing.status_code == 200
        items = listing.json()["items"]
        assert len(items) == 3
        by_label = {item["label"]: item for item in items}
        assert by_label["steward hand-card"]["use_count"] == 1
        assert by_label["steward hand-card"]["status"] == "exhausted"
        assert by_label["union board"]["status"] == "revoked"
    finally:
        api.app.state.platform = prior


def test_invite_admin_endpoints_enforce_union_scope(tmp_path):
    platform = _build_platform(tmp_path)
    prior = _with_app(platform)
    try:
        union_id, _ = _seed_union_with_admin(platform)
        with platform.session_factory() as db:
            other_union = Union(slug="other-local", name="Other Local", union_local_id="other-local")
            db.add(other_union)
            db.commit()
            other_union_id = other_union.id

        client = _admin_client(platform)
        cross = client.post(
            f"/api/admin/unions/{other_union_id}/invites",
            json={"label": "should not work"},
        )
        assert cross.status_code == 403

        # Anonymous users cannot manage invites at all.
        anon = TestClient(api.app)
        assert anon.get(f"/api/admin/unions/{union_id}/invites").status_code == 401
        assert (
            anon.post(f"/api/admin/unions/{union_id}/invites", json={"label": "x"}).status_code
            == 401
        )
    finally:
        api.app.state.platform = prior


def test_guest_join_is_zero_friction_and_disconnectable(tmp_path):
    platform = _build_platform(tmp_path)
    prior = _with_app(platform)
    try:
        union_id, union_slug = _seed_union_with_admin(platform)
        client = _admin_client(platform)
        invite = client.post(
            f"/api/admin/unions/{union_id}/invites",
            json={"label": "break room poster"},
        ).json()

        # One tap: no name, no email, no password — straight to a member session.
        phone = TestClient(api.app)
        joined = phone.post("/api/auth/session/join-guest", json={"code": invite["code"]})
        assert joined.status_code == 200, joined.text
        body = joined.json()
        assert body["guest"] is True
        assert body["user"]["union_slug"] == union_slug
        assert phone.cookies.get(platform.settings.session_cookie_name)

        me = phone.get("/api/auth/session/me")
        assert me.status_code == 200
        assert me.json()["authenticated"] is True
        assert me.json()["role"] == "user"

        # Second scan = second independent guest session/identity.
        phone_two = TestClient(api.app)
        joined_two = phone_two.post("/api/auth/session/join-guest", json={"code": invite["code"]})
        assert joined_two.status_code == 200
        assert joined_two.json()["user"]["id"] != body["user"]["id"]

        with platform.session_factory() as db:
            stored = db.scalar(select(InviteCode).where(InviteCode.code == invite["code"]))
            assert stored.use_count == 2
            sessions = db.scalars(
                select(AuthSession).where(AuthSession.invite_code_id == stored.id)
            ).all()
            assert len(sessions) == 2 and all(s.revoked_at is None for s in sessions)

        # Misuse lever: revoke + disconnect kills the code AND its sessions.
        revoked = client.post(
            f"/api/admin/unions/{union_id}/invites/{invite['id']}/revoke",
            json={"disconnect_sessions": True},
        )
        assert revoked.status_code == 200, revoked.text
        assert revoked.json()["status"] == "revoked"
        assert revoked.json()["sessions_disconnected"] == 2

        with platform.session_factory() as db:
            stored = db.scalar(select(InviteCode).where(InviteCode.code == invite["code"]))
            sessions = db.scalars(
                select(AuthSession).where(AuthSession.invite_code_id == stored.id)
            ).all()
            assert all(s.revoked_at is not None for s in sessions)

        # Disconnected members are signed out; new joins through the code are refused.
        me_after = phone.get("/api/auth/session/me")
        assert me_after.status_code == 200
        assert me_after.json()["authenticated"] is False
        assert (
            TestClient(api.app)
            .post("/api/auth/session/join-guest", json={"code": invite["code"]})
            .status_code
            == 410
        )
    finally:
        api.app.state.platform = prior


def test_invite_pinned_contract_overrides_client_supplied_contract():
    """The invite's contract pin must win over the request body.

    The QR code taped to the meat department's board is the union's control
    over which agreement its scanners can read. The request body is attacker
    controlled, so if the client could override the pin, a member could read
    the other bargaining unit's contract by editing one field.
    """
    from types import SimpleNamespace

    from backend.api import _resolve_scoped_contract_id

    class _FakeDB:
        def __init__(self, objects, filed=None):
            self._objects = objects
            self._filed = filed or set()

        def get(self, model, key):
            return self._objects.get(key)

        def scalar(self, *_args, **_kwargs):
            return "doc-1" if self._filed else None

    session = SimpleNamespace(invite_code_id="invite-1")
    invite = SimpleNamespace(contract_id="meat_2022")
    db = _FakeDB({"session-1": session, "invite-1": invite}, filed={"clerks_2022"})
    auth = SimpleNamespace(session_id="session-1")

    # Even though the union does have clerks documents filed, the pin wins.
    assert _resolve_scoped_contract_id(db, auth, "clerks_2022", "union-1") == "meat_2022"
    assert _resolve_scoped_contract_id(db, auth, None, "union-1") == "meat_2022"


def test_unpinned_invite_falls_back_to_requested_contract():
    from types import SimpleNamespace

    from backend.api import _resolve_scoped_contract_id

    class _FakeDB:
        def __init__(self, objects, has_filed_documents):
            self._objects = objects
            self._has_filed_documents = has_filed_documents

        def get(self, model, key):
            return self._objects.get(key)

        def scalar(self, *_args, **_kwargs):
            return "doc-1" if self._has_filed_documents else None

    session = SimpleNamespace(invite_code_id="invite-1")
    invite = SimpleNamespace(contract_id=None)
    auth = SimpleNamespace(session_id="session-1")
    objects = {"session-1": session, "invite-1": invite}

    filed = _FakeDB(objects, has_filed_documents=True)
    assert _resolve_scoped_contract_id(filed, auth, "clerks_2022", "union-1") == "clerks_2022"
    assert _resolve_scoped_contract_id(filed, auth, "  ", "union-1") is None

    # Corpus predates contract scoping: every document is NULL, so a required
    # request field must not be allowed to filter everything out.
    unfiled = _FakeDB(objects, has_filed_documents=False)
    assert _resolve_scoped_contract_id(unfiled, auth, "clerks_2022", "union-1") is None
