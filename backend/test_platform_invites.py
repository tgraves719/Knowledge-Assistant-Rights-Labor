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
from backend.platform.auth import AuthContext, HeaderAuthAdapter
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
            json={"label": "Pueblo West break room poster", "max_uses": 5, "contract_id": "clerks_2022"},
        )
        assert created.status_code == 200, created.text
        invite = created.json()
        assert invite["status"] == "active"
        assert invite["audience"] == "member"
        assert invite["contract_id"] == "clerks_2022"
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
            json={"label": "steward hand-card", "max_uses": 1, "contract_id": "clerks_2022"},
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
            json={"label": "old poster", "contract_id": "clerks_2022", "expires_at": (datetime.utcnow() - timedelta(days=1)).isoformat()},
        ).json()
        assert TestClient(api.app).get(f"/api/auth/join/{expired['code']}").status_code == 410

        # Revocation closes an active code and the landing page reflects it.
        active = client.post(
            f"/api/admin/unions/{union_id}/invites",
            json={"label": "union board", "contract_id": "clerks_2022"},
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
            json={"label": "break room poster", "contract_id": "clerks_2022"},
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


def test_audience_validation_across_three_tiers(tmp_path):
    platform = _build_platform(tmp_path)
    prior = _with_app(platform)
    try:
        union_id, _ = _seed_union_with_admin(platform)
        client = _admin_client(platform)

        def create(payload):
            return client.post(f"/api/admin/unions/{union_id}/invites", json=payload)

        # Tier 1 member: needs exactly one contract, no set.
        assert create({"audience": "member"}).status_code == 400
        bad = create({"audience": "member", "contract_ids": ["clerks_2022"]})
        assert bad.status_code == 400 and "single contract" in bad.text

        # Tier 2 steward: needs a contract set, forbids a single pin.
        assert create({"audience": "steward"}).status_code == 400
        pinned_steward = create({"audience": "steward", "contract_id": "clerks_2022"})
        assert pinned_steward.status_code == 400 and "contract set" in pinned_steward.text

        # Tier 3 union_rep: forbids any contract scoping.
        assert create({"audience": "union_rep", "contract_id": "clerks_2022"}).status_code == 400
        assert create({"audience": "union_rep", "contract_ids": ["clerks_2022"]}).status_code == 400

        # Unknown audience rejected.
        assert create({"audience": "manager"}).status_code == 400

        # Valid codes for each tier.
        member = create({"label": "poster", "audience": "member", "contract_id": "clerks_2022"})
        assert member.status_code == 200 and member.json()["contract_id"] == "clerks_2022"

        steward = create({"label": "pueblo store", "audience": "steward", "contract_ids": ["clerks_2022", "meat_2022"]})
        assert steward.status_code == 200, steward.text
        assert steward.json()["audience"] == "steward"
        assert steward.json()["contract_ids"] == ["clerks_2022", "meat_2022"]
        assert steward.json()["contract_id"] is None

        union_rep = create({"label": "rep cards", "audience": "union_rep"})
        assert union_rep.status_code == 200
        assert union_rep.json()["audience"] == "union_rep"
        assert union_rep.json()["contract_ids"] == []
    finally:
        api.app.state.platform = prior


def test_union_rep_code_grants_steward_role_on_scan(tmp_path):
    platform = _build_platform(tmp_path)
    prior = _with_app(platform)
    try:
        union_id, union_slug = _seed_union_with_admin(platform)
        client = _admin_client(platform)
        rep_code = client.post(
            f"/api/admin/unions/{union_id}/invites",
            json={"label": "union rep cards", "audience": "union_rep"},
        ).json()

        # Scanning a union-rep card mints a steward_admin session — role decided server-side.
        phone = TestClient(api.app)
        joined = phone.post("/api/auth/session/join-guest", json={"code": rep_code["code"]})
        assert joined.status_code == 200, joined.text
        assert joined.json()["user"]["role"] == Role.STEWARD_ADMIN.value

        me = phone.get("/api/auth/session/me")
        assert me.status_code == 200
        assert me.json()["role"] == Role.STEWARD_ADMIN.value

        with platform.session_factory() as db:
            member = db.get(User, joined.json()["user"]["id"])
            membership = db.scalar(
                select(UnionMembership).where(UnionMembership.user_id == member.id)
            )
            assert membership.role == Role.STEWARD_ADMIN
    finally:
        api.app.state.platform = prior


def test_invite_usage_timestamps_advance(tmp_path):
    platform = _build_platform(tmp_path)
    prior = _with_app(platform)
    try:
        union_id, _ = _seed_union_with_admin(platform)
        client = _admin_client(platform)
        invite = client.post(
            f"/api/admin/unions/{union_id}/invites",
            json={"label": "poster", "contract_id": "clerks_2022"},
        ).json()
        assert invite["first_used_at"] is None and invite["last_used_at"] is None

        TestClient(api.app).post("/api/auth/session/join-guest", json={"code": invite["code"]})
        TestClient(api.app).post("/api/auth/session/join-guest", json={"code": invite["code"]})

        item = next(
            it for it in client.get(f"/api/admin/unions/{union_id}/invites").json()["items"]
            if it["id"] == invite["id"]
        )
        assert item["use_count"] == 2
        assert item["first_used_at"] is not None
        assert item["last_used_at"] is not None
        assert item["last_used_at"] >= item["first_used_at"]
    finally:
        api.app.state.platform = prior


def _seed_contract_docs(platform, union_id):
    """Two single-doc contracts in one union, both member-readable."""
    from backend.platform.models import ChunkEmbedding, Document, DocumentStatus

    with platform.session_factory() as db:
        for cid, article in (("clerks_2022", "12"), ("meat_2022", "7")):
            doc = Document(
                union_id=union_id,
                title=f"{cid} agreement",
                contract_id=cid,
                storage_key=f"{cid}.txt",
                content_type="text/plain",
                status=DocumentStatus.ACTIVE,
                metadata_json={"ready_for_query": True, "member_visible": True},
            )
            db.add(doc)
            db.flush()
            db.add(ChunkEmbedding(
                union_id=union_id,
                document_id=doc.id,
                chunk_index=0,
                chunk_text=f"{cid} article {article} text",
                metadata_json={"article_num": article, "article_title": f"Article {article}", "section_num": "1"},
            ))
        db.commit()


def test_pinned_member_cannot_read_sibling_contract_explorer(tmp_path):
    """Regression: the contract pin must gate the explorer, not just the contract list.

    A member who joined through a clerks-pinned code must get 404 (not the text)
    when requesting the meat contract's outline or section by id directly.
    """
    platform = _build_platform(tmp_path)
    prior = _with_app(platform)
    try:
        union_id, union_slug = _seed_union_with_admin(platform)
        _seed_contract_docs(platform, union_id)
        client = _admin_client(platform)
        pinned = client.post(
            f"/api/admin/unions/{union_id}/invites",
            json={"label": "meat board", "contract_id": "clerks_2022"},
        ).json()

        phone = TestClient(api.app)
        phone.post("/api/auth/session/join-guest", json={"code": pinned["code"]})

        # The list only offers the pinned contract.
        contracts = phone.get("/api/member/contracts")
        assert contracts.status_code == 200
        listed = contracts.json()
        assert listed["pinned_contract_id"] == "clerks_2022"
        assert [c["contract_id"] for c in listed["contracts"]] == ["clerks_2022"]

        # Pinned contract reads fine.
        assert phone.get("/api/member/contracts/clerks_2022/outline").status_code == 200
        # Sibling contract is invisible — 404, indistinguishable from "no such contract".
        assert phone.get("/api/member/contracts/meat_2022/outline").status_code == 404
        assert phone.get("/api/member/contracts/meat_2022/section?article_num=7").status_code == 404
    finally:
        api.app.state.platform = prior


def test_union_rep_session_sees_all_contracts(tmp_path):
    """A union rep (Tier 3) sees every contract in the local and can read each."""
    platform = _build_platform(tmp_path)
    prior = _with_app(platform)
    try:
        union_id, _ = _seed_union_with_admin(platform)
        _seed_contract_docs(platform, union_id)
        client = _admin_client(platform)
        rep_code = client.post(
            f"/api/admin/unions/{union_id}/invites",
            json={"label": "rep cards", "audience": "union_rep"},
        ).json()

        phone = TestClient(api.app)
        phone.post("/api/auth/session/join-guest", json={"code": rep_code["code"]})

        contracts = phone.get("/api/member/contracts")
        assert contracts.status_code == 200
        body = contracts.json()
        assert body["pinned_contract_id"] is None
        assert body["allowed_contract_ids"] is None
        assert {c["contract_id"] for c in body["contracts"]} == {"clerks_2022", "meat_2022"}
        assert phone.get("/api/member/contracts/clerks_2022/outline").status_code == 200
        assert phone.get("/api/member/contracts/meat_2022/outline").status_code == 200
    finally:
        api.app.state.platform = prior


def test_steward_session_scoped_to_store_contracts(tmp_path):
    """A Tier-2 steward sees only their store's contract set, not the whole local.

    With two contracts filed in the local, a steward code scoped to just the
    clerks agreement must offer only clerks in the explorer and 404 on the meat
    agreement — the same isolation a member pin gives, but over a set.
    """
    platform = _build_platform(tmp_path)
    prior = _with_app(platform)
    try:
        union_id, _ = _seed_union_with_admin(platform)
        _seed_contract_docs(platform, union_id)  # clerks_2022 + meat_2022
        client = _admin_client(platform)
        steward_code = client.post(
            f"/api/admin/unions/{union_id}/invites",
            json={"label": "one-book store", "audience": "steward", "contract_ids": ["clerks_2022"]},
        ).json()
        assert steward_code["contract_ids"] == ["clerks_2022"]

        phone = TestClient(api.app)
        joined = phone.post("/api/auth/session/join-guest", json={"code": steward_code["code"]})
        assert joined.json()["user"]["role"] == Role.STEWARD_ADMIN.value

        contracts = phone.get("/api/member/contracts")
        assert contracts.status_code == 200
        body = contracts.json()
        assert body["allowed_contract_ids"] == ["clerks_2022"]
        assert [c["contract_id"] for c in body["contracts"]] == ["clerks_2022"]

        # In-store contract reads; out-of-store contract is invisible (404).
        assert phone.get("/api/member/contracts/clerks_2022/outline").status_code == 200
        assert phone.get("/api/member/contracts/meat_2022/outline").status_code == 404
        assert phone.get("/api/member/contracts/meat_2022/section?article_num=7").status_code == 404
    finally:
        api.app.state.platform = prior


def test_usage_is_metered_per_invite_code(tmp_path):
    """Token/request/cost usage is attributed to the code a member joined through,
    and the admin invite list reports per-code totals."""
    platform = _build_platform(tmp_path)
    prior = _with_app(platform)
    try:
        union_id, union_slug = _seed_union_with_admin(platform)
        client = _admin_client(platform)
        invite = client.post(
            f"/api/admin/unions/{union_id}/invites",
            json={"label": "poster", "contract_id": "clerks_2022"},
        ).json()

        # A session that joined through this code carries its id on the auth context;
        # record_usage stamps it onto the usage event.
        phone = TestClient(api.app)
        phone.post("/api/auth/session/join-guest", json={"code": invite["code"]})
        with platform.session_factory() as db:
            member = db.scalar(select(User).where(User.email.like("guest-%@join.karl.invalid")))
            auth = AuthContext(
                user_id=member.id,
                email=member.email,
                full_name=member.full_name,
                role=Role.USER.value,
                union_id=union_id,
                union_slug=union_slug,
                source="session",
                is_authenticated=True,
                invite_code_id=invite["id"],
            )
            platform.quotas.record_usage(
                db, auth, route="/api/query", token_count=1500, estimated_cost_usd=0.02
            )
            platform.quotas.record_usage(
                db, auth, route="/api/query", token_count=500, estimated_cost_usd=0.01
            )
            db.commit()

        item = next(
            it for it in client.get(f"/api/admin/unions/{union_id}/invites").json()["items"]
            if it["id"] == invite["id"]
        )
        assert item["total_requests"] == 2
        assert item["total_tokens"] == 2000
        assert round(item["total_cost_usd"], 4) == 0.03
    finally:
        api.app.state.platform = prior


def test_authenticated_session_carries_its_invite_code(tmp_path):
    """The QR-minted session resolves to an auth context tagged with its code,
    which is what lets usage be metered per code downstream."""
    platform = _build_platform(tmp_path)
    prior = _with_app(platform)
    try:
        union_id, _ = _seed_union_with_admin(platform)
        invite = _admin_client(platform).post(
            f"/api/admin/unions/{union_id}/invites",
            json={"label": "poster", "contract_id": "clerks_2022"},
        ).json()

        phone = TestClient(api.app)
        phone.post("/api/auth/session/join-guest", json={"code": invite["code"]})
        cookie = phone.cookies.get(platform.settings.session_cookie_name)
        with platform.session_factory() as db:
            resolved = platform.auth_adapter.resolve(
                db=db,
                session_cookie=cookie,
                authorization=None,
                external_auth_id=None,
                email=None,
                full_name=None,
                requested_role=None,
                union_slug=None,
            )
        assert resolved.is_authenticated
        assert resolved.invite_code_id == invite["id"]
    finally:
        api.app.state.platform = prior


def test_invite_qr_and_printable_card_render(tmp_path):
    platform = _build_platform(tmp_path)
    prior = _with_app(platform)
    try:
        union_id, _ = _seed_union_with_admin(platform)
        client = _admin_client(platform)
        invite = client.post(
            f"/api/admin/unions/{union_id}/invites",
            json={"label": "poster", "contract_id": "clerks_2022"},
        ).json()

        svg = client.get(f"/api/admin/unions/{union_id}/invites/{invite['id']}/qr")
        assert svg.status_code == 200
        assert svg.headers["content-type"].startswith("image/svg")
        assert b"<svg" in svg.content

        png = client.get(f"/api/admin/unions/{union_id}/invites/{invite['id']}/qr?format=png")
        assert png.status_code == 200
        assert png.headers["content-type"] == "image/png"
        assert png.content[:8] == b"\x89PNG\r\n\x1a\n"

        card = client.get(f"/api/admin/unions/{union_id}/invites/{invite['id']}/card")
        assert card.status_code == 200
        assert invite["code"] in card.text
        assert f"/j/{invite['code']}" in card.text

        # Anonymous users cannot pull QR assets.
        assert TestClient(api.app).get(
            f"/api/admin/unions/{union_id}/invites/{invite['id']}/qr"
        ).status_code == 401
    finally:
        api.app.state.platform = prior


def test_tenant_member_app_requires_a_session(tmp_path):
    """Navigating straight to /u/{slug}/ without a QR-minted session gets the gate."""
    platform = _build_platform(tmp_path)
    prior = _with_app(platform)
    try:
        union_id, union_slug = _seed_union_with_admin(platform)
        client = _admin_client(platform)
        code = client.post(
            f"/api/admin/unions/{union_id}/invites",
            json={"label": "poster", "contract_id": "clerks_2022"},
        ).json()["code"]

        # Anonymous visitor: no session → access-code gate, not the app.
        anon = TestClient(api.app)
        gated = anon.get(f"/u/{union_slug}/")
        assert gated.status_code == 401
        assert "access code" in gated.text.lower()
        assert "__KARL_ROUTE_CONTEXT__" not in gated.text  # app shell not served

        # After joining through the code, the same client gets the real app.
        phone = TestClient(api.app)
        phone.post("/api/auth/session/join-guest", json={"code": code})
        opened = phone.get(f"/u/{union_slug}/")
        assert opened.status_code == 200
        assert "__KARL_ROUTE_CONTEXT__" in opened.text
    finally:
        api.app.state.platform = prior


def test_tenant_query_requires_a_session(tmp_path):
    """A query aimed at a real union tenant is refused without an entitled session."""
    platform = _build_platform(tmp_path)
    prior = _with_app(platform)
    try:
        union_id, _ = _seed_union_with_admin(platform)
        code = _admin_client(platform).post(
            f"/api/admin/unions/{union_id}/invites",
            json={"label": "poster", "contract_id": "clerks_2022"},
        ).json()["code"]

        payload = {
            "question": "what is my wage?",
            "union_local_id": "ufcw-7",
            "contract_id": "local7_safeway_pueblo_clerks_2022",
            "contract_version": "current",
        }
        anon = TestClient(api.app)
        assert anon.post("/api/query", json=payload).status_code == 401

        # An entitled member session gets past the auth gate (may 4xx later for
        # missing docs, but never 401).
        phone = TestClient(api.app)
        phone.post("/api/auth/session/join-guest", json={"code": code})
        assert phone.post("/api/query", json=payload).status_code != 401
    finally:
        api.app.state.platform = prior


def test_superadmin_and_tenant_admin_shells_require_auth(tmp_path):
    platform = _build_platform(tmp_path)
    prior = _with_app(platform)
    try:
        _, union_slug = _seed_union_with_admin(platform)

        anon = TestClient(api.app)
        karl = anon.get("/karl/")
        assert karl.status_code == 401
        assert "sign in" in karl.text.lower()
        assert "/api/auth/session/login" in karl.text  # real sign-in form, not a dead end

        admin_shell = anon.get(f"/u/{union_slug}/admin")
        assert admin_shell.status_code == 401
        assert "sign in" in admin_shell.text.lower()

        # The canonical /signin entry serves the form to anonymous visitors.
        signin = anon.get("/signin")
        assert signin.status_code == 200
        assert "access code" in signin.text.lower()
        assert "/api/auth/session/login" in signin.text

        # A signed-in union admin reaches their console.
        client = _admin_client(platform)
        assert client.get(f"/u/{union_slug}/admin").status_code == 200

        # /signin routes an already-authenticated admin to their console.
        redirected = client.get("/signin", follow_redirects=False)
        assert redirected.status_code == 303
        assert redirected.headers["location"] == f"/u/{union_slug}/admin"
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


def test_steward_store_allowlist_scopes_query_retrieval():
    """A Tier-2 steward code scopes queries to its store's contract set.

    A chosen contract inside the store narrows to it; anything else (or nothing)
    restricts retrieval to the whole allowlist and never leaks outside it.
    """
    from types import SimpleNamespace

    from backend.api import _resolve_query_contract_scope

    class _FakeDB:
        def __init__(self, objects):
            self._objects = objects

        def get(self, model, key):
            return self._objects.get(key)

        def scalar(self, *_args, **_kwargs):
            return None

    session = SimpleNamespace(invite_code_id="invite-1")
    invite = SimpleNamespace(contract_id=None, contract_ids=["clerks_2022", "meat_2022"])
    db = _FakeDB({"session-1": session, "invite-1": invite})
    auth = SimpleNamespace(session_id="session-1")

    # A contract inside the store narrows to exactly it.
    assert _resolve_query_contract_scope(db, auth, "meat_2022", "union-1") == ("meat_2022", None)
    # No / invalid choice restricts to the whole store allowlist.
    assert _resolve_query_contract_scope(db, auth, None, "union-1") == (None, ["clerks_2022", "meat_2022"])
    assert _resolve_query_contract_scope(db, auth, "deli_2022", "union-1") == (None, ["clerks_2022", "meat_2022"])
