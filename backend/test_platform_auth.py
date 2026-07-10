from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.platform.auth import (
    AuthContext,
    HeaderAuthAdapter,
    get_current_auth_context,
    reset_current_auth_context,
    set_current_auth_context,
)
from backend.platform.db import Base
from backend.platform.db import apply_request_context
from backend.platform.models import AuthSession, Role, SessionType, Union, UnionMembership, User
from backend.platform.session_auth import SessionAuthService
from backend.platform.settings import PlatformSettings


def _settings(*, bootstrap_super_admin_emails: list[str] | None = None) -> PlatformSettings:
    return PlatformSettings(
        project_root=Path("/tmp"),
        postgres_url="",
        auto_create_tables=False,
        apply_rls_policies=False,
        allowed_origins=["*"],
        request_rate_limit_per_minute=60,
        login_rate_limit_per_minute=10,
        query_token_limit=4000,
        hard_cap_default_requests_per_day=500,
        hard_cap_default_tokens_per_day=250000,
        hard_cap_default_cost_usd_per_day=25.0,
        local_storage_root=Path("/tmp"),
        storage_backend="local",
        document_parser_backend="auto",
        liteparse_executable="",
        embedding_backend="deterministic",
        embedding_dimensions=384,
        google_embedding_model="text-embedding-004",
        google_embedding_api_key="",
        inference_request_timeout_seconds=8,
        legacy_contract_pipeline_enabled=False,
        inline_parse_max_bytes=1_000_000,
        ocr_auto_retry_enabled=True,
        ocr_auto_retry_max_attempts=1,
        local_auth_token_ttl_seconds=43_200,
        secret_encryption_key="x",
        sentinel_email_from="sentinel@example.com",
        sentinel_email_enabled=False,
        bootstrap_super_admin_emails=bootstrap_super_admin_emails or [],
    )


def test_header_auth_adapter_bootstrap_super_admin_without_db():
    adapter = HeaderAuthAdapter(_settings(bootstrap_super_admin_emails=["admin@example.com"]))
    auth = adapter.resolve(
        db=None,
        session_cookie=None,
        authorization=None,
        external_auth_id=None,
        email="admin@example.com",
        full_name="Admin",
        requested_role="user",
        union_slug=None,
    )

    assert auth.is_authenticated is True
    assert auth.is_super_admin is True
    assert auth.role == "super_admin"


def test_header_auth_adapter_requires_membership_for_non_super_admin_roles():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        user = User(email="memberless@example.com", full_name="Memberless User")
        db.add_all([union, user])
        db.commit()

        adapter = HeaderAuthAdapter(_settings())
        auth = adapter.resolve(
            db=db,
            session_cookie=None,
            authorization=None,
            external_auth_id=None,
            email="memberless@example.com",
            full_name=None,
            requested_role="union_admin",
            union_slug="local-1",
        )

    assert auth.is_authenticated is True
    assert auth.role == Role.USER.value
    assert auth.union_id is None


def test_header_auth_adapter_uses_membership_role():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        user = User(email="admin@example.com", full_name="Union Admin")
        db.add_all([union, user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=user.id, role=Role.UNION_ADMIN))
        db.commit()

        adapter = HeaderAuthAdapter(_settings())
        auth = adapter.resolve(
            db=db,
            session_cookie=None,
            authorization=None,
            external_auth_id=None,
            email="admin@example.com",
            full_name=None,
            requested_role="user",
            union_slug="local-1",
        )

    assert auth.is_authenticated is True
    assert auth.role == Role.UNION_ADMIN.value
    assert auth.union_id == union.id


def test_apply_request_context_sets_postgres_session_values():
    calls = []

    class FakeSession:
        class _Bind:
            class dialect:
                name = "postgresql"

        def get_bind(self):
            return self._Bind()

        def execute(self, statement, params):
            calls.append((str(statement), params))

    apply_request_context(
        FakeSession(),
        AuthContext(
            user_id="user-1",
            email="user@example.com",
            full_name="User",
            role="union_admin",
            union_id="union-1",
            union_slug="local-1",
            source="header",
            is_authenticated=True,
        ),
    )

    assert len(calls) == 1
    sql, params = calls[0]
    assert "set_config('app.current_role'" in sql
    assert params == {
        "role": "union_admin",
        "union_id": "union-1",
        "user_id": "user-1",
    }


def test_current_auth_context_round_trip():
    auth = AuthContext(
        user_id="user-1",
        email="user@example.com",
        full_name="User",
        role="user",
        union_id="union-1",
        union_slug="local-1",
        source="test",
        is_authenticated=True,
    )

    token = set_current_auth_context(auth)
    try:
        assert get_current_auth_context() is auth
    finally:
        reset_current_auth_context(token)

    assert get_current_auth_context() is None


def test_session_auth_service_creates_and_slides_member_session():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    service = SessionAuthService(
        secret_key="session-secret",
        cookie_name="karl_session",
        member_idle_seconds=604800,
        union_admin_idle_seconds=259200,
        super_admin_idle_seconds=3600,
    )

    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        user = User(email="member@example.com", full_name="Member")
        db.add_all([union, user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=user.id, role=Role.USER))
        db.commit()

        session_row, session_secret = service.create_session(db, user=user, union=union, role=Role.USER.value)
        original_expiry = session_row.expires_at
        session_row.last_seen_at = datetime.utcnow() - timedelta(minutes=5)
        session_row.expires_at = datetime.utcnow() + timedelta(minutes=1)
        db.flush()

        resolved = service.resolve(db, session_secret=session_secret, requested_union_slug="local-1")

        assert resolved.is_authenticated is True
        assert resolved.role == Role.USER.value
        assert resolved.session.session_type == SessionType.MEMBER
        assert resolved.session.expires_at < original_expiry


def test_session_auth_service_clears_cookie_for_expired_session():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    service = SessionAuthService(
        secret_key="session-secret",
        cookie_name="karl_session",
        member_idle_seconds=604800,
        union_admin_idle_seconds=259200,
        super_admin_idle_seconds=3600,
    )

    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        user = User(email="member@example.com", full_name="Member")
        db.add_all([union, user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=user.id, role=Role.USER))
        session_row, session_secret = service.create_session(db, user=user, union=union, role=Role.USER.value)
        session_row.expires_at = datetime.utcnow() - timedelta(seconds=1)
        db.flush()

        resolved = service.resolve(db, session_secret=session_secret, requested_union_slug="local-1")

        assert resolved.is_authenticated is False
        assert resolved.clear_cookie is True
        assert db.get(AuthSession, session_row.id).revoked_at is not None


def test_purge_expired_sessions_removes_terminated_rows_past_window():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    service = SessionAuthService(
        secret_key="session-secret",
        cookie_name="karl_session",
        member_idle_seconds=604800,
        union_admin_idle_seconds=259200,
        super_admin_idle_seconds=3600,
    )

    now = datetime.utcnow()
    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        user = User(email="member@example.com", full_name="Member")
        db.add_all([union, user])
        db.flush()

        def _make(secret_hash, *, expires_at, revoked_at=None):
            row = AuthSession(
                user_id=user.id,
                union_id=union.id,
                session_secret_hash=secret_hash,
                session_type=SessionType.MEMBER,
                created_at=now - timedelta(days=200),
                last_seen_at=now - timedelta(days=200),
                expires_at=expires_at,
                revoked_at=revoked_at,
            )
            db.add(row)
            db.flush()
            return row.id

        active_id = _make("active", expires_at=now + timedelta(days=1))
        recently_expired_id = _make("recent-expired", expires_at=now - timedelta(days=10))
        old_expired_id = _make("old-expired", expires_at=now - timedelta(days=120))
        recently_revoked_id = _make("recent-revoked", expires_at=now + timedelta(days=1), revoked_at=now - timedelta(days=10))
        old_revoked_id = _make("old-revoked", expires_at=now + timedelta(days=1), revoked_at=now - timedelta(days=120))
        db.commit()

        deleted = service.purge_expired_sessions(db, older_than_days=90)
        db.commit()

        assert deleted == 2
        # Active and recently-terminated sessions are retained.
        assert db.get(AuthSession, active_id) is not None
        assert db.get(AuthSession, recently_expired_id) is not None
        assert db.get(AuthSession, recently_revoked_id) is not None
        # Sessions terminated longer ago than the window are deleted (IP/UA removed with them).
        assert db.get(AuthSession, old_expired_id) is None
        assert db.get(AuthSession, old_revoked_id) is None
