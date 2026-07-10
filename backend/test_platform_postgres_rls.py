import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from backend.platform.auth import AuthContext
from backend.platform.db import apply_request_context
from backend.platform.models import (
    AuditEvent,
    AuthSession,
    Document,
    DocumentStatus,
    MemberTrackingChoiceMode,
    Notification,
    NotificationStatus,
    ProviderConfig,
    QuotaPolicy,
    RawQueryRecord,
    RawQueryStorageMode,
    Role,
    SecurityEvent,
    SecuritySeverity,
    SessionType,
    TelemetryEvent,
    TrackingMode,
    TrackingPolicy,
    TrackingPreference,
    TrackingPrivacyMode,
    Union,
    UnionMembership,
    UsageEvent,
    User,
    UserTrackingPreference,
)


ADMIN_URL_ENV = "KARL_TEST_POSTGRES_ADMIN_URL"
TEST_DB_ENV = "KARL_TEST_POSTGRES_DB"


def _admin_url() -> str | None:
    return os.getenv(ADMIN_URL_ENV)


pytestmark = pytest.mark.skipif(
    not _admin_url(),
    reason=f"Set {ADMIN_URL_ENV} to run live PostgreSQL RLS integration tests.",
)


@pytest.fixture()
def postgres_env():
    for sidecar in Path("alembic").rglob("._*"):
        sidecar.unlink(missing_ok=True)

    admin_url = make_url(_admin_url())
    db_name = os.getenv(TEST_DB_ENV, f"karl_rls_test_{uuid.uuid4().hex[:8]}")
    db_url = admin_url.set(database=db_name)
    app_role = f"karl_rls_app_{uuid.uuid4().hex[:8]}"
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
        alembic_cfg = Config("alembic.ini")
        command.upgrade(alembic_cfg, "head")

        owner_engine = create_engine(db_url, future=True, pool_pre_ping=True)
        with owner_engine.connect() as conn:
            conn.execute(text(f'GRANT CONNECT ON DATABASE "{db_name}" TO "{app_role}"'))
            conn.execute(text(f'GRANT USAGE ON SCHEMA public TO "{app_role}"'))
            conn.execute(
                text(
                    f'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO "{app_role}"'
                )
            )
            conn.execute(text(f'GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO "{app_role}"'))
            conn.commit()

        app_engine = create_engine(app_db_url, future=True, pool_pre_ping=True)
        OwnerSessionLocal = sessionmaker(bind=owner_engine, autoflush=False, autocommit=False, future=True)
        AppSessionLocal = sessionmaker(bind=app_engine, autoflush=False, autocommit=False, future=True)
        try:
            yield {
                "owner_engine": owner_engine,
                "app_engine": app_engine,
                "owner_session_factory": OwnerSessionLocal,
                "app_session_factory": AppSessionLocal,
                "db_url": str(db_url),
                "app_db_url": str(app_db_url),
                "db_name": db_name,
                "app_role": app_role,
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


def _auth(*, role: str, union_id: str | None = None, user_id: str | None = None) -> AuthContext:
    return AuthContext(
        user_id=user_id,
        email=None,
        full_name=None,
        role=role,
        union_id=union_id,
        union_slug=None,
        source="test",
        is_authenticated=True,
    )


def _seed_sample_data(session_factory):
    with session_factory() as db:
        apply_request_context(db, _auth(role=Role.SUPER_ADMIN.value))

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
                    encrypted_api_key="encrypted-1",
                    config_json={},
                ),
                ProviderConfig(
                    union_id=union_two.id,
                    provider_name="openai",
                    model_name="gpt-test",
                    encrypted_api_key="encrypted-2",
                    config_json={},
                ),
                QuotaPolicy(union_id=union_one.id),
                QuotaPolicy(union_id=union_two.id),
                Document(
                    union_id=union_one.id,
                    uploaded_by_user_id=user_one.id,
                    title="Local 1 CBA",
                    storage_key="local-1/doc.pdf",
                    content_type="application/pdf",
                    bytes_size=100,
                    status=DocumentStatus.ACTIVE,
                ),
                Document(
                    union_id=union_two.id,
                    uploaded_by_user_id=user_two.id,
                    title="Local 2 CBA",
                    storage_key="local-2/doc.pdf",
                    content_type="application/pdf",
                    bytes_size=100,
                    status=DocumentStatus.ACTIVE,
                ),
                SecurityEvent(
                    union_id=union_one.id,
                    user_id=user_one.id,
                    event_type="quota_warning",
                    severity=SecuritySeverity.WARNING,
                    response_action="notify",
                    details_json={},
                ),
                SecurityEvent(
                    union_id=union_two.id,
                    user_id=user_two.id,
                    event_type="quota_warning",
                    severity=SecuritySeverity.WARNING,
                    response_action="notify",
                    details_json={},
                ),
                Notification(
                    union_id=union_one.id,
                    user_id=user_one.id,
                    channel="in_app",
                    subject="Union One",
                    body="union one",
                    status=NotificationStatus.PENDING,
                ),
                Notification(
                    union_id=union_two.id,
                    user_id=user_two.id,
                    channel="in_app",
                    subject="Union Two",
                    body="union two",
                    status=NotificationStatus.PENDING,
                ),
                AuditEvent(union_id=union_one.id, actor_user_id=user_one.id, event_type="seed", event_payload={}),
                AuditEvent(union_id=union_two.id, actor_user_id=user_two.id, event_type="seed", event_payload={}),
                UsageEvent(union_id=union_one.id, user_id=user_one.id, route="/api/query"),
                UsageEvent(union_id=union_two.id, user_id=user_two.id, route="/api/query"),
            ]
        )
        db.commit()
        return {
            "union_one_id": union_one.id,
            "union_two_id": union_two.id,
            "user_one_id": user_one.id,
            "user_two_id": user_two.id,
        }


def test_postgres_migration_enables_vector_and_rls(postgres_env):
    with postgres_env["owner_session_factory"]() as db:
        extension = db.execute(text("select extname from pg_extension where extname = 'vector'")).scalar()
        rls_row = db.execute(
            text(
                """
                select relrowsecurity, relforcerowsecurity
                from pg_class
                where relname = 'documents'
                """
            )
        ).one()

    assert extension == "vector"
    assert rls_row[0] is True
    assert rls_row[1] is True


def test_postgres_rls_limits_tenant_reads(postgres_env):
    seeded = _seed_sample_data(postgres_env["owner_session_factory"])

    with postgres_env["app_session_factory"]() as db:
        apply_request_context(
            db,
            _auth(
                role=Role.UNION_ADMIN.value,
                union_id=seeded["union_one_id"],
                user_id=seeded["user_one_id"],
            ),
        )

        unions = db.scalars(select(Union).order_by(Union.slug.asc())).all()
        memberships = db.scalars(select(UnionMembership).order_by(UnionMembership.union_id.asc())).all()
        documents = db.scalars(select(Document).order_by(Document.title.asc())).all()
        providers = db.scalars(select(ProviderConfig).order_by(ProviderConfig.union_id.asc())).all()
        quotas = db.scalars(select(QuotaPolicy).order_by(QuotaPolicy.union_id.asc())).all()
        security_events = db.scalars(select(SecurityEvent).order_by(SecurityEvent.union_id.asc())).all()
        notifications = db.scalars(select(Notification).order_by(Notification.union_id.asc())).all()

        cross_union = db.get(Union, seeded["union_two_id"])
        cross_document = db.scalar(select(Document).where(Document.union_id == seeded["union_two_id"]))

    assert [item.slug for item in unions] == ["local-1"]
    assert len(memberships) == 1
    assert len(documents) == 1
    assert len(providers) == 1
    assert len(quotas) == 1
    assert len(security_events) == 1
    assert len(notifications) == 1
    assert cross_union is None
    assert cross_document is None


def test_postgres_rls_allows_super_admin_global_visibility(postgres_env):
    _seed_sample_data(postgres_env["owner_session_factory"])

    with postgres_env["app_session_factory"]() as db:
        apply_request_context(db, _auth(role=Role.SUPER_ADMIN.value))

        unions = db.scalars(select(Union).order_by(Union.slug.asc())).all()
        documents = db.scalars(select(Document).order_by(Document.title.asc())).all()
        notifications = db.scalars(select(Notification).order_by(Notification.subject.asc())).all()

    assert [item.slug for item in unions] == ["local-1", "local-2"]
    assert [item.title for item in documents] == ["Local 1 CBA", "Local 2 CBA"]
    assert [item.subject for item in notifications] == ["Union One", "Union Two"]


def _seed_tracking_and_sessions(session_factory, seeded):
    """Seed tracking policies, telemetry, raw queries, preferences, and sessions for both unions."""
    with session_factory() as db:
        apply_request_context(db, _auth(role=Role.SUPER_ADMIN.value))

        # Global (null-union) tracking policy plus a per-union override for union two.
        db.add(
            TrackingPolicy(
                union_id=None,
                tracking_mode=TrackingMode.BUG_AND_JOURNEY,
                privacy_mode=TrackingPrivacyMode.ANONYMIZED,
                member_choice_mode=MemberTrackingChoiceMode.BUG_ONLY_OR_FULL,
                raw_query_storage_mode=RawQueryStorageMode.DISABLED,
                default_member_preference=TrackingPreference.BUG_ONLY,
                allow_union_override=True,
            )
        )
        db.add(
            TrackingPolicy(
                union_id=seeded["union_two_id"],
                tracking_mode=TrackingMode.BOTH,
                privacy_mode=TrackingPrivacyMode.IDENTIFIED,
                member_choice_mode=MemberTrackingChoiceMode.NONE,
                raw_query_storage_mode=RawQueryStorageMode.ENABLED_IDENTIFIED,
                default_member_preference=TrackingPreference.FULL,
                allow_union_override=True,
            )
        )
        db.add_all(
            [
                UserTrackingPreference(user_id=seeded["user_one_id"], union_id=seeded["union_one_id"], preference=TrackingPreference.FULL),
                UserTrackingPreference(user_id=seeded["user_two_id"], union_id=seeded["union_two_id"], preference=TrackingPreference.OFF),
                TelemetryEvent(union_id=seeded["union_one_id"], category="usage_ux", event_type="query_completed", metadata_json={}),
                TelemetryEvent(union_id=seeded["union_two_id"], category="usage_ux", event_type="query_completed", metadata_json={}),
                TelemetryEvent(union_id=None, category="bug_journey", event_type="anonymous_event", metadata_json={}),
                RawQueryRecord(union_id=seeded["union_one_id"], route="/api/query", question_text="one", answer_text="one", metadata_json={}),
                RawQueryRecord(union_id=seeded["union_two_id"], route="/api/query", question_text="two", answer_text="two", metadata_json={}),
                AuthSession(user_id=seeded["user_one_id"], union_id=seeded["union_one_id"], session_secret_hash="hash-one", session_type=SessionType.MEMBER, expires_at=datetime.utcnow() + timedelta(days=1)),
                AuthSession(user_id=seeded["user_two_id"], union_id=seeded["union_two_id"], session_secret_hash="hash-two", session_type=SessionType.MEMBER, expires_at=datetime.utcnow() + timedelta(days=1)),
                AuthSession(user_id=seeded["user_one_id"], union_id=None, session_secret_hash="hash-super", session_type=SessionType.SUPER_ADMIN, expires_at=datetime.utcnow() + timedelta(days=1)),
            ]
        )
        db.commit()


def test_postgres_rls_isolates_tracking_and_session_tables(postgres_env):
    seeded = _seed_sample_data(postgres_env["owner_session_factory"])
    _seed_tracking_and_sessions(postgres_env["owner_session_factory"], seeded)

    with postgres_env["app_session_factory"]() as db:
        apply_request_context(
            db,
            _auth(role=Role.UNION_ADMIN.value, union_id=seeded["union_one_id"], user_id=seeded["user_one_id"]),
        )

        telemetry = db.scalars(select(TelemetryEvent)).all()
        raw_queries = db.scalars(select(RawQueryRecord)).all()
        preferences = db.scalars(select(UserTrackingPreference)).all()
        sessions = db.scalars(select(AuthSession)).all()
        policies = db.scalars(select(TrackingPolicy)).all()

        cross_raw = db.scalar(select(RawQueryRecord).where(RawQueryRecord.union_id == seeded["union_two_id"]))
        cross_session = db.scalar(select(AuthSession).where(AuthSession.union_id == seeded["union_two_id"]))

    # Union-A sees only its own union rows (anonymous null-union telemetry remains visible).
    assert {item.union_id for item in telemetry} == {seeded["union_one_id"], None}
    assert {item.union_id for item in raw_queries} == {seeded["union_one_id"]}
    assert {item.union_id for item in preferences} == {seeded["union_one_id"]}
    # Sessions: own union only — super-admin (null-union) sessions are NOT exposed to a tenant.
    assert {item.union_id for item in sessions} == {seeded["union_one_id"]}
    # Tracking policies: only the global (null-union) row resolves for union-A; union-two's
    # override is hidden and union-A has no override of its own.
    assert {item.union_id for item in policies} == {None}
    assert cross_raw is None
    assert cross_session is None


def test_postgres_rls_global_tracking_policy_resolves_under_super_admin(postgres_env):
    seeded = _seed_sample_data(postgres_env["owner_session_factory"])
    _seed_tracking_and_sessions(postgres_env["owner_session_factory"], seeded)

    # The shared seed intentionally leaves union one without an override (the isolation test
    # relies on that); add one here so super-admin visibility of *both* overrides is exercised.
    with postgres_env["owner_session_factory"]() as db:
        apply_request_context(db, _auth(role=Role.SUPER_ADMIN.value))
        db.add(
            TrackingPolicy(
                union_id=seeded["union_one_id"],
                tracking_mode=TrackingMode.BUG_AND_JOURNEY,
                privacy_mode=TrackingPrivacyMode.ANONYMIZED,
                member_choice_mode=MemberTrackingChoiceMode.BUG_ONLY_OR_FULL,
                raw_query_storage_mode=RawQueryStorageMode.DISABLED,
                default_member_preference=TrackingPreference.BUG_ONLY,
                allow_union_override=True,
            )
        )
        db.commit()

    with postgres_env["app_session_factory"]() as db:
        apply_request_context(db, _auth(role=Role.SUPER_ADMIN.value))
        policies = db.scalars(select(TrackingPolicy)).all()
        sessions = db.scalars(select(AuthSession)).all()

    # Super-admin resolves the global policy, both union overrides, and all sessions.
    assert None in {item.union_id for item in policies}
    assert {seeded["union_one_id"], seeded["union_two_id"]} <= {item.union_id for item in policies if item.union_id}
    assert any(item.session_type == SessionType.SUPER_ADMIN for item in sessions)


def test_postgres_rls_blocks_cross_tenant_insert(postgres_env):
    seeded = _seed_sample_data(postgres_env["owner_session_factory"])

    with pytest.raises(SQLAlchemyError):
        with postgres_env["app_session_factory"]() as db:
            apply_request_context(
                db,
                _auth(
                    role=Role.UNION_ADMIN.value,
                    union_id=seeded["union_one_id"],
                    user_id=seeded["user_one_id"],
                ),
            )
            db.add(
                Document(
                    union_id=seeded["union_two_id"],
                    uploaded_by_user_id=seeded["user_one_id"],
                    title="Cross Tenant Attempt",
                    storage_key="local-2/cross.pdf",
                    content_type="application/pdf",
                    bytes_size=1,
                    status=DocumentStatus.ACTIVE,
                )
            )
            db.commit()
