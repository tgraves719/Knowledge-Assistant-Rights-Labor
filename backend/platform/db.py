"""Database setup for the production multi-tenant foundation."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from backend.platform.settings import PlatformSettings

if TYPE_CHECKING:
    from backend.platform.auth import AuthContext


Base = declarative_base()

TENANT_RLS_TABLES = (
    "documents",
    "ingestion_jobs",
    "chunk_embeddings",
    "chats",
    "messages",
    "provider_configs",
    "quota_policies",
    "usage_events",
    "audit_events",
    "security_events",
    "notifications",
)


def create_session_factory(settings: PlatformSettings) -> tuple[Engine | None, sessionmaker[Session] | None]:
    if not settings.db_enabled:
        return None, None

    engine = create_engine(
        settings.postgres_url,
        future=True,
        pool_pre_ping=True,
    )
    return engine, sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def create_all(engine: Engine) -> None:
    from backend.platform import models  # noqa: F401

    Base.metadata.create_all(engine)


def get_rls_statements() -> list[str]:
    tenant_tables_sql = ", ".join(f"'{table}'" for table in TENANT_RLS_TABLES)
    return [
        "CREATE EXTENSION IF NOT EXISTS vector",
        "ALTER TABLE unions ENABLE ROW LEVEL SECURITY",
        "ALTER TABLE unions FORCE ROW LEVEL SECURITY",
        "ALTER TABLE union_memberships ENABLE ROW LEVEL SECURITY",
        "ALTER TABLE union_memberships FORCE ROW LEVEL SECURITY",
        *[f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY" for table in TENANT_RLS_TABLES],
        *[f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY" for table in TENANT_RLS_TABLES],
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_policies WHERE policyname = 'tenant_isolation_unions'
            ) THEN
                CREATE POLICY tenant_isolation_unions ON unions
                USING (
                    current_setting('app.current_role', true) = 'super_admin'
                    OR id::text = current_setting('app.current_union_id', true)
                )
                WITH CHECK (
                    current_setting('app.current_role', true) = 'super_admin'
                    OR id::text = current_setting('app.current_union_id', true)
                );
            END IF;
        END $$;
        """,
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_policies WHERE policyname = 'tenant_isolation_union_memberships'
            ) THEN
                CREATE POLICY tenant_isolation_union_memberships ON union_memberships
                USING (
                    current_setting('app.current_role', true) = 'super_admin'
                    OR union_id::text = current_setting('app.current_union_id', true)
                )
                WITH CHECK (
                    current_setting('app.current_role', true) = 'super_admin'
                    OR union_id::text = current_setting('app.current_union_id', true)
                );
            END IF;
        END $$;
        """,
        f"""
        DO $$
        DECLARE
            tbl text;
        BEGIN
            FOREACH tbl IN ARRAY ARRAY[{tenant_tables_sql}]
            LOOP
                IF NOT EXISTS (
                    SELECT 1 FROM pg_policies WHERE policyname = 'tenant_isolation_' || tbl
                ) THEN
                    EXECUTE format(
                        'CREATE POLICY %I ON %I USING (
                            current_setting(''app.current_role'', true) = ''super_admin''
                            OR union_id::text = current_setting(''app.current_union_id'', true)
                        ) WITH CHECK (
                            current_setting(''app.current_role'', true) = ''super_admin''
                            OR union_id::text = current_setting(''app.current_union_id'', true)
                        )',
                        'tenant_isolation_' || tbl,
                        tbl
                    );
                END IF;
            END LOOP;
        END $$;
        """,
    ]


def apply_rls_policies(engine: Engine) -> None:
    with engine.begin() as connection:
        for statement in get_rls_statements():
            connection.execute(text(statement))


def apply_request_context(session: Session | None, auth: "AuthContext | None") -> None:
    if session is None:
        return
    bind = session.get_bind()
    if bind is None or bind.dialect.name != "postgresql":
        return

    role = getattr(auth, "role", None) or ""
    union_id = getattr(auth, "union_id", None) or ""
    user_id = getattr(auth, "user_id", None) or ""

    session.execute(
        text(
            """
            SELECT
                set_config('app.current_role', :role, true),
                set_config('app.current_union_id', :union_id, true),
                set_config('app.current_user_id', :user_id, true)
            """
        ),
        {
            "role": str(role),
            "union_id": str(union_id),
            "user_id": str(user_id),
        },
    )


def apply_service_bootstrap_context(session: Session | None) -> None:
    if session is None:
        return

    apply_request_context(
        session,
        type(
            "BootstrapAuthContext",
            (),
            {
                "role": "super_admin",
                "union_id": "",
                "user_id": "",
            },
        )(),
    )


@contextmanager
def session_scope(session_factory: sessionmaker[Session] | None) -> Iterator[Session | None]:
    if session_factory is None:
        yield None
        return
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
