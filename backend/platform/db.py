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

# Tracking/telemetry tables with a *legitimately* nullable union_id (the global tracking
# policy row, anonymous public telemetry, and super-admin-scoped preferences). A naive
# ``union_id = current_union_id`` predicate would hide those null-union rows from everyone and
# break effective-policy resolution under member context, so null-union rows resolve here for
# any tenant while non-null rows stay isolated to their union (or super-admin).
NULLABLE_UNION_RLS_TABLES = (
    "tracking_policies",
    "user_tracking_preferences",
    "telemetry_events",
    "raw_query_records",
)

# Append-only event/session tables are written by the application under contexts that are not
# yet (or never) tenant-scoped — e.g. telemetry recorded during an unauthenticated/failed login,
# or an auth_session created mid-login before a user context exists. For these we keep a
# restrictive read predicate (USING) but a permissive write predicate (WITH CHECK true): the
# privacy-relevant control is cross-tenant *read* isolation, and writes are fully app-controlled.
PERMISSIVE_WRITE_RLS_TABLES = frozenset(
    {"telemetry_events", "raw_query_records", "auth_sessions"}
)

# auth_sessions carries no global/config rows that tenants should see: super-admin sessions
# (union_id IS NULL) must remain invisible to union tenants, so its read predicate excludes the
# null-union escape hatch used by the tracking tables.
SESSION_RLS_TABLES = ("auth_sessions",)

# Invite codes are strictly tenant-owned (union_id NOT NULL) but must be resolvable by the
# anonymous join flow, which runs before any tenant context exists. Reads therefore permit the
# no-context case (current_setting returns NULL/'' for unset app.current_union_id) while an
# established tenant context stays isolated to its own union. Writes are app-controlled: the
# join endpoint increments use_count pre-auth, so WITH CHECK stays permissive like the other
# app-managed tables.
INVITE_RLS_TABLES = ("invite_codes",)

_SUPER_ADMIN_PREDICATE = "current_setting('app.current_role', true) = 'super_admin'"
_UNION_MATCH_PREDICATE = "union_id::text = current_setting('app.current_union_id', true)"


def _nullable_union_read_predicate() -> str:
    return f"{_SUPER_ADMIN_PREDICATE} OR union_id IS NULL OR {_UNION_MATCH_PREDICATE}"


def _session_read_predicate() -> str:
    return f"{_SUPER_ADMIN_PREDICATE} OR (union_id IS NOT NULL AND {_UNION_MATCH_PREDICATE})"


def _policy_statement(table: str, *, using_expr: str, check_expr: str) -> str:
    """Idempotently create the tenant-isolation policy for ``table`` (matches existing style)."""
    return f"""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_policies WHERE policyname = 'tenant_isolation_{table}'
            ) THEN
                CREATE POLICY tenant_isolation_{table} ON {table}
                USING (
                    {using_expr}
                )
                WITH CHECK (
                    {check_expr}
                );
            END IF;
        END $$;
        """


def tracking_and_session_rls_statements() -> list[str]:
    """RLS enable/force + policy statements for the tracking + session tables.

    Kept separate from :func:`get_rls_statements` so the Alembic migration that introduces this
    control can apply exactly the same SQL to existing databases.
    """
    tables = (*NULLABLE_UNION_RLS_TABLES, *SESSION_RLS_TABLES)
    statements: list[str] = []
    for table in tables:
        statements.append(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        statements.append(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
    for table in NULLABLE_UNION_RLS_TABLES:
        using_expr = _nullable_union_read_predicate()
        check_expr = "true" if table in PERMISSIVE_WRITE_RLS_TABLES else using_expr
        statements.append(_policy_statement(table, using_expr=using_expr, check_expr=check_expr))
    for table in SESSION_RLS_TABLES:
        using_expr = _session_read_predicate()
        check_expr = "true" if table in PERMISSIVE_WRITE_RLS_TABLES else using_expr
        statements.append(_policy_statement(table, using_expr=using_expr, check_expr=check_expr))
    return statements


def invite_rls_statements() -> list[str]:
    """RLS enable/force + policy statements for the invite-code table.

    Kept separate (like :func:`tracking_and_session_rls_statements`) so the Alembic migration
    that introduces invite codes applies exactly the same SQL to existing databases.
    """
    no_context_predicate = "coalesce(current_setting('app.current_union_id', true), '') = ''"
    using_expr = f"{_SUPER_ADMIN_PREDICATE} OR {no_context_predicate} OR {_UNION_MATCH_PREDICATE}"
    statements: list[str] = []
    for table in INVITE_RLS_TABLES:
        statements.append(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        statements.append(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        statements.append(_policy_statement(table, using_expr=using_expr, check_expr="true"))
    return statements


def create_session_factory(settings: PlatformSettings) -> tuple[Engine | None, sessionmaker[Session] | None]:
    if not settings.db_enabled:
        return None, None

    # Pool sizing applies only to real server-backed pools. SQLite (used by the
    # test suite) gets SingletonThreadPool/StaticPool, which reject max_overflow.
    pool_kwargs: dict[str, int] = {}
    if not settings.postgres_url.startswith("sqlite"):
        pool_kwargs["pool_size"] = settings.db_pool_size
        pool_kwargs["max_overflow"] = settings.db_max_overflow

    engine = create_engine(
        settings.postgres_url,
        future=True,
        pool_pre_ping=True,
        **pool_kwargs,
    )
    return engine, sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def create_all(engine: Engine) -> None:
    from backend.platform import models  # noqa: F401

    Base.metadata.create_all(engine)


def foundation_rls_statements() -> list[str]:
    """RLS statements for the migration-0001 foundation tables only.

    Migration 20260320_0001 iterates this list. It must never grow statements for tables
    created by later migrations (a fresh-database upgrade would fail at 0001) — later RLS
    additions belong in their own migration, like 20260613_0005.
    """
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


def get_rls_statements() -> list[str]:
    """The complete RLS statement set for a database whose schema is at head.

    Used by the app-startup path (``apply_rls_policies``), where every table already exists.
    Migrations must NOT use this aggregate — each migration pins its own statement list.
    """
    return [
        *foundation_rls_statements(),
        *tracking_and_session_rls_statements(),
        *invite_rls_statements(),
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
