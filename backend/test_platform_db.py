from backend.platform.db import TENANT_RLS_TABLES, get_rls_statements
from backend.platform.models import IngestionJob
from backend.platform.settings import get_platform_settings


def test_rls_statements_cover_tenant_tables():
    statements = get_rls_statements()
    joined = "\n".join(statements)

    assert "tenant_isolation_unions" in joined
    assert "tenant_isolation_union_memberships" in joined
    for table in TENANT_RLS_TABLES:
        assert f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY" in joined
        assert table in joined


def test_platform_settings_disable_auto_create_by_default(monkeypatch):
    monkeypatch.delenv("KARL_AUTO_CREATE_TABLES", raising=False)
    settings = get_platform_settings()
    assert settings.auto_create_tables is False


def test_ingestion_job_model_registered():
    assert IngestionJob.__tablename__ == "ingestion_jobs"
