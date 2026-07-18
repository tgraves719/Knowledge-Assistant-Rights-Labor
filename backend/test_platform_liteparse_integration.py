import base64
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.platform.db import Base
from backend.platform.ingestion import IngestionService
from backend.platform.models import ChunkEmbedding, Document, DocumentStatus, IngestionJob, IngestionJobStatus, Role, Union, UnionMembership, User
from backend.platform.parsing import LiteParseDocumentParser, ParserRegistry, PlainTextDocumentParser
from backend.platform.retrieval import TenantRetrievalService
from backend.platform.settings import PlatformSettings
from backend.platform.storage import LocalDiskStorage


SAMPLE_PDF = Path("data/contracts/local7_safeway_pueblo_clerks_2022/source/SW+Pueblo+Clerks+2022.2025.pdf")
VENDORED_LITEPARSE = Path("vendor/liteparse/dist/src/index.js")


pytestmark = pytest.mark.skipif(
    not SAMPLE_PDF.exists() or not VENDORED_LITEPARSE.exists(),
    reason="LiteParse integration test requires vendored LiteParse build and sample PDF.",
)


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
        liteparse_executable=f"node {VENDORED_LITEPARSE.resolve()}",
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
        bootstrap_super_admin_emails=[],
    )


def test_liteparse_processes_deferred_pdf_job(tmp_path):
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
    ingestion = IngestionService(
        storage=storage,
        retrieval=retrieval,
        parsers=ParserRegistry(
            [
                PlainTextDocumentParser(),
                LiteParseDocumentParser(settings.liteparse_executable),
            ]
        ),
        inline_parse_max_bytes=settings.inline_parse_max_bytes,
    )

    pdf_bytes = SAMPLE_PDF.read_bytes()

    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        user = User(email="admin@example.com", full_name="Admin")
        db.add_all([union, user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=user.id, role=Role.UNION_ADMIN))
        db.commit()

        stored = storage.save_bytes(union.slug, SAMPLE_PDF.name, pdf_bytes)
        result = ingestion.register_upload(
            db,
            union=union,
            uploaded_by_user_id=user.id,
            filename=SAMPLE_PDF.name,
            content_type="application/pdf",
            payload=pdf_bytes,
            storage_key=stored.key,
        )
        db.commit()

        assert result.ingestion_job.status == IngestionJobStatus.PENDING
        assert result.document.status == DocumentStatus.PROCESSING
        assert result.ingestion_job.metadata_json["parser"] == "liteparse"

        artifact_key = ingestion.process_job(db, union=union, job=result.ingestion_job)
        db.commit()

        document = db.get(Document, result.document.id)
        ingestion_job = db.get(IngestionJob, result.ingestion_job.id)
        chunks = db.query(ChunkEmbedding).filter(ChunkEmbedding.document_id == document.id).order_by(ChunkEmbedding.chunk_index.asc()).all()

        assert artifact_key
        assert document is not None
        assert document.status == DocumentStatus.ACTIVE
        assert ingestion_job is not None
        assert ingestion_job.status == IngestionJobStatus.SUCCEEDED
        assert ingestion_job.metadata_json["artifact_key"] == artifact_key
        assert ingestion_job.metadata_json["page_count"] >= 1
        assert len(chunks) >= 1
        assert storage.open(artifact_key).exists()
