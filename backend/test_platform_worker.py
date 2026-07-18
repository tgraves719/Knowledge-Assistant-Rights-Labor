from types import SimpleNamespace
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.platform.db import Base
from backend.platform.ingestion import IngestionService
from backend.platform.models import Document, DocumentStatus, IngestionJob, IngestionJobStatus, Role, Union, UnionMembership, User
from backend.platform.parsing import ParsedBlock, ParsedDocument, ParsedPage, ParserRegistry, PlainTextDocumentParser
from backend.platform.storage import LocalDiskStorage
from backend.platform.retrieval import TenantRetrievalService
from backend.platform.settings import PlatformSettings
from backend.platform.worker import process_pending_ingestion_jobs


class FakePdfParser:
    name = "fake_pdf"

    def can_parse(self, *, content_type: str | None, filename: str | None) -> bool:
        return str(content_type or "").lower() == "application/pdf" or str(filename or "").lower().endswith(".pdf")

    def parse_bytes(self, payload: bytes, *, content_type: str | None, filename: str | None) -> ParsedDocument:
        text = "Scanned schedule text."
        return ParsedDocument(
            parser_name=self.name,
            content_type=content_type or "application/pdf",
            text=text,
            pages=[ParsedPage(page_number=1, text=text, blocks=[ParsedBlock(text=text)])],
            metadata={"filename": filename},
        )

    def parse_file(self, file_path: Path, *, content_type: str | None, filename: str | None) -> ParsedDocument:
        return self.parse_bytes(file_path.read_bytes(), content_type=content_type, filename=filename or file_path.name)


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
        secret_encryption_key="MDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDA=",
        sentinel_email_from="sentinel@example.com",
        sentinel_email_enabled=False,
        bootstrap_super_admin_emails=[],
        session_cookie_name="karl_session",
        member_session_idle_seconds=604800,
        union_admin_session_idle_seconds=259200,
        super_admin_session_idle_seconds=3600,
    )


def test_worker_prioritizes_lightweight_jobs_before_heavy_ocr_jobs(tmp_path):
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
        parsers=ParserRegistry([PlainTextDocumentParser(), FakePdfParser()]),
        inline_parse_max_bytes=settings.inline_parse_max_bytes,
        ocr_auto_retry_enabled=settings.ocr_auto_retry_enabled,
        ocr_auto_retry_max_attempts=settings.ocr_auto_retry_max_attempts,
    )
    container = SimpleNamespace(session_factory=SessionLocal, ingestion=ingestion)

    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        user = User(email="admin@example.com", full_name="Union Admin")
        db.add_all([union, user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=user.id, role=Role.UNION_ADMIN))
        db.flush()

        heavy_storage = storage.save_bytes(union.slug, "scan.pdf", b"%PDF-1.4 fake pdf")
        light_storage = storage.save_bytes(union.slug, "notice.txt", b"Meeting is at noon.")
        heavy_document = Document(
            union_id=union.id,
            uploaded_by_user_id=user.id,
            title="scan.pdf",
            storage_key=heavy_storage.key,
            content_type="application/pdf",
            bytes_size=12_000_000,
            status=DocumentStatus.PROCESSING,
            metadata_json={"page_count": 120, "scan_likelihood": "high", "review_status": "retry_pending"},
        )
        light_document = Document(
            union_id=union.id,
            uploaded_by_user_id=user.id,
            title="notice.txt",
            storage_key=light_storage.key,
            content_type="text/plain",
            bytes_size=128,
            status=DocumentStatus.PROCESSING,
            metadata_json={"review_status": "pending_ingestion"},
        )
        db.add_all([heavy_document, light_document])
        db.flush()

        heavy_job = IngestionJob(
            union_id=union.id,
            document_id=heavy_document.id,
            requested_by_user_id=user.id,
            status=IngestionJobStatus.PENDING,
            metadata_json={"trigger": "retry", "ocr_enabled": True},
        )
        light_job = IngestionJob(
            union_id=union.id,
            document_id=light_document.id,
            requested_by_user_id=user.id,
            status=IngestionJobStatus.PENDING,
            metadata_json={"trigger": "upload"},
        )
        db.add_all([heavy_job, light_job])
        db.commit()
        heavy_job_id = heavy_job.id
        light_job_id = light_job.id

    summary = process_pending_ingestion_jobs(container, limit=1)

    assert summary["processed"] == 1
    assert summary["failed"] == 0

    with SessionLocal() as db:
        updated_heavy_job = db.get(IngestionJob, heavy_job_id)
        updated_light_job = db.get(IngestionJob, light_job_id)

        assert updated_light_job is not None
        assert updated_light_job.status == IngestionJobStatus.SUCCEEDED
        assert updated_heavy_job is not None
        assert updated_heavy_job.status == IngestionJobStatus.PENDING
