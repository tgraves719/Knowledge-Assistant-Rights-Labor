import base64
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.platform.db import Base
from backend.platform.ingestion import IngestionService
from backend.platform.models import ChunkEmbedding, Document, DocumentStatus, IngestionJob, IngestionJobStatus, Notification, Role, Union, UnionMembership, User
from backend.platform.parsing import LiteParseDocumentParser, ParsedBlock, ParsedDocument, ParsedPage, ParserRegistry, PlainTextDocumentParser
from backend.platform.retrieval import TenantRetrievalService
from backend.platform.settings import PlatformSettings
from backend.platform.storage import LocalDiskStorage


class FakePdfParser:
    name = "fake_pdf"

    def can_parse(self, *, content_type: str | None, filename: str | None) -> bool:
        return str(content_type or "").lower() == "application/pdf" or str(filename or "").lower().endswith(".pdf")

    def parse_bytes(self, payload: bytes, *, content_type: str | None, filename: str | None) -> ParsedDocument:
        text = "Vacation bidding rules are described on page one."
        return ParsedDocument(
            parser_name=self.name,
            content_type=content_type or "application/pdf",
            text=text,
            pages=[ParsedPage(page_number=1, text=text, blocks=[ParsedBlock(text=text)])],
            metadata={"filename": filename},
        )

    def parse_file(self, file_path: Path, *, content_type: str | None, filename: str | None) -> ParsedDocument:
        return self.parse_bytes(file_path.read_bytes(), content_type=content_type, filename=filename or file_path.name)


class EmptyPdfParser(FakePdfParser):
    name = "empty_pdf"

    def parse_bytes(self, payload: bytes, *, content_type: str | None, filename: str | None) -> ParsedDocument:
        return ParsedDocument(
            parser_name=self.name,
            content_type=content_type or "application/pdf",
            text="",
            pages=[ParsedPage(page_number=1, text="", blocks=[])],
            metadata={"filename": filename},
        )


class OcrSensitiveLiteParseParser(LiteParseDocumentParser):
    def __init__(self, executable: str | None, *, ocr_enabled: bool = False):
        super().__init__(executable, ocr_enabled=ocr_enabled)

    def parse_bytes(self, payload: bytes, *, content_type: str | None, filename: str | None) -> ParsedDocument:
        text = "Scanned grievance procedure and wage schedule." if self.ocr_enabled else ""
        return ParsedDocument(
            parser_name=self.name,
            content_type=content_type or "application/pdf",
            text=text,
            pages=[ParsedPage(page_number=1, text=text, blocks=[ParsedBlock(text=text)] if text else [])],
            metadata={"filename": filename, "ocr_enabled": self.ocr_enabled},
        )

    def parse_file(self, file_path: Path, *, content_type: str | None, filename: str | None) -> ParsedDocument:
        return self.parse_bytes(file_path.read_bytes(), content_type=content_type, filename=filename or file_path.name)


class OcrStillWeakLiteParseParser(LiteParseDocumentParser):
    def __init__(self, executable: str | None, *, ocr_enabled: bool = False):
        super().__init__(executable, ocr_enabled=ocr_enabled)

    def parse_bytes(self, payload: bytes, *, content_type: str | None, filename: str | None) -> ParsedDocument:
        return ParsedDocument(
            parser_name=self.name,
            content_type=content_type or "application/pdf",
            text="",
            pages=[ParsedPage(page_number=1, text="", blocks=[])],
            metadata={"filename": filename, "ocr_enabled": self.ocr_enabled},
        )

    def parse_file(self, file_path: Path, *, content_type: str | None, filename: str | None) -> ParsedDocument:
        return self.parse_bytes(file_path.read_bytes(), content_type=content_type, filename=filename or file_path.name)


class UnparseableLiteParseParser(LiteParseDocumentParser):
    def __init__(self, executable: str | None, *, ocr_enabled: bool = False):
        super().__init__(executable, ocr_enabled=ocr_enabled)

    def parse_bytes(self, payload: bytes, *, content_type: str | None, filename: str | None) -> ParsedDocument:
        return ParsedDocument(
            parser_name=self.name,
            content_type=content_type or "application/pdf",
            text="",
            pages=[ParsedPage(page_number=1, text="", blocks=[])],
            warnings=["Document appears encrypted and cannot be parsed."],
            metadata={"filename": filename, "ocr_enabled": self.ocr_enabled},
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
        secret_encryption_key=base64.urlsafe_b64encode(b"0" * 32).decode("utf-8"),
        sentinel_email_from="sentinel@example.com",
        sentinel_email_enabled=False,
        bootstrap_super_admin_emails=[],
    )


def test_ingestion_service_processes_deferred_pdf_job(tmp_path):
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
        parsers=ParserRegistry([FakePdfParser()]),
        inline_parse_max_bytes=settings.inline_parse_max_bytes,
    )

    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        user = User(email="admin@example.com", full_name="Admin")
        db.add_all([union, user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=user.id, role=Role.UNION_ADMIN))
        db.commit()

        stored = storage.save_bytes(union.slug, "scan.pdf", b"%PDF-1.4 fake pdf")
        result = ingestion.register_upload(
            db,
            union=union,
            uploaded_by_user_id=user.id,
            filename="scan.pdf",
            content_type="application/pdf",
            payload=b"%PDF-1.4 fake pdf",
            storage_key=stored.key,
        )
        db.commit()

        assert result.ingestion_job.status == IngestionJobStatus.PENDING
        assert result.document.status == DocumentStatus.PROCESSING

        artifact_key = ingestion.process_job(db, union=union, job=result.ingestion_job)
        db.commit()

        document = db.get(Document, result.document.id)
        ingestion_job = db.get(IngestionJob, result.ingestion_job.id)
        chunks = db.query(ChunkEmbedding).filter(ChunkEmbedding.document_id == document.id).order_by(ChunkEmbedding.chunk_index.asc()).all()
        notifications = db.query(Notification).filter(Notification.union_id == union.id).all()

        assert artifact_key
        assert document is not None
        assert document.status == DocumentStatus.ACTIVE
        assert ingestion_job is not None
        assert ingestion_job.status == IngestionJobStatus.SUCCEEDED
        assert ingestion_job.metadata_json["parser"] == "fake_pdf"
        assert ingestion_job.metadata_json["artifact_key"] == artifact_key
        assert len(chunks) >= 1
        assert "Vacation bidding rules" in chunks[0].chunk_text
        assert storage.open(artifact_key).exists()
        assert notifications


def test_ingestion_service_extracts_legal_structure_for_contract_like_upload(tmp_path):
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
        parsers=ParserRegistry([PlainTextDocumentParser()]),
        inline_parse_max_bytes=settings.inline_parse_max_bytes,
    )

    contract_text = """
    COLLECTIVE BARGAINING AGREEMENT

    ARTICLE 17 VACATIONS
    SECTION 17.1 Eligibility
    Vacation time is earned on the basis of having completed an anniversary year.
    Weekly vacation requests take preference over daily vacation requests.

    SECTION 17.2 Payment
    Employees who have earned vacation may receive pay during the workweek immediately preceding their vacation.

    ARTICLE 27 SENIORITY
    SECTION 27.1 General
    Seniority governs vacation preference when employees request the same vacation period.
    """.strip().encode("utf-8")

    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        user = User(email="admin@example.com", full_name="Admin")
        db.add_all([union, user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=user.id, role=Role.UNION_ADMIN))
        db.commit()

        stored = storage.save_bytes(union.slug, "contract.txt", contract_text)
        result = ingestion.register_upload(
            db,
            union=union,
            uploaded_by_user_id=user.id,
            filename="contract.txt",
            content_type="text/plain",
            payload=contract_text,
            storage_key=stored.key,
        )
        db.commit()

        document = db.get(Document, result.document.id)
        ingestion_job = db.get(IngestionJob, result.ingestion_job.id)
        chunks = db.query(ChunkEmbedding).filter(ChunkEmbedding.document_id == document.id).order_by(ChunkEmbedding.chunk_index.asc()).all()

        assert document is not None
        assert document.status == DocumentStatus.ACTIVE
        assert document.metadata_json["document_type"] == "legal_contract"
        assert document.metadata_json["structure_mode"] == "legal_structured"
        assert document.metadata_json["total_articles"] >= 2
        assert document.metadata_json["total_sections"] >= 3
        assert ingestion_job is not None
        assert ingestion_job.metadata_json["document_type"] == "legal_contract"
        assert chunks
        assert any((chunk.metadata_json or {}).get("article_num") == "17" for chunk in chunks)
        assert any((chunk.metadata_json or {}).get("section_num") == "17.1" for chunk in chunks)
        assert any("vacation" in " ".join((chunk.metadata_json or {}).get("topic_tags") or []).lower() for chunk in chunks)


def test_ingestion_service_marks_low_quality_pdf_for_review(tmp_path):
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
        parsers=ParserRegistry([EmptyPdfParser()]),
        inline_parse_max_bytes=settings.inline_parse_max_bytes,
    )

    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        user = User(email="admin@example.com", full_name="Admin")
        db.add_all([union, user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=user.id, role=Role.UNION_ADMIN))
        db.commit()

        stored = storage.save_bytes(union.slug, "scan.pdf", b"%PDF-1.4 fake pdf")
        result = ingestion.register_upload(
            db,
            union=union,
            uploaded_by_user_id=user.id,
            filename="scan.pdf",
            content_type="application/pdf",
            payload=b"%PDF-1.4 fake pdf",
            storage_key=stored.key,
        )
        artifact_key = ingestion.process_job(db, union=union, job=result.ingestion_job)
        db.commit()

        document = db.get(Document, result.document.id)
        ingestion_job = db.get(IngestionJob, result.ingestion_job.id)
        chunks = db.query(ChunkEmbedding).filter(ChunkEmbedding.document_id == document.id).all()
        notifications = db.query(Notification).filter(Notification.union_id == union.id).all()

        assert artifact_key
        assert document is not None
        assert document.status == DocumentStatus.FAILED
        assert document.metadata_json["quality_status"] == "needs_review"
        assert document.metadata_json["quality_reason"] == "no_text_after_parse"
        assert document.metadata_json["ocr_status"] == "recommended"
        assert document.metadata_json["scan_likelihood"] == "high"
        assert document.metadata_json["recommended_action"] == "retry_with_ocr"
        assert document.metadata_json["ready_for_query"] is False
        assert ingestion_job is not None
        assert ingestion_job.status == IngestionJobStatus.SUCCEEDED
        assert ingestion_job.metadata_json["quality_status"] == "needs_review"
        assert ingestion_job.metadata_json["quality_reason"] == "no_text_after_parse"
        assert ingestion_job.metadata_json["ocr_status"] == "recommended"
        assert ingestion_job.metadata_json["scan_likelihood"] == "high"
        assert ingestion_job.metadata_json["recommended_action"] == "retry_with_ocr"
        assert chunks == []
        assert notifications
        assert "needs review" in notifications[-1].subject.lower()


def test_ingestion_service_auto_queues_ocr_retry_for_low_quality_liteparse_pdf(tmp_path):
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
        parsers=ParserRegistry([OcrSensitiveLiteParseParser("liteparse")]),
        inline_parse_max_bytes=settings.inline_parse_max_bytes,
        ocr_auto_retry_enabled=True,
        ocr_auto_retry_max_attempts=1,
    )

    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        user = User(email="admin@example.com", full_name="Admin")
        db.add_all([union, user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=user.id, role=Role.UNION_ADMIN))
        db.commit()

        stored = storage.save_bytes(union.slug, "scan.pdf", b"%PDF-1.4 fake pdf")
        result = ingestion.register_upload(
            db,
            union=union,
            uploaded_by_user_id=user.id,
            filename="scan.pdf",
            content_type="application/pdf",
            payload=b"%PDF-1.4 fake pdf",
            storage_key=stored.key,
        )
        artifact_key = ingestion.process_job(db, union=union, job=result.ingestion_job)
        db.commit()

        document = db.get(Document, result.document.id)
        initial_job = db.get(IngestionJob, result.ingestion_job.id)
        retry_job = db.get(IngestionJob, document.metadata_json["latest_retry_job_id"])
        notifications = db.query(Notification).filter(Notification.union_id == union.id).order_by(Notification.created_at.asc()).all()

        assert artifact_key
        assert document.status == DocumentStatus.PROCESSING
        assert document.metadata_json["quality_status"] == "retrying_with_ocr"
        assert document.metadata_json["quality_reason"] == "no_text_after_parse"
        assert document.metadata_json["ocr_status"] == "retry_queued"
        assert document.metadata_json["scan_likelihood"] == "high"
        assert document.metadata_json["recommended_action"] == "await_ocr_retry"
        assert document.metadata_json["ready_for_query"] is False
        assert initial_job.metadata_json["auto_retry_enqueued"] is True
        assert initial_job.metadata_json["quality_reason"] == "no_text_after_parse"
        assert initial_job.metadata_json["ocr_status"] == "recommended"
        assert initial_job.metadata_json["scan_likelihood"] == "high"
        assert retry_job is not None
        assert retry_job.metadata_json["automatic_retry"] is True
        assert retry_job.metadata_json["ocr_enabled"] is True
        assert notifications
        assert "ocr retry queued" in notifications[-1].subject.lower()


def test_ingestion_service_marks_manual_review_after_failed_ocr_retry(tmp_path):
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
        parsers=ParserRegistry([OcrStillWeakLiteParseParser("liteparse")]),
        inline_parse_max_bytes=settings.inline_parse_max_bytes,
        ocr_auto_retry_enabled=True,
        ocr_auto_retry_max_attempts=1,
    )

    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        user = User(email="admin@example.com", full_name="Admin")
        db.add_all([union, user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=user.id, role=Role.UNION_ADMIN))
        db.commit()

        stored = storage.save_bytes(union.slug, "scan.pdf", b"%PDF-1.4 fake pdf")
        result = ingestion.register_upload(
            db,
            union=union,
            uploaded_by_user_id=user.id,
            filename="scan.pdf",
            content_type="application/pdf",
            payload=b"%PDF-1.4 fake pdf",
            storage_key=stored.key,
        )
        ingestion.process_job(db, union=union, job=result.ingestion_job)
        db.flush()

        document = db.get(Document, result.document.id)
        retry_job = db.get(IngestionJob, document.metadata_json["latest_retry_job_id"])
        assert retry_job is not None

        artifact_key = ingestion.process_job(db, union=union, job=retry_job)
        db.commit()

        document = db.get(Document, result.document.id)
        final_retry_job = db.get(IngestionJob, retry_job.id)
        notifications = db.query(Notification).filter(Notification.union_id == union.id).order_by(Notification.created_at.asc()).all()

        assert artifact_key
        assert document is not None
        assert document.status == DocumentStatus.FAILED
        assert document.metadata_json["quality_status"] == "needs_review"
        assert document.metadata_json["quality_reason"] == "no_text_after_ocr"
        assert document.metadata_json["ocr_status"] == "attempted_needs_review"
        assert document.metadata_json["scan_likelihood"] == "high"
        assert document.metadata_json["recommended_action"] == "manual_review_after_ocr"
        assert document.metadata_json["ready_for_query"] is False
        assert final_retry_job is not None
        assert final_retry_job.status == IngestionJobStatus.SUCCEEDED
        assert final_retry_job.metadata_json["quality_status"] == "needs_review"
        assert final_retry_job.metadata_json["quality_reason"] == "no_text_after_ocr"
        assert final_retry_job.metadata_json["ocr_status"] == "attempted_needs_review"
        assert final_retry_job.metadata_json["scan_likelihood"] == "high"
        assert final_retry_job.metadata_json["recommended_action"] == "manual_review_after_ocr"
        assert notifications
        assert "needs review" in notifications[-1].subject.lower()


def test_ingestion_service_skips_ocr_retry_for_unparseable_pdf(tmp_path):
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
        parsers=ParserRegistry([UnparseableLiteParseParser("liteparse")]),
        inline_parse_max_bytes=settings.inline_parse_max_bytes,
        ocr_auto_retry_enabled=True,
        ocr_auto_retry_max_attempts=1,
    )

    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        user = User(email="admin@example.com", full_name="Admin")
        db.add_all([union, user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=user.id, role=Role.UNION_ADMIN))
        db.commit()

        stored = storage.save_bytes(union.slug, "broken.pdf", b"%PDF-1.4 fake pdf")
        result = ingestion.register_upload(
            db,
            union=union,
            uploaded_by_user_id=user.id,
            filename="broken.pdf",
            content_type="application/pdf",
            payload=b"%PDF-1.4 fake pdf",
            storage_key=stored.key,
        )
        artifact_key = ingestion.process_job(db, union=union, job=result.ingestion_job)
        db.commit()

        document = db.get(Document, result.document.id)
        ingestion_job = db.get(IngestionJob, result.ingestion_job.id)
        jobs = db.query(IngestionJob).filter(IngestionJob.document_id == document.id).all()

        assert artifact_key
        assert document is not None
        assert document.status == DocumentStatus.FAILED
        assert document.metadata_json["quality_status"] == "needs_review"
        assert document.metadata_json["quality_reason"] == "parser_reported_unparseable"
        assert document.metadata_json["ocr_status"] == "not_recommended"
        assert document.metadata_json["scan_likelihood"] == "high"
        assert document.metadata_json["recommended_action"] == "manual_review_unparseable"
        assert ingestion_job is not None
        assert ingestion_job.metadata_json["quality_reason"] == "parser_reported_unparseable"
        assert ingestion_job.metadata_json["ocr_status"] == "not_recommended"
        assert ingestion_job.metadata_json["recommended_action"] == "manual_review_unparseable"
        assert len(jobs) == 1
