import pytest
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


def test_structure_extraction_keeps_body_text_on_heading_lines():
    """A section whose whole body sits on its heading line must survive.

    The materializer emits sections as single lines ("Section 121. Discharge
    for Just Cause. No employee shall be discharged..."). The old extractor
    swallowed the entire line into section_title and never emitted the text as
    content, silently deleting 22% of the clerks book from the index --
    including the just-cause clause, the strongest protection a member has.
    """
    from backend.platform.document_structure import analyze_parsed_document
    from backend.platform.parsing import PlainTextDocumentParser

    text = (
        "ARTICLE 20 DISCHARGE\n"
        "Section 121. Discharge for Just Cause. No employee shall be discharged "
        "except for just cause.\n"
        "Section 122. Notice. The Employer shall provide written notice.\n"
    )
    parsed = PlainTextDocumentParser().parse_bytes(
        text.encode("utf-8"), content_type="text/markdown", filename="contract.md"
    )
    st = analyze_parsed_document(parsed, filename="contract.md", content_type="text/markdown")

    joined = " ".join(s.text for s in st.sections)
    assert "No employee shall be discharged except for just cause." in joined
    assert "The Employer shall provide written notice." in joined

    s121 = next(s for s in st.sections if s.section_num == "121")
    # Title is the first sentence, not the whole body.
    assert s121.section_title == "Discharge for Just Cause."
    assert "just cause" in s121.text.lower()


def test_reingestion_preserves_standing_safety_approval(tmp_path):
    """A human safety approval must survive re-ingestion of unchanged risks.

    Every retry re-runs the safety scanner. Before this fix that overwrote the
    admin's approve_member_access decision and silently re-locked the document
    -- in the pilot, members lost access to their contract PDFs after each
    re-index because the contract prints the Plan Administrator's phone
    number, which re-trips the sensitive-data pattern every time.
    """
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
    from backend.platform.guardrails import GuardrailService

    ingestion = IngestionService(
        storage=storage,
        retrieval=TenantRetrievalService(),
        parsers=ParserRegistry([PlainTextDocumentParser()]),
        guardrails=GuardrailService(),
        inline_parse_max_bytes=settings.inline_parse_max_bytes,
    )

    # Trips the sensitive-phone pattern the same way the real contracts do.
    contract_text = (
        "ARTICLE 1 BENEFITS\n"
        "Section 1. Contact the Plan Administrator at 303-430-9334 with any "
        "questions about the health plan described in this Agreement.\n"
    ).encode("utf-8")

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
        assert document.metadata_json.get("sensitive_data_risk") is True

        # Admin approves, recording what risks were reviewed.
        metadata = dict(document.metadata_json)
        metadata.update(
            {
                "member_visible": True,
                "ready_for_query": True,
                "sensitive_data_risk": False,
                "safety_review_status": "resolved",
                "safety_status": "reviewed_safe",
                "safety_override": {
                    "approved_by_user_id": user.id,
                    "previous_prompt_injection_risk": False,
                    "previous_sensitive_data_risk": True,
                },
            }
        )
        document.metadata_json = metadata
        db.commit()

        # Re-ingest: same content, scanner re-flags the same phone number.
        retry = ingestion.enqueue_retry(
            db, union=union, document=document, source_job=result.ingestion_job, requested_by_user_id=user.id, ocr_enabled=False
        )
        db.commit()
        ingestion.process_job(db, union=union, job=retry)
        db.commit()

        document = db.get(Document, document.id)
        assert document.metadata_json.get("sensitive_data_risk") is False
        assert document.metadata_json.get("safety_review_status") == "resolved"
        assert document.metadata_json.get("member_visible") is True
        assert document.metadata_json.get("safety_override", {}).get("previous_sensitive_data_risk") is True


def test_contract_pack_upload_preserves_article_hierarchy():
    """A contract pack keeps its articles; flat markdown cannot.

    The materializer's markdown export of the same contract has zero ARTICLE
    heading lines -- only "Section N." -- so text extraction recovered 2
    garbled articles out of cross-reference prose and the contract explorer
    had no tree to render. Ingesting the structured export instead keeps all
    58 articles with their real titles.
    """
    import json

    from backend.platform.document_structure import analyze_contract_pack, _looks_like_contract_pack
    from backend.platform.parsing import ContractPackDocumentParser

    pack = {
        "contract_id": "local7_test_2022",
        "effective_version_id": "effective_test",
        "sections": [
            {
                "anchor_id": "a1_s1",
                "article_num": "1",
                "article_title": "RECOGNITION AND EXCLUSIONS",
                "section_num": "1",
                "content_markdown": "The Employer recognizes the Union as sole bargaining agent.",
                "provenance": {"page": 3},
            },
            {
                "anchor_id": "a20_s121",
                "article_num": "20",
                "article_title": "DISCHARGE",
                "section_num": "121",
                "content_markdown": "Discharge for Just Cause. No employee shall be discharged except for just cause.",
                "provenance": {"page": 20},
            },
        ],
    }
    payload = json.dumps(pack).encode("utf-8")

    parser = ContractPackDocumentParser()
    assert parser.can_parse(content_type="application/json", filename="contract.json")

    parsed = parser.parse_bytes(payload, content_type="application/json", filename="contract.json")
    assert _looks_like_contract_pack(parsed.metadata["contract_pack"])
    # Body text still flows into retrieval/safety/quality exactly as prose does.
    assert "just cause" in parsed.text.lower()

    structure = analyze_contract_pack(parsed.metadata["contract_pack"], filename="contract.json")
    assert structure.structure_mode == "legal_structured"
    assert structure.total_articles == 2
    assert structure.article_titles["20"] == "DISCHARGE"

    discharge = next(s for s in structure.sections if s.section_num == "121")
    assert discharge.article_num == "20"
    assert discharge.section_title == "Discharge for Just Cause."
    # Printed page carried through for citations and the PDF pane.
    assert discharge.page_start == 20


def test_contract_pack_labels_strip_the_leading_section_marker():
    """Section bodies open with their own "Section N." marker.

    Taking the first sentence verbatim produced table-of-contents entries
    reading "Section 4." for every row, which is no help to a member
    scanning for the clause they need.
    """
    from backend.platform.document_structure import analyze_contract_pack

    pack = {
        "contract_id": "local7_test_2022",
        "sections": [
            {
                "article_num": "2",
                "article_title": "JURISDICTION",
                "section_num": "4",
                "content_markdown": "Section 4. Work Jurisdiction. The Employer agrees not to subcontract.",
                "provenance": [{"pdf": "SW+Pueblo+Clerks+2022.2025.pdf", "pdf_page": 4, "source_type": "base"}],
            }
        ],
    }

    structure = analyze_contract_pack(pack, filename="contract.json")
    section = structure.sections[0]

    assert section.section_title == "Work Jurisdiction."
    # provenance is a LIST of source refs, not a dict.
    assert section.page_start == 4


def _wage_row(classification_key, classification_name, step_name, threshold, effective_date, rate, *, amended=False, page=62):
    return {
        "columns": {
            "classification_key": classification_key,
            "classification_name": classification_name,
            "step_name": step_name,
            "step_type": "hours" if threshold is not None else "fixed",
            "threshold_value": threshold,
            "effective_date": effective_date,
            "rate": rate,
            "row_type": "step_row" if threshold is not None else "rate_row",
        },
        "amendments": ["moa_2025"] if amended else [],
        "provenance": [{"pdf": "Book.pdf", "pdf_page": page, "source_type": "base"}],
    }


def test_contract_pack_wage_rows_become_current_rate_sections():
    """Amended pay rates live ONLY in tables.appendix_a_wage_rows.

    The markdown wage tables in sections[] still show the base-book columns
    (2022-2024), so indexing only section text meant KARL could never cite a
    current rate. The structured rows must be rendered into their own
    Appendix A sections, current schedule first and labelled as current.
    """
    from backend.platform.document_structure import (
        APPENDIX_A_ARTICLE_NUM,
        analyze_contract_pack,
    )

    pack = {
        "contract_id": "local7_test_2022",
        "sections": [
            {
                "anchor_id": "a1_s1",
                "article_num": "1",
                "article_title": "RECOGNITION",
                "section_num": "1",
                "content_markdown": "The Employer recognizes the Union.",
                "provenance": [{"pdf": "Book.pdf", "pdf_page": 3, "source_type": "base"}],
            },
        ],
        "tables": {
            "appendix_a_wage_rows": {
                "table_id": "appendix_a_wage_rows",
                "columns": [],
                "rows": [
                    # Deliberately out of ladder order: the pack interleaves steps.
                    _wage_row("all_purpose_clerk", "ALL PURPOSE CLERK", "After 1560 hours", 1560, "2024-01-21", 18.00),
                    _wage_row("all_purpose_clerk", "ALL PURPOSE CLERK", "Start", 0, "2024-01-21", 17.00),
                    _wage_row("all_purpose_clerk", "ALL PURPOSE CLERK", "After 520 hours", 520, "2024-01-21", 17.50),
                    _wage_row("all_purpose_clerk", "ALL PURPOSE CLERK", "Start", 0, "2025-07-05", 17.25, amended=True),
                    _wage_row("head_clerk", "HEAD CLERK", "Rate", None, "2025-07-05", 24.11, amended=True, page=63),
                ],
            }
        },
    }

    structure = analyze_contract_pack(pack, filename="contract.json")
    wage_sections = [s for s in structure.sections if s.article_num == APPENDIX_A_ARTICLE_NUM]
    assert len(wage_sections) == 2
    assert APPENDIX_A_ARTICLE_NUM in structure.article_titles

    apc = next(s for s in wage_sections if s.section_num == "all_purpose_clerk")
    paragraphs = apc.text.split("\n\n")
    # Current (amended) schedule leads and says so; superseded ones say so too.
    assert "effective 7/5/2025 (current rates, as amended)" in paragraphs[0]
    assert "$17.25" in paragraphs[0]
    assert "superseded" in paragraphs[1]
    # Ladder order restored regardless of pack row order.
    start = paragraphs[1].index("Start")
    after_520 = paragraphs[1].index("After 520 hours")
    after_1560 = paragraphs[1].index("After 1560 hours")
    assert start < after_520 < after_1560
    # Row-level provenance carries the real printed page.
    assert apc.page_start == 62
    assert next(s for s in wage_sections if s.section_num == "head_clerk").page_start == 63
    # Regular sections are untouched.
    assert any(s.section_num == "1" and s.article_num == "1" for s in structure.sections)


def test_contract_pack_wage_table_snapshots_are_skipped_when_rows_exist():
    """The pack carries the wage tables THREE ways: structured rows, markdown
    snapshots with doc_type "appendix", and duplicate "cba" sections sharing
    the appendix anchors. The snapshots are mis-anchored to the last article
    and show only base-book rates, so when the rows exist neither copy may be
    indexed — a member asking about pay must never retrieve a superseded rate
    presented as current."""
    from backend.platform.document_structure import APPENDIX_A_ARTICLE_NUM, analyze_contract_pack

    stale_table = "| CLASSIFICATION | 1/23/2022 |\n| --- | --- |\n| HEAD CLERK | $22.51 |"
    pack = {
        "contract_id": "local7_test_2022",
        "sections": [
            {
                "anchor_id": "a58_s175",
                "article_num": "58",
                "article_title": "TERM OF AGREEMENT",
                "section_num": "175",
                "content_markdown": "Section 175. In the event of an Act of God the parties shall meet.",
                "provenance": [{"pdf": "Book.pdf", "pdf_page": 58, "source_type": "base"}],
            },
            {
                "anchor_id": "a58_s175_p10",
                "article_num": "58",
                "article_title": "TERM OF AGREEMENT",
                "section_num": "175",
                "citation": "Article 58, Section 175, Part 10",
                "doc_type": "cba",
                "content_markdown": stale_table,
                "provenance": [{"pdf": "Book.pdf", "pdf_page": 59, "source_type": "base"}],
            },
            {
                "anchor_id": "a58_s175_p10",
                "article_num": "58",
                "article_title": "TERM OF AGREEMENT",
                "section_num": "175",
                "citation": "Appendix A Wage Table - Article 58, Section 175, Part 10",
                "doc_type": "appendix",
                "content_markdown": stale_table,
                "provenance": [{"pdf": "Book.pdf", "pdf_page": 59, "source_type": "base"}],
            },
        ],
        "tables": {
            "appendix_a_wage_rows": {
                "table_id": "appendix_a_wage_rows",
                "columns": [],
                "rows": [_wage_row("head_clerk", "HEAD CLERK", "Rate", None, "2025-07-05", 24.11, amended=True)],
            }
        },
    }

    structure = analyze_contract_pack(pack, filename="contract.json")
    # The real Section 175 survives; both snapshot copies are gone.
    survivors = [s for s in structure.sections if s.article_num == "58"]
    assert len(survivors) == 1
    assert "Act of God" in survivors[0].text
    assert not any("$22.51" in s.text for s in structure.sections)
    assert any(s.article_num == APPENDIX_A_ARTICLE_NUM and "$24.11" in s.text for s in structure.sections)


def test_contract_pack_without_wage_rows_keeps_table_snapshots():
    """With no structured rows, the markdown snapshot is the only wage data
    the pack has — it must keep flowing into the index."""
    from backend.platform.document_structure import analyze_contract_pack

    pack = {
        "contract_id": "local7_test_2022",
        "sections": [
            {
                "anchor_id": "a58_s175_p10",
                "article_num": "58",
                "article_title": "TERM OF AGREEMENT",
                "section_num": "175",
                "citation": "Appendix A Wage Table - Article 58, Section 175, Part 10",
                "doc_type": "appendix",
                "content_markdown": "| CLASSIFICATION | 1/23/2022 |\n| --- | --- |\n| HEAD CLERK | $22.51 |",
                "provenance": [{"pdf": "Book.pdf", "pdf_page": 59, "source_type": "base"}],
            },
        ],
    }

    structure = analyze_contract_pack(pack, filename="contract.json")
    assert any("$22.51" in s.text for s in structure.sections)


def test_contract_pack_sections_inherit_the_previous_page_when_provenance_is_null():
    """~50 sections per book ship pdf_page null in the pack itself (Sunday
    Premium, Travel Pay, Discharge...). Sections run in reading order, so the
    previous section's page is at worst one off — and a citation that opens
    the PDF a page early beats one with no page button at all."""
    from backend.platform.document_structure import analyze_contract_pack

    pack = {
        "contract_id": "local7_test_2022",
        "sections": [
            {
                "article_num": "13",
                "article_title": "SIXTH DAY",
                "section_num": "28",
                "content_markdown": "Section 28. Sixth day premium applies.",
                "provenance": [{"pdf": "Book.pdf", "pdf_page": 11, "source_type": "base"}],
            },
            {
                "article_num": "13",
                "article_title": "SUNDAY PREMIUM",
                "section_num": "29",
                "content_markdown": "Section 29. Sunday work is paid at premium rates.",
                "provenance": [{"pdf": "Book.pdf", "pdf_page": None, "source_type": "base"}],
            },
        ],
    }

    structure = analyze_contract_pack(pack, filename="contract.json")
    sunday = next(s for s in structure.sections if s.section_num == "29")
    assert sunday.page_start == 11
    # A sourced page must never be overwritten by an inherited one.
    assert next(s for s in structure.sections if s.section_num == "28").page_start == 11


def test_contract_pack_letters_of_understanding_get_their_own_outline_group():
    """LOU sections have no article/section numbers, so the outline (which
    groups on article_num) never showed them even though their text was
    indexed. They now group under one heading, one entry per letter."""
    from backend.platform.document_structure import (
        LOU_ARTICLE_NUM,
        LOU_ARTICLE_TITLE,
        analyze_contract_pack,
    )

    pack = {
        "contract_id": "local7_test_2022",
        "sections": [
            {
                "anchor_id": "lou_15_p1",
                "article_num": None,
                "section_num": None,
                "citation": "Letter of Understanding 15: Beverage Stewards, Part 1",
                "doc_type": "lou",
                "content_markdown": "## 15. Beverage Stewards. A Beverage Steward shall be defined as an associate.",
                "provenance": [{"pdf": "Book.pdf", "pdf_page": None, "source_type": "base"}],
            },
            {
                "anchor_id": "lou_15_p2",
                "article_num": None,
                "section_num": None,
                "citation": "Letter of Understanding 15: Beverage Stewards, Part 2",
                "doc_type": "lou",
                "content_markdown": "The Employer retains the right to add the position to stores.",
                "provenance": [{"pdf": "Book.pdf", "pdf_page": None, "source_type": "base"}],
            },
        ],
    }

    structure = analyze_contract_pack(pack, filename="contract.json")
    lou_sections = [s for s in structure.sections if s.article_num == LOU_ARTICLE_NUM]
    assert len(lou_sections) == 2
    assert structure.article_titles[LOU_ARTICLE_NUM] == LOU_ARTICLE_TITLE
    # Parts share the letter's number so the outline shows one entry per letter.
    assert {s.section_num for s in lou_sections} == {"15"}
    assert lou_sections[0].section_title == "Beverage Stewards"


def test_contract_pack_ingest_indexes_current_wage_rates_end_to_end(tmp_path):
    """Upload → parse → structure → chunks, for a pack with wage rows.

    The retrievable index must carry the amended (current) rate with the
    appendix's own article identity and the row-level printed page — this is
    what the member outline, citations and the PDF pane all read."""
    import json

    from backend.platform.document_structure import APPENDIX_A_ARTICLE_NUM
    from backend.platform.parsing import ContractPackDocumentParser

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
        parsers=ParserRegistry([ContractPackDocumentParser(), PlainTextDocumentParser()]),
        inline_parse_max_bytes=settings.inline_parse_max_bytes,
    )

    pack = {
        "contract_id": "local7_test_2022",
        "effective_version_id": "effective_test",
        "sections": [
            {
                "anchor_id": "a8_s17",
                "article_num": "8",
                "article_title": "RATES OF PAY",
                "section_num": "17",
                "content_markdown": "Section 17. The minimum wages shall be as set forth in Appendix A.",
                "provenance": [{"pdf": "Book.pdf", "pdf_page": 8, "source_type": "base"}],
            },
        ],
        "tables": {
            "appendix_a_wage_rows": {
                "table_id": "appendix_a_wage_rows",
                "columns": [],
                "rows": [
                    _wage_row("all_purpose_clerk", "ALL PURPOSE CLERK", "Start", 0, "2024-01-21", 17.00),
                    _wage_row("all_purpose_clerk", "ALL PURPOSE CLERK", "Start", 0, "2025-07-05", 17.25, amended=True),
                ],
            }
        },
    }
    payload = json.dumps(pack).encode("utf-8")

    with SessionLocal() as db:
        union = Union(slug="local-1", name="Local 1", union_local_id="local-1")
        user = User(email="admin@example.com", full_name="Admin")
        db.add_all([union, user])
        db.flush()
        db.add(UnionMembership(union_id=union.id, user_id=user.id, role=Role.UNION_ADMIN))
        db.commit()

        stored = storage.save_bytes(union.slug, "contract.json", payload)
        result = ingestion.register_upload(
            db,
            union=union,
            uploaded_by_user_id=user.id,
            filename="contract.json",
            content_type="application/json",
            payload=payload,
            storage_key=stored.key,
        )
        db.commit()

        # JSON is not inline-ingestable; drive the deferred job like the worker does.
        ingestion.process_job(db, union=union, job=result.ingestion_job)
        db.commit()

        document = db.get(Document, result.document.id)
        assert document.status == DocumentStatus.ACTIVE
        chunks = db.query(ChunkEmbedding).filter(ChunkEmbedding.document_id == document.id).all()
        wage_chunks = [c for c in chunks if (c.metadata_json or {}).get("article_num") == APPENDIX_A_ARTICLE_NUM]
        assert wage_chunks, "wage rows must land in the retrievable index"
        apc = next(c for c in wage_chunks if (c.metadata_json or {}).get("section_num") == "all_purpose_clerk")
        # The current (amended) rate is what the member must retrieve.
        assert "$17.25" in apc.chunk_text
        assert "current rates, as amended" in apc.chunk_text
        # Citations and the PDF pane read source_page.
        assert apc.metadata_json.get("source_page") == 62
        # The outline groups on these; both articles must be present.
        assert any((c.metadata_json or {}).get("article_num") == "8" for c in chunks)


def test_contract_pack_parser_rejects_unrelated_json():
    """Arbitrary JSON must not be mistaken for a contract pack."""
    import json

    from backend.platform.parsing import ContractPackDocumentParser, ParserUnavailableError

    payload = json.dumps({"hello": "world", "items": [1, 2, 3]}).encode("utf-8")
    with pytest.raises(ParserUnavailableError):
        ContractPackDocumentParser().parse_bytes(
            payload, content_type="application/json", filename="notes.json"
        )
