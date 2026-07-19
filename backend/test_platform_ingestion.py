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
