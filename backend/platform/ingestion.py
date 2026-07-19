"""Document ingestion orchestration for tenant uploads."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.platform.auth import AuthContext
from backend.platform.document_structure import analyze_parsed_document
from backend.platform.models import Document, DocumentStatus, IngestionJob, IngestionJobStatus, Notification, NotificationStatus, Union
from backend.platform.parsing import LiteParseDocumentParser, ParsedDocument, ParserRegistry

OCR_CAPABLE_CONTENT_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/tiff",
}

UNPARSEABLE_WARNING_KEYWORDS = (
    "encrypted",
    "password",
    "corrupt",
    "damaged",
    "malformed",
    "unsupported",
    "cannot parse",
    "parse failed",
)


@dataclass
class UploadIngestionResult:
    document: Document
    ingestion_job: IngestionJob
    artifact_key: str | None = None


class IngestionService:
    def __init__(
        self,
        *,
        storage,
        retrieval,
        parsers: ParserRegistry,
        guardrails=None,
        sentinel=None,
        inline_parse_max_bytes: int = 1_000_000,
        ocr_auto_retry_enabled: bool = True,
        ocr_auto_retry_max_attempts: int = 1,
    ):
        self.storage = storage
        self.retrieval = retrieval
        self.parsers = parsers
        self.guardrails = guardrails
        self.sentinel = sentinel
        self.inline_parse_max_bytes = max(1_024, int(inline_parse_max_bytes))
        self.ocr_auto_retry_enabled = bool(ocr_auto_retry_enabled)
        self.ocr_auto_retry_max_attempts = max(0, int(ocr_auto_retry_max_attempts))

    @staticmethod
    def _system_auth_context(*, union_id: str, user_id: str | None) -> AuthContext:
        return AuthContext(
            user_id=user_id,
            email=None,
            full_name=None,
            role="union_admin",
            union_id=union_id,
            union_slug=None,
            source="ingestion_service",
            is_authenticated=True,
        )

    def _resolve_parser(self, *, content_type: str, filename: str, job: IngestionJob | None = None):
        parser = self.parsers.resolve(content_type=content_type, filename=filename)
        if parser is None:
            return None
        if isinstance(parser, LiteParseDocumentParser) and bool(((job.metadata_json or {}) if job is not None else {}).get("ocr_enabled")):
            return parser.__class__(parser.executable, ocr_enabled=True)
        return parser

    def register_upload(
        self,
        db: Session,
        *,
        union: Union,
        uploaded_by_user_id: str | None,
        filename: str,
        content_type: str,
        payload: bytes,
        storage_key: str,
    ) -> UploadIngestionResult:
        document = Document(
            union_id=union.id,
            uploaded_by_user_id=uploaded_by_user_id,
            title=filename,
            storage_key=storage_key,
            content_type=content_type,
            bytes_size=len(payload),
            status=DocumentStatus.PROCESSING,
            metadata_json={"ingestion_mode": "deferred", "review_status": "pending_ingestion"},
        )
        db.add(document)
        db.flush()

        parser = self._resolve_parser(content_type=content_type, filename=filename)
        parser_name = getattr(parser, "name", None)
        should_inline = parser is not None and len(payload) <= self.inline_parse_max_bytes and parser_name == "plain_text"

        job = IngestionJob(
            union_id=union.id,
            document_id=document.id,
            requested_by_user_id=uploaded_by_user_id,
            status=IngestionJobStatus.PENDING,
            metadata_json={
                "trigger": "upload",
                "filename": filename,
                "parser": parser_name,
                "mode": "inline" if should_inline else "deferred",
            },
        )
        db.add(job)
        db.flush()

        artifact_key = None
        if should_inline and parser is not None:
            artifact_key = self._execute_parse(
                db,
                union=union,
                document=document,
                job=job,
                parser_name=parser_name or "unknown",
                parsed=parser.parse_bytes(payload, content_type=content_type, filename=filename),
            )
        elif parser is None:
            job.metadata_json = {
                **(job.metadata_json or {}),
                "deferred_reason": "no_parser_available",
            }
            document.metadata_json = {
                **(document.metadata_json or {}),
                "deferred_reason": "no_parser_available",
            }

        return UploadIngestionResult(document=document, ingestion_job=job, artifact_key=artifact_key)

    def process_job(self, db: Session, *, union: Union, job: IngestionJob) -> str | None:
        document = job.document_id and db.get(Document, job.document_id)
        if document is None:
            raise ValueError("Ingestion job has no document.")

        parser = self._resolve_parser(content_type=document.content_type, filename=document.title, job=job)
        if parser is None:
            raise ValueError("No parser available for document.")

        stored = self.storage.open(document.storage_key)
        parsed = parser.parse_file(stored, content_type=document.content_type, filename=document.title)
        return self._execute_parse(
            db,
            union=union,
            document=document,
            job=job,
            parser_name=parser.name,
            parsed=parsed,
        )

    def _execute_parse(
        self,
        db: Session,
        *,
        union: Union,
        document: Document,
        job: IngestionJob,
        parser_name: str,
        parsed: ParsedDocument,
    ) -> str:
        job.status = IngestionJobStatus.RUNNING
        job.started_at = job.started_at or datetime.utcnow()

        artifact_key = self.storage.save_json(
            union.slug,
            f"{document.id}/parse/{job.id}.json",
            parsed.to_dict(),
        ).key
        structure = analyze_parsed_document(
            parsed,
            filename=document.title,
            content_type=document.content_type,
        )
        quality = self._assess_parse_quality(document=document, parsed=parsed, ocr_enabled=bool((job.metadata_json or {}).get("ocr_enabled")))
        safety = self.guardrails.assess_document_safety(parsed.text) if self.guardrails is not None else None
        auto_retry_job = None
        if self._should_auto_retry_with_ocr(document=document, job=job, quality=quality) and not (safety and safety.prompt_injection_risk):
            auto_retry_job = self.enqueue_retry(
                db,
                union=union,
                document=document,
                source_job=job,
                requested_by_user_id=job.requested_by_user_id,
                ocr_enabled=True,
                automatic=True,
                retry_reason="auto_ocr_retry",
            )
        chunk_count = 0
        if quality["status"] != "needs_review":
            chunk_count = self.retrieval.ingest_document(
                db,
                union_id=union.id,
                document_id=document.id,
                text=parsed.text,
                pages=[
                    {
                        "page_number": page.page_number,
                        "text": page.text,
                    }
                    for page in parsed.pages
                ],
                structured_sections=[{**section.to_metadata(), "text": section.text} for section in structure.sections],
                metadata={
                    "parser_name": parser_name,
                    "page_count": parsed.page_count,
                    "artifact_key": artifact_key,
                    "warnings": list(parsed.warnings),
                    "quality_status": quality["status"],
                    "quality_reason": quality["reason"],
                    "ocr_status": quality["ocr_status"],
                    "scan_likelihood": quality["scan_likelihood"],
                    "document_title": document.title,
                    "safety_status": safety.safety_status if safety is not None else "clear",
                    "prompt_injection_risk": bool(safety.prompt_injection_risk) if safety is not None else False,
                    "sensitive_data_risk": bool(safety.sensitive_data_risk) if safety is not None else False,
                    "member_visible": bool(safety.member_visible) if safety is not None else True,
                    "safety_reasons": list(safety.safety_reasons) if safety is not None else [],
                    "safety_review_status": safety.safety_review_status if safety is not None else "not_required",
                    **structure.to_document_metadata(),
                },
            )

        job.status = IngestionJobStatus.SUCCEEDED
        job.completed_at = datetime.utcnow()
        job.error_message = None
        job.metadata_json = {
            **(job.metadata_json or {}),
            "parser": parser_name,
            "artifact_key": artifact_key,
            "chunk_count": chunk_count,
            "page_count": parsed.page_count,
            "warnings": [*list(parsed.warnings), *quality["warnings"]],
            "quality_status": quality["status"],
            "quality_reason": quality["reason"],
            "ocr_status": quality["ocr_status"],
            "scan_likelihood": quality["scan_likelihood"],
            "recommended_action": quality["recommended_action"],
            "safety_status": safety.safety_status if safety is not None else "clear",
            "prompt_injection_risk": bool(safety.prompt_injection_risk) if safety is not None else False,
            "sensitive_data_risk": bool(safety.sensitive_data_risk) if safety is not None else False,
            "member_visible": bool(safety.member_visible) if safety is not None else True,
            "safety_reasons": list(safety.safety_reasons) if safety is not None else [],
            "safety_review_status": safety.safety_review_status if safety is not None else "not_required",
            "redacted_preview": safety.redacted_preview[:1200] if safety is not None else "",
            **structure.to_document_metadata(),
        }
        document_metadata = {
            **(document.metadata_json or {}),
            "ingestion_mode": job.metadata_json.get("mode"),
            "parser": parser_name,
            "artifact_key": artifact_key,
            "page_count": parsed.page_count,
            "quality_status": quality["status"],
            "quality_reason": quality["reason"],
            "quality_warnings": quality["warnings"],
            "ocr_status": quality["ocr_status"],
            "scan_likelihood": quality["scan_likelihood"],
            "recommended_action": quality["recommended_action"],
            "ready_for_query": quality["status"] != "needs_review" and not bool(safety.prompt_injection_risk if safety is not None else False),
            "review_status": "not_required" if quality["status"] != "needs_review" else "needs_review",
            "safety_status": safety.safety_status if safety is not None else "clear",
            "prompt_injection_risk": bool(safety.prompt_injection_risk) if safety is not None else False,
            "sensitive_data_risk": bool(safety.sensitive_data_risk) if safety is not None else False,
            "member_visible": bool(safety.member_visible) if safety is not None else True,
            "safety_reasons": list(safety.safety_reasons) if safety is not None else [],
            "safety_review_status": safety.safety_review_status if safety is not None else "not_required",
            "redacted_preview": safety.redacted_preview[:1200] if safety is not None else "",
            **structure.to_document_metadata(),
        }
        if safety is not None and safety.prompt_injection_risk:
            document_metadata.update(
                {
                    "review_status": "escalated",
                    "recommended_action": safety.recommended_action,
                    "ready_for_query": False,
                }
            )
        elif safety is not None and safety.sensitive_data_risk:
            document_metadata.update(
                {
                    "review_status": "needs_review",
                    "recommended_action": safety.recommended_action,
                }
            )

        # A standing human approval survives re-ingestion. Without this, every
        # retry re-ran the scanner and silently re-locked documents an admin
        # had already reviewed -- in the pilot, members lost access to their
        # own contract PDFs after each re-index. The override is honored only
        # while the fresh scan finds nothing beyond what the reviewer saw:
        # a newly detected prompt-injection risk, or a sensitive-data risk
        # that was not previously approved, voids it and requires re-review.
        prior_override = (document.metadata_json or {}).get("safety_override")
        if isinstance(prior_override, dict) and safety is not None:
            new_injection = bool(safety.prompt_injection_risk)
            new_sensitive = bool(safety.sensitive_data_risk)
            approved_injection = bool(prior_override.get("previous_prompt_injection_risk"))
            approved_sensitive = bool(prior_override.get("previous_sensitive_data_risk"))
            if (not new_injection or approved_injection) and (not new_sensitive or approved_sensitive):
                document_metadata.update(
                    {
                        "member_visible": True,
                        "ready_for_query": True,
                        "review_status": "resolved",
                        "safety_status": "reviewed_safe",
                        "safety_review_status": "resolved",
                        "safety_reasons": [],
                        "prompt_injection_risk": False,
                        "sensitive_data_risk": False,
                        "recommended_action": "Approved for full member access after manual safety review.",
                        "safety_override": prior_override,
                    }
                )
        if auto_retry_job is not None:
            document.status = DocumentStatus.PROCESSING
            document_metadata.update(
                {
                    "quality_status": "retrying_with_ocr",
                    "recommended_action": "await_ocr_retry",
                    "ready_for_query": False,
                    "auto_retry_pending": True,
                    "latest_retry_job_id": auto_retry_job.id,
                    "ocr_retry_attempts": self._count_ocr_attempts(db, document_id=document.id),
                    "ocr_status": "retry_queued",
                    "scan_likelihood": quality["scan_likelihood"],
                    "review_status": "retrying_with_ocr",
                }
            )
            job.metadata_json = {
                **(job.metadata_json or {}),
                "auto_retry_enqueued": True,
                "auto_retry_job_id": auto_retry_job.id,
            }
            self._notify_job_result(
                db,
                union_id=union.id,
                requested_by_user_id=job.requested_by_user_id,
                document=document,
                outcome="retrying",
                detail="An OCR retry was queued automatically after low-confidence extraction.",
            )
        else:
            document.status = DocumentStatus.ACTIVE if quality["status"] != "needs_review" else DocumentStatus.FAILED
            self._notify_job_result(
                db,
                union_id=union.id,
                requested_by_user_id=job.requested_by_user_id,
                document=document,
                outcome=quality["status"],
                detail=quality["recommended_action"],
            )
        document.metadata_json = document_metadata
        if safety is not None and safety.sensitive_data_risk and self.sentinel is not None:
            self.sentinel.record_event(
                db,
                self._system_auth_context(union_id=union.id, user_id=job.requested_by_user_id),
                event_type="document_sensitive_data_flagged",
                details={
                    "document_id": document.id,
                    "document_title": document.title,
                    "job_id": job.id,
                    "safety_reasons": safety.safety_reasons,
                },
            )
        if safety is not None and safety.prompt_injection_risk and self.sentinel is not None:
            self.sentinel.record_event(
                db,
                self._system_auth_context(union_id=union.id, user_id=job.requested_by_user_id),
                event_type="document_prompt_injection_blocked",
                details={
                    "document_id": document.id,
                    "document_title": document.title,
                    "job_id": job.id,
                    "safety_reasons": safety.safety_reasons,
                },
            )
        if quality["status"] == "needs_review" and auto_retry_job is None and self.sentinel is not None:
            self.sentinel.record_event(
                db,
                self._system_auth_context(union_id=union.id, user_id=job.requested_by_user_id),
                event_type="ingestion_review_required",
                details={
                    "document_id": document.id,
                    "document_title": document.title,
                    "job_id": job.id,
                    "recommended_action": quality["recommended_action"],
                },
            )
        return artifact_key

    def fail_job(self, db: Session, *, document: Document, job: IngestionJob, error: Exception) -> None:
        job.status = IngestionJobStatus.FAILED
        job.completed_at = datetime.utcnow()
        job.error_message = str(error)
        job.metadata_json = {
            **(job.metadata_json or {}),
            "failure": type(error).__name__,
        }
        document.status = DocumentStatus.FAILED
        document.metadata_json = {
            **(document.metadata_json or {}),
            "last_error": str(error),
            "ready_for_query": False,
            "review_status": "failed",
        }
        self._notify_job_result(
            db,
            union_id=document.union_id,
            requested_by_user_id=job.requested_by_user_id,
            document=document,
            outcome="failed",
            detail=str(error),
        )
        if self.sentinel is not None:
            self.sentinel.record_event(
                db,
                self._system_auth_context(union_id=document.union_id, user_id=job.requested_by_user_id),
                event_type="ingestion_failed",
                details={
                    "document_id": document.id,
                    "document_title": document.title,
                    "job_id": job.id,
                    "error": str(error),
                },
            )

    def enqueue_retry(
        self,
        db: Session,
        *,
        union: Union,
        document: Document,
        source_job: IngestionJob,
        requested_by_user_id: str | None,
        ocr_enabled: bool = False,
        automatic: bool = False,
        retry_reason: str | None = None,
    ) -> IngestionJob:
        if source_job.status == IngestionJobStatus.RUNNING and not automatic:
            raise ValueError("Cannot retry a running ingestion job.")

        parser = self._resolve_parser(content_type=document.content_type, filename=document.title, job=source_job)
        retry_job = IngestionJob(
            union_id=union.id,
            document_id=document.id,
            requested_by_user_id=requested_by_user_id,
            status=IngestionJobStatus.PENDING,
            metadata_json={
                "trigger": "retry",
                "source_job_id": source_job.id,
                "filename": document.title,
                "parser": getattr(parser, "name", None),
                "mode": "deferred",
                "ocr_enabled": bool(ocr_enabled),
                "automatic_retry": bool(automatic),
                "retry_reason": retry_reason,
            },
        )
        if parser is None:
            retry_job.metadata_json["deferred_reason"] = "no_parser_available"
        db.add(retry_job)
        db.flush()

        if document.status != DocumentStatus.ACTIVE:
            document.status = DocumentStatus.PROCESSING
        document.metadata_json = {
            **(document.metadata_json or {}),
            "reingest_pending": True,
            "latest_retry_job_id": retry_job.id,
            "ready_for_query": False,
            "review_status": "retry_pending",
        }
        return retry_job

    def _count_ocr_attempts(self, db: Session, *, document_id: str) -> int:
        jobs = db.scalars(select(IngestionJob).where(IngestionJob.document_id == document_id)).all()
        return sum(1 for existing_job in jobs if bool((existing_job.metadata_json or {}).get("ocr_enabled")))

    def _should_auto_retry_with_ocr(self, *, document: Document, job: IngestionJob, quality: dict) -> bool:
        if not self.ocr_auto_retry_enabled:
            return False
        if quality["status"] != "needs_review" or not quality.get("auto_retry_recommended"):
            return False
        if bool((job.metadata_json or {}).get("ocr_enabled")):
            return False
        parser_name = str((job.metadata_json or {}).get("parser") or "")
        if parser_name != LiteParseDocumentParser.name:
            return False
        prior_attempts = int((document.metadata_json or {}).get("ocr_retry_attempts") or 0)
        return prior_attempts < self.ocr_auto_retry_max_attempts

    def _assess_parse_quality(self, *, document: Document, parsed: ParsedDocument, ocr_enabled: bool) -> dict:
        warnings = list(parsed.warnings)
        page_count = max(parsed.page_count, 0)
        text_length = len(str(parsed.text or "").strip())
        nonempty_pages = sum(1 for page in parsed.pages if str(page.text or "").strip())
        empty_page_ratio = 0.0 if page_count <= 0 else max(0.0, (page_count - nonempty_pages) / page_count)
        chars_per_page = text_length / max(page_count, 1)
        bounding_box_count = sum(int((page.metadata or {}).get("bounding_box_count") or 0) for page in parsed.pages)
        avg_bounding_boxes_per_page = bounding_box_count / max(page_count, 1)

        status = "ready"
        recommended_action = "none"
        reason = "acceptable_parse"
        content_type = str(document.content_type or "").lower()
        needs_ocr_retry = content_type in OCR_CAPABLE_CONTENT_TYPES and not ocr_enabled
        ocr_status = "attempted" if ocr_enabled else ("recommended" if needs_ocr_retry else "not_applicable")
        scan_likelihood = self._classify_scan_likelihood(
            content_type=content_type,
            text_length=text_length,
            nonempty_pages=nonempty_pages,
            empty_page_ratio=empty_page_ratio,
            chars_per_page=chars_per_page,
            avg_bounding_boxes_per_page=avg_bounding_boxes_per_page,
        )
        has_unparseable_warning = self._has_unparseable_warning(warnings)
        auto_retry_recommended = False

        if has_unparseable_warning:
            status = "needs_review"
            warnings.append("The parser reported a likely unrecoverable document issue.")
            reason = "parser_reported_unparseable"
            recommended_action = "manual_review_unparseable"
            ocr_status = "not_recommended"
        elif text_length == 0 or nonempty_pages == 0:
            status = "needs_review"
            warnings.append("No usable text was extracted from the document.")
            reason = "no_text_after_parse" if not ocr_enabled else "no_text_after_ocr"
            recommended_action = "retry_with_ocr" if needs_ocr_retry else "manual_review_after_ocr"
            auto_retry_recommended = needs_ocr_retry and scan_likelihood == "high"
        elif page_count >= 3 and empty_page_ratio >= 0.4:
            status = "needs_review"
            warnings.append("A large portion of pages produced little or no text.")
            reason = "sparse_page_coverage" if not ocr_enabled else "sparse_page_coverage_after_ocr"
            recommended_action = "retry_with_ocr" if needs_ocr_retry else "manual_review_after_ocr"
            auto_retry_recommended = needs_ocr_retry and scan_likelihood == "high"
        elif page_count >= 3 and chars_per_page < 200:
            status = "needs_review"
            warnings.append("Extracted text density is too low for reliable retrieval.")
            reason = "low_text_density" if not ocr_enabled else "low_text_density_after_ocr"
            if needs_ocr_retry and scan_likelihood in {"high", "medium"}:
                recommended_action = "retry_with_ocr"
                auto_retry_recommended = scan_likelihood == "high"
            else:
                recommended_action = "manual_review_after_ocr" if ocr_enabled else "manual_review"
        elif chars_per_page < 500:
            status = "warning"
            warnings.append("Extracted text density is lower than expected.")
            reason = "low_text_density_warning"
            recommended_action = "review_parse_quality"

        if has_unparseable_warning:
            ocr_status = "not_recommended"
        elif status == "needs_review":
            ocr_status = "recommended" if needs_ocr_retry else ("attempted_needs_review" if ocr_enabled else "unavailable")
        elif status == "warning":
            ocr_status = "attempted_warning" if ocr_enabled else ("recommended_optional" if needs_ocr_retry else "not_applicable")
        elif ocr_enabled:
            ocr_status = "attempted_ready"
        else:
            ocr_status = "not_needed"

        return {
            "status": status,
            "reason": reason,
            "ocr_status": ocr_status,
            "scan_likelihood": scan_likelihood,
            "auto_retry_recommended": auto_retry_recommended,
            "warnings": warnings,
            "recommended_action": recommended_action,
        }

    @staticmethod
    def _has_unparseable_warning(warnings: list[str]) -> bool:
        warning_blob = " ".join(str(item or "").strip().lower() for item in warnings)
        return any(keyword in warning_blob for keyword in UNPARSEABLE_WARNING_KEYWORDS)

    @staticmethod
    def _classify_scan_likelihood(
        *,
        content_type: str,
        text_length: int,
        nonempty_pages: int,
        empty_page_ratio: float,
        chars_per_page: float,
        avg_bounding_boxes_per_page: float,
    ) -> str:
        if content_type.startswith("image/"):
            return "high"
        if text_length == 0 or nonempty_pages == 0:
            return "high"
        if empty_page_ratio >= 0.4 and avg_bounding_boxes_per_page >= 1:
            return "high"
        if chars_per_page < 120 and avg_bounding_boxes_per_page >= 1:
            return "high"
        if chars_per_page < 200 or empty_page_ratio >= 0.4:
            return "medium"
        return "low"

    def _notify_job_result(
        self,
        db: Session | None,
        *,
        union_id: str,
        requested_by_user_id: str | None,
        document: Document,
        outcome: str,
        detail: str,
    ) -> None:
        if db is None:
            return
        subject = {
            "ready": f"Document ready: {document.title}",
            "warning": f"Document ready with warnings: {document.title}",
            "needs_review": f"Document needs review: {document.title}",
            "retrying": f"OCR retry queued: {document.title}",
            "failed": f"Document failed: {document.title}",
        }.get(outcome, f"Document update: {document.title}")
        db.add(
            Notification(
                union_id=union_id,
                user_id=requested_by_user_id,
                channel="in_app",
                subject=subject,
                body=f"Ingestion outcome: {outcome}. Detail: {detail}",
                status=NotificationStatus.PENDING,
            )
        )
