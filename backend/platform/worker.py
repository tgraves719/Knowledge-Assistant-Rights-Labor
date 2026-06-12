"""Deferred ingestion worker helpers."""

from __future__ import annotations

from sqlalchemy import select

from backend.platform.db import apply_service_bootstrap_context
from backend.platform.models import Document, IngestionJob, IngestionJobStatus, Union
from backend.platform.queueing import ingestion_job_priority
from backend.platform.service_container import build_service_container


def process_pending_ingestion_jobs(container, *, limit: int = 25) -> dict:
    if container.session_factory is None:
        return {"processed": 0, "failed": 0, "queued_retries": 0}

    processed = 0
    failed = 0
    queued_retries = 0
    with container.session_factory() as db:
        apply_service_bootstrap_context(db)
        jobs = db.scalars(
            select(IngestionJob)
            .where(IngestionJob.status == IngestionJobStatus.PENDING)
        ).all()
        jobs = sorted(
            jobs,
            key=lambda job: ingestion_job_priority(job.document_id and db.get(Document, job.document_id), job),
        )[: max(1, int(limit))]

        for job in jobs:
            union = db.get(Union, job.union_id)
            document = job.document_id and db.get(Document, job.document_id)
            if union is None:
                if document is not None:
                    container.ingestion.fail_job(db, document=document, job=job, error=ValueError("Union not found"))
                failed += 1
                continue
            try:
                before_retry_id = (document.metadata_json or {}).get("latest_retry_job_id") if document is not None else None
                container.ingestion.process_job(db, union=union, job=job)
                after_retry_id = (document.metadata_json or {}).get("latest_retry_job_id") if document is not None else None
                if after_retry_id and after_retry_id != before_retry_id:
                    queued_retries += 1
                processed += 1
            except Exception as exc:
                if document is not None:
                    container.ingestion.fail_job(db, document=document, job=job, error=exc)
                failed += 1
        db.commit()
    return {"processed": processed, "failed": failed, "queued_retries": queued_retries}


if __name__ == "__main__":
    container = build_service_container()
    summary = process_pending_ingestion_jobs(container)
    print(summary)
