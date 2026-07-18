"""Deferred ingestion worker helpers."""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from sqlalchemy import select

from backend.platform.db import apply_service_bootstrap_context
from backend.platform.models import Document, IngestionJob, IngestionJobStatus, Union
from backend.platform.queueing import ingestion_job_priority
from backend.platform.service_container import build_service_container


def _stale_running_timeout_seconds() -> int:
    return max(60, int(os.getenv("KARL_INGESTION_STALE_JOB_SECONDS", "1800")))


def reclaim_stale_running_jobs(db, *, now: datetime | None = None) -> int:
    """Return jobs orphaned in RUNNING to PENDING so they can be retried.

    A job is marked RUNNING before processing starts. If the process dies in
    between -- a crash, a container restart, or an inline ingest that raised --
    nothing ever moves it out of RUNNING: the worker only picks up PENDING, and
    the admin retry endpoint refuses to touch a running job. The document is
    then permanently stuck with no supported way to recover it.

    Only jobs whose started_at is older than the timeout are reclaimed, so a
    genuinely long-running ingest is not pulled out from under itself. The
    tradeoff is that an ingest exceeding the timeout could be processed twice;
    the window is deliberately generous to make that unlikely.
    """
    cutoff = (now or datetime.utcnow()) - timedelta(seconds=_stale_running_timeout_seconds())
    stale = db.scalars(
        select(IngestionJob).where(
            IngestionJob.status == IngestionJobStatus.RUNNING,
            IngestionJob.started_at.is_not(None),
            IngestionJob.started_at < cutoff,
        )
    ).all()
    for job in stale:
        job.status = IngestionJobStatus.PENDING
        job.started_at = None
    if stale:
        db.flush()
    return len(stale)


def process_pending_ingestion_jobs(container, *, limit: int = 25) -> dict:
    if container.session_factory is None:
        return {"processed": 0, "failed": 0, "queued_retries": 0}

    processed = 0
    failed = 0
    queued_retries = 0
    with container.session_factory() as db:
        apply_service_bootstrap_context(db)
        reclaimed = reclaim_stale_running_jobs(db)
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
    return {
        "processed": processed,
        "failed": failed,
        "queued_retries": queued_retries,
        "reclaimed": reclaimed,
    }


if __name__ == "__main__":
    container = build_service_container()
    summary = process_pending_ingestion_jobs(container)
    print(summary)
