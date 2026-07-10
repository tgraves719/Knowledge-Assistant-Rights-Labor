"""Shared ingestion queue heuristics for worker ordering and ETA estimates."""

from __future__ import annotations

from backend.platform.models import Document, IngestionJob


def estimate_ingestion_runtime_seconds(document: Document | None, job: IngestionJob) -> int:
    base_seconds = 60
    metadata = dict((document.metadata_json or {}) if document is not None else {})
    content_type = str(document.content_type or "").lower() if document is not None else ""
    bytes_size = int(document.bytes_size or 0) if document is not None else 0
    page_count = int(metadata.get("page_count") or 0)
    scan_likelihood = str(metadata.get("scan_likelihood") or "").strip().lower()
    ocr_enabled = bool((job.metadata_json or {}).get("ocr_enabled"))

    if content_type.startswith("text/"):
        base_seconds = 15
    elif content_type == "application/pdf":
        base_seconds = 180
    elif content_type.startswith("image/"):
        base_seconds = 240
    else:
        base_seconds = 240

    if page_count >= 200:
        base_seconds *= 4
    elif page_count >= 75:
        base_seconds *= 3
    elif page_count >= 25:
        base_seconds *= 2

    if bytes_size > 20_000_000:
        base_seconds *= 3
    elif bytes_size > 10_000_000:
        base_seconds *= 2
    elif bytes_size > 3_000_000:
        base_seconds = int(base_seconds * 1.5)

    if scan_likelihood == "high":
        base_seconds += 120
    elif scan_likelihood == "medium":
        base_seconds += 45

    if ocr_enabled:
        base_seconds = int(base_seconds * 1.75)

    return max(15, int(base_seconds))


def ingestion_job_priority(document: Document | None, job: IngestionJob) -> tuple[int, int]:
    metadata = dict((document.metadata_json or {}) if document is not None else {})
    content_type = str(document.content_type or "").lower() if document is not None else ""
    bytes_size = int(document.bytes_size or 0) if document is not None else 0
    page_count = int(metadata.get("page_count") or 0)
    scan_likelihood = str(metadata.get("scan_likelihood") or "").strip().lower()

    score = 100
    trigger = str((job.metadata_json or {}).get("trigger") or "").strip().lower()
    if trigger == "retry":
        score -= 20

    if content_type.startswith("text/"):
        score -= 40
    elif content_type == "application/pdf":
        score += 30
    elif content_type.startswith("image/"):
        score += 40

    if page_count >= 200:
        score += 60
    elif page_count >= 75:
        score += 40
    elif page_count >= 25:
        score += 20

    if bytes_size > 20_000_000:
        score += 60
    elif bytes_size > 10_000_000:
        score += 40
    elif bytes_size > 3_000_000:
        score += 20

    if scan_likelihood == "high":
        score += 40
    elif scan_likelihood == "medium":
        score += 15

    if bool((job.metadata_json or {}).get("ocr_enabled")):
        score += 50

    return score, int(job.created_at.timestamp())
