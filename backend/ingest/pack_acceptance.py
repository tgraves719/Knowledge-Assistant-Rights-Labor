"""
Contract pack acceptance suite (ingestion-owned quality gate).

Evaluates a package under data/contracts/<contract_id>/ and emits a scorecard
with deterministic artifact hashes plus required/advisory checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.config import DATA_DIR
from backend.effective_contracts import (
    resolve_effective_version_dir,
    resolve_latest_effective_version_id,
)
from backend.miss_records import load_miss_record
from backend.validate_manifests import validate_manifest
from backend.ingest.extract_wages import lookup_wage, normalize_classification_name


CONTRACTS_ROOT = DATA_DIR / "contracts"
MISS_RECORDS_ROOT = DATA_DIR / "miss_records" / "records"
SCORECARD_VERSION = "contract_pack_scorecard_v3"
SIDE_LETTER_BUCKET_HIT_THRESHOLD = 1
MISS_TAXONOMY_CHECK_IDS = {
    "onboarding_taxonomy_defect": {
        "classification_ontology_manifest_decisions",
        "classification_ontology_mapping_coverage",
        "role_catalog_onboarding_default_wage_ready",
        "role_catalog_unresolved_manifest_rate",
        "ingestion_review_queue_issue_coverage",
    },
    "deterministic_answer_binding_defect": {
        "wages_exists",
        "wages_json_load",
        "manifest_classification_wage_coverage",
        "vacation_entitlement_non_empty",
    },
    "trigger_intent_defect": {
        "query_routing_coverage",
        "query_routing_article_ref_integrity",
        "vacation_entitlement_non_empty",
        "manifest_classification_wage_coverage",
    },
    "retrieval_followup_defect": {
        "query_routing_coverage",
        "query_routing_article_ref_integrity",
        "chunks_exists",
        "chunks_non_empty",
    },
    "artifact_scope_defect": {
        "effective_snapshot_contract_exists",
        "effective_chunk_index_input_non_empty",
        "effective_wage_index_input_non_empty",
        "effective_entitlement_index_input_non_empty",
    },
    "genuine_corpus_gap": {
        "source_pdf_exists",
        "chunks_exists",
    },
    "extraction_indexing_gap": {
        "side_letter_doc_type_materialization",
        "appendix_table_coverage_signal",
        "language_feature_coverage",
        "effective_chunk_index_input_non_empty",
        "effective_wage_index_input_non_empty",
        "effective_entitlement_index_input_non_empty",
    },
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _pack_hash(artifact_hashes: dict[str, str]) -> str:
    joined = "\n".join(f"{k}:{artifact_hashes[k]}" for k in sorted(artifact_hashes))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_classification_label(raw: str) -> str:
    return normalize_classification_name(str(raw or ""))


def _check_status(checks: list[dict], check_id: str) -> Optional[str]:
    for row in checks:
        if row.get("id") == check_id:
            return str(row.get("status") or "")
    return None


def _side_letter_chunk_metrics(chunks: list[dict]) -> dict[str, int]:
    doc_type_counts: Counter[str] = Counter()
    loa_text_hits = 0
    lou_text_hits = 0
    for row in chunks:
        if not isinstance(row, dict):
            continue
        doc_type = str(row.get("doc_type") or "cba").strip().lower()
        if doc_type:
            doc_type_counts[doc_type] += 1
        blob = (
            str(row.get("citation") or "")
            + "\n"
            + str(row.get("content_with_tables") or "")
            + "\n"
            + str(row.get("content") or "")
        ).lower()
        if "letter of agreement" in blob:
            loa_text_hits += 1
        if "letter of understanding" in blob:
            lou_text_hits += 1
    return {
        "loa_bucket_count": int(doc_type_counts.get("loa", 0)),
        "lou_bucket_count": int(doc_type_counts.get("lou", 0)),
        "loa_text_hits": loa_text_hits,
        "lou_text_hits": lou_text_hits,
    }


def _load_json_artifact(path: Path) -> tuple[Any, Optional[str]]:
    try:
        return _load_json(path), None
    except Exception as exc:
        return None, str(exc)


def _load_contract_miss_records(contract_id: str) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    if not MISS_RECORDS_ROOT.exists():
        return records, errors
    for path in sorted(MISS_RECORDS_ROOT.rglob("*.json")):
        try:
            record = load_miss_record(path)
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        if str(record.get("contract_id") or "") == contract_id:
            records.append(record)
    return records, errors


def _moa_wage_patch_metrics(package_dir: Path) -> dict[str, Any]:
    amendments_dir = package_dir / "amendments"
    metrics: dict[str, Any] = {
        "patch_count": 0,
        "moa_wage_patch_count": 0,
        "moa_wage_op_count": 0,
        "ops_with_selected_schedule_label": 0,
        "ops_with_source_rate_schedule": 0,
        "patches_missing_sync_metadata": [],
        "patches_missing_config_id": [],
        "source_doc_ids": [],
    }
    if not amendments_dir.exists():
        return metrics

    source_doc_ids: set[str] = set()
    for patch_path in sorted(amendments_dir.glob("*.json")):
        payload, error = _load_json_artifact(patch_path)
        if error or not isinstance(payload, dict):
            continue
        metrics["patch_count"] += 1
        operations = payload.get("operations") or []
        if not isinstance(operations, list):
            continue
        moa_wage_ops = []
        for op in operations:
            if not isinstance(op, dict):
                continue
            if str(op.get("op") or "").strip() != "replace_table_row":
                continue
            target = op.get("target") if isinstance(op.get("target"), dict) else {}
            if str(target.get("table_id") or "").strip() != "appendix_a_wage_rows":
                continue
            refs = op.get("source_refs") or []
            has_moa_ref = any(
                isinstance(ref, dict) and str(ref.get("source_type") or "").strip().lower() == "moa"
                for ref in refs
            )
            if not has_moa_ref:
                continue
            moa_wage_ops.append(op)
            for ref in refs:
                if isinstance(ref, dict):
                    source_doc_id = str(ref.get("source_doc_id") or "").strip()
                    if source_doc_id:
                        source_doc_ids.add(source_doc_id)
        if not moa_wage_ops:
            continue
        metrics["moa_wage_patch_count"] += 1
        metrics["moa_wage_op_count"] += len(moa_wage_ops)

        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        appendix_sync = metadata.get("appendix_a_sync") if isinstance(metadata.get("appendix_a_sync"), dict) else {}
        if not appendix_sync:
            metrics["patches_missing_sync_metadata"].append(str(patch_path))
        elif not str(appendix_sync.get("config_id") or "").strip():
            metrics["patches_missing_config_id"].append(str(patch_path))

        for op in moa_wage_ops:
            new_row = op.get("new_row") if isinstance(op.get("new_row"), dict) else {}
            if str(new_row.get("selected_schedule_label") or "").strip():
                metrics["ops_with_selected_schedule_label"] += 1
            if isinstance(new_row.get("source_rate_schedule"), dict) and bool(new_row.get("source_rate_schedule")):
                metrics["ops_with_source_rate_schedule"] += 1

    metrics["source_doc_ids"] = sorted(source_doc_ids)
    return metrics


def _effective_moa_provenance_page_metrics(package_dir: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "effective_version_id": None,
        "moa_ref_count": 0,
        "missing_page_ref_count": 0,
        "sections_missing_page": [],
        "tables_missing_page": [],
        "source_doc_ids": [],
    }
    latest_path = package_dir / "effective" / "latest.json"
    latest_payload, latest_error = _load_json_artifact(latest_path)
    if latest_error or not isinstance(latest_payload, dict):
        return metrics
    effective_version_id = str(latest_payload.get("effective_version_id") or "").strip()
    if not effective_version_id:
        return metrics
    metrics["effective_version_id"] = effective_version_id
    effective_contract_path = package_dir / "effective" / effective_version_id / "effective_contract.json"
    effective_payload, effective_error = _load_json_artifact(effective_contract_path)
    if effective_error or not isinstance(effective_payload, dict):
        return metrics

    source_doc_ids: set[str] = set()
    sections_missing_page: list[str] = []
    tables_missing_page: list[str] = []

    def _page_value(value: Any) -> Optional[int]:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    def _is_moa_ref(ref: dict[str, Any]) -> bool:
        source_type = str(ref.get("source_type") or "").strip().lower()
        pdf_name = str(ref.get("pdf") or "").strip().lower()
        return "moa" in source_type or "amend" in source_type or "moa" in pdf_name

    for section in effective_payload.get("sections") or []:
        if not isinstance(section, dict):
            continue
        citation = str(section.get("citation") or "").strip() or str(section.get("anchor_id") or "<section>")
        for ref in section.get("provenance") or []:
            if not isinstance(ref, dict) or not _is_moa_ref(ref):
                continue
            metrics["moa_ref_count"] += 1
            source_doc_id = str(ref.get("source_doc_id") or "").strip()
            if source_doc_id:
                source_doc_ids.add(source_doc_id)
            if _page_value(ref.get("pdf_page")) is None:
                metrics["missing_page_ref_count"] += 1
                if citation not in sections_missing_page:
                    sections_missing_page.append(citation)

    raw_tables = effective_payload.get("tables") or {}
    table_iter = raw_tables.values() if isinstance(raw_tables, dict) else raw_tables
    for table in table_iter or []:
        if not isinstance(table, dict):
            continue
        table_id = str(table.get("table_id") or "<table>")
        for row in table.get("rows") or []:
            if not isinstance(row, dict):
                continue
            row_key = str(row.get("row_key") or "").strip() or "<row>"
            for ref in row.get("provenance") or []:
                if not isinstance(ref, dict) or not _is_moa_ref(ref):
                    continue
                metrics["moa_ref_count"] += 1
                source_doc_id = str(ref.get("source_doc_id") or "").strip()
                if source_doc_id:
                    source_doc_ids.add(source_doc_id)
                if _page_value(ref.get("pdf_page")) is None:
                    metrics["missing_page_ref_count"] += 1
                    token = f"{table_id}:{row_key}"
                    if token not in tables_missing_page:
                        tables_missing_page.append(token)

    metrics["sections_missing_page"] = sections_missing_page[:50]
    metrics["tables_missing_page"] = tables_missing_page[:50]
    metrics["source_doc_ids"] = sorted(source_doc_ids)
    return metrics


def _check(
    checks: list[dict],
    check_id: str,
    passed: bool,
    severity: str,
    message: str,
    metrics: Optional[dict] = None,
) -> None:
    checks.append(
        {
            "id": check_id,
            "severity": severity,  # required | advisory
            "status": "pass" if passed else "fail",
            "message": message,
            "metrics": metrics or {},
        }
    )


def _non_empty_article_map_entries(value: Any) -> int:
    if not isinstance(value, dict):
        return 0
    count = 0
    for articles in value.values():
        if _to_article_list(articles):
            count += 1
    return count


def _to_article_list(values: Any) -> list[int]:
    out: list[int] = []
    seen = set()
    for raw in values or []:
        try:
            article_num = int(raw)
        except (TypeError, ValueError):
            continue
        if article_num in seen:
            continue
        seen.add(article_num)
        out.append(article_num)
    return out


def _invalid_article_refs(value: Any, valid_articles: set[int]) -> list[str]:
    if not isinstance(value, dict):
        return ["<map-not-object>"]
    invalid: list[str] = []
    for key, raw_articles in value.items():
        for article_num in _to_article_list(raw_articles):
            if article_num not in valid_articles:
                invalid.append(f"{key}:{article_num}")
    return sorted(invalid)


def _extract_outline_article_numbers(contract_outline: Any) -> set[int]:
    numbers: set[int] = set()
    if not isinstance(contract_outline, dict):
        return numbers

    raw_titles = contract_outline.get("article_titles")
    if isinstance(raw_titles, dict):
        for raw_key in raw_titles.keys():
            try:
                article_num = int(raw_key)
            except (TypeError, ValueError):
                continue
            if article_num > 0:
                numbers.add(article_num)

    raw_articles = contract_outline.get("articles")
    if isinstance(raw_articles, list):
        for row in raw_articles:
            if not isinstance(row, dict):
                continue
            try:
                article_num = int(row.get("article_num"))
            except (TypeError, ValueError):
                continue
            if article_num > 0:
                numbers.add(article_num)
    return numbers


def _resolve_artifacts(package_dir: Path, contract_id: str) -> dict[str, Path]:
    source_dir = package_dir / "source"
    manifests_dir = package_dir / "manifests"
    chunks_dir = package_dir / "chunks"
    tables_dir = package_dir / "tables"
    wages_dir = package_dir / "wages"
    entitlements_dir = package_dir / "entitlements"
    ontology_dir = package_dir / "ontology"
    outline_dir = package_dir / "outline"

    md_candidates = sorted(source_dir.glob("*.md"))
    json_candidates = sorted(source_dir.glob("*.json"))
    pdf_candidates = sorted(source_dir.glob("*.pdf"))

    return {
        "source_md": md_candidates[0] if md_candidates else source_dir / f"{contract_id}.md",
        "source_json": json_candidates[0] if json_candidates else source_dir / f"{contract_id}.json",
        "source_pdf": pdf_candidates[0] if pdf_candidates else source_dir / f"{contract_id}.pdf",
        "manifest": manifests_dir / f"{contract_id}.json",
        "chunks_enriched": chunks_dir / f"contract_chunks_enriched_{contract_id}.json",
        "chunks_smart": chunks_dir / f"contract_chunks_smart_{contract_id}.json",
        "chunks_base": chunks_dir / f"contract_chunks_{contract_id}.json",
        "concept_index": chunks_dir / f"concept_index_{contract_id}.json",
        "tables": tables_dir / "structured_tables.json",
        "wages": wages_dir / f"wage_tables_{contract_id}.json",
        "entitlements": entitlements_dir / f"entitlement_tables_{contract_id}.json",
        "classification_ontology": ontology_dir / "classification_ontology.json",
        "language_lexicon": ontology_dir / "language_lexicon.json",
        "pdf_nav_index": ontology_dir / "pdf_nav_index.json",
        "contract_outline": outline_dir / "contract_outline.json",
        "role_catalog": ontology_dir / "role_catalog.json",
        "ingestion_review_queue": ontology_dir / "ingestion_review_queue.json",
        "manual_classification_overrides": ontology_dir / "manual_classification_overrides.json",
    }


def evaluate_contract_pack(
    package_dir: Path,
    strict: bool = False,
    write_scorecard: bool = True,
) -> dict:
    contract_id = package_dir.name
    artifacts = _resolve_artifacts(package_dir, contract_id)
    checks: list[dict] = []

    artifact_hashes: dict[str, str] = {}
    for name, path in artifacts.items():
        if path.exists():
            artifact_hashes[name] = _sha256_file(path)

    manifest = None
    chunks = []
    wages_data = None
    entitlements_data = None
    table_registry = []
    classification_ontology = None
    pdf_nav_index = None
    contract_outline = None
    role_catalog = None
    ingestion_review_queue = None
    effective_version_id = resolve_latest_effective_version_id(contract_id)
    effective_version_dir = resolve_effective_version_dir(
        contract_id,
        effective_version_id=effective_version_id,
    )

    # Manifest presence + schema integrity
    manifest_path = artifacts["manifest"]
    manifest_ok = manifest_path.exists()
    _check(
        checks,
        "manifest_exists",
        manifest_ok,
        "required",
        "Manifest artifact is present." if manifest_ok else "Manifest artifact is missing.",
        {"path": str(manifest_path)},
    )
    if manifest_ok:
        manifest_errors = validate_manifest(manifest_path)
        manifest_schema_ok = not manifest_errors
        _check(
            checks,
            "manifest_schema_valid",
            manifest_schema_ok,
            "required",
            "Manifest schema passed validation."
            if manifest_schema_ok
            else "Manifest schema validation failed.",
            {"errors": manifest_errors[:25]},
        )
        try:
            manifest = _load_json(manifest_path)
        except Exception as exc:
            _check(
                checks,
                "manifest_json_load",
                False,
                "required",
                f"Manifest JSON load failed: {exc}",
            )

    source_pdf_path = artifacts["source_pdf"]
    source_pdf_exists = source_pdf_path.exists()
    source_pdf_size = source_pdf_path.stat().st_size if source_pdf_exists else 0
    _check(
        checks,
        "source_pdf_exists",
        source_pdf_exists and source_pdf_size > 0,
        "required",
        "Source PDF artifact is present and non-empty."
        if (source_pdf_exists and source_pdf_size > 0)
        else "Source PDF artifact is missing or empty.",
        {
            "path": str(source_pdf_path),
            "size_bytes": source_pdf_size,
        },
    )

    if manifest:
        article_titles = manifest.get("article_titles", {}) or {}
        valid_articles = set()
        for raw_key in article_titles.keys():
            try:
                valid_articles.add(int(raw_key))
            except (TypeError, ValueError):
                continue

        routing = manifest.get("query_routing") or {}
        topic_to_articles = routing.get("topic_to_articles") if isinstance(routing, dict) else {}
        topic_patterns = routing.get("topic_patterns") if isinstance(routing, dict) else {}
        slang_to_contract = routing.get("slang_to_contract") if isinstance(routing, dict) else {}
        classification_to_articles = routing.get("classification_to_articles") if isinstance(routing, dict) else {}

        topic_entries = _non_empty_article_map_entries(topic_to_articles)
        pattern_entries = len(topic_patterns) if isinstance(topic_patterns, dict) else 0
        slang_entries = len(slang_to_contract) if isinstance(slang_to_contract, dict) else 0
        class_entries = _non_empty_article_map_entries(classification_to_articles)

        manifest_classifications = manifest.get("classifications", []) or []
        min_class_entries = 0
        if manifest_classifications:
            min_class_entries = max(1, min(6, int(round(len(manifest_classifications) * 0.4))))

        routing_coverage_ok = (
            topic_entries >= 4
            and pattern_entries >= 4
            and slang_entries >= 8
            and class_entries >= min_class_entries
        )
        _check(
            checks,
            "query_routing_coverage",
            routing_coverage_ok,
            "required",
            "Manifest query_routing coverage meets deterministic minimums."
            if routing_coverage_ok
            else "Manifest query_routing coverage below deterministic minimums.",
            {
                "topic_entries": topic_entries,
                "topic_pattern_entries": pattern_entries,
                "slang_entries": slang_entries,
                "classification_entries": class_entries,
                "min_classification_entries": min_class_entries,
            },
        )

        invalid_topic_refs = _invalid_article_refs(topic_to_articles, valid_articles)
        invalid_class_refs = _invalid_article_refs(classification_to_articles, valid_articles)
        routing_refs_ok = len(invalid_topic_refs) == 0 and len(invalid_class_refs) == 0
        _check(
            checks,
            "query_routing_article_ref_integrity",
            routing_refs_ok,
            "required",
            "Manifest query_routing article references are valid."
            if routing_refs_ok
            else "Manifest query_routing contains invalid article references.",
            {
                "invalid_topic_refs": invalid_topic_refs[:40],
                "invalid_classification_refs": invalid_class_refs[:40],
            },
        )

    # Chunk artifact checks
    chunks_path = artifacts["chunks_enriched"]
    if not chunks_path.exists():
        chunks_path = artifacts["chunks_base"]
    chunks_exists = chunks_path.exists()
    _check(
        checks,
        "chunks_exists",
        chunks_exists,
        "required",
        "Chunk artifact is present." if chunks_exists else "Chunk artifact is missing.",
        {"path": str(chunks_path)},
    )
    if chunks_exists:
        try:
            chunks = _load_json(chunks_path)
        except Exception as exc:
            chunks = []
            _check(
                checks,
                "chunks_json_load",
                False,
                "required",
                f"Chunk JSON load failed: {exc}",
            )

    if chunks:
        non_empty = len(chunks) > 0
        _check(
            checks,
            "chunks_non_empty",
            non_empty,
            "required",
            f"Chunk count = {len(chunks)}.",
            {"chunk_count": len(chunks)},
        )

        missing_contract = sum(1 for c in chunks if not c.get("contract_id"))
        missing_region = sum(1 for c in chunks if not c.get("region_id"))
        _check(
            checks,
            "chunks_contract_scope",
            missing_contract == 0,
            "required",
            "All chunks contain contract_id."
            if missing_contract == 0
            else "Some chunks are missing contract_id.",
            {"missing_contract_id_chunks": missing_contract},
        )
        _check(
            checks,
            "chunks_region_scope",
            missing_region == 0,
            "required",
            "All chunks contain region_id."
            if missing_region == 0
            else "Some chunks are missing region_id.",
            {"missing_region_id_chunks": missing_region},
        )

        duplicate_count = sum(v - 1 for v in Counter(c.get("chunk_id") for c in chunks).values() if v > 1)
        _check(
            checks,
            "chunk_id_uniqueness",
            duplicate_count == 0,
            "advisory",
            "All chunk IDs are unique."
            if duplicate_count == 0
            else "Duplicate chunk IDs detected.",
            {"duplicate_chunk_rows": duplicate_count},
        )

        malformed_citations = []
        legacy_segment_subsections = []
        inconsistent_cba_citations = []
        for c in chunks:
            citation = str(c.get("citation") or "")
            subsection = str(c.get("subsection") or "")
            doc_type = str(c.get("doc_type") or "cba")
            article_num = c.get("article_num")
            section_num = c.get("section_num")

            if re.search(r"\bPart\s+part\d+\b", citation, flags=re.IGNORECASE):
                malformed_citations.append(citation)
            if re.fullmatch(r"part\d+", subsection.strip(), flags=re.IGNORECASE):
                legacy_segment_subsections.append(
                    {"chunk_id": c.get("chunk_id"), "subsection": subsection}
                )
            if (
                doc_type == "cba"
                and article_num is not None
                and section_num is not None
                and not re.search(r"Article\s+\d+\s*,\s*Section\s+\d+", citation)
            ):
                inconsistent_cba_citations.append(
                    {"chunk_id": c.get("chunk_id"), "citation": citation}
                )

        citation_ok = (
            len(malformed_citations) == 0
            and len(legacy_segment_subsections) == 0
            and len(inconsistent_cba_citations) == 0
        )
        _check(
            checks,
            "chunk_citation_normalization",
            citation_ok,
            "advisory",
            "Chunk citations/subsections follow normalized conventions."
            if citation_ok
            else "Detected legacy or malformed chunk citation/subsection conventions.",
            {
                "malformed_citations_sample": malformed_citations[:10],
                "legacy_segment_subsections_sample": legacy_segment_subsections[:10],
                "inconsistent_cba_citations_sample": inconsistent_cba_citations[:10],
            },
        )

        # Deterministic language metadata coverage (required for concept index quality).
        alt_non_empty = 0
        question_non_empty = 0
        for c in chunks:
            alt = c.get("alternative_names")
            q = c.get("worker_questions")
            if isinstance(alt, str):
                alt = [alt] if alt.strip() else []
            if isinstance(q, str):
                q = [q] if q.strip() else []
            if isinstance(alt, list) and len([x for x in alt if str(x).strip()]) > 0:
                alt_non_empty += 1
            if isinstance(q, list) and len([x for x in q if str(x).strip()]) > 0:
                question_non_empty += 1

        chunk_total = max(1, len(chunks))
        alt_coverage = alt_non_empty / chunk_total
        question_coverage = question_non_empty / chunk_total
        min_coverage = 0.25
        _check(
            checks,
            "language_feature_coverage",
            alt_coverage >= min_coverage and question_coverage >= min_coverage,
            "required",
            "Deterministic language features meet minimum coverage."
            if (alt_coverage >= min_coverage and question_coverage >= min_coverage)
            else "Language features coverage below minimum threshold.",
            {
                "alt_non_empty_chunks": alt_non_empty,
                "question_non_empty_chunks": question_non_empty,
                "chunk_total": chunk_total,
                "alt_coverage": round(alt_coverage, 4),
                "question_coverage": round(question_coverage, 4),
                "min_required": min_coverage,
            },
        )

        side_letter_metrics = _side_letter_chunk_metrics(chunks)
        side_letter_text_hits = (
            int(side_letter_metrics.get("loa_text_hits", 0))
            + int(side_letter_metrics.get("lou_text_hits", 0))
        )
        side_letter_bucket_count = (
            int(side_letter_metrics.get("loa_bucket_count", 0))
            + int(side_letter_metrics.get("lou_bucket_count", 0))
        )
        side_letter_materialized = (
            side_letter_text_hits < SIDE_LETTER_BUCKET_HIT_THRESHOLD
            or side_letter_bucket_count > 0
        )
        _check(
            checks,
            "side_letter_doc_type_materialization",
            side_letter_materialized,
            "required",
            "Side-letter lexical hits are backed by LOA/LOU doc_type buckets."
            if side_letter_materialized
            else "Side-letter lexical hits exist but LOA/LOU doc_type buckets were not materialized.",
            {
                "side_letter_hit_threshold": SIDE_LETTER_BUCKET_HIT_THRESHOLD,
                "side_letter_text_hits": side_letter_text_hits,
                "loa_text_hits": side_letter_metrics.get("loa_text_hits", 0),
                "lou_text_hits": side_letter_metrics.get("lou_text_hits", 0),
                "loa_bucket_count": side_letter_metrics.get("loa_bucket_count", 0),
                "lou_bucket_count": side_letter_metrics.get("lou_bucket_count", 0),
            },
        )

    # Article coverage against manifest
    if manifest and chunks:
        article_titles = manifest.get("article_titles", {}) or {}
        expected_articles = set()
        for key in article_titles.keys():
            try:
                article_num = int(key)
                if article_num > 0:
                    expected_articles.add(article_num)
            except (TypeError, ValueError):
                continue

        found_articles = {
            int(c.get("article_num")) for c in chunks
            if isinstance(c.get("article_num"), int) and int(c.get("article_num")) > 0
        }
        missing_articles = sorted(expected_articles - found_articles)
        unindexed_articles = sorted(found_articles - expected_articles)
        coverage = (
            len(expected_articles & found_articles) / len(expected_articles)
            if expected_articles else 0.0
        )
        _check(
            checks,
            "article_coverage",
            coverage >= 0.95,
            "required",
            "Article coverage threshold met."
            if coverage >= 0.95
            else "Article coverage below threshold.",
            {
                "coverage": round(coverage, 4),
                "expected_articles": len(expected_articles),
                "found_articles": len(found_articles),
                "missing_articles": missing_articles[:30],
            },
        )
        _check(
            checks,
            "manifest_article_index_completeness",
            len(unindexed_articles) == 0,
            "required",
            "Manifest indexes all chunked article numbers."
            if len(unindexed_articles) == 0
            else "Manifest is missing article numbers that are present in chunks.",
            {
                "missing_from_manifest": unindexed_articles[:30],
                "missing_from_manifest_count": len(unindexed_articles),
                "found_articles": len(found_articles),
                "indexed_articles": len(expected_articles),
            },
        )

    # Table registry checks
    tables_path = artifacts["tables"]
    tables_exists = tables_path.exists()
    _check(
        checks,
        "tables_registry_exists",
        tables_exists or not artifacts["source_json"].exists(),
        "advisory",
        "Structured table registry is present."
        if tables_exists
        else "Structured table registry missing (JSON source may be absent).",
        {"path": str(tables_path)},
    )
    if tables_exists:
        try:
            table_registry = _load_json(tables_path)
            if isinstance(table_registry, dict):
                table_registry = table_registry.get("tables", [])
        except Exception:
            table_registry = []

    if chunks and table_registry:
        table_ids = {str(t.get("table_id")) for t in table_registry if t.get("table_id")}
        referenced = set()
        for c in chunks:
            for t_id in c.get("table_refs") or []:
                referenced.add(str(t_id))
        unresolved_refs = sorted(referenced - table_ids)
        _check(
            checks,
            "table_ref_integrity",
            len(unresolved_refs) == 0,
            "required",
            "All chunk table_refs resolve to registry table IDs."
            if not unresolved_refs
            else "Some chunk table_refs do not resolve to table registry IDs.",
            {"unresolved_table_refs": unresolved_refs[:30]},
        )

    concept_index_path = artifacts["concept_index"]
    concept_index_exists = concept_index_path.exists()
    _check(
        checks,
        "concept_index_exists",
        concept_index_exists,
        "required",
        "Contract-scoped concept index is present."
        if concept_index_exists
        else "Contract-scoped concept index is missing.",
        {"path": str(concept_index_path)},
    )
    if concept_index_exists:
        try:
            concept_index_data = _load_json(concept_index_path)
        except Exception as exc:
            concept_index_data = {}
            _check(
                checks,
                "concept_index_json_load",
                False,
                "required",
                f"Concept index JSON load failed: {exc}",
            )
        if isinstance(concept_index_data, dict):
            concept_count = len((concept_index_data.get("concept_to_articles") or {}))
            question_count = len((concept_index_data.get("question_to_articles") or {}))
            _check(
                checks,
                "concept_index_non_empty",
                concept_count > 0 or question_count > 0,
                "required",
                "Concept index contains concept/question mappings."
                if (concept_count > 0 or question_count > 0)
                else "Concept index contains no concept/question mappings.",
                {
                    "concept_count": concept_count,
                    "question_count": question_count,
                },
            )

    language_lexicon_path = artifacts["language_lexicon"]
    lex_exists = language_lexicon_path.exists()
    _check(
        checks,
        "language_lexicon_exists",
        lex_exists,
        "required",
        "Language lexicon artifact is present."
        if lex_exists
        else "Language lexicon artifact is missing.",
        {"path": str(language_lexicon_path)},
    )
    if lex_exists:
        try:
            lex_data = _load_json(language_lexicon_path)
        except Exception as exc:
            lex_data = {}
            _check(
                checks,
                "language_lexicon_json_load",
                False,
                "required",
                f"Language lexicon JSON load failed: {exc}",
            )
        if isinstance(lex_data, dict):
            alias_count = len(lex_data.get("alias_to_canonical", {}) or {})
            entry_count = len(lex_data.get("entries", []) or [])
            region_id = str(lex_data.get("region_id") or "").strip()
            _check(
                checks,
                "language_lexicon_non_empty",
                alias_count > 0 and entry_count > 0,
                "required",
                "Language lexicon contains aliases and entries."
                if alias_count > 0 and entry_count > 0
                else "Language lexicon has no usable aliases/entries.",
                {
                    "alias_count": alias_count,
                    "entry_count": entry_count,
                },
            )
            _check(
                checks,
                "language_lexicon_region_id",
                bool(region_id),
                "required",
                "Language lexicon includes region_id."
                if region_id
                else "Language lexicon missing region_id.",
                {"region_id": region_id},
            )

    pdf_nav_path = artifacts["pdf_nav_index"]
    pdf_nav_exists = pdf_nav_path.exists()
    _check(
        checks,
        "pdf_nav_index_exists",
        pdf_nav_exists,
        "required",
        "PDF navigation index artifact is present."
        if pdf_nav_exists
        else "PDF navigation index artifact is missing.",
        {"path": str(pdf_nav_path)},
    )
    if pdf_nav_exists:
        try:
            pdf_nav_index = _load_json(pdf_nav_path)
        except Exception as exc:
            pdf_nav_index = None
            _check(
                checks,
                "pdf_nav_index_json_load",
                False,
                "required",
                f"PDF navigation index JSON load failed: {exc}",
            )

    if pdf_nav_index:
        schema_ok = str(pdf_nav_index.get("schema_version") or "") == "pdf_nav_index_v1"
        contract_ok = str(pdf_nav_index.get("contract_id") or "") == contract_id
        article_pages = pdf_nav_index.get("article_pages") or {}
        section_pages = pdf_nav_index.get("section_pages") or {}
        article_count = len(article_pages) if isinstance(article_pages, dict) else 0
        section_count = 0
        if isinstance(section_pages, dict):
            for raw_val in section_pages.values():
                if isinstance(raw_val, dict):
                    section_count += len(raw_val)
        _check(
            checks,
            "pdf_nav_index_schema_valid",
            schema_ok and contract_ok,
            "required",
            "PDF navigation index schema and contract_id are valid."
            if schema_ok and contract_ok
            else "PDF navigation index schema_version/contract_id are invalid.",
            {
                "schema_version": pdf_nav_index.get("schema_version"),
                "artifact_contract_id": pdf_nav_index.get("contract_id"),
            },
        )
        _check(
            checks,
            "pdf_nav_index_non_empty",
            article_count > 0,
            "required",
            "PDF navigation index includes article page mappings."
            if article_count > 0
            else "PDF navigation index has no article page mappings.",
            {
                "article_pages": article_count,
                "section_pages": section_count,
            },
        )

    contract_outline_path = artifacts["contract_outline"]
    contract_outline_exists = contract_outline_path.exists()
    _check(
        checks,
        "contract_outline_exists",
        contract_outline_exists,
        "required",
        "Contract outline artifact is present."
        if contract_outline_exists
        else "Contract outline artifact is missing.",
        {"path": str(contract_outline_path)},
    )
    if contract_outline_exists:
        try:
            contract_outline = _load_json(contract_outline_path)
        except Exception as exc:
            contract_outline = None
            _check(
                checks,
                "contract_outline_json_load",
                False,
                "required",
                f"Contract outline JSON load failed: {exc}",
            )

    if contract_outline:
        schema_ok = str(contract_outline.get("schema_version") or "") == "contract_outline_v1"
        contract_ok = str(contract_outline.get("contract_id") or "") == contract_id
        outline_article_numbers = _extract_outline_article_numbers(contract_outline)
        outline_articles = contract_outline.get("articles") or []
        outline_non_empty = bool(outline_article_numbers) and bool(outline_articles)
        _check(
            checks,
            "contract_outline_schema_valid",
            schema_ok and contract_ok,
            "required",
            "Contract outline schema and contract_id are valid."
            if schema_ok and contract_ok
            else "Contract outline schema_version/contract_id are invalid.",
            {
                "schema_version": contract_outline.get("schema_version"),
                "artifact_contract_id": contract_outline.get("contract_id"),
            },
        )
        _check(
            checks,
            "contract_outline_non_empty",
            outline_non_empty,
            "required",
            "Contract outline includes article metadata."
            if outline_non_empty
            else "Contract outline has no usable article metadata.",
            {
                "article_count": len(outline_article_numbers),
                "raw_articles_count": len(outline_articles) if isinstance(outline_articles, list) else 0,
            },
        )
        if chunks:
            chunk_articles = {
                int(c.get("article_num")) for c in chunks
                if isinstance(c.get("article_num"), int) and int(c.get("article_num")) > 0
            }
            missing_from_outline = sorted(chunk_articles - outline_article_numbers)
            _check(
                checks,
                "contract_outline_chunk_coverage",
                len(missing_from_outline) == 0,
                "required",
                "Contract outline indexes all chunked article numbers."
                if len(missing_from_outline) == 0
                else "Contract outline is missing chunked article numbers.",
                {
                    "missing_from_outline": missing_from_outline[:30],
                    "missing_from_outline_count": len(missing_from_outline),
                    "chunk_article_count": len(chunk_articles),
                    "outline_article_count": len(outline_article_numbers),
                },
            )

    # Wage artifact checks
    wages_path = artifacts["wages"]
    wages_exists = wages_path.exists()
    requires_wages = bool(manifest and manifest.get("has_appendix_a"))
    _check(
        checks,
        "wages_exists",
        wages_exists or not requires_wages,
        "required",
        "Wage artifact is present."
        if wages_exists
        else "Wage artifact missing for manifest with has_appendix_a=true."
        if requires_wages
        else "Wage artifact missing but Appendix A is not marked required.",
        {"path": str(wages_path), "requires_wages": requires_wages},
    )

    if wages_exists:
        try:
            wages_data = _load_json(wages_path)
        except Exception as exc:
            _check(
                checks,
                "wages_json_load",
                False,
                "required",
                f"Wage JSON load failed: {exc}",
            )
            wages_data = None

    if wages_data:
        classes = wages_data.get("classifications", {}) or {}
        _check(
            checks,
            "wage_classifications_non_empty",
            len(classes) > 0,
            "required",
            "Wage classifications extracted."
            if len(classes) > 0
            else "Wage classifications are empty.",
            {"classification_count": len(classes)},
        )

        unresolved_primary = []
        for class_key in sorted(classes.keys()):
            if lookup_wage(wages_data, class_key, 0, 0) is None:
                unresolved_primary.append(class_key)
        _check(
            checks,
            "wage_primary_lookup_integrity",
            len(unresolved_primary) == 0,
            "required",
            "All primary wage classification keys resolve."
            if not unresolved_primary
            else "Some primary wage classification keys do not resolve.",
            {"unresolved_keys": unresolved_primary[:30]},
        )

        canonical_rows = wages_data.get("canonical_wage_rows", []) or []
        requires_canonical_rows = len(classes) > 0
        _check(
            checks,
            "canonical_wage_rows_present",
            (len(canonical_rows) > 0) or (not requires_canonical_rows),
            "required",
            "Canonical wage rows are present."
            if len(canonical_rows) > 0
            else "Canonical wage rows missing despite extracted wage classifications."
            if requires_canonical_rows
            else "Canonical wage rows not required (no wage classifications).",
            {
                "canonical_row_count": len(canonical_rows),
                "requires_canonical_rows": requires_canonical_rows,
            },
        )

        invalid_rows = []
        contradictory_keys = {}
        date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        allowed_step_types = {"hours", "months", "fixed"}
        for idx, row in enumerate(canonical_rows):
            if not isinstance(row, dict):
                invalid_rows.append({"index": idx, "reason": "row_not_object"})
                continue

            required_fields = (
                "classification_key",
                "step_name",
                "step_type",
                "effective_date",
                "rate",
                "source_method",
                "source_reference",
            )
            missing = [f for f in required_fields if row.get(f) is None or row.get(f) == ""]
            if missing:
                invalid_rows.append({"index": idx, "reason": "missing_fields", "fields": missing})
                continue

            if str(row.get("step_type")) not in allowed_step_types:
                invalid_rows.append({"index": idx, "reason": "invalid_step_type", "value": row.get("step_type")})
                continue

            if not date_re.match(str(row.get("effective_date"))):
                invalid_rows.append({"index": idx, "reason": "invalid_effective_date", "value": row.get("effective_date")})
                continue

            try:
                rate_value = float(row.get("rate"))
                if rate_value <= 0:
                    raise ValueError("non_positive_rate")
            except Exception:
                invalid_rows.append({"index": idx, "reason": "invalid_rate", "value": row.get("rate")})
                continue

            row_key = (
                str(row.get("classification_key")),
                str(row.get("step_name")),
                str(row.get("step_type")),
                row.get("threshold_value"),
                str(row.get("effective_date")),
            )
            contradictory_keys.setdefault(row_key, set()).add(round(rate_value, 6))

        _check(
            checks,
            "canonical_wage_row_schema_valid",
            len(invalid_rows) == 0,
            "required",
            "Canonical wage rows pass schema validation."
            if not invalid_rows
            else "Canonical wage rows contain schema errors.",
            {"invalid_row_count": len(invalid_rows), "invalid_rows": invalid_rows[:40]},
        )

        contradictory = [
            {
                "classification_key": key[0],
                "step_name": key[1],
                "step_type": key[2],
                "threshold_value": key[3],
                "effective_date": key[4],
                "rates": sorted(list(rates)),
            }
            for key, rates in contradictory_keys.items()
            if len(rates) > 1
        ]
        _check(
            checks,
            "canonical_wage_row_contradictions",
            len(contradictory) == 0,
            "advisory",
            "No contradictory canonical wage rows detected."
            if not contradictory
            else "Contradictory canonical wage row rates detected.",
            {"contradiction_count": len(contradictory), "contradictions": contradictory[:30]},
        )

        extraction_meta = wages_data.get("extraction_metadata", {}) or {}
        canonical_conflicts = extraction_meta.get("canonical_conflicts", []) or []
        canonical_ambiguities = extraction_meta.get("canonical_ambiguities", []) or []
        denom = max(len(canonical_rows), 1)
        conflict_rate = len(canonical_conflicts) / denom
        ambiguity_rate = len(canonical_ambiguities) / denom
        _check(
            checks,
            "canonical_wage_conflict_rate",
            conflict_rate <= 0.05,
            "advisory",
            "Canonical wage conflict rate is within advisory threshold."
            if conflict_rate <= 0.05
            else "Canonical wage conflict rate exceeds advisory threshold.",
            {
                "conflict_rate": round(conflict_rate, 4),
                "conflict_count": len(canonical_conflicts),
                "canonical_row_count": len(canonical_rows),
            },
        )
        _check(
            checks,
            "canonical_wage_ambiguity_rate",
            ambiguity_rate <= 0.05,
            "advisory",
            "Canonical wage ambiguity rate is within advisory threshold."
            if ambiguity_rate <= 0.05
            else "Canonical wage ambiguity rate exceeds advisory threshold.",
            {
                "ambiguity_rate": round(ambiguity_rate, 4),
                "ambiguity_count": len(canonical_ambiguities),
                "canonical_row_count": len(canonical_rows),
            },
        )

        if manifest:
            manifest_classes = manifest.get("classifications", []) or []
            normalized_manifest_classes = [
                _normalize_classification_label(v) for v in manifest_classes if str(v).strip()
            ]
            ontology_decisions_by_source = {}
            ontology_for_coverage_path = artifacts["classification_ontology"]
            if ontology_for_coverage_path.exists():
                try:
                    ontology_for_coverage = _load_json(ontology_for_coverage_path)
                    ontology_decisions_by_source = {
                        str(d.get("source_key") or "").strip(): d
                        for d in (ontology_for_coverage.get("decisions") or [])
                        if isinstance(d, dict) and str(d.get("source_key") or "").strip()
                    }
                except Exception:
                    ontology_decisions_by_source = {}
            resolved_manifest = []
            unresolved_manifest = []
            out_of_scope_manifest = []
            clarification_manifest = []
            actionable_manifest = []
            for raw_label in normalized_manifest_classes:
                decision = ontology_decisions_by_source.get(raw_label, {})
                review_state = str(decision.get("review_state") or "").strip()
                if review_state == "out_of_scope":
                    out_of_scope_manifest.append(raw_label)
                    continue
                actionable_manifest.append(raw_label)
                if (
                    lookup_wage(wages_data, raw_label, 0, 0) is not None
                    or bool(decision.get("mapped_wage_key"))
                    or review_state == "needs_clarification"
                ):
                    resolved_manifest.append(raw_label)
                    if review_state == "needs_clarification":
                        clarification_manifest.append(raw_label)
                else:
                    unresolved_manifest.append(raw_label)
            ratio = (
                len(resolved_manifest) / len(actionable_manifest)
                if actionable_manifest else 1.0
            )
            _check(
                checks,
                "manifest_classification_wage_coverage",
                ratio >= 0.6,
                "advisory",
                "Manifest classification wage coverage meets advisory threshold."
                if ratio >= 0.6
                else "Manifest classification wage coverage below advisory threshold.",
                {
                    "coverage": round(ratio, 4),
                    "resolved_or_clarified": len(resolved_manifest),
                    "clarification_required": len(clarification_manifest),
                    "out_of_scope": len(out_of_scope_manifest),
                    "total": len(actionable_manifest),
                    "unresolved": unresolved_manifest[:30],
                },
            )

            critical_aliases = [
                c for c in normalized_manifest_classes
                if any(tok in c for tok in ("dug", "shopper", "drive_up", "clicklist"))
            ]
            unresolved_critical = [
                c for c in critical_aliases if lookup_wage(wages_data, c, 0, 0) is None
            ]
            _check(
                checks,
                "critical_alias_resolution",
                len(unresolved_critical) == 0,
                "required",
                "Critical shopper/dug aliases resolve in wage lookup."
                if not unresolved_critical
                else "Critical shopper/dug aliases failed wage resolution.",
                {"unresolved_critical_aliases": unresolved_critical},
            )

    # Entitlement artifact checks
    entitlements_path = artifacts["entitlements"]
    entitlements_exists = entitlements_path.exists()
    requires_entitlements = False
    if manifest:
        article_titles = manifest.get("article_titles", {}) or {}
        requires_entitlements = any("vacation" in str(v or "").lower() for v in article_titles.values())
    _check(
        checks,
        "entitlements_exists",
        entitlements_exists or not requires_entitlements,
        "required",
        "Entitlement artifact is present."
        if entitlements_exists
        else "Entitlement artifact missing for manifest with vacation language."
        if requires_entitlements
        else "Entitlement artifact missing but vacation entitlement extraction is not required.",
        {"path": str(entitlements_path), "requires_entitlements": requires_entitlements},
    )
    if entitlements_exists:
        try:
            entitlements_data = _load_json(entitlements_path)
        except Exception as exc:
            entitlements_data = None
            _check(
                checks,
                "entitlements_json_load",
                False,
                "required",
                f"Entitlement JSON load failed: {exc}",
            )

    if entitlements_data:
        schema_ok = str(entitlements_data.get("schema_version") or "") == "entitlement_tables_v1"
        contract_ok = str(entitlements_data.get("contract_id") or "") == contract_id
        region_ok = bool(str(entitlements_data.get("region_id") or "").strip())
        schedules = list(entitlements_data.get("vacation_entitlements") or [])
        schedule_count = len(schedules)
        _check(
            checks,
            "entitlements_schema_valid",
            schema_ok and contract_ok and region_ok,
            "required",
            "Entitlement artifact schema/contract/region are valid."
            if schema_ok and contract_ok and region_ok
            else "Entitlement artifact schema/contract/region validation failed.",
            {
                "schema_version": entitlements_data.get("schema_version"),
                "artifact_contract_id": entitlements_data.get("contract_id"),
                "region_id": entitlements_data.get("region_id"),
            },
        )
        _check(
            checks,
            "vacation_entitlement_non_empty",
            schedule_count > 0 or not requires_entitlements,
            "required",
            "Vacation entitlement schedules are present."
            if schedule_count > 0
            else "Vacation entitlement schedules are empty.",
            {
                "schedule_count": schedule_count,
                "requires_entitlements": requires_entitlements,
            },
        )

    # Effective snapshot checks
    effective_contract_path = (
        effective_version_dir / "effective_contract.json" if effective_version_dir else None
    )
    effective_contract_exists = bool(
        effective_contract_path is not None and effective_contract_path.exists()
    )
    _check(
        checks,
        "effective_snapshot_contract_exists",
        (not effective_version_dir) or effective_contract_exists,
        "required",
        "Effective snapshot includes effective_contract.json."
        if effective_contract_exists
        else "No effective snapshot present."
        if not effective_version_dir
        else "Effective snapshot is missing effective_contract.json.",
        {
            "effective_version_id": effective_version_id,
            "path": str(effective_contract_path) if effective_contract_path else None,
        },
    )

    effective_index_inputs_dir = (
        effective_version_dir / "index_inputs" if effective_version_dir else None
    )
    effective_chunk_input = None
    effective_chunk_candidates = []
    if effective_index_inputs_dir:
        effective_chunk_candidates = [
            effective_index_inputs_dir / artifacts["chunks_enriched"].name,
            effective_index_inputs_dir / artifacts["chunks_base"].name,
        ]
        for candidate in effective_chunk_candidates:
            if candidate.exists():
                effective_chunk_input = candidate
                break

    _check(
        checks,
        "effective_chunk_index_input_exists",
        (not effective_version_dir) or (effective_chunk_input is not None),
        "required",
        "Effective snapshot includes chunk index input."
        if effective_chunk_input is not None
        else "No effective snapshot present."
        if not effective_version_dir
        else "Effective snapshot is missing chunk index input.",
        {
            "effective_version_id": effective_version_id,
            "path": str(effective_chunk_input) if effective_chunk_input else None,
            "candidates": [str(v) for v in effective_chunk_candidates],
        },
    )
    if effective_chunk_input is not None:
        effective_chunk_payload, effective_chunk_error = _load_json_artifact(effective_chunk_input)
        if effective_chunk_error:
            _check(
                checks,
                "effective_chunk_index_input_json_load",
                False,
                "required",
                f"Effective chunk index input JSON load failed: {effective_chunk_error}",
                {"path": str(effective_chunk_input)},
            )
        effective_chunk_count = (
            len(effective_chunk_payload)
            if isinstance(effective_chunk_payload, list)
            else 0
        )
        _check(
            checks,
            "effective_chunk_index_input_non_empty",
            effective_chunk_count > 0,
            "required",
            "Effective chunk index input is non-empty."
            if effective_chunk_count > 0
            else "Effective chunk index input is empty or invalid.",
            {
                "path": str(effective_chunk_input),
                "chunk_count": effective_chunk_count,
            },
        )

    effective_wage_input = (
        effective_index_inputs_dir / artifacts["wages"].name
        if effective_index_inputs_dir
        else None
    )
    effective_wage_required = bool(effective_version_dir and wages_exists)
    _check(
        checks,
        "effective_wage_index_input_exists",
        (not effective_wage_required)
        or (effective_wage_input is not None and effective_wage_input.exists()),
        "required",
        "Effective snapshot includes wage index input."
        if effective_wage_required and effective_wage_input is not None and effective_wage_input.exists()
        else "Effective wage index input not required."
        if not effective_wage_required
        else "Effective snapshot is missing wage index input.",
        {
            "effective_version_id": effective_version_id,
            "path": str(effective_wage_input) if effective_wage_input else None,
            "required": effective_wage_required,
        },
    )
    if effective_wage_required and effective_wage_input is not None and effective_wage_input.exists():
        effective_wage_payload, effective_wage_error = _load_json_artifact(effective_wage_input)
        if effective_wage_error:
            _check(
                checks,
                "effective_wage_index_input_json_load",
                False,
                "required",
                f"Effective wage index input JSON load failed: {effective_wage_error}",
                {"path": str(effective_wage_input)},
            )
        effective_wage_rows = (
            len((effective_wage_payload or {}).get("canonical_wage_rows") or [])
            if isinstance(effective_wage_payload, dict)
            else 0
        )
        _check(
            checks,
            "effective_wage_index_input_non_empty",
            effective_wage_rows > 0,
            "required",
            "Effective wage index input is non-empty."
            if effective_wage_rows > 0
            else "Effective wage index input is empty or invalid.",
            {
                "path": str(effective_wage_input),
                "canonical_wage_row_count": effective_wage_rows,
            },
        )

    effective_entitlement_input = (
        effective_index_inputs_dir / artifacts["entitlements"].name
        if effective_index_inputs_dir
        else None
    )
    effective_entitlement_required = bool(
        effective_version_dir and (entitlements_exists or requires_entitlements)
    )
    _check(
        checks,
        "effective_entitlement_index_input_exists",
        (not effective_entitlement_required)
        or (
            effective_entitlement_input is not None
            and effective_entitlement_input.exists()
        ),
        "required",
        "Effective snapshot includes entitlement index input."
        if (
            effective_entitlement_required
            and effective_entitlement_input is not None
            and effective_entitlement_input.exists()
        )
        else "Effective entitlement index input not required."
        if not effective_entitlement_required
        else "Effective snapshot is missing entitlement index input.",
        {
            "effective_version_id": effective_version_id,
            "path": str(effective_entitlement_input) if effective_entitlement_input else None,
            "required": effective_entitlement_required,
        },
    )
    if (
        effective_entitlement_required
        and effective_entitlement_input is not None
        and effective_entitlement_input.exists()
    ):
        effective_entitlement_payload, effective_entitlement_error = _load_json_artifact(
            effective_entitlement_input
        )
        if effective_entitlement_error:
            _check(
                checks,
                "effective_entitlement_index_input_json_load",
                False,
                "required",
                f"Effective entitlement index input JSON load failed: {effective_entitlement_error}",
                {"path": str(effective_entitlement_input)},
            )
        effective_schedule_count = (
            len((effective_entitlement_payload or {}).get("vacation_entitlements") or [])
            if isinstance(effective_entitlement_payload, dict)
            else 0
        )
        _check(
            checks,
            "effective_entitlement_index_input_non_empty",
            effective_schedule_count > 0,
            "required",
            "Effective entitlement index input is non-empty."
            if effective_schedule_count > 0
            else "Effective entitlement index input is empty or invalid.",
            {
                "path": str(effective_entitlement_input),
                "vacation_schedule_count": effective_schedule_count,
            },
        )

    # Classification ontology checks
    ontology_path = artifacts["classification_ontology"]
    requires_ontology = bool(manifest and (manifest.get("classifications") or []))
    ontology_exists = ontology_path.exists()
    _check(
        checks,
        "classification_ontology_exists",
        ontology_exists or not requires_ontology,
        "required",
        "Classification ontology artifact is present."
        if ontology_exists
        else "Classification ontology artifact missing for manifest classifications."
        if requires_ontology
        else "Classification ontology not required (no manifest classifications).",
        {"path": str(ontology_path), "requires_ontology": requires_ontology},
    )

    if ontology_exists:
        try:
            classification_ontology = _load_json(ontology_path)
        except Exception as exc:
            classification_ontology = None
            _check(
                checks,
                "classification_ontology_json_load",
                False,
                "required",
                f"Classification ontology JSON load failed: {exc}",
            )

    if classification_ontology:
        schema_ok = classification_ontology.get("schema_version") in {
            "classification_ontology_v1",
            "classification_ontology_v2",
        }
        contract_match_ok = classification_ontology.get("contract_id") == contract_id
        _check(
            checks,
            "classification_ontology_schema_valid",
            schema_ok and contract_match_ok,
            "required",
            "Classification ontology schema and contract_id are valid."
            if schema_ok and contract_match_ok
            else "Classification ontology schema_version or contract_id mismatch.",
            {
                "schema_version": classification_ontology.get("schema_version"),
                "ontology_contract_id": classification_ontology.get("contract_id"),
            },
        )

        alias_map = classification_ontology.get("alias_to_wage_key") or {}
        alias_map = alias_map if isinstance(alias_map, dict) else {}
        wage_keys = set((wages_data or {}).get("classifications", {}).keys())
        invalid_alias_targets = sorted(
            f"{src}->{dst}"
            for src, dst in alias_map.items()
            if str(dst) and wage_keys and str(dst) not in wage_keys
        )
        _check(
            checks,
            "classification_ontology_alias_integrity",
            len(invalid_alias_targets) == 0,
            "required",
            "All ontology alias targets resolve to wage keys."
            if not invalid_alias_targets
            else "Some ontology alias targets do not resolve to wage keys.",
            {"invalid_alias_targets": invalid_alias_targets[:40]},
        )

        manifest_keys = set()
        for raw in (manifest.get("classifications", []) if manifest else []):
            normalized = _normalize_classification_label(raw)
            if normalized:
                manifest_keys.add(normalized)

        decisions = classification_ontology.get("decisions") or []
        decisions = decisions if isinstance(decisions, list) else []
        decision_keys = {
            str(d.get("source_key"))
            for d in decisions
            if isinstance(d, dict) and str(d.get("source_key") or "").strip()
        }
        missing_decisions = sorted(manifest_keys - decision_keys)
        _check(
            checks,
            "classification_ontology_manifest_decisions",
            len(missing_decisions) == 0,
            "required",
            "Ontology contains deterministic mapping decisions for all manifest classes."
            if not missing_decisions
            else "Ontology is missing mapping decisions for some manifest classes.",
            {"missing_decisions": missing_decisions[:40]},
        )

        ontology_summary = classification_ontology.get("summary") or {}
        ontology_coverage = float(ontology_summary.get("coverage", 0.0) or 0.0)
        _check(
            checks,
            "classification_ontology_mapping_coverage",
            ontology_coverage >= 0.5,
            "advisory",
            "Ontology mapping coverage meets advisory threshold."
            if ontology_coverage >= 0.5
            else "Ontology mapping coverage below advisory threshold.",
            {
                "coverage": round(ontology_coverage, 4),
                "covered_manifest_classes": ontology_summary.get("covered_manifest_classes"),
                "actionable_manifest_classes": ontology_summary.get("actionable_manifest_classes"),
                "resolved_manifest_classes": ontology_summary.get("resolved_manifest_classes"),
                "clarification_manifest_classes": ontology_summary.get("clarification_manifest_classes"),
                "out_of_scope_manifest_classes": ontology_summary.get("out_of_scope_manifest_classes"),
                "total_manifest_classes": ontology_summary.get("total_manifest_classes"),
                "unresolved_manifest_keys": (ontology_summary.get("unresolved_manifest_keys") or [])[:40],
            },
        )

    # Role catalog checks
    role_catalog_path = artifacts["role_catalog"]
    requires_role_catalog = bool(manifest and (manifest.get("classifications") or []))
    role_catalog_exists = role_catalog_path.exists()
    _check(
        checks,
        "role_catalog_exists",
        role_catalog_exists or not requires_role_catalog,
        "required",
        "Role catalog artifact is present."
        if role_catalog_exists
        else "Role catalog artifact missing for manifest classifications."
        if requires_role_catalog
        else "Role catalog not required (no manifest classifications).",
        {"path": str(role_catalog_path), "requires_role_catalog": requires_role_catalog},
    )
    if role_catalog_exists:
        try:
            role_catalog = _load_json(role_catalog_path)
        except Exception as exc:
            role_catalog = None
            _check(
                checks,
                "role_catalog_json_load",
                False,
                "required",
                f"Role catalog JSON load failed: {exc}",
            )

    if role_catalog:
        roles = role_catalog.get("roles") or []
        schema_ok = role_catalog.get("schema_version") in {
            "role_catalog_v1",
            "role_catalog_v2",
        }
        contract_match_ok = role_catalog.get("contract_id") == contract_id
        roles_ok = isinstance(roles, list)
        roles_non_empty_ok = bool(roles) if requires_role_catalog else True
        _check(
            checks,
            "role_catalog_schema_valid",
            schema_ok and contract_match_ok and roles_ok and roles_non_empty_ok,
            "required",
            "Role catalog schema and contract_id are valid."
            if schema_ok and contract_match_ok and roles_ok and roles_non_empty_ok
            else "Role catalog schema_version/contract_id/roles are invalid.",
            {
                "schema_version": role_catalog.get("schema_version"),
                "catalog_contract_id": role_catalog.get("contract_id"),
                "role_count": len(roles) if isinstance(roles, list) else None,
            },
        )

        default_unmapped_roles = []
        unresolved_manifest_roles = []
        manifest_roles_total = 0
        if isinstance(roles, list):
            for role in roles:
                if not isinstance(role, dict):
                    continue
                role_value = str(role.get("value") or "").strip()
                is_default = bool(role.get("onboarding_default"))
                wage_available = bool(role.get("wage_available"))
                manifest_present = bool(role.get("manifest_present"))

                if is_default and not wage_available:
                    default_unmapped_roles.append(role_value)
                if manifest_present:
                    manifest_roles_total += 1
                    if not wage_available:
                        unresolved_manifest_roles.append(role_value)

        _check(
            checks,
            "role_catalog_onboarding_default_wage_ready",
            len(default_unmapped_roles) == 0,
            "required",
            "All onboarding-default roles are wage-available."
            if len(default_unmapped_roles) == 0
            else "Some onboarding-default roles are not wage-available.",
            {"default_unmapped_roles": default_unmapped_roles[:40]},
        )

        role_catalog_summary = role_catalog.get("summary") or {}
        summary_unresolved_manifest_roles = role_catalog_summary.get("unresolved_manifest_roles")
        if summary_unresolved_manifest_roles is not None:
            unresolved_manifest_roles = list(summary_unresolved_manifest_roles)
        manifest_roles_total = int(
            role_catalog_summary.get("actionable_manifest_roles")
            or role_catalog_summary.get("manifest_roles")
            or manifest_roles_total
            or 0
        )
        clarification_manifest_roles = list(role_catalog_summary.get("clarification_manifest_roles") or [])
        out_of_scope_manifest_roles = list(role_catalog_summary.get("out_of_scope_manifest_roles") or [])
        unresolved_rate = (
            len(unresolved_manifest_roles) / manifest_roles_total
            if manifest_roles_total > 0 else 0.0
        )
        _check(
            checks,
            "role_catalog_unresolved_manifest_rate",
            unresolved_rate <= 0.4,
            "advisory",
            "Role catalog unresolved-manifest rate is within advisory threshold."
            if unresolved_rate <= 0.4
            else "Role catalog unresolved-manifest rate exceeds advisory threshold.",
            {
                "manifest_roles_total": manifest_roles_total,
                "unresolved_manifest_roles": len(unresolved_manifest_roles),
                "unresolved_manifest_rate": round(unresolved_rate, 4),
                "clarification_manifest_roles": clarification_manifest_roles[:40],
                "out_of_scope_manifest_roles": out_of_scope_manifest_roles[:40],
                "unresolved_manifest_role_values": unresolved_manifest_roles[:40],
            },
        )

    # Ingestion review queue checks (required when unresolved/ambiguous issues exist)
    review_queue_path = artifacts["ingestion_review_queue"]
    review_required = False
    unresolved_manifest_count = 0
    canonical_ambiguities_count = 0
    canonical_conflicts_count = 0
    unresolved_rows_count = 0

    if classification_ontology:
        summary = classification_ontology.get("summary", {}) or {}
        unresolved_manifest_count = int(summary.get("unresolved_manifest_classes", 0) or 0)
        if unresolved_manifest_count > 0:
            review_required = True
    if wages_data:
        extraction_meta = wages_data.get("extraction_metadata", {}) or {}
        canonical_ambiguities_count = len(extraction_meta.get("canonical_ambiguities", []) or [])
        canonical_conflicts_count = len(extraction_meta.get("canonical_conflicts", []) or [])
        unresolved_rows_count = len(extraction_meta.get("unresolved_rows", []) or [])
        if canonical_ambiguities_count > 0 or canonical_conflicts_count > 0 or unresolved_rows_count > 0:
            review_required = True

    review_queue_exists = review_queue_path.exists()
    _check(
        checks,
        "ingestion_review_queue_exists",
        review_queue_exists or not review_required,
        "required",
        "Ingestion review queue artifact is present."
        if review_queue_exists
        else "Ingestion review queue missing despite unresolved ingestion issues."
        if review_required
        else "Ingestion review queue not required (no unresolved/ambiguous ingestion issues).",
        {
            "path": str(review_queue_path),
            "review_required": review_required,
            "unresolved_manifest_classes": unresolved_manifest_count,
            "canonical_ambiguities": canonical_ambiguities_count,
            "canonical_conflicts": canonical_conflicts_count,
            "unresolved_rows": unresolved_rows_count,
        },
    )

    if review_queue_exists:
        try:
            ingestion_review_queue = _load_json(review_queue_path)
        except Exception as exc:
            ingestion_review_queue = None
            _check(
                checks,
                "ingestion_review_queue_json_load",
                False,
                "required",
                f"Ingestion review queue JSON load failed: {exc}",
            )

    if ingestion_review_queue:
        schema_ok = ingestion_review_queue.get("schema_version") == "ingestion_review_queue_v1"
        contract_match_ok = ingestion_review_queue.get("contract_id") == contract_id
        items = ingestion_review_queue.get("items", [])
        items_ok = isinstance(items, list)
        _check(
            checks,
            "ingestion_review_queue_schema_valid",
            schema_ok and contract_match_ok and items_ok,
            "required",
            "Ingestion review queue schema and contract_id are valid."
            if schema_ok and contract_match_ok and items_ok
            else "Ingestion review queue schema/contract_id/items are invalid.",
            {
                "schema_version": ingestion_review_queue.get("schema_version"),
                "queue_contract_id": ingestion_review_queue.get("contract_id"),
                "item_count": len(items) if isinstance(items, list) else None,
            },
        )

        queue_issue_ids = {
            str(item.get("issue_id") or "").strip()
            for item in items
            if isinstance(item, dict) and str(item.get("issue_id") or "").strip()
        }
        required_issue_ids = []
        if unresolved_manifest_count > 0:
            required_issue_ids.append("ontology_unresolved_manifest_classes")
        if canonical_ambiguities_count > 0:
            required_issue_ids.append("canonical_wage_ambiguities")
        if canonical_conflicts_count > 0:
            required_issue_ids.append("canonical_wage_conflicts")
        if unresolved_rows_count > 0:
            required_issue_ids.append("canonical_wage_unresolved_rows")
        missing_issue_ids = sorted(set(required_issue_ids) - queue_issue_ids)
        _check(
            checks,
            "ingestion_review_queue_issue_coverage",
            len(missing_issue_ids) == 0,
            "required",
            "Ingestion review queue covers all unresolved ingestion issue categories."
            if len(missing_issue_ids) == 0
            else "Ingestion review queue is missing unresolved ingestion issue categories.",
            {
                "required_issue_ids": sorted(set(required_issue_ids)),
                "queue_issue_ids": sorted(queue_issue_ids),
                "missing_issue_ids": missing_issue_ids,
            },
        )

    # Advisory: Appendix table evidence coverage
    if manifest and manifest.get("has_appendix_a"):
        table_count = len(table_registry) if isinstance(table_registry, list) else 0
        non_toc_tables = []
        appendix_like_tables = []
        for t in table_registry if isinstance(table_registry, list) else []:
            heading_path = " ".join(t.get("heading_path", []) or []).lower()
            markdown = str(t.get("markdown") or "").lower()
            if "table of contents" not in heading_path:
                non_toc_tables.append(t)
            if any(tok in heading_path or tok in markdown for tok in ("appendix", "wage", "rates of pay")):
                appendix_like_tables.append(t)

        article8_table_chunks = 0
        for c in chunks:
            if c.get("article_num") == 8 and (c.get("table_refs") or []):
                article8_table_chunks += 1

        appendix_signal = bool(appendix_like_tables) or article8_table_chunks > 0
        _check(
            checks,
            "appendix_table_coverage_signal",
            appendix_signal,
            "advisory",
            "Appendix/wage table coverage signal detected."
            if appendix_signal
            else "Appendix marked present but wage table coverage signal is weak.",
            {
                "table_count": table_count,
                "non_toc_table_count": len(non_toc_tables),
                "appendix_like_table_count": len(appendix_like_tables),
                "article8_table_chunks": article8_table_chunks,
            },
        )

    moa_wage_patch_metrics = _moa_wage_patch_metrics(package_dir)
    moa_wage_schedule_required = int(moa_wage_patch_metrics.get("moa_wage_op_count") or 0) > 0
    schedule_label_ops = int(moa_wage_patch_metrics.get("ops_with_selected_schedule_label") or 0)
    schedule_map_ops = int(moa_wage_patch_metrics.get("ops_with_source_rate_schedule") or 0)
    sync_metadata_missing = list(moa_wage_patch_metrics.get("patches_missing_sync_metadata") or [])
    sync_config_missing = list(moa_wage_patch_metrics.get("patches_missing_config_id") or [])

    _check(
        checks,
        "moa_wage_schedule_metadata_integrity",
        (not moa_wage_schedule_required)
        or (
            schedule_label_ops == int(moa_wage_patch_metrics.get("moa_wage_op_count") or 0)
            and schedule_map_ops == int(moa_wage_patch_metrics.get("moa_wage_op_count") or 0)
            and len(sync_metadata_missing) == 0
        ),
        "required",
        "MOA wage row ops preserve raw schedule metadata."
        if moa_wage_schedule_required
        and schedule_label_ops == int(moa_wage_patch_metrics.get("moa_wage_op_count") or 0)
        and schedule_map_ops == int(moa_wage_patch_metrics.get("moa_wage_op_count") or 0)
        and len(sync_metadata_missing) == 0
        else "MOA wage schedule metadata not required."
        if not moa_wage_schedule_required
        else "MOA wage row ops are missing selected schedule metadata or sync metadata.",
        {
            "required": moa_wage_schedule_required,
            "moa_wage_patch_count": moa_wage_patch_metrics.get("moa_wage_patch_count"),
            "moa_wage_op_count": moa_wage_patch_metrics.get("moa_wage_op_count"),
            "ops_with_selected_schedule_label": schedule_label_ops,
            "ops_with_source_rate_schedule": schedule_map_ops,
            "patches_missing_sync_metadata": sync_metadata_missing[:20],
            "source_doc_ids": moa_wage_patch_metrics.get("source_doc_ids") or [],
        },
    )
    _check(
        checks,
        "moa_wage_schedule_sync_registration",
        (not moa_wage_schedule_required) or len(sync_config_missing) == 0,
        "advisory",
        "MOA wage schedule sync metadata records a config id."
        if moa_wage_schedule_required and len(sync_config_missing) == 0
        else "MOA wage schedule sync registration not required."
        if not moa_wage_schedule_required
        else "MOA wage schedule sync metadata is missing config id.",
        {
            "required": moa_wage_schedule_required,
            "patches_missing_config_id": sync_config_missing[:20],
            "moa_wage_patch_count": moa_wage_patch_metrics.get("moa_wage_patch_count"),
            "source_doc_ids": moa_wage_patch_metrics.get("source_doc_ids") or [],
        },
    )

    effective_moa_provenance_metrics = _effective_moa_provenance_page_metrics(package_dir)
    moa_provenance_required = int(effective_moa_provenance_metrics.get("moa_ref_count") or 0) > 0
    missing_moa_pages = int(effective_moa_provenance_metrics.get("missing_page_ref_count") or 0)
    _check(
        checks,
        "effective_moa_provenance_page_integrity",
        (not moa_provenance_required) or missing_moa_pages == 0,
        "required",
        "Effective amended MOA provenance refs include navigable PDF pages."
        if moa_provenance_required and missing_moa_pages == 0
        else "Effective amended MOA provenance page integrity not required."
        if not moa_provenance_required
        else "Effective amended MOA provenance includes refs without PDF pages.",
        {
            "required": moa_provenance_required,
            "effective_version_id": effective_moa_provenance_metrics.get("effective_version_id"),
            "moa_ref_count": effective_moa_provenance_metrics.get("moa_ref_count"),
            "missing_page_ref_count": missing_moa_pages,
            "sections_missing_page": effective_moa_provenance_metrics.get("sections_missing_page") or [],
            "tables_missing_page": effective_moa_provenance_metrics.get("tables_missing_page") or [],
            "source_doc_ids": effective_moa_provenance_metrics.get("source_doc_ids") or [],
        },
    )

    contract_miss_records, miss_record_load_errors = _load_contract_miss_records(contract_id)
    miss_taxonomy_counts = Counter(
        str(row.get("taxonomy_type") or "").strip().lower()
        for row in contract_miss_records
        if isinstance(row, dict)
    )
    regression_added_records = [
        row for row in contract_miss_records
        if str(row.get("regression_status") or "").strip().lower() == "regression_added"
    ]
    missing_regression_case_ids = sorted(
        str(row.get("miss_id") or row.get("operator_label") or "<unknown>")
        for row in regression_added_records
        if not str(row.get("regression_case_id") or "").strip()
    )
    miss_backlog_linkage_ok = (
        len(miss_record_load_errors) == 0 and len(missing_regression_case_ids) == 0
    )
    _check(
        checks,
        "miss_record_backlog_linkage",
        miss_backlog_linkage_ok,
        "advisory",
        "Contract-scoped reviewed miss backlog is linked cleanly."
        if miss_backlog_linkage_ok and contract_miss_records
        else "No contract-scoped miss backlog records present."
        if not contract_miss_records and len(miss_record_load_errors) == 0
        else "Contract-scoped miss backlog has load or regression-linkage issues.",
        {
            "miss_record_count": len(contract_miss_records),
            "regression_added_count": len(regression_added_records),
            "missing_regression_case_ids": missing_regression_case_ids,
            "load_errors": miss_record_load_errors[:20],
            "taxonomy_counts": dict(sorted(miss_taxonomy_counts.items())),
        },
    )

    failing_check_ids = {
        str(row.get("id") or "")
        for row in checks
        if str(row.get("status") or "") == "fail"
    }
    mapped_related_checks = sorted(
        {
            check_id
            for taxonomy_type in miss_taxonomy_counts.keys()
            for check_id in MISS_TAXONOMY_CHECK_IDS.get(str(taxonomy_type), set())
        }
    )
    active_related_failures = sorted(set(mapped_related_checks) & failing_check_ids)
    _check(
        checks,
        "miss_record_signal_alignment",
        len(active_related_failures) == 0,
        "advisory",
        "Known contract-scoped miss signals align with current green pack capabilities."
        if len(active_related_failures) == 0 and contract_miss_records
        else "No contract-scoped miss signals to align."
        if not contract_miss_records
        else "Contract pack still fails capability checks associated with reviewed miss patterns.",
        {
            "miss_record_count": len(contract_miss_records),
            "mapped_related_checks": mapped_related_checks,
            "active_related_failures": active_related_failures,
            "taxonomy_counts": dict(sorted(miss_taxonomy_counts.items())),
        },
    )

    required_failed = [c["id"] for c in checks if c["severity"] == "required" and c["status"] == "fail"]
    advisory_failed = [c["id"] for c in checks if c["severity"] == "advisory" and c["status"] == "fail"]
    scorecard = {
        "scorecard_version": SCORECARD_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "strict_mode": strict,
        "contract_id": contract_id,
        "package_path": str(package_dir),
        "artifacts": {k: str(v) for k, v in artifacts.items()},
        "artifact_hashes": artifact_hashes,
        "pack_hash": _pack_hash(artifact_hashes) if artifact_hashes else None,
        "checks": checks,
        "capabilities": {
            "side_letter_doc_type_materialization": _check_status(
                checks, "side_letter_doc_type_materialization"
            ),
            "effective_snapshot_present": bool(effective_version_dir),
            "effective_chunk_index_input": _check_status(
                checks, "effective_chunk_index_input_non_empty"
            ) or _check_status(checks, "effective_chunk_index_input_exists"),
            "effective_wage_index_input": _check_status(
                checks, "effective_wage_index_input_non_empty"
            ) or _check_status(checks, "effective_wage_index_input_exists"),
            "effective_entitlement_index_input": _check_status(
                checks, "effective_entitlement_index_input_non_empty"
            ) or _check_status(checks, "effective_entitlement_index_input_exists"),
            "moa_wage_schedule_metadata_integrity": _check_status(
                checks, "moa_wage_schedule_metadata_integrity"
            ),
            "moa_wage_schedule_sync_registration": _check_status(
                checks, "moa_wage_schedule_sync_registration"
            ),
            "effective_moa_provenance_page_integrity": _check_status(
                checks, "effective_moa_provenance_page_integrity"
            ),
            "ingestion_review_queue_issue_coverage": _check_status(
                checks, "ingestion_review_queue_issue_coverage"
            ),
            "miss_records_present": len(contract_miss_records) > 0,
            "miss_record_count": len(contract_miss_records),
            "miss_record_backlog_linkage": _check_status(
                checks, "miss_record_backlog_linkage"
            ),
            "miss_record_signal_alignment": _check_status(
                checks, "miss_record_signal_alignment"
            ),
        },
        "summary": {
            "required_failed": required_failed,
            "advisory_failed": advisory_failed,
            "pass": (len(required_failed) == 0 and (not strict or len(advisory_failed) == 0)),
        },
    }

    if write_scorecard:
        out_dir = package_dir / "pack"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "pack_scorecard.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(scorecard, f, indent=2, ensure_ascii=False)
        scorecard["scorecard_path"] = str(out_path)

    return scorecard


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate contract package acceptance gates.")
    parser.add_argument(
        "--package",
        required=True,
        help="Package directory name under data/contracts (or absolute path).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat advisory failures as blocking failures.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write pack_scorecard.json to package/pack/",
    )
    args = parser.parse_args()

    pkg_arg = Path(args.package)
    package_dir = pkg_arg if pkg_arg.is_absolute() else (CONTRACTS_ROOT / args.package)
    if not package_dir.exists():
        print(f"[FAIL] Package not found: {package_dir}")
        return 1

    scorecard = evaluate_contract_pack(
        package_dir=package_dir,
        strict=args.strict,
        write_scorecard=not args.no_write,
    )

    summary = scorecard.get("summary", {})
    print("=" * 72)
    print("KARL Contract Pack Acceptance")
    print("=" * 72)
    print(f"Contract: {scorecard.get('contract_id')}")
    print(f"Pack hash: {scorecard.get('pack_hash')}")
    print(f"Strict mode: {args.strict}")
    print(f"Pass: {summary.get('pass')}")
    print(f"Required failed: {summary.get('required_failed', [])}")
    print(f"Advisory failed: {summary.get('advisory_failed', [])}")
    if scorecard.get("scorecard_path"):
        print(f"Scorecard: {scorecard['scorecard_path']}")

    return 0 if summary.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
