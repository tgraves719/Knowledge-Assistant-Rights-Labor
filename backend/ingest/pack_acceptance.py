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
from backend.validate_manifests import validate_manifest
from backend.ingest.extract_wages import lookup_wage, normalize_classification_name


CONTRACTS_ROOT = DATA_DIR / "contracts"
SCORECARD_VERSION = "contract_pack_scorecard_v1"


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


def _resolve_artifacts(package_dir: Path, contract_id: str) -> dict[str, Path]:
    source_dir = package_dir / "source"
    manifests_dir = package_dir / "manifests"
    chunks_dir = package_dir / "chunks"
    tables_dir = package_dir / "tables"
    wages_dir = package_dir / "wages"
    entitlements_dir = package_dir / "entitlements"
    ontology_dir = package_dir / "ontology"

    md_candidates = sorted(source_dir.glob("*.md"))
    json_candidates = sorted(source_dir.glob("*.json"))

    return {
        "source_md": md_candidates[0] if md_candidates else source_dir / f"{contract_id}.md",
        "source_json": json_candidates[0] if json_candidates else source_dir / f"{contract_id}.json",
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
    role_catalog = None
    ingestion_review_queue = None

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

    # Article coverage against manifest
    if manifest and chunks:
        article_titles = manifest.get("article_titles", {}) or {}
        expected_articles = set()
        for key in article_titles.keys():
            try:
                expected_articles.add(int(key))
            except (TypeError, ValueError):
                continue

        found_articles = {
            int(c.get("article_num")) for c in chunks
            if isinstance(c.get("article_num"), int)
        }
        missing_articles = sorted(expected_articles - found_articles)
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
            resolved_manifest = []
            unresolved_manifest = []
            for raw_label in normalized_manifest_classes:
                if lookup_wage(wages_data, raw_label, 0, 0) is None:
                    unresolved_manifest.append(raw_label)
                else:
                    resolved_manifest.append(raw_label)
            ratio = (
                len(resolved_manifest) / len(normalized_manifest_classes)
                if normalized_manifest_classes else 1.0
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
                    "resolved": len(resolved_manifest),
                    "total": len(normalized_manifest_classes),
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
        schema_ok = classification_ontology.get("schema_version") == "classification_ontology_v1"
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
                "resolved_manifest_classes": ontology_summary.get("resolved_manifest_classes"),
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
        schema_ok = role_catalog.get("schema_version") == "role_catalog_v1"
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
