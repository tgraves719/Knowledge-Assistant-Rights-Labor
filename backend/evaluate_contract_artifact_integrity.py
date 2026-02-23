"""Cross-contract artifact integrity evaluator.

Audits, per contract:
- chunk representation (doc_type mix, LOU/LOA lexical presence)
- wage table row representation/linkage
- PDF/table navigation artifact presence
- effective snapshot integrity when present
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import DATA_DIR
from backend.effective_contracts import resolve_latest_effective_version_id


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _discover_contract_ids(explicit: Optional[list[str]]) -> list[str]:
    if explicit:
        return sorted({str(v).strip() for v in explicit if str(v).strip()})
    root = DATA_DIR / "contracts"
    if not root.exists():
        return []
    out = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if p.name == "contractid":
            continue
        out.append(p.name)
    return out


def _discover_base_chunks_path(contract_id: str) -> Optional[Path]:
    root = DATA_DIR / "contracts" / contract_id
    candidates = [
        root / "chunks" / f"contract_chunks_enriched_{contract_id}.json",
        root / "chunks" / "contract_chunks_enriched.json",
        root / "chunks" / f"contract_chunks_smart_{contract_id}.json",
        root / "chunks" / "contract_chunks_smart.json",
        DATA_DIR / "chunks" / f"contract_chunks_enriched_{contract_id}.json",
        root / "base" / "contract_chunks_enriched.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _discover_base_wages_path(contract_id: str) -> Optional[Path]:
    root = DATA_DIR / "contracts" / contract_id
    candidates = [
        root / "base" / "wage_tables.json",
        root / "wages" / f"wage_tables_{contract_id}.json",
        root / "wages" / "wage_tables.json",
        DATA_DIR / "wages" / f"wage_tables_{contract_id}.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _discover_pdf_nav_path(contract_id: str) -> Optional[Path]:
    candidates = [
        DATA_DIR / "ontologies" / f"pdf_nav_index_{contract_id}.json",
        DATA_DIR / "contracts" / contract_id / "ontology" / f"pdf_nav_index_{contract_id}.json",
        DATA_DIR / "contracts" / contract_id / "ontology" / "pdf_nav_index.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _discover_table_nav_path(contract_id: str) -> Optional[Path]:
    candidates = [
        DATA_DIR / "ontologies" / f"table_nav_index_{contract_id}.json",
        DATA_DIR / "contracts" / contract_id / "ontology" / f"table_nav_index_{contract_id}.json",
        DATA_DIR / "contracts" / contract_id / "ontology" / "table_nav_index.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _effective_chunks_path(contract_id: str, effective_version_id: Optional[str]) -> Optional[Path]:
    if not effective_version_id:
        return None
    path = (
        DATA_DIR
        / "contracts"
        / contract_id
        / "effective"
        / effective_version_id
        / "index_inputs"
        / f"contract_chunks_enriched_{contract_id}.json"
    )
    return path if path.exists() else None


def _effective_wages_path(contract_id: str, effective_version_id: Optional[str]) -> Optional[Path]:
    if not effective_version_id:
        return None
    path = (
        DATA_DIR
        / "contracts"
        / contract_id
        / "effective"
        / effective_version_id
        / "index_inputs"
        / f"wage_tables_{contract_id}.json"
    )
    return path if path.exists() else None


def _chunk_metrics(chunks: list[dict]) -> dict:
    total = len(chunks)
    doc_type_counts = Counter((str(c.get("doc_type") or "").lower() or "unknown") for c in chunks if isinstance(c, dict))
    with_anchor = sum(1 for c in chunks if isinstance(c, dict) and str(c.get("anchor_id") or "").strip())
    with_provenance = sum(
        1 for c in chunks
        if isinstance(c, dict) and isinstance(c.get("provenance"), list) and len(c.get("provenance")) > 0
    )
    with_prov_page = sum(
        1 for c in chunks
        if isinstance(c, dict)
        and any(isinstance(r, dict) and r.get("pdf_page") is not None for r in (c.get("provenance") or []))
    )

    loa_text_hits = 0
    lou_text_hits = 0
    for c in chunks:
        if not isinstance(c, dict):
            continue
        blob = (
            str(c.get("citation") or "")
            + "\n"
            + str(c.get("content_with_tables") or "")
            + "\n"
            + str(c.get("content") or "")
        ).lower()
        if "letter of agreement" in blob:
            loa_text_hits += 1
        if "letter of understanding" in blob:
            lou_text_hits += 1

    return {
        "chunk_total": total,
        "doc_type_counts": dict(sorted(doc_type_counts.items())),
        "with_anchor_count": with_anchor,
        "with_provenance_count": with_provenance,
        "with_provenance_page_count": with_prov_page,
        "loa_text_hits": loa_text_hits,
        "lou_text_hits": lou_text_hits,
    }


def _wage_metrics(payload: dict) -> dict:
    rows = payload.get("canonical_wage_rows") if isinstance(payload, dict) else []
    rows = rows if isinstance(rows, list) else []
    total = len(rows)
    with_table_id = 0
    with_row_index = 0
    with_row_key = 0
    with_provenance = 0
    for r in rows:
        if not isinstance(r, dict):
            continue
        source_ref = r.get("source_reference") if isinstance(r.get("source_reference"), dict) else {}
        if str(source_ref.get("table_id") or "").strip():
            with_table_id += 1
        if source_ref.get("row_index") is not None:
            with_row_index += 1
        if str(r.get("row_key") or "").strip():
            with_row_key += 1
        if isinstance(r.get("provenance"), list) and len(r.get("provenance")) > 0:
            with_provenance += 1
    return {
        "canonical_wage_row_total": total,
        "rows_with_table_id": with_table_id,
        "rows_with_row_index": with_row_index,
        "rows_with_row_key": with_row_key,
        "rows_with_provenance": with_provenance,
    }


def _nav_metrics(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return {}
    stats = payload.get("stats") if isinstance(payload.get("stats"), dict) else {}
    table_pages = payload.get("table_pages") if isinstance(payload.get("table_pages"), dict) else {}
    section_pages = payload.get("section_pages") if isinstance(payload.get("section_pages"), dict) else {}
    article_pages = payload.get("article_pages") if isinstance(payload.get("article_pages"), dict) else {}
    section_entries = 0
    if section_pages:
        for value in section_pages.values():
            if isinstance(value, dict):
                section_entries += len(value)
            else:
                section_entries += 1
    return {
        "pdf_filename": payload.get("pdf_filename"),
        "total_pages": payload.get("total_pages"),
        "stats": stats,
        "table_pages_count": len(table_pages),
        "section_pages_count": len(section_pages),
        "section_page_entries": section_entries,
        "article_pages_count": len(article_pages),
    }


def run(
    contract_ids: Optional[list[str]] = None,
    *,
    strict_side_letter_buckets: bool = False,
    side_letter_hit_threshold: int = 1,
) -> dict:
    ids = _discover_contract_ids(contract_ids)
    results = []

    for contract_id in ids:
        base_chunks_path = _discover_base_chunks_path(contract_id)
        base_wages_path = _discover_base_wages_path(contract_id)
        pdf_nav_path = _discover_pdf_nav_path(contract_id)
        table_nav_path = _discover_table_nav_path(contract_id)
        effective_version_id = resolve_latest_effective_version_id(contract_id)
        effective_chunks_path = _effective_chunks_path(contract_id, effective_version_id)
        effective_wages_path = _effective_wages_path(contract_id, effective_version_id)

        missing = []
        if not base_chunks_path:
            missing.append("base_chunks")
        if not base_wages_path:
            missing.append("base_wages")
        if not pdf_nav_path:
            missing.append("pdf_nav_index")
        if not table_nav_path:
            missing.append("table_nav_index")

        base_chunks_payload = _load_json(base_chunks_path) if base_chunks_path else None
        base_chunks_list = base_chunks_payload if isinstance(base_chunks_payload, list) else []
        base_chunk_metrics = _chunk_metrics(base_chunks_list)

        base_wages_payload = _load_json(base_wages_path) if base_wages_path else None
        base_wage_metrics = _wage_metrics(base_wages_payload if isinstance(base_wages_payload, dict) else {})

        pdf_nav_payload = _load_json(pdf_nav_path) if pdf_nav_path else None
        table_nav_payload = _load_json(table_nav_path) if table_nav_path else None

        effective_chunk_metrics = None
        effective_wage_metrics = None
        effective_integrity = None
        if effective_version_id and effective_chunks_path:
            payload = _load_json(effective_chunks_path)
            arr = payload if isinstance(payload, list) else []
            effective_chunk_metrics = _chunk_metrics(arr)
        if effective_version_id and effective_wages_path:
            payload = _load_json(effective_wages_path)
            effective_wage_metrics = _wage_metrics(payload if isinstance(payload, dict) else {})

        if effective_chunk_metrics:
            base_counts = base_chunk_metrics.get("doc_type_counts") or {}
            eff_counts = effective_chunk_metrics.get("doc_type_counts") or {}
            missing_doc_types = [
                k for k, v in sorted(base_counts.items())
                if int(v) > 0 and int(eff_counts.get(k, 0)) == 0
            ]
            reduced_doc_types = [
                k for k, v in sorted(base_counts.items())
                if int(eff_counts.get(k, 0)) < int(v)
            ]
            effective_integrity = {
                "missing_doc_types_vs_base": missing_doc_types,
                "reduced_doc_types_vs_base": reduced_doc_types,
                "base_chunk_total": int(base_chunk_metrics.get("chunk_total", 0)),
                "effective_chunk_total": int(effective_chunk_metrics.get("chunk_total", 0)),
            }

        doc_type_counts = base_chunk_metrics.get("doc_type_counts") or {}
        loa_bucket_count = int(doc_type_counts.get("loa", 0))
        lou_bucket_count = int(doc_type_counts.get("lou", 0))
        side_letter_bucket_count = loa_bucket_count + lou_bucket_count
        side_letter_text_hits = int(base_chunk_metrics.get("loa_text_hits", 0)) + int(base_chunk_metrics.get("lou_text_hits", 0))
        side_letter_bucket_required = side_letter_text_hits >= int(side_letter_hit_threshold)
        side_letter_bucket_ok = (not side_letter_bucket_required) or side_letter_bucket_count > 0

        checks = {
            "has_base_chunks": bool(base_chunks_path and base_chunk_metrics.get("chunk_total", 0) > 0),
            "has_base_wages": bool(base_wages_path and base_wage_metrics.get("canonical_wage_row_total", 0) > 0),
            "has_pdf_nav": bool(pdf_nav_path),
            "has_table_nav": bool(table_nav_path),
            "wage_rows_have_table_id": bool(
                base_wage_metrics.get("canonical_wage_row_total", 0) == base_wage_metrics.get("rows_with_table_id", -1)
            ),
            "wage_rows_have_row_index": bool(
                base_wage_metrics.get("canonical_wage_row_total", 0) == base_wage_metrics.get("rows_with_row_index", -1)
            ),
        }
        if effective_integrity is not None:
            checks["effective_has_missing_doc_types"] = len(effective_integrity.get("missing_doc_types_vs_base") or []) == 0
        if strict_side_letter_buckets:
            checks["side_letter_hits_have_bucket"] = side_letter_bucket_ok

        passed = all(bool(v) for v in checks.values())

        results.append(
            {
                "contract_id": contract_id,
                "effective_version_id": effective_version_id,
                "paths": {
                    "base_chunks": str(base_chunks_path) if base_chunks_path else None,
                    "base_wages": str(base_wages_path) if base_wages_path else None,
                    "pdf_nav_index": str(pdf_nav_path) if pdf_nav_path else None,
                    "table_nav_index": str(table_nav_path) if table_nav_path else None,
                    "effective_chunks": str(effective_chunks_path) if effective_chunks_path else None,
                    "effective_wages": str(effective_wages_path) if effective_wages_path else None,
                },
                "missing_artifacts": missing,
                "base_chunks": base_chunk_metrics,
                "base_wages": base_wage_metrics,
                "pdf_nav": _nav_metrics(pdf_nav_payload if isinstance(pdf_nav_payload, dict) else {}),
                "table_nav": _nav_metrics(table_nav_payload if isinstance(table_nav_payload, dict) else {}),
                "effective_chunks": effective_chunk_metrics,
                "effective_wages": effective_wage_metrics,
                "effective_integrity": effective_integrity,
                "side_letter_policy": {
                    "strict_side_letter_buckets": bool(strict_side_letter_buckets),
                    "side_letter_hit_threshold": int(side_letter_hit_threshold),
                    "side_letter_text_hits": side_letter_text_hits,
                    "side_letter_bucket_required": side_letter_bucket_required,
                    "side_letter_bucket_ok": side_letter_bucket_ok,
                    "doc_type_side_letter_counts": {
                        "loa": loa_bucket_count,
                        "lou": lou_bucket_count,
                    },
                },
                "checks": checks,
                "pass": passed,
            }
        )

    passed = sum(1 for r in results if bool(r.get("pass")))
    total = len(results)
    return {
        "schema_version": "contract_artifact_integrity_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "strict_side_letter_buckets": bool(strict_side_letter_buckets),
            "side_letter_hit_threshold": int(side_letter_hit_threshold),
        },
        "overall": {
            "passed": passed,
            "total": total,
            "pass_rate": round((passed / total) if total else 0.0, 4),
        },
        "results": results,
    }


def _write_report(report: dict, out_path: Optional[Path] = None) -> Path:
    target = out_path or (DATA_DIR / "test_set" / "contract_artifact_integrity_results.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit contract artifact integrity across contracts.")
    parser.add_argument("--contract-id", action="append", default=None, help="Contract ID (repeatable)")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON path")
    parser.add_argument(
        "--strict-side-letter-buckets",
        action="store_true",
        help="Fail when LOA/LOU lexical hits exceed threshold but neither doc_type=loa nor doc_type=lou exists.",
    )
    parser.add_argument(
        "--side-letter-hit-threshold",
        type=int,
        default=1,
        help="Minimum LOA/LOU lexical hits before strict side-letter bucket rule applies (default: 1).",
    )
    args = parser.parse_args()

    report = run(
        contract_ids=args.contract_id,
        strict_side_letter_buckets=bool(args.strict_side_letter_buckets),
        side_letter_hit_threshold=max(1, int(args.side_letter_hit_threshold)),
    )
    out_path = _write_report(report, out_path=Path(args.output) if args.output else None)
    overall = report.get("overall") or {}
    policy = report.get("policy") or {}
    print("=" * 72)
    print("KARL Contract Artifact Integrity")
    print("=" * 72)
    print(
        f"Pass: {int(overall.get('passed', 0))}/{int(overall.get('total', 0))} "
        f"({float(overall.get('pass_rate', 0.0)):.1%})"
    )
    print(
        "Policy: "
        f"strict_side_letter_buckets={bool(policy.get('strict_side_letter_buckets'))}, "
        f"side_letter_hit_threshold={int(policy.get('side_letter_hit_threshold', 1))}"
    )
    print(f"Results: {out_path}")
    for row in report.get("results") or []:
        side_letter = row.get("side_letter_policy") if isinstance(row.get("side_letter_policy"), dict) else {}
        print(
            f"- {row.get('contract_id')}: pass={row.get('pass')} "
            f"base_chunks={row.get('base_chunks', {}).get('chunk_total')} "
            f"effective={row.get('effective_chunks', {}).get('chunk_total') if isinstance(row.get('effective_chunks'), dict) else None} "
            f"lou_bucket={side_letter.get('doc_type_side_letter_counts', {}).get('lou', 0)} "
            f"loa_hits={row.get('base_chunks', {}).get('loa_text_hits', 0)} "
            f"lou_hits={row.get('base_chunks', {}).get('lou_text_hits', 0)}"
        )
    return 0 if int(overall.get("passed", 0)) == int(overall.get("total", 0)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
