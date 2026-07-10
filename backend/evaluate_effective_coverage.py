"""Evaluate base-vs-effective chunk coverage integrity."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import DATA_DIR
from backend.effective_contracts import resolve_latest_effective_version_id


def _load_json_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    out: list[dict] = []
    for row in payload:
        if isinstance(row, dict):
            out.append(row)
    return out


def _doc_type_counts(chunks: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in chunks:
        doc_type = str(row.get("doc_type") or "").strip().lower() or "unknown"
        counts[doc_type] = int(counts.get(doc_type, 0)) + 1
    return dict(sorted(counts.items()))


def _discover_contract_ids(explicit: Optional[list[str]]) -> list[str]:
    if explicit:
        return sorted({str(v).strip() for v in explicit if str(v).strip()})

    out: list[str] = []
    contracts_root = DATA_DIR / "contracts"
    if not contracts_root.exists():
        return []
    for contract_dir in sorted(contracts_root.iterdir()):
        if not contract_dir.is_dir():
            continue
        contract_id = contract_dir.name
        if (contract_dir / "effective" / "latest.json").exists():
            out.append(contract_id)
    return out


def run(contract_ids: Optional[list[str]] = None) -> dict:
    rows: list[dict] = []
    ids = _discover_contract_ids(contract_ids)

    for contract_id in ids:
        base_path = DATA_DIR / "contracts" / contract_id / "base" / "contract_chunks_enriched.json"
        effective_version_id = resolve_latest_effective_version_id(contract_id)
        effective_path = (
            DATA_DIR
            / "contracts"
            / contract_id
            / "effective"
            / str(effective_version_id or "")
            / "index_inputs"
            / f"contract_chunks_enriched_{contract_id}.json"
        )

        base_chunks = _load_json_list(base_path) if base_path.exists() else []
        effective_chunks = _load_json_list(effective_path) if effective_path.exists() else []
        base_counts = _doc_type_counts(base_chunks)
        effective_counts = _doc_type_counts(effective_chunks)

        missing_files = []
        if not base_path.exists():
            missing_files.append(str(base_path))
        if not effective_version_id:
            missing_files.append("latest effective version")
        if effective_version_id and not effective_path.exists():
            missing_files.append(str(effective_path))

        reduced_doc_types = []
        missing_doc_types = []
        increased_doc_types = []
        for doc_type, base_count in sorted(base_counts.items()):
            eff_count = int(effective_counts.get(doc_type, 0))
            if eff_count < base_count:
                reduced_doc_types.append(doc_type)
            if base_count > 0 and eff_count == 0:
                missing_doc_types.append(doc_type)
        for doc_type, eff_count in sorted(effective_counts.items()):
            base_count = int(base_counts.get(doc_type, 0))
            if eff_count > base_count:
                increased_doc_types.append(doc_type)

        total_ok = len(base_chunks) == len(effective_chunks)
        coverage_ok = len(reduced_doc_types) == 0 and len(missing_doc_types) == 0
        pass_row = (not missing_files) and total_ok and coverage_ok

        rows.append(
            {
                "contract_id": contract_id,
                "effective_version_id": effective_version_id,
                "base_chunk_total": len(base_chunks),
                "effective_chunk_total": len(effective_chunks),
                "base_doc_type_counts": base_counts,
                "effective_doc_type_counts": effective_counts,
                "missing_files": missing_files,
                "reduced_doc_types": reduced_doc_types,
                "missing_doc_types": missing_doc_types,
                "increased_doc_types": increased_doc_types,
                "pass": pass_row,
            }
        )

    passed = sum(1 for row in rows if bool(row.get("pass")))
    total = len(rows)
    return {
        "schema_version": "effective_snapshot_coverage_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall": {
            "passed": passed,
            "total": total,
            "pass_rate": round((passed / total) if total else 0.0, 4),
        },
        "results": rows,
    }


def _write_report(report: dict, out_path: Optional[Path] = None) -> Path:
    target = out_path or (DATA_DIR / "test_set" / "effective_snapshot_coverage_results.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate base/effective chunk coverage consistency.")
    parser.add_argument("--contract-id", action="append", default=None, help="Contract ID to validate (repeatable)")
    parser.add_argument("--output", type=str, default=None, help="Optional output results path")
    args = parser.parse_args()

    report = run(contract_ids=args.contract_id)
    out_path = _write_report(report, out_path=Path(args.output) if args.output else None)
    overall = report.get("overall") or {}
    print("=" * 72)
    print("KARL Effective Snapshot Coverage")
    print("=" * 72)
    print(
        f"Pass: {int(overall.get('passed', 0))}/{int(overall.get('total', 0))} "
        f"({float(overall.get('pass_rate', 0.0)):.1%})"
    )
    print(f"Results: {out_path}")
    for row in report.get("results") or []:
        print(
            f"- {row.get('contract_id')}: pass={row.get('pass')} "
            f"base={row.get('base_chunk_total')} effective={row.get('effective_chunk_total')} "
            f"missing_doc_types={row.get('missing_doc_types')}"
        )
    return 0 if int(overall.get("passed", 0)) == int(overall.get("total", 0)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
