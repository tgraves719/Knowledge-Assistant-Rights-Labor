"""Deterministic artifact-level regression for Contract-tab effective/base compare targets.

Validates that known amended sections produce distinct text between base chunks and
effective snapshot chunks, which is the data prerequisite for the frontend compare UI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Optional

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.chunk_files import resolve_chunk_file
from backend.config import DATA_DIR
from backend.api import _resolve_base_chunk_file


SCHEMA_VERSION = "contract_text_compare_amended_eval_v1"
DEFAULT_INPUT = DATA_DIR / "test_set" / "contract_text_compare_amended_targets_test.json"
DEFAULT_OUTPUT = DATA_DIR / "test_set" / "contract_text_compare_amended_results.json"


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_approved_replace_section_inventory(*, data_dir: Optional[Path] = None) -> dict[str, Any]:
    root = data_dir or DATA_DIR
    contracts_root = root / "contracts"
    inventory_rows: list[dict[str, Any]] = []
    if not contracts_root.exists():
        return {
            "contracts": [],
            "operations": [],
            "summary": {
                "contracts_with_approved_replace_section_ops": 0,
                "approved_replace_section_ops": 0,
            },
        }

    for contract_dir in sorted(contracts_root.iterdir(), key=lambda p: p.name):
        if not contract_dir.is_dir():
            continue
        amendments_dir = contract_dir / "amendments"
        if not amendments_dir.exists():
            continue
        for patch_path in sorted(amendments_dir.glob("*.json")):
            try:
                payload = _load_json(patch_path)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            patch_id = str(payload.get("patch_id") or patch_path.stem)
            contract_id = str(payload.get("contract_id") or contract_dir.name).strip()
            for op_index, op in enumerate(payload.get("operations") or []):
                if not isinstance(op, dict):
                    continue
                if str(op.get("op") or "").strip() != "replace_section":
                    continue
                if str(op.get("review_status") or "").strip().lower() != "approved":
                    continue
                target = op.get("target") or {}
                try:
                    article_num = int(target.get("article_num"))
                    section_num = int(target.get("section_num"))
                except Exception:
                    continue
                if article_num <= 0 or section_num <= 0:
                    continue
                inventory_rows.append(
                    {
                        "contract_id": contract_id,
                        "patch_id": patch_id,
                        "op_index": int(op_index),
                        "article_num": article_num,
                        "section_num": section_num,
                        "anchor_id": str(target.get("anchor_id") or ""),
                        "patch_path": str(patch_path),
                    }
                )

    inventory_rows.sort(
        key=lambda r: (
            str(r.get("contract_id") or ""),
            str(r.get("patch_id") or ""),
            int(r.get("article_num") or 0),
            int(r.get("section_num") or 0),
            int(r.get("op_index") or 0),
        )
    )
    contract_counts: dict[str, int] = {}
    for row in inventory_rows:
        cid = str(row.get("contract_id") or "")
        contract_counts[cid] = contract_counts.get(cid, 0) + 1
    contracts = [
        {"contract_id": cid, "approved_replace_section_ops": contract_counts[cid]}
        for cid in sorted(contract_counts)
    ]
    return {
        "contracts": contracts,
        "operations": inventory_rows,
        "summary": {
            "contracts_with_approved_replace_section_ops": len(contracts),
            "approved_replace_section_ops": len(inventory_rows),
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    return path


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_chunk_rows(path: Optional[Path], contract_id: str) -> list[dict[str, Any]]:
    if not path or not path.exists():
        return []
    try:
        payload = _load_json(path)
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    rows: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        row_contract_id = str(row.get("contract_id") or "").strip()
        if row_contract_id and row_contract_id != contract_id:
            continue
        rows.append(row)
    return rows


def _section_text_from_rows(rows: list[dict[str, Any]], article_num: int, section_num: int) -> Optional[str]:
    matches = [
        row
        for row in rows
        if int(row.get("article_num") or 0) == article_num and int(row.get("section_num") or 0) == section_num
    ]
    if not matches:
        return None
    matches.sort(key=lambda row: str(row.get("subsection") or ""))
    parts: list[str] = []
    for row in matches:
        content = str(row.get("content_with_tables") or row.get("content") or "").strip()
        if not content:
            continue
        subsection = str(row.get("subsection") or "").strip()
        if subsection and len(matches) > 1:
            parts.append(f"{subsection}: {content}")
        else:
            parts.append(content)
    if not parts:
        return None
    return "\n\n".join(parts).strip()


def _contains_any(text: Optional[str], needles: list[str]) -> bool:
    if not needles:
        return True
    hay = _normalize_text(text or "").lower()
    return any(_normalize_text(n).lower() in hay for n in needles if str(n or "").strip())


def _evaluate_case(case: dict[str, Any], chunk_cache: dict[tuple[str, str], list[dict[str, Any]]]) -> dict[str, Any]:
    case_id = str(case.get("id") or "").strip()
    contract_id = str(case.get("contract_id") or "").strip()
    article_num = int(case.get("article_num") or 0)
    section_num = int(case.get("section_num") or 0)
    expect_difference = bool(case.get("expect_difference", True))
    base_must_contain_any = [str(x) for x in (case.get("base_must_contain_any") or []) if str(x or "").strip()]
    effective_must_contain_any = [str(x) for x in (case.get("effective_must_contain_any") or []) if str(x or "").strip()]

    base_chunk_path = _resolve_base_chunk_file(contract_id)
    effective_chunk_path = resolve_chunk_file(contract_id=contract_id, allow_shared_fallback=True)

    base_key = ("base", contract_id)
    if base_key not in chunk_cache:
        chunk_cache[base_key] = _load_chunk_rows(base_chunk_path, contract_id)
    effective_key = ("effective", contract_id)
    if effective_key not in chunk_cache:
        chunk_cache[effective_key] = _load_chunk_rows(effective_chunk_path, contract_id)

    base_text = _section_text_from_rows(chunk_cache[base_key], article_num, section_num)
    effective_text = _section_text_from_rows(chunk_cache[effective_key], article_num, section_num)
    base_found = base_text is not None
    effective_found = effective_text is not None

    norm_base = _normalize_text(base_text or "")
    norm_effective = _normalize_text(effective_text or "")
    texts_differ = bool(base_found and effective_found and norm_base != norm_effective)
    diff_expectation_ok = texts_differ == expect_difference if (base_found and effective_found) else False
    base_snippet_ok = _contains_any(base_text, base_must_contain_any)
    effective_snippet_ok = _contains_any(effective_text, effective_must_contain_any)

    passed = base_found and effective_found and diff_expectation_ok and base_snippet_ok and effective_snippet_ok
    return {
        "id": case_id,
        "description": str(case.get("description") or ""),
        "contract_id": contract_id,
        "article_num": article_num,
        "section_num": section_num,
        "expect_difference": expect_difference,
        "base_chunk_path": str(base_chunk_path) if base_chunk_path else None,
        "effective_chunk_path": str(effective_chunk_path) if effective_chunk_path else None,
        "base_found": base_found,
        "effective_found": effective_found,
        "texts_differ": texts_differ,
        "diff_expectation_ok": diff_expectation_ok,
        "base_snippet_ok": base_snippet_ok,
        "effective_snippet_ok": effective_snippet_ok,
        "base_len": len(norm_base),
        "effective_len": len(norm_effective),
        "base_hash": _sha256_text(norm_base) if base_found else None,
        "effective_hash": _sha256_text(norm_effective) if effective_found else None,
        "base_preview": (base_text or "")[:220] if base_text else None,
        "effective_preview": (effective_text or "")[:220] if effective_text else None,
        "pass": passed,
    }


def _dataset_target_key(case: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(case.get("contract_id") or "").strip(),
        int(case.get("article_num") or 0),
        int(case.get("section_num") or 0),
    )


def _build_coverage(payload: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    inventory = discover_approved_replace_section_inventory()
    inventory_ops = list(inventory.get("operations") or [])
    inventory_contracts = [str((row or {}).get("contract_id") or "") for row in list(inventory.get("contracts") or [])]
    dataset_cases = [row for row in list(payload.get("test_cases") or []) if isinstance(row, dict)]
    dataset_keys = {_dataset_target_key(row) for row in dataset_cases}
    dataset_contracts = sorted({key[0] for key in dataset_keys if key[0]})

    inventory_keys = {
        (
            str(row.get("contract_id") or "").strip(),
            int(row.get("article_num") or 0),
            int(row.get("section_num") or 0),
        )
        for row in inventory_ops
    }
    covered_ops = sum(1 for key in inventory_keys if key in dataset_keys)
    total_ops = len(inventory_keys)
    covered_contracts = sorted({key[0] for key in inventory_keys if key in dataset_keys and key[0]})
    total_contracts = len(set(inventory_contracts))
    uncovered_contracts = sorted(set(inventory_contracts) - set(covered_contracts))
    missing_targets = [
        {
            "contract_id": str(row.get("contract_id") or ""),
            "patch_id": str(row.get("patch_id") or ""),
            "article_num": int(row.get("article_num") or 0),
            "section_num": int(row.get("section_num") or 0),
            "op_index": int(row.get("op_index") or 0),
        }
        for row in inventory_ops
        if (
            str(row.get("contract_id") or "").strip(),
            int(row.get("article_num") or 0),
            int(row.get("section_num") or 0),
        )
        not in dataset_keys
    ]
    result_failures = [
        {
            "id": str(row.get("id") or ""),
            "contract_id": str(row.get("contract_id") or ""),
            "article_num": int(row.get("article_num") or 0),
            "section_num": int(row.get("section_num") or 0),
        }
        for row in results
        if not bool(row.get("pass"))
    ]

    return {
        "inventory_summary": inventory.get("summary") or {},
        "dataset_contracts": dataset_contracts,
        "covered_contracts": covered_contracts,
        "uncovered_contracts": uncovered_contracts,
        "contract_coverage_rate": round((len(covered_contracts) / total_contracts) if total_contracts else 0.0, 4),
        "approved_replace_section_ops_total": total_ops,
        "approved_replace_section_ops_covered": covered_ops,
        "operation_coverage_rate": round((covered_ops / total_ops) if total_ops else 0.0, 4),
        "missing_targets_count": len(missing_targets),
        "missing_targets_sample": missing_targets[:10],
        "failing_cases_count": len(result_failures),
        "failing_cases_sample": result_failures[:10],
    }


def run(*, input_path: Optional[Path] = None, output_path: Optional[Path] = None) -> dict[str, Any]:
    test_file = input_path or DEFAULT_INPUT
    payload = _load_json(test_file)
    cases = [row for row in list(payload.get("test_cases") or []) if isinstance(row, dict)]
    chunk_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
    results = [_evaluate_case(case, chunk_cache) for case in cases]

    passed = sum(1 for row in results if bool(row.get("pass")))
    total = len(results)
    missing_base = sum(1 for row in results if not bool(row.get("base_found")))
    missing_effective = sum(1 for row in results if not bool(row.get("effective_found")))
    non_diff_failures = sum(
        1
        for row in results
        if bool(row.get("base_found")) and bool(row.get("effective_found")) and not bool(row.get("diff_expectation_ok"))
    )

    report = {
        "schema_version": SCHEMA_VERSION,
        "dataset_schema_version": str(payload.get("schema_version") or ""),
        "test_file": str(test_file),
        "overall": {
            "passed": passed,
            "total": total,
            "pass_rate": round((passed / total) if total else 0.0, 4),
            "pass": passed == total,
        },
        "summary": {
            "missing_base_count": missing_base,
            "missing_effective_count": missing_effective,
            "diff_expectation_failures": non_diff_failures,
        },
        "coverage": _build_coverage(payload, results),
        "results": results,
    }
    _write_json(output_path or DEFAULT_OUTPUT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate amended section text compare targets (base vs effective).")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    report = run(input_path=Path(args.input), output_path=Path(args.output))
    overall = report.get("overall") or {}
    print("=" * 72)
    print("KARL Contract Text Compare (Amended Targets)")
    print("=" * 72)
    print(
        f"Overall: {int(overall.get('passed', 0))}/{int(overall.get('total', 0))} "
        f"({float(overall.get('pass_rate', 0.0)):.1%})"
    )
    print(f"Pass: {bool(overall.get('pass'))}")
    print(f"Results: {args.output}")
    return 0 if bool(overall.get("pass")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
