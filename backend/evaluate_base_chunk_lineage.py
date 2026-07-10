"""Audit base-chunk lineage risk for Contract-tab `source_view=base` text."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.api import _build_contract_history_payload, _resolve_base_chunk_file  # reuse viewer logic
from backend.chunk_files import resolve_chunk_file
from backend.config import DATA_DIR, MANIFESTS_DIR, TEST_SET_DIR


SCHEMA_VERSION = "base_chunk_lineage_report_v1"
DEFAULT_OUTPUT = TEST_SET_DIR / "base_chunk_lineage_report.json"


def _path_str(path: Optional[Path]) -> Optional[str]:
    if not path:
        return None
    try:
        return str(path.resolve())
    except Exception:
        return str(path)


def _is_under_effective_snapshot(path: Optional[Path]) -> bool:
    if not path:
        return False
    parts = [str(p).lower() for p in path.parts]
    return "effective" in parts and "index_inputs" in parts


def _risk_level(*, has_effective_snapshot: bool, patch_count: int, base_missing: bool, base_equals_effective: bool, base_under_effective_snapshot: bool) -> str:
    if not has_effective_snapshot and patch_count <= 0:
        return "low"
    if base_missing:
        return "high"
    if base_under_effective_snapshot:
        return "high"
    if has_effective_snapshot and (patch_count > 0) and base_equals_effective:
        return "high"
    if has_effective_snapshot and base_equals_effective:
        return "medium"
    return "low"


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _load_json(path: Optional[Path]) -> Optional[object]:
    if not path or not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_chunks_from_path(path: Optional[Path], contract_id: str) -> list[dict]:
    payload = _load_json(path)
    if not isinstance(payload, list):
        return []
    rows: list[dict] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        row_contract = str(row.get("contract_id") or "").strip()
        if row_contract and row_contract != contract_id:
            continue
        rows.append(row)
    return rows


def _section_content_from_chunks(chunks: list[dict], article_num: int, section_num: int) -> Optional[str]:
    matches = [
        row for row in chunks
        if int(row.get("article_num") or 0) == article_num
        and int(row.get("section_num") or 0) == section_num
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
    return "\n\n".join(parts).strip() if parts else None


def _load_replace_section_targets(contract_id: str) -> list[dict]:
    amendments_dir = DATA_DIR / "contracts" / contract_id / "amendments"
    if not amendments_dir.exists():
        return []
    targets: list[dict] = []
    for patch_path in sorted(amendments_dir.glob("*.json")):
        payload = _load_json(patch_path)
        if not isinstance(payload, dict):
            continue
        patch_id = str(payload.get("patch_id") or patch_path.stem)
        for idx, op in enumerate(payload.get("operations") or []):
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
            targets.append(
                {
                    "patch_id": patch_id,
                    "op_index": idx,
                    "article_num": article_num,
                    "section_num": section_num,
                }
            )
    return targets


def _patched_section_content_check(contract_id: str, base_chunk_path: Optional[Path], effective_chunk_path: Optional[Path]) -> dict:
    targets = _load_replace_section_targets(contract_id)
    if not targets:
        return {
            "replace_section_ops": 0,
            "checked": 0,
            "equal_content_count": 0,
            "different_content_count": 0,
            "missing_in_base_count": 0,
            "missing_in_effective_count": 0,
            "samples": [],
        }

    base_chunks = _load_chunks_from_path(base_chunk_path, contract_id)
    effective_chunks = _load_chunks_from_path(effective_chunk_path, contract_id)
    samples: list[dict] = []
    checked = 0
    equal_count = 0
    diff_count = 0
    missing_base = 0
    missing_effective = 0

    for target in targets:
        article_num = int(target["article_num"])
        section_num = int(target["section_num"])
        base_text = _section_content_from_chunks(base_chunks, article_num, section_num)
        effective_text = _section_content_from_chunks(effective_chunks, article_num, section_num)
        if base_text is None:
            missing_base += 1
        if effective_text is None:
            missing_effective += 1
        if base_text is None or effective_text is None:
            status = "missing"
        else:
            checked += 1
            if _normalize_text(base_text) == _normalize_text(effective_text):
                equal_count += 1
                status = "equal"
            else:
                diff_count += 1
                status = "different"
        if len(samples) < 8:
            samples.append(
                {
                    **target,
                    "status": status,
                    "base_has_content": base_text is not None,
                    "effective_has_content": effective_text is not None,
                    "base_preview": (str(base_text or "")[:180] if base_text else None),
                    "effective_preview": (str(effective_text or "")[:180] if effective_text else None),
                }
            )

    return {
        "replace_section_ops": len(targets),
        "checked": checked,
        "equal_content_count": equal_count,
        "different_content_count": diff_count,
        "missing_in_base_count": missing_base,
        "missing_in_effective_count": missing_effective,
        "samples": samples,
    }


def _audit_contract(contract_id: str) -> dict:
    effective_chunk_path = resolve_chunk_file(contract_id=contract_id, allow_shared_fallback=True)
    base_chunk_path = _resolve_base_chunk_file(contract_id)
    history = _build_contract_history_payload(contract_id)
    patch_count = int(history.get("patch_count") or 0)
    has_effective_snapshot = bool(history.get("has_effective_snapshot"))

    effective_str = _path_str(effective_chunk_path)
    base_str = _path_str(base_chunk_path)
    base_equals_effective = bool(effective_str and base_str and effective_str == base_str)
    base_under_effective = _is_under_effective_snapshot(base_chunk_path)
    effective_under_effective = _is_under_effective_snapshot(effective_chunk_path)
    base_missing = base_chunk_path is None
    patched_content_check = _patched_section_content_check(contract_id, base_chunk_path, effective_chunk_path)
    patched_equal_all = (
        int(patched_content_check.get("replace_section_ops") or 0) > 0
        and int(patched_content_check.get("checked") or 0) > 0
        and int(patched_content_check.get("different_content_count") or 0) == 0
        and int(patched_content_check.get("equal_content_count") or 0) == int(patched_content_check.get("checked") or 0)
    )
    risk = _risk_level(
        has_effective_snapshot=has_effective_snapshot,
        patch_count=patch_count,
        base_missing=base_missing,
        base_equals_effective=base_equals_effective,
        base_under_effective_snapshot=base_under_effective,
    )
    if patched_equal_all and patch_count > 0:
        risk = "high"

    findings: list[str] = []
    if base_missing:
        findings.append("base_chunk_missing")
    if base_under_effective:
        findings.append("base_chunk_points_into_effective_snapshot")
    if base_equals_effective:
        findings.append("base_chunk_path_equals_effective_chunk_path")
    if has_effective_snapshot and patch_count > 0 and base_equals_effective:
        findings.append("amended_contract_base_text_likely_not_immutable")
    if patched_equal_all and patch_count > 0:
        findings.append("patched_replace_section_targets_equal_between_base_and_effective")

    return {
        "contract_id": contract_id,
        "has_effective_snapshot": has_effective_snapshot,
        "patch_count": patch_count,
        "effective_chunk_path": effective_str,
        "base_chunk_path": base_str,
        "effective_chunk_under_effective_snapshot": effective_under_effective,
        "base_chunk_under_effective_snapshot": base_under_effective,
        "base_chunk_missing": base_missing,
        "base_chunk_path_equals_effective_chunk_path": base_equals_effective,
        "risk_level": risk,
        "findings": findings,
        "patched_section_content_check": patched_content_check,
    }


def build_report() -> dict:
    contract_ids = sorted(p.stem for p in MANIFESTS_DIR.glob("*.json"))
    contracts = [_audit_contract(cid) for cid in contract_ids]
    summary = {
        "total_contracts": len(contracts),
        "high_risk": sum(1 for c in contracts if c["risk_level"] == "high"),
        "medium_risk": sum(1 for c in contracts if c["risk_level"] == "medium"),
        "low_risk": sum(1 for c in contracts if c["risk_level"] == "low"),
        "missing_base_chunk": sum(1 for c in contracts if c["base_chunk_missing"]),
        "base_equals_effective": sum(1 for c in contracts if c["base_chunk_path_equals_effective_chunk_path"]),
        "patched_section_equal_all": sum(
            1
            for c in contracts
            if int(((c.get("patched_section_content_check") or {}).get("replace_section_ops") or 0)) > 0
            and "patched_replace_section_targets_equal_between_base_and_effective" in (c.get("findings") or [])
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "summary": summary,
        "contracts": contracts,
    }


def save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def main() -> None:
    report = build_report()
    save_report(DEFAULT_OUTPUT, report)
    summary = report.get("summary") or {}
    print(
        "[OK] base chunk lineage audit:",
        f"contracts={summary.get('total_contracts', 0)}",
        f"high={summary.get('high_risk', 0)}",
        f"medium={summary.get('medium_risk', 0)}",
        f"low={summary.get('low_risk', 0)}",
    )
    for row in report.get("contracts") or []:
        if row.get("risk_level") != "high":
            continue
        print(
            "  -",
            row.get("contract_id"),
            "| findings:",
            ",".join(row.get("findings") or []) or "none",
        )


if __name__ == "__main__":
    main()
