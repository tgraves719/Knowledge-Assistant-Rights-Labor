"""Backfill side-letter doc_type buckets in chunk artifacts.

Purpose:
- Promote likely side-letter chunks to `doc_type=loa` or `doc_type=lou`
- Keep changes minimal and deterministic
- Support dry-run + JSON report for auditability
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.config import DATA_DIR


_DATE_TOKEN_RE = re.compile(r"\bdated\s+\d{1,2}/\d{1,2}/\d{2,4}\b", re.IGNORECASE)
_SEG_TOKEN_RE = re.compile(r"seg[_-]?\d+", re.IGNORECASE)


def _discover_contract_ids(explicit: Optional[list[str]]) -> list[str]:
    if explicit:
        return sorted({str(v).strip() for v in explicit if str(v).strip()})
    root = DATA_DIR / "contracts"
    if not root.exists():
        return []
    out = []
    for p in sorted(root.iterdir()):
        if not p.is_dir() or p.name == "contractid":
            continue
        out.append(p.name)
    return out


def _discover_chunk_files(contract_id: str) -> list[Path]:
    root = DATA_DIR / "contracts" / contract_id / "chunks"
    shared = DATA_DIR / "chunks"
    candidates = [
        root / f"contract_chunks_enriched_{contract_id}.json",
        root / "contract_chunks_enriched.json",
        root / f"contract_chunks_smart_{contract_id}.json",
        root / "contract_chunks_smart.json",
        shared / f"contract_chunks_enriched_{contract_id}.json",
        shared / f"contract_chunks_smart_{contract_id}.json",
    ]
    return [p for p in candidates if p.exists()]


def _is_side_letter_term(text: str) -> tuple[bool, bool]:
    lower = str(text or "").lower()
    return ("letter of agreement" in lower, "letter of understanding" in lower)


def _classify_target_doc_type(meta_text: str, content_text: str) -> str:
    loa_meta, lou_meta = _is_side_letter_term(meta_text)
    loa_content, lou_content = _is_side_letter_term(content_text)
    loa_hits = int(loa_meta) + int(loa_content)
    lou_hits = int(lou_meta) + int(lou_content)
    if loa_hits > lou_hits:
        return "loa"
    if lou_hits > loa_hits:
        return "lou"
    if loa_hits > 0:
        return "loa"
    return "lou"


def _is_segment_like(citation: str, chunk_id: str) -> bool:
    citation_l = str(citation or "").lower()
    chunk_id_l = str(chunk_id or "").lower()
    if " part " in f" {citation_l} ":
        return True
    if _SEG_TOKEN_RE.search(chunk_id_l):
        return True
    return False


def _is_strong_side_letter_signal(content_text: str) -> bool:
    lower = str(content_text or "").lower()
    if "original document was signed" in lower:
        return True
    if _DATE_TOKEN_RE.search(lower):
        return True
    return False


def _select_candidate_indexes(chunks: list[dict], max_changes: int) -> list[int]:
    meta_candidates: list[int] = []
    strong_candidates: list[int] = []
    segmented_candidates: list[int] = []

    for idx, row in enumerate(chunks):
        if not isinstance(row, dict):
            continue
        current_doc_type = str(row.get("doc_type") or "").strip().lower()
        if current_doc_type in {"loa", "lou"}:
            continue

        citation = str(row.get("citation") or "")
        article_title = str(row.get("article_title") or "")
        parent_context = str(row.get("parent_context") or "")
        content = str(row.get("content_with_tables") or row.get("content") or "")

        meta_text = "\n".join([citation, article_title, parent_context])
        loa_meta, lou_meta = _is_side_letter_term(meta_text)
        loa_content, lou_content = _is_side_letter_term(content)
        has_any = loa_meta or lou_meta or loa_content or lou_content
        if not has_any:
            continue

        if loa_meta or lou_meta:
            meta_candidates.append(idx)
            continue
        if _is_strong_side_letter_signal(content):
            strong_candidates.append(idx)
            continue
        if _is_segment_like(citation=citation, chunk_id=str(row.get("chunk_id") or "")):
            segmented_candidates.append(idx)
            continue

    ordered = meta_candidates + strong_candidates + segmented_candidates
    if max_changes > 0:
        return ordered[:max_changes]
    return ordered


def _update_chunk_payload(chunks: list[dict], max_changes: int) -> tuple[list[dict], list[dict]]:
    updated = [dict(row) if isinstance(row, dict) else row for row in chunks]
    selected = _select_candidate_indexes(chunks=updated, max_changes=max_changes)
    changes: list[dict] = []
    for idx in selected:
        row = updated[idx]
        if not isinstance(row, dict):
            continue
        citation = str(row.get("citation") or "")
        article_title = str(row.get("article_title") or "")
        parent_context = str(row.get("parent_context") or "")
        content = str(row.get("content_with_tables") or row.get("content") or "")
        meta_text = "\n".join([citation, article_title, parent_context])
        new_doc_type = _classify_target_doc_type(meta_text=meta_text, content_text=content)
        old_doc_type = str(row.get("doc_type") or "cba").strip().lower() or "cba"
        if old_doc_type == new_doc_type:
            continue
        row["doc_type"] = new_doc_type
        changes.append(
            {
                "index": idx,
                "chunk_id": str(row.get("chunk_id") or ""),
                "citation": citation,
                "old_doc_type": old_doc_type,
                "new_doc_type": new_doc_type,
            }
        )
    return updated, changes


def run(contract_ids: Optional[list[str]], apply: bool, max_changes_per_file: int) -> dict:
    ids = _discover_contract_ids(contract_ids)
    results = []
    for contract_id in ids:
        files = _discover_chunk_files(contract_id)
        file_results = []
        total_changes = 0
        for path in files:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                file_results.append(
                    {
                        "path": str(path),
                        "status": "error",
                        "error": "invalid_json",
                        "changes": [],
                    }
                )
                continue
            if not isinstance(payload, list):
                file_results.append(
                    {
                        "path": str(path),
                        "status": "skipped",
                        "error": "not_list_payload",
                        "changes": [],
                    }
                )
                continue

            updated, changes = _update_chunk_payload(payload, max_changes=max_changes_per_file)
            total_changes += len(changes)
            if apply and changes:
                path.write_text(json.dumps(updated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            file_results.append(
                {
                    "path": str(path),
                    "status": "updated" if changes else "no_change",
                    "change_count": len(changes),
                    "changes": changes,
                }
            )

        results.append(
            {
                "contract_id": contract_id,
                "file_count": len(files),
                "change_count": total_changes,
                "files": file_results,
            }
        )

    changed_contracts = sum(1 for row in results if int(row.get("change_count") or 0) > 0)
    return {
        "schema_version": "side_letter_doc_type_backfill_v1",
        "apply": bool(apply),
        "max_changes_per_file": int(max_changes_per_file),
        "contracts_total": len(results),
        "contracts_changed": changed_contracts,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill side-letter doc_type buckets in chunk artifacts.")
    parser.add_argument("--contract-id", action="append", default=None, help="Contract ID (repeatable)")
    parser.add_argument("--apply", action="store_true", help="Write changes in place; default is dry-run")
    parser.add_argument(
        "--max-changes-per-file",
        type=int,
        default=32,
        help="Safety cap for updates per file (default: 32; set <=0 for unlimited)",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional report JSON path")
    args = parser.parse_args()

    max_changes = int(args.max_changes_per_file)
    report = run(
        contract_ids=args.contract_id,
        apply=bool(args.apply),
        max_changes_per_file=max_changes,
    )
    out_path = Path(args.output) if args.output else (DATA_DIR / "test_set" / "side_letter_doc_type_backfill_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
