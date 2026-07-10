"""Sync missing patch source page refs from a regenerated draft patch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional


def _load_patch(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Patch payload at {path} is not an object")
    return payload


def _operation_signature(operation: dict[str, Any]) -> Optional[tuple[str, ...]]:
    op = str(operation.get("op") or "").strip()
    target = operation.get("target") or {}
    if not isinstance(target, dict) or not op:
        return None
    if op == "replace_section":
        anchor_id = str(target.get("anchor_id") or "").strip()
        if anchor_id:
            return (op, "anchor_id", anchor_id)
        article_num = str(target.get("article_num") or "").strip()
        section_num = str(target.get("section_num") or "").strip()
        if article_num and section_num:
            return (op, "article_section", article_num, section_num)
        return None
    if op == "replace_table_row":
        table_id = str(target.get("table_id") or "").strip()
        row_key = str(target.get("row_key") or "").strip()
        if table_id and row_key:
            return (op, table_id, row_key)
    return None


def _matching_draft_ref(source_ref: dict[str, Any], draft_refs: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    source_doc_id = str(source_ref.get("source_doc_id") or "").strip()
    source_type = str(source_ref.get("source_type") or "").strip()
    source_pdf = str(source_ref.get("pdf") or "").strip()
    for draft_ref in draft_refs:
        if not isinstance(draft_ref, dict):
            continue
        if source_doc_id and str(draft_ref.get("source_doc_id") or "").strip() != source_doc_id:
            continue
        if source_type and str(draft_ref.get("source_type") or "").strip() != source_type:
            continue
        draft_pdf = str(draft_ref.get("pdf") or "").strip()
        if source_pdf and draft_pdf and draft_pdf != source_pdf:
            continue
        if isinstance(draft_ref.get("pdf_page"), int) and int(draft_ref["pdf_page"]) > 0:
            return draft_ref
    for draft_ref in draft_refs:
        if not isinstance(draft_ref, dict):
            continue
        if source_doc_id:
            draft_doc_id = str(draft_ref.get("source_doc_id") or "").strip()
            if draft_doc_id and draft_doc_id != source_doc_id:
                continue
        if isinstance(draft_ref.get("pdf_page"), int) and int(draft_ref["pdf_page"]) > 0:
            return draft_ref
    return None


def sync_patch_source_pages(approved_payload: dict[str, Any], draft_payload: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    approved_operations = approved_payload.get("operations") or []
    draft_operations = draft_payload.get("operations") or []
    if not isinstance(approved_operations, list) or not isinstance(draft_operations, list):
        raise ValueError("Patch payloads must include operations arrays")

    draft_map: dict[tuple[str, ...], dict[str, Any]] = {}
    for draft_op in draft_operations:
        if not isinstance(draft_op, dict):
            continue
        signature = _operation_signature(draft_op)
        if signature is not None:
            draft_map[signature] = draft_op

    changed: list[dict[str, Any]] = []
    for operation in approved_operations:
        if not isinstance(operation, dict):
            continue
        signature = _operation_signature(operation)
        if signature is None:
            continue
        draft_op = draft_map.get(signature)
        if not isinstance(draft_op, dict):
            continue
        source_refs = operation.get("source_refs") or []
        draft_refs = draft_op.get("source_refs") or []
        if not isinstance(source_refs, list) or not isinstance(draft_refs, list):
            continue
        for idx, source_ref in enumerate(source_refs):
            if not isinstance(source_ref, dict):
                continue
            if isinstance(source_ref.get("pdf_page"), int) and int(source_ref["pdf_page"]) > 0:
                continue
            match = _matching_draft_ref(source_ref, draft_refs)
            if match is None:
                continue
            source_ref["pdf_page"] = int(match["pdf_page"])
            changed.append(
                {
                    "operation": operation.get("op"),
                    "signature": list(signature),
                    "source_ref_index": idx,
                    "pdf_page": int(match["pdf_page"]),
                }
            )
    return approved_payload, changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync missing patch source pages from a regenerated draft patch")
    parser.add_argument("--approved-patch", required=True, help="Approved patch JSON path")
    parser.add_argument("--draft-patch", required=True, help="Draft patch JSON path")
    parser.add_argument("--output-path", default=None, help="Optional output path; defaults to approved path with .synced suffix")
    parser.add_argument("--in-place", action="store_true", help="Overwrite the approved patch in place")
    args = parser.parse_args()

    approved_path = Path(args.approved_patch)
    draft_path = Path(args.draft_patch)
    approved_payload = _load_patch(approved_path)
    draft_payload = _load_patch(draft_path)
    synced_payload, changes = sync_patch_source_pages(approved_payload, draft_payload)

    if args.in_place:
        output_path = approved_path
    elif args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = approved_path.with_name(f"{approved_path.stem}.synced.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(synced_payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "approved_patch": str(approved_path),
                "draft_patch": str(draft_path),
                "output_path": str(output_path),
                "changed_count": len(changes),
                "changes": changes,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
