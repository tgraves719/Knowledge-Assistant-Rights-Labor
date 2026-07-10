"""Deterministic table navigation index helpers."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend.config import DATA_DIR
from backend.pdf_nav_index import resolve_contract_source_json_path


TABLE_NAV_INDEX_SCHEMA_VERSION = "table_nav_index_v1"

_TABLE_PATH_RE = re.compile(r"pages\[(\d+)\]", re.IGNORECASE)


def source_dir_for_contract(contract_id: str) -> Path:
    return DATA_DIR / "contracts" / str(contract_id or "") / "source"


def resolve_structured_tables_path(contract_id: str) -> Optional[Path]:
    package_path = DATA_DIR / "contracts" / str(contract_id or "") / "tables" / "structured_tables.json"
    if package_path.exists():
        return package_path

    shared_contract = DATA_DIR / "tables" / f"structured_tables_{contract_id}.json"
    if shared_contract.exists():
        return shared_contract

    shared_default = DATA_DIR / "tables" / "structured_tables.json"
    if shared_default.exists():
        return shared_default
    return None


def _to_positive_int(value: object) -> Optional[int]:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _clean_heading_path(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for row in value:
        text = " ".join(str(row or "").split()).strip()
        if text:
            out.append(text)
    return out


def build_table_nav_index(
    contract_id: str,
    structured_tables_path: Optional[Path] = None,
    source_json_path: Optional[Path] = None,
) -> dict:
    structured_tables_path = structured_tables_path or resolve_structured_tables_path(contract_id)
    source_json_path = source_json_path or resolve_contract_source_json_path(contract_id)

    out = {
        "schema_version": TABLE_NAV_INDEX_SCHEMA_VERSION,
        "contract_id": contract_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_json": source_json_path.name if source_json_path else None,
        "structured_tables": structured_tables_path.name if structured_tables_path else None,
        "total_pages": None,
        "table_pages": {},
        "table_meta": {},
        "stats": {
            "tables_total": 0,
            "tables_with_page": 0,
            "tables_with_article": 0,
            "tables_with_section": 0,
            "source_pages": 0,
        },
    }

    if not source_json_path or not source_json_path.exists():
        return out
    if not structured_tables_path or not structured_tables_path.exists():
        return out

    try:
        with open(source_json_path, "r", encoding="utf-8") as f:
            source_payload = json.load(f)
    except Exception:
        return out

    pages = source_payload.get("pages") if isinstance(source_payload, dict) else None
    if not isinstance(pages, list):
        return out
    out["total_pages"] = len(pages)
    out["stats"]["source_pages"] = len(pages)

    source_page_numbers: dict[int, int] = {}
    for idx, page in enumerate(pages):
        if not isinstance(page, dict):
            source_page_numbers[idx] = idx + 1
            continue
        page_number = (
            _to_positive_int(page.get("page_number"))
            or _to_positive_int(page.get("page"))
            or idx + 1
        )
        source_page_numbers[idx] = page_number

    try:
        with open(structured_tables_path, "r", encoding="utf-8") as f:
            table_rows = json.load(f)
    except Exception:
        return out
    if not isinstance(table_rows, list):
        return out

    out["stats"]["tables_total"] = len(table_rows)

    for row in table_rows:
        if not isinstance(row, dict):
            continue
        table_id = str(row.get("table_id") or "").strip()
        if not table_id:
            continue

        page_number: Optional[int] = None
        json_path = str(row.get("json_path") or "").strip()
        match = _TABLE_PATH_RE.search(json_path)
        if match:
            try:
                page_idx = int(match.group(1))
            except ValueError:
                page_idx = -1
            if page_idx >= 0:
                page_number = source_page_numbers.get(page_idx)
        if page_number is not None:
            out["table_pages"][table_id] = page_number
            out["stats"]["tables_with_page"] += 1

        article_num = _to_positive_int(row.get("parent_article"))
        section_num = _to_positive_int(row.get("parent_section"))
        if article_num is not None:
            out["stats"]["tables_with_article"] += 1
        if section_num is not None:
            out["stats"]["tables_with_section"] += 1

        heading_path = _clean_heading_path(row.get("heading_path"))
        out["table_meta"][table_id] = {
            "article_num": article_num,
            "section_num": section_num,
            "json_path": json_path or None,
            "heading_path": heading_path,
            "heading_leaf": heading_path[-1] if heading_path else None,
            "row_count": len(row.get("rows") or []),
            "is_perfect_table": bool(row.get("is_perfect")),
        }

    return out


def save_table_nav_index(path: Path, index: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


def load_table_nav_index(path: Path) -> Optional[dict]:
    if not path or not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def to_runtime_table_navigation_maps(index: Optional[dict]) -> dict:
    out = {
        "total_pages": None,
        "table_pages": {},
        "table_meta": {},
    }
    if not isinstance(index, dict):
        return out

    total_pages = _to_positive_int(index.get("total_pages"))
    if total_pages:
        out["total_pages"] = total_pages

    raw_table_pages = index.get("table_pages") or {}
    if isinstance(raw_table_pages, dict):
        for raw_table_id, raw_page in raw_table_pages.items():
            table_id = str(raw_table_id or "").strip()
            page_number = _to_positive_int(raw_page)
            if not table_id or page_number is None:
                continue
            out["table_pages"][table_id] = page_number

    raw_meta = index.get("table_meta") or {}
    if isinstance(raw_meta, dict):
        for raw_table_id, raw_value in raw_meta.items():
            table_id = str(raw_table_id or "").strip()
            if not table_id or not isinstance(raw_value, dict):
                continue
            out["table_meta"][table_id] = {
                "article_num": _to_positive_int(raw_value.get("article_num")),
                "section_num": _to_positive_int(raw_value.get("section_num")),
                "json_path": str(raw_value.get("json_path") or "").strip() or None,
                "heading_leaf": str(raw_value.get("heading_leaf") or "").strip() or None,
                "row_count": _to_positive_int(raw_value.get("row_count")),
            }
    return out
