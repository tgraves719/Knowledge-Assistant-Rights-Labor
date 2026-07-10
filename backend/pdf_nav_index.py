"""Deterministic PDF navigation index helpers."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend.config import DATA_DIR


PDF_NAV_INDEX_SCHEMA_VERSION = "pdf_nav_index_v1"

_ARTICLE_RE = re.compile(r"^\s*article\s+(\d+)\b", re.IGNORECASE)
_SECTION_RE = re.compile(r"^\s*section\s+(\d+)\b", re.IGNORECASE)


def source_dir_for_contract(contract_id: str) -> Path:
    return DATA_DIR / "contracts" / str(contract_id or "") / "source"


def resolve_contract_source_json_path(contract_id: str) -> Optional[Path]:
    source_dir = source_dir_for_contract(contract_id)
    candidates: list[Path] = []
    if source_dir.exists():
        candidates.extend(sorted(source_dir.glob("*.json")))
    if not candidates:
        # Fallback: accept top-level JSONs in the contract pack if source/ is missing.
        pack_dir = DATA_DIR / "contracts" / str(contract_id or "")
        if pack_dir.exists():
            candidates.extend(sorted(pack_dir.glob("*.json")))
    if not candidates:
        return None
    preferred = f"{contract_id}.json".lower()
    for path in candidates:
        if path.name.lower() == preferred:
            return path
    return candidates[0]


def resolve_contract_pdf_path(contract_id: str) -> Optional[Path]:
    source_dir = source_dir_for_contract(contract_id)
    candidates: list[Path] = []
    if source_dir.exists():
        candidates.extend(sorted(source_dir.glob("*.pdf")))
    if not candidates:
        # Fallback: accept top-level PDFs in the contract pack if source/ is missing.
        pack_dir = DATA_DIR / "contracts" / str(contract_id or "")
        if pack_dir.exists():
            candidates.extend(sorted(pack_dir.glob("*.pdf")))
    if not candidates:
        return None

    # Prefer base CBA PDF for article/section navigation index generation.
    non_moa = [p for p in candidates if "moa" not in p.name.lower()]
    if non_moa:
        return non_moa[0]
    return candidates[0]


def _to_positive_int(value: object) -> Optional[int]:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _clean_source_item_text(item: dict) -> str:
    raw = str(item.get("value") or item.get("md") or "")
    if not raw:
        return ""
    text = re.sub(r"<[^>]+>", " ", raw)
    text = text.replace("&nbsp;", " ")
    text = re.sub(r"[`*_#>\[\]\(\)]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_pdf_nav_index(
    contract_id: str,
    source_json_path: Optional[Path] = None,
    pdf_path: Optional[Path] = None,
) -> dict:
    source_json_path = source_json_path or resolve_contract_source_json_path(contract_id)
    pdf_path = pdf_path or resolve_contract_pdf_path(contract_id)

    out = {
        "schema_version": PDF_NAV_INDEX_SCHEMA_VERSION,
        "contract_id": contract_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_json": source_json_path.name if source_json_path else None,
        "pdf_filename": pdf_path.name if pdf_path else None,
        "total_pages": None,
        "article_pages": {},
        "section_pages": {},
        "stats": {
            "articles_mapped": 0,
            "sections_mapped": 0,
            "source_pages": 0,
        },
    }

    if not source_json_path or not source_json_path.exists():
        return out

    try:
        with open(source_json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return out

    pages = payload.get("pages") if isinstance(payload, dict) else None
    if not isinstance(pages, list):
        return out

    out["total_pages"] = len(pages)
    out["stats"]["source_pages"] = len(pages)

    article_pages: dict[str, int] = {}
    section_pages: dict[str, dict[str, int]] = {}
    current_article: Optional[int] = None

    for idx, page in enumerate(pages):
        if not isinstance(page, dict):
            continue

        raw_page_number = page.get("page_number")
        page_number = _to_positive_int(raw_page_number) or (idx + 1)

        items = page.get("items") or []
        if not isinstance(items, list):
            continue

        cleaned_items: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            text = _clean_source_item_text(item)
            if text:
                cleaned_items.append(text)

        page_text = " ".join(t.lower() for t in cleaned_items)
        is_toc_page = "table of contents" in page_text

        for text in cleaned_items:
            text_lower = text.lower()
            if "table of contents" in text_lower:
                is_toc_page = True
            if is_toc_page:
                continue

            article_match = _ARTICLE_RE.search(text)
            if article_match:
                article_num = int(article_match.group(1))
                current_article = article_num
                article_key = str(article_num)
                article_pages.setdefault(article_key, page_number)
                continue

            section_match = _SECTION_RE.search(text)
            if section_match and current_article is not None:
                section_num = int(section_match.group(1))
                article_key = str(current_article)
                section_key = str(section_num)
                section_map = section_pages.setdefault(article_key, {})
                section_map.setdefault(section_key, page_number)

    out["article_pages"] = dict(
        sorted(article_pages.items(), key=lambda kv: int(kv[0]))
    )
    out["section_pages"] = {
        art: dict(sorted(sec_map.items(), key=lambda kv: int(kv[0])))
        for art, sec_map in sorted(section_pages.items(), key=lambda kv: int(kv[0]))
    }
    out["stats"]["articles_mapped"] = len(out["article_pages"])
    out["stats"]["sections_mapped"] = sum(len(v) for v in out["section_pages"].values())
    return out


def save_pdf_nav_index(path: Path, index: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


def load_pdf_nav_index(path: Path) -> Optional[dict]:
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


def to_runtime_navigation_maps(index: Optional[dict]) -> dict:
    out = {
        "total_pages": None,
        "article_pages": {},
        "section_pages": {},
    }
    if not isinstance(index, dict):
        return out

    total_pages = _to_positive_int(index.get("total_pages"))
    if total_pages:
        out["total_pages"] = total_pages

    article_map = index.get("article_pages") or {}
    if isinstance(article_map, dict):
        for raw_article, raw_page in article_map.items():
            article_num = _to_positive_int(raw_article)
            page_num = _to_positive_int(raw_page)
            if article_num and page_num:
                out["article_pages"][article_num] = page_num

    section_map = index.get("section_pages") or {}
    if isinstance(section_map, dict):
        for raw_article, value in section_map.items():
            if isinstance(value, dict):
                article_num = _to_positive_int(raw_article)
                if not article_num:
                    continue
                for raw_section, raw_page in value.items():
                    section_num = _to_positive_int(raw_section)
                    page_num = _to_positive_int(raw_page)
                    if section_num and page_num:
                        out["section_pages"][f"{article_num}:{section_num}"] = page_num
                continue

            if isinstance(raw_article, str) and ":" in raw_article:
                parts = raw_article.split(":", 1)
                article_num = _to_positive_int(parts[0])
                section_num = _to_positive_int(parts[1])
                page_num = _to_positive_int(value)
                if article_num and section_num and page_num:
                    out["section_pages"][f"{article_num}:{section_num}"] = page_num

    return out
