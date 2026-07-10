"""Contract outline artifact helpers.

Canonicalizes article/section/page navigation metadata into a single artifact
to reduce drift between manifest TOC, chunks, and PDF navigation indices.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend.chunk_files import resolve_chunk_file
from backend.config import DATA_DIR, MANIFESTS_DIR, ONTOLOGIES_DIR
from backend.pdf_nav_files import resolve_pdf_nav_index_file
from backend.pdf_nav_index import load_pdf_nav_index, to_runtime_navigation_maps


CONTRACT_OUTLINE_SCHEMA_VERSION = "contract_outline_v1"


def package_contract_outline_path(contract_id: str) -> Path:
    """Preferred package-scoped outline artifact path."""
    return DATA_DIR / "contracts" / str(contract_id or "") / "outline" / "contract_outline.json"


def shared_contract_outline_path(contract_id: str) -> Path:
    """Shared runtime outline artifact path."""
    return ONTOLOGIES_DIR / f"contract_outline_{contract_id}.json"


def candidate_contract_outline_files(contract_id: Optional[str] = None) -> list[Path]:
    """Return candidate outline files in priority order."""
    names: list[Path] = []
    if contract_id:
        names.append(package_contract_outline_path(contract_id))
        names.append(shared_contract_outline_path(contract_id))
    names.append(ONTOLOGIES_DIR / "contract_outline.json")
    return names


def resolve_contract_outline_file(
    contract_id: Optional[str] = None,
    allow_shared_fallback: bool = True,
) -> Optional[Path]:
    """Resolve best available contract outline artifact."""
    manifest_count = len(list(MANIFESTS_DIR.glob("*.json")))
    effective_allow_shared = allow_shared_fallback and manifest_count <= 1
    package_path = package_contract_outline_path(str(contract_id or ""))

    for path in candidate_contract_outline_files(contract_id=contract_id):
        is_shared = path == (ONTOLOGIES_DIR / "contract_outline.json")
        if is_shared and not effective_allow_shared:
            continue
        # Package path is contract-scoped and always allowed.
        if path == package_path and contract_id:
            if path.exists():
                return path
            continue
        # Other shared paths are allowed in any mode when they are contract-scoped.
        if path.exists():
            return path
    return None


def save_contract_outline(path: Path, outline: dict) -> None:
    """Write outline artifact to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(outline, f, indent=2, ensure_ascii=False)


def load_contract_outline(path: Path) -> Optional[dict]:
    """Load outline artifact from disk."""
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


def _to_positive_int(value: object) -> Optional[int]:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _clean_title(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def article_titles_from_outline(outline: Optional[dict]) -> dict[str, str]:
    """Extract sorted article title map from an outline payload."""
    if not isinstance(outline, dict):
        return {}

    raw_map = outline.get("article_titles")
    out: dict[str, str] = {}
    if isinstance(raw_map, dict):
        for raw_key, raw_title in raw_map.items():
            num = _to_positive_int(raw_key)
            title = _clean_title(raw_title)
            if num is None:
                continue
            out[str(num)] = title or f"Article {num}"
        if out:
            return dict(sorted(out.items(), key=lambda kv: int(kv[0])))

    raw_articles = outline.get("articles")
    if not isinstance(raw_articles, list):
        return {}
    for row in raw_articles:
        if not isinstance(row, dict):
            continue
        num = _to_positive_int(row.get("article_num"))
        if num is None:
            continue
        title = _clean_title(row.get("article_title"))
        out[str(num)] = title or f"Article {num}"
    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


def build_contract_outline(
    contract_id: str,
    manifest: Optional[dict] = None,
    chunks: Optional[list[dict]] = None,
    pdf_nav_index: Optional[dict] = None,
    manifest_path: Optional[Path] = None,
    chunks_path: Optional[Path] = None,
    pdf_nav_path: Optional[Path] = None,
) -> dict:
    """Build a canonical contract outline artifact."""
    resolved_manifest_path = manifest_path or (MANIFESTS_DIR / f"{contract_id}.json")
    resolved_chunks_path = chunks_path or resolve_chunk_file(
        contract_id=contract_id,
        allow_shared_fallback=True,
    )
    resolved_pdf_nav_path = pdf_nav_path or resolve_pdf_nav_index_file(
        contract_id=contract_id,
        allow_shared_fallback=True,
    )

    if manifest is None and resolved_manifest_path and resolved_manifest_path.exists():
        try:
            with open(resolved_manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}
    manifest = manifest or {}

    if chunks is None and resolved_chunks_path and resolved_chunks_path.exists():
        try:
            with open(resolved_chunks_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            chunks = raw if isinstance(raw, list) else []
        except Exception:
            chunks = []
    chunks = chunks or []

    if pdf_nav_index is None and resolved_pdf_nav_path and resolved_pdf_nav_path.exists():
        pdf_nav_index = load_pdf_nav_index(resolved_pdf_nav_path)
    runtime_nav = to_runtime_navigation_maps(pdf_nav_index)
    article_pages = runtime_nav.get("article_pages") or {}
    section_pages = runtime_nav.get("section_pages") or {}

    article_map: dict[int, dict] = {}

    raw_titles = manifest.get("article_titles") or {}
    if isinstance(raw_titles, dict):
        for raw_key, raw_title in raw_titles.items():
            article_num = _to_positive_int(raw_key)
            if article_num is None:
                continue
            row = article_map.setdefault(
                article_num,
                {
                    "article_num": article_num,
                    "manifest_title": "",
                    "chunk_title": "",
                    "section_nums": set(),
                },
            )
            row["manifest_title"] = _clean_title(raw_title)

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        chunk_contract_id = str(chunk.get("contract_id") or "").strip()
        if chunk_contract_id and chunk_contract_id != contract_id:
            continue

        article_num = _to_positive_int(chunk.get("article_num"))
        if article_num is None:
            continue

        row = article_map.setdefault(
            article_num,
            {
                "article_num": article_num,
                "manifest_title": "",
                "chunk_title": "",
                "section_nums": set(),
            },
        )
        chunk_title = _clean_title(chunk.get("article_title"))
        if chunk_title and not row["chunk_title"]:
            row["chunk_title"] = chunk_title

        section_num = _to_positive_int(chunk.get("section_num"))
        if section_num is not None:
            row["section_nums"].add(section_num)

    articles: list[dict] = []
    article_titles: dict[str, str] = {}
    section_count = 0
    sections_with_pages = 0
    articles_with_pages = 0

    for article_num in sorted(article_map):
        row = article_map[article_num]
        title = (
            row.get("manifest_title")
            or row.get("chunk_title")
            or f"Article {article_num}"
        )
        section_nums = sorted(int(v) for v in row.get("section_nums") or set())

        sections: list[dict] = []
        for section_num in section_nums:
            page = _to_positive_int(section_pages.get(f"{article_num}:{section_num}"))
            if page is not None:
                sections_with_pages += 1
            sections.append(
                {
                    "section_num": section_num,
                    "page_number": page,
                }
            )

        page_number = _to_positive_int(article_pages.get(article_num))
        if page_number is not None:
            articles_with_pages += 1

        section_count += len(sections)
        article_titles[str(article_num)] = title
        articles.append(
            {
                "article_num": article_num,
                "article_title": title,
                "page_number": page_number,
                "sections": sections,
            }
        )

    return {
        "schema_version": CONTRACT_OUTLINE_SCHEMA_VERSION,
        "contract_id": contract_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "manifest": str(resolved_manifest_path) if resolved_manifest_path else None,
            "chunks": str(resolved_chunks_path) if resolved_chunks_path else None,
            "pdf_nav_index": str(resolved_pdf_nav_path) if resolved_pdf_nav_path else None,
        },
        "article_titles": article_titles,
        "articles": articles,
        "stats": {
            "article_count": len(articles),
            "section_count": section_count,
            "articles_with_pages": articles_with_pages,
            "sections_with_pages": sections_with_pages,
            "total_pages": runtime_nav.get("total_pages"),
        },
    }
