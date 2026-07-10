"""Generate draft MOA patch ops from parsed source-doc markdown/json."""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from backend.config import DATA_DIR
from backend.ingest.materializer import ContractMaterializer, ensure_base_snapshot, load_base_contract_state
from backend.source_docs import load_source_doc_metadata, resolve_source_doc_dir, resolve_source_doc_pdf_name


_ARTICLE_RE = re.compile(r"^\s{0,3}(?:#+\s*)?(?:<u>)?\s*ARTICLE\s+(\d+)\b", re.IGNORECASE)
_SECTION_RE = re.compile(r"\bSection\s+(\d{1,5})\b", re.IGNORECASE)
_PAGE_RE = re.compile(r"\bPage\s+(\d+)\s+of\s+\d+\b", re.IGNORECASE)
_REDLINE_HINT_RE = re.compile(r"(~~|<u>|</u>|<font|</font>|\(\s*TA\s+\d+/\d+/\d+\s*\))", re.IGNORECASE)
_REVISION_HEADER_RE = re.compile(r"\bRevisions?\s+to\s+Article\b", re.IGNORECASE)
_TA_ARTICLE_RE = re.compile(r"\(\s*TA\s+\d+/\d+/\d+\s*\)\s*\*?\s*\(?Applies[^)]*\)?\s*Revisions?\s+to\s+Article\b", re.IGNORECASE)


def _norm(value: Any) -> str:
    return str(value or "").strip()


def _slug(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", _norm(value).lower()).strip("_")
    return token or "unknown"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sorted_unique(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in values or []:
        value = _norm(raw)
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return sorted(out)


def _metadata_contract_ids(metadata: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    applies_to_contract_ids = metadata.get("applies_to_contract_ids")
    if isinstance(applies_to_contract_ids, list):
        ids.extend(applies_to_contract_ids)
    contract_ids = metadata.get("contract_ids")
    if isinstance(contract_ids, list):
        ids.extend(contract_ids)
    linked_contract_ids = metadata.get("linked_contract_ids")
    if isinstance(linked_contract_ids, list):
        ids.extend(linked_contract_ids)
    contract_id = metadata.get("contract_id")
    if isinstance(contract_id, str):
        ids.append(contract_id)
    return _sorted_unique(ids)


def resolve_target_contract_ids(
    *,
    source_doc_id: str,
    metadata: dict[str, Any],
    explicit_contract_ids: Optional[list[str]],
    exclude_contract_ids: Optional[list[str]],
) -> tuple[list[str], str]:
    explicit = _sorted_unique(list(explicit_contract_ids or []))
    excluded = set(_sorted_unique(list(exclude_contract_ids or [])))
    metadata_ids = _metadata_contract_ids(metadata)
    selection_mode = "explicit" if explicit else "metadata"
    selected = list(explicit) if explicit else list(metadata_ids)
    targets = [cid for cid in selected if cid not in excluded]
    if not targets:
        if explicit:
            raise ValueError("No target contract IDs remain after applying exclusions")
        raise ValueError(
            f"No contract targets found for source_doc_id='{source_doc_id}' "
            "(provide --contract-id or set metadata.contract_ids)"
        )
    return targets, selection_mode


def _render_page_aware_markdown_from_source_json(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    raw_pages = payload.get("pages")
    if not isinstance(raw_pages, list) or not raw_pages:
        return ""

    page_entries: list[tuple[int, dict[str, Any]]] = []
    for page in raw_pages:
        if not isinstance(page, dict):
            continue
        page_number = page.get("page_number")
        if not isinstance(page_number, int) or page_number <= 0:
            continue
        page_entries.append((page_number, page))
    if not page_entries:
        return ""

    total_pages = max(page_number for page_number, _page in page_entries)
    parts: list[str] = []
    for page_number, page in sorted(page_entries, key=lambda row: row[0]):
        parts.append(f"Page {page_number} of {total_pages}")
        for item in page.get("items") or []:
            if not isinstance(item, dict):
                continue
            md = str(item.get("md") or "").strip()
            if not md:
                continue
            parts.append(md)
    return "\n\n".join(parts).strip()


def _load_source_doc_markdown(source_dir: Path) -> str:
    extracted_md_path = source_dir / "extracted.md"
    output_md_path = source_dir / "output.md"

    md_text = ""
    for path in (extracted_md_path, output_md_path):
        if path.exists():
            md_text = path.read_text(encoding="utf-8")
            if md_text.strip():
                break

    if _PAGE_RE.search(md_text):
        return md_text

    for json_name in ("output.json", "extracted.json"):
        json_path = source_dir / json_name
        if not json_path.exists():
            continue
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rendered = _render_page_aware_markdown_from_source_json(payload)
        if rendered:
            return rendered

    return md_text


def _extract_effective_candidates(md_text: str) -> list[dict[str, Any]]:
    lines = str(md_text or "").splitlines()
    out: list[dict[str, Any]] = []
    current_article: Optional[int] = None
    current_page: Optional[int] = None
    i = 0
    while i < len(lines):
        line = lines[i]
        page_match = _PAGE_RE.search(line)
        if page_match:
            try:
                current_page = int(page_match.group(1))
            except Exception:
                current_page = None

        article_match = _ARTICLE_RE.match(line)
        if article_match:
            try:
                current_article = int(article_match.group(1))
            except Exception:
                current_article = None

        section_match = _SECTION_RE.search(line)
        if not section_match or current_article is None:
            i += 1
            continue

        try:
            section_num = int(section_match.group(1))
        except Exception:
            i += 1
            continue

        block_lines = [line]
        j = i + 1
        block_page = current_page
        while j < len(lines):
            next_line = lines[j]
            page_match_next = _PAGE_RE.search(next_line)
            if page_match_next:
                try:
                    block_page = int(page_match_next.group(1))
                except Exception:
                    pass
            if _ARTICLE_RE.match(next_line):
                break
            if _SECTION_RE.search(next_line):
                break
            block_lines.append(next_line)
            j += 1

        raw_block = "\n".join(block_lines).strip()
        has_redline = bool(_REDLINE_HINT_RE.search(raw_block))
        effective_text = _to_effective_text(raw_block)
        out.append(
            {
                "article_num": current_article,
                "section_num": section_num,
                "raw_markdown": raw_block,
                "effective_text_markdown": effective_text,
                "source_page": block_page,
                "has_redline": has_redline,
            }
        )
        i = j

    return out


def _to_effective_text(value: str) -> str:
    text = str(value or "")
    # Remove struck-through content.
    text = re.sub(r"~~.*?~~", "", text, flags=re.DOTALL)
    # Remove HTML/format wrappers while preserving inserted text.
    text = re.sub(r"</?u>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?font[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?sup[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("**", "").replace("__", "")
    text = text.replace("[]", "")
    text = html.unescape(text)
    # Remove isolated numeric footnote markers that often survive HTML stripping.
    text = re.sub(r"(?<=\D)(\d)(?=\n|$)", "", text)
    text = re.sub(r"(?<=[A-Za-z])(\d{1,2})(?=[\s\.,;:])", "", text)
    # Normalize whitespace while preserving paragraph breaks.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _trim_revision_tail(text: str) -> str:
    value = str(text or "")
    for pattern in (_TA_ARTICLE_RE, _REVISION_HEADER_RE):
        match = pattern.search(value)
        if match:
            value = value[: match.start()]
    return value.strip()


def _normalize_section_text_for_target(text: str, section_num: int) -> str:
    value = str(text or "").strip()
    if not value:
        return value
    # Trim known inter-article transition boilerplate.
    value = _trim_revision_tail(value)
    # Prefer starting from the section header if prefixed with TA note.
    anchor = re.search(rf"\bSection\s+{int(section_num)}\b", value, re.IGNORECASE)
    if anchor and anchor.start() > 0:
        prefix = value[: anchor.start()]
        if "(TA " in prefix or "BARGAINING NOTE" in prefix.upper():
            value = value[anchor.start():].lstrip()
    # Fix common OCR artifact: "Section 494" when section is 49 + footnote marker.
    value = re.sub(rf"\bSection\s+{int(section_num)}\d\b", f"Section {int(section_num)}", value, flags=re.IGNORECASE)
    value = re.sub(r"\n{3,}", "\n\n", value).strip()
    return value


def _quality_flags(text: str, section_num: int) -> list[str]:
    flags: list[str] = []
    value = str(text or "")
    if "..." in value:
        flags.append("contains_ellipsis")
    if _REVISION_HEADER_RE.search(value):
        flags.append("contains_revision_header")
    if len(value) < 80:
        flags.append("too_short")
    if not re.search(rf"\bSection\s+{int(section_num)}\b", value, re.IGNORECASE):
        flags.append("missing_section_header")
    return flags


def _select_candidates(candidates: list[dict[str, Any]], include_unmarked: bool) -> list[dict[str, Any]]:
    by_target: dict[tuple[int, int], dict[str, Any]] = {}
    for candidate in candidates:
        if not include_unmarked and not candidate.get("has_redline"):
            continue
        article_num = candidate.get("article_num")
        section_num = candidate.get("section_num")
        if not isinstance(article_num, int) or not isinstance(section_num, int):
            continue
        key = (article_num, section_num)
        prev = by_target.get(key)
        if prev is None or len(str(candidate.get("effective_text_markdown") or "")) > len(str(prev.get("effective_text_markdown") or "")):
            by_target[key] = candidate
    return [by_target[key] for key in sorted(by_target)]


def _base_section_index(base_state: dict[str, Any]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    out: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for section in base_state.get("sections") or []:
        if not isinstance(section, dict):
            continue
        article_num = section.get("article_num")
        section_num = section.get("section_num")
        if isinstance(article_num, int) and isinstance(section_num, int):
            out[(article_num, section_num)].append(section)
    return out


def _build_patch_payload(
    *,
    contract_id: str,
    source_doc_id: str,
    source_pdf: Optional[str],
    effective_date: str,
    ratified_date: Optional[str],
    parent_effective_version_id: str,
    selected_candidates: list[dict[str, Any]],
    base_index: dict[tuple[int, int], list[dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    materializer = ContractMaterializer()
    operations: list[dict[str, Any]] = []
    skipped = []
    skipped_reason_counts = Counter()
    quality_flag_counts = Counter()

    for candidate in selected_candidates:
        article_num = int(candidate["article_num"])
        section_num = int(candidate["section_num"])
        target_sections = base_index.get((article_num, section_num)) or []
        if not target_sections:
            skipped.append({"article_num": article_num, "section_num": section_num, "reason": "target_not_found_in_base"})
            skipped_reason_counts["target_not_found_in_base"] += 1
            continue
        if len(target_sections) > 1:
            skipped.append({"article_num": article_num, "section_num": section_num, "reason": "ambiguous_target_multiple_base_sections"})
            skipped_reason_counts["ambiguous_target_multiple_base_sections"] += 1
            continue

        base_section = target_sections[0]
        new_text = _normalize_section_text_for_target(
            text=str(candidate.get("effective_text_markdown") or ""),
            section_num=section_num,
        )
        if not new_text:
            skipped.append({"article_num": article_num, "section_num": section_num, "reason": "empty_effective_text"})
            skipped_reason_counts["empty_effective_text"] += 1
            continue
        quality_flags = _quality_flags(new_text, section_num=section_num)
        if quality_flags:
            skipped.append(
                {
                    "article_num": article_num,
                    "section_num": section_num,
                    "reason": "low_quality_candidate",
                    "quality_flags": quality_flags,
                }
            )
            skipped_reason_counts["low_quality_candidate"] += 1
            for flag in quality_flags:
                quality_flag_counts[flag] += 1
            continue

        expected_prev_hash = materializer.hash_text(base_section.get("content_markdown", ""))
        source_ref = {
            "source_type": "moa",
            "source_doc_id": source_doc_id,
        }
        if source_pdf:
            source_ref["pdf"] = source_pdf
        source_page = candidate.get("source_page")
        if isinstance(source_page, int) and source_page > 0:
            source_ref["pdf_page"] = source_page

        confidence = 0.55
        if candidate.get("has_redline"):
            confidence += 0.2
        if isinstance(source_page, int) and source_page > 0:
            confidence += 0.1
        confidence = min(confidence, 0.95)

        operations.append(
            {
                "op": "replace_section",
                "target": {
                    "anchor_id": str(base_section.get("anchor_id") or ""),
                    "article_num": article_num,
                    "section_num": section_num,
                },
                "expected_prev_hash": expected_prev_hash,
                "new_text_markdown": new_text,
                "source_refs": [source_ref],
                "confidence": round(confidence, 2),
                "review_status": "pending",
            }
        )

    date_token = _slug(effective_date.replace("-", "_"))
    patch_id = f"draft_{_slug(source_doc_id)}_{_slug(contract_id)}_{date_token}"
    payload = {
        "schema_version": "moa_patch_v0_9_0",
        "patch_id": patch_id,
        "contract_id": contract_id,
        "source_doc_id": source_doc_id,
        "source_pdf": source_pdf,
        "effective_date": effective_date,
        "ratified_date": ratified_date,
        "parent_effective_version_id": parent_effective_version_id,
        "operations": operations,
    }
    report = {
        "contract_id": contract_id,
        "patch_id": patch_id,
        "generated_at_utc": _now_utc(),
        "selected_candidate_count": len(selected_candidates),
        "operation_count": len(operations),
        "skipped_count": len(skipped),
        "skipped_reason_counts": dict(sorted(skipped_reason_counts.items())),
        "quality_flag_counts": dict(sorted(quality_flag_counts.items())),
        "skipped": skipped[:250],
    }
    return payload, report


def generate_patch_draft_for_contract(
    *,
    contract_id: str,
    source_doc_id: str,
    include_unmarked: bool = False,
    output_dir: Optional[Path] = None,
    effective_date_override: Optional[str] = None,
    ratified_date_override: Optional[str] = None,
) -> dict[str, Any]:
    source_dir = resolve_source_doc_dir(source_doc_id)
    if not source_dir:
        raise FileNotFoundError(f"Source doc not found: {source_doc_id}")

    metadata = load_source_doc_metadata(source_doc_id)
    source_pdf = resolve_source_doc_pdf_name(source_doc_id)
    effective_date = _norm(effective_date_override or metadata.get("document_date"))
    ratified_date = _norm(ratified_date_override or metadata.get("ratified_date")) or None
    if not effective_date:
        raise ValueError(f"Missing effective date (provide --effective-date or metadata.document_date) for {source_doc_id}")

    md_text = _load_source_doc_markdown(source_dir)
    if not md_text.strip():
        raise FileNotFoundError(f"Missing extracted markdown/json content under {source_dir}")
    raw_candidates = _extract_effective_candidates(md_text)
    selected_candidates = _select_candidates(raw_candidates, include_unmarked=include_unmarked)

    base_paths = ensure_base_snapshot(contract_id)
    base_state = load_base_contract_state(contract_id=contract_id, base_paths=base_paths)
    base_index = _base_section_index(base_state)
    parent_effective_version_id = str(base_state.get("base_version_id") or "base_snapshot_v0_9_0")

    payload, report = _build_patch_payload(
        contract_id=contract_id,
        source_doc_id=source_doc_id,
        source_pdf=source_pdf,
        effective_date=effective_date,
        ratified_date=ratified_date,
        parent_effective_version_id=parent_effective_version_id,
        selected_candidates=selected_candidates,
        base_index=base_index,
    )

    patch_id = str(payload.get("patch_id") or "")
    if output_dir is None:
        output_dir = DATA_DIR / "contracts" / contract_id / "amendments" / "drafts"
    output_dir.mkdir(parents=True, exist_ok=True)
    patch_path = output_dir / f"{patch_id}.json"
    report_path = output_dir / f"{patch_id}.report.json"

    with open(patch_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    with open(report_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")

    return {
        "contract_id": contract_id,
        "source_doc_id": source_doc_id,
        "patch_path": str(patch_path),
        "report_path": str(report_path),
        "operation_count": len(payload.get("operations") or []),
        "selected_candidate_count": len(selected_candidates),
        "source_pdf": source_pdf,
        "effective_date": effective_date,
        "ratified_date": ratified_date,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate draft MOA patch files from source-doc markdown.")
    parser.add_argument("--source-doc-id", required=True, help="Shared source doc id")
    parser.add_argument("--contract-id", action="append", default=None, help="Contract id to draft against (repeatable)")
    parser.add_argument("--exclude-contract-id", action="append", default=None, help="Contract id to exclude (repeatable)")
    parser.add_argument("--effective-date", default=None, help="Override effective date (YYYY-MM-DD)")
    parser.add_argument("--ratified-date", default=None, help="Override ratified date (YYYY-MM-DD)")
    parser.add_argument("--include-unmarked", action="store_true", help="Include section candidates without obvious redline markers")
    parser.add_argument("--output-dir", default=None, help="Optional output directory override")
    parser.add_argument("--report-out", default=None, help="Optional summary report output path")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any target contract fails")
    args = parser.parse_args()

    source_doc_id = _norm(args.source_doc_id)
    metadata = load_source_doc_metadata(source_doc_id)
    targets, selection_mode = resolve_target_contract_ids(
        source_doc_id=source_doc_id,
        metadata=metadata,
        explicit_contract_ids=args.contract_id,
        exclude_contract_ids=args.exclude_contract_id,
    )
    output_dir = Path(args.output_dir) if args.output_dir else None
    results = []
    failures = []
    for contract_id in targets:
        try:
            result = generate_patch_draft_for_contract(
                contract_id=_norm(contract_id),
                source_doc_id=source_doc_id,
                include_unmarked=bool(args.include_unmarked),
                output_dir=output_dir,
                effective_date_override=args.effective_date,
                ratified_date_override=args.ratified_date,
            )
            results.append(result)
        except Exception as exc:
            failures.append(
                {
                    "contract_id": contract_id,
                    "error": str(exc),
                }
            )

    report = {
        "source_doc_id": source_doc_id,
        "generated_at_utc": _now_utc(),
        "selection_mode": selection_mode,
        "requested_contract_ids": targets,
        "excluded_contract_ids": _sorted_unique(list(args.exclude_contract_id or [])),
        "succeeded": results,
        "failed": failures,
        "ok_count": len(results),
        "error_count": len(failures),
    }

    report_out: Optional[Path] = None
    if args.report_out:
        report_out = Path(args.report_out)
    else:
        source_dir = resolve_source_doc_dir(source_doc_id)
        if source_dir:
            report_out = source_dir / "patch_draft_generation_report.json"
    if report_out:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        with open(report_out, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
        report["report_path"] = str(report_out)

    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    if failures and args.strict:
        return 1
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
