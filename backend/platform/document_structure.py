"""Structure extraction helpers for tenant-uploaded documents.

This ports the most useful legal-document ideas from the legacy ingest stack
into the platform runtime without reviving the old artifact pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from backend.platform.parsing import ParsedDocument


_TOPIC_PATTERNS = {
    "vacation": (r"\bvacation\b", r"\bpaid time off\b", r"\banniversary year\b"),
    "seniority": (r"\bseniority\b", r"\byears? of service\b", r"\blength of service\b"),
    "wages": (r"\bwages?\b", r"\brate of pay\b", r"\bhourly rate\b", r"\$\d"),
    "grievance": (r"\bgrievance\b", r"\barbitrat", r"\bdispute procedure\b"),
    "discipline": (r"\bdisciplin", r"\bdischarge\b", r"\btermination\b", r"\bjust cause\b"),
    "scheduling": (r"\bschedul", r"\bshift\b", r"\bposted\b", r"\bworkweek\b"),
    "benefits": (r"\bhealth\b", r"\bbenefits?\b", r"\bpension\b", r"\binsurance\b"),
    "leave": (r"\bleave\b", r"\bmedical leave\b", r"\bbereavement\b", r"\bsick leave\b"),
}

_LEGAL_SIGNALS = (
    r"\bcollective bargaining agreement\b",
    r"\barticle\s+\d+\b",
    r"\bsection\s+\d+",
    r"\bthis agreement\b",
    r"\bemployer\b",
    r"\bunion\b",
)

_POLICY_SIGNALS = (
    r"\bpolicy\b",
    r"\bhandbook\b",
    r"\bprocedure\b",
    r"\bemployee handbook\b",
)

_NOTICE_SIGNALS = (
    r"\bnotice\b",
    r"\bannouncement\b",
    r"\beffective immediately\b",
)

_MEMO_SIGNALS = (
    r"\bmemo\b",
    r"\bmemorandum\b",
    r"\bto:\b",
    r"\bfrom:\b",
)


def _normalize_lines(text: str) -> list[str]:
    normalized = re.sub(r"\r\n?", "\n", str(text or ""))
    return [line.strip() for line in normalized.splitlines()]


def _normalize_phrase(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _first_sentence(text: str, *, fallback_limit: int = 180) -> str:
    normalized = _normalize_phrase(text)
    if not normalized:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    sentence = parts[0].strip() if parts else normalized
    if len(sentence) <= fallback_limit:
        return sentence
    clipped = sentence[:fallback_limit].rstrip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return f"{clipped}..."


def _infer_topics(*values: str) -> list[str]:
    haystack = " ".join(_normalize_phrase(value).lower() for value in values if value)
    topics: list[str] = []
    for topic, patterns in _TOPIC_PATTERNS.items():
        if any(re.search(pattern, haystack) for pattern in patterns):
            topics.append(topic)
    return topics


def _extract_cross_references(text: str, *, article_num: str | None = None) -> list[str]:
    refs: list[str] = []
    current = str(article_num or "").strip()
    for match in re.finditer(r"\barticle\s+([0-9]+[a-z]?)\b", str(text or ""), re.IGNORECASE):
        ref = f"article_{match.group(1).lower()}"
        if current and ref == f"article_{current.lower()}":
            continue
        if ref not in refs:
            refs.append(ref)
    return refs


@dataclass
class StructuredSection:
    article_num: str | None
    article_title: str | None
    section_num: str | None
    section_title: str | None
    text: str
    page_start: int | None = None
    page_end: int | None = None
    subsection: str | None = None
    summary: str | None = None
    topic_tags: list[str] = field(default_factory=list)
    cross_references: list[str] = field(default_factory=list)
    search_aliases: list[str] = field(default_factory=list)

    def to_metadata(self) -> dict:
        return {
            "article_num": self.article_num,
            "article_title": self.article_title,
            "section_num": self.section_num,
            "section_title": self.section_title,
            "subsection": self.subsection,
            "summary": self.summary,
            "topic_tags": list(self.topic_tags),
            "cross_references": list(self.cross_references),
            "search_aliases": list(self.search_aliases),
            "page_start": self.page_start,
            "page_end": self.page_end,
            "structure_mode": "legal_structured",
        }


@dataclass
class DocumentStructureAnalysis:
    document_type: str
    document_type_confidence: float
    structure_mode: str
    extraction_status: str
    article_titles: dict[str, str] = field(default_factory=dict)
    total_articles: int = 0
    total_sections: int = 0
    topic_hints: list[str] = field(default_factory=list)
    sections: list[StructuredSection] = field(default_factory=list)

    def to_document_metadata(self) -> dict:
        return {
            "document_type": self.document_type,
            "document_type_confidence": round(float(self.document_type_confidence), 3),
            "structure_mode": self.structure_mode,
            "structure_extraction_status": self.extraction_status,
            "article_titles": dict(self.article_titles),
            "total_articles": int(self.total_articles),
            "total_sections": int(self.total_sections),
            "topic_hints": list(self.topic_hints),
        }


def _classify_document(*, title: str, content_type: str, text: str) -> tuple[str, float, str]:
    haystack = f"{title}\n{text}".lower()
    legal_hits = sum(1 for pattern in _LEGAL_SIGNALS if re.search(pattern, haystack))
    if legal_hits >= 2:
        return "legal_contract", min(0.7 + legal_hits * 0.08, 0.98), "legal_structured"
    if any(re.search(pattern, haystack) for pattern in _POLICY_SIGNALS):
        return "policy_handbook", 0.78, "generic"
    if any(re.search(pattern, haystack) for pattern in _NOTICE_SIGNALS):
        return "general_notice", 0.72, "generic"
    if any(re.search(pattern, haystack) for pattern in _MEMO_SIGNALS):
        return "general_memo", 0.7, "generic"
    if str(content_type or "").lower().startswith("text/") and len(text.split()) < 250:
        return "general_notice", 0.58, "generic"
    return "unknown", 0.42, "generic"


def _split_heading_remainder(remainder: str) -> tuple[str | None, str]:
    """Split a heading line's remainder into (short title, body text).

    In this corpus a "Section N." line usually carries the ENTIRE section body
    on the same line ("Section 121. Discharge for Just Cause. No employee
    shall be discharged..."). The old regexes swallowed all of it into the
    title and never returned it as content, so any section whose whole body
    sat on its heading line was silently deleted from the index -- measured at
    22% of the clerks book, including the just-cause clause. The title is the
    first sentence; everything on the line, title included, stays body text.
    """
    text = _normalize_phrase(remainder)
    if not text:
        return None, ""
    title = _first_sentence(text, fallback_limit=90) or None
    return title, text


def _extract_article_heading(line: str) -> tuple[str, str | None, str] | None:
    match = re.match(
        r"^(?:#+\s*)?article\s+([0-9]+[a-z]?)\s*[:\-.]?\s*(.*)$",
        str(line or "").strip(),
        re.IGNORECASE,
    )
    if not match:
        return None
    article_num = match.group(1).strip()
    title, body = _split_heading_remainder(match.group(2) or "")
    return article_num, title, body


def _extract_section_heading(line: str) -> tuple[str, str | None, str] | None:
    match = re.match(
        r"^(?:#+\s*)?(?:section|sec\.?)\s+([0-9]+(?:\.[0-9a-z]+)?)\s*[:\-.]?\s*(.*)$",
        str(line or "").strip(),
        re.IGNORECASE,
    )
    if not match:
        return None
    section_num = match.group(1).strip()
    title, body = _split_heading_remainder(match.group(2) or "")
    return section_num, title, body


def _detect_page_for_offset(pages: list[dict], offset: int) -> int | None:
    if not pages:
        return None
    running = 0
    for page in pages:
        page_text = str(page.get("text") or "")
        page_length = len(page_text) + 2
        if offset <= running + page_length:
            page_number = page.get("page_number")
            if isinstance(page_number, int):
                return page_number
            if str(page_number).isdigit():
                return int(page_number)
            return None
        running += page_length
    page_number = pages[-1].get("page_number")
    if isinstance(page_number, int):
        return page_number
    if str(page_number).isdigit():
        return int(page_number)
    return None


CONTRACT_PACK_SCHEMA_MARKERS = ("sections", "contract_id")


def _looks_like_contract_pack(payload: dict) -> bool:
    """True for a materialized effective-contract export.

    These carry the article/section hierarchy the contract explorer needs.
    Text extraction cannot recover it: the flat markdown export of the same
    contract has zero ARTICLE headings, only 160 "Section N." lines, so
    everything below Article level was being lost on the way into a tenant
    workspace.
    """
    if not isinstance(payload, dict):
        return False
    if not all(marker in payload for marker in CONTRACT_PACK_SCHEMA_MARKERS):
        return False
    sections = payload.get("sections")
    if not isinstance(sections, list) or not sections:
        return False
    first = sections[0]
    return isinstance(first, dict) and ("content_markdown" in first or "raw_chunk" in first)


def analyze_contract_pack(payload: dict, *, filename: str) -> DocumentStructureAnalysis:
    """Build structure directly from a contract pack instead of regex-parsing text."""
    raw_sections = payload.get("sections") or []
    article_titles: dict[str, str] = {}
    sections: list[StructuredSection] = []

    for raw in raw_sections:
        if not isinstance(raw, dict):
            continue
        content = _normalize_phrase(str(raw.get("content_markdown") or raw.get("raw_chunk") or ""))
        if not content:
            continue
        article_num = str(raw.get("article_num") or "").strip() or None
        article_title = _normalize_phrase(str(raw.get("article_title") or "")) or None
        section_num = str(raw.get("section_num") or "").strip() or None
        # The pack has no separate section title; the first sentence of the
        # body is what reads as one in a table of contents.
        section_title = _first_sentence(content, fallback_limit=90) or None
        if article_num and article_title:
            article_titles.setdefault(article_num, article_title)

        provenance = raw.get("provenance") or {}
        page_start = None
        if isinstance(provenance, dict):
            for key in ("page", "page_start", "pdf_page"):
                value = provenance.get(key)
                if isinstance(value, int) and value > 0:
                    page_start = value
                    break

        aliases = [
            alias
            for alias in [
                article_title,
                section_title,
                f"article {article_num}" if article_num else None,
                f"section {section_num}" if section_num else None,
                str(raw.get("citation") or "").strip() or None,
            ]
            if alias
        ]
        sections.append(
            StructuredSection(
                article_num=article_num,
                article_title=article_title,
                section_num=section_num,
                section_title=section_title,
                text=content,
                page_start=page_start,
                page_end=page_start,
                summary=_first_sentence(content) or None,
                topic_tags=_infer_topics(article_title or "", section_title or "", content[:2500]),
                cross_references=_extract_cross_references(content, article_num=article_num),
                search_aliases=list(dict.fromkeys(_normalize_phrase(a) for a in aliases if _normalize_phrase(a))),
            )
        )

    if not sections:
        return DocumentStructureAnalysis(
            document_type="legal_contract",
            document_type_confidence=0.6,
            structure_mode="generic",
            extraction_status="generic_fallback",
            topic_hints=_infer_topics(filename),
        )

    return DocumentStructureAnalysis(
        document_type="legal_contract",
        document_type_confidence=0.99,
        structure_mode="legal_structured",
        extraction_status="contract_pack_ready",
        article_titles=article_titles,
        total_articles=len(article_titles),
        total_sections=len(sections),
        topic_hints=_infer_topics(filename, " ".join(article_titles.values())[:4000]),
        sections=sections,
    )


def analyze_parsed_document(
    parsed: ParsedDocument,
    *,
    filename: str,
    content_type: str,
) -> DocumentStructureAnalysis:
    text = str(parsed.text or "")
    document_type, confidence, structure_mode = _classify_document(
        title=filename,
        content_type=content_type,
        text=text,
    )
    topic_hints = _infer_topics(filename, text[:4000])
    if structure_mode != "legal_structured":
        return DocumentStructureAnalysis(
            document_type=document_type,
            document_type_confidence=confidence,
            structure_mode="generic",
            extraction_status="generic_ready",
            topic_hints=topic_hints,
        )

    pages = [
        {"page_number": page.page_number, "text": page.text}
        for page in parsed.pages
    ]
    lines = _normalize_lines(text)
    sections: list[StructuredSection] = []
    article_titles: dict[str, str] = {}
    current_article_num: str | None = None
    current_article_title: str | None = None
    current_section_num: str | None = None
    current_section_title: str | None = None
    current_buffer: list[str] = []
    current_offset = 0
    section_start_offset = 0

    def flush_section(end_offset: int) -> None:
        nonlocal current_buffer, section_start_offset
        content = _normalize_phrase("\n".join(current_buffer))
        if not content:
            current_buffer = []
            section_start_offset = end_offset
            return
        summary = _first_sentence(content)
        topic_tags = _infer_topics(current_article_title or "", current_section_title or "", content[:2500])
        aliases = [
            alias
            for alias in [
                current_article_title,
                current_section_title,
                f"article {current_article_num}" if current_article_num else None,
                f"section {current_section_num}" if current_section_num else None,
                *topic_tags,
            ]
            if alias
        ]
        sections.append(
            StructuredSection(
                article_num=current_article_num,
                article_title=current_article_title,
                section_num=current_section_num,
                section_title=current_section_title,
                text=content,
                page_start=_detect_page_for_offset(pages, section_start_offset),
                page_end=_detect_page_for_offset(pages, end_offset),
                summary=summary or None,
                topic_tags=topic_tags,
                cross_references=_extract_cross_references(content, article_num=current_article_num),
                search_aliases=list(dict.fromkeys(_normalize_phrase(alias) for alias in aliases if _normalize_phrase(alias))),
            )
        )
        current_buffer = []
        section_start_offset = end_offset

    for line in lines:
        article_hit = _extract_article_heading(line)
        section_hit = _extract_section_heading(line)
        if article_hit:
            flush_section(current_offset)
            current_article_num, current_article_title, heading_body = article_hit
            article_titles[current_article_num] = current_article_title or f"Article {current_article_num}"
            current_section_num = None
            current_section_title = None
            if heading_body:
                # The heading line often carries the whole section body;
                # dropping it deleted real contract language.
                current_buffer.append(heading_body)
        elif section_hit:
            flush_section(current_offset)
            current_section_num, current_section_title, heading_body = section_hit
            if heading_body:
                current_buffer.append(heading_body)
        elif line:
            current_buffer.append(line)
        current_offset += len(line) + 1

    flush_section(current_offset)

    if not sections:
        return DocumentStructureAnalysis(
            document_type=document_type,
            document_type_confidence=max(confidence - 0.12, 0.5),
            structure_mode="generic",
            extraction_status="generic_fallback",
            topic_hints=topic_hints,
        )

    return DocumentStructureAnalysis(
        document_type=document_type,
        document_type_confidence=confidence,
        structure_mode="legal_structured",
        extraction_status="structured_ready",
        article_titles=article_titles,
        total_articles=len(article_titles),
        total_sections=len(sections),
        topic_hints=topic_hints,
        sections=sections,
    )
