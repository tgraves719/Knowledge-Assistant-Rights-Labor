"""Cleanup of machine annotations before contract text reaches a member.

The materializer emits provenance markers into effective_markdown.md:

    PROV(contract_id=..., anchor_id=a10_s25_p2, effective_version_id=...,
         sources=[base:SW+Pueblo+Clerks+2022.2025.pdf#p11])

They are machine-readable anchors, not contract language. The legacy pipeline
understands them; the platform ingestion path did not, so they flowed through
chunk text into the model's context and onto the member's screen mid-sentence.

The anchor is genuinely useful, so it is parked in chunk metadata rather than
discarded -- it points at the exact section a citation came from.
"""

from __future__ import annotations

import re


PROV_PATTERN = re.compile(r"PROV\((?P<body>[^)]*(?:\)[^)]*)*?)\)\s*", re.DOTALL)

# A running header/footer like "9 PUEBLO CLERKS 2022-2025": a page number, an
# all-caps title, and a term year range.
#
# NOT line-anchored, on purpose. Structured section extraction collapses the
# source's line breaks, so by the time text reaches here the furniture is
# stranded mid-sentence ("...twenty-four (24) hours or less 9 PUEBLO CLERKS
# 2022-2025 --- 10 PUEBLO CLERKS 2022-2025 for the involved workweek").
#
# Kept narrow by requiring an all-caps run *and* a term year range: contract
# prose is mixed case, so "The term of this Agreement is 2022-2025" cannot
# match. Over-stripping here would silently delete terms members rely on.
# The separator after the page number is optional: the source runs them
# together ("23PUEBLO CLERKS 2022-2025") often enough that requiring
# whitespace left furniture in member-visible chunks.
PAGE_FURNITURE_PATTERN = re.compile(
    r"\s*\d{1,4}\s*[A-Z][A-Z0-9 \-&'/]{3,60}?\s+\d{4}\s*[-–—]\s*\d{4}\s*(?:[-–—]{2,}\s*)?"
)

_SOURCES_PATTERN = re.compile(r"sources=\[(?P<sources>[^\]]*)\]")
_FIELD_PATTERN = re.compile(r"(?P<key>[a-z_]+)=(?P<value>[^,)\]]*)")


def _parse_prov_body(body: str) -> dict:
    fields: dict = {}
    sources_match = _SOURCES_PATTERN.search(body)
    if sources_match:
        raw = sources_match.group("sources").strip()
        fields["sources"] = [part.strip() for part in raw.split(";") if part.strip()]
        body = body[: sources_match.start()] + body[sources_match.end() :]
    for match in _FIELD_PATTERN.finditer(body):
        key = match.group("key").strip()
        value = match.group("value").strip()
        if key and value and key != "sources":
            fields[key] = value
    return fields


def extract_provenance(text: str) -> tuple[str, dict]:
    """Strip PROV markers and page furniture; return the text and the anchor.

    The first marker is the one that owns the passage -- the materializer emits
    it immediately before the section body -- so its fields describe the chunk.
    Any later markers belong to text that got merged in and are dropped from
    the output without overwriting the owning anchor.
    """
    source = str(text or "")
    if not source.strip():
        return "", {}

    provenance: dict = {}
    first = PROV_PATTERN.search(source)
    if first is not None:
        provenance = _parse_prov_body(first.group("body"))

    cleaned = PROV_PATTERN.sub(" ", source)
    cleaned = PAGE_FURNITURE_PATTERN.sub(" ", cleaned)
    cleaned = re.sub(r"^\s*[-–—]{3,}\s*$", "", cleaned, flags=re.MULTILINE)
    # Collapse the blank runs left behind, without joining separate paragraphs.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = "\n".join(line.rstrip() for line in cleaned.splitlines())

    return cleaned.strip(), provenance


def provenance_metadata(provenance: dict) -> dict:
    """Map parsed provenance onto the chunk metadata keys we keep."""
    if not provenance:
        return {}
    metadata: dict = {}
    for key in ("anchor_id", "effective_version_id"):
        value = str(provenance.get(key) or "").strip()
        if value:
            metadata[key] = value
    sources = provenance.get("sources") or []
    if sources:
        metadata["provenance_sources"] = list(sources)
    return metadata
