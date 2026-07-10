"""
Table Extractor - JSON-first structured table extraction.

Phase A: Build table registry from LlamaIndex JSON export (canonical source).
Phase B: Match JSON tables to chunks, replace HTML with clean markdown.

The JSON export is guaranteed for every contract. HTML-in-markdown parsing
exists only as a fallback for unmatched tables.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import CONTRACT_JSON_FILE, TABLES_DIR, CHUNKS_DIR


@dataclass
class StructuredTable:
    """First-class table representation extracted from JSON."""
    table_id: str               # "tbl_art{article}_{seq}" e.g. "tbl_art40_1"
    content_hash: str           # SHA-256 of normalized rows
    headers: list[str]          # First row (or detected headers)
    rows: list[list[str]]       # Data rows (excluding header)
    markdown: str               # Pipe-table markdown from JSON md field
    csv: str                    # CSV from JSON csv field
    is_perfect: bool            # From JSON isPerfectTable
    json_path: str              # "pages[{page_idx}].items[{item_idx}]"
    heading_path: list[str]     # Full heading stack from outermost to nearest
    parent_article: Optional[int] = None
    parent_section: Optional[int] = None
    parent_chunk_id: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'StructuredTable':
        return cls(
            table_id=d['table_id'],
            content_hash=d['content_hash'],
            headers=d.get('headers', []),
            rows=d.get('rows', []),
            markdown=d.get('markdown', ''),
            csv=d.get('csv', ''),
            is_perfect=d.get('is_perfect', False),
            json_path=d.get('json_path', ''),
            heading_path=d.get('heading_path', []),
            parent_article=d.get('parent_article'),
            parent_section=d.get('parent_section'),
            parent_chunk_id=d.get('parent_chunk_id'),
        )


def _normalize_cell(cell: str) -> str:
    """Normalize a cell value for hashing."""
    return re.sub(r'\s+', ' ', str(cell).strip().lower())


def _compute_content_hash(all_rows: list[list[str]]) -> str:
    """Compute SHA-256 hash of normalized row content."""
    normalized = [[_normalize_cell(c) for c in row] for row in all_rows]
    return hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()[:16]


def _extract_article_num(text: str) -> Optional[int]:
    """Extract article number from heading text."""
    match = re.search(r'ARTICLE\s+(\d+)', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _extract_section_num(text: str) -> Optional[int]:
    """Extract section number from heading text."""
    match = re.search(r'Section\s+(\d+)', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


# ============================================================================
# Phase A: Build table registry from JSON
# ============================================================================

def build_table_registry(json_path: Path = None) -> list[StructuredTable]:
    """
    Walk the JSON page-by-page, tracking heading context.
    For each table item, extract a StructuredTable.

    Args:
        json_path: Path to LlamaIndex JSON export. Defaults to CONTRACT_JSON_FILE.

    Returns:
        List of StructuredTable objects.
    """
    if json_path is None:
        json_path = CONTRACT_JSON_FILE

    with open(json_path, 'r', encoding='utf-8') as f:
        doc = json.load(f)

    pages = doc.get('pages', [])
    tables = []

    # Heading stack: list of (level, text) tuples
    heading_stack = []
    current_article = None
    current_section = None
    article_table_counts = {}  # Track sequence per article

    for page_idx, page in enumerate(pages):
        items = page.get('items', [])

        for item_idx, item in enumerate(items):
            item_type = item.get('type', '')

            if item_type == 'heading':
                heading_text = item.get('value', item.get('md', '')).strip()
                lvl = item.get('lvl', 1)

                # Update heading stack: same-or-higher level replaces from that position down
                heading_stack = [
                    (l, t) for l, t in heading_stack if l < lvl
                ]
                heading_stack.append((lvl, heading_text))

                # Update article/section tracking
                art_num = _extract_article_num(heading_text)
                if art_num is not None:
                    current_article = art_num
                    current_section = None  # Reset section on new article

                sec_num = _extract_section_num(heading_text)
                if sec_num is not None:
                    current_section = sec_num

            elif item_type in ('table', 'layout_v2_table'):
                raw_rows = item.get('rows', [])
                if not raw_rows:
                    continue

                # Headers = first row, data = remaining rows
                headers = [str(c) for c in raw_rows[0]] if raw_rows else []
                data_rows = [[str(c) for c in row] for row in raw_rows[1:]] if len(raw_rows) > 1 else []

                # Compute content hash from all rows
                content_hash = _compute_content_hash(raw_rows)

                # Get markdown and CSV from JSON
                md = item.get('md', '')
                csv_str = item.get('csv', '')
                is_perfect = item.get('isPerfectTable', False)

                # Build heading path (just the text values)
                heading_path = [t for _, t in heading_stack]

                # Table ID: tbl_art{N}_{seq}
                article_key = current_article or 0
                if article_key not in article_table_counts:
                    article_table_counts[article_key] = 0
                article_table_counts[article_key] += 1
                seq = article_table_counts[article_key]

                table_id = f"tbl_art{article_key}_{seq}"

                table = StructuredTable(
                    table_id=table_id,
                    content_hash=content_hash,
                    headers=headers,
                    rows=data_rows,
                    markdown=md,
                    csv=csv_str,
                    is_perfect=is_perfect,
                    json_path=f"pages[{page_idx}].items[{item_idx}]",
                    heading_path=heading_path,
                    parent_article=current_article,
                    parent_section=current_section,
                )
                tables.append(table)

            elif item_type == 'text':
                # Check text items for section references too
                text = item.get('value', item.get('md', ''))
                sec_num = _extract_section_num(text)
                if sec_num is not None:
                    current_section = sec_num

    return tables


def save_table_registry(tables: list[StructuredTable], output_dir: Path = None):
    """Save table registry to JSON."""
    if output_dir is None:
        output_dir = TABLES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "structured_tables.json"
    data = [t.to_dict() for t in tables]

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(data)} tables to {output_file}")
    return output_file


# ============================================================================
# Phase B: Match tables to chunks and replace HTML
# ============================================================================

class _SimpleHTMLTableParser(HTMLParser):
    """Minimal HTML parser to extract cell text from <table> blocks."""

    def __init__(self):
        super().__init__()
        self.tables = []       # list of list[list[str]] (table -> rows -> cells)
        self._current_table = None
        self._current_row = None
        self._current_cell = None
        self._in_cell = False

    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            self._current_table = []
        elif tag == 'tr' and self._current_table is not None:
            self._current_row = []
        elif tag in ('td', 'th') and self._current_row is not None:
            self._current_cell = ''
            self._in_cell = True

    def handle_endtag(self, tag):
        if tag in ('td', 'th') and self._in_cell:
            self._current_row.append(self._current_cell.strip())
            self._current_cell = None
            self._in_cell = False
        elif tag == 'tr' and self._current_row is not None:
            if self._current_row:
                self._current_table.append(self._current_row)
            self._current_row = None
        elif tag == 'table' and self._current_table is not None:
            if self._current_table:
                self.tables.append(self._current_table)
            self._current_table = None

    def handle_data(self, data):
        if self._in_cell and self._current_cell is not None:
            self._current_cell += data


def _extract_html_tables(content: str) -> list[tuple[str, list[list[str]]]]:
    """
    Extract HTML table blocks from content.

    Returns list of (html_block, parsed_rows) tuples.
    """
    results = []
    # Find all <table>...</table> blocks
    table_pattern = re.compile(r'<table.*?>.*?</table>', re.DOTALL | re.IGNORECASE)

    for match in table_pattern.finditer(content):
        html_block = match.group(0)
        parser = _SimpleHTMLTableParser()
        try:
            parser.feed(html_block)
            if parser.tables:
                results.append((html_block, parser.tables[0]))
        except Exception:
            # HTML too malformed, add with empty rows
            results.append((html_block, []))

    return results


def _match_table_by_hash(
    parsed_rows: list[list[str]],
    registry_by_hash: dict[str, StructuredTable]
) -> Optional[StructuredTable]:
    """Match an HTML table to the registry by content hash."""
    if not parsed_rows:
        return None
    content_hash = _compute_content_hash(parsed_rows)
    return registry_by_hash.get(content_hash)


def _match_table_by_article(
    article_num: Optional[int],
    heading_text: str,
    registry_by_article: dict[int, list[StructuredTable]],
    used_ids: set
) -> Optional[StructuredTable]:
    """Fallback: match by article number and heading overlap."""
    if article_num is None:
        return None

    candidates = registry_by_article.get(article_num, [])
    heading_lower = heading_text.lower()

    for table in candidates:
        if table.table_id in used_ids:
            continue
        # Check heading path overlap
        for hp in table.heading_path:
            if hp.lower() in heading_lower or heading_lower in hp.lower():
                return table

    # If only one unused candidate for this article, use it
    unused = [t for t in candidates if t.table_id not in used_ids]
    if len(unused) == 1:
        return unused[0]

    return None


def _rows_to_markdown(rows: list[list[str]]) -> str:
    """Convert a 2D row array to pipe-table markdown."""
    if not rows:
        return ""

    lines = []
    # Header row
    header = rows[0]
    lines.append("| " + " | ".join(str(c) for c in header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")

    # Data rows
    for row in rows[1:]:
        # Pad or truncate to match header width
        padded = list(row) + [""] * max(0, len(header) - len(row))
        lines.append("| " + " | ".join(str(c) for c in padded[:len(header)]) + " |")

    return "\n".join(lines)


def _rows_to_plaintext(headers: list[str], rows: list[list[str]]) -> str:
    """
    Convert table rows to compact plain text for embedding/search.

    Embeddings generally perform better on sentence-like text than raw pipe tables.
    """
    lines = []
    clean_headers = [str(h).strip() for h in (headers or [])]
    if clean_headers:
        lines.append("Columns: " + " | ".join(h for h in clean_headers if h))

    for row in rows or []:
        parts = []
        for idx, cell in enumerate(row):
            value = str(cell).strip()
            if not value:
                continue
            label = clean_headers[idx] if idx < len(clean_headers) and clean_headers[idx] else f"Column {idx + 1}"
            parts.append(f"{label}: {value}")
        if parts:
            lines.append("; ".join(parts))

    return "\n".join(lines)


def _rows_to_csv(rows: list[list[str]]) -> str:
    """Convert table rows into a simple CSV string."""
    out: list[str] = []
    for row in rows:
        cells: list[str] = []
        for cell in row:
            text = str(cell or "").replace('"', '""')
            if any(ch in text for ch in [",", '"', "\n"]):
                text = f'"{text}"'
            cells.append(text)
        out.append(",".join(cells))
    return "\n".join(out)


def _normalize_heading_text(text: str) -> str:
    normalized = re.sub(r'</?u>|</?strong>|</?em>|</?b>|</?i>', ' ', str(text or ''), flags=re.IGNORECASE)
    normalized = re.sub(r'[*_`]+', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized.strip(" .-\t")


def _build_fallback_table_from_rows(
    parsed_rows: list[list[str]],
    chunk: dict,
    chunk_idx: int,
    table_idx: int,
    existing_ids: set[str],
) -> Optional[StructuredTable]:
    """
    Build a deterministic fallback StructuredTable when JSON registry misses an HTML table.

    This keeps table_refs and table-backed retrieval available when JSON exports
    under-detect tables for a contract.
    """
    if not parsed_rows:
        return None

    rows = [[str(cell or "").strip() for cell in row] for row in parsed_rows]
    headers = rows[0] if rows else []
    data_rows = rows[1:] if len(rows) > 1 else []
    content_hash = _compute_content_hash(rows)

    article_num = chunk.get("article_num")
    section_num = chunk.get("section_num")
    article_key = article_num if isinstance(article_num, int) else 0

    base_id = f"tbl_fallback_art{article_key}_{content_hash[:8]}"
    table_id = base_id
    suffix = 2
    while table_id in existing_ids:
        table_id = f"{base_id}_{suffix}"
        suffix += 1

    heading_path: list[str] = []
    for raw in (
        chunk.get("parent_context"),
        chunk.get("article_title"),
        chunk.get("citation"),
    ):
        text = _normalize_heading_text(raw)
        if text and text not in heading_path:
            heading_path.append(text)

    return StructuredTable(
        table_id=table_id,
        content_hash=content_hash,
        headers=headers,
        rows=data_rows,
        markdown=_rows_to_markdown(rows),
        csv=_rows_to_csv(rows),
        is_perfect=False,
        json_path=f"chunks[{chunk_idx}].fallback_html_tables[{table_idx}]",
        heading_path=heading_path,
        parent_article=article_num if isinstance(article_num, int) else None,
        parent_section=section_num if isinstance(section_num, int) else None,
    )


def _is_toc_table(table: StructuredTable) -> bool:
    heading = " ".join(table.heading_path or []).lower()
    md = (table.markdown or "").lower()
    csv = (table.csv or "").lower()
    return "table of contents" in heading or "table of contents" in md or "table of contents" in csv


def _infer_table_doc_type(table: StructuredTable) -> str:
    heading = " ".join(table.heading_path or []).lower()
    if "appendix" in heading:
        return "appendix"
    # Many wage tables are exported under trailing "additional provisions"
    # headings without explicit "appendix" wording.
    raw = f"{heading}\n{(table.markdown or '').lower()}\n{(table.csv or '').lower()}"
    wage_like = (
        ("classification" in raw and "effective" in raw and raw.count("$") >= 3)
        or "rates of pay" in raw
    )
    if wage_like and isinstance(table.parent_article, int) and table.parent_article >= 50:
        return "appendix"
    return "cba"


def _infer_table_topics(text: str) -> list[str]:
    t = (text or "").lower()
    topics = []
    if any(tok in t for tok in ("wage", "rate", "hourly", "progression", "journeyman", "classification")):
        topics.append("wages")
    if any(tok in t for tok in ("vacation", "holiday", "paid time")):
        topics.append("vacation")
    if any(tok in t for tok in ("benefit", "medical", "dental", "vision", "insurance")):
        topics.append("health_benefits")
    return topics


def _appendix_citation_label(table: StructuredTable) -> str:
    heading = " ".join(table.heading_path or "")
    match = re.search(r'appendix\s*"?([a-z0-9]+)"?', heading, re.IGNORECASE)
    if match:
        return f"Appendix {match.group(1).upper()}"
    raw = f"{heading}\n{table.markdown or ''}\n{table.csv or ''}".lower()
    if re.search(r'appendix\s*"?a"?', raw, re.IGNORECASE):
        return "Appendix A"
    wage_like = (
        ("classification" in raw and "effective" in raw and raw.count("$") >= 2)
        or "rate of pay" in raw
        or "hourly rate" in raw
        or "wage progression" in raw
    )
    if wage_like:
        return "Appendix A Wage Table"
    return "Appendix"


def apply_tables_to_chunks(
    chunks: list[dict],
    table_registry: list[StructuredTable]
) -> tuple[list[dict], list[StructuredTable]]:
    """
    For each chunk containing HTML <table> tags:
    1. Match to JSON registry by content_hash (primary) or article+heading (fallback)
    2. Replace HTML with JSON-sourced clean markdown in content_with_tables
    3. Replace HTML with placeholder in content (for clean embedding)
    4. Set table_refs on chunk

    Args:
        chunks: List of chunk dicts (from enriched JSON)
        table_registry: List of StructuredTable objects

    Returns:
        Tuple of (modified chunks, updated registry with parent_chunk_ids set)
    """
    # Convert dicts to StructuredTable objects if needed
    table_registry = [
        StructuredTable.from_dict(t) if isinstance(t, dict) else t
        for t in table_registry
    ]

    # Build lookup indices
    registry_by_hash = {t.content_hash: t for t in table_registry}
    registry_by_article = {}
    for t in table_registry:
        art = t.parent_article or 0
        if art not in registry_by_article:
            registry_by_article[art] = []
        registry_by_article[art].append(t)

    used_ids = set()
    existing_ids = {str(t.table_id) for t in table_registry}
    matched_count = 0
    fallback_count = 0
    synthesized_count = 0

    for chunk_idx, chunk in enumerate(chunks):
        content = chunk.get('content', '')

        # Skip chunks without HTML tables
        if '<table' not in content.lower():
            chunk['content_with_tables'] = content
            chunk['table_refs'] = []
            continue

        # Extract HTML table blocks
        html_tables = _extract_html_tables(content)

        table_refs = []
        content_clean = content
        content_rich = content

        for table_idx, (html_block, parsed_rows) in enumerate(html_tables):
            # Try primary match: content hash
            matched = _match_table_by_hash(parsed_rows, registry_by_hash)

            # Try fallback match: article + heading
            if matched is None:
                article_num = chunk.get('article_num')
                heading = chunk.get('parent_context', '') + ' ' + chunk.get('article_title', '')
                matched = _match_table_by_article(
                    article_num, heading, registry_by_article, used_ids
                )
                if matched:
                    fallback_count += 1

            if matched:
                matched.parent_chunk_id = chunk.get('chunk_id', '')
                used_ids.add(matched.table_id)
                table_refs.append(matched.table_id)
                matched_count += 1

                # Use JSON-sourced markdown (canonical)
                clean_md = matched.markdown
                if not clean_md and (matched.headers or matched.rows):
                    all_rows = [matched.headers] + matched.rows if matched.headers else matched.rows
                    clean_md = _rows_to_markdown(all_rows)

                # Build placeholder for clean embedding
                table_label = matched.heading_path[-1] if matched.heading_path else matched.table_id
                placeholder = f"[Table: {table_label}]"

                content_clean = content_clean.replace(html_block, placeholder)
                content_rich = content_rich.replace(html_block, clean_md)
            else:
                # HTML fallback: convert HTML to markdown best-effort
                if parsed_rows:
                    synthesized = _build_fallback_table_from_rows(
                        parsed_rows=parsed_rows,
                        chunk=chunk,
                        chunk_idx=chunk_idx,
                        table_idx=table_idx,
                        existing_ids=existing_ids,
                    )
                    if synthesized:
                        synthesized.parent_chunk_id = chunk.get('chunk_id', '')
                        table_registry.append(synthesized)
                        existing_ids.add(synthesized.table_id)
                        used_ids.add(synthesized.table_id)
                        table_refs.append(synthesized.table_id)
                        synthesized_count += 1

                        registry_by_hash.setdefault(synthesized.content_hash, synthesized)
                        article_key = synthesized.parent_article or 0
                        registry_by_article.setdefault(article_key, []).append(synthesized)

                        table_label = (
                            synthesized.heading_path[-1]
                            if synthesized.heading_path
                            else synthesized.table_id
                        )
                        placeholder = f"[Table: {table_label}]"
                        content_clean = content_clean.replace(html_block, placeholder)
                        content_rich = content_rich.replace(
                            html_block,
                            synthesized.markdown or _rows_to_markdown(parsed_rows),
                        )
                    else:
                        fallback_md = _rows_to_markdown(parsed_rows)
                        content_clean = content_clean.replace(html_block, "[Table]")
                        content_rich = content_rich.replace(html_block, fallback_md)
                else:
                    # Cannot parse at all, just remove
                    content_clean = content_clean.replace(html_block, "[Table: unparseable]")
                    content_rich = content_rich.replace(html_block, "[Table: unparseable]")

        chunk['content'] = content_clean
        chunk['content_with_tables'] = content_rich
        chunk['table_refs'] = table_refs

    # Set content_with_tables for chunks without tables
    for chunk in chunks:
        if 'content_with_tables' not in chunk:
            chunk['content_with_tables'] = chunk.get('content', '')
        if 'table_refs' not in chunk:
            chunk['table_refs'] = []

    print(
        f"Matched {matched_count} HTML tables to JSON registry "
        f"({fallback_count} via fallback, {synthesized_count} synthesized from HTML)"
    )
    unmatched = len(table_registry) - len(used_ids)
    if unmatched:
        print(f"  {unmatched} registry tables were not found in any chunk")

    return chunks, table_registry


def synthesize_unmatched_table_chunks(
    chunks: list[dict],
    table_registry: list[StructuredTable],
) -> tuple[list[dict], dict]:
    """
    Ensure meaningful structured tables are retrievable as first-class chunks.

    Behavior:
    - Always materialize non-TOC appendix/wage/vacation tables as dedicated chunks.
    - Materialize any other non-TOC table that is still unreferenced after HTML
      table matching.
    """
    table_registry = [
        StructuredTable.from_dict(t) if isinstance(t, dict) else t
        for t in table_registry
    ]
    if not chunks or not table_registry:
        return chunks, {
            "added_table_chunks": 0,
            "unmatched_before": 0,
            "unmatched_after": 0,
            "skipped_toc_tables": 0,
        }

    existing_ids = {str(c.get("chunk_id")) for c in chunks if c.get("chunk_id")}
    materialized_table_ids = {
        str(table_refs[0])
        for c in chunks
        for table_refs in [c.get("table_refs") or []]
        if len(table_refs) == 1
        and table_refs[0]
        and str(c.get("chunk_id") or "").startswith("tbl_")
        and "_chunk" in str(c.get("chunk_id") or "")
    }
    referenced = {
        str(t_id)
        for c in chunks
        for t_id in (c.get("table_refs") or [])
        if t_id
    }
    article_titles = {}
    for c in chunks:
        article_num = c.get("article_num")
        if isinstance(article_num, int) and c.get("article_title") and article_num not in article_titles:
            article_titles[article_num] = c.get("article_title")

    contract_id = str(chunks[0].get("contract_id") or "")
    unmatched_before = len([t for t in table_registry if t.table_id not in referenced and not _is_toc_table(t)])
    skipped_toc = 0
    added = 0
    skipped_existing = 0

    for table in table_registry:
        table_id = str(table.table_id)
        if _is_toc_table(table):
            skipped_toc += 1
            continue
        if table_id in materialized_table_ids:
            skipped_existing += 1
            continue

        heading_path = [_normalize_heading_text(h) for h in (table.heading_path or []) if _normalize_heading_text(h)]
        heading_leaf = heading_path[-1] if heading_path else table_id
        doc_type = _infer_table_doc_type(table)

        rich_markdown = (table.markdown or "").strip()
        if not rich_markdown and (table.headers or table.rows):
            all_rows = [table.headers] + table.rows if table.headers else table.rows
            rich_markdown = _rows_to_markdown(all_rows)

        plain_rows = _rows_to_plaintext(table.headers, table.rows).strip()
        if not rich_markdown and not plain_rows:
            continue

        combined_topic_text = " ".join(
            [
                heading_leaf,
                " > ".join(heading_path),
                rich_markdown,
                plain_rows,
            ]
        )
        topics = _infer_table_topics(combined_topic_text)

        should_materialize = (
            table_id not in referenced
            or doc_type == "appendix"
            or any(t in {"wages", "vacation"} for t in topics)
        )
        if not should_materialize:
            continue

        if doc_type == "appendix":
            citation = _appendix_citation_label(table)
            if heading_leaf and heading_leaf.lower() not in citation.lower():
                citation = f"{citation} - {heading_leaf}"
        else:
            if isinstance(table.parent_article, int):
                citation = f"Article {table.parent_article}"
                if isinstance(table.parent_section, int):
                    citation += f", Section {table.parent_section}"
                citation += f" (Table: {heading_leaf})"
            else:
                citation = f"Table: {heading_leaf}"

        parent_context = " > ".join(heading_path) if heading_path else f"Structured Table {table_id}"
        summary_line = f"{citation}\n{parent_context}"
        embed_content = "\n\n".join([summary_line, plain_rows or rich_markdown]).strip()
        display_content = "\n\n".join([summary_line, rich_markdown or plain_rows]).strip()

        base_chunk_id = f"{table_id}_chunk"
        chunk_id = base_chunk_id
        suffix = 1
        while chunk_id in existing_ids:
            chunk_id = f"{base_chunk_id}__dup{suffix}"
            suffix += 1
        existing_ids.add(chunk_id)

        combined_topic_text = " ".join([citation, parent_context, plain_rows, rich_markdown])
        topics = list(dict.fromkeys(topics + _infer_table_topics(combined_topic_text)))

        article_num = table.parent_article if isinstance(table.parent_article, int) else None
        section_num = table.parent_section if isinstance(table.parent_section, int) else None

        chunks.append(
            {
                "chunk_id": chunk_id,
                "contract_id": contract_id,
                "doc_type": doc_type,
                "article_num": article_num,
                "article_title": article_titles.get(article_num, heading_leaf if article_num else "Appendix"),
                "section_num": section_num,
                "subsection": None,
                "subsection_title": None,
                "citation": citation,
                "parent_context": parent_context,
                "content": embed_content,
                "content_with_tables": display_content,
                "char_count": len(embed_content),
                "table_refs": [table_id],
                "applies_to": ["all"],
                "topics": topics,
                "cross_references": [],
                "summary": None,
                "is_definition": False,
                "is_exception": False,
                "hire_date_sensitive": False,
                "is_high_stakes": False,
            }
        )
        referenced.add(table_id)
        added += 1

    unmatched_after = len([t for t in table_registry if str(t.table_id) not in referenced and not _is_toc_table(t)])
    return chunks, {
        "added_table_chunks": added,
        "unmatched_before": unmatched_before,
        "unmatched_after": unmatched_after,
        "skipped_toc_tables": skipped_toc,
        "skipped_already_materialized": skipped_existing,
    }


# ============================================================================
# CLI entry point
# ============================================================================

def main():
    """Run table extraction pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Table extraction pipeline")
    parser.add_argument('--build-registry', action='store_true',
                        help='Build table registry from JSON export')
    parser.add_argument('--apply-to-chunks', action='store_true',
                        help='Apply tables to enriched chunks')
    parser.add_argument('--json-path', type=Path, default=None,
                        help='Path to JSON export (default: config CONTRACT_JSON_FILE)')
    parser.add_argument('--chunks-path', type=Path, default=None,
                        help='Path to enriched chunks JSON')
    args = parser.parse_args()

    # Default: run both phases
    run_both = not args.build_registry and not args.apply_to_chunks

    registry = None

    if args.build_registry or run_both:
        print("Phase A: Building table registry from JSON...")
        registry = build_table_registry(args.json_path)
        save_table_registry(registry)

        # Print summary
        for t in registry:
            art = f"Art {t.parent_article}" if t.parent_article else "Unknown"
            heading = t.heading_path[-1][:50] if t.heading_path else "No heading"
            print(f"  {t.table_id}: {art} | {heading} | {len(t.rows)} rows | perfect={t.is_perfect}")

    if args.apply_to_chunks or run_both:
        print("\nPhase B: Applying tables to chunks...")

        # Load registry if not already built
        if registry is None:
            registry_file = TABLES_DIR / "structured_tables.json"
            if not registry_file.exists():
                print("Error: No table registry found. Run --build-registry first.")
                return
            with open(registry_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            registry = [StructuredTable(**d) for d in data]

        # Load chunks
        chunks_path = args.chunks_path
        if chunks_path is None:
            chunks_path = CHUNKS_DIR / "contract_chunks_enriched.json"
            if not chunks_path.exists():
                chunks_path = CHUNKS_DIR / "contract_chunks_smart.json"

        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        print(f"  Loaded {len(chunks)} chunks from {chunks_path}")

        # Apply tables
        chunks, registry = apply_tables_to_chunks(chunks, registry)

        # Save modified chunks
        output_path = chunks_path  # Overwrite in place
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(chunks)} chunks to {output_path}")

        # Save updated registry with parent_chunk_ids
        save_table_registry(registry)

        # Print stats
        with_tables = sum(1 for c in chunks if c.get('table_refs'))
        print(f"  {with_tables} chunks now have table references")


if __name__ == "__main__":
    main()
