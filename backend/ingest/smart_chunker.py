"""
Smart Chunker - Subsection-aware hierarchical chunking.

Improvements over the original parser:
1. Splits sections with lettered subsections (a, b, c, d...)
2. Splits sections with numbered subsections (1, 2, 3...)
3. Respects size limits (target: 500-1500 chars)
4. Preserves hierarchical context in each chunk
5. Extracts article/section titles
"""

import re
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

from backend.config import CONTRACT_ID


@dataclass
class RawChunk:
    """Intermediate chunk before LLM enrichment."""
    chunk_id: str
    contract_id: str = CONTRACT_ID
    doc_type: str = "cba"  # "cba" for contract articles, "lou" for letters of understanding

    # Hierarchy
    article_num: Optional[int] = None
    article_title: Optional[str] = None
    section_num: Optional[int] = None
    subsection: Optional[str] = None
    subsection_title: Optional[str] = None
    citation: str = ""
    parent_context: str = ""

    # Content
    content: str = ""
    char_count: int = 0

    # Placeholder metadata (to be filled by enricher)
    applies_to: list = field(default_factory=lambda: ["all"])
    topics: list = field(default_factory=list)
    cross_references: list = field(default_factory=list)
    summary: Optional[str] = None
    is_definition: bool = False
    is_exception: bool = False
    hire_date_sensitive: bool = False
    is_high_stakes: bool = False
    
    def __post_init__(self):
        self.char_count = len(self.content)
    
    def to_dict(self) -> dict:
        return asdict(self)


class SmartChunker:
    """
    Parses contract markdown into enrichment-ready chunks.
    
    Key features:
    - Detects and splits lettered subsections (a. TITLE, b. TITLE)
    - Detects and splits numbered subsections (1. Item, 2. Item)
    - Splits long sections without subsections into paragraphs
    - Maintains hierarchical context for each chunk
    """
    
    # Patterns
    # Article header patterns - match both # and ## formats
    # Title class allows uppercase, digits, spaces, &, commas, hyphens, slashes, parens
    # to handle titles like "401K PLAN", "SECTION 125 PLANS", "HEALTH & WELFARE"
    ARTICLE_HEADER = re.compile(
        r'^#{1,3}\s*ARTICLE\s+(\d+)\s*\n#{1,3}\s*([A-Z0-9][A-Z0-9 \t&,/()\-\'".:]+)',
        re.MULTILINE
    )
    ARTICLE_HEADER_SINGLE = re.compile(
        r'^#{1,3}\s*ARTICLE\s+(\d+)\s+([A-Z0-9][A-Z0-9 \t&,/()\-\'".:]+)',
        re.MULTILINE
    )
    SECTION_HEADER = re.compile(
        r'^\s*\*{0,2}Section\s+\*{0,2}(\d+)\*{0,2}\s*(?:[.)]\s*)?(?:\*{0,2}([^.\n]{1,180}))?',
        re.IGNORECASE | re.MULTILINE
    )
    LOU_HEADER = re.compile(
        r'^##\s*Letter\s+of\s+Understanding\s+#?(\d+)',
        re.MULTILINE | re.IGNORECASE
    )
    # Detect the LOU section boundary (standard CBA language)
    LOU_SECTION_HEADER = re.compile(
        r'LETTERS\s+OF\s+UNDERSTANDING',
        re.IGNORECASE
    )
    # Individual LOU items within the full-text LOU section: ## N. Title
    LOU_ITEM_HEADER = re.compile(
        r'^##\s*(\d+)\.\s*(.+?)$',
        re.MULTILINE
    )
    
    # Subsection patterns - match full titles like "DRIVE UP AND GO"
    LETTERED_SUBSECTION = re.compile(
        r'\n\s*\*{0,2}([a-z])[\.\)]\s*\*{0,2}\s*([A-Z][A-Z\s&]+?)(?:\s*\.|\s*\n|\s*\*|$)',
        re.MULTILINE
    )
    NUMBERED_SUBSECTION = re.compile(
        r'\n\s*\*{0,2}(\d+)[\.\)]\s*\*{0,2}\s*(.+?)(?:\n|$)',
        re.MULTILINE
    )
    
    # Size limits
    MIN_CHUNK_SIZE = 100
    TARGET_CHUNK_SIZE = 800
    MAX_CHUNK_SIZE = 2000
    
    def __init__(self, contract_id: str = CONTRACT_ID):
        self.contract_id = contract_id
        self.chunks: list[RawChunk] = []
        self.current_article_num = None
        self.current_article_title = None
        self._chunk_id_counts: dict[str, int] = {}

    @staticmethod
    def _mask_inline_tags(text: str) -> str:
        """
        Replace inline HTML tags with same-length spaces so regex scanning
        preserves original character offsets.
        """
        return re.sub(r'<[^>\n]+>', lambda m: " " * len(m.group(0)), text)

    @staticmethod
    def _normalize_heading_text(text: str) -> str:
        """Normalize heading text by removing lightweight markup noise."""
        normalized = re.sub(r'</?u>|</?strong>|</?em>|</?b>|</?i>', ' ', text, flags=re.IGNORECASE)
        normalized = re.sub(r'[*_`]+', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip(" .-\t")
        return normalized

    @staticmethod
    def _normalize_subsection_token(subsection: Optional[str]) -> Optional[str]:
        """Normalize legal subsection identifiers for stable sorting/filtering."""
        if subsection is None:
            return None
        value = str(subsection).strip()
        if not value:
            return None
        if len(value) == 1 and value.isalpha():
            return value.lower()
        value = re.sub(r"\s+", " ", value)
        return value

    @staticmethod
    def _slugify_id_fragment(value: str) -> str:
        """Convert arbitrary token to safe lowercase ID fragment."""
        slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
        return slug or "x"

    @classmethod
    def _normalize_segment_token(cls, segment_token: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        """
        Normalize section-split segment token into:
        - chunk-id fragment (stable, machine-safe)
        - human citation suffix (readable)
        """
        if segment_token is None:
            return None, None
        raw = str(segment_token).strip()
        if not raw:
            return None, None
        token = raw.lower()

        part_match = re.fullmatch(r"part[\s_-]*(\d+)", token)
        if part_match:
            n = int(part_match.group(1))
            return f"seg_{n:03d}", f"Part {n}"

        range_match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", token)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            return f"seg_{start:03d}_{end:03d}", f"Part {start}-{end}"

        plus_match = re.fullmatch(r"(\d+)\+", token)
        if plus_match:
            start = int(plus_match.group(1))
            return f"seg_{start:03d}_plus", f"Part {start}+"

        number_match = re.fullmatch(r"(\d+)", token)
        if number_match:
            n = int(number_match.group(1))
            return f"seg_{n:03d}", f"Part {n}"

        slug = cls._slugify_id_fragment(token)
        return f"seg_{slug}", f"Part {raw}"

    def _allocate_chunk_id(self, base_id: str) -> str:
        """
        Allocate a collision-safe chunk ID.

        Ingestion used to emit duplicate IDs for some subsection layouts.
        We keep the first canonical ID and suffix later collisions.
        """
        seen = self._chunk_id_counts.get(base_id, 0)
        self._chunk_id_counts[base_id] = seen + 1
        if seen == 0:
            return base_id
        return f"{base_id}__dup{seen}"
    
    def parse_markdown(self, markdown_path: Path) -> list[RawChunk]:
        """Parse markdown file into smart chunks."""
        with open(markdown_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return self.parse_content(content)
    
    def parse_content(self, content: str) -> list[RawChunk]:
        """Parse markdown content into smart chunks."""
        self.chunks = []
        self._chunk_id_counts = {}
        scan_content = self._mask_inline_tags(content)

        # Find where actual article content starts (skip TOC references).
        first_article_match = self.ARTICLE_HEADER.search(scan_content)
        if first_article_match is None:
            first_article_match = self.ARTICLE_HEADER_SINGLE.search(scan_content)
        first_article_pos = first_article_match.start() if first_article_match else 0

        # Detect LOU section boundary and separate it.
        # Find the first non-HTML occurrence that starts a line (the section header).
        lou_content = None
        best_lou_match = None
        for m in self.LOU_SECTION_HEADER.finditer(scan_content):
            # Ignore TOC/header references that appear before first actual article heading.
            if m.start() < first_article_pos:
                continue
            line_start = scan_content.rfind('\n', 0, m.start()) + 1
            line = scan_content[line_start:m.end() + 20]
            if '<' not in line:  # Skip matches inside HTML tags
                # Check this is a heading line (starts with # or is all caps)
                line_prefix = scan_content[line_start:m.start()].strip()
                if not line_prefix or line_prefix.startswith('#') or line_prefix.isupper():
                    best_lou_match = m
                    break  # Take the first valid match (section header)
        if best_lou_match:
            # Back up to the start of the line
            lou_start = content.rfind('\n', 0, best_lou_match.start())
            if lou_start < 0:
                lou_start = 0
            lou_content = content[lou_start:]
            content = content[:lou_start]

        # Split by article headers
        articles = self._split_by_articles(content)

        for article_num, article_title, article_content in articles:
            self.current_article_num = article_num
            self.current_article_title = article_title

            if article_num is None:
                # Letter of Understanding or other
                self._process_lou(article_content)
            else:
                self._process_article(article_num, article_title, article_content)

        # Process LOU section separately
        if lou_content:
            self._process_lou_section(lou_content)

        return self.chunks
    
    def _split_by_articles(self, content: str) -> list[tuple]:
        """Split content into (article_num, title, content) tuples."""
        articles = []
        scan_content = self._mask_inline_tags(content)

        # Find all article positions
        positions_by_key = {}

        def add_position(pos: int, num: Optional[int], title: str):
            clean_title = self._normalize_heading_text(title or "")
            if num is not None and not clean_title:
                clean_title = f"ARTICLE {num}"
            key = (pos, num)
            existing = positions_by_key.get(key)
            if existing is None or len(clean_title) > len(existing[2]):
                positions_by_key[key] = (pos, num, clean_title)

        # Two-line headers: ## ARTICLE N\n## TITLE
        for match in self.ARTICLE_HEADER.finditer(scan_content):
            add_position(match.start(), int(match.group(1)), match.group(2))

        # Single-line headers: ## ARTICLE N TITLE
        for match in self.ARTICLE_HEADER_SINGLE.finditer(scan_content):
            add_position(match.start(), int(match.group(1)), match.group(2))

        # LOUs
        for match in self.LOU_HEADER.finditer(scan_content):
            add_position(match.start(), None, f"Letter of Understanding {match.group(1)}")

        positions = sorted(positions_by_key.values(), key=lambda x: x[0])

        # Sort by position
        # Extract content between positions
        for i, (pos, num, title) in enumerate(positions):
            end_pos = positions[i + 1][0] if i + 1 < len(positions) else len(content)
            article_content = content[pos:end_pos]
            articles.append((num, title, article_content))
        
        return articles
    
    def _process_article(self, article_num: int, article_title: str, content: str):
        """Process an article into chunks."""
        # Find all sections
        sections = self._split_by_sections(content)
        
        if not sections:
            # No sections found, treat whole article as one chunk
            self._create_chunk(
                article_num=article_num,
                article_title=article_title,
                content=content
            )
            return
        
        for section_num, section_title, section_content in sections:
            self._process_section(
                article_num, article_title,
                section_num, section_title, section_content
            )
    
    def _split_by_sections(self, content: str) -> list[tuple]:
        """Split article content by sections."""
        sections = []
        scan_content = self._mask_inline_tags(content)

        # Find section headers from tag-masked text so variants like
        # "<u>Section 48</u>." are detected with stable offsets.
        matches = list(self.SECTION_HEADER.finditer(scan_content))
        
        for i, match in enumerate(matches):
            section_num = int(match.group(1))
            raw_title = (match.group(2) or "").strip().rstrip('.')
            section_title = self._normalize_heading_text(raw_title) or f"Section {section_num}"
            
            start_pos = match.start()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            section_content = content[start_pos:end_pos].strip()
            
            sections.append((section_num, section_title, section_content))
        
        return sections
    
    def _process_section(self, article_num: int, article_title: str,
                         section_num: int, section_title: str, content: str):
        """Process a section, potentially splitting into subsections."""
        
        # Check for lettered subsections (a. TITLE, b. TITLE)
        lettered = list(self.LETTERED_SUBSECTION.finditer(content))
        
        if len(lettered) >= 2:
            # Has multiple lettered subsections - split them
            self._split_lettered_subsections(
                article_num, article_title,
                section_num, section_title,
                content, lettered
            )
            return
        
        # Check for numbered subsections (1. Item, 2. Item) - mainly in Article 35
        if len(content) > self.MAX_CHUNK_SIZE:
            numbered = list(self.NUMBERED_SUBSECTION.finditer(content))
            if len(numbered) >= 3:
                self._split_numbered_subsections(
                    article_num, article_title,
                    section_num, section_title,
                    content, numbered
                )
                return
        
        # No subsections or too few - keep as single chunk
        # But split if too long
        if len(content) > self.MAX_CHUNK_SIZE:
            self._split_by_paragraphs(
                article_num, article_title,
                section_num, section_title,
                content
            )
        else:
            self._create_chunk(
                article_num=article_num,
                article_title=article_title,
                section_num=section_num,
                section_title=section_title,
                content=content
            )
    
    def _split_lettered_subsections(self, article_num: int, article_title: str,
                                     section_num: int, section_title: str,
                                     content: str, matches: list):
        """Split section by lettered subsections."""
        
        for i, match in enumerate(matches):
            letter = match.group(1).lower()
            subsection_title = self._normalize_heading_text(match.group(2))
            
            start_pos = match.start()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            subsection_content = content[start_pos:end_pos].strip()
            
            self._create_chunk(
                article_num=article_num,
                article_title=article_title,
                section_num=section_num,
                section_title=section_title,
                subsection=letter,
                subsection_title=subsection_title,
                content=subsection_content
            )
    
    def _split_numbered_subsections(self, article_num: int, article_title: str,
                                     section_num: int, section_title: str,
                                     content: str, matches: list):
        """Split section by numbered subsections (for things like sick leave rules)."""
        
        # First, create a chunk for content before the first numbered item
        if matches and matches[0].start() > 100:
            intro_content = content[:matches[0].start()].strip()
            if intro_content:
                self._create_chunk(
                    article_num=article_num,
                    article_title=article_title,
                    section_num=section_num,
                    section_title=section_title,
                    content=intro_content
                )
        
        # Group numbered items into reasonable chunks
        current_chunk = ""
        current_start_num = None
        
        for i, match in enumerate(matches):
            num = match.group(1)
            
            start_pos = match.start()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            item_content = content[start_pos:end_pos].strip()
            
            if current_start_num is None:
                current_start_num = num
            
            # Check if adding this item would exceed max size
            if len(current_chunk) + len(item_content) > self.TARGET_CHUNK_SIZE and current_chunk:
                # Save current chunk
                self._create_chunk(
                    article_num=article_num,
                    article_title=article_title,
                    section_num=section_num,
                    section_title=section_title,
                    segment_token=f"{current_start_num}-{int(num)-1}",
                    content=current_chunk
                )
                current_chunk = item_content
                current_start_num = num
            else:
                current_chunk += "\n\n" + item_content if current_chunk else item_content
        
        # Save remaining chunk
        if current_chunk:
            self._create_chunk(
                article_num=article_num,
                article_title=article_title,
                section_num=section_num,
                section_title=section_title,
                segment_token=f"{current_start_num}+" if current_start_num != num else current_start_num,
                content=current_chunk
            )
    
    def _split_by_paragraphs(self, article_num: int, article_title: str,
                              section_num: int, section_title: str, content: str):
        """Split long section by paragraphs."""
        paragraphs = content.split('\n\n')
        
        current_chunk = ""
        part_num = 1
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(current_chunk) + len(para) > self.TARGET_CHUNK_SIZE and current_chunk:
                # Save current chunk
                self._create_chunk(
                    article_num=article_num,
                    article_title=article_title,
                    section_num=section_num,
                    section_title=section_title,
                    segment_token=f"part{part_num}",
                    content=current_chunk
                )
                current_chunk = para
                part_num += 1
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        # Save remaining
        if current_chunk:
            if part_num == 1:
                # Only one part, don't add part number
                self._create_chunk(
                    article_num=article_num,
                    article_title=article_title,
                    section_num=section_num,
                    section_title=section_title,
                    content=current_chunk
                )
            else:
                self._create_chunk(
                    article_num=article_num,
                    article_title=article_title,
                    section_num=section_num,
                    section_title=section_title,
                    segment_token=f"part{part_num}",
                    content=current_chunk
                )
    
    def _process_lou(self, content: str):
        """Process Letter of Understanding."""
        # Extract LOU number from content
        lou_match = self.LOU_HEADER.search(content)
        if lou_match:
            lou_num = lou_match.group(1)
            
            # Split if too long
            if len(content) > self.MAX_CHUNK_SIZE:
                paragraphs = content.split('\n\n')
                current_chunk = ""
                part_num = 1
                
                for para in paragraphs:
                    para = para.strip()
                    if not para:
                        continue
                    
                    if len(current_chunk) + len(para) > self.TARGET_CHUNK_SIZE and current_chunk:
                        self._create_lou_chunk(lou_num, part_num, current_chunk)
                        current_chunk = para
                        part_num += 1
                    else:
                        current_chunk += "\n\n" + para if current_chunk else para
                
                if current_chunk:
                    self._create_lou_chunk(lou_num, part_num, current_chunk)
            else:
                self._create_lou_chunk(lou_num, 1, content)
    
    def _create_lou_chunk(self, lou_num: str, part_num: int, content: str):
        """Create a Letter of Understanding chunk."""
        base_chunk_id = f"lou{lou_num}_part{part_num}" if part_num > 1 else f"lou{lou_num}"
        chunk_id = self._allocate_chunk_id(base_chunk_id)
        citation = f"Letter of Understanding {lou_num}"
        if part_num > 1:
            citation += f", Part {part_num}"

        chunk = RawChunk(
            chunk_id=chunk_id,
            contract_id=self.contract_id,
            doc_type="lou",
            citation=citation,
            parent_context=f"Letter of Understanding {lou_num}",
            content=content,
        )
        self.chunks.append(chunk)

    def _process_lou_section(self, content: str):
        """
        Process the full Letters of Understanding section.

        The LOU section has two parts:
        1. Summary list (brief descriptions, no ## headers)
        2. Full text (detailed content with ## N. or plain N. headers)

        We process the full-text section by finding LOU boundaries using both
        ## N. and plain N. headers, then using increasing-number heuristic to
        distinguish top-level LOUs from sub-items within LOUs.
        """
        # Find the start of the full-text section.
        # The LOU section has a summary list followed by full-text content.
        # We use the first ## N. header as the reliable full-text section start,
        # since only full-text LOUs have ## heading markers.
        first_heading_item = self.LOU_ITEM_HEADER.search(content)
        if first_heading_item:
            full_text_start = first_heading_item.start()
        else:
            # No ## N. headers at all
            return

        full_text = content[full_text_start:]

        # Split by ## N. headers only (reliable boundaries).
        # Some LOUs between ## boundaries (e.g., 9-13 between ## 8 and ## 14)
        # will be embedded in the nearest ## LOU's content. This is acceptable
        # for retrieval since the content is still findable.
        lou_boundaries = list(self.LOU_ITEM_HEADER.finditer(full_text))

        if not lou_boundaries:
            return

        for i, match in enumerate(lou_boundaries):
            lou_num = match.group(1)
            lou_title = match.group(2).strip().rstrip('.')

            # Extract content from this LOU to the next
            start_pos = match.start()
            end_pos = lou_boundaries[i + 1].start() if i + 1 < len(lou_boundaries) else len(full_text)
            item_content = full_text[start_pos:end_pos].strip()

            # Clean page number artifacts (e.g., "63 PUEBLO CLERKS\n2022-2025\n---")
            item_content = re.sub(
                r'\n\d{2,3}\s+PUEBLO CLERKS\s*\n\s*2022-2025\s*\n+---\s*\n*',
                '\n',
                item_content
            )
            item_content = re.sub(
                r'\n\d{2,3}\s+PUEBLO CLERKS\s*\n\s*2022-2025\s*$',
                '',
                item_content
            )
            # Clean standalone page numbers and separators
            item_content = re.sub(r'\n\d{2,3}\s*\n', '\n', item_content)
            item_content = re.sub(r'\n---\s*\n+', '\n\n', item_content)
            item_content = item_content.strip()

            if not item_content:
                continue

            # Extract a short title for the citation
            short_title = lou_title.split('.')[0].strip()
            if len(short_title) > 60:
                short_title = short_title[:57] + "..."

            # For small items, prepend the LOU title as context for better embedding
            if len(item_content) < self.MIN_CHUNK_SIZE:
                item_content = f"Letter of Understanding {lou_num}: {short_title}\n\n{item_content}"

            self._emit_lou_chunk(lou_num, short_title, item_content)

    def _emit_lou_chunk(self, lou_num: str, short_title: str, item_content: str):
        """Emit one or more chunks for a single LOU item."""
        # Split long LOUs by paragraphs if needed
        if len(item_content) > self.MAX_CHUNK_SIZE:
            paragraphs = item_content.split('\n\n')
            current_chunk = ""
            part_num = 1

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                if len(current_chunk) + len(para) > self.TARGET_CHUNK_SIZE and current_chunk:
                    chunk = RawChunk(
                        chunk_id=self._allocate_chunk_id(f"lou_{lou_num}_part{part_num}"),
                        contract_id=self.contract_id,
                        doc_type="lou",
                        citation=f"Letter of Understanding {lou_num}: {short_title}, Part {part_num}",
                        parent_context=f"Letters of Understanding > Item {lou_num}: {short_title}",
                        content=current_chunk,
                    )
                    self.chunks.append(chunk)
                    current_chunk = para
                    part_num += 1
                else:
                    current_chunk += "\n\n" + para if current_chunk else para

            if current_chunk:
                chunk = RawChunk(
                    chunk_id=self._allocate_chunk_id(
                        f"lou_{lou_num}_part{part_num}" if part_num > 1 else f"lou_{lou_num}"
                    ),
                    contract_id=self.contract_id,
                    doc_type="lou",
                    citation=f"Letter of Understanding {lou_num}: {short_title}" + (f", Part {part_num}" if part_num > 1 else ""),
                    parent_context=f"Letters of Understanding > Item {lou_num}: {short_title}",
                    content=current_chunk,
                )
                self.chunks.append(chunk)
        else:
            chunk = RawChunk(
                chunk_id=self._allocate_chunk_id(f"lou_{lou_num}"),
                contract_id=self.contract_id,
                doc_type="lou",
                citation=f"Letter of Understanding {lou_num}: {short_title}",
                parent_context=f"Letters of Understanding > Item {lou_num}: {short_title}",
                content=item_content,
            )
            self.chunks.append(chunk)
    
    def _create_chunk(self, article_num: int, article_title: str,
                       section_num: int = None, section_title: str = None,
                       subsection: str = None, subsection_title: str = None,
                       segment_token: str = None,
                       content: str = ""):
        """Create a chunk with proper ID and context."""
        subsection_norm = self._normalize_subsection_token(subsection)
        segment_id_fragment, segment_citation_label = self._normalize_segment_token(segment_token)

        # Build chunk ID
        chunk_id = f"art{article_num}"
        if section_num:
            chunk_id += f"_sec{section_num}"
        if subsection_norm:
            chunk_id += f"_sub_{self._slugify_id_fragment(subsection_norm)}"
        if segment_id_fragment:
            chunk_id += f"_{segment_id_fragment}"
        chunk_id = self._allocate_chunk_id(chunk_id)
        
        # Build citation
        citation = f"Article {article_num}"
        if section_num:
            citation += f", Section {section_num}"
        if subsection_norm and subsection_title:
            citation += f", Subsection {subsection_norm} ({subsection_title})"
        elif subsection_norm:
            citation += f", Subsection {subsection_norm}"
        if segment_citation_label:
            citation += f", {segment_citation_label}"
        
        # Build parent context
        context_parts = [f"Article {article_num} ({article_title})"]
        if section_num and section_title:
            context_parts.append(f"Section {section_num} ({section_title})")
        elif section_num:
            context_parts.append(f"Section {section_num}")
        if subsection_norm and subsection_title:
            context_parts.append(f"Subsection {subsection_norm} ({subsection_title})")
        elif subsection_norm:
            context_parts.append(f"Subsection {subsection_norm}")
        if segment_citation_label:
            context_parts.append(segment_citation_label)
        parent_context = " > ".join(context_parts)
        
        chunk = RawChunk(
            chunk_id=chunk_id,
            contract_id=self.contract_id,
            article_num=article_num,
            article_title=article_title,
            section_num=section_num,
            subsection=subsection_norm,
            subsection_title=subsection_title,
            citation=citation,
            parent_context=parent_context,
            content=content,
        )
        self.chunks.append(chunk)
    
    def save_chunks(self, output_path: Path):
        """Save chunks to JSON file."""
        chunks_data = [chunk.to_dict() for chunk in self.chunks]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(chunks_data)} chunks to {output_path}")
        
        # Print size distribution
        sizes = [c.char_count for c in self.chunks]
        print(f"\nSize stats:")
        print(f"  Min: {min(sizes)} chars")
        print(f"  Max: {max(sizes)} chars")
        print(f"  Avg: {sum(sizes) // len(sizes)} chars")
        print(f"  Chunks > 1500: {sum(1 for s in sizes if s > 1500)}")


def main():
    """Re-chunk the Pueblo contract with smart chunking."""
    from backend.config import CONTRACT_MD_FILE, CHUNKS_DIR
    
    print("Smart Chunker - Subsection-Aware Parsing")
    print("=" * 60)
    
    chunker = SmartChunker()
    chunks = chunker.parse_markdown(CONTRACT_MD_FILE)
    
    print(f"\nGenerated {len(chunks)} chunks")
    
    # Show sample chunks
    print("\n--- Sample Chunks ---")
    for chunk in chunks[:5]:
        print(f"\n{chunk.citation}")
        print(f"  ID: {chunk.chunk_id}")
        print(f"  Chars: {chunk.char_count}")
        print(f"  Context: {chunk.parent_context}")
    
    # Show Article 7 Section 14 chunks specifically
    print("\n--- Article 7, Section 14 (Job Definitions) ---")
    art7_chunks = [c for c in chunks if c.article_num == 7 and c.section_num == 14]
    for chunk in art7_chunks:
        print(f"  {chunk.citation}: {chunk.char_count} chars")
        if chunk.subsection_title:
            print(f"    Title: {chunk.subsection_title}")
    
    # Save to new file
    output_path = CHUNKS_DIR / "contract_chunks_smart.json"
    chunker.save_chunks(output_path)


if __name__ == "__main__":
    main()

