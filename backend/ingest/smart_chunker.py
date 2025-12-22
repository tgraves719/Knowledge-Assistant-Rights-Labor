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


@dataclass
class RawChunk:
    """Intermediate chunk before LLM enrichment."""
    chunk_id: str
    contract_id: str = "safeway_pueblo_clerks_2022"
    
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
    ARTICLE_HEADER = re.compile(
        r'^#{1,2}\s*ARTICLE\s+(\d+)\s*\n#{1,2}\s*([A-Z][A-Z\s&,]+)',
        re.MULTILINE
    )
    ARTICLE_HEADER_SINGLE = re.compile(
        r'^#{1,2}\s*ARTICLE\s+(\d+)\s+([A-Z][A-Z\s&,]+)',
        re.MULTILINE
    )
    SECTION_HEADER = re.compile(
        r'Section\s+\*{0,2}(\d+)\*{0,2}[.\s]+\*{0,2}([^.\n]+)',
        re.IGNORECASE
    )
    LOU_HEADER = re.compile(
        r'^##\s*Letter\s+of\s+Understanding\s+#?(\d+)',
        re.MULTILINE | re.IGNORECASE
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
    
    def __init__(self, contract_id: str = "safeway_pueblo_clerks_2022"):
        self.contract_id = contract_id
        self.chunks: list[RawChunk] = []
        self.current_article_num = None
        self.current_article_title = None
    
    def parse_markdown(self, markdown_path: Path) -> list[RawChunk]:
        """Parse markdown file into smart chunks."""
        with open(markdown_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return self.parse_content(content)
    
    def parse_content(self, content: str) -> list[RawChunk]:
        """Parse markdown content into smart chunks."""
        self.chunks = []
        
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
        
        return self.chunks
    
    def _split_by_articles(self, content: str) -> list[tuple]:
        """Split content into (article_num, title, content) tuples."""
        articles = []
        
        # Find all article positions
        positions = []
        
        # Two-line headers: ## ARTICLE N\n## TITLE
        for match in self.ARTICLE_HEADER.finditer(content):
            positions.append((match.start(), int(match.group(1)), match.group(2).strip()))
        
        # Single-line headers: ## ARTICLE N TITLE
        for match in self.ARTICLE_HEADER_SINGLE.finditer(content):
            positions.append((match.start(), int(match.group(1)), match.group(2).strip()))
        
        # LOUs
        for match in self.LOU_HEADER.finditer(content):
            positions.append((match.start(), None, f"Letter of Understanding {match.group(1)}"))
        
        # Sort by position
        positions.sort(key=lambda x: x[0])
        
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
        
        # Find section headers
        matches = list(self.SECTION_HEADER.finditer(content))
        
        for i, match in enumerate(matches):
            section_num = int(match.group(1))
            section_title = match.group(2).strip().rstrip('.')
            
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
            subsection_title = match.group(2).strip()
            
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
                    subsection=f"{current_start_num}-{int(num)-1}",
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
                subsection=f"{current_start_num}+" if current_start_num != num else current_start_num,
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
                    subsection=f"part{part_num}",
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
                    subsection=f"part{part_num}",
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
        chunk_id = f"lou{lou_num}_part{part_num}" if part_num > 1 else f"lou{lou_num}"
        citation = f"Letter of Understanding {lou_num}"
        if part_num > 1:
            citation += f", Part {part_num}"
        
        chunk = RawChunk(
            chunk_id=chunk_id,
            contract_id=self.contract_id,
            citation=citation,
            parent_context=f"Letter of Understanding {lou_num}",
            content=content,
        )
        self.chunks.append(chunk)
    
    def _create_chunk(self, article_num: int, article_title: str,
                       section_num: int = None, section_title: str = None,
                       subsection: str = None, subsection_title: str = None,
                       content: str = ""):
        """Create a chunk with proper ID and context."""
        
        # Build chunk ID
        chunk_id = f"art{article_num}"
        if section_num:
            chunk_id += f"_sec{section_num}"
        if subsection:
            chunk_id += f"_{subsection}"
        
        # Build citation
        citation = f"Article {article_num}"
        if section_num:
            citation += f", Section {section_num}"
        if subsection and subsection_title:
            citation += f", Subsection {subsection} ({subsection_title})"
        elif subsection:
            citation += f", Part {subsection}"
        
        # Build parent context
        context_parts = [f"Article {article_num} ({article_title})"]
        if section_num and section_title:
            context_parts.append(f"Section {section_num} ({section_title})")
        elif section_num:
            context_parts.append(f"Section {section_num}")
        if subsection and subsection_title:
            context_parts.append(f"Subsection {subsection} ({subsection_title})")
        parent_context = " > ".join(context_parts)
        
        chunk = RawChunk(
            chunk_id=chunk_id,
            contract_id=self.contract_id,
            article_num=article_num,
            article_title=article_title,
            section_num=section_num,
            subsection=subsection,
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

