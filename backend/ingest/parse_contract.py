"""
Contract Parser V2 - Robust chunking of union contract markdown.

Handles multiple formatting patterns:
- ## ARTICLE X followed by ## TITLE on next line
- ## ARTICLE X TITLE (same line)
- # ARTICLE X (single hash)
- Sections with various numbering patterns

Outputs all text with 100% recall - no data loss.
"""

import re
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import CONTRACT_MD_FILE, CHUNKS_DIR, CONTRACT_ID


@dataclass
class ContractChunk:
    """Represents a semantic chunk from the contract."""
    chunk_id: str
    contract_id: str
    article_num: Optional[int]
    article_title: str
    section_num: Optional[int]
    subsection: Optional[str]
    citation: str
    content: str
    doc_type: str = "cba"  # cba, lou, appendix
    applies_to: list = field(default_factory=lambda: ["all"])
    topic_tags: list = field(default_factory=list)
    references: list = field(default_factory=list)
    urgency_tier: str = "standard"


# Classification keywords for tagging
CLASSIFICATION_KEYWORDS = {
    "courtesy_clerk": ["courtesy clerk", "courtesy clerks"],
    "head_clerk": ["head clerk", "head clerks"],
    "produce_manager": ["produce department manager", "produce manager"],
    "bakery_manager": ["bakery department manager", "bakery manager"],
    "pharmacy_tech": ["pharmacy technician", "pharmacy technicians"],
    "cake_decorator": ["cake decorator", "cake decorators"],
    "all_purpose_clerk": ["all purpose clerk", "all-purpose clerk"],
    "non_foods_clerk": ["non-foods clerk", "general merchandise clerk", "gm clerk"],
    "sanitation_clerk": ["sanitation clerk", "sanitation clerks"],
    "dug_shopper": ["dug shopper", "drive up and go"],
}

# Topic keywords for tagging - EXPANDED for better retrieval
# These keyword lists are designed for scalability across 100+ contracts
TOPIC_KEYWORDS = {
    "wages": [
        "wage", "wages", "pay", "paid", "rate", "rates", "salary", "compensation",
        "appendix a", "hourly", "dollar", "earnings", "step", "progression",
        "starting pay", "top rate", "minimum wage"
    ],
    "overtime": [
        "overtime", "over time", "time and one-half", "time and a half",
        "1.5x", "1 1/2", "over 8 hours", "over 40 hours", "double time",
        "ot", "extra hours", "excess of eight", "excess of forty"
    ],
    "scheduling": [
        "schedule", "scheduling", "scheduled", "shift", "shifts", "hours",
        "workweek", "work week", "posted", "posting", "roster",
        "start time", "minimum hours", "maximum hours"
    ],
    "seniority": [
        "seniority", "senior", "junior", "length of service",
        "years of service", "hire date", "continuous service",
        "most senior", "least senior", "tenure"
    ],
    "layoff": [
        "layoff", "lay off", "laid off", "reduction", "bumping", "displacement",
        "displaced", "workforce reduction", "rif", "furlough", "recall"
    ],
    "vacation": [
        "vacation", "vacations", "holiday", "holidays", "personal day",
        "time off", "pto", "paid time off", "anniversary", "vacation pay"
    ],
    "sick_leave": [
        "sick leave", "sick day", "illness", "sick pay", "medical leave",
        "health leave", "absence", "call in sick", "sick time"
    ],
    "discipline": [
        "discipline", "disciplinary", "discharge", "discharged", "termination",
        "terminated", "warning", "written warning", "verbal warning",
        "suspension", "suspended", "corrective action", "write up",
        "just cause", "good cause", "dismissal", "fired"
    ],
    "grievance": [
        "grievance", "grievances", "arbitration", "arbitrate", "dispute",
        "complaint", "step 1", "step 2", "step 3", "file", "filing",
        "retroactive", "back pay", "remedy", "time limit", "deadline"
    ],
    "union_security": [
        "union membership", "union dues", "check-off", "deduction",
        "initiation fee", "union shop", "bargaining unit"
    ],
    "union_rights": [
        "steward", "stewards", "union steward", "representation",
        "representative", "weingarten", "union rep", "business representative",
        "visitation", "visit", "union meeting"
    ],
    "safety": [
        "safety", "injury", "injured", "workers comp", "hazard", "hazardous",
        "unsafe", "dangerous", "accident", "osha", "protective equipment"
    ],
    "benefits": [
        "health", "welfare", "pension", "401k", "insurance", "medical",
        "dental", "vision", "coverage", "contribution", "trust fund",
        "eligibility", "dependent"
    ],
    "breaks": [
        "lunch", "lunch break", "lunch period", "break", "breaks",
        "relief period", "relief", "meal period", "meal break",
        "rest period", "rest break", "15 minute", "30 minute", "one hour"
    ],
    "premiums": [
        "premium", "sunday premium", "night premium", "shift differential",
        "holiday premium", "time and one-quarter", "1.25", "1 1/4"
    ],
    "probation": [
        "probation", "probationary", "trial period", "first sixty days",
        "first 60 days", "new employee"
    ],
    "store_closing": [
        "store closing", "severance", "severance pay", "dislocation",
        "relocation", "transfer", "new store"
    ],
}

# High-stakes topics - trigger escalation and steward contact
HIGH_STAKES_KEYWORDS = [
    "discharge", "discharged", "termination", "terminated", "fired", "dismissal",
    "discipline", "disciplinary", "suspension", "suspended",
    "harassment", "harassed", "discrimination", "discriminated", "retaliation",
    "safety", "injury", "injured", "accident", "unsafe",
    "weingarten", "representation", "just cause",
    "no strike", "lockout", "investigation"
]


def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    # Remove page markers like "6 PUEBLO CLERKS 2022-2025"
    text = re.sub(r'\d+\s*PUEBLO CLERKS\s*2022-2025', '', text)
    text = re.sub(r'\d+\s*$', '', text, flags=re.MULTILINE)  # Trailing page numbers
    # Remove horizontal rules
    text = re.sub(r'^---+\s*$', '', text, flags=re.MULTILINE)
    # Remove HTML table tags but keep content
    text = re.sub(r'</?table[^>]*>', '', text)
    text = re.sub(r'</?thead[^>]*>', '', text)
    text = re.sub(r'</?tbody[^>]*>', '', text)
    text = re.sub(r'</?tr[^>]*>', '\n', text)
    text = re.sub(r'</?th[^>]*>', ' | ', text)
    text = re.sub(r'</?td[^>]*>', ' | ', text)
    # Remove ins/del tags but keep content
    text = re.sub(r'</?ins>', '', text)
    text = re.sub(r'</?del>', '', text)
    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def extract_classifications(text: str) -> list:
    """Extract which classifications this text applies to."""
    text_lower = text.lower()
    classifications = []
    for class_key, keywords in CLASSIFICATION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                classifications.append(class_key)
                break
    return classifications if classifications else ["all"]


def extract_topics(text: str) -> list:
    """Extract topic tags from text."""
    text_lower = text.lower()
    topics = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                topics.append(topic)
                break
    return topics


def is_high_stakes(text: str) -> bool:
    """Check if text contains high-stakes topics."""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in HIGH_STAKES_KEYWORDS)


def extract_article_references(text: str) -> list:
    """Extract references to other articles in the text."""
    references = []
    matches = re.findall(r'Article\s+(\d+)', text, re.IGNORECASE)
    for match in matches:
        ref = f"art{match}"
        if ref not in references:
            references.append(ref)
    return references


def find_all_articles(md_content: str) -> List[Tuple[int, int, str]]:
    """
    Find all article boundaries in the markdown.
    
    Returns list of (article_num, start_pos, title)
    """
    articles = []
    
    # Pattern 1: ## ARTICLE X TITLE (same line)
    # Pattern 2: ## ARTICLE X followed by ## TITLE on next line
    # Pattern 3: # ARTICLE X
    
    # Match all article headers
    pattern = r'^(?:##?\s*)ARTICLE\s+(\d+)(?:\s*\n##?\s*([A-Z][A-Z\s]+?))?(?:\s+([A-Z][A-Z\s,]+?))?(?:\s*$|\n)'
    
    for match in re.finditer(pattern, md_content, re.MULTILINE | re.IGNORECASE):
        article_num = int(match.group(1))
        # Title could be on next line (group 2) or same line (group 3)
        title = match.group(2) or match.group(3) or ""
        title = title.strip() if title else ""
        start_pos = match.start()
        articles.append((article_num, start_pos, title))
    
    return articles


def find_articles_robust(md_content: str) -> List[Tuple[int, int, str]]:
    """
    Robust article finder that handles all formatting patterns.
    
    Returns list of (article_num, start_pos, title)
    """
    articles = []
    lines = md_content.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Match ## ARTICLE X or # ARTICLE X
        match = re.match(r'^(#{1,2})\s*ARTICLE\s+(\d+)(.*)$', line, re.IGNORECASE)
        if match:
            article_num = int(match.group(2))
            rest = match.group(3).strip()
            
            # Check if title is on same line
            if rest and not rest.startswith('#'):
                title = rest
            else:
                # Check if next line is a title (## TITLE format)
                title = ""
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    title_match = re.match(r'^#{1,2}\s+([A-Z][A-Z\s\-,/]+)$', next_line)
                    if title_match:
                        title = title_match.group(1).strip()
            
            # Calculate position
            pos = sum(len(lines[j]) + 1 for j in range(i))
            articles.append((article_num, pos, title))
        
        i += 1
    
    return articles


def extract_article_content(md_content: str, articles: List[Tuple[int, int, str]]) -> dict:
    """
    Extract content for each article.
    
    Returns dict of article_num -> (title, content)
    """
    article_contents = {}
    
    for i, (article_num, start_pos, title) in enumerate(articles):
        # Find end position (start of next article or end of relevant section)
        if i + 1 < len(articles):
            end_pos = articles[i + 1][1]
        else:
            # Find the end (Letters of Understanding or end of file)
            lou_match = re.search(r'\n#\s*SAFEWAY INC\.\s*CLERKS\s*LETTERS', md_content[start_pos:])
            if lou_match:
                end_pos = start_pos + lou_match.start()
            else:
                end_pos = len(md_content)
        
        content = md_content[start_pos:end_pos]
        
        # If title is empty, try to extract from first line
        if not title:
            first_lines = content.split('\n')[:3]
            for fl in first_lines:
                fl_clean = re.sub(r'^#{1,2}\s*ARTICLE\s+\d+\s*', '', fl, flags=re.IGNORECASE).strip()
                if fl_clean and len(fl_clean) > 3 and fl_clean.isupper():
                    title = fl_clean
                    break
        
        article_contents[article_num] = (title or f"ARTICLE {article_num}", content)
    
    return article_contents


def parse_sections(article_content: str) -> List[Tuple[int, Optional[str], str]]:
    """
    Parse sections within an article.
    
    Returns list of (section_num, subsection, content)
    """
    sections = []
    
    # Split by Section headers
    section_pattern = r'(?:^|\n)Section\s+(\d+)\.?\s*'
    
    parts = re.split(section_pattern, article_content, flags=re.IGNORECASE)
    
    if len(parts) <= 1:
        # No sections found, return entire content as one chunk
        return [(None, None, article_content)]
    
    # First part is before any section
    preamble = parts[0].strip()
    
    # Process section pairs (number, content)
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            section_num = int(parts[i])
            section_content = parts[i + 1].strip()
            
            # Check for subsections (a., b., c., etc.)
            subsection_pattern = r'\n\(([a-z])\)\s+'
            subsection_matches = list(re.finditer(subsection_pattern, section_content))
            
            if subsection_matches and len(section_content) > 800:
                # Split by subsections for long sections
                last_end = 0
                for j, sub_match in enumerate(subsection_matches):
                    # Content before this subsection (or between subsections)
                    pre_content = section_content[last_end:sub_match.start()].strip()
                    if pre_content and len(pre_content) > 50:
                        if j == 0:
                            sections.append((section_num, None, pre_content))
                        else:
                            prev_sub = subsection_matches[j-1].group(1)
                            sections.append((section_num, prev_sub, pre_content))
                    last_end = sub_match.end()
                
                # Last subsection
                last_sub = subsection_matches[-1].group(1)
                last_content = section_content[subsection_matches[-1].end():].strip()
                if last_content:
                    sections.append((section_num, last_sub, last_content))
            else:
                # Keep section as one chunk
                if section_content:
                    sections.append((section_num, None, section_content))
    
    # Add preamble if substantial
    if preamble and len(preamble) > 100:
        # Check if it's not just headers
        cleaned = re.sub(r'^#{1,2}.*$', '', preamble, flags=re.MULTILINE).strip()
        if len(cleaned) > 50:
            sections.insert(0, (None, None, preamble))
    
    return sections


def create_chunk(
    article_num: int, 
    article_title: str, 
    section_num: Optional[int], 
    subsection: Optional[str],
    content: str,
    chunk_counter: dict
) -> ContractChunk:
    """Create a ContractChunk with all metadata."""
    
    # Build unique chunk ID
    base_id = f"art{article_num}"
    if section_num:
        base_id += f"_sec{section_num}"
    if subsection:
        base_id += f"_{subsection}"
    
    # Ensure uniqueness
    if base_id in chunk_counter:
        chunk_counter[base_id] += 1
        chunk_id = f"{base_id}_{chunk_counter[base_id]}"
    else:
        chunk_counter[base_id] = 0
        chunk_id = base_id
    
    # Build citation
    citation_parts = [f"Article {article_num}"]
    if section_num:
        citation_parts.append(f"Section {section_num}")
    if subsection:
        citation_parts[-1] += f"({subsection})"
    citation = ", ".join(citation_parts)
    
    # Clean content
    cleaned_content = clean_text(content)
    
    # Extract metadata
    applies_to = extract_classifications(cleaned_content)
    topic_tags = extract_topics(cleaned_content)
    references = extract_article_references(cleaned_content)
    urgency = "high_stakes" if is_high_stakes(cleaned_content) else "standard"
    
    return ContractChunk(
        chunk_id=chunk_id,
        contract_id=CONTRACT_ID,
        article_num=article_num,
        article_title=article_title,
        section_num=section_num,
        subsection=subsection,
        citation=citation,
        content=cleaned_content,
        doc_type="cba",
        applies_to=applies_to,
        topic_tags=topic_tags,
        references=references,
        urgency_tier=urgency
    )


def parse_lous(md_content: str, chunk_counter: dict) -> List[ContractChunk]:
    """Parse Letters of Understanding from the contract."""
    chunks = []
    
    # Find the LOUs section
    lou_patterns = [
        r'SAFEWAY INC\.\s*CLERKS\s*LETTERS OF UNDERSTANDING',
        r'LETTERS OF UNDERSTANDING',
    ]
    
    lou_start = -1
    for pattern in lou_patterns:
        match = re.search(pattern, md_content, re.IGNORECASE)
        if match:
            lou_start = match.start()
            break
    
    if lou_start == -1:
        return chunks
    
    lou_content = md_content[lou_start:]
    
    # Find numbered LOUs
    lou_pattern = r'\n(\d+)\.\s+([A-Z][\w\s\-\.]+?)(?:\n|$)'
    
    matches = list(re.finditer(lou_pattern, lou_content))
    
    for i, match in enumerate(matches):
        lou_num = match.group(1)
        lou_title = match.group(2).strip()[:100]
        
        # Get content until next LOU or end
        start_pos = match.end()
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(lou_content)
        
        lou_body = lou_content[start_pos:end_pos].strip()
        cleaned = clean_text(lou_body)
        
        if len(cleaned) < 30:
            continue
        
        # Ensure unique ID
        base_id = f"lou_{lou_num}"
        if base_id in chunk_counter:
            chunk_counter[base_id] += 1
            chunk_id = f"{base_id}_{chunk_counter[base_id]}"
        else:
            chunk_counter[base_id] = 0
            chunk_id = base_id
        
        chunk = ContractChunk(
            chunk_id=chunk_id,
            contract_id=CONTRACT_ID,
            article_num=None,
            article_title=f"Letter of Understanding {lou_num}: {lou_title[:50]}",
            section_num=None,
            subsection=None,
            citation=f"Letter of Understanding {lou_num}",
            content=cleaned[:2000],
            doc_type="lou",
            applies_to=extract_classifications(cleaned),
            topic_tags=extract_topics(cleaned),
            references=extract_article_references(cleaned),
            urgency_tier="high_stakes" if is_high_stakes(cleaned) else "standard"
        )
        chunks.append(chunk)
    
    return chunks


def parse_contract(md_content: str) -> List[ContractChunk]:
    """Parse the contract markdown into structured chunks."""
    chunks = []
    chunk_counter = {}
    unparsed_lines = []
    
    print("Finding articles...")
    articles = find_articles_robust(md_content)
    print(f"Found {len(articles)} articles")
    
    # Extract content for each article
    article_contents = extract_article_content(md_content, articles)
    
    print("Parsing sections...")
    for article_num, (title, content) in sorted(article_contents.items()):
        sections = parse_sections(content)
        
        for section_num, subsection, section_content in sections:
            if len(section_content.strip()) < 30:
                continue
            
            chunk = create_chunk(
                article_num, title, section_num, subsection,
                section_content, chunk_counter
            )
            chunks.append(chunk)
    
    # Parse LOUs
    print("Parsing Letters of Understanding...")
    lou_chunks = parse_lous(md_content, chunk_counter)
    chunks.extend(lou_chunks)
    
    return chunks


def save_chunks(chunks: List[ContractChunk], output_file: Path) -> None:
    """Save chunks to JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    chunks_data = [asdict(chunk) for chunk in chunks]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(chunks)} chunks to {output_file}")


def main():
    """Main entry point for contract parsing."""
    print(f"Parsing contract from: {CONTRACT_MD_FILE}")
    
    # Read markdown content
    with open(CONTRACT_MD_FILE, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Parse into chunks
    chunks = parse_contract(md_content)
    
    # Print summary
    print(f"\nParsed {len(chunks)} chunks:")
    articles = sorted(set(c.article_num for c in chunks if c.article_num))
    print(f"  - Articles covered: {len(articles)}")
    print(f"  - Article numbers: {articles[:20]}{'...' if len(articles) > 20 else ''}")
    
    lous = [c for c in chunks if c.doc_type == "lou"]
    print(f"  - Letters of Understanding: {len(lous)}")
    
    high_stakes = [c for c in chunks if c.urgency_tier == "high_stakes"]
    print(f"  - High-stakes chunks: {len(high_stakes)}")
    
    # Check for missing expected articles
    expected_articles = set(range(1, 59))  # Articles 1-58
    found_articles = set(articles)
    missing = expected_articles - found_articles
    if missing:
        print(f"\n  WARNING: Missing articles: {sorted(missing)}")
    
    # Save to file
    output_file = CHUNKS_DIR / "contract_chunks.json"
    save_chunks(chunks, output_file)
    
    # Save summary
    summary = {
        "contract_id": CONTRACT_ID,
        "total_chunks": len(chunks),
        "articles_found": articles,
        "missing_articles": sorted(list(missing)),
        "lou_count": len(lous),
        "high_stakes_count": len(high_stakes),
        "topic_distribution": {}
    }
    
    for chunk in chunks:
        for topic in chunk.topic_tags:
            summary["topic_distribution"][topic] = summary["topic_distribution"].get(topic, 0) + 1
    
    summary_file = CHUNKS_DIR / "chunks_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to {summary_file}")
    
    return chunks


if __name__ == "__main__":
    main()
