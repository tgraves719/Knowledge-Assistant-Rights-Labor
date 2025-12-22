"""
LLM Metadata Enricher for Contract Chunks.

Processes each chunk through an LLM to extract:
- applies_to: Job classifications this affects
- topics: Topic tags from taxonomy
- cross_references: References to other articles/sections
- summary: One-sentence description
- flags: is_definition, is_exception, hire_date_sensitive, is_high_stakes
"""

import json
import os
import time
import asyncio
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from backend.ingest.schema import (
    EnrichedChunk, 
    TOPICS, 
    CLASSIFICATIONS,
    validate_chunk
)
from backend.config import GEMINI_API_KEY, CHUNKS_DIR


# =============================================================================
# ENRICHMENT PROMPT
# =============================================================================

ENRICHMENT_PROMPT = """You are analyzing a union contract provision for a RAG search system.

Contract: Safeway Pueblo Clerks 2022-2025 (UFCW Local 7)
Citation: {citation}
Parent Context: {parent_context}

Content:
{content}

---

Analyze this provision and respond with a JSON object containing:

1. "applies_to": Which job classifications does this SPECIFICALLY mention or apply to?
   - Choose from: {classifications}
   - Use "all" ONLY if it genuinely applies to all employees with no exceptions
   - Be specific - if it mentions "Courtesy Clerks", include "courtesy_clerk"
   - If it mentions DUG/Drive Up & Go, include "dug_shopper"

2. "topics": What topics does this cover?
   - Choose from: {topics}
   - Can select multiple
   - Be comprehensive - a vacation scheduling section might have both "vacation" and "scheduling"

3. "cross_references": Does this explicitly reference other Articles, Sections, or Appendices?
   - Format as list: ["art40_sec116", "appendix_a"]
   - Only include EXPLICIT references in the text
   - If none, use empty list []

4. "summary": One sentence (max 100 chars) describing what this provision does
   - Be specific and actionable
   - Example: "Defines night premium as $2/hr for hours between midnight and 6am"

5. "is_definition": Does this define a term, role, or classification? (true/false)
   - True for sections that define what something IS

6. "is_exception": Does this contain override language like "except", "notwithstanding", "shall not apply", "unless"? (true/false)

7. "hire_date_sensitive": Are there different rules for employees hired before/after a specific date (usually March 27, 2005)? (true/false)

8. "is_high_stakes": Does this involve discipline, termination, harassment, discrimination, safety hazards, or grievance procedures? (true/false)

Respond with ONLY valid JSON, no markdown or explanation."""


# =============================================================================
# LLM CLIENT
# =============================================================================

class GeminiEnricher:
    """Enricher using Google Gemini API."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
        self.model = None
        self._init_client()
    
    def _init_client(self):
        """Initialize Gemini client."""
        if not self.api_key:
            raise ValueError("No Gemini API key provided")
        
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash-lite")
    
    def enrich_chunk(self, chunk: dict, max_retries: int = 3) -> dict:
        """
        Enrich a single chunk with LLM-generated metadata.
        
        Returns the original chunk with added metadata fields.
        """
        prompt = ENRICHMENT_PROMPT.format(
            citation=chunk.get("citation", "Unknown"),
            parent_context=chunk.get("parent_context", ""),
            content=chunk.get("content", "")[:3000],  # Limit content length
            classifications=", ".join(CLASSIFICATIONS),
            topics=", ".join(TOPICS),
        )
        
        # Retry with exponential backoff for rate limits
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                result_text = response.text.strip()
                break
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    wait_time = (2 ** attempt) * 2  # 2, 4, 8 seconds
                    print(f"Rate limited, waiting {wait_time}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait_time)
                    if attempt == max_retries - 1:
                        raise
                else:
                    raise
        
        try:
            
            # Clean up response - remove markdown code blocks if present
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            result_text = result_text.strip()
            
            # Parse JSON response
            enrichment = json.loads(result_text)
            
            # Validate and clean up values
            enriched = self._apply_enrichment(chunk, enrichment)
            return enriched
            
        except json.JSONDecodeError as e:
            print(f"  JSON parse error for {chunk.get('citation')}: {e}")
            print(f"  Raw response: {result_text[:200]}...")
            return self._default_enrichment(chunk)
        except Exception as e:
            print(f"  LLM error for {chunk.get('citation')}: {e}")
            return self._default_enrichment(chunk)
    
    def _apply_enrichment(self, chunk: dict, enrichment: dict) -> dict:
        """Apply enrichment data to chunk with validation."""
        result = chunk.copy()
        
        # applies_to - validate against taxonomy
        applies_to = enrichment.get("applies_to", ["all"])
        if isinstance(applies_to, str):
            applies_to = [applies_to]
        result["applies_to"] = [c for c in applies_to if c in CLASSIFICATIONS] or ["all"]
        
        # topics - validate against taxonomy
        topics = enrichment.get("topics", [])
        if isinstance(topics, str):
            topics = [topics]
        result["topics"] = [t for t in topics if t in TOPICS]
        
        # cross_references
        cross_refs = enrichment.get("cross_references", [])
        if isinstance(cross_refs, str):
            cross_refs = [cross_refs] if cross_refs else []
        result["cross_references"] = cross_refs
        
        # summary
        summary = enrichment.get("summary", "")
        result["summary"] = summary[:150] if summary else None
        
        # flags
        result["is_definition"] = bool(enrichment.get("is_definition", False))
        result["is_exception"] = bool(enrichment.get("is_exception", False))
        result["hire_date_sensitive"] = bool(enrichment.get("hire_date_sensitive", False))
        result["is_high_stakes"] = bool(enrichment.get("is_high_stakes", False))
        
        return result
    
    def _default_enrichment(self, chunk: dict) -> dict:
        """Return chunk with default metadata when LLM fails."""
        result = chunk.copy()
        result["applies_to"] = ["all"]
        result["topics"] = []
        result["cross_references"] = []
        result["summary"] = None
        result["is_definition"] = False
        result["is_exception"] = False
        result["hire_date_sensitive"] = False
        result["is_high_stakes"] = False
        return result


# =============================================================================
# BATCH ENRICHMENT
# =============================================================================

def enrich_chunks(
    input_path: Path,
    output_path: Path,
    batch_size: int = 10,
    delay_between_batches: float = 2.0,
    start_from: int = 0,
    limit: int = None,
) -> dict:
    """
    Enrich all chunks in a file.
    
    Args:
        input_path: Path to input chunks JSON
        output_path: Path to save enriched chunks
        batch_size: Number of chunks per batch (for rate limiting)
        delay_between_batches: Seconds to wait between batches
        start_from: Index to start from (for resuming)
        limit: Maximum chunks to process (for testing)
    
    Returns:
        Summary dict with counts
    """
    print(f"Loading chunks from {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    total = len(chunks)
    if limit:
        chunks = chunks[start_from:start_from + limit]
    else:
        chunks = chunks[start_from:]
    
    print(f"Processing {len(chunks)} chunks (starting from {start_from})")
    
    # Initialize enricher
    enricher = GeminiEnricher()
    
    # Load existing enriched chunks if resuming
    enriched_chunks = []
    if start_from > 0 and output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            enriched_chunks = json.load(f)
        print(f"Loaded {len(enriched_chunks)} previously enriched chunks")
    
    # Process in batches
    stats = {"success": 0, "errors": 0, "total": len(chunks)}
    
    for i, chunk in enumerate(chunks):
        chunk_num = start_from + i + 1
        print(f"[{chunk_num}/{total}] {chunk.get('citation', 'Unknown')}...", end=" ")
        
        try:
            enriched = enricher.enrich_chunk(chunk)
            enriched_chunks.append(enriched)
            
            # Show key metadata
            topics = enriched.get("topics", [])
            applies = enriched.get("applies_to", [])
            print(f"OK (topics: {topics[:2]}, applies: {applies[:2]})")
            
            stats["success"] += 1
            
        except Exception as e:
            print(f"ERROR: {e}")
            enriched_chunks.append(enricher._default_enrichment(chunk))
            stats["errors"] += 1
        
        # Rate limiting
        if (i + 1) % batch_size == 0:
            print(f"  [Batch complete, waiting {delay_between_batches}s...]")
            time.sleep(delay_between_batches)
            
            # Save progress
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(enriched_chunks, f, indent=2, ensure_ascii=False)
    
    # Final save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enriched_chunks, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Enrichment complete!")
    print(f"  Success: {stats['success']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  Saved to: {output_path}")
    
    return stats


def validate_enriched_chunks(path: Path) -> dict:
    """Validate enriched chunks against schema."""
    from backend.ingest.schema import validate_chunks
    
    with open(path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    return validate_chunks(chunks)


# =============================================================================
# CLI
# =============================================================================

def main():
    """Run enrichment on smart-chunked contract."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enrich contract chunks with LLM")
    parser.add_argument("--input", type=str, default="data/chunks/contract_chunks_smart.json")
    parser.add_argument("--output", type=str, default="data/chunks/contract_chunks_enriched.json")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--start-from", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if args.validate_only:
        print(f"Validating {output_path}...")
        results = validate_enriched_chunks(output_path)
        print(f"Valid: {results['valid']}/{results['total']}")
        if results['errors']:
            print("Errors:")
            for err in results['errors'][:5]:
                print(f"  {err['citation']}: {err['errors']}")
        return
    
    # Run enrichment
    stats = enrich_chunks(
        input_path=input_path,
        output_path=output_path,
        batch_size=args.batch_size,
        delay_between_batches=args.delay,
        start_from=args.start_from,
        limit=args.limit,
    )
    
    # Validate results
    print("\nValidating enriched chunks...")
    validation = validate_enriched_chunks(output_path)
    print(f"Schema validation: {validation['valid']}/{validation['total']} valid")


if __name__ == "__main__":
    main()

