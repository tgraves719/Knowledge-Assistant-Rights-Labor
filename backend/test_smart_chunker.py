"""
Regression checks for SmartChunker section parsing.

Validates section-header variants used in real contract exports:
- <u>Section 48</u>.
- <u>Section 49.</u>
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.ingest.smart_chunker import SmartChunker


def _build_fixture() -> str:
    return """
## ARTICLE 17
## VACATIONS

<u>Section 48</u>. All regular full-time employees shall receive one (1) week's paid vacation after one (1) year's service.

<u>Section 49.</u> Effective January 1, 1991, the Employer will convert the employees' weekly vacation allotment to a daily vacation allotment.
""".strip()


def main() -> None:
    chunker = SmartChunker(contract_id="test_contract")
    chunks = chunker.parse_content(_build_fixture())

    sec48 = [c for c in chunks if c.article_num == 17 and c.section_num == 48]
    sec49 = [c for c in chunks if c.article_num == 17 and c.section_num == 49]

    assert sec48, "Expected Section 48 chunk to be parsed from <u>Section 48</u>."
    assert sec49, "Expected Section 49 chunk to be parsed from <u>Section 49.</u>"
    assert any("one (1) week's paid vacation" in (c.content or "") for c in sec48), (
        "Section 48 content missing expected vacation accrual sentence"
    )

    # Regression: no legacy "Part partN" citation format.
    assert not any("Part part" in (c.citation or "") for c in chunks), (
        "Found legacy citation format 'Part partN'"
    )

    # Regression: no legacy segment markers stored in subsection field.
    assert not any(
        str(c.subsection or "").lower().startswith("part")
        for c in chunks
    ), "Found legacy segment token in subsection field"

    # Force paragraph splitting to validate normalized segment labels.
    chunker2 = SmartChunker(contract_id="test_contract")
    chunker2.MAX_CHUNK_SIZE = 180
    chunker2.TARGET_CHUNK_SIZE = 90
    split_fixture = """
## ARTICLE 99
## TEST ARTICLE

Section 1. TEST.
Paragraph one with enough text to force chunk splitting when thresholds are low.

Paragraph two with enough text to force chunk splitting when thresholds are low.

Paragraph three with enough text to force chunk splitting when thresholds are low.
""".strip()
    split_chunks = chunker2.parse_content(split_fixture)
    split_citations = [c.citation for c in split_chunks if c.article_num == 99 and c.section_num == 1]
    assert any(", Part 1" in c for c in split_citations), "Expected normalized 'Part 1' citation"
    assert not any("Part part" in c for c in split_citations), "Split citation used legacy part token"

    print("[OK] SmartChunker section parser handles underlined section header variants")


if __name__ == "__main__":
    main()
