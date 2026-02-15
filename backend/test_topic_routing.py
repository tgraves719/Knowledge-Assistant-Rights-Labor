"""
Deterministic topic-routing smoke checks.

These checks run BM25-only to avoid embedding-model/network dependencies.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import backend.retrieval.router as router_module
from backend.retrieval.router import HybridRetriever, classify_intent, extract_classifications_for_contract


def _run_bm25_retrieval(query: str, contract_id: str, n_results: int = 8) -> tuple[object, list[str]]:
    intent = classify_intent(query, contract_id=contract_id)

    retriever = HybridRetriever(vector_store=None)
    result = retriever.retrieve(
        query=query,
        intent=intent,
        n_results=n_results,
        use_hybrid=True,
        contract_id=contract_id,
    )
    citations = [str(c.get("citation") or "") for c in result.get("chunks", [])[:n_results]]
    return intent, citations


def _test_float_days_maps_to_personal_holiday_article() -> None:
    contract_id = "local7_safeway_pueblo_clerks_2022"
    query = "float days"

    intent, citations = _run_bm25_retrieval(query=query, contract_id=contract_id, n_results=8)
    joined = " | ".join(citations)

    assert intent.topic == "personal_holiday", f"Expected personal_holiday topic, got: {intent.topic}"
    assert 16 in set(intent.relevant_articles), (
        f"Expected Article 16 in relevant articles, got: {intent.relevant_articles}"
    )
    assert any(c.startswith("Article 16") for c in citations), (
        f"Expected Article 16 in top results for 'float days'. Got: {joined}"
    )
    assert any("Section 38" in c for c in citations), (
        f"Expected Section 38 personal-holiday content in top results. Got: {joined}"
    )


def _test_contract_term_formal_rewrites_map_to_term_article() -> None:
    contract_id = "local7_safeway_pueblo_clerks_2022"
    queries = [
        "When does this union contract start and end?",
        "How long is the current CBA effective for, and what are the start and expiration dates?",
    ]

    for query in queries:
        intent, citations = _run_bm25_retrieval(query=query, contract_id=contract_id, n_results=8)
        joined = " | ".join(citations)

        assert intent.topic == "term", f"Expected topic 'term' for '{query}', got: {intent.topic}"
        assert 58 in set(intent.relevant_articles), (
            f"Expected Article 58 in relevant articles for '{query}'. Got: {intent.relevant_articles}"
        )
        assert any(c.startswith("Article 58") for c in citations), (
            f"Expected Article 58 in top results for '{query}'. Got: {joined}"
        )


def _test_inter_shift_rest_formal_rewrite_maps_to_break_articles() -> None:
    contract_id = "local7_safeway_pueblo_clerks_2022"
    query = (
        "What are the contractual requirements for minimum hours between the end "
        "of one shift and the start of the next?"
    )

    intent, citations = _run_bm25_retrieval(query=query, contract_id=contract_id, n_results=8)
    joined = " | ".join(citations)

    assert intent.topic == "breaks", f"Expected topic 'breaks', got: {intent.topic}"
    articles = set(intent.relevant_articles)
    assert 24 in articles and 25 in articles, (
        f"Expected Articles 24 and 25 in relevant articles, got: {intent.relevant_articles}"
    )
    assert any(c.startswith("Article 24") for c in citations), (
        f"Expected Article 24 in top results. Got: {joined}"
    )
    assert any(c.startswith("Article 25") for c in citations), (
        f"Expected Article 25 in top results. Got: {joined}"
    )


def _test_vacation_entitlement_query_surfaces_section_42() -> None:
    contract_id = "local7_kingsoopers_loveland_meat_2019"
    queries = [
        "How much vacation do I get per year?",
        "Please provide the annual paid vacation entitlement schedule based on years of continuous service.",
        "At 33 months of service, what weekly vacation entitlement applies under Article 17?",
    ]

    for query in queries:
        intent, citations = _run_bm25_retrieval(query=query, contract_id=contract_id, n_results=10)
        joined = " | ".join(citations)

        assert intent.topic == "vacation", f"Expected topic 'vacation' for '{query}', got: {intent.topic}"
        assert 17 in set(intent.relevant_articles), (
            f"Expected Article 17 in relevant articles for '{query}', got: {intent.relevant_articles}"
        )
        assert any(c.startswith("Article 17, Section 42") for c in citations), (
            f"Expected vacation entitlement Section 42 in top results for '{query}'. Got: {joined}"
        )


def _test_holiday_work_premium_query_maps_to_premium_article() -> None:
    contract_id = "local7_kingsoopers_loveland_meat_2019"
    query = (
        "For a post-2005 hire who works a contractual holiday, "
        "what premium applies to hours worked on that holiday?"
    )

    intent, citations = _run_bm25_retrieval(query=query, contract_id=contract_id, n_results=10)
    joined = " | ".join(citations)

    assert intent.topic == "premiums", f"Expected topic 'premiums', got: {intent.topic}"
    assert any(c.startswith("Article 16, Section 40") for c in citations), (
        f"Expected holiday-work premium Section 40 in top results. Got: {joined}"
    )


def _test_role_comparison_query_maps_multiple_classifications() -> None:
    contract_id = "local7_safeway_pueblo_clerks_2022"
    query = "what is the difference between cc's and dug workers"

    classes = extract_classifications_for_contract(query, contract_id=contract_id, max_matches=4)
    intent = classify_intent(query, contract_id=contract_id)

    assert "courtesy_clerk" in classes, f"Expected courtesy_clerk mention, got: {classes}"
    assert any(c in classes for c in ("drive_up_and_go", "dug_shopper", "all_purpose_clerk")), (
        f"Expected DUG-related classification mention, got: {classes}"
    )
    assert intent.relevant_articles, (
        "Expected non-empty relevant_articles for role-comparison query "
        f"with detected classes: {classes}"
    )
    assert 4 in set(intent.relevant_articles), (
        f"Expected role-comparison anchors to include article 4. Got: {intent.relevant_articles}"
    )
    assert intent.topic in (None, ""), f"Expected role comparison topic to route via entities, got: {intent.topic}"
    assert intent.comparison_mode is True, (
        f"Expected comparison_mode=True for comparison query; got {intent.comparison_mode}"
    )
    assert "classification_definition" in set(intent.required_evidence_slots), (
        "Expected comparison query to require classification_definition slot. "
        f"Got: {intent.required_evidence_slots}"
    )


def main() -> None:
    original_vector = router_module.HYBRID_VECTOR_WEIGHT
    original_keyword = router_module.HYBRID_KEYWORD_WEIGHT
    original_reranker = router_module.CAG_ENABLE_RERANKER
    try:
        router_module.HYBRID_VECTOR_WEIGHT = 0.0
        router_module.HYBRID_KEYWORD_WEIGHT = 1.0
        router_module.CAG_ENABLE_RERANKER = False

        _test_float_days_maps_to_personal_holiday_article()
        _test_contract_term_formal_rewrites_map_to_term_article()
        _test_inter_shift_rest_formal_rewrite_maps_to_break_articles()
        _test_vacation_entitlement_query_surfaces_section_42()
        _test_holiday_work_premium_query_maps_to_premium_article()
        _test_role_comparison_query_maps_multiple_classifications()
    finally:
        router_module.HYBRID_VECTOR_WEIGHT = original_vector
        router_module.HYBRID_KEYWORD_WEIGHT = original_keyword
        router_module.CAG_ENABLE_RERANKER = original_reranker

    print("[OK] Topic routing checks passed")


if __name__ == "__main__":
    main()
