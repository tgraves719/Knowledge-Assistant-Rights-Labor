"""
Deterministic topic-routing smoke checks.

These checks run BM25-only to avoid embedding-model/network dependencies.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import backend.retrieval.router as router_module
from backend.retrieval.router import HybridRetriever, classify_intent


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
    query = "How much vacation do I get per year?"

    intent, citations = _run_bm25_retrieval(query=query, contract_id=contract_id, n_results=10)
    joined = " | ".join(citations)

    assert intent.topic == "vacation", f"Expected topic 'vacation', got: {intent.topic}"
    assert 17 in set(intent.relevant_articles), (
        f"Expected Article 17 in relevant articles, got: {intent.relevant_articles}"
    )
    assert any(c.startswith("Article 17, Section 42") for c in citations), (
        f"Expected vacation entitlement Section 42 in top results. Got: {joined}"
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
    finally:
        router_module.HYBRID_VECTOR_WEIGHT = original_vector
        router_module.HYBRID_KEYWORD_WEIGHT = original_keyword
        router_module.CAG_ENABLE_RERANKER = original_reranker

    print("[OK] Topic routing checks passed")


if __name__ == "__main__":
    main()
