"""
Deterministic topic-routing smoke checks.

These checks run BM25-only to avoid embedding-model/network dependencies.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import backend.retrieval.router as router_module
from backend.retrieval.router import (
    HybridRetriever,
    build_followup_routing_plan,
    classify_intent,
    extract_classifications_for_contract,
    infer_side_letter_articles,
)


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


def _test_post_2005_holiday_work_premium_query_prioritizes_section_40() -> None:
    contract_id = "local7_kingsoopers_loveland_meat_2019"
    query = (
        "For a post-2005 hire who works a contractual holiday, "
        "what premium applies to hours worked on that holiday?"
    )

    intent, citations = _run_bm25_retrieval(query=query, contract_id=contract_id, n_results=8)
    joined = " | ".join(citations)

    assert intent.topic == "premiums", f"Expected topic 'premiums', got: {intent.topic}"
    assert any(c.startswith("Article 16, Section 40") for c in citations[:8]), (
        f"Expected Article 16, Section 40 in top-8 results. Got: {joined}"
    )


def _test_overtime_definition_query_prefers_article_12() -> None:
    contract_id = "local7_safeway_pueblo_meat_2022"
    query = "Where are overtime rules defined in this agreement?"

    intent, citations = _run_bm25_retrieval(query=query, contract_id=contract_id, n_results=8)
    joined = " | ".join(citations)

    assert intent.topic == "overtime", f"Expected topic 'overtime', got: {intent.topic}"
    assert any(c.startswith("Article 12") for c in citations[:3]), (
        f"Expected Article 12 overtime section in top-3 results. Got: {joined}"
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


def _test_role_targeted_wage_followup_prefers_explicit_role() -> None:
    contract_id = "local7_safeway_pueblo_clerks_2022"
    profile_classification = "nonfood_gm_floral"
    retriever = HybridRetriever(vector_store=None)

    followup_query = "you dont know what the pay is for cc's"
    intent = classify_intent(
        followup_query,
        user_classification=profile_classification,
        contract_id=contract_id,
    )
    assert intent.intent_type == "wage", (
        f"Expected wage intent for role-targeted follow-up, got: {intent.intent_type}"
    )
    assert intent.classification == "courtesy_clerk", (
        f"Expected explicit CC role targeting, got: {intent.classification}"
    )

    result = retriever.retrieve(
        query=followup_query,
        intent=intent,
        n_results=8,
        use_hybrid=True,
        contract_id=contract_id,
    )
    wage_info = result.get("wage_info") or {}
    assert wage_info, "Expected deterministic wage lookup for explicit CC follow-up query."
    assert wage_info.get("classification_key") == "courtesy_clerk", (
        f"Expected courtesy_clerk wage lookup key, got: {wage_info.get('classification_key')}"
    )

    formal_query = "what does a courtesy clerk make"
    formal_intent = classify_intent(
        formal_query,
        user_classification=profile_classification,
        contract_id=contract_id,
    )
    assert formal_intent.intent_type == "wage", (
        f"Expected wage intent for formal role-targeted phrasing, got: {formal_intent.intent_type}"
    )
    assert formal_intent.classification == "courtesy_clerk", (
        f"Expected courtesy_clerk classification for formal role-targeted phrasing, got: {formal_intent.classification}"
    )

    self_query = "what is my pay"
    self_intent = classify_intent(
        self_query,
        user_classification=profile_classification,
        contract_id=contract_id,
    )
    assert self_intent.intent_type == "wage", (
        f"Expected wage intent for self-pay query, got: {self_intent.intent_type}"
    )
    assert self_intent.classification == profile_classification, (
        "Expected profile classification to remain active when no explicit target role "
        f"is mentioned. Got: {self_intent.classification}"
    )


def _test_premium_rate_legal_text_not_routed_as_wage() -> None:
    contract_id = "local7_safeway_pueblo_clerks_2022"
    query = (
        "The Sunday premium shall not be averaged into the employee's straight-time rate "
        "for the purpose of determining the rate upon which daily or weekly overtime is based."
    )

    intent = classify_intent(query, contract_id=contract_id)
    assert intent.intent_type == "contract", (
        f"Expected contract intent for legal premium/overtime language, got: {intent.intent_type}"
    )

    retriever = HybridRetriever(vector_store=None)
    result = retriever.retrieve(
        query=query,
        intent=intent,
        n_results=8,
        use_hybrid=True,
        contract_id=contract_id,
    )
    assert result.get("wage_info") is None, (
        "Expected no wage_info for premium/overtime legal-language query."
    )


def _test_steward_union_meeting_schedule_routes_article_45() -> None:
    contract_id = "local7_safeway_pueblo_clerks_2022"
    query = "Can appointed stewards be scheduled later than 6:00 p.m. on regular union meeting nights?"

    intent, citations = _run_bm25_retrieval(query=query, contract_id=contract_id, n_results=10)
    joined = " | ".join(citations)

    assert intent.topic == "scheduling", f"Expected scheduling topic, got: {intent.topic}"
    assert 45 in set(intent.relevant_articles), (
        f"Expected Article 45 in relevant articles, got: {intent.relevant_articles}"
    )
    assert any(c.startswith("Article 45, Section 130") for c in citations), (
        f"Expected Article 45, Section 130 in top results. Got: {joined}"
    )


def _test_query_embedded_hours_drive_wage_lookup_step() -> None:
    contract_id = "local7_safeway_pueblo_clerks_2022"
    query = "How much should I be making after 520 hours as non-food gm floral?"

    intent = classify_intent(query, contract_id=contract_id)
    assert intent.intent_type == "wage", f"Expected wage intent, got: {intent.intent_type}"

    retriever = HybridRetriever(vector_store=None)
    result = retriever.retrieve(
        query=query,
        intent=intent,
        n_results=10,
        use_hybrid=True,
        contract_id=contract_id,
    )
    wage_info = result.get("wage_info") or {}
    assert wage_info, "Expected wage_info for query with explicit hours and classification."
    assert wage_info.get("classification_key") == "nonfood_gm_floral", (
        f"Expected nonfood_gm_floral wage lookup key, got: {wage_info.get('classification_key')}"
    )
    assert str(wage_info.get("step") or "").strip().lower() == "after 520 hours", (
        f"Expected 'After 520 hours' step, got: {wage_info.get('step')}"
    )
    # An undated query resolves to the rate in effect *today*, which shifts as the
    # MOA's scheduled raises take effect (FSAR 2025-07-05 -> OE+52 2026-07-04 ->
    # OE+104 2027-07-03). To assert the wage math deterministically, pin the date
    # to the FSAR schedule and check that exact dated row is selected.
    fsar = retriever.lookup_wage(
        classification=intent.classification,
        hours_worked=520,
        contract_id=contract_id,
        effective_date="2025-07-05",
    )
    assert float(fsar.get("rate") or 0.0) == 17.75, (
        f"Expected 17.75 FSAR rate for 520-hour step, got: {fsar.get('rate')}"
    )
    assert str(fsar.get("selected_schedule_label") or "").strip().upper() == "FSAR", (
        f"Expected FSAR-selected wage step, got: {fsar.get('selected_schedule_label')}"
    )
    # And an undated lookup must never return a future scheduled raise.
    import datetime as _dt
    today = _dt.date.today().isoformat()
    undated = wage_info.get("effective_date")
    assert str(undated or "") <= today, (
        f"Undated wage lookup returned a future effective date {undated} (today {today})"
    )


def _test_side_letter_anchor_inference_discovers_expected_articles() -> None:
    ks_articles = infer_side_letter_articles("local7_kingsoopers_loveland_meat_2019")
    sw_articles = infer_side_letter_articles("local7_safeway_pueblo_meat_2022")

    assert 57 in set(ks_articles), (
        f"Expected Article 57 side-letter anchors for KS Loveland Meat, got: {ks_articles}"
    )
    assert 32 in set(sw_articles), (
        f"Expected Article 32 side-letter anchors for Safeway Pueblo Meat, got: {sw_articles}"
    )


def _test_side_letter_followup_query_routes_side_letter_anchor() -> None:
    contract_id = "local7_safeway_pueblo_meat_2022"
    query = "How much written notice is required if either party wants to discontinue that agreement?"

    intent = classify_intent(query, contract_id=contract_id)
    retriever = HybridRetriever(vector_store=None)
    result = retriever.retrieve(
        query=query,
        intent=intent,
        n_results=8,
        use_hybrid=True,
        contract_id=contract_id,
    )
    chunks = list(result.get("chunks") or [])
    citations = [str(c.get("citation") or "") for c in chunks[:8]]
    joined = " | ".join(citations)

    assert 32 in set(intent.relevant_articles), (
        f"Expected side-letter anchor Article 32 in relevant articles, got: {intent.relevant_articles}"
    )
    assert any(c.startswith("Article 32, Section 94, Part 5+") for c in citations), (
        f"Expected side-letter notice clause in top results. Got: {joined}"
    )
    match = next(
        (chunk for chunk in chunks if str(chunk.get("citation") or "").startswith("Article 32, Section 94, Part 5+")),
        None,
    )
    assert match is not None, f"Expected to recover the side-letter notice chunk. Got: {joined}"
    assert str(match.get("doc_type") or "").strip().lower() == "lou", (
        f"Expected returned side-letter notice chunk to expose inferred doc_type 'lou', got: {match.get('doc_type')}"
    )


def _test_followup_routing_plan_rewrites_short_vacation_turn() -> None:
    plan = build_followup_routing_plan(
        question="what about 4 years",
        prior_topic="vacation",
        prior_citations=["Article 17, Section 42"],
        prior_article_anchors=[17],
    )
    plan_dict = plan.to_dict()
    assert plan.followup_context_used is True, f"Expected follow-up plan to reuse context, got: {plan_dict}"
    assert plan.strategy == "followup_anchor_seeded", (
        f"Expected anchor-seeded follow-up strategy, got: {plan_dict}"
    )
    assert 17 in set(plan.article_anchors), f"Expected follow-up plan to preserve Article 17 anchor, got: {plan_dict}"
    assert "vacation" in plan.routing_query.lower(), f"Expected routing query to inject prior topic, got: {plan_dict}"
    assert "article 17" in plan.routing_query.lower(), f"Expected routing query to inject prior article anchor, got: {plan_dict}"


def _test_explicit_side_letter_query_infers_doc_type_without_pack_backfill() -> None:
    contract_id = "local7_safeway_pueblo_meat_2022"
    query = "What does the Floater Pool Letter of Understanding say?"

    intent = classify_intent(query, contract_id=contract_id)
    retriever = HybridRetriever(vector_store=None)
    result = retriever.retrieve(
        query=query,
        intent=intent,
        n_results=8,
        use_hybrid=True,
        contract_id=contract_id,
    )
    retrieval_policy = dict(result.get("retrieval_policy") or {})
    retrieval_plan = dict(result.get("retrieval_plan") or {})
    chunks = list(result.get("chunks") or [])
    citations = [str(c.get("citation") or "") for c in chunks[:8]]
    joined = " | ".join(citations)

    match = next(
        (chunk for chunk in chunks if str(chunk.get("citation") or "").startswith("Article 32, Section 94, Part 5+")),
        None,
    )
    assert match is not None, (
        f"Expected explicit LOU query to surface Article 32, Section 94, Part 5+. Got: {joined}"
    )
    assert str(match.get("doc_type") or "").strip().lower() == "lou", (
        f"Expected inferred doc_type 'lou' for explicit side-letter query, got: {match.get('doc_type')}"
    )
    assert retrieval_plan.get("planned_strategy") == "side_letter_query", (
        f"Expected explicit LOU query to declare side-letter query planning, got: {retrieval_plan}"
    )
    assert retrieval_plan.get("apply_side_letter_promotion") is True, (
        f"Expected plan to include side-letter promotion, got: {retrieval_plan}"
    )
    assert retrieval_plan.get("side_letter_filter_supported") is True, (
        f"Expected refreshed pack to report raw side-letter filter support, got: {retrieval_plan}"
    )
    assert retrieval_policy.get("search_mode") == "single_angle_hybrid", (
        f"Expected router policy to record single-angle hybrid retrieval, got: {retrieval_policy}"
    )
    assert retrieval_policy.get("doc_type_filter") == "lou", (
        f"Expected refreshed pack to use raw lou doc-type filtering after normalization, got: {retrieval_policy}"
    )
    assert retrieval_policy.get("side_letter_seeded") is True, (
        f"Expected explicit LOU query to record side-letter promotion, got: {retrieval_policy}"
    )
    assert retrieval_policy.get("strategy") == "side_letter_promoted", (
        f"Expected side-letter promotion strategy, got: {retrieval_policy}"
    )
    executed_stages = list(retrieval_policy.get("executed_stages") or [])
    assert "side_letter_promotion" in executed_stages, (
        f"Expected staged executor to record side-letter promotion, got: {retrieval_policy}"
    )
    assert "full_article_expansion" in executed_stages, (
        f"Expected staged executor to record full-article expansion, got: {retrieval_policy}"
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
        _test_post_2005_holiday_work_premium_query_prioritizes_section_40()
        _test_overtime_definition_query_prefers_article_12()
        _test_role_comparison_query_maps_multiple_classifications()
        _test_role_targeted_wage_followup_prefers_explicit_role()
        _test_premium_rate_legal_text_not_routed_as_wage()
        _test_steward_union_meeting_schedule_routes_article_45()
        _test_query_embedded_hours_drive_wage_lookup_step()
        _test_side_letter_anchor_inference_discovers_expected_articles()
        _test_side_letter_followup_query_routes_side_letter_anchor()
        _test_followup_routing_plan_rewrites_short_vacation_turn()
        _test_explicit_side_letter_query_infers_doc_type_without_pack_backfill()
    finally:
        router_module.HYBRID_VECTOR_WEIGHT = original_vector
        router_module.HYBRID_KEYWORD_WEIGHT = original_keyword
        router_module.CAG_ENABLE_RERANKER = original_reranker

    print("[OK] Topic routing checks passed")


if __name__ == "__main__":
    main()
