"""Regression coverage for the first real-user error-correction cycle."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import fastapi as _real_fastapi  # noqa: F401
    _HAVE_FASTAPI = True
except ImportError:
    _HAVE_FASTAPI = False

if not _HAVE_FASTAPI and "fastapi" not in sys.modules:
    fastapi_module = types.ModuleType("fastapi")
    middleware_module = types.ModuleType("fastapi.middleware")
    cors_module = types.ModuleType("fastapi.middleware.cors")
    staticfiles_module = types.ModuleType("fastapi.staticfiles")
    responses_module = types.ModuleType("fastapi.responses")

    class _DummyFastAPI:
        def __init__(self, *args, **kwargs):
            pass

        def add_middleware(self, *args, **kwargs):
            return None

        def mount(self, *args, **kwargs):
            return None

        def _decorator(self, *args, **kwargs):
            def _wrap(func):
                return func
            return _wrap

        get = post = put = delete = patch = _decorator

    class _DummyHTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class _DummyCORSMiddleware:
        def __init__(self, *args, **kwargs):
            pass

    class _DummyStaticFiles:
        def __init__(self, *args, **kwargs):
            pass

    class _DummyFileResponse:
        def __init__(self, *args, **kwargs):
            pass

    class _DummyRequest:
        def __init__(self, *args, **kwargs):
            pass

    class _DummyResponse:
        def __init__(self, *args, **kwargs):
            pass

    fastapi_module.FastAPI = _DummyFastAPI
    fastapi_module.HTTPException = _DummyHTTPException
    fastapi_module.Request = _DummyRequest
    cors_module.CORSMiddleware = _DummyCORSMiddleware
    staticfiles_module.StaticFiles = _DummyStaticFiles
    responses_module.FileResponse = _DummyFileResponse
    responses_module.HTMLResponse = _DummyResponse
    responses_module.JSONResponse = _DummyResponse
    responses_module.RedirectResponse = _DummyResponse
    responses_module.Response = _DummyResponse

    sys.modules["fastapi"] = fastapi_module
    sys.modules["fastapi.middleware"] = middleware_module
    sys.modules["fastapi.middleware.cors"] = cors_module
    sys.modules["fastapi.staticfiles"] = staticfiles_module
    sys.modules["fastapi.responses"] = responses_module

try:
    import pydantic as _real_pydantic  # noqa: F401
    _HAVE_PYDANTIC = True
except ImportError:
    _HAVE_PYDANTIC = False

if not _HAVE_PYDANTIC and "pydantic" not in sys.modules:
    pydantic_module = types.ModuleType("pydantic")

    class _DummyBaseModel:
        def __init__(self, **kwargs):
            annotations = getattr(self.__class__, "__annotations__", {})
            for key in annotations:
                if key in kwargs:
                    setattr(self, key, kwargs[key])
                elif hasattr(self.__class__, key):
                    setattr(self, key, getattr(self.__class__, key))
                else:
                    setattr(self, key, None)

        def model_dump(self, exclude_none: bool = False):
            payload = dict(self.__dict__)
            if exclude_none:
                payload = {k: v for k, v in payload.items() if v is not None}
            return payload

    def _dummy_field(default=None, **_kwargs):
        return default

    pydantic_module.BaseModel = _DummyBaseModel
    pydantic_module.Field = _dummy_field
    sys.modules["pydantic"] = pydantic_module

import backend.api as api_module
import backend.config as runtime_config
import backend.entitlement_files as entitlement_files_module
import backend.retrieval.router as router_module
from backend.api import ProfileUpdateRequest, QueryRequest, query_contract, update_profile
from backend.generation.context import clear_session_context, get_session_context
from backend.retrieval.router import HybridRetriever
from backend.user.profile import clear_user_profile, get_classification_options, update_user_profile


def _load_manifest(contract_id: str) -> dict:
    path = Path("data/manifests") / f"{contract_id}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_request(
    contract_id: str,
    question: str,
    session_id: str | None = None,
    user_classification: str | None = None,
    hours_worked: int = 0,
    months_employed: int = 0,
) -> QueryRequest:
    manifest = _load_manifest(contract_id)
    return QueryRequest(
        question=question,
        union_local_id=str(manifest.get("union_local") or ""),
        contract_id=contract_id,
        contract_version=str(
            manifest.get("contract_version")
            or f"{manifest.get('term_start', '')}__{manifest.get('term_end', '')}"
        ),
        user_classification=user_classification,
        hours_worked=hours_worked,
        months_employed=months_employed,
        session_id=session_id,
    )


def _run_query(request: QueryRequest):
    original_retriever = api_module.retriever
    original_generate = api_module.generate_response
    original_vector = runtime_config.HYBRID_VECTOR_WEIGHT
    original_router_vector = router_module.HYBRID_VECTOR_WEIGHT
    original_reranker = getattr(runtime_config, "CAG_ENABLE_RERANKER", False)
    original_router_reranker = getattr(router_module, "CAG_ENABLE_RERANKER", False)
    api_module.retriever = HybridRetriever(vector_store=None)
    runtime_config.HYBRID_VECTOR_WEIGHT = 0.0
    router_module.HYBRID_VECTOR_WEIGHT = 0.0
    runtime_config.CAG_ENABLE_RERANKER = False
    router_module.CAG_ENABLE_RERANKER = False

    async def _should_not_generate(*_args, **_kwargs):
        raise AssertionError("generate_response should not run for deterministic regression cases")

    api_module.generate_response = _should_not_generate
    try:
        return asyncio.run(query_contract(request))
    finally:
        api_module.retriever = original_retriever
        api_module.generate_response = original_generate
        runtime_config.HYBRID_VECTOR_WEIGHT = original_vector
        router_module.HYBRID_VECTOR_WEIGHT = original_router_vector
        runtime_config.CAG_ENABLE_RERANKER = original_reranker
        router_module.CAG_ENABLE_RERANKER = original_router_reranker


def test_wage_answer_is_bound_to_structured_wage_info() -> None:
    request = _build_request(
        contract_id="local7_safeway_pueblo_clerks_2022",
        question="what should i be making right now?",
        user_classification="courtesy_clerk",
        months_employed=19,
    )
    response = _run_query(request)
    wage_info = dict(response.wage_info or {})
    assert wage_info, "Expected deterministic wage info for courtesy clerk regression."
    expected_rate = f"${float(wage_info.get('rate') or 0.0):.2f}"
    assert expected_rate in response.answer, "Answer must reuse the structured wage rate exactly."
    assert "Appendix A" in response.answer, "Deterministic wage answer should cite Appendix A."
    assert any("appendix a" in str(citation).lower() for citation in response.citations), (
        "Deterministic wage response should surface Appendix A citation metadata."
    )


def test_courtesy_clerk_fsar_rate_surfaces_in_answer() -> None:
    request = _build_request(
        contract_id="local7_safeway_pueblo_clerks_2022",
        question="how much should i be making right now?",
        user_classification="courtesy_clerk",
        months_employed=1,
    )
    response = _run_query(request)
    wage_info = dict(response.wage_info or {})
    assert wage_info, "Expected deterministic wage info for courtesy clerk."

    # "Right now" must resolve to the MOA-amended rate in effect TODAY, not the
    # pre-MOA base and not a future scheduled raise. The MOA lays out dated steps
    # (FSAR 2025-07-05 -> OE+52 2026-07-04 -> OE+104 2027-07-03), so the exact
    # rate advances over time; assert the invariants that must always hold rather
    # than a single value that goes stale as raises take effect.
    import datetime as _dt
    import re as _re

    label = str(wage_info.get("selected_schedule_label") or "")
    assert _re.match(r"^(FSAR|OE\+\d+)$", label), (
        f"Expected a MOA schedule label (FSAR/OE+N), got: {label!r}"
    )
    assert wage_info.get("amendments_applied"), "Expected the MOA amendment to be applied to the rate."
    effective = str(wage_info.get("effective_date") or "")
    assert effective <= _dt.date.today().isoformat(), (
        f"'Right now' returned a future scheduled rate (effective {effective})."
    )
    rate = float(wage_info.get("rate") or 0.0)
    assert rate > 0.0, "Expected a positive resolved rate."
    # The user-facing answer must disclose the resolved rate and MOA schedule.
    assert f"${rate:.2f}" in response.answer, "Expected the answer to reflect the resolved rate."
    assert label in response.answer, "Expected the answer to disclose the selected MOA schedule label."


def test_wage_progression_followup_stays_deterministic() -> None:
    contract_id = "local7_safeway_pueblo_clerks_2022"
    session_id = "real-user-wage-progression"
    clear_user_profile(session_id)
    clear_session_context(session_id)

    first = _run_query(
        _build_request(
            contract_id=contract_id,
            question="how much should i be making?",
            session_id=session_id,
            user_classification="nonfood_gm_floral",
            months_employed=19,
            hours_worked=1558,
        )
    )
    assert first.wage_info, "Expected initial wage lookup to stay deterministic."

    followup = _run_query(
        _build_request(
            contract_id=contract_id,
            question="whats the second grade tier?",
            session_id=session_id,
        )
    )
    assert followup.wage_info is None, "Progression-step follow-up should avoid showing the generic current-rate card."
    assert "Appendix A" in followup.answer, "Expected progression follow-up answer to stay bound to Appendix A."
    assert "After 520 hours" in followup.answer, "Expected second progression step to resolve from Appendix A rows."
    assert any("appendix a" in str(citation).lower() for citation in followup.citations), (
        "Expected progression follow-up to surface Appendix A citations."
    )
    retrieval_context = get_session_context(session_id).get_last_retrieval_context()
    assert retrieval_context.get("wage_context"), "Expected wage follow-up context to persist for subsequent turns."


def test_wage_next_bracket_followup_uses_appendix_rows() -> None:
    contract_id = "local7_safeway_pueblo_clerks_2022"
    session_id = "real-user-wage-next-bracket"
    clear_user_profile(session_id)
    clear_session_context(session_id)

    _run_query(
        _build_request(
            contract_id=contract_id,
            question="how much should i be making?",
            session_id=session_id,
            user_classification="nonfood_gm_floral",
            months_employed=19,
            hours_worked=1558,
        )
    )
    followup = _run_query(
        _build_request(
            contract_id=contract_id,
            question="i havent made the next bracket yet??",
            session_id=session_id,
        )
    )
    assert "current progression step" in followup.answer.lower(), "Expected follow-up answer to explain the current bracket."
    assert "next step is" in followup.answer.lower(), "Expected next bracket resolution to identify the next Appendix A step."
    assert "$" in followup.answer, "Expected next bracket follow-up to include the next contractual rate."
    assert any("appendix a" in str(citation).lower() for citation in followup.citations), (
        "Expected next-bracket answer to cite Appendix A."
    )


def test_vacation_followup_four_years_reuses_topic_and_articles() -> None:
    contract_id = "local7_safeway_pueblo_clerks_2022"
    session_id = "real-user-vacation-4-years"
    clear_user_profile(session_id)
    clear_session_context(session_id)
    update_user_profile(
        session_id,
        {
            "contract_id": contract_id,
            "classification": "courtesy_clerk",
            "hire_date": "2024-03-08",
            "employment_type": "part_time",
        },
    )

    first = _run_query(
        _build_request(
            contract_id=contract_id,
            question="how much vacation am i supposed to get currently?",
            session_id=session_id,
        )
    )
    assert first.entitlement_info, "Expected deterministic vacation result for the first turn."
    ctx = get_session_context(session_id)
    retrieval_context = ctx.get_last_retrieval_context()
    assert retrieval_context.get("topic") == "vacation", "Expected vacation topic to persist in session context."
    assert retrieval_context.get("article_anchors"), "Expected prior article anchors to be stored for follow-ups."
    assert retrieval_context.get("retrieval_strategy") in {"topic_article_seeded", "multi_angle_interpreted"}, (
        "Expected initial vacation turn to record a router-owned retrieval strategy."
    )
    assert retrieval_context.get("followup_context_used") is False, (
        "Expected initial vacation turn not to claim prior-context reuse."
    )

    followup = _run_query(
        _build_request(
            contract_id=contract_id,
            question="what about 4 years",
            session_id=session_id,
        )
    )
    entitlement_info = dict(followup.entitlement_info or {})
    assert entitlement_info, "Expected deterministic vacation follow-up resolution."
    assert int(entitlement_info.get("months_employed") or 0) == 48, (
        "Expected follow-up override to map 4 years to 48 months."
    )
    assert "Using 4 years of service" in followup.answer, "Expected follow-up answer to surface override semantics."
    assert any("Estimated hours for 4 years of service" in str(note) for note in entitlement_info.get("assumption_notes") or []), (
        "Expected hypothetical years follow-up to rescale estimated hours."
    )
    assert followup.followup_context_used is True, "Expected short follow-up to reuse prior session context."
    assert followup.retrieval_strategy == "followup_anchor_seeded", (
        "Expected follow-up retrieval to be labeled as anchor-seeded."
    )
    retrieval_plan = dict(followup.retrieval_plan or {})
    assert retrieval_plan.get("apply_topic_seed_coverage") is True, (
        "Expected follow-up retrieval plan to include topic-anchor seeding."
    )
    followup_plan = dict(retrieval_plan.get("followup") or {})
    assert followup_plan.get("strategy") == "followup_anchor_seeded", (
        "Expected retrieval plan to preserve router-owned follow-up strategy."
    )
    assert "vacation" in str(followup_plan.get("routing_query") or "").lower(), (
        "Expected retrieval plan to preserve router-owned follow-up routing query."
    )
    assert int(retrieval_plan.get("article_anchor_count") or 0) >= 1, (
        "Expected retrieval plan to preserve prior article anchor count."
    )
    assert int(followup.retrieval_anchor_count or 0) >= 1, (
        "Expected follow-up retrieval metadata to report reused article anchors."
    )
    assert followup.retrieval_retry_used is False, (
        "Expected deterministic vacation follow-up to resolve without retry recovery."
    )


def test_vacation_followup_full_time_override_is_deterministic() -> None:
    contract_id = "local7_safeway_pueblo_clerks_2022"
    session_id = "real-user-vacation-full-time"
    clear_user_profile(session_id)
    clear_session_context(session_id)
    update_user_profile(
        session_id,
        {
            "contract_id": contract_id,
            "classification": "courtesy_clerk",
            "hire_date": "2024-03-08",
            "employment_type": "part_time",
        },
    )
    _run_query(
        _build_request(
            contract_id=contract_id,
            question="how much vacation am i supposed to get currently?",
            session_id=session_id,
        )
    )
    followup = _run_query(
        _build_request(
            contract_id=contract_id,
            question="what if i was full time",
            session_id=session_id,
        )
    )
    entitlement_info = dict(followup.entitlement_info or {})
    assert entitlement_info, "Expected deterministic entitlement info for full-time follow-up."
    # Mirror UserProfile.months_employed: calendar months from the fixed hire_date to today,
    # so the expectation tracks the clock the same way production does.
    from datetime import date as _date

    _hire = _date(2024, 3, 8)
    _today = _date.today()
    _months = max(0, (_today.year - _hire.year) * 12 + (_today.month - _hire.month))
    expected_hours = int(_months * 4.33 * 36)
    assert int(entitlement_info.get("hours_worked") or 0) == expected_hours, (
        "Expected full-time override to replace part-time hour estimate."
    )
    assert "full-time schedule" in followup.answer.lower(), "Expected answer to explain the full-time assumption."


def test_vacation_estimate_followup_explains_hour_assumption() -> None:
    contract_id = "local7_safeway_pueblo_clerks_2022"
    session_id = "real-user-vacation-estimate"
    clear_user_profile(session_id)
    clear_session_context(session_id)
    update_user_profile(
        session_id,
        {
            "contract_id": contract_id,
            "classification": "courtesy_clerk",
            "hire_date": "2024-03-08",
            "employment_type": "part_time",
        },
    )
    _run_query(
        _build_request(
            contract_id=contract_id,
            question="how much vacation am i supposed to get currently?",
            session_id=session_id,
        )
    )
    followup = _run_query(
        _build_request(
            contract_id=contract_id,
            question="you may have to estimate because its based on hours served and that should be approximated based on part time full time right",
            session_id=session_id,
        )
    )
    entitlement_info = dict(followup.entitlement_info or {})
    assert entitlement_info, "Expected deterministic entitlement info for estimate follow-up."
    assert any("Estimated hours" in str(note) for note in entitlement_info.get("assumption_notes") or []), (
        "Expected follow-up answer path to surface hour-estimation assumptions."
    )


def test_effective_entitlement_resolution_prefers_effective_snapshot() -> None:
    original_entitlements_dir = entitlement_files_module.ENTITLEMENTS_DIR
    original_manifests_dir = entitlement_files_module.MANIFESTS_DIR
    original_resolve_effective_index_input = entitlement_files_module.resolve_effective_index_input
    Path("tmp_test_work").mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="entitlement_effective_resolution_", dir="tmp_test_work"))
    try:
        entitlements_dir = tmp_root / "entitlements"
        manifests_dir = tmp_root / "manifests"
        effective_file = tmp_root / "effective" / "entitlement_tables_test_contract.json"
        entitlements_dir.mkdir(parents=True, exist_ok=True)
        manifests_dir.mkdir(parents=True, exist_ok=True)
        effective_file.parent.mkdir(parents=True, exist_ok=True)
        (manifests_dir / "test_contract.json").write_text("{}", encoding="utf-8")
        effective_file.write_text("{}", encoding="utf-8")
        (entitlements_dir / "entitlement_tables_test_contract.json").write_text("{}", encoding="utf-8")

        entitlement_files_module.ENTITLEMENTS_DIR = entitlements_dir
        entitlement_files_module.MANIFESTS_DIR = manifests_dir
        entitlement_files_module.resolve_effective_index_input = (
            lambda contract_id, filename: effective_file
            if contract_id == "test_contract" and filename == "entitlement_tables_test_contract.json"
            else None
        )

        resolved = entitlement_files_module.resolve_entitlement_file(
            contract_id="test_contract",
            allow_shared_fallback=False,
        )
        assert resolved == effective_file, "Expected effective entitlement artifact to win over base artifact."
    finally:
        entitlement_files_module.ENTITLEMENTS_DIR = original_entitlements_dir
        entitlement_files_module.MANIFESTS_DIR = original_manifests_dir
        entitlement_files_module.resolve_effective_index_input = original_resolve_effective_index_input
        if tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)


def test_ambiguous_management_roles_are_not_default_onboarding_choices() -> None:
    contract_id = "local7_safeway_pueblo_clerks_2022"
    default_options = {
        str(opt.get("value") or ""): opt
        for opt in get_classification_options(contract_id=contract_id)
    }
    all_options = {
        str(opt.get("value") or ""): opt
        for opt in get_classification_options(contract_id=contract_id, include_unmapped=True)
    }

    for value in ("manager_trainee", "other_assistant_managers"):
        assert value not in default_options, f"Expected ambiguous management role {value} to be hidden by default."
        assert all_options[value].get("requires_role_clarification") is True, (
            f"Expected role {value} to require clarification."
        )


def test_ambiguous_management_profile_update_returns_clarification() -> None:
    contract_id = "local7_safeway_pueblo_clerks_2022"
    session_id = "real-user-ambiguous-management-profile"
    clear_user_profile(session_id)
    response = asyncio.run(
        update_profile(
            session_id,
            ProfileUpdateRequest(
                contract_id=contract_id,
                classification="assistant manager",
            ),
        )
    )
    role_clarification = dict(response.role_clarification or {})
    assert response.classification is None, "Expected ambiguous classification update to avoid storing a role."
    assert role_clarification.get("needs_clarification") is True, (
        "Expected ambiguous management onboarding to request clarification."
    )
    option_values = {str(opt.get("value") or "") for opt in role_clarification.get("options") or []}
    assert {"head_clerk", "manager_trainee", "other_assistant_managers"}.issubset(option_values), (
        "Expected management clarification payload to surface concrete contract choices."
    )


def test_ambiguous_management_wage_query_requests_clarification() -> None:
    request = _build_request(
        contract_id="local7_safeway_pueblo_clerks_2022",
        question="what do assistant managers make?",
    )
    response = _run_query(request)
    role_clarification = dict(response.role_clarification or {})
    assert role_clarification.get("needs_clarification") is True, (
        "Expected ambiguous management wage query to return clarification metadata."
    )
    assert "Head Clerk" in response.answer, "Expected clarification answer to enumerate concrete contract choices."
    assert response.wage_info is None, "Clarification responses should not fabricate wage info."
    assert not response.citations, "Clarification response should not pretend it has resolved article evidence yet."


def test_generic_clerk_wage_query_requests_clarification() -> None:
    request = _build_request(
        contract_id="local7_safeway_pueblo_clerks_2022",
        question="what do clerks make?",
    )
    response = _run_query(request)
    role_clarification = dict(response.role_clarification or {})
    assert role_clarification.get("needs_clarification") is True, (
        "Expected generic clerk wage query to request role clarification."
    )
    option_values = {str(opt.get("value") or "") for opt in role_clarification.get("options") or []}
    assert {"courtesy_clerk", "all_purpose_clerk", "head_clerk"}.issubset(option_values), (
        "Expected generic clerk clarification payload to enumerate the contract clerk roles."
    )
    assert response.wage_info is None, "Generic clerk clarification response should not invent wage info."


def test_dug_wage_query_still_resolves_nonfood_gm_floral() -> None:
    request = _build_request(
        contract_id="local7_safeway_pueblo_clerks_2022",
        question="what does dug make per hour?",
        user_classification="courtesy_clerk",
    )
    response = _run_query(request)
    wage_info = dict(response.wage_info or {})
    assert not response.role_clarification, "Explicit DUG role queries should not be over-clarified."
    assert wage_info, "Expected explicit DUG wage query to stay on the deterministic wage path."
    resolved_class = str(wage_info.get("classification_key") or wage_info.get("classification") or "").lower()
    assert resolved_class == "nonfood_gm_floral", (
        "Expected DUG wording to resolve to the contract nonfood_gm_floral wage row."
    )


def test_night_premium_query_does_not_take_wage_fast_path() -> None:
    contract_id = "local7_safeway_pueblo_clerks_2022"
    query = "how much do cc's get paid at night"
    intent = router_module.classify_intent(query, contract_id=contract_id)
    assert api_module._should_suppress_deterministic_wage_path(query, intent.topic) is True, (
        "Expected night-premium wording to suppress the deterministic base-wage answer path."
    )


def main() -> None:
    test_wage_answer_is_bound_to_structured_wage_info()
    test_courtesy_clerk_fsar_rate_surfaces_in_answer()
    test_wage_progression_followup_stays_deterministic()
    test_wage_next_bracket_followup_uses_appendix_rows()
    test_vacation_followup_four_years_reuses_topic_and_articles()
    test_vacation_followup_full_time_override_is_deterministic()
    test_vacation_estimate_followup_explains_hour_assumption()
    test_effective_entitlement_resolution_prefers_effective_snapshot()
    test_ambiguous_management_roles_are_not_default_onboarding_choices()
    test_ambiguous_management_profile_update_returns_clarification()
    test_ambiguous_management_wage_query_requests_clarification()
    test_generic_clerk_wage_query_requests_clarification()
    test_dug_wage_query_still_resolves_nonfood_gm_floral()
    test_night_premium_query_does_not_take_wage_fast_path()
    print("[OK] Real-user error correction regressions passed")


if __name__ == "__main__":
    main()
