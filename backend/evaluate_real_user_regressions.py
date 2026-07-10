"""
Canonical evaluator for the first real-user error-correction regressions.

Wraps the deterministic regression tests that capture production-like misses:
- wage answer binding
- vacation follow-up carryover and overrides
- effective entitlement artifact preference
- ambiguous management-role onboarding containment
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import DATA_DIR
import backend.test_real_user_error_corrections as regression_tests


_CASES: list[tuple[str, Callable[[], None]]] = [
    (
        "wage_answer_binding",
        regression_tests.test_wage_answer_is_bound_to_structured_wage_info,
    ),
    (
        "wage_progression_followup_deterministic",
        regression_tests.test_wage_progression_followup_stays_deterministic,
    ),
    (
        "wage_next_bracket_followup_appendix_rows",
        regression_tests.test_wage_next_bracket_followup_uses_appendix_rows,
    ),
    (
        "vacation_followup_4_years",
        regression_tests.test_vacation_followup_four_years_reuses_topic_and_articles,
    ),
    (
        "vacation_followup_full_time_override",
        regression_tests.test_vacation_followup_full_time_override_is_deterministic,
    ),
    (
        "vacation_followup_estimate_assumption",
        regression_tests.test_vacation_estimate_followup_explains_hour_assumption,
    ),
    (
        "effective_entitlement_prefers_effective_snapshot",
        regression_tests.test_effective_entitlement_resolution_prefers_effective_snapshot,
    ),
    (
        "ambiguous_management_not_default",
        regression_tests.test_ambiguous_management_roles_are_not_default_onboarding_choices,
    ),
    (
        "ambiguous_management_profile_clarification",
        regression_tests.test_ambiguous_management_profile_update_returns_clarification,
    ),
    (
        "ambiguous_management_wage_query_clarification",
        regression_tests.test_ambiguous_management_wage_query_requests_clarification,
    ),
    (
        "generic_clerk_wage_query_clarification",
        regression_tests.test_generic_clerk_wage_query_requests_clarification,
    ),
    (
        "dug_query_still_resolves_nonfood_gm_floral",
        regression_tests.test_dug_wage_query_still_resolves_nonfood_gm_floral,
    ),
    (
        "night_premium_suppresses_wage_fast_path",
        regression_tests.test_night_premium_query_does_not_take_wage_fast_path,
    ),
    (
        "courtesy_clerk_fsar_rate_surfaces_in_answer",
        regression_tests.test_courtesy_clerk_fsar_rate_surfaces_in_answer,
    ),
]


def case_ids() -> set[str]:
    return {case_id for case_id, _ in _CASES}


def run() -> dict:
    rows: list[dict] = []
    passed = 0

    for case_id, fn in _CASES:
        try:
            fn()
            row = {
                "id": case_id,
                "pass": True,
                "error": None,
            }
            passed += 1
        except Exception as exc:
            row = {
                "id": case_id,
                "pass": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)

    total = len(rows)
    return {
        "schema_version": "real_user_regressions_eval_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall": {
            "passed": passed,
            "total": total,
            "pass_rate": round((passed / total) if total else 0.0, 4),
        },
        "results": rows,
    }


def _write_report(report: dict) -> Path:
    out_path = DATA_DIR / "test_set" / "real_user_regressions_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate real-user correction regressions.")
    parser.parse_args()

    report = run()
    out_path = _write_report(report)
    overall = report["overall"]

    print("=" * 72)
    print("KARL Real-User Regression Evaluation")
    print("=" * 72)
    print(f"Overall: {overall['passed']}/{overall['total']} ({overall['pass_rate']:.1%})")
    for row in report["results"]:
        status = "PASS" if row["pass"] else "FAIL"
        suffix = "" if row["pass"] else f" :: {row['error']}"
        print(f"- {status} {row['id']}{suffix}")
    print(f"Results: {out_path}")
    return 0 if overall["passed"] == overall["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
