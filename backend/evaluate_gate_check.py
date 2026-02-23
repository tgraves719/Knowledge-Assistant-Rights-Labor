"""
Release-gate checker for evaluation artifacts.

Fails (exit code 1) if thresholds are not met.
"""

import argparse
import json
import sys
from pathlib import Path

from backend.validate_adversarial_dataset import run as validate_adversarial_dataset_run
from backend.validate_cross_contract_mentions_dataset import run as validate_cross_contract_mentions_dataset_run
from backend.validate_unanswerable_dataset import run as validate_unanswerable_dataset_run


def _load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _check_v2(results: dict, min_accuracy: float) -> tuple[bool, str]:
    acc = results.get("summary", {}).get("accuracy")
    if acc is None:
        return False, "v2 summary.accuracy missing"
    ok = acc >= min_accuracy
    return ok, f"v2 accuracy={acc:.3f} threshold>={min_accuracy:.3f}"


def _check_escalation(results: dict, min_precision: float, min_recall: float, max_fpr: float) -> list[tuple[bool, str]]:
    m = results.get("escalation_metrics", {})
    precision = m.get("precision")
    recall = m.get("recall")
    fpr = m.get("false_positive_rate")
    checks: list[tuple[bool, str]] = []

    if precision is None or recall is None or fpr is None:
        checks.append((False, "escalation metrics missing required fields"))
        return checks

    checks.append((precision >= min_precision, f"escalation precision={precision:.3f} threshold>={min_precision:.3f}"))
    checks.append((recall >= min_recall, f"escalation recall={recall:.3f} threshold>={min_recall:.3f}"))
    checks.append((fpr <= max_fpr, f"escalation false_positive_rate={fpr:.3f} threshold<={max_fpr:.3f}"))
    return checks


def _check_multi_contract(
    results: dict,
    min_overall_accuracy: float,
    min_per_contract_accuracy: float,
) -> list[tuple[bool, str]]:
    checks: list[tuple[bool, str]] = []
    overall = results.get("overall", {}) or {}
    overall_acc = overall.get("pass_rate")
    if overall_acc is None:
        checks.append((False, "multi-contract overall.pass_rate missing"))
        return checks
    checks.append(
        (
            overall_acc >= min_overall_accuracy,
            f"multi-contract overall pass_rate={overall_acc:.3f} threshold>={min_overall_accuracy:.3f}",
        )
    )

    by_contract = results.get("by_contract", {}) or {}
    if not by_contract:
        checks.append((False, "multi-contract by_contract summary missing"))
        return checks

    for contract_id, stats in sorted(by_contract.items()):
        acc = (stats or {}).get("pass_rate")
        if acc is None:
            checks.append((False, f"multi-contract {contract_id} pass_rate missing"))
            continue
        checks.append(
            (
                acc >= min_per_contract_accuracy,
                f"multi-contract {contract_id} pass_rate={acc:.3f} threshold>={min_per_contract_accuracy:.3f}",
            )
        )
    return checks


def _check_cross_contamination(results: dict) -> list[tuple[bool, str]]:
    checks: list[tuple[bool, str]] = []
    skipped = results.get("skipped")
    failures_count = results.get("failures_count")
    passed = results.get("pass")

    if skipped is None:
        checks.append((False, "cross-contamination skipped flag missing"))
    else:
        checks.append((not bool(skipped), f"cross-contamination skipped={bool(skipped)} expected=False"))

    if failures_count is None:
        checks.append((False, "cross-contamination failures_count missing"))
    else:
        checks.append((int(failures_count) == 0, f"cross-contamination failures_count={int(failures_count)} expected=0"))

    if passed is None:
        checks.append((False, "cross-contamination pass flag missing"))
    else:
        checks.append((bool(passed), f"cross-contamination pass={bool(passed)} expected=True"))

    return checks


def _check_paraphrase(
    results: dict,
    min_family_pass_rate: float,
    min_worker_slang_pass_rate: float,
    min_formal_rewrite_pass_rate: float,
) -> list[tuple[bool, str]]:
    checks: list[tuple[bool, str]] = []
    overall = results.get("overall", {}) or {}
    by_variant_type = results.get("by_variant_type", {}) or {}
    family_rate = overall.get("family_pass_rate")
    slang_rate = overall.get("worker_slang_pass_rate")
    formal_rate = overall.get("formal_rewrite_pass_rate")
    if formal_rate is None:
        formal_rate = (by_variant_type.get("formal_rewrite") or {}).get("pass_rate")

    if family_rate is None:
        checks.append((False, "paraphrase overall.family_pass_rate missing"))
    else:
        checks.append(
            (
                family_rate >= min_family_pass_rate,
                f"paraphrase family_pass_rate={family_rate:.3f} threshold>={min_family_pass_rate:.3f}",
            )
        )

    if slang_rate is None:
        checks.append((False, "paraphrase overall.worker_slang_pass_rate missing"))
    else:
        checks.append(
            (
                slang_rate >= min_worker_slang_pass_rate,
                f"paraphrase worker_slang_pass_rate={slang_rate:.3f} threshold>={min_worker_slang_pass_rate:.3f}",
            )
        )

    if formal_rate is None:
        checks.append((False, "paraphrase formal_rewrite pass_rate missing"))
    else:
        checks.append(
            (
                formal_rate >= min_formal_rewrite_pass_rate,
                f"paraphrase formal_rewrite_pass_rate={formal_rate:.3f} threshold>={min_formal_rewrite_pass_rate:.3f}",
            )
        )
    return checks


def _check_adversarial(
    results: dict,
    required_dataset_schema_version: str,
    min_total_cases: int,
    min_cases_per_contract: int,
    min_pass_rate: float,
    min_per_contract_pass_rate: float,
    min_precedence_pass_rate: float,
) -> list[tuple[bool, str]]:
    checks: list[tuple[bool, str]] = []
    overall = results.get("overall", {}) or {}
    by_contract = results.get("by_contract", {}) or {}

    dataset_schema_version = str(results.get("dataset_schema_version") or "")
    total_cases = overall.get("total")
    pass_rate = overall.get("pass_rate")
    precedence_rate = overall.get("precedence_pass_rate")
    precedence_total = overall.get("precedence_total")

    if not dataset_schema_version:
        checks.append((False, "adversarial dataset_schema_version missing"))
    else:
        checks.append(
            (
                dataset_schema_version == required_dataset_schema_version,
                f"adversarial dataset_schema_version={dataset_schema_version} expected={required_dataset_schema_version}",
            )
        )

    if total_cases is None:
        checks.append((False, "adversarial overall.total missing"))
    else:
        checks.append(
            (
                int(total_cases) >= min_total_cases,
                f"adversarial total_cases={int(total_cases)} threshold>={min_total_cases}",
            )
        )

    if pass_rate is None:
        checks.append((False, "adversarial overall.pass_rate missing"))
    else:
        checks.append(
            (
                pass_rate >= min_pass_rate,
                f"adversarial overall pass_rate={pass_rate:.3f} threshold>={min_pass_rate:.3f}",
            )
        )

    if precedence_total is None:
        checks.append((False, "adversarial overall.precedence_total missing"))
    else:
        checks.append((int(precedence_total) > 0, f"adversarial precedence_total={int(precedence_total)} expected>0"))

    if precedence_rate is None:
        checks.append((False, "adversarial overall.precedence_pass_rate missing"))
    else:
        checks.append(
            (
                precedence_rate >= min_precedence_pass_rate,
                f"adversarial precedence_pass_rate={precedence_rate:.3f} threshold>={min_precedence_pass_rate:.3f}",
            )
        )

    if not by_contract:
        checks.append((False, "adversarial by_contract summary missing"))
        return checks

    for contract_id, stats in sorted(by_contract.items()):
        rate = (stats or {}).get("pass_rate")
        if rate is None:
            checks.append((False, f"adversarial {contract_id} pass_rate missing"))
        else:
            checks.append(
                (
                    rate >= min_per_contract_pass_rate,
                    f"adversarial {contract_id} pass_rate={rate:.3f} threshold>={min_per_contract_pass_rate:.3f}",
                )
            )
        total = (stats or {}).get("total")
        if total is None:
            checks.append((False, f"adversarial {contract_id} total missing"))
        else:
            checks.append(
                (
                    int(total) >= min_cases_per_contract,
                    f"adversarial {contract_id} total={int(total)} threshold>={min_cases_per_contract}",
                )
            )
    return checks


def _check_unanswerable(
    results: dict,
    required_dataset_schema_version: str,
    min_total_cases: int,
    min_cases_per_contract: int,
    min_pass_rate: float,
    min_per_contract_pass_rate: float,
) -> list[tuple[bool, str]]:
    checks: list[tuple[bool, str]] = []
    overall = results.get("overall", {}) or {}
    by_contract = results.get("by_contract", {}) or {}

    dataset_schema_version = str(results.get("dataset_schema_version") or "")
    total_cases = overall.get("total")
    pass_rate = overall.get("pass_rate")
    abstention_rate = overall.get("abstention_rate")

    if not dataset_schema_version:
        checks.append((False, "unanswerable dataset_schema_version missing"))
    else:
        checks.append(
            (
                dataset_schema_version == required_dataset_schema_version,
                f"unanswerable dataset_schema_version={dataset_schema_version} expected={required_dataset_schema_version}",
            )
        )

    if total_cases is None:
        checks.append((False, "unanswerable overall.total missing"))
    else:
        checks.append(
            (
                int(total_cases) >= min_total_cases,
                f"unanswerable total_cases={int(total_cases)} threshold>={min_total_cases}",
            )
        )

    if pass_rate is None:
        checks.append((False, "unanswerable overall.pass_rate missing"))
    else:
        checks.append(
            (
                pass_rate >= min_pass_rate,
                f"unanswerable overall pass_rate={pass_rate:.3f} threshold>={min_pass_rate:.3f}",
            )
        )

    if abstention_rate is None:
        checks.append((False, "unanswerable overall.abstention_rate missing"))
    else:
        checks.append(
            (
                abstention_rate >= min_pass_rate,
                f"unanswerable abstention_rate={abstention_rate:.3f} threshold>={min_pass_rate:.3f}",
            )
        )

    if not by_contract:
        checks.append((False, "unanswerable by_contract summary missing"))
        return checks

    for contract_id, stats in sorted(by_contract.items()):
        rate = (stats or {}).get("pass_rate")
        if rate is None:
            checks.append((False, f"unanswerable {contract_id} pass_rate missing"))
        else:
            checks.append(
                (
                    rate >= min_per_contract_pass_rate,
                    f"unanswerable {contract_id} pass_rate={rate:.3f} threshold>={min_per_contract_pass_rate:.3f}",
                )
            )
        total = (stats or {}).get("total")
        if total is None:
            checks.append((False, f"unanswerable {contract_id} total missing"))
        else:
            checks.append(
                (
                    int(total) >= min_cases_per_contract,
                    f"unanswerable {contract_id} total={int(total)} threshold>={min_cases_per_contract}",
                )
            )

    return checks


def _check_cross_contract_mentions(
    results: dict,
    required_dataset_schema_version: str,
    min_total_cases: int,
    min_cases_per_contract: int,
    min_pass_rate: float,
    min_per_contract_pass_rate: float,
    min_no_citation_rate: float,
) -> list[tuple[bool, str]]:
    checks: list[tuple[bool, str]] = []
    overall = results.get("overall", {}) or {}
    by_contract = results.get("by_contract", {}) or {}

    dataset_schema_version = str(results.get("dataset_schema_version") or "")
    total_cases = overall.get("total")
    pass_rate = overall.get("pass_rate")
    no_citation_rate = overall.get("no_citation_rate")

    if not dataset_schema_version:
        checks.append((False, "cross-contract-mentions dataset_schema_version missing"))
    else:
        checks.append(
            (
                dataset_schema_version == required_dataset_schema_version,
                "cross-contract-mentions dataset_schema_version="
                f"{dataset_schema_version} expected={required_dataset_schema_version}",
            )
        )

    if total_cases is None:
        checks.append((False, "cross-contract-mentions overall.total missing"))
    else:
        checks.append(
            (
                int(total_cases) >= min_total_cases,
                f"cross-contract-mentions total_cases={int(total_cases)} threshold>={min_total_cases}",
            )
        )

    if pass_rate is None:
        checks.append((False, "cross-contract-mentions overall.pass_rate missing"))
    else:
        checks.append(
            (
                pass_rate >= min_pass_rate,
                f"cross-contract-mentions overall pass_rate={pass_rate:.3f} threshold>={min_pass_rate:.3f}",
            )
        )

    if no_citation_rate is None:
        checks.append((False, "cross-contract-mentions overall.no_citation_rate missing"))
    else:
        checks.append(
            (
                no_citation_rate >= min_no_citation_rate,
                f"cross-contract-mentions no_citation_rate={no_citation_rate:.3f} threshold>={min_no_citation_rate:.3f}",
            )
        )

    if not by_contract:
        checks.append((False, "cross-contract-mentions by_contract summary missing"))
        return checks

    for contract_id, stats in sorted(by_contract.items()):
        rate = (stats or {}).get("pass_rate")
        if rate is None:
            checks.append((False, f"cross-contract-mentions {contract_id} pass_rate missing"))
        else:
            checks.append(
                (
                    rate >= min_per_contract_pass_rate,
                    "cross-contract-mentions "
                    f"{contract_id} pass_rate={rate:.3f} threshold>={min_per_contract_pass_rate:.3f}",
                )
            )
        total = (stats or {}).get("total")
        if total is None:
            checks.append((False, f"cross-contract-mentions {contract_id} total missing"))
        else:
            checks.append(
                (
                    int(total) >= min_cases_per_contract,
                    f"cross-contract-mentions {contract_id} total={int(total)} threshold>={min_cases_per_contract}",
                )
            )

    return checks


def _check_v3(results: dict, min_components_pass_rate: float) -> list[tuple[bool, str]]:
    checks: list[tuple[bool, str]] = []
    overall = results.get("overall", {}) or {}
    pass_rate = overall.get("pass_rate")
    passed = overall.get("pass")

    if pass_rate is None:
        checks.append((False, "v3 overall.pass_rate missing"))
    else:
        checks.append(
            (
                pass_rate >= min_components_pass_rate,
                f"v3 components pass_rate={pass_rate:.3f} threshold>={min_components_pass_rate:.3f}",
            )
        )

    if passed is None:
        checks.append((False, "v3 overall.pass missing"))
    else:
        checks.append((bool(passed), f"v3 overall.pass={bool(passed)} expected=True"))

    return checks


def _check_release_090(
    results: dict,
    required_schema_version: str,
    min_components_pass_rate: float,
) -> list[tuple[bool, str]]:
    checks: list[tuple[bool, str]] = []
    schema_version = str(results.get("schema_version") or "")
    overall = results.get("overall", {}) or {}
    pass_rate = overall.get("pass_rate")
    passed = overall.get("pass")

    if not schema_version:
        checks.append((False, "release_090 schema_version missing"))
    else:
        checks.append(
            (
                schema_version == required_schema_version,
                f"release_090 schema_version={schema_version} expected={required_schema_version}",
            )
        )

    if pass_rate is None:
        checks.append((False, "release_090 overall.pass_rate missing"))
    else:
        checks.append(
            (
                pass_rate >= min_components_pass_rate,
                f"release_090 components pass_rate={pass_rate:.3f} threshold>={min_components_pass_rate:.3f}",
            )
        )

    if passed is None:
        checks.append((False, "release_090 overall.pass missing"))
    else:
        checks.append((bool(passed), f"release_090 overall.pass={bool(passed)} expected=True"))

    return checks


def _release_090_advisory_messages(results: dict) -> list[tuple[bool, str]]:
    advisories = results.get("advisories") or {}
    if not isinstance(advisories, dict) or not advisories:
        return [(True, "release_090 advisories missing or empty (non-blocking)")]

    messages: list[tuple[bool, str]] = []
    for key in sorted(advisories.keys()):
        row = advisories.get(key) or {}
        if not isinstance(row, dict):
            messages.append((True, f"release_090 advisory {key}: malformed entry (non-blocking)"))
            continue
        status = str(row.get("status") or "")
        warning = bool(row.get("warning"))
        path = str(row.get("path") or "")
        schema_version = str(row.get("schema_version") or "")

        if key == "contract_text_compare_amended":
            coverage = row.get("coverage") or {}
            overall = row.get("overall") or {}
            op_cov = coverage.get("operation_coverage_rate")
            contract_cov = coverage.get("contract_coverage_rate")
            total = overall.get("total")
            passed = overall.get("passed")
            detail = (
                "release_090 advisory contract_text_compare_amended "
                f"status={status} warning={warning} "
                f"coverage(contract={contract_cov}, ops={op_cov}) "
                f"overall={passed}/{total} schema={schema_version}"
            )
            if path:
                detail += f" path={path}"
            messages.append((warning, detail))
            continue

        if key == "base_chunk_lineage":
            summary = row.get("summary") or {}
            detail = (
                "release_090 advisory base_chunk_lineage "
                f"status={status} warning={warning} "
                f"high={summary.get('high_risk')} medium={summary.get('medium_risk')} "
                f"missing_base={summary.get('missing_base_chunk')} schema={schema_version}"
            )
            if path:
                detail += f" path={path}"
            messages.append((warning, detail))
            continue

        detail = f"release_090 advisory {key} status={status} warning={warning} schema={schema_version}"
        if path:
            detail += f" path={path}"
        messages.append((warning, detail))

    return messages


def _check_moa_deleted_vs_updated(
    results: dict,
    required_schema_version: str,
    required_dataset_schema_version: str,
    min_overall_pass_rate: float,
    min_updated_pass_rate: float,
    min_deleted_pass_rate: float,
    min_updated_moa_source_type_match_rate: float,
) -> list[tuple[bool, str]]:
    checks: list[tuple[bool, str]] = []
    schema_version = str(results.get("schema_version") or "")
    dataset_schema_version = str(results.get("dataset_schema_version") or "")
    overall = results.get("overall", {}) or {}
    buckets = results.get("buckets", {}) or {}
    gate = results.get("gate", {}) or {}
    updated = buckets.get("updated", {}) or {}
    deleted = buckets.get("deleted", {}) or {}

    if not schema_version:
        checks.append((False, "moa_deleted_vs_updated schema_version missing"))
    else:
        checks.append(
            (
                schema_version == required_schema_version,
                f"moa_deleted_vs_updated schema_version={schema_version} expected={required_schema_version}",
            )
        )

    if not dataset_schema_version:
        checks.append((False, "moa_deleted_vs_updated dataset_schema_version missing"))
    else:
        checks.append(
            (
                dataset_schema_version == required_dataset_schema_version,
                "moa_deleted_vs_updated dataset_schema_version="
                f"{dataset_schema_version} expected={required_dataset_schema_version}",
            )
        )

    gate_pass = gate.get("pass")
    if gate_pass is None:
        checks.append((False, "moa_deleted_vs_updated gate.pass missing"))
    else:
        checks.append((bool(gate_pass), f"moa_deleted_vs_updated gate.pass={bool(gate_pass)} expected=True"))

    overall_pass_rate = overall.get("pass_rate")
    if overall_pass_rate is None:
        checks.append((False, "moa_deleted_vs_updated overall.pass_rate missing"))
    else:
        checks.append(
            (
                float(overall_pass_rate) >= min_overall_pass_rate,
                "moa_deleted_vs_updated overall.pass_rate="
                f"{float(overall_pass_rate):.3f} threshold>={min_overall_pass_rate:.3f}",
            )
        )

    updated_rate = updated.get("pass_rate")
    if updated_rate is None:
        checks.append((False, "moa_deleted_vs_updated updated.pass_rate missing"))
    else:
        checks.append(
            (
                float(updated_rate) >= min_updated_pass_rate,
                "moa_deleted_vs_updated updated.pass_rate="
                f"{float(updated_rate):.3f} threshold>={min_updated_pass_rate:.3f}",
            )
        )

    deleted_rate = deleted.get("pass_rate")
    if deleted_rate is None:
        checks.append((False, "moa_deleted_vs_updated deleted.pass_rate missing"))
    else:
        checks.append(
            (
                float(deleted_rate) >= min_deleted_pass_rate,
                "moa_deleted_vs_updated deleted.pass_rate="
                f"{float(deleted_rate):.3f} threshold>={min_deleted_pass_rate:.3f}",
            )
        )

    source_cases = int(updated.get("source_type_cases") or 0)
    source_rate = updated.get("moa_source_type_match_rate")
    if source_cases == 0:
        checks.append((True, "moa_deleted_vs_updated updated.moa_source_type_match_rate skipped (0 cases)"))
    elif source_rate is None:
        checks.append((False, "moa_deleted_vs_updated updated.moa_source_type_match_rate missing"))
    else:
        checks.append(
            (
                float(source_rate) >= min_updated_moa_source_type_match_rate,
                "moa_deleted_vs_updated updated.moa_source_type_match_rate="
                f"{float(source_rate):.3f} threshold>={min_updated_moa_source_type_match_rate:.3f}",
            )
        )

    return checks


def _check_needle(
    results: dict,
    min_pass_rate: float,
    min_position_pass_rate: float,
) -> list[tuple[bool, str]]:
    checks: list[tuple[bool, str]] = []
    overall = results.get("overall", {}) or {}
    pass_rate = overall.get("pass_rate")

    if pass_rate is None:
        checks.append((False, "needle overall.pass_rate missing"))
    else:
        checks.append(
            (
                pass_rate >= min_pass_rate,
                f"needle pass_rate={pass_rate:.3f} threshold>={min_pass_rate:.3f}",
            )
        )

    by_position = results.get("by_position", {}) or {}
    for position in ("top", "middle", "bottom"):
        stats = by_position.get(position) or {}
        pos_rate = stats.get("pass_rate")
        if pos_rate is None:
            checks.append((False, f"needle by_position.{position}.pass_rate missing"))
            continue
        checks.append(
            (
                pos_rate >= min_position_pass_rate,
                f"needle {position} pass_rate={pos_rate:.3f} threshold>={min_position_pass_rate:.3f}",
            )
        )

    return checks


def _check_wage_table_evidence(
    results: dict,
    required_dataset_schema_version: str,
    min_total_cases: int,
    min_cases_per_contract: int,
    min_pass_rate: float,
    min_per_contract_pass_rate: float,
    min_source_method_pass_rate: float,
    min_table_evidence_presence_rate: float,
    min_table_id_presence_rate: float,
) -> list[tuple[bool, str]]:
    checks: list[tuple[bool, str]] = []
    overall = results.get("overall", {}) or {}
    by_contract = results.get("by_contract", {}) or {}

    dataset_schema_version = str(results.get("dataset_schema_version") or "")
    total_cases = overall.get("total")
    pass_rate = overall.get("pass_rate")
    source_method_rate = overall.get("source_method_pass_rate")
    table_evidence_rate = overall.get("table_evidence_presence_rate")
    table_id_rate = overall.get("table_id_presence_rate")

    if not dataset_schema_version:
        checks.append((False, "wage-table-evidence dataset_schema_version missing"))
    else:
        checks.append(
            (
                dataset_schema_version == required_dataset_schema_version,
                "wage-table-evidence dataset_schema_version="
                f"{dataset_schema_version} expected={required_dataset_schema_version}",
            )
        )

    if total_cases is None:
        checks.append((False, "wage-table-evidence overall.total missing"))
    else:
        checks.append(
            (
                int(total_cases) >= min_total_cases,
                f"wage-table-evidence total_cases={int(total_cases)} threshold>={min_total_cases}",
            )
        )

    if pass_rate is None:
        checks.append((False, "wage-table-evidence overall.pass_rate missing"))
    else:
        checks.append(
            (
                pass_rate >= min_pass_rate,
                f"wage-table-evidence overall pass_rate={pass_rate:.3f} threshold>={min_pass_rate:.3f}",
            )
        )

    if source_method_rate is None:
        checks.append((False, "wage-table-evidence overall.source_method_pass_rate missing"))
    else:
        checks.append(
            (
                source_method_rate >= min_source_method_pass_rate,
                "wage-table-evidence source_method_pass_rate="
                f"{source_method_rate:.3f} threshold>={min_source_method_pass_rate:.3f}",
            )
        )

    if table_evidence_rate is None:
        checks.append((False, "wage-table-evidence overall.table_evidence_presence_rate missing"))
    else:
        checks.append(
            (
                table_evidence_rate >= min_table_evidence_presence_rate,
                "wage-table-evidence table_evidence_presence_rate="
                f"{table_evidence_rate:.3f} threshold>={min_table_evidence_presence_rate:.3f}",
            )
        )

    if table_id_rate is None:
        checks.append((False, "wage-table-evidence overall.table_id_presence_rate missing"))
    else:
        checks.append(
            (
                table_id_rate >= min_table_id_presence_rate,
                "wage-table-evidence table_id_presence_rate="
                f"{table_id_rate:.3f} threshold>={min_table_id_presence_rate:.3f}",
            )
        )

    if not by_contract:
        checks.append((False, "wage-table-evidence by_contract summary missing"))
        return checks

    for contract_id, stats in sorted(by_contract.items()):
        rate = (stats or {}).get("pass_rate")
        if rate is None:
            checks.append((False, f"wage-table-evidence {contract_id} pass_rate missing"))
        else:
            checks.append(
                (
                    rate >= min_per_contract_pass_rate,
                    "wage-table-evidence "
                    f"{contract_id} pass_rate={rate:.3f} threshold>={min_per_contract_pass_rate:.3f}",
                )
            )
        total = (stats or {}).get("total")
        if total is None:
            checks.append((False, f"wage-table-evidence {contract_id} total missing"))
        else:
            checks.append(
                (
                    int(total) >= min_cases_per_contract,
                    f"wage-table-evidence {contract_id} total={int(total)} threshold>={min_cases_per_contract}",
                )
            )

    return checks


def _check_entitlement_table_evidence(
    results: dict,
    required_dataset_schema_version: str,
    min_total_cases: int,
    min_cases_per_contract: int,
    min_pass_rate: float,
    min_per_contract_pass_rate: float,
    min_weeks_resolution_pass_rate: float,
    min_source_method_pass_rate: float,
    min_evidence_presence_rate: float,
) -> list[tuple[bool, str]]:
    checks: list[tuple[bool, str]] = []
    overall = results.get("overall", {}) or {}
    by_contract = results.get("by_contract", {}) or {}

    dataset_schema_version = str(results.get("dataset_schema_version") or "")
    total_cases = overall.get("total")
    pass_rate = overall.get("pass_rate")
    weeks_rate = overall.get("weeks_resolution_pass_rate")
    source_method_rate = overall.get("source_method_pass_rate")
    evidence_rate = overall.get("evidence_presence_rate")

    if not dataset_schema_version:
        checks.append((False, "entitlement-table-evidence dataset_schema_version missing"))
    else:
        checks.append(
            (
                dataset_schema_version == required_dataset_schema_version,
                "entitlement-table-evidence dataset_schema_version="
                f"{dataset_schema_version} expected={required_dataset_schema_version}",
            )
        )

    if total_cases is None:
        checks.append((False, "entitlement-table-evidence overall.total missing"))
    else:
        checks.append(
            (
                int(total_cases) >= min_total_cases,
                f"entitlement-table-evidence total_cases={int(total_cases)} threshold>={min_total_cases}",
            )
        )

    if pass_rate is None:
        checks.append((False, "entitlement-table-evidence overall.pass_rate missing"))
    else:
        checks.append(
            (
                pass_rate >= min_pass_rate,
                "entitlement-table-evidence overall pass_rate="
                f"{pass_rate:.3f} threshold>={min_pass_rate:.3f}",
            )
        )

    if weeks_rate is None:
        checks.append((False, "entitlement-table-evidence overall.weeks_resolution_pass_rate missing"))
    else:
        checks.append(
            (
                weeks_rate >= min_weeks_resolution_pass_rate,
                "entitlement-table-evidence weeks_resolution_pass_rate="
                f"{weeks_rate:.3f} threshold>={min_weeks_resolution_pass_rate:.3f}",
            )
        )

    if source_method_rate is None:
        checks.append((False, "entitlement-table-evidence overall.source_method_pass_rate missing"))
    else:
        checks.append(
            (
                source_method_rate >= min_source_method_pass_rate,
                "entitlement-table-evidence source_method_pass_rate="
                f"{source_method_rate:.3f} threshold>={min_source_method_pass_rate:.3f}",
            )
        )

    if evidence_rate is None:
        checks.append((False, "entitlement-table-evidence overall.evidence_presence_rate missing"))
    else:
        checks.append(
            (
                evidence_rate >= min_evidence_presence_rate,
                "entitlement-table-evidence evidence_presence_rate="
                f"{evidence_rate:.3f} threshold>={min_evidence_presence_rate:.3f}",
            )
        )

    if not by_contract:
        checks.append((False, "entitlement-table-evidence by_contract summary missing"))
        return checks

    for contract_id, stats in sorted(by_contract.items()):
        rate = (stats or {}).get("pass_rate")
        if rate is None:
            checks.append((False, f"entitlement-table-evidence {contract_id} pass_rate missing"))
        else:
            checks.append(
                (
                    rate >= min_per_contract_pass_rate,
                    "entitlement-table-evidence "
                    f"{contract_id} pass_rate={rate:.3f} threshold>={min_per_contract_pass_rate:.3f}",
                )
            )
        total = (stats or {}).get("total")
        if total is None:
            checks.append((False, f"entitlement-table-evidence {contract_id} total missing"))
        else:
            checks.append(
                (
                    int(total) >= min_cases_per_contract,
                    f"entitlement-table-evidence {contract_id} total={int(total)} threshold>={min_cases_per_contract}",
                )
            )

    return checks


def _check_role_catalog_integrity(
    results: dict,
    required_dataset_schema_version: str,
    min_total_cases: int,
    min_cases_per_contract: int,
    min_pass_rate: float,
    min_per_contract_pass_rate: float,
    min_dataset_case_pass_rate: float,
    min_default_wage_ready_rate: float,
    min_unresolved_not_default_rate: float,
    min_default_wage_key_unique_rate: float,
) -> list[tuple[bool, str]]:
    checks: list[tuple[bool, str]] = []
    overall = results.get("overall", {}) or {}
    by_contract = results.get("by_contract", {}) or {}

    dataset_schema_version = str(results.get("dataset_schema_version") or "")
    total_cases = overall.get("total")
    pass_rate = overall.get("pass_rate")
    dataset_case_rate = overall.get("dataset_case_pass_rate")
    default_wage_ready_rate = overall.get("default_wage_ready_rate")
    unresolved_not_default_rate = overall.get("unresolved_not_default_rate")
    default_wage_key_unique_rate = overall.get("default_wage_key_unique_rate")

    if not dataset_schema_version:
        checks.append((False, "role-catalog-integrity dataset_schema_version missing"))
    else:
        checks.append(
            (
                dataset_schema_version == required_dataset_schema_version,
                "role-catalog-integrity dataset_schema_version="
                f"{dataset_schema_version} expected={required_dataset_schema_version}",
            )
        )

    if total_cases is None:
        checks.append((False, "role-catalog-integrity overall.total missing"))
    else:
        checks.append(
            (
                int(total_cases) >= min_total_cases,
                f"role-catalog-integrity total_cases={int(total_cases)} threshold>={min_total_cases}",
            )
        )

    if pass_rate is None:
        checks.append((False, "role-catalog-integrity overall.pass_rate missing"))
    else:
        checks.append(
            (
                pass_rate >= min_pass_rate,
                "role-catalog-integrity overall pass_rate="
                f"{pass_rate:.3f} threshold>={min_pass_rate:.3f}",
            )
        )

    if dataset_case_rate is None:
        checks.append((False, "role-catalog-integrity overall.dataset_case_pass_rate missing"))
    else:
        checks.append(
            (
                dataset_case_rate >= min_dataset_case_pass_rate,
                "role-catalog-integrity dataset_case_pass_rate="
                f"{dataset_case_rate:.3f} threshold>={min_dataset_case_pass_rate:.3f}",
            )
        )

    if default_wage_ready_rate is None:
        checks.append((False, "role-catalog-integrity overall.default_wage_ready_rate missing"))
    else:
        checks.append(
            (
                default_wage_ready_rate >= min_default_wage_ready_rate,
                "role-catalog-integrity default_wage_ready_rate="
                f"{default_wage_ready_rate:.3f} threshold>={min_default_wage_ready_rate:.3f}",
            )
        )

    if unresolved_not_default_rate is None:
        checks.append((False, "role-catalog-integrity overall.unresolved_not_default_rate missing"))
    else:
        checks.append(
            (
                unresolved_not_default_rate >= min_unresolved_not_default_rate,
                "role-catalog-integrity unresolved_not_default_rate="
                f"{unresolved_not_default_rate:.3f} threshold>={min_unresolved_not_default_rate:.3f}",
            )
        )

    if default_wage_key_unique_rate is None:
        checks.append((False, "role-catalog-integrity overall.default_wage_key_unique_rate missing"))
    else:
        checks.append(
            (
                default_wage_key_unique_rate >= min_default_wage_key_unique_rate,
                "role-catalog-integrity default_wage_key_unique_rate="
                f"{default_wage_key_unique_rate:.3f} threshold>={min_default_wage_key_unique_rate:.3f}",
            )
        )

    if not by_contract:
        checks.append((False, "role-catalog-integrity by_contract summary missing"))
        return checks

    for contract_id, stats in sorted(by_contract.items()):
        rate = (stats or {}).get("pass_rate")
        if rate is None:
            checks.append((False, f"role-catalog-integrity {contract_id} pass_rate missing"))
        else:
            checks.append(
                (
                    rate >= min_per_contract_pass_rate,
                    "role-catalog-integrity "
                    f"{contract_id} pass_rate={rate:.3f} threshold>={min_per_contract_pass_rate:.3f}",
                )
            )
        total = (stats or {}).get("total")
        if total is None:
            checks.append((False, f"role-catalog-integrity {contract_id} total missing"))
        else:
            checks.append(
                (
                    int(total) >= min_cases_per_contract,
                    f"role-catalog-integrity {contract_id} total={int(total)} threshold>={min_cases_per_contract}",
                )
            )

    return checks


def _check_followup_role_wage(
    results: dict,
    required_dataset_schema_version: str,
    min_total_cases: int,
    min_cases_per_contract: int,
    min_pass_rate: float,
    min_per_contract_pass_rate: float,
    min_target_resolution_rate: float,
    min_table_evidence_presence_rate: float,
    min_appendix_citation_rate: float,
    min_intent_wage_rate: float,
    min_no_unavailable_rate: float,
    min_explicit_override_rate: float,
    min_profile_fallback_rate: float,
) -> list[tuple[bool, str]]:
    checks: list[tuple[bool, str]] = []
    overall = results.get("overall", {}) or {}
    by_contract = results.get("by_contract", {}) or {}

    dataset_schema_version = str(results.get("dataset_schema_version") or "")
    total_cases = overall.get("total")
    pass_rate = overall.get("pass_rate")
    target_resolution_rate = overall.get("target_resolution_rate")
    table_evidence_rate = overall.get("table_evidence_presence_rate")
    appendix_citation_rate = overall.get("appendix_citation_rate")
    intent_wage_rate = overall.get("intent_wage_rate")
    no_unavailable_rate = overall.get("no_unavailable_rate")
    explicit_override_rate = overall.get("explicit_override_rate")
    profile_fallback_rate = overall.get("profile_fallback_rate")

    if not dataset_schema_version:
        checks.append((False, "followup-role-wage dataset_schema_version missing"))
    else:
        checks.append(
            (
                dataset_schema_version == required_dataset_schema_version,
                "followup-role-wage dataset_schema_version="
                f"{dataset_schema_version} expected={required_dataset_schema_version}",
            )
        )

    if total_cases is None:
        checks.append((False, "followup-role-wage overall.total missing"))
    else:
        checks.append(
            (
                int(total_cases) >= min_total_cases,
                f"followup-role-wage total_cases={int(total_cases)} threshold>={min_total_cases}",
            )
        )

    if pass_rate is None:
        checks.append((False, "followup-role-wage overall.pass_rate missing"))
    else:
        checks.append(
            (
                pass_rate >= min_pass_rate,
                f"followup-role-wage overall pass_rate={pass_rate:.3f} threshold>={min_pass_rate:.3f}",
            )
        )

    if target_resolution_rate is None:
        checks.append((False, "followup-role-wage overall.target_resolution_rate missing"))
    else:
        checks.append(
            (
                target_resolution_rate >= min_target_resolution_rate,
                "followup-role-wage target_resolution_rate="
                f"{target_resolution_rate:.3f} threshold>={min_target_resolution_rate:.3f}",
            )
        )

    if table_evidence_rate is None:
        checks.append((False, "followup-role-wage overall.table_evidence_presence_rate missing"))
    else:
        checks.append(
            (
                table_evidence_rate >= min_table_evidence_presence_rate,
                "followup-role-wage table_evidence_presence_rate="
                f"{table_evidence_rate:.3f} threshold>={min_table_evidence_presence_rate:.3f}",
            )
        )

    if appendix_citation_rate is None:
        checks.append((False, "followup-role-wage overall.appendix_citation_rate missing"))
    else:
        checks.append(
            (
                appendix_citation_rate >= min_appendix_citation_rate,
                "followup-role-wage appendix_citation_rate="
                f"{appendix_citation_rate:.3f} threshold>={min_appendix_citation_rate:.3f}",
            )
        )

    if intent_wage_rate is None:
        checks.append((False, "followup-role-wage overall.intent_wage_rate missing"))
    else:
        checks.append(
            (
                intent_wage_rate >= min_intent_wage_rate,
                f"followup-role-wage intent_wage_rate={intent_wage_rate:.3f} threshold>={min_intent_wage_rate:.3f}",
            )
        )

    if no_unavailable_rate is None:
        checks.append((False, "followup-role-wage overall.no_unavailable_rate missing"))
    else:
        checks.append(
            (
                no_unavailable_rate >= min_no_unavailable_rate,
                f"followup-role-wage no_unavailable_rate={no_unavailable_rate:.3f} threshold>={min_no_unavailable_rate:.3f}",
            )
        )

    if explicit_override_rate is None:
        checks.append((False, "followup-role-wage overall.explicit_override_rate missing"))
    else:
        checks.append(
            (
                explicit_override_rate >= min_explicit_override_rate,
                "followup-role-wage explicit_override_rate="
                f"{explicit_override_rate:.3f} threshold>={min_explicit_override_rate:.3f}",
            )
        )

    if profile_fallback_rate is None:
        checks.append((False, "followup-role-wage overall.profile_fallback_rate missing"))
    else:
        checks.append(
            (
                profile_fallback_rate >= min_profile_fallback_rate,
                "followup-role-wage profile_fallback_rate="
                f"{profile_fallback_rate:.3f} threshold>={min_profile_fallback_rate:.3f}",
            )
        )

    if not by_contract:
        checks.append((False, "followup-role-wage by_contract summary missing"))
        return checks

    for contract_id, stats in sorted(by_contract.items()):
        rate = (stats or {}).get("pass_rate")
        if rate is None:
            checks.append((False, f"followup-role-wage {contract_id} pass_rate missing"))
        else:
            checks.append(
                (
                    rate >= min_per_contract_pass_rate,
                    "followup-role-wage "
                    f"{contract_id} pass_rate={rate:.3f} threshold>={min_per_contract_pass_rate:.3f}",
                )
            )
        total = (stats or {}).get("total")
        if total is None:
            checks.append((False, f"followup-role-wage {contract_id} total missing"))
        else:
            checks.append(
                (
                    int(total) >= min_cases_per_contract,
                    f"followup-role-wage {contract_id} total={int(total)} threshold>={min_cases_per_contract}",
                )
            )

    return checks


def _check_false_unavailable(
    results: dict,
    required_dataset_schema_version: str,
    min_total_cases: int,
    min_recover_cases: int,
    min_uncertain_cases: int,
    min_cases_per_contract: int,
    min_pass_rate: float,
    min_per_contract_pass_rate: float,
    min_recovered_rate: float,
    min_proper_uncertainty_rate: float,
) -> list[tuple[bool, str]]:
    checks: list[tuple[bool, str]] = []
    overall = results.get("overall", {}) or {}
    pass_rate = overall.get("pass_rate")
    recovered_rate = overall.get("false_unavailable_recovered_rate")
    uncertainty_rate = overall.get("proper_uncertainty_rate")
    total_cases = overall.get("total")
    recover_total = overall.get("recover_total")
    uncertain_total = overall.get("uncertain_total")
    dataset_schema_version = str(results.get("dataset_schema_version") or "")

    if not dataset_schema_version:
        checks.append((False, "false-unavailable dataset_schema_version missing"))
    else:
        checks.append(
            (
                dataset_schema_version == required_dataset_schema_version,
                "false-unavailable dataset_schema_version="
                f"{dataset_schema_version} expected={required_dataset_schema_version}",
            )
        )

    if total_cases is None:
        checks.append((False, "false-unavailable overall.total missing"))
    else:
        checks.append(
            (
                int(total_cases) >= min_total_cases,
                f"false-unavailable total_cases={int(total_cases)} threshold>={min_total_cases}",
            )
        )

    if recover_total is None:
        checks.append((False, "false-unavailable overall.recover_total missing"))
    else:
        checks.append(
            (
                int(recover_total) >= min_recover_cases,
                f"false-unavailable recover_cases={int(recover_total)} threshold>={min_recover_cases}",
            )
        )

    if uncertain_total is None:
        checks.append((False, "false-unavailable overall.uncertain_total missing"))
    else:
        checks.append(
            (
                int(uncertain_total) >= min_uncertain_cases,
                f"false-unavailable uncertain_cases={int(uncertain_total)} threshold>={min_uncertain_cases}",
            )
        )

    if pass_rate is None:
        checks.append((False, "false-unavailable overall.pass_rate missing"))
    else:
        checks.append(
            (
                pass_rate >= min_pass_rate,
                f"false-unavailable overall pass_rate={pass_rate:.3f} threshold>={min_pass_rate:.3f}",
            )
        )

    if recovered_rate is None:
        checks.append((False, "false-unavailable overall.false_unavailable_recovered_rate missing"))
    else:
        checks.append(
            (
                recovered_rate >= min_recovered_rate,
                "false-unavailable recovered_rate="
                f"{recovered_rate:.3f} threshold>={min_recovered_rate:.3f}",
            )
        )

    if uncertainty_rate is None:
        checks.append((False, "false-unavailable overall.proper_uncertainty_rate missing"))
    else:
        checks.append(
            (
                uncertainty_rate >= min_proper_uncertainty_rate,
                "false-unavailable proper_uncertainty_rate="
                f"{uncertainty_rate:.3f} threshold>={min_proper_uncertainty_rate:.3f}",
            )
        )

    by_contract = results.get("by_contract", {}) or {}
    if not by_contract:
        checks.append((False, "false-unavailable by_contract summary missing"))
        return checks

    for contract_id, stats in sorted(by_contract.items()):
        rate = (stats or {}).get("pass_rate")
        if rate is None:
            checks.append((False, f"false-unavailable {contract_id} pass_rate missing"))
            continue
        checks.append(
            (
                rate >= min_per_contract_pass_rate,
                f"false-unavailable {contract_id} pass_rate={rate:.3f} threshold>={min_per_contract_pass_rate:.3f}",
            )
        )
        total = (stats or {}).get("total")
        if total is None:
            checks.append((False, f"false-unavailable {contract_id} total missing"))
        else:
            checks.append(
                (
                    int(total) >= min_cases_per_contract,
                    f"false-unavailable {contract_id} total={int(total)} threshold>={min_cases_per_contract}",
                )
            )

    return checks


def main():
    parser = argparse.ArgumentParser(description="Check KARL release-gate thresholds from evaluation artifacts.")
    parser.add_argument("--v2-results", default="data/test_set/comprehensive_results.json")
    parser.add_argument("--escalation-results", default="data/test_set/escalation_precision_results.json")
    parser.add_argument("--cross-contamination-results", default="data/test_set/cross_contamination_results.json")
    parser.add_argument("--multi-contract-results", default="data/test_set/multi_contract_v2_results.json")
    parser.add_argument("--v3-results", default="data/test_set/v3_results.json")
    parser.add_argument("--paraphrase-results", default="data/test_set/paraphrase_results.json")
    parser.add_argument("--adversarial-results", default="data/test_set/adversarial_results.json")
    parser.add_argument("--adversarial-dataset", default="data/test_set/adversarial_test.json")
    parser.add_argument("--unanswerable-results", default="data/test_set/unanswerable_results.json")
    parser.add_argument("--unanswerable-dataset", default="data/test_set/unanswerable_multi_contract_test.json")
    parser.add_argument("--cross-contract-mentions-results", default="data/test_set/cross_contract_mentions_results.json")
    parser.add_argument("--cross-contract-mentions-dataset", default="data/test_set/cross_contract_mentions_test.json")
    parser.add_argument("--false-unavailable-results", default="data/test_set/false_unavailable_results.json")
    parser.add_argument("--needle-results", default="data/test_set/needle_results.json")
    parser.add_argument("--wage-table-evidence-results", default="data/test_set/wage_table_evidence_results.json")
    parser.add_argument("--entitlement-table-evidence-results", default="data/test_set/entitlement_table_evidence_results.json")
    parser.add_argument("--role-catalog-integrity-results", default="data/test_set/role_catalog_integrity_results.json")
    parser.add_argument("--followup-role-wage-results", default="data/test_set/followup_role_wage_results.json")
    parser.add_argument("--moa-deleted-vs-updated-results", default="data/test_set/moa_deleted_vs_updated_results.json")
    parser.add_argument("--release-090-results", default="data/test_set/release_0_9_0_scorecard.json")

    parser.add_argument("--min-v2-accuracy", type=float, default=0.80)
    parser.add_argument("--min-escalation-precision", type=float, default=0.90)
    parser.add_argument("--min-escalation-recall", type=float, default=0.70)
    parser.add_argument("--max-escalation-fpr", type=float, default=0.10)
    parser.add_argument("--min-v3-components-pass-rate", type=float, default=1.0)
    parser.add_argument("--min-multi-contract-accuracy", type=float, default=0.80)
    parser.add_argument("--min-multi-contract-per-contract", type=float, default=0.75)
    parser.add_argument("--min-paraphrase-family-pass-rate", type=float, default=0.85)
    parser.add_argument("--min-paraphrase-worker-slang-pass-rate", type=float, default=0.80)
    parser.add_argument("--min-paraphrase-formal-rewrite-pass-rate", type=float, default=0.90)
    parser.add_argument("--required-adversarial-dataset-schema-version", default="adversarial_precedence_test_v1")
    parser.add_argument("--min-adversarial-total-cases", type=int, default=12)
    parser.add_argument("--min-adversarial-cases-per-contract", type=int, default=3)
    parser.add_argument("--min-adversarial-precedence-cases", type=int, default=4)
    parser.add_argument("--min-adversarial-pass-rate", type=float, default=0.90)
    parser.add_argument("--min-adversarial-per-contract", type=float, default=0.80)
    parser.add_argument("--min-adversarial-precedence-pass-rate", type=float, default=0.90)
    parser.add_argument("--required-unanswerable-dataset-schema-version", default="unanswerable_multi_contract_test_v1")
    parser.add_argument("--min-unanswerable-total-cases", type=int, default=12)
    parser.add_argument("--min-unanswerable-cases-per-contract", type=int, default=3)
    parser.add_argument("--min-unanswerable-scenario-types", type=int, default=3)
    parser.add_argument("--min-unanswerable-pass-rate", type=float, default=0.90)
    parser.add_argument("--min-unanswerable-per-contract", type=float, default=0.80)
    parser.add_argument("--required-cross-contract-mentions-dataset-schema-version", default="cross_contract_mentions_test_v1")
    parser.add_argument("--min-cross-contract-mentions-total-cases", type=int, default=9)
    parser.add_argument("--min-cross-contract-mentions-cases-per-contract", type=int, default=3)
    parser.add_argument("--min-cross-contract-mentions-pass-rate", type=float, default=0.90)
    parser.add_argument("--min-cross-contract-mentions-per-contract", type=float, default=0.80)
    parser.add_argument("--min-cross-contract-mentions-no-citation-rate", type=float, default=0.90)
    parser.add_argument("--required-false-unavailable-dataset-schema-version", default="false_unavailable_test_v1")
    parser.add_argument("--min-false-unavailable-total-cases", type=int, default=12)
    parser.add_argument("--min-false-unavailable-recover-cases", type=int, default=9)
    parser.add_argument("--min-false-unavailable-uncertain-cases", type=int, default=3)
    parser.add_argument("--min-false-unavailable-cases-per-contract", type=int, default=3)
    parser.add_argument("--min-false-unavailable-pass-rate", type=float, default=0.90)
    parser.add_argument("--min-false-unavailable-per-contract", type=float, default=0.80)
    parser.add_argument("--min-false-unavailable-recovered-rate", type=float, default=0.90)
    parser.add_argument("--min-false-unavailable-proper-uncertainty-rate", type=float, default=0.90)
    parser.add_argument("--min-needle-pass-rate", type=float, default=0.80)
    parser.add_argument("--min-needle-position-pass-rate", type=float, default=0.80)
    parser.add_argument("--required-wage-table-evidence-dataset-schema-version", default="wage_table_evidence_test_v1")
    parser.add_argument("--min-wage-table-evidence-total-cases", type=int, default=12)
    parser.add_argument("--min-wage-table-evidence-cases-per-contract", type=int, default=3)
    parser.add_argument("--min-wage-table-evidence-pass-rate", type=float, default=0.90)
    parser.add_argument("--min-wage-table-evidence-per-contract", type=float, default=0.80)
    parser.add_argument("--min-wage-table-evidence-source-method-pass-rate", type=float, default=0.95)
    parser.add_argument("--min-wage-table-evidence-presence-rate", type=float, default=0.95)
    parser.add_argument("--min-wage-table-evidence-table-id-presence-rate", type=float, default=0.95)
    parser.add_argument("--required-entitlement-table-evidence-dataset-schema-version", default="entitlement_table_evidence_test_v1")
    parser.add_argument("--min-entitlement-table-evidence-total-cases", type=int, default=12)
    parser.add_argument("--min-entitlement-table-evidence-cases-per-contract", type=int, default=3)
    parser.add_argument("--min-entitlement-table-evidence-pass-rate", type=float, default=0.90)
    parser.add_argument("--min-entitlement-table-evidence-per-contract", type=float, default=0.80)
    parser.add_argument("--min-entitlement-table-evidence-weeks-resolution-pass-rate", type=float, default=0.90)
    parser.add_argument("--min-entitlement-table-evidence-source-method-pass-rate", type=float, default=0.95)
    parser.add_argument("--min-entitlement-table-evidence-presence-rate", type=float, default=0.95)
    parser.add_argument("--required-role-catalog-integrity-dataset-schema-version", default="role_catalog_integrity_test_v1")
    parser.add_argument("--min-role-catalog-integrity-total-cases", type=int, default=12)
    parser.add_argument("--min-role-catalog-integrity-cases-per-contract", type=int, default=3)
    parser.add_argument("--min-role-catalog-integrity-pass-rate", type=float, default=0.95)
    parser.add_argument("--min-role-catalog-integrity-per-contract", type=float, default=0.90)
    parser.add_argument("--min-role-catalog-integrity-dataset-case-pass-rate", type=float, default=0.95)
    parser.add_argument("--min-role-catalog-integrity-default-wage-ready-rate", type=float, default=1.0)
    parser.add_argument("--min-role-catalog-integrity-unresolved-not-default-rate", type=float, default=1.0)
    parser.add_argument("--min-role-catalog-integrity-default-wage-key-unique-rate", type=float, default=1.0)
    parser.add_argument("--required-followup-role-wage-dataset-schema-version", default="followup_role_wage_test_v1")
    parser.add_argument("--min-followup-role-wage-total-cases", type=int, default=12)
    parser.add_argument("--min-followup-role-wage-cases-per-contract", type=int, default=3)
    parser.add_argument("--min-followup-role-wage-pass-rate", type=float, default=0.90)
    parser.add_argument("--min-followup-role-wage-per-contract", type=float, default=0.80)
    parser.add_argument("--min-followup-role-wage-target-resolution-rate", type=float, default=0.95)
    parser.add_argument("--min-followup-role-wage-table-evidence-presence-rate", type=float, default=0.95)
    parser.add_argument("--min-followup-role-wage-appendix-citation-rate", type=float, default=0.95)
    parser.add_argument("--min-followup-role-wage-intent-wage-rate", type=float, default=0.95)
    parser.add_argument("--min-followup-role-wage-no-unavailable-rate", type=float, default=0.95)
    parser.add_argument("--min-followup-role-wage-explicit-override-rate", type=float, default=0.90)
    parser.add_argument("--min-followup-role-wage-profile-fallback-rate", type=float, default=0.90)
    parser.add_argument("--required-moa-deleted-vs-updated-schema-version", default="moa_deleted_vs_updated_eval_v1")
    parser.add_argument("--required-moa-deleted-vs-updated-dataset-schema-version", default="moa_deleted_vs_updated_test_v1")
    parser.add_argument("--min-moa-deleted-vs-updated-overall-pass-rate", type=float, default=1.0)
    parser.add_argument("--min-moa-deleted-vs-updated-updated-pass-rate", type=float, default=1.0)
    parser.add_argument("--min-moa-deleted-vs-updated-deleted-pass-rate", type=float, default=1.0)
    parser.add_argument("--min-moa-deleted-vs-updated-updated-moa-source-type-match-rate", type=float, default=1.0)
    parser.add_argument("--required-release-090-schema-version", default="release_090_scorecard_v1")
    parser.add_argument("--min-release-090-components-pass-rate", type=float, default=1.0)
    parser.add_argument(
        "--allow-missing-cross-contamination",
        action="store_true",
        help="Allow missing cross-contamination artifact (non-release debugging only).",
    )
    parser.add_argument(
        "--allow-missing-v3",
        action="store_true",
        help="Allow missing v3 artifact (non-release debugging only).",
    )
    parser.add_argument(
        "--allow-missing-multi-contract",
        action="store_true",
        help="Allow missing multi-contract artifact (non-release debugging only).",
    )
    parser.add_argument(
        "--allow-missing-paraphrase",
        action="store_true",
        help="Allow missing paraphrase artifact (non-release debugging only).",
    )
    parser.add_argument(
        "--allow-missing-adversarial",
        action="store_true",
        help="Allow missing adversarial artifact (non-release debugging only).",
    )
    parser.add_argument(
        "--allow-missing-unanswerable",
        action="store_true",
        help="Allow missing unanswerable artifact (non-release debugging only).",
    )
    parser.add_argument(
        "--allow-missing-cross-contract-mentions",
        action="store_true",
        help="Allow missing cross-contract-mentions artifact (non-release debugging only).",
    )
    parser.add_argument(
        "--allow-missing-needle",
        action="store_true",
        help="Allow missing needle artifact (non-release debugging only).",
    )
    parser.add_argument(
        "--allow-missing-false-unavailable",
        action="store_true",
        help="Allow missing false-unavailable artifact (non-release debugging only).",
    )
    parser.add_argument(
        "--allow-missing-wage-table-evidence",
        action="store_true",
        help="Allow missing wage-table-evidence artifact (non-release debugging only).",
    )
    parser.add_argument(
        "--allow-missing-entitlement-table-evidence",
        action="store_true",
        help="Allow missing entitlement-table-evidence artifact (non-release debugging only).",
    )
    parser.add_argument(
        "--allow-missing-role-catalog-integrity",
        action="store_true",
        help="Allow missing role-catalog-integrity artifact (non-release debugging only).",
    )
    parser.add_argument(
        "--allow-missing-followup-role-wage",
        action="store_true",
        help="Allow missing followup-role-wage artifact (non-release debugging only).",
    )
    parser.add_argument(
        "--allow-missing-moa-deleted-vs-updated",
        action="store_true",
        help="Allow missing moa_deleted_vs_updated artifact (non-release debugging only).",
    )
    parser.add_argument(
        "--allow-missing-release-090",
        action="store_true",
        help="Allow missing release_090 scorecard artifact (non-release debugging only).",
    )
    args = parser.parse_args()

    failures: list[str] = []
    print("=" * 72)
    print("KARL Release-Gate Check")
    print("=" * 72)

    try:
        v2 = _load_json(args.v2_results)
        ok, msg = _check_v2(v2, args.min_v2_accuracy)
        print(f"[{'OK' if ok else 'XX'}] {msg}")
        if not ok:
            failures.append(msg)
    except Exception as e:
        msg = f"v2 check error: {e}"
        print(f"[XX] {msg}")
        failures.append(msg)

    try:
        esc = _load_json(args.escalation_results)
        for ok, msg in _check_escalation(
            esc,
            min_precision=args.min_escalation_precision,
            min_recall=args.min_escalation_recall,
            max_fpr=args.max_escalation_fpr,
        ):
            print(f"[{'OK' if ok else 'XX'}] {msg}")
            if not ok:
                failures.append(msg)
    except Exception as e:
        msg = f"escalation check error: {e}"
        print(f"[XX] {msg}")
        failures.append(msg)

    cross_path = Path(args.cross_contamination_results)
    if cross_path.exists():
        try:
            cross = _load_json(args.cross_contamination_results)
            for ok, msg in _check_cross_contamination(cross):
                print(f"[{'OK' if ok else 'XX'}] {msg}")
                if not ok:
                    failures.append(msg)
        except Exception as e:
            msg = f"cross-contamination check error: {e}"
            print(f"[XX] {msg}")
            failures.append(msg)
    else:
        msg = f"cross-contamination artifact missing: {args.cross_contamination_results}"
        if args.allow_missing_cross_contamination:
            print(f"[--] {msg}")
        else:
            print(f"[XX] {msg}")
            failures.append(msg)

    v3_path = Path(args.v3_results)
    if v3_path.exists():
        try:
            v3 = _load_json(args.v3_results)
            for ok, msg in _check_v3(v3, min_components_pass_rate=args.min_v3_components_pass_rate):
                print(f"[{'OK' if ok else 'XX'}] {msg}")
                if not ok:
                    failures.append(msg)
        except Exception as e:
            msg = f"v3 check error: {e}"
            print(f"[XX] {msg}")
            failures.append(msg)
    else:
        msg = f"v3 artifact missing: {args.v3_results}"
        if args.allow_missing_v3:
            print(f"[--] {msg}")
        else:
            print(f"[XX] {msg}")
            failures.append(msg)

    multi_path = Path(args.multi_contract_results)
    if multi_path.exists():
        try:
            multi = _load_json(args.multi_contract_results)
            for ok, msg in _check_multi_contract(
                multi,
                min_overall_accuracy=args.min_multi_contract_accuracy,
                min_per_contract_accuracy=args.min_multi_contract_per_contract,
            ):
                print(f"[{'OK' if ok else 'XX'}] {msg}")
                if not ok:
                    failures.append(msg)
        except Exception as e:
            msg = f"multi-contract check error: {e}"
            print(f"[XX] {msg}")
            failures.append(msg)
    else:
        msg = f"multi-contract artifact missing: {args.multi_contract_results}"
        if args.allow_missing_multi_contract:
            print(f"[--] {msg}")
        else:
            print(f"[XX] {msg}")
            failures.append(msg)

    paraphrase_path = Path(args.paraphrase_results)
    if paraphrase_path.exists():
        try:
            paraphrase = _load_json(args.paraphrase_results)
            for ok, msg in _check_paraphrase(
                paraphrase,
                min_family_pass_rate=args.min_paraphrase_family_pass_rate,
                min_worker_slang_pass_rate=args.min_paraphrase_worker_slang_pass_rate,
                min_formal_rewrite_pass_rate=args.min_paraphrase_formal_rewrite_pass_rate,
            ):
                print(f"[{'OK' if ok else 'XX'}] {msg}")
                if not ok:
                    failures.append(msg)
        except Exception as e:
            msg = f"paraphrase check error: {e}"
            print(f"[XX] {msg}")
            failures.append(msg)
    else:
        msg = f"paraphrase artifact missing: {args.paraphrase_results}"
        if args.allow_missing_paraphrase:
            print(f"[--] {msg}")
        else:
            print(f"[XX] {msg}")
            failures.append(msg)

    adversarial_path = Path(args.adversarial_results)
    if adversarial_path.exists():
        try:
            adversarial = _load_json(args.adversarial_results)
            for ok, msg in _check_adversarial(
                adversarial,
                required_dataset_schema_version=args.required_adversarial_dataset_schema_version,
                min_total_cases=args.min_adversarial_total_cases,
                min_cases_per_contract=args.min_adversarial_cases_per_contract,
                min_pass_rate=args.min_adversarial_pass_rate,
                min_per_contract_pass_rate=args.min_adversarial_per_contract,
                min_precedence_pass_rate=args.min_adversarial_precedence_pass_rate,
            ):
                print(f"[{'OK' if ok else 'XX'}] {msg}")
                if not ok:
                    failures.append(msg)
        except Exception as e:
            msg = f"adversarial check error: {e}"
            print(f"[XX] {msg}")
            failures.append(msg)
    else:
        msg = f"adversarial artifact missing: {args.adversarial_results}"
        if args.allow_missing_adversarial:
            print(f"[--] {msg}")
        else:
            print(f"[XX] {msg}")
            failures.append(msg)

    adversarial_dataset_path = Path(args.adversarial_dataset)
    if adversarial_dataset_path.exists():
        try:
            ok, issues, summary = validate_adversarial_dataset_run(
                dataset_path=adversarial_dataset_path,
                required_schema_version=args.required_adversarial_dataset_schema_version,
                min_total_cases=args.min_adversarial_total_cases,
                min_cases_per_contract=args.min_adversarial_cases_per_contract,
                min_precedence_cases=args.min_adversarial_precedence_cases,
            )
            schema = str(summary.get("schema_version") or "")
            total = int(summary.get("total_cases") or 0)
            precedence_cases = int(summary.get("precedence_cases") or 0)
            print(
                f"[{'OK' if schema == args.required_adversarial_dataset_schema_version else 'XX'}] "
                "adversarial dataset schema_version="
                f"{schema or '<missing>'} expected={args.required_adversarial_dataset_schema_version}"
            )
            if schema != args.required_adversarial_dataset_schema_version:
                failures.append(
                    "adversarial dataset schema_version="
                    f"{schema or '<missing>'} expected={args.required_adversarial_dataset_schema_version}"
                )
            print(
                f"[{'OK' if total >= args.min_adversarial_total_cases else 'XX'}] "
                f"adversarial dataset total_cases={total} threshold>={args.min_adversarial_total_cases}"
            )
            if total < args.min_adversarial_total_cases:
                failures.append(
                    f"adversarial dataset total_cases={total} threshold>={args.min_adversarial_total_cases}"
                )
            print(
                f"[{'OK' if precedence_cases >= args.min_adversarial_precedence_cases else 'XX'}] "
                "adversarial dataset precedence_cases="
                f"{precedence_cases} threshold>={args.min_adversarial_precedence_cases}"
            )
            if precedence_cases < args.min_adversarial_precedence_cases:
                failures.append(
                    "adversarial dataset precedence_cases="
                    f"{precedence_cases} threshold>={args.min_adversarial_precedence_cases}"
                )

            by_contract = summary.get("by_contract_total") or {}
            for contract_id, total_cases in sorted(by_contract.items()):
                count = int(total_cases or 0)
                ok_case = count >= args.min_adversarial_cases_per_contract
                print(
                    f"[{'OK' if ok_case else 'XX'}] adversarial dataset {contract_id} total={count} "
                    f"threshold>={args.min_adversarial_cases_per_contract}"
                )
                if not ok_case:
                    failures.append(
                        f"adversarial dataset {contract_id} total={count} "
                        f"threshold>={args.min_adversarial_cases_per_contract}"
                    )

            if not ok:
                for issue in issues:
                    msg = f"adversarial dataset issue: {issue}"
                    print(f"[XX] {msg}")
                    failures.append(msg)
        except Exception as e:
            msg = f"adversarial dataset validation error: {e}"
            print(f"[XX] {msg}")
            failures.append(msg)
    else:
        msg = f"adversarial dataset missing: {args.adversarial_dataset}"
        print(f"[XX] {msg}")
        failures.append(msg)

    unanswerable_path = Path(args.unanswerable_results)
    if unanswerable_path.exists():
        try:
            unanswerable = _load_json(args.unanswerable_results)
            for ok, msg in _check_unanswerable(
                unanswerable,
                required_dataset_schema_version=args.required_unanswerable_dataset_schema_version,
                min_total_cases=args.min_unanswerable_total_cases,
                min_cases_per_contract=args.min_unanswerable_cases_per_contract,
                min_pass_rate=args.min_unanswerable_pass_rate,
                min_per_contract_pass_rate=args.min_unanswerable_per_contract,
            ):
                print(f"[{'OK' if ok else 'XX'}] {msg}")
                if not ok:
                    failures.append(msg)
        except Exception as e:
            msg = f"unanswerable check error: {e}"
            print(f"[XX] {msg}")
            failures.append(msg)
    else:
        msg = f"unanswerable artifact missing: {args.unanswerable_results}"
        if args.allow_missing_unanswerable:
            print(f"[--] {msg}")
        else:
            print(f"[XX] {msg}")
            failures.append(msg)

    unanswerable_dataset_path = Path(args.unanswerable_dataset)
    if unanswerable_dataset_path.exists():
        try:
            ok, issues, summary = validate_unanswerable_dataset_run(
                dataset_path=unanswerable_dataset_path,
                required_schema_version=args.required_unanswerable_dataset_schema_version,
                min_total_cases=args.min_unanswerable_total_cases,
                min_cases_per_contract=args.min_unanswerable_cases_per_contract,
                min_scenario_types=args.min_unanswerable_scenario_types,
            )
            schema = str(summary.get("schema_version") or "")
            total = int(summary.get("total_cases") or 0)
            scenario_types = len(summary.get("scenario_counts") or {})
            print(
                f"[{'OK' if schema == args.required_unanswerable_dataset_schema_version else 'XX'}] "
                "unanswerable dataset schema_version="
                f"{schema or '<missing>'} expected={args.required_unanswerable_dataset_schema_version}"
            )
            if schema != args.required_unanswerable_dataset_schema_version:
                failures.append(
                    "unanswerable dataset schema_version="
                    f"{schema or '<missing>'} expected={args.required_unanswerable_dataset_schema_version}"
                )
            print(
                f"[{'OK' if total >= args.min_unanswerable_total_cases else 'XX'}] "
                f"unanswerable dataset total_cases={total} threshold>={args.min_unanswerable_total_cases}"
            )
            if total < args.min_unanswerable_total_cases:
                failures.append(
                    f"unanswerable dataset total_cases={total} threshold>={args.min_unanswerable_total_cases}"
                )
            print(
                f"[{'OK' if scenario_types >= args.min_unanswerable_scenario_types else 'XX'}] "
                "unanswerable dataset scenario_types="
                f"{scenario_types} threshold>={args.min_unanswerable_scenario_types}"
            )
            if scenario_types < args.min_unanswerable_scenario_types:
                failures.append(
                    "unanswerable dataset scenario_types="
                    f"{scenario_types} threshold>={args.min_unanswerable_scenario_types}"
                )
            by_contract = summary.get("by_contract_total") or {}
            for contract_id, total_cases in sorted(by_contract.items()):
                count = int(total_cases or 0)
                ok_case = count >= args.min_unanswerable_cases_per_contract
                print(
                    f"[{'OK' if ok_case else 'XX'}] unanswerable dataset {contract_id} total={count} "
                    f"threshold>={args.min_unanswerable_cases_per_contract}"
                )
                if not ok_case:
                    failures.append(
                        f"unanswerable dataset {contract_id} total={count} "
                        f"threshold>={args.min_unanswerable_cases_per_contract}"
                    )
            if not ok:
                for issue in issues:
                    msg = f"unanswerable dataset issue: {issue}"
                    print(f"[XX] {msg}")
                    failures.append(msg)
        except Exception as e:
            msg = f"unanswerable dataset validation error: {e}"
            print(f"[XX] {msg}")
            failures.append(msg)
    else:
        msg = f"unanswerable dataset missing: {args.unanswerable_dataset}"
        print(f"[XX] {msg}")
        failures.append(msg)

    ccm_path = Path(args.cross_contract_mentions_results)
    if ccm_path.exists():
        try:
            ccm = _load_json(args.cross_contract_mentions_results)
            for ok, msg in _check_cross_contract_mentions(
                ccm,
                required_dataset_schema_version=args.required_cross_contract_mentions_dataset_schema_version,
                min_total_cases=args.min_cross_contract_mentions_total_cases,
                min_cases_per_contract=args.min_cross_contract_mentions_cases_per_contract,
                min_pass_rate=args.min_cross_contract_mentions_pass_rate,
                min_per_contract_pass_rate=args.min_cross_contract_mentions_per_contract,
                min_no_citation_rate=args.min_cross_contract_mentions_no_citation_rate,
            ):
                print(f"[{'OK' if ok else 'XX'}] {msg}")
                if not ok:
                    failures.append(msg)
        except Exception as e:
            msg = f"cross-contract-mentions check error: {e}"
            print(f"[XX] {msg}")
            failures.append(msg)
    else:
        msg = f"cross-contract-mentions artifact missing: {args.cross_contract_mentions_results}"
        if args.allow_missing_cross_contract_mentions:
            print(f"[--] {msg}")
        else:
            print(f"[XX] {msg}")
            failures.append(msg)

    ccm_dataset_path = Path(args.cross_contract_mentions_dataset)
    if ccm_dataset_path.exists():
        try:
            ok, issues, summary = validate_cross_contract_mentions_dataset_run(
                dataset_path=ccm_dataset_path,
                required_schema_version=args.required_cross_contract_mentions_dataset_schema_version,
                min_total_cases=args.min_cross_contract_mentions_total_cases,
                min_cases_per_contract=args.min_cross_contract_mentions_cases_per_contract,
            )
            schema = str(summary.get("schema_version") or "")
            total = int(summary.get("total_cases") or 0)
            print(
                f"[{'OK' if schema == args.required_cross_contract_mentions_dataset_schema_version else 'XX'}] "
                "cross-contract-mentions dataset schema_version="
                f"{schema or '<missing>'} expected={args.required_cross_contract_mentions_dataset_schema_version}"
            )
            if schema != args.required_cross_contract_mentions_dataset_schema_version:
                failures.append(
                    "cross-contract-mentions dataset schema_version="
                    f"{schema or '<missing>'} expected={args.required_cross_contract_mentions_dataset_schema_version}"
                )
            print(
                f"[{'OK' if total >= args.min_cross_contract_mentions_total_cases else 'XX'}] "
                "cross-contract-mentions dataset total_cases="
                f"{total} threshold>={args.min_cross_contract_mentions_total_cases}"
            )
            if total < args.min_cross_contract_mentions_total_cases:
                failures.append(
                    "cross-contract-mentions dataset total_cases="
                    f"{total} threshold>={args.min_cross_contract_mentions_total_cases}"
                )

            by_contract = summary.get("by_contract_total") or {}
            for contract_id, total_cases in sorted(by_contract.items()):
                count = int(total_cases or 0)
                ok_case = count >= args.min_cross_contract_mentions_cases_per_contract
                print(
                    f"[{'OK' if ok_case else 'XX'}] cross-contract-mentions dataset {contract_id} total={count} "
                    f"threshold>={args.min_cross_contract_mentions_cases_per_contract}"
                )
                if not ok_case:
                    failures.append(
                        f"cross-contract-mentions dataset {contract_id} total={count} "
                        f"threshold>={args.min_cross_contract_mentions_cases_per_contract}"
                    )
            if not ok:
                for issue in issues:
                    msg = f"cross-contract-mentions dataset issue: {issue}"
                    print(f"[XX] {msg}")
                    failures.append(msg)
        except Exception as e:
            msg = f"cross-contract-mentions dataset validation error: {e}"
            print(f"[XX] {msg}")
            failures.append(msg)
    else:
        msg = f"cross-contract-mentions dataset missing: {args.cross_contract_mentions_dataset}"
        print(f"[XX] {msg}")
        failures.append(msg)

    false_unavailable_path = Path(args.false_unavailable_results)
    if false_unavailable_path.exists():
        try:
            false_unavailable = _load_json(args.false_unavailable_results)
            for ok, msg in _check_false_unavailable(
                false_unavailable,
                required_dataset_schema_version=args.required_false_unavailable_dataset_schema_version,
                min_total_cases=args.min_false_unavailable_total_cases,
                min_recover_cases=args.min_false_unavailable_recover_cases,
                min_uncertain_cases=args.min_false_unavailable_uncertain_cases,
                min_cases_per_contract=args.min_false_unavailable_cases_per_contract,
                min_pass_rate=args.min_false_unavailable_pass_rate,
                min_per_contract_pass_rate=args.min_false_unavailable_per_contract,
                min_recovered_rate=args.min_false_unavailable_recovered_rate,
                min_proper_uncertainty_rate=args.min_false_unavailable_proper_uncertainty_rate,
            ):
                print(f"[{'OK' if ok else 'XX'}] {msg}")
                if not ok:
                    failures.append(msg)
        except Exception as e:
            msg = f"false-unavailable check error: {e}"
            print(f"[XX] {msg}")
            failures.append(msg)
    else:
        msg = f"false-unavailable artifact missing: {args.false_unavailable_results}"
        if args.allow_missing_false_unavailable:
            print(f"[--] {msg}")
        else:
            print(f"[XX] {msg}")
            failures.append(msg)

    needle_path = Path(args.needle_results)
    if needle_path.exists():
        try:
            needle = _load_json(args.needle_results)
            for ok, msg in _check_needle(
                needle,
                min_pass_rate=args.min_needle_pass_rate,
                min_position_pass_rate=args.min_needle_position_pass_rate,
            ):
                print(f"[{'OK' if ok else 'XX'}] {msg}")
                if not ok:
                    failures.append(msg)
        except Exception as e:
            msg = f"needle check error: {e}"
            print(f"[XX] {msg}")
            failures.append(msg)
    else:
        msg = f"needle artifact missing: {args.needle_results}"
        if args.allow_missing_needle:
            print(f"[--] {msg}")
        else:
            print(f"[XX] {msg}")
            failures.append(msg)

    wage_table_path = Path(args.wage_table_evidence_results)
    if wage_table_path.exists():
        try:
            wage_table = _load_json(args.wage_table_evidence_results)
            for ok, msg in _check_wage_table_evidence(
                wage_table,
                required_dataset_schema_version=args.required_wage_table_evidence_dataset_schema_version,
                min_total_cases=args.min_wage_table_evidence_total_cases,
                min_cases_per_contract=args.min_wage_table_evidence_cases_per_contract,
                min_pass_rate=args.min_wage_table_evidence_pass_rate,
                min_per_contract_pass_rate=args.min_wage_table_evidence_per_contract,
                min_source_method_pass_rate=args.min_wage_table_evidence_source_method_pass_rate,
                min_table_evidence_presence_rate=args.min_wage_table_evidence_presence_rate,
                min_table_id_presence_rate=args.min_wage_table_evidence_table_id_presence_rate,
            ):
                print(f"[{'OK' if ok else 'XX'}] {msg}")
                if not ok:
                    failures.append(msg)
        except Exception as e:
            msg = f"wage-table-evidence check error: {e}"
            print(f"[XX] {msg}")
            failures.append(msg)
    else:
        msg = f"wage-table-evidence artifact missing: {args.wage_table_evidence_results}"
        if args.allow_missing_wage_table_evidence:
            print(f"[--] {msg}")
        else:
            print(f"[XX] {msg}")
            failures.append(msg)

    entitlement_table_path = Path(args.entitlement_table_evidence_results)
    if entitlement_table_path.exists():
        try:
            entitlement_table = _load_json(args.entitlement_table_evidence_results)
            for ok, msg in _check_entitlement_table_evidence(
                entitlement_table,
                required_dataset_schema_version=args.required_entitlement_table_evidence_dataset_schema_version,
                min_total_cases=args.min_entitlement_table_evidence_total_cases,
                min_cases_per_contract=args.min_entitlement_table_evidence_cases_per_contract,
                min_pass_rate=args.min_entitlement_table_evidence_pass_rate,
                min_per_contract_pass_rate=args.min_entitlement_table_evidence_per_contract,
                min_weeks_resolution_pass_rate=args.min_entitlement_table_evidence_weeks_resolution_pass_rate,
                min_source_method_pass_rate=args.min_entitlement_table_evidence_source_method_pass_rate,
                min_evidence_presence_rate=args.min_entitlement_table_evidence_presence_rate,
            ):
                print(f"[{'OK' if ok else 'XX'}] {msg}")
                if not ok:
                    failures.append(msg)
        except Exception as e:
            msg = f"entitlement-table-evidence check error: {e}"
            print(f"[XX] {msg}")
            failures.append(msg)
    else:
        msg = f"entitlement-table-evidence artifact missing: {args.entitlement_table_evidence_results}"
        if args.allow_missing_entitlement_table_evidence:
            print(f"[--] {msg}")
        else:
            print(f"[XX] {msg}")
            failures.append(msg)

    role_integrity_path = Path(args.role_catalog_integrity_results)
    if role_integrity_path.exists():
        try:
            role_integrity = _load_json(args.role_catalog_integrity_results)
            for ok, msg in _check_role_catalog_integrity(
                role_integrity,
                required_dataset_schema_version=args.required_role_catalog_integrity_dataset_schema_version,
                min_total_cases=args.min_role_catalog_integrity_total_cases,
                min_cases_per_contract=args.min_role_catalog_integrity_cases_per_contract,
                min_pass_rate=args.min_role_catalog_integrity_pass_rate,
                min_per_contract_pass_rate=args.min_role_catalog_integrity_per_contract,
                min_dataset_case_pass_rate=args.min_role_catalog_integrity_dataset_case_pass_rate,
                min_default_wage_ready_rate=args.min_role_catalog_integrity_default_wage_ready_rate,
                min_unresolved_not_default_rate=args.min_role_catalog_integrity_unresolved_not_default_rate,
                min_default_wage_key_unique_rate=args.min_role_catalog_integrity_default_wage_key_unique_rate,
            ):
                print(f"[{'OK' if ok else 'XX'}] {msg}")
                if not ok:
                    failures.append(msg)
        except Exception as e:
            msg = f"role-catalog-integrity check error: {e}"
            print(f"[XX] {msg}")
            failures.append(msg)
    else:
        msg = f"role-catalog-integrity artifact missing: {args.role_catalog_integrity_results}"
        if args.allow_missing_role_catalog_integrity:
            print(f"[--] {msg}")
        else:
            print(f"[XX] {msg}")
            failures.append(msg)

    followup_role_wage_path = Path(args.followup_role_wage_results)
    if followup_role_wage_path.exists():
        try:
            followup_role_wage = _load_json(args.followup_role_wage_results)
            for ok, msg in _check_followup_role_wage(
                followup_role_wage,
                required_dataset_schema_version=args.required_followup_role_wage_dataset_schema_version,
                min_total_cases=args.min_followup_role_wage_total_cases,
                min_cases_per_contract=args.min_followup_role_wage_cases_per_contract,
                min_pass_rate=args.min_followup_role_wage_pass_rate,
                min_per_contract_pass_rate=args.min_followup_role_wage_per_contract,
                min_target_resolution_rate=args.min_followup_role_wage_target_resolution_rate,
                min_table_evidence_presence_rate=args.min_followup_role_wage_table_evidence_presence_rate,
                min_appendix_citation_rate=args.min_followup_role_wage_appendix_citation_rate,
                min_intent_wage_rate=args.min_followup_role_wage_intent_wage_rate,
                min_no_unavailable_rate=args.min_followup_role_wage_no_unavailable_rate,
                min_explicit_override_rate=args.min_followup_role_wage_explicit_override_rate,
                min_profile_fallback_rate=args.min_followup_role_wage_profile_fallback_rate,
            ):
                print(f"[{'OK' if ok else 'XX'}] {msg}")
                if not ok:
                    failures.append(msg)
        except Exception as e:
            msg = f"followup-role-wage check error: {e}"
            print(f"[XX] {msg}")
            failures.append(msg)
    else:
        msg = f"followup-role-wage artifact missing: {args.followup_role_wage_results}"
        if args.allow_missing_followup_role_wage:
            print(f"[--] {msg}")
        else:
            print(f"[XX] {msg}")
            failures.append(msg)

    moa_delupd_path = Path(args.moa_deleted_vs_updated_results)
    if moa_delupd_path.exists():
        try:
            moa_delupd = _load_json(args.moa_deleted_vs_updated_results)
            for ok, msg in _check_moa_deleted_vs_updated(
                moa_delupd,
                required_schema_version=args.required_moa_deleted_vs_updated_schema_version,
                required_dataset_schema_version=args.required_moa_deleted_vs_updated_dataset_schema_version,
                min_overall_pass_rate=args.min_moa_deleted_vs_updated_overall_pass_rate,
                min_updated_pass_rate=args.min_moa_deleted_vs_updated_updated_pass_rate,
                min_deleted_pass_rate=args.min_moa_deleted_vs_updated_deleted_pass_rate,
                min_updated_moa_source_type_match_rate=args.min_moa_deleted_vs_updated_updated_moa_source_type_match_rate,
            ):
                print(f"[{'OK' if ok else 'XX'}] {msg}")
                if not ok:
                    failures.append(msg)
        except Exception as e:
            msg = f"moa_deleted_vs_updated check error: {e}"
            print(f"[XX] {msg}")
            failures.append(msg)
    else:
        msg = f"moa_deleted_vs_updated artifact missing: {args.moa_deleted_vs_updated_results}"
        if args.allow_missing_moa_deleted_vs_updated:
            print(f"[--] {msg}")
        else:
            print(f"[XX] {msg}")
            failures.append(msg)

    release_090_path = Path(args.release_090_results)
    if release_090_path.exists():
        try:
            release_090 = _load_json(args.release_090_results)
            for ok, msg in _check_release_090(
                release_090,
                required_schema_version=args.required_release_090_schema_version,
                min_components_pass_rate=args.min_release_090_components_pass_rate,
            ):
                print(f"[{'OK' if ok else 'XX'}] {msg}")
                if not ok:
                    failures.append(msg)
            for warn, msg in _release_090_advisory_messages(release_090):
                print(f"[{'!!' if warn else 'OK'}] {msg}")
        except Exception as e:
            msg = f"release_090 check error: {e}"
            print(f"[XX] {msg}")
            failures.append(msg)
    else:
        msg = f"release_090 artifact missing: {args.release_090_results}"
        if args.allow_missing_release_090:
            print(f"[--] {msg}")
        else:
            print(f"[XX] {msg}")
            failures.append(msg)

    if failures:
        print("\nGate status: BLOCKED")
        for f in failures:
            print(f"- {f}")
        sys.exit(1)

    print("\nGate status: PASS")


if __name__ == "__main__":
    main()
