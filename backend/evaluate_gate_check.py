"""
Release-gate checker for evaluation artifacts.

Fails (exit code 1) if thresholds are not met.
"""

import argparse
import json
import sys
from pathlib import Path


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


def _check_paraphrase(
    results: dict,
    min_family_pass_rate: float,
    min_worker_slang_pass_rate: float,
) -> list[tuple[bool, str]]:
    checks: list[tuple[bool, str]] = []
    overall = results.get("overall", {}) or {}
    family_rate = overall.get("family_pass_rate")
    slang_rate = overall.get("worker_slang_pass_rate")

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


def main():
    parser = argparse.ArgumentParser(description="Check KARL release-gate thresholds from evaluation artifacts.")
    parser.add_argument("--v2-results", default="data/test_set/comprehensive_results.json")
    parser.add_argument("--escalation-results", default="data/test_set/escalation_precision_results.json")
    parser.add_argument("--multi-contract-results", default="data/test_set/multi_contract_v2_results.json")
    parser.add_argument("--paraphrase-results", default="data/test_set/paraphrase_results.json")
    parser.add_argument("--needle-results", default="data/test_set/needle_results.json")

    parser.add_argument("--min-v2-accuracy", type=float, default=0.80)
    parser.add_argument("--min-escalation-precision", type=float, default=0.90)
    parser.add_argument("--min-escalation-recall", type=float, default=0.70)
    parser.add_argument("--max-escalation-fpr", type=float, default=0.10)
    parser.add_argument("--min-multi-contract-accuracy", type=float, default=0.80)
    parser.add_argument("--min-multi-contract-per-contract", type=float, default=0.75)
    parser.add_argument("--min-paraphrase-family-pass-rate", type=float, default=0.85)
    parser.add_argument("--min-paraphrase-worker-slang-pass-rate", type=float, default=0.80)
    parser.add_argument("--min-needle-pass-rate", type=float, default=0.80)
    parser.add_argument("--min-needle-position-pass-rate", type=float, default=0.80)
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
        print(f"[--] multi-contract check skipped (missing {args.multi_contract_results})")

    paraphrase_path = Path(args.paraphrase_results)
    if paraphrase_path.exists():
        try:
            paraphrase = _load_json(args.paraphrase_results)
            for ok, msg in _check_paraphrase(
                paraphrase,
                min_family_pass_rate=args.min_paraphrase_family_pass_rate,
                min_worker_slang_pass_rate=args.min_paraphrase_worker_slang_pass_rate,
            ):
                print(f"[{'OK' if ok else 'XX'}] {msg}")
                if not ok:
                    failures.append(msg)
        except Exception as e:
            msg = f"paraphrase check error: {e}"
            print(f"[XX] {msg}")
            failures.append(msg)
    else:
        print(f"[--] paraphrase check skipped (missing {args.paraphrase_results})")

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

    if failures:
        print("\nGate status: BLOCKED")
        for f in failures:
            print(f"- {f}")
        sys.exit(1)

    print("\nGate status: PASS")


if __name__ == "__main__":
    main()
