"""
Escalation precision evaluation for deterministic two-stage policy.

Outputs:
- Confusion matrix for active_urgent_context (escalation trigger)
- Precision/recall/FPR for escalation decisions
- High-stakes topic detection accuracy
- Slice-level metrics
- Threshold tradeoff simulation (active pattern count >= N)
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import DATA_DIR, CONTRACT_ID
from backend.retrieval.router import classify_intent, classify_high_stakes_context


@dataclass
class EscalationCaseResult:
    case_id: str
    slice_name: str
    question: str
    expected_high_stakes_topic: bool
    expected_active_urgent_context: bool
    predicted_high_stakes_topic: bool
    predicted_active_urgent_context: bool
    active_pattern_hits: int
    matched_patterns: list[str]


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _confusion(y_true: list[bool], y_pred: list[bool]) -> dict[str, int]:
    tp = fp = tn = fn = 0
    for truth, pred in zip(y_true, y_pred):
        if truth and pred:
            tp += 1
        elif not truth and pred:
            fp += 1
        elif not truth and not pred:
            tn += 1
        else:
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _metrics_from_confusion(cm: dict[str, int]) -> dict[str, float]:
    tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    fpr = _safe_div(fp, fp + tn)
    accuracy = _safe_div(tp + tn, tp + fp + tn + fn)
    return {
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fpr,
        "accuracy": accuracy,
    }


def evaluate(test_file: Path | None = None) -> dict:
    if test_file is None:
        test_file = DATA_DIR / "test_set" / "escalation_precision_test.json"

    with open(test_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    cases = payload["test_cases"]
    results: list[EscalationCaseResult] = []

    for case in cases:
        question = case["question"]
        intent = classify_intent(question, contract_id=CONTRACT_ID)
        _, _, matched = classify_high_stakes_context(question)
        active_hits = sum(1 for m in matched if m.startswith("active:"))

        results.append(
            EscalationCaseResult(
                case_id=case["id"],
                slice_name=case["slice"],
                question=question,
                expected_high_stakes_topic=case["expected_high_stakes_topic"],
                expected_active_urgent_context=case["expected_active_urgent_context"],
                predicted_high_stakes_topic=intent.high_stakes_topic,
                predicted_active_urgent_context=intent.active_urgent_context,
                active_pattern_hits=active_hits,
                matched_patterns=intent.keywords_matched,
            )
        )

    expected_escalate = [r.expected_active_urgent_context for r in results]
    predicted_escalate = [r.predicted_active_urgent_context for r in results]
    escalation_cm = _confusion(expected_escalate, predicted_escalate)
    escalation_metrics = _metrics_from_confusion(escalation_cm)

    expected_topic = [r.expected_high_stakes_topic for r in results]
    predicted_topic = [r.predicted_high_stakes_topic for r in results]
    topic_cm = _confusion(expected_topic, predicted_topic)
    topic_metrics = _metrics_from_confusion(topic_cm)

    slice_metrics = {}
    grouped = defaultdict(list)
    for row in results:
        grouped[row.slice_name].append(row)
    for slice_name, rows in grouped.items():
        cm = _confusion(
            [r.expected_active_urgent_context for r in rows],
            [r.predicted_active_urgent_context for r in rows],
        )
        slice_metrics[slice_name] = {
            "count": len(rows),
            "confusion_matrix": cm,
            "metrics": _metrics_from_confusion(cm),
        }

    tradeoffs = []
    for threshold in [1, 2, 3]:
        preds = [r.active_pattern_hits >= threshold for r in results]
        cm = _confusion(expected_escalate, preds)
        tradeoffs.append(
            {
                "active_pattern_threshold": threshold,
                "confusion_matrix": cm,
                "metrics": _metrics_from_confusion(cm),
            }
        )

    failures = [
        asdict(r) for r in results
        if (
            r.expected_active_urgent_context != r.predicted_active_urgent_context
            or r.expected_high_stakes_topic != r.predicted_high_stakes_topic
        )
    ]

    return {
        "metadata": {
            "policy": payload.get("policy", "deterministic"),
            "total_cases": len(results),
            "source_file": str(test_file),
        },
        "escalation_confusion_matrix": escalation_cm,
        "escalation_metrics": escalation_metrics,
        "high_stakes_topic_confusion_matrix": topic_cm,
        "high_stakes_topic_metrics": topic_metrics,
        "slice_metrics": slice_metrics,
        "threshold_tradeoffs": tradeoffs,
        "failures": failures,
        "results": [asdict(r) for r in results],
    }


def main():
    report = evaluate()
    out_file = DATA_DIR / "test_set" / "escalation_precision_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    cm = report["escalation_confusion_matrix"]
    m = report["escalation_metrics"]

    print("=" * 72)
    print("ESCALATION PRECISION EVALUATION (deterministic two-stage)")
    print("=" * 72)
    print(f"Cases: {report['metadata']['total_cases']}")
    print(f"Confusion Matrix (Active Escalation): TP={cm['tp']} FP={cm['fp']} TN={cm['tn']} FN={cm['fn']}")
    print(
        f"Precision={m['precision']:.3f} "
        f"Recall={m['recall']:.3f} "
        f"FPR={m['false_positive_rate']:.3f} "
        f"Accuracy={m['accuracy']:.3f}"
    )

    print("\nSlice Metrics:")
    for slice_name, info in report["slice_metrics"].items():
        sm = info["metrics"]
        scm = info["confusion_matrix"]
        print(
            f"- {slice_name}: n={info['count']} "
            f"(TP={scm['tp']} FP={scm['fp']} TN={scm['tn']} FN={scm['fn']}) "
            f"precision={sm['precision']:.3f} recall={sm['recall']:.3f} fpr={sm['false_positive_rate']:.3f}"
        )

    print("\nThreshold Tradeoffs (active pattern hits >= N):")
    for row in report["threshold_tradeoffs"]:
        tm = row["metrics"]
        tcm = row["confusion_matrix"]
        print(
            f"- N={row['active_pattern_threshold']}: "
            f"TP={tcm['tp']} FP={tcm['fp']} TN={tcm['tn']} FN={tcm['fn']} "
            f"precision={tm['precision']:.3f} recall={tm['recall']:.3f} fpr={tm['false_positive_rate']:.3f}"
        )

    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
