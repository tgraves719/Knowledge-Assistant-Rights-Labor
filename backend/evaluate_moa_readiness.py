"""MOA readiness gate runner.

Runs key deterministic checks and emits a single gate report:
- materializer integrity tests
- effective snapshot coverage (base vs effective index)
- contract-history/pdf-source API tests
- MOA effective retrieval benchmark
- deleted-vs-updated MOA regression slice (deleted clauses absent; updated clauses retrieved from MOA provenance)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import DATA_DIR


def _run_cmd(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    return {
        "command": " ".join(cmd),
        "return_code": int(proc.returncode),
        "stdout_tail": proc.stdout[-6000:],
        "stderr_tail": proc.stderr[-6000:],
    }


def _rate(passed: int, total: int) -> float:
    return round((passed / total) if total else 0.0, 4)


def run(
    *,
    moa_pass_rate_threshold: float = 0.9,
    include_topic_routing_test: bool = True,
    deep_input: Optional[Path] = None,
    deep_pass_rate_threshold: Optional[float] = None,
) -> dict[str, Any]:
    py = sys.executable
    baseline_output = DATA_DIR / "test_set" / "moa_effective_results.json"
    deleted_vs_updated_output = DATA_DIR / "test_set" / "moa_deleted_vs_updated_results.json"
    deleted_vs_updated_answer_output = DATA_DIR / "test_set" / "moa_deleted_vs_updated_answer_results.json"
    deep_output = DATA_DIR / "test_set" / "moa_effective_deep_results.json"
    commands = [
        [py, "-m", "backend.test_moa_materializer"],
        [py, "-m", "backend.evaluate_effective_coverage"],
        [py, "-m", "backend.test_contract_history_api"],
        [py, "-m", "backend.evaluate_moa_effective", "--bm25-only", "--output", str(baseline_output)],
        [py, "-m", "backend.evaluate_moa_deleted_vs_updated", "--bm25-only", "--output", str(deleted_vs_updated_output)],
        [py, "-m", "backend.evaluate_moa_deleted_vs_updated_answer", "--bm25-only", "--output", str(deleted_vs_updated_answer_output)],
    ]
    if include_topic_routing_test:
        commands.insert(2, [py, "-m", "backend.test_topic_routing"])
    commands.insert(3 if include_topic_routing_test else 2, [py, "-m", "backend.test_false_unavailable_fallback"])
    if deep_input:
        commands.append(
            [
                py,
                "-m",
                "backend.evaluate_moa_effective",
                "--bm25-only",
                "--input",
                str(deep_input),
                "--output",
                str(deep_output),
            ]
        )

    results = [_run_cmd(cmd) for cmd in commands]
    all_cmds_ok = all(row.get("return_code") == 0 for row in results)

    moa_results_file = baseline_output
    moa_eval = {}
    if moa_results_file.exists():
        try:
            moa_eval = json.loads(moa_results_file.read_text(encoding="utf-8"))
        except Exception:
            moa_eval = {}

    moa_overall = moa_eval.get("overall") if isinstance(moa_eval, dict) else {}
    moa_passed = int((moa_overall or {}).get("passed") or 0)
    moa_total = int((moa_overall or {}).get("total") or 0)
    moa_pass_rate = float((moa_overall or {}).get("pass_rate") or _rate(moa_passed, moa_total))
    moa_gate_ok = moa_pass_rate >= float(moa_pass_rate_threshold)

    delupd_eval = _load_json_or_empty(deleted_vs_updated_output)
    delupd_gate = delupd_eval.get("gate") if isinstance(delupd_eval, dict) else {}
    delupd_gate_ok = bool((delupd_gate or {}).get("pass"))
    delupd_answer_eval = _load_json_or_empty(deleted_vs_updated_answer_output)
    delupd_answer_gate = delupd_answer_eval.get("gate") if isinstance(delupd_answer_eval, dict) else {}
    delupd_answer_gate_ok = bool((delupd_answer_gate or {}).get("pass"))

    deep_gate_block = None
    if deep_input:
        deep_eval = _load_json_or_empty(deep_output)
        deep_overall = deep_eval.get("overall") if isinstance(deep_eval, dict) else {}
        deep_passed = int((deep_overall or {}).get("passed") or 0)
        deep_total = int((deep_overall or {}).get("total") or 0)
        deep_pass_rate = float((deep_overall or {}).get("pass_rate") or _rate(deep_passed, deep_total))
        if deep_pass_rate_threshold is None:
            deep_pass_rate_threshold = 0.8
        deep_gate_block = {
            "pass": deep_pass_rate >= float(deep_pass_rate_threshold),
            "threshold": deep_pass_rate_threshold,
            "observed": deep_pass_rate,
            "passed": deep_passed,
            "total": deep_total,
            "results_file": str(deep_output),
            "input_file": str(deep_input),
        }

    gates = {
        "tests_and_commands_ok": {
            "pass": all_cmds_ok,
            "detail": "All required test/eval commands returned 0.",
        },
        "moa_effective_pass_rate": {
            "pass": moa_gate_ok,
            "threshold": moa_pass_rate_threshold,
            "observed": moa_pass_rate,
            "detail": "MOA effective benchmark pass rate.",
        },
        "moa_deleted_vs_updated_gate": {
            "pass": delupd_gate_ok,
            "detail": "Deleted-vs-updated MOA regression slice gate.",
            "results_file": str(deleted_vs_updated_output),
        },
        "moa_deleted_vs_updated_answer_gate": {
            "pass": delupd_answer_gate_ok,
            "detail": "End-to-end answer behavior for deleted-vs-updated MOA clauses.",
            "results_file": str(deleted_vs_updated_answer_output),
        },
    }
    if deep_gate_block:
        gates["moa_effective_deep_pass_rate"] = deep_gate_block
    gate_pass = all(bool(row.get("pass")) for row in gates.values())

    return {
        "schema_version": "moa_readiness_eval_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gate_pass": gate_pass,
        "gates": gates,
        "moa_effective_summary": {
            "passed": moa_passed,
            "total": moa_total,
            "pass_rate": moa_pass_rate,
            "results_file": str(moa_results_file),
        },
        "moa_deleted_vs_updated_summary": {
            "gate_pass": delupd_gate_ok,
            "results_file": str(deleted_vs_updated_output),
            "overall": (delupd_eval.get("overall") if isinstance(delupd_eval, dict) else None),
            "buckets": (delupd_eval.get("buckets") if isinstance(delupd_eval, dict) else None),
        },
        "moa_deleted_vs_updated_answer_summary": {
            "gate_pass": delupd_answer_gate_ok,
            "results_file": str(deleted_vs_updated_answer_output),
            "overall": (delupd_answer_eval.get("overall") if isinstance(delupd_answer_eval, dict) else None),
            "by_bucket": (delupd_answer_eval.get("by_bucket") if isinstance(delupd_answer_eval, dict) else None),
        },
        "commands": results,
    }


def _write_report(report: dict[str, Any]) -> Path:
    out_path = DATA_DIR / "test_set" / "moa_readiness_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return out_path


def _load_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MOA readiness gates.")
    parser.add_argument("--moa-pass-rate-threshold", type=float, default=0.9)
    parser.add_argument("--deep-input", type=str, default=None, help="Optional deep MOA-effective dataset path")
    parser.add_argument("--deep-pass-rate-threshold", type=float, default=None)
    parser.add_argument("--skip-topic-routing-test", action="store_true")
    args = parser.parse_args()

    report = run(
        moa_pass_rate_threshold=float(args.moa_pass_rate_threshold),
        include_topic_routing_test=not bool(args.skip_topic_routing_test),
        deep_input=Path(args.deep_input) if args.deep_input else None,
        deep_pass_rate_threshold=(float(args.deep_pass_rate_threshold) if args.deep_pass_rate_threshold is not None else None),
    )
    out_path = _write_report(report)

    print("=" * 72)
    print("KARL MOA Readiness")
    print("=" * 72)
    print(f"Gate pass: {report.get('gate_pass')}")
    for name, gate in (report.get("gates") or {}).items():
        print(f"- {name}: {gate.get('pass')}")
    moa_summary = report.get("moa_effective_summary") or {}
    print(
        "MOA effective pass rate: "
        f"{moa_summary.get('passed', 0)}/{moa_summary.get('total', 0)} "
        f"({float(moa_summary.get('pass_rate', 0.0)):.1%})"
    )
    print(f"Results: {out_path}")
    return 0 if report.get("gate_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
