"""Run a deep MOA-centric evaluation suite and aggregate results.

Includes effective snapshot coverage integrity to catch missing doc types
between base and latest effective index inputs.
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


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _summary_from_result(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if not payload:
        return {"result_file": str(path), "summary": None}
    gate = payload.get("gate")
    gate_pass = None
    if isinstance(gate, dict):
        gate_pass = gate.get("pass")
    overall = payload.get("overall")
    if isinstance(overall, dict):
        return {
            "result_file": str(path),
            "summary": {
                "passed": overall.get("passed"),
                "total": overall.get("total"),
                "pass_rate": overall.get("pass_rate"),
                "gate_pass": gate_pass,
            },
        }
    return {"result_file": str(path), "summary": None}


def run(skip_topic_routing_in_readiness: bool = False) -> dict[str, Any]:
    py = sys.executable
    baseline_out = DATA_DIR / "test_set" / "moa_effective_results.json"
    deep_out = DATA_DIR / "test_set" / "moa_effective_deep_results.json"
    deep_input = DATA_DIR / "test_set" / "moa_effective_deep_test.json"
    commands: list[dict[str, Any]] = [
        {
            "name": "moa_effective_baseline",
            "cmd": [py, "-m", "backend.evaluate_moa_effective", "--bm25-only", "--output", str(baseline_out)],
            "result_file": baseline_out,
        },
        {
            "name": "moa_effective_deep",
            "cmd": [
                py,
                "-m",
                "backend.evaluate_moa_effective",
                "--bm25-only",
                "--input",
                str(deep_input),
                "--output",
                str(deep_out),
            ],
            "result_file": deep_out,
        },
        {
            "name": "moa_deleted_vs_updated",
            "cmd": [py, "-m", "backend.evaluate_moa_deleted_vs_updated", "--bm25-only"],
            "result_file": DATA_DIR / "test_set" / "moa_deleted_vs_updated_results.json",
        },
        {
            "name": "moa_deleted_vs_updated_answer",
            "cmd": [py, "-m", "backend.evaluate_moa_deleted_vs_updated_answer", "--bm25-only"],
            "result_file": DATA_DIR / "test_set" / "moa_deleted_vs_updated_answer_results.json",
        },
        {
            "name": "moa_readiness",
            "cmd": (
                [
                    py,
                    "-m",
                    "backend.evaluate_moa_readiness",
                    "--deep-input",
                    str(deep_input),
                    "--deep-pass-rate-threshold",
                    "0.85",
                ]
                + (["--skip-topic-routing-test"] if skip_topic_routing_in_readiness else [])
            ),
            "result_file": DATA_DIR / "test_set" / "moa_readiness_results.json",
        },
        {
            "name": "effective_snapshot_coverage",
            "cmd": [py, "-m", "backend.evaluate_effective_coverage"],
            "result_file": DATA_DIR / "test_set" / "effective_snapshot_coverage_results.json",
        },
        {
            "name": "effective_wage_snapshot_coverage",
            "cmd": [py, "-m", "backend.evaluate_effective_wage_coverage"],
            "result_file": DATA_DIR / "test_set" / "effective_wage_snapshot_coverage_results.json",
        },
        {
            "name": "false_unavailable_fallback_test",
            "cmd": [py, "-m", "backend.test_false_unavailable_fallback"],
            "result_file": None,
        },
        {
            "name": "cross_contract_mentions",
            "cmd": [py, "-m", "backend.evaluate_cross_contract_mentions", "--bm25-only"],
            "result_file": DATA_DIR / "test_set" / "cross_contract_mentions_results.json",
        },
        {
            "name": "false_unavailable",
            "cmd": [py, "-m", "backend.evaluate_false_unavailable", "--bm25-only"],
            "result_file": DATA_DIR / "test_set" / "false_unavailable_results.json",
        },
        {
            "name": "unanswerable",
            "cmd": [py, "-m", "backend.evaluate_unanswerable", "--bm25-only"],
            "result_file": DATA_DIR / "test_set" / "unanswerable_results.json",
        },
        {
            "name": "adversarial_precedence",
            "cmd": [py, "-m", "backend.evaluate_adversarial_precedence", "--bm25-only"],
            "result_file": DATA_DIR / "test_set" / "adversarial_results.json",
        },
        {
            "name": "wage_table_evidence",
            "cmd": [py, "-m", "backend.evaluate_wage_table_evidence", "--bm25-only"],
            "result_file": DATA_DIR / "test_set" / "wage_table_evidence_results.json",
        },
        {
            "name": "entitlement_table_evidence",
            "cmd": [py, "-m", "backend.evaluate_entitlement_table_evidence"],
            "result_file": DATA_DIR / "test_set" / "entitlement_table_evidence_results.json",
        },
        {
            "name": "side_letter_retrieval",
            "cmd": [py, "-m", "backend.evaluate_side_letter_retrieval", "--bm25-only"],
            "result_file": DATA_DIR / "test_set" / "side_letter_retrieval_results.json",
        },
        {
            "name": "role_catalog_integrity",
            "cmd": [py, "-m", "backend.evaluate_role_catalog_integrity"],
            "result_file": DATA_DIR / "test_set" / "role_catalog_integrity_results.json",
        },
        {
            "name": "followup_role_wage",
            "cmd": [py, "-m", "backend.evaluate_followup_role_wage", "--bm25-only"],
            "result_file": DATA_DIR / "test_set" / "followup_role_wage_results.json",
        },
        {
            "name": "needle",
            "cmd": [py, "-m", "backend.evaluate_needle", "--bm25-only"],
            "result_file": DATA_DIR / "test_set" / "needle_results.json",
        },
        {
            "name": "topic_routing_test",
            "cmd": [py, "-m", "backend.test_topic_routing"],
            "result_file": None,
        },
    ]

    command_results = []
    for row in commands:
        cmd_result = _run_cmd(row["cmd"])
        summary = _summary_from_result(row["result_file"]) if row.get("result_file") else None
        command_results.append(
            {
                "name": row["name"],
                "return_code": cmd_result["return_code"],
                "command": cmd_result["command"],
                "stdout_tail": cmd_result["stdout_tail"],
                "stderr_tail": cmd_result["stderr_tail"],
                "result_file": str(row["result_file"]) if row.get("result_file") else None,
                "result_summary": summary["summary"] if isinstance(summary, dict) else None,
            }
        )

    passing = sum(1 for row in command_results if row.get("return_code") == 0)
    total = len(command_results)

    return {
        "schema_version": "moa_deep_eval_suite_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall": {
            "commands_passed": passing,
            "commands_total": total,
            "pass_rate": round((passing / total) if total else 0.0, 4),
            "all_passed": passing == total,
        },
        "commands": command_results,
    }


def _write_report(report: dict[str, Any]) -> Path:
    out_path = DATA_DIR / "test_set" / "moa_deep_eval_suite_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return out_path


def _command_row(report: dict[str, Any], name: str) -> Optional[dict[str, Any]]:
    for row in report.get("commands") or []:
        if str(row.get("name") or "") == name:
            return row
    return None


def _write_markdown_summary(report: dict[str, Any]) -> Path:
    out_path = DATA_DIR / "test_set" / "moa_deep_eval_summary.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    overall = report.get("overall") or {}
    baseline_cmd = _command_row(report, "moa_effective_baseline") or {}
    deep_cmd = _command_row(report, "moa_effective_deep") or {}
    baseline_summary = baseline_cmd.get("result_summary") or {}
    deep_summary = deep_cmd.get("result_summary") or {}

    lines: list[str] = [
        "# MOA Deep Eval Summary",
        "",
        f"Generated: {str(report.get('timestamp_utc') or '')}",
        "",
        "## Suite Status",
        f"- Commands passed: {int(overall.get('commands_passed', 0))}/{int(overall.get('commands_total', 0))} ({float(overall.get('pass_rate', 0.0)):.1%})",
        f"- All commands passed: {bool(overall.get('all_passed'))}",
        "",
        "## MOA Effective Scores",
        f"- Baseline dataset: {int(baseline_summary.get('passed') or 0)}/{int(baseline_summary.get('total') or 0)} ({float(baseline_summary.get('pass_rate') or 0.0):.1%})",
        f"- Deep dataset: {int(deep_summary.get('passed') or 0)}/{int(deep_summary.get('total') or 0)} ({float(deep_summary.get('pass_rate') or 0.0):.1%})",
        "",
        "## Track Scores",
    ]

    for row in report.get("commands") or []:
        name = str(row.get("name") or "")
        rc = int(row.get("return_code") or 0)
        summary = row.get("result_summary") or {}
        if isinstance(summary, dict) and summary:
            passed = int(summary.get("passed") or 0)
            total = int(summary.get("total") or 0)
            rate = float(summary.get("pass_rate") or 0.0)
            gate_suffix = ""
            if summary.get("gate_pass") is not None:
                gate_suffix = f", gate={bool(summary.get('gate_pass'))}"
            lines.append(f"- `{name}`: rc={rc}, pass={passed}/{total} ({rate:.1%}){gate_suffix}")
        else:
            lines.append(f"- `{name}`: rc={rc}")

    deep_failures: list[dict[str, Any]] = []
    deep_result_file = Path(str(deep_cmd.get("result_file") or ""))
    deep_payload = _load_json(deep_result_file) if deep_result_file else None
    if isinstance(deep_payload, dict):
        for row in deep_payload.get("results") or []:
            if not bool(row.get("pass")):
                deep_failures.append(row)

    lines.extend(["", "## Deep Dataset Failures"])
    if not deep_failures:
        lines.append("- None. All deep MOA cases passed.")
    else:
        for row in deep_failures:
            case_id = str(row.get("id") or "")
            question = str(row.get("question") or "").strip()
            retrieved = ", ".join((row.get("retrieved_citations") or [])[:4])
            lines.append(f"- `{case_id}`: {question}")
            lines.append(
                "  - "
                f"citation_hit={bool(row.get('citation_hit'))} "
                f"keywords_ok={bool(row.get('keywords_ok'))} "
                f"source_type_ok={bool(row.get('source_type_ok'))} "
                f"intent={str(row.get('observed_intent_type') or '')}"
            )
            if retrieved:
                lines.append(f"  - top citations: {retrieved}")

    lines.extend(["", "## Priority Gaps"])
    risk_rows = []
    for row in report.get("commands") or []:
        summary = row.get("result_summary") or {}
        if not isinstance(summary, dict) or not summary:
            continue
        pass_rate = float(summary.get("pass_rate") or 0.0)
        if pass_rate < 1.0:
            risk_rows.append((pass_rate, row))
    risk_rows.sort(key=lambda t: t[0])
    if not risk_rows:
        lines.append("- No tracked gaps in this suite run.")
    else:
        for pass_rate, row in risk_rows:
            name = str(row.get("name") or "")
            summary = row.get("result_summary") or {}
            lines.append(
                f"- `{name}`: {int(summary.get('passed') or 0)}/{int(summary.get('total') or 0)} "
                f"({pass_rate:.1%})"
            )

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines).rstrip() + "\n")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deep MOA eval suite.")
    parser.add_argument("--skip-topic-routing-in-readiness", action="store_true")
    args = parser.parse_args()

    report = run(skip_topic_routing_in_readiness=bool(args.skip_topic_routing_in_readiness))
    out_path = _write_report(report)
    summary_path = _write_markdown_summary(report)
    overall = report.get("overall") or {}
    print("=" * 72)
    print("KARL MOA Deep Eval Suite")
    print("=" * 72)
    print(
        "Commands: "
        f"{overall.get('commands_passed', 0)}/{overall.get('commands_total', 0)} "
        f"({float(overall.get('pass_rate', 0.0)):.1%})"
    )
    print(f"All passed: {overall.get('all_passed')}")
    print(f"Results: {out_path}")
    print(f"Summary: {summary_path}")
    return 0 if overall.get("all_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
