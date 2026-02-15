"""
Canonical evaluation runner for KARL.

Tracks:
- v1: legacy golden benchmark
- v2: comprehensive benchmark (ablation-capable)
- v2_multi_contract: multi-contract benchmark slice with per-contract reporting
- v3: canonical multi-contract phase suite
- escalation: escalation precision slice
- paraphrase: paraphrase/slang robustness slice
- adversarial: deterministic formal-precedence near-miss slice
- unanswerable: deterministic multi-contract abstention slice
- cross_contract_mentions: deterministic /api/query guard for foreign-contract references
- false_unavailable: deterministic guard against false "not available" responses
- wage_table_evidence: deterministic canonical wage-row table evidence slice
- entitlement_table_evidence: deterministic canonical entitlement schedule evidence slice
- role_catalog_integrity: deterministic contract-scoped role integrity slice
- needle: needle retrieval integrity slice
- all: run v1 + v2 + v2_multi_contract + escalation + paraphrase + adversarial + unanswerable + cross_contract_mentions + false_unavailable + wage_table_evidence + entitlement_table_evidence + role_catalog_integrity + needle + v3

This runner records deterministic run metadata for auditability:
- timestamp, command, cwd
- git commit/branch/dirty state
- config/model snapshot
- corpus and manifest SHA256 hashes
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend import config
from backend.contracts import list_contract_catalog, get_contract_catalog_entry, resolve_default_contract_id
from backend.validate_manifests import main as validate_manifests_main


PROJECT_ROOT = Path(__file__).parent.parent
OUT_DIR = config.DATA_DIR / "test_set"
PACK_REGISTRY_FILE = config.DATA_DIR / "contracts" / "pack_registry.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_hashes() -> dict:
    hashes = {}
    for pattern in [
        config.CHUNKS_DIR / "*.json",
        config.MANIFESTS_DIR / "*.json",
        config.WAGES_DIR / "*.json",
    ]:
        for path in sorted(pattern.parent.glob(pattern.name)):
            hashes[str(path.relative_to(PROJECT_ROOT))] = _sha256_file(path)
    return hashes


def _pack_registry_snapshot() -> dict:
    """
    Capture accepted contract-pack hashes for active manifests.

    This ties benchmark metadata to ingestion acceptance evidence.
    """
    active_contract_ids = [c.get("contract_id") for c in list_contract_catalog() if c.get("contract_id")]

    registry_data = {}
    if PACK_REGISTRY_FILE.exists():
        try:
            with open(PACK_REGISTRY_FILE, "r", encoding="utf-8") as f:
                registry_data = json.load(f)
        except Exception:
            registry_data = {}

    accepted = (registry_data or {}).get("accepted_packs", {}) if isinstance(registry_data, dict) else {}
    by_contract = {}
    for contract_id in active_contract_ids:
        info = accepted.get(contract_id)
        if isinstance(info, dict):
            by_contract[contract_id] = {
                "pack_hash": info.get("pack_hash"),
                "scorecard_path": info.get("scorecard_path"),
                "accepted_at_utc": info.get("accepted_at_utc"),
            }
        else:
            by_contract[contract_id] = None

    return {
        "registry_path": str(PACK_REGISTRY_FILE),
        "active_contract_ids": active_contract_ids,
        "contracts": by_contract,
    }


def _git_info() -> dict:
    def _run(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=PROJECT_ROOT, text=True).strip()
        except Exception:
            return ""

    commit = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    return {
        "commit": commit or None,
        "branch": branch or None,
        "dirty": bool(status),
    }


def _config_snapshot() -> dict:
    configured_contract_id = config.CONTRACT_ID
    catalog_entry = get_contract_catalog_entry(configured_contract_id)
    resolved_contract_id = (
        catalog_entry.get("contract_id")
        if catalog_entry
        else (resolve_default_contract_id() or configured_contract_id)
    )
    return {
        "contract_id_configured": configured_contract_id,
        "contract_id_resolved": resolved_contract_id,
        "embedding_model": config.EMBEDDING_MODEL,
        "llm_model": config.LLM_MODEL,
        "interpreter_model": getattr(config, "INTERPRETER_MODEL", None),
        "reranker_model": getattr(config, "RERANKER_MODEL", None),
        "hypothesis_model": getattr(config, "HYPOTHESIS_MODEL", None),
        "cag_flags": {
            "hypothesis": getattr(config, "CAG_ENABLE_HYPOTHESIS_LAYER", None),
            "full_article_expansion": getattr(config, "CAG_ENABLE_FULL_ARTICLE_EXPANSION", None),
            "query_interpreter": getattr(config, "CAG_ENABLE_QUERY_INTERPRETER", None),
            "reranker": getattr(config, "CAG_ENABLE_RERANKER", None),
        },
        "hybrid_weights": {
            "vector": getattr(config, "HYBRID_VECTOR_WEIGHT", None),
            "keyword": getattr(config, "HYBRID_KEYWORD_WEIGHT", None),
        },
    }


def _run_cmd(cmd: list[str]) -> dict:
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    return {
        "command": " ".join(cmd),
        "return_code": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def _build_commands(track: str, ablation_mode: str, bucket_filter: str | None, seed: int) -> list[list[str]]:
    py = sys.executable
    if track == "v1":
        return [[py, "-m", "backend.evaluate"]]
    if track == "v2":
        cmd = [py, "-m", "backend.evaluate_comprehensive", "--ablation-mode", ablation_mode, "--seed", str(seed)]
        if bucket_filter:
            cmd.extend(["--bucket-filter", bucket_filter])
        return [cmd]
    if track == "v2_multi_contract":
        return [[py, "-m", "backend.evaluate_multi_contract", "--bm25-only"]]
    if track == "v3":
        return [[py, "-m", "backend.evaluate_v3", "--bm25-only"]]
    if track == "escalation":
        return [[py, "-m", "backend.evaluate_escalation_precision"]]
    if track == "paraphrase":
        return [[py, "-m", "backend.evaluate_paraphrase", "--bm25-only"]]
    if track == "adversarial":
        return [[py, "-m", "backend.evaluate_adversarial_precedence", "--bm25-only"]]
    if track == "unanswerable":
        return [[py, "-m", "backend.evaluate_unanswerable", "--bm25-only"]]
    if track == "cross_contract_mentions":
        return [[py, "-m", "backend.evaluate_cross_contract_mentions", "--bm25-only"]]
    if track == "false_unavailable":
        return [[py, "-m", "backend.evaluate_false_unavailable", "--bm25-only"]]
    if track == "wage_table_evidence":
        return [[py, "-m", "backend.evaluate_wage_table_evidence", "--bm25-only"]]
    if track == "entitlement_table_evidence":
        return [[py, "-m", "backend.evaluate_entitlement_table_evidence"]]
    if track == "role_catalog_integrity":
        return [[py, "-m", "backend.evaluate_role_catalog_integrity"]]
    if track == "needle":
        return [[py, "-m", "backend.evaluate_needle", "--bm25-only"]]
    if track == "all":
        return (
            _build_commands("v1", ablation_mode, bucket_filter, seed)
            + _build_commands("v2", ablation_mode, bucket_filter, seed)
            + _build_commands("v2_multi_contract", ablation_mode, bucket_filter, seed)
            + _build_commands("escalation", ablation_mode, bucket_filter, seed)
            + _build_commands("paraphrase", ablation_mode, bucket_filter, seed)
            + _build_commands("adversarial", ablation_mode, bucket_filter, seed)
            + _build_commands("unanswerable", ablation_mode, bucket_filter, seed)
            + _build_commands("cross_contract_mentions", ablation_mode, bucket_filter, seed)
            + _build_commands("false_unavailable", ablation_mode, bucket_filter, seed)
            + _build_commands("wage_table_evidence", ablation_mode, bucket_filter, seed)
            + _build_commands("entitlement_table_evidence", ablation_mode, bucket_filter, seed)
            + _build_commands("role_catalog_integrity", ablation_mode, bucket_filter, seed)
            + _build_commands("needle", ablation_mode, bucket_filter, seed)
            + _build_commands("v3", ablation_mode, bucket_filter, seed)
        )
    raise ValueError(f"Unsupported track: {track}")


def run(track: str, ablation_mode: str, bucket_filter: str | None, seed: int, dry_run: bool) -> dict:
    ts = datetime.now(timezone.utc).isoformat()
    commands = _build_commands(track, ablation_mode, bucket_filter, seed)

    report = {
        "metadata_version": "eval_run_v1",
        "timestamp_utc": ts,
        "track": track,
        "cwd": str(PROJECT_ROOT),
        "git": _git_info(),
        "config_snapshot": _config_snapshot(),
        "artifact_hashes": _collect_hashes(),
        "pack_registry_snapshot": _pack_registry_snapshot(),
        "planned_commands": [" ".join(c) for c in commands],
        "results": [],
        "dry_run": dry_run,
    }

    if not dry_run:
        manifest_rc = validate_manifests_main()
        report["manifest_validation_return_code"] = manifest_rc
        if manifest_rc != 0:
            report["results"].append(
                {
                    "command": "python -m backend.validate_manifests",
                    "return_code": manifest_rc,
                    "stdout_tail": "Manifest validation failed before evaluation run.",
                    "stderr_tail": "",
                }
            )
            return report

        for cmd in commands:
            result = _run_cmd(cmd)
            report["results"].append(result)
            if result["return_code"] != 0:
                break

    return report


def _write_report(report: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = OUT_DIR / f"eval_run_metadata_{report['track']}_{stamp}.json"
    latest = OUT_DIR / "eval_run_metadata_latest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(latest, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Canonical KARL evaluation runner with metadata capture.")
    parser.add_argument(
        "--track",
        choices=["v1", "v2", "v2_multi_contract", "v3", "escalation", "paraphrase", "adversarial", "unanswerable", "cross_contract_mentions", "false_unavailable", "wage_table_evidence", "entitlement_table_evidence", "role_catalog_integrity", "needle", "all"],
        default="v2",
    )
    parser.add_argument("--ablation-mode", default="normal")
    parser.add_argument("--bucket-filter", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = run(
        track=args.track,
        ablation_mode=args.ablation_mode,
        bucket_filter=args.bucket_filter,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    out_path = _write_report(report)

    print("=" * 72)
    print("KARL Canonical Evaluation Runner")
    print("=" * 72)
    print(f"Track: {args.track}")
    print(f"Dry run: {args.dry_run}")
    print(f"Metadata: {out_path}")
    if report["results"]:
        for i, r in enumerate(report["results"], start=1):
            print(f"[{i}] rc={r['return_code']} {r['command']}")

    failed = any(r["return_code"] != 0 for r in report["results"])
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
