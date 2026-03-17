"""
Apply reviewed classification override decisions with validation and diff preview.

Usage examples:
  python scripts/apply_review_overrides.py --contract-id local7_safeway_pueblo_clerks_2022 --emit-template
  python scripts/apply_review_overrides.py --contract-id <id> --decision-file <path> --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import DATA_DIR
from backend.ingest.classification_ontology import (
    MANUAL_OVERRIDE_SCHEMA_VERSION,
    build_classification_ontology,
    load_manual_classification_review_overrides,
    normalize_classification_name,
)


REVIEW_DECISION_SCHEMA_VERSION = "classification_review_decisions_v1"


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _package_paths(contract_id: str) -> dict[str, Path]:
    root = DATA_DIR / "contracts" / contract_id
    return {
        "package_root": root,
        "manifest": root / "manifests" / f"{contract_id}.json",
        "wages": root / "wages" / f"wage_tables_{contract_id}.json",
        "ontology": root / "ontology" / "classification_ontology.json",
        "review_queue": root / "ontology" / "ingestion_review_queue.json",
        "manual_overrides": root / "ontology" / "manual_classification_overrides.json",
        "review_template": root / "ontology" / "review_decisions_template.json",
    }


def _build_template(contract_id: str, ontology: dict) -> dict:
    unresolved = []
    for d in ontology.get("decisions", []) or []:
        if d.get("mapped_wage_key"):
            continue
        suggested_action = (
            "needs_clarification"
            if str(d.get("review_state") or "") == "needs_clarification"
            or str(d.get("mapping_method") or "") == "unresolved_ambiguous"
            else "out_of_scope"
        )
        unresolved.append(
            {
                "source_key": d.get("source_key"),
                "action": suggested_action,
                "comment": (
                    "Use 'map' with target_wage_key, 'out_of_scope', or "
                    "'needs_clarification' with clarification_wage_keys after human review."
                ),
                "candidate_scores": (d.get("candidate_scores") or [])[:5],
                "clarification_wage_keys": (d.get("clarification_wage_keys") or [])[:5],
            }
        )
    return {
        "schema_version": REVIEW_DECISION_SCHEMA_VERSION,
        "contract_id": contract_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "set_aliases": {},
        "remove_aliases": [],
        "review_overrides": {},
        "decision_notes": unresolved,
    }


def _parse_decision_file(path: Path) -> tuple[dict[str, str], set[str], dict[str, dict]]:
    raw = _load_json(path)
    set_aliases: dict[str, str] = {}
    remove_aliases: set[str] = set()
    review_overrides: dict[str, dict] = {}

    if not isinstance(raw, dict):
        raise ValueError("Decision file must be a JSON object.")

    if "set_aliases" in raw or "remove_aliases" in raw:
        for src, dst in (raw.get("set_aliases") or {}).items():
            src_key = normalize_classification_name(str(src or ""))
            dst_key = normalize_classification_name(str(dst or ""))
            if not src_key or not dst_key:
                continue
            set_aliases[src_key] = dst_key
        for src in (raw.get("remove_aliases") or []):
            src_key = normalize_classification_name(str(src or ""))
            if src_key:
                remove_aliases.add(src_key)
        for src, cfg in (raw.get("review_overrides") or {}).items():
            src_key = normalize_classification_name(str(src or ""))
            if not src_key:
                continue
            if isinstance(cfg, str):
                action = normalize_classification_name(cfg)
                if action in {"out_of_scope", "needs_clarification"}:
                    review_overrides[src_key] = {"action": action}
                continue
            if not isinstance(cfg, dict):
                continue
            action = normalize_classification_name(str(cfg.get("action") or ""))
            if action not in {"out_of_scope", "needs_clarification"}:
                continue
            payload = {"action": action}
            candidates = [
                normalize_classification_name(str(v or ""))
                for v in (cfg.get("clarification_wage_keys") or [])
                if normalize_classification_name(str(v or ""))
            ]
            if candidates:
                payload["clarification_wage_keys"] = candidates
            comment = str(cfg.get("comment") or "").strip()
            if comment:
                payload["comment"] = comment
            review_overrides[src_key] = payload
        for row in (raw.get("decision_notes") or []):
            if not isinstance(row, dict):
                continue
            src_key = normalize_classification_name(str(row.get("source_key") or ""))
            action = normalize_classification_name(str(row.get("action") or ""))
            if not src_key or not action:
                continue
            if action == "map":
                dst_key = normalize_classification_name(str(row.get("target_wage_key") or ""))
                if dst_key:
                    set_aliases[src_key] = dst_key
                continue
            if action in {"remove", "clear"}:
                remove_aliases.add(src_key)
                review_overrides.pop(src_key, None)
                continue
            if action in {"out_of_scope", "needs_clarification"}:
                payload = {"action": action}
                candidates = [
                    normalize_classification_name(str(v or ""))
                    for v in (row.get("clarification_wage_keys") or [])
                    if normalize_classification_name(str(v or ""))
                ]
                if candidates:
                    payload["clarification_wage_keys"] = candidates
                comment = str(row.get("comment") or "").strip()
                if comment:
                    payload["comment"] = comment
                review_overrides[src_key] = payload
        return set_aliases, remove_aliases, review_overrides

    if "classification_alias_overrides" in raw:
        raw_map = raw.get("classification_alias_overrides") or {}
        if not isinstance(raw_map, dict):
            raise ValueError("classification_alias_overrides must be an object.")
        for src, dst in raw_map.items():
            src_key = normalize_classification_name(str(src or ""))
            if not src_key:
                continue
            if dst is None or str(dst).strip() == "":
                remove_aliases.add(src_key)
                continue
            dst_key = normalize_classification_name(str(dst))
            if dst_key:
                set_aliases[src_key] = dst_key
        for src, cfg in (raw.get("classification_review_overrides") or {}).items():
            src_key = normalize_classification_name(str(src or ""))
            if not src_key:
                continue
            if isinstance(cfg, str):
                action = normalize_classification_name(cfg)
                if action in {"out_of_scope", "needs_clarification"}:
                    review_overrides[src_key] = {"action": action}
                continue
            if not isinstance(cfg, dict):
                continue
            action = normalize_classification_name(str(cfg.get("action") or ""))
            if action not in {"out_of_scope", "needs_clarification"}:
                continue
            payload = {"action": action}
            candidates = [
                normalize_classification_name(str(v or ""))
                for v in (cfg.get("clarification_wage_keys") or [])
                if normalize_classification_name(str(v or ""))
            ]
            if candidates:
                payload["clarification_wage_keys"] = candidates
            comment = str(cfg.get("comment") or "").strip()
            if comment:
                payload["comment"] = comment
            review_overrides[src_key] = payload
        return set_aliases, remove_aliases, review_overrides

    # Fallback: treat object as source->target map, null/empty means remove.
    for src, dst in raw.items():
        src_key = normalize_classification_name(str(src or ""))
        if not src_key:
            continue
        if dst is None or str(dst).strip() == "":
            remove_aliases.add(src_key)
            continue
        dst_key = normalize_classification_name(str(dst))
        if dst_key:
            set_aliases[src_key] = dst_key
    return set_aliases, remove_aliases, review_overrides


def _print_changes(current: dict[str, str], merged: dict[str, str]) -> None:
    added = sorted(k for k in merged if k not in current)
    updated = sorted(k for k in merged if k in current and current[k] != merged[k])
    removed = sorted(k for k in current if k not in merged)
    print("Override diff:")
    print(f"- added: {len(added)}")
    for key in added[:20]:
        print(f"  + {key} -> {merged[key]}")
    print(f"- updated: {len(updated)}")
    for key in updated[:20]:
        print(f"  ~ {key}: {current[key]} -> {merged[key]}")
    print(f"- removed: {len(removed)}")
    for key in removed[:20]:
        print(f"  - {key} (was {current[key]})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply reviewed classification overrides.")
    parser.add_argument("--contract-id", required=True, help="Contract package id under data/contracts/")
    parser.add_argument("--decision-file", help="Path to reviewed decision JSON.")
    parser.add_argument("--apply", action="store_true", help="Write merged overrides to manual override file.")
    parser.add_argument("--emit-template", action="store_true", help="Emit review decision template.")
    parser.add_argument("--template-out", help="Optional output path for template.")
    args = parser.parse_args()

    paths = _package_paths(args.contract_id)
    if not paths["manifest"].exists():
        print(f"[FAIL] Missing manifest: {paths['manifest']}")
        return 1
    if not paths["wages"].exists():
        print(f"[FAIL] Missing wages: {paths['wages']}")
        return 1

    manifest = _load_json(paths["manifest"])
    wages_data = _load_json(paths["wages"])

    current_overrides, current_review_overrides, current_warnings = load_manual_classification_review_overrides(
        paths["manual_overrides"], contract_id=args.contract_id
    )
    if current_warnings:
        print("Current override file warnings:")
        for w in current_warnings:
            print(f"- {w}")

    base_ontology = build_classification_ontology(
        contract_id=args.contract_id,
        manifest_classifications=manifest.get("classifications", []) or [],
        wages_data=wages_data,
        manual_alias_overrides=current_overrides,
        manual_review_overrides=current_review_overrides,
    )

    if args.emit_template:
        template = _build_template(args.contract_id, base_ontology)
        template_out = Path(args.template_out) if args.template_out else paths["review_template"]
        _write_json(template_out, template)
        print(f"[OK] Review template written: {template_out}")

    if not args.decision_file:
        print("No decision file provided; preview complete.")
        print(f"Current override entries: {len(current_overrides)}")
        print(f"Current reviewed entries: {len(current_review_overrides)}")
        print(f"Current ontology coverage: {(base_ontology.get('summary') or {}).get('coverage')}")
        return 0

    decision_path = Path(args.decision_file)
    if not decision_path.exists():
        print(f"[FAIL] Decision file not found: {decision_path}")
        return 1

    set_aliases, remove_aliases, review_overrides = _parse_decision_file(decision_path)

    wage_keys = set((wages_data.get("classifications") or {}).keys())
    invalid_targets = sorted(
        f"{src}->{dst}" for src, dst in set_aliases.items() if dst not in wage_keys
    )
    if invalid_targets:
        print("[FAIL] Some decision targets are not wage keys:")
        for item in invalid_targets[:50]:
            print(f"- {item}")
        return 1
    invalid_review_targets = sorted(
        f"{src}->{dst}"
        for src, cfg in review_overrides.items()
        for dst in (cfg.get("clarification_wage_keys") or [])
        if dst not in wage_keys
    )
    if invalid_review_targets:
        print("[FAIL] Some clarification targets are not wage keys:")
        for item in invalid_review_targets[:50]:
            print(f"- {item}")
        return 1

    merged = dict(current_overrides)
    merged_reviews = dict(current_review_overrides)
    for src in sorted(remove_aliases):
        merged.pop(src, None)
        merged_reviews.pop(src, None)
    for src, dst in sorted(set_aliases.items()):
        merged[src] = dst
        merged_reviews.pop(src, None)
    for src, cfg in sorted(review_overrides.items()):
        if src not in merged:
            merged_reviews[src] = cfg

    after_ontology = build_classification_ontology(
        contract_id=args.contract_id,
        manifest_classifications=manifest.get("classifications", []) or [],
        wages_data=wages_data,
        manual_alias_overrides=merged,
        manual_review_overrides=merged_reviews,
    )

    before_summary = base_ontology.get("summary", {}) or {}
    after_summary = after_ontology.get("summary", {}) or {}
    before_cov = float(before_summary.get("coverage", 0.0) or 0.0)
    after_cov = float(after_summary.get("coverage", 0.0) or 0.0)
    before_unresolved = set(before_summary.get("unresolved_manifest_keys", []) or [])
    after_unresolved = set(after_summary.get("unresolved_manifest_keys", []) or [])
    newly_resolved = sorted(before_unresolved - after_unresolved)
    newly_unresolved = sorted(after_unresolved - before_unresolved)

    print("=" * 72)
    print(f"Override Preview: {args.contract_id}")
    print("=" * 72)
    _print_changes(current_overrides, merged)
    print(f"Coverage before: {before_cov:.4f}")
    print(f"Coverage after:  {after_cov:.4f}")
    print(f"Coverage delta:  {after_cov - before_cov:+.4f}")
    print(f"Reviewed entries before: {len(current_review_overrides)}")
    print(f"Reviewed entries after:  {len(merged_reviews)}")
    print(f"Newly resolved keys: {newly_resolved}")
    print(f"Newly unresolved keys: {newly_unresolved}")

    if not args.apply:
        print("Dry run only. Re-run with --apply to write overrides.")
        return 0

    if paths["manual_overrides"].exists():
        backup_path = paths["manual_overrides"].with_suffix(
            f".json.bak_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )
        shutil.copy2(paths["manual_overrides"], backup_path)
        print(f"Backup written: {backup_path}")

    payload = {
        "schema_version": MANUAL_OVERRIDE_SCHEMA_VERSION,
        "contract_id": args.contract_id,
        "classification_alias_overrides": {k: merged[k] for k in sorted(merged)},
        "classification_review_overrides": {k: merged_reviews[k] for k in sorted(merged_reviews)},
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_decision_file": str(decision_path),
    }
    _write_json(paths["manual_overrides"], payload)
    print(f"[OK] Manual overrides updated: {paths['manual_overrides']}")
    print(f"Next step: rerun onboarding for this contract to materialize updated ontology/wages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

