"""
Onboard contract packages from data/contracts into KARL runtime artifacts.

Pipeline per package:
1) detect source markdown/json
2) generate manifest
3) generate smart chunks
4) apply structured tables (if JSON source present)
5) extract wage table artifact (best-effort)
6) sync canonical artifacts into data/{manifests,chunks,wages,tables}

By default, only packages that have source markdown and no manifest JSON yet
are processed.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import DATA_DIR
from backend.ingest.smart_chunker import SmartChunker
from backend.ingest.manifest import extract_manifest
from backend.ingest.extract_wages import extract_wages
from backend.ingest.extract_entitlements import extract_entitlements, save_entitlements
from backend.ingest.classification_ontology import (
    build_classification_ontology,
    apply_ontology_aliases,
    save_classification_ontology,
    load_manual_classification_overrides,
    write_manual_override_template,
)
from backend.ingest.review_queue import (
    build_ingestion_review_queue,
    save_ingestion_review_queue,
)
from backend.ingest.table_extractor import (
    build_table_registry,
    save_table_registry,
    apply_tables_to_chunks,
    synthesize_unmatched_table_chunks,
)
from backend.ingest.toc_index import build_concept_index
from backend.ingest.language_lexicon import (
    ensure_manifest_region_id,
    apply_deterministic_language_enrichment,
    build_language_lexicon,
    save_language_lexicon,
)
from backend.ingest.query_routing import (
    synthesize_query_routing,
    merge_query_routing,
)
from backend.ingest.role_catalog import (
    build_role_catalog,
    save_role_catalog,
)
from backend.ingest.pack_acceptance import evaluate_contract_pack


CONTRACTS_ROOT = DATA_DIR / "contracts"
RUNTIME_MANIFESTS = DATA_DIR / "manifests"
RUNTIME_CHUNKS = DATA_DIR / "chunks"
RUNTIME_WAGES = DATA_DIR / "wages"
RUNTIME_TABLES = DATA_DIR / "tables"
RUNTIME_ONTOLOGIES = DATA_DIR / "ontologies"
RUNTIME_ENTITLEMENTS = DATA_DIR / "entitlements"
PACK_REGISTRY_FILE = CONTRACTS_ROOT / "pack_registry.json"


def _find_source_file(source_dir: Path, suffix: str, preferred_stem: str) -> Optional[Path]:
    files = sorted(source_dir.glob(f"*{suffix}"))
    if not files:
        return None
    for f in files:
        if f.stem == preferred_stem:
            return f
    for f in files:
        if preferred_stem in f.stem:
            return f
    return files[0]


def _infer_term_dates(md_text: str, md_name: str, contract_id: str) -> tuple[str, str]:
    month_date = re.findall(
        r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}",
        md_text,
        flags=re.IGNORECASE,
    )
    if len(month_date) >= 2:
        return month_date[0], month_date[1]

    year_pair = re.search(r"(20\d{2})[.\-_](20\d{2})", md_name)
    if not year_pair:
        year_pair = re.search(r"(20\d{2}).{0,10}(20\d{2})", contract_id)
    if year_pair:
        start_year, end_year = year_pair.group(1), year_pair.group(2)
        return f"January 1, {start_year}", f"December 31, {end_year}"

    return "UNKNOWN_START", "UNKNOWN_END"


def _infer_union_local(contract_id: str, current: str) -> str:
    if current and current != "Unknown Union":
        return current
    m = re.search(r"local[_-]?(\d+)", contract_id, flags=re.IGNORECASE)
    if m:
        return f"UFCW Local {m.group(1)}"
    return "Unknown Union"


def _ensure_manifest_shape(manifest: dict, md_text: str, md_name: str, contract_id: str) -> dict:
    term_start = manifest.get("term_start")
    term_end = manifest.get("term_end")
    if not term_start or not term_end:
        inferred_start, inferred_end = _infer_term_dates(md_text, md_name, contract_id)
        term_start = term_start or inferred_start
        term_end = term_end or inferred_end
    manifest["term_start"] = term_start
    manifest["term_end"] = term_end
    manifest["contract_id"] = contract_id
    manifest["union_local"] = _infer_union_local(contract_id, manifest.get("union_local", ""))
    manifest["contract_version"] = f"{term_start}__{term_end}"

    article_titles = manifest.get("article_titles", {}) or {}
    manifest["article_titles"] = {str(k): v for k, v in article_titles.items()}
    manifest["total_articles"] = len(manifest["article_titles"])

    routing = manifest.get("query_routing") or {}
    manifest["query_routing"] = {
        "slang_to_contract": routing.get("slang_to_contract", {}),
        "topic_to_articles": routing.get("topic_to_articles", {}),
        "topic_patterns": routing.get("topic_patterns", {}),
        "classification_to_articles": routing.get("classification_to_articles", {}),
    }
    return ensure_manifest_region_id(manifest, contract_id=contract_id)


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _sync(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _load_pack_registry() -> dict:
    if not PACK_REGISTRY_FILE.exists():
        return {"schema_version": "contract_pack_registry_v1", "accepted_packs": {}}
    try:
        with open(PACK_REGISTRY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {"schema_version": "contract_pack_registry_v1", "accepted_packs": {}}
    if not isinstance(data, dict):
        return {"schema_version": "contract_pack_registry_v1", "accepted_packs": {}}
    data.setdefault("schema_version", "contract_pack_registry_v1")
    data.setdefault("accepted_packs", {})
    return data


def _write_pack_registry(registry: dict) -> None:
    PACK_REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PACK_REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def _record_pack_acceptance(contract_id: str, scorecard: dict) -> None:
    registry = _load_pack_registry()
    accepted = registry.setdefault("accepted_packs", {})
    accepted[contract_id] = {
        "contract_id": contract_id,
        "pack_hash": scorecard.get("pack_hash"),
        "scorecard_path": scorecard.get("scorecard_path"),
        "strict_mode": scorecard.get("strict_mode", False),
        "accepted_at_utc": datetime.now(timezone.utc).isoformat(),
        "required_failed": scorecard.get("summary", {}).get("required_failed", []),
        "advisory_failed": scorecard.get("summary", {}).get("advisory_failed", []),
    }
    _write_pack_registry(registry)


def _filter_plausible_wage_tables(wages_data: dict) -> dict:
    """
    Drop likely non-wage tables (benefits/plan tables) from wage artifact.

    Keep role keys when they have real numeric rates, even if the label is
    uncommon (e.g., `5star_cake_decorator`). Avoid brittle token whitelists.
    """
    classes = wages_data.get("classifications", {}) or {}
    if not classes:
        return wages_data

    generic_drop_labels = {
        "grandfathered",
        "minimum_wage",
        "effective",
        "classification",
        "rate",
        "rates",
    }
    exclude_tokens = {
        "plan", "medical", "medicare", "prescription", "dental", "vision",
        "premium", "investment", "income", "matrix",
    }

    filtered = {}
    for key, cls in classes.items():
        name = (cls.get("name") or key or "").lower()
        if any(tok in name for tok in exclude_tokens):
            continue

        steps = cls.get("steps", []) or []
        has_numeric_rate = False
        for step in steps:
            rates = step.get("rates", {}) or {}
            if any(isinstance(v, (int, float)) and v > 0 for v in rates.values()):
                has_numeric_rate = True
                break
        if not has_numeric_rate:
            continue
        if key in generic_drop_labels:
            continue

        filtered[key] = cls

    wages_data = dict(wages_data)
    wages_data["classifications"] = filtered
    return wages_data


def _process_package(
    package_dir: Path,
    build_tables: bool,
    build_wages: bool,
    sync_runtime: bool,
    run_pack_gates: bool,
    enforce_pack_gates: bool,
    strict_pack_gates: bool,
) -> dict:
    contract_id = package_dir.name
    source_dir = package_dir / "source"
    chunks_dir = package_dir / "chunks"
    manifests_dir = package_dir / "manifests"
    tables_dir = package_dir / "tables"
    wages_dir = package_dir / "wages"
    entitlements_dir = package_dir / "entitlements"
    ontology_dir = package_dir / "ontology"

    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source directory: {source_dir}")

    md_path = _find_source_file(source_dir, ".md", preferred_stem=contract_id)
    if md_path is None:
        raise FileNotFoundError(f"No markdown source found in {source_dir}")

    json_path = _find_source_file(source_dir, ".json", preferred_stem=contract_id)

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Manifest
    manifest_obj = extract_manifest(md_path, contract_id=contract_id)
    manifest = _ensure_manifest_shape(manifest_obj.to_dict(), md_text, md_path.name, contract_id)
    manifest_path = manifests_dir / f"{contract_id}.json"
    _write_json(manifest_path, manifest)

    # Chunks
    chunker = SmartChunker(contract_id=contract_id)
    raw_chunks = [c.to_dict() for c in chunker.parse_markdown(md_path)]
    smart_chunks = raw_chunks

    table_registry_path = None
    table_registry_for_wages = None
    table_chunk_synthesis = {
        "added_table_chunks": 0,
        "unmatched_before": 0,
        "unmatched_after": 0,
        "skipped_toc_tables": 0,
    }
    enriched_chunks = [dict(c) for c in smart_chunks]

    if build_tables and json_path is not None:
        registry = build_table_registry(json_path=json_path)
        table_registry_path = save_table_registry(registry, output_dir=tables_dir)
        enriched_chunks, updated_registry = apply_tables_to_chunks(enriched_chunks, registry)
        enriched_chunks, table_chunk_synthesis = synthesize_unmatched_table_chunks(
            enriched_chunks,
            updated_registry,
        )
        save_table_registry(updated_registry, output_dir=tables_dir)
        table_registry_for_wages = updated_registry
    else:
        for c in enriched_chunks:
            c.setdefault("content_with_tables", c.get("content", ""))
            c.setdefault("table_refs", [])

    if table_registry_for_wages is None:
        existing_registry_path = tables_dir / "structured_tables.json"
        if existing_registry_path.exists():
            with open(existing_registry_path, "r", encoding="utf-8") as f:
                table_registry_for_wages = json.load(f)
    if (not build_tables or json_path is None) and table_registry_for_wages is not None:
        enriched_chunks, table_chunk_synthesis = synthesize_unmatched_table_chunks(
            enriched_chunks,
            table_registry_for_wages,
        )

    enriched_chunks, language_enrichment_stats = apply_deterministic_language_enrichment(
        enriched_chunks,
        contract_id=contract_id,
        manifest=manifest,
    )

    smart_path = chunks_dir / f"contract_chunks_smart_{contract_id}.json"
    enriched_path = chunks_dir / f"contract_chunks_enriched_{contract_id}.json"
    base_path = chunks_dir / f"contract_chunks_{contract_id}.json"
    concept_index_path = chunks_dir / f"concept_index_{contract_id}.json"
    language_lexicon_path = ontology_dir / "language_lexicon.json"

    _write_json(smart_path, smart_chunks)
    _write_json(enriched_path, enriched_chunks)
    _write_json(base_path, enriched_chunks)
    language_lexicon = build_language_lexicon(
        enriched_chunks,
        contract_id=contract_id,
        manifest=manifest,
    )
    save_language_lexicon(language_lexicon_path, language_lexicon)
    build_concept_index(
        chunks_path=enriched_path,
        output_path=concept_index_path,
        manifest=manifest,
    )
    concept_index_data = {}
    if concept_index_path.exists():
        with open(concept_index_path, "r", encoding="utf-8") as f:
            concept_index_data = json.load(f)

    # Package-local compatibility aliases
    _write_json(chunks_dir / "contract_chunks_smart.json", smart_chunks)
    _write_json(chunks_dir / "contract_chunks_enriched.json", enriched_chunks)
    _write_json(chunks_dir / "contract_chunks.json", enriched_chunks)

    wage_path = None
    ontology_path = None
    role_catalog_path = ontology_dir / "role_catalog.json"
    review_queue_path = None
    ontology_data = None
    role_catalog = None
    manual_override_path = ontology_dir / "manual_classification_overrides.json"
    wage_classification_count = 0
    ontology_coverage = None
    role_catalog_summary = {}
    review_queue_items = 0
    if build_wages:
        write_manual_override_template(manual_override_path, contract_id=contract_id)
        manual_alias_overrides, manual_override_warnings = load_manual_classification_overrides(
            path=manual_override_path,
            contract_id=contract_id,
        )
        wages_data = extract_wages(
            md_content=md_text,
            tables=table_registry_for_wages,
            contract_id=contract_id,
        )
        wages_data = _filter_plausible_wage_tables(wages_data)
        ontology_data = build_classification_ontology(
            contract_id=contract_id,
            manifest_classifications=manifest.get("classifications", []),
            wages_data=wages_data,
            manual_alias_overrides=manual_alias_overrides,
        )
        ontology_path = ontology_dir / "classification_ontology.json"
        save_classification_ontology(ontology_path, ontology_data)
        ontology_coverage = (ontology_data.get("summary") or {}).get("coverage")
        wages_data = apply_ontology_aliases(wages_data, ontology_data)
        review_queue = build_ingestion_review_queue(
            contract_id=contract_id,
            ontology=ontology_data,
            wages_data=wages_data,
            manual_override_warnings=manual_override_warnings,
        )
        review_queue_path = ontology_dir / "ingestion_review_queue.json"
        save_ingestion_review_queue(review_queue_path, review_queue)
        review_queue_items = int((review_queue.get("summary") or {}).get("total_items", 0) or 0)
        wage_classification_count = len(wages_data.get("classifications", {}))
        wage_path = wages_dir / f"wage_tables_{contract_id}.json"
        _write_json(wage_path, wages_data)

    if manifest.get("classifications"):
        role_catalog_wages = None
        if build_wages and wage_path is not None and wage_path.exists():
            role_catalog_wages = wages_data
        else:
            existing_wage_path = wages_dir / f"wage_tables_{contract_id}.json"
            if existing_wage_path.exists():
                with open(existing_wage_path, "r", encoding="utf-8") as f:
                    role_catalog_wages = json.load(f)

        role_catalog_ontology = None
        if ontology_path is not None and ontology_path.exists():
            role_catalog_ontology = ontology_data
        else:
            existing_ontology_path = ontology_dir / "classification_ontology.json"
            if existing_ontology_path.exists():
                with open(existing_ontology_path, "r", encoding="utf-8") as f:
                    role_catalog_ontology = json.load(f)

        role_catalog = build_role_catalog(
            contract_id=contract_id,
            manifest=manifest,
            wages_data=role_catalog_wages or {},
            classification_ontology=role_catalog_ontology or {},
        )
        save_role_catalog(role_catalog_path, role_catalog)
        role_catalog_summary = role_catalog.get("summary", {}) or {}

    entitlement_data = extract_entitlements(
        chunks=enriched_chunks,
        contract_id=contract_id,
        manifest=manifest,
    )
    entitlement_path = entitlements_dir / f"entitlement_tables_{contract_id}.json"
    save_entitlements(entitlement_data, entitlement_path)
    save_entitlements(entitlement_data, entitlements_dir / "entitlement_tables.json")

    generated_routing, routing_stats = synthesize_query_routing(
        manifest=manifest,
        concept_index=concept_index_data,
        language_lexicon=language_lexicon,
        classification_ontology=ontology_data,
    )
    manifest["query_routing"] = merge_query_routing(
        generated=generated_routing,
        existing=manifest.get("query_routing") or {},
    )
    _write_json(manifest_path, manifest)

    pack_scorecard = None
    pack_gate_pass = True
    if run_pack_gates:
        pack_scorecard = evaluate_contract_pack(
            package_dir=package_dir,
            strict=strict_pack_gates,
            write_scorecard=True,
        )
        pack_gate_pass = bool(pack_scorecard.get("summary", {}).get("pass"))

    runtime_sync_allowed = sync_runtime and (
        (not enforce_pack_gates) or pack_gate_pass
    )
    if runtime_sync_allowed:
        _sync(manifest_path, RUNTIME_MANIFESTS / f"{contract_id}.json")
        _sync(base_path, RUNTIME_CHUNKS / f"contract_chunks_{contract_id}.json")
        _sync(smart_path, RUNTIME_CHUNKS / f"contract_chunks_smart_{contract_id}.json")
        _sync(enriched_path, RUNTIME_CHUNKS / f"contract_chunks_enriched_{contract_id}.json")
        if concept_index_path.exists():
            _sync(concept_index_path, RUNTIME_CHUNKS / f"concept_index_{contract_id}.json")
        if wage_path is not None and wage_path.exists():
            _sync(wage_path, RUNTIME_WAGES / f"wage_tables_{contract_id}.json")
        if table_registry_path is not None and table_registry_path.exists():
            _sync(table_registry_path, RUNTIME_TABLES / f"structured_tables_{contract_id}.json")
        if ontology_path is not None and ontology_path.exists():
            _sync(ontology_path, RUNTIME_ONTOLOGIES / f"classification_ontology_{contract_id}.json")
        if language_lexicon_path.exists():
            _sync(language_lexicon_path, RUNTIME_ONTOLOGIES / f"language_lexicon_{contract_id}.json")
        if role_catalog_path.exists():
            _sync(role_catalog_path, RUNTIME_ONTOLOGIES / f"role_catalog_{contract_id}.json")
        if entitlement_path.exists():
            _sync(entitlement_path, RUNTIME_ENTITLEMENTS / f"entitlement_tables_{contract_id}.json")
        if run_pack_gates and pack_gate_pass and pack_scorecard is not None:
            _record_pack_acceptance(contract_id, pack_scorecard)

    return {
        "contract_id": contract_id,
        "source_md": str(md_path),
        "source_json": str(json_path) if json_path else None,
        "manifest_path": str(manifest_path),
        "chunk_count": len(enriched_chunks),
        "concept_index_path": str(concept_index_path),
        "language_lexicon_path": str(language_lexicon_path),
        "language_lexicon_entries": len(language_lexicon.get("entries", [])),
        "query_routing_stats": routing_stats,
        "language_enrichment_stats": language_enrichment_stats,
        "table_chunk_synthesis": table_chunk_synthesis,
        "wage_classification_count": wage_classification_count,
        "classification_ontology_path": str(ontology_path) if ontology_path else None,
        "classification_ontology_coverage": ontology_coverage,
        "role_catalog_path": str(role_catalog_path) if role_catalog_path.exists() else None,
        "role_catalog_total_roles": int(role_catalog_summary.get("total_roles", 0) or 0),
        "role_catalog_unresolved_manifest_roles": (
            role_catalog_summary.get("unresolved_manifest_roles", []) if role_catalog_summary else []
        ),
        "ingestion_review_queue_path": str(review_queue_path) if review_queue_path else None,
        "ingestion_review_queue_items": review_queue_items,
        "entitlement_path": str(entitlement_path),
        "vacation_schedule_count": len(entitlement_data.get("vacation_entitlements") or []),
        "runtime_synced": runtime_sync_allowed,
        "pack_gate_ran": run_pack_gates,
        "pack_gate_pass": pack_gate_pass if run_pack_gates else None,
        "pack_scorecard_path": pack_scorecard.get("scorecard_path") if pack_scorecard else None,
        "pack_hash": pack_scorecard.get("pack_hash") if pack_scorecard else None,
        "pack_required_failed": (
            pack_scorecard.get("summary", {}).get("required_failed", [])
            if pack_scorecard else []
        ),
        "pack_advisory_failed": (
            pack_scorecard.get("summary", {}).get("advisory_failed", [])
            if pack_scorecard else []
        ),
    }


def _discover_default_packages() -> list[Path]:
    packages = []
    for p in sorted(CONTRACTS_ROOT.iterdir()):
        if not p.is_dir():
            continue
        if p.name.lower() == "contractid":
            continue
        source_dir = p / "source"
        if not source_dir.exists():
            continue
        has_md = any(source_dir.glob("*.md"))
        has_manifest = any((p / "manifests").glob("*.json")) if (p / "manifests").exists() else False
        if has_md and not has_manifest:
            packages.append(p)
    return packages


def main() -> int:
    parser = argparse.ArgumentParser(description="Onboard contract packages into KARL runtime artifacts.")
    parser.add_argument(
        "--package",
        action="append",
        default=[],
        help="Package directory name under data/contracts (repeatable).",
    )
    parser.add_argument("--no-tables", action="store_true", help="Skip table registry/apply pipeline.")
    parser.add_argument("--no-wages", action="store_true", help="Skip wage extraction.")
    parser.add_argument("--no-sync-runtime", action="store_true", help="Do not copy artifacts into runtime dirs.")
    parser.add_argument("--no-pack-gates", action="store_true", help="Skip contract-pack acceptance scorecard generation.")
    parser.add_argument("--enforce-pack-gates", action="store_true", help="Block runtime sync when required pack gates fail.")
    parser.add_argument("--strict-pack-gates", action="store_true", help="Treat advisory pack-gate failures as blocking.")
    args = parser.parse_args()

    if args.package:
        package_dirs = [CONTRACTS_ROOT / name for name in args.package]
    else:
        package_dirs = _discover_default_packages()

    if not package_dirs:
        print("No contract packages selected for onboarding.")
        return 0

    print("=" * 72)
    print("KARL Contract Package Onboarding")
    print("=" * 72)
    print(f"Packages: {[p.name for p in package_dirs]}")

    results = []
    failures = []
    for package_dir in package_dirs:
        try:
            result = _process_package(
                package_dir=package_dir,
                build_tables=not args.no_tables,
                build_wages=not args.no_wages,
                sync_runtime=not args.no_sync_runtime,
                run_pack_gates=not args.no_pack_gates,
                enforce_pack_gates=args.enforce_pack_gates,
                strict_pack_gates=args.strict_pack_gates,
            )
            results.append(result)
            gate_suffix = ""
            if result.get("pack_gate_ran"):
                gate_suffix = (
                    f" gates={'PASS' if result.get('pack_gate_pass') else 'FAIL'} "
                    f"hash={result.get('pack_hash')}"
                )
            print(
                f"[OK] {package_dir.name} -> chunks={result['chunk_count']} "
                f"wages={result['wage_classification_count']} synced={result['runtime_synced']} "
                f"table_chunks+={result.get('table_chunk_synthesis', {}).get('added_table_chunks', 0)} "
                f"routing_topics={result.get('query_routing_stats', {}).get('topic_entries')} "
                f"ontology_cov={result.get('classification_ontology_coverage')} "
                f"role_catalog={result.get('role_catalog_total_roles', 0)} "
                f"review_items={result.get('ingestion_review_queue_items', 0)}{gate_suffix}"
            )
        except Exception as exc:
            failures.append((package_dir.name, str(exc)))
            print(f"[FAIL] {package_dir.name}: {exc}")

    print("\nSummary:")
    print(f"- processed: {len(results)}")
    print(f"- failed: {len(failures)}")
    for r in results:
        print(
            f"  - {r['contract_id']}: chunks={r['chunk_count']} "
            f"wages={r['wage_classification_count']} synced={r['runtime_synced']} "
            f"routing_topics={r.get('query_routing_stats', {}).get('topic_entries')} "
            f"ontology_cov={r.get('classification_ontology_coverage')} "
            f"role_catalog={r.get('role_catalog_total_roles', 0)} "
            f"review_items={r.get('ingestion_review_queue_items', 0)} "
            f"gates={r.get('pack_gate_pass')}"
        )
        if r.get("pack_scorecard_path"):
            print(f"    scorecard={r['pack_scorecard_path']}")
        if r.get("pack_required_failed"):
            print(f"    required_failed={r['pack_required_failed']}")
        if r.get("pack_advisory_failed"):
            print(f"    advisory_failed={r['pack_advisory_failed']}")
    for name, err in failures:
        print(f"  - FAIL {name}: {err}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
