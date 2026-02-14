"""
Validate contract manifests for multi-contract runtime and evaluation integrity.

Checks:
- Required fields exist and are non-empty
- contract_id matches manifest filename stem
- contract_version exists and matches term_start__term_end
- article title map integrity and article references in query routing
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import MANIFESTS_DIR


REQUIRED_TOP_LEVEL = [
    "contract_id",
    "union_local",
    "term_start",
    "term_end",
    "contract_version",
    "article_titles",
    "query_routing",
]

REQUIRED_ROUTING_KEYS = [
    "slang_to_contract",
    "topic_to_articles",
    "topic_patterns",
    "classification_to_articles",
]


def _is_non_empty_string(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_article_refs(
    refs: dict,
    valid_article_nums: set[int],
    error_prefix: str,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(refs, dict):
        return [f"{error_prefix} must be an object"]

    for key, articles in refs.items():
        if not isinstance(articles, list):
            errors.append(f"{error_prefix}.{key} must be a list of article numbers")
            continue
        for article_num in articles:
            if not isinstance(article_num, int):
                errors.append(f"{error_prefix}.{key} contains non-integer article '{article_num}'")
                continue
            if article_num not in valid_article_nums:
                errors.append(
                    f"{error_prefix}.{key} references unknown article {article_num}"
                )
    return errors


def validate_manifest(path: Path) -> list[str]:
    errors: list[str] = []

    try:
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as exc:
        return [f"Invalid JSON: {exc}"]

    for key in REQUIRED_TOP_LEVEL:
        if key not in manifest:
            errors.append(f"Missing required field '{key}'")

    contract_id = manifest.get("contract_id")
    if not _is_non_empty_string(contract_id):
        errors.append("contract_id must be a non-empty string")
    elif contract_id != path.stem:
        errors.append(
            f"contract_id '{contract_id}' must match filename '{path.stem}.json'"
        )

    union_local = manifest.get("union_local")
    if not _is_non_empty_string(union_local):
        errors.append("union_local must be a non-empty string")

    term_start = manifest.get("term_start")
    term_end = manifest.get("term_end")
    if not _is_non_empty_string(term_start):
        errors.append("term_start must be a non-empty string")
    if not _is_non_empty_string(term_end):
        errors.append("term_end must be a non-empty string")

    expected_version = f"{term_start}__{term_end}"
    contract_version = manifest.get("contract_version")
    if not _is_non_empty_string(contract_version):
        errors.append("contract_version must be a non-empty string")
    elif contract_version != expected_version:
        errors.append(
            "contract_version mismatch: "
            f"expected '{expected_version}', got '{contract_version}'"
        )

    article_titles = manifest.get("article_titles")
    if not isinstance(article_titles, dict) or not article_titles:
        errors.append("article_titles must be a non-empty object")
        valid_article_nums: set[int] = set()
    else:
        valid_article_nums = set()
        for article_key, title in article_titles.items():
            if not _is_non_empty_string(str(article_key)):
                errors.append(f"Invalid article key '{article_key}'")
                continue
            try:
                article_num = int(article_key)
            except ValueError:
                errors.append(f"article_titles key '{article_key}' must be numeric")
                continue
            if article_num <= 0:
                errors.append(f"article_titles key '{article_key}' must be > 0")
                continue
            if not _is_non_empty_string(title):
                errors.append(f"article_titles[{article_key}] must be a non-empty string")
            valid_article_nums.add(article_num)

    total_articles = manifest.get("total_articles")
    if total_articles is not None and total_articles != len(valid_article_nums):
        errors.append(
            "total_articles mismatch: "
            f"expected {len(valid_article_nums)}, got {total_articles}"
        )

    routing = manifest.get("query_routing")
    if not isinstance(routing, dict):
        errors.append("query_routing must be an object")
        return errors

    for key in REQUIRED_ROUTING_KEYS:
        if key not in routing:
            errors.append(f"query_routing missing '{key}'")

    slang_to_contract = routing.get("slang_to_contract", {})
    if not isinstance(slang_to_contract, dict):
        errors.append("query_routing.slang_to_contract must be an object")
    else:
        for key, value in slang_to_contract.items():
            if not _is_non_empty_string(key) or not _is_non_empty_string(value):
                errors.append(
                    "query_routing.slang_to_contract entries must map non-empty strings"
                )

    topic_patterns = routing.get("topic_patterns", {})
    if not isinstance(topic_patterns, dict):
        errors.append("query_routing.topic_patterns must be an object")
    else:
        for key, value in topic_patterns.items():
            if not _is_non_empty_string(key) or not _is_non_empty_string(value):
                errors.append(
                    "query_routing.topic_patterns entries must map non-empty strings"
                )

    topic_to_articles = routing.get("topic_to_articles", {})
    errors.extend(
        _validate_article_refs(
            topic_to_articles,
            valid_article_nums,
            "query_routing.topic_to_articles",
        )
    )

    classification_to_articles = routing.get("classification_to_articles", {})
    errors.extend(
        _validate_article_refs(
            classification_to_articles,
            valid_article_nums,
            "query_routing.classification_to_articles",
        )
    )

    return errors


def main() -> int:
    manifests = sorted(MANIFESTS_DIR.glob("*.json"))
    if not manifests:
        print(f"[FAIL] No manifests found in {MANIFESTS_DIR}")
        return 1

    failures = 0
    for manifest_path in manifests:
        errors = validate_manifest(manifest_path)
        if errors:
            failures += 1
            print(f"[FAIL] {manifest_path}")
            for err in errors:
                print(f"  - {err}")
        else:
            print(f"[OK] {manifest_path}")

    if failures:
        print(f"\nManifest validation failed: {failures}/{len(manifests)} invalid.")
        return 1

    print(f"\nManifest validation passed: {len(manifests)}/{len(manifests)} valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
