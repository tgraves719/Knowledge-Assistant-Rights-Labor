"""
Contract-scoped role-catalog integrity checks.

Verifies role-catalog artifacts are present, schema-valid, and enforce
onboarding defaults that are wage-resolvable.
"""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.role_catalog_files import resolve_role_catalog_file
from backend.user.profile import get_classification_options


CONTRACT_IDS = [
    "local7_kingsoopers_loveland_meat_2019",
    "local7_safeway_pueblo_meat_2022",
    "local7_safeway_pueblo_clerks_2022",
]


def _load_role_catalog(contract_id: str) -> dict:
    path = resolve_role_catalog_file(contract_id=contract_id, allow_shared_fallback=False)
    assert path is not None and path.exists(), f"Missing role catalog for {contract_id}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _test_role_catalog_schema_and_default_wage_integrity() -> None:
    for contract_id in CONTRACT_IDS:
        catalog = _load_role_catalog(contract_id)
        assert catalog.get("schema_version") == "role_catalog_v2", (
            f"Unexpected role-catalog schema for {contract_id}"
        )
        assert catalog.get("contract_id") == contract_id, (
            f"Role-catalog contract_id mismatch for {contract_id}"
        )
        roles = catalog.get("roles") or []
        assert isinstance(roles, list), f"Role catalog roles must be a list for {contract_id}"
        default_unmapped = [
            str(role.get("value") or "")
            for role in roles
            if isinstance(role, dict)
            and bool(role.get("onboarding_default"))
            and not bool(role.get("wage_available"))
        ]
        assert not default_unmapped, (
            f"Onboarding default roles must be wage-available for {contract_id}: {default_unmapped}"
        )
        defaults_by_wage_key: dict[str, int] = {}
        for role in roles:
            if not isinstance(role, dict):
                continue
            if not bool(role.get("onboarding_default")):
                continue
            wage_key = str(role.get("wage_key") or "").strip()
            if not wage_key:
                continue
            defaults_by_wage_key[wage_key] = defaults_by_wage_key.get(wage_key, 0) + 1
        duplicate_default_keys = sorted(
            key for key, count in defaults_by_wage_key.items() if count > 1
        )
        assert not duplicate_default_keys, (
            "Expected at most one onboarding-default role per wage_key "
            f"for {contract_id}; duplicates: {duplicate_default_keys}"
        )


def _test_default_classification_options_exclude_unresolved_manifest_roles() -> None:
    for contract_id in CONTRACT_IDS:
        default_values = {
            str(opt.get("value") or "")
            for opt in get_classification_options(contract_id=contract_id)
        }
        all_options = get_classification_options(contract_id=contract_id, include_unmapped=True)
        unresolved_manifest = {
            str(opt.get("value") or "")
            for opt in all_options
            if bool(opt.get("manifest_present")) and not bool(opt.get("wage_available"))
        }
        assert unresolved_manifest.isdisjoint(default_values), (
            f"Unresolved manifest roles leaked into default options for {contract_id}: "
            f"{sorted(unresolved_manifest & default_values)}"
        )


def _test_clerks_dug_aliases_not_duplicated_in_default_options() -> None:
    contract_id = "local7_safeway_pueblo_clerks_2022"
    default_values = {
        str(opt.get("value") or "")
        for opt in get_classification_options(contract_id=contract_id)
    }
    alias_values = {"dug_shopper", "drive_up_and_go"}
    assert len(default_values & alias_values) <= 1, (
        "Expected wage-equivalent DUG aliases to collapse to a single default "
        f"option; got {sorted(default_values & alias_values)}"
    )


def _test_reviewed_manifest_roles_are_not_counted_as_unresolved() -> None:
    kings_catalog = _load_role_catalog("local7_kingsoopers_loveland_meat_2019")
    kings_summary = kings_catalog.get("summary") or {}
    assert "head_clerk" in set(kings_summary.get("clarification_manifest_roles") or []), (
        "Expected head_clerk to be preserved as a reviewed clarification role."
    )
    assert "head_clerk" not in set(kings_summary.get("unresolved_manifest_roles") or []), (
        "Reviewed clarification roles should not count as unresolved."
    )

    meat_catalog = _load_role_catalog("local7_safeway_pueblo_meat_2022")
    meat_summary = meat_catalog.get("summary") or {}
    expected_out_of_scope = {"cake_decorator", "courtesy_clerk", "pharmacy_technician"}
    assert expected_out_of_scope.issubset(set(meat_summary.get("out_of_scope_manifest_roles") or [])), (
        "Expected reviewed out-of-scope manifest classes to be preserved in the role catalog."
    )
    assert expected_out_of_scope.isdisjoint(set(meat_summary.get("unresolved_manifest_roles") or [])), (
        "Reviewed out-of-scope roles should not count as unresolved."
    )


def main() -> None:
    _test_role_catalog_schema_and_default_wage_integrity()
    _test_default_classification_options_exclude_unresolved_manifest_roles()
    _test_clerks_dug_aliases_not_duplicated_in_default_options()
    _test_reviewed_manifest_roles_are_not_counted_as_unresolved()
    print("[OK] Role catalog checks passed")


if __name__ == "__main__":
    main()
