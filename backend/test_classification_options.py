"""
Deterministic classification option integrity checks.

Verifies contract-scoped role options distinguish wage-resolvable vs unresolved
roles and avoid surfacing unresolved options in default onboarding lists.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.ingest.extract_wages import normalize_classification_name
from backend.user.profile import get_classification_options


def _by_value(options: list[dict]) -> dict[str, dict]:
    return {str(o.get("value") or ""): o for o in options}


def _test_kingsoopers_meat_hides_unmapped_by_default() -> None:
    contract_id = "local7_kingsoopers_loveland_meat_2019"
    default_opts = _by_value(get_classification_options(contract_id=contract_id))
    all_opts = _by_value(get_classification_options(contract_id=contract_id, include_unmapped=True))

    assert "meat_manager" in default_opts, "Expected wage role in default options."
    for value, opt in default_opts.items():
        assert opt.get("wage_available") is not False, (
            f"Default onboarding option must be wage-resolvable: {value}"
        )

    unresolved_all = [v for v, o in all_opts.items() if o.get("wage_available") is False]
    for value in unresolved_all:
        assert value not in default_opts, (
            f"Unmapped role should be excluded from default onboarding options: {value}"
        )


def _test_clerks_contract_keeps_cake_decorator_wage_role() -> None:
    contract_id = "local7_safeway_pueblo_clerks_2022"
    default_opts = _by_value(get_classification_options(contract_id=contract_id))

    assert "cake_decorator" in default_opts, "Expected cake_decorator in clerks wage options."
    assert default_opts["cake_decorator"].get("wage_available") is True, (
        "Expected clerks cake_decorator to be wage-available."
    )


def _test_pueblo_meat_normalized_role_values_are_unique() -> None:
    contract_id = "local7_safeway_pueblo_meat_2022"
    options = get_classification_options(contract_id=contract_id)

    normalized_values = [normalize_classification_name(str(o.get("value") or "")) for o in options]
    duplicates = sorted(
        v for v in set(normalized_values)
        if v and normalized_values.count(v) > 1
    )
    assert not duplicates, (
        "Expected normalized role values to be unique in default options for "
        f"{contract_id}; duplicates: {duplicates}"
    )


def main() -> None:
    _test_kingsoopers_meat_hides_unmapped_by_default()
    _test_clerks_contract_keeps_cake_decorator_wage_role()
    _test_pueblo_meat_normalized_role_values_are_unique()
    print("[OK] Classification option checks passed")


if __name__ == "__main__":
    main()
