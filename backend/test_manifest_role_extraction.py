"""
Deterministic manifest role-extraction checks.

Guards against incidental role mentions (exclusion clauses, experience-credit
matrices) polluting manifest.classifications for onboarding/runtime routing.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.ingest.manifest import extract_manifest


def _test_kingsoopers_meat_filters_incidental_roles() -> None:
    contract_id = "local7_kingsoopers_loveland_meat_2019"
    source_path = Path(
        "data/contracts/local7_kingsoopers_loveland_meat_2019/source/local7_kingsoopers_loveland_meat_2019.md"
    )
    manifest = extract_manifest(source_path, contract_id=contract_id)
    classes = set(manifest.classifications)

    assert "Meat Cutter" in classes, "Expected core meat role in extracted classifications."
    assert "Cake Decorator" not in classes, (
        "Cake Decorator should not be extracted from incidental experience-credit mentions."
    )
    assert "Pharmacy Technician" not in classes, (
        "Pharmacy Technician should not be extracted from non-rate matrix references."
    )


def main() -> None:
    _test_kingsoopers_meat_filters_incidental_roles()
    print("[OK] Manifest role extraction checks passed")


if __name__ == "__main__":
    main()

