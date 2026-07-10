from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.ingest.moa_wage_schedule_configs import PUEBLO_CLERKS_2025_07_05
from backend.ingest.moa_wage_schedule_sync import build_row_ops, extract_schedule_rates
from backend.ingest.seed_moa_wage_rollforward_ops import _seed_ops


def test_generic_moa_schedule_sync_preserves_raw_schedule_columns() -> None:
    rate_map = extract_schedule_rates(PUEBLO_CLERKS_2025_07_05)
    courtesy_start = rate_map[("courtesy_clerk", "hours", 0)]
    assert float(courtesy_start["rate"]) == 17.25, "Expected generic sync engine to select the FSAR payable rate."
    assert str(courtesy_start["selected_schedule_label"] or "") == "FSAR", (
        "Expected generic sync engine to preserve the selected schedule label."
    )
    source_schedule = dict(courtesy_start["source_rate_schedule"] or {})
    assert float(source_schedule.get("Current") or 0.0) == 17.0, "Expected raw Current column to be preserved."
    assert float(source_schedule.get("FSAR") or 0.0) == 17.25, "Expected raw FSAR column to be preserved."
    assert float(source_schedule.get("OE+52") or 0.0) == 17.5, "Expected raw OE+52 column to be preserved."


def test_generic_moa_schedule_sync_builds_contract_row_ops() -> None:
    ops = build_row_ops(PUEBLO_CLERKS_2025_07_05)
    courtesy_op = next(
        op
        for op in ops
        if str(((op.get("target") or {}).get("row_key") or "")) == "courtesy_clerk|hours:0|2024-01-21"
    )
    new_row = courtesy_op.get("new_row") or {}
    assert float(new_row.get("rate") or 0.0) == 17.25, "Expected generated row op to use the selected payable rate."
    assert str(new_row.get("selected_schedule_label") or "") == "FSAR", (
        "Expected generated row op to include selected schedule label."
    )
    assert float(((new_row.get("source_rate_schedule") or {}).get("OE+104") or 0.0)) == 17.75, (
        "Expected generated row op to include the full raw source schedule."
    )


def test_seed_rollforward_preserves_schedule_metadata() -> None:
    patch_payload = {
        "contract_id": "local7_safeway_pueblo_clerks_2022",
        "effective_date": "2026-07-05",
        "source_pdf": "Signed+MOA+-+July+5,+2025+(Safeway).pdf",
        "source_doc_id": "albertsons_safeway_moa_2025_07_05",
        "operations": [],
    }
    wages_payload = {
        "canonical_wage_rows": [
            {
                "row_key": "courtesy_clerk|hours:0|2025-07-05",
                "effective_date": "2025-07-05",
                "row_hash": "abc123",
                "rate": 17.25,
                "selected_schedule_label": "FSAR",
                "source_rate_schedule": {
                    "Current": 17.0,
                    "FSAR": 17.25,
                    "OE+52": 17.5,
                    "OE+104": 17.75,
                },
            }
        ]
    }
    ops, summary = _seed_ops(
        patch_payload=patch_payload,
        wages_payload=wages_payload,
        source_effective_date="2025-07-05",
        review_status="approved",
        confidence=0.7,
    )
    assert summary["generated_ops"] == 1, "Expected exactly one seeded roll-forward op."
    new_row = (ops[0].get("new_row") or {})
    assert str(new_row.get("selected_schedule_label") or "") == "FSAR", (
        "Expected seeded roll-forward op to preserve selected schedule label."
    )
    assert float(((new_row.get("source_rate_schedule") or {}).get("OE+52") or 0.0)) == 17.5, (
        "Expected seeded roll-forward op to preserve raw schedule values."
    )


def main() -> None:
    test_generic_moa_schedule_sync_preserves_raw_schedule_columns()
    test_generic_moa_schedule_sync_builds_contract_row_ops()
    test_seed_rollforward_preserves_schedule_metadata()
    print("[OK] MOA wage schedule sync checks passed")


if __name__ == "__main__":
    main()
