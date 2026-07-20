"""Generate Appendix A wage-row ops for the Pueblo Meat 2025-07-05 MOA patch.

Source: signed MOA wage booklet, "Safeway Denver Metro Meat" schedule
(output.json pages 39-40; the store list explicitly includes Pueblo).

A dedicated script rather than a MoaWageScheduleSyncConfig because the meat
schedule has two quirks the generic config cannot express:
- the managers' fixed table is the SECOND table on page 39 (courtesy clerks,
  a clerks-unit table, comes first), and
- one progression table is shared by three classifications
  ("Meat Wrappers/ Seafood Clerks /Butcher Block / Deli Clerks").
The Pueblo Meat base book also has no Start row for meat_wrappers, so that
one MOA row is tolerated as uncovered (reported, not applied).
"Meat Clean Up (ABS only)" on page 40 is skipped: Albertsons-only, and the
Safeway Pueblo book has no such classification.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.ingest.materializer import ContractMaterializer
from backend.ingest.moa_wage_schedule_sync import (
    DEFAULT_SCHEDULE_EFFECTIVE_DATES,
    DEFAULT_SCHEDULE_LABELS,
    extract_step_and_schedule_from_row,
    normalize_label,
    parse_step,
    schedule_ops_for_row,
    schedule_rates_from_cells,
    selected_schedule_rate,
)

CONTRACT_ID = "local7_safeway_pueblo_meat_2022"
PATCH_PATH = Path(f"data/contracts/{CONTRACT_ID}/amendments/local7_safeway_moa_2025_07_05.json")
EFFECTIVE_CONTRACT_PATH = Path(
    f"data/contracts/{CONTRACT_ID}/effective/effective_local7_safeway_moa_2025_07_05/effective_contract.json"
)
OUTPUT_JSON_PATH = Path("data/source_docs/moa/albertsons_safeway_moa_2025_07_05/output.json")
SOURCE_DOC_ID = "albertsons_safeway_moa_2025_07_05"
SOURCE_PDF = "Signed+MOA+-+July+5,+2025+(Safeway).pdf"
BASE_EFFECTIVE_DATE = "2024-02-25"
WAGE_TABLE_ID = "appendix_a_wage_rows"
PREFERRED_SCHEDULE_LABELS = ("FSAR", "Current")

MANAGER_CLASS_MAP = {
    normalize_label("Head Meat Cutter"): "head_meat_cutter",
    normalize_label("First Cutter"): "first_cutter",
    normalize_label("Seafood Manager /Butcher Block"): "seafood_manager",
    normalize_label("Deli Manager"): "deli_manager",
    normalize_label("Deli Manager 5 < deli ee's"): "deli_manager_after_5_20_77_directing_5_or_less_deli_employees",
    normalize_label("Assist. Deli Manager"): "assistant_deli_manager",
    normalize_label("Starbucks Lead"): "starbucks_lead",
}

# heading text on the page -> classification keys fed by the table that follows
PROGRESSION_HEADINGS = {
    normalize_label("Meat Cutters"): ("meat_cutters",),
    normalize_label("Meat Wrappers/ Seafood Clerks /Butcher Block / Deli Clerks"): (
        "meat_wrappers",
        "seafood_clerks",
        "deli_clerks",
    ),
    normalize_label("Starbucks Clerks"): ("starbucks_clerks",),
}

MEAT_SECTION_MARKER = "safeway denver metro meat"


def build_rate_map() -> dict[tuple[str, str, object], dict[str, object]]:
    pages = {int(p["page_number"]): p for p in json.loads(OUTPUT_JSON_PATH.read_text(encoding="utf-8"))["pages"]}
    rate_map: dict[tuple[str, str, object], dict[str, object]] = {}

    def add(classification_key, step_type, threshold, schedule_rates, page_num):
        label, rate = selected_schedule_rate(schedule_rates, PREFERRED_SCHEDULE_LABELS)
        key = (classification_key, step_type, threshold)
        if key in rate_map and abs(rate_map[key]["rate"] - rate) > 1e-9:
            raise ValueError(f"conflicting rates for {key}")
        rate_map[key] = {
            "rate": rate,
            "selected_schedule_label": label,
            "source_rate_schedule": dict(schedule_rates),
            "page_number": page_num,
        }

    in_meat_section = False
    managers_done = False
    pending_classes: tuple[str, ...] | None = None
    for page_num in (39, 40):
        for item in pages[page_num]["items"]:
            kind = item.get("type")
            if kind in {"heading", "text"}:
                text = " ".join(str(item.get("value") or "").split())
                if MEAT_SECTION_MARKER in text.lower():
                    in_meat_section = True
                    continue
                norm = normalize_label(text)
                if in_meat_section and norm in PROGRESSION_HEADINGS:
                    pending_classes = PROGRESSION_HEADINGS[norm]
                elif in_meat_section and norm and norm != normalize_label("Albertson Denver Meat"):
                    # any other heading ("Meat Clean Up (ABS only)", next locality) ends the pending table mapping
                    pending_classes = None
                continue
            if kind != "table" or not in_meat_section:
                continue
            rows = item.get("rows") or []
            if pending_classes is None and not managers_done:
                # first meat-section table = fixed managers table
                for row in rows:
                    label = str(row[0] or "").strip()
                    classification_key = MANAGER_CLASS_MAP.get(normalize_label(label))
                    if not classification_key:
                        continue
                    schedule_rates = schedule_rates_from_cells(row[1:], list(DEFAULT_SCHEDULE_LABELS))
                    if schedule_rates:
                        add(classification_key, "fixed", None, schedule_rates, page_num)
                managers_done = True
                continue
            if pending_classes:
                for row in rows:
                    step_label, schedule_rates = extract_step_and_schedule_from_row(row, list(DEFAULT_SCHEDULE_LABELS))
                    if not step_label or not schedule_rates:
                        continue
                    step_type, threshold = parse_step(step_label, "hours")
                    for classification_key in pending_classes:
                        add(classification_key, step_type, threshold, schedule_rates, page_num)
                pending_classes = None
    return rate_map


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    rate_map = build_rate_map()

    effective = json.loads(EFFECTIVE_CONTRACT_PATH.read_text(encoding="utf-8"))
    rows = ((effective.get("tables") or {}).get(WAGE_TABLE_ID) or {}).get("rows") or []
    latest_rows = {}
    for row in rows:
        cols = row.get("columns") or {}
        if cols.get("effective_date") != BASE_EFFECTIVE_DATE:
            continue
        latest_rows[(cols.get("classification_key"), cols.get("step_type"), cols.get("threshold_value"))] = row

    missing = sorted(str(k) for k in latest_rows if k not in rate_map)
    extras = sorted(str(k) for k in rate_map if k not in latest_rows)
    known_extras = {str(("meat_wrappers", "hours", 0))}
    if missing:
        raise RuntimeError(f"base rows with no MOA rate: {missing}")
    unexpected = [e for e in extras if e not in known_extras]
    if unexpected:
        raise RuntimeError(f"MOA rates with no base row: {unexpected}")
    for e in extras:
        print(f"[tolerated] MOA rate with no base row (base book quirk): {e}")

    materializer = ContractMaterializer()
    ops = []
    for key in sorted(latest_rows, key=lambda k: (str(k[0]), str(k[1]), -1 if k[2] is None else int(k[2]))):
        row = latest_rows[key]
        cols = row.get("columns") or {}
        info = rate_map[key]
        # One op per dated schedule (FSAR / OE+52 / OE+104): the MOA raises
        # this row three times, and each raise must be its own dated row or
        # "current rate" goes stale when the next one takes effect.
        ops.extend(
            schedule_ops_for_row(
                table_id=WAGE_TABLE_ID,
                row_key=str(row.get("row_key")),
                expected_prev_hash=materializer.hash_row(cols),
                schedule_rates=dict(info["source_rate_schedule"]),
                source_refs=[
                    {
                        "source_type": "moa",
                        "pdf": SOURCE_PDF,
                        "source_doc_id": SOURCE_DOC_ID,
                        "pdf_page": int(info["page_number"]),
                    }
                ],
            )
        )

    patch = json.loads(PATCH_PATH.read_text(encoding="utf-8"))
    existing = patch.get("operations") or []
    kept = [
        op
        for op in existing
        if not (op.get("op") == "replace_table_row" and (op.get("target") or {}).get("table_id") == WAGE_TABLE_ID)
    ]
    patch["operations"] = kept + ops
    patch.setdefault("metadata", {})["appendix_a_sync"] = {
        "config_id": "local7_safeway_pueblo_meat_2025_07_05",
        "source_doc_id": SOURCE_DOC_ID,
        "source": "output.json Safeway Denver Metro Meat pages 39-40 wage schedule (store list includes Pueblo)",
        "schedule_effective_dates": dict(DEFAULT_SCHEDULE_EFFECTIVE_DATES),
        "row_ops_generated": len(ops),
    }
    print(f"wage ops generated: {len(ops)} (existing non-wage ops kept: {len(kept)})")
    if not dry_run:
        PATCH_PATH.write_text(json.dumps(patch, indent=2) + "\n", encoding="utf-8")
        print(f"patch updated: {PATCH_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
