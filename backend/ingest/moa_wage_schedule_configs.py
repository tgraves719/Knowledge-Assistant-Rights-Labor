from __future__ import annotations

from pathlib import Path

from backend.ingest.moa_wage_schedule_sync import (
    HeadingTableConfig,
    MoaWageScheduleSyncConfig,
    SummaryRowPlan,
    SummaryTableConfig,
    normalize_label,
)


_CLERKS_CLASS_MAP = {
    normalize_label("Other Assist. Managers"): "other_assistant_managers",
    normalize_label("Head Clerk"): "head_clerk",
    normalize_label("Produce Dept. Manager"): "produce_department_manager",
    normalize_label("Floral Manager"): "floral_manager",
    normalize_label("Head Baker (SWY only)"): "head_baker",
    normalize_label("Bakery Manager (SWY only)"): "bakery_manager",
    normalize_label("Variety Manager (SWY only)"): "variety_manager",
    normalize_label("Manager Trainee"): "manager_trainee",
    normalize_label("Fuel Lead"): "fuel_lead",
    normalize_label("Fresh Cut Supervisor"): "fresh_cut_supervisor",
    normalize_label("All Purpose Clerk"): "all_purpose_clerk",
    normalize_label("Grandfathered"): "grandfathered",
    normalize_label("Bakery/Fresh Cut (Salad Bar)/Liquor Clerk"): "bakery_fresh_cut_liquor_clerk",
    normalize_label("Cake Decorator (SWY only)"): "cake_decorator",
    normalize_label("5 Star Cake Decorator (SWY only)"): "5star_cake_decorator",
    normalize_label("Non-Food/GM/Floral"): "nonfood_gm_floral",
    normalize_label("COURTESY CLERKS"): "courtesy_clerk",
}


PUEBLO_CLERKS_2025_07_05 = MoaWageScheduleSyncConfig(
    config_id="local7_safeway_pueblo_clerks_2025_07_05",
    contract_id="local7_safeway_pueblo_clerks_2022",
    patch_path=Path("data/contracts/local7_safeway_pueblo_clerks_2022/amendments/local7_safeway_moa_2025_07_05.json"),
    effective_contract_path=Path(
        "data/contracts/local7_safeway_pueblo_clerks_2022/effective/effective_local7_safeway_moa_2025_07_05/effective_contract.json"
    ),
    source_doc_id="albertsons_safeway_moa_2025_07_05",
    output_json_path=Path("data/source_docs/moa/albertsons_safeway_moa_2025_07_05/output.json"),
    source_pdf="Signed+MOA+-+July+5,+2025+(Safeway).pdf",
    base_effective_date="2024-01-21",
    pdf_page_offset=27,
    metadata_source="output.json Denver Metro clerks pages 37-39 wage schedule",
    summary_tables=(
        SummaryTableConfig(
            page_number=37,
            classification_map=_CLERKS_CLASS_MAP,
            row_plans=(
                SummaryRowPlan(
                    row_start=1,
                    row_end=11,
                    mode="mapped_fixed",
                    skip_labels=("Fuel Lead Stand-Alone",),
                ),
                SummaryRowPlan(
                    row_start=13,
                    row_end=21,
                    mode="single_class_progression",
                    classification_key="all_purpose_clerk",
                    default_step_type="hours",
                ),
                SummaryRowPlan(
                    row_start=22,
                    row_end=22,
                    mode="single_class_fixed",
                    classification_key="grandfathered",
                ),
            ),
        ),
    ),
    heading_tables=(
        HeadingTableConfig(
            page_numbers=(38, 39),
            heading_map=_CLERKS_CLASS_MAP,
            heading_modes={
                "5star_cake_decorator": "fixed_first_row",
                "courtesy_clerk": "progression_auto",
            },
            stop_heading="Safeway Denver Metro Meat",
        ),
    ),
)


CONFIGS = {
    PUEBLO_CLERKS_2025_07_05.config_id: PUEBLO_CLERKS_2025_07_05,
}


def get_config(config_id: str) -> MoaWageScheduleSyncConfig:
    try:
        return CONFIGS[config_id]
    except KeyError as exc:
        raise KeyError(f"Unknown MOA wage schedule config_id: {config_id}") from exc
