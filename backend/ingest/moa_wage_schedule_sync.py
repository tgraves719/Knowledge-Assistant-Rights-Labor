from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from backend.ingest.materializer import ContractMaterializer


DEFAULT_SCHEDULE_LABELS = ("Current", "FSAR", "OE+52", "OE+104")

# The 2025 MOA raises the same wage row three times: at ratification (FSAR,
# "First Sunday After Ratification") and on the 52- and 104-week marks. Every
# non-"Current" schedule must materialize as its own dated superseding row —
# selecting just the schedule that was active at materialization time made
# every "current rate" answer silently stale when OE+52 arrived a year later.
# Dates follow the pack's established FSAR anchor (2025-07-05) plus 364/728
# days; confirm exact Sundays with the steward if payroll disputes arise.
DEFAULT_SCHEDULE_EFFECTIVE_DATES = {
    "FSAR": "2025-07-05",
    "OE+52": "2026-07-04",
    "OE+104": "2027-07-03",
}


def normalize_label(value: str) -> str:
    text = (value or "").strip().lower()
    text = text.replace("&", "and")
    text = re.sub(r"\(swy only\)", "", text)
    text = text.replace("(salad bar)", "")
    text = text.replace("dept.", "department")
    text = text.replace("assist.", "assistant")
    text = text.replace("5 star", "5-star")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def money_to_float(raw: str) -> float:
    txt = str(raw or "").strip().replace("$", "").replace(",", "")
    return float(Decimal(txt))


def parse_step(label: str, default_type: str) -> tuple[str, int | None]:
    label = " ".join(str(label or "").split())
    if label.lower() == "start":
        return default_type, 0
    match = re.match(r"After\s+(\d+)\s+(Hours|Months)\b", label, flags=re.I)
    if match:
        return ("hours" if match.group(2).lower().startswith("hour") else "months"), int(match.group(1))
    if default_type == "fixed" and label.lower() == "rate":
        return "fixed", None
    raise ValueError(f"Unrecognized step label: {label}")


def _normalized_schedule_label(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").strip()).upper()


def _is_money_cell(raw: Any) -> bool:
    txt = str(raw or "").strip()
    return bool(re.fullmatch(r"\$?\d+(?:\.\d{1,2})?", txt))


def extract_schedule_labels(row: list[Any], default_labels: tuple[str, ...]) -> list[str]:
    known = {_normalized_schedule_label(v) for v in default_labels}
    labels = [str(cell or "").strip() for cell in row if _normalized_schedule_label(str(cell or "")) in known]
    if len(labels) >= 2:
        return labels[: len(default_labels)]
    return list(default_labels)


def schedule_rates_from_cells(rate_cells: list[Any], schedule_labels: list[str]) -> dict[str, float]:
    numeric_cells = [str(cell or "").strip() for cell in rate_cells if _is_money_cell(cell)]
    out: dict[str, float] = {}
    for idx, label in enumerate(schedule_labels):
        if idx >= len(numeric_cells):
            break
        out[str(label)] = money_to_float(numeric_cells[idx])
    return out


def selected_schedule_rate(schedule_rates: dict[str, float], preferred_labels: tuple[str, ...]) -> tuple[str, float]:
    for preferred in preferred_labels:
        preferred_norm = _normalized_schedule_label(preferred)
        for label, rate in schedule_rates.items():
            if _normalized_schedule_label(label) == preferred_norm:
                return label, rate
    first_label = next(iter(schedule_rates))
    return first_label, schedule_rates[first_label]


def schedule_ops_for_row(
    *,
    table_id: str,
    row_key: str,
    expected_prev_hash: str,
    schedule_rates: dict[str, float],
    source_refs: list[dict[str, Any]],
    schedule_effective_dates: dict[str, str] | None = None,
    confidence: float = 0.95,
) -> list[dict[str, Any]]:
    """One replace_table_row op per dated MOA schedule for a single base row.

    Each op carries an explicit new_row.effective_date, which the
    materializer treats as a dated supersession — the base row stays for
    history and every schedule becomes its own row, so "current rate" stays
    correct as each raise takes effect. All ops share the base row's prev
    hash because supersession never mutates the base row.
    """
    effective_dates = schedule_effective_dates or DEFAULT_SCHEDULE_EFFECTIVE_DATES
    ops: list[dict[str, Any]] = []
    for label, effective_date in effective_dates.items():
        label_norm = _normalized_schedule_label(label)
        rate = next(
            (value for cand, value in schedule_rates.items() if _normalized_schedule_label(cand) == label_norm),
            None,
        )
        if rate is None:
            continue
        ops.append(
            {
                "op": "replace_table_row",
                "target": {"table_id": table_id, "row_key": row_key},
                "expected_prev_hash": expected_prev_hash,
                "new_row": {
                    "rate": float(rate),
                    "effective_date": str(effective_date),
                    "selected_schedule_label": str(label),
                    "source_rate_schedule": dict(schedule_rates),
                },
                "source_refs": [dict(ref) for ref in source_refs],
                "confidence": confidence,
                "review_status": "approved",
            }
        )
    return ops


def extract_step_and_schedule_from_row(
    row: list[Any],
    schedule_labels: list[str],
) -> tuple[str | None, dict[str, float]]:
    cells = [str(c or "").strip() for c in row]
    if len(cells) >= 2 and (cells[1].lower() == "start" or cells[1].lower().startswith("after ")):
        return cells[1], schedule_rates_from_cells(cells[2:], schedule_labels)
    if len(cells) >= 1 and (cells[0].lower() == "start" or cells[0].lower().startswith("after ")):
        return cells[0], schedule_rates_from_cells(cells[1:], schedule_labels)
    return None, {}


@dataclass(frozen=True)
class SummaryRowPlan:
    row_start: int
    row_end: int
    mode: str
    classification_key: str | None = None
    default_step_type: str = "hours"
    skip_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class SummaryTableConfig:
    page_number: int
    classification_map: dict[str, str]
    row_plans: tuple[SummaryRowPlan, ...]


@dataclass(frozen=True)
class HeadingTableConfig:
    page_numbers: tuple[int, ...]
    heading_map: dict[str, str]
    heading_modes: dict[str, str]
    stop_heading: str | None = None
    default_schedule_labels: tuple[str, ...] = DEFAULT_SCHEDULE_LABELS


@dataclass(frozen=True)
class MoaWageScheduleSyncConfig:
    config_id: str
    contract_id: str
    patch_path: Path
    effective_contract_path: Path
    source_doc_id: str
    output_json_path: Path
    source_pdf: str
    base_effective_date: str
    pdf_page_offset: int
    metadata_source: str
    summary_tables: tuple[SummaryTableConfig, ...]
    heading_tables: tuple[HeadingTableConfig, ...]
    preferred_active_schedule_labels: tuple[str, ...] = ("FSAR", "Current")
    wage_table_id: str = "appendix_a_wage_rows"


def load_output_pages(output_json_path: Path) -> dict[int, dict[str, Any]]:
    data = json.loads(output_json_path.read_text(encoding="utf-8"))
    return {int(p["page_number"]): p for p in data["pages"]}


def _add_rate(
    rate_map: dict[tuple[str, str, Any], dict[str, Any]],
    *,
    page_num: int,
    classification_key: str,
    step_type: str,
    threshold_value: Any,
    schedule_rates: dict[str, float],
    preferred_active_schedule_labels: tuple[str, ...],
) -> None:
    key = (classification_key, step_type, threshold_value)
    selected_label, selected_rate = selected_schedule_rate(schedule_rates, preferred_active_schedule_labels)
    existing = rate_map.get(key)
    if existing and abs(existing["rate"] - selected_rate) > 1e-9:
        raise ValueError(f"Conflicting MOA rates for {key}: {existing['rate']} vs {selected_rate}")
    rate_map[key] = {
        "rate": selected_rate,
        "page_number": page_num,
        "selected_schedule_label": selected_label,
        "source_rate_schedule": dict(schedule_rates),
    }


def extract_schedule_rates(config: MoaWageScheduleSyncConfig) -> dict[tuple[str, str, Any], dict[str, Any]]:
    pages = load_output_pages(config.output_json_path)
    rate_map: dict[tuple[str, str, Any], dict[str, Any]] = {}

    for summary in config.summary_tables:
        table_item = next(item for item in pages[summary.page_number]["items"] if item.get("type") == "table")
        rows = table_item.get("rows") or []
        schedule_labels = extract_schedule_labels(rows[0] if rows else [], DEFAULT_SCHEDULE_LABELS)
        for index, row in enumerate(rows[1:], start=1):
            label = str(row[0] or "").strip()
            for plan in summary.row_plans:
                if not (plan.row_start <= index <= plan.row_end):
                    continue
                if label in plan.skip_labels:
                    break
                if plan.mode == "mapped_fixed":
                    classification_key = summary.classification_map.get(normalize_label(label))
                    schedule_rates = schedule_rates_from_cells(row[1:], schedule_labels)
                    if classification_key and schedule_rates:
                        _add_rate(
                            rate_map,
                            page_num=summary.page_number,
                            classification_key=classification_key,
                            step_type="fixed",
                            threshold_value=None,
                            schedule_rates=schedule_rates,
                            preferred_active_schedule_labels=config.preferred_active_schedule_labels,
                        )
                    break
                if plan.mode == "single_class_progression":
                    if not plan.classification_key:
                        raise ValueError("single_class_progression requires classification_key")
                    schedule_rates = schedule_rates_from_cells(row[1:], schedule_labels)
                    if schedule_rates:
                        step_type, threshold = parse_step(label, plan.default_step_type)
                        _add_rate(
                            rate_map,
                            page_num=summary.page_number,
                            classification_key=plan.classification_key,
                            step_type=step_type,
                            threshold_value=threshold,
                            schedule_rates=schedule_rates,
                            preferred_active_schedule_labels=config.preferred_active_schedule_labels,
                        )
                    break
                if plan.mode == "single_class_fixed":
                    if not plan.classification_key:
                        raise ValueError("single_class_fixed requires classification_key")
                    schedule_rates = schedule_rates_from_cells(row[1:], schedule_labels)
                    if schedule_rates:
                        _add_rate(
                            rate_map,
                            page_num=summary.page_number,
                            classification_key=plan.classification_key,
                            step_type="fixed",
                            threshold_value=None,
                            schedule_rates=schedule_rates,
                            preferred_active_schedule_labels=config.preferred_active_schedule_labels,
                        )
                    break
                raise ValueError(f"Unsupported summary row plan mode: {plan.mode}")

    for heading_table in config.heading_tables:
        current_heading: str | None = None
        schedule_labels = list(heading_table.default_schedule_labels)
        stop_heading_norm = normalize_label(heading_table.stop_heading or "")
        for page_num in heading_table.page_numbers:
            stop_page = False
            for item in pages[page_num]["items"]:
                item_type = item.get("type")
                if item_type in {"heading", "text"}:
                    text = " ".join(str(item.get("value") or item.get("md") or "").split())
                    if stop_heading_norm and normalize_label(text).startswith(stop_heading_norm):
                        stop_page = True
                        break
                    if normalize_label(text) in heading_table.heading_map:
                        current_heading = text
                    continue
                if item_type != "table" or not current_heading:
                    continue
                rows = item.get("rows") or []
                classification_key = heading_table.heading_map.get(normalize_label(current_heading))
                if not classification_key or not rows:
                    continue
                detected = extract_schedule_labels(rows[0], heading_table.default_schedule_labels)
                if detected:
                    schedule_labels = detected
                mode = heading_table.heading_modes.get(classification_key, "progression_hours")
                if mode == "fixed_first_row":
                    schedule_rates = schedule_rates_from_cells(rows[0], schedule_labels)
                    if schedule_rates:
                        _add_rate(
                            rate_map,
                            page_num=page_num,
                            classification_key=classification_key,
                            step_type="fixed",
                            threshold_value=None,
                            schedule_rates=schedule_rates,
                            preferred_active_schedule_labels=config.preferred_active_schedule_labels,
                        )
                    continue
                for row in rows:
                    step_label, schedule_rates = extract_step_and_schedule_from_row(row, schedule_labels)
                    if not step_label or not schedule_rates:
                        continue
                    default_type = "hours"
                    if mode == "progression_auto":
                        default_type = "months" if "Months" in step_label else "hours"
                    step_type, threshold = parse_step(step_label, default_type)
                    _add_rate(
                        rate_map,
                        page_num=page_num,
                        classification_key=classification_key,
                        step_type=step_type,
                        threshold_value=threshold,
                        schedule_rates=schedule_rates,
                        preferred_active_schedule_labels=config.preferred_active_schedule_labels,
                    )
            if stop_page:
                break

    return rate_map


def load_latest_base_rows(config: MoaWageScheduleSyncConfig) -> dict[tuple[str, str, Any], dict[str, Any]]:
    effective = json.loads(config.effective_contract_path.read_text(encoding="utf-8"))
    rows = ((effective.get("tables") or {}).get(config.wage_table_id) or {}).get("rows") or []
    out: dict[tuple[str, str, Any], dict[str, Any]] = {}
    for row in rows:
        cols = row.get("columns") or {}
        if cols.get("effective_date") != config.base_effective_date:
            continue
        key = (cols.get("classification_key"), cols.get("step_type"), cols.get("threshold_value"))
        out[key] = row
    return out


def build_row_ops(config: MoaWageScheduleSyncConfig) -> list[dict[str, Any]]:
    rate_map = extract_schedule_rates(config)
    latest_rows = load_latest_base_rows(config)
    materializer = ContractMaterializer()

    missing = sorted(k for k in latest_rows if k not in rate_map)
    extras = sorted(k for k in rate_map if k not in latest_rows)
    if missing or extras:
        raise RuntimeError(
            f"Coverage mismatch building MOA Appendix A ops for {config.contract_id}. "
            f"missing={len(missing)} extras={len(extras)}"
        )

    ops: list[dict[str, Any]] = []
    for key, row in sorted(
        latest_rows.items(),
        key=lambda kv: (
            str((kv[1].get("columns") or {}).get("classification_key") or ""),
            str((kv[1].get("columns") or {}).get("step_type") or ""),
            -1
            if (kv[1].get("columns") or {}).get("threshold_value") is None
            else int((kv[1].get("columns") or {}).get("threshold_value")),
        ),
    ):
        cols = row.get("columns") or {}
        page_num = int(rate_map[key]["page_number"])
        ops.extend(
            schedule_ops_for_row(
                table_id=config.wage_table_id,
                row_key=str(row.get("row_key")),
                expected_prev_hash=materializer.hash_row(cols),
                schedule_rates=dict(rate_map[key]["source_rate_schedule"]),
                source_refs=[
                    {
                        "source_type": "moa",
                        "pdf": config.source_pdf,
                        "source_doc_id": config.source_doc_id,
                        "pdf_page": page_num - config.pdf_page_offset,
                    }
                ],
            )
        )
    return ops


def update_patch_file(config: MoaWageScheduleSyncConfig, dry_run: bool = False) -> dict[str, Any]:
    patch = json.loads(config.patch_path.read_text(encoding="utf-8"))
    ops = patch.get("operations") or []
    new_wage_ops = build_row_ops(config)

    rebuilt_ops: list[dict[str, Any]] = []
    inserted = False
    for op in ops:
        is_target = op.get("op") == "replace_table_row" and (op.get("target") or {}).get("table_id") == config.wage_table_id
        if is_target and not inserted:
            rebuilt_ops.extend(new_wage_ops)
            inserted = True
            continue
        if not is_target:
            rebuilt_ops.append(op)
    if not inserted:
        rebuilt_ops.extend(new_wage_ops)

    patch["operations"] = rebuilt_ops
    patch.setdefault("metadata", {})
    patch["metadata"]["appendix_a_sync"] = {
        "config_id": config.config_id,
        "source_doc_id": config.source_doc_id,
        "source": config.metadata_source,
        "schedule_effective_dates": dict(DEFAULT_SCHEDULE_EFFECTIVE_DATES),
        "row_ops_generated": len(new_wage_ops),
    }

    summary = {
        "config_id": config.config_id,
        "contract_id": config.contract_id,
        "patch_path": str(config.patch_path),
        "previous_operations": len(ops),
        "new_wage_operations": len(new_wage_ops),
        "total_operations": len(rebuilt_ops),
        "dry_run": dry_run,
    }
    if not dry_run:
        config.patch_path.write_text(json.dumps(patch, indent=2) + "\n", encoding="utf-8")
    return summary
