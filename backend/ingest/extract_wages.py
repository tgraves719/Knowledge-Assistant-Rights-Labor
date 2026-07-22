"""Deterministic wage-table extraction for contract ingestion/runtime lookups."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import CONTRACT_ID, CONTRACT_MD_FILE, WAGES_DIR


EFFECTIVE_DATES = ["2022-01-23", "2023-01-22", "2024-01-21"]
_WAGE_DEFAULT_EFFECTIVE_DATES = EFFECTIVE_DATES.copy()

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _default_effective_date(effective_dates: list[str], *, as_of: Optional[str] = None) -> Optional[str]:
    """The wage date an undated query resolves to: the currently-effective one.

    When a member asks "what should I be making?" with no date, KARL quotes the
    rate in effect *today*, not the furthest-future scheduled raise. Once a
    contract carries MOA-dated future rows (e.g. raises through 2027), the old
    "latest date wins" default would quote a rate that hasn't started yet. So we
    take the latest effective date that is on or before `as_of` (today by
    default); if every date is still in the future, fall back to the earliest.
    """
    normalized = sorted({str(d).strip() for d in (effective_dates or []) if _ISO_DATE.match(str(d or "").strip())})
    if not normalized:
        return effective_dates[-1] if effective_dates else None
    today = str(as_of or "").strip() or datetime.now().date().isoformat()
    past_or_present = [d for d in normalized if d <= today]
    return past_or_present[-1] if past_or_present else normalized[0]

_CLASSIFICATION_HINTS = (
    "classification", "clerk", "manager", "cutter", "wrapper", "deli", "meat",
    "baker", "lead", "shopper", "assistant", "head", "starbucks", "courtesy",
    "cashier", "chef", "trainee", "apprentice", "seafood",
)
_STEP_RE = re.compile(
    r"^(start|after\s+\d+\s+(?:hours?|months?)|first\s+\d+\s+hours?\s+worked|"
    r"next\s+\d+\s+hours?\s+worked|thereafter)\b",
    re.IGNORECASE,
)

# Deterministic role aliases used only for wage-table key resolution.
# Order matters: first match present in the contract's wage table wins.
_CLASSIFICATION_ALIASES: dict[str, tuple[str, ...]] = {
    # Keep only lexical variants here. Contract-specific semantic remapping
    # must be provided via per-contract ontology/manual overrides.
    "pharmacy_tech": ("pharmacy_technician",),
    "meatcutter": ("meat_cutter", "meat_cutters"),
}
_DATE_TOKEN_RE = re.compile(
    r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4})",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
CANONICAL_WAGE_SCHEMA_VERSION = "wage_canonical_rows_v1"


def _step_type_and_threshold(
    step_name: str,
    hours_required: Optional[int],
    months_required: Optional[int],
) -> tuple[str, Optional[int]]:
    if hours_required is not None:
        return "hours", int(hours_required)
    if months_required is not None:
        return "months", int(months_required)
    if _strip_html(step_name).lower() == "start":
        return "hours", 0
    return "fixed", None


def _canonical_row_key(row: dict) -> tuple:
    return (
        str(row.get("classification_key", "")),
        str(row.get("step_name", "")),
        str(row.get("step_type", "")),
        row.get("threshold_value"),
        str(row.get("effective_date", "")),
    )


def _register_canonical_row(
    canonical_rows: list[dict],
    canonical_index: dict[tuple, int],
    conflicts: list[dict],
    ambiguities: list[dict],
    row: dict,
) -> None:
    key = _canonical_row_key(row)
    idx = canonical_index.get(key)
    if idx is None:
        canonical_index[key] = len(canonical_rows)
        canonical_rows.append(row)
        return

    existing = canonical_rows[idx]
    existing_rate = float(existing.get("rate"))
    new_rate = float(row.get("rate"))
    if abs(existing_rate - new_rate) < 1e-9:
        return

    existing_conf = float(existing.get("confidence", 0.0))
    new_conf = float(row.get("confidence", 0.0))
    if new_conf > existing_conf + 0.05:
        canonical_rows[idx] = row
        conflicts.append(
            {
                "key": {
                    "classification_key": key[0],
                    "step_name": key[1],
                    "step_type": key[2],
                    "threshold_value": key[3],
                    "effective_date": key[4],
                },
                "existing_rate": existing_rate,
                "new_rate": new_rate,
                "chosen": "new",
                "existing_confidence": existing_conf,
                "new_confidence": new_conf,
                "existing_source": existing.get("source_reference", {}),
                "new_source": row.get("source_reference", {}),
            }
        )
        return

    if existing_conf > new_conf + 0.05:
        conflicts.append(
            {
                "key": {
                    "classification_key": key[0],
                    "step_name": key[1],
                    "step_type": key[2],
                    "threshold_value": key[3],
                    "effective_date": key[4],
                },
                "existing_rate": existing_rate,
                "new_rate": new_rate,
                "chosen": "existing",
                "existing_confidence": existing_conf,
                "new_confidence": new_conf,
                "existing_source": existing.get("source_reference", {}),
                "new_source": row.get("source_reference", {}),
            }
        )
        return

    ambiguities.append(
        {
            "key": {
                "classification_key": key[0],
                "step_name": key[1],
                "step_type": key[2],
                "threshold_value": key[3],
                "effective_date": key[4],
            },
            "existing_rate": existing_rate,
            "new_rate": new_rate,
            "existing_confidence": existing_conf,
            "new_confidence": new_conf,
            "resolution": "keep_existing_first_seen",
            "existing_source": existing.get("source_reference", {}),
            "new_source": row.get("source_reference", {}),
        }
    )


def _append_step_with_conflict_resolution(
    class_entry: dict,
    step_payload: dict,
    conflicts: list[dict],
    source_reference: dict,
) -> None:
    steps = class_entry.setdefault("steps", [])
    incoming_key = (
        _strip_html(step_payload.get("step_name", "")).lower(),
        step_payload.get("hours_required"),
        step_payload.get("months_required"),
    )
    for i, existing in enumerate(steps):
        existing_key = (
            _strip_html(existing.get("step_name", "")).lower(),
            existing.get("hours_required"),
            existing.get("months_required"),
        )
        if existing_key != incoming_key:
            continue

        existing_rates = existing.get("rates", {}) or {}
        incoming_rates = step_payload.get("rates", {}) or {}
        if existing_rates == incoming_rates:
            return

        existing_populated = len(existing_rates)
        incoming_populated = len(incoming_rates)
        choose_incoming = incoming_populated > existing_populated
        chosen = "incoming" if choose_incoming else "existing"
        if choose_incoming:
            steps[i] = step_payload
        conflicts.append(
            {
                "classification_key": class_entry.get("normalized_name"),
                "step_name": step_payload.get("step_name"),
                "source_reference": source_reference,
                "existing_rates": existing_rates,
                "incoming_rates": incoming_rates,
                "resolution": f"keep_{chosen}",
            }
        )
        return

    steps.append(step_payload)


def _classify_structured_row(
    label: str,
    row: list[str],
    expected_cols: int,
    current_class_key: Optional[str],
) -> dict:
    lower = label.lower()
    if _is_separator_row(label, row):
        return {"row_type": "separator", "confidence": 1.0, "reason": "separator_row"}
    if "classification" in lower and "effective" in " ".join(_strip_html(c) for c in row).lower():
        return {"row_type": "header", "confidence": 1.0, "reason": "classification_effective_header"}

    row_dates = _extract_dates_from_cells(row)
    if _is_effective_date_row(label, row, row_dates):
        return {"row_type": "effective_date", "confidence": 0.98, "reason": "effective_date_row", "dates": row_dates}

    rates = _extract_row_rates(row, expected_cols=max(expected_cols, 1))
    numeric_count = sum(v is not None for v in rates)

    if _looks_like_step_label(label):
        if numeric_count == 0:
            return {"row_type": "non_wage", "confidence": 0.55, "reason": "step_label_without_rates"}
        if current_class_key:
            return {"row_type": "step_row", "confidence": 0.95, "reason": "step_label_with_rates", "rates": rates}
        return {"row_type": "ambiguous_step", "confidence": 0.52, "reason": "step_without_active_class", "rates": rates}

    if _looks_like_classification_label(label) and numeric_count < max(2, expected_cols - 1):
        return {"row_type": "classification_header", "confidence": 0.92, "reason": "classification_header"}

    if numeric_count >= max(2, expected_cols - 1):
        return {"row_type": "rate_row", "confidence": 0.90, "reason": "direct_rate_row", "rates": rates}

    if numeric_count > 0:
        return {"row_type": "ambiguous_rate_like", "confidence": 0.5, "reason": "partial_rate_row", "rates": rates}

    return {"row_type": "non_wage", "confidence": 0.35, "reason": "no_rates"}


def _build_canonical_rows_from_classes(
    classes: dict,
    contract_id: str,
    source_method: str,
    confidence_default: float = 0.9,
) -> list[dict]:
    rows: list[dict] = []
    for class_key, class_data in (classes or {}).items():
        class_name = class_data.get("name", class_key)
        for step in class_data.get("steps", []) or []:
            step_name = step.get("step_name", "Rate")
            hours_required = step.get("hours_required")
            months_required = step.get("months_required")
            step_type, threshold_value = _step_type_and_threshold(step_name, hours_required, months_required)
            for effective_date, rate in (step.get("rates") or {}).items():
                if not _ISO_DATE_RE.match(str(effective_date or "")):
                    continue
                if rate is None:
                    continue
                rows.append(
                    {
                        "schema_version": CANONICAL_WAGE_SCHEMA_VERSION,
                        "contract_id": contract_id,
                        "classification_key": class_key,
                        "classification_name": class_name,
                        "step_name": step_name,
                        "step_type": step_type,
                        "threshold_value": threshold_value,
                        "effective_date": effective_date,
                        "rate": float(rate),
                        "source_method": source_method,
                        "row_type": "step_row" if step_name != "Rate" else "rate_row",
                        "confidence": confidence_default,
                        "source_reference": {"type": "derived_from_classifications"},
                    }
                )
    return rows


def _strip_html(text: str) -> str:
    """Convert simple HTML fragments to plain text for deterministic parsing."""
    if text is None:
        return ""
    value = str(text)
    value = value.replace("<br/>", " ").replace("<br>", " ").replace("<br />", " ")
    value = re.sub(r"<[^>]+>", " ", value)
    value = value.replace("&nbsp;", " ").replace("&amp;", "&")
    return re.sub(r"\s+", " ", value).strip()


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def normalize_classification_name(name: str) -> str:
    """Convert classification names to stable lookup keys."""
    clean = _strip_html(name).upper()
    clean = re.sub(r"[/\s]+", "_", clean)
    clean = re.sub(r"[^A-Z0-9_]", "", clean)
    clean = re.sub(r"_+", "_", clean)
    return clean.strip("_").lower()


def parse_rate(rate_text: str) -> Optional[float]:
    """Extract an hourly-rate value from a cell."""
    text = _strip_html(rate_text)
    if not text or text in {"-", "--"}:
        return None
    if "%" in text:
        return None

    # Prefer explicit currency patterns.
    m = re.search(r"\$+\s*([\d,]+(?:\.\d{1,2})?)", text)
    if m:
        return float(m.group(1).replace(",", ""))

    # Accept plain numeric cells if they look like wage rates.
    m = re.match(r"^\s*([\d,]+(?:\.\d{1,2})?)\s*$", text)
    if not m:
        return None
    value = float(m.group(1).replace(",", ""))
    if 0 < value < 200:
        return value
    return None


def _normalize_effective_date(date_text: str) -> Optional[str]:
    """Normalize date text to YYYY-MM-DD when possible."""
    text = _strip_html(date_text)
    if not text:
        return None

    # Numeric dates: m/d/yyyy, m-d-yyyy, m/d/yy, m-d-yy
    m = re.match(r"^\s*(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\s*$", text)
    if m:
        month = int(m.group(1))
        day = int(m.group(2))
        year = int(m.group(3))
        if year < 100:
            year += 2000 if year <= 69 else 1900
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{year:04d}-{month:02d}-{day:02d}"

    # Month-name formats.
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except Exception:
            continue
    return None


def _extract_dates_from_cells(cells: list[str]) -> list[str]:
    """Extract ordered, unique normalized dates from table cells."""
    found: list[str] = []
    for raw in cells:
        text = _strip_html(raw)
        if not text:
            continue
        for token in _DATE_TOKEN_RE.findall(text):
            iso = _normalize_effective_date(token)
            if iso and iso not in found:
                found.append(iso)
    return found


def _is_effective_date_row(label: str, row: list[str], dates: list[str]) -> bool:
    """
    Identify rows that are actually effective-date rows.

    Guards against classification labels that contain incidental dates
    (e.g., "AFTER 5/20/77"), which should not reset table-wide dates.
    """
    if len(dates) < 2:
        return False

    clean_label = _strip_html(label).lower()
    row_text = " ".join(_strip_html(c) for c in row).lower()

    if clean_label in {"", "-", "--", "_"}:
        return True
    if "effective" in row_text:
        return True

    non_first_cells = row[1:] if len(row) > 1 else []
    date_cell_count = 0
    for cell in non_first_cells:
        if _extract_dates_from_cells([cell]):
            date_cell_count += 1
    if date_cell_count >= 2 and not _looks_like_classification_label(label):
        return True

    return False


def _looks_like_step_label(label: str) -> bool:
    return bool(_STEP_RE.match(_strip_html(label)))


def _looks_like_classification_label(label: str) -> bool:
    text = _strip_html(label)
    if not text:
        return False
    lower = text.lower()
    if _looks_like_step_label(text):
        return False
    if any(tok in lower for tok in _CLASSIFICATION_HINTS):
        return True

    # Fallback heuristic for compact all-caps labels (for example,
    # "NON-FOOD/GM/FLORAL") that often appear in wage progression tables.
    if re.search(r"\d", text):
        return False
    if any(stop in lower for stop in ("effective", "current", "year", "hours payable")):
        return False

    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    uppercase_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
    if uppercase_ratio >= 0.7 and len(text.split()) <= 8:
        return True
    if ("/" in text or "&" in text or "-" in text) and len(text.split()) <= 8:
        return True
    return False


def _is_separator_row(label: str, row: list[str]) -> bool:
    clean_label = _strip_html(label)
    if clean_label in {"", "-", "--", "_"}:
        return True
    cleaned = [_strip_html(c) for c in row]
    if cleaned and all(c in {"", "-", "--", "_"} for c in cleaned):
        return True
    return False


def _extract_row_rates(row: list[str], expected_cols: int) -> list[Optional[float]]:
    cells = [_strip_html(c) for c in row[1:]]
    values = [parse_rate(c) for c in cells[:expected_cols]]
    if len(values) < expected_cols:
        values.extend([None] * (expected_cols - len(values)))
    return values


def _build_rate_map(
    effective_dates: list[str],
    rates: list[Optional[float]],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for idx, d in enumerate(effective_dates):
        if idx >= len(rates):
            break
        value = rates[idx]
        if value is not None:
            out[d] = value
    return out


def _derive_step_requirements(step_name: str, state: dict) -> tuple[Optional[int], Optional[int]]:
    """
    Derive threshold requirements from step labels.

    Supports:
    - Start
    - After N hours / months
    - First N hours worked
    - Next N hours worked
    - Thereafter
    """
    label = _strip_html(step_name)
    lower = label.lower()

    if lower == "start":
        state["accumulated_hours"] = 0
        return 0, 0

    m = re.search(r"after\s+(\d+)\s+hours?", lower)
    if m:
        hours = int(m.group(1))
        state["accumulated_hours"] = max(state.get("accumulated_hours", 0), hours)
        return hours, None

    m = re.search(r"after\s+(\d+)\s+months?", lower)
    if m:
        return None, int(m.group(1))

    m = re.search(r"first\s+(\d+)\s+hours?\s+worked", lower)
    if m:
        span = int(m.group(1))
        state["accumulated_hours"] = span
        return 0, None

    m = re.search(r"next\s+(\d+)\s+hours?\s+worked", lower)
    if m:
        span = int(m.group(1))
        threshold = int(state.get("accumulated_hours", 0))
        state["accumulated_hours"] = threshold + span
        return threshold, None

    if lower.startswith("thereafter"):
        threshold = state.get("accumulated_hours")
        if threshold is None:
            return None, None
        return int(threshold), None

    return None, None


def _table_to_dict(table: Any) -> dict:
    if isinstance(table, dict):
        return table
    if hasattr(table, "to_dict"):
        return table.to_dict()
    return {
        "table_id": getattr(table, "table_id", ""),
        "headers": getattr(table, "headers", []),
        "rows": getattr(table, "rows", []),
        "heading_path": getattr(table, "heading_path", []),
        "parent_article": getattr(table, "parent_article", None),
    }


def _is_wage_candidate_table(table: dict) -> bool:
    rows = table.get("rows") or []
    if not rows:
        return False

    heading = " ".join(_strip_html(h) for h in (table.get("heading_path") or []))
    heading_lower = heading.lower()
    headers = [_strip_html(h) for h in (table.get("headers") or [])]
    header_text = " ".join(headers).lower()

    first_col = [_strip_html(r[0]) for r in rows if r]
    has_step_rows = any(_looks_like_step_label(v) for v in first_col)
    has_classification_signal = (
        "classification" in header_text
        or any(_looks_like_classification_label(v) for v in first_col)
    )
    has_effective_signal = (
        "effective" in header_text
        or bool(_extract_dates_from_cells(headers))
        or bool(_extract_dates_from_cells(rows[0] if rows else []))
    )
    has_rates_signal = any("rate" in h.lower() for h in (table.get("heading_path") or []))
    has_appendix_signal = "appendix" in heading_lower

    money_cells = 0
    for row in rows:
        for cell in row[1:]:
            if parse_rate(cell) is not None:
                money_cells += 1

    # Avoid benefits/copay/pension tables that contain dollars but are not wage grids.
    non_wage_heading = any(
        token in heading_lower
        for token in (
            "health", "benefit", "deductible", "co-pay", "copay", "dental",
            "vision", "prescription", "pension", "401k", "plan",
        )
    )
    if non_wage_heading and not (has_step_rows or has_appendix_signal or has_rates_signal):
        return False

    if money_cells < 6:
        return False
    if has_step_rows:
        return True
    if has_classification_signal and has_effective_signal:
        return True
    if has_classification_signal and (has_appendix_signal or has_rates_signal):
        return True
    return False


def _extract_wages_from_structured_tables(
    tables: list[Any],
    contract_id: str,
) -> dict:
    classes: dict[str, dict] = {}
    warnings: list[str] = []
    row_type_counts: dict[str, int] = {}
    unresolved_rows: list[dict] = []
    canonical_rows: list[dict] = []
    canonical_index: dict[tuple, int] = {}
    canonical_conflicts: list[dict] = []
    canonical_ambiguities: list[dict] = []
    step_conflicts: list[dict] = []

    table_dicts = [_table_to_dict(t) for t in tables]
    candidates = [t for t in table_dicts if _is_wage_candidate_table(t)]
    candidates.sort(key=lambda t: ((t.get("parent_article") or 999), t.get("table_id", "")))

    current_effective_dates: list[str] = []
    current_class_key: Optional[str] = None
    step_state: dict = {"accumulated_hours": 0}
    source_table_ids: list[str] = []

    for table in candidates:
        table_id = table.get("table_id", "")
        source_table_ids.append(table_id)
        rows = table.get("rows") or []
        headers = table.get("headers") or []

        found_dates = _extract_dates_from_cells(headers)
        if not found_dates and rows:
            found_dates = _extract_dates_from_cells(rows[0])
        if found_dates:
            current_effective_dates = found_dates
        if not current_effective_dates:
            current_effective_dates = _WAGE_DEFAULT_EFFECTIVE_DATES.copy()

        expected_cols = len(current_effective_dates)
        if expected_cols == 0:
            expected_cols = 3
            current_effective_dates = _WAGE_DEFAULT_EFFECTIVE_DATES.copy()

        for row_idx, row in enumerate(rows):
            if not row:
                continue
            label = _strip_html(row[0])
            lower = label.lower()
            classification = _classify_structured_row(
                label=label,
                row=row,
                expected_cols=expected_cols,
                current_class_key=current_class_key,
            )
            row_type = classification.get("row_type", "non_wage")
            row_type_counts[row_type] = row_type_counts.get(row_type, 0) + 1

            if row_type in {"separator", "header", "non_wage"}:
                continue

            if row_type == "effective_date":
                row_dates = classification.get("dates") or _extract_dates_from_cells(row)
                if row_dates:
                    current_effective_dates = row_dates
                    expected_cols = len(current_effective_dates)
                continue

            rates = classification.get("rates")
            if rates is None:
                rates = _extract_row_rates(row, expected_cols=expected_cols)
            rate_map = _build_rate_map(current_effective_dates, rates)
            if not rate_map and row_type in {"step_row", "rate_row"}:
                unresolved_rows.append(
                    {
                        "table_id": table_id,
                        "row_index": row_idx,
                        "label": label,
                        "reason": "row_type_without_valid_rates",
                        "row_type": row_type,
                    }
                )
                continue

            if row_type in {"ambiguous_step", "ambiguous_rate_like"}:
                unresolved_rows.append(
                    {
                        "table_id": table_id,
                        "row_index": row_idx,
                        "label": label,
                        "reason": classification.get("reason", row_type),
                        "row_type": row_type,
                    }
                )
                warnings.append(f"{table_id}: ambiguous row '{label}' ({classification.get('reason', row_type)})")
                continue

            # Step rows belong to the currently active classification.
            if row_type == "step_row":
                if current_class_key is None:
                    warnings.append(f"{table_id}: step '{label}' without active classification header")
                    unresolved_rows.append(
                        {
                            "table_id": table_id,
                            "row_index": row_idx,
                            "label": label,
                            "reason": "step_without_active_classification",
                            "row_type": row_type,
                        }
                    )
                    continue

                hours_req, months_req = _derive_step_requirements(label, step_state)
                step_payload = {
                    "step_name": label,
                    "hours_required": hours_req,
                    "months_required": months_req,
                    "rates": rate_map,
                }
                _append_step_with_conflict_resolution(
                    classes[current_class_key],
                    step_payload,
                    step_conflicts,
                    {"table_id": table_id, "row_index": row_idx, "row_type": row_type},
                )

                step_type, threshold_value = _step_type_and_threshold(label, hours_req, months_req)
                for effective_date, rate in rate_map.items():
                    _register_canonical_row(
                        canonical_rows,
                        canonical_index,
                        canonical_conflicts,
                        canonical_ambiguities,
                        {
                            "schema_version": CANONICAL_WAGE_SCHEMA_VERSION,
                            "contract_id": contract_id,
                            "classification_key": current_class_key,
                            "classification_name": classes[current_class_key].get("name", current_class_key),
                            "step_name": label,
                            "step_type": step_type,
                            "threshold_value": threshold_value,
                            "effective_date": effective_date,
                            "rate": float(rate),
                            "source_method": "table_registry",
                            "row_type": row_type,
                            "confidence": float(classification.get("confidence", 0.9)),
                            "source_reference": {"table_id": table_id, "row_index": row_idx},
                        },
                    )
                continue

            # Classification header row (for progression blocks).
            if row_type == "classification_header":
                key = normalize_classification_name(label)
                classes.setdefault(
                    key,
                    {
                        "name": label,
                        "normalized_name": key,
                        "is_manager": ("manager" in lower) or ("head" in lower),
                        "steps": [],
                    },
                )
                current_class_key = key
                step_state = {"accumulated_hours": 0}
                continue

            # Single-rate row (classification with direct rates).
            if row_type == "rate_row":
                if not label:
                    continue
                key = normalize_classification_name(label)
                class_entry = classes.setdefault(
                    key,
                    {
                        "name": label,
                        "normalized_name": key,
                        "is_manager": ("manager" in lower) or ("head" in lower),
                        "steps": [],
                    },
                )
                step_payload = {
                    "step_name": "Rate",
                    "hours_required": None,
                    "months_required": None,
                    "rates": rate_map,
                }
                _append_step_with_conflict_resolution(
                    class_entry,
                    step_payload,
                    step_conflicts,
                    {"table_id": table_id, "row_index": row_idx, "row_type": row_type},
                )
                for effective_date, rate in rate_map.items():
                    _register_canonical_row(
                        canonical_rows,
                        canonical_index,
                        canonical_conflicts,
                        canonical_ambiguities,
                        {
                            "schema_version": CANONICAL_WAGE_SCHEMA_VERSION,
                            "contract_id": contract_id,
                            "classification_key": key,
                            "classification_name": class_entry.get("name", label),
                            "step_name": "Rate",
                            "step_type": "fixed",
                            "threshold_value": None,
                            "effective_date": effective_date,
                            "rate": float(rate),
                            "source_method": "table_registry",
                            "row_type": row_type,
                            "confidence": float(classification.get("confidence", 0.9)),
                            "source_reference": {"table_id": table_id, "row_index": row_idx},
                        },
                    )
                current_class_key = None
                step_state = {"accumulated_hours": 0}
                continue

    effective_dates = current_effective_dates or _WAGE_DEFAULT_EFFECTIVE_DATES.copy()
    # Remove classification headers that never got any step/rate rows.
    classes = {k: v for k, v in classes.items() if v.get("steps")}
    if not canonical_rows:
        canonical_rows = _build_canonical_rows_from_classes(
            classes=classes,
            contract_id=contract_id,
            source_method="table_registry",
            confidence_default=0.85,
        )
    return {
        "contract_id": contract_id,
        "effective_dates": effective_dates,
        "classifications": classes,
        "canonical_wage_rows": canonical_rows,
        "extraction_metadata": {
            "method": "table_registry",
            "candidate_tables": len(candidates),
            "source_table_ids": source_table_ids,
            "warnings": warnings,
            "row_type_counts": row_type_counts,
            "unresolved_rows": unresolved_rows[:200],
            "canonical_conflicts": canonical_conflicts[:200],
            "canonical_ambiguities": canonical_ambiguities[:200],
            "step_conflicts": step_conflicts[:200],
        },
    }


def _extract_wages_from_markdown(md_content: str, contract_id: str) -> dict:
    """Fallback markdown-table extraction path (legacy parser)."""
    start_markers = [
        'APPENDIX "A"',
        "APPENDIX A",
        "RATES OF PAY",
        "ARTICLE 8",
        "HEAD CLERK",
    ]
    appendix_start = -1
    for marker in start_markers:
        idx = md_content.find(marker)
        if idx != -1:
            appendix_start = idx
            break

    if appendix_start == -1:
        money_row_match = re.search(r"<tr[^>]*>.*?\$[\d,]+\.\d{2}.*?</tr>", md_content, re.DOTALL | re.IGNORECASE)
        if money_row_match:
            appendix_start = money_row_match.start()
        else:
            return {
                "contract_id": contract_id,
                "effective_dates": _WAGE_DEFAULT_EFFECTIVE_DATES.copy(),
                "classifications": {},
                "canonical_wage_rows": [],
                "extraction_metadata": {
                    "method": "markdown_fallback",
                    "warnings": ["could_not_find_wage_tables"],
                    "row_type_counts": {},
                    "unresolved_rows": [],
                    "canonical_conflicts": [],
                    "canonical_ambiguities": [],
                },
            }

    appendix_content = md_content[appendix_start:]
    end_markers = [
        "LETTERS OF UNDERSTANDING",
        "LETTERS OF AGREEMENT",
        "SAFEWAY INC. CLERKS LETTERS OF UNDERSTANDING",
        "SAFEWAY INC. MEAT LETTERS OF UNDERSTANDING",
    ]
    for marker in end_markers:
        idx = appendix_content.find(marker)
        if idx > 0:
            appendix_content = appendix_content[:idx]
            break

    classifications = {}
    current_classification = None
    current_steps = []
    effective_dates = _WAGE_DEFAULT_EFFECTIVE_DATES.copy()

    tr_pattern = r"<tr[^>]*>(.*?)</tr>"
    td_pattern = r"<td[^>]*>([^<]*)</td>"
    colspan_pattern = r"<td\s+colspan[^>]*>([^<]+)</td>"

    tr_matches = re.findall(tr_pattern, appendix_content, re.DOTALL | re.IGNORECASE)

    for tr_content in tr_matches:
        colspan_match = re.search(colspan_pattern, tr_content)
        if colspan_match:
            class_name = _strip_html(colspan_match.group(1))
            if class_name and len(class_name) > 2:
                if current_classification and current_steps:
                    classifications[current_classification["normalized_name"]] = {
                        "name": current_classification["name"],
                        "normalized_name": current_classification["normalized_name"],
                        "is_manager": current_classification["is_manager"],
                        "steps": current_steps,
                    }
                current_classification = {
                    "name": class_name,
                    "normalized_name": normalize_classification_name(class_name),
                    "is_manager": ("MANAGER" in class_name.upper()) or ("HEAD" in class_name.upper()),
                }
                current_steps = []
                continue

        td_values = [_strip_html(c) for c in re.findall(td_pattern, tr_content)]
        if len(td_values) < 4:
            continue

        label = td_values[0]
        row_dates = _extract_dates_from_cells(td_values[1:4])
        if len(row_dates) >= 3:
            effective_dates = row_dates[:3]
            continue

        rates = [parse_rate(td_values[1]), parse_rate(td_values[2]), parse_rate(td_values[3])]
        if sum(v is not None for v in rates) < 3:
            continue

        rates_map = _build_rate_map(effective_dates[:3], rates)
        if not rates_map:
            continue

        if current_classification is None:
            key = normalize_classification_name(label)
            classifications[key] = {
                "name": label,
                "normalized_name": key,
                "is_manager": ("manager" in label.lower()) or ("head" in label.lower()),
                "steps": [
                    {
                        "step_name": "Rate",
                        "hours_required": None,
                        "months_required": None,
                        "rates": rates_map,
                    }
                ],
            }
            continue

        hours_req, months_req = _derive_step_requirements(label, {"accumulated_hours": 0})
        current_steps.append(
            {
                "step_name": label,
                "hours_required": hours_req,
                "months_required": months_req,
                "rates": rates_map,
            }
        )

    if current_classification and current_steps:
        classifications[current_classification["normalized_name"]] = {
            "name": current_classification["name"],
            "normalized_name": current_classification["normalized_name"],
            "is_manager": current_classification["is_manager"],
            "steps": current_steps,
        }

    canonical_rows = _build_canonical_rows_from_classes(
        classes=classifications,
        contract_id=contract_id,
        source_method="markdown_fallback",
        confidence_default=0.75,
    )

    return {
        "contract_id": contract_id,
        "effective_dates": effective_dates,
        "classifications": classifications,
        "canonical_wage_rows": canonical_rows,
        "extraction_metadata": {
            "method": "markdown_fallback",
            "warnings": [],
            "row_type_counts": {},
            "unresolved_rows": [],
            "canonical_conflicts": [],
            "canonical_ambiguities": [],
        },
    }


def extract_wages(
    md_content: str,
    tables: Optional[list[Any]] = None,
    contract_id: str = CONTRACT_ID,
) -> dict:
    """
    Extract wages with deterministic precedence:
    1) structured tables (when available)
    2) markdown-table fallback
    """
    table_result = None
    if tables:
        table_result = _extract_wages_from_structured_tables(tables=tables, contract_id=contract_id)
        if table_result.get("classifications"):
            return table_result

    markdown_result = _extract_wages_from_markdown(md_content=md_content, contract_id=contract_id)
    if table_result is not None:
        markdown_meta = markdown_result.setdefault("extraction_metadata", {})
        warnings = markdown_meta.setdefault("warnings", [])
        warnings.append("table_registry_extraction_empty_fallback_to_markdown")
        warnings.extend(table_result.get("extraction_metadata", {}).get("warnings", []))
    return markdown_result


def _resolve_classification_key(wages_data: dict, classification: str) -> Optional[str]:
    """Resolve user-provided classification to a canonical wage class key."""
    norm_class = normalize_classification_name(classification)
    classes = wages_data.get("classifications", {}) or {}
    if not classes:
        return None

    alias_map = {
        normalize_classification_name(k): normalize_classification_name(v)
        for k, v in (wages_data.get("classification_aliases") or {}).items()
        if str(k).strip() and str(v).strip()
    }
    alias_target = alias_map.get(norm_class)
    if alias_target and alias_target in classes:
        norm_class = alias_target

    if norm_class not in classes:
        # 1) Explicit deterministic lexical aliases.
        for alias in _CLASSIFICATION_ALIASES.get(norm_class, ()):
            if alias in classes:
                norm_class = alias
                break

    if norm_class not in classes:
        # 2) Broad deterministic containment fallback.
        for key in classes:
            if norm_class in key or key in norm_class:
                norm_class = key
                break
        else:
            return None

    return norm_class


def _pick_effective_date(
    available_dates: list[str],
    requested_date: Optional[str],
    fallback_dates: list[str],
) -> Optional[str]:
    if not available_dates:
        return None

    normalized = sorted({d for d in available_dates if _ISO_DATE_RE.match(str(d or ""))})
    if not normalized:
        return None

    if requested_date and _ISO_DATE_RE.match(str(requested_date)):
        if requested_date in normalized:
            return requested_date
        prior_or_equal = [d for d in normalized if d <= requested_date]
        if prior_or_equal:
            return prior_or_equal[-1]

    for d in reversed(fallback_dates or []):
        if d in normalized:
            return d

    return normalized[-1]


def _preferred_wage_provenance_ref(provenance: Any) -> dict:
    refs = provenance if isinstance(provenance, list) else []
    normalized_refs = [ref for ref in refs if isinstance(ref, dict)]
    if not normalized_refs:
        return {}

    def _rank(ref: dict) -> tuple[int, int]:
        source_type = str(ref.get("source_type") or "").strip().lower()
        pdf_name = str(ref.get("pdf") or "").strip().lower()
        is_moa = int("moa" in source_type or "amend" in source_type or "moa" in pdf_name)
        has_page = int(ref.get("pdf_page") not in (None, "", 0))
        return (is_moa, has_page)

    return sorted(normalized_refs, key=_rank, reverse=True)[0]


def _lookup_wage_from_canonical_rows(
    wages_data: dict,
    class_key: str,
    class_data: dict,
    hours_worked: int,
    months_employed: int,
    effective_date: Optional[str],
) -> Optional[dict]:
    """Deterministically resolve wage from canonical row artifact first."""
    canonical_rows = wages_data.get("canonical_wage_rows", []) or []
    if not isinstance(canonical_rows, list) or not canonical_rows:
        return None

    target_rows = [
        row for row in canonical_rows
        if isinstance(row, dict)
        and normalize_classification_name(str(row.get("classification_key") or "")) == class_key
    ]
    if not target_rows:
        return None

    selected_date = _pick_effective_date(
        available_dates=[str(r.get("effective_date") or "") for r in target_rows],
        requested_date=effective_date,
        fallback_dates=wages_data.get("effective_dates") or _WAGE_DEFAULT_EFFECTIVE_DATES,
    )
    if not selected_date:
        return None

    rows_at_date = [
        row for row in target_rows
        if str(row.get("effective_date") or "") == selected_date
        and row.get("rate") is not None
    ]
    if not rows_at_date:
        return None

    step_types = {str(r.get("step_type") or "") for r in rows_at_date}
    if "hours" in step_types:
        progression_type = "hours"
        comparator_value = int(hours_worked or 0)
    elif "months" in step_types:
        progression_type = "months"
        comparator_value = int(months_employed or 0)
    else:
        progression_type = "fixed"
        comparator_value = 0

    applicable: list[dict] = []
    for row in rows_at_date:
        step_type = str(row.get("step_type") or "fixed")
        threshold_raw = row.get("threshold_value")
        threshold = 0
        if threshold_raw not in (None, ""):
            try:
                threshold = int(threshold_raw)
            except (TypeError, ValueError):
                continue

        if progression_type == "hours":
            if step_type == "hours" and comparator_value >= threshold:
                applicable.append(row)
            elif step_type == "fixed":
                applicable.append(row)
        elif progression_type == "months":
            if step_type == "months" and comparator_value >= threshold:
                applicable.append(row)
            elif step_type == "fixed":
                applicable.append(row)
        else:
            applicable.append(row)

    if not applicable:
        applicable = rows_at_date

    def _row_rank(row: dict) -> tuple:
        step_type = str(row.get("step_type") or "fixed")
        threshold_raw = row.get("threshold_value")
        threshold = 0
        if threshold_raw not in (None, ""):
            try:
                threshold = int(threshold_raw)
            except (TypeError, ValueError):
                threshold = 0
        confidence = float(row.get("confidence", 0.0) or 0.0)
        source_ref = row.get("source_reference") if isinstance(row.get("source_reference"), dict) else {}
        has_table_ref = 0 if source_ref.get("table_id") else 1
        progression_priority = 0 if step_type == progression_type else (1 if step_type == "fixed" else 2)
        return (
            progression_priority,
            -threshold,
            -confidence,
            has_table_ref,
            str(row.get("step_name") or ""),
        )

    chosen = sorted(applicable, key=_row_rank)[0]
    rate = chosen.get("rate")
    if rate is None:
        return None

    chosen_step_name = str(chosen.get("step_name") or "Rate")
    chosen_rate = float(rate)
    evidence_rows = []
    evidence_seen = set()
    for row in rows_at_date:
        if str(row.get("step_name") or "") != chosen_step_name:
            continue
        try:
            row_rate = float(row.get("rate"))
        except (TypeError, ValueError):
            continue
        if abs(row_rate - chosen_rate) > 1e-9:
            continue
        source_ref = row.get("source_reference") if isinstance(row.get("source_reference"), dict) else {}
        table_id = str(source_ref.get("table_id") or "").strip()
        row_index_raw = source_ref.get("row_index")
        row_index = None
        if row_index_raw is not None:
            try:
                row_index = int(row_index_raw)
            except (TypeError, ValueError):
                row_index = None
        evidence_key = (table_id, row_index)
        if evidence_key in evidence_seen:
            continue
        evidence_seen.add(evidence_key)
        preferred_ref = _preferred_wage_provenance_ref(row.get("provenance"))
        source_page_raw = preferred_ref.get("pdf_page")
        source_page = None
        if source_page_raw not in (None, "", 0):
            try:
                source_page = int(source_page_raw)
            except (TypeError, ValueError):
                source_page = None
        evidence_rows.append(
            {
                "table_id": table_id or None,
                "row_index": row_index,
                "classification_key": class_key,
                "step_name": chosen_step_name,
                "effective_date": selected_date,
                "rate": chosen_rate,
                "row_type": str(row.get("row_type") or ""),
                "source_method": str(row.get("source_method") or "canonical_rows"),
                "confidence": float(row.get("confidence", 0.0) or 0.0),
                "provenance": row.get("provenance") if isinstance(row.get("provenance"), list) else [],
                "source_type": str(preferred_ref.get("source_type") or "").strip() or None,
                "source_pdf": str(preferred_ref.get("pdf") or "").strip() or None,
                "source_page": source_page,
                "source_doc_id": str(preferred_ref.get("source_doc_id") or "").strip() or None,
                "effective_version_id": str(row.get("effective_version_id") or "").strip() or None,
                "amendments_applied": list(row.get("amendments_applied") or []),
                "selected_schedule_label": str(row.get("selected_schedule_label") or "").strip() or None,
                "source_rate_schedule": dict(row.get("source_rate_schedule") or {})
                if isinstance(row.get("source_rate_schedule"), dict)
                else {},
            }
        )

    display_name = (
        class_data.get("name")
        or chosen.get("classification_name")
        or class_key
    )
    return {
        "classification": display_name,
        "classification_key": class_key,
        "step": chosen_step_name,
        "rate": chosen_rate,
        "effective_date": selected_date,
        "citation": "Appendix A",
        "source_method": "canonical_rows",
        "table_evidence": evidence_rows,
        "effective_version_id": str(wages_data.get("effective_version_id") or "").strip() or None,
        "amendments_applied": list(wages_data.get("amendments_applied") or []),
        "selected_schedule_label": str(chosen.get("selected_schedule_label") or "").strip() or None,
        "source_rate_schedule": dict(chosen.get("source_rate_schedule") or {})
        if isinstance(chosen.get("source_rate_schedule"), dict)
        else {},
    }


def lookup_wage(
    wages_data: dict,
    classification: str,
    hours_worked: int = 0,
    months_employed: int = 0,
    effective_date: str = None,
) -> Optional[dict]:
    """Deterministically resolve wage step/rate by classification and tenure."""
    effective_dates = wages_data.get("effective_dates") or _WAGE_DEFAULT_EFFECTIVE_DATES
    if effective_date is None:
        # Undated query → the rate in effect today, never a future scheduled raise.
        effective_date = _default_effective_date(effective_dates)

    classes = wages_data.get("classifications", {}) or {}
    norm_class = _resolve_classification_key(wages_data, classification)
    if not norm_class or norm_class not in classes:
        return None

    class_data = classes[norm_class]
    steps = class_data.get("steps", [])
    if not steps:
        return None

    canonical_match = _lookup_wage_from_canonical_rows(
        wages_data=wages_data,
        class_key=norm_class,
        class_data=class_data,
        hours_worked=hours_worked,
        months_employed=months_employed,
        effective_date=effective_date,
    )
    if canonical_match:
        return canonical_match

    applicable_step = steps[0]
    for step in steps:
        hours_required = step.get("hours_required")
        months_required = step.get("months_required")
        if hours_required is not None and hours_worked >= hours_required:
            applicable_step = step
        elif months_required is not None and months_employed >= months_required:
            applicable_step = step

    rate_map = applicable_step.get("rates", {})
    rate = rate_map.get(effective_date)
    if rate is None and rate_map:
        # If requested date is missing, use latest available from this step.
        for d in reversed(effective_dates):
            if d in rate_map:
                effective_date = d
                rate = rate_map[d]
                break
    if rate is None:
        return None

    return {
        "classification": class_data.get("name", classification),
        "classification_key": norm_class,
        "step": applicable_step.get("step_name", "Rate"),
        "rate": rate,
        "effective_date": effective_date,
        "citation": "Appendix A",
        "source_method": "classification_steps",
        "table_evidence": [],
    }


def save_wages(wages_data: dict, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(wages_data, f, indent=2, ensure_ascii=False)
    print(f"Saved wages data to {output_file}")


def main():
    print(f"Extracting wages from: {CONTRACT_MD_FILE}")
    with open(CONTRACT_MD_FILE, "r", encoding="utf-8") as f:
        md_content = f.read()

    wages_data = extract_wages(md_content, contract_id=CONTRACT_ID)
    print(f"\nExtracted {len(wages_data.get('classifications', {}))} classifications:")
    for _, data in wages_data.get("classifications", {}).items():
        print(f"  - {data.get('name')}: {len(data.get('steps', []))} steps")

    output_file = WAGES_DIR / "wage_tables.json"
    save_wages(wages_data, output_file)

    print("\n--- Test Lookups ---")
    test_cases = [
        ("all_purpose_clerk", 0, 0),
        ("all_purpose_clerk", 3000, 0),
        ("all_purpose_clerk", 8000, 0),
        ("courtesy_clerk", 0, 0),
        ("courtesy_clerk", 0, 48),
        ("head_clerk", 0, 0),
    ]
    for classification, hours, months in test_cases:
        result = lookup_wage(wages_data, classification, hours, months)
        if result:
            print(f"  {classification} ({hours}hrs/{months}mo): ${result['rate']:.2f} ({result['step']})")
        else:
            print(f"  {classification}: Not found")
    return wages_data


if __name__ == "__main__":
    main()
