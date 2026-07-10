"""Deterministic extraction/lookup for contract entitlement schedules."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import CONTRACT_ID, ENTITLEMENTS_DIR


ENTITLEMENT_SCHEMA_VERSION = "entitlement_tables_v1"
_WORD_TO_INT = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
_VACATION_TIER_RE = re.compile(
    r"(?:(?P<week_word>one|two|three|four|five|six|seven|eight|nine|ten)\s+)?"
    r"(?:\((?P<week_num>\d+)\)\s*)?"
    r"week(?:s)?(?:['’]s?|['’])?\s+paid\s+vacation\s+after\s+"
    r"(?:(?P<year_word>one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|twenty)\s+)?"
    r"(?:\((?P<year_num>\d+)\)\s*)?"
    r"year(?:s)?",
    re.IGNORECASE,
)
_HIRE_BEFORE_RE = re.compile(
    r"hired\s+on\s+or\s+before\s+"
    r"(?P<date>(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4})",
    re.IGNORECASE,
)
_HIRE_AFTER_RE = re.compile(
    r"hired\s+on\s+or\s+after\s+"
    r"(?P<date>(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4})",
    re.IGNORECASE,
)
_ANNIVERSARY_HOURS_RE = re.compile(
    r"worked[^.]{0,220}\((?P<hours>\d[\d,]*)\)\s+or\s+more\s+hours?\s+in\s+their\s+anniversary\s+year",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _strip_html(text: str) -> str:
    if text is None:
        return ""
    value = str(text)
    value = value.replace("<br/>", " ").replace("<br>", " ").replace("<br />", " ")
    value = re.sub(r"<[^>]+>", " ", value)
    value = value.replace("&nbsp;", " ").replace("&amp;", "&")
    return re.sub(r"\s+", " ", value).strip()


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", _strip_html(text)).strip()


def _to_iso_date(value: Optional[str]) -> Optional[str]:
    text = _normalize_text(value or "")
    if not text:
        return None
    if _ISO_DATE_RE.match(text):
        return text
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except Exception:
            continue
    return None


def _extract_vacation_tiers(text: str) -> list[dict]:
    tiers: list[dict] = []
    seen: set[tuple[int, int]] = set()
    normalized = _normalize_text(text)
    for m in _VACATION_TIER_RE.finditer(normalized):
        week_num = m.group("week_num")
        year_num = m.group("year_num")
        week_word = str(m.group("week_word") or "").lower()
        year_word = str(m.group("year_word") or "").lower()
        weeks = int(week_num) if week_num else _WORD_TO_INT.get(week_word)
        years = int(year_num) if year_num else _WORD_TO_INT.get(year_word)
        if weeks is None or years is None:
            continue
        key = (int(years), int(weeks))
        if key in seen:
            continue
        seen.add(key)
        tiers.append(
            {
                "years_of_service": int(years),
                "weeks_per_year": int(weeks),
            }
        )
    tiers.sort(key=lambda t: (t["years_of_service"], t["weeks_per_year"]))
    return tiers


def _extract_conditions(text: str) -> dict:
    normalized = _normalize_text(text)
    before_match = _HIRE_BEFORE_RE.search(normalized)
    after_match = _HIRE_AFTER_RE.search(normalized)
    hours_match = _ANNIVERSARY_HOURS_RE.search(normalized)
    hours_min = None
    if hours_match:
        raw = str(hours_match.group("hours") or "").replace(",", "").strip()
        if raw.isdigit():
            hours_min = int(raw)
    return {
        "hire_date_on_or_before": _to_iso_date(before_match.group("date")) if before_match else None,
        "hire_date_on_or_after": _to_iso_date(after_match.group("date")) if after_match else None,
        "anniversary_hours_min": hours_min,
    }


def _vacation_clauses(text: str) -> list[str]:
    normalized = _normalize_text(text)
    if "paid vacation after" not in normalized.lower():
        return []

    parts = re.split(r"(?=all regular full-time employees)", normalized, flags=re.IGNORECASE)
    clauses: list[str] = []
    for part in parts:
        lower = part.lower()
        if "paid vacation after" not in lower:
            continue
        m = re.search(
            r"(all regular full-time employees.*?paid vacation after.*?(?:continuous service|year'?s service)\.)",
            part,
            flags=re.IGNORECASE,
        )
        if m:
            clauses.append(_normalize_text(m.group(1)))
        else:
            clauses.append(_normalize_text(part))

    if clauses:
        return clauses
    return [normalized]


def _schedule_key(entry: dict) -> tuple:
    cond = entry.get("conditions") or {}
    tiers = tuple(
        (int(t.get("years_of_service") or 0), int(t.get("weeks_per_year") or 0))
        for t in (entry.get("tiers") or [])
    )
    return (
        cond.get("hire_date_on_or_before"),
        cond.get("hire_date_on_or_after"),
        cond.get("anniversary_hours_min"),
        tiers,
    )


def extract_entitlements(
    chunks: list[dict],
    contract_id: str = CONTRACT_ID,
    manifest: Optional[dict] = None,
) -> dict:
    """
    Build deterministic contract-scoped entitlement artifact from chunks.

    Currently extracts vacation-accrual tier ladders (years -> weeks per year).
    """
    region_id = None
    if isinstance(manifest, dict):
        region_id = manifest.get("region_id")
    if not region_id:
        for chunk in chunks or []:
            region_id = chunk.get("region_id")
            if region_id:
                break

    candidates: list[dict] = []
    for chunk in chunks or []:
        chunk_text = _normalize_text(
            chunk.get("content_with_tables")
            or chunk.get("content")
            or ""
        )
        for content in _vacation_clauses(chunk_text):
            tiers = _extract_vacation_tiers(content)
            if len(tiers) < 2:
                continue

            citation = str(chunk.get("citation") or "").strip()
            evidence = {
                "chunk_id": chunk.get("chunk_id"),
                "citation": citation,
                "article_num": chunk.get("article_num"),
                "section_num": chunk.get("section_num"),
                "subsection": chunk.get("subsection"),
                "source_excerpt": content[:600],
            }
            candidates.append(
                {
                    "entitlement_type": "vacation_accrual",
                    "conditions": _extract_conditions(content),
                    "tiers": tiers,
                    "citation": citation or "Article 17",
                    "source_method": "chunk_pattern_v1",
                    "confidence": 0.9,
                    "source_evidence": [evidence],
                }
            )

    merged: list[dict] = []
    key_to_index: dict[tuple, int] = {}
    for row in candidates:
        key = _schedule_key(row)
        idx = key_to_index.get(key)
        if idx is None:
            key_to_index[key] = len(merged)
            merged.append(row)
            continue
        merged[idx]["source_evidence"].extend(row.get("source_evidence") or [])
        if len(str(row.get("citation") or "")) > len(str(merged[idx].get("citation") or "")):
            merged[idx]["citation"] = row.get("citation")
        merged[idx]["confidence"] = max(
            float(merged[idx].get("confidence", 0.0) or 0.0),
            float(row.get("confidence", 0.0) or 0.0),
        )

    for idx, row in enumerate(merged, start=1):
        row["schedule_id"] = f"vacation_schedule_{idx}"
        row["source_evidence"] = (row.get("source_evidence") or [])[:6]

    return {
        "schema_version": ENTITLEMENT_SCHEMA_VERSION,
        "contract_id": contract_id,
        "region_id": region_id,
        "vacation_entitlements": merged,
        "extraction_metadata": {
            "source_method": "chunk_pattern_v1",
            "candidate_count": len(candidates),
            "schedule_count": len(merged),
        },
    }


def _schedule_specificity(schedule: dict) -> int:
    cond = schedule.get("conditions") or {}
    score = 0
    if cond.get("hire_date_on_or_before"):
        score += 1
    if cond.get("hire_date_on_or_after"):
        score += 1
    if cond.get("anniversary_hours_min") is not None:
        score += 1
    return score


def _select_schedule(
    schedules: list[dict],
    hire_date_iso: Optional[str],
    hours_worked: int,
) -> tuple[Optional[dict], list[dict], list[str]]:
    considered: list[dict] = []
    elimination_reasons: list[str] = []

    for schedule in schedules:
        cond = schedule.get("conditions") or {}
        before = cond.get("hire_date_on_or_before")
        after = cond.get("hire_date_on_or_after")
        hours_min = cond.get("anniversary_hours_min")

        if hire_date_iso:
            if before and hire_date_iso > before:
                elimination_reasons.append(
                    f"{schedule.get('schedule_id')}: hire_date>{before}"
                )
                continue
            if after and hire_date_iso < after:
                elimination_reasons.append(
                    f"{schedule.get('schedule_id')}: hire_date<{after}"
                )
                continue
        if hours_worked > 0 and isinstance(hours_min, int) and hours_worked < hours_min:
            elimination_reasons.append(
                f"{schedule.get('schedule_id')}: hours<{hours_min}"
            )
            continue
        considered.append(schedule)

    if not considered:
        considered = list(schedules)

    selected = None
    if len(considered) == 1:
        selected = considered[0]
    elif not hire_date_iso:
        has_hire_windows = any(
            bool((s.get("conditions") or {}).get("hire_date_on_or_before"))
            or bool((s.get("conditions") or {}).get("hire_date_on_or_after"))
            for s in considered
        )
        if has_hire_windows:
            selected = None
            return selected, considered, elimination_reasons
    elif hire_date_iso:
        considered_sorted = sorted(
            considered,
            key=lambda s: (_schedule_specificity(s), s.get("schedule_id") or ""),
            reverse=True,
        )
        selected = considered_sorted[0]
    elif hours_worked > 0:
        considered_sorted = sorted(
            considered,
            key=lambda s: (
                int((s.get("conditions") or {}).get("anniversary_hours_min") or 0),
                _schedule_specificity(s),
            ),
            reverse=True,
        )
        top = considered_sorted[0]
        top_hours = int((top.get("conditions") or {}).get("anniversary_hours_min") or 0)
        ties = [
            s for s in considered_sorted
            if int((s.get("conditions") or {}).get("anniversary_hours_min") or 0) == top_hours
        ]
        if len(ties) == 1:
            selected = top
    return selected, considered, elimination_reasons


def lookup_vacation_entitlement(
    entitlements_data: dict,
    months_employed: int = 0,
    hours_worked: int = 0,
    hire_date: Optional[str] = None,
) -> Optional[dict]:
    schedules = list((entitlements_data or {}).get("vacation_entitlements") or [])
    if not schedules:
        return None

    months_value = int(months_employed or 0)
    years_completed = (months_value // 12) if months_value > 0 else None
    hire_date_iso = _to_iso_date(hire_date)
    selected, considered, elimination_reasons = _select_schedule(
        schedules=schedules,
        hire_date_iso=hire_date_iso,
        hours_worked=int(hours_worked or 0),
    )

    estimated_weeks = None
    if selected and years_completed is not None:
        best = 0
        for tier in selected.get("tiers") or []:
            years = int(tier.get("years_of_service") or 0)
            weeks = int(tier.get("weeks_per_year") or 0)
            if years_completed >= years:
                best = max(best, weeks)
        estimated_weeks = best

    if selected and years_completed is not None:
        confidence = "exact" if hire_date_iso or len(considered) == 1 else "conditional"
    elif selected:
        confidence = "insufficient_profile"
    else:
        confidence = "conditional"

    citations = []
    for row in ([selected] if selected else considered):
        citation = str((row or {}).get("citation") or "").strip()
        if citation and citation not in citations:
            citations.append(citation)
    citation = "; ".join(citations[:3]) if citations else "Article 17"

    evidence = []
    selected_rows = [selected] if selected else considered[:2]
    for row in selected_rows:
        for ev in (row or {}).get("source_evidence") or []:
            if isinstance(ev, dict):
                evidence.append(ev)
    evidence = evidence[:6]

    return {
        "entitlement": "vacation",
        "source_method": "vacation_entitlement_tiers",
        "months_employed": months_value,
        "years_completed": years_completed,
        "hours_worked": int(hours_worked or 0),
        "hire_date": hire_date_iso,
        "confidence": confidence,
        "estimated_weeks_per_year": estimated_weeks,
        "selected_schedule": selected,
        "schedules_considered": considered,
        "citation": citation,
        "entitlement_evidence": evidence,
        "elimination_reasons": elimination_reasons[:12],
    }


def save_entitlements(entitlements_data: dict, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(entitlements_data, f, indent=2, ensure_ascii=False)


def main() -> int:
    # Local utility run against existing runtime chunk artifacts.
    from backend.chunk_files import resolve_chunk_file

    chunk_file = resolve_chunk_file(contract_id=CONTRACT_ID, allow_shared_fallback=True)
    if not chunk_file or not chunk_file.exists():
        print(f"[FAIL] Could not locate chunk file for contract: {CONTRACT_ID}")
        return 1
    with open(chunk_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    artifact = extract_entitlements(chunks=chunks, contract_id=CONTRACT_ID, manifest=None)
    out = ENTITLEMENTS_DIR / f"entitlement_tables_{CONTRACT_ID}.json"
    save_entitlements(artifact, out)
    print(f"[OK] Saved entitlement artifact: {out}")
    print(f"Schedules: {len(artifact.get('vacation_entitlements') or [])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
