"""
User Profile Management for KARL.

Stores user context (hire date, classification, employment type) to enable:
- Accurate wage estimates
- Hire-date sensitive provision lookups
- Personalized responses

Privacy Note: Profile data is stored in session only, not persisted.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional, Literal
from enum import Enum
import math
import json
import re

from backend.contracts import get_contract_catalog_entry
from backend.wage_files import resolve_wage_file
from backend.classification_ontology_files import resolve_classification_ontology_file
from backend.role_catalog_files import resolve_role_catalog_file
from backend.ingest.extract_wages import normalize_classification_name


class EmploymentType(str, Enum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    UNKNOWN = "unknown"


@dataclass
class UserProfile:
    """
    User profile for personalized contract queries.

    All fields are optional - KARL works with partial info but
    gives more accurate answers with complete profiles.
    """
    # Core identification
    session_id: str = ""

    # Contract context
    contract_id: str = ""
    union_local: str = ""
    employer: str = ""

    # Job info
    classification: Optional[str] = None
    employment_type: EmploymentType = EmploymentType.UNKNOWN

    # Tenure info
    hire_date: Optional[date] = None

    # Cached calculations
    _estimated_hours: Optional[int] = field(default=None, repr=False)

    # Profile completeness
    def __post_init__(self):
        """Fill contract metadata when a contract is explicitly selected."""
        meta = get_contract_catalog_entry(self.contract_id)
        if meta:
            if not self.union_local:
                self.union_local = meta.get("union_local_id", "")
            if not self.employer:
                self.employer = meta.get("employer", "")

    @property
    def is_complete(self) -> bool:
        """Check if we have enough info for accurate wage lookups."""
        return all([
            self.classification,
            self.employment_type != EmploymentType.UNKNOWN,
            self.hire_date
        ])

    @property
    def has_basic_info(self) -> bool:
        """Check if we have at least classification."""
        return self.classification is not None

    @property
    def months_employed(self) -> Optional[int]:
        """Calculate months since hire date."""
        if not self.hire_date:
            return None

        today = date.today()
        months = (today.year - self.hire_date.year) * 12 + (today.month - self.hire_date.month)
        return max(0, months)

    @property
    def estimated_hours(self) -> Optional[int]:
        """
        Estimate total hours worked based on tenure and employment type.

        Assumptions:
        - Full-time: ~36 hours/week average (accounting for vacation, sick)
        - Part-time: ~20 hours/week average
        - 4.33 weeks per month

        Returns conservative (floor) estimate.
        """
        if self._estimated_hours is not None:
            return self._estimated_hours

        if not self.hire_date:
            return None

        months = self.months_employed
        if months is None:
            return None

        # Average weekly hours by employment type
        if self.employment_type == EmploymentType.FULL_TIME:
            avg_weekly_hours = 36
        elif self.employment_type == EmploymentType.PART_TIME:
            avg_weekly_hours = 20
        else:
            # Conservative estimate if unknown
            avg_weekly_hours = 20

        # Calculate: months * weeks_per_month * hours_per_week
        estimated = int(months * 4.33 * avg_weekly_hours)

        return estimated

    def set_estimated_hours(self, hours: int):
        """Allow manual override of hours if user knows exact value."""
        self._estimated_hours = hours

    @property
    def is_grandfathered(self) -> Optional[bool]:
        """
        Check if hired before March 27, 2005 (grandfathered provisions).

        Many contract provisions have different rules for employees
        hired before this date.
        """
        if not self.hire_date:
            return None

        grandfather_date = date(2005, 3, 27)
        return self.hire_date < grandfather_date

    def to_dict(self) -> dict:
        """Serialize profile for API responses."""
        return {
            "session_id": self.session_id,
            "contract_id": self.contract_id,
            "union_local": self.union_local,
            "employer": self.employer,
            "classification": self.classification,
            "employment_type": self.employment_type.value if self.employment_type else None,
            "hire_date": self.hire_date.isoformat() if self.hire_date else None,
            "months_employed": self.months_employed,
            "estimated_hours": self.estimated_hours,
            "is_grandfathered": self.is_grandfathered,
            "is_complete": self.is_complete,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UserProfile":
        """Deserialize profile from API request."""
        hire_date = None
        if data.get("hire_date"):
            if isinstance(data["hire_date"], str):
                hire_date = date.fromisoformat(data["hire_date"])
            elif isinstance(data["hire_date"], date):
                hire_date = data["hire_date"]

        employment_type = EmploymentType.UNKNOWN
        if data.get("employment_type"):
            try:
                employment_type = EmploymentType(data["employment_type"])
            except ValueError:
                pass

        return cls(
            session_id=data.get("session_id", ""),
            contract_id=data.get("contract_id", ""),
            union_local=data.get("union_local", ""),
            employer=data.get("employer", ""),
            classification=data.get("classification"),
            employment_type=employment_type,
            hire_date=hire_date,
        )


# =============================================================================
# HOURS ESTIMATOR
# =============================================================================

@dataclass
class HoursEstimate:
    """Result of hours estimation with transparency metadata."""
    estimated_hours: int
    confidence: Literal["exact", "high", "medium", "low"]
    basis: str  # Human-readable explanation
    current_step: Optional[str] = None
    next_step: Optional[str] = None
    hours_to_next_step: Optional[int] = None

    @property
    def disclaimer(self) -> str:
        """Generate appropriate disclaimer based on confidence."""
        if self.confidence == "exact":
            return ""
        elif self.confidence == "high":
            return "This is an estimate based on your tenure. Your actual hours may vary."
        else:
            return (
                "This is a rough estimate. Your actual wage depends on total hours worked. "
                "Check your pay stub or the Company HR Portal for exact figures."
            )


def estimate_hours_worked(profile: UserProfile) -> Optional[HoursEstimate]:
    """
    Estimate hours worked with confidence level and transparency.

    Returns HoursEstimate with explanation of how it was calculated.
    """
    if profile._estimated_hours is not None:
        return HoursEstimate(
            estimated_hours=profile._estimated_hours,
            confidence="exact",
            basis="You provided your exact hours worked."
        )

    if not profile.hire_date:
        return None

    months = profile.months_employed
    estimated = profile.estimated_hours

    if estimated is None:
        return None

    # Determine confidence based on what we know
    if profile.employment_type == EmploymentType.UNKNOWN:
        confidence = "low"
        basis = f"Based on ~{months} months employed, assuming part-time (~20 hrs/week)."
    elif profile.employment_type == EmploymentType.FULL_TIME:
        confidence = "medium"
        basis = f"Based on ~{months} months full-time (~36 hrs/week average)."
    else:
        confidence = "medium"
        basis = f"Based on ~{months} months part-time (~20 hrs/week average)."

    return HoursEstimate(
        estimated_hours=estimated,
        confidence=confidence,
        basis=basis
    )


# =============================================================================
# SESSION PROFILE STORAGE
# =============================================================================

# In-memory storage for session profiles
_session_profiles: dict[str, UserProfile] = {}


def get_user_profile(session_id: str) -> UserProfile:
    """Get or create user profile for session."""
    if session_id not in _session_profiles:
        _session_profiles[session_id] = UserProfile(session_id=session_id)
    return _session_profiles[session_id]


def _normalize_profile_classification_value(
    value: Optional[str],
    contract_id: Optional[str] = None,
) -> Optional[str]:
    """
    Normalize classification input from UI/API into canonical option value.

    Accepts canonical values, display labels, or free-form text and returns a
    stable snake_case key for runtime wage lookup.
    """
    if value is None:
        return None

    raw = str(value).strip()
    if not raw:
        return None
    lowered = raw.lower()

    if contract_id:
        options = get_classification_options(contract_id=contract_id, include_unmapped=True)
        for opt in options:
            candidate = str(opt.get("value") or "").strip().lower()
            if candidate and candidate == lowered:
                return str(opt.get("wage_key") or candidate).strip().lower() or candidate
        for opt in options:
            label = str(opt.get("label") or "").strip().lower()
            candidate = str(opt.get("value") or "").strip().lower()
            if label and label == lowered and candidate:
                return str(opt.get("wage_key") or candidate).strip().lower() or candidate

    for key, label in CLASSIFICATION_DISPLAY_NAMES.items():
        if lowered == key.lower() or lowered == str(label).strip().lower():
            return key

    normalized = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return normalized or lowered


def update_user_profile(session_id: str, updates: dict) -> UserProfile:
    """Update user profile with new data."""
    profile = get_user_profile(session_id)

    if "contract_id" in updates:
        profile.contract_id = updates["contract_id"]
        meta = get_contract_catalog_entry(profile.contract_id)
        if meta:
            # Keep profile context aligned with selected contract.
            profile.union_local = meta.get("union_local_id", profile.union_local)
            profile.employer = meta.get("employer", profile.employer)

    if "classification" in updates:
        profile.classification = _normalize_profile_classification_value(
            updates["classification"],
            contract_id=profile.contract_id,
        )

    if "employment_type" in updates:
        try:
            profile.employment_type = EmploymentType(updates["employment_type"])
        except ValueError:
            pass

    if "hire_date" in updates:
        if isinstance(updates["hire_date"], str):
            profile.hire_date = date.fromisoformat(updates["hire_date"])
        elif isinstance(updates["hire_date"], date):
            profile.hire_date = updates["hire_date"]

    if "exact_hours" in updates:
        profile.set_estimated_hours(int(updates["exact_hours"]))

    if "employer" in updates:
        profile.employer = updates["employer"]

    return profile


def clear_user_profile(session_id: str):
    """Clear profile data for session."""
    if session_id in _session_profiles:
        del _session_profiles[session_id]


# =============================================================================
# CLASSIFICATION HELPERS
# =============================================================================

# Human-readable classification names for onboarding UI
CLASSIFICATION_DISPLAY_NAMES = {
    "all_purpose_clerk": "All-Purpose Clerk",
    "courtesy_clerk": "Courtesy Clerk",
    "head_clerk": "Head Clerk",
    "produce_department_manager": "Produce Manager",
    "bakery_manager": "Bakery Manager",
    "floral_manager": "Floral Manager",
    "cake_decorator": "Cake Decorator",
    "nonfood_gm_floral": "Non-Food/GM/Floral Clerk",
    "bakery_fresh_cut_liquor_clerk": "Bakery/Fresh Cut/Liquor Clerk",
    "pharmacy_tech": "Pharmacy Technician",
    "dug_shopper": "DUG Shopper (Drive Up & Go)",
    "fuel_lead": "Fuel Lead",
    "fresh_cut_supervisor": "Fresh Cut Supervisor",
    "head_baker": "Head Baker",
    "variety_manager": "Variety Manager",
    "manager_trainee": "Manager Trainee",
    "other_assistant_managers": "Other Assistant Managers",
}


_classification_cache_by_contract: dict[str, list[dict]] = {}


def _prettify_classification_label(value: str) -> str:
    """Generate a readable fallback label from a normalized classification value."""
    cleaned = re.sub(r"[_\s]+", " ", (value or "").strip()).strip()
    if not cleaned:
        return "Unknown Classification"

    words = []
    for token in cleaned.split(" "):
        upper = token.upper()
        if upper in {"HR", "GM", "DUG", "PTO", "ASST"}:
            words.append(upper)
        else:
            words.append(token.capitalize())
    return " ".join(words)


def _normalize_classification_label(label: str) -> str:
    """Normalize extracted classification labels for consistent UI display."""
    cleaned = re.sub(r"\s+", " ", (label or "").strip()).strip(" .")
    if not cleaned:
        return ""

    # Expand two-digit years in slash dates to avoid ambiguity (e.g., 5/20/77 -> 5/20/1977).
    def _expand_year(m: re.Match) -> str:
        month, day, year2 = int(m.group(1)), int(m.group(2)), int(m.group(3))
        year4 = 1900 + year2 if year2 >= 50 else 2000 + year2
        return f"{month}/{day}/{year4}"

    cleaned = re.sub(r"(?<!\d)(\d{1,2})/(\d{1,2})/(\d{2})(?!\d)", _expand_year, cleaned)

    # Title-case all-caps OCR labels while preserving a few acronyms.
    letters = [ch for ch in cleaned if ch.isalpha()]
    is_mostly_upper = bool(letters) and (sum(1 for ch in letters if ch.isupper()) / len(letters) >= 0.6)
    if is_mostly_upper:
        cleaned = cleaned.title()
        for acronym in ("Hr", "Gm", "Dug", "Pto", "Ufcw", "Asst"):
            cleaned = re.sub(rf"\b{acronym}\b", acronym.upper(), cleaned)

    return cleaned


def _load_contract_classification_options_from_wages(contract_id: str) -> list[dict]:
    """Build classification options from contract-scoped wage artifacts when available."""
    wages_file = resolve_wage_file(contract_id=contract_id, allow_shared_fallback=False)
    if not wages_file or not wages_file.exists():
        return []

    try:
        with open(wages_file, "r", encoding="utf-8") as f:
            wages_data = json.load(f)
    except Exception:
        return []

    classes = wages_data.get("classifications") or {}
    options: list[dict] = []
    for key, cls in classes.items():
        value = normalize_classification_name(str(cls.get("normalized_name") or key or ""))
        if not value:
            continue
        label = (cls.get("name") or "").strip()
        if not label:
            label = CLASSIFICATION_DISPLAY_NAMES.get(value) or _prettify_classification_label(value)
        label = _normalize_classification_label(label) or _prettify_classification_label(value)
        options.append(
            {
                "value": value,
                "label": label,
                "wage_available": True,
                "wage_key": value,
                "source": "wage_table",
            }
        )

    # Preserve first-seen uniqueness by value.
    seen = set()
    deduped: list[dict] = []
    for opt in options:
        if opt["value"] in seen:
            continue
        seen.add(opt["value"])
        deduped.append(opt)
    return deduped


def _load_contract_classification_options_from_role_catalog(contract_id: str) -> list[dict]:
    """Build classification options from contract-scoped role catalog when available."""
    role_catalog_file = resolve_role_catalog_file(
        contract_id=contract_id,
        allow_shared_fallback=False,
    )
    if not role_catalog_file or not role_catalog_file.exists():
        return []

    try:
        with open(role_catalog_file, "r", encoding="utf-8") as f:
            role_catalog_data = json.load(f)
    except Exception:
        return []

    if str(role_catalog_data.get("contract_id") or "").strip().lower() != contract_id.strip().lower():
        return []

    roles = role_catalog_data.get("roles") or []
    if not isinstance(roles, list):
        return []

    options: list[dict] = []
    for role in roles:
        if not isinstance(role, dict):
            continue
        value = normalize_classification_name(str(role.get("value") or ""))
        if not value:
            continue
        label = _normalize_classification_label(str(role.get("label") or ""))
        if not label:
            label = CLASSIFICATION_DISPLAY_NAMES.get(value) or _prettify_classification_label(value)
        alias_labels = role.get("alias_labels") if isinstance(role.get("alias_labels"), list) else []
        alias_labels = [str(x).strip() for x in alias_labels if str(x).strip()]
        if alias_labels and bool(role.get("onboarding_default")):
            preview = " / ".join(alias_labels[:2])
            if preview and preview.lower() not in label.lower():
                label = f"{label} ({preview})"

        wage_available = bool(role.get("wage_available"))
        wage_key = normalize_classification_name(str(role.get("wage_key") or "")) or None
        if not wage_available:
            wage_key = None
        onboarding_default = (
            bool(role.get("onboarding_default"))
            if "onboarding_default" in role
            else wage_available
        )

        options.append(
            {
                "value": value,
                "label": label,
                "wage_available": wage_available,
                "wage_key": wage_key,
                "source": str(role.get("source") or "role_catalog"),
                "mapping_method": str(role.get("mapping_method") or ""),
                "manifest_present": bool(role.get("manifest_present")),
                "onboarding_default": onboarding_default,
                "alias_labels": alias_labels,
            }
        )

    seen = set()
    deduped: list[dict] = []
    for opt in options:
        if opt["value"] in seen:
            continue
        seen.add(opt["value"])
        deduped.append(opt)
    return deduped


def _load_contract_classification_options_from_ontology(contract_id: str) -> list[dict]:
    """Build classification options from contract ontology decisions when available."""
    ontology_file = resolve_classification_ontology_file(
        contract_id=contract_id,
        allow_shared_fallback=False,
    )
    if not ontology_file or not ontology_file.exists():
        return []

    try:
        with open(ontology_file, "r", encoding="utf-8") as f:
            ontology_data = json.load(f)
    except Exception:
        return []

    decisions = ontology_data.get("decisions") or []
    if not isinstance(decisions, list):
        return []

    options: list[dict] = []
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        value = normalize_classification_name(str(decision.get("source_key") or ""))
        if not value:
            continue
        labels = decision.get("source_labels") or []
        label = ""
        if isinstance(labels, list):
            for raw in labels:
                label = _normalize_classification_label(str(raw or ""))
                if label:
                    break
        if not label:
            label = CLASSIFICATION_DISPLAY_NAMES.get(value) or _prettify_classification_label(value)
        mapped_wage_key = normalize_classification_name(str(decision.get("mapped_wage_key") or "")) or None
        options.append(
            {
                "value": value,
                "label": label,
                "wage_available": bool(mapped_wage_key),
                "wage_key": mapped_wage_key,
                "source": "ontology_resolved" if mapped_wage_key else "ontology_unresolved",
            }
        )

    # Stable first-seen dedupe.
    seen = set()
    deduped: list[dict] = []
    for opt in options:
        if opt["value"] in seen:
            continue
        seen.add(opt["value"])
        deduped.append(opt)
    return deduped


def get_classification_options(
    contract_id: Optional[str] = None,
    include_unmapped: bool = False,
) -> list[dict]:
    """
    Get classification options for onboarding/profile UI.

    Resolution order:
    1) contract-specific wage classifications (when contract_id provided)
    2) static legacy map fallback
    """
    def _option_rank(opt: dict) -> tuple[int, int]:
        source_rank = {
            "role_catalog": 5,
            "wage_table": 4,
            "ontology_resolved": 3,
            "legacy": 2,
            "ontology_unresolved": 1,
        }
        return (
            1 if bool(opt.get("wage_available")) else 0,
            source_rank.get(str(opt.get("source") or ""), 0),
        )

    def _merge(existing: dict, candidate: dict) -> dict:
        ex = dict(existing or {})
        cand = dict(candidate or {})
        if _option_rank(cand) > _option_rank(ex):
            merged = dict(ex)
            merged.update(cand)
            return merged

        if not ex.get("wage_key") and cand.get("wage_key"):
            ex["wage_key"] = cand.get("wage_key")
        if cand.get("wage_available"):
            ex["wage_available"] = True
        if not ex.get("source") and cand.get("source"):
            ex["source"] = cand.get("source")
        if not ex.get("label") and cand.get("label"):
            ex["label"] = cand.get("label")
        return ex

    if contract_id:
        if contract_id in _classification_cache_by_contract:
            cached = _classification_cache_by_contract[contract_id]
            if include_unmapped:
                return cached
            default_cached = [
                o for o in cached
                if bool(o.get("onboarding_default", o.get("wage_available", True)))
            ]
            if default_cached:
                return default_cached
            filtered_cached = [o for o in cached if o.get("wage_available", True)]
            return filtered_cached or cached

        role_catalog_options = _load_contract_classification_options_from_role_catalog(contract_id)
        if role_catalog_options:
            _classification_cache_by_contract[contract_id] = role_catalog_options
            if include_unmapped:
                return role_catalog_options
            default_opts = [
                o for o in role_catalog_options
                if bool(o.get("onboarding_default", o.get("wage_available", True)))
            ]
            if default_opts:
                return default_opts
            filtered = [o for o in role_catalog_options if o.get("wage_available", True)]
            return filtered or role_catalog_options

        options: list[dict] = []
        option_index: dict[str, int] = {}

        def _upsert(opt: dict) -> None:
            value = str(opt.get("value") or "").strip().lower()
            if not value:
                return
            normalized_opt = dict(opt)
            normalized_opt["value"] = value
            if not normalized_opt.get("label"):
                normalized_opt["label"] = _prettify_classification_label(value)
            if value not in option_index:
                option_index[value] = len(options)
                options.append(normalized_opt)
                return
            idx = option_index[value]
            options[idx] = _merge(options[idx], normalized_opt)

        for opt in _load_contract_classification_options_from_wages(contract_id):
            _upsert(opt)

        ontology_options = _load_contract_classification_options_from_ontology(contract_id)
        for opt in ontology_options:
            _upsert(opt)
        contract_id_lower = contract_id.lower()

        # Clerks contracts often have role variants not present in wage tables.
        # Include legacy clerks role map as supplemental options.
        if "clerks" in contract_id_lower:
            for key, label in CLASSIFICATION_DISPLAY_NAMES.items():
                _upsert(
                    {
                        "value": key,
                        "label": label,
                        "wage_available": False,
                        "wage_key": None,
                        "source": "legacy",
                    }
                )

        if options:
            _classification_cache_by_contract[contract_id] = options
            if include_unmapped:
                return options
            filtered = [o for o in options if o.get("wage_available", True)]
            return filtered or options

    return [{"value": key, "label": label} for key, label in CLASSIFICATION_DISPLAY_NAMES.items()]


def resolve_classification_display_name(
    classification: Optional[str],
    contract_id: Optional[str] = None,
) -> Optional[str]:
    """Resolve a human-readable classification label for API responses."""
    if not classification:
        return None

    key = str(classification).strip().lower()
    if not key:
        return None

    if contract_id:
        options = get_classification_options(contract_id=contract_id)
        for opt in options:
            if opt.get("value") == key:
                return opt.get("label")

    return CLASSIFICATION_DISPLAY_NAMES.get(key) or _prettify_classification_label(key)
