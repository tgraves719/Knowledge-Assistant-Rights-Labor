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
    contract_id: str = "safeway_pueblo_clerks_2022"  # Default for now
    union_local: str = "UFCW Local 7"
    employer: str = "Safeway"

    # Job info
    classification: Optional[str] = None
    employment_type: EmploymentType = EmploymentType.UNKNOWN

    # Tenure info
    hire_date: Optional[date] = None

    # Cached calculations
    _estimated_hours: Optional[int] = field(default=None, repr=False)

    # Profile completeness
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
            contract_id=data.get("contract_id", "safeway_pueblo_clerks_2022"),
            union_local=data.get("union_local", "UFCW Local 7"),
            employer=data.get("employer", "Safeway"),
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


def update_user_profile(session_id: str, updates: dict) -> UserProfile:
    """Update user profile with new data."""
    profile = get_user_profile(session_id)

    if "classification" in updates:
        profile.classification = updates["classification"]

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

    if "contract_id" in updates:
        profile.contract_id = updates["contract_id"]

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


def get_classification_options() -> list[dict]:
    """Get classification options for onboarding UI."""
    return [
        {"value": key, "label": label}
        for key, label in CLASSIFICATION_DISPLAY_NAMES.items()
    ]
