"""User profile and context management for KARL."""

from backend.user.profile import (
    UserProfile,
    EmploymentType,
    HoursEstimate,
    get_user_profile,
    update_user_profile,
    clear_user_profile,
    estimate_hours_worked,
    get_classification_options,
    CLASSIFICATION_DISPLAY_NAMES,
)

__all__ = [
    "UserProfile",
    "EmploymentType",
    "HoursEstimate",
    "get_user_profile",
    "update_user_profile",
    "clear_user_profile",
    "estimate_hours_worked",
    "get_classification_options",
    "CLASSIFICATION_DISPLAY_NAMES",
]
