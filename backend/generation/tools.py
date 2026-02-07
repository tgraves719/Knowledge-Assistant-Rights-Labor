"""
LLM Tool Definitions for KARL.

Defines tools that the LLM can call during response generation,
enabling access to structured data like wage tables.

Uses Gemini's function calling capabilities.
"""

import json
from typing import Optional, Callable
from dataclasses import dataclass

from backend.user.profile import (
    UserProfile,
    get_user_profile,
    estimate_hours_worked,
    CLASSIFICATION_DISPLAY_NAMES,
)


# =============================================================================
# TOOL DEFINITIONS (for Gemini function calling)
# =============================================================================

WAGE_LOOKUP_TOOL = {
    "name": "lookup_wage",
    "description": (
        "Look up the exact wage rate from Appendix A of the union contract. "
        "Call this whenever the user asks about pay, wages, salary, or hourly rate. "
        "Returns the wage rate, step, and citation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "classification": {
                "type": "string",
                "description": "Job classification (e.g., 'courtesy_clerk', 'all_purpose_clerk')"
            },
            "use_profile": {
                "type": "boolean",
                "description": "If true, use the user's profile to estimate hours. If false, use provided values."
            },
            "hours_worked": {
                "type": "integer",
                "description": "Total hours worked (only if use_profile is false)"
            },
            "months_employed": {
                "type": "integer",
                "description": "Months employed (only if use_profile is false)"
            }
        },
        "required": ["classification"]
    }
}


PROFILE_INFO_TOOL = {
    "name": "get_user_info",
    "description": (
        "Get the user's profile information including their job classification, "
        "hire date, and employment type. Use this to personalize responses."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}


ASK_FOR_INFO_TOOL = {
    "name": "ask_for_info",
    "description": (
        "Ask the user for information needed to answer their question accurately. "
        "Use this when you need their hire date, classification, or hours worked."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "info_needed": {
                "type": "string",
                "enum": ["hire_date", "classification", "hours_worked", "employment_type"],
                "description": "What information is needed"
            },
            "reason": {
                "type": "string",
                "description": "Why this information is needed"
            }
        },
        "required": ["info_needed", "reason"]
    }
}


# All available tools
AVAILABLE_TOOLS = [
    WAGE_LOOKUP_TOOL,
    PROFILE_INFO_TOOL,
    ASK_FOR_INFO_TOOL,
]


# =============================================================================
# TOOL EXECUTION
# =============================================================================

@dataclass
class ToolResult:
    """Result of executing a tool."""
    success: bool
    data: dict
    display_text: str  # Human-readable summary for the LLM to incorporate
    disclaimer: Optional[str] = None


class ToolExecutor:
    """
    Executes tool calls from the LLM.

    Bridges between Gemini's function calling and KARL's backend services.
    """

    def __init__(self, retriever, session_id: str = None):
        self.retriever = retriever
        self.session_id = session_id
        self.profile = get_user_profile(session_id) if session_id else None

    def execute(self, tool_name: str, arguments: dict) -> ToolResult:
        """Execute a tool and return the result."""
        if tool_name == "lookup_wage":
            return self._lookup_wage(arguments)
        elif tool_name == "get_user_info":
            return self._get_user_info(arguments)
        elif tool_name == "ask_for_info":
            return self._ask_for_info(arguments)
        else:
            return ToolResult(
                success=False,
                data={"error": f"Unknown tool: {tool_name}"},
                display_text=f"Error: Unknown tool '{tool_name}'"
            )

    def _lookup_wage(self, arguments: dict) -> ToolResult:
        """Execute wage lookup tool."""
        classification = arguments.get("classification")
        use_profile = arguments.get("use_profile", True)

        if not classification:
            # Try to get from profile
            if self.profile and self.profile.classification:
                classification = self.profile.classification
            else:
                return ToolResult(
                    success=False,
                    data={"error": "No classification provided"},
                    display_text="I need to know your job classification to look up your wage."
                )

        # Get hours - from profile or arguments
        if use_profile and self.profile:
            hours_estimate = estimate_hours_worked(self.profile)
            hours_worked = hours_estimate.estimated_hours if hours_estimate else 0
            months_employed = self.profile.months_employed or 0
            is_estimate = True
            confidence = hours_estimate.confidence if hours_estimate else "low"
        else:
            hours_worked = arguments.get("hours_worked", 0)
            months_employed = arguments.get("months_employed", 0)
            is_estimate = False
            confidence = "exact"

        # Look up wage
        wage_info = self.retriever.lookup_wage(
            classification=classification,
            hours_worked=hours_worked,
            months_employed=months_employed,
        )

        if not wage_info:
            return ToolResult(
                success=False,
                data={"error": f"No wage data for {classification}"},
                display_text=f"I couldn't find wage information for {classification}."
            )

        # Build display text
        display_name = CLASSIFICATION_DISPLAY_NAMES.get(classification, classification)
        rate = wage_info["rate"]
        step = wage_info["step"]
        citation = wage_info["citation"]

        if is_estimate and confidence != "exact":
            display_text = (
                f"Based on your profile, as a {display_name} at {step}, "
                f"your estimated wage is **${rate:.2f}/hour** ({citation})."
            )
            disclaimer = (
                "This is an estimate based on your tenure. Your actual rate depends on "
                "total hours worked. Check your pay stub or Company HR Portal to verify."
            )
        else:
            display_text = (
                f"As a {display_name} at {step}, "
                f"your wage is **${rate:.2f}/hour** ({citation})."
            )
            disclaimer = None

        return ToolResult(
            success=True,
            data={
                "classification": classification,
                "rate": rate,
                "step": step,
                "effective_date": wage_info["effective_date"],
                "citation": citation,
                "is_estimate": is_estimate,
                "hours_worked": hours_worked,
            },
            display_text=display_text,
            disclaimer=disclaimer
        )

    def _get_user_info(self, arguments: dict) -> ToolResult:
        """Get user profile information."""
        if not self.profile:
            return ToolResult(
                success=False,
                data={"error": "No profile available"},
                display_text="I don't have any information about you yet."
            )

        data = self.profile.to_dict()
        display_name = CLASSIFICATION_DISPLAY_NAMES.get(self.profile.classification) if self.profile.classification else None

        parts = []
        if display_name:
            parts.append(f"Classification: {display_name}")
        if self.profile.hire_date:
            parts.append(f"Hire date: {self.profile.hire_date}")
            parts.append(f"Months employed: ~{self.profile.months_employed}")
        if self.profile.employment_type.value != "unknown":
            parts.append(f"Employment type: {self.profile.employment_type.value}")

        display_text = "User profile: " + ", ".join(parts) if parts else "No profile information available."

        return ToolResult(
            success=True,
            data=data,
            display_text=display_text
        )

    def _ask_for_info(self, arguments: dict) -> ToolResult:
        """Generate a request for user information."""
        info_needed = arguments.get("info_needed")
        reason = arguments.get("reason", "to answer your question accurately")

        prompts = {
            "hire_date": "When did you start working at Safeway? (approximate date is fine)",
            "classification": "What is your job classification? (e.g., Courtesy Clerk, All-Purpose Clerk)",
            "hours_worked": "Approximately how many total hours have you worked?",
            "employment_type": "Are you full-time or part-time?",
        }

        display_text = prompts.get(info_needed, f"Could you provide your {info_needed}?")
        display_text += f" I need this {reason}."

        return ToolResult(
            success=True,
            data={"info_needed": info_needed, "reason": reason},
            display_text=display_text
        )


# =============================================================================
# GEMINI TOOL INTEGRATION
# =============================================================================

def get_gemini_tools():
    """Get tool definitions in Gemini format."""
    try:
        from google import genai
        return [
            genai.types.Tool(
                function_declarations=[
                    genai.types.FunctionDeclaration(
                        name=tool["name"],
                        description=tool["description"],
                        parameters=tool["parameters"],
                    )
                    for tool in AVAILABLE_TOOLS
                ]
            )
        ]
    except Exception:
        return None


def format_tool_result_for_llm(result: ToolResult) -> str:
    """Format tool result for inclusion in LLM context."""
    output = result.display_text

    if result.disclaimer:
        output += f"\n\n⚠️ {result.disclaimer}"

    return output
