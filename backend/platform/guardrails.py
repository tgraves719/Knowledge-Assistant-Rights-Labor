"""Guardrail pipeline with LLM Guard integration and deterministic fallbacks."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field


PROMPT_INJECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    ("prompt_injection_phrase", r"ignore (all|previous|prior) instructions"),
    ("prompt_injection_phrase", r"(disregard|override).{0,40}(instructions|rules|policy)"),
    ("credential_exfiltration", r"(reveal|print|show|return).{0,40}(api key|password|secret|token)"),
    ("jailbreak_attempt", r"developer mode|system prompt|bypass guard"),
    ("prompt_injection_phrase", r"(reveal|show).{0,40}(system prompt|hidden instructions?)"),
)

PII_PATTERN_DEFINITIONS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("ssn", re.compile(r"\b(\d{3})[- ]?(\d{2})[- ]?(\d{4})\b"), "ssn"),
    ("email", re.compile(r"\b([A-Za-z0-9._%+-])([A-Za-z0-9._%+-]{0,62})@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b"), "email"),
    ("phone", re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(?(\d{3})\)?[-.\s]?)(\d{3})[-.\s]?(\d{4})(?!\d)"), "phone"),
    ("dob", re.compile(r"\b(?:dob|date of birth|birth date)\s*[:#-]?\s*((?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2})\b", re.IGNORECASE), "dob"),
    ("street_address", re.compile(r"\b(?:address|home address|mailing address)\s*[:#-]?\s*(\d{1,6}\s+[A-Za-z0-9.\- ]{3,80}\s+(?:street|st|avenue|ave|road|rd|lane|ln|drive|dr|court|ct|boulevard|blvd|way)\b[^\n,;]*)", re.IGNORECASE), "address"),
    ("routing_number", re.compile(r"\b(?:routing number|routing #)\s*[:#-]?\s*(\d{9})\b", re.IGNORECASE), "routing_number"),
    ("account_number", re.compile(r"\b(?:account number|acct(?:ount)? #?|checking account|savings account)\s*[:#-]?\s*(\d{6,17})\b", re.IGNORECASE), "account_number"),
    ("card_number", re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "card_number"),
    ("member_id", re.compile(r"\b(?:member id|employee id|employee number|member number)\s*[:#-]?\s*([A-Za-z0-9-]{5,24})\b", re.IGNORECASE), "member_id"),
    ("api_key", re.compile(r"\b(?:api[_ -]?key|secret|token|password)\s*(?::|=|\bis\b)\s*([A-Za-z0-9_\-]{6,}|sk-[A-Za-z0-9_-]{8,})\b", re.IGNORECASE), "secret"),
)

PROMPT_INJECTION_DOCUMENT_LABEL = "prompt_injection_risk"


@dataclass
class GuardrailResult:
    allowed: bool
    sanitized_text: str
    reasons: list[str] = field(default_factory=list)
    risk_score: float = 0.0


@dataclass
class DocumentSafetyAssessment:
    safety_status: str
    prompt_injection_risk: bool
    sensitive_data_risk: bool
    member_visible: bool
    safety_reasons: list[str] = field(default_factory=list)
    safety_review_status: str = "not_required"
    recommended_action: str | None = None
    redacted_preview: str = ""


@dataclass
class SafetyFinding:
    category: str
    label: str
    match_preview: str
    context: str


def _mask_ssn(match: re.Match[str]) -> str:
    groups = match.groups()
    if len(groups) >= 3:
        return f"***-**-{groups[2]}"
    digits = re.sub(r"\D", "", match.group(0))
    return f"***-**-{digits[-4:]}" if len(digits) >= 4 else "[redacted ssn]"


def _mask_email(match: re.Match[str]) -> str:
    first, _, domain = match.groups()
    return f"{first}***@{domain}"


def _mask_phone(match: re.Match[str]) -> str:
    groups = match.groups()
    if len(groups) >= 3:
        return f"(***) ***-{groups[2]}"
    digits = re.sub(r"\D", "", match.group(0))
    return f"(***) ***-{digits[-4:]}" if len(digits) >= 4 else "[redacted phone]"


def _mask_labeled_value(label: str) -> str:
    return f"[redacted {label.replace('_', ' ')}]"


def _mask_card_number(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    if len(digits) < 13 or len(digits) > 19:
        return match.group(0)
    return f"[redacted card ending {digits[-4:]}]"


def detect_prompt_injection_markers(text: str) -> list[str]:
    normalized = str(text or "").lower()
    reasons: list[str] = []
    for label, pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, normalized):
            reasons.append(label)
    return sorted(set(reasons))


def _compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _finding_context(text: str, start: int, end: int, *, context_chars: int = 80) -> str:
    snippet = str(text or "")[max(0, start - context_chars): min(len(str(text or "")), end + context_chars)]
    return _compact_whitespace(snippet)


class GuardrailService:
    def __init__(self, token_limit: int = 4000):
        self.token_limit = token_limit
        self.enable_llm_guard = str(
            os.getenv("KARL_ENABLE_LLM_GUARD", "")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.enable_model_scanners = str(
            os.getenv("KARL_ENABLE_MODEL_GUARDRAILS", "")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._input_scanners = []
        self._output_scanners = []
        self._llm_guard_enabled = False
        self._configure_llm_guard()

    def _configure_llm_guard(self) -> None:
        if not self.enable_llm_guard:
            return
        try:
            from llm_guard.input_scanners import InvisibleText, PromptInjection, TokenLimit
            from llm_guard.input_scanners.prompt_injection import MatchType
            from llm_guard.output_scanners import Sensitive
        except Exception:
            return

        scanners = []
        for factory in (
            lambda: TokenLimit(limit=self.token_limit),
            InvisibleText,
        ):
            try:
                scanners.append(factory())
            except Exception:
                continue
        if self.enable_model_scanners:
            try:
                scanners.append(PromptInjection(threshold=0.8, match_type=MatchType.FULL))
            except Exception:
                pass
        self._input_scanners = scanners
        if self.enable_model_scanners:
            try:
                self._output_scanners = [Sensitive()]
            except Exception:
                self._output_scanners = []
        else:
            self._output_scanners = []
        self._llm_guard_enabled = bool(self._input_scanners or self._output_scanners)

    def scan_prompt(self, prompt: str) -> GuardrailResult:
        redaction = self.redact_sensitive_text(prompt)
        sanitized = redaction.sanitized_text
        reasons: list[str] = []
        risk_score = 0.0

        if self._llm_guard_enabled:
            for scanner in self._input_scanners:
                sanitized, is_valid, scanner_risk = scanner.scan(sanitized)
                risk_score = max(risk_score, float(scanner_risk or 0.0))
                if not is_valid:
                    reasons.append(scanner.__class__.__name__)

        for label in detect_prompt_injection_markers(sanitized):
            reasons.append(label)
            risk_score = max(risk_score, 0.95)

        if len(sanitized.split()) > self.token_limit:
            reasons.append("token_limit")
            risk_score = max(risk_score, 1.0)

        return GuardrailResult(
            allowed=not reasons,
            sanitized_text=sanitized,
            reasons=reasons,
            risk_score=risk_score,
        )

    def scan_output(self, response_text: str) -> GuardrailResult:
        redaction = self.redact_sensitive_text(response_text)
        sanitized = redaction.sanitized_text
        reasons: list[str] = list(redaction.reasons)
        risk_score = 0.9 if reasons else 0.0

        for scanner in self._output_scanners:
            sanitized, is_valid, scanner_risk = scanner.scan(sanitized)
            risk_score = max(risk_score, float(scanner_risk or 0.0))
            if not is_valid:
                reasons.append(scanner.__class__.__name__)

        return GuardrailResult(
            allowed=not reasons,
            sanitized_text=sanitized,
            reasons=sorted(set(reasons)),
            risk_score=risk_score,
        )

    def redact_sensitive_text(self, text: str) -> GuardrailResult:
        sanitized = str(text or "")
        reasons: list[str] = []
        risk_score = 0.0

        for name, pattern, label in PII_PATTERN_DEFINITIONS:
            if name == "ssn":
                sanitized, count = pattern.subn(_mask_ssn, sanitized)
            elif name == "email":
                sanitized, count = pattern.subn(_mask_email, sanitized)
            elif name == "phone":
                sanitized, count = pattern.subn(_mask_phone, sanitized)
            elif name == "card_number":
                sanitized, count = pattern.subn(_mask_card_number, sanitized)
            else:
                sanitized, count = pattern.subn(_mask_labeled_value(label), sanitized)
            if count:
                reasons.append(label)
                risk_score = max(risk_score, 0.9)

        return GuardrailResult(
            allowed=True,
            sanitized_text=sanitized,
            reasons=sorted(set(reasons)),
            risk_score=risk_score,
        )

    def assess_document_safety(self, text: str) -> DocumentSafetyAssessment:
        prompt_markers = detect_prompt_injection_markers(text)
        redaction = self.redact_sensitive_text(text)
        prompt_injection_risk = bool(prompt_markers)
        sensitive_data_risk = bool(redaction.reasons)
        safety_reasons = sorted(set(prompt_markers + [f"sensitive_{reason}" for reason in redaction.reasons]))

        if prompt_injection_risk:
            return DocumentSafetyAssessment(
                safety_status="blocked_prompt_injection",
                prompt_injection_risk=True,
                sensitive_data_risk=sensitive_data_risk,
                member_visible=False,
                safety_reasons=safety_reasons,
                safety_review_status="blocked_pending_superadmin",
                recommended_action="Blocked for member use pending superadmin safety review.",
                redacted_preview=redaction.sanitized_text,
            )
        if sensitive_data_risk:
            return DocumentSafetyAssessment(
                safety_status="flagged_sensitive_data",
                prompt_injection_risk=False,
                sensitive_data_risk=True,
                member_visible=True,
                safety_reasons=safety_reasons,
                safety_review_status="needs_review",
                recommended_action="Sensitive data detected. Member excerpts will be redacted and retrieval will be down-ranked until reviewed.",
                redacted_preview=redaction.sanitized_text,
            )
        return DocumentSafetyAssessment(
            safety_status="clear",
            prompt_injection_risk=False,
            sensitive_data_risk=False,
            member_visible=True,
            safety_reasons=[],
            safety_review_status="not_required",
            recommended_action=None,
            redacted_preview=redaction.sanitized_text,
        )

    def review_findings(self, text: str, *, limit: int = 12) -> list[dict]:
        source = str(text or "")
        findings: list[SafetyFinding] = []

        for label, pattern in PROMPT_INJECTION_PATTERNS:
            for match in re.finditer(pattern, source, re.IGNORECASE):
                findings.append(
                    SafetyFinding(
                        category="prompt_injection",
                        label=label,
                        match_preview=_compact_whitespace(match.group(0)),
                        context=_finding_context(source, match.start(), match.end()),
                    )
                )

        for name, pattern, label in PII_PATTERN_DEFINITIONS:
            for match in pattern.finditer(source):
                findings.append(
                    SafetyFinding(
                        category="sensitive_data",
                        label=label,
                        match_preview=_compact_whitespace(match.group(0)),
                        context=_finding_context(source, match.start(), match.end()),
                    )
                )

        deduped: list[dict] = []
        seen: set[tuple[str, str, str]] = set()
        for finding in findings:
            key = (finding.category, finding.label, finding.match_preview)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(
                {
                    "category": finding.category,
                    "label": finding.label,
                    "match_preview": finding.match_preview,
                    "context": finding.context,
                }
            )
            if len(deduped) >= limit:
                break
        return deduped
