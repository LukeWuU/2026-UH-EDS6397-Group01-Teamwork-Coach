"""Evidence and citation validation helpers."""

from __future__ import annotations

import re

from config.safety_policy import (
    MOTIVE_CLAIM_PATTERNS,
    OVERCONFIDENCE_PATTERNS,
    PROHIBITED_ADVICE_PATTERNS,
)
from contract import CoachingRecommendation, RetrievedEvidence, ValidationResult
from guardrails.citation_validation import validate_citations
from guardrails.pii_redaction import contains_pii


def validate_recommendation(
    recommendation: CoachingRecommendation,
    evidence: list[RetrievedEvidence],
    *,
    escalation_required: bool = False,
    reflection: str = "",
    student_goal: str | None = None,
) -> ValidationResult:
    checks: dict[str, bool] = {}
    reasons: list[str] = []

    if escalation_required:
        return ValidationResult(
            safe_to_display=False,
            repairable=False,
            escalation_required=True,
            reasons=["High-risk content requires escalation, not coaching display."],
            checks={"escalation_clear": False},
        )

    checks["has_evidence"] = len(evidence) > 0
    if not checks["has_evidence"]:
        reasons.append("No retrieved evidence available.")

    # Citations required whenever retrieval returned anything. Advice attach
    # prefers on-case chunks, then falls back to top-scored retrieved.
    require_citations = bool(evidence)

    if require_citations:
        citation_checks, citation_reasons = validate_citations(recommendation, evidence)
        checks.update(citation_checks)
        reasons.extend(citation_reasons)
    else:
        checks["citations_present"] = True
        checks["citations_from_retrieved_sources"] = True
        checks["cited_chunks_from_retrieved"] = True
        checks["cited_sources_match_chunks"] = True
        checks["citations_lexically_grounded"] = True

    full_text = _flatten(recommendation)
    checks["no_pii"] = not contains_pii(full_text)
    if not checks["no_pii"]:
        reasons.append("Draft contains possible PII.")

    prohibited = _matches_any(full_text, PROHIBITED_ADVICE_PATTERNS)
    checks["no_prohibited_advice"] = not prohibited
    if prohibited:
        reasons.append("Draft contains prohibited advice patterns.")

    motive = _matches_any(full_text, MOTIVE_CLAIM_PATTERNS)
    checks["no_motive_claims"] = not motive
    if motive:
        reasons.append("Draft contains unsupported motive/character claims.")

    overconfident = _matches_any(full_text, OVERCONFIDENCE_PATTERNS)
    checks["not_overconfident"] = not overconfident
    if overconfident:
        reasons.append("Draft overstates certainty.")

    checks["has_observation"] = bool(
        (recommendation.what_may_be_happening or "").strip()
    )
    if not checks["has_observation"]:
        reasons.append("Draft lacks an observational summary (what_may_be_happening).")

    checks["has_actions"] = len(
        [a for a in recommendation.what_you_could_do_next if str(a).strip()]
    ) >= 2
    if not checks["has_actions"]:
        reasons.append("Draft needs at least two concrete next-step options.")

    checks["has_phrases"] = len(
        [p for p in recommendation.how_you_might_say_it if str(p).strip()]
    ) >= 1
    if not checks["has_phrases"]:
        reasons.append("Draft lacks an example phrase (how_you_might_say_it).")

    checks["has_why"] = bool((recommendation.why_this_may_help or "").strip())
    if not checks["has_why"]:
        reasons.append("Draft lacks why_this_may_help.")

    checks["has_escalation_guidance"] = bool(
        (recommendation.when_to_involve_someone_else or "").strip()
    )
    if not checks["has_escalation_guidance"]:
        reasons.append("Draft lacks when_to_involve_someone_else.")

    # Hard failures vs repairable
    hard_keys = [
        "has_evidence",
        "no_prohibited_advice",
        "no_motive_claims",
    ]
    repairable_keys = [
        "citations_present",
        "citations_from_retrieved_sources",
        "cited_chunks_from_retrieved",
        "cited_sources_match_chunks",
        "citations_lexically_grounded",
        "no_pii",
        "not_overconfident",
        "has_observation",
        "has_actions",
        "has_phrases",
        "has_why",
        "has_escalation_guidance",
    ]

    hard_fail = any(not checks[k] for k in hard_keys if k in checks)
    repairable_fail = any(not checks[k] for k in repairable_keys if k in checks)

    if hard_fail:
        return ValidationResult(
            safe_to_display=False,
            repairable=False,
            escalation_required=False,
            reasons=reasons,
            checks=checks,
        )

    if repairable_fail:
        return ValidationResult(
            safe_to_display=False,
            repairable=True,
            escalation_required=False,
            reasons=reasons,
            checks=checks,
        )

    return ValidationResult(
        safe_to_display=True,
        repairable=False,
        escalation_required=False,
        reasons=[],
        checks=checks,
    )


def _flatten(rec: CoachingRecommendation) -> str:
    parts = [
        rec.what_may_be_happening,
        " ".join(rec.what_you_could_do_next),
        " ".join(rec.how_you_might_say_it),
        rec.why_this_may_help,
        " ".join(rec.what_to_watch_for),
        rec.when_to_involve_someone_else,
    ]
    return "\n".join(parts)


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)
