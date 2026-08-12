"""Validation agent."""

from __future__ import annotations

from typing import Any

from agents.state_utils import state_get
from contract import ValidationResult
from guardrails.evidence_validation import validate_recommendation


def validation_agent(state: Any) -> dict[str, Any]:
    recommendation = state_get(state, "draft_recommendation")
    evidence = state_get(state, "retrieved_evidence", []) or []
    escalation_required = bool(state_get(state, "escalation_required"))

    if recommendation is None:
        result = ValidationResult(
            safe_to_display=False,
            repairable=False,
            escalation_required=escalation_required,
            reasons=["Missing draft recommendation."],
            checks={"has_draft": False},
        )
        return {
            "validation_result": result,
            "safe_to_display": False,
        }

    result = validate_recommendation(
        recommendation,
        evidence,
        escalation_required=escalation_required,
        reflection=str(state_get(state, "redacted_input", "") or ""),
        student_goal=state_get(state, "student_goal"),
    )
    return {
        "validation_result": result,
        "safe_to_display": result.safe_to_display,
        "escalation_required": result.escalation_required or escalation_required,
    }
