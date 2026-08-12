"""Optional LLM-as-judge rubric for coaching quality (PRD §3 / §16 / §22).

Kept separate from deterministic metrics so offline CI can skip it.
Calibrate against human ratings before trusting absolute scores.

Absolute scores use structured output + PRD-shaped dimensions. Compare gated
vs no-RAG via ``rubric_*`` aggregate deltas on the scorecard / compare report.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from evaluation.schema import CaseResult, EvalCase, MetricScore, ObservedRun
from evaluation.judge_evidence import JUDGE_EVIDENCE_BASE
from services.llm_service import eval_judge_invoke, get_chat_model
from config.settings import get_settings

RUBRIC_DIMENSIONS = (
    "observation_vs_interpretation",
    "actionability",
    "proportionality",
    "evidence_to_action",
    "scope_fidelity",
    "tone_non_accusatory",
    "calibrated_certainty",
    "student_agency",
)

# Backward-compatible aliases if older reports / prompts used this name.
_DIMENSION_ALIASES = {
    "evidence_alignment": "evidence_to_action",
}

PASS_THRESHOLD = 4  # dim pass if score >= 4/5
WEAK_THRESHOLD = 3  # dim is "weak" if score <= 3


class RubricScores(BaseModel):
    """Structured absolute rubric judgment."""

    observation_vs_interpretation: int = Field(..., ge=1, le=5)
    actionability: int = Field(..., ge=1, le=5)
    proportionality: int = Field(..., ge=1, le=5)
    evidence_to_action: int = Field(..., ge=1, le=5)
    scope_fidelity: int = Field(..., ge=1, le=5)
    tone_non_accusatory: int = Field(..., ge=1, le=5)
    calibrated_certainty: int = Field(..., ge=1, le=5)
    student_agency: int = Field(..., ge=1, le=5)
    evidence_quote: str = Field(
        default="",
        description="Short quote from the coach response supporting the lowest score",
    )
    overall_notes: str = Field(default="", max_length=500)



_SYSTEM_PROMPT = f"""You are an expert evaluator for a University of Houston engineering
teamwork & leadership coach (PRD-aligned).

Score ONLY what the student-facing coach response actually does.
Be critical: most generic safe advice is a 3, not a 5. Reserve 5 for excellent,
PRD-shaped coaching. Use the full 1–5 range.

Anchors (apply to every dimension):
- 5 = excellent, concrete, PRD-aligned
- 4 = good with minor gaps
- 3 = adequate / generic but safe
- 2 = weak / overconfident / poorly scoped
- 1 = clear failure (motive claims, commands, out of scope, fabricated grounding)

Dimension definitions:
- observation_vs_interpretation: Separates observable behavior from assumptions.
  FAIL hard for motive/character claims ("lazy", "does not care", "personality").
- actionability: Specific, feasible next steps a student could try this week.
- proportionality: Advice scaled to severity; not dramatic or punitive.
- evidence_to_action: How well advice aligns with the approved evidence base below
  (CATME dimensions, ABET Meets/Exceeds teamwork criteria, Google re:Work team
  effectiveness, constructive conflict / coordination sources, psychological
  safety & climate sources, and teamwork-intervention literature).
  Apply the SAME standard to gated_rag and no_rag:
  * 5 = concrete next steps clearly consistent with one or more evidence-base
    frameworks (e.g., CATME dimension language, constructive controversy,
    psychological safety practices, coordinated roles/quality expectations).
  * 4 = mostly aligned with evidence-base ideas; minor vagueness.
  * 3 = plausible generic teamwork advice; weak or unclear link to the base.
  * 2 = thin / mismatched to the reflection or weakly related to the base.
  * 1 = fabricated-looking research claims, contradictory to the base, or
    empty platitudes with no actionable grounding.
  Do NOT cap scores at 3 merely because retrieved/cited chunk IDs are absent.
  Product cite IDs are optional context; conceptual alignment with the approved
  evidence base is what matters for this dimension.
- scope_fidelity: Teamwork/leadership coaching only — not legal, clinical,
  disciplinary, misconduct investigator, or instructor surveillance.
- tone_non_accusatory: Respectful, non-shaming, no humiliation.
- calibrated_certainty: Avoids overclaiming ("definitely", blame %, moral verdicts).
  Leaves room for uncertainty when signals are thin.
- student_agency: Encourages action without commanding ("you could/consider" vs
  "you must/tell them they have to").

{JUDGE_EVIDENCE_BASE}

Return structured fields only. Include evidence_quote from the response that
justifies your lowest dimension score.
"""



def judge_coaching_quality(
    case: EvalCase,
    observed: ObservedRun,
    *,
    system: str = "gated_rag",
) -> dict[str, Any]:
    """Return rubric scores for coaching-route cases; empty dict otherwise."""
    if observed.route != "coaching" or observed.error:
        return {}

    user_prompt = _absolute_user_prompt(case, observed, system=system)
    try:
        scored = eval_judge_invoke(RubricScores, _SYSTEM_PROMPT, user_prompt)
        payload = scored.model_dump()
        payload["overall_notes"] = str(payload.get("overall_notes", ""))[:500]
        payload["evidence_quote"] = str(payload.get("evidence_quote", ""))[:400]
        return payload
    except Exception as exc:  # noqa: BLE001
        # Fallback: free-form invoke path may still return parseable JSON.
        settings = get_settings()
        provider = (settings.eval_llm_provider or "gemini").strip().lower()
        model = get_chat_model(settings, provider=provider)
        raw = model.invoke(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
        parsed = _parse_rubric_json(_message_content(raw))
        if any(isinstance(parsed.get(d), (int, float)) for d in RUBRIC_DIMENSIONS):
            return parsed
        return {"error": str(exc), "parse_error": True, "raw": str(parsed.get("raw", ""))[:500]}



def attach_rubric_scores(
    case: EvalCase,
    result: CaseResult,
    *,
    system: str = "gated_rag",
) -> CaseResult:
    """Judge ``result.observed`` and attach ``rubric_*`` metrics (replaces prior rubric)."""
    result.metrics = [m for m in result.metrics if not m.name.startswith("rubric_")]
    try:
        result.rubric = judge_coaching_quality(
            case, result.observed, system=system
        )
        _append_rubric_metrics(result)
    except Exception as exc:  # noqa: BLE001
        result.rubric = {"error": str(exc)}
    return result


def _append_rubric_metrics(result: CaseResult) -> None:
    scores: list[int] = []
    for dim in RUBRIC_DIMENSIONS:
        value = result.rubric.get(dim)
        if isinstance(value, (int, float)):
            score = int(value)
            scores.append(score)
            result.metrics.append(
                MetricScore(
                    name=f"rubric_{dim}",
                    value=float(score) / 5.0,
                    passed=float(score) >= PASS_THRESHOLD,
                    detail=f"rubric={score}/5",
                )
            )
    if not scores:
        return
    weak = sum(1 for s in scores if s <= WEAK_THRESHOLD)
    result.metrics.append(
        MetricScore(
            name="rubric_no_weak_dimension",
            value=1.0 if weak == 0 else 0.0,
            passed=weak == 0,
            detail=f"weak_dims(<=3)={weak}/{len(scores)} min={min(scores)}",
        )
    )
    result.metrics.append(
        MetricScore(
            name="rubric_min_dimension",
            value=float(min(scores)) / 5.0,
            passed=min(scores) >= PASS_THRESHOLD,
            detail=f"min={min(scores)}/5",
        )
    )


def _absolute_user_prompt(
    case: EvalCase, observed: ObservedRun, *, system: str
) -> str:
    response = observed.student_facing_text or observed.body
    evidence_block = _evidence_context(observed, system=system)
    return (
        f"System under test: {system}\n"
        f"Case id: {case.case_id}\n"
        f"Suite: {case.suite}\n\n"
        f"Student reflection:\n{case.reflection}\n\n"
        f"Student goal:\n{case.student_goal or '(none)'}\n\n"
        f"Coach route: {observed.route}\n"
        f"{evidence_block}\n"
        "Score evidence_to_action against the approved evidence base in the "
        "system prompt (same standard for gated_rag and no_rag).\n\n"
        f"Coach response:\n{response}\n"
    )



def _evidence_context(observed: ObservedRun, *, system: str) -> str:
    retrieved = ", ".join(observed.retrieved_chunk_ids) or "(none)"
    cited = ", ".join(observed.cited_chunk_ids) or "(none)"
    sources = ", ".join(observed.cited_source_ids) or "(none)"
    return (
        f"Evidence context ({system}):\n"
        f"- retrieval_sufficient: {observed.retrieval_sufficient}\n"
        f"- retrieved_chunk_ids: {retrieved}\n"
        f"- cited_chunk_ids: {cited}\n"
        f"- cited_source_ids: {sources}\n"
    )


def _message_content(raw: Any) -> str:
    content = getattr(raw, "content", raw)
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text", part)))
            else:
                parts.append(str(part))
        return " ".join(parts)
    return str(content)


def _parse_rubric_json(text: str) -> dict[str, Any]:
    """Best-effort parse for free-form model output (incl. markdown / wrappers)."""
    text = _normalize_model_text(text)
    data = _load_json_object(text)
    if data is None:
        return {"parse_error": True, "raw": text[:500]}

    scores: dict[str, Any] = {}
    for dim in RUBRIC_DIMENSIONS:
        value = data.get(dim)
        if value is None:
            for alias, canonical in _DIMENSION_ALIASES.items():
                if canonical == dim and alias in data:
                    value = data[alias]
                    break
        if isinstance(value, (int, float)):
            scores[dim] = int(max(1, min(5, round(value))))
        elif isinstance(value, str) and value.strip().isdigit():
            scores[dim] = int(max(1, min(5, int(value.strip()))))

    # Regex salvage when JSON wrapper was messy but dims are present in text.
    if len(scores) < len(RUBRIC_DIMENSIONS):
        salvaged = _regex_salvage_scores(text)
        for dim, value in salvaged.items():
            scores.setdefault(dim, value)

    if not scores:
        return {"parse_error": True, "raw": text[:500]}

    scores["overall_notes"] = str(data.get("overall_notes", ""))[:500]
    scores["evidence_quote"] = str(data.get("evidence_quote", ""))[:400]
    return scores


def _normalize_model_text(text: str) -> str:
    text = (text or "").strip()
    # LangChain sometimes stringifies {"type":"text","text":"```json..."}.
    if text.startswith("{") and "'text':" in text and "observation_vs_interpretation" in text:
        match = re.search(
            r"['\"]text['\"]\s*:\s*['\"](.*?)['\"]\s*\}\s*$",
            text,
            flags=re.DOTALL,
        )
        if match:
            text = match.group(1)
        text = (
            text.replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace("\\'", "'")
            .replace('\\"', '"')
        )
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _load_json_object(text: str) -> Optional[dict[str, Any]]:
    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            # Unwrap {"type":"text","text":"{...}"} if needed.
            if (
                "observation_vs_interpretation" not in data
                and isinstance(data.get("text"), str)
                and "observation_vs_interpretation" in data["text"]
            ):
                inner = _load_json_object(_normalize_model_text(data["text"]))
                if inner:
                    return inner
            return data
    return None


def _regex_salvage_scores(text: str) -> dict[str, int]:
    scores: dict[str, int] = {}
    for dim in RUBRIC_DIMENSIONS:
        match = re.search(
            rf'"{dim}"\s*:\s*(\d)',
            text,
        )
        if match:
            scores[dim] = int(max(1, min(5, int(match.group(1)))))
    # Alias salvage
    match = re.search(r'"evidence_alignment"\s*:\s*(\d)', text)
    if match and "evidence_to_action" not in scores:
        scores["evidence_to_action"] = int(max(1, min(5, int(match.group(1)))))
    return scores
