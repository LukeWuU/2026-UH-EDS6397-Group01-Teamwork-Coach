"""Advice generation agent."""

from __future__ import annotations

import json
import re
from typing import Any

from agents.state_utils import state_get
from contract import CoachingRecommendation, RetrievedEvidence, TeamworkDiagnosis
from services.llm_service import structured_invoke

# Safe, blame-free default — only used when the model omits the field.
DEFAULT_WHEN_TO_INVOLVE = (
    "If the issue continues after you try a clear team process, ask your "
    "instructor or advisor to help facilitate a check-in without assigning blame."
)

SYSTEM_PROMPT = """You are a peer teamwork and leadership coach for engineering students.
Speak directly to the student as a teammate ("you" / "your teammate").
Never address them as an instructor, TA, or external coach.
Never call the reader "the student" or advise them to "check in with the student."

Create practical, proportionate coaching. The student's stated GOAL is binding:
advice must serve that goal and the redacted reflection. Diagnosis and retrieved
evidence are optional aids — not a script you must follow.

Rules:
- Stay within teamwork/leadership coaching.
- Encourage action without commands (prefer "you could", "one option is").
- No unsupported diagnoses, accusations, or motive claims.
- No overconfidence words like definitely/certainly/obviously.
- No PII (no names, emails, phones, student IDs, URLs that identify a person).
- REQUIRED: what_may_be_happening must be a non-empty observational summary.
- REQUIRED: what_you_could_do_next must include at least 2 concrete next steps.
- REQUIRED: how_you_might_say_it must include at least 1 example phrase in quotes-ready form.
- REQUIRED: why_this_may_help must be non-empty.
- REQUIRED: when_to_involve_someone_else must be non-empty.
- Use a retrieved chunk ONLY if it matches this reflection + goal (same problem,
  same kind of intervention). If a chunk is off-topic, ignore it completely —
  do not stretch it into a paper-shaped plan.
- cited_source_ids / cited_chunk_ids: copy ids EXACTLY from the evidence list,
  and only for chunks you actually used. If matching chunks are listed, cite
  1-2 of them. If NO chunk matches this case, leave both lists empty and still
  complete the coaching fields — do not invent citations or force-cite junk.
- Each cited_chunk_id's source_id must appear in cited_source_ids.
- How-you-might-say-it phrases should sound like peer-to-peer talk, not
  instructor-to-student talk.
- Note when to involve an instructor/advisor without determining fault.
- Never invent scenario details that are not grounded in the reflection.

Return every required coaching field populated. Incomplete drafts are rejected.
"""


_CASE_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "being",
        "could",
        "from",
        "have",
        "into",
        "just",
        "like",
        "make",
        "more",
        "most",
        "need",
        "next",
        "only",
        "other",
        "same",
        "some",
        "such",
        "than",
        "that",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "very",
        "what",
        "when",
        "where",
        "which",
        "while",
        "with",
        "would",
        "your",
        "team",
        "teams",
        "work",
        "help",
        "student",
        "students",
    }
)


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z]{4,}", (text or "").lower())
        if token not in _CASE_STOPWORDS
    }


def chunk_supports_case(
    chunk_text: str,
    *,
    reflection: str,
    student_goal: str | None = None,
) -> bool:
    """True when a chunk shares distinctive vocabulary with this case.

    Soft match: prefer goal overlap, but accept light reflection overlap so
    retrieval misses do not starve citations.
    """
    chunk_tokens = _content_tokens(chunk_text)
    reflection_tokens = _content_tokens(reflection)
    goal_tokens = _content_tokens(student_goal or "")
    case_tokens = reflection_tokens | goal_tokens
    if not chunk_tokens or not case_tokens:
        return False
    case_overlap = chunk_tokens & case_tokens
    if goal_tokens:
        goal_overlap = chunk_tokens & goal_tokens
        return len(goal_overlap) >= 1 or len(case_overlap) >= 2
    return len(case_overlap) >= 1 or (len(case_overlap) / len(case_tokens) >= 0.08)


def on_case_evidence(
    evidence: list[RetrievedEvidence],
    *,
    reflection: str,
    student_goal: str | None = None,
) -> list[RetrievedEvidence]:
    """Retrieved chunks that lexically match this reflection + goal, score desc."""
    matched = [
        item
        for item in evidence
        if item.chunk_id
        and chunk_supports_case(
            item.text or "",
            reflection=reflection,
            student_goal=student_goal,
        )
    ]
    return sorted(matched, key=lambda item: float(item.score or 0.0), reverse=True)


def _claim_tokens(recommendation: CoachingRecommendation) -> set[str]:
    return _content_tokens(
        " ".join(
            [
                recommendation.what_may_be_happening or "",
                recommendation.why_this_may_help or "",
                " ".join(recommendation.what_you_could_do_next or []),
            ]
        )
    )


def _top_scored_evidence(
    evidence: list[RetrievedEvidence],
    *,
    limit: int,
) -> list[RetrievedEvidence]:
    ranked = sorted(
        [item for item in evidence if item.chunk_id],
        key=lambda item: float(item.score or 0.0),
        reverse=True,
    )
    return ranked[: max(1, limit)]


def _best_claim_aligned_evidence(
    recommendation: CoachingRecommendation,
    evidence: list[RetrievedEvidence],
    *,
    limit: int,
) -> list[RetrievedEvidence]:
    """Prefer retrieved chunks that share claim vocabulary; else top by score."""
    claim = _claim_tokens(recommendation)
    scored: list[tuple[int, float, RetrievedEvidence]] = []
    for item in evidence:
        if not item.chunk_id:
            continue
        overlap = len(_content_tokens(item.text or "") & claim) if claim else 0
        scored.append((overlap, float(item.score or 0.0), item))
    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
    aligned = [row[2] for row in scored if row[0] > 0]
    pool = aligned if aligned else [row[2] for row in scored]
    return pool[: max(1, limit)]


def attach_supporting_citations(
    recommendation: CoachingRecommendation,
    evidence: list[RetrievedEvidence],
    *,
    reflection: str,
    student_goal: str | None = None,
    limit: int = 2,
) -> CoachingRecommendation:
    """If the draft omitted cites, attach up to ``limit`` supporting chunks.

    Prefer on-case lexical matches; else claim-aligned retrieved chunks; else
    top retrieved by score so coaching still carries citation gates.
    """
    if recommendation.cited_chunk_ids:
        return recommendation
    if not evidence:
        return recommendation
    matched = on_case_evidence(
        evidence, reflection=reflection, student_goal=student_goal
    )
    if matched:
        chosen = matched[: max(1, limit)]
    else:
        chosen = _best_claim_aligned_evidence(
            recommendation, evidence, limit=limit
        )
    if not chosen:
        return recommendation
    sources: list[str] = []
    for item in chosen:
        if item.source_id and item.source_id not in sources:
            sources.append(item.source_id)
    data = recommendation.model_dump()
    data["cited_chunk_ids"] = [item.chunk_id for item in chosen if item.chunk_id]
    data["cited_source_ids"] = sources
    return CoachingRecommendation.model_validate(data)


def _sanitize_recommendation(
    recommendation: CoachingRecommendation,
    evidence: list[RetrievedEvidence],
    *,
    reflection: str = "",
    student_goal: str | None = None,
) -> CoachingRecommendation:
    """Drop hallucinated or off-case citation ids and trim whitespace.

    Never invent coaching content — thin drafts must fail validation / repair / fallback.
    Off-topic retrieved chunks may be ignored; matching chunks can be attached later.
    """
    data = recommendation.model_dump()
    allowed_sources = {e.source_id for e in evidence if e.source_id}
    allowed_chunks = {e.chunk_id for e in evidence if e.chunk_id}
    chunk_to_source = {e.chunk_id: e.source_id for e in evidence if e.chunk_id}
    chunk_text = {e.chunk_id: (e.text or "") for e in evidence if e.chunk_id}

    cited_chunks = [
        cid for cid in data.get("cited_chunk_ids", []) if cid in allowed_chunks
    ]
    if reflection or student_goal:
        cited_chunks = [
            cid
            for cid in cited_chunks
            if chunk_supports_case(
                chunk_text.get(cid, ""),
                reflection=reflection,
                student_goal=student_goal,
            )
        ]
    cited_sources = [
        sid for sid in data.get("cited_source_ids", []) if sid in allowed_sources
    ]
    matched_sources: list[str] = []
    for cid in cited_chunks:
        sid = chunk_to_source.get(cid)
        if sid and sid not in matched_sources:
            matched_sources.append(sid)
    if cited_chunks:
        cited_sources = matched_sources
    else:
        cited_sources = []

    actions = [a.strip() for a in data.get("what_you_could_do_next", []) if str(a).strip()]
    phrases = [p.strip() for p in data.get("how_you_might_say_it", []) if str(p).strip()]
    watch = [w.strip() for w in data.get("what_to_watch_for", []) if str(w).strip()]

    return CoachingRecommendation(
        what_may_be_happening=(data.get("what_may_be_happening") or "").strip(),
        what_you_could_do_next=actions,
        how_you_might_say_it=phrases,
        why_this_may_help=(data.get("why_this_may_help") or "").strip(),
        what_to_watch_for=watch,
        when_to_involve_someone_else=(
            data.get("when_to_involve_someone_else") or ""
        ).strip(),
        cited_source_ids=cited_sources,
        cited_chunk_ids=cited_chunks,
    )


def structural_gaps(
    recommendation: CoachingRecommendation,
    *,
    require_citations: bool = True,
) -> list[str]:
    """Return human-readable gaps that make a draft fail completeness checks.

    Set ``require_citations=False`` for the no-RAG baseline (same coaching
    completeness bar, but citations must stay empty).
    """
    gaps: list[str] = []
    if not (recommendation.what_may_be_happening or "").strip():
        gaps.append("what_may_be_happening is empty")
    if len([a for a in recommendation.what_you_could_do_next if str(a).strip()]) < 2:
        gaps.append("what_you_could_do_next needs at least 2 items")
    if len([p for p in recommendation.how_you_might_say_it if str(p).strip()]) < 1:
        gaps.append("how_you_might_say_it needs at least 1 phrase")
    if not (recommendation.why_this_may_help or "").strip():
        gaps.append("why_this_may_help is empty")
    if not (recommendation.when_to_involve_someone_else or "").strip():
        gaps.append("when_to_involve_someone_else is empty")
    if require_citations:
        if not recommendation.cited_chunk_ids:
            gaps.append("cited_chunk_ids is empty")
        if not recommendation.cited_source_ids:
            gaps.append("cited_source_ids is empty")
    return gaps


def apply_safe_defaults(recommendation: CoachingRecommendation) -> CoachingRecommendation:
    """Fill only the safe escalation-guidance default when omitted.

    Does not invent actions, phrases, observations, or citations.
    """
    data = recommendation.model_dump()
    when = (data.get("when_to_involve_someone_else") or "").strip()
    if not when:
        data["when_to_involve_someone_else"] = DEFAULT_WHEN_TO_INVOLVE
    return CoachingRecommendation.model_validate(data)


def _build_user_prompt(
    *,
    reflection: str,
    diagnosis: TeamworkDiagnosis,
    evidence_payload: list[dict[str, Any]],
    allowed_source_ids: list[str],
    allowed_chunk_ids: list[str],
    student_goal: str | None = None,
    repair_block: str = "",
) -> str:
    goal = (student_goal or getattr(diagnosis, "student_goal", None) or "").strip() or (
        "not provided"
    )
    return f"""
Student goal (BINDING — advice must serve this):
{goal}

Redacted reflection:
{reflection}

Diagnosis JSON (hypotheses only; ignore if it conflicts with the goal/reflection):
{diagnosis.model_dump_json(indent=2)}

Retrieved evidence JSON (use only chunks that match this case; ignore the rest):
{json.dumps(evidence_payload, indent=2)}

ALLOWED cited_source_ids (copy exactly IF you used that chunk; else []):
{json.dumps(allowed_source_ids)}

ALLOWED cited_chunk_ids (copy exactly IF you used that chunk; else []):
{json.dumps(allowed_chunk_ids)}
{repair_block}
Coach the reader as a peer teammate. Do not invent unrelated scenarios or
instructor-facing advice. You MUST populate ALL coaching fields:
- what_may_be_happening (non-empty, observational)
- what_you_could_do_next (array with >= 2 concrete steps for THIS goal)
- how_you_might_say_it (array with >= 1 peer-to-peer example phrase)
- why_this_may_help (non-empty)
- when_to_involve_someone_else (non-empty)
- cited_source_ids / cited_chunk_ids: 1-2 matching ALLOWED ids if any match,
  or both empty if none match — still return complete coaching either way
"""


def advice_agent(state: Any) -> dict[str, Any]:
    diagnosis: TeamworkDiagnosis = state_get(state, "diagnosis_payload")
    evidence: list[RetrievedEvidence] = state_get(state, "retrieved_evidence", []) or []
    reflection = state_get(state, "redacted_input", "") or ""
    student_goal = state_get(state, "student_goal") or getattr(
        diagnosis, "student_goal", None
    )
    repair_notes = state_get(state, "validation_result")

    matched_evidence = on_case_evidence(
        evidence, reflection=reflection, student_goal=student_goal
    )
    prompt_evidence = matched_evidence or evidence
    evidence_payload = [
        {
            "chunk_id": e.chunk_id,
            "source_id": e.source_id,
            "text": e.text[:1200],
            "supported_intervention_tags": e.supported_intervention_tags,
            "limitations": e.limitations,
            "citation_text": e.citation.citation_text if e.citation else "",
        }
        for e in prompt_evidence
    ]
    allowed_source_ids = [e.source_id for e in prompt_evidence if e.source_id]
    allowed_chunk_ids = [e.chunk_id for e in prompt_evidence if e.chunk_id]
    # Always require cites when anything was retrieved — attach_supporting_citations
    # fills on-case matches first, else top-scored retrieved.
    require_citations = bool(evidence)

    repair_block = ""
    if repair_notes and getattr(repair_notes, "repairable", False):
        cite_line = (
            "Cite 1-2 ALLOWED ids (prefer chunks that match this case)."
            if evidence
            else "Leave cited_source_ids and cited_chunk_ids empty."
        )
        repair_block = (
            "\nPrevious draft failed validation for repairable reasons:\n"
            + "\n".join(f"- {r}" for r in repair_notes.reasons)
            + "\nRevise to fix those issues. Keep peer-to-peer voice and serve the "
            "stated student goal. Include non-empty what_may_be_happening, "
            "why_this_may_help, when_to_involve_someone_else, at least two "
            "what_you_could_do_next items, and at least one how_you_might_say_it "
            f"phrase. {cite_line} Remove any PII.\n"
        )

    user_prompt = _build_user_prompt(
        reflection=reflection,
        diagnosis=diagnosis,
        evidence_payload=evidence_payload,
        allowed_source_ids=allowed_source_ids,
        allowed_chunk_ids=allowed_chunk_ids,
        student_goal=student_goal,
        repair_block=repair_block,
    )

    recommendation = structured_invoke(
        CoachingRecommendation, SYSTEM_PROMPT, user_prompt
    )
    recommendation = attach_supporting_citations(
        apply_safe_defaults(
            _sanitize_recommendation(
                recommendation,
                evidence,
                reflection=reflection,
                student_goal=student_goal,
            )
        ),
        evidence,
        reflection=reflection,
        student_goal=student_goal,
    )

    gaps = structural_gaps(recommendation, require_citations=require_citations)
    if gaps:
        retry_block = (
            "\nYour previous draft was incomplete:\n"
            + "\n".join(f"- {g}" for g in gaps)
            + "\nReturn a COMPLETE peer coaching recommendation.\n"
        )
        retry_prompt = _build_user_prompt(
            reflection=reflection,
            diagnosis=diagnosis,
            evidence_payload=evidence_payload,
            allowed_source_ids=allowed_source_ids,
            allowed_chunk_ids=allowed_chunk_ids,
            student_goal=student_goal,
            repair_block=repair_block + retry_block,
        )
        recommendation = attach_supporting_citations(
            apply_safe_defaults(
                _sanitize_recommendation(
                    structured_invoke(
                        CoachingRecommendation, SYSTEM_PROMPT, retry_prompt
                    ),
                    evidence,
                    reflection=reflection,
                    student_goal=student_goal,
                )
            ),
            evidence,
            reflection=reflection,
            student_goal=student_goal,
        )

    return {"draft_recommendation": recommendation}
