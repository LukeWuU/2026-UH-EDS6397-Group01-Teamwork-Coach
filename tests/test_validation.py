"""Tests for high-risk detection and validation guardrails."""

from agents.advice_agent import (
    DEFAULT_WHEN_TO_INVOLVE,
    _build_user_prompt,
    _sanitize_recommendation,
    apply_safe_defaults,
    attach_supporting_citations,
    chunk_supports_case,
    structural_gaps,
)
from agents.finalize_node import finalize_coaching_node
from contract import (
    CitationMetadata,
    CoachingRecommendation,
    RetrievedEvidence,
    TeamworkDiagnosis,
)
from guardrails.citation_validation import validate_citations
from guardrails.evidence_validation import validate_recommendation
from guardrails.harmful_advice_validation import detect_high_risk


def test_high_risk_self_harm():
    flagged, hits = detect_high_risk("I want to kill myself after this team conflict.")
    assert flagged is True
    assert hits


def test_ordinary_teamwork_not_high_risk():
    flagged, hits = detect_high_risk(
        "We have unclear roles and missed deadlines on the lab report."
    )
    assert flagged is False
    assert hits == []


def test_shoot_for_grade_is_not_high_risk():
    flagged, hits = detect_high_risk(
        "I want to shoot for an A on the milestone and clarify ownership."
    )
    assert flagged is False
    assert hits == []


def test_illegal_to_ignore_policy_is_not_high_risk():
    flagged, hits = detect_high_risk(
        "It would be illegal to ignore the syllabus late-work policy."
    )
    assert flagged is False
    assert hits == []


def test_illegal_lab_access_is_high_risk():
    flagged, hits = detect_high_risk(
        "Is there an illegal way to access the lab overnight without authorization?"
    )
    assert flagged is True
    assert "illegal" in hits


def test_shoot_threat_is_high_risk():
    flagged, hits = detect_high_risk("I want to shoot my teammate after the meeting.")
    assert flagged is True
    assert "shoot" in hits


def _sample_evidence() -> list[RetrievedEvidence]:
    return [
        RetrievedEvidence(
            chunk_id="chk_accountability_01",
            source_id="src_catme_dimensions",
            text=(
                "Accountability interventions include checkpoints, named owners, "
                "and shared task boards so deliverables are not dropped."
            ),
            score=0.8,
            citation=CitationMetadata(
                source_id="src_catme_dimensions",
                citation_key="catme",
                citation_text="CATME teamwork dimensions.",
            ),
        ),
        RetrievedEvidence(
            chunk_id="chk_psych_safety_01",
            source_id="src_google_rework",
            text=(
                "Psychological safety means teammates can speak up about mistakes "
                "without fear of embarrassment or punishment."
            ),
            score=0.7,
            citation=CitationMetadata(
                source_id="src_google_rework",
                citation_key="rework",
                citation_text="Google re:Work psychological safety.",
            ),
        ),
    ]


def _grounded_recommendation() -> CoachingRecommendation:
    return CoachingRecommendation(
        what_may_be_happening=(
            "Task ownership may be unclear, so deliverables risk being dropped."
        ),
        what_you_could_do_next=[
            "You could propose a shared task board with named owners.",
            "One option is to add midpoint checkpoints before the deadline.",
        ],
        how_you_might_say_it=["Could we assign owners for each deliverable?"],
        why_this_may_help=(
            "Clear ownership and checkpoints often reduce dropped tasks."
        ),
        what_to_watch_for=["Whether checkpoints are kept"],
        when_to_involve_someone_else="If the team cannot agree on a process after trying.",
        cited_source_ids=["src_catme_dimensions"],
        cited_chunk_ids=["chk_accountability_01"],
    )


def test_validation_passes_clean_recommendation():
    result = validate_recommendation(_grounded_recommendation(), _sample_evidence())
    assert result.safe_to_display is True
    assert result.checks.get("citations_present") is True
    assert result.checks.get("citations_lexically_grounded") is True


def test_validation_blocks_motive_and_retaliation():
    rec = CoachingRecommendation(
        what_may_be_happening="They are lazy and do not care.",
        what_you_could_do_next=["Publicly shame them in the group chat."],
        cited_source_ids=["src_catme_dimensions"],
        cited_chunk_ids=["chk_accountability_01"],
    )
    result = validate_recommendation(rec, _sample_evidence())
    assert result.safe_to_display is False
    assert result.repairable is False


def test_validation_marks_missing_citations_repairable():
    rec = CoachingRecommendation(
        what_may_be_happening="Coordination may be difficult.",
        what_you_could_do_next=["You could clarify roles."],
        cited_source_ids=[],
        cited_chunk_ids=[],
    )
    result = validate_recommendation(rec, _sample_evidence())
    assert result.safe_to_display is False
    assert result.repairable is True
    assert result.checks.get("citations_present") is False


def test_validation_requires_chunk_cites_not_sources_alone():
    rec = CoachingRecommendation(
        what_may_be_happening=(
            "Task ownership may be unclear, so deliverables risk being dropped."
        ),
        what_you_could_do_next=[
            "You could propose a shared task board with named owners.",
        ],
        why_this_may_help="Clear ownership and checkpoints often reduce dropped tasks.",
        cited_source_ids=["src_catme_dimensions"],
        cited_chunk_ids=[],
    )
    result = validate_recommendation(rec, _sample_evidence())
    assert result.safe_to_display is False
    assert result.repairable is True
    assert result.checks.get("cited_chunks_from_retrieved") is False


def test_validation_rejects_ungrounded_cites():
    """Citing a retrieved chunk that does not support the claims should fail."""
    rec = CoachingRecommendation(
        what_may_be_happening=(
            "Task ownership may be unclear, so deliverables risk being dropped."
        ),
        what_you_could_do_next=[
            "You could propose a shared task board with named owners.",
            "One option is to add midpoint checkpoints before the deadline.",
        ],
        why_this_may_help=(
            "Clear ownership and checkpoints often reduce dropped tasks."
        ),
        cited_source_ids=["src_google_rework"],
        cited_chunk_ids=["chk_psych_safety_01"],
    )
    checks, reasons = validate_citations(rec, _sample_evidence())
    assert checks["citations_present"] is True
    assert checks["cited_chunks_from_retrieved"] is True
    assert checks["citations_lexically_grounded"] is False
    assert any("vocabulary" in r for r in reasons)

    result = validate_recommendation(rec, _sample_evidence())
    assert result.safe_to_display is False
    assert result.repairable is True


def test_advice_sanitize_does_not_invent_content_or_citations():
    thin = CoachingRecommendation(
        what_may_be_happening="",
        what_you_could_do_next=[],
        how_you_might_say_it=[],
        why_this_may_help="",
        what_to_watch_for=[],
        when_to_involve_someone_else="",
        cited_source_ids=[],
        cited_chunk_ids=[],
    )
    cleaned = _sanitize_recommendation(thin, _sample_evidence())
    assert cleaned.cited_source_ids == []
    assert cleaned.cited_chunk_ids == []
    assert cleaned.what_may_be_happening == ""
    assert cleaned.what_you_could_do_next == []
    assert cleaned.how_you_might_say_it == []
    assert cleaned.why_this_may_help == ""
    assert cleaned.when_to_involve_someone_else == ""


def test_apply_safe_defaults_only_fills_when_to_involve():
    thin = CoachingRecommendation(
        what_may_be_happening="",
        what_you_could_do_next=[],
        how_you_might_say_it=[],
        why_this_may_help="",
        when_to_involve_someone_else="",
        cited_source_ids=[],
        cited_chunk_ids=[],
    )
    filled = apply_safe_defaults(thin)
    assert filled.when_to_involve_someone_else == DEFAULT_WHEN_TO_INVOLVE
    assert filled.what_you_could_do_next == []
    assert filled.how_you_might_say_it == []
    assert filled.what_may_be_happening == ""


def test_structural_gaps_lists_missing_required_fields():
    thin = CoachingRecommendation(
        what_may_be_happening="",
        what_you_could_do_next=["only one"],
        how_you_might_say_it=[],
        why_this_may_help="",
        when_to_involve_someone_else="",
        cited_source_ids=[],
        cited_chunk_ids=[],
    )
    gaps = structural_gaps(apply_safe_defaults(thin))
    assert any("what_may_be_happening" in g for g in gaps)
    assert any("what_you_could_do_next" in g for g in gaps)
    assert any("how_you_might_say_it" in g for g in gaps)
    assert any("why_this_may_help" in g for g in gaps)
    assert not any("when_to_involve_someone_else" in g for g in gaps)


def test_structural_gaps_empty_when_complete():
    rec = CoachingRecommendation(
        what_may_be_happening="Roles are unclear.",
        what_you_could_do_next=["Propose owners.", "Add a checkpoint."],
        how_you_might_say_it=["Could we name owners?"],
        why_this_may_help="Clear ownership helps.",
        when_to_involve_someone_else="If process talks stall.",
        cited_source_ids=["src_catme_dimensions"],
        cited_chunk_ids=["chk_accountability_01"],
    )
    assert structural_gaps(rec) == []


def test_structural_gaps_can_skip_citation_requirement_for_no_rag():
    rec = CoachingRecommendation(
        what_may_be_happening="Roles are unclear.",
        what_you_could_do_next=["Propose owners.", "Add a checkpoint."],
        how_you_might_say_it=["Could we name owners?"],
        why_this_may_help="Clear ownership helps.",
        when_to_involve_someone_else="If process talks stall.",
        cited_source_ids=[],
        cited_chunk_ids=[],
    )
    assert structural_gaps(rec, require_citations=False) == []
    assert any("cited_" in g for g in structural_gaps(rec, require_citations=True))


def test_validation_marks_incomplete_coaching_repairable():
    rec = CoachingRecommendation(
        what_may_be_happening="",
        what_you_could_do_next=["Only one step"],
        how_you_might_say_it=[],
        why_this_may_help="",
        when_to_involve_someone_else="",
        cited_source_ids=["src_catme_dimensions"],
        cited_chunk_ids=["chk_accountability_01"],
    )
    result = validate_recommendation(rec, _sample_evidence())
    assert result.safe_to_display is False
    assert result.repairable is True
    assert result.checks.get("has_observation") is False
    assert result.checks.get("has_actions") is False
    assert result.checks.get("has_phrases") is False
    assert result.checks.get("has_why") is False
    assert result.checks.get("has_escalation_guidance") is False


def test_advice_sanitize_drops_hallucinated_ids_only():
    rec = CoachingRecommendation(
        what_may_be_happening="Roles are fuzzy.",
        what_you_could_do_next=["Propose owners.", "Add a checkpoint."],
        how_you_might_say_it=["Could we name owners?"],
        why_this_may_help="Clear ownership helps.",
        when_to_involve_someone_else="If process talks stall.",
        cited_source_ids=["src_catme_dimensions", "src_fabricated"],
        cited_chunk_ids=["chk_accountability_01", "chk_fake"],
    )
    cleaned = _sanitize_recommendation(rec, _sample_evidence())
    assert cleaned.cited_chunk_ids == ["chk_accountability_01"]
    assert cleaned.cited_source_ids == ["src_catme_dimensions"]
    assert cleaned.what_you_could_do_next == ["Propose owners.", "Add a checkpoint."]


def test_chunk_supports_case_requires_overlap_with_reflection_or_goal():
    accountability = (
        "Accountability interventions include checkpoints, named owners, "
        "and shared task boards so deliverables are not dropped."
    )
    unrelated = (
        "Psychological safety means people can take interpersonal risks "
        "without fear of humiliation when speaking up about mistakes."
    )
    assert chunk_supports_case(
        accountability,
        reflection="Teammate missed FEA checkpoints and named owners slipped.",
        student_goal="Restore follow-through with checkpoints",
    )
    assert not chunk_supports_case(
        unrelated,
        reflection="We keep arguing about how to divide design review prep.",
        student_goal="Reduce process friction before review week",
    )


def test_advice_sanitize_drops_off_case_citations():
    rec = CoachingRecommendation(
        what_may_be_happening="Prep work is split unevenly.",
        what_you_could_do_next=["Propose a shared checklist.", "Name a prep lead."],
        how_you_might_say_it=["Could we use one checklist for review prep?"],
        why_this_may_help="A shared checklist may reduce process friction.",
        when_to_involve_someone_else="If the team cannot agree on a process.",
        cited_source_ids=["src_google_rework"],
        cited_chunk_ids=["chk_psych_safety_01"],
    )
    evidence = _sample_evidence() + [
        RetrievedEvidence(
            chunk_id="chk_psych_safety_01",
            source_id="src_google_rework",
            text=(
                "Psychological safety means people can take interpersonal risks "
                "without fear of humiliation."
            ),
            score=0.7,
            citation=CitationMetadata(
                source_id="src_google_rework",
                citation_key="rework",
                citation_text="Google re:Work.",
            ),
        )
    ]
    cleaned = _sanitize_recommendation(
        rec,
        evidence,
        reflection=(
            "We keep arguing about how to divide the design review prep. "
            "The technical design itself is not the issue."
        ),
        student_goal="Reduce process friction before review week",
    )
    assert cleaned.cited_chunk_ids == []
    assert cleaned.cited_source_ids == []
    assert cleaned.what_you_could_do_next == [
        "Propose a shared checklist.",
        "Name a prep lead.",
    ]


def test_attach_supporting_citations_fills_on_case_chunks():
    rec = CoachingRecommendation(
        what_may_be_happening="Missed FEA checkpoints are visible.",
        what_you_could_do_next=["Ask for a new date.", "Add a shared checkpoint."],
        how_you_might_say_it=["Can we confirm a new FEA date?"],
        why_this_may_help="Named checkpoints restore follow-through.",
        when_to_involve_someone_else="If dates keep slipping.",
        cited_source_ids=[],
        cited_chunk_ids=[],
    )
    filled = attach_supporting_citations(
        rec,
        _sample_evidence(),
        reflection="Teammate missed FEA checkpoints and named owners slipped.",
        student_goal="Restore follow-through with checkpoints",
    )
    assert filled.cited_chunk_ids == ["chk_accountability_01"]
    assert filled.cited_source_ids == ["src_catme_dimensions"]


def test_attach_supporting_citations_falls_back_to_top_retrieved():
    rec = CoachingRecommendation(
        what_may_be_happening="Prep work is split unevenly.",
        what_you_could_do_next=["Propose a shared checklist.", "Name a prep lead."],
        how_you_might_say_it=["Could we use one checklist for review prep?"],
        why_this_may_help="A shared checklist may reduce process friction.",
        when_to_involve_someone_else="If the team cannot agree on a process.",
        cited_source_ids=[],
        cited_chunk_ids=[],
    )
    off_case = [
        RetrievedEvidence(
            chunk_id="chk_psych_safety_01",
            source_id="src_google_rework",
            text=(
                "Psychological safety means people can take interpersonal risks "
                "without fear of humiliation."
            ),
            score=0.9,
            citation=CitationMetadata(
                source_id="src_google_rework",
                citation_key="rework",
                citation_text="Google re:Work.",
            ),
        ),
        RetrievedEvidence(
            chunk_id="chk_other_01",
            source_id="src_other",
            text="Unrelated laboratory safety procedures for chemical handling.",
            score=0.2,
            citation=CitationMetadata(
                source_id="src_other",
                citation_key="other",
                citation_text="Other.",
            ),
        ),
    ]
    filled = attach_supporting_citations(
        rec,
        off_case,
        reflection="We keep arguing about how to divide design review prep.",
        student_goal="Reduce process friction before review week",
    )
    assert filled.cited_chunk_ids[0] == "chk_psych_safety_01"
    assert "chk_psych_safety_01" in filled.cited_chunk_ids
    assert filled.cited_source_ids[0] == "src_google_rework"


def test_validation_requires_cites_when_evidence_exists():
    rec = CoachingRecommendation(
        what_may_be_happening="Prep work is split unevenly.",
        what_you_could_do_next=["Propose a shared checklist.", "Name a prep lead."],
        how_you_might_say_it=["Could we use one checklist for review prep?"],
        why_this_may_help="A shared checklist may reduce process friction.",
        when_to_involve_someone_else="If the team cannot agree on a process.",
        cited_source_ids=[],
        cited_chunk_ids=[],
    )
    off_case = [
        RetrievedEvidence(
            chunk_id="chk_psych_safety_01",
            source_id="src_google_rework",
            text=(
                "Psychological safety means people can take interpersonal risks "
                "without fear of humiliation."
            ),
            score=0.7,
            citation=CitationMetadata(
                source_id="src_google_rework",
                citation_key="rework",
                citation_text="Google re:Work.",
            ),
        )
    ]
    empty = validate_recommendation(
        rec,
        off_case,
        reflection="We keep arguing about how to divide design review prep.",
        student_goal="Reduce process friction before review week",
    )
    assert empty.safe_to_display is False
    assert empty.repairable is True

    filled = attach_supporting_citations(
        rec,
        off_case,
        reflection="We keep arguing about how to divide design review prep.",
        student_goal="Reduce process friction before review week",
    )
    result = validate_recommendation(
        filled,
        off_case,
        reflection="We keep arguing about how to divide design review prep.",
        student_goal="Reduce process friction before review week",
    )
    assert result.safe_to_display is True
    assert filled.cited_chunk_ids == ["chk_psych_safety_01"]


def test_advice_prompt_binds_goal_and_peer_voice():
    diagnosis = TeamworkDiagnosis(
        primary_challenge="decision_making",
        student_goal="Reduce process friction before review week",
        confidence=0.7,
    )
    prompt = _build_user_prompt(
        reflection="The technical design itself is not the issue.",
        diagnosis=diagnosis,
        evidence_payload=[],
        allowed_source_ids=["src_catme_dimensions"],
        allowed_chunk_ids=["chk_accountability_01"],
        student_goal="Reduce process friction before review week",
    )
    assert "BINDING" in prompt
    assert "Reduce process friction before review week" in prompt
    assert "peer teammate" in prompt
    assert "else []" in prompt
    assert "include at least one" not in prompt.lower()


def test_finalize_only_surfaces_cited_chunks():
    state = {
        "draft_recommendation": _grounded_recommendation(),
        "diagnosis_payload": TeamworkDiagnosis(
            primary_challenge="accountability",
            confidence=0.7,
        ),
        "retrieved_evidence": _sample_evidence(),
        "redacted_input": "Roles are unclear.",
        "pii_detected": False,
    }
    out = finalize_coaching_node(state)
    final = out["final_response"]
    assert [e.chunk_id for e in final.supporting_evidence] == ["chk_accountability_01"]
    assert [c.source_id for c in final.citations] == ["src_catme_dimensions"]
    assert (
        '- "Could we assign owners for each deliverable?"' in final.body
    )


def test_finalize_does_not_dump_all_evidence_when_cites_missing():
    rec = CoachingRecommendation(
        what_may_be_happening="Coordination gaps.",
        what_you_could_do_next=["Sync once."],
        cited_source_ids=[],
        cited_chunk_ids=[],
    )
    state = {
        "draft_recommendation": rec,
        "diagnosis_payload": None,
        "retrieved_evidence": _sample_evidence(),
        "redacted_input": "x",
        "pii_detected": False,
    }
    out = finalize_coaching_node(state)
    final = out["final_response"]
    assert final.supporting_evidence == []
    assert final.citations == []
