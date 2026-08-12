"""Citation presence, membership, consistency, and light lexical grounding."""

from __future__ import annotations

import re

from contract import CoachingRecommendation, RetrievedEvidence

# Short / generic tokens that should not count as grounding evidence.
_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "again",
        "being",
        "could",
        "does",
        "each",
        "from",
        "have",
        "into",
        "just",
        "like",
        "make",
        "more",
        "most",
        "much",
        "need",
        "next",
        "only",
        "other",
        "over",
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
        "through",
        "under",
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
        "might",
        "option",
        "options",
        "suggest",
        "propose",
        "clarify",
        "shared",
    }
)


def validate_citations(
    recommendation: CoachingRecommendation,
    evidence: list[RetrievedEvidence],
) -> tuple[dict[str, bool], list[str]]:
    """Return (checks, reasons) for citation-related gates.

    Checks:
    - citations_present: at least one source id and one chunk id
    - citations_from_retrieved_sources: source ids ⊆ retrieved and non-empty
    - cited_chunks_from_retrieved: chunk ids ⊆ retrieved and non-empty
    - cited_sources_match_chunks: every cited chunk's source is among cited sources
    - citations_lexically_grounded: claim text shares content words with cited chunks
    """
    checks: dict[str, bool] = {}
    reasons: list[str] = []

    retrieved_source_ids = {e.source_id for e in evidence if e.source_id}
    retrieved_chunk_ids = {e.chunk_id for e in evidence if e.chunk_id}
    chunk_to_source = {e.chunk_id: e.source_id for e in evidence if e.chunk_id}

    cited_sources = list(recommendation.cited_source_ids or [])
    cited_chunks = list(recommendation.cited_chunk_ids or [])

    checks["citations_present"] = bool(cited_sources) and bool(cited_chunks)
    if not checks["citations_present"]:
        reasons.append(
            "Citations missing: cited_source_ids and cited_chunk_ids must both be non-empty."
        )

    checks["citations_from_retrieved_sources"] = bool(cited_sources) and all(
        sid in retrieved_source_ids for sid in cited_sources
    )
    if cited_sources and not checks["citations_from_retrieved_sources"]:
        reasons.append("Cited source ids are not limited to retrieved sources.")
    elif not cited_sources:
        # Already covered by citations_present; keep check False.
        pass

    checks["cited_chunks_from_retrieved"] = bool(cited_chunks) and all(
        cid in retrieved_chunk_ids for cid in cited_chunks
    )
    if cited_chunks and not checks["cited_chunks_from_retrieved"]:
        reasons.append("Cited chunk ids are not in retrieved evidence.")

    if checks["cited_chunks_from_retrieved"] and checks["citations_from_retrieved_sources"]:
        checks["cited_sources_match_chunks"] = all(
            chunk_to_source.get(cid) in cited_sources for cid in cited_chunks
        )
        if not checks["cited_sources_match_chunks"]:
            reasons.append(
                "Cited source ids do not match the sources of the cited chunks."
            )
    else:
        checks["cited_sources_match_chunks"] = False

    cited_evidence = [e for e in evidence if e.chunk_id in set(cited_chunks)]
    grounded = _lexical_grounding_ok(recommendation, cited_evidence)
    if (
        not grounded
        and checks.get("cited_chunks_from_retrieved")
        and cited_evidence
        and not _any_retrieved_grounds_claim(recommendation, evidence)
    ):
        # Corpus miss: no retrieved chunk overlaps the claim — keep membership cites.
        grounded = True
    checks["citations_lexically_grounded"] = grounded
    if (
        checks["cited_chunks_from_retrieved"]
        and cited_evidence
        and not checks["citations_lexically_grounded"]
    ):
        reasons.append(
            "Cited chunks share too little vocabulary with the coaching claims "
            "(pick chunks that actually support what you wrote)."
        )

    return checks, reasons


def _any_retrieved_grounds_claim(
    recommendation: CoachingRecommendation,
    evidence: list[RetrievedEvidence],
) -> bool:
    return any(
        _lexical_grounding_ok(recommendation, [item])
        for item in evidence
        if item.chunk_id
    )


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z]{4,}", (text or "").lower())
        if token not in _STOPWORDS
    }


def _lexical_grounding_ok(
    recommendation: CoachingRecommendation,
    cited_evidence: list[RetrievedEvidence],
) -> bool:
    """Lightweight claim↔chunk overlap. Not NLI — blocks empty/unrelated cites."""
    if not cited_evidence:
        return False

    claim_text = " ".join(
        [
            recommendation.what_may_be_happening or "",
            recommendation.why_this_may_help or "",
            " ".join(recommendation.what_you_could_do_next or []),
        ]
    )
    claim_tokens = _content_tokens(claim_text)
    if len(claim_tokens) < 4:
        # Too little claim text to judge; presence/membership already enforced.
        return True

    evidence_tokens: set[str] = set()
    for item in cited_evidence:
        evidence_tokens |= _content_tokens(item.text or "")

    if not evidence_tokens:
        return False

    overlap = claim_tokens & evidence_tokens
    # At least two distinctive shared terms, or ~10% of claim content words.
    return len(overlap) >= 2 or (len(overlap) / len(claim_tokens) >= 0.10)
