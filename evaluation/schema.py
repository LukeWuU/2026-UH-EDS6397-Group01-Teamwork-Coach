"""Evaluation case and result contracts for the teamwork coach.

Contract changes should stay aligned with PRD §22 and `contract.py`.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

SuiteName = Literal["coaching", "safety", "privacy", "abstention", "refusal"]
ExpectedRoute = Literal["coaching", "fallback", "escalation"]


class ExpectedOutcome(BaseModel):
    """Gold labels for automatic scoring."""

    route: ExpectedRoute
    acceptable_routes: list[ExpectedRoute] = Field(default_factory=list)
    primary_challenge: Optional[str] = None
    acceptable_primary: list[str] = Field(default_factory=list)
    # Deprecated / unused: corpus is instructor-pluggable, so chunk-id IR gold
    # is not scored. Kept empty for backward-compatible case JSON.
    gold_chunk_ids: list[str] = Field(default_factory=list)
    expect_pii_detected: bool = False
    expect_high_risk: bool = False
    min_actions: int = 0
    must_not_contain: list[str] = Field(default_factory=list)
    notes: str = ""


class EvalCase(BaseModel):
    """One synthetic / de-identified evaluation scenario."""

    case_id: str
    suite: SuiteName
    reflection: str = Field(..., min_length=1)
    student_goal: Optional[str] = None
    expected: ExpectedOutcome
    tags: list[str] = Field(default_factory=list)
    difficulty: Literal["easy", "medium", "hard"] = "medium"


class EvalCaseFile(BaseModel):
    """JSON fixture wrapper."""

    version: str = "1.0"
    description: str = ""
    cases: list[EvalCase]


class ObservedRun(BaseModel):
    """Captured workflow outputs used by scorers."""

    route: Optional[str] = None
    title: str = ""
    body: str = ""
    primary_challenge: Optional[str] = None
    secondary_challenges: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    cited_chunk_ids: list[str] = Field(default_factory=list)
    cited_source_ids: list[str] = Field(default_factory=list)
    action_count: int = 0
    pii_detected: bool = False
    high_risk_detected: bool = False
    retrieval_sufficient: bool = False
    safe_to_display: bool = False
    escalation_required: bool = False
    validation_checks: dict[str, bool] = Field(default_factory=dict)
    redacted_input: str = ""
    student_facing_text: str = ""
    latency_ms: float = 0.0
    error: Optional[str] = None


class MetricScore(BaseModel):
    """Single metric outcome. None means not applicable for this case."""

    name: str
    value: Optional[float] = None
    passed: Optional[bool] = None
    detail: str = ""


class CaseResult(BaseModel):
    case_id: str
    suite: SuiteName
    tags: list[str] = Field(default_factory=list)
    observed: ObservedRun
    metrics: list[MetricScore] = Field(default_factory=list)
    failure_codes: list[str] = Field(default_factory=list)
    rubric: dict[str, Any] = Field(default_factory=dict)


class AggregateMetric(BaseModel):
    name: str
    n: int = 0
    mean: Optional[float] = None
    pass_rate: Optional[float] = None


class EvalReport(BaseModel):
    version: str = "1.0"
    system: str = "gated_rag"
    case_count: int = 0
    suite_counts: dict[str, int] = Field(default_factory=dict)
    aggregates: list[AggregateMetric] = Field(default_factory=list)
    failure_code_counts: dict[str, int] = Field(default_factory=dict)
    cases: list[CaseResult] = Field(default_factory=list)


class SystemCompareRow(BaseModel):
    metric: str
    gated_rag_mean: Optional[float] = None
    gated_rag_pass_rate: Optional[float] = None
    no_rag_mean: Optional[float] = None
    no_rag_pass_rate: Optional[float] = None
    delta_pass_rate: Optional[float] = None


class PairwiseSide(BaseModel):
    """One system's observed response + scoring for a paired case review."""

    route: Optional[str] = None
    title: str = ""
    body: str = ""
    student_facing_text: str = ""
    failure_codes: list[str] = Field(default_factory=list)
    error: Optional[str] = None
    latency_ms: float = 0.0


class PairwiseCase(BaseModel):
    """Side-by-side human review of one eval case across gated_rag vs no_rag."""

    case_id: str
    suite: SuiteName
    tags: list[str] = Field(default_factory=list)
    reflection: str = ""
    student_goal: Optional[str] = None
    gated_rag: PairwiseSide
    no_rag: PairwiseSide


class PairwiseReport(BaseModel):
    """Paired full-text responses for human review (not aggregate metrics)."""

    version: str = "1.0"
    case_count: int = 0
    suite_counts: dict[str, int] = Field(default_factory=dict)
    cases: list[PairwiseCase] = Field(default_factory=list)


class CompareReport(BaseModel):
    version: str = "1.0"
    case_count: int = 0
    suite_counts: dict[str, int] = Field(default_factory=dict)
    rows: list[SystemCompareRow] = Field(default_factory=list)
    gated_rag: EvalReport
    no_rag: EvalReport

