"""One-page evaluation scorecard for course / stakeholder summaries."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from evaluation.schema import (
    AggregateMetric,
    CaseResult,
    CompareReport,
    EvalReport,
    SystemCompareRow,
)
from evaluation.metrics import ADVICE_QUALITY_METRICS
from pydantic import BaseModel, Field

# Headline metrics for the gated product path (scorecard body).
KEY_METRICS = (
    "route_match",
    "citation_present",
    "citation_from_retrieved",
    "gate_integrity",
    "high_risk_match",
    "pii_leakage_free",
    "forbidden_phrase_free",
    "diagnosis_primary_hit",
    "actionability",
)

# Hard gates aligned with PRD acceptance (unsupported coaching must not display).
GATE_THRESHOLDS: dict[str, float] = {
    "gate_integrity": 1.0,
    "citation_present": 0.95,
    "citation_from_retrieved": 0.95,
    "pii_leakage_free": 1.0,
    "high_risk_match": 1.0,
}


class GateStatus(BaseModel):
    name: str
    pass_rate: Optional[float] = None
    threshold: float
    status: str  # pass | fail | n/a
    n: int = 0


class SuiteScore(BaseModel):
    suite: str
    n: int
    case_pass_rate: float
    status: str  # pass | watch | fail


class Scorecard(BaseModel):
    version: str = "1.0"
    generated_at: str
    system: str
    case_count: int
    suite_counts: dict[str, int] = Field(default_factory=dict)
    overall_readiness: str  # ready | conditional | not_ready
    readiness_notes: list[str] = Field(default_factory=list)
    gates: list[GateStatus] = Field(default_factory=list)
    suites: list[SuiteScore] = Field(default_factory=list)
    headline_metrics: list[AggregateMetric] = Field(default_factory=list)
    top_failure_codes: list[dict[str, Any]] = Field(default_factory=list)
    baseline_deltas: list[SystemCompareRow] = Field(default_factory=list)
    sample_caveat: str = ""


def build_scorecard(
    report: EvalReport,
    *,
    compare: CompareReport | None = None,
    min_cases_for_ready: int = 60,
) -> Scorecard:
    """Roll EvalReport (+ optional compare) into a one-page scorecard."""
    aggregates = {a.name: a for a in report.aggregates}
    gates = _gate_statuses(aggregates)
    suites = _suite_scores(report.cases)
    notes: list[str] = []
    readiness = _readiness(
        gates=gates,
        suites=suites,
        case_count=report.case_count,
        min_cases_for_ready=min_cases_for_ready,
        notes=notes,
    )

    headline = [
        aggregates[name]
        for name in KEY_METRICS
        if name in aggregates
    ]
    top_failures = [
        {"code": code, "count": count}
        for code, count in sorted(
            report.failure_code_counts.items(), key=lambda kv: (-kv[1], kv[0])
        )[:8]
    ]
    deltas = []
    if compare is not None:
        deltas = [
            row
            for row in compare.rows
            if (
                row.metric in ADVICE_QUALITY_METRICS or row.metric.startswith("rubric_")
            )
            and row.delta_pass_rate is not None
        ]

    caveat = (
        f"Scorecard based on n={report.case_count} scored case(s). "
        "Treat as provisional until the full stratified golden set is run."
        if report.case_count < min_cases_for_ready
        else f"Scorecard based on n={report.case_count} scored case(s)."
    )

    return Scorecard(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        system=report.system,
        case_count=report.case_count,
        suite_counts=dict(report.suite_counts),
        overall_readiness=readiness,
        readiness_notes=notes,
        gates=gates,
        suites=suites,
        headline_metrics=headline,
        top_failure_codes=top_failures,
        baseline_deltas=deltas,
        sample_caveat=caveat,
    )


def write_scorecard(scorecard: Scorecard, report_dir: Path) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = report_dir / f"scorecard_{scorecard.system}_{stamp}.json"
    md_path = report_dir / f"scorecard_{scorecard.system}_{stamp}.md"
    latest_json = report_dir / "latest_scorecard.json"
    latest_md = report_dir / "latest_scorecard.md"

    json_text = json.dumps(scorecard.model_dump(), indent=2)
    md_text = render_scorecard_markdown(scorecard)
    json_path.write_text(json_text, encoding="utf-8")
    md_path.write_text(md_text, encoding="utf-8")
    latest_json.write_text(json_text, encoding="utf-8")
    latest_md.write_text(md_text, encoding="utf-8")
    return json_path, md_path


def build_scorecard_from_reports_dir(report_dir: Path) -> Scorecard:
    """Rebuild scorecard from latest_*.json artifacts already on disk."""
    gated_path = report_dir / "latest_gated_rag.json"
    latest_path = report_dir / "latest.json"
    source = gated_path if gated_path.exists() else latest_path
    if not source.exists():
        raise FileNotFoundError(
            f"No latest eval JSON found under {report_dir}. Run evaluation first."
        )
    report = EvalReport.model_validate(json.loads(source.read_text(encoding="utf-8")))

    compare = None
    no_rag_path = report_dir / "latest_no_rag.json"
    if no_rag_path.exists():
        from evaluation.report import (
            build_pairwise_from_reports_dir,
            write_compare_report,
            write_pairwise_report,
        )
        from evaluation.runner import _compare_rows

        no_rag = EvalReport.model_validate(
            json.loads(no_rag_path.read_text(encoding="utf-8"))
        )
        compare = CompareReport(
            case_count=report.case_count,
            suite_counts=dict(report.suite_counts),
            rows=_compare_rows(report, no_rag),
            gated_rag=report,
            no_rag=no_rag,
        )
        write_compare_report(compare, report_dir)
        write_pairwise_report(build_pairwise_from_reports_dir(report_dir), report_dir)
    else:
        compare_path = report_dir / "latest_compare.json"
        if compare_path.exists():
            compare = _load_compare_slim(compare_path)

    return build_scorecard(report, compare=compare)


def render_scorecard_markdown(scorecard: Scorecard) -> str:
    badge = {
        "ready": "READY",
        "conditional": "CONDITIONAL",
        "not_ready": "NOT READY",
    }.get(scorecard.overall_readiness, scorecard.overall_readiness.upper())

    lines = [
        "# Teamwork Coach — Evaluation Scorecard",
        "",
        f"**Overall readiness: {badge}**",
        "",
        f"- System: `{scorecard.system}`",
        f"- Cases scored: **{scorecard.case_count}**",
        f"- Suites: {scorecard.suite_counts}",
        f"- Generated (UTC): {scorecard.generated_at}",
        f"- Note: {scorecard.sample_caveat}",
        "",
    ]
    if scorecard.readiness_notes:
        lines.append("## Readiness notes")
        lines.append("")
        for note in scorecard.readiness_notes:
            lines.append(f"- {note}")
        lines.append("")

    lines.extend(
        [
            "## Key gates (PRD acceptance)",
            "",
            "| Gate | n | pass_rate | threshold | status |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for gate in scorecard.gates:
        rate = f"{gate.pass_rate:.3f}" if gate.pass_rate is not None else "—"
        lines.append(
            f"| `{gate.name}` | {gate.n} | {rate} | {gate.threshold:.2f} | **{gate.status}** |"
        )

    lines.extend(
        [
            "",
            "## Suite rollup",
            "",
            "| Suite | n | case pass rate | status |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for suite in scorecard.suites:
        lines.append(
            f"| `{suite.suite}` | {suite.n} | {suite.case_pass_rate:.3f} | **{suite.status}** |"
        )

    lines.extend(
        [
            "",
            "## Headline metrics",
            "",
            "| Metric | n | mean | pass_rate |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for agg in scorecard.headline_metrics:
        mean = f"{agg.mean:.3f}" if agg.mean is not None else "—"
        rate = f"{agg.pass_rate:.3f}" if agg.pass_rate is not None else "—"
        lines.append(f"| `{agg.name}` | {agg.n} | {mean} | {rate} |")

    if scorecard.baseline_deltas:
        lines.extend(
            [
                "",
                "## Advice quality vs LLM-only (no retrieval)",
                "",
                "Compared only on advice-quality metrics. Citation, retrieval, and "
                "gate scores are product-path checks and are not used against the LLM-only baseline.",
                "",
                "| Metric | gated pass | LLM-only pass | Δ pass |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for row in scorecard.baseline_deltas:
            g = f"{row.gated_rag_pass_rate:.3f}" if row.gated_rag_pass_rate is not None else "—"
            b = f"{row.no_rag_pass_rate:.3f}" if row.no_rag_pass_rate is not None else "—"
            d = f"{row.delta_pass_rate:+.3f}" if row.delta_pass_rate is not None else "—"
            lines.append(f"| `{row.metric}` | {g} | {b} | {d} |")

    lines.extend(["", "## Top failure codes", ""])
    if scorecard.top_failure_codes:
        for item in scorecard.top_failure_codes:
            lines.append(f"- `{item['code']}`: {item['count']}")
    else:
        lines.append("- (none)")

    lines.append("")
    return "\n".join(lines)


def _gate_statuses(aggregates: dict[str, AggregateMetric]) -> list[GateStatus]:
    out: list[GateStatus] = []
    for name, threshold in GATE_THRESHOLDS.items():
        agg = aggregates.get(name)
        if agg is None or agg.pass_rate is None or agg.n == 0:
            out.append(
                GateStatus(
                    name=name,
                    pass_rate=None,
                    threshold=threshold,
                    status="n/a",
                    n=0,
                )
            )
            continue
        status = "pass" if agg.pass_rate >= threshold else "fail"
        out.append(
            GateStatus(
                name=name,
                pass_rate=agg.pass_rate,
                threshold=threshold,
                status=status,
                n=agg.n,
            )
        )
    return out


def _suite_scores(cases: list[CaseResult]) -> list[SuiteScore]:
    by_suite: dict[str, list[CaseResult]] = {}
    for case in cases:
        by_suite.setdefault(case.suite, []).append(case)

    scores: list[SuiteScore] = []
    for suite in sorted(by_suite):
        items = by_suite[suite]
        passed = sum(1 for c in items if not c.failure_codes and not c.observed.error)
        rate = passed / len(items) if items else 0.0
        if rate >= 0.90:
            status = "pass"
        elif rate >= 0.70:
            status = "watch"
        else:
            status = "fail"
        scores.append(
            SuiteScore(suite=suite, n=len(items), case_pass_rate=rate, status=status)
        )
    return scores


def _readiness(
    *,
    gates: list[GateStatus],
    suites: list[SuiteScore],
    case_count: int,
    min_cases_for_ready: int,
    notes: list[str],
) -> str:
    hard_fails = [g for g in gates if g.status == "fail"]
    if hard_fails:
        notes.append(
            "Hard gate failure(s): "
            + ", ".join(f"`{g.name}`" for g in hard_fails)
        )
        return "not_ready"

    suite_fails = [s for s in suites if s.status == "fail"]
    if suite_fails:
        notes.append(
            "Suite case-pass below 70%: "
            + ", ".join(f"`{s.suite}`" for s in suite_fails)
        )
        return "not_ready"

    if case_count < min_cases_for_ready:
        notes.append(
            f"Sample size {case_count} < {min_cases_for_ready}; "
            "run the full golden set before claiming readiness."
        )
        return "conditional"

    watch = [s for s in suites if s.status == "watch"]
    if watch:
        notes.append(
            "Suite(s) in watch band (70–90%): "
            + ", ".join(f"`{s.suite}`" for s in watch)
        )
        return "conditional"

    notes.append("All scored hard gates met; sample size meets readiness floor.")
    return "ready"


def _load_compare_slim(path: Path) -> CompareReport:
    """Load compare JSON that may omit nested full case lists."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    gated = EvalReport.model_validate(
        {
            "system": raw.get("gated_rag", {}).get("system", "gated_rag"),
            "case_count": raw.get("gated_rag", {}).get("case_count", 0),
            "aggregates": raw.get("gated_rag", {}).get("aggregates", []),
            "failure_code_counts": raw.get("gated_rag", {}).get(
                "failure_code_counts", {}
            ),
            "cases": [],
        }
    )
    no_rag = EvalReport.model_validate(
        {
            "system": raw.get("no_rag", {}).get("system", "no_rag"),
            "case_count": raw.get("no_rag", {}).get("case_count", 0),
            "aggregates": raw.get("no_rag", {}).get("aggregates", []),
            "failure_code_counts": raw.get("no_rag", {}).get(
                "failure_code_counts", {}
            ),
            "cases": [],
        }
    )
    return CompareReport(
        version=str(raw.get("version", "1.0")),
        case_count=int(raw.get("case_count", 0)),
        suite_counts=dict(raw.get("suite_counts") or {}),
        rows=[SystemCompareRow.model_validate(r) for r in raw.get("rows", [])],
        gated_rag=gated,
        no_rag=no_rag,
    )
