"""Serialize evaluation and comparison reports to JSON and Markdown."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional

from evaluation.schema import (
    CaseResult,
    CompareReport,
    EvalCase,
    EvalReport,
    ObservedRun,
    PairwiseCase,
    PairwiseReport,
    PairwiseSide,
)

_ADVICE_FAILURE_CODES = frozenset({"weak_actionability", "forbidden_phrase"})


def write_report(
    report: EvalReport,
    report_dir: Path,
    *,
    prefix: str = "eval_report",
) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = report_dir / f"{prefix}_{stamp}.json"
    md_path = report_dir / f"{prefix}_{stamp}.md"
    latest_json = report_dir / f"latest_{report.system}.json"
    latest_md = report_dir / f"latest_{report.system}.md"
    # Keep generic latest aliases for the default gated system.
    payload = report.model_dump()
    json_text = json.dumps(payload, indent=2)
    md_text = render_markdown(report)

    json_path.write_text(json_text, encoding="utf-8")
    md_path.write_text(md_text, encoding="utf-8")
    latest_json.write_text(json_text, encoding="utf-8")
    latest_md.write_text(md_text, encoding="utf-8")
    if report.system == "gated_rag":
        (report_dir / "latest.json").write_text(json_text, encoding="utf-8")
        (report_dir / "latest.md").write_text(md_text, encoding="utf-8")
    return json_path, md_path


def write_compare_report(
    report: CompareReport, report_dir: Path
) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = report_dir / f"compare_gated_vs_norag_{stamp}.json"
    md_path = report_dir / f"compare_gated_vs_norag_{stamp}.md"
    latest_json = report_dir / "latest_compare.json"
    latest_md = report_dir / "latest_compare.md"

    # Avoid nesting full case dumps twice in the compare JSON blob size explosion:
    # keep aggregates + rows; attach slim case failure summaries.
    payload = {
        "version": report.version,
        "case_count": report.case_count,
        "suite_counts": report.suite_counts,
        "rows": [r.model_dump() for r in report.rows],
        "gated_rag": {
            "system": report.gated_rag.system,
            "case_count": report.gated_rag.case_count,
            "aggregates": [a.model_dump() for a in report.gated_rag.aggregates],
            "failure_code_counts": report.gated_rag.failure_code_counts,
        },
        "no_rag": {
            "system": report.no_rag.system,
            "case_count": report.no_rag.case_count,
            "aggregates": [
                a.model_dump()
                for a in report.no_rag.aggregates
                if a.name in {"actionability", "forbidden_phrase_free"}
                or a.name.startswith("rubric_")
            ],
            "failure_code_counts": {
                k: v
                for k, v in report.no_rag.failure_code_counts.items()
                if k in _ADVICE_FAILURE_CODES
            },
        },
    }
    json_text = json.dumps(payload, indent=2)
    md_text = render_compare_markdown(report)

    json_path.write_text(json_text, encoding="utf-8")
    md_path.write_text(md_text, encoding="utf-8")
    latest_json.write_text(json_text, encoding="utf-8")
    latest_md.write_text(md_text, encoding="utf-8")
    return json_path, md_path


def render_markdown(report: EvalReport) -> str:
    lines = [
        f"# Teamwork Coach Evaluation Report (`{report.system}`)",
        "",
        f"- System: **{report.system}**",
        f"- Cases: **{report.case_count}**",
        f"- Suites: {report.suite_counts}",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | n | mean | pass_rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for agg in report.aggregates:
        mean = f"{agg.mean:.3f}" if agg.mean is not None else "—"
        rate = f"{agg.pass_rate:.3f}" if agg.pass_rate is not None else "—"
        lines.append(f"| `{agg.name}` | {agg.n} | {mean} | {rate} |")

    lines.extend(["", "## Failure codes", ""])
    if report.failure_code_counts:
        for code, count in report.failure_code_counts.items():
            lines.append(f"- `{code}`: {count}")
    else:
        lines.append("- (none)")

    lines.extend(["", "## Per-case summary", ""])
    for case in report.cases:
        failed = ", ".join(case.failure_codes) if case.failure_codes else "ok"
        route = case.observed.route or "error"
        latency = f"{case.observed.latency_ms:.0f}ms"
        err = f" error={case.observed.error}" if case.observed.error else ""
        lines.append(
            f"- `{case.case_id}` [{case.suite}] route={route} "
            f"failures=[{failed}] latency={latency}{err}"
        )

    lines.append("")
    return "\n".join(lines)


def render_compare_markdown(report: CompareReport) -> str:
    lines = [
        "# Advice Quality: Gated Coach vs LLM-only",
        "",
        f"- Cases: **{report.case_count}**",
        f"- Suites: {report.suite_counts}",
        "",
        "This comparison is limited to **advice-quality** metrics "
        "(`actionability`, `forbidden_phrase_free`, and optional `rubric_*`).",
        "Retrieval, citation, and gate metrics are scored on the gated product path only.",
        "",
        "Positive `Δ pass` means the gated coach scored higher on that advice metric.",
        "",
        "| Metric | gated pass | LLM-only pass | Δ pass | gated mean | LLM-only mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.rows:
        g_rate = f"{row.gated_rag_pass_rate:.3f}" if row.gated_rag_pass_rate is not None else "—"
        b_rate = f"{row.no_rag_pass_rate:.3f}" if row.no_rag_pass_rate is not None else "—"
        delta = f"{row.delta_pass_rate:+.3f}" if row.delta_pass_rate is not None else "—"
        g_mean = f"{row.gated_rag_mean:.3f}" if row.gated_rag_mean is not None else "—"
        b_mean = f"{row.no_rag_mean:.3f}" if row.no_rag_mean is not None else "—"
        lines.append(
            f"| `{row.metric}` | {g_rate} | {b_rate} | {delta} | {g_mean} | {b_mean} |"
        )

    lines.extend(
        [
            "",
            "## Advice-quality failure codes",
            "",
            "### gated_rag",
        ]
    )
    if report.gated_rag.failure_code_counts:
        for code, count in report.gated_rag.failure_code_counts.items():
            lines.append(f"- `{code}`: {count}")
    else:
        lines.append("- (none)")

    lines.extend(["", "### LLM-only (no_rag)"])
    advice_failures = {
        k: v
        for k, v in report.no_rag.failure_code_counts.items()
        if k in _ADVICE_FAILURE_CODES
    }
    if advice_failures:
        for code, count in advice_failures.items():
            lines.append(f"- `{code}`: {count}")
    else:
        lines.append("- (none)")

    lines.append("")
    return "\n".join(lines)


def _facing_text(observed: ObservedRun) -> str:
    text = (observed.student_facing_text or "").strip()
    if text:
        return text
    parts = [p for p in (observed.title, observed.body) if p]
    return "\n".join(parts).strip()


def _side_from_case(result: CaseResult | None) -> PairwiseSide:
    if result is None:
        return PairwiseSide(error="missing case in report")
    obs = result.observed
    return PairwiseSide(
        route=obs.route,
        title=obs.title,
        body=obs.body,
        student_facing_text=_facing_text(obs),
        failure_codes=list(result.failure_codes),
        error=obs.error,
        latency_ms=obs.latency_ms,
    )


def build_pairwise_report(
    gated: EvalReport,
    no_rag: EvalReport,
    *,
    case_inputs: Mapping[str, EvalCase] | None = None,
) -> PairwiseReport:
    """Pair gated_rag vs no_rag case results for human side-by-side review.

    ``case_inputs`` supplies reflection / student_goal (not stored on CaseResult).
    When omitted, falls back to ``observed.redacted_input`` when present.
    """
    gated_map = {c.case_id: c for c in gated.cases}
    no_rag_map = {c.case_id: c for c in no_rag.cases}
    case_ids = sorted(set(gated_map) | set(no_rag_map))
    inputs = case_inputs or {}

    pairs: list[PairwiseCase] = []
    suite_counts: dict[str, int] = {}
    for case_id in case_ids:
        gated_case = gated_map.get(case_id)
        no_rag_case = no_rag_map.get(case_id)
        source = gated_case or no_rag_case
        assert source is not None
        suite = source.suite
        suite_counts[suite] = suite_counts.get(suite, 0) + 1

        eval_case = inputs.get(case_id)
        reflection = ""
        goal: Optional[str] = None
        if eval_case is not None:
            reflection = eval_case.reflection
            goal = eval_case.student_goal
        else:
            for side in (gated_case, no_rag_case):
                if side and side.observed.redacted_input:
                    reflection = side.observed.redacted_input
                    break

        pairs.append(
            PairwiseCase(
                case_id=case_id,
                suite=suite,
                tags=list(source.tags),
                reflection=reflection,
                student_goal=goal,
                gated_rag=_side_from_case(gated_case),
                no_rag=_side_from_case(no_rag_case),
            )
        )

    return PairwiseReport(
        case_count=len(pairs),
        suite_counts=suite_counts,
        cases=pairs,
    )


def write_pairwise_report(
    report: PairwiseReport, report_dir: Path
) -> tuple[Path, Path]:
    """Write stamped + latest pairwise JSON/Markdown under report_dir."""
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = report_dir / f"pairwise_gated_vs_norag_{stamp}.json"
    md_path = report_dir / f"pairwise_gated_vs_norag_{stamp}.md"
    latest_json = report_dir / "latest_pairwise.json"
    latest_md = report_dir / "latest_pairwise.md"

    json_text = json.dumps(report.model_dump(), indent=2)
    md_text = render_pairwise_markdown(report)

    json_path.write_text(json_text, encoding="utf-8")
    md_path.write_text(md_text, encoding="utf-8")
    latest_json.write_text(json_text, encoding="utf-8")
    latest_md.write_text(md_text, encoding="utf-8")
    return json_path, md_path


def render_pairwise_markdown(report: PairwiseReport) -> str:
    lines = [
        "# Side-by-side: Gated Coach vs LLM-only",
        "",
        f"- Cases: **{report.case_count}**",
        f"- Suites: {report.suite_counts}",
        "",
        "Paired full responses for human review. Aggregate advice-quality deltas "
        "remain in `latest_compare.md`.",
        "",
    ]
    for case in report.cases:
        lines.extend(_render_pairwise_case(case))
    return "\n".join(lines)


def _render_pairwise_case(case: PairwiseCase) -> list[str]:
    goal = case.student_goal if case.student_goal else "(none)"
    gated_fails = ", ".join(case.gated_rag.failure_codes) or "ok"
    no_rag_fails = ", ".join(case.no_rag.failure_codes) or "ok"
    gated_route = case.gated_rag.route or "—"
    no_rag_route = case.no_rag.route or "—"
    return [
        f"## `{case.case_id}` [{case.suite}]",
        "",
        "### Input",
        "",
        f"**Reflection:** {case.reflection or '(unavailable)'}",
        "",
        f"**Student goal:** {goal}",
        "",
        "| | gated_rag | no_rag (LLM-only) |",
        "| --- | --- | --- |",
        f"| Route | `{gated_route}` | `{no_rag_route}` |",
        f"| Failures | {gated_fails} | {no_rag_fails} |",
        "",
        "### gated_rag response",
        "",
        _fence(case.gated_rag.student_facing_text, case.gated_rag.error),
        "",
        "### no_rag response",
        "",
        _fence(case.no_rag.student_facing_text, case.no_rag.error),
        "",
    ]


def _fence(text: str, error: Optional[str]) -> str:
    if error:
        return f"_Error: {error}_"
    body = (text or "").strip() or "(empty)"
    return f"```text\n{body}\n```"


def build_pairwise_from_reports_dir(
    report_dir: Path,
    *,
    cases_path: Path | None = None,
) -> PairwiseReport:
    """Rebuild pairwise artifact from ``latest_gated_rag.json`` + ``latest_no_rag.json``.

    Loads reflection/goal from the golden case file when available.
    """
    gated_path = report_dir / "latest_gated_rag.json"
    no_rag_path = report_dir / "latest_no_rag.json"
    if not gated_path.exists():
        gated_path = report_dir / "latest.json"
    if not gated_path.exists() or not no_rag_path.exists():
        raise FileNotFoundError(
            f"Need latest_gated_rag.json (or latest.json) and latest_no_rag.json "
            f"under {report_dir} to rebuild pairwise review."
        )

    gated = EvalReport.model_validate(
        json.loads(gated_path.read_text(encoding="utf-8"))
    )
    no_rag = EvalReport.model_validate(
        json.loads(no_rag_path.read_text(encoding="utf-8"))
    )

    case_inputs: dict[str, EvalCase] = {}
    if cases_path is None:
        cases_path = Path(__file__).resolve().parent / "cases" / "golden_seed.json"
    if cases_path.exists():
        from evaluation.runner import load_cases

        case_inputs = {c.case_id: c for c in load_cases(cases_path)}

    return build_pairwise_report(gated, no_rag, case_inputs=case_inputs)

