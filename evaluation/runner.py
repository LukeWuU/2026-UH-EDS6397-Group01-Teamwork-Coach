"""Run golden-set evaluation against coach and baseline systems."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from config.settings import get_settings
from evaluation.baselines import SystemName, invoke_system
from evaluation.metrics import aggregate_results, count_failure_codes, score_case
from evaluation.report import (
    build_pairwise_report,
    write_compare_report,
    write_pairwise_report,
    write_report,
)
from evaluation.schema import (
    CaseResult,
    CompareReport,
    EvalCase,
    EvalCaseFile,
    EvalReport,
    SystemCompareRow,
)
from evaluation.scorecard import build_scorecard, write_scorecard

ROOT = Path(__file__).resolve().parent
DEFAULT_CASES = ROOT / "cases" / "golden_seed.json"
DEFAULT_REPORT_DIR = ROOT / "reports"


def configure_eval_llm() -> str:
    """Log which models eval uses: Ollama for systems under test, judge separately."""
    settings = get_settings()
    judge = (settings.eval_llm_provider or "gemini").strip().lower()
    judge_model = (
        settings.gemini_model if judge == "gemini" else settings.ollama_model
    )
    print(
        f"[eval] systems-under-test: Ollama model={settings.ollama_model} "
        f"host={settings.ollama_host}",
        flush=True,
    )
    print(
        f"[eval] LLM-as-judge: provider={judge} model={judge_model}",
        flush=True,
    )
    return judge



def load_cases(path: Path | str) -> list[EvalCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return EvalCaseFile.model_validate(data).cases


def filter_cases(
    cases: Iterable[EvalCase],
    *,
    suites: Optional[set[str]] = None,
    case_ids: Optional[set[str]] = None,
) -> list[EvalCase]:
    selected: list[EvalCase] = []
    for case in cases:
        if suites and case.suite not in suites:
            continue
        if case_ids and case.case_id not in case_ids:
            continue
        selected.append(case)
    return selected


def run_eval(
    cases: list[EvalCase],
    *,
    system: SystemName = "gated_rag",
    with_rubric: bool = False,
    report_dir: Path | str = DEFAULT_REPORT_DIR,
    write: bool = True,
) -> EvalReport:
    """Execute cases on one system, score, optionally LLM-judge, write report."""
    configure_eval_llm()
    results: list[CaseResult] = []
    total = len(cases)
    print(f"\n[{system}] running {total} case(s)...", flush=True)
    for index, case in enumerate(cases, start=1):
        print(
            f"[{system}] {index}/{total} {case.case_id} [{case.suite}] ...",
            flush=True,
        )
        try:
            observed = invoke_system(case, system)
        except Exception as exc:  # noqa: BLE001
            print(f"[{system}] {index}/{total} EXCEPTION: {exc}", flush=True)
            from evaluation.schema import ObservedRun

            observed = ObservedRun(error=str(exc))
        result = score_case(case, observed, system=system)
        if with_rubric:
            from evaluation.rubric import attach_rubric_scores

            attach_rubric_scores(case, result, system=system)
        status = "ok" if not result.failure_codes and not observed.error else (
            f"error={observed.error}" if observed.error else f"fail={result.failure_codes}"
        )
        print(
            f"[{system}] {index}/{total} done route={observed.route or '—'} "
            f"latency={observed.latency_ms:.0f}ms {status}",
            flush=True,
        )
        results.append(result)

    suite_counts: dict[str, int] = {}
    for case in cases:
        suite_counts[case.suite] = suite_counts.get(case.suite, 0) + 1

    report = EvalReport(
        system=system,
        case_count=len(results),
        suite_counts=suite_counts,
        aggregates=aggregate_results(results),
        failure_code_counts=count_failure_codes(results),
        cases=results,
    )
    if write:
        write_report(report, Path(report_dir), prefix=f"eval_{system}")
        if system == "gated_rag":
            write_scorecard(build_scorecard(report), Path(report_dir))
    return report


def run_compare(
    cases: list[EvalCase],
    *,
    with_rubric: bool = False,
    report_dir: Path | str = DEFAULT_REPORT_DIR,
) -> CompareReport:
    """Run gated RAG vs no-RAG baseline on the same cases."""
    print(
        f"\nCompare mode: {len(cases)} case(s) × 2 systems "
        "(gated_rag first, then LLM-only). This can take a long time.",
        flush=True,
    )
    gated = run_eval(
        cases,
        system="gated_rag",
        with_rubric=with_rubric,
        report_dir=report_dir,
        write=True,
    )
    baseline = run_eval(
        cases,
        system="no_rag",
        with_rubric=with_rubric,
        report_dir=report_dir,
        write=True,
    )
    return _write_compare_artifacts(
        gated, baseline, cases=cases, report_dir=Path(report_dir)
    )


def load_latest_report(report_dir: Path, system: str) -> EvalReport | None:
    """Load ``latest_{system}.json``; gated_rag also falls back to ``latest.json``."""
    path = report_dir / f"latest_{system}.json"
    if not path.exists() and system == "gated_rag":
        path = report_dir / "latest.json"
    if not path.exists():
        return None
    return EvalReport.model_validate(json.loads(path.read_text(encoding="utf-8")))


def run_rubric_on_reports(
    cases: list[EvalCase],
    *,
    report_dir: Path | str = DEFAULT_REPORT_DIR,
    suites: Optional[set[str]] = None,
    case_ids: Optional[set[str]] = None,
    systems: tuple[str, ...] = ("gated_rag", "no_rag"),
) -> list[EvalReport]:
    """Apply LLM rubric to coaching answers already saved in report JSON.

    Does not re-invoke the coach. Rebuilds compare + scorecard when both
    gated_rag and no_rag reports are available after judging.

    Defaults to judging the coaching suite only when ``suites`` is None
    (PRD advice-quality focus). Pass an explicit suites set to override.
    """
    from evaluation.rubric import attach_rubric_scores
    from evaluation.schema import ExpectedOutcome

    configure_eval_llm()
    out = Path(report_dir)
    case_map = {c.case_id: c for c in cases}
    updated: list[EvalReport] = []
    # Prefer coaching-suite absolute scores unless caller widens the filter.
    effective_suites = {"coaching"} if suites is None else suites

    for system in systems:
        report = load_latest_report(out, system)
        if report is None:
            print(f"[rubric] skip {system}: no latest report under {out}", flush=True)
            continue

        eligible = [
            result
            for result in report.cases
            if result.observed.route == "coaching"
            and not result.observed.error
            and result.suite in effective_suites
            and (not case_ids or result.case_id in case_ids)
        ]
        total = len(eligible)
        print(
            f"\n[rubric] {system}: judging {total} coaching answer(s) "
            f"(suites={sorted(effective_suites)}; "
            f"of {report.case_count} saved case(s))...",
            flush=True,
        )
        by_id = {r.case_id: r for r in report.cases}
        for index, result in enumerate(eligible, start=1):
            case = case_map.get(result.case_id)
            if case is None:
                reflection = (result.observed.redacted_input or "").strip()
                if not reflection:
                    print(
                        f"[rubric] {system} {index}/{total} {result.case_id} "
                        "SKIPPED: case not in golden set and no redacted_input",
                        flush=True,
                    )
                    continue
                case = EvalCase(
                    case_id=result.case_id,
                    suite=result.suite,
                    reflection=reflection,
                    expected=ExpectedOutcome(route="coaching"),
                )
            print(
                f"[rubric] {system} {index}/{total} {result.case_id} ...",
                flush=True,
            )
            attach_rubric_scores(case, result, system=system)
            dims = [
                f"{k}={v}"
                for k, v in result.rubric.items()
                if isinstance(v, (int, float))
            ]
            status = (
                f"error={result.rubric.get('error')}"
                if result.rubric.get("error")
                else (", ".join(dims) if dims else "no scores")
            )
            print(
                f"[rubric] {system} {index}/{total} done {status}",
                flush=True,
            )
            by_id[result.case_id] = result

        report.cases = [by_id[r.case_id] for r in report.cases]
        report.aggregates = aggregate_results(report.cases)
        report.failure_code_counts = count_failure_codes(report.cases)
        write_report(report, out, prefix=f"eval_{system}")
        updated.append(report)

    gated = next((r for r in updated if r.system == "gated_rag"), None)
    baseline = next((r for r in updated if r.system == "no_rag"), None)
    if gated is None:
        gated = load_latest_report(out, "gated_rag")
    if baseline is None:
        baseline = load_latest_report(out, "no_rag")

    if gated is not None and baseline is not None:
        _write_compare_artifacts(
            gated,
            baseline,
            cases=cases,
            report_dir=out,
        )
    elif gated is not None:
        write_scorecard(build_scorecard(gated), out)

    return updated


def _write_compare_artifacts(
    gated: EvalReport,
    baseline: EvalReport,
    *,
    cases: list[EvalCase],
    report_dir: Path,
) -> CompareReport:
    compare = CompareReport(
        case_count=gated.case_count,
        suite_counts=dict(gated.suite_counts),
        rows=_compare_rows(gated, baseline),
        gated_rag=gated,
        no_rag=baseline,
    )
    write_compare_report(compare, report_dir)
    write_pairwise_report(
        build_pairwise_report(
            gated,
            baseline,
            case_inputs={c.case_id: c for c in cases},
        ),
        report_dir,
    )
    write_scorecard(build_scorecard(gated, compare=compare), report_dir)
    return compare


def _compare_rows(gated: EvalReport, baseline: EvalReport) -> list[SystemCompareRow]:
    """Compare only advice-quality metrics (fair LLM-only baseline)."""
    from evaluation.metrics import ADVICE_QUALITY_METRICS

    gated_map = {a.name: a for a in gated.aggregates}
    base_map = {a.name: a for a in baseline.aggregates}
    names = sorted(
        name
        for name in (set(gated_map) | set(base_map))
        if name in ADVICE_QUALITY_METRICS or name.startswith("rubric_")
    )
    rows: list[SystemCompareRow] = []
    for name in names:
        g = gated_map.get(name)
        b = base_map.get(name)
        g_rate = g.pass_rate if g else None
        b_rate = b.pass_rate if b else None
        delta = None
        if g_rate is not None and b_rate is not None:
            delta = g_rate - b_rate
        rows.append(
            SystemCompareRow(
                metric=name,
                gated_rag_mean=g.mean if g else None,
                gated_rag_pass_rate=g_rate,
                no_rag_mean=b.mean if b else None,
                no_rag_pass_rate=b_rate,
                delta_pass_rate=delta,
            )
        )
    return rows
