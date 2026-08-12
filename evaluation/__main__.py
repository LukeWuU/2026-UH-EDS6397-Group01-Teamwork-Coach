"""CLI: python -m evaluation [--system gated_rag|no_rag|compare|scorecard|rubric]."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evaluation.runner import (
    DEFAULT_CASES,
    DEFAULT_REPORT_DIR,
    filter_cases,
    load_cases,
    run_compare,
    run_eval,
    run_rubric_on_reports,
)
from evaluation.scorecard import build_scorecard_from_reports_dir, write_scorecard


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run golden-set evaluation for the Teamwork & Leadership Coach."
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES,
        help="Path to EvalCaseFile JSON (default: evaluation/cases/golden_seed.json)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Report output directory (default: evaluation/reports)",
    )
    parser.add_argument(
        "--suites",
        type=str,
        default="",
        help="Comma-separated suites to run (coaching,safety,privacy,abstention,refusal)",
    )
    parser.add_argument(
        "--case-ids",
        type=str,
        default="",
        help="Comma-separated case_id filter",
    )
    parser.add_argument(
        "--system",
        choices=("gated_rag", "no_rag", "compare", "scorecard", "rubric"),
        default="gated_rag",
        help=(
            "System under test (default: gated_rag). "
            "Use compare for gated vs no-RAG. "
            "Use scorecard to rebuild the one-page scorecard from latest reports. "
            "Use rubric to LLM-judge coaching-suite answers already saved in "
            "reports (absolute rubric scores + gated vs no-RAG compare; "
            "no coach re-run)."
        ),
    )
    parser.add_argument(
        "--rubric",
        action="store_true",
        help=(
            "Also run optional LLM-as-judge during a live eval (costs tokens). "
            "Prefer --system rubric on saved reports after compare finishes."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate/load cases and print selection without invoking the coach",
    )
    args = parser.parse_args(argv)

    if args.system == "scorecard":
        try:
            scorecard = build_scorecard_from_reports_dir(args.out)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        paths = write_scorecard(scorecard, args.out)
        print(f"Overall readiness: {scorecard.overall_readiness.upper()}")
        print(f"Cases: {scorecard.case_count}")
        for note in scorecard.readiness_notes:
            print(f"  - {note}")
        print(f"Wrote {paths[1]}")
        return 0

    cases = load_cases(args.cases)
    suites = {s.strip() for s in args.suites.split(",") if s.strip()} or None
    case_ids = {s.strip() for s in args.case_ids.split(",") if s.strip()} or None

    if args.system == "rubric":
        if args.dry_run:
            print(
                f"Would judge saved coaching answers under {args.out} "
                f"(suites={suites or 'all'}, case_ids={case_ids or 'all'})"
            )
            return 0
        updated = run_rubric_on_reports(
            cases,
            report_dir=args.out,
            suites=suites,
            case_ids=case_ids,
        )
        if not updated:
            print(
                f"No latest gated_rag / no_rag reports found under {args.out}.",
                file=sys.stderr,
            )
            return 1
        print(f"\nWrote rubric-updated reports under {args.out}")
        for report in updated:
            rubric_aggs = [
                a for a in report.aggregates if a.name.startswith("rubric_")
            ]
            print(f"System: {report.system} (n={report.case_count})")
            if not rubric_aggs:
                print("  (no rubric_* aggregates — no coaching answers judged)")
                continue
            for agg in rubric_aggs:
                mean = f"{agg.mean:.3f}" if agg.mean is not None else "—"
                rate = f"{agg.pass_rate:.3f}" if agg.pass_rate is not None else "—"
                print(f"  {agg.name}: mean={mean} pass_rate={rate} (n={agg.n})")
        print(
            "See evaluation/reports/latest_compare.md and "
            "latest_scorecard.md (when both systems were available)."
        )
        return 0

    selected = filter_cases(cases, suites=suites, case_ids=case_ids)

    if not selected:
        print("No cases selected.", file=sys.stderr)
        return 1

    print(f"Loaded {len(selected)} case(s) from {args.cases}")
    suite_counts: dict[str, int] = {}
    for case in selected:
        suite_counts[case.suite] = suite_counts.get(case.suite, 0) + 1
    print(f"Suite counts: {suite_counts}")
    for case in selected:
        print(f"  - {case.case_id} [{case.suite}]")

    if args.dry_run:
        return 0

    if args.system == "compare":
        compare = run_compare(
            selected, with_rubric=args.rubric, report_dir=args.out
        )
        print(f"\nWrote comparison reports under {args.out}")
        print(f"Cases compared: {compare.case_count}")
        for row in compare.rows:
            if row.delta_pass_rate is None:
                continue
            print(
                f"  {row.metric}: gated={row.gated_rag_pass_rate:.3f} "
                f"no_rag={row.no_rag_pass_rate:.3f} "
                f"delta={row.delta_pass_rate:+.3f}"
            )
        print(
            "See evaluation/reports/latest_compare.md, "
            "latest_pairwise.md, and latest_scorecard.md"
        )
        return 0

    report = run_eval(
        selected,
        system=args.system,
        with_rubric=args.rubric,
        report_dir=args.out,
    )
    print(f"\nWrote reports under {args.out}")
    print(f"System: {report.system}")
    print(f"Cases scored: {report.case_count}")
    for agg in report.aggregates:
        mean = f"{agg.mean:.3f}" if agg.mean is not None else "—"
        rate = f"{agg.pass_rate:.3f}" if agg.pass_rate is not None else "—"
        print(f"  {agg.name}: mean={mean} pass_rate={rate} (n={agg.n})")
    if report.failure_code_counts:
        print("Failure codes:")
        for code, count in report.failure_code_counts.items():
            print(f"  {code}: {count}")
    if args.system == "gated_rag":
        print("See evaluation/reports/latest_scorecard.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
