"""Compare two experiment results side-by-side.

Usage::

    from rllm_telemetry import Eval, ExactMatch, compare

    baseline = Eval("v1", data=..., task=agent_v1, scores=[ExactMatch()])
    candidate = Eval("v2", data=..., task=agent_v2, scores=[ExactMatch()])

    diff = compare(baseline, candidate)
    diff.print_report()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .eval import ExperimentResult
from .schemas import ExperimentCaseRecord


@dataclass
class CaseDiff:
    """Score differences for a single test case between two experiments."""

    input: Any
    expected: Any | None
    baseline_output: Any | None
    candidate_output: Any | None
    score_diffs: dict[str, float] = field(default_factory=dict)
    """Scorer name → (candidate_score - baseline_score). Positive = improvement."""
    baseline_error: str | None = None
    candidate_error: str | None = None


@dataclass
class ComparisonResult:
    """Result of comparing two experiments."""

    baseline_name: str
    candidate_name: str
    score_diffs: dict[str, float] = field(default_factory=dict)
    """Scorer name → mean score change (candidate - baseline)."""
    improvements: int = 0
    regressions: int = 0
    unchanged: int = 0
    case_diffs: list[CaseDiff] = field(default_factory=list)

    def print_report(self) -> None:
        """Print a formatted comparison report."""
        _print_report(self)


def _match_cases(
    baseline: list[ExperimentCaseRecord],
    candidate: list[ExperimentCaseRecord],
) -> list[tuple[ExperimentCaseRecord, ExperimentCaseRecord]]:
    """Match cases between two experiments by input value.

    Falls back to positional matching if inputs don't align.
    """
    # Try matching by input
    cand_by_input: dict[str, ExperimentCaseRecord] = {}
    for case in candidate:
        key = str(case.input)
        cand_by_input[key] = case

    matched: list[tuple[ExperimentCaseRecord, ExperimentCaseRecord]] = []
    unmatched_baseline: list[ExperimentCaseRecord] = []

    for b_case in baseline:
        key = str(b_case.input)
        c_case = cand_by_input.pop(key, None)
        if c_case is not None:
            matched.append((b_case, c_case))
        else:
            unmatched_baseline.append(b_case)

    # Fall back to positional for unmatched
    remaining_candidate = list(cand_by_input.values())
    for b_case, c_case in zip(unmatched_baseline, remaining_candidate, strict=False):
        matched.append((b_case, c_case))

    return matched


def compare(
    baseline: ExperimentResult,
    candidate: ExperimentResult,
) -> ComparisonResult:
    """Compare two experiment results and compute score differences.

    Args:
        baseline: The reference experiment (e.g. current production version).
        candidate: The new experiment to evaluate against the baseline.

    Returns:
        :class:`ComparisonResult` with per-case diffs and aggregate changes.
    """
    pairs = _match_cases(baseline.cases, candidate.cases)

    case_diffs: list[CaseDiff] = []
    score_totals: dict[str, float] = {}
    score_counts: dict[str, int] = {}
    improvements = 0
    regressions = 0
    unchanged = 0

    for b_case, c_case in pairs:
        b_scores = {s.name: s.value for s in b_case.scores}
        c_scores = {s.name: s.value for s in c_case.scores}

        all_scorer_names = list(dict.fromkeys(list(b_scores) + list(c_scores)))
        diffs: dict[str, float] = {}

        case_improved = False
        case_regressed = False

        for scorer_name in all_scorer_names:
            b_val = b_scores.get(scorer_name, 0.0)
            c_val = c_scores.get(scorer_name, 0.0)
            diff = c_val - b_val
            diffs[scorer_name] = diff
            score_totals[scorer_name] = score_totals.get(scorer_name, 0.0) + diff
            score_counts[scorer_name] = score_counts.get(scorer_name, 0) + 1

            if diff > 0:
                case_improved = True
            elif diff < 0:
                case_regressed = True

        if case_regressed:
            regressions += 1
        elif case_improved:
            improvements += 1
        else:
            unchanged += 1

        case_diffs.append(
            CaseDiff(
                input=b_case.input,
                expected=b_case.expected,
                baseline_output=b_case.output,
                candidate_output=c_case.output,
                score_diffs=diffs,
                baseline_error=b_case.error,
                candidate_error=c_case.error,
            )
        )

    avg_diffs = {name: score_totals[name] / score_counts[name] for name in score_totals}

    return ComparisonResult(
        baseline_name=baseline.record.name,
        candidate_name=candidate.record.name,
        score_diffs=avg_diffs,
        improvements=improvements,
        regressions=regressions,
        unchanged=unchanged,
        case_diffs=case_diffs,
    )


def _print_report(result: ComparisonResult) -> None:
    """Print a formatted comparison report."""
    scorer_names = list(result.score_diffs.keys())

    print(f"\n{'=' * 60}")
    print(f"Comparison: {result.baseline_name} → {result.candidate_name}")
    print(f"Improved: {result.improvements} | Regressed: {result.regressions} | Unchanged: {result.unchanged}")
    print(f"{'=' * 60}")

    if not scorer_names:
        print("(no scores to compare)")
        return

    # Header
    input_w = 30
    header = f"{'Input':<{input_w}}"
    for name in scorer_names:
        header += f"  {name:>12}"
    print(header)
    print("-" * len(header))

    # Rows — show diff per case
    for case_diff in result.case_diffs:
        input_str = str(case_diff.input)[:input_w].ljust(input_w)
        row = input_str
        for name in scorer_names:
            diff = case_diff.score_diffs.get(name, 0.0)
            if diff > 0:
                row += f"  {'+' + f'{diff:.3f}':>12}"
            elif diff < 0:
                row += f"  {f'{diff:.3f}':>12}"
            else:
                row += f"  {'=':>12}"
        print(row)

    # Average diffs
    print("-" * len(header))
    avg_row = f"{'Mean delta':<{input_w}}"
    for name in scorer_names:
        diff = result.score_diffs[name]
        if diff > 0:
            avg_row += f"  {'+' + f'{diff:.3f}':>12}"
        elif diff < 0:
            avg_row += f"  {f'{diff:.3f}':>12}"
        else:
            avg_row += f"  {'=':>12}"
    print(avg_row)
    print()
