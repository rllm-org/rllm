"""Tests for experiment comparison."""

from rllm_telemetry.compare import _match_cases, compare
from rllm_telemetry.eval import Eval
from rllm_telemetry.schemas import ExperimentCaseRecord
from rllm_telemetry.scorers import Contains, ExactMatch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA = [
    {"input": "Alice", "expected": "Hello, Alice!"},
    {"input": "Bob", "expected": "Hello, Bob!"},
    {"input": "Charlie", "expected": "Hello, Charlie!"},
]


# ---------------------------------------------------------------------------
# _match_cases
# ---------------------------------------------------------------------------


class TestMatchCases:
    def test_match_by_input(self):
        b = [
            ExperimentCaseRecord(case_id="1", experiment_id="e", input="Alice", scores=[]),
            ExperimentCaseRecord(case_id="2", experiment_id="e", input="Bob", scores=[]),
        ]
        c = [
            ExperimentCaseRecord(case_id="3", experiment_id="e", input="Bob", scores=[]),
            ExperimentCaseRecord(case_id="4", experiment_id="e", input="Alice", scores=[]),
        ]
        pairs = _match_cases(b, c)
        assert len(pairs) == 2
        assert pairs[0][0].input == "Alice" and pairs[0][1].input == "Alice"
        assert pairs[1][0].input == "Bob" and pairs[1][1].input == "Bob"

    def test_positional_fallback(self):
        b = [
            ExperimentCaseRecord(case_id="1", experiment_id="e", input="A", scores=[]),
        ]
        c = [
            ExperimentCaseRecord(case_id="2", experiment_id="e", input="X", scores=[]),
        ]
        pairs = _match_cases(b, c)
        # No input match, falls back to positional
        assert len(pairs) == 1


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


class TestCompare:
    def test_identical_experiments(self):
        baseline = Eval("v1", data=DATA, task=lambda x: f"Hello, {x}!", scores=[ExactMatch()])
        candidate = Eval("v2", data=DATA, task=lambda x: f"Hello, {x}!", scores=[ExactMatch()])

        diff = compare(baseline, candidate)
        assert diff.improvements == 0
        assert diff.regressions == 0
        assert diff.unchanged == 3
        assert diff.score_diffs["ExactMatch"] == 0.0

    def test_regression(self):
        baseline = Eval("v1", data=DATA, task=lambda x: f"Hello, {x}!", scores=[ExactMatch()])
        candidate = Eval("v2", data=DATA, task=lambda x: f"Hi, {x}!", scores=[ExactMatch()])

        diff = compare(baseline, candidate)
        assert diff.regressions == 3
        assert diff.improvements == 0
        assert diff.score_diffs["ExactMatch"] == -1.0

    def test_improvement(self):
        baseline = Eval("v1", data=DATA, task=lambda x: f"Hi, {x}!", scores=[ExactMatch()])
        candidate = Eval("v2", data=DATA, task=lambda x: f"Hello, {x}!", scores=[ExactMatch()])

        diff = compare(baseline, candidate)
        assert diff.improvements == 3
        assert diff.regressions == 0
        assert diff.score_diffs["ExactMatch"] == 1.0

    def test_mixed_results(self):
        data = [
            {"input": "Alice", "expected": "Hello, Alice!"},
            {"input": "Bob", "expected": "Hi, Bob!"},
        ]
        baseline = Eval("v1", data=data, task=lambda x: f"Hello, {x}!", scores=[ExactMatch()])
        candidate = Eval("v2", data=data, task=lambda x: f"Hi, {x}!", scores=[ExactMatch()])

        diff = compare(baseline, candidate)
        # Alice: 1.0 → 0.0 = regression; Bob: 0.0 → 1.0 = improvement
        assert diff.improvements == 1
        assert diff.regressions == 1
        assert diff.unchanged == 0
        assert diff.score_diffs["ExactMatch"] == 0.0  # net zero

    def test_multiple_scorers(self):
        baseline = Eval("v1", data=DATA, task=lambda x: f"Hello, {x}!", scores=[ExactMatch(), Contains()])
        candidate = Eval("v2", data=DATA, task=lambda x: f"Greetings, {x}!", scores=[ExactMatch(), Contains()])

        diff = compare(baseline, candidate)
        assert "ExactMatch" in diff.score_diffs
        assert "Contains" in diff.score_diffs

    def test_case_diffs_populated(self):
        baseline = Eval("v1", data=DATA[:1], task=lambda x: f"Hello, {x}!", scores=[ExactMatch()])
        candidate = Eval("v2", data=DATA[:1], task=lambda x: f"Hi, {x}!", scores=[ExactMatch()])

        diff = compare(baseline, candidate)
        assert len(diff.case_diffs) == 1
        cd = diff.case_diffs[0]
        assert cd.input == "Alice"
        assert cd.baseline_output == "Hello, Alice!"
        assert cd.candidate_output == "Hi, Alice!"
        assert cd.score_diffs["ExactMatch"] == -1.0

    def test_print_report(self, capsys):
        baseline = Eval("v1", data=DATA[:1], task=lambda x: x, scores=[ExactMatch()])
        candidate = Eval("v2", data=DATA[:1], task=lambda x: x, scores=[ExactMatch()])

        diff = compare(baseline, candidate)
        diff.print_report()
        captured = capsys.readouterr()
        assert "Comparison:" in captured.out
        assert "v1" in captured.out
        assert "v2" in captured.out
