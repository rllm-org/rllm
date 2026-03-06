"""Tests for Eval and AsyncEval."""

import asyncio

from rllm_telemetry.eval import AsyncEval, Eval, ExperimentResult, _compute_summary
from rllm_telemetry.schemas import ExperimentCaseRecord, ScoreRecord
from rllm_telemetry.scorers import Contains, ExactMatch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIMPLE_DATA = [
    {"input": "Alice", "expected": "Hello, Alice!"},
    {"input": "Bob", "expected": "Hello, Bob!"},
]


def greet(input: str) -> str:
    return f"Hello, {input}!"


async def async_greet(input: str) -> str:
    await asyncio.sleep(0.001)
    return f"Hello, {input}!"


def failing_task(input: str) -> str:
    raise ValueError("boom")


# ---------------------------------------------------------------------------
# Eval (sync)
# ---------------------------------------------------------------------------


class TestEval:
    def test_basic(self):
        result = Eval("test", data=SIMPLE_DATA, task=greet, scores=[ExactMatch()])
        assert isinstance(result, ExperimentResult)
        assert len(result.cases) == 2
        assert result.summary is not None
        assert result.summary.scores["ExactMatch"] == 1.0

    def test_multiple_scorers(self):
        result = Eval(
            "test",
            data=SIMPLE_DATA,
            task=greet,
            scores=[ExactMatch(), Contains()],
        )
        assert "ExactMatch" in result.summary.scores
        assert "Contains" in result.summary.scores

    def test_callable_data(self):
        result = Eval(
            "test",
            data=lambda: SIMPLE_DATA,
            task=greet,
            scores=[ExactMatch()],
        )
        assert len(result.cases) == 2

    def test_limit(self):
        result = Eval(
            "test",
            data=SIMPLE_DATA,
            task=greet,
            scores=[ExactMatch()],
            limit=1,
        )
        assert len(result.cases) == 1

    def test_task_failure_captured(self):
        result = Eval(
            "test",
            data=[{"input": "x", "expected": "y"}],
            task=failing_task,
            scores=[ExactMatch()],
        )
        assert result.cases[0].error is not None
        assert "ValueError" in result.cases[0].error
        assert result.summary.error_count == 1

    def test_scorer_still_runs_on_error(self):
        result = Eval(
            "test",
            data=[{"input": "x", "expected": "y"}],
            task=failing_task,
            scores=[ExactMatch()],
        )
        # Scorer should have run (output=None vs expected="y")
        assert len(result.cases[0].scores) == 1
        assert result.cases[0].scores[0].value == 0.0

    def test_metadata(self):
        result = Eval(
            "test",
            data=SIMPLE_DATA,
            task=greet,
            scores=[ExactMatch()],
            metadata={"model": "test-model"},
        )
        assert result.record.metadata == {"model": "test-model"}

    def test_no_expected(self):
        result = Eval(
            "test",
            data=[{"input": "Alice"}],
            task=greet,
            scores=[ExactMatch()],
        )
        assert len(result.cases) == 1
        assert result.cases[0].expected is None

    def test_with_async_task(self):
        result = Eval(
            "test",
            data=SIMPLE_DATA,
            task=async_greet,
            scores=[ExactMatch()],
        )
        assert result.summary.scores["ExactMatch"] == 1.0

    def test_experiment_record_fields(self):
        result = Eval("my_exp", data=SIMPLE_DATA, task=greet, scores=[ExactMatch()])
        assert result.record.name == "my_exp"
        assert result.record.experiment_id is not None
        assert result.record.created_at > 0
        assert result.record.ended_at > result.record.created_at
        assert result.record.duration_ms >= 0


# ---------------------------------------------------------------------------
# AsyncEval
# ---------------------------------------------------------------------------


class TestAsyncEval:
    def test_basic(self):
        result = asyncio.run(AsyncEval("test", data=SIMPLE_DATA, task=async_greet, scores=[ExactMatch()]))
        assert result.summary.scores["ExactMatch"] == 1.0
        assert len(result.cases) == 2

    def test_sync_task(self):
        result = asyncio.run(AsyncEval("test", data=SIMPLE_DATA, task=greet, scores=[ExactMatch()]))
        assert result.summary.scores["ExactMatch"] == 1.0

    def test_limit(self):
        result = asyncio.run(AsyncEval("test", data=SIMPLE_DATA, task=async_greet, scores=[ExactMatch()], limit=1))
        assert len(result.cases) == 1


# ---------------------------------------------------------------------------
# Exporter integration
# ---------------------------------------------------------------------------


class TestExporterIntegration:
    def test_stdout_backend(self, capsys):
        Eval(
            "test",
            data=[{"input": "x", "expected": "x"}],
            task=lambda x: x,
            scores=[ExactMatch()],
            backend="stdout",
        )
        captured = capsys.readouterr()
        assert '"experiment.start"' in captured.out
        assert '"experiment.case"' in captured.out
        assert '"experiment.end"' in captured.out

    def test_no_exporter(self, capsys):
        Eval(
            "test",
            data=[{"input": "x", "expected": "x"}],
            task=lambda x: x,
            scores=[ExactMatch()],
        )
        captured = capsys.readouterr()
        # Should only have the summary table, no JSON envelopes
        assert '"experiment.start"' not in captured.out
        assert "Experiment: test" in captured.out


# ---------------------------------------------------------------------------
# _compute_summary
# ---------------------------------------------------------------------------


class TestComputeSummary:
    def test_basic(self):
        cases = [
            ExperimentCaseRecord(
                case_id="1",
                experiment_id="e",
                input="x",
                scores=[ScoreRecord(name="s1", value=1.0)],
                duration_ms=10.0,
            ),
            ExperimentCaseRecord(
                case_id="2",
                experiment_id="e",
                input="y",
                scores=[ScoreRecord(name="s1", value=0.0)],
                duration_ms=20.0,
            ),
        ]
        summary = _compute_summary(cases)
        assert summary.scores["s1"] == 0.5
        assert summary.total_cases == 2
        assert summary.error_count == 0
        assert summary.total_duration_ms == 30.0

    def test_with_errors(self):
        cases = [
            ExperimentCaseRecord(
                case_id="1",
                experiment_id="e",
                input="x",
                error="ValueError: boom",
                scores=[],
                duration_ms=5.0,
            ),
        ]
        summary = _compute_summary(cases)
        assert summary.error_count == 1
        assert summary.scores == {}
