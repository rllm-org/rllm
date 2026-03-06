"""Experiment evaluation orchestrator.

Usage (sync)::

    from rllm_telemetry import Eval, ExactMatch

    result = Eval(
        "My Experiment",
        data=[{"input": "hello", "expected": "Hello!"}],
        task=lambda input: my_fn(input),
        scores=[ExactMatch()],
    )

Usage (async)::

    from rllm_telemetry import AsyncEval, Contains

    result = await AsyncEval(
        "My Experiment",
        data="eval_data.jsonl",
        task=my_async_agent,
        scores=[Contains()],
    )
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
import uuid
from collections.abc import Callable, Sequence
from typing import Any

from .config import RllmConfig
from .dataset import load_dataset
from .exporter import BaseExporter, create_exporter
from .schemas import (
    ExperimentCaseRecord,
    ExperimentRecord,
    ExperimentSummary,
    ScoreRecord,
)
from .scorers import Scorer

logger = logging.getLogger("rllm_telemetry.eval")


DataFn = Callable[[], list[dict[str, Any]]]
TaskFn = Callable[..., Any]


# ---------------------------------------------------------------------------
# ExperimentResult
# ---------------------------------------------------------------------------


class ExperimentResult:
    """Container for experiment results, returned by :func:`Eval`."""

    def __init__(
        self,
        record: ExperimentRecord,
        cases: list[ExperimentCaseRecord],
    ) -> None:
        self.record = record
        self.cases = cases

    @property
    def summary(self) -> ExperimentSummary | None:
        return self.record.summary

    def print_summary(self) -> None:
        """Print a formatted results table to stdout."""
        _print_summary(self.record, self.cases)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _maybe_await(value: Any) -> Any:
    """Await the value if it's awaitable, otherwise return as-is."""
    if inspect.isawaitable(value):
        return await value
    return value


def _ensure_sync(coro_or_value: Any) -> Any:
    """If the value is awaitable, run it to completion. Otherwise return as-is."""
    if not inspect.isawaitable(coro_or_value):
        return coro_or_value
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro_or_value).result()
    return asyncio.run(coro_or_value)


def _compute_summary(cases: list[ExperimentCaseRecord]) -> ExperimentSummary:
    """Aggregate scores across all cases."""
    score_totals: dict[str, float] = {}
    score_counts: dict[str, int] = {}
    error_count = 0
    total_duration = 0.0

    for case in cases:
        if case.error is not None:
            error_count += 1
        if case.duration_ms is not None:
            total_duration += case.duration_ms
        for score in case.scores:
            score_totals[score.name] = score_totals.get(score.name, 0.0) + score.value
            score_counts[score.name] = score_counts.get(score.name, 0) + 1

    avg_scores = {name: score_totals[name] / score_counts[name] for name in score_totals}

    return ExperimentSummary(
        scores=avg_scores,
        total_cases=len(cases),
        error_count=error_count,
        total_duration_ms=total_duration,
    )


def _resolve_exporter(
    exporter: BaseExporter | None,
    backend: str | None,
) -> BaseExporter | None:
    if exporter is not None:
        return exporter
    if backend is not None:
        config = RllmConfig(backend=backend)
        return create_exporter(config)
    return None


def _resolve_data(
    data: str | DataFn | list[dict[str, Any]],
    split: str | None,
    field_map: dict[str, str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    if isinstance(data, str):
        return load_dataset(data, split=split, field_map=field_map, limit=limit)
    test_cases = data() if callable(data) else data
    if limit is not None:
        test_cases = test_cases[:limit]
    return test_cases


def _print_summary(record: ExperimentRecord, cases: list[ExperimentCaseRecord]) -> None:
    """Print a formatted results table."""
    summary = record.summary
    if summary is None:
        return

    scorer_names = list(summary.scores.keys())

    print(f"\n{'=' * 60}")
    print(f"Experiment: {record.name}")
    print(f"ID: {record.experiment_id}")
    print(f"Cases: {summary.total_cases} | Errors: {summary.error_count} | Duration: {summary.total_duration_ms:.0f}ms")
    print(f"{'=' * 60}")

    if not scorer_names:
        print("(no scorers)")
        return

    # Header
    input_w = 30
    header = f"{'Input':<{input_w}}"
    for name in scorer_names:
        header += f"  {name:>12}"
    print(header)
    print("-" * len(header))

    # Rows
    for case in cases:
        input_str = str(case.input)[:input_w].ljust(input_w)
        row = input_str
        score_map = {s.name: s.value for s in case.scores}
        for name in scorer_names:
            val = score_map.get(name)
            if val is not None:
                row += f"  {val:>12.3f}"
            elif case.error:
                row += f"  {'ERROR':>12}"
            else:
                row += f"  {'-':>12}"
        print(row)

    # Averages
    print("-" * len(header))
    avg_row = f"{'Average':<{input_w}}"
    for name in scorer_names:
        avg_row += f"  {summary.scores[name]:>12.3f}"
    print(avg_row)
    print()


# ---------------------------------------------------------------------------
# Core eval loop (async)
# ---------------------------------------------------------------------------

_EVAL_ARGS_DOC = """
    Args:
        name: Human-readable experiment name.
        data: Test cases. One of:
            - A list of dicts (each with ``"input"`` and optionally ``"expected"``)
            - A callable that returns such a list
            - A file path (``.json``, ``.jsonl``, ``.csv``, ``.parquet``, ``.arrow``)
            - An rllm registered dataset name (e.g. ``"gsm8k"``)
        task: Function that takes an input and returns an output.
            Can be sync or async.
        scores: Sequence of :class:`~rllm_telemetry.scorers.Scorer` instances.
        split: Dataset split (for registry datasets, e.g. ``"test"``).
        field_map: Custom field mapping from source fields to ``"input"``/``"expected"``.
            Defaults to ``{"question": "input", "ground_truth": "expected"}``.
        limit: Max number of test cases to run.
        metadata: Optional metadata to attach to the experiment record.
        exporter: An exporter instance to emit experiment records to.
            Mutually exclusive with ``backend``.
        backend: Shorthand to create an exporter (e.g. ``"stdout"``).
            Mutually exclusive with ``exporter``.

    Returns:
        :class:`ExperimentResult` with the full experiment record and per-case results.
"""


async def _run_eval(
    name: str,
    *,
    data: str | DataFn | list[dict[str, Any]],
    task: TaskFn,
    scores: Sequence[Scorer],
    split: str | None = None,
    field_map: dict[str, str] | None = None,
    limit: int | None = None,
    metadata: dict[str, Any] | None = None,
    exporter: BaseExporter | None = None,
    backend: str | None = None,
) -> ExperimentResult:
    """Shared async implementation for Eval and AsyncEval."""
    exp_exporter = _resolve_exporter(exporter, backend)

    experiment_id = str(uuid.uuid4())
    started_at = time.time()

    test_cases = _resolve_data(data, split, field_map, limit)

    # Emit experiment.start
    record = ExperimentRecord(
        experiment_id=experiment_id,
        name=name,
        created_at=started_at,
        metadata=metadata or {},
    )
    if exp_exporter is not None:
        exp_exporter.enqueue("experiment.start", record.model_dump(exclude_none=True))

    case_records: list[ExperimentCaseRecord] = []

    for test_case in test_cases:
        case_id = str(uuid.uuid4())
        input_val = test_case["input"]
        expected_val = test_case.get("expected")

        case_start = time.time()
        output_val = None
        error_str = None
        case_scores: list[ScoreRecord] = []

        # Run task
        try:
            output_val = await _maybe_await(task(input_val))
        except Exception as exc:
            error_str = f"{type(exc).__name__}: {exc}"
            logger.warning("Task failed for case %s: %s", case_id, error_str)

        # Run scorers (even on error — scorer sees output=None)
        for scorer in scores:
            try:
                score = await _maybe_await(scorer(input=input_val, output=output_val, expected=expected_val))
                case_scores.append(
                    ScoreRecord(
                        name=score.name,
                        value=score.value,
                        metadata=score.metadata,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Scorer %s failed for case %s: %s",
                    scorer.name,
                    case_id,
                    exc,
                )

        case_end = time.time()
        case_record = ExperimentCaseRecord(
            case_id=case_id,
            experiment_id=experiment_id,
            input=input_val,
            expected=expected_val,
            output=output_val,
            scores=case_scores,
            error=error_str,
            duration_ms=(case_end - case_start) * 1000,
        )
        case_records.append(case_record)

        if exp_exporter is not None:
            exp_exporter.enqueue("experiment.case", case_record.model_dump(exclude_none=True))

    ended_at = time.time()
    summary = _compute_summary(case_records)

    record.ended_at = ended_at
    record.duration_ms = (ended_at - started_at) * 1000
    record.summary = summary

    if exp_exporter is not None:
        exp_exporter.enqueue("experiment.end", record.model_dump(exclude_none=True))

    result = ExperimentResult(record=record, cases=case_records)
    result.print_summary()
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def Eval(
    name: str,
    *,
    data: str | DataFn | list[dict[str, Any]],
    task: TaskFn,
    scores: Sequence[Scorer],
    split: str | None = None,
    field_map: dict[str, str] | None = None,
    limit: int | None = None,
    metadata: dict[str, Any] | None = None,
    exporter: BaseExporter | None = None,
    backend: str | None = None,
) -> ExperimentResult:
    """Run an experiment synchronously."""
    coro = _run_eval(
        name,
        data=data,
        task=task,
        scores=scores,
        split=split,
        field_map=field_map,
        limit=limit,
        metadata=metadata,
        exporter=exporter,
        backend=backend,
    )
    return _ensure_sync(coro)


Eval.__doc__ = f"Run an experiment synchronously.\n{_EVAL_ARGS_DOC}"


async def AsyncEval(
    name: str,
    *,
    data: str | DataFn | list[dict[str, Any]],
    task: TaskFn,
    scores: Sequence[Scorer],
    split: str | None = None,
    field_map: dict[str, str] | None = None,
    limit: int | None = None,
    metadata: dict[str, Any] | None = None,
    exporter: BaseExporter | None = None,
    backend: str | None = None,
) -> ExperimentResult:
    """Run an experiment asynchronously."""
    return await _run_eval(
        name,
        data=data,
        task=task,
        scores=scores,
        split=split,
        field_map=field_map,
        limit=limit,
        metadata=metadata,
        exporter=exporter,
        backend=backend,
    )


AsyncEval.__doc__ = f"Run an experiment asynchronously.\n{_EVAL_ARGS_DOC}"
