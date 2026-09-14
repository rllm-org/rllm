"""Eval results: data classes for storing and reporting evaluation outcomes."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from rllm import paths


@dataclass
class EvalItem:
    """Result for a single rollout. ``idx`` is the task index; with multiple
    attempts per task (pass@k) sibling rollouts share an ``idx`` and are told
    apart by ``attempt``."""

    idx: int
    reward: float
    is_correct: bool
    error: str | None = None
    signals: dict[str, float] = field(default_factory=dict)
    # Provider-reported solver usage for this rollout. Judge usage is
    # intentionally excluded: this describes the model being evaluated.
    metrics: dict[str, Any] = field(default_factory=dict)
    attempt: int = 0
    # Coarse outcome bucket (TerminationReason.value), e.g. "env_done", "timeout",
    # "verifier_timeout", "grading_error". Lets the report break failures down by
    # cause — agent timeouts vs infra/grading errors vs genuine wrong answers.
    termination_reason: str | None = None


def _pass_at_k(per_task_counts: list[tuple[int, int]], k: int) -> float:
    """Unbiased pass@k (Chen et al. 2021) over ``(n_attempts, n_correct)`` pairs.

    Per task: ``1 - C(n-c, k)/C(n, k)``, the probability that a random size-k
    subset of the n attempts contains at least one success. Tasks with fewer
    than ``k`` attempts fall back to their empirical any-pass rate.
    """
    from math import comb

    scores = []
    for n, c in per_task_counts:
        if n < k:
            scores.append(1.0 if c > 0 else 0.0)
        elif n - c < k:
            scores.append(1.0)
        else:
            scores.append(1.0 - comb(n - c, k) / comb(n, k))
    return sum(scores) / len(scores) if scores else 0.0


@dataclass
class EvalResult:
    """Aggregated evaluation results for a benchmark run."""

    dataset_name: str
    model: str
    agent: str
    score: float
    total: int
    correct: int
    errors: int
    items: list[EvalItem] = field(default_factory=list)
    signal_averages: dict[str, float] = field(default_factory=dict)
    timestamp: str = ""
    attempts: int = 1
    pass_at: dict[int, float] = field(default_factory=dict)
    # Count of rollouts per termination_reason (e.g. {"env_done": 95,
    # "timeout": 66, "verifier_timeout": 3}). The per-category error rate.
    termination_breakdown: dict[str, int] = field(default_factory=dict)
    # Solver token usage: turns and input/answer/reasoning/output counts.
    # No cost — vendor rates move, so pricing is left to the reader.
    usage: dict[str, float | int | None] = field(default_factory=dict)

    @classmethod
    def from_items(cls, dataset_name: str, model: str, agent: str, items: list[EvalItem], attempts: int = 1) -> EvalResult:
        """Create an EvalResult from a list of EvalItems.

        ``score``/``correct``/``total`` are always per-rollout (with equal
        attempts everywhere, ``score`` equals unbiased pass@1). When
        ``attempts > 1``, ``pass_at[k]`` is additionally computed for
        ``k = 1..attempts`` by grouping items on their task ``idx``.
        """
        total = len(items)
        correct = sum(1 for item in items if item.is_correct)
        errors = sum(1 for item in items if item.error is not None)
        score = correct / total if total > 0 else 0.0
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

        pass_at: dict[int, float] = {}
        if attempts > 1 and items:
            by_task: dict[int, list[EvalItem]] = {}
            for item in items:
                by_task.setdefault(item.idx, []).append(item)
            counts = [(len(group), sum(1 for i in group if i.is_correct)) for group in by_task.values()]
            pass_at = {k: _pass_at_k(counts, k) for k in range(1, attempts + 1)}

        # Compute signal averages
        signal_sums: dict[str, float] = {}
        signal_counts: dict[str, int] = {}
        for item in items:
            for name, value in item.signals.items():
                signal_sums[name] = signal_sums.get(name, 0.0) + value
                signal_counts[name] = signal_counts.get(name, 0) + 1
        signal_averages = {name: signal_sums[name] / signal_counts[name] for name in signal_sums}

        # Per-category outcome counts (agent timeouts vs infra/grading errors vs done).
        termination_breakdown: dict[str, int] = {}
        for item in items:
            key = item.termination_reason or "unknown"
            termination_breakdown[key] = termination_breakdown.get(key, 0) + 1

        usage = _usage_summary(items)

        return cls(
            dataset_name=dataset_name,
            model=model,
            agent=agent,
            score=score,
            total=total,
            correct=correct,
            errors=errors,
            items=items,
            signal_averages=signal_averages,
            timestamp=timestamp,
            attempts=attempts,
            pass_at=pass_at,
            termination_breakdown=termination_breakdown,
            usage=usage,
        )

    def summary_table(self) -> str:
        """Format a human-readable summary table."""
        pct = f"{self.score * 100:.1f}%"
        lines = [
            "",
            "Results:",
            f"  Accuracy:  {pct} ({self.correct}/{self.total})",
            f"  Errors:    {self.errors}",
        ]
        for k, v in sorted(self.pass_at.items()):
            lines.append(f"  pass@{k}:    {v * 100:.1f}%")
        if self.termination_breakdown:
            lines.append("  Terminations:")
            for reason, count in sorted(self.termination_breakdown.items(), key=lambda kv: (-kv[1], kv[0])):
                lines.append(f"    {reason}: {count}")
        if self.usage:
            lines.extend(
                [
                    "  Solver usage:",
                    f"    Average turns/task: {self.usage['average_turns_per_task']:.2f}",
                    f"    Input tokens/task: {self.usage['average_input_tokens_per_task']:.2f}",
                    f"    Output tokens/task: {self.usage['average_output_tokens_per_task']:.2f}",
                    f"    Total input tokens: {self.usage['total_input_tokens']}",
                    f"    Total output tokens: {self.usage['total_output_tokens']}",
                ]
            )
        return "\n".join(lines)

    def save(self, path: str | None = None) -> str:
        """Save results to a JSON file.

        Args:
            path: Optional output path. Defaults to ~/.rllm/eval_results/<dataset>_<model>_<timestamp>.json.

        Returns:
            The path the results were saved to.
        """
        if path is None:
            results_dir = paths.eval_results_dir()
            os.makedirs(results_dir, exist_ok=True)
            # Sanitize names for filename
            model_safe = self.model.replace("/", "_").replace("\\", "_")
            dataset_safe = self.dataset_name.replace("/", "_").replace("\\", "_")
            path = os.path.join(results_dir, f"{dataset_safe}_{model_safe}_{self.timestamp}.json")

        data = {
            "dataset_name": self.dataset_name,
            "model": self.model,
            "agent": self.agent,
            "score": self.score,
            "total": self.total,
            "correct": self.correct,
            "errors": self.errors,
            "timestamp": self.timestamp,
            "signal_averages": self.signal_averages,
            "attempts": self.attempts,
            "pass_at": {str(k): v for k, v in self.pass_at.items()},
            "termination_breakdown": self.termination_breakdown,
            "usage": self.usage,
            "items": [
                {
                    "idx": item.idx,
                    "attempt": item.attempt,
                    "reward": item.reward,
                    "is_correct": item.is_correct,
                    "error": item.error,
                    "signals": item.signals,
                    "metrics": item.metrics,
                    "termination_reason": item.termination_reason,
                }
                for item in self.items
            ],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return path

    @classmethod
    def load(cls, path: str) -> EvalResult:
        """Load results from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        items = [
            EvalItem(
                idx=item["idx"],
                reward=item["reward"],
                is_correct=item["is_correct"],
                error=item.get("error"),
                signals=item.get("signals", {}),
                metrics=item.get("metrics", {}),
                attempt=item.get("attempt", 0),
                termination_reason=item.get("termination_reason"),
            )
            for item in data["items"]
        ]

        return cls(
            dataset_name=data["dataset_name"],
            model=data["model"],
            agent=data["agent"],
            score=data["score"],
            total=data["total"],
            correct=data["correct"],
            errors=data["errors"],
            items=items,
            signal_averages=data.get("signal_averages", {}),
            timestamp=data.get("timestamp", ""),
            attempts=data.get("attempts", 1),
            pass_at={int(k): v for k, v in data.get("pass_at", {}).items()},
            termination_breakdown=data.get("termination_breakdown", {}),
            usage=data.get("usage", _usage_summary(items)),
        )


def _usage_summary(items: list[EvalItem]) -> dict[str, float | int | None]:
    """Aggregate solver-only token usage.

    Averages use every rollout as the denominator, including failed rollouts.
    This matches the benchmark's per-task reporting and prevents failures from
    making the averages look artificially high by silently shrinking the run.

    Tokens only: cost depends on vendor rates that move and vary per account,
    so it is left to whoever reads these counts rather than frozen into them.
    """
    count = len(items)

    def total(name: str) -> int:
        return sum(int(item.metrics.get(name, 0) or 0) for item in items)

    turns = total("turns")
    answer = total("answer_tokens")
    reasoning = total("reasoning_tokens")
    output = total("output_tokens")
    input_tokens = total("input_tokens")
    divisor = count or 1

    return {
        "tasks": count,
        "average_turns_per_task": turns / divisor,
        "average_answer_tokens_per_task": answer / divisor,
        "average_reasoning_tokens_per_task": reasoning / divisor,
        "average_output_tokens_per_task": output / divisor,
        "average_input_tokens_per_task": input_tokens / divisor,
        "total_input_tokens": input_tokens,
        "total_answer_tokens": answer,
        "total_reasoning_tokens": reasoning,
        "total_output_tokens": output,
    }
