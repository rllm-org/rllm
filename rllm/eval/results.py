"""Eval results: data classes for storing and reporting evaluation outcomes."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class EvalItem:
    """Result for a single evaluation example."""

    idx: int
    reward: float
    is_correct: bool
    error: str | None = None
    signals: dict[str, float] = field(default_factory=dict)


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
    # Reward statistics across non-errored items. ``mean_reward`` is the
    # always-present aggregate; ``reward_min``/``reward_max`` are populated
    # only when the eval scored at least one item without errors.
    mean_reward: float = 0.0
    reward_min: float | None = None
    reward_max: float | None = None
    # Wall-clock duration of the eval run, in seconds. Set by the CLI just
    # before the result panel renders.
    runtime_sec: float = 0.0
    # Histogram of error-typed exceptions. Populated when ``errors > 0``.
    # Keys are the exception class names (or "Error" for legacy untyped
    # error strings); values are the number of items that hit that type.
    exception_counts: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_items(cls, dataset_name: str, model: str, agent: str, items: list[EvalItem]) -> EvalResult:
        """Create an EvalResult from a list of EvalItems."""
        total = len(items)
        correct = sum(1 for item in items if item.is_correct)
        errors = sum(1 for item in items if item.error is not None)
        score = correct / total if total > 0 else 0.0
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Compute signal averages
        signal_sums: dict[str, float] = {}
        signal_counts: dict[str, int] = {}
        for item in items:
            for name, value in item.signals.items():
                signal_sums[name] = signal_sums.get(name, 0.0) + value
                signal_counts[name] = signal_counts.get(name, 0) + 1
        signal_averages = {name: signal_sums[name] / signal_counts[name] for name in signal_sums}

        # Reward stats over the items that actually produced a score (skip
        # errored items so a single timeout doesn't drag the mean to 0).
        scored_rewards = [item.reward for item in items if item.error is None]
        if scored_rewards:
            mean_reward = sum(scored_rewards) / len(scored_rewards)
            reward_min: float | None = min(scored_rewards)
            reward_max: float | None = max(scored_rewards)
        else:
            mean_reward = 0.0
            reward_min = None
            reward_max = None

        # Group errors by exception type. Errors are stored as
        # ``"<TypeName>: <message>"`` (see eval/runner.py); strip the
        # message to count by type. Legacy untyped strings bucket as
        # ``"Error"``.
        exception_counts: dict[str, int] = {}
        for item in items:
            if item.error is None:
                continue
            head = item.error.split(":", 1)[0].strip()
            key = head if head and " " not in head else "Error"
            exception_counts[key] = exception_counts.get(key, 0) + 1

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
            mean_reward=mean_reward,
            reward_min=reward_min,
            reward_max=reward_max,
            exception_counts=exception_counts,
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
        return "\n".join(lines)

    def save(self, path: str | None = None) -> str:
        """Save results to a JSON file.

        Args:
            path: Optional output path. Defaults to ~/.rllm/eval_results/<dataset>_<model>_<timestamp>.json.

        Returns:
            The path the results were saved to.
        """
        if path is None:
            rllm_home = os.path.expanduser(os.environ.get("RLLM_HOME", "~/.rllm"))
            results_dir = os.path.join(rllm_home, "eval_results")
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
            "mean_reward": self.mean_reward,
            "reward_min": self.reward_min,
            "reward_max": self.reward_max,
            "runtime_sec": self.runtime_sec,
            "exception_counts": self.exception_counts,
            "signal_averages": self.signal_averages,
            "items": [{"idx": item.idx, "reward": item.reward, "is_correct": item.is_correct, "error": item.error, "signals": item.signals} for item in self.items],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return path
