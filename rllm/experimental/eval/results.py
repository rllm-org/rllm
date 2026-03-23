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

        return cls(dataset_name=dataset_name, model=model, agent=agent, score=score, total=total, correct=correct, errors=errors, items=items, signal_averages=signal_averages, timestamp=timestamp)

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
            # Sanitize model name for filename
            model_safe = self.model.replace("/", "_").replace("\\", "_")
            path = os.path.join(results_dir, f"{self.dataset_name}_{model_safe}_{self.timestamp}.json")

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
            "items": [{"idx": item.idx, "reward": item.reward, "is_correct": item.is_correct, "error": item.error, "signals": item.signals} for item in self.items],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return path
