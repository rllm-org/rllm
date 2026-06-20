"""Curate eval trajectories into SFT datasets.

Reads one or more ``rllm eval`` run directories, groups the per-rollout results
by task (pooling attempts across runs), filters tasks by an aggregate-metric
expression (see :mod:`rllm.eval.filter_dsl`), selects which trajectories to keep
per surviving task, and emits ``{"messages": [...]}`` rows ready for
:meth:`rllm.data.DatasetRegistry.register_dataset`.

This is the engine behind ``rllm dataset from-eval``. It is pure and
GPU-free — everything works off the JSON a run dir already contains:

    <run_dir>/
        results.json                       # EvalResult: per-rollout items + attempts
        episodes/episode_NNNNNN_<task>.json # full Episode (trajectories → steps → chat_completions)

Filtering and reward/correctness-based selection run off the lightweight
``results.json`` items; only the chosen episodes are deserialized.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

from rllm import paths
from rllm.eval.filter_dsl import FilterError, compile_filter
from rllm.eval.results import EvalResult, _pass_at_k

# Per-task trajectory selection strategies.
SELECT_STRATEGIES = ("correct", "best", "best-n", "shortest", "all")

# episode_000123_<task_id>.json
_EP_RE = re.compile(r"^episode_(\d+)_(.*)\.json$")


class CurationError(Exception):
    """Raised for unresolvable run dirs or invalid curation configuration."""


# ---------------------------------------------------------------------------
# Config / stats
# ---------------------------------------------------------------------------


@dataclass
class CurationConfig:
    """Knobs for :func:`curate`. Mirrors the ``rllm dataset from-eval`` flags."""

    metric: str = "is_correct"  # what avg/best/worst aggregate: is_correct | reward | <signal name>
    filter_expr: str = "solved"  # task-level boolean over aggregates (filter_dsl)
    select: str = "correct"  # correct | best | best-n | shortest | all
    max_per_task: int | None = None
    min_reward: float | None = None  # passing predicate; None → use is_correct
    dedup: bool = False
    trajectory: str | None = None  # named trajectory to extract; None → first

    def validate(self) -> None:
        if self.select not in SELECT_STRATEGIES:
            raise CurationError(f"Unknown --select {self.select!r}. Choose from: {', '.join(SELECT_STRATEGIES)}.")
        if self.max_per_task is not None and self.max_per_task < 1:
            raise CurationError("--max-per-task must be >= 1.")
        if self.select == "best-n" and self.max_per_task is None:
            raise CurationError("--select best-n requires --max-per-task N.")


@dataclass
class CurationStats:
    """Summary of a curation pass, for reporting / ``--dry-run``."""

    runs: int = 0
    tasks_total: int = 0
    tasks_kept: int = 0
    attempts_total: int = 0
    rows_emitted: int = 0
    rows_skipped_no_messages: int = 0
    rows_deduped: int = 0


# ---------------------------------------------------------------------------
# Internal data model
# ---------------------------------------------------------------------------


@dataclass
class _AttemptRef:
    run_id: str
    eval_idx: int
    attempt: int
    score: float  # value of the chosen metric for this attempt
    is_correct: bool
    reward: float
    signals: dict
    error: str | None
    episode_path: Path | None


@dataclass
class AttemptGroup:
    """One task's pooled attempts, with the aggregates the filter DSL reads."""

    task_id: str
    metric: str
    attempts: list[_AttemptRef] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.attempts)

    @property
    def n_correct(self) -> int:
        return sum(1 for a in self.attempts if a.is_correct)

    @property
    def solved(self) -> bool:
        return self.n_correct > 0

    @property
    def _scores(self) -> list[float]:
        return [a.score for a in self.attempts]

    @property
    def avg(self) -> float:
        return mean(self._scores) if self.attempts else 0.0

    @property
    def best(self) -> float:
        return max(self._scores) if self.attempts else 0.0

    @property
    def worst(self) -> float:
        return min(self._scores) if self.attempts else 0.0

    def _at(self, name: str, k: int) -> float:
        """Accessor for ``name@k`` filter forms."""
        if name == "pass":
            return _pass_at_k([(self.n, self.n_correct)], k)
        if name == "avg":  # avg is k-invariant; @k is cosmetic
            return self.avg
        if name == "best":
            return self.best
        if name == "worst":
            return self.worst
        raise FilterError(f"Unknown metric {name!r} in '{name}@{k}'. Use pass@k, avg@k, best@k, or worst@k.")

    def filter_namespace(self) -> dict:
        return {
            "avg": self.avg,
            "best": self.best,
            "worst": self.worst,
            "solved": self.solved,
            "n": self.n,
            "n_correct": self.n_correct,
            "_at": self._at,
        }


@dataclass
class _RunData:
    run_id: str
    run_dir: Path
    attempts: int
    items: list  # list[EvalItem]
    episodes: dict[int, tuple[Path, str]]  # eval_idx -> (path, task_id)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _resolve_run_dir(ref: str | Path) -> Path:
    """Resolve a run reference to a directory containing ``results.json``.

    Accepts a path to a run dir, or a bare ``run_id`` under
    ``~/.rllm/eval_results/``.
    """
    p = Path(ref).expanduser()
    if (p / "results.json").is_file():
        return p
    cand = Path(paths.eval_results_dir()) / str(ref)
    if (cand / "results.json").is_file():
        return cand
    raise CurationError(f"No eval run found at {ref!r} (looked for results.json in {p} and {cand}).")


def _index_episodes(episodes_dir: Path) -> dict[int, tuple[Path, str]]:
    out: dict[int, tuple[Path, str]] = {}
    if not episodes_dir.is_dir():
        return out
    for f in episodes_dir.iterdir():
        m = _EP_RE.match(f.name)
        if m:
            out[int(m.group(1))] = (f, m.group(2))
    return out


def _load_run(ref: str | Path) -> _RunData:
    run_dir = _resolve_run_dir(ref)
    result = EvalResult.load(str(run_dir / "results.json"))
    attempts = max(1, int(result.attempts or 1))
    episodes = _index_episodes(run_dir / "episodes")
    return _RunData(run_id=run_dir.name, run_dir=run_dir, attempts=attempts, items=list(result.items), episodes=episodes)


def _metric_value(item, metric: str) -> float:
    if metric in ("is_correct", "correct"):
        return 1.0 if item.is_correct else 0.0
    if metric == "reward":
        return float(item.reward or 0.0)
    return float((item.signals or {}).get(metric, 0.0))


def _build_groups(runs: list[_RunData], metric: str) -> list[AttemptGroup]:
    """Group every rollout by stable task id, pooling across runs."""
    groups: dict[str, AttemptGroup] = {}
    for run in runs:
        for item in run.items:
            # The runner expands task `idx` into `attempts` adjacent rollouts,
            # so the on-disk episode index is idx*attempts + attempt.
            eval_idx = item.idx * run.attempts + item.attempt
            ep = run.episodes.get(eval_idx)
            path = ep[0] if ep else None
            task_id = ep[1] if ep else None
            key = task_id if task_id is not None else f"{run.run_id}:t{item.idx}"
            ref = _AttemptRef(
                run_id=run.run_id,
                eval_idx=eval_idx,
                attempt=item.attempt,
                score=_metric_value(item, metric),
                is_correct=bool(item.is_correct),
                reward=float(item.reward or 0.0),
                signals=dict(item.signals or {}),
                error=item.error,
                episode_path=path,
            )
            groups.setdefault(key, AttemptGroup(task_id=key, metric=metric)).attempts.append(ref)
    return list(groups.values())


# ---------------------------------------------------------------------------
# Message extraction
# ---------------------------------------------------------------------------


def _clean_message(m: dict) -> dict | None:
    """Normalize one chat-completion message; drop empties (no content/tool_calls)."""
    if not isinstance(m, dict):
        return None
    role = m.get("role")
    if not role:
        return None
    content = m.get("content")
    if isinstance(content, (str, list)):
        norm = content
    elif content is None:
        norm = ""
    else:
        norm = str(content)
    out = {"role": role, "content": norm}
    for k in ("tool_calls", "tool_call_id", "name"):
        if m.get(k) is not None:
            out[k] = m[k]
    has_text = bool(norm.strip()) if isinstance(norm, str) else bool(norm)
    if not (has_text or out.get("tool_calls")):
        return None
    return out


def _episode_to_messages(episode: dict, trajectory_name: str | None) -> list[dict] | None:
    """Extract the conversation from an episode's chosen trajectory.

    Uses the last step that carries ``chat_completions`` (the full conversation),
    matching how eval scores trajectories. Returns ``None`` if no usable
    (>=2-turn) conversation is found.
    """
    trajs = episode.get("trajectories") or []
    if not trajs:
        return None
    traj = None
    if trajectory_name:
        for t in trajs:
            if t.get("name") == trajectory_name:
                traj = t
                break
        if traj is None:
            return None
    else:
        traj = trajs[0]
    steps = traj.get("steps") or []
    for step in reversed(steps):
        cc = step.get("chat_completions")
        if cc:
            clean = [c for c in (_clean_message(m) for m in cc) if c is not None]
            if len(clean) >= 2:
                return clean
    return None


def _load_messages(ref: _AttemptRef, trajectory_name: str | None) -> list[dict] | None:
    if ref.episode_path is None or not ref.episode_path.is_file():
        return None
    try:
        with open(ref.episode_path, encoding="utf-8") as f:
            episode = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return _episode_to_messages(episode, trajectory_name)


def _content_len(messages: list[dict]) -> int:
    return sum(len(m["content"]) if isinstance(m.get("content"), str) else len(str(m.get("content"))) for m in messages)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def _passes(ref: _AttemptRef, min_reward: float | None) -> bool:
    if min_reward is not None:
        return ref.score >= min_reward
    return ref.is_correct


def _ranked_candidates(group: AttemptGroup, config: CurationConfig) -> list[_AttemptRef]:
    """Passing attempts for a task, ordered best-first (by score, then attempt)."""
    if config.select == "all":
        cands = list(group.attempts)
    else:
        cands = [a for a in group.attempts if _passes(a, config.min_reward)]
    cands.sort(key=lambda a: (a.score, -a.attempt), reverse=True)
    return cands


def _make_row(ref: _AttemptRef, group: AttemptGroup, messages: list[dict]) -> dict:
    return {
        "messages": messages,
        "source_run": ref.run_id,
        "task_id": group.task_id,
        "attempt": ref.attempt,
        "score": ref.score,
        "reward": ref.reward,
    }


def _assistant_signature(messages: list[dict]) -> str:
    parts = [str(m.get("content", "")) for m in messages if m.get("role") == "assistant"]
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def curate(run_dirs: list[str | Path], config: CurationConfig | None = None) -> tuple[list[dict], CurationStats]:
    """Curate eval trajectories into SFT ``{"messages": ...}`` rows.

    Args:
        run_dirs: run ids (under ``~/.rllm/eval_results``) or paths to run dirs.
        config: curation knobs; defaults to "keep any solved task, take correct trajectories".

    Returns:
        ``(rows, stats)`` — ``rows`` is ready for ``DatasetRegistry.register_dataset``.
    """
    config = config or CurationConfig()
    config.validate()
    if not run_dirs:
        raise CurationError("No eval runs provided.")

    flt = compile_filter(config.filter_expr)
    runs = [_load_run(r) for r in run_dirs]
    groups = _build_groups(runs, config.metric)

    stats = CurationStats(runs=len(runs), tasks_total=len(groups), attempts_total=sum(g.n for g in groups))
    rows: list[dict] = []

    for group in groups:
        if not flt.evaluate(group.filter_namespace()):
            continue
        stats.tasks_kept += 1
        cands = _ranked_candidates(group, config)

        if config.select == "shortest":
            loaded = []
            for ref in cands:
                messages = _load_messages(ref, config.trajectory)
                if messages is None:
                    stats.rows_skipped_no_messages += 1
                    continue
                loaded.append((ref, messages, _content_len(messages)))
            loaded.sort(key=lambda t: t[2])
            if config.max_per_task is not None:
                loaded = loaded[: config.max_per_task]
            rows.extend(_make_row(ref, group, messages) for ref, messages, _ in loaded)
        else:
            limit = 1 if config.select == "best" else config.max_per_task
            taken = 0
            for ref in cands:
                if limit is not None and taken >= limit:
                    break
                messages = _load_messages(ref, config.trajectory)
                if messages is None:
                    stats.rows_skipped_no_messages += 1
                    continue
                rows.append(_make_row(ref, group, messages))
                taken += 1

    if config.dedup:
        seen: set[str] = set()
        deduped: list[dict] = []
        for row in rows:
            key = _assistant_signature(row["messages"])
            if key in seen:
                stats.rows_deduped += 1
                continue
            seen.add(key)
            deduped.append(row)
        rows = deduped

    stats.rows_emitted = len(rows)
    return rows, stats
