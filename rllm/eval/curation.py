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
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

from rllm import paths
from rllm.eval.filter_dsl import FilterError, compile_filter
from rllm.eval.results import EvalResult, _pass_at_k

logger = logging.getLogger(__name__)

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
    rows_skipped_no_messages: int = 0  # attempts whose automerge walk yielded zero segments
    rows_deduped: int = 0
    rows_invalid: int = 0  # rows dropped because they failed SFT schema validation
    # Automerge-walk telemetry (from-eval): how steps merged/split into rows.
    segments_merged: int = 0  # steps merged into an already-open segment
    segments_split: int = 0  # times a new segment was started while one was already open (same attempt)
    steps_skipped_no_assistant: int = 0  # steps with no assistant turn to train
    targets_skipped_empty: int = 0  # steps whose last assistant turn had no content and no tool_calls


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
# Message extraction — automerge walk (always on; the data decides).
#
# Each attempt becomes one or more self-describing rows: every message carries a
# ``trainable`` flag, and an automerge walk (message-level analogue of
# ``rllm.trainer.verl.transform._process_trajectory``) merges steps that form a
# clean prefix chain into one row (all their turns trained) and splits where they
# don't. This is deterministic — no flag decides it:
#   - A NON-thinking trajectory (each turn's history-form == its target-form) is
#     one prefix chain and merges into a single row training every turn.
#   - A THINKING trajectory follows what the data shows: when the harness strips a
#     turn's reasoning from history (non-interleaved runs), the reasoning-aware
#     prefix breaks and each turn becomes its own row; when history RETAINS
#     reasoning (interleaved-thinking runs), steps keep merging and context keeps
#     its ThinkingParts. Either way rows match what the model saw at inference.
# Every turn is a trained target; only a step with no assistant turn is skipped.
# Reasoning is preserved MODEL-AGNOSTICALLY as a structured ``ThinkingPart`` (not
# a hardcoded ``<think>`` string) so the model's renderer picks the reasoning
# format at training time (deepseek ``<think>``, qwen, harmony, ...). The SFT
# loader renders these rows with tinker's ``CUSTOMIZED`` mode, driven by the
# per-message ``trainable`` flag.
# ---------------------------------------------------------------------------


def _reasoning(msg: dict) -> str:
    return msg.get("reasoning_content") or msg.get("reasoning") or ""


def _text_content(content) -> str:
    """The plain-text view of a message's content (str or list of parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def _parts_thinking(content) -> str:
    """The thinking-text view of a parts-list content."""
    if isinstance(content, list):
        return "".join(p.get("thinking", "") for p in content if isinstance(p, dict) and p.get("type") == "thinking")
    return ""


def _passthrough(out: dict, msg: dict) -> dict:
    """Carry provider fields (tool calls etc.) through to the rebuilt message."""
    for k in ("tool_calls", "tool_call_id", "name"):
        if msg.get(k) is not None:
            out[k] = msg[k]
    return out


def _to_parts(msg: dict) -> list[dict]:
    """Message content as a parts list, reasoning preserved as a ``ThinkingPart``.

    NOT a hardcoded ``<think>`` string: the model's *renderer* decides the
    reasoning format at training time (deepseek ``<think>``, qwen, harmony, ...).
    Content is always a list of parts so the dataset's content column stays a
    uniform type for parquet.
    """
    parts: list[dict] = []
    rc = _reasoning(msg).strip()
    if rc:
        parts.append({"type": "thinking", "thinking": rc})
    parts.append({"type": "text", "text": msg.get("content") or ""})
    return parts


def _step_target(msg: dict) -> dict:
    """An assistant turn as a trained target — model-agnostic."""
    return _passthrough({"role": "assistant", "content": _to_parts(msg), "trainable": True}, msg)


def _step_context(msg: dict) -> dict:
    """A history/context message: never trained. Reasoning stays a ThinkingPart
    when the episode data carries it (interleaved-thinking runs), so context
    matches what the model actually saw."""
    return _passthrough({"role": msg["role"], "content": _to_parts(msg), "trainable": False}, msg)


def _step_has_content(msg: dict) -> bool:
    return bool(_text_content(msg.get("content")).strip())


def _keep(msg: dict) -> bool:
    """A message worth keeping in a conversation: it carries text or tool calls.

    The one keep-predicate applied SYMMETRICALLY on both sides of the prefix
    walk — a message dropped from the running segment must also be dropped from
    the step view it is compared against, or the positional window desyncs."""
    return _step_has_content(msg) or bool(msg.get("tool_calls"))


def _prefix_matches(seg_messages: list[dict], step_cc: list[dict]) -> bool:
    """Whether the running segment's turns appear identically — text, reasoning,
    and tool calls — at the start of the new step's conversation. A context reset
    (summarization) breaks this, and so does a turn whose thinking the harness
    stripped from history (non-interleaved runs)."""
    seg_view = [(m["role"], _text_content(m["content"]), _parts_thinking(m["content"]), str(m.get("tool_calls"))) for m in seg_messages]
    step_view = [(m.get("role"), _text_content(m.get("content") or ""), _reasoning(m).strip(), str(m.get("tool_calls"))) for m in step_cc[: len(seg_view)]]
    return len(step_cc) >= len(seg_view) and step_view == seg_view


def _episode_to_step_message_lists(episode: dict, trajectory_name: str | None, stats: CurationStats | None = None) -> list[list[dict]]:
    """Extract per-turn (automerged) training conversations from an episode.

    Returns one message-list per segment; every message carries ``trainable``.
    A segment merges the next step iff its turns — text, reasoning, and tool
    calls — appear identically as the new step's history. Purely data-driven:
    a context reset splits, and a turn whose thinking was stripped from history
    (non-interleaved harness) splits, while interleaved-thinking histories keep
    merging with their ThinkingParts intact. No flag or format string decides it.

    The prefix walk filters both sides with the same ``_keep`` predicate: the
    running segment and the step's history view are compared over their KEPT
    messages, never a mix of filtered-vs-raw positions (which would desync the
    window on an empty history message). ``stats`` (optional) accumulates
    merge/split/skip telemetry across attempts.
    """
    trajs = episode.get("trajectories") or []
    traj = None
    if trajectory_name:
        traj = next((t for t in trajs if t.get("name") == trajectory_name), None)
    elif trajs:
        traj = trajs[0]
    if traj is None:
        return []

    segments: list[list[dict]] = []
    seg: list[dict] | None = None
    targets = 0  # steps that became a trained target in THIS attempt
    merges = 0  # merges in THIS attempt
    for step in traj.get("steps") or []:
        cc = step.get("chat_completions") or []
        last = next((i for i in range(len(cc) - 1, -1, -1) if cc[i].get("role") == "assistant"), -1)
        if last < 0:
            if stats is not None:
                stats.steps_skipped_no_assistant += 1
            continue  # no assistant turn to train
        if not _keep(cc[last]):
            # An empty final assistant (no text, no tool_calls) carries no
            # trainable signal; never emit a segment with an empty target.
            if stats is not None:
                stats.targets_skipped_empty += 1
            continue
        # KEPT view of the history before the target; the target is always the
        # last assistant message regardless of _keep.
        kept = [m for m in cc[:last] if _keep(m)]
        target = _step_target(cc[last])
        if seg is not None and _prefix_matches(seg, kept):
            tail = [_step_context(m) for m in kept[len(seg) :]]
            seg = seg + tail + [target]
            merges += 1
            if stats is not None:
                stats.segments_merged += 1
        else:
            if seg is not None:
                segments.append(seg)
                if stats is not None:
                    stats.segments_split += 1
            seg = [_step_context(m) for m in kept] + [target]
        targets += 1
    if seg is not None:
        segments.append(seg)
    if targets >= 3 and merges == 0:
        logger.warning(
            "from-eval automerge: attempt produced %d trained steps but 0 merges (degenerate splitting — likely a per-step-varying history element).",
            targets,
        )
    return segments


def _load_step_message_lists(ref: _AttemptRef, trajectory_name: str | None, stats: CurationStats | None = None) -> list[list[dict]]:
    if ref.episode_path is None or not ref.episode_path.is_file():
        return []
    try:
        with open(ref.episode_path, encoding="utf-8") as f:
            episode = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    # Schema-2 episodes store deduplicated messages; identity on legacy files.
    from rllm.eval.episode_compact import expand_episode

    return _episode_to_step_message_lists(expand_episode(episode), trajectory_name, stats)


def _content_len(messages: list[dict]) -> int:
    return sum(len(_text_content(m.get("content"))) for m in messages)


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


def _rows_for_attempt(ref: _AttemptRef, group: AttemptGroup, trajectory_name: str | None, stats: CurationStats | None = None) -> list[dict]:
    """All rows one attempt (episode) contributes, via the automerge walk.

    Each row is validated through the SFT schema (``rllm.data.sft_schema``)
    before emission, so malformed provider payloads are caught here — at
    creation time — rather than deep inside a training backend.
    """
    from rllm.data.sft_schema import SFTSchemaError, normalize_row

    rows: list[dict] = []
    for seg in _load_step_message_lists(ref, trajectory_name, stats):
        raw = _make_row(ref, group, seg)
        try:
            rows.append(normalize_row(raw).to_record())
        except SFTSchemaError as e:
            logger.warning("Dropping curated row (task %s, attempt %d): %s", group.task_id, ref.attempt, e)
            if stats is not None:
                stats.rows_invalid += 1
    return rows


def _row_signature(row: dict) -> str:
    """Dedup key for a curated row: its task_id plus every message's role, text,
    thinking, and tool calls. Keyed on the full conversation (not just assistant
    content) so two distinct tasks that happen to share an assistant string do
    not collide, while identical attempts of the SAME task still dedup."""
    parts = [str(row.get("task_id"))]
    for m in row.get("messages") or []:
        content = m.get("content")
        parts.append(str(m.get("role") or ""))
        parts.append(_text_content(content))
        parts.append(_parts_thinking(content))
        parts.append(str(m.get("tool_calls")))
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

        # Each attempt becomes one or more self-describing rows via the automerge
        # walk (below): steps that form a clean prefix chain merge into one row
        # (all turns trained), others split — deterministic, decided by the data.
        if config.select == "shortest":
            loaded = []
            for ref in cands:
                attempt_rows = _rows_for_attempt(ref, group, config.trajectory, stats)
                if not attempt_rows:
                    stats.rows_skipped_no_messages += 1
                    continue
                total_len = sum(_content_len(r["messages"]) for r in attempt_rows)
                loaded.append((attempt_rows, total_len))
            loaded.sort(key=lambda t: t[1])
            if config.max_per_task is not None:
                loaded = loaded[: config.max_per_task]
            for attempt_rows, _ in loaded:
                rows.extend(attempt_rows)
        else:
            limit = 1 if config.select == "best" else config.max_per_task
            taken = 0
            for ref in cands:
                if limit is not None and taken >= limit:
                    break
                attempt_rows = _rows_for_attempt(ref, group, config.trajectory, stats)
                if not attempt_rows:
                    stats.rows_skipped_no_messages += 1
                    continue
                rows.extend(attempt_rows)
                taken += 1

    if config.dedup:
        seen: set[str] = set()
        deduped: list[dict] = []
        for row in rows:
            key = _row_signature(row)
            if key in seen:
                stats.rows_deduped += 1
                continue
            seen.add(key)
            deduped.append(row)
        rows = deduped

    stats.rows_emitted = len(rows)
    return rows, stats
