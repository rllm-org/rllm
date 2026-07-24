"""Colorful one-line summary of a finished task group.

A *task group* is all rollouts sharing a ``task_id`` (created by
``interleave_tasks`` for GRPO). Both execution paths reduce to the same readout:

- the sync/eval path (``AgentFlowEngine.execute_tasks``) prints it once a group's
  last rollout lands, and
- the async training path (``TrajectoryGroupBuffer`` in ``rllm.trainer.buffer``)
  prints it from the per-group completion hook, passing the extra ``status`` /
  ``reason`` it knows (queued vs. filtered, and why).

Kept dependency-light (only ``rllm.types``) so both engine and trainer can import
it without a cycle.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import click

from rllm.types import INFRA_ERROR_REASONS

if TYPE_CHECKING:
    from rllm.types import Episode


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _pstd(xs: list[float]) -> float:
    """Population standard deviation (0 for <2 samples)."""
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def rewards_by_trajectory_name(episodes: list[Episode]) -> dict[str, list[float]]:
    """Per-trajectory-name reward lists across a task group.

    GRPO groups trajectories by (task_id, trajectory name) and computes advantage
    within each name-group, so reward stats are only meaningful *per name* — not
    collapsed to one scalar per episode. Falls back to a trajectory's last step
    reward when ``traj.reward`` is unset (mirrors the per-episode log line). Name
    order is first-seen, for a stable readout.
    """
    by_name: dict[str, list[float]] = {}
    for ep in episodes:
        for traj in ep.trajectories:
            reward = traj.reward
            if reward is None and traj.steps and traj.steps[-1].reward is not None:
                reward = traj.steps[-1].reward
            if reward is None:
                continue
            by_name.setdefault(traj.name, []).append(float(reward))
    return by_name


def format_group_finished(
    task_id: str,
    episodes: list[Episode],
    *,
    status: str | None = None,
    reason: str | None = None,
) -> str:
    """One-line colorful summary of a finished task group (all rollouts sharing a task_id).

    Surfaces the GRPO-relevant signal at a glance: solve rate, reward spread
    **per trajectory name** (GRPO forms one advantage group per name — a *flat*
    name, all rewards equal, gives zero advantage and gets filtered), straggler
    timing (the group only finishes when its slowest rollout does), and the
    termination-reason breakdown. Segments are individually styled with
    ``click.style`` and joined, so the line is multi-color.

    ``status`` / ``reason`` are optional and come from the async training buffer:
    ``queued`` (accepted into training, green) or ``filtered`` (dropped, with the
    reason — e.g. ``uniform_reward`` — in yellow).
    """
    n = len(episodes)
    n_correct = sum(1 for e in episodes if e.is_correct)
    rewards_by_name = rewards_by_trajectory_name(episodes)
    rollout_s = [e.metrics["time/rollout_s"] for e in episodes if "time/rollout_s" in e.metrics]
    llm_s = [e.metrics["time/agentflow_llm_wall_s"] for e in episodes if "time/agentflow_llm_wall_s" in e.metrics]
    steps = [e.metrics["n_turns"] for e in episodes if "n_turns" in e.metrics]
    terms = Counter(e.termination_reason.value if e.termination_reason is not None else "None" for e in episodes)

    short_id = task_id.split("-")[0]  # first uuid segment — enough to eyeball

    seg = [click.style(f"█ group {short_id} ×{n}", fg="cyan", bold=True)]

    # Buffer verdict (async path only): did this group make it into training?
    if status == "queued":
        seg.append(click.style("✓ queued", fg="green", bold=True))
    elif status == "filtered":
        seg.append(click.style(f"⊘ filtered:{reason}" if reason else "⊘ filtered", fg="yellow", bold=True))

    seg.append(click.style(f"solved {n_correct}/{n} ({100.0 * n_correct / n if n else 0.0:.0f}%)", fg="bright_white", bold=True))

    # Reward per trajectory name — this is the GRPO advantage group. A name whose
    # rewards are all equal (flat) yields zero advantage and gets filtered, so mark
    # it yellow; a name with spread carries a learning signal, so green.
    if rewards_by_name:
        for name, rs in rewards_by_name.items():
            flat = len(rs) >= 2 and _pstd(rs) < 1e-9
            txt = f"{name} μ{_mean(rs):.2f} σ{_pstd(rs):.2f} [{min(rs):.1f}, {max(rs):.1f}]"
            seg.append(click.style(txt, fg="yellow" if flat else "green"))
    else:
        seg.append(click.style("reward N/A", fg="white"))

    if rollout_s:
        seg.append(click.style(f"rollout μ{_mean(rollout_s):.0f}s max{max(rollout_s):.0f}s", fg="blue"))
    if llm_s:
        seg.append(click.style(f"llm μ{_mean(llm_s):.0f}s", fg="blue"))
    if steps:
        seg.append(click.style(f"steps μ{_mean(steps):.0f}", fg="blue"))

    # Termination — red if any rollout hit an infra/grading failure (a wasted,
    # untrustworthy rollout that training filters), yellow otherwise. Normal
    # agent outcomes (env_done, max_turns, timeout, length limits) stay yellow.
    _infra = {r.value for r in INFRA_ERROR_REASONS}
    any_infra = any(k in _infra for k in terms)
    seg.append(click.style(" ".join(f"{k}×{v}" for k, v in terms.most_common()), fg="red" if any_infra else "yellow"))

    return "  ".join(seg)
