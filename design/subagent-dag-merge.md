# Design: Subagent lineages as trajectories (gateway-tagged DAG)

- **Status:** implemented. Supersedes the transform-side multi-slot merge (#773, closed). Gateway `SessionSlots` + `lineage_id` tag in #774; collection split + advantage dedup in #775.
- **Related:** `rllm-model-gateway` (`token_accumulator.py` `SessionSlots`, `proxy.py`, `models.py`); `rllm/engine/agentflow_engine.py` (`enrich_episode_with_traces`), `rllm/engine/trace_converter.py`; `rllm/trainer/algorithms/{transform.py,advantage.py,rl_algo.py}`; the gateway-dag-token-storage RFC.

---

## Problem

Under the opencode/claude-code harness a subagent (`task` tool) runs under the **same** gateway session
as the parent but with a **different system prompt/tool set**, so its turns are not prefix-extensions of
the parent conversation. All of a rollout's turns land in **one** `Trajectory`, interleaved in time, and
the trainer's linear prefix-merge can't merge the interleaved lineages → one training row per turn →
`batch/merge_compression_ratio → 1`. Under `cumulative_token_mode` it is worse: the single accumulator
`reset()`s on every parent↔subagent switch and re-tokenizes the parent's resumed turn (token drift).

## Principles (the chosen model)

1. **A subagent is a fresh `Trajectory`.** The lineage/DAG structure lives on the **collection** side; the
   trainer only ever linear-merges the steps *within* a `Trajectory`.
2. **An episode may hold multiple same-named trajectories; the advantage baseline deduplicates them by
   rollout.** So one rollout = one GRPO baseline sample regardless of how many lineages it spawned.

These two together let the trainer transform stay a plain linear per-`Trajectory` merge (revert #773) with
**no** GRPO skew.

## Architecture (four layers)

### 1. Gateway — tag each trace with a lineage id
`SessionSlots` (PR #774) already opens one accumulator ("slot") per lineage: a request that continues a
slot extends it; one that continues none opens a new slot. Assign each slot a stable `lineage_id`
(`f"{session_id}#{n}"`, monotonic `n`) and stamp it on every `TraceRecord` (`TraceRecord.lineage_id`).
This is the single source of truth for lineage boundaries — no prefix re-derivation in the trainer, and it
stays correct under drift. (Non-cumulative mode: no slots ⇒ no tag ⇒ collection falls back to one
trajectory, as today.)

### 2. Collection — split traces into one Trajectory per lineage
`trace_converter.trace_record_to_step` carries `lineage_id` onto the `Step`. In
`enrich_episode_with_traces`, the "agent produced no steps → absorb all traces" branch (the CLI-harness
path) groups the session's traces by `lineage_id` and emits **one `Trajectory` per lineage**, in
first-appearance order, each carrying the episode reward. Each lineage's steps are a linear prefix chain
(time-ordered within the lineage), so the existing linear merge handles them.

### 3. Naming — lineage splits share the originating role
`_impute_trajectory_names` currently renames each *unnamed* trajectory to `f"{default}_{position}"`, which
would scatter K lineage-splits into K roles. Fix: name by the **originating** trajectory, not post-split
position — all lineage-splits of one original trajectory share one role name (single-agent → one role;
genuine multi-agent → distinct roles, unchanged). Concretely, the split stamps each lineage trajectory with
its originating trajectory index and imputation uses that; lineage-tagged trajectories are never
position-split against each other.

### 4. Advantage — dedup by rollout in the baseline
In `collect_reward_and_advantage_from_trajectory_groups`, build the per-group reward array with **one entry
per rollout** (dedup by `group.metadata[i]["rollout_idx"]`, which is already the per-trajectory parallel
metadata), pass that to the estimator unchanged, then **fan the per-rollout advantage back to every
trajectory of that rollout** (set `step.advantage` for all its steps). All estimators
(`advantage.py`) take a per-group reward array and return same-shape output, so nothing in the estimator
math changes. For today's single-trajectory runs each rollout already contributes exactly one trajectory,
so dedup is a **no-op** — no regression. It only bites when a rollout has multiple same-named trajectories,
i.e. the new subagent case.

### Transform — revert #773
Back to a single-segment linear per-`Trajectory` merge; each lineage-trajectory → one row. The forest /
multi-slot logic is removed from `verl/transform.py` and `tinker/transform.py`.

## Equivalence & correctness

- Produces the **same rows/advantages** as #773 (K rows per rollout, all sharing the rollout's advantage),
  but with the lineage concept owned by the gateway/collection and the trainer kept simple.
- GRPO baseline is over **rollouts** (dedup), so it is unbiased regardless of per-rollout lineage count —
  the exact defect that blocked moving the split above the advantage layer.
- `merge_compression_ratio` = total turns / #lineages, same as #773.

## Rollout / PRs
- #774 (base `terminal-rl`): gateway `SessionSlots` multi-slot accumulator + `lineage_id` on
  `SessionSlots`/`TraceRecord` + proxy stamping.
- #775 (stacked on #774): collection split + imputation fix + advantage dedup; the transform stays the plain
  linear merge (#773's change not included). #773 closed as superseded.

## Testing
- Gateway: each lineage's traces carry a distinct stable `lineage_id`; parent-resume traces carry the
  parent's id.
- Collection: a parent→subagent→parent-resume session yields 2 trajectories (parent merged, subagent
  separate); K subagents → K+1 trajectories.
- Advantage: dedup is a no-op for single-trajectory groups (bit-identical advantages); for a group with a
  multi-lineage rollout the baseline mean/std match the per-rollout values, and every lineage of a rollout
  gets that rollout's advantage.
- Transform: linear merge only; one row per (linear) trajectory.

## Out of scope
- `parent_trace_id` delta-chain **storage** (O(N²) trace bytes) — composes with the gateway-dag-token-storage
  RFC; independent of this.
- Non-cumulative-mode lineage splitting (no gateway tags there) — falls back to one trajectory as today.
