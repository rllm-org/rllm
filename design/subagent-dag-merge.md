# Design: Subagent lineages as gateway-tagged partitions of one trajectory

- **Status:** implemented. Gateway `SessionSlots` + `lineage_id` tag in #774; trainer transform partition in #775. Supersedes the transform-side byte-prefix multi-slot merge (#773, closed) and the collection-split-into-multiple-trajectories variant (never merged).
- **Related:** `rllm-model-gateway` (`token_accumulator.py` `SessionSlots`, `proxy.py`, `models.py`); `rllm/engine/trace_converter.py`; `rllm/trainer/verl/transform.py`, `rllm/trainer/tinker/transform.py`; the gateway-dag-token-storage RFC.

---

## Problem

Under the opencode/claude-code harness a subagent (`task` tool) runs under the **same** gateway session
as the parent but with a **different system prompt/tool set**, so its turns are not prefix-extensions of
the parent conversation. All of a rollout's turns land in **one** `Trajectory`, interleaved in time, and
the trainer's linear prefix-merge (single running segment) can't merge the interleaved lineages: it
reseeds on every parent↔subagent switch → one training row per turn → `batch/merge_compression_ratio → 1`.
Under `cumulative_token_mode` it is worse: the single accumulator `reset()`s on every switch and
re-tokenizes the parent's resumed turn (token drift).

## Model

A subagent-spawning rollout is **one attempt** that delegates — so it stays **one `Trajectory`** (one GRPO
sample). The lineage/DAG structure is carried as a per-step **tag** and consumed only where training rows
are packed:

- **The gateway** already distinguishes lineages (`SessionSlots`, #774); it assigns each a stable
  `lineage_id` and stamps it on every trace.
- **The trainer transform** partitions a trajectory's steps by `lineage_id` and linear-merges each
  partition independently → one row per lineage. Parent turns merge among themselves; each subagent among
  itself.
- **Everything else is untouched**: enrichment still builds one trajectory, grouping/naming/imputation are
  unchanged, and the GRPO advantage baseline stays per-trajectory (= per-rollout) — **no dedup needed**.

This is why it beats the two alternatives considered: it needs no byte-prefix inference (#773 — the gateway
tag is authoritative and drift-proof for *grouping*) and no per-rollout advantage dedup or group-size
surgery (the "subagent = its own trajectory" variant, which broke "one rollout = one baseline sample").

## Architecture

1. **Gateway (#774).** `SessionSlots` keeps one `TokenAccumulator` per lineage (drift-free generation under
   cumulative mode) and assigns each a stable `lineage_id` (`f"{session}#{n}"`, monotonic). The proxy stamps
   it on `TraceRecord.lineage_id` at build time (resolved from the active slot on the event loop, never from
   the deferred store task).
2. **Collection.** `trace_converter.trace_record_to_step` copies `lineage_id` onto `Step.metadata`.
   `enrich_episode_with_traces` is **unchanged** — still one trajectory per session.
3. **Transform (verl + tinker).** `_partition_steps_by_lineage(steps)` groups by `step.metadata["lineage_id"]`
   (first-appearance order; untagged → one partition = today's behavior). The existing single-segment linear
   merge runs **within** each partition. verl emits one masked row per partition (all keyed by
   `trajectory.uid`, sharing the trajectory's broadcast advantage); tinker emits one Datum per partition.
4. **Advantage / grouping / imputation.** Unchanged. One trajectory = one rollout = one baseline sample.

## Equivalence & correctness

- Emits the **same rows and the same loss** as any split-based approach: a rollout with a parent + K
  subagent lineages becomes K+1 rows (different-prefix lineages can't share one masked sequence).
- Because the trajectory stays 1:1 with the rollout, the GRPO baseline is correct **by construction** and
  the shared advantage/grouping code is not touched — a no-op for existing single-lineage runs.
- Within a partition the linear merge is byte-exact (the gateway kept each lineage drift-free); a genuine
  mid-lineage break (compaction) still splits into an extra row inside that partition, as before.
- `merge_compression_ratio` = total turns / total rows recovers from ~1; `steps_per_traj` reads as the
  number of lineage rows the trajectory produced.

## PRs
- #774 (base `terminal-rl`): gateway `SessionSlots` multi-slot accumulator + `lineage_id` on
  `SessionSlots`/`TraceRecord` + proxy stamping.
- #775 (stacked on #774): `trace_converter` tag pass-through + `_partition_steps_by_lineage` in the verl and
  tinker transforms. #773 closed.

## Testing
- Gateway (#774): distinct stable `lineage_id` per lineage; parent-resume reuses the parent's id.
- Transform (#775): one trajectory with interleaved lineage-tagged steps →
  parent-subagent-parent = 2 rows/Datums (parent merged, mask `[1,1,0,1,1]`); parent + 2 subagents = 3;
  untagged steps → single partition = original behavior. verl module isn't importable in this env; the
  partition+merge logic is additionally validated with a standalone replica.

## Out of scope
- `parent_trace_id` delta-chain **storage** (O(N²) trace bytes) — composes with the gateway-dag-token-storage
  RFC; independent of this.
- Non-cumulative-mode lineage splitting: no gateway tags there → one partition (one trajectory), as today.
