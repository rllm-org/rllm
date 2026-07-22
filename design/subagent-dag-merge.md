# Design: Multi-lineage (DAG) prefix merge for subagent trajectories

- **Status:** Layer 1 implemented (trainer-side multi-segment merger in `rllm/trainer/verl/transform.py` + `rllm/trainer/tinker/transform.py`, with tests). Layer 2 (gateway multi-slot accumulator) proposed.
- **Related:** `design/gateway-dag-token-storage.md` (delta-chain token storage — this is its deferred §7 "full branching DAG", motivated by subagents); `PR_dev-multiturn-merged-rows.md` (the `merge_compression_ratio` metric); `rllm-model-gateway` `token_accumulator.py` / `proxy.py`; `rllm/harnesses/{opencode,cli_harness}.py`.
- **Scope:** how a multi-turn rollout whose turns do **not** all share one growing prefix (a subagent runs mid-rollout) is merged into training rows, and (proposed) how the gateway should tag such turns.

---

## Summary

`merge_compression_ratio = total_agent_steps / total_emitted_rows` measures how well a
trajectory's turns collapse into shared-prefix training rows (`= N` for a clean N-turn
cumulative rollout → 1 row; `= 1` when nothing merges). Under the **opencode** harness with
**subagents** the ratio collapses toward 1.

Root cause: opencode's `task`-tool subagent runs **inside the same process**, which points at a
single session-scoped gateway URL `/sessions/<uid>/v1`. The gateway derives session id purely
from that path, so every LLM call — parent agent *and* subagent — lands in **one** session /
**one** trajectory, interleaved in time. But a subagent turn is a *fresh conversation* with a
*different system prompt and tool set*, so it is **not** a byte-prefix-extension of the parent.
Both the gateway accumulator and the trainer transforms assume a **single** growing prefix, so
the interleaving `A1, B1, A2, B2, …` breaks merging at every switch.

The fix is the one the [gateway-dag-token-storage RFC §7](gateway-dag-token-storage.md) deferred:
maintain a **set of open prefix slots** (a forest / DAG of lineages) instead of one linear chain.
Each turn attaches to the deepest slot it prefix-extends; a turn that extends none opens a new
slot. Parent turns merge among themselves, subagent turns merge among themselves.

Two layers, independent and separately shippable:

- **Layer 1 — trainer transform (implemented here).** Directly restores the metric, works on
  already-stored `prompt_ids`, needs **no gateway change and no rerun**.
- **Layer 2 — gateway multi-slot accumulator (proposed).** Keeps generation drift-free across
  lineages and materializes the DAG (`parent_trace_id`), composing with the RFC's Phase-1
  delta-chain storage.

---

## 1. Why it breaks today

**One session per rollout, subagent shares it.** `rllm/engine/agentflow_engine.py` mints one
`uid` per rollout and hands opencode `base_url = /sessions/<uid>/v1`; the subagent uses the same
process/URL. `middleware.py` parses the sid from the path (there is no per-subagent header), so
all traces are stored under that one session and — because the CLI-harness episode has an empty
trajectory — assigned wholesale to **one** `Trajectory`, ordered by time
(`enrich_episode_with_traces`).

**Single-prefix assumption, two places:**

- **Gateway** `TokenAccumulator` holds one snapshot (`prev_prompt_ids`, `_prefix_fps`). A subagent
  turn → `_REL_PREFIX_CHANGED` → `reset()`; returning to the parent → `reset()` again. A reset
  storm, and each reset drops drift-free token state (next turn re-renders from chat text).
- **Trainer** `_process_trajectory` (verl) / `trajectory_to_datums` (tinker) keep one running
  `full_seq` and walk steps in order: a step that doesn't extend it **emits the row and reseeds**,
  discarding the prior prefix. So `A1,B1,A2,B2` → B1 doesn't extend A1; once A1 is emitted, A2
  can't rejoin it → one row per turn → ratio → 1.

## 2. Layer 1 — trainer multi-segment merger (implemented)

Replace the single running segment with a **list of open segments**. For each step:

1. Find the open segment whose `full_seq` is a byte-prefix of the step's `prompt_ids`; pick the
   **longest** match.
2. Match → extend that segment (append `delta_obs` mask-0, then action mask-1) — unchanged logic.
3. No match → open a **new** segment (a new lineage root).
4. At the end, emit **every** open segment as one row.

`A1,B1,A2,B2` → `{A1,A2}` + `{B1,B2}` = 2 rows, each fully merged.

**Why it's safe / contained:**

- **Byte-exact prefix match** is the same cumulative-extension invariant used today; merging is
  never lossy. Distinct lineages have distinct roots (different system prompt), so at most one
  segment matches — `longest` also keeps a genuine DAG branch attached to its nearest ancestor.
- **Backward compatible:** a single-lineage trajectory keeps exactly one open segment and emits
  one row — byte-identical to the previous behavior. Sequential non-prefix breaks (context reset
  with no later return) still produce the same row count.
- **Multi-row-per-trajectory is already supported.** `_emit` keys every row by `trajectory.uid`
  with the same broadcast scalar advantage; the batch builder and advantage join already handle
  it. Layer 1 only *reduces* row count (from `#turns` to `#lineages`), improving per-trajectory
  weight uniformity — it does not introduce a new code path downstream.
- **Cost** is `~O(#open_segments · prefix_len)` per step (length-guarded before the slice);
  `#open_segments` is tiny (parent + a few subagent types).
- The `merge_compression_ratio` / `steps_per_traj` metrics need no change — `total_emitted_rows`
  now counts `#lineages` and the ratio rises accordingly.

**Open question (not blocking):** whether all lineages of one trajectory should share gradient
weight (scale advantage by `1/#lineages`) or keep per-row weighting. Layer 1 works either way;
default keeps current behavior, revisit if it skews.

## 3. Layer 2 — gateway multi-slot accumulator (proposed)

This is the user's "dict of prefix tokens → new prefix takes a new slot → the trajectory becomes a
DAG," and the fix for the *generation-time* reset storm.

Replace the single-prefix `TokenAccumulator` with a **registry of live slots**, each carrying its
own `prev_prompt_ids` / `prev_completion_ids` / `_prefix_fps`. `plan_turn` matches an incoming
request against **all** slots (message-prefix match): extend the matching slot; if none matches,
**open a new slot instead of `reset()`**. `DUPLICATE`/replay is handled per-slot. A genuine
compaction (history shrank under an existing slot's snapshot) still resets that slot.

Wins beyond Layer 1:

- Generation stays **drift-free** (cumulative `/v1/completions`) for *both* the parent and each
  subagent lineage, instead of falling back to chat re-tokenization at every switch.
- The reset storm (and its log spam) disappears.
- Each trace can be tagged `parent_trace_id` = the tip of its slot's chain, composing directly
  with the RFC's Phase-1 delta-chain storage (`token_chain.py`). The DAG is then **materialized**,
  so a Phase-2 trainer transform could group by lineage from the stored parent pointers instead of
  re-deriving by byte-match (Layer 1 becomes an optimization, not a necessity).

Slot lifetime is session-local; `delete_session` drops the whole forest. Bounded by a max-slots
guard (log + fall back to reset on overflow) so a pathological session can't grow slots unbounded.

## 4. Relationship to the token-storage RFC

`design/gateway-dag-token-storage.md` implemented a **linear** delta chain (one accumulator; a
reset starts a new root) and explicitly deferred (§7) the **branching DAG** for "compaction
branches" and "cross-rollout prefix sharing." Subagents are exactly the branching case. Layer 2
generalizes that RFC's single chain into a forest keyed by prefix; the same machinery later
unlocks its other deferred win (G GRPO rollouts sharing one system prefix, stored once).

## 5. Testing

- **tinker** (`tests/unified_trainer/test_tinker_transform.py`):
  `test_interleaved_subagent_lineage_remerges` (parent–subagent–parent → 2 Datums, parent mask
  intact), `test_two_subagents_between_parent_turns` (parent + 2 subagents → 3 Datums). Existing
  sequential-break and single-lineage tests unchanged.
- **verl** (`tests/unified_trainer/test_verl_transform.py`):
  `test_interleaved_subagent_lineage_remerges` (→ 2 rows, parent mask `[1,1,0,1,1]`). The verl
  module isn't importable in the doc env; the merge algorithm was additionally validated with a
  standalone replica (interleaved → 2, single-lineage → 1, sequential non-prefix → 2, two
  subagents → 3).
- **Layer 2** (proposed): `TokenAccumulator` multi-slot routing across extend / new-slot / replay /
  reset; parent-pointer round-trip; parity of reconstructed prompts against full-mode output.

## 6. Impact

For a rollout with `S` subagent invocations interleaved through `T` parent turns, the trajectory
drops from ~`T + S` unmerged rows back to `1 + (#distinct subagent lineages)` rows —
`merge_compression_ratio` recovers from ~1 toward its no-subagent value. Not opencode-specific:
claude-code and any subagent-spawning CLI harness benefit.
