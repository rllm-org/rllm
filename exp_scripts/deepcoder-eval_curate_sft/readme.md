# DeepCoder Eval → Curate → SFT

This directory is a peripheral, benchmark-specific pipeline over rLLM's native
eval curation and SFT infrastructure.  It does not modify the DeepCoder flow,
verifier, reward, or rLLM/veRL training implementation.

## Directory and dependencies

```text
deepcoder-eval_curate_sft/
├── eval_rollouts.py       # collect k verifier-scored trajectories per task
├── curate_dataset.py      # coverage probe, filtering and leakage audit
├── run_eval.sh            # launch the base-model vLLM server and collection
├── run_eval_dual.sh       # formal two-GPU, two-shard collection
├── run_curate.sh          # build messages-only SFT parquet
├── run_curate_dual.sh     # merge and curate both formal shards
├── run_sft.sh             # launch native rLLM/veRL LoRA SFT
├── run_pipeline.sh        # fail-fast stage orchestrator
├── run_formal_dual.sh     # dual-shard formal collection → curation (no SFT)
├── validate_shards.py     # prove disjointness and complete task coverage
├── repair_attempt_episode_names.py # audited repair for the first smoke only
└── readme.md
```

This experiment directory contains only orchestration and audit code. It
intentionally reuses:

- `/home/yanan/agents/rllm` for the rLLM source tree, DeepCoder flow and SFT
  backend;
- sibling `deepcoder-run/runtime` for the registered 24,287/687 dataset;
- sibling `deepcoder-run/compatible_grader.py` for the verified forkserver
  wrapper around the unchanged deterministic verifier;
- sibling `finqa-grpo-run/env.sh` for the existing rLLM virtual environment.

Run the commands below from this directory. Generated rollouts, curated data,
logs and checkpoints are deliberately stored outside the Git repository.

## Frozen protocol

| Item | Value |
|---|---|
| Base model | Qwen3-4B-Instruct-2507 (`cdbee75f…`) |
| Source split | DeepCoder train (24,287 tasks) |
| Attempts/task | 8 |
| Durable flush | 32 source tasks per shard (256 rollouts) |
| Sampling | temperature 0.6, top-p 1.0, max completion 16,384 |
| Verifier | upstream deterministic DeepCoder verifier through `ForkserverEvaluator` |
| Task filter | `avg(is_correct) > 0` (at least one success out of eight) |
| Trajectory selection | shortest successful trajectories; `max_select_rollouts_cnt=1` by default |
| SFT | LoRA rank 32, veRL/FSDP, model-native `hf_template` masking |
| SFT maximum sequence | 32,768 tokens, including the complete conversation |

A single fixed request seed is intentionally not sent during stochastic
rollout generation: identical prompts with an identical backend seed can
collapse the eight attempts. Task subsampling remains reproducible with seed
1234. The formal SFT dataset itself has no additional validation holdout.

The data scope is fixed as follows:

- smoke: randomly sample 16–32 tasks from the train split with seed 1234;
- formal rollout/curation: use all 24,287 train tasks, with 8 attempts per task;
- formal SFT: train on every leak-free, in-budget trajectory retained by the
  stated difficulty filter and selection rule. There is no additional task
  subsampling after curation.

## Stages

The three stages can be invoked independently, or through the fail-fast
orchestrator:

```bash
MAX_EXAMPLES=32 RUN_NAME=smoke-k8 bash run_pipeline.sh all smoke-k8
```

That command is documented for the later two-GPU smoke; it has not been run.

Generate native rLLM eval artifacts (`results.json` and individual Episode
files):

```bash
MAX_EXAMPLES=1000 RUN_NAME=base-train-1000-k8 bash run_eval.sh
```

`MAX_EXAMPLES` is mandatory.  Use a positive integer for a bounded run.  The
full 24,287-task run is deliberately gated behind `MAX_EXAMPLES=all` because it
means 194,296 verifier-scored rollouts.

Formal rollout-and-curation command:

```bash
bash run_formal_dual.sh
```

The formal path runs two independent TP=1 vLLM servers: GPU0 receives source
indices `0,2,4,...`, GPU1 receives `1,3,5,...`. All eight attempts for a task
remain on the same shard. Before curation, `validate_shards.py` fails closed
unless the shards are disjoint and their union is exactly all 24,287 source
indices. Tensor parallelism is intentionally not used for this 4B model.
On `.16`, this orchestrator stops after curation by design. It never invokes
SFT; a later SFT experiment must call `run_sft.sh` explicitly on the chosen
machine and curated dataset or snapshot.

`run_dataset` only calls its episode callback after its entire input returns.
The peripheral collector therefore feeds it bounded 32-task chunks. At every
chunk boundary it first writes all native Episode JSON files, then atomically
replaces `results.json` and `progress.json`. Consequently a running formal job
has durable, internally consistent snapshots suitable for curation/SFT on a
different machine; raw in-flight tasks are never advertised in `results.json`.

Curate and audit:

```bash
bash run_curate.sh \
  /home/yanan/.deepcoder-sft-pipeline/eval_runs/base-train-1000-k8
```

The curation stage uses rLLM's `curate()` implementation, then:

- writes `coverage_probe.json`, aggregating all eight verifier outcomes per
  task into `0/8`, `1–7/8`, and `8/8` coverage classes plus the full success
  histogram and rollout-error rates;
- verifies every row maps to a source task;
- rejects exact leakage of `tests`, `ground_truth`, or `solutions` into chat
  messages;
- measures length with the actual Qwen tokenizer;
- removes conversations longer than 32,768 tokens rather than silently
  truncating away their final code;
- performs no additional SFT holdout or task subsampling: every retained
  trajectory is written to the training dataset;
- writes only the `messages` column to JSONL for SFT, plus a Parquet audit
  copy. JSONL is intentional: with the installed Pandas/PyArrow stack, nested
  Parquet arrays are read back as `numpy.ndarray`, while rLLM SFT requires
  `messages` to be a Python list.

`MAX_SELECT_ROLLOUTS_CNT` is an external curation parameter in the inclusive
range 1–8 and defaults to 1. If a task has `s` successful trajectories, the
curator takes the shortest `min(MAX_SELECT_ROLLOUTS_CNT, s)` successes. For
example, with six successes and a value of four, that task contributes four
SFT rows. Set it when running curation or the combined pipeline:

```bash
MAX_SELECT_ROLLOUTS_CNT=4 \
  bash run_pipeline.sh curate base-train-full-k8
```

Selected trajectories are not silently deduplicated, because doing so would
violate this per-task contribution rule. A selected trajectory may still be
excluded if its complete rendered conversation exceeds 32,768 tokens; that
reduction is recorded in the curation manifest.

Reward/correctness and curation statistics are audit metadata, not model input.
The raw per-Episode eval JSON files do retain the source `Task` metadata needed
by the verifier, including private tests/reference material. Treat that eval
directory as restricted benchmark state; only the audited `messages` parquet
crosses into SFT.

The primary coverage quantity requested for the experiment is
`zero_success_task_share`: the fraction of source tasks with no successful
trajectory among eight attempts. Those tasks cannot contribute a successful
demonstration and are excluded. Tasks with `1–7/8` and `8/8` successes are both
eligible. Under the default `MAX_SELECT_ROLLOUTS_CNT=1`, an `8/8` task still
contributes only its shortest successful trajectory.

Train after a smoke test and after the target two-GPU host is idle:

```bash
bash run_sft.sh \
  /home/yanan/.deepcoder-sft-pipeline/eval_runs/base-train-1000-k8-sft-data
```

Generated eval, dataset, server-log, and checkpoint artifacts live outside the
repository by default. On `.16`, the default artifact root is the machine-local
`/home/yanan/.deepcoder-sft-pipeline`; no NFS checkpoint path is used. Existing
DeepCoder `eval_base.py` output is not accepted
directly because it uses `result.json + episodes.jsonl`; this pipeline emits the
native per-Episode format required by rLLM curation.

## Live status and audit log (2026-09-18)

The first 32-task smoke has completed rollout collection on `.16`:

| Item | Result |
|---|---:|
| Tasks / attempts | 32 / 8 |
| Episodes | 256 / 256 |
| Correct episodes | 89 |
| Rollout errors | 0 |
| Mean correctness | 0.34765625 |
| Eval wall time | 23m 28s |
| Eval artifact size | 41 MiB |
| OOM | none |

Run directory:
`/home/yanan/.deepcoder-sft-pipeline-smoke-20260918/eval_runs/smoke32-k8`.

Curation then failed closed before SFT with:

```text
RuntimeError: curated task 'deepcoder_14441_7' cannot be mapped to source data
```

Root cause: rLLM's Episode store receives `Episode.task` as a dictionary for
this flow, but its filename helper only performs attribute access. It therefore
falls back to the attempt-qualified Episode id (`deepcoder_14441:7`), sanitized
as `deepcoder_14441_7`. Native curation groups by the filename task id, so this
would incorrectly form eight one-attempt groups instead of one eight-attempt
group. No SFT process was started and no model checkpoint was produced.

The local eval entry now writes the same native Episode JSON under filenames
derived from the immutable source task id and asserts the attempt order. The
one-off repair utility validates filename suffix, eval index, all eight slots,
and the embedded source `task_id` before it renames any smoke artifact. The
existing smoke must pass that repair, curation, leakage/length audit, and SFT
before the formal run is allowed to start.

The first post-repair curation correctly recovered 32 task groups and found 17
tasks with at least one success (15 tasks, or 46.875%, were 0/8). It also
exposed a second fail-closed audit defect before SFT: with the installed
Transformers version, `apply_chat_template(tokenize=True)` returns a
`BatchEncoding`; taking `len()` measured its two fields rather than sequence
tokens. The length audit reported the impossible value of 2 for every full
conversation. The audit now unwraps `input_ids` before measuring. The produced
messages themselves were complete, but that first curated dataset is archived
and must not be used because its overlength gate was ineffective.

The next SFT launch reached rLLM's input validation and stopped before model
loading because Parquet returned nested `messages` as `numpy.ndarray` rather
than `list`. No optimizer step/checkpoint was produced. Curation now emits a
JSONL training file (list semantics preserved) and keeps Parquet only as an
audit copy; the peripheral SFT launcher consumes JSONL. This avoids changing
rLLM core validation.

After switching the peripheral SFT input to JSONL, the two-step SFT smoke
completed successfully on both `.16` GPUs:

| Item | Step 1 | Step 2 |
|---|---:|---:|
| Loss | 0.130647 | 0.147958 |
| Gradient norm | 0.329597 | 0.465199 |
| Global tokens | 19,467 | 14,522 |
| Max allocated GPU memory/rank | 20.54 GiB | 20.54 GiB |
| Checkpoint | complete `global_step_1` | complete `global_step_2` |

Each checkpoint contains both rank model shards, optimizer shards, extra state,
dataloader state, Hugging Face config/tokenizer files, and LoRA metadata. There
were no OOMs or non-finite metrics. Smoke checkpoint root:
`/home/yanan/.deepcoder-checkpoints/deepcoder-self-sft-smoke32-k8`.

The bounded runtime test of the two-server/two-shard launcher and its
disjoint-union validator has passed.

That bounded dual-shard test generated all 32 requested episodes (four tasks,
eight attempts each; two tasks per GPU) with zero rollout errors. The eval
stages completed, but the wrapper then exposed a peripheral invocation bug:
it called bare `python` for `validate_shards.py` without activating the venv.
The wrapper now calls the explicit rLLM venv interpreter. Existing trajectories
remain valid; validator and joint-curation checks are rerun on those artifacts
without regeneration.

Operational note: do not rewrite a shell script while a process is currently
executing that script from NFS. An earlier live edit made Bash read a mixture of
old and new file offsets and caused a transient syntax error after the eval
stage. The checked-in scripts themselves pass syntax validation; future
changes must occur only between stage processes.

The formal dual-shard path passed its bounded runtime smoke: shard sizes were
2+2, overlap was zero, union was four, joint curation recovered four task
groups and 32 rollouts, and both successful tasks contributed one row. The full
24,287-task collection starts as a separate `eval → curate` job on `.16`; SFT
is intentionally excluded from that host's formal orchestrator.

A first attempt to start the full collector was stopped before any Episode was
written after a live audit found that upstream `run_dataset` defers its
completion callback until *all* supplied tasks return. Supplying all 24,287
tasks at once would therefore retain 194,296 Episodes in RAM and provide no
mid-run SFT data. That zero-artifact attempt is archived as
`/home/yanan/.deepcoder-sft-pipeline-formal-aborted-nondurable-20260918`.

The replacement bounded-chunk implementation was then tested with eight random
tasks split across two GPUs and `flush_tasks=2`. While the second chunks were
still running, each shard already exposed exactly 16 Episode files plus an
atomic `results.json` with 16 items and `progress.json` marked `running 2/4`.
A concurrent joint curation read exactly four complete task groups / 32
rollouts, emitted a valid one-row JSONL dataset, and saw no partial group. The
run then completed 64/64 rollouts, zero errors, shard sizes 4+4, zero overlap,
and union size eight. This is the acceptance evidence for incremental handoff.

### Current formal run

Started on `.16` on 2026-09-18 after all smoke gates passed:

| Item | Value |
|---|---|
| Orchestrator PID | `2548945` |
| Artifact root | `/home/yanan/.deepcoder-sft-pipeline-formal` |
| Shards | GPU0/port 8992 = shard 0; GPU1/port 8993 = shard 1 |
| Per-shard task count | 12,144 / 12,143 |
| Rollouts | 8 per task, 194,296 total |
| Durable interval | 32 tasks / 256 rollouts per shard |
| Terminal stage on `.16` | curation; no SFT |
| Main log | `formal_pipeline.log` |
| Progress | each run directory's `progress.json` |

At handoff, both independent vLLM servers were starting normally and no fault
signature had appeared. A consumer should wait for at least one durable flush,
then use the atomic `results.json` plus matching `episodes/` files; do not copy
`.tmp` files or infer completeness from console lines.

### Partial formal-run audit (2026-09-21)

The collector is no longer running. It stopped after the per-chunk local
gateway repeatedly attempted to bind a port already in use; shard 0's defining
failure is near `formal_pipeline.log:155439`:

```text
ERROR: [Errno 98] ... bind ... address already in use
TimeoutError: Gateway thread did not start within 30.0s
```

Shard 0 stopped at 2,432 tasks / 19,456 rollouts. Shard 1 continued alone and
later stopped at 5,856 tasks / 46,848 rollouts. `progress.json` still says
`running`, so process existence and logs—not that stale status field—determine
run liveness. No curation stage ran automatically and `curated/` does not yet
exist.

The last durable boundaries are nevertheless internally complete and usable:

| Check | Result |
|---|---:|
| Durable source tasks | 8,288 |
| Episode JSON / result items | 66,304 / 66,304 |
| Missing or extra episode indices | 0 |
| Incomplete 8-attempt groups | 0 |
| Inconsistent task IDs within groups | 0 |
| Cross-shard completed-task overlap | 0 |
| Rollout-level error Episodes | 78 / 66,304 (0.118%) |
| Tasks with at least one success | 4,360 / 8,288 (52.61%) |
| Tasks with zero successes | 3,928 / 8,288 (47.39%) |

Because full-split collection preserves source order before even/odd sharding,
the unequal stopping points make the 8,288-task union an unbalanced prefix:
shard 0 contributes 2,432 tasks while shard 1 contributes 5,856. It is suitable
for an explicitly exploratory "use all currently available experience" SFT,
but not a clean balanced comparison. A matched-frontier snapshot should retain
the first 2,432 task groups from each shard (4,864 tasks / 38,912 rollouts).
That balanced subset has 1,331 + 1,344 = 2,675 tasks with at least one success;
the two halves have nearly identical rollout-correct totals (7,556 vs 7,588).

In-memory native curation (`avg(is_correct)>0`, shortest success, at most one
row/task) emitted exactly 4,360 unique rows, with zero missing conversations or
dedup losses. Every row had system/user/assistant turns, a non-empty final
assistant message, and a fenced Python solution. Rendered lengths were p50
1,570, p90 3,123, p95 4,418, p99 8,111, max 17,065; zero exceeded 32,768.

The leak audit originally compared private references against the entire
conversation, including the sampled assistant answer. It therefore
false-positively rejected three
trivial successful solutions whose generated code exactly matches the short
reference solution (`deepcoder_2576`, `deepcoder_2675`, `deepcoder_10609`). A
separate all-row audit found **0/4,360 private matches in policy inputs**
(system/user) and exactly those three matches only in assistant outputs. The
peripheral curator now applies private-reference checks only to non-assistant
policy inputs, while retaining provenance that every assistant label came from
a verifier-scored rollout. This is an input-leakage correction only; rLLM core
is unchanged.

The user selected the full currently available snapshot for the first SFT:
4,360 shortest successful trajectories from all 8,288 durable tasks. This is
explicitly an exploratory unbalanced snapshot, not the matched-frontier
protocol described above.

### Formal SFT attempts

The first full-data SFT attempt used two GPUs, one epoch, global batch 32, and
`max_length=32768`. Step 1 completed (loss 0.130320, gradient norm 0.422332),
but both ranks OOMed during step 2: they held roughly 40–42 GiB and requested an
additional 8.5–9.0 GiB. No checkpoint was produced. The failure directory and
log remain available for audit and are not resumed.

The approved restart protocol is four epochs, global batch 32,
`max_length=18000`, and `save_freq=136` (one checkpoint per epoch). The curated
maximum is 17,065 tokens, so this lower runtime ceiling does not truncate any
current training row. `MAX_LENGTH` is an explicit peripheral launcher variable;
its default remains 32,768 for other runs.

The restart was launched on `.16` at 2026-09-21 11:21 EDT. Its output directory
is `/home/yanan/.deepcoder-checkpoints/deepcoder-self-sft-current4360-r32-e4-len18000`
and its log is
`/home/yanan/.deepcoder-sft-pipeline-formal/sft-current4360-e4-len18000.log`.
The resolved trainer configuration reports 4,360 examples, batch 32, 136 steps
per epoch, 544 total steps, `max_length=18000`, and checkpoints every 136 steps.
The first three optimizer steps completed without OOM or numerical faults:

| Step | Loss | Gradient norm | Global tokens | Peak allocated GPU memory |
|---:|---:|---:|---:|---:|
| 1 | 0.130320 | 0.428679 | 57,162 | 24.70 GiB |
| 2 | 0.136775 | 0.480399 | 61,709 | 29.63 GiB |
| 3 | 0.126012 | 0.402832 | 60,101 | 29.68 GiB |

At this early acceptance point both GPUs were at 100% utilization and used
about 32.7/33.4 GiB of 49.1 GiB. This clears the exact step-2 failure boundary
from the 32,768-token attempt, but does not replace continued monitoring through
the first epoch and its checkpoint.

The four-epoch restart completed successfully on 2026-09-21 at approximately
13:38 EDT. It reached step 544 with no OOM, traceback, NCCL fault, or non-finite
metric. Checkpoints were written at steps 136, 272, 408, and 544; the checkpoint
tracker contains `544`, and the final checkpoint includes both model shards,
both optimizer shards, both dataloader/extra-state files, the Hugging Face
configuration/tokenizer directory, and LoRA metadata. The final reported loss
was 0.0935 with gradient norm 0.15689. Peak allocated/reserved GPU memory was
34.69/36.19 GiB. No validation was run by design (`Final validation metrics:
None`). Both GPUs were released after completion.

## Smoke acceptance sequence

On idle `.16` GPUs:

1. Generate 8 attempts for 16–32 randomly selected train tasks.
2. Confirm at least one task passes the `avg(is_correct) > 0` filter.
3. Inspect `coverage_probe.json`, the curation manifest, and several messages
   manually.
4. Run two SFT optimizer steps on both GPUs at `max_length=32768`.
5. Check peak memory, finite loss/gradients, assistant-only masking, checkpoint
   completeness, and absence of private verifier content.
6. Only then choose the formal task count and launch a full SFT run.
