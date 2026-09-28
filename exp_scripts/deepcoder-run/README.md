# DeepCoder: GRPO versus synchronous PRPO

Upstream `cookbooks/deepcoder/train.py`, `deepcoder_flow`, and
`deepcoder_evaluator` are reused without changes. The evaluator executes hidden
unit tests locally; no LLM judge. `prepare.sh` registers the full upstream
24,287 train / 687 test rows in an isolated shared registry. Test is loaded by
the upstream entrypoint but online validation is disabled in both namespaces.

| Setting | GRPO (.16) | PRPO (.24) |
| --- | ---: | ---: |
| GPUs | 2 | 2 |
| Task batch | 16 | 64 |
| Rollouts/task | 8 | 1 |
| Rollouts/update | 128 | 64 |
| Steps/epoch (upstream drop_last) | 1,517 | 379 |
| Max epochs | 30 | 30 |
| Save interval | 1,517 | 379 |

These are NOT compute-budget-matched arms: GRPO uses roughly eight times the
environment calls per epoch. Report environment calls alongside epochs and
wall time. A shared seed/shuffle ensures a common input dataset/order, not
bitwise identical generation across batch layouts/hardware.

Common: Qwen3-4B-Instruct-2507 local snapshot, LoRA rank/alpha 32, lr 1e-6,
temperature .6/top_p 1, prompt 8,192/response 16,384/model context 32,768,
seq-mean-token-mean loss, clip .2/.28, no KL, synchronous updates, TIS off.
Both machines reuse the same rLLM venv and FSDP2 implementation. Fused torch
kernels and bf16 base/offload follow the validated FinQA hardware setup and
avoid large logits materialization. The cookbook's eight-GPU layout is changed
to two GPUs per host; .24 additionally disables NCCL P2P (validated host issue).

## Eval → Curate → SFT

The independent rejection-sampling/SFT experiment now lives in the sibling
directory `../deepcoder-eval_curate_sft/`. It samples eight base-model attempts
per train task, keeps every task with at least one successful rollout,
selects the configured number of shortest successful trajectories, performs a
private-verifier-data leakage audit, and trains a Qwen3-4B-Instruct-2507 LoRA
with veRL at a 32,768-token maximum sequence length. See
`../deepcoder-eval_curate_sft/readme.md`; no SFT smoke or training run has been
performed yet.
PPO mini-batch retains the upstream nominal value 64 in both arms; the rLLM
backend derives one optimizer update per generation batch for these settings.

GPU utilization initially follows the user's .9 override (upstream .8).
`DEEPCODER_GPU_MEMORY_UTILIZATION` can explicitly override it; any hardware
fallback must be documented and applied equally to both arms. No silent retry.

`train.sh grpo|prpo` launches one arm. `--cfg job` resolves Hydra without GPU
training. `monitor.sh` records GPU/error/progress snapshots without killing or
restarting anything. Logs and checkpoints are under this directory; full data
under `runtime/`, which is ignored by git.

## Launch record (2026-09-15 EDT)

Full upstream data preparation first failed with Arrow
`Uncompressed data page size overflows INT32_MAX`. `prepare_full.py` retries the
same upstream preparation with process-local pandas parquet writer options:
row_group_size=32, use_dictionary=False, data_page_size=1MB. The registered
dataset retains all upstream columns, including private tests/solutions;
upstream flow exposes only `question` to the policy. Those other columns are
not skill/policy input. Total parquet storage, including verl companions, is
about 29GB. Registry confirms 24,287 train / 687 test rows.

Both arms launched at approximately 15:38 EDT. Initial trainer PIDs:
.16 GRPO 2170099, .24 PRPO 1396208. Local read-only monitor PID 1155975
(managed exec session 86443; the initial detached PID did not persist).
Startup/model loading is NOT evidence of a completed optimizer step.
GRPO task batch starts at 16; user authorizes an 8-task retry only after a
confirmed OOM. Each retry must use a fresh attempt rather than overwrite logs.

Storage warning: 30 epochs in both arms may need roughly 500GB of checkpoint
space if each is comparable to the 8.3GB FinQA checkpoints. Current shared
disk free space is below that; do not assume the 30-epoch ceiling is covered.

### Startup blocked: concurrent grading/fork (15:49 EDT)

GRPO initialized vLLM, synchronized weights and began sampling at utilization
.9 (about 46.6GB/card, no CUDA OOM). During batch 1, upstream grading repeatedly
raised `os.fork is unsafe while filelock is changing descriptor ownership`;
29 occurrences were logged before stopping. `code_reward.py` uses default
`multiprocessing.Manager()` and `Process()`, while AgentFlowEngine evaluates
in concurrent executor threads. Installed filelock's process-wide audit hook
rejects fork during descriptor transitions. Retries rerun the agent, so this
is not a benign warning or a model failure reward. No optimizer step/checkpoint
completed. Both process trees were stopped with SIGKILL; PRPO was still in
inference initialization. Logs remain as evidence. Do not resume this attempt.

At that point a peripheral forkserver fix had not been implemented; see the
subsequent verified restart below. Task batch=8 retry authorization is for OOM;
the observed failure was not OOM.

### Forkserver compatibility verified and restart (2026-09-15)

User authorized a process-start-only peripheral compatibility entrypoint.
`train_compatible.py` invokes the unchanged upstream Hydra main and replaces
only its evaluator object with `compatible_grader.ForkserverEvaluator`. That
adapter delegates to the original evaluator and rebinds only code_reward's
module-local multiprocessing/Manager references to a forkserver context.
Python's global start method, Ray/vLLM contexts, hidden tests, code extraction,
timeouts, reward and correctness remain unchanged. No core source file edits.

The server is prewarmed during construction and fresh-process deserialization,
before concurrent gateway filelock activity. It preloads only the CPU grader,
not the launcher/data/models. No fallback to unsafe fork.

Five tests passed on the workstation, .16 and .24 (standard-library runner;
no new dependencies): original-fork vs compatible reward agreement on 8 code
cases; deterministic filelock transition plus 32 concurrent score calls;
pickle/idempotence; fresh spawn-process cloudpickle reconstruction preserving
the worker's global spawn policy; timeout remains failure. Tests cover stdin,
functional calls, correct/wrong code, syntax/runtime errors, no fence and
last-fence selection. They do not prove bitwise equivalence of arbitrary
nondeterministic submitted programs.

New run suffix: `-forkserver`. GRPO PID 2191016 on .16 (batch16/n8), PRPO PID
1411195 on .24 (batch64/n1). Both initial logs confirm forkserver warmed.
Monitor PID 1163554 / managed session 61835. Old failed logs renamed to
`logs/grpo-before-forkserver-20260915.log` and
`logs/prpo-before-forkserver-20260915.log`; current logs are grpo.log/prpo.log.
Save intervals 1517/379, max epochs30, online evaluation disabled, all other
parameters unchanged. Still in startup; no optimizer-step acceptance yet.

### `.16` PRPO restart protocol (2026-09-16)

The stopped GRPO arm produced no checkpoint. A separate PRPO run uses batch
64, n=1, shuffled train data with seed 1234, 10 epochs, no online evaluation,
and saves every 100 optimizer steps (6,400 task attempts). CPU grading is
capped at 64 concurrent tasks and Ray workers start at nice 0. Before launch,
the identical 48-reference-solution probe was repeated three times each at
concurrency 4, 8, and 16 on `.16`; all 432 scores produced zero alarms, and
concurrency 16 was the fastest among those initial probes. Additional tests at
32 (5 x 48 scores) and 64 (5 x 96 scores) also produced zero alarms; concurrency
64 was selected to retain rollout throughput. On `.16`, checkpoints go to its
local ext4 path `/home/yanan/.deepcoder-checkpoints/deepcoder-prpo-b64-n1-16-c64-ni0-save100`;
that host's `/mnt/disk1t` is an unrelated root-owned 98-GiB filesystem.

### `.24` GRPO restart: full concurrency with verifier-timeout dropping (2026-09-18)

Three `.24` GRPO attempts were stopped and their empty checkpoint dirs removed; logs are archived under `/home/yanan/.deepcoder-checkpoints/archive-grpo24-attempts-20260917/` on `.24`.

| Attempt (all batch 16, n=8) | n_parallel_tasks | Real verifier timeouts | Step time | Outcome |
|---|---:|---:|---:|---|
| `...-24-save64` | 256 | 5.95% (213/3582) | ~250 s | stopped at step 27 |
| `...-24-p4-ni0-save64` | 4 | 1.47% (31/2108) | ~1,080 s | stopped at step 16 |
| `...-24-p1-ni0-save64` | 1 | 0.10% excl. batch 1 | ~2,905 s, GPU1 idle (gateway pins one task to one replica) | stopped at step 20 |

Reference: `.16` PRPO at n_parallel_tasks=64 shows 0.06% timeouts. Same reference solutions graded on both hosts (`grader_ab/`) reproduce the difference; `.24` stalls in ~5 s quanta under concurrent grading (kernel 7.0.0-28 vs 6.8.0-138 on `.16` is the only host difference found).

New run `deepcoder-grpo-b16-n8-24-c64-drop-save64` (launcher `train_24_grpo_c64_drop.sh`, PID in `/home/yanan/.deepcoder-checkpoints/*.pid` on `.24`): n_parallel_tasks=64, both GPUs serve rollouts, save every 64 steps (1,024 tasks). Verifier timeouts are no longer scored 0: `compatible_grader.ForkserverEvaluator(drop_timeouts=True)` (set by `DEEPCODER_DROP_TIMEOUTS=1` in `train_compatible.py`) raises `GraderTimeout` when the verifier metadata carries `Time Limit Exceeded` / `global timeout`; the AgentFlow engine retries the rollout up to `retry_limit=3` fresh samples and otherwise returns a `TerminationReason.ERROR` episode, which `rllm.compact_filtering.enable=true` + `mask_error=true` excludes from the GRPO group (`transform.py:_build_trajectory_groups`). Genuine slow-code timeouts are dropped too; their background rate is ~0.06%. The default `drop_timeouts=False` keeps the upstream behaviour for validation, standalone eval and the `.16` SFT pipeline. Unit and end-to-end tests: `_timed_out` classification, pickle round-trip of the flag, reference solution passes, infinite loop raises. Metric to watch: `batch/termination_reason/error` = dropped rollouts per step.

### `.24` GRPO: host-RAM OOM at step 233 and the fixes (2026-09-19)

At 18:50 EDT Ray's memory monitor killed the training workers: node memory 238.9 / 250.8 GB (95% threshold). The OOM report's top consumer was a **grading child** of the forkserver (cmdline `multiprocessing.forkserver ... main`, 148.7 GB RSS) — a model-generated program allocating without bound. `livecodebench.run_test` calls `reliability_guard()` with `maximum_memory_bytes=None`, so no address-space limit existed. Steady-state host RAM without that child is ~90 GB.

Fixes (peripheral only):
1. `grader_preload.py` sets `RLIMIT_AS` = 16 GB (env `GRADER_MAX_AS_GB`) and is added to `set_forkserver_preload` in `compatible_grader.py`; the forkserver server inherits it to every grading child. Test: reference solution still passes; a 40 GB allocation now fails in 0.2 s with `MemoryError -> Runtime Error` and no host-memory spike. Applies to every user of `compatible_grader` (.16 pipeline included).
2. `train_24_grpo_c64_drop.sh` adds `trainer.resume_mode=auto` (verified with `--cfg job` that this later override beats train.sh's `resume_mode=disable`), so crashes resume from `latest_checkpointed_iteration.txt`. The 18:57 relaunch resumed from `global_step_192` (41 steps ≈ 4.3 h lost instead of 233).

Earlier the same day: launches at 16:33 and 16:58 hung at FSDP init because the launcher lacked `NCCL_P2P_DISABLE=1` (train.sh sets it only in its prpo branch); the 17:20 run crashed at step 5 because `rllm.workflow.raise_on_error` defaulted to True and re-raised the third consecutive `GraderTimeout`; the 17:55 launch failed at Hydra parse (`+` prefix on an existing key). All logs are archived under `.24:/home/yanan/.deepcoder-checkpoints/archive-grpo24-attempts-20260917/`.
