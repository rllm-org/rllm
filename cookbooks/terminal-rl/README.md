# Terminal-RL

End-to-end agentic-RL recipe for terminal agents: train on **a local set of
Harbor-format terminal-agent tasks** that you provide (a `.tar.zst` of task
directories) and validate on
[Terminal-Bench](https://www.tbench.ai) pulled from the Harbor registry. The
agent harness defaults to Harbor's
[Terminus-2](https://www.harborframework.com/docs/agents/terminus-2) (a
tmux-driven mono-tool terminal agent), and the Fireworks GLM-5.2 launcher can
also select the pinned OpenCode CLI. The base model is chosen by the launcher.

This cookbook ships **no custom AgentFlow and no custom evaluator** — it's a thin
wrapper around primitives that already live in `rllm/`. The selected CLI agent
runs *inside* each task's sandbox, and the evaluator is each task's own
`tests/test.sh`. Both the training tasks and the Terminal-Bench eval tasks ship
that verifier with the dataset. This is the same machinery as the
[Terminal-Bench eval cookbook](../../docs/cookbooks/terminal_bench.mdx), packaged
as a versioned, installable *training* recipe (the sibling of `cookbooks/swe-rl`).

## Architecture

```
AgentTrainer.train()
  │
  ├── for each task: launch a sandbox (Modal / Daytona / Docker)
  │       │
  │       └── selected CLI harness runs IN the sandbox
  │             │   (multi-turn agent loop; each LLM call → gateway)
  │             │
  │             └── rLLM gateway routes to the trainer-hosted policy,
  │                  capturing the full trajectory (prompt + response
  │                  tokens + sampling params per turn).
  │
  └── verifier: tests/test.sh inside the sandbox
        │   writes 1.0 / 0.0 to /logs/verifier/reward.txt
        │
        └──  →  RL reward signal
```

The trainer never parses tool calls or model outputs directly. The agent harness
owns the action loop; the gateway owns the trajectory; the in-sandbox verifier
owns the reward.

## Installation

```bash
uv pip install -e ".[tinker,harbor]"                  # rllm + tinker backend + harbor
uv pip install --no-deps -e cookbooks/terminal-rl     # this cookbook (registers prepare_data)
```

The `harbor` extra lets the CLI resolve `harbor:` dataset names and ships the
Terminus-2 agent code. Terminus-2 itself (an isolated Python 3.12 venv with
`harbor` + tmux) is installed automatically into each task sandbox on first run —
no host-side agent install required.

## Datasets

```bash
python cookbooks/terminal-rl/prepare_data.py
# or, faster smoke run:
python cookbooks/terminal-rl/prepare_data.py --train-limit 50
```

This prepares:

| Dataset | Role | Source | Verifier |
|---|---|---|---|
| `tb-opus-pass/train` | train (all 1,200 archive tasks) | local `.tar.zst` or `.zip` archive (set via `TB_TRAIN_TARBALL`) | in-sandbox `tests/test.sh` → `/logs/verifier/reward.txt` |
| `terminal-bench@2.0/default` | existing debug/comparison eval (89) | `harbor:terminal-bench@2.0` | in-sandbox `tests/test.sh` → `/logs/verifier/reward.txt` |
| `terminal-bench@2.1/midtest` | production periodic eval (fixed 8) | deterministic subset of pinned Terminal-Bench 2.1 | in-sandbox `tests/test.sh` → `/logs/verifier/reward.txt` |
| `terminal-bench@2.1/default` | production boundary benchmark (89) | `harbor:terminal-bench/terminal-bench-2-1@6` | in-sandbox `tests/test.sh` → `/logs/verifier/reward.txt` |

All materialize as Harbor-format task rows (each row points at a task directory
holding `task.toml`, `instruction.md`, prebuilt `docker_image`, and
`tests/test.sh`). The training tarball is extracted once under the rLLM datasets
dir and each task directory becomes one row. The eight-task mid-test is external
Terminal-Bench 2.1 data; it is not carved out of `tb-opus-pass`, so all 1,200
internal tasks remain available for training.

Terminal-Bench 2.1 is pinned to immutable Harbor package revision `6`. The
periodic subset is selected deterministically by task ID; reproduce or change
it with:

```bash
python cookbooks/terminal-rl/prepare_data.py \
  --midtest-size 8 \
  --midtest-seed 20260723
```

Point `TB_TRAIN_TARBALL` at your training tarball (or pass `--tarball`); it
extracts on first run and is a no-op thereafter.

## Training

### Tinker (single-machine, LoRA)

```bash
bash cookbooks/terminal-rl/train_tinker.sh
```

Defaults: Qwen/Qwen3.5-4B + LoRA rank 32, GRPO with compact filtering, 128
parallel Modal sandboxes, async rollout/training. Override anything via Hydra:

```bash
TERMINAL_SANDBOX_BACKEND=docker bash cookbooks/terminal-rl/train_tinker.sh \
    model.name=Qwen/Qwen3-8B \
    rllm.workflow.n_parallel_tasks=32
```

For a simpler on-policy loop (generate a full batch, then one optimizer step —
easier to debug), use the synchronous variant:

```bash
bash cookbooks/terminal-rl/train_tinker_sync.sh
```

It drops `async_training` and uses a real `data.train_batch_size` (default 16;
effective batch = `train_batch_size × group_size`).

### Verl (distributed GPU)

```bash
uv pip install -e ".[verl,harbor]"
bash scripts/install_megatron.sh <cu128|cu129|...>
bash cookbooks/terminal-rl/train_verl.sh
```

vLLM rollouts + FSDP/Megatron training. Sandboxes still run Terminus-2 — only the
trainer hosting changes.

### Fireworks (managed, LoRA)

```bash
uv pip install -e ".[fireworks,harbor]"
export FIREWORKS_API_KEY=...
bash cookbooks/terminal-rl/train_fireworks.sh
```

Same async GRPO + compact-filtering recipe as `train_tinker.sh`, but the trainer
job and inference deployment are provisioned on Fireworks at startup and torn
down on shutdown. Defaults to `accounts/fireworks/models/qwen3p5-9b` + LoRA rank
32 on the `qwen3p5-9b-256k-lora` training shape (Fireworks ships a 3.5-9B LoRA
shape but no 3.5-4B; swap `model.name` / `model.tokenizer_model` /
`fireworks_config.policy_trainer_shape_id` together to change it — see
[`docs/backends/fireworks.mdx`](../../docs/backends/fireworks.mdx)). The
synchronous (on-policy) variant is `train_fireworks_sync.sh`.

### Fireworks GLM-5.2 four-run comparison

`train_fireworks_glm5p2.sh` defines the validated comparison matrix:

| Mode | Training shape | LoRA rank | Learning rate |
|---|---|---:|---:|
| `lora` | `accounts/fireworks/trainingShapes/glm-5p2-200k-lora` | 128 | `2e-5` |
| `full` | `accounts/fireworks/trainingShapes/glm-5p2-200k` | 0 | `1e-6` |

Both modes use `accounts/fireworks/models/glm-5p2-fp8`, train on
`tb-opus-pass`, and evaluate on all 89 `terminal-bench@2.0` tasks. The
comparison profiles request one policy-trainer replica plus one
rollout-deployment replica. The two supported harnesses are `opencode` and
`terminus-2`. Region placement is explicit at launch through
`TB_TRAINER_REGION`; the cookbook does not hard-code a client-side default.
The GLM launcher uses strict synchronous on-policy updates: rollout collection
and training do not overlap, and the next rollout starts only after the updated
weights finish hot-loading.

Install the Fireworks and Harbor dependencies, then register the standalone
debug set and the full training set. The full archive may be either `.tar.zst`
or `.zip`:

```bash
uv pip install -e ".[fireworks,harbor]"
uv pip install --no-deps -e cookbooks/terminal-rl

python cookbooks/terminal-rl/prepare_data.py \
  --debug-only --tarball /path/to/tb_v2_debug_tasks.tar.zst
python cookbooks/terminal-rl/prepare_data.py \
  --tarball /path/to/tb_v2_opus_pass.zip
```

Export credentials without putting their values in source files or command
arguments. For the internal four-run comparison, `FIREWORKS_API_KEY` must
belong to the intended human creator in the `training` account. The key selects
both the account and the audit identity recorded in each resource's
`Created By` field; merely selecting account `training` does not make the
signed-in human the creator. The launcher does not require a `firectl` profile
flag. In particular, do not add `-p fw-prod`; it is not a valid option for this
launch path.

For an internal admin launch, fail closed by matching the actual exported key
against server-side key metadata before starting a detached process. Do not use
`firectl whoami --api-key ...` for this check: current `firectl` reads the local
ID token for `whoami`, so that output does not identify the supplied API key.

```bash
expected_creator="your-email@fireworks.ai"
actual_creator="$(
  firectl-admin -a training api-key list --all-users -o json |
    jq -r '
      [.[] |
       .prefix as $prefix |
       select(env.FIREWORKS_API_KEY | startswith($prefix)) |
       .email][0] // empty
    '
)"
test "$actual_creator" = "$expected_creator" || {
  echo "FIREWORKS_API_KEY belongs to '$actual_creator', expected '$expected_creator'" >&2
  exit 1
}
```

```bash
export FIREWORKS_API_KEY=...
export WANDB_API_KEY=...
export WANDB_ENTITY=...
export TB_STATE_ROOT="/shared/${USER}/rllm-terminal-rl-glm5p2"
export TB_TRAINER_REGION=AP_MALAYSIA_2
```

Run all four debug combinations first. These use the eight-task debug split,
one optimizer batch, and two validation tasks:

```bash
bash cookbooks/terminal-rl/train_fireworks_glm5p2.sh lora opencode debug
bash cookbooks/terminal-rl/train_fireworks_glm5p2.sh lora terminus-2 debug
bash cookbooks/terminal-rl/train_fireworks_glm5p2.sh full opencode debug
bash cookbooks/terminal-rl/train_fireworks_glm5p2.sh full terminus-2 debug
```

After all four debug runs finish successfully and emit W&B metrics, launch the
full matrix in parallel. Each run starts a four-worker local gateway, so base
ports must be at least five ports apart; the example uses ten-port spacing.

```bash
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
comparison="glm5p2-tb-${run_stamp}"
mkdir -p "${TB_STATE_ROOT}/logs"

launch_glm5p2_run() {
  mode="$1"
  harness="$2"
  port="$3"
  run_name="${comparison}-train-${mode}-${harness}"
  log_path="${TB_STATE_ROOT}/logs/${run_name}.log"

  nohup env \
    TB_RUN_STAMP="${run_stamp}" \
    TB_COMPARISON_ID="${comparison}" \
    TB_RUN_NAME="${run_name}" \
    RLLM_GATEWAY_PORT="${port}" \
    bash cookbooks/terminal-rl/train_fireworks_glm5p2.sh \
      "${mode}" "${harness}" train \
      >"${log_path}" 2>&1 &
  echo "$! ${run_name} ${log_path}"
}

launch_glm5p2_run lora opencode 9400
launch_glm5p2_run lora terminus-2 9410
launch_glm5p2_run full opencode 9420
launch_glm5p2_run full terminus-2 9430
```

All four runs share `WANDB_RUN_GROUP=${comparison}` and use distinct run names,
job types, tags, gateway ports, and generated deployment IDs. The training
phase evaluates every 50 optimizer steps and once more at the end of the epoch.
Inspect the printed trainer job and deployment IDs plus the four log files
before detaching from the host.

### Full-parameter GLM-5.2 OpenCode production run

The production profile is a separate, guarded launch contract:

- full-parameter `accounts/fireworks/trainingShapes/glm-5p2-200k` with LoRA
  rank `0`
- OpenCode harness
- all 1,200 `tb-opus-pass/train` tasks for training
- one training epoch
- trainer sequence length resolved from the selected shape (`204736` for the
  current GLM-5.2 200k full-parameter and LoRA shapes)
- compact filtering enabled for invalid or infrastructure-failed trajectories;
  an agent wall-clock timeout remains a valid, partially graded RL outcome
- strict synchronous on-policy GRPO with a fixed eight prompt groups × 16
  rollouts (128 trajectories) per optimizer step
- original group-standardized GRPO advantages and a PPO-clipped policy loss
- one forward/backward pass over the complete optimizer batch, followed by one
  optimizer step and an awaited deployment hot-load before new rollouts
- all 89 `terminal-bench@2.1/default` tasks at step 0, every 10 optimizer
  steps, and final weights
- no separate boundary-benchmark invocation, so step 0 evaluates the full set
  exactly once
- four policy-trainer replicas and twelve rollout-deployment replicas
- explicit `AP_MALAYSIA_2` trainer placement

Prepare the full training archive and evaluation dataset:

```bash
RLLM_HOME="${TB_STATE_ROOT}/state" \
python cookbooks/terminal-rl/prepare_data.py \
  --tarball /path/to/tb_v2_opus_pass.zip
```

Then launch. The script rejects `production` with any mode/harness other than
`full opencode`. It does not use or accept a `firectl -p fw-prod` profile.

```bash
export FIREWORKS_API_KEY=...  # key for the training account
export WANDB_API_KEY=...
export WANDB_ENTITY=...
export TB_STATE_ROOT="/shared/${USER}/rllm-terminal-rl-glm5p2"
export TB_TRAINER_REGION=AP_MALAYSIA_2
export TB_TRAINER_REPLICAS=4
export TB_ROLLOUT_REPLICAS=12
export TB_TRAIN_GROUPS_PER_STEP=8
export TB_RUN_NAME="glm5p2-full-opencode-tb21-production"

bash cookbooks/terminal-rl/train_fireworks_glm5p2.sh \
  full opencode production
```

The 4+12 replica values are also production defaults. The full-parameter
training shape uses two physical nodes per logical trainer replica, while each
rollout replica uses one node, so this requests `4 × 2 + 12 × 1 = 20` nodes.
Every production evaluation logs under `val/*`, giving one directly comparable
89-task curve from step 0 through the final checkpoint. If the last optimizer
step is itself a multiple of 10, the trainer recognizes that the final policy
was already validated and does not run the same 89 tasks a second time.
Use `val/reward/mean` as the fixed-suite learning curve: it includes all 89
attempts in the denominator and is accompanied by
`val/reward/{num_correct,num_episodes}`. Role-specific metrics such as
`val/reward/opencode/mean` describe only trajectories that remained scorable
after compact filtering and are diagnostic rather than the benchmark score.

The production profile keeps uniform-reward GRPO groups in the fixed optimizer
batch. For a prompt with 16 identical rewards, group-relative standardization
gives every trajectory zero advantage, so it contributes no policy-gradient
numerator without changing the sampled task distribution or the batch
denominator. Compact filtering remains separate and removes only trajectories
whose reward is not a trustworthy training signal.

Before step-0 evaluation, the trainer saves and hot-loads its initial policy
(including a newly initialized LoRA adapter) and waits for deployment warmup.
The same awaited hot-load barrier runs after every optimizer step.

Before spending full-parameter capacity, run the matching LoRA sanity profile.
It keeps the same training data, step-0 boundary evaluation, and four-plus-four
replica layout, but uses the rank-128 LoRA shape and stops after one optimizer
batch by default. The step-0 89-task benchmark already exercises the evaluation
stack, so this smoke test disables the roughly 20-minute periodic mid-test and
does not repeat the boundary benchmark at the end:

```bash
export TB_TRAINER_REGION=AP_MALAYSIA_2
export TB_TRAINER_REPLICAS=4
export TB_ROLLOUT_REPLICAS=4
export TB_RUN_NAME="glm5p2-lora-opencode-tb21-sanity"

bash cookbooks/terminal-rl/train_fireworks_glm5p2.sh \
  lora opencode sanity
```

### ECHO (train on environment feedback)

[ECHO](https://arxiv.org/abs/2605.24517) adds a cross-entropy loss on the
environment-observation tokens (the terminal/tool output) that the policy
already conditions on but GRPO never trains. A terminal agent is the ideal case
for it: rollouts are dominated by terminal output, and Terminal-Bench is hard
enough that many rollouts fail — ECHO turns every rollout, including the
failures, into dense supervision at no extra rollout or forward-pass cost. It
uses GRPO's advantages unchanged; the only difference is the extra loss term.

Flip GRPO → ECHO with one override on any backend (verl / tinker / fireworks):

```bash
# tinker (async or sync), verl, or fireworks — same switch:
bash cookbooks/terminal-rl/train_tinker.sh    rllm.algorithm.adv_estimator=echo
bash cookbooks/terminal-rl/train_verl.sh      algorithm.adv_estimator=echo
bash cookbooks/terminal-rl/train_fireworks.sh rllm.algorithm.adv_estimator=echo
```

`adv_estimator=echo` defaults the env-loss weight λ to the paper's 0.05. Tune it
explicitly with `rllm.algorithm.env_loss_coef=<λ>` (productive range 0.01–0.05;
`0.0` reproduces plain GRPO). It is implemented as an `env_prediction`
[auxiliary loss](../../design/auxiliary-losses.md); watch
`actor/aux_env_prediction_loss` (verl) / `train/aux_*` (tinker, fireworks) to
confirm the environment-prediction loss is falling.

> On verl the env term shares GRPO's single forward pass (free, exact). On
> tinker/fireworks (managed training services with fixed server-side loss
> kernels) it is a second, gradient-accumulated `cross_entropy` pass over the
> same rollouts — no extra rollouts, but one extra backward. λ may need
> per-backend retuning since loss normalization differs across services.

## Evaluation (no training)

```bash
rllm eval harbor:terminal-bench@2.0 \
    --agent terminus2 --sandbox-backend modal \
    --max-tokens 4096 --temperature 0.7 \
    --max-examples 20
```

Per-task results land in `~/.rllm/eval_results/`; aggregated resolve rate is
printed at the end. `rllm view` opens the per-task trajectory UI. See the
[Terminal-Bench eval cookbook](../../docs/cookbooks/terminal_bench.mdx) for the
full benchmark run (snapshots, pass@k, sandbox lifetimes).

## Sandbox backend

Training uses rLLM's own `SandboxedAgentFlow` path (`AgentFlowEngine`) — not the
remote Harbor runtime. Terminus-2 runs inside one sandbox per task, created by
`SandboxTaskHooks`. Pick a backend via the `TERMINAL_SANDBOX_BACKEND` env var:

| Backend | Setup | Notes |
|---|---|---|
| `modal` | `pip install modal` + `modal token new` | Default for training — per-task billing, scales to many parallel sandboxes. |
| `daytona` | `pip install daytona` + `DAYTONA_API_KEY` | Cloud sandboxes; scales to thousands in parallel. |
| `docker` | local | Fastest iteration; needs the Docker daemon and ~20 GB free disk. |

The scripts set two timeouts that must stay ordered — **`RLLM_SANDBOX_TIMEOUT_S`
(sandbox lifetime, default 2400s / 40 min) > `RLLM_HARNESS_RUN_TIMEOUT_S` (agent
run cap, default 1800s / 30 min)**. `RLLM_SANDBOX_TIMEOUT_S` is provider-agnostic
(seconds) — every backend honors it (Modal as a hard lifetime, Daytona as an idle
auto-stop, converted to minutes); the old `RLLM_MODAL_SANDBOX_TIMEOUT_S` remains a
deprecated alias. The agent cap is the knob that actually
bounds a rollout's duration/cost; the sandbox lifetime is a *ceiling*, not a
fixed duration — a sandbox is torn down as soon as its rollout + verifier
finish, so a higher ceiling costs nothing for normal rollouts. The two clocks
start at different points (sandbox lifetime at **boot**, agent cap after
**setup**, ~1–3 min), and the per-task verifier can take up to 300s, so the
lifetime needs that margin above the agent cap. If you make them equal (or the
lifetime shorter), Modal reaps the longest rollouts *before* their verifier runs
and you get a storm of `NotFoundError: Sandbox has already shut down` (plus
`exit 137` on the command that was running when the axe fell) — those rollouts
then error out and get dropped instead of scored.

## Files

| File | Description |
|------|-------------|
| `prepare_data.py` | Extracts your local training tarball (train), pulls `harbor:terminal-bench@<ver>` (eval) |
| `train.py` | Loads the two datasets, hands them to `AgentTrainer` |
| `train_tinker.sh` | Tinker backend — Qwen3.5-4B LoRA, GRPO + async, Modal sandboxes |
| `train_tinker_sync.sh` | Tinker backend — synchronous (on-policy) variant, simpler for testing |
| `train_fireworks.sh` | Fireworks backend — Qwen3.5-9B LoRA, GRPO + async, managed trainer/deployment |
| `train_fireworks_glm5p2.sh` | Fireworks GLM-5.2 — LoRA/full × OpenCode/Terminus-2 debug and full-run matrix |
| `train_fireworks_sync.sh` | Fireworks backend — synchronous (on-policy) variant |
| `train_verl.sh` | Verl backend — same recipe with vLLM + FSDP |
| `test.py` | Harness/loader import + script-wiring smoke tests |
| `pyproject.toml` | Cookbook metadata (registers `prepare_data`) |

## Why no custom flow or evaluator?

Other cookbooks in this repo (`finqa`, `math`, `deepcoder`, …) ship a custom
AgentFlow because their workloads either fit in a single LLM turn or need bespoke
tool wiring. A terminal agent doesn't — the existing in-tree primitives already
cover it:

- **`rllm.harnesses.terminus2`** is the agent. It runs Harbor's Terminus-2
  *inside* the sandbox (installs an isolated Python 3.12 venv on first run; reads
  the gateway URL from the env; drives a tmux session locally).
- **Per-task `tests/test.sh`** is the evaluator. The sandbox-shell verifier kind
  reads `/logs/verifier/reward.txt` and returns it as the RL reward. Every
  Terminal-Bench task (train and eval) ships that script.

The only thing this cookbook adds on top is the recipe: dataset pairing,
sampling/optimizer hyperparams, and the `terminus2` harness selection. Forking
`train_tinker.sh` is the place to start customizing.
