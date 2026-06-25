# Terminal-RL

End-to-end agentic-RL recipe for terminal agents: train on **a local set of
Harbor-format terminal-agent tasks** that you provide (a `.tar.zst` of task
directories) and validate on
[Terminal-Bench](https://www.tbench.ai) pulled from the Harbor registry. The
agent harness is Harbor's [Terminus-2](https://www.harborframework.com/docs/agents/terminus-2)
(a tmux-driven mono-tool terminal agent); the base model is `Qwen/Qwen3.5-4B`.

This cookbook ships **no custom AgentFlow and no custom evaluator** — it's a thin
wrapper around primitives that already live in `rllm/`. The flow is Terminus-2
running *inside* each task's sandbox, and the evaluator is each task's own
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
  │       └── terminus2 driver runs IN the sandbox
  │             │   (multi-turn tmux loop; each LLM call → gateway)
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

This pulls:

| Dataset | Role | Source | Verifier |
|---|---|---|---|
| `tb-opus-pass` | train | local tarball (set via `TB_TRAIN_TARBALL`) | in-sandbox `tests/test.sh` → `/logs/verifier/reward.txt` |
| `terminal-bench@2.0` | eval (89) | `harbor:terminal-bench@2.0` | in-sandbox `tests/test.sh` → `/logs/verifier/reward.txt` |

Both materialize as Harbor-format task rows (each row points at a task directory
holding `task.toml`, `instruction.md`, prebuilt `docker_image`, and
`tests/test.sh`). The training tarball is extracted once under the rLLM datasets
dir and each task directory becomes one row.

**Eval version.** `TB_EVAL_VERSION` selects the Terminal-Bench eval version
(default `2.0`). The Harbor registry only publishes `2.0` today; once `2.1`
lands, switch with a single env var — `prepare_data.py` and `train.py` both read
it so the pulled and loaded dataset names stay in sync:

```bash
TB_EVAL_VERSION=2.1 python cookbooks/terminal-rl/prepare_data.py
TB_EVAL_VERSION=2.1 bash cookbooks/terminal-rl/train_tinker.sh
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
