# SWE-RL

End-to-end agentic-RL recipe for software engineering: train on [R2E-Gym](https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset) (R2E-Gym Subset, 4,578 bug-fix tasks across 12 Python repos), validate on [SWE-bench Verified](https://www.swebench.com/) (500 real GitHub issues). The agent harness is [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent); the base model is `Qwen/Qwen3.5-9B`.

This cookbook deliberately ships **no custom AgentFlow and no custom evaluator** — it's a thin wrapper around primitives that already live in `rllm/`. The flow is `mini-swe-agent` running as a CLI inside per-task sandboxes, and the evaluator is each task's own `tests/test.sh`. Both `r2egym` and `harbor:swebench-verified` ship that verifier with the dataset.

## Architecture

```
AgentTrainer.train()
  │
  ├── for each task: launch a sandbox (Modal / Daytona / Docker)
  │       │
  │       └── mini-swe-agent --yolo --task=<problem_statement>
  │             │   (multi-turn shell loop; each LLM call → gateway)
  │             │
  │             └── rLLM gateway routes to the trainer-hosted policy,
  │                  capturing the full trajectory (prompt + response
  │                  tokens + sampling params per turn).
  │
  └── verifier: tests/test.sh inside the sandbox
        │   run_tests.sh for r2egym;  SWE-bench harness for verified.
        │
        └── writes /logs/verifier/reward.txt  →  RL reward signal
```

The trainer never parses tool calls or model outputs directly. The agent harness owns the action loop; the gateway owns the trajectory; the in-sandbox verifier owns the reward. This is the same setup as `examples/harbor_swe/` — the cookbook packages it as a versioned, installable recipe.

## Installation

```bash
uv pip install -e ".[tinker]"                       # rllm + tinker backend
uv pip install --no-deps -e cookbooks/swe-rl        # this cookbook (registers prepare_data)
```

`mini-swe-agent` is installed automatically into each task sandbox on first run — no host-side install required.

## Datasets

```bash
python cookbooks/swe-rl/prepare_data.py
# or, faster smoke run:
python cookbooks/swe-rl/prepare_data.py --train-limit 50 --val-limit 20
```

This pulls:

| Dataset | Role | Source | Verifier |
|---|---|---|---|
| `r2egym` | train (4,578) | `R2E-Gym/R2E-Gym-Subset` | in-sandbox `run_tests.sh`, pytest-output equality (`tests/test.sh`) |
| `harbor:swebench-verified` | eval (500) | `princeton-nlp/SWE-bench_Verified` | official SWE-bench harness (F2P / P2P) |

Both materialize as Harbor-format task directories (`task.toml`, `instruction.md`, `environment/Dockerfile`, `tests/test.sh`).

## Training

### Tinker (single-machine, LoRA)

```bash
bash cookbooks/swe-rl/train_tinker.sh
```

Defaults: Qwen/Qwen3.5-9B + LoRA rank 32, GRPO with compact filtering, 64 parallel Modal sandboxes, async rollout/training. Override anything via Hydra:

```bash
SWE_SANDBOX_BACKEND=docker bash cookbooks/swe-rl/train_tinker.sh \
    model.name=Qwen/Qwen3-8B \
    rllm.workflow.n_parallel_tasks=32
```

For a simpler on-policy loop (generate a full batch, then one optimizer step — easier to debug), use the synchronous variant:

```bash
bash cookbooks/swe-rl/train_tinker_sync.sh
```

It drops `async_training` and uses a real `data.train_batch_size` (default 4; effective batch = `train_batch_size × group_size`).

### Verl (distributed GPU)

```bash
uv pip install -e ".[verl]"
bash scripts/install_megatron.sh <cu128|cu129|...>
bash cookbooks/swe-rl/train_verl.sh
```

vLLM rollouts + FSDP/Megatron training. Sandboxes still run mini-swe-agent — only the trainer hosting changes.

## Evaluation (no training)

```bash
rllm eval harbor:swebench-verified \
    --agent mini-swe-agent \
    --model Qwen/Qwen3.5-9B \
    --base-url http://localhost:8000/v1 \
    --max-examples 20
```

Per-task results land in `~/.rllm/eval_results/`; aggregated resolve rate is printed at the end. `rllm view` opens the per-task trajectory UI.

## Sandbox backend

Training uses rLLM's own `SandboxedAgentFlow` path (`AgentFlowEngine`) — not the
remote Harbor runtime. `mini-swe-agent` runs inside one sandbox per task, created
by `SandboxTaskHooks`. Pick a backend via the `SWE_SANDBOX_BACKEND` env var:

| Backend | Setup | Notes |
|---|---|---|
| `modal` | `pip install modal` + `modal token new` | Default for training — per-task billing, scales to many parallel sandboxes. |
| `daytona` | `pip install daytona` + `DAYTONA_API_KEY` | Cloud sandboxes; scales to thousands in parallel. |
| `docker` | local | Fastest iteration; needs the Docker daemon and ~20 GB free disk. |

## Files

| File | Description |
|------|-------------|
| `prepare_data.py` | Pulls `r2egym` (train) and `harbor:swebench-verified` (eval) |
| `train.py` | Loads the two datasets, hands them to `AgentTrainer` |
| `train_tinker.sh` | Tinker backend — Qwen3.5-9B LoRA, GRPO + async, Modal sandboxes |
| `train_tinker_sync.sh` | Tinker backend — synchronous (on-policy) variant, simpler for testing |
| `train_verl.sh` | Verl backend — same recipe with vLLM + FSDP |
| `test.py` | Catalog wiring + harness import smoke tests |
| `pyproject.toml` | Cookbook metadata (no entry points — the harness is in-tree) |

## Why no custom flow or evaluator?

Other cookbooks in this repo (`finqa`, `math`, `deepcoder`, …) ship a custom AgentFlow because their workloads either fit in a single LLM turn or need bespoke tool wiring. SWE doesn't — the existing in-tree primitives already cover it:

- **`rllm.harnesses.mini_swe_agent`** is the agent. It exposes the `mini-swe-agent` CLI as an rLLM harness (installs in-sandbox on first run; reads the gateway URL from the env; logs to `/tmp/mini-swe-agent.log`).
- **Per-task `tests/test.sh`** is the evaluator. The sandbox-shell verifier kind (`rllm.eval.script_evaluator`) reads `/logs/verifier/reward.txt` and returns it as the RL reward. For `harbor:swebench-verified`, that script invokes the official SWE-bench harness; for `r2egym`, it runs the image's own `/testbed/run_tests.sh` and scores 1.0 iff the pytest output matches the row's expected output.

The only thing this cookbook adds on top is the recipe: dataset pairing, sampling/optimizer hyperparams, and the `mini-swe-agent` harness selection. Forking `train_tinker.sh` is the place to start customizing.
