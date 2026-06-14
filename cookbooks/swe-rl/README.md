# SWE-RL

End-to-end agentic-RL recipe for software engineering: train on [rllm-swesmith](https://huggingface.co/datasets/kylemontgomery/swesmith-filtered) (filtered SWE-smith, ~4.7K bug-fix tasks across 105 Python repos), validate on [SWE-bench Verified](https://www.swebench.com/) (500 real GitHub issues). The agent harness is [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent); the base model is `Qwen/Qwen3.5-9B`.

This cookbook deliberately ships **no custom AgentFlow and no custom evaluator** — it's a thin wrapper around primitives that already live in `rllm/`. The flow is `mini-swe-agent` running as a CLI inside per-task sandboxes, and the evaluator is each task's own `tests/test.sh`. Both `rllm-swesmith` and `harbor:swebench-verified` ship that verifier with the dataset.

## Architecture

```
AgentTrainer.train()
  │
  ├── for each task: launch a sandbox (Docker / Daytona)
  │       │
  │       └── mini-swe-agent --yolo --task=<problem_statement>
  │             │   (multi-turn shell loop; each LLM call → gateway)
  │             │
  │             └── rLLM gateway routes to the trainer-hosted policy,
  │                  capturing the full trajectory (prompt + response
  │                  tokens + sampling params per turn).
  │
  └── verifier: tests/test.sh inside the sandbox
        │   pytest for rllm-swesmith;  SWE-bench harness for verified.
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
| `rllm-swesmith` | train (~4.7K) | `kylemontgomery/swesmith-filtered` | in-sandbox pytest (`tests/test.sh`) |
| `harbor:swebench-verified` | eval (500) | `princeton-nlp/SWE-bench_Verified` | official SWE-bench harness (F2P / P2P) |

Both materialize as Harbor-format task directories (`task.toml`, `instruction.md`, `environment/Dockerfile`, `tests/test.sh`).

## Training

### Tinker (single-machine, LoRA)

```bash
bash cookbooks/swe-rl/train_tinker.sh
```

Defaults: Qwen/Qwen3.5-9B + LoRA rank 32, GRPO with compact filtering, 64 parallel Daytona sandboxes, async rollout/training. Override anything via Hydra:

```bash
bash cookbooks/swe-rl/train_tinker.sh \
    model.name=Qwen/Qwen3-8B \
    rllm.workflow.n_parallel_tasks=32 \
    rllm.remote_runtime.harbor.environment_type=docker
```

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

The harness runs inside one sandbox per task. Pick a backend via `rllm.remote_runtime.harbor.environment_type`:

| Backend | Setup | Notes |
|---|---|---|
| `docker` | local | Fastest iteration; needs the Docker daemon and ~20 GB free disk. |
| `daytona` | `DAYTONA_API_KEY` | Default for training — scales to thousands of parallel sandboxes. |
| `modal` | `modal token new` | Per-task billing; good for one-shot eval. |

## Files

| File | Description |
|------|-------------|
| `prepare_data.py` | Pulls `rllm-swesmith` (train) and `harbor:swebench-verified` (eval) |
| `train.py` | Loads the two datasets, hands them to `AgentTrainer` |
| `train_tinker.sh` | Tinker backend — Qwen3.5-9B LoRA, GRPO + async, Daytona sandboxes |
| `train_verl.sh` | Verl backend — same recipe with vLLM + FSDP |
| `test.py` | Catalog wiring + harness import smoke tests |
| `pyproject.toml` | Cookbook metadata (no entry points — the harness is in-tree) |

## Why no custom flow or evaluator?

Other cookbooks in this repo (`finqa`, `math`, `deepcoder`, …) ship a custom AgentFlow because their workloads either fit in a single LLM turn or need bespoke tool wiring. SWE doesn't — the existing in-tree primitives already cover it:

- **`rllm.harnesses.mini_swe_agent`** is the agent. It exposes the `mini-swe-agent` CLI as an rLLM harness (installs in-sandbox on first run; reads the gateway URL from the env; logs to `/tmp/mini-swe-agent.log`).
- **Per-task `tests/test.sh`** is the evaluator. The sandbox-shell verifier kind (`rllm.eval.script_evaluator`) reads `/logs/verifier/reward.txt` and returns it as the RL reward. For `harbor:swebench-verified`, that script invokes the official SWE-bench harness; for `rllm-swesmith`, it runs pytest against the FAIL_TO_PASS suite.

The only thing this cookbook adds on top is the recipe: dataset pairing, sampling/optimizer hyperparams, and the `mini-swe-agent` harness selection. Forking `train_tinker.sh` is the place to start customizing.
