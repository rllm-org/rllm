# FrozenLake Agent

A multi-turn agent flow for rLLM that trains a model to navigate procedurally-generated FrozenLake puzzles end-to-end via the **AgentFlow protocol**, demonstrating gym-environment integration without the legacy `Agent + Environment + AgentExecutionEngine` stack.

## Overview

Each task is a randomly-generated `size x size` FrozenLake grid (with `S`tart, `G`oal, frozen tiles `_`, and `H`oles `O`). The agent reads the rendered grid each turn and outputs `Up`/`Down`/`Left`/`Right`. Reward is `1.0` if the player reaches the goal, `0.0` otherwise.

The map is regenerated deterministically from `(seed, size, p)` inside the flow, so the dataset stores only those parameters — no map serialization.

## Architecture

```
AgentFlow.run(task, config)
  │
  ├── generate_random_map(seed, size, p)   # deterministic, in-process
  ├── env = gymnasium.make("FrozenLake-v1", desc=…, is_slippery=…)
  └── Multi-turn loop (up to max_steps turns)
        │
        ├── LLM call via OpenAI(base_url=config.base_url)
        │     Model outputs reasoning + ```Up|Down|Left|Right```
        │
        ├── parse_action() → env.step(action)
        │
        └── Repeat until terminated (G or H), truncated, or max_steps
```

The evaluator just reads `episode.artifacts["won"]` set by the flow.

## Installation

```bash
uv pip install -e ".[tinker]"                          # rllm + tinker backend
uv pip install --no-deps -e cookbooks/frozenlake       # this cookbook (gymnasium pulled in transitively)
```

After installation:

```bash
rllm agent list      # should show "frozenlake"
```

## Dataset

The dataset is procedurally generated — there's nothing to download. Run once:

```bash
python cookbooks/frozenlake/prepare_data.py
# or, with custom sizes:
python cookbooks/frozenlake/prepare_data.py --train-size 5000 --test-size 200 --slippery
```

This registers `frozenlake/{train,test}` with `DatasetRegistry`.

## Eval (rllm CLI)

Once the cookbook is installed and the dataset is generated, evaluate a model on the test split with one command:

```bash
rllm eval frozenlake \
    --agent frozenlake \
    --evaluator frozenlake \
    --split test \
    --max-examples 20            # smoke-test on 20 puzzles before running the full split
```

`--agent` and `--evaluator` resolve via the entry points declared in `pyproject.toml`. Drop `--base-url` if you want the CLI to auto-start a proxy from your `rllm setup` config. Episode JSONs are written under `~/.rllm/eval_results/` for inspection (`rllm view`).

## Training (rllm CLI)

The simplest training entrypoint is the CLI, which mirrors the eval flags:

```bash
rllm train frozenlake \
    --agent frozenlake \
    --evaluator frozenlake \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --group-size 8 \
    --batch-size 32 \
    --lora-rank 32 \
    --epochs 1 \
    --val-freq 10
```

This dispatches through `AgentTrainer` with the tinker backend by default. For deeper customization (verl backend, full PPO knobs, megatron, …) use the shell scripts below.

## Training (shell scripts)

### Tinker (single-machine)

```bash
bash cookbooks/frozenlake/train_tinker.sh
```

Or directly via the Python API with Hydra overrides:

```bash
python cookbooks/frozenlake/train.py \
    rllm/backend=tinker \
    model.name=Qwen/Qwen3-4B-Instruct-2507 \
    training.group_size=8
```

### Verl (distributed GPU)

```bash
uv pip install -e ".[verl]"
bash scripts/install_megatron.sh <cu128|cu129|...>
bash cookbooks/frozenlake/train_verl.sh
```

## Tests

```bash
pytest cookbooks/frozenlake/test.py -v
```

## Files

| File | Description |
|------|-------------|
| `frozenlake_flow.py` | `frozenlake_flow` — multi-turn AgentFlow over a gym env |
| `evaluator.py` | `frozenlake_evaluator` — reward = won? 1 : 0 |
| `prepare_data.py` | Generate + register the train/test splits |
| `train.py` | Python API training script (Hydra config) |
| `train_tinker.sh` | Tinker backend — single-machine training |
| `train_verl.sh` | Verl backend — distributed multi-GPU training |
| `pyproject.toml` | Plugin metadata and entry points |
| `test.py` | Unit tests for map gen, action parsing, rendering, evaluator |

## Migration notes

This cookbook is the AgentFlow port of `examples/frozenlake/`, which uses the legacy `BaseAgent` + `Environment` + `AgentExecutionEngine` stack. The two are functionally equivalent on FrozenLake; the cookbook is fully self-contained — no imports from `rllm.agents`, `rllm.environments`, or `rllm.engine.agent_execution_engine` — so the legacy stack can be deleted without touching this directory.
