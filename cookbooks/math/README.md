# Math Agent

A single-turn math agent for rLLM that solves competition / textbook math problems via the **AgentFlow protocol**, replacing the four legacy `examples/{deepscaler, simple_math, gsm8k_lora, math_tinker}/` examples (all of which used `MathAgent` + `SingleTurnEnvironment` + `AgentExecutionEngine`).

This is the no-tool counterpart to [`cookbooks/math_tool_agent/`](../math_tool_agent/), which uses a calculator tool and multi-turn reasoning. Use `math` for chain-of-thought-only training; use `math_tool_agent` if you want the model to learn explicit tool use.

## Overview

Each task is a math problem. The agent emits one assistant message containing reasoning followed by a final answer in `\boxed{...}` notation. The evaluator extracts the boxed answer and grades it against ground truth via [`rllm.eval.reward_fns.math`](../../rllm/eval/reward_fns/math.py) (which uses symbolic + numeric equivalence — `0.5` matches `\frac{1}{2}`, etc.).

## Architecture

```
AgentFlow.run(task, config)
  │
  ├── one LLM call via OpenAI(base_url=config.base_url)
  │     model outputs reasoning + \boxed{ANSWER}
  │
  └── store full response in episode.artifacts["answer"]

Evaluator.evaluate(task, episode)
  │
  └── extract last \boxed{...}, grade against task.metadata["ground_truth"]
      via mathd + sympy.
```

## Installation

```bash
uv pip install -e ".[tinker]"                  # rllm + tinker backend
uv pip install --no-deps -e cookbooks/math     # this cookbook
```

After installation:

```bash
rllm agent list      # should show "math"
```

## Datasets

```bash
rllm dataset pull hendrycks_math    # train (Hendrycks MATH)
rllm dataset pull math500           # 500-problem test
rllm dataset pull gsm8k             # alternative train
rllm dataset pull deepscaler_math   # ~40K AIME/AMC/Omni-MATH/STILL
rllm dataset pull aime2024          # AIME 2024 (eval)
```

## Eval (rllm CLI)

```bash
rllm eval math500 \
    --agent math \
    --evaluator math \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --base-url http://localhost:8000/v1 \
    --max-examples 20
```

For `aime2024` pass@k-style eval (the original `examples/deepscaler/` use case), use `--max-examples` and run multiple times, or use the harness's repeat flag if exposed.

Episode JSONs land under `~/.rllm/eval_results/` for inspection (`rllm view`).

## Training (rllm CLI)

```bash
rllm train hendrycks_math \
    --agent math \
    --evaluator math \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --group-size 8 \
    --batch-size 32 \
    --lora-rank 32 \
    --epochs 1 \
    --val-freq 10
```

For LoRA-only training (the legacy `examples/gsm8k_lora/` use case), set `--lora-rank` higher and pass `--train-dataset gsm8k`. For deeper customization (verl backend, full PPO knobs), use the shell scripts.

## Training (shell scripts)

### Tinker (single-machine)

```bash
bash cookbooks/math/train_tinker.sh
```

### Verl (distributed GPU)

```bash
uv pip install -e ".[verl]"
bash scripts/install_megatron.sh <cu128|cu129|...>
bash cookbooks/math/train_verl.sh
```

## Tests

```bash
pytest cookbooks/math/test.py -v
```

## Files

| File | Description |
|------|-------------|
| `math_flow.py` | `math_flow` — single-turn AgentFlow |
| `evaluator.py` | `math_evaluator` — wraps `rllm.eval.reward_fns.math` |
| `train.py` | Python API training script (Hydra config) |
| `train_tinker.sh` | Tinker backend — single-machine training |
| `train_verl.sh` | Verl backend — distributed multi-GPU training |
| `pyproject.toml` | Plugin metadata and entry points |
| `test.py` | Unit tests for evaluator scoring |

## Migration notes

This cookbook replaces:

- **`examples/deepscaler/`** — `MathAgent` eval on `aime2024` → use `rllm eval aime2024 --agent math --evaluator math`
- **`examples/simple_math/`** — `MathAgent` train on `hendrycks_math` → see `train_tinker.sh` / `train_verl.sh`
- **`examples/gsm8k_lora/`** — `MathAgent` LoRA train on `gsm8k` → `rllm train gsm8k --agent math --evaluator math --lora-rank 32`
- **`examples/math_tinker/`** — `MathAgent` + few-shot prompts → fork `math_flow.py` and customize the `SYSTEM_PROMPT`

All four shared the same `MathAgent` + `SingleTurnEnvironment` + `AgentExecutionEngine` plumbing. The cookbook collapses them into one self-contained AgentFlow + entry-point with no dependency on `rllm.agents` or `rllm.environments`.
