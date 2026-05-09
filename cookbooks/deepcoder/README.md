# Deepcoder Agent

A single-turn coding agent for rLLM that solves competition-style programming problems via the **AgentFlow protocol**, replacing the legacy `examples/deepcoder/` (which used `BaseAgent` + `SingleTurnEnvironment` + `AgentExecutionEngine`).

## Overview

The agent receives a problem (LiveCodeBench / Codeforces / TACO / APPS / primeintellect formats) and emits a single response containing reasoning followed by a fenced ```` ```python ```` code block. The full assistant response is stored on the episode; the evaluator extracts the last fenced block and runs it against the hidden test cases. Reward is `1.0` if all tests pass, `0.0` otherwise.

Long chain-of-thought reasoning happens *inside the assistant message* — there is no multi-turn revise/feedback loop. This matches the original deepcoder training setup.

## Architecture

```
AgentFlow.run(task, config)
  │
  ├── one LLM call via OpenAI(base_url=config.base_url)
  │     model outputs reasoning + ```python ... ```
  │
  └── store full response in episode.artifacts["answer"]

Evaluator.evaluate(task, episode)
  │
  └── RewardCodeFn extracts last ```python``` block, runs against
      task.metadata["ground_truth"] (hidden tests).
```

## Installation

```bash
uv pip install -e ".[tinker]"                          # rllm + tinker backend
uv pip install --no-deps -e cookbooks/deepcoder        # this cookbook
```

After installation:

```bash
rllm agent list      # should show "deepcoder"
```

## Dataset

```bash
python cookbooks/deepcoder/prepare_deepcoder_data.py
# Smoke-size run:
python cookbooks/deepcoder/prepare_deepcoder_data.py --train-size 200 --test-size 50
```

This pulls `agentica-org/DeepCoder-Preview-Dataset` (primeintellect + taco + lcbv5 train; codeforces + lcbv5 test), normalizes the test schemas (TACO's nested dict → flat list), and registers `deepcoder/{train,test}` with `DatasetRegistry`.

## Eval (rllm CLI)

```bash
rllm eval deepcoder \
    --agent deepcoder \
    --evaluator deepcoder \
    --model agentica-org/DeepCoder-14B-Preview \
    --base-url http://localhost:8000/v1 \
    --split test \
    --max-examples 20            # smoke-test before running the full split
```

Episode JSONs land under `~/.rllm/eval_results/` for inspection (`rllm view`).

## Training (rllm CLI)

```bash
rllm train deepcoder \
    --agent deepcoder \
    --evaluator deepcoder \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --group-size 4 \
    --batch-size 16 \
    --lora-rank 32 \
    --epochs 1 \
    --val-freq 20
```

For deeper customization (verl backend, full PPO knobs, megatron) use the shell scripts.

## Training (shell scripts)

### Tinker (single-machine)

```bash
bash cookbooks/deepcoder/train_tinker.sh
```

### Verl (distributed GPU)

```bash
uv pip install -e ".[verl]"
bash scripts/install_megatron.sh <cu128|cu129|...>
bash cookbooks/deepcoder/train_verl.sh
```

## Tests

```bash
pytest cookbooks/deepcoder/test.py -v
```

## Files

| File | Description |
|------|-------------|
| `deepcoder_flow.py` | `deepcoder_flow` — multi-turn iterative-coding AgentFlow |
| `deepcoder_eval.py` | `deepcoder_evaluator` — wraps `rllm.eval.reward_fns.code` |
| `prepare_deepcoder_data.py` | Pull + normalize Deepcoder splits via `DatasetRegistry` |
| `train.py` | Python API training script (Hydra config) |
| `train_tinker.sh` | Tinker backend — single-machine training |
| `train_verl.sh` | Verl backend — distributed multi-GPU training |
| `pyproject.toml` | Plugin metadata and entry points |
| `test.py` | Unit tests for code extraction, feedback rendering, evaluator |

## Migration notes

This cookbook replaces `examples/deepcoder/`, which used `BaseAgent` + `SingleTurnEnvironment` + `AgentExecutionEngine`. The two are functionally equivalent on competition-coding workloads. Multi-turn revisions are now driven by a plain async function with explicit message construction; no env subclass, no agent subclass, no execution engine. The cookbook depends only on `rllm.types`, `rllm.rewards.code_reward`, and `rllm.eval.reward_fns.code` — none of the legacy stack.
