# Math Tool Agent

A multi-turn agent for rLLM that trains a math agent to solve arithmetic problems using a **calculator tool**. Ships in two flavors:

* **AgentFlow path** (`math_tool_agent.py` + `train.py`) — the original protocol-based example. Talks to the model via OpenAI Chat Completions through the rLLM gateway.
* **Workflow path** (`workflow.py` + `train_workflow.py`) — the rLLM `Workflow` subclass form. Drives `rllm.experimental.rollout.RolloutEngine` directly via `TITOCompleter`, which is what makes token-level **step merging** available in the training pipeline.

## Overview

The agent solves competition math problems from [DeepScaleR-Preview](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) (train, ~40K problems blending AIME/AMC/Omni-MATH) and [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) (test) by calling a calculator tool for each arithmetic step. These problems are harder than GSM8K and require multi-turn reasoning, making them a better test for multi-turn RL training. This cookbook serves three purposes:

1. **End-to-end system test** — a multi-turn tool-use example that exercises the full training loop
2. **Onboarding** — shows new users how to build a tool-calling agent with rLLM
3. **Step-merging reference** — the workflow path exercises rLLM's TITO completer so consecutive turns can share a token prefix at training time (target metric: `batch/merge_compression_ratio > 1.0`)

## Which path should I use?

| Concern | AgentFlow (`train.py`) | Workflow (`train_workflow.py`) |
| --- | --- | --- |
| Talks to LLM via | OpenAI Chat Completions (gateway) | rLLM `RolloutEngine` directly |
| Token-level step merging | No — each turn re-tokenizes from scratch | Yes — `TITOCompleter` keeps a running buffer |
| `batch/merge_compression_ratio` in logs | ≈ 1.0 (no merging) | > 1.0 if turns share a prefix |
| Trace/observability via gateway | Yes (full HTTP capture) | No (engine-direct) |
| Backend | Tinker or Verl | Tinker or Verl |
| Recommended for | onboarding, generic agents | multi-turn RL where you want step merging |

## Architecture

### AgentFlow path

```
AgentFlow.run(task, config)
  │
  └── Multi-turn loop (up to 5 turns)
        │
        ├── LLM call via OpenAI(base_url=config.base_url)
        │     Model outputs reasoning + <tool_call>...</tool_call>
        │
        ├── Parse tool call → execute calculator → inject result
        │
        └── Repeat until model outputs <answer>NUMBER</answer>
```

The evaluator checks the final `<answer>` against the ground truth via numeric comparison.

### Workflow path

```
MathToolWorkflow.run(task, uid)
  │
  ├── MessageList state ([system, user])
  │
  └── Multi-turn loop (up to max_turns)
        │
        ├── TITOCompleter.complete(messages, tools=TOOLS)
        │     → ModelOutput (content, reasoning, tool_calls, prompt_ids, completion_ids)
        │     → Step (captured with prompt_ids + response_ids for training)
        │
        ├── Append assistant message to MessageList
        │
        ├── If tool_calls present: run _safe_eval() on each, append tool message → loop
        │
        └── Else: extract \boxed{ANSWER}, reward = (answer == task["answer"]) ? 1.0 : 0.0
```

Per-turn `Step.prompt_ids` starts where the previous turn's `prompt_ids + response_ids` left off (token-level extension). The transform path detects this and avoids re-encoding, surfacing as `batch/merge_compression_ratio > 1.0`.

## Installation

```bash
# From the rllm repo root
uv pip install -e ".[tinker]"                              # rllm + tinker backend
uv pip install --no-deps -e cookbooks/math_tool_agent      # this cookbook
```

After installation, the agent and evaluator are discoverable by the CLI:

```bash
rllm agent list    # should show "math_tool_agent" as a plugin
```

## Dataset

Pull the datasets (one-time):

```bash
rllm dataset pull deepscaler_math   # ~40K competition math problems (AIME/AMC/Omni-MATH/STILL)
rllm dataset pull math500           # 500-problem test benchmark
```

## Training

### AgentFlow path

#### Tinker (single-machine)

```bash
bash cookbooks/math_tool_agent/train_tinker.sh
```

Or directly via the Python API:

```bash
python cookbooks/math_tool_agent/train.py \
    rllm/backend=tinker \
    model.name=Qwen/Qwen3-4B-Instruct-2507 \
    model.lora_rank=32 \
    training.group_size=8
```

#### Verl (distributed GPU)

Requires verl extras and megatron:

```bash
uv pip install -e ".[verl]"
bash scripts/install_megatron.sh <cu128|cu129|...>
```

Then:

```bash
bash cookbooks/math_tool_agent/train_verl.sh
```

### Workflow path

The workflow entry script is `train_workflow.py`. It mirrors `train.py`'s Hydra config surface but routes through `AgentTrainer(workflow_class=MathToolWorkflow, ...)` instead of `AgentTrainer(agent_flow=..., evaluator=...)`. Same backend selection (`rllm/backend=verl` or `tinker`).

```bash
python cookbooks/math_tool_agent/train_workflow.py \
    rllm/backend=verl \
    model.name=Qwen/Qwen3-4B-Instruct-2507 \
    rllm.workflow.max_turns=8
```

Watch for `batch/merge_compression_ratio` in the training log. On a working multi-turn rollout with shared prefixes, it should rise above 1.0 within the first training step. A value stuck at 1.0 means the completer's prefix check is failing every turn — investigate the chat-parser's `parse_completion(parse_assistant(...))` round-trip for the assistant messages the workflow builds.

## Tests

```bash
pytest cookbooks/math_tool_agent/test.py -v      # AgentFlow unit tests
pytest tests/cookbooks/test_math_tool_workflow.py -v   # Workflow end-to-end smoke tests
```

The workflow smoke tests use a stub `RolloutEngine` that returns pre-canned completions; they don't need a GPU. They pin two properties:

1. **Reward correctness** — answer `\boxed{87}` against task answer `"87"` produces `reward=1.0`; `\boxed{99}` produces `reward=0.0`.
2. **Step-merging witness** — turn 2's `Step.prompt_ids` byte-extends turn 1's `prompt_ids + response_ids`. This is the byte-level property the transform layer reads to compute `merge_compression_ratio`.

## Files

| File | Description |
|------|-------------|
| `math_tool_agent.py` | `math_tool_agent` — multi-turn AgentFlow with calculator tool |
| `math_tool_eval.py` | `math_tool_evaluator` — numeric answer comparison |
| `workflow.py` | `MathToolWorkflow` — `Workflow` subclass with TITO step merging |
| `train.py` | Python API training script (AgentFlow path, Hydra config) |
| `train_workflow.py` | Python API training script (Workflow path, Hydra config) |
| `train_tinker.sh` | Tinker backend — single-machine training (AgentFlow) |
| `train_verl.sh` | Verl backend — distributed multi-GPU training (AgentFlow) |
| `pyproject.toml` | Plugin metadata and entry points |
| `test.py` | Unit tests for calculator, parsing, and evaluation |
