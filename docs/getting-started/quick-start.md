# Quick Start with rLLM

This guide walks you through using rLLM to build AI agents with tool usage capabilities. We'll use the **math tool agent** example to demonstrate the complete workflow from dataset preparation through model training.

## Overview

In this tutorial, you'll create a math reasoning agent that can:

- Access a Python interpreter to solve mathematical problems
- Perform step-by-step reasoning with interleaved tool usage
- Learn and improve its math problem solving ability through reinforcement learning

The example uses:

- **Base Model**: Qwen3-4B
- **Training Data**: DeepScaleR-Preview-Math dataset 
- **Evaluation Data**: AIME 2024 mathematics competition problems
- **Tools**: Python interpreter for mathematical computations

## Prerequisites

Before starting, ensure you have:

1. **rLLM Installation**: Follow the [installation guide](./installation.md)
2. **GPU Requirements**: At least 1 GPU with 16GB+ memory for inference, 8+ GPUs for training
3. **Model Server**: We'll use vLLM or SGLang to serve the base model

## Step 1: Dataset Preparation

rLLM's `DatasetRegistry` provides a centralized way to manage datasets. Let's prepare the math datasets:

```python title="examples/math_tool/prepare_math_data.py"
--8<-- "examples/math_tool/prepare_math_data.py"
```

This registers the training dataset `deepscaler_math` and the testing dataset `aime2024`. Under the hood, rLLM stores the processed data as parquet files in a format suitable for both inference and training. Later, you can easily load the registered datasets using `DatasetRegistry.load_dataset`.

**Run the preparation script:**
```bash
cd examples/math_tool
python prepare_math_data.py
```

## Step 2: Model Server Setup

rLLM requires a model server for inference. Choose one of these options:

### Option A: vLLM Server
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16 \
    --tensor-parallel-size 1
# For multi-GPU: set --tensor-parallel-size to the number of GPUs (e.g., 4 for 4 GPUs)
```

### Option B: SGLang Server
```bash
python -m sglang_router.launch_server \
    --model-path Qwen/Qwen3-4B \
    --dp-size 1 \
    --dtype bfloat16
# increase dp_size to enable data-parallel processing on multi-GPU
```

The server provides an OpenAI-compatible API at `http://localhost:30000/v1`.

## Step 3: Model Inference

Now let's run inference to see how agents solve math problems using tools:

```python title="examples/math_tool/run_math_with_tool.py"
--8<-- "examples/math_tool/run_math_with_tool.py"
```

**Run the inference script:**
```bash
cd examples/math_tool
python run_math_with_tool.py
```

The script above configures a `ToolAgent` from rLLM with access to the `python` tool for solving math problems in AIME2024, and a `ToolEnvironment` for handling Python tool calls and returning results. 

The `AgentExecutionEngine` orchestrates the interaction between the `ToolAgent` and `ToolEnvironment`. The `execute_tasks` function launches 64 agent-environment pairs in parallel (`n_parallel_agents=64`) for rollout generation and returns results after all problems from the AIME2024 dataset are processed. Finally, the Pass@1 and Pass@K metrics for AIME are computed and printed. 

## Step 4: Agent Training with GRPO

Training improves the agent's ability to use tools effectively. rLLM uses verl as its training backend, which supports training language models with GRPO and various other RL algorithms.

```python title="examples/math_tool/train_math_with_tool.py"
--8<-- "examples/math_tool/train_math_with_tool.py"
```

**Run the training script:**
```bash
cd examples/math_tool
bash train_math_with_tool.sh
```

The script above launches an RL training job for our ToolAgent, using `deepscaler_math` as the training set and `aime2024` as the test set. Under the hood, rLLM handles agent trajectory generation using our `AgentExecutionEngine` and transforms the trajectories into `verl`'s format for model training using FSDP or Megatron. The training process works as follows:

1. **Rollout Generation**: A batch of data is passed to `AgentExecutionEngine`, which launches multiple agent-environment pairs in parallel to process the batch. The engine returns all trajectories along with rewards computed by the environment.
2. **Transform Trajectories**: Agent trajectories are transformed into the corresponding format for our training backend `verl`. 
3. **Advantage Calculation with GRPO**: `verl` uses GRPO for advantage calculation.
4. **Model Update**: `verl` updates the model parameters to increase the probability of successful actions. The updated model is then used to generate trajectories for the next batch of data.

### Key rLLM Components in This Example

| Component | Purpose | Example Usage |
|-----------|---------|---------------|
| `ToolAgent` | Agent with tool usage capabilities | Reasoning + Python execution |
| `ToolEnvironment` | Safe tool execution environment | Sandboxed Python interpreter |
| `DatasetRegistry` | Centralized dataset management | Load/register math datasets |
| `AgentExecutionEngine` | Parallel agent execution | Efficient batch inference |
| `AgentTrainer` | RL training orchestration | PPO-based agent improvement |

## Next Steps

Congratulations! You've successfully used rLLM to run and train a ToolAgent for math problem solving. For a deeper dive into rLLM's main components, check out [Core Concepts in rLLM](../core-concepts/overview.md).