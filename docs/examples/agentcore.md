# Agentic RL Training with AWS Bedrock AgentCore Runtime

This guide walks you through training an arbitrary agent using rLLM with AWS Bedrock AgentCore Runtime (ACR). You will adapt an agent for RL using [agentcore-rl-toolkit](https://github.com/awslabs/agentcore-rl-toolkit), deploy it to ACR, and run GRPO training using either the **Tinker backend** (CPU only) or the **verl backend** (GPU needed). We use the GSM8K math agent as a running example. Training any other agent only requires an agent ARN config change.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                  Training Orchestrator (rLLM)                  │
│       ┌─────────────────────┐    ┌─────────────────────┐       │
│       │   Training Engine   ├───►│  Inference Servers  │       │
│       │   (Tinker / veRL)   │    │   (Tinker / vLLM)   │       │
│       └───────┬─────────────┘    └───────▲─────────────┘       │
│               │                          │                     │
│               │                  ┌───────┴─────────────┐       │
│               │                  │ rllm-model-gateway  │       │
│               │                  │   (token capture)   │       │
│               │                  └───────▲─────────────┘       │
└───────────────┼──────────────────────────┼─────────────────────┘
                │ 1. Submit N prompts      │ 2. Model inference
                │    to ACR                │    via standard OpenAI API
                ▼                          │
┌────────────────────────────────────────────────────────────────┐
│                  AWS Bedrock AgentCore Runtime                 │
│        ┌───────────┐  ┌───────────┐       ┌───────────┐        │
│        │   Agent   │  │   Agent   │  ...  │   Agent   │        │
│        │ Session 1 │  │ Session 2 │       │ Session N │        │
│        └─────┬─────┘  └─────┬─────┘       └─────┬─────┘        │
│              └──────────────┴───────────────────┘              │
│                             │ 3. Save rewards to S3            │
│                             ▼                                  │
│                   ┌───────────────────┐                        │
│                   │    S3 (results)   │                        │
│                   └─────────┬─────────┘                        │
└─────────────────────────────┼──────────────────────────────────┘
                              │ 4. Poll S3 for results
                              ▼
                  Training engine reads tokens
                  from gateway + rewards from S3
                  for policy update
```

**Components:**

- **[agentcore-rl-toolkit (ART)](https://github.com/awslabs/agentcore-rl-toolkit)** — SDK for adapting agents for RL training on ACR. Provides `AgentCoreRLApp` (agent side) and `RolloutClient` (training side).
- **[rllm-model-gateway](https://github.com/rllm-org/rllm/tree/main/rllm-model-gateway)** — HTTP proxy that transparently captures token IDs and logprobs from inference servers. Managed automatically by rLLM.
- **[AWS Bedrock AgentCore Runtime (ACR)](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agents-tools-runtime.html)** — Serverless runtime for deploying agents with auto-scaling and session isolation.

## Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) package manager
- AWS account with ACR access, an ECR repository, and an S3 bucket
- AWS credentials configured (run `aws configure`; the credentials need permissions for Bedrock AgentCore, ECR, and S3)
- Docker (for building agent containers)
- For verl: GPU machine(s) with CUDA 12.8+

---

## Step 1: Install rLLM

### Option A: Tinker Backend

Tinker is a lightweight backend suitable for hosted training. No local GPU setup needed.

```bash
git clone https://github.com/rllm-org/rllm.git
cd rllm
uv venv --python 3.11
uv pip install -e ".[tinker]" --torch-backend=cpu
```

You will also need a Tinker API key. Sign up at [https://tinker-console.thinkingmachines.ai](https://tinker-console.thinkingmachines.ai) and note your `TINKER_API_KEY`.

### Option B: verl Backend

verl is a distributed backend for production-scale GPU training. Megatron enables efficient model parallelism.

```bash
git clone https://github.com/rllm-org/rllm.git
cd rllm
uv venv --python 3.11
uv pip install -e ".[verl]" --torch-backend=<cu128|cu129|cu130>
```

Then install Megatron dependencies (nvidia-modelopt, transformer-engine, megatron-core, megatron-bridge, NVIDIA Apex):

```bash
bash scripts/install_megatron.sh <cu128|cu129|cu130>
```

The `--torch-backend` flag must match your CUDA version: `cu128` for CUDA 12.8, `cu129` for CUDA 12.9, `cu130` for CUDA 13.0. The Megatron install compiles CUDA extensions and may take a while.

### AgentCore Integration (Both Backends)

Install the AgentCore extra to get `RolloutClient` on the training side:

```bash
uv pip install -e ".[agentcore]"
```

---

## Step 2: Build Your Agent

Your agent runs as a container on ACR. It receives prompts, calls the model via a standard OpenAI-compatible API (through `rllm-model-gateway` during training), executes tools in a sandboxed environment, computes a reward, and returns it.

The [agentcore-rl-toolkit](https://github.com/awslabs/agentcore-rl-toolkit) makes this a small adaptation from a production agent:

1. `BedrockAgentCoreApp` → `AgentCoreRLApp`
2. `@app.entrypoint` → `@app.rollout_entrypoint`
3. Create model inside the entrypoint using `base_url` and `model_id` from the `_rollout` payload
4. Return `{"rewards": ...}` instead of text

### Example: Math Agent (`rl_app.py`)

```python
from reward import GSM8KReward
from strands import Agent
from strands.models.openai import OpenAIModel
from strands_tools import calculator

from agentcore_rl_toolkit import AgentCoreRLApp

app = AgentCoreRLApp()

system_prompt = (
    "Your task is to solve the math problem. "
    + "Use the calculator tool to compute all mathematical expressions. "
    + 'Let\'s think step by step and output the final answer after "####".'
)

reward_fn = GSM8KReward()


@app.rollout_entrypoint
def invoke_agent(payload: dict):
    base_url = payload["_rollout"]["base_url"]
    model_id = payload["_rollout"]["model_id"]
    params = payload["_rollout"].get("sampling_params", {})

    model = OpenAIModel(
        client_args={"api_key": "EMPTY", "base_url": base_url},
        model_id=model_id,
        params=params,
    )

    agent = Agent(
        model=model,
        tools=[calculator],
        system_prompt=system_prompt,
    )

    user_input = payload.get("prompt")
    answer = payload.get("answer")

    response = agent(user_input)

    rewards = reward_fn(
        response_text=response.message["content"][0]["text"],
        ground_truth=answer,
    )

    return {"rewards": rewards}


if __name__ == "__main__":
    app.run()
```

### Example: Reward Function (`reward.py`)

```python
import re

from agentcore_rl_toolkit import RewardFunction


class GSM8KReward(RewardFunction):
    def __call__(
        self,
        response_text="",
        ground_truth="",
        method="strict",
        format_score=0.0,
        score=1.0,
        **kwargs,
    ):
        answer = self.extract_solution(solution_str=response_text, method=method)
        if answer is None:
            reward = 0
        else:
            if answer == ground_truth:
                reward = score
            else:
                reward = format_score
        return reward

    @staticmethod
    def extract_solution(solution_str, method="strict"):
        _SOLUTION_CLIP_CHARS = 300

        if len(solution_str) > _SOLUTION_CLIP_CHARS:
            solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

        if method == "strict":
            solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
            if len(solutions) == 0:
                final_answer = None
            else:
                final_answer = solutions[-1].replace(",", "").replace("$", "")
        elif method == "flexible":
            answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
            final_answer = None
            if len(answer) > 0:
                invalid_str = ["", "."]
                for final_answer in reversed(answer):
                    if final_answer not in invalid_str:
                        break
        return final_answer
```

> **Token capture** is handled automatically by `rllm-model-gateway` — a transparent HTTP proxy between your agent and the inference server. It captures token IDs and logprobs without any changes to your agent code. You don't need to configure it; the training infrastructure manages the gateway.

---

## Step 3: Deploy to ACR

Follow the deployment instructions in the [agentcore-rl-toolkit repository](https://github.com/awslabs/agentcore-rl-toolkit):

1. Prepare a Dockerfile
2. Build and push the container image to ECR
3. Create an ACR runtime

After deployment, note two values you'll need for training:

- **`AGENTCORE_AGENT_ARN`** — the ARN of your deployed agent runtime (e.g., `arn:aws:bedrock-agentcore:us-west-2:123456789:runtime/my-agent`)
- **`AGENTCORE_S3_BUCKET`** — the S3 bucket for storing rollout results

---

## Step 4: Prepare Data and Configure

### Prepare the Dataset

From the rLLM repo root, prepare the GSM8K dataset:

```bash
uv run python -m examples.agentcore_math.prepare_gsm8k_data
```

This downloads GSM8K from HuggingFace and registers it as `gsm8k_agentcore` with `{"prompt": ..., "answer": ...}` fields matching what the agent expects.

### Create `.env`

Create a `.env` file at the rLLM repo root with your credentials:

```bash
# Required for Tinker backend only
TINKER_API_KEY=your_tinker_api_key

# Required for AgentCore integration (both backends)
AGENTCORE_AGENT_ARN=arn:aws:bedrock-agentcore:us-west-2:123456789:runtime/your-agent
AGENTCORE_S3_BUCKET=your-s3-bucket
```

---

## Step 5: Run Training

### Option A: Tinker Backend

**Training script** (`examples/agentcore_math/train_agentcore_math_tinker.py`).

**Launch script** (`examples/agentcore_math/train_agentcore_math_tinker.sh`):

Key parameters:

- `rllm.remote_runtime.enabled=true` + `backend=agentcore` — enables ACR as the rollout runtime
- `tps_limit=25` — ACR rate limit (transactions per second)
- `session_timeout=300` — 5-minute timeout per agent session

Run:
```bash
bash examples/agentcore_math/train_agentcore_math_tinker.sh
```

### Option B: verl Backend

**Training script** (`examples/agentcore_math/train_agentcore_math_verl.py`).

**Launch script** (`examples/agentcore_math/train_agentcore_math_verl.sh`):

Key parameters (beyond those shared with Tinker):

- `model_engine=megatron` — use Megatron for model parallelism
- `actor_rollout_ref.hybrid_engine=True` — co-locate actor and rollout on the same GPUs
- `actor_rollout_ref.model.lora.*` — LoRA configuration (rank, alpha, merge)
- `actor_rollout_ref.rollout.name=vllm` + `mode=async` — async vLLM rollout engine
- `actor_rollout_ref.actor.megatron.use_mbridge=True` — enable Megatron-Bridge
- `trainer.n_gpus_per_node=8` — number of GPUs per node


Run:
```bash
bash examples/agentcore_math/train_agentcore_math_verl.sh
```

---

## What Happens During Training

1. rLLM loads a batch of prompts from the dataset
2. `RolloutClient` submits prompts to ACR, each as a separate agent session
3. ACR auto-scales containers. Each agent runs `rl_app.py`, calling the model via `base_url` (routed through `rllm-model-gateway`)
4. The gateway captures token IDs and logprobs from inference server responses
5. Each agent computes a reward and returns `{"rewards": ...}`. The `@rollout_entrypoint` decorator saves results to S3
6. `RolloutFuture` polls S3 until results are available
7. The training engine combines token data from the gateway with rewards from S3 to compute advantages and update the policy

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| ACR sessions timing out | Increase `rllm.remote_runtime.session_timeout` (default 300s) |
| Rate limiting / throttling errors | ACR has a default 25 TPS limit. Ensure `tps_limit=25` is set. Reduce `tps_limit` if needed |
| Model not found errors in agent | Ensure the model path in your training config matches what the inference server is serving |
| Megatron build failures | CUDA toolkit version must match `--torch-backend` (e.g., cu128 needs CUDA 12.8) |
| S3 permission errors | The ACR execution role needs `s3:PutObject` and `s3:GetObject` on the configured bucket |
