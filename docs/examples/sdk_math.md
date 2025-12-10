# Train Math Agent with rLLM SDK

In this tutorial, you'll build and train a single-step agent that solves math problems. This is the simplest way to get started with RL training using rLLM SDK.

## Overview

By the end of this tutorial, you will have:

1. Created a simple agent function that solves math problems
2. Connected it to rLLM's automatic tracing system
3. Trained the agent using GRPO on the Hendrycks MATH dataset

Training an RL agent requires two components:

1. **Rollout function**: Perform a sequence of actions using the LLM
2. **Reward function**: Evaluate how good the outcome is

The rLLM SDK handles the plumbing—you just define what to generate and how to score it.

---

## Setup

Install rLLM if you haven't already, and prepare the dataset:

```bash
cd rllm
python -m examples.simple_math.prepare_math_dataset
```

Launch a vLLM server for testing:

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --host 0.0.0.0 \
    --port 4000
```

---

## 1. Define the Rollout Function

The rollout function generates a response from the LLM. This is **what you want to train**.

### 1.1 Import dependencies and Launch Proxy

```python
from rllm.sdk import get_chat_client
```

### 1.2 Create the generation logic

```python
def generate_response(question: str) -> str:
    """Generate a response to a math question.
    
    This is the core behavior you want to improve via RL.
    """
    # Create client INSIDE the function (important for Ray serialization)
    client = get_chat_client(
        base_url="http://localhost:4000/v1",
        api_key="token-abc123"
    )
    
    # Make the LLM call - automatically traced!
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        messages=[
            {"role": "user", "content": question},
        ],
    )
    
    return response.choices[0].message.content
```

### 1.3 Test the generation

```python
print(generate_response("What is 2 + 2?"))
```

**Expected output:**
```
"\boxed{4}"
```

> **⚠️ Important**: Always create `get_chat_client()` *inside* the function. Creating it at module level causes Ray serialization errors.

---

## 2. Define the Reward Function

The reward function evaluates how good the response is. This is **the training signal**.

### 2.1 What the reward function does

The reward function is simple—it just does two things:

1. **Parse**: Extract the answer from the model's response (looks for `\boxed{}`, numbers, etc.)
2. **Compare**: Check if the extracted answer matches the ground truth

```
Model Response: "Let me solve this step by step... The answer is \boxed{4}"
                                                              ↓
                                               extract_answer() → "4"
                                                              ↓
                                               compare with ground_truth "4"
                                                              ↓
                                               Match? → reward = 1.0
```

### 2.2 Using the built-in math reward

rLLM provides `math_reward_fn` which handles common math answer formats:

```python
from rllm.rewards.reward_fn import math_reward_fn

def evaluate_response(response: str, ground_truth: str) -> float:
    """Evaluate how correct the response is.
    
    Returns:
        1.0 if correct, 0.0 if incorrect
    """
    result = math_reward_fn(
        {"ground_truth": ground_truth}, 
        response  # The model's full response
    )
    return result.reward
```

### 2.3 Test the evaluation

```python
# Correct answer (boxed format)
reward = evaluate_response("The answer is \\boxed{4}", ground_truth="4")
print(f"Reward for correct: {reward}")  # 1.0

# Correct answer (plain number)
reward = evaluate_response("After calculation, I get 4", ground_truth="4")
print(f"Reward for correct: {reward}")  # 0.0

# Wrong answer
reward = evaluate_response("The answer is \\boxed{5}", ground_truth="4")
print(f"Reward for wrong: {reward}")  # 0.0
```

**Expected output:**
```
Reward for correct: 1.0
Reward for format incorrect: 0.0
Reward for wrong: 0.0
```

---

## 3. Combine into a Rollout Function

Now combine generation + reward into a single rollout function that the trainer can call:

```python
from rllm.sdk import get_chat_client
from rllm.rewards.reward_fn import math_reward_fn

def rollout(**kwargs):
    """Complete training function: generate + evaluate.
    
    Args:
        question: The math problem to solve
        ground_truth: The correct answer
        
    Returns:
        float: Reward (1.0 for correct, 0.0 for incorrect)
    """
    question = kwargs["question"]
    ground_truth = kwargs["ground_truth"]
    
    # Step 1: Generate response (rollout)
    client = get_chat_client(
        base_url="http://localhost:4000/v1",
        api_key="EMPTY"
    )
    
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        messages=[{"role": "user", "content": question}],
    )
    response_text = response.choices[0].message.content
    
    # Step 2: Evaluate result (reward)
    reward = math_reward_fn(
        {"ground_truth": ground_truth}, 
        response_text
    ).reward
    
    return reward
```

### 3.1 Test the complete function

```python
result = rollout(
    question="What is 2 + 2?",
    ground_truth="4"
)
print(f"Reward: {result}")
```

**Expected output:**
```
Reward: 1.0
```

---

## 4. Set Up the Trainer

Now wrap the agent function with `AgentTrainer`:

```python
import hydra
from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer

@hydra.main(
    config_path="pkg://rllm.trainer.config", 
    config_name="agent_ppo_trainer", 
    version_base=None
)
def main(config):
    # Load datasets
    train_dataset = DatasetRegistry.load_dataset("hendrycks_math", "train")
    test_dataset = DatasetRegistry.load_dataset("math500", "test")
    
    # Create trainer with your agent function
    trainer = AgentTrainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        agent_run_func=rollout,  # Your function from step 3
    )
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
```

---

## 5. Configure Training Hyperparameters

Create a shell script with training configuration:

```bash
#!/bin/bash
# train_hendrycks_math.sh
set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

python train_hendrycks_math.py \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=32 \
    data.val_batch_size=512 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=30000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    rllm.mask_truncated_samples=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='sdk-math' \
    trainer.experiment_name='sdk-math' \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=10 \
    rllm.agent.max_steps=1 \
    rllm.stepwise_advantage.enable=False \
    rllm.workflow.use_workflow=True \
    trainer.total_epochs=100 \
    rllm.sdk.proxy.host=127.0.0.1 \
    rllm.sdk.proxy.port=4000 \
    rllm.sdk.proxy.mode=subprocess \
    rllm.sdk.store.path="/tmp/rllm-traces.db"
```

---

## 6. Run Training

Launch the training:

```bash
chmod +x train_hendrycks_math.sh
./train_hendrycks_math.sh
```


---

## 7. Monitor Training

Training logs to WandB by default. Key metrics:

| Metric | Description |
|--------|-------------|
| `critic/score/mean` | Average reward per batch |
| `val/pass@1` | Validation accuracy |

## Next Steps

- **[Tutorial 2](sdk_solver_judge.md)**: Multi-agent solver-judge with `@trajectory` decorator
- **[Tutorial 3](sdk_langgraph_rag.md)**: Train a LangGraph RAG agent with tool use
