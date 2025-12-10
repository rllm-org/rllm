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
python -m rllm.data.prepare_hendrycks_math
```

---

## 1. Define the Rollout Function

The rollout function generates a response from the LLM. This is **what you want to train**.

### 1.1 Import dependencies

```python
from rllm.sdk.shortcuts import get_chat_client
```

### 1.2 Define the prompt

The prompt tells the model how to format its answer. This is crucial—the reward function needs to find the answer in the response:

```python
MATH_PROMPT = """Solve the following math problem step by step.
Put your final answer in \\boxed{} format.

Problem: {question}
"""
```

> **💡 Why `\boxed{}`?** This LaTeX format is standard in math benchmarks. The reward function looks for `\boxed{answer}` to extract the final answer. Without it, the reward function may fail to find the answer.

### 1.3 Create the generation logic

```python
def generate_response(question: str) -> str:
    """Generate a response to a math question.
    
    This is the core behavior you want to improve via RL.
    """
    # Create client INSIDE the function (important for Ray serialization)
    client = get_chat_client(
        base_url="http://localhost:4000/v1",
        api_key="EMPTY"
    )
    
    # Format the prompt with the question
    prompt = MATH_PROMPT.format(question=question)
    
    # Make the LLM call - automatically traced!
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    
    return response.choices[0].message.content
```

### 1.4 Test the generation

```python
response = generate_response("What is 2 + 2?")
print(f"Response: {response}")
```

**Expected output:**
```
Response: Let me solve this step by step.
2 + 2 = 4
The answer is \boxed{4}
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

Under the hood, `math_reward_fn`:
- Extracts the answer from `\boxed{}` tags (common in math models)
- Falls back to finding the last number in the response
- Normalizes both answers (handles fractions, LaTeX, etc.)
- Compares using both string matching and symbolic math (via SymPy)

### 2.3 Test the evaluation

```python
# Correct answer (boxed format)
reward = evaluate_response("The answer is \\boxed{4}", ground_truth="4")
print(f"Reward for correct: {reward}")  # 1.0

# Correct answer (plain number)
reward = evaluate_response("After calculation, I get 4", ground_truth="4")
print(f"Reward for correct: {reward}")  # 1.0

# Wrong answer
reward = evaluate_response("The answer is \\boxed{5}", ground_truth="4")
print(f"Reward for wrong: {reward}")  # 0.0
```

**Expected output:**
```
Reward for correct: 1.0
Reward for wrong: 0.0
```

---

## 3. Combine into the Training Function

Now combine rollout + reward into a single function that the trainer can call:

```python
from rllm.sdk.shortcuts import get_chat_client
from rllm.rewards.reward_fn import math_reward_fn

MATH_PROMPT = """Solve the following math problem step by step.
Put your final answer in \\boxed{} format.

Problem: {question}
"""

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
    
    prompt = MATH_PROMPT.format(question=question)
    
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        messages=[{"role": "user", "content": prompt}],
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

# vLLM environment setup
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1

MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

python3 -m examples.sdk.simple_math.train_hendrycks_math \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=32 \
    data.val_batch_size=512 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    trainer.total_epochs=3 \
    trainer.project_name=simple-math \
    trainer.experiment_name=grpo-baseline
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
