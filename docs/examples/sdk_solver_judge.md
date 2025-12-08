# Multi-Agent Solver-Judge with @trajectory Decorator

In this tutorial, you'll build a **two-agent system** where:

- **Solver**: Generates candidate solutions to a problem
- **Judge**: Evaluates and selects the best solution

This pattern is powerful for training agents that can both generate and verify solutions.

## Overview

By the end of this tutorial, you will have:
1. Built a Solver agent that generates multiple solution candidates
2. Built a Judge agent that selects the best solution
3. Assigned separate rewards to each agent using `@trajectory`
4. Trained the multi-agent system end-to-end

**Dataset**: [Countdown](https://huggingface.co/datasets/Jiayi-Pan/Countdown) - Given numbers, reach a target using arithmetic operations.

---

## Why Multi-Agent?

Training an RL agent requires two components:

1. **Rollout function**: Perform a sequence of actions using the LLM
2. **Reward function**: Evaluate how good the outcome is

In a multi-agent system, you have **multiple rollout functions** (Solver and Judge), and each gets its own reward:

```
┌──────────────────────────────────────────────────────────────────┐
│                        Training Loop                              │
│                                                                   │
│   ┌──────────┐      ┌──────────┐      ┌──────────┐              │
│   │  Task    │ ──▶  │ Solver   │ ──▶  │  Reward  │ (per solver) │
│   │ (target, │      │ (generate│      │ (correct │              │
│   │  numbers)│      │ solution)│      │ equation?)│              │
│   └──────────┘      └──────────┘      └──────────┘              │
│                           │                                       │
│                           ▼                                       │
│                     ┌──────────┐      ┌──────────┐              │
│                     │  Judge   │ ──▶  │  Reward  │ (selection)  │
│                     │ (select  │      │ (picked  │              │
│                     │   best)  │      │ correct?)│              │
│                     └──────────┘      └──────────┘              │
└──────────────────────────────────────────────────────────────────┘

Result: Multiple trajectories, each with its own reward signal
```

**Key insight**: Each agent gets its OWN trajectory and reward signal. The trainer collects ALL trajectories and trains both agents simultaneously.

### Concepts

We will cover:

- **`@trajectory` decorator**: Automatic session management and trace capture
- **`TrajectoryView`**: Access to steps, results, and rewards
- **Multi-agent workflows**: Composing multiple agents with independent rewards

---

## Setup

Install rLLM if you haven't already, and prepare the Countdown dataset:

```bash
python -m rllm.data.prepare_countdown
```

---

## 1. Understanding @trajectory

The `@trajectory` decorator automatically:
- Creates a session for each function call
- Tracks all LLM calls as steps
- Returns a `TrajectoryView` with steps and result

### 1.1 Basic usage

```python
from rllm.sdk import trajectory, get_chat_client_async

@trajectory(name="my_agent")
async def my_agent(prompt: str):
    client = get_chat_client_async(base_url="http://localhost:4000/v1", api_key="EMPTY")
    response = await client.chat.completions.create(
        model="Qwen/Qwen3-4B",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

### 1.2 What you get back

```python
# Call the decorated function
traj = await my_agent("What is 2+2?")

# traj is a TrajectoryView with:
print(traj.name)        # "my_agent"
print(traj.result)      # "4" (your return value)
print(traj.steps)       # [StepView(...)] - one per LLM call
print(traj.reward)      # 0.0 (default, you can set this)
```

---

## 2. Understand the Task

The **Countdown** task: Given a target number and a list of numbers, create an equation using the given numbers to reach the target.

**Example:**
- Target: `150`
- Numbers: `[25, 50, 75, 100]`
- Valid solution: `100 + 50 = 150`

The reward function checks:
1. Does the equation use only the given numbers (each once)?
2. Does the equation evaluate to the target?

---

## 3. Build the Solver Agent

The Solver generates solution candidates for Countdown puzzles.

### 3.1 Define the prompt template

Just like Tutorial 1 used `\boxed{}` for math problems, the Countdown task uses `<answer>...</answer>` tags:

```python
SOLVER_PROMPT = """{problem}

Think through this step by step. Use only the given numbers, each at most once.
Output your final equation within <answer>...</answer> tags.

Example: <answer>100 + 50 = 150</answer>
"""
```

> **💡 Why `<answer>` tags?** The reward function looks for `<answer>equation</answer>` to extract the solution. Without it, the reward function cannot find your answer—similar to `\boxed{}` in math problems.

### 3.2 Define the Solver class

```python
import asyncio
import re
from rllm.sdk import trajectory, get_chat_client_async

SOLVER_PROMPT = """{problem}

Think through this step by step. Use only the given numbers, each at most once.
Output your final equation within <answer>...</answer> tags.

Example: <answer>100 + 50 = 150</answer>
"""

class Solver:
    def __init__(self):
        self.client = get_chat_client_async(
            base_url="http://localhost:4000/v1", 
            api_key="EMPTY"
        )
        self.model = "Qwen/Qwen3-4B"

    @trajectory(name="solver")
    async def generate_solution(self, problem: str):
        """Generate a single solution. Returns TrajectoryView automatically."""
        prompt = SOLVER_PROMPT.format(problem=problem)
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,  # Higher temperature for diverse solutions
            max_tokens=1000,
        )
        
        response_text = response.choices[0].message.content
        return self._parse_answer(response_text)

    def _parse_answer(self, response: str) -> str:
        """Extract answer from <answer>...</answer> tags."""
        match = re.search(r"<answer>(.*?)</answer>", response, re.IGNORECASE | re.DOTALL)
        if match:
            return f"<answer>{match.group(1).strip()}</answer>"
        return ""

    async def generate_solutions(self, problem: str, n_solutions: int = 2):
        """Generate multiple solutions concurrently."""
        tasks = [
            asyncio.create_task(self.generate_solution(problem))
            for _ in range(n_solutions)
        ]
        return await asyncio.gather(*tasks)
```

### 3.4 Test the Solver

```python
solver = Solver()

# Generate 2 solutions for a Countdown puzzle
problem = "Using numbers [25, 50, 75, 100], reach target 150"
trajs = await solver.generate_solutions(problem, n_solutions=2)

for i, traj in enumerate(trajs):
    print(f"Solution {i+1}: {traj.result}")
    print(f"  Steps: {len(traj.steps)}")
```

**Expected output:**
```
Solution 1: <answer>100 + 50 = 150</answer>
  Steps: 1
Solution 2: <answer>75 + 75 = 150</answer>
  Steps: 1
```

---

## 4. Build the Judge Agent

The Judge evaluates solutions and selects the best one.

### 4.1 Define the Judge prompt template

The Judge needs to compare solutions and pick the correct one:

```python
JUDGE_PROMPT = """You are an expert verifier. Given a problem and candidate solutions, select the correct one.

Problem: {problem}

Solutions:
{solutions}

Analyze each solution:
1. Does it use only the given numbers?
2. Does it use each number at most once?
3. Does the equation equal the target?

Output the index of the correct solution within <answer>...</answer> tags.
"""
```

### 4.2 Define the Judge class

```python
class Judge:
    def __init__(self):
        self.client = get_chat_client_async(
            base_url="http://localhost:4000/v1", 
            api_key="EMPTY"
        )
        self.model = "Qwen/Qwen3-4B"

    @trajectory(name="judge")
    async def judge_solutions(self, problem: str, solutions: list[str]):
        """Evaluate solutions and select the best one."""
        # Format solutions list
        solutions_text = ""
        for i, sol in enumerate(solutions, 1):
            solutions_text += f"\nSolution {i}:\n{sol}\n"
        
        prompt = JUDGE_PROMPT.format(problem=problem, solutions=solutions_text)
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            max_tokens=1000,
        )
        
        response_text = response.choices[0].message.content
        return self._parse_selection(response_text, solutions)

    def _parse_selection(self, response: str, solutions: list[str]) -> str:
        """Extract selected solution index."""
        match = re.search(r"<answer>(\d+)</answer>", response)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(solutions):
                return solutions[idx]
        return ""
```

### 4.3 Test the Judge

```python
judge = Judge()

solutions = [
    "<answer>100 + 50 = 150</answer>",
    "<answer>Wrong answer</answer>"
]
judge_traj = await judge.judge_solutions(problem, solutions)

print(f"Selected: {judge_traj.result}")
print(f"Steps: {len(judge_traj.steps)}")
```

**Expected output:**
```
Selected: <answer>100 + 50 = 150</answer>
Steps: 1
```

---

## 5. Compose the Workflow

Now combine Solver and Judge, assigning rewards to each trajectory.


```python
from rllm.sdk import TrajectoryView
from rllm.rewards.countdown_reward import countdown_reward_fn

class SolverJudgeWorkflow:
    def __init__(self, n_solutions: int = 2, reward_function=None, **kwargs):
        self.n_solutions = n_solutions
        self.reward_function = reward_function
        self.solver = Solver()
        self.judge = Judge()

    async def run(self, task: dict, uid: str, **kwargs) -> list[TrajectoryView]:
        """Run the full workflow and return all trajectories."""
        problem = task["question"]

        # Step 1: Generate multiple solutions
        solver_trajs = await self.solver.generate_solutions(problem, self.n_solutions)

        # Step 2: Assign rewards to each solver
        solutions = []
        for traj in solver_trajs:
            parsed_answer = traj.result
            reward = self.reward_function(task, parsed_answer).reward
            
            # Assign reward to the trajectory AND its steps
            traj.steps[0].reward = reward
            traj.reward = reward
            solutions.append(parsed_answer)

        # Step 3: Judge selects the best solution
        judge_traj = await self.judge.judge_solutions(problem, solutions)
        selected = judge_traj.result
        
        # Judge reward based on final selection quality
        judge_reward = self.reward_function(task, selected).reward
        judge_traj.steps[0].reward = judge_reward
        judge_traj.reward = judge_reward

        # Return ALL trajectories for training
        return solver_trajs + [judge_traj]
```

### 5.3 Reward assignment strategy

```
Example run:
┌─────────────────────────────────────────────────┐
│ Problem: Reach 150 with [25, 50, 75, 100]       │
├─────────────────────────────────────────────────┤
│ Solver 1: "100 + 50 = 150"  → reward = 1.0 ✓   │
│ Solver 2: "25 + 75 = 100"   → reward = 0.0 ✗   │
│ Judge: selects Solver 1     → reward = 1.0 ✓   │
└─────────────────────────────────────────────────┘

Training signal:
• Solver 1 is reinforced (correct answer)
• Solver 2 learns to improve (wrong answer)
• Judge learns to identify correct solutions
```

---

## 6. Set Up Training

### 6.1 Create the training wrapper

```python
import hydra
from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer

async def run_workflow(**kwargs) -> list[TrajectoryView]:
    """Training wrapper that returns trajectories."""
    workflow = SolverJudgeWorkflow(
        n_solutions=2,
        reward_function=countdown_reward_fn
    )
    return await workflow.run(kwargs, uid="")

@hydra.main(
    config_path="pkg://rllm.trainer.config", 
    config_name="agent_ppo_trainer", 
    version_base=None
)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("countdown", "train")
    test_dataset = DatasetRegistry.load_dataset("countdown", "test")

    trainer = AgentTrainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        agent_run_func=run_workflow,
    )
    trainer.train()

if __name__ == "__main__":
    main()
```

### 6.2 Launch script

```bash
#!/bin/bash
# train_solver_judge.sh
set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1

MODEL_PATH=Qwen/Qwen3-4B

python3 -m examples.sdk.solver_judge.train_decorator \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=32 \
    data.val_batch_size=256 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    trainer.total_epochs=3 \
    trainer.project_name=solver-judge \
    trainer.experiment_name=countdown-grpo
```

---

## 7. Run Training

```bash
chmod +x train_solver_judge.sh
./train_solver_judge.sh
```

---

## Next Steps

- **[Tutorial 1](sdk_math.md)**: Review the basics with a single-step agent
- **[Tutorial 3](sdk_langgraph_rag.md)**: Train a LangGraph RAG agent with tool use