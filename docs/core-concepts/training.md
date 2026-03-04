# RL Training with AgentTrainer

!!! warning "Deprecation warning"
    The current `AgentTrainer` will be soon replaced with a newer, [`UnifiedTrainer`](../experimental/unified-trainer.md)-based agent trainer. If you are working with `Tinker`, we highly recommend you to migrate to the new [`UnifiedTrainer`](../experimental/unified-trainer.md)-based workflow trainer. See [RL Training with Tinker (with Unified Trainer)](../examples/tinker_rl.md) for an upgraded solver-judge workflow example.

Training in rLLM uses reinforcement learning algorithms to update agent policies based on rewards. This page explains the training architecture, available algorithms, and how to configure and run training jobs.

## Overview

The `AgentTrainer` is the high-level interface for training reinforcement learning agents in rLLM. It provides a simplified API that wraps the underlying training infrastructure (verl), allowing you to train custom agents in custom environments without directly managing the complex distributed training setup.

## Architecture

### Core Components

The AgentTrainer orchestrates several key components:

1. **Agent**: The learning policy that generates actions based on observations
2. **Environment**: The task environment that provides observations and rewards
3. **RL Trainer**: The underlying reinforcement learning algorithm implementation

### Training Flow

The AgentTrainer serves as a wrapper over the training engine `verl`. When `trainer.train()` is called, the following process occurs:

**Initialization**: The system initializes the `AgentPPOTrainer`, which inherits from `verl`'s `RayPPOTrainer`. We replace the original trajectory generation logic with rLLM's AgentExecutionEngine.

**Setup Phase**: The `AgentPPOTrainer` performs the following setup:

   - Sets up Ray workers for distributed model training
   - Initializes the AgentExecutionEngine
   - Loads the dataset and splits it into mini-batches

**Training Loop**: For each mini-batch:

   - Data is passed to rLLM's AgentExecutionEngine
   - The engine initializes agent-environment pairs to process the mini-batch in parallel
   - Agent trajectories are collected through environment interactions

**Update Phase**: After a mini-batch is sampled:

   - The trainer transforms trajectories into `verl`'s format
   - Gradient updates are performed using the collected trajectories

For more details, reference `rllm/trainer/agent_ppo_trainer.py`, where we implement our custom RL training flow for agents.

## Basic Usage

### Simple Training Setup

```python
import hydra
from rllm.train.agent_trainer import AgentTrainer
from rllm.agents import YourCustomAgent
from rllm.environments import YourCustomEnvironment
from rllm.data import DatasetRegistry

@hydra.main(config_path="pkg://rllm.train.config", config_name="ppo_trainer")
def main(config):
    # Load datasets
    train_dataset = DatasetRegistry.load_dataset("your_dataset", "train")
    val_dataset = DatasetRegistry.load_dataset("your_dataset", "test")
    
    # Initialize trainer
    trainer = AgentTrainer(
        agent_class=YourCustomAgent,
        env_class=YourCustomEnvironment,
        agent_args={},
        env_args={},
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    
    # Start training
    trainer.train()
```

## Configuration

### Main Configuration File

rLLM adopts the same configuration structure as verl's `ppo_trainer.yaml`, with additional rLLM-specific configurations for our AgentExecutionEngine.

#### Agent-Specific Configuration
```yaml
agent:
  max_steps: 10              # Maximum steps per episode
  n_parallel_agents: 8       # Number of parallel agent instances
  use_stepwise_advantage: true  # Enable step-wise advantage calculation
  trajectory_timeout: 300    # Timeout for trajectory collection (seconds)
```