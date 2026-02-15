"""
Training script for the remote Solver-Judge Workflow.

This is identical to the standard Tinker solver-judge training, except it
enables remote agent mode via Hydra config overrides.  The trainer:

  1. Starts an inference API server that wraps the training model.
  2. Dispatches tasks to the remote agent server(s).
  3. The remote agent calls back to the inference API for model inference,
     runs the SolverJudgeWorkflow, and returns Episode objects.
  4. The trainer performs the usual RL update (advantages, policy gradient, etc.).

Usage:
    # Step 1: Start one or more remote agent servers:
    python -m examples.remote_solver_judge.remote_agent_server --port 5100
    python -m examples.remote_solver_judge.remote_agent_server --port 5101  # optional extra

    # Step 2: Launch training (see train_remote_solver_judge.sh for full example):
    python -m examples.remote_solver_judge.train_remote_solver_judge \\
        rllm/backend=tinker \\
        rllm.remote_agent.enabled=true \\
        'rllm.remote_agent.endpoints=["http://localhost:5100"]' \\
        rllm.remote_agent.inference_api.port=8089 \\
        ...
"""

import hydra

from examples.solver_judge.solver_judge_flow import SolverJudgeWorkflow
from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer
from rllm.rewards.countdown_reward import countdown_reward_fn


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("countdown", "train")
    test_dataset = DatasetRegistry.load_dataset("countdown", "test")

    trainer = AgentTrainer(
        workflow_class=SolverJudgeWorkflow,
        workflow_args={
            "n_solutions": 2,
            "reward_function": countdown_reward_fn,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        backend="tinker",
    )
    trainer.train()


if __name__ == "__main__":
    main()
