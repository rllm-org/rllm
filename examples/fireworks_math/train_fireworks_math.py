import hydra

from rllm.agents.agent import Action
from rllm.data.dataset import DatasetRegistry
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.rewards.reward_fn import math_reward_fn
from rllm.rewards.reward_types import RewardOutput
from rllm.trainer.pipeline_agent_trainer import PipelineAgentTrainer
from rllm.workflows.single_turn_workflow import SingleTurnWorkflow


def math_workflow_reward_fn(task_info: dict, action: str) -> RewardOutput:
    if isinstance(action, Action):
        action = action.action
    if isinstance(action, ModelOutput):
        action = action.text
    return math_reward_fn(task_info, action)

@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("hendrycks_math", "train")
    test_dataset = DatasetRegistry.load_dataset("math500", "test")

    trainer = PipelineAgentTrainer(
        workflow_class=SingleTurnWorkflow,
        workflow_args={
            "reward_function": math_workflow_reward_fn,
            "max_prompt_length": config.data.max_prompt_length,
            "max_response_length": config.data.max_response_length,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
