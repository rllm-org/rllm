import hydra

from examples.sdk.solver_judge.solver_judge_flow_session import SolverJudgeWorkflow
from rllm.data.dataset import DatasetRegistry
from rllm.rewards.countdown_reward import countdown_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer


async def run_workflow(**kwargs) -> float:
    task = kwargs
    workflow = SolverJudgeWorkflow(rollout_engine=None, executor=None, n_solutions=2, reward_function=countdown_reward_fn)
    return await workflow.run(task, "")


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("countdown", "train")
    test_dataset = DatasetRegistry.load_dataset("countdown", "test")

    assert train_dataset, "Train dataset not found. Please run examples/solver_judge/prepare_countdown_data.py first."
    assert test_dataset, "Test dataset not found. Please run examples/solver_judge/prepare_countdown_data.py first."

    trainer = AgentTrainer(
        agent_run_func=run_workflow,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
