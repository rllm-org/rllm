import asyncio

import rllm.experimental.unified_trainer as unified_trainer
from rllm.agents.agent import Episode
from rllm.experimental.unified_trainer import TrainerState, UnifiedTrainer
from rllm.workflows.workflow import TerminationReason


class _WorkflowEngine:
    def set_training_step(self, step, mode="train", epoch=0):
        pass


class _Backend:
    async def generate_episodes(self, batch, agent_workflow_engine, is_validation=False):
        return [
            Episode(id="task:0", termination_reason=None, metrics={"custom": 1.0}),
            Episode(id="task:1", termination_reason=TerminationReason.ERROR, metrics={"custom": 3.0}),
        ]


def test_train_batch_populates_termination_metrics_before_early_return(monkeypatch):
    monkeypatch.setattr(
        unified_trainer,
        "transform_episodes_to_trajectory_groups",
        lambda episodes, transform_config, compact_filtering_config, traj_grouping_hook=None: ([], {}),
    )
    monkeypatch.setattr(
        unified_trainer,
        "apply_rejection_sampling_and_filtering",
        lambda episodes, groups, config, state: ([], episodes, {}),
    )

    trainer = object.__new__(UnifiedTrainer)
    trainer.agent_workflow_engine = _WorkflowEngine()
    trainer.backend = _Backend()
    trainer.transform_config = None
    trainer.cf_config = None
    trainer.traj_grouping_hook = None
    trainer.rs_config = None

    state = TrainerState()
    asyncio.run(trainer._train_batch_async(batch={}, trainer_state=state))

    assert state.metrics["batch/custom"] == 2.0
    assert state.metrics["batch/termination_reason/unknown"] == 0.5
    assert state.metrics["batch/termination_reason/error"] == 0.5
    assert state.metrics["batch/termination_reason/env_done"] == 0.0
