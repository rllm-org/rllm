from rllm.agents.agent import Episode
from rllm.experimental.unified_trainer import UnifiedTrainer
from rllm.workflows.workflow import TerminationReason


def test_collect_workflow_metrics_counts_unknown_termination_reason():
    episodes = [
        Episode(id="task:0", termination_reason=None, metrics={"custom": 1.0}),
        Episode(id="task:1", termination_reason=TerminationReason.ERROR, metrics={"custom": 3.0}),
    ]

    workflow_metrics, termination_counts = UnifiedTrainer._collect_workflow_metrics_from_episodes(episodes)

    assert workflow_metrics["custom"] == 2.0
    assert termination_counts[TerminationReason.UNKNOWN.value] == 1
    assert termination_counts[TerminationReason.ERROR.value] == 1
