"""Tests for the model-free no-op harness."""

from rllm.eval.agent_loader import load_agent
from rllm.harnesses.nop import NopHarness
from rllm.types import AgentConfig, Task


class _UntouchedEnv:
    def __getattr__(self, name):
        raise AssertionError(f"nop harness accessed sandbox attribute {name!r}")


def test_nop_harness_leaves_sandbox_untouched():
    harness = NopHarness()
    task = Task(id="task-1", instruction="Do nothing")
    config = AgentConfig(base_url="", model="none", session_uid="task-1:0")

    episode = harness.run(task, config, env=_UntouchedEnv())

    assert episode.id == "task-1:0"
    assert episode.task == "task-1"
    assert episode.trajectories[0].name == "nop"
    assert episode.trajectories[0].steps[0].output == "[nop] no changes applied"
    assert harness.makes_llm_calls is False


def test_nop_harness_is_available_by_builtin_name():
    assert isinstance(load_agent("nop"), NopHarness)
