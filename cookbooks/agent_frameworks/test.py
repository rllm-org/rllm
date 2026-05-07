"""Tests for the framework cookbook's shared modules.

The flow files themselves are smoke-tested only via the import + protocol
check at the bottom; the flows depend on external framework SDKs whose
behavior we don't test here.
"""

import pytest
from calculator import safe_eval
from evaluator import _extract_answer, _extract_boxed, math_evaluator

from rllm.types import Episode, Step, Trajectory

# -- Calculator ---------------------------------------------------------------


def test_safe_eval_basic():
    assert safe_eval("2 + 3") == "5"
    assert safe_eval("34 * 17") == "578"


def test_safe_eval_float():
    assert safe_eval("10 / 4") == "2.5"
    assert safe_eval("10 / 2") == "5"


def test_safe_eval_sqrt():
    assert safe_eval("sqrt(64 + 225)") == "17"


def test_safe_eval_combinatorics():
    assert safe_eval("comb(5, 2)") == "10"
    assert safe_eval("factorial(5)") == "120"


def test_safe_eval_rejects_invalid():
    assert "Error" in safe_eval("import os")
    assert "Error" in safe_eval("__import__('os')")
    assert "Error" in safe_eval("(1).bit_length()")
    assert "Error" in safe_eval("a" * 201)


# -- Answer extraction --------------------------------------------------------


def test_extract_boxed():
    assert _extract_boxed(r"Therefore \boxed{42}") == "42"
    assert _extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert _extract_boxed("no boxed here") is None


def test_extract_answer_natural():
    assert _extract_answer("The final answer is 42.") == "42"


# -- Evaluator ----------------------------------------------------------------


def _episode_with_last_response(text: str, name: str = "math") -> Episode:
    return Episode(trajectories=[Trajectory(name=name, steps=[Step(model_response=text)])])


def test_evaluator_correct():
    task = {"answer": "10"}
    episode = _episode_with_last_response(r"After computing, \boxed{10}")
    result = math_evaluator.evaluate(task, episode)
    assert result.is_correct is True
    assert result.reward == 1.0


def test_evaluator_wrong():
    task = {"answer": "10"}
    episode = _episode_with_last_response(r"\boxed{9}")
    result = math_evaluator.evaluate(task, episode)
    assert result.is_correct is False


def test_evaluator_walks_back_past_empty_steps():
    task = {"answer": "10"}
    episode = Episode(
        trajectories=[
            Trajectory(
                name="math",
                steps=[
                    Step(model_response="I'll calculate 5 + 5"),
                    Step(model_response=""),  # tool message
                    Step(model_response=r"The answer is \boxed{10}"),
                ],
            )
        ]
    )
    result = math_evaluator.evaluate(task, episode)
    assert result.is_correct is True


def test_evaluator_no_response():
    task = {"answer": "10"}
    episode = Episode(trajectories=[Trajectory(name="math", steps=[])])
    result = math_evaluator.evaluate(task, episode)
    assert result.is_correct is False


# -- Flow protocol smoke tests ------------------------------------------------
#
# Each flow imports its framework SDK at module import time. We import each
# under a try/except so missing optional deps don't crash the suite.


@pytest.mark.parametrize(
    "module_name,attr_name,traj_name",
    [
        ("agentflow.langgraph", "langgraph_math", "langgraph-math"),
        ("agentflow.openai_agents", "openai_agents_math", "openai-agents-math"),
        ("agentflow.smolagents", "smolagents_math", "smolagents-math"),
        ("agentflow.strands", "strands_math", "strands-math"),
    ],
)
def test_flow_satisfies_agentflow_protocol(module_name, attr_name, traj_name):
    pytest.importorskip(
        {
            "agentflow.langgraph": "langgraph",
            "agentflow.openai_agents": "agents",
            "agentflow.smolagents": "smolagents",
            "agentflow.strands": "strands",
        }[module_name]
    )
    import importlib

    mod = importlib.import_module(module_name)
    flow = getattr(mod, attr_name)
    from rllm.types import AgentFlow

    assert isinstance(flow, AgentFlow)
    assert flow._name == traj_name
