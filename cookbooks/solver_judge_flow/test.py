"""Tests for solver-judge flow."""

from evaluator import solver_judge_countdown_evaluator
from solver_judge_flow import _parse_answer, _parse_judge_response

from rllm.types import Episode, Step, Trajectory


def test_parse_answer_extracts_boxed():
    assert _parse_answer("blah <answer>(1+2)*3</answer> blah") == "<answer>(1+2)*3</answer>"


def test_parse_answer_no_match():
    assert _parse_answer("no answer here") == "No solution found"


def test_parse_judge_response():
    assert _parse_judge_response("<answer>2</answer>", ["sol1", "sol2"]) == "sol2"


def test_parse_judge_response_invalid():
    assert _parse_judge_response("<answer>bad</answer>", ["sol1"]) == ""


def test_evaluator_scores_trajectories():
    task = {"question": "reach 6", "target": 6, "nums": [1, 2, 3]}

    episode = Episode(
        trajectories=[
            Trajectory(name="solver", steps=[Step(action="<answer>1 + 2 + 3</answer>")]),
            Trajectory(name="solver", steps=[Step(action="<answer>wrong</answer>")]),
            Trajectory(name="judge", steps=[Step(action="<answer>1 + 2 + 3</answer>")]),
        ],
    )

    result = solver_judge_countdown_evaluator.evaluate(task, episode)

    assert result.is_correct is True
    assert result.reward == 1.0
    assert episode.trajectories[0].reward == 1.0
    assert episode.trajectories[1].reward == 0.0
    assert episode.trajectories[2].reward == 1.0


def test_evaluator_wrong_judge():
    task = {"question": "reach 6", "target": 6, "nums": [1, 2, 3]}

    episode = Episode(
        trajectories=[
            Trajectory(name="solver", steps=[Step(action="<answer>1 + 2 + 3</answer>")]),
            Trajectory(name="judge", steps=[Step(action="<answer>wrong</answer>")]),
        ],
    )

    result = solver_judge_countdown_evaluator.evaluate(task, episode)

    assert result.is_correct is False
    assert result.reward == 0.0
