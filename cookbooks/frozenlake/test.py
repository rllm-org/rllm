"""Tests for the frozenlake cookbook."""

from __future__ import annotations

import pytest

# Skip the whole module if gymnasium isn't installed (optional dep).
gymnasium = pytest.importorskip("gymnasium")

from frozenlake_eval import frozenlake_evaluator  # noqa: E402
from frozenlake_flow import (  # noqa: E402
    _is_solvable,
    generate_random_map,
    parse_action,
    render_grid,
)

from rllm.types import Episode, Step, Trajectory  # noqa: E402

# -- Map generation ------------------------------------------------------------


def test_random_map_size():
    desc = generate_random_map(size=5, p=0.8, seed=1)
    assert len(desc) == 5
    assert all(len(row) == 5 for row in desc)


def test_random_map_has_start_and_goal():
    desc = generate_random_map(size=5, p=0.8, seed=2)
    flat = "".join(desc)
    assert flat.count("S") == 1
    assert flat.count("G") == 1


def test_random_map_is_solvable():
    desc = generate_random_map(size=6, p=0.7, seed=3)
    board = [list(row) for row in desc]
    assert _is_solvable(board, size=6)


def test_random_map_deterministic():
    a = generate_random_map(size=5, p=0.8, seed=42)
    b = generate_random_map(size=5, p=0.8, seed=42)
    assert a == b


# -- Action parsing ------------------------------------------------------------


def test_parse_action_basic():
    assert parse_action("Going up. ```Up```") == 3
    assert parse_action("```Down```") == 1
    assert parse_action("```Left```") == 0
    assert parse_action("```Right```") == 2


def test_parse_action_case_insensitive():
    assert parse_action("```up```") == 3
    assert parse_action("```DOWN```") == 1


def test_parse_action_takes_last_block():
    assert parse_action("First ```Left``` then actually ```Right```") == 2


def test_parse_action_invalid():
    assert parse_action("no action here") is None
    assert parse_action("```diagonal```") is None
    assert parse_action("```42```") is None


def test_parse_action_with_thinking():
    text = "I think ```Left``` is wrong. Let me try ```Up``` instead."
    assert parse_action(text) == 3


# -- Render --------------------------------------------------------------------


def test_render_grid_shape():
    desc = generate_random_map(size=4, p=0.85, seed=7)
    env = gymnasium.make("FrozenLake-v1", desc=desc, is_slippery=False)
    env.reset(seed=7)
    rendered = render_grid(env)
    assert rendered.count("\n") == 3  # 4 rows -> 3 newlines
    assert "P" in rendered  # player is always shown
    assert "G" in rendered  # goal is always shown


# -- Evaluator -----------------------------------------------------------------


def _ep(won: bool, turns: int = 3) -> Episode:
    return Episode(
        trajectories=[Trajectory(name="frozenlake", steps=[Step()] * turns)],
        artifacts={"won": won, "turns": turns},
        is_correct=won,
    )


def test_evaluator_won():
    out = frozenlake_evaluator.evaluate({}, _ep(won=True, turns=4))
    assert out.is_correct is True
    assert out.reward == 1.0
    accuracy = next(s for s in out.signals if s.name == "accuracy")
    assert accuracy.value == 1.0


def test_evaluator_lost():
    out = frozenlake_evaluator.evaluate({}, _ep(won=False, turns=8))
    assert out.is_correct is False
    assert out.reward == 0.0
