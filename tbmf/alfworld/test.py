"""Tests for the alfworld cookbook."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from alfworld_eval import alfworld_evaluator
from alfworld_flow import _run_env_init, parse_action

from rllm.types import Episode, Step, Trajectory

# -- Action parsing ------------------------------------------------------------


def test_parse_action_basic():
    text = "<think>\nI need to go to the countertop.\n</think>\n```action\ngo to countertop 1\n```"
    assert parse_action(text) == "go to countertop 1"


def test_parse_action_multiple_blocks():
    text = "```action\ntake apple 1 from table 1\n```\nActually:\n```action\ngo to fridge 1\n```"
    assert parse_action(text) == "go to fridge 1"


def test_parse_action_case_insensitive_block():
    text = "```Action\nopen fridge 1\n```"
    assert parse_action(text) == "open fridge 1"


def test_parse_action_no_block():
    text = "I think I should go to the fridge."
    assert parse_action(text) is None


def test_parse_action_generic_backticks():
    text = "Let me try:\n```\nlook\n```"
    assert parse_action(text) == "look"


def test_parse_action_with_whitespace():
    text = "```action\n  heat egg 1 with microwave 1  \n```"
    assert parse_action(text) == "heat egg 1 with microwave 1"


def test_parse_action_empty_block():
    text = "```action\n\n```"
    assert parse_action(text) is None


def test_env_init_lock_does_not_starve_default_executor():
    async def scenario():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=2))

        started = threading.Event()
        release = threading.Event()

        def slow_init(name: str) -> str:
            if name == "first":
                started.set()
                assert release.wait(timeout=2), "test did not release slow init"
            return name

        first = asyncio.create_task(_run_env_init(slow_init, "first"))
        assert await asyncio.to_thread(started.wait, 2), "first init did not start"

        second = asyncio.create_task(_run_env_init(slow_init, "second"))
        await asyncio.sleep(0.05)

        marker = await asyncio.wait_for(asyncio.to_thread(lambda: "executor-free"), timeout=0.5)
        assert marker == "executor-free"

        release.set()
        assert await first == "first"
        assert await second == "second"

    asyncio.run(scenario())


# -- Evaluator -----------------------------------------------------------------


def _ep(won: bool, turns: int = 5, task_type: str = "pick_and_place_simple") -> Episode:
    return Episode(
        trajectories=[Trajectory(name="alfworld", steps=[Step()] * turns)],
        artifacts={"won": won, "turns": turns, "task_type": task_type},
        is_correct=won,
    )


def test_evaluator_won():
    out = alfworld_evaluator.evaluate({}, _ep(won=True, turns=8))
    assert out.is_correct is True
    assert out.reward == 1.0
    accuracy = next(s for s in out.signals if s.name == "accuracy")
    assert accuracy.value == 1.0


def test_evaluator_lost():
    out = alfworld_evaluator.evaluate({}, _ep(won=False, turns=50))
    assert out.is_correct is False
    assert out.reward == 0.0


def test_evaluator_signals():
    out = alfworld_evaluator.evaluate({}, _ep(won=True, turns=12, task_type="look_at_obj_in_light"))
    turns_signal = next(s for s in out.signals if s.name == "turns")
    assert turns_signal.value == 12.0
    task_signal = next(s for s in out.signals if s.name == "task_type")
    assert task_signal.value == "look_at_obj_in_light"
