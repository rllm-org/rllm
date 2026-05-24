"""Tests for the Sokoban tbmf plugin."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gym_sokoban")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sokoban_flow as sokoban_flow_module  # noqa: E402
from prepare_sokoban_data import LAMER_SOKOBAN_CONFIG, prepare_sokoban_data  # noqa: E402
from sokoban_eval import sokoban_evaluator  # noqa: E402
from sokoban_flow import _build_observation_prompt, _build_system_prompt, _parse_dim_room, parse_actions, sokoban_flow  # noqa: E402
from sokoban_pkg import SokobanEnv  # noqa: E402

from rllm.data.dataset import DatasetRegistry  # noqa: E402
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory  # noqa: E402


def test_parse_actions_tagged_list():
    assert parse_actions("reason\n<action>up, right, down</action>", max_actions=3) == [1, 4, 2]


def test_parse_actions_takes_last_block():
    text = "<action>left</action>\nActually:\n<action>right, up</action>"
    assert parse_actions(text, max_actions=3) == [4, 1]


def test_parse_actions_code_fence_fallback():
    assert parse_actions("```action\nleft, down\n```", max_actions=3) == [3, 2]


def test_parse_actions_invalid():
    assert parse_actions("move somewhere") is None
    assert parse_actions("<action>jump</action>") is None


def test_parse_actions_truncates():
    assert parse_actions("<action>up, down, left, right</action>", max_actions=2) == [1, 2]


def test_parse_dim_room():
    assert _parse_dim_room("7") == (7, 7)
    assert _parse_dim_room("6x8") == (6, 8)
    assert _parse_dim_room([5, 6]) == (5, 6)


def test_prompt_helpers_keep_lamer_markers():
    system_prompt = _build_system_prompt(actions_per_turn=3)
    user_prompt = _build_observation_prompt("0: # # #\n1: # P #", turn=0, max_turns=7, actions_per_turn=3)

    assert "You are an expert agent operating in the Sokoban environment." in system_prompt
    assert "# Symbols and Their Meaning" in system_prompt
    assert "Now it's your turn to make moves" in system_prompt
    assert "The initial state of the game is:" in user_prompt
    assert "choose the next THREE actions" in user_prompt


def test_prepare_data_defaults_match_lamer_config(tmp_path, monkeypatch):
    monkeypatch.setattr(DatasetRegistry, "_RLLM_HOME", str(tmp_path))
    monkeypatch.setattr(DatasetRegistry, "_DATASET_DIR", str(tmp_path / "datasets"))
    monkeypatch.setattr(DatasetRegistry, "_REGISTRY_FILE", str(tmp_path / "datasets" / "registry.json"))

    train, test = prepare_sokoban_data()
    train_rows = train.get_data()
    test_rows = test.get_data()
    cfg = LAMER_SOKOBAN_CONFIG

    assert len(train_rows) == cfg["train_size"]
    assert len(test_rows) == cfg["test_size"]

    row = train_rows[0]
    assert row["dim_room"] == list(cfg["dim_room"])
    assert row["num_boxes"] == cfg["num_boxes"]
    assert row["max_steps"] == cfg["max_steps"]
    assert row["search_depth"] == cfg["search_depth"]
    assert row["min_steps"] == cfg["min_steps"]
    assert row["max_sol_steps"] == cfg["max_sol_steps"]
    assert row["actions_per_turn"] == cfg["actions_per_turn"]
    assert row["max_turns"] == cfg["max_turns"]
    assert row["mode"] == cfg["mode"]

    expected_train = np.random.RandomState(cfg["env_seed"]).randint(0, 2**16 - 1, size=cfg["train_size"])
    expected_test = np.random.RandomState(cfg["env_seed"] + 1000).randint(
        2**16, 2**32 - 1, size=cfg["test_size"]
    )
    assert [r["seed"] for r in train_rows] == [int(v) for v in expected_train]
    assert [r["seed"] for r in test_rows] == [int(v) for v in expected_test]


def test_env_reset_renders_board():
    env = SokobanEnv(
        mode="text_with_row_numbers",
        dim_room=(6, 6),
        num_boxes=1,
        max_steps=30,
        search_depth=80,
        min_steps=2,
        max_sol_steps=20,
    )
    obs, info = env.reset(seed=42)
    assert info["won"] is False
    assert "0:" in obs
    assert "P" in obs
    assert "X" in obs or "√" in obs


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content: str):
        self.content = content
        self.calls: list[list[dict]] = []

    async def create(self, **kwargs):
        self.calls.append(list(kwargs["messages"]))
        return _FakeResponse(self.content)


class _FakeChat:
    def __init__(self, content: str):
        self.completions = _FakeCompletions(content)


class _FakeClient:
    def __init__(self, content: str):
        self.chat = _FakeChat(content)


def test_flow_smoke_with_mocked_llm(monkeypatch):
    fake_client = _FakeClient("<action>up</action>")
    monkeypatch.setattr(sokoban_flow_module, "AsyncOpenAI", lambda **_kwargs: fake_client)
    task = Task(
        id="sokoban-smoke",
        instruction="",
        metadata={
            "seed": 42,
            "dim_room": [6, 6],
            "num_boxes": 1,
            "max_steps": 2,
            "search_depth": 80,
            "min_steps": 1,
            "max_sol_steps": 20,
            "actions_per_turn": 1,
            "max_turns": 1,
        },
    )
    config = AgentConfig(base_url="http://fake", model="fake", session_uid="test")
    episode = asyncio.run(sokoban_flow.arun(task, config))

    assert episode.trajectories[0].name == "sokoban"
    step = episode.trajectories[0].steps[0]
    assert "The initial state of the game is:" in step.observation
    assert step.model_response == "<action>up</action>"
    assert [m["role"] for m in step.chat_completions] == ["system", "user", "assistant"]
    assert episode.artifacts["turns"] == 1
    assert episode.artifacts["env_steps"] == 1


def test_flow_accumulates_system_and_chat_history(monkeypatch):
    fake_client = _FakeClient("<action>jump</action>")
    monkeypatch.setattr(sokoban_flow_module, "AsyncOpenAI", lambda **_kwargs: fake_client)
    task = Task(
        id="sokoban-chat-history",
        instruction="",
        metadata={
            "seed": 42,
            "dim_room": [6, 6],
            "num_boxes": 1,
            "max_steps": 2,
            "search_depth": 80,
            "min_steps": 1,
            "max_sol_steps": 20,
            "actions_per_turn": 1,
            "max_turns": 2,
        },
    )
    config = AgentConfig(base_url="http://fake", model="fake", session_uid="test")
    episode = asyncio.run(sokoban_flow.arun(task, config))
    calls = fake_client.chat.completions.calls

    assert episode.artifacts["turns"] == 2
    assert [m["role"] for m in calls[0]] == ["system", "user"]
    assert [m["role"] for m in calls[1]] == ["system", "user", "assistant", "user"]
    assert "You are an expert agent operating in the Sokoban environment." in calls[0][0]["content"]
    assert "The initial state of the game is:" in calls[0][1]["content"]
    assert "Your last response did not contain a valid action" in calls[1][-1]["content"]
    assert episode.trajectories[0].steps[0].observation == calls[0][1]["content"]
    assert episode.trajectories[0].steps[1].observation == calls[1][-1]["content"]


def _ep(won: bool, turns: int = 3, env_steps: int = 5) -> Episode:
    return Episode(
        trajectories=[Trajectory(name="sokoban", steps=[Step()] * turns)],
        artifacts={"won": won, "turns": turns, "env_steps": env_steps, "num_boxes": 1},
        is_correct=won,
    )


def test_evaluator_won():
    out = sokoban_evaluator.evaluate({}, _ep(won=True))
    assert out.is_correct is True
    assert out.reward == 1.0
    accuracy = next(s for s in out.signals if s.name == "accuracy")
    assert accuracy.value == 1.0


def test_evaluator_lost():
    out = sokoban_evaluator.evaluate({}, _ep(won=False))
    assert out.is_correct is False
    assert out.reward == 0.0
