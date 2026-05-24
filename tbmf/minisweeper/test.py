"""Tests for the MiniSweeper tbmf plugin."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import minisweeper_flow as minisweeper_flow_module
from prepare_minisweeper_data import LAMER_MINISWEEPER_CONFIG, prepare_minisweeper_data
from minisweeper_eval import minisweeper_evaluator
from minisweeper_flow import _build_observation_prompt, _build_system_prompt, minisweeper_flow, parse_action
from minisweeper_pkg import MineSweeper

from rllm.data.dataset import DatasetRegistry
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory


def test_parse_action_tagged_coordinate():
    assert parse_action("reason\n<action>(2, 3)</action>", board_size=5) == (2, 3)


def test_parse_action_takes_last_coordinate():
    text = "<action>(1, 1)</action>\nActually:\n<action>(4, 5)</action>"
    assert parse_action(text, board_size=5) == (4, 5)


def test_parse_action_code_fence_fallback():
    assert parse_action("```action\n3, 2\n```", board_size=5) == (3, 2)


def test_parse_action_rejects_out_of_bounds():
    assert parse_action("<action>(0, 1)</action>", board_size=5) is None
    assert parse_action("<action>(6, 1)</action>", board_size=5) is None


def test_parse_action_invalid():
    assert parse_action("open the corner", board_size=5) is None
    assert parse_action("<action>top left</action>", board_size=5) is None


def test_prompt_helpers_keep_lamer_markers():
    system_prompt = _build_system_prompt(board_size=6, n_mines=3)
    user_prompt = _build_observation_prompt("Row 1: ? ? ?", turn=0, max_turns=7)

    assert "You are an expert agent operating in the Minesweeper game." in system_prompt
    assert "# Cell States" in system_prompt
    assert "Now it's your turn to make a move" in system_prompt
    assert "The initial state of the game is:" in user_prompt
    assert "Row 1: ? ? ?" in user_prompt


def test_prepare_data_defaults_match_lamer_config(tmp_path, monkeypatch):
    monkeypatch.setattr(DatasetRegistry, "_RLLM_HOME", str(tmp_path))
    monkeypatch.setattr(DatasetRegistry, "_DATASET_DIR", str(tmp_path / "datasets"))
    monkeypatch.setattr(DatasetRegistry, "_REGISTRY_FILE", str(tmp_path / "datasets" / "registry.json"))

    train, test = prepare_minisweeper_data()
    train_rows = train.get_data()
    test_rows = test.get_data()
    cfg = LAMER_MINISWEEPER_CONFIG

    assert len(train_rows) == cfg["train_size"]
    assert len(test_rows) == cfg["test_size"]

    row = train_rows[0]
    assert row["board_size"] == cfg["board_size"]
    assert row["n_mines"] == cfg["n_mines"]
    assert row["board_type"] == cfg["board_type"]
    assert row["mode"] == cfg["mode"]
    assert row["max_steps"] == cfg["max_steps"]
    assert row["max_turns"] == cfg["max_turns"]

    expected_train = np.random.RandomState(cfg["env_seed"]).randint(0, 2**16 - 1, size=cfg["train_size"])
    expected_test = np.random.RandomState(cfg["env_seed"] + 1000).randint(
        2**16, 2**32 - 1, size=cfg["test_size"]
    )
    assert [r["seed"] for r in train_rows] == [int(v) for v in expected_train]
    assert [r["seed"] for r in test_rows] == [int(v) for v in expected_test]


def test_env_reset_is_seeded_and_renders_board():
    a = MineSweeper(board_size=5, n_mines=5, board_type="board")
    b = MineSweeper(board_size=5, n_mines=5, board_type="board")
    obs_a, info_a = a.reset(seed=123)
    obs_b, info_b = b.reset(seed=123)

    assert obs_a == obs_b
    assert info_a["won"] is False
    assert info_b["won"] is False
    assert "Row 1:" in obs_a
    assert "?" in obs_a


def test_env_step_reveals_or_penalizes():
    env = MineSweeper(board_size=5, n_mines=5, board_type="board")
    env.reset(seed=123)
    obs, reward, done, info = env.step("L", 1, 1)
    assert isinstance(obs, str)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "won" in info


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
    fake_client = _FakeClient("<action>(1, 1)</action>")
    monkeypatch.setattr(minisweeper_flow_module, "AsyncOpenAI", lambda **_kwargs: fake_client)
    task = Task(
        id="minisweeper-smoke",
        instruction="",
        metadata={"seed": 123, "board_size": 5, "n_mines": 5, "max_steps": 1},
    )
    config = AgentConfig(base_url="http://fake", model="fake", session_uid="test")
    episode = asyncio.run(minisweeper_flow.arun(task, config))

    assert episode.trajectories[0].name == "minisweeper"
    assert episode.artifacts["turns"] == 1
    assert episode.artifacts["env_steps"] == 1


def test_flow_accumulates_system_and_chat_history(monkeypatch):
    fake_client = _FakeClient("<action>(0, 1)</action>")
    monkeypatch.setattr(minisweeper_flow_module, "AsyncOpenAI", lambda **_kwargs: fake_client)
    task = Task(
        id="minisweeper-chat-history",
        instruction="",
        metadata={"seed": 123, "board_size": 5, "n_mines": 5, "max_steps": 2, "max_turns": 2},
    )
    config = AgentConfig(base_url="http://fake", model="fake", session_uid="test")
    episode = asyncio.run(minisweeper_flow.arun(task, config))
    calls = fake_client.chat.completions.calls

    assert episode.artifacts["turns"] == 2
    assert [m["role"] for m in calls[0]] == ["system", "user"]
    assert [m["role"] for m in calls[1]] == ["system", "user", "assistant", "user"]
    assert "You are an expert agent operating in the Minesweeper game." in calls[0][0]["content"]
    assert "The initial state of the game is:" in calls[0][1]["content"]
    assert "Your last response did not contain a valid coordinate" in calls[1][-1]["content"]


def _ep(won: bool, turns: int = 3, env_steps: int = 5) -> Episode:
    return Episode(
        trajectories=[Trajectory(name="minisweeper", steps=[Step()] * turns)],
        artifacts={"won": won, "turns": turns, "env_steps": env_steps, "n_mines": 5},
        is_correct=won,
    )


def test_evaluator_won():
    out = minisweeper_evaluator.evaluate({}, _ep(won=True))
    assert out.is_correct is True
    assert out.reward == 1.0
    accuracy = next(s for s in out.signals if s.name == "accuracy")
    assert accuracy.value == 1.0


def test_evaluator_lost():
    out = minisweeper_evaluator.evaluate({}, _ep(won=False))
    assert out.is_correct is False
    assert out.reward == 0.0
