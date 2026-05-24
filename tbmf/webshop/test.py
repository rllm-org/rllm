"""Tests for the WebShop tbmf plugin."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ray")

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PACKAGE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import webshop_flow as webshop_flow_module  # noqa: E402
from prepare_webshop_data import LAMER_WEBSHOP_CONFIG, prepare_webshop_data  # noqa: E402
from webshop_eval import webshop_evaluator  # noqa: E402
from webshop_flow import _build_system_prompt, _build_user_prompt, parse_action, webshop_flow  # noqa: E402

from rllm.data.dataset import DatasetRegistry  # noqa: E402
from webshop_env import WebShopEnv  # noqa: E402
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory  # noqa: E402


def test_parse_action_action_block():
    assert parse_action("reason\n```action\nsearch[red shoes]\n```", has_search_bar=True) == "search[red shoes]"


def test_parse_action_last_block_wins():
    text = "<think>search</think>\n```action\nclick[Product 1]\n```\n```action\nclick[Buy Now]\n```"
    assert parse_action(text, has_search_bar=False) == "click[Buy Now]"


def test_parse_action_inline_fallback():
    assert parse_action("I should search[laptop] now", has_search_bar=False) == "search[laptop]"


def test_parse_action_wraps_plain_text():
    assert parse_action("blue jacket", has_search_bar=True) == "search[blue jacket]"
    assert parse_action("blue jacket", has_search_bar=False) == "click[back to search]"


def test_prompt_helpers_include_actions():
    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(
        observation="Welcome",
        instruction="Find a jacket",
        available_actions={"has_search_bar": True, "clickables": ["Product 1", "Next"]},
        turn=0,
        max_turns=5,
    )

    assert "search[query]" in system_prompt
    assert "click[element]" in system_prompt
    assert "Shopping Task" in user_prompt
    assert "Available Actions" in user_prompt
    assert "Product 1" in user_prompt


def test_prepare_data_defaults_match_lamer_config(tmp_path, monkeypatch):
    monkeypatch.setattr(DatasetRegistry, "_RLLM_HOME", str(tmp_path))
    monkeypatch.setattr(DatasetRegistry, "_DATASET_DIR", str(tmp_path / "datasets"))
    monkeypatch.setattr(DatasetRegistry, "_REGISTRY_FILE", str(tmp_path / "datasets" / "registry.json"))

    train, test = prepare_webshop_data()
    train_rows = train.get_data()
    test_rows = test.get_data()
    cfg = LAMER_WEBSHOP_CONFIG

    assert len(train_rows) == cfg["train_size"]
    assert len(test_rows) == cfg["test_size"]

    row = train_rows[0]
    assert row["seed"] == cfg["env_seed"]
    assert row["observation_mode"] == cfg["observation_mode"]
    assert row["num_products"] == cfg["num_products"]
    assert row["human_goals"] == cfg["human_goals"]
    assert row["max_steps"] == cfg["max_steps"]
    assert row["max_turns"] == cfg["max_turns"]
    assert row["use_available_actions"] == cfg["use_available_actions"]

    expected_train = np.random.RandomState(cfg["env_seed"]).choice(
        np.arange(1500, 12000), size=cfg["train_size"], replace=False
    )
    expected_test = np.random.RandomState(cfg["env_seed"] + 1000).choice(
        np.arange(0, 500), size=cfg["test_size"], replace=False
    )
    assert [r["session_id"] for r in train_rows] == [int(v) for v in expected_train]
    assert [r["session_id"] for r in test_rows] == [int(v) for v in expected_test]


def test_env_basic_construction():
    env = WebShopEnv(observation_mode="text", max_steps=12, num_products=1000, session_id=7, seed=42)
    assert env.max_steps == 12
    assert env.observation_mode == "text"
    assert env.num_products == 1000
    assert env.session_id == 7
    assert env.seed == 42
    assert env._initialized is False


def test_env_from_dict():
    env = WebShopEnv.from_dict(
        {
            "observation_mode": "html",
            "max_steps": 99,
            "seed": 123,
            "session_id": 5,
            "max_workers": 8,
            "ignored": "value",
        }
    )
    assert env.observation_mode == "html"
    assert env.max_steps == 99
    assert env.seed == 123
    assert env.session_id == 5


def test_env_repr():
    env = WebShopEnv(observation_mode="text_rich", max_steps=30, num_products=1000)
    text = repr(env)
    assert "WebShopEnv" in text
    assert "observation_mode=text_rich" in text
    assert "max_steps=30" in text


def test_env_thread_safe_flag():
    assert WebShopEnv.is_multithread_safe() is True


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


class _FakeEnv:
    def __init__(self):
        self.closed = False
        self.reset_calls = 0
        self.step_calls: list[str] = []

    def reset(self, seed: int | None = None):
        self.reset_calls += 1
        return "WebShop start page", {
            "instruction": "Find a red jacket",
            "available_actions": {"has_search_bar": True, "clickables": []},
        }

    def step(self, action: str):
        self.step_calls.append(action)
        return (
            "WebShop product page",
            1.0,
            True,
            {
                "instruction": "Find a red jacket",
                "available_actions": {"has_search_bar": False, "clickables": ["Buy Now"]},
                "task_score": 1.0,
                "won": True,
                "success": True,
            },
        )

    def close(self):
        self.closed = True


def test_flow_smoke_with_mocked_env(monkeypatch):
    fake_env = _FakeEnv()
    fake_client = _FakeClient("<action>search[red jacket]</action>")
    monkeypatch.setattr(webshop_flow_module, "WebShopEnv", lambda **_kwargs: fake_env)
    monkeypatch.setattr(webshop_flow_module, "AsyncOpenAI", lambda **_kwargs: fake_client)

    task = Task(
        id="webshop-smoke",
        instruction="WebShop session 7",
        metadata={
            "session_id": 7,
            "seed": 42,
            "observation_mode": "text",
            "num_products": 1000,
            "human_goals": False,
            "max_steps": 2,
            "max_turns": 2,
            "file_path": "/tmp/items_shuffle_1000.json",
            "attr_path": "/tmp/items_ins_v2_1000.json",
        },
    )
    config = AgentConfig(base_url="http://fake", model="fake", session_uid="test")
    episode = asyncio.run(webshop_flow.arun(task, config))

    assert episode.trajectories[0].name == "webshop"
    assert episode.is_correct is True
    assert episode.artifacts["won"] is True
    assert episode.artifacts["task_score"] == 1.0
    assert fake_env.closed is True
    assert fake_env.step_calls == ["search[red jacket]"]
    assert episode.trajectories[0].steps[0].action == "search[red jacket]"
    assert "Shopping Task" in episode.trajectories[0].steps[0].observation


def test_evaluator_won():
    ep = Episode(
        trajectories=[Trajectory(name="webshop", steps=[Step()])],
        artifacts={"won": True, "task_score": 1.0, "turns": 1, "env_steps": 1},
        is_correct=True,
    )
    out = webshop_evaluator.evaluate({}, ep)
    assert out.reward == 1.0
    assert out.is_correct is True
    assert any(sig.name == "task_score" for sig in out.signals)
