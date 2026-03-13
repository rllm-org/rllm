"""Tests for the FrozenLake agent plugin."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent.agent import FrozenLakeAgentFlow, _parse_action
from agent.env import FrozenLakeEnv, generate_random_map
from eval.evaluator import FrozenLakeEvaluator

from rllm.experimental.eval.types import AgentConfig
from rllm.types import Episode

# ---------------------------------------------------------------------------
# Environment tests
# ---------------------------------------------------------------------------


class TestGenerateRandomMap:
    def test_deterministic(self):
        m1, g1 = generate_random_map(size=4, p=0.8, seed=42)
        m2, g2 = generate_random_map(size=4, p=0.8, seed=42)
        assert m1 == m2
        assert g1 == g2

    def test_contains_start_and_goal(self):
        map_rows, goal = generate_random_map(size=4, p=0.8, seed=0)
        flat = "".join(map_rows)
        assert "S" in flat
        assert "G" in flat

    def test_different_seeds_differ(self):
        m1, _ = generate_random_map(size=4, p=0.8, seed=0)
        m2, _ = generate_random_map(size=4, p=0.8, seed=99)
        # Very unlikely to be the same
        assert m1 != m2

    def test_size(self):
        for sz in [2, 4, 6]:
            map_rows, _ = generate_random_map(size=sz, p=0.8, seed=42)
            assert len(map_rows) == sz
            assert all(len(r) == sz for r in map_rows)


class TestFrozenLakeEnv:
    def test_reset_returns_string(self):
        env = FrozenLakeEnv(size=4, p=0.8, seed=42)
        obs = env.reset()
        assert isinstance(obs, str)
        assert "P" in obs

    def test_invalid_action_noop(self):
        env = FrozenLakeEnv(size=4, p=0.8, seed=42)
        env.reset()
        obs1 = env.render()
        obs2, reward, done, info = env.step(0)
        assert obs1 == obs2
        assert reward == 0.0
        assert not done
        assert not info["action_is_effective"]

    def test_step_moves_player(self):
        env = FrozenLakeEnv(size=4, p=0.8, seed=42)
        env.reset()
        # Try all 4 directions; at least one should move
        moved = False
        for action in [1, 2, 3, 4]:
            env.reset()
            _, _, _, info = env.step(action)
            if info["action_is_effective"]:
                moved = True
                break
        assert moved, "At least one direction should be effective"

    def test_goal_detection(self):
        env = FrozenLakeEnv(size=4, p=0.8, seed=42)
        env.reset()
        # Manually place player on goal
        env._player = env._goal
        env._done = True
        assert env.success()
        assert env.finished()

    def test_render_symbols(self):
        env = FrozenLakeEnv(size=4, p=0.8, seed=42)
        obs = env.reset()
        # Should contain standard symbols
        assert " P " in obs
        # Should contain frozen or hole or goal
        has_expected = " _ " in obs or " O " in obs or " G " in obs
        assert has_expected


# ---------------------------------------------------------------------------
# Action parsing tests
# ---------------------------------------------------------------------------


class TestParseAction:
    def test_simple_directions(self):
        assert _parse_action("I should go ```Up```") == 4
        assert _parse_action("```Down```") == 2
        assert _parse_action("```Left```") == 1
        assert _parse_action("```Right```") == 3

    def test_case_insensitive(self):
        assert _parse_action("```up```") == 4
        assert _parse_action("```DOWN```") == 2

    def test_numeric(self):
        assert _parse_action("```1```") == 1
        assert _parse_action("```4```") == 4

    def test_no_backticks(self):
        assert _parse_action("I want to go Up") == 0

    def test_invalid_content(self):
        assert _parse_action("```invalid```") == 0

    def test_last_match_wins(self):
        assert _parse_action("```Up``` then ```Down```") == 2


# ---------------------------------------------------------------------------
# Agent flow tests
# ---------------------------------------------------------------------------


class TestFrozenLakeAgentFlow:
    def test_protocol_conformance(self):
        from rllm.experimental.eval.types import AgentFlow

        assert isinstance(FrozenLakeAgentFlow(), AgentFlow)

    @patch("agent.agent.openai.OpenAI")
    def test_run_with_mock_llm(self, mock_openai_cls):
        # Set up mock to always return "Up"
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = "I should go up. ```Up```"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        agent = FrozenLakeAgentFlow()
        config = AgentConfig(base_url="http://localhost:8000/v1", model="test", session_uid="test")
        task = {"seed": 42, "size": 4, "p": 0.8, "max_steps": 5}

        episode = agent.run(task, config)

        assert isinstance(episode, Episode)
        assert len(episode.trajectories) == 1
        assert episode.trajectories[0].name == "navigator"
        assert len(episode.trajectories[0].steps) > 0
        assert "success" in episode.artifacts
        assert "num_steps" in episode.artifacts


# ---------------------------------------------------------------------------
# Evaluator tests
# ---------------------------------------------------------------------------


class TestFrozenLakeEvaluator:
    def test_success(self):
        evaluator = FrozenLakeEvaluator()
        episode = Episode(
            id="test:0",
            artifacts={"success": True, "num_steps": 3},
        )
        result = evaluator.evaluate({}, episode)
        assert result.reward == 1.0
        assert result.is_correct is True
        assert any(s.name == "success" and s.value == 1.0 for s in result.signals)

    def test_failure(self):
        evaluator = FrozenLakeEvaluator()
        episode = Episode(
            id="test:0",
            artifacts={"success": False, "num_steps": 5},
        )
        result = evaluator.evaluate({}, episode)
        assert result.reward == 0.0
        assert result.is_correct is False

    def test_missing_artifacts(self):
        evaluator = FrozenLakeEvaluator()
        episode = Episode(id="test:0")
        result = evaluator.evaluate({}, episode)
        assert result.reward == 0.0
        assert result.is_correct is False

    def test_num_steps_signal(self):
        evaluator = FrozenLakeEvaluator()
        episode = Episode(
            id="test:0",
            artifacts={"success": True, "num_steps": 7},
        )
        result = evaluator.evaluate({}, episode)
        step_signal = next(s for s in result.signals if s.name == "num_steps")
        assert step_signal.value == 7.0
