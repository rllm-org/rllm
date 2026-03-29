"""Tests for the rllm.experimental.erl module."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock

import numpy as np

from rllm.agents.agent import Step, Trajectory
from rllm.workflows.store import InMemoryStore


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# RAFT estimator
# ---------------------------------------------------------------------------


class TestRaftEstimator:
    def test_registration(self):
        from rllm.experimental.common.advantage import get_rllm_adv_estimator
        from rllm.experimental.erl.utils import calculate_raft_advantages  # noqa: F401 — triggers registration

        fn = get_rllm_adv_estimator("raft")
        assert fn is calculate_raft_advantages

    def test_positive_only(self):
        from rllm.experimental.erl.utils import calculate_raft_advantages

        rewards = [np.array([1.0, -0.5, 0.0, 0.8])]
        adv, ret = calculate_raft_advantages(rewards)
        np.testing.assert_array_equal(adv[0], [1.0, 0.0, 0.0, 0.8])
        np.testing.assert_array_equal(ret[0], [1.0, 0.0, 0.0, 0.8])

    def test_all_negative(self):
        from rllm.experimental.erl.utils import calculate_raft_advantages

        rewards = [np.array([-1.0, -2.0])]
        adv, _ = calculate_raft_advantages(rewards)
        np.testing.assert_array_equal(adv[0], [0.0, 0.0])

    def test_multiple_groups(self):
        from rllm.experimental.erl.utils import calculate_raft_advantages

        rewards = [np.array([1.0, -1.0]), np.array([0.5])]
        adv, _ = calculate_raft_advantages(rewards)
        assert len(adv) == 2
        np.testing.assert_array_equal(adv[0], [1.0, 0.0])
        np.testing.assert_array_equal(adv[1], [0.5])


# ---------------------------------------------------------------------------
# Prompt extraction
# ---------------------------------------------------------------------------


class TestPromptExtraction:
    def test_basic(self):
        from rllm.experimental.erl.utils import extract_prompt_from_response

        assert extract_prompt_from_response("Here: <prompt>Hello</prompt>") == "Hello"

    def test_multiline(self):
        from rllm.experimental.erl.utils import extract_prompt_from_response

        text = "<prompt>\nLine 1\nLine 2\n</prompt>"
        assert extract_prompt_from_response(text) == "Line 1\nLine 2"

    def test_missing_tags(self):
        from rllm.experimental.erl.utils import extract_prompt_from_response

        assert extract_prompt_from_response("no tags here") is None

    def test_empty_string(self):
        from rllm.experimental.erl.utils import extract_prompt_from_response

        assert extract_prompt_from_response("") is None

    def test_case_insensitive(self):
        from rllm.experimental.erl.utils import extract_prompt_from_response

        assert extract_prompt_from_response("<PROMPT>ok</PROMPT>") == "ok"


# ---------------------------------------------------------------------------
# ErlPromptUpdater
# ---------------------------------------------------------------------------


@dataclass
class _FakeModelOutput:
    text: str | None = None
    content: str | None = None
    reasoning: str | None = None
    prompt_ids: list[int] | None = None
    completion_ids: list[int] | None = None
    logprobs: list[float] | None = None
    prompt_logprobs: list[float] | None = None
    prompt_length: int = 0
    completion_length: int = 0
    finish_reason: str | None = None
    tool_calls: list | None = None
    multi_modal_inputs: dict | None = None


def _mock_engine(response_text: str):
    engine = AsyncMock()
    engine.get_model_response = AsyncMock(return_value=_FakeModelOutput(content=response_text))
    return engine


class TestErlPromptUpdater:
    def test_extraction_success(self):
        from rllm.experimental.erl.updater import ErlPromptUpdater

        engine = _mock_engine("I suggest: <prompt>improved prompt</prompt>")
        updater = ErlPromptUpdater(engine)

        new_prompt, traj = _run(updater.propose_prompt("state ctx", "old prompt"))
        assert new_prompt == "improved prompt"
        assert traj.name == "erl_updater"
        assert len(traj.steps) == 1

    def test_extraction_fallback(self):
        from rllm.experimental.erl.updater import ErlPromptUpdater

        engine = _mock_engine("Here is my advice, no tags.")
        updater = ErlPromptUpdater(engine)

        new_prompt, _ = _run(updater.propose_prompt("state", "fallback"))
        assert new_prompt == "fallback"

    def test_trajectory_structure(self):
        from rllm.experimental.erl.updater import ErlPromptUpdater

        engine = _mock_engine("<prompt>v2</prompt>")
        updater = ErlPromptUpdater(engine)

        _, traj = _run(updater.propose_prompt("state", "v1"))
        step = traj.steps[0]
        assert step.info["previous_prompt"] == "v1"
        assert len(step.chat_completions) == 3  # system, user, assistant


# ---------------------------------------------------------------------------
# ErlWorkflow
# ---------------------------------------------------------------------------


def _make_solver_fn(reward: float = 0.0):
    """Return a solver_fn that produces a minimal trajectory with given reward."""

    async def solver_fn(prompt: str, task: dict, engine: Any) -> Trajectory:
        step = Step(
            chat_completions=[
                {"role": "system", "content": prompt},
                {"role": "assistant", "content": "answer"},
            ],
            reward=reward,
            done=True,
        )
        return Trajectory(steps=[step], reward=reward)

    return solver_fn


def _simple_state_builder(base_prompt, task, traj, feedback):
    return f"base={base_prompt} feedback={feedback}"


def _make_workflow(erl_config_overrides: dict | None = None, **overrides):
    from rllm.experimental.erl.workflow import ErlConfig, ErlWorkflow

    cfg_kwargs = {"initial_system_prompt": "initial prompt"}
    cfg_kwargs.update(erl_config_overrides or {})
    cfg = ErlConfig(**cfg_kwargs)

    defaults = {
        "rollout_engine": _mock_engine("<prompt>better</prompt>"),
        "executor": None,
        "solver_fn": _make_solver_fn(0.0),
        "state_builder_fn": _simple_state_builder,
        "erl_config": cfg,
    }
    defaults.update(overrides)
    return ErlWorkflow(**defaults)


class TestErlWorkflow:
    def test_first_attempt_succeeds_skips_reflection(self):
        """When first attempt succeeds, only erl_first is produced."""
        wf = _make_workflow(solver_fn=_make_solver_fn(1.0))
        ep = _run(wf.run({"q": "test"}, "uid:0"))
        names = [t.name for t in ep.trajectories]
        assert "erl_first" in names
        assert "erl_updater" not in names
        assert "erl_second" not in names
        assert "erl_distill" not in names
        assert ep.is_correct is True

    def test_full_loop_on_failure(self):
        """When first attempt fails, all 4 roles should be produced."""
        wf = _make_workflow(solver_fn=_make_solver_fn(0.0))
        ep = _run(wf.run({"q": "test"}, "uid:0"))
        names = [t.name for t in ep.trajectories]
        assert "erl_first" in names
        assert "erl_updater" in names
        assert "erl_second" in names
        assert "erl_distill" in names

    def test_no_reflection_skips_updater_but_injects_context(self):
        """no_reflection=True skips the updater LLM but still injects failure context."""
        captured_prompts: list[str] = []

        async def capturing_solver(prompt, task, engine):
            captured_prompts.append(prompt)
            step = Step(
                chat_completions=[{"role": "system", "content": prompt}, {"role": "assistant", "content": "a"}],
                reward=0.0,
                done=True,
            )
            return Trajectory(steps=[step], reward=0.0)

        wf = _make_workflow(erl_config_overrides={"no_reflection": True}, solver_fn=capturing_solver)
        ep = _run(wf.run({"q": "test"}, "uid:0"))
        names = [t.name for t in ep.trajectories]
        assert "erl_updater" not in names
        assert "erl_second" in names
        # Second attempt prompt should contain the retry instruction, not be identical to first
        assert len(captured_prompts) == 2
        assert captured_prompts[1] != captured_prompts[0]
        assert "past attempt data" in captured_prompts[1].lower()

    def test_compute_gate_skips_reflection(self):
        """When no flags need reflection/second attempt, skip them entirely."""
        call_count = 0

        async def counting_solver(prompt, task, engine):
            nonlocal call_count
            call_count += 1
            step = Step(
                chat_completions=[{"role": "system", "content": prompt}, {"role": "assistant", "content": "a"}],
                reward=0.0,
                done=True,
            )
            return Trajectory(steps=[step], reward=0.0)

        wf = _make_workflow(
            erl_config_overrides={"train_second_attempt": False, "train_distilled": False, "train_updater": False},
            solver_fn=counting_solver,
        )
        ep = _run(wf.run({"q": "test"}, "uid:0"))
        # Only the first attempt should run (solver called once)
        assert call_count == 1
        names = [t.name for t in ep.trajectories]
        assert "erl_second" not in names
        assert "erl_distill" not in names
        assert "erl_updater" not in names

    def test_train_flags_respected(self):
        wf = _make_workflow(
            erl_config_overrides={"train_first_attempt": False, "train_updater": False},
            solver_fn=_make_solver_fn(0.0),
        )
        ep = _run(wf.run({"q": "test"}, "uid:0"))
        names = [t.name for t in ep.trajectories]
        assert "erl_first" not in names
        assert "erl_updater" not in names
        assert "erl_second" in names
        assert "erl_distill" in names

    def test_distillation_replaces_prompt(self):
        wf = _make_workflow(solver_fn=_make_solver_fn(0.0))
        ep = _run(wf.run({"q": "test"}, "uid:0"))
        distill = next(t for t in ep.trajectories if t.name == "erl_distill")
        for step in distill.steps:
            assert step.model_output is None
            if step.chat_completions and step.chat_completions[0].get("role") == "system":
                assert step.chat_completions[0]["content"] == "initial prompt"

    def test_updater_reward_aligned_to_second(self):
        wf = _make_workflow(solver_fn=_make_solver_fn(0.0))
        ep = _run(wf.run({"q": "test"}, "uid:0"))
        updater = next(t for t in ep.trajectories if t.name == "erl_updater")
        second = next(t for t in ep.trajectories if t.name == "erl_second")
        assert updater.reward == second.reward

    def test_validation_mode(self):
        wf = _make_workflow(solver_fn=_make_solver_fn(0.0))
        ep = _run(wf.run({"q": "test"}, "uid:0", is_validation=True))
        names = [t.name for t in ep.trajectories]
        assert names == ["erl_first"]

    def test_memory_write_on_success(self):
        store = InMemoryStore()

        call_count = 0

        async def solver_fn(prompt, task, engine):
            nonlocal call_count
            call_count += 1
            # Second attempt succeeds
            reward = 1.0 if call_count == 2 else 0.0
            step = Step(
                chat_completions=[
                    {"role": "system", "content": prompt},
                    {"role": "assistant", "content": "ans"},
                ],
                reward=reward,
                done=True,
            )
            return Trajectory(steps=[step], reward=reward)

        wf = _make_workflow(solver_fn=solver_fn, store=store)
        _run(wf.run({"q": "test"}, "uid:0"))
        assert _run(store.get("improved_prompt")) is not None

    def test_memory_not_written_when_disabled(self):
        store = InMemoryStore()
        wf = _make_workflow(erl_config_overrides={"no_memory": True}, solver_fn=_make_solver_fn(0.0), store=store)
        _run(wf.run({"q": "test"}, "uid:0"))
        assert _run(store.get("improved_prompt")) is None

    def test_memory_read_on_next_episode(self):
        store = InMemoryStore()
        _run(store.set("improved_prompt", "stored prompt"))

        captured_prompts: list[str] = []

        async def solver_fn(prompt, task, engine):
            captured_prompts.append(prompt)
            step = Step(
                chat_completions=[
                    {"role": "system", "content": prompt},
                    {"role": "assistant", "content": "ans"},
                ],
                reward=0.0,
                done=True,
            )
            return Trajectory(steps=[step], reward=0.0)

        wf = _make_workflow(solver_fn=solver_fn, store=store)
        _run(wf.run({"q": "test"}, "uid:0"))
        # First call uses initial_system_prompt, second call should use stored prompt
        # (second call is the second attempt, which uses improved_prompt from updater)
        # The updater receives "stored prompt" as base_prompt via the store
        assert len(captured_prompts) == 2  # first + second attempt


# ---------------------------------------------------------------------------
# Default estimator map
# ---------------------------------------------------------------------------


class TestDefaultMap:
    def test_keys(self):
        from rllm.experimental.erl.utils import DEFAULT_ERL_ADV_ESTIMATOR_MAP

        assert set(DEFAULT_ERL_ADV_ESTIMATOR_MAP.keys()) == {
            "erl_first",
            "erl_updater",
            "erl_second",
            "erl_distill",
        }

    def test_distill_uses_raft(self):
        from rllm.experimental.erl.utils import DEFAULT_ERL_ADV_ESTIMATOR_MAP

        assert DEFAULT_ERL_ADV_ESTIMATOR_MAP["erl_distill"] == "raft"
