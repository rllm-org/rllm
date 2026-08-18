"""Offline tests for the SWE-bench Pro GLM-5.2 OpenHands profile."""

from __future__ import annotations

import pytest

from rllm.eval.agent_loader import load_agent
from rllm.harnesses.openhands import OpenHandsHarness
from rllm.harnesses.swebench_pro_openhands_glm52 import (
    CONTEXT_WINDOW_TOKENS,
    FIREWORKS_MODEL_ID,
    LLM_STATS_TARGET_SCORE,
    MAX_ITERATIONS,
    MAX_OUTPUT_TOKENS,
    OPENHANDS_BENCHMARKS_REVISION,
    OPENROUTER_MODEL_ID,
    REPRODUCTION_PROFILE,
    SwebenchProOpenHandsGLM52Harness,
    _render_prompt,
)
from rllm.types import AgentConfig, Task


def _task() -> Task:
    return Task(
        id="task-1",
        instruction="Fix the issue.\n\nRequirements:\nKeep the API stable.",
        metadata={
            "workdir": "/app",
            "agent_timeout": 3600.0,
            "metadata": {"base_commit": "a" * 40},
        },
    )


def _config(model: str = "openai/glm-5.2") -> AgentConfig:
    return AgentConfig(
        base_url="http://gateway/sessions/test/v1",
        model=model,
        session_uid="test",
        metadata={"gateway_auth_token": "gateway-token"},
    )


def test_profile_is_a_thin_openhands_specialization():
    harness = SwebenchProOpenHandsGLM52Harness()

    assert isinstance(harness, OpenHandsHarness)
    assert harness.temperature == 1.0
    assert harness.top_p == 1.0
    assert harness.max_input_tokens == CONTEXT_WINDOW_TOKENS == 400_000
    assert harness.max_output_tokens == MAX_OUTPUT_TOKENS == 32_768
    assert harness.max_iterations == MAX_ITERATIONS == 500


def test_profile_records_public_score_and_disclosure_gap():
    assert LLM_STATS_TARGET_SCORE == 0.621
    assert len(OPENHANDS_BENCHMARKS_REVISION) == 40
    assert REPRODUCTION_PROFILE["status"] == "best_effort"
    assert REPRODUCTION_PROFILE["score_status"] == "self_reported_unverified"
    assert "undisclosed" in REPRODUCTION_PROFILE["prompt_provenance"]


@pytest.mark.parametrize(
    "model",
    ["openai/glm-5.2", OPENROUTER_MODEL_ID, FIREWORKS_MODEL_ID],
)
def test_profile_accepts_glm52_routes_and_preserves_gateway_alias(model):
    env = SwebenchProOpenHandsGLM52Harness().build_env(_task(), _config(model))

    assert env["LLM_MODEL"] == f"openai/{model}"
    assert env["LLM_API_KEY"] == "gateway-token"
    assert env["LLM_BASE_URL"] == "http://gateway/sessions/test/v1"
    assert env["RLLM_OPENHANDS_MAX_INPUT_TOKENS"] == "400000"
    assert env["RLLM_OPENHANDS_MAX_OUTPUT_TOKENS"] == "32768"


def test_profile_rejects_non_glm52_models():
    with pytest.raises(ValueError, match="requires a GLM-5.2 model"):
        SwebenchProOpenHandsGLM52Harness().build_env(_task(), _config("openai/gpt-5.4"))


def test_public_prompt_wraps_task_and_base_commit():
    prompt = _render_prompt("Fix it", "/app", "b" * 40)

    assert "code repository in the directory /app" in prompt
    assert "<issue_description>\nFix it\n</issue_description>" in prompt
    assert "b" * 40 in prompt
    assert "minimal changes to non-test files" in prompt


def test_render_prompt_reads_builder_metadata_shape():
    prompt = SwebenchProOpenHandsGLM52Harness().render_prompt(_task(), "/app")

    assert "Keep the API stable." in prompt
    assert "a" * 40 in prompt


def test_registry_loads_profile_and_episode_records_provenance():
    harness = load_agent("swebench-pro-openhands-glm52")
    episode = harness._outcome_episode(_task())

    assert isinstance(harness, SwebenchProOpenHandsGLM52Harness)
    assert episode.metadata["reproduction_profile"] == REPRODUCTION_PROFILE
