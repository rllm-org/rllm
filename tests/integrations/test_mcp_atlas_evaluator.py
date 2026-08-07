from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from rllm.integrations.mcp_atlas.constants import DIRECT_GEMINI_JUDGE_MODEL, JUDGE_BASE_URL, JUDGE_MODEL
from rllm.integrations.mcp_atlas.evaluator import MCPAtlasClaimsEvaluator, _OfficialPromptLiteLLMClient, resolve_judge_settings
from rllm.types import Episode, Task, Trajectory
from rllm.workflows.workflow import TerminationReason


def _task():
    return Task(
        id="atlas-1",
        instruction="prompt",
        metadata={"GTFA_CLAIMS": ["First claim", "Second claim", "Third claim", "Fourth claim"]},
        dataset_dir=Path("."),
    )


def _episode(answer="answer"):
    return Episode(
        id="episode-1",
        task=_task(),
        trajectories=[Trajectory(uid="t", output=answer)],
        artifacts={"answer": answer},
    )


def _fake_scorer(result):
    @dataclass
    class Config:
        model_name: str
        api_key: str
        base_url: str | None
        verbose: bool

    class Coverage:
        def __init__(self, client, config):
            self.client = client
            self.config = config

        async def evaluate(self, claims, answer):
            assert claims
            assert answer == "answer"
            return result

    return SimpleNamespace(
        extract_claims=lambda value: list(value),
        EvaluatorConfig=Config,
        CoverageEvaluator=Coverage,
    )


def test_claim_scores_and_exact_pass_boundary(monkeypatch):
    result = {
        "per_claim": [
            {"claim": "First claim", "score": 1.0, "covered": True, "reason": "yes"},
            {"claim": "Second claim", "score": 1.0, "covered": True, "reason": "yes"},
            {"claim": "Third claim", "score": 0.5, "covered": "partial", "reason": "some"},
            {"claim": "Fourth claim", "score": 0.5, "covered": "partial", "reason": "some"},
        ],
        "coverage_score": 0.75,
        "total_claims": 4,
        "fully_covered_claims": 2,
        "partially_covered_claims": 2,
        "confidence": 0.9,
    }
    monkeypatch.setattr("rllm.integrations.mcp_atlas.evaluator.load_official_scorer", lambda: _fake_scorer(result))
    monkeypatch.setenv("MCP_ATLAS_JUDGE_API_KEY", "judge-key")

    output = MCPAtlasClaimsEvaluator().evaluate(_task(), _episode())

    assert output.reward == 1.0
    assert output.is_correct is True
    assert {signal.name: signal.value for signal in output.signals}["coverage"] == 0.75
    labels = [item["label"] for item in result["per_claim"]]
    assert labels == ["correct", "correct", "partially_correct", "partially_correct"]
    assert result["per_claim"][0]["justification"] == "yes"
    assert output.metadata["judge_model"] == JUDGE_MODEL
    assert output.metadata["judge_provider"] == "openrouter"
    assert output.metadata["judge_base_url"] == JUDGE_BASE_URL


def test_openrouter_is_default_and_gemini_key_falls_back_to_direct(monkeypatch):
    for name in (
        "MCP_ATLAS_JUDGE_MODEL",
        "MCP_ATLAS_JUDGE_BASE_URL",
        "MCP_ATLAS_JUDGE_API_KEY",
        "OPENROUTER_API_KEY",
        "GEMINI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    openrouter = resolve_judge_settings()
    assert openrouter.model == JUDGE_MODEL
    assert openrouter.base_url == JUDGE_BASE_URL
    assert openrouter.api_key == "openrouter-key"
    assert openrouter.provider == "openrouter"

    monkeypatch.delenv("OPENROUTER_API_KEY")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    direct = resolve_judge_settings()
    assert direct.model == DIRECT_GEMINI_JUDGE_MODEL
    assert direct.base_url is None
    assert direct.api_key == "gemini-key"
    assert direct.provider == "gemini"


def test_coverage_below_boundary_fails(monkeypatch):
    result = {
        "per_claim": [{"claim": "First claim", "score": 0.0, "covered": False, "reason": "no"}],
        "coverage_score": 0.749,
        "fully_covered_claims": 0,
        "partially_covered_claims": 0,
    }
    scorer = _fake_scorer(result)
    scorer.extract_claims = lambda value: ["First claim"]
    monkeypatch.setattr("rllm.integrations.mcp_atlas.evaluator.load_official_scorer", lambda: scorer)
    monkeypatch.setenv("GEMINI_API_KEY", "judge-key")

    output = MCPAtlasClaimsEvaluator().evaluate(
        Task(id="one", instruction="", metadata={"GTFA_CLAIMS": ["First claim"]}, dataset_dir=Path(".")),
        _episode(),
    )

    assert output.reward == 0.0
    assert output.is_correct is False


def test_empty_and_agent_error_responses_do_not_call_judge(monkeypatch):
    scorer = SimpleNamespace(extract_claims=lambda value: list(value))
    monkeypatch.setattr("rllm.integrations.mcp_atlas.evaluator.load_official_scorer", lambda: scorer)
    monkeypatch.delenv("MCP_ATLAS_JUDGE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    empty = MCPAtlasClaimsEvaluator().evaluate(_task(), _episode(""))
    failed_episode = _episode("partial answer")
    failed_episode.termination_reason = TerminationReason.ERROR
    failed = MCPAtlasClaimsEvaluator().evaluate(_task(), failed_episode)

    assert empty.reward == 0.0
    assert failed.reward == 0.0
    assert failed_episode.artifacts["mcp_atlas"]["claim_evaluation"]["per_claim"][0]["reason"] == "Agent execution failed"


def test_exhausted_official_claim_failure_is_retryable_eval_error(monkeypatch):
    result = {
        "per_claim": [{"claim": "First claim", "score": 0.0, "covered": False, "reason": "Evaluation failed: rate limit"}],
        "coverage_score": 0.0,
    }
    scorer = _fake_scorer(result)
    scorer.extract_claims = lambda value: ["First claim"]
    monkeypatch.setattr("rllm.integrations.mcp_atlas.evaluator.load_official_scorer", lambda: scorer)
    monkeypatch.setenv("GEMINI_API_KEY", "judge-key")

    with pytest.raises(RuntimeError, match="judge failed"):
        MCPAtlasClaimsEvaluator().evaluate(
            Task(id="one", instruction="", metadata={"GTFA_CLAIMS": ["First claim"]}, dataset_dir=Path(".")),
            _episode(),
        )


def test_judge_client_retries_and_uses_exact_structured_request(monkeypatch):
    calls = []

    async def acompletion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("transient")
        message = SimpleNamespace(content=json.dumps({"coverage_outcome": "fulfilled"}))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    async def no_sleep(_seconds):
        return None

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(acompletion=acompletion))
    monkeypatch.setattr("rllm.integrations.mcp_atlas.evaluator.asyncio.sleep", no_sleep)
    client = _OfficialPromptLiteLLMClient(model="gemini/gemini-2.5-pro", api_key="key", base_url=None, attempts=2)
    schema = {"type": "object", "properties": {"coverage_outcome": {"type": "string"}}}

    value = asyncio.run(client.generate_structured_content("official prompt", schema, temperature=0.0))

    assert value == {"coverage_outcome": "fulfilled"}
    assert len(calls) == 2
    assert calls[-1]["model"] == "gemini/gemini-2.5-pro"
    assert calls[-1]["temperature"] == 0.0
    assert calls[-1]["response_format"]["json_schema"]["schema"] == schema
