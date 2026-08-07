"""MCP-Atlas claim-coverage evaluator backed by the pinned official scorer."""

from __future__ import annotations

import asyncio
import json
import os
import random
from dataclasses import dataclass
from typing import Any

from rllm.eval.reward_fns._helpers import extract_answer_text
from rllm.eval.types import EvalOutput, Signal
from rllm.integrations.mcp_atlas.constants import DIRECT_GEMINI_JUDGE_MODEL, JUDGE_BASE_URL, JUDGE_MODEL
from rllm.integrations.mcp_atlas.source import load_official_scorer
from rllm.types import Episode, Task
from rllm.workflows.workflow import TerminationReason


@dataclass(frozen=True)
class JudgeSettings:
    model: str
    base_url: str | None
    api_key: str | None
    provider: str


def resolve_judge_settings(*, require_key: bool = True) -> JudgeSettings:
    """Resolve the OpenRouter default while retaining direct-Gemini fallback."""
    model_override = os.environ.get("MCP_ATLAS_JUDGE_MODEL")
    base_url_override = os.environ.get("MCP_ATLAS_JUDGE_BASE_URL")
    explicit_key = os.environ.get("MCP_ATLAS_JUDGE_API_KEY")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")

    # A direct Gemini key remains a zero-config compatibility fallback when no
    # MCP-Atlas/OpenRouter judge configuration was supplied.
    direct_fallback = not any((model_override, base_url_override, explicit_key, openrouter_key)) and bool(gemini_key)
    if direct_fallback:
        settings = JudgeSettings(
            model=DIRECT_GEMINI_JUDGE_MODEL,
            base_url=None,
            api_key=gemini_key,
            provider="gemini",
        )
    else:
        model = model_override or JUDGE_MODEL
        if base_url_override:
            base_url = base_url_override
        elif model.startswith("gemini/"):
            base_url = None
        else:
            base_url = JUDGE_BASE_URL
        provider = "gemini" if base_url is None and model.startswith("gemini/") else "openrouter"
        api_key = explicit_key or (gemini_key if provider == "gemini" else openrouter_key)
        settings = JudgeSettings(model=model, base_url=base_url, api_key=api_key, provider=provider)

    if require_key and not settings.api_key:
        raise RuntimeError("MCP-Atlas scoring requires OPENROUTER_API_KEY (default), MCP_ATLAS_JUDGE_API_KEY, or GEMINI_API_KEY for direct-Gemini fallback")
    return settings


class _OfficialPromptLiteLLMClient:
    """Implement the official scorer client interface using LiteLLM directly."""

    def __init__(self, *, model: str, api_key: str, base_url: str | None, attempts: int = 8) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.attempts = attempts

    async def generate_structured_content(self, prompt: str, response_schema: dict, temperature: float = 0.0) -> dict:
        import litellm

        last_error: Exception | None = None
        min_wait = float(os.environ.get("MCP_ATLAS_JUDGE_RETRY_MIN_SECONDS", "60"))
        max_wait = float(os.environ.get("MCP_ATLAS_JUDGE_RETRY_MAX_SECONDS", "120"))
        for attempt in range(self.attempts):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "api_key": self.api_key,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "claim_evaluation", "schema": response_schema},
                    },
                }
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                response = await litellm.acompletion(**kwargs)
                content = response.choices[0].message.content
                if isinstance(content, dict):
                    return content
                return json.loads(content or "")
            except Exception as exc:
                last_error = exc
                if attempt + 1 >= self.attempts:
                    break
                await asyncio.sleep(random.uniform(min_wait, max_wait))
        assert last_error is not None
        raise last_error


class MCPAtlasClaimsEvaluator:
    """Score one final answer with Scale's claim prompt and aggregation code."""

    def evaluate(self, task: Task, episode: Episode) -> EvalOutput:
        scorer = load_official_scorer()
        claims = scorer.extract_claims(task.metadata.get("GTFA_CLAIMS"))
        if not claims:
            raise RuntimeError(f"MCP-Atlas task {task.id} has no scoreable claims")
        answer = str(episode.artifacts.get("answer") or extract_answer_text(episode) or "")
        terminal_error = episode.termination_reason in {TerminationReason.ERROR, TerminationReason.TIMEOUT}
        response_truncated = False
        if terminal_error or not answer.strip() or answer.startswith("ERROR:"):
            reason = "Agent execution failed" if terminal_error else "Empty or error response"
            per_claim = [{"claim": claim, "score": 0.0, "covered": False, "reason": reason} for claim in claims]
            result = {
                "per_claim": per_claim,
                "coverage_score": 0.0,
                "total_claims": len(claims),
                "fully_covered_claims": 0,
                "partially_covered_claims": 0,
                "confidence": 1.0,
            }
        else:
            settings = resolve_judge_settings()
            model = settings.model
            api_key = settings.api_key
            assert api_key is not None
            base_url = settings.base_url
            config = scorer.EvaluatorConfig(model_name=model, api_key=api_key, base_url=base_url, verbose=False)
            client = _OfficialPromptLiteLLMClient(model=model, api_key=api_key, base_url=base_url)
            evaluator = scorer.CoverageEvaluator(client, config)
            judge_answer = answer
            response_truncated = len(judge_answer) > 500_000
            if response_truncated:
                judge_answer = judge_answer[:500_000] + "\n\n[TRUNCATED — original response was too long]"
            result = asyncio.run(evaluator.evaluate(claims, judge_answer))
            failed = [claim_result for claim_result in result.get("per_claim", []) if str(claim_result.get("reason", "")).startswith("Evaluation failed:")]
            if failed:
                raise RuntimeError(f"MCP-Atlas judge failed for {len(failed)} claim(s)")

        coverage = float(result.get("coverage_score") or 0.0)
        for claim_result in result.get("per_claim", []):
            score = float(claim_result.get("score") or 0.0)
            claim_result["label"] = "correct" if score >= 1.0 else "partially_correct" if score >= 0.5 else "incorrect"
            claim_result.setdefault("justification", claim_result.get("reason", ""))
        is_correct = coverage >= 0.75
        settings = resolve_judge_settings(require_key=False)
        judge_model = settings.model
        episode.artifacts.setdefault("mcp_atlas", {})["claim_evaluation"] = result
        return EvalOutput(
            reward=1.0 if is_correct else 0.0,
            is_correct=is_correct,
            signals=[
                Signal(name="coverage", value=coverage),
                Signal(name="fully_covered_claims", value=float(result.get("fully_covered_claims", 0))),
                Signal(name="partially_covered_claims", value=float(result.get("partially_covered_claims", 0))),
            ],
            metadata={
                "judge_model": judge_model,
                "judge_provider": settings.provider,
                "judge_base_url": settings.base_url,
                "judge_temperature": 0.0,
                "judge_response_truncated": response_truncated,
                "claim_evaluation": result,
            },
        )


__all__ = ["JudgeSettings", "MCPAtlasClaimsEvaluator", "resolve_judge_settings"]
