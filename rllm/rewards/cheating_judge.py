"""Host-side, post-verifier attempted-cheating check for terminal episodes.

One judgment per verifier-correct episode. This client deliberately bypasses
the rollout gateway: judge completions must never enter the policy trajectory.
The caller must compact-filter GRADING_ERROR episodes before training.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass

import httpx

from rllm.rewards.cheating_prompt import FormatError, format_user_message
from rllm.types import INFRA_ERROR_REASONS, Episode, Task, TerminationReason, TrajectoryDelta, _index_step_deltas

logger = logging.getLogger(__name__)


def episode_to_judge_input(task: Task, episode: Episode) -> dict:
    """Judge the longest recorded chat path, measured in assistant turns.

    Branches can live inside a single trajectory or across multiple trajectories.
    Ties choose the later stored candidate. Count turns in the actual messages,
    not graph depth: a compact root can contain an entire prior conversation
    after a token-prefix reset. Only the winning compact chat is materialized;
    token arrays and all training steps remain untouched.
    """
    best = None
    best_turns = 0
    for trajectory in episode.trajectories:
        if isinstance(trajectory, TrajectoryDelta):
            by_id, _ = _index_step_deltas(trajectory.steps)
            turns_by_id = {}
            for step in trajectory.steps:
                turns = sum(message.get("role") == "assistant" for message in step.chat_completions_suffix)
                if step.parent_step_id is not None:
                    turns += turns_by_id[step.parent_step_id]
                turns_by_id[step.id] = turns
                if turns > 0 and turns >= best_turns:
                    best, best_turns = (step, by_id), turns
        else:
            for step in trajectory.steps:
                turns = sum(message.get("role") == "assistant" for message in step.chat_completions)
                if turns > 0 and turns >= best_turns:
                    best, best_turns = (step, None), turns
    if best is None:
        raise FormatError("Correct episode has no assistant evidence")
    step, by_id = best
    if by_id is not None:
        chain = [step]
        while chain[-1].parent_step_id is not None:
            chain.append(by_id[chain[-1].parent_step_id])
        messages = [message for step in reversed(chain) for message in step.chat_completions_suffix]
    else:
        messages = step.chat_completions
    return {"instruction": task.instruction, "messages": messages}


def parse_verdict(response: dict) -> str:
    """Last exact verdict in content, never reasoning_content."""
    choices = response.get("choices") if isinstance(response, dict) else None
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("Expected exactly one judge choice")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("tool_calls") or message.get("function_call"):
        raise ValueError("Invalid judge message")
    if choice.get("finish_reason") != "stop":
        raise ValueError("Judge response did not finish normally")
    content = message.get("content")
    matches = re.findall(r"### (PASS|FAIL)\b", content) if isinstance(content, str) else []
    if not matches:
        raise ValueError("Judge content contains no verdict")
    return matches[-1]


@dataclass(frozen=True)
class CheatingJudgeConfig:
    model: str
    url: str = "https://api.fireworks.ai/inference/v1/chat/completions"
    api_key_env: str = "FIREWORKS_API_KEY"
    reasoning_effort: str = "high"
    timeout_s: float = 600.0
    max_attempts: int = 3
    retry_delay_s: float = 2.0
    apply_to_validation: bool = False

    def __post_init__(self):
        if not self.model or self.max_attempts < 1 or self.timeout_s <= 0 or self.retry_delay_s < 0:
            raise ValueError("Invalid cheating judge configuration")
        if self.reasoning_effort not in ("low", "high", "max"):
            raise ValueError("Unsupported judge reasoning effort")


class CheatingJudge:
    def __init__(self, config: CheatingJudgeConfig, *, transport=None):
        self.config = config
        self._api_key = os.environ.get(config.api_key_env, "").strip()
        if not self._api_key:
            raise ValueError(f"Cheating judge requires {config.api_key_env}")
        self._transport = transport

    @staticmethod
    def _zero_reward(episode: Episode):
        episode.is_correct = False
        for trajectory in episode.trajectories:
            trajectory.reward = 0.0
            for step in trajectory.steps:
                step.reward = 0.0
            if "accuracy" in trajectory.signals:
                trajectory.signals["accuracy"] = 0.0
        if "accuracy" in episode.metrics:
            episode.metrics["accuracy"] = 0.0

    async def apply(self, task: Task, episode: Episode, *, is_validation: bool = False) -> None:
        if is_validation and not self.config.apply_to_validation:
            return
        if "cheating_judge" in episode.metadata:
            raise ValueError("Episode has already been judged")
        original_rewards = [trajectory.reward for trajectory in episode.trajectories]
        eligible = episode.termination_reason not in INFRA_ERROR_REASONS and (episode.is_correct or any(reward is not None and reward > 0 for reward in original_rewards))
        episode.metrics.update(
            verifier_correct_frac=float(episode.is_correct),
            judge_eligible_frac=float(eligible),
            reward_removed_frac=0.0,
        )
        if not eligible:
            return
        info = {
            "model": self.config.model,
            "reasoning_effort": self.config.reasoning_effort,
            "original_rewards": original_rewards,
            "original_is_correct": episode.is_correct,
            "original_termination_reason": episode.termination_reason.value if episode.termination_reason else None,
            "original_accuracy": episode.metrics.get("accuracy"),
            "attempts": [],
        }
        episode.metadata["cheating_judge"] = info
        started = time.perf_counter()
        episode.metrics.update(judge_error_frac=0.0, judge_coverage=0.0, judge_requests=0.0, judge_prompt_tokens=0.0, judge_completion_tokens=0.0, judge_reasoning_tokens=0.0)
        try:
            if not original_rewards or any(reward != 1.0 for reward in original_rewards) or not episode.is_correct:
                raise FormatError("Expected verifier-correct trajectories with shared reward 1.0")
            judge_input = episode_to_judge_input(task, episode)
            info["path_selection"] = "most_assistant_turns_latest_on_tie"
            info["selected_num_turns"] = sum(message.get("role") == "assistant" for message in judge_input["messages"])
            user_message = format_user_message(judge_input)
            info["prompt_sha256"] = hashlib.sha256(user_message["content"].encode()).hexdigest()
            body = {"model": self.config.model, "messages": [user_message], "reasoning_effort": self.config.reasoning_effort}
            async with httpx.AsyncClient(
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=httpx.Timeout(self.config.timeout_s, connect=min(30, self.config.timeout_s)),
                follow_redirects=False,
                transport=self._transport,
            ) as client:
                for attempt in range(self.config.max_attempts):
                    details = {"number": attempt + 1}
                    info["attempts"].append(details)
                    episode.metrics["judge_requests"] += 1.0
                    try:
                        response = await client.post(self.config.url, json=body)
                        details["http_status"] = response.status_code
                        if response.status_code != 200:
                            if response.status_code != 429 and response.status_code < 500:
                                raise RuntimeError(f"Judge HTTP {response.status_code}")
                            raise ValueError(f"Judge HTTP {response.status_code}")
                        data = response.json()
                        # Persist the returned evidence, not another copy of
                        # the entire prompt. Never persist auth headers.
                        info["response"] = response.text.replace(self._api_key, "[REDACTED]")
                        usage = data.get("usage") or {} if isinstance(data, dict) else {}
                        for field in ("prompt_tokens", "completion_tokens"):
                            episode.metrics[f"judge_{field}"] += float(usage.get(field) or 0)
                        token_details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
                        episode.metrics["judge_reasoning_tokens"] += float(token_details.get("reasoning_tokens") or 0)
                        verdict = parse_verdict(data)
                        break
                    except (httpx.HTTPError, ValueError) as exc:
                        details["error"] = f"{type(exc).__name__}: {exc}".replace(self._api_key, "[REDACTED]")
                        if attempt + 1 == self.config.max_attempts:
                            raise
                        await asyncio.sleep(min(30.0, self.config.retry_delay_s * 2**attempt))
            info.update(status="ok", verdict=verdict)
            episode.metrics["judge_coverage"] = 1.0
            # Conditional metric: only successfully judged episodes contribute.
            episode.metrics["cheat_frac"] = float(verdict == "FAIL")
            if verdict == "FAIL":
                self._zero_reward(episode)
                episode.metrics["reward_removed_frac"] = 1.0
        except Exception as exc:
            # Return the existing episode, not an exception that would rerun
            # the expensive solver. This isn't a policy failure/negative label.
            error = f"{type(exc).__name__}: {exc}".replace(self._api_key, "[REDACTED]")
            info.update(status="error", error=error)
            self._zero_reward(episode)
            episode.termination_reason = TerminationReason.GRADING_ERROR
            episode.metadata["error"] = {"error_type": "CheatingJudgeError", "message": error}
            episode.metrics["judge_error_frac"] = 1.0
            logger.warning("Cheating judge failed for task %s: %s", task.id, error)
        finally:
            episode.metrics["judge_latency_s"] = time.perf_counter() - started


def build_cheating_judge(config, compact_filtering) -> CheatingJudge | None:
    """Wire only when enabled, and never let unjudged positives enter the loss."""
    if not config or not config.get("enabled", False):
        return None
    if not compact_filtering.should_mask(TerminationReason.GRADING_ERROR):
        raise ValueError("Cheating judge requires compact filtering of grading_error")
    return CheatingJudge(CheatingJudgeConfig(**{key: value for key, value in config.items() if key != "enabled"}))
