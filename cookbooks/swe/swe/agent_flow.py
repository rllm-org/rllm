#!/usr/bin/env python3
"""SWE AgentFlow — runs mini-swe-agent on a task, returns an Episode.

Creates a Modal sandbox, runs mini-swe-agent with an OpenAI-compatible
model, and returns an Episode with the agent's trajectory and patch.

Works identically for eval and training — the only difference is what
``config.base_url`` points at:

* **Eval**: any OpenAI-compatible endpoint
* **Training**: the rllm model gateway (captures traces transparently)
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import time
from typing import Any, Callable
from uuid import uuid4

from rllm.types import AgentConfig, Episode, Task, Trajectory

from swe.environment import (
    create_env,
    ensure_bootstrapped,
    load_runtime_configs,
)
from swe.tasks.common import make_log, short_id

ensure_bootstrapped()

from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import LimitsExceeded

from swe.flow_config import SWEAgentFlowConfig
from swe.openai_model import (
    MaxPromptLengthExceeded,
    MaxResponseLengthExceeded,
    OpenAIClientModel,
)
from swe.tasks.swesmith import setup_swesmith_agent_env
from swe.utils import (
    build_error_details,
    classify_termination,
    close_env,
    tool_response_user_message,
)


def _agent_exit_message(exit_status: str, submission: str = "") -> dict:
    return {
        "role": "exit",
        "content": exit_status,
        "extra": {"exit_status": exit_status, "submission": submission},
    }


def _raise_agent_exit(exit_status: str, submission: str = "") -> None:
    raise LimitsExceeded(_agent_exit_message(exit_status, submission=submission))


# -------------------------------------------------------------------
# Agent wrapper: adds per-step logging, timeout, and FormatError handling
# -------------------------------------------------------------------

class ProgressLoggingAgent(DefaultAgent):
    """DefaultAgent with per-step logging, wall-clock timeout, and context compaction."""

    def __init__(
        self,
        *args,
        instance_id: str = "",
        agent_timeout: float = 0,
        log_fn: Callable[[str], None] = print,
        # Compaction config (passed through from YAML via agent_config dict)
        compaction_enabled: bool = False,
        compaction_token_trigger: int = 28000,
        compaction_keep_recent_turns: int = 1,
        compaction_summary_prompt: str = "",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.instance_id = instance_id
        self._step_count = 0
        self._agent_timeout = agent_timeout
        self._start_time: float | None = None
        self._log_fn = log_fn
        # Compaction state
        self._compaction_enabled = compaction_enabled
        self._compaction_token_trigger = compaction_token_trigger
        self._compaction_keep_recent_turns = compaction_keep_recent_turns
        self._compaction_summary_prompt = compaction_summary_prompt
        self._last_prompt_tokens: int = 0
        self._compaction_count: int = 0
        self._segments: list[dict] = []
        self._current_segment_messages: list[dict] = []

    def _get_prompt_tokens(self, message: dict) -> int:
        """Extract prompt_tokens from the last model response usage."""
        usage = (
            message.get("extra", {})
            .get("response", {})
            .get("usage", {})
        )
        return usage.get("prompt_tokens", 0)

    def _should_compact(self) -> bool:
        if not self._compaction_enabled:
            return False
        return self._last_prompt_tokens >= self._compaction_token_trigger

    @staticmethod
    def _find_turn_boundary(messages: list[dict], n_turns: int) -> int:
        """Find the message index where the last N complete turns start.

        A "turn" is an assistant message plus any tool/user response messages
        that follow it (before the next assistant message). We scan backward
        to find clean boundaries so we never split a tool-call / tool-response pair.
        """
        # Find indices of all assistant messages
        assistant_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
        if not assistant_indices or n_turns <= 0:
            return len(messages)
        # Take the last n_turns assistant messages; the boundary is the first of those
        start = max(0, len(assistant_indices) - n_turns)
        return assistant_indices[start]

    def _close_current_segment(self, kind: str = "solver") -> None:
        msgs = self._current_segment_messages or self.messages
        if msgs:
            self._segments.append({
                "kind": kind,
                "messages": list(msgs),
            })
            self._current_segment_messages = []

    def _compact_messages(self) -> None:
        """Summarize old messages and replace history, keeping recent turns verbatim.

        Message layout before compaction:
            [system, task, a1, t1, ..., a8, t8, a9, t9, a10, t10]
                          |______old_turns___|  |____recent____|

        Summarizer sees: [system, task, old_turns] + summary_prompt
        After compaction: [system, task, compact_summary, recent_turns]

        On repeated compactions the previous summary is part of old_turns,
        so the new summary is a summary-of-summary + new work.
        """
        prefix = self.messages[:2]  # system + task

        conversation = self.messages[2:]
        boundary = self._find_turn_boundary(conversation, self._compaction_keep_recent_turns)
        old_turns = conversation[:boundary]
        recent_turns = conversation[boundary:]

        if not old_turns:
            return

        self._close_current_segment("solver")

        summarizer_input = prefix + old_turns
        summary_msg = self.model.summarize_context(
            summarizer_input,
            summary_prompt=self._compaction_summary_prompt,
        )

        self._segments.append({
            "kind": "summarizer",
            "messages": [
                *[{k: v for k, v in m.items() if k != "extra"} for m in summarizer_input],
                tool_response_user_message(self._compaction_summary_prompt),
                summary_msg,
            ],
        })

        self.messages = prefix + [summary_msg] + recent_turns

        self._compaction_count += 1
        self._log_fn(
            f"[{self.instance_id}] Compaction #{self._compaction_count}: "
            f"{self._last_prompt_tokens} prompt tokens -> compacted "
            f"({len(old_turns)} messages summarized, {len(recent_turns)} kept)"
        )


    def get_segments(self) -> list[dict]:
        self._close_current_segment("solver")
        return list(self._segments)

    def query(self) -> dict:
        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            _raise_agent_exit("MaxTurnsExceeded")

        if self._should_compact():
            self._compact_messages()

        self.n_calls += 1
        message = self.model.query(self.messages)
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self._last_prompt_tokens = self._get_prompt_tokens(message)
        self.add_messages(message)
        self._current_segment_messages = list(self.messages)

        return message

    def step(self) -> list[dict[str, Any]]:
        if self._start_time is None:
            self._start_time = time.monotonic()

        if self._agent_timeout > 0:
            elapsed = time.monotonic() - self._start_time
            if elapsed >= self._agent_timeout:
                self._log_fn(
                    f"[{self.instance_id}] Agent timeout after {elapsed:.1f}s "
                    f"(limit: {self._agent_timeout}s)"
                )
                _raise_agent_exit("Timeout")

        self._step_count += 1
        self._log_fn(f"[{self.instance_id}] Step {self._step_count:3d} - querying model...")
        try:
            result_messages = super().step()
        except MaxPromptLengthExceeded:
            _raise_agent_exit("MaxPromptLengthExceeded")
        except MaxResponseLengthExceeded:
            _raise_agent_exit("MaxResponseLengthExceeded")

        preview = ""
        if result_messages:
            preview = str(result_messages[-1].get("content", ""))
        self._log_fn(
            f"[{self.instance_id}] Step {self._step_count:3d} - got response, "
            f"output preview: {preview[:100]}..."
        )
        return result_messages

    def execute_actions(self, message: dict) -> list[dict]:
        """Handle format errors: return feedback instead of executing."""
        format_error = message.get("extra", {}).get("format_error")
        if format_error is not None:
            self._log_fn(f"[{self.instance_id}] Format error, sending feedback")
            return self.add_messages(tool_response_user_message(format_error))
        return super().execute_actions(message)


# -------------------------------------------------------------------
# AgentFlow
# -------------------------------------------------------------------

class SWEAgentFlow:
    """AgentFlow that runs mini-swe-agent to generate a patch.

    Constructor args configure the agent limits and sandbox settings.
    The ``run()`` method receives the task and inference config from
    the runner/engine.
    """

    def __init__(self, config: SWEAgentFlowConfig | None = None):
        self.config = config if config is not None else SWEAgentFlowConfig()

    @property
    def verbose(self) -> bool:
        return self.config.verbose

    def _get_params(self, is_validation: bool) -> dict:
        """Return agent parameters, applying validation overrides if applicable."""
        c = self.config
        v = c.validation

        def _pick(override, base):
            return override if is_validation and override is not None else base

        return {
            "step_limit": _pick(v.step_limit, c.step_limit),
            "agent_timeout": _pick(v.agent_timeout, c.agent_timeout),
            "command_timeout": _pick(v.command_timeout, c.command_timeout),
            "sandbox_timeout": _pick(v.sandbox_timeout, c.sandbox_timeout),
            "startup_jitter_s": _pick(v.startup_jitter_s, c.startup_jitter_s),
        }

    @staticmethod
    def _task_data(task: Task | dict) -> dict:
        """Return the raw dataset row for both old and current rllm Task shapes."""
        if isinstance(task, dict):
            return task
        data = getattr(task, "data", None)
        if isinstance(data, dict):
            return data
        metadata = getattr(task, "metadata", None)
        if isinstance(metadata, dict):
            return metadata
        raise TypeError(f"Unsupported task type: {type(task).__name__}")

    @staticmethod
    def _safe_filename(value: str) -> str:
        return "".join(c if c.isalnum() or c in "._-" else "_" for c in value) or "task"

    @staticmethod
    def _clean_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep trajectory JSON useful without dumping full API responses."""
        cleaned = []
        for msg in messages:
            item: dict[str, Any] = {
                "role": msg.get("role", "unknown"),
                "content": msg.get("content", ""),
            }
            extra = msg.get("extra") or {}
            if extra:
                clean_extra = {
                    key: extra[key]
                    for key in (
                        "actions",
                        "format_error",
                        "exit_status",
                        "submission",
                        "interrupt_type",
                        "model_response",
                        "returncode",
                        "raw_output",
                        "summary",
                        "raw_transcript",
                    )
                    if key in extra
                }
                if clean_extra:
                    item["extra"] = clean_extra
            cleaned.append(item)
        return cleaned

    @classmethod
    def _clean_segments(cls, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "kind": segment.get("kind", "solver"),
                "messages": cls._clean_messages(segment.get("messages", [])),
            }
            for segment in segments
        ]

    def _save_trajectory(self, episode: Episode) -> None:
        if not self.config.save_trajectories:
            return

        output_dir = Path(self.config.trajectory_output_dir or "training_trajs")
        output_dir.mkdir(parents=True, exist_ok=True)

        task = episode.task if isinstance(episode.task, dict) else {}
        instance_id = str(task.get("instance_id") or episode.id)
        filename = f"{self._safe_filename(instance_id)}__{self._safe_filename(episode.id)}.json"
        path = output_dir / filename
        tmp_path = output_dir / f".{filename}.{uuid4().hex}.tmp"

        artifacts = episode.artifacts
        data = {
            "episode_id": episode.id,
            "instance_id": instance_id,
            "exit_status": artifacts.get("exit_status"),
            "patch": artifacts.get("patch", ""),
            "messages": self._clean_messages(artifacts.get("messages", [])),
            "segments": self._clean_segments(artifacts.get("segments", [])),
            "termination_reason": (
                episode.termination_reason.value
                if getattr(episode.termination_reason, "value", None) is not None
                else episode.termination_reason
            ),
            "metrics": episode.metrics,
            "metadata": episode.metadata,
        }

        with tmp_path.open("w") as f:
            json.dump(data, f, indent=2, default=str)
        tmp_path.replace(path)

    def run(self, task: Task, config: AgentConfig) -> Episode:
        """Run mini-swe-agent on a single task.

        Returns an Episode with:
        - One trajectory ("solver") with a Step per assistant message
        - artifacts: patch, exit_status, env (if kept alive for grading)
        """

        data = self._task_data(task)
        is_validation = getattr(config, "is_validation", False)
        params = self._get_params(is_validation)

        # Startup jitter to avoid Modal sandbox creation thundering-herd at step start.
        jitter_max = params["startup_jitter_s"]
        if jitter_max > 0:
            time.sleep(random.uniform(0, jitter_max))
        log = make_log(self.verbose)
        sid = short_id(data["instance_id"])
        log(f"\n[{sid}] Starting patch generation{'  [val]' if is_validation else ''}...")

        env = None
        agent = None
        keep_env_alive = data["eval_type"] in {"swebench", "swe_rebench_v2"}

        try:
            # --- Create sandbox (retry on transient Modal failures) ---
            env_start = time.monotonic()
            retry = self.config.sandbox_retry
            max_attempts = max(1, retry.attempts)
            for attempt in range(max_attempts):
                try:
                    log(f"[{sid}] Creating sandbox environment...")
                    env = create_env(data, command_timeout=params["command_timeout"], sandbox_timeout=params["sandbox_timeout"])
                    break
                except Exception as e:
                    if attempt < max_attempts - 1:
                        delay = random.uniform(retry.backoff_min_s, retry.backoff_max_s)
                        log(f"[{sid}] Sandbox creation failed (attempt {attempt+1}/{max_attempts}), retrying in {delay:.0f}s: {type(e).__name__}")
                        time.sleep(delay)
                        continue
                    raise
            log(f"[{sid}] Sandbox ready in {time.monotonic() - env_start:.1f}s")

            if data["eval_type"] == "swesmith":
                setup_start = time.monotonic()
                log(f"[{sid}] Preparing SWE-smith sandbox...")
                setup_swesmith_agent_env(env, data, sid, log)
                log(f"[{sid}] SWE-smith sandbox ready in {time.monotonic() - setup_start:.1f}s")

            # --- Create model ---
            agent_config, model_config, _ = load_runtime_configs()
            model_overrides = self.config.model.as_model_config_overrides()
            if "model_kwargs" in model_overrides:
                merged_kwargs = dict(model_config.get("model_kwargs") or {})
                merged_kwargs.update(model_overrides.pop("model_kwargs"))
                model_config["model_kwargs"] = merged_kwargs
            model_config.update(model_overrides)
            model = OpenAIClientModel(
                base_url=config.base_url,
                model_name=config.model,
                verbose=self.verbose,
                model_config=model_config,
            )

            # --- Run agent ---
            agent_config["cost_limit"] = self.config.cost_limit
            agent_config["step_limit"] = params["step_limit"]
            agent_config.update(self.config.compaction.as_agent_config_overrides())

            agent = ProgressLoggingAgent(
                model=model,
                env=env,
                instance_id=sid,
                agent_timeout=params["agent_timeout"],
                log_fn=log,
                **agent_config,
            )

            problem_stmt = data["problem_statement"]
            log(f"[{sid}] Starting agent run...")
            result = agent.run(problem_stmt, problem_statement=problem_stmt)
            exit_status = result["exit_status"]
            patch = result["submission"]
            messages = agent.messages
            segments = agent.get_segments()
            num_summaries = agent._compaction_count
            log(f"[{sid}] Patch generation complete: {exit_status}, {len(patch)} chars")

        except Exception as e:
            log(f"[{sid}] Agent error: {type(e).__name__}: {e}")
            exit_status = f"Error: {type(e).__name__}: {e}"
            patch = ""
            messages = agent.messages if agent else []
            segments = agent.get_segments() if agent else []
            num_summaries = agent._compaction_count if agent else 0

        # --- Build Episode ---
        # Return steps=[] so that _enrich_episode assigns all gateway traces
        # to this trajectory (the canonical rllm cookbook pattern).  The gateway
        # already captured every chat.completions.create() call with full
        # prompt_ids, completion_ids, and logprobs.
        trajectory = Trajectory(name="solver", steps=[])

        # Keep env alive for graders that reuse the sandbox
        return_env = None
        if keep_env_alive and env is not None:
            return_env = env
            env = None  # prevent cleanup below

        # Classify termination reason from exit_status
        error_details = build_error_details(exit_status)
        termination_reason = classify_termination(exit_status)

        artifacts: dict[str, Any] = {
            "patch": patch,
            "exit_status": exit_status,
            "messages": messages,
            "segments": segments,
            "env": return_env,
        }

        if env is not None:
            close_env(env, log_fn=log)

        episode = Episode(
            task=data,
            trajectories=[trajectory],
            artifacts=artifacts,
        )
        episode.termination_reason = termination_reason
        episode.metrics["num_summaries"] = num_summaries
        if error_details:
            episode.metadata["error"] = error_details
        self._save_trajectory(episode)
        return episode
