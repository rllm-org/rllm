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

import time
from typing import Any, Callable

from rllm.experimental.eval.types import AgentConfig, Task
from rllm.types import Episode, Step, Trajectory

from environment import (
    create_env,
    ensure_bootstrapped,
    load_runtime_configs,
)
from tasks.common import make_log, short_id

ensure_bootstrapped()

from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import LimitsExceeded

from openai_model import OpenAIClientModel
from tasks.swesmith import setup_swesmith_agent_env


# -------------------------------------------------------------------
# Agent wrapper: adds per-step logging, timeout, and FormatError handling
# -------------------------------------------------------------------

class ProgressLoggingAgent(DefaultAgent):
    """DefaultAgent with per-step logging and wall-clock timeout."""

    def __init__(
        self,
        *args,
        instance_id: str = "",
        agent_timeout: float = 0,
        log_fn: Callable[[str], None] = print,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.instance_id = instance_id
        self._step_count = 0
        self._agent_timeout = agent_timeout
        self._start_time: float | None = None
        self._log_fn = log_fn

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
                raise LimitsExceeded({
                    "role": "exit",
                    "content": "AgentTimeout",
                    "extra": {"exit_status": "AgentTimeout", "submission": ""},
                })

        self._step_count += 1
        self._log_fn(f"[{self.instance_id}] Step {self._step_count:3d} - querying model...")
        result_messages = super().step()

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
            return self.add_messages({"role": "user", "content": format_error})
        return super().execute_actions(message)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _close_env(env: Any, sid: str, log_fn: Callable) -> None:
    close = getattr(env, "close", None) or getattr(env, "stop", None)
    if close is None:
        return
    try:
        close()
    except Exception as exc:
        log_fn(f"[{sid}] Env cleanup error: {type(exc).__name__}: {exc}")


# -------------------------------------------------------------------
# AgentFlow
# -------------------------------------------------------------------

class SWEAgentFlow:
    """AgentFlow that runs mini-swe-agent to generate a patch.

    Constructor args configure the agent limits and sandbox settings.
    The ``run()`` method receives the task and inference config from
    the runner/engine.
    """

    def __init__(
        self,
        cost_limit: float = 3.0,
        step_limit: int = 0,
        command_timeout: int = 120,
        sandbox_timeout: int = 3600,
        agent_timeout: int = 0,
        verbose: bool = False,
    ):
        self.cost_limit = cost_limit
        self.step_limit = step_limit
        self.command_timeout = command_timeout
        self.sandbox_timeout = sandbox_timeout
        self.agent_timeout = agent_timeout
        self.verbose = verbose

    def run(self, task: Task, config: AgentConfig) -> Episode:
        """Run mini-swe-agent on a single task.

        Returns an Episode with:
        - One trajectory ("solver") with a Step per assistant message
        - artifacts: patch, exit_status, env (if kept alive for grading)
        """
        data = task.data
        log = make_log(self.verbose)
        sid = short_id(data["instance_id"])
        log(f"\n[{sid}] Starting patch generation...")

        env = None
        agent = None
        keep_env_alive = data["eval_type"] in {"swebench"}

        try:
            # --- Create sandbox ---
            env_start = time.monotonic()
            log(f"[{sid}] Creating sandbox environment...")
            env = create_env(data, command_timeout=self.command_timeout, sandbox_timeout=self.sandbox_timeout)
            log(f"[{sid}] Sandbox ready in {time.monotonic() - env_start:.1f}s")

            if data["eval_type"] == "swesmith":
                setup_start = time.monotonic()
                log(f"[{sid}] Preparing SWE-smith sandbox...")
                setup_swesmith_agent_env(env, data, sid, log)
                log(f"[{sid}] SWE-smith sandbox ready in {time.monotonic() - setup_start:.1f}s")

            # --- Create model ---
            agent_config, model_config, _ = load_runtime_configs()
            model = OpenAIClientModel(
                base_url=config.base_url,
                model_name=config.model,
                verbose=self.verbose,
                model_config=model_config,
            )

            # --- Run agent ---
            agent_config["cost_limit"] = self.cost_limit
            agent_config["step_limit"] = self.step_limit

            agent = ProgressLoggingAgent(
                model=model,
                env=env,
                instance_id=sid,
                agent_timeout=self.agent_timeout,
                log_fn=log,
                **agent_config,
            )

            problem_stmt = data["problem_statement"]
            log(f"[{sid}] Starting agent run...")
            result = agent.run(problem_stmt, problem_statement=problem_stmt)
            exit_status = result["exit_status"]
            patch = result["submission"]
            messages = agent.messages
            log(f"[{sid}] Patch generation complete: {exit_status}, {len(patch)} chars")

        except Exception as e:
            log(f"[{sid}] Agent error: {type(e).__name__}: {e}")
            exit_status = f"Error: {type(e).__name__}: {e}"
            patch = ""
            messages = agent.messages if agent else []

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

        artifacts: dict[str, Any] = {
            "patch": patch,
            "exit_status": exit_status,
            "messages": messages,
            "env": return_env,
        }

        if env is not None:
            _close_env(env, sid, log)

        return Episode(
            task=data,
            trajectories=[trajectory],
            artifacts=artifacts,
        )
