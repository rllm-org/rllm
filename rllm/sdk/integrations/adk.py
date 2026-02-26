"""Google ADK integration for rLLM trajectory tracking.

Provides ``RLLMTrajectoryPlugin``, a Google ADK ``BasePlugin`` that captures
all LLM calls during agent execution and builds rLLM ``Trajectory``
objects for SFT distillation and RL training.

Usage::

    from google.adk.agents import Agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from rllm.sdk.integrations.adk import RLLMTrajectoryPlugin

    agent = Agent(name="solver", model="gemini-2.5-flash", ...)
    plugin = RLLMTrajectoryPlugin()
    runner = Runner(
        app_name="my_app",
        agent=agent,
        session_service=InMemorySessionService(),
        plugins=[plugin],
    )

    for event in runner.run(user_id="u1", session_id="s1", new_message=msg):
        pass

    traj = plugin.get_trajectory()
    traj.reward = my_reward_fn(traj.output, expected)
"""

from __future__ import annotations

import json
import time
import uuid
from typing import TYPE_CHECKING, Any

from rllm.sdk.protocol import (
    LLMInput,
    LLMOutput,
    Trace,
    Trajectory,
    trace_to_step,
)

try:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.adk.plugins.base_plugin import BasePlugin
    from google.adk.tools.base_tool import BaseTool
    from google.genai import types

    _ADK_AVAILABLE = True
except ImportError:
    _ADK_AVAILABLE = False
    BasePlugin = object  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.tools.tool_context import ToolContext


# ---------------------------------------------------------------------------
# Gemini <-> OpenAI format converters
# ---------------------------------------------------------------------------


_GEMINI_TO_OPENAI_ROLE = {"model": "assistant", "user": "user"}


def _gemini_role_to_openai(role: str | None) -> str:
    if role is None:
        return "user"
    return _GEMINI_TO_OPENAI_ROLE.get(role, role)


def _safe_json(obj: Any) -> str:
    """Serialize *obj* to a JSON string, falling back to ``str()``."""
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(obj)


def _gemini_content_to_openai_messages(
    contents: list,
    system_instruction: Any = None,
) -> list[dict]:
    """Convert a list of Gemini ``types.Content`` to OpenAI-style messages.

    Handles text, ``function_call`` (tool_calls), ``function_response``
    (tool messages) and system instructions.
    """
    messages: list[dict] = []

    # ---- system instruction ---------------------------------------------------
    if system_instruction:
        if isinstance(system_instruction, str):
            messages.append({"role": "system", "content": system_instruction})
        elif hasattr(system_instruction, "parts"):
            text_parts = [p.text for p in system_instruction.parts if getattr(p, "text", None)]
            if text_parts:
                messages.append({"role": "system", "content": "\n".join(text_parts)})

    # ---- conversation contents ------------------------------------------------
    for content in contents:
        role = _gemini_role_to_openai(getattr(content, "role", None))
        parts = getattr(content, "parts", None) or []
        if not parts:
            continue

        text_parts: list[str] = []
        function_calls: list = []
        function_responses: list = []

        for part in parts:
            if getattr(part, "function_call", None):
                function_calls.append(part.function_call)
            elif getattr(part, "function_response", None):
                function_responses.append(part.function_response)
            elif getattr(part, "text", None):
                if not getattr(part, "thought", False):
                    text_parts.append(part.text)

        # Function responses -> separate ``tool`` role messages
        for fr in function_responses:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": getattr(fr, "id", None) or "",
                    "content": _safe_json(fr.response),
                }
            )

        # Function calls -> ``assistant`` message with ``tool_calls``
        if function_calls:
            msg: dict[str, Any] = {"role": "assistant"}
            if text_parts:
                msg["content"] = "\n".join(text_parts)
            tool_calls = []
            for fc in function_calls:
                tool_calls.append(
                    {
                        "id": getattr(fc, "id", None) or f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": fc.name,
                            "arguments": _safe_json(fc.args or {}),
                        },
                    }
                )
            msg["tool_calls"] = tool_calls
            messages.append(msg)
        elif text_parts and not function_responses:
            messages.append({"role": role, "content": "\n".join(text_parts)})

    return messages


def _gemini_response_to_openai_message(content: Any | None) -> dict:
    """Convert a Gemini response ``Content`` to an OpenAI-style message dict."""
    if content is None:
        return {"role": "assistant", "content": ""}

    text_parts: list[str] = []
    function_calls: list = []

    for part in getattr(content, "parts", []) or []:
        if getattr(part, "function_call", None):
            function_calls.append(part.function_call)
        elif getattr(part, "text", None) and not getattr(part, "thought", False):
            text_parts.append(part.text)

    msg: dict[str, Any] = {
        "role": "assistant",
        "content": "\n".join(text_parts) if text_parts else None,
    }

    if function_calls:
        tool_calls = []
        for fc in function_calls:
            tool_calls.append(
                {
                    "id": getattr(fc, "id", None) or f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": fc.name,
                        "arguments": _safe_json(fc.args or {}),
                    },
                }
            )
        msg["tool_calls"] = tool_calls

    return msg


def _extract_finish_reason(llm_response: Any) -> str:
    fr = getattr(llm_response, "finish_reason", None)
    if fr is None:
        return "stop"
    if isinstance(fr, str):
        return fr
    return str(fr.name).lower() if hasattr(fr, "name") else str(fr)


def _extract_usage(llm_response: Any) -> dict[str, int]:
    usage = getattr(llm_response, "usage_metadata", None)
    if usage is None:
        return {"prompt": 0, "completion": 0, "total": 0}
    prompt = getattr(usage, "prompt_token_count", 0) or 0
    completion = getattr(usage, "candidates_token_count", 0) or 0
    total = getattr(usage, "total_token_count", 0) or (prompt + completion)
    return {"prompt": prompt, "completion": completion, "total": total}


def _extract_tools_from_request(llm_request: Any) -> list[dict] | None:
    """Extract tool definitions from ``LlmRequest`` in OpenAI format."""
    config = getattr(llm_request, "config", None)
    if config is None:
        return None
    tools_list = getattr(config, "tools", None)
    if not tools_list:
        return None

    tools: list[dict] = []
    for tool in tools_list:
        fds = getattr(tool, "function_declarations", None)
        if not fds:
            continue
        for fd in fds:
            tool_def: dict[str, Any] = {
                "type": "function",
                "function": {"name": fd.name},
            }
            if getattr(fd, "description", None):
                tool_def["function"]["description"] = fd.description
            params = getattr(fd, "parameters", None)
            if params is not None:
                if hasattr(params, "model_dump"):
                    tool_def["function"]["parameters"] = params.model_dump(exclude_none=True)
                elif isinstance(params, dict):
                    tool_def["function"]["parameters"] = params
            tools.append(tool_def)
    return tools if tools else None


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------


class RLLMTrajectoryPlugin(BasePlugin):
    """ADK Plugin that captures LLM calls and builds rLLM Trajectory objects.

    Each model invocation is recorded as an rLLM ``Trace``, with Gemini-native
    types automatically converted to the OpenAI-compatible format that rLLM's
    training pipeline expects.

    After the runner finishes, call :py:meth:`get_trajectory` to obtain a
    ``Trajectory`` ready for reward assignment and training.

    Args:
        name: Plugin identifier (default ``"rllm_trajectory"``).

    Example::

        plugin = RLLMTrajectoryPlugin()
        runner = Runner(
            app_name="math_agent",
            agent=agent,
            session_service=InMemorySessionService(),
            plugins=[plugin],
        )

        for event in runner.run(user_id="u1", session_id="s1", new_message=msg):
            pass

        traj = plugin.get_trajectory()
        traj.reward = compute_reward(traj.output, expected_answer)
    """

    def __init__(self, name: str = "rllm_trajectory"):
        if not _ADK_AVAILABLE:
            raise ImportError("Google ADK is required for RLLMTrajectoryPlugin. Install it with: pip install google-adk")
        super().__init__(name=name)
        self._traces: list[Trace] = []
        self._trajectories: list[Trajectory] = []
        self._pending_request: Any = None
        self._pending_agent_name: str = "adk_agent"
        self._request_start_time: float = 0.0
        self._user_input: dict | None = None
        self._last_output: Any = None
        self._agent_name: str = "adk_agent"
        self._model_name: str = "unknown"
        self._per_agent_trajectories: dict[str, Trajectory] = {}

    # ---- lifecycle callbacks --------------------------------------------------

    async def before_run_callback(self, *, invocation_context: InvocationContext) -> types.Content | None:
        """Reset collection state at the start of each invocation."""
        self._traces = []
        self._pending_request = None
        self._user_input = None
        self._last_output = None
        self._per_agent_trajectories = {}
        self._agent_name = getattr(invocation_context.agent, "name", "adk_agent")
        return None

    async def after_run_callback(self, *, invocation_context: InvocationContext) -> None:
        """Build the ``Trajectory`` once the invocation completes.

        A combined trajectory (all agents) is always produced.  In addition,
        per-agent trajectories are built so multi-agent systems can assign
        rewards to each sub-agent independently via
        :py:meth:`get_trajectories_by_agent`.
        """
        # -- combined trajectory (all agents) --
        steps = [trace_to_step(t) for t in self._traces]
        traj = Trajectory(
            name=self._agent_name,
            steps=steps,
            reward=0.0,
            input=self._user_input,
            output=self._last_output,
            metadata={
                "source": "google_adk",
                "model": self._model_name,
                "num_llm_calls": len(self._traces),
            },
        )
        self._trajectories.append(traj)

        # -- per-agent trajectories --
        agent_traces: dict[str, list[Trace]] = {}
        for t in self._traces:
            a = t.metadata.get("agent", self._agent_name)
            agent_traces.setdefault(a, []).append(t)

        per_agent: dict[str, Trajectory] = {}
        for agent, traces in agent_traces.items():
            agent_steps = [trace_to_step(t) for t in traces]
            last_content = None
            for t in reversed(traces):
                msg = t.output.message if hasattr(t.output, "message") else {}
                c = msg.get("content") if isinstance(msg, dict) else None
                if c:
                    last_content = c
                    break
            per_agent[agent] = Trajectory(
                name=agent,
                steps=agent_steps,
                reward=0.0,
                input=self._user_input,
                output=last_content,
                metadata={
                    "source": "google_adk",
                    "model": self._model_name,
                    "num_llm_calls": len(traces),
                },
            )
        self._per_agent_trajectories = per_agent

    # ---- model callbacks ------------------------------------------------------

    async def before_model_callback(self, *, callback_context: CallbackContext, llm_request: LlmRequest) -> LlmResponse | None:
        """Store the outgoing ``LlmRequest`` so we can pair it with the response."""
        self._pending_request = llm_request
        self._pending_agent_name = getattr(callback_context, "agent_name", self._agent_name)
        self._request_start_time = time.perf_counter()

        if getattr(llm_request, "model", None):
            self._model_name = llm_request.model

        # Capture user input from the first model call
        if self._user_input is None:
            for content in llm_request.contents or []:
                if getattr(content, "role", None) == "user":
                    texts = [p.text for p in (getattr(content, "parts", None) or []) if getattr(p, "text", None)]
                    if texts:
                        self._user_input = {"message": "\n".join(texts)}
                        break

        return None

    async def after_model_callback(self, *, callback_context: CallbackContext, llm_response: LlmResponse) -> LlmResponse | None:
        """Pair the stored request with its response and create a ``Trace``."""
        if self._pending_request is None:
            return None

        latency_ms = (time.perf_counter() - self._request_start_time) * 1000
        request = self._pending_request
        agent_name = self._pending_agent_name
        self._pending_request = None

        # Convert request contents to OpenAI messages
        system_instruction = None
        config = getattr(request, "config", None)
        if config is not None:
            system_instruction = getattr(config, "system_instruction", None)

        openai_messages = _gemini_content_to_openai_messages(request.contents or [], system_instruction)

        # Convert response to OpenAI message
        response_msg = _gemini_response_to_openai_message(llm_response.content)

        if response_msg.get("content"):
            self._last_output = response_msg["content"]

        tokens = _extract_usage(llm_response)
        finish_reason = _extract_finish_reason(llm_response)
        model = getattr(request, "model", None) or self._model_name

        trace = Trace(
            trace_id=f"adk_{uuid.uuid4().hex[:16]}",
            session_name=agent_name,
            name=f"adk/{model}",
            input=LLMInput(
                messages=openai_messages,
                prompt_token_ids=[],
            ),
            output=LLMOutput(
                message=response_msg,
                finish_reason=finish_reason,
                output_token_ids=[],
            ),
            model=model,
            latency_ms=latency_ms,
            tokens=tokens,
            metadata={"source": "google_adk", "agent": agent_name},
            timestamp=time.time(),
            tools=_extract_tools_from_request(request),
        )

        self._traces.append(trace)
        return None

    # ---- tool callbacks -------------------------------------------------------

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict | None:
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict,
    ) -> dict | None:
        """Annotate the most recent trace with tool execution metadata."""
        if self._traces:
            last = self._traces[-1]
            last.metadata.setdefault("tool_executions", []).append(
                {
                    "tool_name": tool.name,
                    "tool_args": tool_args,
                }
            )
        return None

    # ---- public API -----------------------------------------------------------

    def get_trajectory(self) -> Trajectory:
        """Return the trajectory from the most recent completed invocation.

        Raises:
            ValueError: If no invocation has completed yet.
        """
        if not self._trajectories:
            raise ValueError("No trajectories collected yet. Make sure the runner has completed at least one invocation.")
        return self._trajectories[-1]

    def get_trajectories(self) -> list[Trajectory]:
        """Return all collected trajectories (one per completed invocation)."""
        return list(self._trajectories)

    def get_trajectories_by_agent(self) -> dict[str, Trajectory]:
        """Return per-agent trajectories from the most recent invocation.

        In a multi-agent system each sub-agent that made at least one LLM call
        gets its own ``Trajectory``.  The keys are agent names.

        Returns:
            Mapping of agent name to its ``Trajectory``.

        Raises:
            ValueError: If no invocation has completed yet.
        """
        if not self._trajectories:
            raise ValueError("No trajectories collected yet. Make sure the runner has completed at least one invocation.")
        return dict(self._per_agent_trajectories)

    def get_traces(self) -> list[Trace]:
        """Return the raw ``Trace`` objects from the most recent invocation."""
        return list(self._traces)

    def clear(self) -> None:
        """Clear all collected trajectories and traces."""
        self._traces.clear()
        self._trajectories.clear()
        self._per_agent_trajectories.clear()
