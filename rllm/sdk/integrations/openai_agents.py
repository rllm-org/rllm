"""OpenAI Agents SDK integration for rLLM trajectory tracking.

Provides ``RLLMTrajectoryHooks``, an OpenAI Agents SDK ``RunHooks`` subclass
that captures all LLM calls during agent execution and builds rLLM
``Trajectory`` objects for SFT distillation and RL training.

Usage::

    from agents import Agent, Runner
    from rllm.sdk.integrations.openai_agents import RLLMTrajectoryHooks

    agent = Agent(name="solver", model="gpt-4o", instructions="...")
    hooks = RLLMTrajectoryHooks()
    result = Runner.run_sync(agent, "What is 15 * 7 + 23?", hooks=hooks)

    traj = hooks.get_trajectory()
    traj.reward = my_reward_fn(traj.output, expected)
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from rllm.sdk.protocol import (
    LLMInput,
    LLMOutput,
    Trace,
    Trajectory,
    trace_to_step,
)

try:
    from agents.lifecycle import RunHooksBase

    _AGENTS_SDK_AVAILABLE = True
except ImportError:
    _AGENTS_SDK_AVAILABLE = False
    RunHooksBase = object  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Responses API <-> Chat Completions format converters
# ---------------------------------------------------------------------------


def _safe_json(obj: Any) -> str:
    """Serialize *obj* to a JSON string, falling back to ``str()``."""
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(obj)


def _responses_input_to_messages(
    system_prompt: str | None,
    input_items: list,
) -> list[dict]:
    """Convert OpenAI Responses API input items to Chat Completions messages.

    Handles ``message`` items (user/system/assistant), ``function_call`` items,
    ``function_call_output`` items, and the system prompt string.
    """
    messages: list[dict] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    pending_tool_calls: list[dict] = []
    pending_text: str | None = None

    for item in input_items:
        if not isinstance(item, dict):
            # Pydantic model -- dump to dict
            if hasattr(item, "model_dump"):
                item = item.model_dump(exclude_unset=True)
            elif hasattr(item, "__dict__"):
                item = vars(item)
            else:
                continue

        item_type = item.get("type")
        role = item.get("role")

        if item_type == "message":
            content = item.get("content")
            text = _extract_text_from_content(content)
            if text is not None:
                messages.append({"role": role or "user", "content": text})

        elif item_type == "function_call":
            # Assistant issued a tool call in a previous turn
            call_id = item.get("call_id") or item.get("id") or ""
            name = item.get("name", "")
            arguments = item.get("arguments", "")
            if isinstance(arguments, dict):
                arguments = _safe_json(arguments)
            # Merge consecutive function_calls into one assistant message
            pending_tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            )

        elif item_type == "function_call_output":
            # Flush any pending tool_calls as an assistant message first
            if pending_tool_calls:
                msg: dict[str, Any] = {"role": "assistant"}
                if pending_text:
                    msg["content"] = pending_text
                msg["tool_calls"] = pending_tool_calls
                messages.append(msg)
                pending_tool_calls = []
                pending_text = None

            call_id = item.get("call_id") or ""
            output = item.get("output", "")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": output if isinstance(output, str) else _safe_json(output),
                }
            )

        # Skip reasoning, mcp, and other non-message items

    # Flush any remaining pending tool_calls
    if pending_tool_calls:
        msg = {"role": "assistant"}
        if pending_text:
            msg["content"] = pending_text
        msg["tool_calls"] = pending_tool_calls
        messages.append(msg)

    return messages


def _extract_text_from_content(content: Any) -> str | None:
    """Extract plain text from a Responses API content field.

    Content can be a plain string, a list of content parts (dicts with
    ``type`` = ``input_text`` / ``output_text`` / ``text``), or ``None``.
    """
    if content is None:
        return None
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                t = part.get("text")
                if t:
                    text_parts.append(t)
            elif hasattr(part, "text"):
                t = getattr(part, "text", None)
                if t:
                    text_parts.append(t)
        return "\n".join(text_parts) if text_parts else None

    return str(content)


def _model_response_to_message(response: Any) -> dict:
    """Convert a ``ModelResponse`` to an OpenAI Chat Completions message dict."""
    output_items = getattr(response, "output", None) or []

    text_parts: list[str] = []
    tool_calls: list[dict] = []

    for item in output_items:
        item_type = getattr(item, "type", None)

        if item_type == "message":
            # ResponseOutputMessage
            for part in getattr(item, "content", []) or []:
                part_type = getattr(part, "type", None)
                if part_type in ("output_text", "text"):
                    text = getattr(part, "text", None)
                    if text:
                        text_parts.append(text)

        elif item_type == "function_call":
            # ResponseFunctionToolCall
            tool_calls.append(
                {
                    "id": getattr(item, "call_id", None) or f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": getattr(item, "name", ""),
                        "arguments": getattr(item, "arguments", "") or "",
                    },
                }
            )

    msg: dict[str, Any] = {
        "role": "assistant",
        "content": "\n".join(text_parts) if text_parts else None,
    }
    if tool_calls:
        msg["tool_calls"] = tool_calls

    return msg


def _extract_usage(response: Any) -> dict[str, int]:
    """Extract token usage from a ``ModelResponse``."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt": 0, "completion": 0, "total": 0}
    prompt = getattr(usage, "input_tokens", 0) or 0
    completion = getattr(usage, "output_tokens", 0) or 0
    total = getattr(usage, "total_tokens", 0) or (prompt + completion)
    return {"prompt": prompt, "completion": completion, "total": total}


def _extract_finish_reason(response: Any) -> str:
    """Infer finish reason from a ``ModelResponse``."""
    output_items = getattr(response, "output", None) or []
    has_tool_calls = any(getattr(i, "type", None) == "function_call" for i in output_items)
    if has_tool_calls:
        return "tool_calls"
    return "stop"


def _extract_tools_from_agent(agent: Any) -> list[dict] | None:
    """Extract tool definitions from an ``Agent`` in OpenAI format."""
    agent_tools = getattr(agent, "tools", None)
    if not agent_tools:
        return None

    tools: list[dict] = []
    for tool in agent_tools:
        name = getattr(tool, "name", None)
        if not name:
            continue
        tool_def: dict[str, Any] = {
            "type": "function",
            "function": {"name": name},
        }
        desc = getattr(tool, "description", None)
        if desc:
            tool_def["function"]["description"] = desc
        params = getattr(tool, "params_json_schema", None)
        if params is not None:
            tool_def["function"]["parameters"] = params
        tools.append(tool_def)
    return tools if tools else None


def _extract_model_name(agent: Any) -> str:
    """Extract the model name from an agent, with a safe fallback."""
    model = getattr(agent, "model", None)
    if model is None:
        return "unknown"
    if isinstance(model, str):
        return model
    return getattr(model, "model", None) or str(model)


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


class RLLMTrajectoryHooks(RunHooksBase):  # type: ignore[misc]
    """OpenAI Agents SDK RunHooks that captures LLM calls and builds rLLM Trajectory objects.

    Each model invocation is recorded as an rLLM ``Trace``, with the
    Responses API types automatically converted to the Chat Completions
    message format that rLLM's training pipeline expects.

    After the runner finishes, call :py:meth:`get_trajectory` to obtain a
    ``Trajectory`` ready for reward assignment and training.

    Example::

        from agents import Agent, Runner
        from rllm.sdk.integrations.openai_agents import RLLMTrajectoryHooks

        agent = Agent(name="math_solver", model="gpt-4o", instructions="...")
        hooks = RLLMTrajectoryHooks()
        result = Runner.run_sync(agent, "What is 15 * 7 + 23?", hooks=hooks)

        traj = hooks.get_trajectory()
        traj.reward = 1.0 if "128" in str(traj.output) else 0.0
    """

    def __init__(self) -> None:
        if not _AGENTS_SDK_AVAILABLE:
            raise ImportError("OpenAI Agents SDK is required for RLLMTrajectoryHooks. Install it with: pip install openai-agents")
        self._traces: list[Trace] = []
        self._trajectories: list[Trajectory] = []
        self._pending_system_prompt: str | None = None
        self._pending_input_items: list | None = None
        self._pending_agent_name: str = "openai_agent"
        self._request_start_time: float = 0.0
        self._user_input: dict | None = None
        self._last_output: Any = None
        self._agent_name: str = "openai_agent"
        self._model_name: str = "unknown"
        self._per_agent_trajectories: dict[str, Trajectory] = {}
        self._trajectory_built: bool = False

    # ---- agent lifecycle ------------------------------------------------------

    async def on_agent_start(
        self,
        context: Any,
        agent: Any,
    ) -> None:
        """Track agent name at the start of each agent invocation."""
        name = getattr(agent, "name", None)
        if name:
            if self._agent_name == "openai_agent":
                self._agent_name = name
            self._model_name = _extract_model_name(agent)

    async def on_agent_end(
        self,
        context: Any,
        agent: Any,
        output: Any,
    ) -> None:
        """Capture agent output.  Trajectory is built lazily by get_trajectory()."""
        if isinstance(output, str):
            self._last_output = output
        elif output is not None:
            self._last_output = str(output)
        # Mark dirty so the next get_trajectory() rebuilds
        self._trajectory_built = False

    # ---- LLM lifecycle --------------------------------------------------------

    async def on_llm_start(
        self,
        context: Any,
        agent: Any,
        system_prompt: str | None,
        input_items: list,
    ) -> None:
        """Store the outgoing LLM request so we can pair it with the response."""
        self._pending_system_prompt = system_prompt
        self._pending_input_items = list(input_items) if input_items else []
        self._pending_agent_name = getattr(agent, "name", self._agent_name)
        self._request_start_time = time.perf_counter()
        self._model_name = _extract_model_name(agent)
        self._trajectory_built = False

        if self._user_input is None:
            self._user_input = self._extract_user_input(input_items)

    async def on_llm_end(
        self,
        context: Any,
        agent: Any,
        response: Any,
    ) -> None:
        """Pair the stored request with the response and create a ``Trace``."""
        if self._pending_input_items is None:
            return

        latency_ms = (time.perf_counter() - self._request_start_time) * 1000
        system_prompt = self._pending_system_prompt
        input_items = self._pending_input_items
        agent_name = self._pending_agent_name
        self._pending_input_items = None
        self._pending_system_prompt = None

        openai_messages = _responses_input_to_messages(system_prompt, input_items)
        response_msg = _model_response_to_message(response)

        if response_msg.get("content"):
            self._last_output = response_msg["content"]

        tokens = _extract_usage(response)
        finish_reason = _extract_finish_reason(response)
        model = self._model_name

        trace = Trace(
            trace_id=f"oai_{uuid.uuid4().hex[:16]}",
            session_name=agent_name,
            name=f"openai_agents/{model}",
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
            metadata={"source": "openai_agents", "agent": agent_name},
            timestamp=time.time(),
            tools=_extract_tools_from_agent(agent),
        )

        self._traces.append(trace)

    # ---- tool lifecycle -------------------------------------------------------

    async def on_tool_start(
        self,
        context: Any,
        agent: Any,
        tool: Any,
    ) -> None:
        pass

    async def on_tool_end(
        self,
        context: Any,
        agent: Any,
        tool: Any,
        result: str,
    ) -> None:
        """Annotate the most recent trace with tool execution metadata."""
        if self._traces:
            last = self._traces[-1]
            tool_name = getattr(tool, "name", "unknown")
            last.metadata.setdefault("tool_executions", []).append(
                {
                    "tool_name": tool_name,
                    "tool_result": result[:500] if isinstance(result, str) else str(result)[:500],
                }
            )

    # ---- handoff lifecycle ----------------------------------------------------

    async def on_handoff(
        self,
        context: Any,
        from_agent: Any,
        to_agent: Any,
    ) -> None:
        """Record handoff metadata on the most recent trace."""
        if self._traces:
            last = self._traces[-1]
            from_name = getattr(from_agent, "name", "unknown")
            to_name = getattr(to_agent, "name", "unknown")
            last.metadata.setdefault("handoffs", []).append({"from": from_name, "to": to_name})

    # ---- internal helpers -----------------------------------------------------

    @staticmethod
    def _extract_user_input(input_items: list) -> dict | None:
        """Extract user message text from the first user message in input_items."""
        for item in input_items or []:
            if isinstance(item, dict):
                if item.get("type") == "message" and item.get("role") == "user":
                    content = item.get("content")
                    text = _extract_text_from_content(content)
                    if text:
                        return {"message": text}
            elif hasattr(item, "model_dump"):
                d = item.model_dump(exclude_unset=True)
                if d.get("type") == "message" and d.get("role") == "user":
                    text = _extract_text_from_content(d.get("content"))
                    if text:
                        return {"message": text}
        return None

    def _ensure_trajectories(self) -> None:
        """Rebuild trajectories from traces if they are stale.

        Always builds a trajectory even when ``_traces`` is empty (e.g. the
        model call raised an exception before any trace was recorded).  This
        ensures that ``output`` / ``input`` are preserved on the resulting
        Trajectory so downstream code never sees a null output.
        """
        if self._trajectory_built:
            return
        self._trajectory_built = True

        # -- combined trajectory (all agents) --
        steps = [trace_to_step(t) for t in self._traces]
        traj = Trajectory(
            name=self._agent_name,
            steps=steps,
            reward=0.0,
            input=self._user_input,
            output=self._last_output,
            metadata={
                "source": "openai_agents",
                "model": self._model_name,
                "num_llm_calls": len(self._traces),
            },
        )
        # Replace the most recent trajectory if one already exists for
        # this run, otherwise append a new one.
        if self._trajectories:
            self._trajectories[-1] = traj
        else:
            self._trajectories.append(traj)

        # -- per-agent trajectories --
        agent_traces: dict[str, list[Trace]] = {}
        for t in self._traces:
            a = t.metadata.get("agent", self._agent_name)
            agent_traces.setdefault(a, []).append(t)

        per_agent: dict[str, Trajectory] = {}
        for agent_name, traces in agent_traces.items():
            agent_steps = [trace_to_step(t) for t in traces]
            last_content = None
            for t in reversed(traces):
                msg = t.output.message if hasattr(t.output, "message") else {}
                c = msg.get("content") if isinstance(msg, dict) else None
                if c:
                    last_content = c
                    break
            per_agent[agent_name] = Trajectory(
                name=agent_name,
                steps=agent_steps,
                reward=0.0,
                input=self._user_input,
                output=last_content,
                metadata={
                    "source": "openai_agents",
                    "model": self._model_name,
                    "num_llm_calls": len(traces),
                },
            )
        self._per_agent_trajectories = per_agent

    # ---- public API -----------------------------------------------------------

    def get_trajectory(self) -> Trajectory:
        """Return the trajectory from the most recent completed run.

        The trajectory is built (or rebuilt) on demand from the collected
        traces, so it is always up to date even if called multiple times.

        Raises:
            ValueError: If no LLM calls have been captured yet.
        """
        self._ensure_trajectories()
        if not self._trajectories:
            raise ValueError("No trajectories collected yet. Make sure the runner has completed at least one invocation.")
        return self._trajectories[-1]

    def get_trajectories(self) -> list[Trajectory]:
        """Return all collected trajectories (one per completed run)."""
        self._ensure_trajectories()
        return list(self._trajectories)

    def get_trajectories_by_agent(self) -> dict[str, Trajectory]:
        """Return per-agent trajectories from the most recent run.

        In a multi-agent system each sub-agent that made at least one LLM call
        gets its own ``Trajectory``.  The keys are agent names.

        Returns:
            Mapping of agent name to its ``Trajectory``.

        Raises:
            ValueError: If no LLM calls have been captured yet.
        """
        self._ensure_trajectories()
        if not self._trajectories:
            raise ValueError("No trajectories collected yet. Make sure the runner has completed at least one invocation.")
        return dict(self._per_agent_trajectories)

    def get_traces(self) -> list[Trace]:
        """Return the raw ``Trace`` objects from the most recent run."""
        return list(self._traces)

    def clear(self) -> None:
        """Clear all collected trajectories and traces, resetting for a new run."""
        self._traces.clear()
        self._trajectories.clear()
        self._per_agent_trajectories.clear()
        self._user_input = None
        self._last_output = None
        self._agent_name = "openai_agent"
        self._model_name = "unknown"
        self._trajectory_built = False
