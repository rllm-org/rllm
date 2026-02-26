"""Strands Agents SDK integration for rLLM trajectory tracking.

Provides ``RLLMTrajectoryHookProvider``, a Strands ``HookProvider`` that
captures all LLM calls during agent execution and builds rLLM
``Trajectory`` objects for SFT distillation and RL training.

Usage::

    from strands import Agent
    from rllm.sdk.integrations.strands import RLLMTrajectoryHookProvider

    hook_provider = RLLMTrajectoryHookProvider()
    agent = Agent(model="...", hooks=[hook_provider])

    result = agent("What is 15 * 7 + 23?")

    traj = hook_provider.get_trajectory()
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
    from strands.hooks.events import (
        AfterInvocationEvent,
        AfterModelCallEvent,
        AfterToolCallEvent,
        BeforeInvocationEvent,
        BeforeModelCallEvent,
    )
    from strands.hooks.registry import HookRegistry

    _STRANDS_AVAILABLE = True
except ImportError:
    _STRANDS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Bedrock (Strands) <-> OpenAI format converters
# ---------------------------------------------------------------------------


def _safe_json(obj: Any) -> str:
    """Serialize *obj* to a JSON string, falling back to ``str()``."""
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(obj)


def _strands_messages_to_openai(
    messages: list[dict],
    system_prompt: str | None = None,
) -> list[dict]:
    """Convert Strands/Bedrock-style messages to OpenAI Chat Completions format.

    Strands ``Message`` is a ``TypedDict`` with ``role`` (``"user"`` |
    ``"assistant"``) and ``content`` (``list[ContentBlock]``).  Each
    ``ContentBlock`` may contain ``text``, ``toolUse``, ``toolResult``,
    or ``reasoningContent`` keys.
    """
    openai_messages: list[dict] = []

    if system_prompt:
        openai_messages.append({"role": "system", "content": system_prompt})

    for msg in messages:
        role = msg.get("role", "user")
        content_blocks = msg.get("content", [])
        if not content_blocks:
            continue

        text_parts: list[str] = []
        tool_calls: list[dict] = []
        tool_results: list[dict] = []

        for block in content_blocks:
            if not isinstance(block, dict):
                continue

            if "text" in block:
                text_parts.append(block["text"])

            elif "toolUse" in block:
                tu = block["toolUse"]
                tool_calls.append(
                    {
                        "id": tu.get("toolUseId", f"call_{uuid.uuid4().hex[:8]}"),
                        "type": "function",
                        "function": {
                            "name": tu.get("name", ""),
                            "arguments": _safe_json(tu.get("input", {})),
                        },
                    }
                )

            elif "toolResult" in block:
                tr = block["toolResult"]
                result_content_parts = []
                for rc in tr.get("content", []):
                    if isinstance(rc, dict) and "text" in rc:
                        result_content_parts.append(rc["text"])
                    elif isinstance(rc, dict) and "json" in rc:
                        result_content_parts.append(_safe_json(rc["json"]))
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tr.get("toolUseId", ""),
                        "content": "\n".join(result_content_parts) if result_content_parts else _safe_json(tr),
                    }
                )

        # Emit tool results as separate messages
        for tr_msg in tool_results:
            openai_messages.append(tr_msg)

        if tool_calls:
            msg_dict: dict[str, Any] = {"role": "assistant"}
            if text_parts:
                msg_dict["content"] = "\n".join(text_parts)
            msg_dict["tool_calls"] = tool_calls
            openai_messages.append(msg_dict)
        elif text_parts and not tool_results:
            openai_messages.append({"role": role, "content": "\n".join(text_parts)})

    return openai_messages


def _strands_message_to_openai(message: dict) -> dict:
    """Convert a single Strands response ``Message`` to an OpenAI-style assistant message."""
    if not message:
        return {"role": "assistant", "content": ""}

    content_blocks = message.get("content", [])
    text_parts: list[str] = []
    tool_calls: list[dict] = []

    for block in content_blocks:
        if not isinstance(block, dict):
            continue

        if "text" in block:
            text_parts.append(block["text"])
        elif "toolUse" in block:
            tu = block["toolUse"]
            tool_calls.append(
                {
                    "id": tu.get("toolUseId", f"call_{uuid.uuid4().hex[:8]}"),
                    "type": "function",
                    "function": {
                        "name": tu.get("name", ""),
                        "arguments": _safe_json(tu.get("input", {})),
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


_STRANDS_TO_OPENAI_STOP_REASON = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "content_filtered": "content_filter",
    "guardrail_intervened": "content_filter",
    "interrupt": "stop",
}


def _extract_finish_reason(stop_reason: str | None) -> str:
    if stop_reason is None:
        return "stop"
    return _STRANDS_TO_OPENAI_STOP_REASON.get(stop_reason, stop_reason)


def _extract_usage(agent: Any) -> dict[str, int]:
    """Extract token usage from the agent's event loop metrics."""
    metrics = getattr(agent, "event_loop_metrics", None)
    if metrics is None:
        return {"prompt": 0, "completion": 0, "total": 0}
    prompt = getattr(metrics, "input_tokens", 0) or 0
    completion = getattr(metrics, "output_tokens", 0) or 0
    total = prompt + completion
    return {"prompt": prompt, "completion": completion, "total": total}


def _extract_tools_from_agent(agent: Any) -> list[dict] | None:
    """Extract tool definitions from the agent's tool registry in OpenAI format."""
    registry = getattr(agent, "tool_registry", None)
    if registry is None:
        return None

    try:
        all_tools = registry.get_all_tools_config()
    except Exception:
        return None

    if not all_tools:
        return None

    tools: list[dict] = []
    for name, config in all_tools.items():
        tool_spec = config.get("toolSpec", {}) if isinstance(config, dict) else {}
        tool_def: dict[str, Any] = {
            "type": "function",
            "function": {"name": name},
        }
        desc = tool_spec.get("description")
        if desc:
            tool_def["function"]["description"] = desc
        input_schema = tool_spec.get("inputSchema", {}).get("json")
        if input_schema:
            tool_def["function"]["parameters"] = input_schema
        tools.append(tool_def)

    return tools if tools else None


def _extract_model_name(agent: Any) -> str:
    """Extract the model name from the agent."""
    model = getattr(agent, "model", None)
    if model is None:
        return "unknown"
    model_id = getattr(model, "config", {})
    if isinstance(model_id, dict):
        mid = model_id.get("model_id")
        if mid:
            return mid
    if hasattr(model, "model_id"):
        return model.model_id
    return str(type(model).__name__)


def _extract_text_from_message(message: dict | None) -> str | None:
    """Extract plain text content from a Strands Message."""
    if not message:
        return None
    for block in message.get("content", []):
        if isinstance(block, dict) and "text" in block:
            return block["text"]
    return None


# ---------------------------------------------------------------------------
# HookProvider
# ---------------------------------------------------------------------------


class RLLMTrajectoryHookProvider:
    """Strands HookProvider that captures LLM calls and builds rLLM Trajectory objects.

    Each model invocation is recorded as an rLLM ``Trace``, with the
    Bedrock-style message types automatically converted to the OpenAI Chat
    Completions format that rLLM's training pipeline expects.

    After the agent finishes, call :py:meth:`get_trajectory` to obtain a
    ``Trajectory`` ready for reward assignment and training.

    Args:
        name: Name for the hook provider (used in trajectory metadata).

    Example::

        from strands import Agent
        from rllm.sdk.integrations.strands import RLLMTrajectoryHookProvider

        hook_provider = RLLMTrajectoryHookProvider()
        agent = Agent(model="...", hooks=[hook_provider])
        result = agent("What is 15 * 7 + 23?")

        traj = hook_provider.get_trajectory()
        traj.reward = 1.0 if "128" in str(traj.output) else 0.0
    """

    def __init__(self, name: str = "rllm_trajectory") -> None:
        if not _STRANDS_AVAILABLE:
            raise ImportError("Strands Agents SDK is required for RLLMTrajectoryHookProvider. Install it with: pip install strands-agents")
        self._name = name
        self._traces: list[Trace] = []
        self._trajectories: list[Trajectory] = []
        self._pending_messages: list[dict] | None = None
        self._request_start_time: float = 0.0
        self._user_input: dict | None = None
        self._last_output: Any = None
        self._agent_name: str = "strands_agent"
        self._model_name: str = "unknown"
        self._system_prompt: str | None = None
        self._trajectory_built: bool = False

    # ---- HookProvider protocol ------------------------------------------------

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Register callbacks for Strands agent lifecycle events."""
        registry.add_callback(BeforeInvocationEvent, self._on_before_invocation)
        registry.add_callback(BeforeModelCallEvent, self._on_before_model)
        registry.add_callback(AfterModelCallEvent, self._on_after_model)
        registry.add_callback(AfterToolCallEvent, self._on_after_tool)
        registry.add_callback(AfterInvocationEvent, self._on_after_invocation)

    # ---- Invocation lifecycle -------------------------------------------------

    def _on_before_invocation(self, event: BeforeInvocationEvent) -> None:
        """Reset collection state at the start of each invocation."""
        self._traces = []
        self._pending_messages = None
        self._user_input = None
        self._last_output = None
        self._trajectory_built = False

        agent = event.agent
        self._agent_name = getattr(agent, "name", None) or "strands_agent"
        self._model_name = _extract_model_name(agent)
        self._system_prompt = getattr(agent, "system_prompt", None)

        # Capture user input from the messages
        messages = getattr(event, "messages", None) or getattr(agent, "messages", [])
        if messages:
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    text = _extract_text_from_message(msg)
                    if text:
                        self._user_input = {"message": text}
                        break

    def _on_after_invocation(self, event: AfterInvocationEvent) -> None:
        """Build the Trajectory once the invocation completes."""
        result = event.result
        if result is not None:
            output_text = str(result)
            if output_text:
                self._last_output = output_text

        self._build_trajectories()

    # ---- Model lifecycle ------------------------------------------------------

    def _on_before_model(self, event: BeforeModelCallEvent) -> None:
        """Snapshot the current messages and start the timer."""
        agent = event.agent
        self._pending_messages = list(getattr(agent, "messages", []))
        self._request_start_time = time.perf_counter()
        self._model_name = _extract_model_name(agent)
        self._trajectory_built = False

        if self._user_input is None:
            messages = getattr(agent, "messages", [])
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    text = _extract_text_from_message(msg)
                    if text:
                        self._user_input = {"message": text}
                        break

    def _on_after_model(self, event: AfterModelCallEvent) -> None:
        """Pair the stored messages with the response and create a Trace."""
        if self._pending_messages is None:
            return

        if event.exception is not None:
            self._pending_messages = None
            return

        latency_ms = (time.perf_counter() - self._request_start_time) * 1000
        pending_messages = self._pending_messages
        self._pending_messages = None

        stop_response = event.stop_response
        if stop_response is None:
            return

        response_message = stop_response.message
        stop_reason = stop_response.stop_reason

        openai_messages = _strands_messages_to_openai(pending_messages, self._system_prompt)
        response_msg = _strands_message_to_openai(response_message)

        if response_msg.get("content"):
            self._last_output = response_msg["content"]

        agent = event.agent
        tokens = _extract_usage(agent)
        finish_reason = _extract_finish_reason(stop_reason)
        model = self._model_name

        trace = Trace(
            trace_id=f"strands_{uuid.uuid4().hex[:16]}",
            session_name=self._agent_name,
            name=f"strands/{model}",
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
            metadata={"source": "strands", "agent": self._agent_name},
            timestamp=time.time(),
            tools=_extract_tools_from_agent(agent),
        )

        self._traces.append(trace)

    # ---- Tool lifecycle -------------------------------------------------------

    def _on_after_tool(self, event: AfterToolCallEvent) -> None:
        """Annotate the most recent trace with tool execution metadata."""
        if self._traces:
            last = self._traces[-1]
            tool_name = "unknown"
            selected = getattr(event, "selected_tool", None)
            if selected is not None:
                tool_name = getattr(selected, "tool_name", "unknown")
            else:
                tool_use = getattr(event, "tool_use", None)
                if tool_use:
                    tool_name = tool_use.get("name", "unknown")

            tool_result = getattr(event, "result", None)
            result_text = ""
            if tool_result and isinstance(tool_result, dict):
                for rc in tool_result.get("content", []):
                    if isinstance(rc, dict) and "text" in rc:
                        result_text = rc["text"][:500]
                        break

            last.metadata.setdefault("tool_executions", []).append(
                {
                    "tool_name": tool_name,
                    "tool_result": result_text,
                }
            )

    # ---- internal helpers -----------------------------------------------------

    def _build_trajectories(self) -> None:
        """Build Trajectory from collected traces."""
        if self._trajectory_built or not self._traces:
            return
        self._trajectory_built = True

        steps = [trace_to_step(t) for t in self._traces]
        traj = Trajectory(
            name=self._agent_name,
            steps=steps,
            reward=0.0,
            input=self._user_input,
            output=self._last_output,
            metadata={
                "source": "strands",
                "model": self._model_name,
                "num_llm_calls": len(self._traces),
            },
        )
        self._trajectories.append(traj)

    # ---- public API -----------------------------------------------------------

    def get_trajectory(self) -> Trajectory:
        """Return the trajectory from the most recent completed invocation.

        Raises:
            ValueError: If no invocation has completed yet.
        """
        self._build_trajectories()
        if not self._trajectories:
            raise ValueError("No trajectories collected yet. Make sure the agent has completed at least one invocation.")
        return self._trajectories[-1]

    def get_trajectories(self) -> list[Trajectory]:
        """Return all collected trajectories (one per completed invocation)."""
        self._build_trajectories()
        return list(self._trajectories)

    def get_traces(self) -> list[Trace]:
        """Return the raw ``Trace`` objects from the most recent invocation."""
        return list(self._traces)

    def clear(self) -> None:
        """Clear all collected trajectories and traces."""
        self._traces.clear()
        self._trajectories.clear()
        self._user_input = None
        self._last_output = None
        self._agent_name = "strands_agent"
        self._model_name = "unknown"
        self._trajectory_built = False
