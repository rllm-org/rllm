"""SmolAgents integration for rLLM trajectory tracking.

Provides ``RLLMSmolAgentsTracer``, a model-wrapping tracer that captures
all LLM calls during SmolAgents agent execution and builds rLLM
``Trajectory`` objects for SFT distillation and RL training.

Usage::

    from smolagents import ToolCallingAgent, OpenAIServerModel
    from rllm.sdk.integrations.smolagents import RLLMSmolAgentsTracer

    tracer = RLLMSmolAgentsTracer()
    model = OpenAIServerModel(model_id="gpt-4o")
    wrapped = tracer.wrap_model(model)

    agent = ToolCallingAgent(tools=[], model=wrapped)
    result = agent.run("What is 15 * 7 + 23?")

    traj = tracer.get_trajectory()
    traj.reward = 1.0 if "128" in str(traj.output) else 0.0
"""

from __future__ import annotations

import json
import logging
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
    import smolagents  # noqa: F401

    _SMOLAGENTS_AVAILABLE = True
except ImportError:
    _SMOLAGENTS_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SmolAgents <-> OpenAI format converters
# ---------------------------------------------------------------------------


def _safe_json(obj: Any) -> str:
    """Serialize *obj* to a JSON string, falling back to ``str()``."""
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(obj)


def _smolagents_messages_to_openai(messages: list) -> list[dict]:
    """Convert SmolAgents ChatMessage list to OpenAI Chat Completions format.

    SmolAgents ``ChatMessage`` has ``.role``, ``.content``, and optionally
    ``.tool_calls`` (list of ``ChatMessageToolCall`` with ``.id``,
    ``.type``, ``.function``).
    """
    openai_messages: list[dict] = []

    for msg in messages:
        if isinstance(msg, dict):
            openai_messages.append(msg)
            continue

        role = getattr(msg, "role", "user")
        content = getattr(msg, "content", None)
        tool_calls = getattr(msg, "tool_calls", None)
        tool_call_id = getattr(msg, "tool_call_id", None)

        # Handle tool result messages
        if role == "tool":
            oai_msg: dict[str, Any] = {
                "role": "tool",
                "content": str(content) if content is not None else "",
            }
            if tool_call_id:
                oai_msg["tool_call_id"] = tool_call_id
            openai_messages.append(oai_msg)
            continue

        oai_msg = {"role": role}

        if content is not None:
            if isinstance(content, str):
                oai_msg["content"] = content
            elif isinstance(content, list):
                # Multimodal content blocks
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "image":
                            pass  # Skip image blocks for trace
                    elif isinstance(block, str):
                        text_parts.append(block)
                oai_msg["content"] = "\n".join(text_parts) if text_parts else str(content)
            else:
                oai_msg["content"] = str(content)

        if tool_calls:
            oai_tool_calls = []
            for tc in tool_calls:
                tc_dict: dict[str, Any]
                if isinstance(tc, dict):
                    tc_dict = tc
                else:
                    tc_id = getattr(tc, "id", None) or f"call_{uuid.uuid4().hex[:8]}"
                    tc_type = getattr(tc, "type", "function")
                    func = getattr(tc, "function", None)
                    if func is not None:
                        if isinstance(func, dict):
                            func_name = func.get("name", "")
                            func_args = func.get("arguments", "")
                        else:
                            func_name = getattr(func, "name", "")
                            func_args = getattr(func, "arguments", "")
                    else:
                        func_name = getattr(tc, "name", "")
                        func_args = getattr(tc, "arguments", "")
                    if not isinstance(func_args, str):
                        func_args = _safe_json(func_args)
                    tc_dict = {
                        "id": tc_id,
                        "type": tc_type,
                        "function": {
                            "name": func_name,
                            "arguments": func_args,
                        },
                    }
                oai_tool_calls.append(tc_dict)
            oai_msg["tool_calls"] = oai_tool_calls

        openai_messages.append(oai_msg)

    return openai_messages


def _smolagents_response_to_openai(output: Any) -> dict:
    """Convert a SmolAgents model output to an OpenAI-style assistant message."""
    if output is None:
        return {"role": "assistant", "content": ""}

    if isinstance(output, dict):
        return output

    # ChatMessage response
    role = getattr(output, "role", "assistant")
    content = getattr(output, "content", None)
    tool_calls = getattr(output, "tool_calls", None)

    msg: dict[str, Any] = {"role": role}

    if content is not None:
        if isinstance(content, str):
            msg["content"] = content
        elif isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            msg["content"] = "\n".join(text_parts) if text_parts else str(content)
        else:
            msg["content"] = str(content)
    else:
        msg["content"] = None

    if tool_calls:
        oai_tool_calls = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                oai_tool_calls.append(tc)
            else:
                tc_id = getattr(tc, "id", None) or f"call_{uuid.uuid4().hex[:8]}"
                func = getattr(tc, "function", None)
                if func is not None:
                    if isinstance(func, dict):
                        func_name = func.get("name", "")
                        func_args = func.get("arguments", "")
                    else:
                        func_name = getattr(func, "name", "")
                        func_args = getattr(func, "arguments", "")
                else:
                    func_name = getattr(tc, "name", "")
                    func_args = getattr(tc, "arguments", "")
                if not isinstance(func_args, str):
                    func_args = _safe_json(func_args)
                oai_tool_calls.append(
                    {
                        "id": tc_id,
                        "type": "function",
                        "function": {
                            "name": func_name,
                            "arguments": func_args,
                        },
                    }
                )
        msg["tool_calls"] = oai_tool_calls

    return msg


def _extract_finish_reason(output: Any) -> str:
    """Infer finish reason from a SmolAgents model response."""
    if output is None:
        return "stop"
    tool_calls = getattr(output, "tool_calls", None)
    if tool_calls:
        return "tool_calls"
    return "stop"


# ---------------------------------------------------------------------------
# Tracing model wrapper
# ---------------------------------------------------------------------------


class _TracingModelWrapper:
    """Transparent proxy around a SmolAgents model that records traces."""

    def __init__(self, inner_model: Any, tracer: RLLMSmolAgentsTracer) -> None:
        self._inner = inner_model
        self._tracer = tracer

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __call__(self, messages: list, **kwargs: Any) -> Any:
        # Snapshot input
        openai_input = _smolagents_messages_to_openai(messages)
        start_time = time.perf_counter()

        # Delegate
        output = self._inner(messages, **kwargs)

        # Record trace
        latency_ms = (time.perf_counter() - start_time) * 1000
        response_msg = _smolagents_response_to_openai(output)
        finish_reason = _extract_finish_reason(output)

        if response_msg.get("content"):
            self._tracer._last_output = response_msg["content"]

        model_name = self._tracer._model_name

        trace = Trace(
            trace_id=f"smol_{uuid.uuid4().hex[:16]}",
            session_name="smolagents",
            name=f"smolagents/{model_name}",
            input=LLMInput(
                messages=openai_input,
                prompt_token_ids=[],
            ),
            output=LLMOutput(
                message=response_msg,
                finish_reason=finish_reason,
                output_token_ids=[],
            ),
            model=model_name,
            latency_ms=latency_ms,
            tokens={"prompt": 0, "completion": 0, "total": 0},
            metadata={"source": "smolagents"},
            timestamp=time.time(),
        )

        self._tracer._traces.append(trace)
        return output


# ---------------------------------------------------------------------------
# Public tracer class
# ---------------------------------------------------------------------------


class RLLMSmolAgentsTracer:
    """SmolAgents tracer that captures LLM calls and builds rLLM Trajectory objects.

    Wraps a SmolAgents model via :py:meth:`wrap_model` so that every
    ``__call__`` is intercepted and recorded as an rLLM ``Trace``.  After
    the agent finishes, call :py:meth:`get_trajectory` to obtain a
    ``Trajectory`` ready for reward assignment and training.

    Example::

        from smolagents import ToolCallingAgent, OpenAIServerModel
        from rllm.sdk.integrations.smolagents import RLLMSmolAgentsTracer

        tracer = RLLMSmolAgentsTracer()
        model = OpenAIServerModel(model_id="gpt-4o")
        wrapped = tracer.wrap_model(model)

        agent = ToolCallingAgent(tools=[], model=wrapped)
        result = agent.run("What is 15 * 7 + 23?")

        traj = tracer.get_trajectory()
    """

    def __init__(self) -> None:
        self._traces: list[Trace] = []
        self._trajectories: list[Trajectory] = []
        self._user_input: dict | None = None
        self._last_output: Any = None
        self._model_name: str = "unknown"
        self._trajectory_built: bool = False

    def wrap_model(self, model: Any) -> _TracingModelWrapper:
        """Wrap a SmolAgents model to intercept LLM calls.

        Args:
            model: A SmolAgents model instance (e.g., ``OpenAIServerModel``).

        Returns:
            A wrapped model that can be passed to SmolAgents agents.
        """
        # Try to extract model name
        model_id = getattr(model, "model_id", None)
        if model_id:
            self._model_name = model_id
        elif hasattr(model, "model"):
            self._model_name = str(model.model)
        return _TracingModelWrapper(model, self)

    def _build_trajectories(self) -> None:
        """Build Trajectory from collected traces."""
        if self._trajectory_built:
            return
        self._trajectory_built = True

        steps = [trace_to_step(t) for t in self._traces]
        traj = Trajectory(
            name="smolagents",
            steps=steps,
            reward=0.0,
            input=self._user_input,
            output=self._last_output,
            metadata={
                "source": "smolagents",
                "model": self._model_name,
                "num_llm_calls": len(self._traces),
            },
        )
        if self._trajectories:
            self._trajectories[-1] = traj
        else:
            self._trajectories.append(traj)

    def get_trajectory(self) -> Trajectory:
        """Return the trajectory from the most recent agent run.

        Raises:
            ValueError: If no LLM calls have been captured yet.
        """
        self._build_trajectories()
        if not self._trajectories:
            raise ValueError("No trajectories collected yet. Make sure the agent has completed at least one run.")
        return self._trajectories[-1]

    def get_trajectories(self) -> list[Trajectory]:
        """Return all collected trajectories."""
        self._build_trajectories()
        return list(self._trajectories)

    def get_traces(self) -> list[Trace]:
        """Return the raw ``Trace`` objects."""
        return list(self._traces)

    def clear(self) -> None:
        """Clear all collected trajectories and traces."""
        self._traces.clear()
        self._trajectories.clear()
        self._user_input = None
        self._last_output = None
        self._model_name = "unknown"
        self._trajectory_built = False
