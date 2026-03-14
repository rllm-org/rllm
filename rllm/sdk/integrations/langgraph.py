"""LangGraph / LangChain integration for rLLM trajectory tracking.

Provides ``RLLMTrajectoryCallbackHandler``, a LangChain
``BaseCallbackHandler`` that captures all LLM calls during LangGraph
agent execution and builds rLLM ``Trajectory`` objects for SFT
distillation and RL training.

Usage::

    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from rllm.sdk.integrations.langgraph import RLLMTrajectoryCallbackHandler

    cb = RLLMTrajectoryCallbackHandler()
    llm = ChatOpenAI(model="gpt-4o")
    agent = create_react_agent(llm, tools=[])

    result = agent.invoke(
        {"messages": [("user", "What is 15 * 7 + 23?")]},
        config={"callbacks": [cb]},
    )

    traj = cb.get_trajectory()
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
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    BaseCallbackHandler = object  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LangChain <-> OpenAI format converters
# ---------------------------------------------------------------------------


def _safe_json(obj: Any) -> str:
    """Serialize *obj* to a JSON string, falling back to ``str()``."""
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(obj)


def _langchain_messages_to_openai(messages: list[list]) -> list[dict]:
    """Convert LangChain BaseMessage lists to OpenAI Chat Completions format.

    ``on_chat_model_start`` receives ``messages`` as a list of lists (one
    list per batch item).  We flatten the first batch.
    """
    if not messages:
        return []
    # Take the first batch
    batch = messages[0] if messages else []
    return [_langchain_message_to_openai(m) for m in batch]


def _langchain_message_to_openai(msg: Any) -> dict:
    """Convert a single LangChain BaseMessage to an OpenAI message dict."""
    if isinstance(msg, dict):
        return msg

    role = "user"
    content = ""
    tool_calls_list = None
    tool_call_id = None

    if _LANGCHAIN_AVAILABLE:
        if isinstance(msg, SystemMessage):
            role = "system"
        elif isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
            tc = getattr(msg, "tool_calls", None)
            if tc:
                tool_calls_list = []
                for call in tc:
                    if isinstance(call, dict):
                        tool_calls_list.append(
                            {
                                "id": call.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                                "type": "function",
                                "function": {
                                    "name": call.get("name", ""),
                                    "arguments": _safe_json(call.get("args", {})),
                                },
                            }
                        )
        elif isinstance(msg, ToolMessage):
            role = "tool"
            tool_call_id = getattr(msg, "tool_call_id", None)
    else:
        role = getattr(msg, "type", "user")
        if role == "human":
            role = "user"
        elif role == "ai":
            role = "assistant"

    raw_content = getattr(msg, "content", "")
    if isinstance(raw_content, str):
        content = raw_content
    elif isinstance(raw_content, list):
        text_parts = []
        for block in raw_content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        content = "\n".join(text_parts) if text_parts else str(raw_content)
    else:
        content = str(raw_content) if raw_content else ""

    oai_msg: dict[str, Any] = {"role": role, "content": content}
    if tool_calls_list:
        oai_msg["tool_calls"] = tool_calls_list
    if tool_call_id:
        oai_msg["tool_call_id"] = tool_call_id

    return oai_msg


def _langchain_generation_to_openai(generation: Any) -> dict:
    """Convert a LangChain ChatGeneration to an OpenAI assistant message dict."""
    msg: dict[str, Any] = {"role": "assistant", "content": None}

    message = getattr(generation, "message", None)
    if message is None:
        text = getattr(generation, "text", "")
        msg["content"] = text
        return msg

    return _langchain_message_to_openai(message)


def _extract_finish_reason(generation: Any) -> str:
    """Infer finish reason from a LangChain generation."""
    message = getattr(generation, "message", None)
    if message is not None:
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            return "tool_calls"
        # Check response_metadata
        meta = getattr(message, "response_metadata", {})
        if isinstance(meta, dict):
            fr = meta.get("finish_reason")
            if fr:
                return fr
    return "stop"


def _extract_usage(result: Any) -> dict[str, int]:
    """Extract token usage from an LLMResult."""
    usage = {"prompt": 0, "completion": 0, "total": 0}
    llm_output = getattr(result, "llm_output", None)
    if isinstance(llm_output, dict):
        token_usage = llm_output.get("token_usage", {})
        if isinstance(token_usage, dict):
            usage["prompt"] = token_usage.get("prompt_tokens", 0) or 0
            usage["completion"] = token_usage.get("completion_tokens", 0) or 0
            usage["total"] = token_usage.get("total_tokens", 0) or 0
    if usage["total"] == 0:
        usage["total"] = usage["prompt"] + usage["completion"]
    return usage


def _extract_model_name(kwargs: dict) -> str:
    """Extract model name from callback kwargs."""
    invocation = kwargs.get("invocation_params", {})
    if isinstance(invocation, dict):
        model = invocation.get("model_name") or invocation.get("model")
        if model:
            return model
    return "unknown"


# ---------------------------------------------------------------------------
# Callback handler
# ---------------------------------------------------------------------------


class RLLMTrajectoryCallbackHandler(BaseCallbackHandler):  # type: ignore[misc]
    """LangChain callback handler that captures LLM calls and builds rLLM Trajectory objects.

    Each chat model invocation is recorded as an rLLM ``Trace``, with
    LangChain message types automatically converted to the OpenAI Chat
    Completions format that rLLM's training pipeline expects.

    After the agent finishes, call :py:meth:`get_trajectory` to obtain a
    ``Trajectory`` ready for reward assignment and training.

    Example::

        from langchain_openai import ChatOpenAI
        from rllm.sdk.integrations.langgraph import RLLMTrajectoryCallbackHandler

        cb = RLLMTrajectoryCallbackHandler()
        llm = ChatOpenAI(model="gpt-4o")
        result = llm.invoke("Hello", config={"callbacks": [cb]})

        traj = cb.get_trajectory()
    """

    name = "rllm_trajectory"

    def __init__(self) -> None:
        if not _LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required for RLLMTrajectoryCallbackHandler. Install it with: pip install langchain-core langchain-openai")
        super().__init__()
        self._traces: list[Trace] = []
        self._trajectories: list[Trajectory] = []
        self._pending_messages: list[dict] | None = None
        self._request_start_time: float = 0.0
        self._user_input: dict | None = None
        self._last_output: Any = None
        self._model_name: str = "unknown"
        self._trajectory_built: bool = False

    # ---- LLM lifecycle --------------------------------------------------------

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """Snapshot the input messages and start the timer."""
        self._pending_messages = _langchain_messages_to_openai(messages)
        self._request_start_time = time.perf_counter()
        self._model_name = _extract_model_name(kwargs)
        self._trajectory_built = False

        # Capture user input from the first user message
        if self._user_input is None and self._pending_messages:
            for m in self._pending_messages:
                if m.get("role") == "user":
                    self._user_input = {"message": m.get("content", "")}
                    break

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """Pair the stored messages with the response and create a Trace."""
        if self._pending_messages is None:
            return

        latency_ms = (time.perf_counter() - self._request_start_time) * 1000
        pending_messages = self._pending_messages
        self._pending_messages = None

        # Extract the first generation
        generations = getattr(response, "generations", None)
        if not generations or not generations[0]:
            return

        generation = generations[0][0]
        response_msg = _langchain_generation_to_openai(generation)
        finish_reason = _extract_finish_reason(generation)
        tokens = _extract_usage(response)

        if response_msg.get("content"):
            self._last_output = response_msg["content"]

        model = self._model_name

        trace = Trace(
            trace_id=f"lg_{uuid.uuid4().hex[:16]}",
            session_name="langgraph",
            name=f"langgraph/{model}",
            input=LLMInput(
                messages=pending_messages,
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
            metadata={"source": "langgraph"},
            timestamp=time.time(),
        )

        self._traces.append(trace)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """Discard pending messages on error."""
        self._pending_messages = None

    # ---- Tool lifecycle (metadata only) ---------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        pass

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """Annotate the most recent trace with tool execution metadata."""
        if self._traces:
            last = self._traces[-1]
            tool_name = kwargs.get("name", "unknown")
            last.metadata.setdefault("tool_executions", []).append(
                {
                    "tool_name": tool_name,
                    "tool_result": str(output)[:500],
                }
            )

    # ---- internal helpers -----------------------------------------------------

    def _build_trajectories(self) -> None:
        """Build Trajectory from collected traces."""
        if self._trajectory_built:
            return
        self._trajectory_built = True

        steps = [trace_to_step(t) for t in self._traces]
        traj = Trajectory(
            name="langgraph",
            steps=steps,
            reward=0.0,
            input=self._user_input,
            output=self._last_output,
            metadata={
                "source": "langgraph",
                "model": self._model_name,
                "num_llm_calls": len(self._traces),
            },
        )
        if self._trajectories:
            self._trajectories[-1] = traj
        else:
            self._trajectories.append(traj)

    # ---- public API -----------------------------------------------------------

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
