"""Rllm observability plugin for Google ADK.

Subclasses ``google.adk.plugins.BasePlugin`` to capture the full agent
execution trace — invocations, agent spans, LLM calls, tool executions,
and raw events — and stream them to the Rllm backend via the async
:class:`~rllm_telemetry.exporter.RllmExporter`.

Every callback is wrapped with ``_safe_callback`` so that plugin errors
never crash the agent run.
"""

from __future__ import annotations

import functools
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.events.event import Event
from google.adk.models.llm_request import LlmRequest as AdkLlmRequest
from google.adk.models.llm_response import LlmResponse as AdkLlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.genai import types

from .config import RllmConfig
from .exporter import create_exporter
from .schemas import (
    AgentInfo,
    AgentSpanRecord,
    EventActionsData,
    EventRecord,
    GenerationConfig,
    InvocationRecord,
    LlmRequest,
    LlmResponseData,
    LlmSpanRecord,
    SessionRecord,
    ToolInfo,
    ToolSpanRecord,
    UsageMetadata,
)

if TYPE_CHECKING:
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.tools.tool_context import ToolContext

logger = logging.getLogger("rllm_telemetry.plugin")


# ---------------------------------------------------------------------------
# Decorator — swallow plugin errors so the agent keeps running
# ---------------------------------------------------------------------------


def _safe_callback(func):
    """Catch and log exceptions in plugin callbacks."""

    @functools.wraps(func)
    async def wrapper(self, **kwargs):
        try:
            return await func(self, **kwargs)
        except Exception:
            logger.exception(
                "Rllm telemetry plugin error in %s; skipping.",
                func.__name__,
            )
            return None

    return wrapper


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _serialize_content(
    content: types.Content | None,
    *,
    max_len: int = -1,
) -> dict[str, Any] | None:
    """Safely serialise ``types.Content`` to a JSON-compatible dict."""
    if content is None:
        return None
    try:
        data = content.model_dump(mode="json", exclude_none=True)
    except Exception:
        data = {"_raw": str(content)}
    if max_len > 0:
        data = _truncate_strings(data, max_len)
    return data


def _truncate_strings(obj: Any, max_len: int) -> Any:
    """Recursively truncate string values in dicts/lists."""
    if isinstance(obj, str):
        return obj[:max_len] + "...[TRUNCATED]" if len(obj) > max_len else obj
    if isinstance(obj, dict):
        return {k: _truncate_strings(v, max_len) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_truncate_strings(v, max_len) for v in obj]
    return obj


def _build_agent_info(agent: BaseAgent) -> AgentInfo:
    """Recursively build an ``AgentInfo`` tree from a ``BaseAgent``."""
    return AgentInfo(
        name=agent.name,
        description=getattr(agent, "description", None),
        type=type(agent).__name__,
        sub_agents=[_build_agent_info(sa) for sa in (agent.sub_agents or [])],
    )


def _extract_tool_infos(
    adk_request: AdkLlmRequest,
) -> list[ToolInfo] | None:
    """Extract tool names and descriptions from an LLM request."""
    if not adk_request.config or not adk_request.config.tools:
        return None
    infos: list[ToolInfo] = []
    for tool in adk_request.config.tools:
        if not isinstance(tool, types.Tool):
            continue
        for decl in tool.function_declarations or []:
            infos.append(ToolInfo(name=decl.name, description=getattr(decl, "description", None)))
    return infos or None


def _extract_generation_config(
    adk_request: AdkLlmRequest,
) -> GenerationConfig | None:
    """Pull relevant generation params from the request config."""
    cfg = adk_request.config
    if cfg is None:
        return None
    return GenerationConfig(
        temperature=getattr(cfg, "temperature", None),
        top_p=getattr(cfg, "top_p", None),
        top_k=getattr(cfg, "top_k", None),
        max_output_tokens=getattr(cfg, "max_output_tokens", None),
        stop_sequences=getattr(cfg, "stop_sequences", None),
        candidate_count=getattr(cfg, "candidate_count", None),
    )


def _safe_dict(obj: Any) -> dict[str, Any] | None:
    """Best-effort conversion of an arbitrary object to a JSON-safe dict."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    try:
        return obj.model_dump(mode="json", exclude_none=True)
    except Exception:
        pass
    try:
        return dict(obj)
    except Exception:
        return {"_raw": str(obj)}


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------


class RllmTelemetryPlugin(BasePlugin):
    """ADK plugin that streams full agent traces to the Rllm backend.

    Usage::

        from rllm_telemetry import RllmTelemetryPlugin

        plugin = RllmTelemetryPlugin(api_key="sk-...")
        runner.plugin_manager.register_plugin(plugin)
    """

    def __init__(
        self,
        api_key: str = "",
        endpoint: str = "http://localhost:8100/v1/traces",
        *,
        config: RllmConfig | None = None,
    ) -> None:
        super().__init__(name="rllm_telemetry")

        if config is not None:
            self._config = config
        else:
            self._config = RllmConfig(api_key=api_key, endpoint=endpoint)

        self._exporter = create_exporter(self._config)
        self._started = False

        # Track seen sessions so we emit SessionRecord only once
        self._seen_sessions: set[str] = set()

        # Open invocations: invocation_id → InvocationRecord
        self._invocations: dict[str, InvocationRecord] = {}

        # Open spans keyed for later lookup on the "after" callback.
        # Agent spans:  (invocation_id, agent_name, branch) → record
        self._agent_spans: dict[tuple[str, str, str | None], AgentSpanRecord] = {}
        # LLM spans:  id(callback_context) → record
        self._llm_spans: dict[int, LlmSpanRecord] = {}
        # Tool spans: id(tool_context) → record
        self._tool_spans: dict[int, ToolSpanRecord] = {}

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    async def _ensure_started(self) -> None:
        if not self._started:
            await self._exporter.start()
            self._started = True

    def _emit_session_once(self, ctx: InvocationContext) -> None:
        sid = ctx.session.id
        if sid in self._seen_sessions:
            return
        self._seen_sessions.add(sid)
        record = SessionRecord(
            session_id=sid,
            app_name=ctx.session.app_name,
            user_id=ctx.session.user_id,
            created_at=time.time(),
        )
        self._exporter.enqueue("session", record.model_dump(exclude_none=True))

    # ------------------------------------------------------------------
    # Invocation callbacks
    # ------------------------------------------------------------------

    @_safe_callback
    async def on_user_message_callback(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content,
    ) -> types.Content | None:
        await self._ensure_started()
        self._emit_session_once(invocation_context)

        inv_id = invocation_context.invocation_id
        # Store user message so we can attach it to the invocation record
        if inv_id in self._invocations:
            self._invocations[inv_id].user_message = _serialize_content(user_message, max_len=self._config.max_content_length)
        return None

    @_safe_callback
    async def before_run_callback(self, *, invocation_context: InvocationContext) -> types.Content | None:
        await self._ensure_started()
        self._emit_session_once(invocation_context)

        ctx = invocation_context
        record = InvocationRecord(
            invocation_id=ctx.invocation_id,
            session_id=ctx.session.id,
            app_name=ctx.session.app_name,
            user_id=ctx.session.user_id,
            root_agent=_build_agent_info(ctx.agent),
            started_at=time.time(),
        )
        self._invocations[ctx.invocation_id] = record
        self._exporter.enqueue("invocation.start", record.model_dump(exclude_none=True))
        return None

    @_safe_callback
    async def after_run_callback(self, *, invocation_context: InvocationContext) -> None:
        inv_id = invocation_context.invocation_id
        record = self._invocations.pop(inv_id, None)
        if record is None:
            return
        record.ended_at = time.time()
        record.duration_ms = (record.ended_at - record.started_at) * 1000
        self._exporter.enqueue("invocation.end", record.model_dump(exclude_none=True))

    # ------------------------------------------------------------------
    # Agent callbacks
    # ------------------------------------------------------------------

    @_safe_callback
    async def before_agent_callback(self, *, agent: BaseAgent, callback_context: CallbackContext) -> types.Content | None:
        ctx = callback_context._invocation_context
        record = AgentSpanRecord(
            span_id=str(uuid.uuid4()),
            invocation_id=ctx.invocation_id,
            session_id=ctx.session.id,
            agent_name=agent.name,
            agent_description=getattr(agent, "description", None),
            agent_type=type(agent).__name__,
            branch=ctx.branch,
            started_at=time.time(),
        )
        key = (ctx.invocation_id, agent.name, ctx.branch)
        self._agent_spans[key] = record
        self._exporter.enqueue("agent.start", record.model_dump(exclude_none=True))
        return None

    @_safe_callback
    async def after_agent_callback(self, *, agent: BaseAgent, callback_context: CallbackContext) -> types.Content | None:
        ctx = callback_context._invocation_context
        key = (ctx.invocation_id, agent.name, ctx.branch)
        record = self._agent_spans.pop(key, None)
        if record is None:
            return None
        record.ended_at = time.time()
        record.duration_ms = (record.ended_at - record.started_at) * 1000
        self._exporter.enqueue("agent.end", record.model_dump(exclude_none=True))
        return None

    # ------------------------------------------------------------------
    # Model callbacks
    # ------------------------------------------------------------------

    @_safe_callback
    async def before_model_callback(self, *, callback_context: CallbackContext, llm_request: AdkLlmRequest) -> AdkLlmResponse | None:
        ctx = callback_context._invocation_context

        request_data: LlmRequest | None = None
        if self._config.capture_content:
            max_len = self._config.max_content_length
            contents_serialized: list[dict[str, Any]] | None = None
            if llm_request.contents:
                contents_serialized = [_serialize_content(c, max_len=max_len) for c in llm_request.contents if c is not None]

            sys_instruction = None
            if llm_request.config and llm_request.config.system_instruction:
                si = llm_request.config.system_instruction
                sys_instruction = si if isinstance(si, str) else str(si)

            request_data = LlmRequest(
                model=llm_request.model,
                system_instruction=sys_instruction,
                contents=contents_serialized,
                tools=_extract_tool_infos(llm_request),
                generation_config=_extract_generation_config(llm_request),
            )
        else:
            request_data = LlmRequest(
                model=llm_request.model,
                tools=_extract_tool_infos(llm_request),
                generation_config=_extract_generation_config(llm_request),
            )

        record = LlmSpanRecord(
            span_id=str(uuid.uuid4()),
            invocation_id=ctx.invocation_id,
            session_id=ctx.session.id,
            agent_name=ctx.agent.name,
            request=request_data,
            started_at=time.time(),
        )
        self._llm_spans[id(callback_context)] = record
        self._exporter.enqueue("llm.start", record.model_dump(exclude_none=True))
        return None

    @_safe_callback
    async def after_model_callback(self, *, callback_context: CallbackContext, llm_response: AdkLlmResponse) -> AdkLlmResponse | None:
        record = self._llm_spans.pop(id(callback_context), None)
        if record is None:
            return None

        usage: UsageMetadata | None = None
        if llm_response.usage_metadata:
            um = llm_response.usage_metadata
            usage = UsageMetadata(
                input_tokens=getattr(um, "prompt_token_count", None),
                output_tokens=getattr(um, "candidates_token_count", None),
                total_tokens=getattr(um, "total_token_count", None),
                cached_content_token_count=getattr(um, "cached_content_token_count", None),
            )

        content_data: dict[str, Any] | None = None
        if self._config.capture_content:
            content_data = _serialize_content(llm_response.content, max_len=self._config.max_content_length)

        record.response = LlmResponseData(
            content=content_data,
            finish_reason=(str(llm_response.finish_reason) if llm_response.finish_reason else None),
            model_version=llm_response.model_version,
            error_code=llm_response.error_code,
            error_message=llm_response.error_message,
            usage=usage,
        )
        record.ended_at = time.time()
        record.duration_ms = (record.ended_at - record.started_at) * 1000

        # Update invocation-level aggregates
        ctx = callback_context._invocation_context
        inv = self._invocations.get(ctx.invocation_id)
        if inv is not None:
            inv.llm_call_count += 1
            if usage:
                inv.total_input_tokens += usage.input_tokens or 0
                inv.total_output_tokens += usage.output_tokens or 0
            if llm_response.error_code:
                inv.error_count += 1

        self._exporter.enqueue("llm.end", record.model_dump(exclude_none=True))
        return None

    @_safe_callback
    async def on_model_error_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: AdkLlmRequest,
        error: Exception,
    ) -> AdkLlmResponse | None:
        record = self._llm_spans.pop(id(callback_context), None)
        if record is not None:
            record.response = LlmResponseData(
                error_code=type(error).__name__,
                error_message=str(error),
            )
            record.ended_at = time.time()
            record.duration_ms = (record.ended_at - record.started_at) * 1000
            self._exporter.enqueue("llm.end", record.model_dump(exclude_none=True))

        ctx = callback_context._invocation_context
        inv = self._invocations.get(ctx.invocation_id)
        if inv is not None:
            inv.error_count += 1
            inv.llm_call_count += 1
        return None

    # ------------------------------------------------------------------
    # Tool callbacks
    # ------------------------------------------------------------------

    @_safe_callback
    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict | None:
        ctx = tool_context._invocation_context

        args_data: dict[str, Any] | None = None
        if self._config.capture_tools:
            args_data = _safe_dict(tool_args)
            if self._config.max_content_length > 0:
                args_data = _truncate_strings(args_data, self._config.max_content_length)

        record = ToolSpanRecord(
            span_id=str(uuid.uuid4()),
            invocation_id=ctx.invocation_id,
            session_id=ctx.session.id,
            agent_name=ctx.agent.name,
            tool_name=tool.name,
            tool_description=getattr(tool, "description", None),
            tool_type=type(tool).__name__,
            is_long_running=getattr(tool, "is_long_running", False),
            args=args_data,
            started_at=time.time(),
        )
        self._tool_spans[id(tool_context)] = record
        self._exporter.enqueue("tool.start", record.model_dump(exclude_none=True))
        return None

    @_safe_callback
    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict,
    ) -> dict | None:
        record = self._tool_spans.pop(id(tool_context), None)
        if record is None:
            return None

        if self._config.capture_tools:
            result_data = _safe_dict(result)
            if self._config.max_content_length > 0:
                result_data = _truncate_strings(result_data, self._config.max_content_length)
            record.result = result_data

        record.ended_at = time.time()
        record.duration_ms = (record.ended_at - record.started_at) * 1000

        ctx = tool_context._invocation_context
        inv = self._invocations.get(ctx.invocation_id)
        if inv is not None:
            inv.tool_call_count += 1

        self._exporter.enqueue("tool.end", record.model_dump(exclude_none=True))
        return None

    @_safe_callback
    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> dict | None:
        record = self._tool_spans.pop(id(tool_context), None)
        if record is not None:
            record.error = f"{type(error).__name__}: {error}"
            record.ended_at = time.time()
            record.duration_ms = (record.ended_at - record.started_at) * 1000
            self._exporter.enqueue("tool.end", record.model_dump(exclude_none=True))

        ctx = tool_context._invocation_context
        inv = self._invocations.get(ctx.invocation_id)
        if inv is not None:
            inv.error_count += 1
            inv.tool_call_count += 1
        return None

    # ------------------------------------------------------------------
    # Event callback
    # ------------------------------------------------------------------

    @_safe_callback
    async def on_event_callback(self, *, invocation_context: InvocationContext, event: Event) -> Event | None:
        actions_data: EventActionsData | None = None
        if event.actions:
            a = event.actions
            actions_data = EventActionsData(
                state_delta=a.state_delta if a.state_delta else None,
                artifact_delta=a.artifact_delta if a.artifact_delta else None,
                transfer_to_agent=a.transfer_to_agent,
                escalate=a.escalate,
                end_of_agent=a.end_of_agent,
            )

        content_data: dict[str, Any] | None = None
        if self._config.capture_content:
            content_data = _serialize_content(event.content, max_len=self._config.max_content_length)

        record = EventRecord(
            event_id=event.id,
            invocation_id=event.invocation_id,
            session_id=invocation_context.session.id,
            author=event.author,
            timestamp=event.timestamp,
            branch=event.branch,
            content=content_data,
            partial=event.partial,
            is_final_response=event.is_final_response(),
            actions=actions_data,
            error_code=event.error_code,
            error_message=event.error_message,
        )
        self._exporter.enqueue("event", record.model_dump(exclude_none=True))
        return None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Flush pending records and release HTTP resources."""
        await self._exporter.close()
