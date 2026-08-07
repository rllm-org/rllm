"""AgentFlow adapter for Scale's pinned MCP-Atlas TypeScript harness."""

from __future__ import annotations

import ast
import json
import threading
from typing import Any

import requests

from rllm.integrations.mcp_atlas.constants import (
    DATASET_REVISION,
    DATASET_SHA256,
    DEFAULT_CONCURRENCY,
    DEFAULT_SERVERS,
    GATEWAY_API_KEY,
    IMAGE,
    MAX_TOOL_CALLS,
    MAX_TURNS,
    TASK_TIMEOUT_SECONDS,
)
from rllm.integrations.mcp_atlas.evaluator import resolve_judge_settings
from rllm.integrations.mcp_atlas.service import MCPAtlasServiceManager
from rllm.types import AgentConfig, Episode, Task, Trajectory
from rllm.workflows.workflow import TerminationReason


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
    return []


def _enabled_tools(value: Any) -> list[str]:
    names: list[str] = []
    for item in _as_list(value):
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict) and item.get("name"):
            names.append(str(item["name"]))
    return names


class MCPAtlasHarness:
    """Delegate the agent loop to the official MCP-Atlas service."""

    needs_env = False
    max_concurrent = DEFAULT_CONCURRENCY

    def __init__(self) -> None:
        self.image = IMAGE
        self.env_file: str | None = None
        self.preflight = "strict"
        self.task_filter = "all"
        self.external_harness_url: str | None = None
        self.startup_timeout = 180.0
        self.health_timeout = 30.0
        self.task_timeout = float(TASK_TIMEOUT_SECONDS)
        self.max_turns = MAX_TURNS
        self.max_tool_calls = MAX_TOOL_CALLS
        self._manager: MCPAtlasServiceManager | None = None
        self._last_manager_metadata: dict[str, Any] = {}
        self._context_metadata: dict[str, Any] = {}
        self._has_run_metadata = False
        self._task_filter_metadata: dict[str, Any] = {}
        self._lock = threading.Lock()

    def configure(self, overrides: dict) -> dict:
        remaining = dict(overrides)
        aliases = {"harness_url": "external_harness_url"}
        fields = {
            "image",
            "env_file",
            "preflight",
            "task_filter",
            "external_harness_url",
            "startup_timeout",
            "health_timeout",
            "task_timeout",
            "max_turns",
            "max_tool_calls",
        }
        for key in list(remaining):
            attr = aliases.get(key, key)
            if attr in fields:
                setattr(self, attr, remaining.pop(key))
        if self.task_filter == "no_credentials":
            self.task_filter = "default_servers"
        if self.task_filter not in {"all", "default_servers"}:
            raise ValueError("MCP-Atlas task_filter must be 'all' or 'default_servers'")
        self.max_concurrent = int(remaining.pop("sandbox_concurrency", DEFAULT_CONCURRENCY))
        return remaining

    def filter_eval_tasks(self, tasks: list[Task]) -> list[Task]:
        """Select a configured MCP-Atlas cohort before lifecycle/model calls."""
        source_count = len(tasks)
        if self.task_filter == "all":
            selected = list(tasks)
            excluded_servers: set[str] = set()
        else:
            selected = []
            excluded_servers = set()
            for task in tasks:
                servers = {tool.split("_", 1)[0] for tool in _enabled_tools(getattr(task, "metadata", {}).get("ENABLED_TOOLS"))}
                blocked = servers - DEFAULT_SERVERS
                if servers and not blocked:
                    selected.append(task)
                else:
                    excluded_servers.update(blocked)
            if not selected:
                raise RuntimeError("MCP-Atlas default_servers filter selected no tasks")

        self._task_filter_metadata = {
            "name": self.task_filter,
            "source_task_count": source_count,
            "selected_task_count": len(selected),
            "excluded_task_count": source_count - len(selected),
        }
        if self.task_filter == "default_servers":
            self._task_filter_metadata.update(
                {
                    "allowed_servers": sorted(DEFAULT_SERVERS),
                    "excluded_servers": sorted(excluded_servers),
                }
            )
        return selected

    def eval_task_filter_metadata(self) -> dict[str, Any]:
        return dict(self._task_filter_metadata)

    def prepare_eval(self, context) -> None:
        with self._lock:
            if self._manager is not None:
                return
            run_dir = getattr(context, "run_dir", None)
            tasks = tuple(getattr(context, "tasks", ()))
            if self.preflight == "smoke" and len(tasks) >= 500:
                raise RuntimeError("MCP-Atlas smoke preflight requires an explicitly selected public-task subset")
            required_servers = {tool.split("_", 1)[0] for task in tasks for tool in _enabled_tools(getattr(task, "metadata", {}).get("ENABLED_TOOLS"))}
            if not required_servers:
                raise RuntimeError("MCP-Atlas selected tasks do not declare any MCP servers")
            self._context_metadata = {
                "model": str(getattr(context, "model", "")),
                "task_count": int(getattr(context, "task_count", len(tasks))),
                "pending_task_count": len(tasks),
                "concurrency": int(getattr(context, "concurrency", self.max_concurrent)),
                "sampling_params": dict(getattr(context, "sampling_params", {}) or {}),
            }
            self._has_run_metadata = True
            manager = MCPAtlasServiceManager(
                image=str(self.image),
                env_file=self.env_file,
                preflight=str(self.preflight),
                external_harness_url=self.external_harness_url,
                startup_timeout=float(self.startup_timeout),
                health_timeout=float(self.health_timeout),
                run_dir=run_dir,
                required_servers=required_servers,
            )
            self._manager = manager
            try:
                manager.start()
            except BaseException:
                self._last_manager_metadata = manager.metadata()
                manager.stop()
                self._manager = None
                raise

    def finalize_eval(self, context=None, error: BaseException | None = None) -> None:  # noqa: ARG002
        with self._lock:
            if self._manager is not None:
                self._last_manager_metadata = self._manager.metadata()
                self._manager.stop()
                self._manager = None

    def eval_run_metadata(self) -> dict[str, Any]:
        if not self._has_run_metadata:
            return {}
        manager = self._manager
        metadata = manager.metadata() if manager is not None else dict(self._last_manager_metadata)
        judge = resolve_judge_settings(require_key=False)
        metadata.update(
            {
                **self._context_metadata,
                "dataset_revision": DATASET_REVISION,
                "dataset_sha256": DATASET_SHA256,
                "judge_model": judge.model,
                "judge_provider": judge.provider,
                "judge_base_url": judge.base_url,
                "judge_temperature": 0.0,
                "pass_coverage_threshold": 0.75,
                "max_turns": int(self.max_turns),
                "max_tool_calls": int(self.max_tool_calls),
                "task_timeout_seconds": float(self.task_timeout),
                "system_prompt": None,
                "task_filter": dict(self._task_filter_metadata),
                "tool_output_cap": None,
                "context_window_management": None,
            }
        )
        return {"mcp_atlas": metadata}

    def _harness_url(self) -> str:
        if self._manager is None or not self._manager.harness_url:
            raise RuntimeError("MCP-Atlas harness was not prepared by the eval runner")
        return self._manager.harness_url

    def run(self, task: Task, config: AgentConfig) -> Episode:
        metadata = task.metadata
        task_id = str(metadata.get("TASK") or task.id)
        prompt = str(metadata.get("PROMPT") or task.instruction)
        tools = _enabled_tools(metadata.get("ENABLED_TOOLS"))
        if not tools:
            raise ValueError(f"MCP-Atlas task {task_id} has no enabled tools")
        payload = {
            "task_id": task_id,
            # The official harness strips Fireworks-incompatible JSON-schema
            # fields based on this prefix; rLLM's gateway still pins the
            # upstream request to the configured model alias.
            "model": f"fireworks_ai/{config.model}" if config.model.startswith("accounts/fireworks/") else config.model,
            "messages": [{"role": "user", "content": prompt}],
            "enabledTools": tools,
            "image": self.image,
            "tags": {"task_id": task_id, "rllm_session_uid": config.session_uid},
            "max_turns": int(self.max_turns),
            "max_tool_calls": int(self.max_tool_calls),
            # Pinned upstream exposes this field. rLLM's recorded compatibility
            # patch makes it authoritative for every completion request.
            "llm_base_url": config.base_url,
            "extra_llm_params": {"api_key": GATEWAY_API_KEY},
        }
        try:
            response = requests.post(
                f"{self._harness_url()}/v2/mcp_eval/run_agent",
                json=payload,
                timeout=float(self.task_timeout),
            )
        except requests.Timeout:
            return Episode(
                id=config.session_uid,
                task=task,
                termination_reason=TerminationReason.TIMEOUT,
                trajectories=[Trajectory(uid=config.session_uid, name="mcp-atlas", task=task_id, output="")],
                artifacts={
                    "answer": "",
                    "raw_conversation_history": [],
                    "raw_harness_events": [],
                    "mcp_atlas": {
                        "task_id": task_id,
                        "enabled_tools": tools,
                        "turn_count": 0,
                        "tool_call_count": 0,
                        "timed_out": True,
                        "harness_errors": [{"reason": "task_timeout", "timeout_seconds": float(self.task_timeout)}],
                        "tool_trajectory": [],
                    },
                },
                metadata={"error": {"message": f"MCP-Atlas task timed out after {self.task_timeout:g}s"}},
            )
        if response.status_code != 200:
            body = response.text[:500]
            raise RuntimeError(f"MCP-Atlas harness HTTP {response.status_code}: {body}")
        try:
            events = response.json()
        except ValueError as exc:
            raise RuntimeError("MCP-Atlas harness returned invalid JSON") from exc
        if not isinstance(events, list):
            raise RuntimeError("MCP-Atlas harness response must be an event list")

        conversation = [event.get("data") for event in events if event.get("type") == "message" and isinstance(event.get("data"), dict)]
        errors = [event.get("data") for event in events if event.get("type") == "error"]
        final_answer = ""
        for message in reversed(conversation):
            if message.get("role") == "assistant" and message.get("content"):
                final_answer = str(message["content"])
                break
        tool_calls = sum(len(message.get("tool_calls") or []) for message in conversation if message.get("role") == "assistant")
        assistant_turns = sum(1 for message in conversation if message.get("role") == "assistant")
        tool_trajectory = [message for message in conversation if message.get("role") == "tool" or (message.get("role") == "assistant" and message.get("tool_calls"))]
        terminal_error = bool(errors)

        trajectory = Trajectory(
            uid=config.session_uid,
            name="mcp-atlas",
            task=task_id,
            output=final_answer,
            metadata={"harness_errors": errors},
        )
        return Episode(
            id=config.session_uid,
            task=task,
            termination_reason=TerminationReason.ERROR if terminal_error else None,
            trajectories=[trajectory],
            artifacts={
                "answer": final_answer,
                "raw_conversation_history": conversation,
                "raw_harness_events": events,
                "mcp_atlas": {
                    "task_id": task_id,
                    "enabled_tools": tools,
                    "turn_count": assistant_turns,
                    "tool_call_count": tool_calls,
                    "harness_errors": errors,
                    "timed_out": False,
                    "tool_trajectory": tool_trajectory,
                },
            },
            metadata={"error": {"message": f"MCP-Atlas harness terminated: {errors!r}"}} if terminal_error else {},
        )


__all__ = ["MCPAtlasHarness"]
