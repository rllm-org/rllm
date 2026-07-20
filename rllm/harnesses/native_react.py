"""Minimal native-tool ReAct loop for terminal tasks.

The policy runs on the host through the rLLM gateway while commands run in one
persistent bash process inside the task sandbox. Model-specific reasoning and
tool syntax is owned by the configured rLLM renderer.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shlex
import time
import uuid
from dataclasses import dataclass
from typing import Any

from rllm.env import env_float
from rllm.sandbox.protocol import Sandbox, SandboxCommandTimeout
from rllm.sandbox.sandboxed_flow import SandboxedAgentFlow
from rllm.types import AgentConfig, Episode, Task, TerminationReason, Trajectory

logger = logging.getLogger(__name__)


NATIVE_SYSTEM_PROMPT = """You are an expert software engineer solving a task inside a Linux container.

You have two tools available:
- **bash**: Execute a shell command and see the output. Use this to explore, edit files, compile, run programs, etc.
- **submit**: Mark the task as complete. Only call this when you are confident the task is fully solved.

Guidelines:
- Read the task description carefully before starting.
- Explore the environment first (ls, cat, pwd) to understand the setup.
- Work step by step. Run one command at a time and check the output before proceeding.
- If a command fails, analyze the error and try a different approach.
- When editing files, use heredocs (cat > file << 'EOF'), sed, or echo/printf. There is no interactive editor.
- Test your solution before submitting.
- When done, call the submit tool."""

NATIVE_USER_PROMPT = """Task:
{instruction}

Current directory:
{terminal_state}"""


NATIVE_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command inside the task container; returns combined stdout/stderr.",
            "parameters": {
                "properties": {
                    "command": {
                        "description": 'The shell command to execute, e.g. "ls -la /app".',
                        "title": "Command",
                        "type": "string",
                    }
                },
                "required": ["command"],
                "title": "bash_args",
                "type": "object",
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Submit the task as complete. Call this only once the task is fully solved.",
            "parameters": {
                "properties": {},
                "title": "submit_args",
                "type": "object",
                "additionalProperties": False,
                "required": [],
            },
            "strict": True,
        },
    },
]

_PARSE_ERROR_OBSERVATION = "Your last message did not contain a valid tool call. Call the `bash` or `submit` function using the provided tools."


def tool_result(tool_call_id: str, content: str) -> dict[str, str]:
    """Build a native OpenAI tool-result message."""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content,
    }


def initial_messages(instruction: str, terminal_state: str) -> list[dict[str, Any]]:
    """Build the initial terminal-agent prompt."""
    return [
        {"role": "system", "content": NATIVE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": NATIVE_USER_PROMPT.format(
                instruction=instruction,
                terminal_state=terminal_state,
            ),
        },
    ]


def preserve_assistant_message(message: Any) -> dict[str, Any]:
    """Round-trip every assistant field returned by the OpenAI-compatible SDK.

    Some backends still expose reasoning as ``reasoning`` while native
    renderers consume ``reasoning_content``.  Add the latter as an alias but do
    not remove or reconstruct any field from the original response.
    """
    if isinstance(message, dict):
        result = dict(message)
    elif hasattr(message, "model_dump"):
        # ``exclude_unset`` retains every field the provider actually sent,
        # including explicit nulls and Pydantic extra/vendor fields.  Using
        # ``exclude_none`` here would silently mutate the message boundary.
        result = message.model_dump(exclude_unset=True)
    else:
        raise TypeError(f"Unsupported assistant message type: {type(message).__name__}")

    reasoning = result.get("reasoning_content")
    if reasoning is None:
        reasoning = result.get("reasoning")
        if reasoning is None:
            reasoning = getattr(message, "reasoning_content", None)
        if reasoning is None:
            reasoning = getattr(message, "reasoning", None)
        if reasoning is not None:
            result["reasoning_content"] = reasoning
    return result


_TIMEOUT_STATUS_CODES = frozenset({408, 504, 524, 598, 599})
_TIMEOUT_ERROR_NAMES = frozenset(
    {
        "APITimeoutError",
        "ConnectTimeout",
        "PoolTimeout",
        "ReadTimeout",
        "WriteTimeout",
    }
)


def _is_timeout_exception(error: BaseException) -> bool:
    """Recognize timeout failures raised by model clients and sandbox backends."""
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, TimeoutError | SandboxCommandTimeout):
            return True
        if type(current).__name__ in _TIMEOUT_ERROR_NAMES:
            return True
        if getattr(current, "status_code", None) in _TIMEOUT_STATUS_CODES:
            return True
        message = str(current).lower()
        if any(marker in message for marker in ("timed out", "timeout exceeded", "deadline exceeded")):
            return True
        current = current.__cause__ or current.__context__
    return False


@dataclass(frozen=True)
class NativeToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    error: str | None = None


def parse_native_tool_calls(message: dict[str, Any]) -> list[NativeToolCall]:
    """Parse renderer-produced OpenAI tool calls without changing their order."""
    parsed: list[NativeToolCall] = []
    for index, tool_call in enumerate(message.get("tool_calls") or []):
        if hasattr(tool_call, "model_dump"):
            tool_call = tool_call.model_dump(exclude_none=True)
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") or tool_call
        if hasattr(function, "model_dump"):
            function = function.model_dump(exclude_none=True)
        if not isinstance(function, dict):
            continue

        name = function.get("name") or tool_call.get("name")
        call_id = str(tool_call.get("id") or f"call_{index}")
        raw_arguments = function.get("arguments", tool_call.get("arguments", {}))
        error = None
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments) if raw_arguments.strip() else {}
            except json.JSONDecodeError:
                arguments = {}
                error = "Tool arguments must be valid JSON."
        elif isinstance(raw_arguments, dict):
            arguments = raw_arguments
        else:
            arguments = {}
            error = "Tool arguments must be a JSON object."
        parsed.append(
            NativeToolCall(
                id=call_id,
                name=str(name or ""),
                arguments=arguments,
                error=error,
            )
        )
    return parsed


def limit_output_length(text: str, max_length: int = 15_000) -> str:
    """Truncate long terminal output while preserving its head and tail."""
    if len(text) <= max_length:
        return text
    half = max_length // 2
    elided = len(text) - max_length
    return f"{text[:half]}\n\n[... {elided} characters elided ...]\n\n{text[-half:]}"


@dataclass(frozen=True)
class ShellResult:
    output: str
    exit_code: int
    timed_out: bool = False


class PersistentBashSession:
    """Persistent non-interactive bash driven through sandbox ``exec`` calls.

    Each command body is staged in a file and sourced by one detached bash
    process.  Sourcing keeps cwd, exports, virtualenv activation, and shell
    functions across turns while redirecting command stdin away from the
    control FIFO.
    """

    def __init__(
        self,
        sandbox: Sandbox,
        *,
        user: str | None = None,
        max_buffer_size: int = 480 * 1024,
    ) -> None:
        self.sandbox = sandbox
        self.user = user
        self.max_buffer_size = max_buffer_size
        self.session_dir = f"/tmp/.rllm_native_react_{uuid.uuid4().hex[:12]}"
        self._sequence = 0
        self._started = False

    def start(self) -> None:
        directory = shlex.quote(self.session_dir)
        nonce = self.session_dir.rsplit("_", 1)[-1]
        reader = (
            f'__rllm_d={directory}; echo $$ > "$__rllm_d/pgid"; __rllm_last=0; '
            "while IFS= read -r __rllm_seq; do "
            "set +e; "
            'case "$__rllm_seq" in (""|*[!0-9]*) continue ;; esac; '
            'if [ "$__rllm_seq" -le "$__rllm_last" ] 2>/dev/null; then continue; fi; '
            '__rllm_last="$__rllm_seq"; '
            ': > "$__rllm_d/s$__rllm_seq"; '
            'source "$__rllm_d/c$__rllm_seq" < /dev/null > "$__rllm_d/o$__rllm_seq" 2>&1; '
            "__rllm_rc=$?; "
            'printf %s "$__rllm_rc" > "$__rllm_d/e$__rllm_seq"; '
            ': > "$__rllm_d/d$__rllm_seq"; '
            'done < "$__rllm_d/cmd"'
        )
        hold = f'D={directory}; exec -a rllm_native_react_hold_{nonce} sleep 2147483647 > "$D/cmd"'
        setup = (
            "set -e; "
            "command -v bash >/dev/null; command -v mkfifo >/dev/null; "
            "command -v setsid >/dev/null; command -v base64 >/dev/null; "
            f"rm -rf {directory}; mkdir -p {directory}; mkfifo {directory}/cmd; "
            f"bash -c {shlex.quote(hold)} >/dev/null 2>&1 & echo $! > {directory}/holdpid; "
            f"setsid bash -c {shlex.quote(reader)} </dev/null >/dev/null 2>&1 &"
        )
        self.sandbox.exec(setup, timeout=30, user=self.user)

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if self.is_alive():
                self._started = True
                return
            time.sleep(0.1)
        self.close()
        raise RuntimeError("native_react persistent bash did not start")

    def is_alive(self) -> bool:
        directory = shlex.quote(self.session_dir)
        check = (
            f'D={directory}; r=$(cat "$D/pgid" 2>/dev/null); '
            'h=$(cat "$D/holdpid" 2>/dev/null); '
            '[ -n "$r" ] && kill -0 "$r" 2>/dev/null && '
            '[ -n "$h" ] && kill -0 "$h" 2>/dev/null && echo ALIVE || true'
        )
        try:
            return "ALIVE" in self.sandbox.exec(check, timeout=10, user=self.user)
        except Exception:
            return False

    def run(self, command: str, timeout: float) -> ShellResult:
        if not self._started:
            raise RuntimeError("persistent bash has not been started")
        if not self.is_alive():
            raise RuntimeError("native_react persistent bash exited")

        self._sequence += 1
        sequence = self._sequence
        previous = sequence - 1
        directory = shlex.quote(self.session_dir)
        encoded = base64.b64encode((command if command.strip() else ":").encode()).decode("ascii")
        deadline_seconds = max(1, int(timeout))
        max_bytes = self.max_buffer_size + 1
        control = (
            f"D={directory}; s={sequence}; p={previous}; "
            'rm -f "$D/c$p" "$D/o$p" "$D/e$p" "$D/d$p" "$D/s$p" 2>/dev/null; '
            'rm -f "$D/s$s" "$D/o$s" "$D/e$s" "$D/d$s"; '
            f'printf %s {shlex.quote(encoded)} | base64 -d > "$D/c$s.tmp" && '
            'mv -f "$D/c$s.tmp" "$D/c$s"; '
            'printf "\\n%s\\n" "$s" > "$D/cmd"; '
            f"deadline=$(( $(date +%s) + {deadline_seconds} )); "
            "while :; do "
            '[ -e "$D/d$s" ] && { state=ok; break; }; '
            '[ "$(date +%s)" -ge "$deadline" ] && { state=timeout; break; }; '
            "sleep 0.1; done; "
            'printf "__RLLM_NATIVE_REACT_STATUS__ %s %s\\n" "$state" "$(cat "$D/e$s" 2>/dev/null)"; '
            f'head -c {max_bytes} "$D/o$s" 2>/dev/null; '
            'printf "\\n__RLLM_NATIVE_REACT_END__\\n"'
        )
        raw = self.sandbox.exec(control, timeout=timeout + 40, user=self.user)
        result = self._parse_result(raw)
        if result.timed_out:
            self.close()
        return result

    def _parse_result(self, raw: str) -> ShellResult:
        status_marker = "__RLLM_NATIVE_REACT_STATUS__"
        end_marker = "\n__RLLM_NATIVE_REACT_END__"
        status_start = raw.find(status_marker)
        output_end = raw.rfind(end_marker)
        if status_start < 0 or output_end < 0 or status_start >= output_end:
            raise RuntimeError("native_react persistent bash returned a truncated response")
        status_end = raw.find("\n", status_start)
        if status_end < 0 or status_end > output_end:
            raise RuntimeError("native_react persistent bash returned a malformed response")

        status_parts = raw[status_start:status_end].split()
        state = status_parts[1] if len(status_parts) > 1 else "timeout"
        try:
            exit_code = int(status_parts[2]) if len(status_parts) > 2 else -1
        except ValueError:
            exit_code = -1
        output = raw[status_end + 1 : output_end]
        if len(output) > self.max_buffer_size:
            output = output[: self.max_buffer_size]
            exit_code = -1
        return ShellResult(output=output, exit_code=exit_code, timed_out=state == "timeout")

    def close(self) -> None:
        directory = shlex.quote(self.session_dir)
        cleanup = (
            f'D={directory}; p=$(cat "$D/pgid" 2>/dev/null); '
            'h=$(cat "$D/holdpid" 2>/dev/null); '
            '[ -n "$p" ] && kill -KILL -"$p" 2>/dev/null || true; '
            '[ -n "$p" ] && kill -KILL "$p" 2>/dev/null || true; '
            '[ -n "$h" ] && kill -KILL "$h" 2>/dev/null || true; '
            'rm -rf "$D" 2>/dev/null; true'
        )
        try:
            self.sandbox.exec(cleanup, timeout=10, user=self.user)
        except Exception:
            logger.debug("native_react persistent bash cleanup failed", exc_info=True)
        self._started = False


class NativeReactHarness(SandboxedAgentFlow):
    """Minimal native-tool ReAct loop without compaction."""

    name = "native_react"
    sandbox_backend = "docker"
    max_turns: int | None = 100
    command_timeout: float = 300.0
    max_output_length: int = 15_000
    max_parse_retries: int = 5
    run_timeout: float = 3600.0
    run_timeout_is_cap: bool = False

    def __init__(self, **kwargs: Any) -> None:
        values = dict(kwargs)
        values.setdefault("run_timeout", env_float("RLLM_HARNESS_RUN_TIMEOUT_S", self.run_timeout))
        values.setdefault("run_timeout_is_cap", "RLLM_HARNESS_RUN_TIMEOUT_S" in os.environ)
        super().__init__(**values)

    def configure(self, overrides: dict) -> dict:
        leftovers = super().configure(overrides)
        timeout = leftovers.pop("agent_timeout", None)
        if timeout is not None:
            self.run_timeout = float(timeout)
            self.run_timeout_is_cap = True
        return leftovers

    def _effective_timeout(self, task: Task) -> float:
        per_task = task.metadata.get("agent_timeout")
        if per_task is None:
            return float(self.run_timeout)
        if self.run_timeout_is_cap:
            return min(float(per_task), float(self.run_timeout))
        return float(per_task)

    def _outcome_episode(
        self,
        task: Task,
        config: AgentConfig,
        *,
        termination_reason: TerminationReason | None = None,
        turns: int = 0,
        parse_errors: int = 0,
        error: dict[str, str] | None = None,
    ) -> Episode:
        metadata: dict[str, Any] = {"native_react": {"turns": turns, "parse_errors": parse_errors}}
        if error is not None:
            metadata["error"] = error
        return Episode(
            id=config.session_uid,
            task=task.id,
            termination_reason=termination_reason,
            trajectories=[
                Trajectory(
                    uid=config.session_uid,
                    name=self.name,
                    task=task.id,
                    steps=[],
                )
            ],
            metadata=metadata,
        )

    def _timeout_episode(
        self,
        task: Task,
        config: AgentConfig,
        *,
        message: str,
        turns: int = 0,
        parse_errors: int = 0,
        cause: BaseException | None = None,
    ) -> Episode:
        if cause is not None:
            message = f"{message} ({type(cause).__name__}: {cause})"
        return self._outcome_episode(
            task,
            config,
            termination_reason=TerminationReason.TIMEOUT,
            turns=turns,
            parse_errors=parse_errors,
            error={"error_type": "AgentTimeoutError", "message": message},
        )

    def run(self, task: Task, config: AgentConfig, *, env: Sandbox) -> Episode:
        from openai import OpenAI

        client = OpenAI(base_url=config.base_url, api_key="EMPTY", max_retries=0)
        agent_user = task.metadata.get("agent_user")
        workdir = str(task.metadata.get("workdir") or "/app")
        task_max_turns = task.metadata.get("rllm", {}).get("max_turns")
        max_turns = int(task_max_turns) if task_max_turns else self.max_turns
        agent_timeout = self._effective_timeout(task)
        shell = PersistentBashSession(env, user=agent_user)
        turns = 0
        parse_errors = 0

        try:
            shell.start()
            deadline = time.monotonic() + agent_timeout
            initial = shell.run(
                f"cd {shlex.quote(workdir)} && pwd && ls -la",
                timeout=min(30.0, max(1.0, deadline - time.monotonic())),
            )
            if initial.timed_out:
                return self._timeout_episode(
                    task,
                    config,
                    message=f"Agent execution timed out after {agent_timeout:g}s",
                )
            messages = initial_messages(str(task.instruction), initial.output)

            while max_turns is None or turns < max_turns:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return self._timeout_episode(
                        task,
                        config,
                        message=f"Agent execution timed out after {agent_timeout:g}s",
                        turns=turns,
                        parse_errors=parse_errors,
                    )

                response = client.chat.completions.create(
                    model=config.model,
                    messages=messages,
                    tools=NATIVE_TOOL_SCHEMAS,
                    timeout=remaining,
                )
                assistant = preserve_assistant_message(response.choices[0].message)
                messages.append(assistant)
                turns += 1

                tool_calls = parse_native_tool_calls(assistant)
                if not tool_calls:
                    parse_errors += 1
                    messages.append({"role": "user", "content": _PARSE_ERROR_OBSERVATION})
                    if parse_errors > self.max_parse_retries:
                        break
                    continue

                for tool_call in tool_calls:
                    if tool_call.error:
                        messages.append(tool_result(tool_call.id, tool_call.error))
                        continue
                    if tool_call.name == "submit":
                        return self._outcome_episode(
                            task,
                            config,
                            turns=turns,
                            parse_errors=parse_errors,
                        )
                    if tool_call.name != "bash":
                        messages.append(tool_result(tool_call.id, f"Unknown tool: {tool_call.name}"))
                        continue

                    command = tool_call.arguments.get("command")
                    if not isinstance(command, str) or not command:
                        messages.append(tool_result(tool_call.id, "The `bash` tool requires a non-empty `command` string."))
                        continue
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return self._timeout_episode(
                            task,
                            config,
                            message=f"Agent execution timed out after {agent_timeout:g}s",
                            turns=turns,
                            parse_errors=parse_errors,
                        )
                    command_timeout = min(self.command_timeout, remaining)
                    result = shell.run(command, timeout=command_timeout)
                    if result.timed_out:
                        return self._timeout_episode(
                            task,
                            config,
                            message=f"Shell command timed out after {command_timeout:g}s",
                            turns=turns,
                            parse_errors=parse_errors,
                        )
                    messages.append(
                        tool_result(
                            tool_call.id,
                            limit_output_length(result.output, self.max_output_length),
                        )
                    )

            return self._outcome_episode(
                task,
                config,
                termination_reason=(TerminationReason.MAX_TURNS_EXCEEDED if max_turns is not None and turns >= max_turns else None),
                turns=turns,
                parse_errors=parse_errors,
            )
        except Exception as error:
            if _is_timeout_exception(error):
                return self._timeout_episode(
                    task,
                    config,
                    message=f"Agent execution timed out after {agent_timeout:g}s",
                    turns=turns,
                    parse_errors=parse_errors,
                    cause=error,
                )
            raise
        finally:
            shell.close()
            client.close()


__all__ = [
    "NATIVE_SYSTEM_PROMPT",
    "NATIVE_TOOL_SCHEMAS",
    "NATIVE_USER_PROMPT",
    "NativeToolCall",
    "NativeReactHarness",
    "PersistentBashSession",
    "initial_messages",
    "limit_output_length",
    "parse_native_tool_calls",
    "preserve_assistant_message",
    "tool_result",
]
