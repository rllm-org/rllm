"""Minimal native-tool ReAct loop for terminal tasks.

The policy runs on the host through the rLLM gateway while commands run in one
persistent bash process inside the task sandbox. Model-specific reasoning and
tool syntax is owned by the configured rLLM renderer.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import math
import os
import re
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


class NativeModelResponseError(RuntimeError):
    """Malformed/error gateway response that should use model-error retries."""

    _rllm_termination_reason = TerminationReason.MODEL_ERROR

    def __init__(self, message: str, *, body: Any = None) -> None:
        super().__init__(message)
        self.body = body


def _response_payload(response: Any) -> Any:
    """Extract an OpenAI response payload for error classification."""
    if isinstance(response, dict):
        return response
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(exclude_none=True)
        except Exception:
            pass
    return {"response_type": type(response).__name__, "response": str(response)}


def _invalid_model_response(response: Any, message: str) -> NativeModelResponseError:
    payload = _response_payload(response)
    detail = json.dumps(payload, default=str)
    if len(detail) > 2_000:
        detail = f"{detail[:2_000]}..."
    return NativeModelResponseError(f"{message}: {detail}", body=payload)


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
        parts = [str(current)]
        body = getattr(current, "body", None)
        if body is not None:
            parts.append(json.dumps(body, default=str))
        message = " ".join(parts).lower()
        if any(marker in message for marker in ("timed out", "timeout exceeded", "deadline exceeded", "gateway_upstream_timeout")):
            return True
        current = current.__cause__ or current.__context__
    return False


def _is_context_length_exception(error: BaseException) -> bool:
    """Recognize the gateway's OpenAI-style prompt-window rejection."""
    current: BaseException | None = error
    seen: set[int] = set()
    markers = (
        "context_length_exceeded",
        "max_prompt_length_exceeded",
        "maximum context length",
        "exceeded the model's prompt window",
    )
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts = [str(current)]
        body = getattr(current, "body", None)
        if body is not None:
            parts.append(json.dumps(body, default=str))
        if any(marker in " ".join(parts).lower() for marker in markers):
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
    recovered: bool = False
    recovery_reason: str | None = None
    truncated: bool = False
    capture_error: str | None = None


@dataclass(frozen=True)
class ToolNotice:
    """One machine-classifiable fact attached to a tool observation."""

    code: str
    message: str


@dataclass(frozen=True)
class ToolObservation:
    """Transport-neutral tool output plus structured execution facts.

    The OpenAI wire protocol still receives ordinary, model-friendly text.
    Keeping notice codes internally gives the harness one extensible path for
    shell outcomes, validation failures, and protocol feedback without making
    prompt construction depend on renderer-specific markup.
    """

    output: str = ""
    notices: tuple[ToolNotice, ...] = ()


def render_tool_observation(observation: ToolObservation, *, max_length: int | None = None) -> str:
    """Render a typed observation once at the chat-protocol boundary."""
    rendered = observation.output
    for notice in observation.notices:
        separator = "\n" if rendered and not rendered.endswith("\n") else ""
        rendered = f"{rendered}{separator}[{notice.message}]"
    if max_length is not None:
        rendered = limit_output_length(rendered, max_length)
    return rendered


def observation_message(
    observation: ToolObservation,
    *,
    tool_call_id: str | None = None,
    max_length: int | None = None,
) -> dict[str, str]:
    """Build either a real tool result or standard user protocol feedback."""
    message = {
        "role": "tool" if tool_call_id is not None else "user",
        "content": render_tool_observation(observation, max_length=max_length),
    }
    if tool_call_id is not None:
        message["tool_call_id"] = tool_call_id
    return message


def tool_result(tool_call_id: str, content: str | ToolObservation) -> dict[str, str]:
    """Build a native OpenAI tool-result message through the observation path."""
    observation = content if isinstance(content, ToolObservation) else ToolObservation(output=content)
    return observation_message(observation, tool_call_id=tool_call_id)


def shell_result_observation(result: ShellResult, *, command_timeout: float) -> ToolObservation:
    """Convert a shell outcome into output and independently composable facts."""
    notices: list[ToolNotice] = []
    if result.timed_out:
        notices.append(
            ToolNotice(
                code="command_timeout",
                message=(f"Command timed out after {command_timeout:g}s. The terminal was automatically restarted; filesystem changes were preserved, but shell-local state was lost."),
            )
        )
    elif result.recovered:
        reason = result.recovery_reason or "the previous shell became unusable"
        notices.append(
            ToolNotice(
                code="shell_restarted",
                message=(f"The terminal was automatically restarted because {reason}. Filesystem changes were preserved, but shell-local state was lost."),
            )
        )
    if result.truncated:
        notices.append(
            ToolNotice(
                code="output_truncated",
                message="Command output exceeded the capture limit and was truncated.",
            )
        )
    if result.capture_error:
        notices.append(
            ToolNotice(
                code="output_unavailable",
                message=f"Some command output could not be recovered: {result.capture_error}.",
            )
        )
    if not result.timed_out and not result.recovered and result.exit_code not in {0, -1}:
        notices.append(
            ToolNotice(
                code="nonzero_exit",
                message=f"Command exited with status {result.exit_code}.",
            )
        )
    return ToolObservation(output=result.output, notices=tuple(notices))


def format_shell_result(result: ShellResult, *, command_timeout: float) -> str:
    """Backward-compatible text renderer for callers outside the harness."""
    return render_tool_observation(shell_result_observation(result, command_timeout=command_timeout))


class PersistentShellError(RuntimeError):
    """Infrastructure failure in the native-react tmux session."""

    _rllm_termination_reason = TerminationReason.SANDBOX_ERROR


class PersistentShellProtocolError(PersistentShellError):
    """The tmux session returned corrupt or incomplete control data."""


class _PersistentShellExited(PersistentShellError):
    """The agent command replaced or exited the tmux-owned Bash."""


class _CompletionPollingFailed(PersistentShellError):
    """The sandbox lost repeated completion-poll responses."""


class _CommandResultProtocolError(PersistentShellProtocolError):
    """The durable command-result metadata could not be decoded."""


class _OutputTransportProtocolError(PersistentShellProtocolError):
    """The bounded output transport could not be decoded."""


_TMUX_INSTALL_SCRIPT = r"""
set -e
export DEBIAN_FRONTEND=noninteractive
if command -v bash >/dev/null 2>&1 && command -v tmux >/dev/null 2>&1 && command -v timeout >/dev/null 2>&1 && command -v base64 >/dev/null 2>&1 && command -v stty >/dev/null 2>&1; then
    exit 0
fi
if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y -qq bash tmux coreutils
elif command -v apk >/dev/null 2>&1; then
    apk add --no-cache tmux coreutils bash
elif command -v dnf >/dev/null 2>&1; then
    dnf install -y bash tmux coreutils
elif command -v yum >/dev/null 2>&1; then
    yum install -y bash tmux coreutils
fi
command -v bash >/dev/null
command -v tmux >/dev/null
command -v timeout >/dev/null
command -v base64 >/dev/null
command -v stty >/dev/null
"""

_COMPLETION_POLL_SLICE_S = 5.0
_COMPLETION_POLL_KILL_GRACE_S = 1.0
# Remote backends include process scheduling, stdout draining, and teardown in
# their hard exec deadline. Keep that deadline well outside the in-sandbox
# watchdog so transport latency cannot masquerade as three lost poll responses.
_COMPLETION_POLL_TRANSPORT_GRACE_S = 10.0
_COMPLETION_POLL_FAILURE_LIMIT = 3
_PROTOCOL_READ_RETRIES = 3
_COMMAND_STAGE_CHUNK_CHARS = 8 * 1024
_ANSI_ESCAPE_RE = re.compile(
    r"\x1b(?:\][^\x07]*(?:\x07|\x1b\\)|P.*?\x1b\\|\[[0-?]*[ -/]*[@-~]|[@-_])",
    re.DOTALL,
)


def _sanitize_terminal_output(text: str) -> str:
    """Render common terminal controls and remove escape sequences.

    Command output is captured from file descriptors rather than tmux history,
    so carriage returns and backspaces have not yet been applied by a terminal.
    A small line renderer keeps progress output readable without pulling a
    terminal-emulation dependency into every benchmark sandbox.
    """
    text = _ANSI_ESCAPE_RE.sub("", text)
    rendered: list[str] = []
    line: list[str] = []
    cursor = 0

    for character in text:
        if character == "\n":
            rendered.append("".join(line))
            rendered.append("\n")
            line = []
            cursor = 0
        elif character == "\r":
            cursor = 0
        elif character == "\b":
            cursor = max(0, cursor - 1)
        elif character == "\t" or ord(character) >= 0x20:
            if cursor < len(line):
                line[cursor] = character
            else:
                if cursor > len(line):
                    line.extend(" " for _ in range(cursor - len(line)))
                line.append(character)
            cursor += 1
        # Other C0 controls have no useful textual representation here.

    rendered.append("".join(line))
    return "".join(rendered)


class PersistentBashSession:
    """Persistent Bash backed by a self-contained tmux session.

    This follows Harbor Terminus-2's terminal design without importing Harbor:
    one detached interactive Bash owns all shell state, commands are delivered
    through tmux, and atomic files signal command completion. Command bodies
    are staged in bounded chunks, while a FIFO collector captures a bounded
    byte prefix and drains excess output without relying on pane history.
    """

    def __init__(
        self,
        sandbox: Sandbox,
        *,
        user: str | None = None,
        max_buffer_size: int = 480 * 1024,
    ) -> None:
        if isinstance(max_buffer_size, bool) or not isinstance(max_buffer_size, int) or max_buffer_size <= 0:
            raise ValueError("max_buffer_size must be a positive integer")
        self.sandbox = sandbox
        self.user = user
        self.max_buffer_size = max_buffer_size
        self.session_dir = f"/tmp/.rllm_native_react_{uuid.uuid4().hex[:12]}"
        self._nonce = self.session_dir.rsplit("_", 1)[-1]
        self.session_name = f"rllm_native_react_{self._nonce}"
        self.socket_name = self.session_name
        self._tmux_path = "tmux"
        self._stty_path = "stty"
        self._sequence = 0
        self._started = False
        self._workdir: str | None = None

    def _exec(self, command: str, *, timeout: float = 30, operation: str) -> str:
        try:
            return self.sandbox.exec(command, timeout=timeout, user=self.user)
        except Exception as error:
            raise PersistentShellError(f"native_react tmux {operation} failed") from error

    def _tmux(self, *, clean_config: bool = False) -> str:
        command = f"{shlex.quote(self._tmux_path)} -u -L {shlex.quote(self.socket_name)}"
        if clean_config:
            command += " -f /dev/null"
        return command

    def _send_line(self, line: str, *, operation: str) -> None:
        tmux = self._tmux()
        target = shlex.quote(self.session_name)
        literal = shlex.quote(line)
        self._exec(
            f"{tmux} send-keys -l -t {target} -- {literal} && {tmux} send-keys -t {target} Enter",
            operation=operation,
        )

    def _wait_for(self, channel: str, timeout: float) -> int:
        tmux = self._tmux()
        seconds = f"{max(timeout, 0.001):.6f}"
        raw = self._exec(
            f'status=0; timeout -k 2s {seconds}s {tmux} wait-for {shlex.quote(channel)} || status=$?; printf %s "$status"',
            timeout=timeout + 5,
            operation="completion wait",
        )
        try:
            return int(raw.strip())
        except ValueError as error:
            raise PersistentShellProtocolError(f"native_react tmux returned invalid wait status {raw!r}") from error

    def _wait_for_completion(self, exit_path: str, timeout: float) -> bool:
        """Wait for one durable completion marker without a long sandbox RPC.

        The command has already been dispatched exactly once. Each poll is
        bounded independently, so a backend timeout cannot strand the harness
        behind a command-sized ``tmux wait-for`` call. A timeout-shaped control
        failure is retried while the command deadline remains; repeated failures
        are infrastructure errors, not agent timeouts.
        """
        deadline = time.monotonic() + timeout
        exit_file = shlex.quote(exit_path)
        tmux = self._tmux()
        target = shlex.quote(self.session_name)
        consecutive_failures = 0

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return self._probe_completion(exit_path)
            poll_slice = min(_COMPLETION_POLL_SLICE_S, remaining)
            seconds = f"{max(poll_slice, 0.001):.6f}"
            wait_script = f"while [ ! -e {exit_file} ]; do {tmux} has-session -t {target} >/dev/null 2>&1 || exit 3; sleep 0.1; done"
            control = (
                f"status=0; timeout -k {_COMPLETION_POLL_KILL_GRACE_S:g}s {seconds}s "
                f"sh -c {shlex.quote(wait_script)} || status=$?; "
                f"if [ -e {exit_file} ]; then printf complete; "
                'elif [ "$status" -eq 124 ] || [ "$status" -eq 137 ] || [ "$status" -eq 143 ]; '
                "then printf pending; "
                'elif [ "$status" -eq 3 ]; then printf dead; '
                'else printf "error:%s" "$status"; fi'
            )
            try:
                state = self._exec(
                    control,
                    timeout=poll_slice + _COMPLETION_POLL_KILL_GRACE_S + _COMPLETION_POLL_TRANSPORT_GRACE_S,
                    operation="completion poll",
                ).strip()
            except PersistentShellError as error:
                if not _is_timeout_exception(error):
                    raise
                consecutive_failures += 1
                if time.monotonic() >= deadline:
                    return self._probe_completion(exit_path)
                if consecutive_failures >= _COMPLETION_POLL_FAILURE_LIMIT:
                    raise _CompletionPollingFailed("native_react completion polling repeatedly timed out") from error
                continue

            if state == "complete":
                consecutive_failures = 0
                return True
            if state == "pending":
                consecutive_failures = 0
                continue
            if state == "dead":
                raise _PersistentShellExited("native_react tmux bash exited while running a command")
            consecutive_failures += 1
            if consecutive_failures >= _COMPLETION_POLL_FAILURE_LIMIT:
                raise _CompletionPollingFailed("native_react completion polling repeatedly returned invalid state")

    def _probe_completion(self, exit_path: str) -> bool:
        """Resolve the command state once without waiting or redispatching it."""
        exit_file = shlex.quote(exit_path)
        tmux = self._tmux()
        target = shlex.quote(self.session_name)
        for _attempt in range(_PROTOCOL_READ_RETRIES):
            state = self._exec(
                f"if [ -e {exit_file} ]; then printf complete; elif {tmux} has-session -t {target} >/dev/null 2>&1; then printf pending; else printf dead; fi",
                timeout=10,
                operation="completion probe",
            ).strip()
            if state == "complete":
                return True
            if state == "pending":
                return False
            if state == "dead":
                raise _PersistentShellExited("native_react tmux bash exited while running a command")
        raise _CompletionPollingFailed("native_react completion probe repeatedly returned invalid state")

    def start(self, *, workdir: str | None = None) -> None:
        if workdir is not None:
            self._workdir = str(workdir)
        if self._started and self.is_alive():
            return
        if self._started:
            self.close()

        directory = shlex.quote(self.session_dir)
        target = shlex.quote(self.session_name)
        try:
            tmux_path = self._exec("command -v tmux", operation="executable lookup").strip()
            bash_path = self._exec("command -v bash", operation="Bash executable lookup").strip()
            stty_path = self._exec("command -v stty", operation="stty executable lookup").strip()
            if not tmux_path or "\n" in tmux_path:
                raise PersistentShellProtocolError("native_react could not resolve a single tmux executable")
            if not bash_path or "\n" in bash_path:
                raise PersistentShellProtocolError("native_react could not resolve a single Bash executable")
            if not stty_path or "\n" in stty_path:
                raise PersistentShellProtocolError("native_react could not resolve a single stty executable")
            self._tmux_path = tmux_path
            self._stty_path = stty_path
            tmux = self._tmux()
            clean_tmux = self._tmux(clean_config=True)
            shell_command = shlex.quote(f"{bash_path} --noprofile --norc")
            start_directory = f" -c {shlex.quote(self._workdir)}" if self._workdir is not None else ""
            self._exec(
                "set -e; command -v timeout >/dev/null; command -v base64 >/dev/null; "
                f"{tmux} kill-session -t {target} 2>/dev/null || true; "
                f"rm -rf {directory}; mkdir -p {directory}; "
                f"TMUX= TERM=xterm-256color SHELL={shlex.quote(bash_path)} {clean_tmux} new-session -x 240 -y 50 -d -s {target}{start_directory} {shell_command}; "
                f"{tmux} set-option -t {target}:0 history-limit 10000000",
                operation="session startup",
            )

            # Wait until Bash is accepting input. Disable terminal echo and the
            # prompt so captured observations contain command output, not the
            # private completion wrapper.
            ready_channel = f"{self.session_name}_ready"
            self._send_line(
                "stty -echo; PS1=''; PS2=''; PROMPT_COMMAND=''; "
                "PAGER=cat; GIT_PAGER=cat; SYSTEMD_PAGER=cat; MANPAGER=cat; "
                "export PAGER GIT_PAGER SYSTEMD_PAGER MANPAGER; "
                f"{tmux} wait-for -S {shlex.quote(ready_channel)}",
                operation="initialization",
            )
            if self._wait_for(ready_channel, 10) != 0:
                raise PersistentShellError("native_react tmux shell initialization timed out")
            self._started = True
        except Exception:
            self.close()
            raise

    def is_alive(self) -> bool:
        try:
            self.sandbox.exec(
                f"{self._tmux()} has-session -t {shlex.quote(self.session_name)}",
                timeout=10,
                user=self.user,
            )
            return True
        except Exception:
            return False

    def _interrupt_and_reap_jobs(self, sequence: int) -> None:
        """Interrupt the foreground command and kill jobs owned by this Bash.

        The in-pane cleanup preserves normal interactive-shell semantics when
        Bash is responsive. The out-of-pane ``/proc`` sweep is the hard
        backstop: it does not depend on an INT-trapping or otherwise wedged Bash
        accepting another command.
        """
        tmux = self._tmux()
        target = shlex.quote(self.session_name)
        pane_pid: int | None = None
        try:
            raw_pane = self._exec(
                f"{tmux} display-message -p -t {target} '#{{pane_dead}} #{{pane_pid}}'",
                timeout=10,
                operation="pane process lookup",
            ).strip()
            raw_dead, raw_pid = raw_pane.split()
            if raw_dead == "0":
                pane_pid = int(raw_pid)
        except Exception:
            logger.debug("native_react tmux pane process lookup failed", exc_info=True)

        try:
            self._exec(f"{tmux} send-keys -t {target} C-c", operation="timeout interrupt")

            # Interactive Bash gives background pipelines their own process
            # groups, so killing the tmux pane alone can leave them alive. Reap
            # them through Bash's job table while the shell still exists.
            cleanup_channel = f"{self.session_name}_cleanup_{sequence}"
            jobs_variable = f"__rllm_jobs_{self._nonce}"
            job_variable = f"__rllm_job_{self._nonce}"
            cleanup = (
                "IFS=$' \\t\\n'; "
                f"{jobs_variable}=$(builtin jobs -pr); "
                f"for {job_variable} in ${jobs_variable}; do "
                f'builtin kill -KILL "${job_variable}" 2>/dev/null; '
                "done; builtin wait 2>/dev/null; "
                f"builtin unset {jobs_variable} {job_variable}; "
                f"{tmux} wait-for -S {shlex.quote(cleanup_channel)}"
            )
            self._send_line(cleanup, operation="timeout job cleanup")
            self._wait_for(cleanup_channel, 2)
        except Exception:
            logger.debug("native_react tmux job cleanup did not complete", exc_info=True)

        if pane_pid is not None:
            self._reap_pane_processes(pane_pid)

    def _reap_pane_processes(self, pane_pid: int) -> None:
        """Hard-kill processes belonging to one tmux pane from outside it.

        Only exact descendants of a pane that tmux reports as currently alive
        are eligible. Three quick passes close ordinary fork/exit races without
        risking process-group or reused-TTY collateral damage.
        """
        root = int(pane_pid)
        script = f"""
root={root}
for round in 1 2 3; do
    unset parent owned
    declare -A parent=() owned=()
    for stat_file in /proc/[0-9]*/stat; do
        [ -r "$stat_file" ] || continue
        stat=$(<"$stat_file") || continue
        pid=${{stat%% *}}
        rest=${{stat##*) }}
        set -- $rest
        [ "$#" -ge 3 ] || continue
        parent[$pid]=$2
    done
    owned[$root]=1
    for ((pass = 0; pass <= ${{#parent[@]}}; pass++)); do
        for pid in "${{!parent[@]}}"; do
            p=${{parent[$pid]}}
            [ -n "${{owned[$p]+yes}}" ] && owned[$pid]=1
        done
    done
    victims=()
    for pid in "${{!parent[@]}}"; do
        [ "$pid" = "$root" ] && continue
        pane_owned=${{owned[$pid]+yes}}
        [ -n "$pane_owned" ] || continue
        victims+=("$pid")
    done
    [ "${{#victims[@]}}" -gt 0 ] || break
    kill -KILL -- "${{victims[@]}}" 2>/dev/null || true
    sleep 0.05
done
"""
        try:
            self._exec(
                f"bash -c {shlex.quote(script)}",
                timeout=10,
                operation="pane process cleanup",
            )
        except PersistentShellError:
            logger.debug("native_react tmux pane process cleanup failed", exc_info=True)

    def _read_output_file(self, output_path: str, *, byte_count: int | None = None) -> tuple[str, bool]:
        """Read bounded command bytes through a base64 transport."""
        quoted_output = shlex.quote(output_path)
        if byte_count is None:
            raw_size = ""
            for _attempt in range(_PROTOCOL_READ_RETRIES):
                raw_size = self._exec(
                    f"if [ -f {quoted_output} ]; then wc -c < {quoted_output}; else printf 0; fi",
                    operation="output-size read",
                ).strip()
                try:
                    byte_count = int(raw_size)
                except ValueError:
                    continue
                break
            else:
                raise _OutputTransportProtocolError(f"native_react returned invalid output size {raw_size!r}")
        if byte_count < 0:
            raise _OutputTransportProtocolError(f"native_react returned invalid output size {byte_count!r}")

        read_size = min(byte_count, self.max_buffer_size)
        if not read_size:
            return "", False

        max_encoded = 4 * ((read_size + 2) // 3)
        encoded = ""
        output_bytes: bytes | None = None
        for _attempt in range(_PROTOCOL_READ_RETRIES):
            raw = self._exec(
                f"head -c {read_size} {quoted_output} 2>/dev/null | base64",
                operation="command-output read",
            )
            encoded = "".join(raw.split())
            if len(encoded) > max_encoded:
                continue
            try:
                output_bytes = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError):
                continue
            if len(output_bytes) != read_size:
                output_bytes = None
                continue
            break
        if output_bytes is None:
            raise _OutputTransportProtocolError("native_react command output transport was invalid")
        output = output_bytes.decode("utf-8", errors="replace")
        return _sanitize_terminal_output(output), byte_count > self.max_buffer_size

    def _read_command_result(self, exit_path: str) -> tuple[int, int, bool]:
        """Read and validate atomic ``exit_code output_size truncated`` metadata."""
        raw_exit = ""
        for _attempt in range(_PROTOCOL_READ_RETRIES):
            raw_exit = self._exec(f"cat {shlex.quote(exit_path)}", operation="exit-code read").strip()
            try:
                raw_code, raw_size, raw_truncated = raw_exit.split()
                exit_code = int(raw_code)
                output_size = int(raw_size)
                truncated = bool(int(raw_truncated))
            except (ValueError, TypeError):
                continue
            if output_size >= 0 and raw_truncated in {"0", "1"}:
                return exit_code, output_size, truncated
        raise _CommandResultProtocolError(f"native_react returned invalid command result metadata {raw_exit!r}")

    def _output_was_truncated(self, truncation_path: str) -> bool:
        """Best-effort read of the collector's durable truncation flag."""
        state = self._exec(
            f"if [ -e {shlex.quote(truncation_path)} ]; then printf 1; else printf 0; fi",
            operation="output-truncation read",
        ).strip()
        if state not in {"0", "1"}:
            raise _OutputTransportProtocolError(f"native_react returned invalid truncation state {state!r}")
        return state == "1"

    def _restart_with_partial_result(
        self,
        output_path: str,
        *,
        reason: str,
        timed_out: bool = False,
        truncation_path: str | None = None,
    ) -> ShellResult:
        """Salvage output, replace the shell, and keep the rollout alive."""
        capture_error: str | None = None
        try:
            output, truncated = self._read_output_file(output_path)
            # A control response can fail immediately after dispatch, before
            # the interactive Bash has had a scheduling turn. Give an alive
            # pane one brief chance to publish already-requested output; never
            # redispatch the command.
            if not output and self.is_alive():
                time.sleep(0.05)
                output, truncated = self._read_output_file(output_path)
            if truncation_path is not None:
                truncated = truncated or self._output_was_truncated(truncation_path)
        except PersistentShellError as error:
            logger.warning("native_react could not recover command output after %s: %s", reason, error)
            output = ""
            truncated = False
            capture_error = "the terminal output transport remained invalid"
        self.close()
        self.start()
        return ShellResult(
            output=output,
            exit_code=-1,
            timed_out=timed_out,
            recovered=True,
            recovery_reason=reason,
            truncated=truncated,
            capture_error=capture_error,
        )

    def _stage_command(self, command_path: str, command: str) -> None:
        """Stage an arbitrary command using bounded control-plane requests."""
        encoded = base64.b64encode((command if command.strip() else ":").encode("utf-8", errors="replace")).decode("ascii")
        encoded_path = f"{command_path}.b64"
        temporary_path = f"{command_path}.tmp"
        quoted_encoded = shlex.quote(encoded_path)
        self._exec(
            f"rm -f {shlex.quote(command_path)} {shlex.quote(temporary_path)} {quoted_encoded}; : > {quoted_encoded}",
            operation="command staging initialization",
        )
        for offset in range(0, len(encoded), _COMMAND_STAGE_CHUNK_CHARS):
            chunk = encoded[offset : offset + _COMMAND_STAGE_CHUNK_CHARS]
            self._exec(
                f"printf %s {shlex.quote(chunk)} >> {quoted_encoded}",
                operation="command staging chunk",
            )
        self._exec(
            f"base64 -d {quoted_encoded} > {shlex.quote(temporary_path)} && mv -f {shlex.quote(temporary_path)} {shlex.quote(command_path)}; rm -f {quoted_encoded}",
            operation="command staging finalization",
        )

    def run(self, command: str, timeout: float) -> ShellResult:
        if not self._started:
            raise PersistentShellError("persistent tmux bash has not been started")
        try:
            timeout_seconds = float(timeout)
        except (TypeError, ValueError) as error:
            raise ValueError("timeout must be a positive finite number") from error
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout must be a positive finite number")

        recovery_before_dispatch: str | None = None
        if not self.is_alive():
            # No command can still be running in a dead tmux session, so this
            # restart is unambiguous and the requested command remains safe to
            # dispatch exactly once in the replacement shell.
            self.close()
            self.start()
            recovery_before_dispatch = "shell was not alive before command dispatch"

        self._sequence += 1
        sequence = self._sequence
        command_path = f"{self.session_dir}/c{sequence}"
        output_path = f"{self.session_dir}/o{sequence}"
        exit_path = f"{self.session_dir}/e{sequence}"
        stream_path = f"{self.session_dir}/p{sequence}"
        truncation_path = f"{self.session_dir}/t{sequence}"
        frame_id = uuid.uuid4().hex
        rc_variable = f"__rllm_rc_{frame_id}"
        size_variable = f"__rllm_size_{frame_id}"
        collector_variable = f"__rllm_collector_{frame_id}"
        extra_variable = f"__rllm_extra_{frame_id}"
        truncated_variable = f"__rllm_truncated_{frame_id}"
        self._exec(
            f"rm -f {shlex.quote(output_path)} {shlex.quote(exit_path)} {shlex.quote(exit_path + '.tmp')} {shlex.quote(stream_path)} {shlex.quote(truncation_path)}",
            operation="command-result initialization",
        )
        self._stage_command(command_path, command)
        wrapper = (
            f"command mkfifo {shlex.quote(stream_path)}; "
            f"(command dd bs=1 count={self.max_buffer_size} status=none > {shlex.quote(output_path)}; "
            f"{extra_variable}=$(command dd bs=1 count=1 status=none | command wc -c); "
            f'if [ "${{{extra_variable}}}" -gt 0 ]; then builtin : > {shlex.quote(truncation_path)}; fi; '
            f"command cat >/dev/null) < {shlex.quote(stream_path)} & {collector_variable}=$!; "
            f'builtin disown "${{{collector_variable}}}" 2>/dev/null || true; '
            f"builtin source {shlex.quote(command_path)} < /dev/null > {shlex.quote(stream_path)} 2>&1; {rc_variable}=$?; "
            f"for __rllm_wait_{frame_id} in 1 2 3 4 5 6 7 8 9 10; do "
            f'command kill -0 "${{{collector_variable}}}" 2>/dev/null || {{ builtin wait "${{{collector_variable}}}" 2>/dev/null; break; }}; '
            "command sleep 0.01; done; "
            f"{shlex.quote(self._stty_path)} sane -echo </dev/tty 2>/dev/null || true; "
            f"{size_variable}=$(command wc -c < {shlex.quote(output_path)}); "
            f"{truncated_variable}=0; [ -e {shlex.quote(truncation_path)} ] && {truncated_variable}=1; "
            f'builtin printf "%s %s %s" "${rc_variable}" "${size_variable}" "${truncated_variable}" > {shlex.quote(exit_path + ".tmp")}; '
            f"command mv -f {shlex.quote(exit_path + '.tmp')} {shlex.quote(exit_path)}; "
            f"builtin unset {rc_variable} {size_variable} {collector_variable} {extra_variable} {truncated_variable} __rllm_wait_{frame_id}"
        )
        self._send_line(wrapper, operation="command dispatch")
        try:
            completed = self._wait_for_completion(exit_path, timeout_seconds)
        except (_PersistentShellExited, _CompletionPollingFailed) as error:
            if isinstance(error, _PersistentShellExited):
                reason = "shell exited while running the command"
            else:
                reason = str(error).removeprefix("native_react ")
            return self._restart_with_partial_result(
                output_path,
                reason=reason,
                truncation_path=truncation_path,
            )
        timed_out = not completed

        if timed_out:
            exit_code = -1
            try:
                output, truncated = self._read_output_file(output_path)
            except _OutputTransportProtocolError:
                return self._restart_with_partial_result(
                    output_path,
                    reason="command timed out and the output transport became invalid",
                    timed_out=True,
                    truncation_path=truncation_path,
                )
            try:
                truncated = truncated or self._output_was_truncated(truncation_path)
            except _OutputTransportProtocolError:
                return self._restart_with_partial_result(
                    output_path,
                    reason="command timed out and the truncation metadata became invalid",
                    timed_out=True,
                    truncation_path=truncation_path,
                )
        else:
            try:
                exit_code, output_size, capture_truncated = self._read_command_result(exit_path)
            except _CommandResultProtocolError:
                return self._restart_with_partial_result(
                    output_path,
                    reason="command result metadata remained invalid",
                    truncation_path=truncation_path,
                )
            try:
                output, truncated = self._read_output_file(output_path, byte_count=output_size)
            except _OutputTransportProtocolError:
                return self._restart_with_partial_result(
                    output_path,
                    reason="command output transport remained invalid",
                    truncation_path=truncation_path,
                )
            truncated = truncated or capture_truncated

        if timed_out:
            self.close()
            self.start()
            result = ShellResult(
                output=output,
                exit_code=exit_code,
                timed_out=True,
                recovered=True,
                recovery_reason="command timed out",
                truncated=truncated,
            )
        else:
            result = ShellResult(
                output=output,
                exit_code=exit_code,
                recovered=recovery_before_dispatch is not None,
                recovery_reason=recovery_before_dispatch,
                truncated=truncated,
            )
            try:
                self._exec(
                    f"rm -f {shlex.quote(command_path)} {shlex.quote(output_path)} {shlex.quote(exit_path)} "
                    f"{shlex.quote(exit_path + '.tmp')} {shlex.quote(stream_path)} {shlex.quote(truncation_path)}",
                    timeout=10,
                    operation="command cleanup",
                )
            except PersistentShellError:
                logger.debug("native_react tmux command-file cleanup failed", exc_info=True)
        return result

    def close(self) -> None:
        directory = shlex.quote(self.session_dir)
        cleanup = f"{self._tmux()} kill-session -t {shlex.quote(self.session_name)} 2>/dev/null || true; rm -rf {directory}; true"
        try:
            if self.is_alive():
                self._interrupt_and_reap_jobs(self._sequence)
        except Exception:
            logger.debug("native_react tmux process cleanup failed", exc_info=True)
        try:
            self.sandbox.exec(cleanup, timeout=10, user=self.user)
        except Exception:
            logger.debug("native_react tmux cleanup failed", exc_info=True)
        self._started = False


class NativeReactHarness(SandboxedAgentFlow):
    """Minimal native-tool ReAct loop without compaction."""

    name = "native_react"
    sandbox_backend = "docker"
    max_turns: int | None = 300
    max_tokens: int = 32_768
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

    def install_script(self) -> str:
        """Install the only external runtime dependency used by this harness."""
        return _TMUX_INSTALL_SCRIPT

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
        from openai import OpenAI, OpenAIError

        client = OpenAI(base_url=config.base_url, api_key="EMPTY", max_retries=0)
        agent_user = task.metadata.get("agent_user")
        workdir = str(task.metadata.get("workdir") or "/app")
        task_max_turns = task.metadata.get("rllm", {}).get("max_turns")
        max_turns = int(task_max_turns) if task_max_turns else self.max_turns
        agent_timeout = self._effective_timeout(task)
        shell = PersistentBashSession(env, user=agent_user)
        turns = 0
        parse_errors = 0
        consecutive_parse_errors = 0

        try:
            shell.start(workdir=workdir)
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
                    tool_choice="required",
                    max_tokens=self.max_tokens,
                    timeout=remaining,
                )
                choices = response.get("choices") if isinstance(response, dict) else getattr(response, "choices", None)
                if not choices:
                    raise _invalid_model_response(response, "model response contained no choices")
                choice = choices[0]
                response_message = choice.get("message") if isinstance(choice, dict) else getattr(choice, "message", None)
                if response_message is None:
                    raise _invalid_model_response(response, "model response choice contained no assistant message")
                try:
                    assistant = preserve_assistant_message(response_message)
                except Exception as error:
                    raise _invalid_model_response(response, "model response contained an invalid assistant message") from error
                messages.append(assistant)
                turns += 1

                finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else getattr(choice, "finish_reason", None)
                if finish_reason == "length":
                    return self._outcome_episode(
                        task,
                        config,
                        termination_reason=TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED,
                        turns=turns,
                        parse_errors=parse_errors,
                        error={
                            "error_type": "OutputLengthExceededError",
                            "message": f"model response reached the per-turn output limit of {self.max_tokens} tokens",
                        },
                    )

                tool_calls = parse_native_tool_calls(assistant)
                if not tool_calls:
                    parse_errors += 1
                    consecutive_parse_errors += 1
                    messages.append(observation_message(ToolObservation(output=_PARSE_ERROR_OBSERVATION)))
                    if consecutive_parse_errors > self.max_parse_retries:
                        return self._outcome_episode(
                            task,
                            config,
                            termination_reason=TerminationReason.TOOL_PROTOCOL_ERROR,
                            turns=turns,
                            parse_errors=parse_errors,
                            error={
                                "error_type": "ToolProtocolError",
                                "message": (f"model produced {consecutive_parse_errors} consecutive responses without a valid native tool call (retry limit: {self.max_parse_retries})"),
                            },
                        )
                    continue
                consecutive_parse_errors = 0

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
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            return self._timeout_episode(
                                task,
                                config,
                                message=f"Agent execution timed out after {agent_timeout:g}s",
                                turns=turns,
                                parse_errors=parse_errors,
                            )
                    messages.append(
                        observation_message(
                            shell_result_observation(result, command_timeout=command_timeout),
                            tool_call_id=tool_call.id,
                            max_length=self.max_output_length,
                        )
                    )

            return self._outcome_episode(
                task,
                config,
                termination_reason=(TerminationReason.MAX_TURNS_EXCEEDED if max_turns is not None and turns >= max_turns else None),
                turns=turns,
                parse_errors=parse_errors,
            )
        except PersistentShellError:
            # A sandbox/control-plane timeout is an infrastructure failure. The
            # command deadline is converted into ShellResult.timed_out inside
            # PersistentBashSession.run and must never reach this boundary.
            raise
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
            if _is_context_length_exception(error):
                return self._outcome_episode(
                    task,
                    config,
                    termination_reason=TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED,
                    turns=turns,
                    parse_errors=parse_errors,
                    error={"error_type": "ContextLengthExceededError", "message": str(error)},
                )
            if isinstance(error, OpenAIError):
                # Preserve retries for transient upstream failures, but make an
                # exhausted retry budget resolve to MODEL_ERROR rather than the
                # engine's generic ERROR bucket.
                try:
                    error._rllm_termination_reason = TerminationReason.MODEL_ERROR
                except Exception:
                    raise NativeModelResponseError(str(error), body=getattr(error, "body", None)) from error
            raise
        finally:
            # The verifier runs in this same task sandbox after the harness
            # returns. Leave the tmux shell and agent-started background
            # services intact until the task context tears the sandbox down
            # after evaluation. Hard shell cleanup remains scoped to timeout
            # recovery and explicit PersistentBashSession.close() callers.
            client.close()


__all__ = [
    "NATIVE_SYSTEM_PROMPT",
    "NATIVE_TOOL_SCHEMAS",
    "NATIVE_USER_PROMPT",
    "NativeToolCall",
    "NativeReactHarness",
    "NativeModelResponseError",
    "PersistentBashSession",
    "PersistentShellError",
    "PersistentShellProtocolError",
    "ShellResult",
    "ToolNotice",
    "ToolObservation",
    "format_shell_result",
    "initial_messages",
    "limit_output_length",
    "observation_message",
    "parse_native_tool_calls",
    "preserve_assistant_message",
    "render_tool_observation",
    "shell_result_observation",
    "tool_result",
]
