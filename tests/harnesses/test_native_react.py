from __future__ import annotations

import base64
import copy
import json
import shlex
import shutil
import socket
import time
from types import SimpleNamespace

import httpx
import pytest
from openai import APITimeoutError, OpenAI

from rllm.harnesses.native_react import (
    NATIVE_SYSTEM_PROMPT,
    NATIVE_TOOL_SCHEMAS,
    NativeReactHarness,
    PersistentBashSession,
    PersistentShellError,
    PersistentShellProtocolError,
    ShellResult,
    ToolNotice,
    ToolObservation,
    _is_timeout_exception,
    format_shell_result,
    initial_messages,
    limit_output_length,
    observation_message,
    parse_native_tool_calls,
    preserve_assistant_message,
    render_tool_observation,
    shell_result_observation,
    tool_result,
)
from rllm.sandbox.backends.local import LocalSandbox
from rllm.sandbox.protocol import SandboxCommandTimeout
from rllm.types import INFRA_ERROR_REASONS, AgentConfig, Task, TerminationReason

requires_tmux = pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is required for the native-react terminal integration")


class _OuterWaitTimeoutSandbox:
    """Emulate a backend killing the long completion wait before it returns 124."""

    def __init__(self, name: str) -> None:
        self._delegate = LocalSandbox(name)

    def exec(self, command: str, timeout: float | None = None, user: str | None = None) -> str:
        if "status=0; timeout " in command and " wait-for " in command and "_ready" not in command and "_cleanup_" not in command:
            raise SandboxCommandTimeout("outer sandbox killed completion wait")
        return self._delegate.exec(command, timeout=timeout, user=user)

    def close(self) -> None:
        self._delegate.close()


class _TransientPollTimeoutSandbox:
    """Lose one bounded poll response while the tmux command keeps running."""

    def __init__(self, name: str) -> None:
        self._delegate = LocalSandbox(name)
        self.poll_failures = 0
        self.command_dispatches = 0

    def exec(self, command: str, timeout: float | None = None, user: str | None = None) -> str:
        if "send-keys -l" in command and "builtin source" in command:
            self.command_dispatches += 1
        if "while [ ! -e" in command and "has-session" in command and self.poll_failures == 0:
            self.poll_failures += 1
            raise SandboxCommandTimeout("poll response was lost")
        return self._delegate.exec(command, timeout=timeout, user=user)

    def close(self) -> None:
        self._delegate.close()


class _RepeatedPollTimeoutSandbox:
    """Keep losing poll responses to exercise the infrastructure cutoff."""

    def __init__(self, name: str) -> None:
        self._delegate = LocalSandbox(name)
        self.poll_failures = 0
        self.command_dispatches = 0

    def exec(self, command: str, timeout: float | None = None, user: str | None = None) -> str:
        if "send-keys -l" in command and "builtin source" in command:
            self.command_dispatches += 1
        if "while [ ! -e" in command and "has-session" in command and self.poll_failures < 3:
            self.poll_failures += 1
            raise SandboxCommandTimeout("poll response was repeatedly lost")
        return self._delegate.exec(command, timeout=timeout, user=user)

    def close(self) -> None:
        self._delegate.close()


class _RemotePollOverheadSandbox:
    """Model a remote backend whose hard deadline includes transport overhead."""

    def __init__(self, name: str, *, minimum_poll_timeout: float) -> None:
        self._delegate = LocalSandbox(name)
        self.minimum_poll_timeout = minimum_poll_timeout
        self.poll_failures = 0
        self.poll_timeouts: list[float | None] = []
        self.command_dispatches = 0

    def exec(self, command: str, timeout: float | None = None, user: str | None = None) -> str:
        if "send-keys -l" in command and "builtin source" in command:
            self.command_dispatches += 1
        if "while [ ! -e" in command and "has-session" in command:
            self.poll_timeouts.append(timeout)
            if timeout is None or timeout < self.minimum_poll_timeout:
                self.poll_failures += 1
                raise SandboxCommandTimeout("remote backend killed the poll during transport teardown")
        return self._delegate.exec(command, timeout=timeout, user=user)

    def close(self) -> None:
        self._delegate.close()


class _DeadlinePollTimeoutSandbox:
    """Lose a poll across the deadline after the command has actually finished."""

    def __init__(self, name: str) -> None:
        self._delegate = LocalSandbox(name)
        self.poll_failures = 0
        self.command_dispatches = 0

    def exec(self, command: str, timeout: float | None = None, user: str | None = None) -> str:
        if "send-keys -l" in command and "builtin source" in command:
            self.command_dispatches += 1
        if "while [ ! -e" in command and "has-session" in command and self.poll_failures == 0:
            self.poll_failures += 1
            time.sleep(0.2)
            raise SandboxCommandTimeout("poll response was lost across command deadline")
        return self._delegate.exec(command, timeout=timeout, user=user)

    def close(self) -> None:
        self._delegate.close()


class _PaddedCaptureSandbox:
    """Reproduce tmux 3.1 capture-pane rows padded to the pane width."""

    def __init__(self, name: str) -> None:
        self._delegate = LocalSandbox(name)
        self.capture_calls = 0

    def exec(self, command: str, timeout: float | None = None, user: str | None = None) -> str:
        raw = self._delegate.exec(command, timeout=timeout, user=user)
        if "capture-pane" not in command or "| base64" not in command:
            return raw

        self.capture_calls += 1
        pane = base64.b64decode("".join(raw.split()))
        padded = b"\n".join(line.ljust(60, b" ") for line in pane.split(b"\n"))
        return base64.b64encode(padded).decode("ascii")

    def close(self) -> None:
        self._delegate.close()


class _CommandLengthLimitedSandbox:
    """Model a backend with a strict argv/control-command size limit."""

    def __init__(self, name: str, max_command_length: int) -> None:
        self._delegate = LocalSandbox(name)
        self.max_command_length = max_command_length
        self.max_seen = 0

    def exec(self, command: str, timeout: float | None = None, user: str | None = None) -> str:
        self.max_seen = max(self.max_seen, len(command))
        if len(command) > self.max_command_length:
            raise RuntimeError(f"control command exceeded {self.max_command_length} bytes")
        return self._delegate.exec(command, timeout=timeout, user=user)

    def close(self) -> None:
        self._delegate.close()


class _CorruptShellProtocolSandbox:
    """Inject malformed control responses without disrupting the command."""

    def __init__(self, name: str, *, corrupt: str, failures: int = 10) -> None:
        self._delegate = LocalSandbox(name)
        self.corrupt = corrupt
        self.failures_remaining = failures

    def exec(self, command: str, timeout: float | None = None, user: str | None = None) -> str:
        raw = self._delegate.exec(command, timeout=timeout, user=user)
        should_corrupt = (
            (self.corrupt == "completion" and "while [ ! -e" in command and "has-session" in command)
            or (self.corrupt == "result" and command.startswith("cat /tmp/.rllm_native_react_") and "/e" in command)
            or (self.corrupt == "output" and command.startswith("head -c ") and command.endswith(" | base64"))
        )
        if should_corrupt and self.failures_remaining > 0:
            self.failures_remaining -= 1
            return "not-valid-protocol-data"
        return raw

    def close(self) -> None:
        self._delegate.close()


class _ResultMetadataRecordingSandbox:
    """Record the collector's durable output size before cleanup."""

    def __init__(self, name: str) -> None:
        self._delegate = LocalSandbox(name)
        self.output_sizes: list[int] = []

    def exec(self, command: str, timeout: float | None = None, user: str | None = None) -> str:
        raw = self._delegate.exec(command, timeout=timeout, user=user)
        if command.startswith("cat /tmp/.rllm_native_react_") and "/e" in command:
            fields = raw.strip().split()
            if len(fields) == 3:
                self.output_sizes.append(int(fields[1]))
        return raw

    def close(self) -> None:
        self._delegate.close()


def test_initial_prompt_and_native_tool_result_shape():
    messages = initial_messages("fix it", "/app\nfile.txt")

    assert (
        NATIVE_SYSTEM_PROMPT
        == """You are an expert software engineer solving a task inside a Linux container.

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
    )
    assert messages == [
        {"role": "system", "content": NATIVE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Task:\nfix it\n\nCurrent directory:\n/app\nfile.txt",
        },
    ]
    assert tool_result("call-7", "ok") == {
        "role": "tool",
        "tool_call_id": "call-7",
        "content": "ok",
    }


def test_assistant_message_round_trips_all_fields():
    original = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "inspect first",
        "tool_calls": [
            {
                "id": "call-7",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
            }
        ],
        "provider_extension": {"future": [1, 2, 3]},
    }

    preserved = preserve_assistant_message(original)

    assert preserved == original
    assert preserved is not original


def test_assistant_message_adds_reasoning_content_alias_without_dropping_reasoning():
    original = {"role": "assistant", "content": "", "reasoning": "legacy field", "vendor": True}

    preserved = preserve_assistant_message(original)

    assert preserved["reasoning"] == "legacy field"
    assert preserved["reasoning_content"] == "legacy field"
    assert preserved["vendor"] is True


def test_assistant_sdk_message_keeps_explicit_null_and_extra_fields():
    class SDKMessage:
        def model_dump(self, **kwargs):
            assert kwargs == {"exclude_unset": True}
            return {
                "role": "assistant",
                "content": None,
                "reasoning_content": "think",
                "tool_calls": [],
                "provider_extension": {"keep": None},
            }

    assert preserve_assistant_message(SDKMessage()) == {
        "role": "assistant",
        "content": None,
        "reasoning_content": "think",
        "tool_calls": [],
        "provider_extension": {"keep": None},
    }


def test_openai_client_sends_preserved_fields_without_filtering():
    captured = {}

    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
            },
        )

    client = OpenAI(
        base_url="http://gateway.test/v1",
        api_key="EMPTY",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assistant = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "think",
        "tool_calls": [],
        "provider_extension": {"keep": None},
    }
    try:
        client.chat.completions.create(model="test", messages=[assistant])
    finally:
        client.close()

    assert captured["messages"] == [assistant]


def test_native_tool_calls_preserve_ids_order_and_arguments():
    tool_calls = parse_native_tool_calls(
        {
            "tool_calls": [
                {
                    "id": "call-bash",
                    "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
                },
                {
                    "id": "call-submit",
                    "function": {"name": "submit", "arguments": "{}"},
                },
            ]
        }
    )

    assert [(call.id, call.name, call.arguments) for call in tool_calls] == [
        ("call-bash", "bash", {"command": "pwd"}),
        ("call-submit", "submit", {}),
    ]


def test_output_truncation_preserves_head_and_tail():
    text = "a" * 10 + "b" * 10
    assert limit_output_length(text, 10) == "aaaaa\n\n[... 10 characters elided ...]\n\nbbbbb"


def test_typed_observation_has_one_renderer_for_tool_and_protocol_messages():
    observation = ToolObservation(
        output="partial output",
        notices=(
            ToolNotice(code="command_timeout", message="Command timed out."),
            ToolNotice(code="shell_restarted", message="Shell-local state was lost."),
        ),
    )

    assert [notice.code for notice in observation.notices] == ["command_timeout", "shell_restarted"]
    assert render_tool_observation(observation) == ("partial output\n[Command timed out.]\n[Shell-local state was lost.]")
    assert observation_message(observation, tool_call_id="call-1") == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "partial output\n[Command timed out.]\n[Shell-local state was lost.]",
    }
    assert observation_message(observation) == {
        "role": "user",
        "content": "partial output\n[Command timed out.]\n[Shell-local state was lost.]",
    }


def test_recovered_shell_result_tells_agent_what_was_preserved():
    result = ShellResult(
        output="partial",
        exit_code=-1,
        recovered=True,
        recovery_reason="shell exited while running the command",
    )

    assert format_shell_result(result, command_timeout=300) == (
        "partial\n[The terminal was automatically restarted because shell exited while running the command. Filesystem changes were preserved, but shell-local state was lost.]"
    )


def test_shell_result_observation_surfaces_exit_status_and_composes_notices():
    assert format_shell_result(ShellResult(output="failed", exit_code=23), command_timeout=300) == ("failed\n[Command exited with status 23.]")

    observation = shell_result_observation(
        ShellResult(output="partial", exit_code=-1, timed_out=True, recovered=True, truncated=True),
        command_timeout=2,
    )
    assert [notice.code for notice in observation.notices] == ["command_timeout", "output_truncated"]

    truncated_failure = shell_result_observation(
        ShellResult(output="lots", exit_code=23, truncated=True),
        command_timeout=2,
    )
    assert [notice.code for notice in truncated_failure.notices] == ["output_truncated", "nonzero_exit"]


def test_rollout_timeout_env_is_a_hard_cap(monkeypatch):
    monkeypatch.setenv("RLLM_HARNESS_RUN_TIMEOUT_S", "30")
    harness = NativeReactHarness()

    assert harness._effective_timeout(Task(id="long", instruction="x", metadata={"agent_timeout": 90})) == 30
    assert harness._effective_timeout(Task(id="short", instruction="x", metadata={"agent_timeout": 10})) == 10
    assert harness._effective_timeout(Task(id="unset", instruction="x", metadata={})) == 30


def test_max_tokens_defaults_to_32768_and_is_configurable():
    assert NativeReactHarness().max_tokens == 32_768
    assert NativeReactHarness(max_tokens=4096).max_tokens == 4096


def test_max_turns_defaults_to_official_terminal_bench_budget():
    assert NativeReactHarness().max_turns == 300


def test_timeout_exception_detection_covers_backend_and_sandbox_timeouts():
    class BackendTimeoutError(RuntimeError):
        status_code = 504

    assert _is_timeout_exception(BackendTimeoutError("gateway deadline exceeded"))
    assert _is_timeout_exception(SandboxCommandTimeout("command timed out"))
    assert not _is_timeout_exception(RuntimeError("ordinary failure"))


@requires_tmux
def test_persistent_bash_preserves_cwd_environment_and_functions():
    sandbox = LocalSandbox("native-react-persistence")
    shell = PersistentBashSession(sandbox)
    try:
        shell.start()
        first = shell.run(
            "cd /tmp; export NATIVE_REACT_STATE=kept; function state_probe(){ printf function-ok; }; pwd",
            timeout=10,
        )
        second = shell.run('pwd; printf "%s\\n" "$NATIVE_REACT_STATE"; state_probe', timeout=10)
    finally:
        shell.close()
        sandbox.close()

    assert first.output == "/tmp\n"
    assert second.output == "/tmp\nkept\nfunction-ok"


@requires_tmux
def test_persistent_bash_isolates_output_when_tmux_capture_rows_are_padded():
    sandbox = _PaddedCaptureSandbox("native-react-padded-capture")
    shell = PersistentBashSession(sandbox)
    try:
        shell.start()
        first = shell.run("printf old-secret", timeout=10)
        second = shell.run("printf current-only", timeout=10)
    finally:
        shell.close()
        sandbox.close()

    assert first.output == "old-secret"
    assert first.exit_code == 0
    assert second.output == "current-only"
    assert second.exit_code == 0
    assert "old-secret" not in second.output
    assert sandbox.capture_calls == 0


@requires_tmux
def test_persistent_bash_safely_transports_non_utf8_output():
    sandbox = LocalSandbox("native-react-binary-output")
    shell = PersistentBashSession(sandbox)
    try:
        shell.start()
        result = shell.run(r"printf '\377\376\375text'", timeout=10)
    finally:
        shell.close()
        sandbox.close()

    # Command bytes cross the strict sandbox text boundary as base64, then are
    # decoded lossily at the model boundary rather than crashing the rollout.
    assert result.output == "\ufffd\ufffd\ufffdtext"
    assert result.exit_code == 0


@requires_tmux
def test_persistent_bash_round_trips_every_byte_and_marker_like_output():
    sandbox = LocalSandbox("native-react-all-bytes")
    shell = PersistentBashSession(sandbox)
    octal_bytes = "".join(f"\\{value:03o}" for value in range(256))
    marker_text = "__RLLM_NATIVE_REACT_V1__ fake 9 ok 0 0\n__RLLM_NATIVE_REACT_END_V1__ fake 9"
    try:
        shell.start()
        binary = shell.run(f"printf '{octal_bytes}'", timeout=10)
        markers = shell.run(f"printf '%s' '{marker_text}'", timeout=10)
    finally:
        shell.close()
        sandbox.close()

    assert isinstance(binary.output, str)
    assert binary.exit_code == 0
    assert markers.output == marker_text
    assert markers.exit_code == 0


@requires_tmux
def test_persistent_bash_bounds_capture_in_bytes_and_does_not_leak_between_commands():
    sandbox = LocalSandbox("native-react-bounded-output")
    shell = PersistentBashSession(sandbox, max_buffer_size=16)
    try:
        shell.start()
        exact = shell.run("printf 0123456789abcdef", timeout=10)
        oversized = shell.run("yes x | head -c 100000", timeout=10)
        empty = shell.run(":", timeout=10)
    finally:
        shell.close()
        sandbox.close()

    assert exact.output == "0123456789abcdef"
    assert exact.exit_code == 0
    assert oversized.output == "x\n" * 8
    assert oversized.exit_code == 0
    assert oversized.truncated
    assert empty.output == ""
    assert empty.exit_code == 0


@requires_tmux
def test_output_capture_file_never_grows_past_configured_limit():
    sandbox = _ResultMetadataRecordingSandbox("native-react-bounded-output-file")
    shell = PersistentBashSession(sandbox, max_buffer_size=4096)
    try:
        shell.start()
        result = shell.run("yes x | head -c 2000000", timeout=10)
    finally:
        shell.close()
        sandbox.close()

    assert result.truncated
    assert len(result.output.encode()) == 4096
    assert sandbox.output_sizes == [4096]


@requires_tmux
def test_persistent_bash_preserves_soft_wrapped_long_lines():
    sandbox = LocalSandbox("native-react-long-lines")
    shell = PersistentBashSession(sandbox, max_buffer_size=4096)
    try:
        shell.start()
        result = shell.run("printf 'a%.0s' {1..1000}", timeout=10)
    finally:
        shell.close()
        sandbox.close()

    assert result.output == "a" * 1000
    assert result.exit_code == 0


@requires_tmux
def test_persistent_bash_stages_large_multiline_and_unicode_commands():
    sandbox = LocalSandbox("native-react-large-command")
    shell = PersistentBashSession(sandbox)
    command = "# " + ("x" * 20_000) + "\ncat <<'RLLM_EOF'\n" + "literal $HOME and unicode 雪\n" + "RLLM_EOF"
    try:
        shell.start()
        result = shell.run(command, timeout=10)
    finally:
        shell.close()
        sandbox.close()

    assert result.output == "literal $HOME and unicode 雪\n"
    assert result.exit_code == 0


@requires_tmux
def test_persistent_bash_replaces_unpaired_surrogates_in_model_commands():
    sandbox = LocalSandbox("native-react-surrogate-command")
    shell = PersistentBashSession(sandbox)
    try:
        shell.start()
        result = shell.run("# unpaired: \ud800\nprintf surrogate-ok", timeout=5)
    finally:
        shell.close()
        sandbox.close()

    assert result.output == "surrogate-ok"
    assert result.exit_code == 0


@requires_tmux
def test_persistent_bash_chunks_commands_below_backend_transport_limit():
    sandbox = _CommandLengthLimitedSandbox("native-react-chunked-command", max_command_length=16_000)
    shell = PersistentBashSession(sandbox)
    command = "# " + ("雪" * 100_000) + "\nprintf 'chunked-ok'"
    try:
        shell.start()
        result = shell.run(command, timeout=10)
    finally:
        shell.close()
        sandbox.close()

    assert result.output == "chunked-ok"
    assert result.exit_code == 0
    assert sandbox.max_seen <= sandbox.max_command_length


@requires_tmux
def test_persistent_bash_recovers_terminal_mode_after_each_command():
    sandbox = LocalSandbox("native-react-terminal-mode")
    shell = PersistentBashSession(sandbox)
    try:
        shell.start()
        raw = shell.run("stty raw </dev/tty; printf raw-ok", timeout=10)
        recovered = shell.run("printf recovered", timeout=10)
    finally:
        shell.close()
        sandbox.close()

    assert raw.output == "raw-ok"
    assert raw.exit_code == 0
    assert recovered.output == "recovered"
    assert recovered.exit_code == 0


@requires_tmux
def test_persistent_bash_control_survives_persistent_fd_redirection():
    sandbox = LocalSandbox("native-react-fd-redirection")
    shell = PersistentBashSession(sandbox)
    redirected_path = f"{shell.session_dir}/redirected"
    try:
        shell.start()
        redirected = shell.run(f"exec > {redirected_path}; printf hidden", timeout=10)
        closed = shell.run("exec 1>&-; true", timeout=10)
        terminal_only = shell.run(f"exec >/dev/tty 2>/dev/tty; cat {redirected_path}; printf '|terminal'", timeout=10)
        recovered = shell.run(f"cat {redirected_path}; printf '|visible'", timeout=10)
    finally:
        shell.close()
        sandbox.close()

    assert redirected.output == ""
    assert redirected.exit_code == 0
    assert closed.output == ""
    assert closed.exit_code == 0
    assert terminal_only.output == ""
    assert terminal_only.exit_code == 0
    assert recovered.output == "hidden|visible"
    assert recovered.exit_code == 0


@requires_tmux
def test_shell_options_persist_and_errexit_recovers_like_an_interactive_shell():
    sandbox = LocalSandbox("native-react-shell-options")
    shell = PersistentBashSession(sandbox)
    try:
        shell.start()
        enabled = shell.run("set -e", timeout=5)
        observed = shell.run("printf '%s' \"$-\"", timeout=5)
        failed = shell.run("false", timeout=5)
        recovered = shell.run("printf recovered", timeout=5)
    finally:
        shell.close()
        sandbox.close()

    assert enabled.exit_code == 0
    assert "e" in observed.output
    assert failed.recovered
    assert failed.recovery_reason == "shell exited while running the command"
    assert recovered.output == "recovered"


@requires_tmux
def test_persistent_bash_close_reaps_background_jobs():
    sandbox = LocalSandbox("native-react-close-jobs")
    shell = PersistentBashSession(sandbox)
    pids = []
    states = []
    try:
        shell.start()
        result = shell.run('sleep 30 | sleep 30 & printf \'%s %s\' "$(jobs -p)" "$!"', timeout=10)
        pids = [int(value) for value in result.output.split()]
        shell.close()
        states = [sandbox.exec(f"ps -o stat= -p {pid} 2>/dev/null | tr -d ' '", timeout=10) for pid in pids]
    finally:
        shell.close()
        for pid in pids:
            sandbox.exec(f"kill -KILL {pid} 2>/dev/null || true", timeout=10)
        sandbox.close()

    assert len(pids) == 2
    assert all(not state or state.startswith("Z") for state in states)


@requires_tmux
def test_background_output_cannot_leak_into_a_later_command():
    sandbox = LocalSandbox("native-react-background-output")
    shell = PersistentBashSession(sandbox)
    try:
        shell.start()
        first = shell.run("(sleep 0.2; printf late-output) & printf current-output", timeout=5)
        time.sleep(0.4)
        second = shell.run("printf next-output", timeout=5)
    finally:
        shell.close()
        sandbox.close()

    assert first.output == "current-output"
    assert second.output == "next-output"


@requires_tmux
def test_alternate_screen_sequences_are_sanitized_without_losing_output():
    sandbox = LocalSandbox("native-react-alternate-screen")
    shell = PersistentBashSession(sandbox)
    try:
        shell.start()
        shell.run("printf old-secret", timeout=5)
        fullscreen = shell.run(r"printf '\033[?1049hfullscreen\033[?1049l'", timeout=5)
        recovered = shell.run("printf clean", timeout=5)
    finally:
        shell.close()
        sandbox.close()

    assert fullscreen.output == "fullscreen"
    assert "old-secret" not in fullscreen.output
    assert recovered.output == "clean"


@requires_tmux
def test_terminal_carriage_returns_backspaces_and_color_are_normalized():
    sandbox = LocalSandbox("native-react-terminal-controls")
    shell = PersistentBashSession(sandbox)
    try:
        shell.start()
        result = shell.run(r"printf '10%%\r20%%\nabc\bZ\n\033[31mred\033[0m\n'", timeout=5)
    finally:
        shell.close()
        sandbox.close()

    assert result.output == "20%\nabZ\nred\n"


@requires_tmux
def test_noninteractive_pager_defaults_are_stable_across_commands():
    sandbox = LocalSandbox("native-react-pager-defaults")
    shell = PersistentBashSession(sandbox)
    try:
        shell.start()
        result = shell.run(
            'printf \'%s|%s|%s|%s\' "$PAGER" "$GIT_PAGER" "$SYSTEMD_PAGER" "$MANPAGER"',
            timeout=5,
        )
    finally:
        shell.close()
        sandbox.close()

    assert result.output == "cat|cat|cat|cat"


@requires_tmux
def test_exec_replacing_shell_is_recovered_without_ending_rollout():
    sandbox = LocalSandbox("native-react-exec-recovery")
    shell = PersistentBashSession(sandbox)
    try:
        shell.start()
        replaced = shell.run("exec sh -c 'printf before-replace'", timeout=5)
        recovered = shell.run("printf after-recovery", timeout=5)
    finally:
        shell.close()
        sandbox.close()

    assert replaced.output == "before-replace"
    assert replaced.recovered
    assert not replaced.timed_out
    assert replaced.recovery_reason == "shell exited while running the command"
    assert recovered.output == "after-recovery"
    assert recovered.exit_code == 0


@requires_tmux
def test_long_running_exec_is_detected_and_recovered_without_waiting_for_process():
    sandbox = LocalSandbox("native-react-exec-timeout")
    shell = PersistentBashSession(sandbox)
    try:
        shell.start()
        started = time.monotonic()
        replaced = shell.run("exec sleep 30", timeout=0.2)
        elapsed = time.monotonic() - started
        recovered = shell.run("printf recovered", timeout=5)
    finally:
        shell.close()
        sandbox.close()

    assert replaced.recovered
    assert replaced.timed_out
    assert replaced.recovery_reason == "command timed out"
    assert elapsed < 2
    assert recovered.output == "recovered"


@requires_tmux
def test_persistent_bash_combines_stderr_and_preserves_nonzero_exit_code():
    sandbox = LocalSandbox("native-react-stderr-exit")
    shell = PersistentBashSession(sandbox)
    try:
        shell.start()
        result = shell.run("printf stdout; printf stderr >&2; (exit 23)", timeout=10)
    finally:
        shell.close()
        sandbox.close()

    assert result.output == "stdoutstderr"
    assert result.exit_code == 23
    assert not result.timed_out


@requires_tmux
def test_external_completion_wait_timeout_recovers_shell_automatically():
    sandbox = _OuterWaitTimeoutSandbox("native-react-outer-wait-timeout")
    shell = PersistentBashSession(sandbox)
    try:
        shell.start()
        result = shell.run("printf partial-output; sleep 30", timeout=0.2)

        assert result.output == "partial-output"
        assert result.exit_code == -1
        assert result.timed_out
        assert result.recovered
        assert result.recovery_reason == "command timed out"

        recovered = shell.run("printf recovered", timeout=5)
        assert recovered.output == "recovered"
        assert recovered.exit_code == 0
        assert not recovered.timed_out
    finally:
        shell.close()
        sandbox.close()


@requires_tmux
def test_transient_poll_timeout_recovers_completed_command_without_redispatch():
    sandbox = _TransientPollTimeoutSandbox("native-react-transient-poll-timeout")
    shell = PersistentBashSession(sandbox)
    try:
        shell.start()
        result = shell.run("sleep 0.2; printf recovered-result", timeout=5)
    finally:
        shell.close()
        sandbox.close()

    assert result.output == "recovered-result"
    assert result.exit_code == 0
    assert not result.timed_out
    assert sandbox.poll_failures == 1
    assert sandbox.command_dispatches == 1


@requires_tmux
def test_completion_poll_budget_includes_remote_transport_overhead():
    sandbox = _RemotePollOverheadSandbox(
        "native-react-remote-poll-overhead",
        minimum_poll_timeout=10,
    )
    shell = PersistentBashSession(sandbox)
    try:
        shell.start()
        result = shell.run("sleep 0.2; printf completed", timeout=5)
    finally:
        shell.close()
        sandbox.close()

    assert result.output == "completed"
    assert result.exit_code == 0
    assert not result.recovered
    assert sandbox.poll_failures == 0
    assert sandbox.command_dispatches == 1
    assert sandbox.poll_timeouts


@requires_tmux
def test_repeated_poll_timeouts_restart_shell_and_continue_rollout():
    sandbox = _RepeatedPollTimeoutSandbox("native-react-repeated-poll-timeout")
    shell = PersistentBashSession(sandbox)
    try:
        shell.start()
        result = shell.run("printf before-control-loss; sleep 30", timeout=10)
        recovered = shell.run("printf recovered", timeout=5)
    finally:
        shell.close()
        sandbox.close()

    assert sandbox.poll_failures == 3
    assert sandbox.command_dispatches == 2
    assert result.output == "before-control-loss"
    assert result.exit_code == -1
    assert result.recovered
    assert result.recovery_reason == "completion polling repeatedly timed out"
    assert recovered.output == "recovered"


@requires_tmux
def test_malformed_completion_responses_restart_shell_instead_of_failing_rollout():
    sandbox = _CorruptShellProtocolSandbox(
        "native-react-corrupt-completion",
        corrupt="completion",
        failures=3,
    )
    shell = PersistentBashSession(sandbox)
    try:
        shell.start()
        result = shell.run("printf command-finished", timeout=5)
        recovered = shell.run("printf next-command", timeout=5)
    finally:
        shell.close()
        sandbox.close()

    assert result.output == "command-finished"
    assert result.recovered
    assert "completion" in (result.recovery_reason or "")
    assert recovered.output == "next-command"


@requires_tmux
def test_malformed_exit_metadata_restart_shell_instead_of_failing_rollout():
    sandbox = _CorruptShellProtocolSandbox(
        "native-react-corrupt-exit-result",
        corrupt="result",
        failures=3,
    )
    shell = PersistentBashSession(sandbox)
    try:
        shell.start()
        result = shell.run("printf output-survives", timeout=5)
        recovered = shell.run("printf next-command", timeout=5)
    finally:
        shell.close()
        sandbox.close()

    assert result.output == "output-survives"
    assert result.recovered
    assert "result metadata" in (result.recovery_reason or "")
    assert recovered.output == "next-command"


@requires_tmux
def test_malformed_output_transport_restart_shell_instead_of_failing_rollout():
    sandbox = _CorruptShellProtocolSandbox(
        "native-react-corrupt-output",
        corrupt="output",
        failures=3,
    )
    shell = PersistentBashSession(sandbox)
    try:
        shell.start()
        result = shell.run("printf unavailable-output", timeout=5)
        recovered = shell.run("printf next-command", timeout=5)
    finally:
        shell.close()
        sandbox.close()

    assert result.output == "unavailable-output"
    assert result.recovered
    assert "output transport" in (result.recovery_reason or "")
    assert recovered.output == "next-command"


@requires_tmux
def test_persistently_malformed_output_becomes_recoverable_observation():
    sandbox = _CorruptShellProtocolSandbox(
        "native-react-persistent-corrupt-output",
        corrupt="output",
        failures=100,
    )
    shell = PersistentBashSession(sandbox)
    try:
        shell.start()
        result = shell.run("printf cannot-be-decoded", timeout=5)
        sandbox.failures_remaining = 0
        recovered = shell.run("printf next-command", timeout=5)
    finally:
        shell.close()
        sandbox.close()

    assert result.output == ""
    assert result.recovered
    assert result.capture_error == "the terminal output transport remained invalid"
    observation = shell_result_observation(result, command_timeout=5)
    assert any(notice.code == "output_unavailable" for notice in observation.notices)
    assert recovered.output == "next-command"


@requires_tmux
def test_deadline_poll_timeout_uses_durable_result_instead_of_redispatching():
    sandbox = _DeadlinePollTimeoutSandbox("native-react-deadline-poll-timeout")
    shell = PersistentBashSession(sandbox)
    try:
        shell.start()
        result = shell.run("sleep 0.05; printf completed-at-deadline", timeout=0.1)
    finally:
        shell.close()
        sandbox.close()

    assert result.output == "completed-at-deadline"
    assert result.exit_code == 0
    assert not result.timed_out
    assert sandbox.poll_failures == 1
    assert sandbox.command_dispatches == 1


@requires_tmux
def test_command_ignoring_sigint_is_hard_reset_with_bounded_cleanup():
    sandbox = LocalSandbox("native-react-ignore-sigint")
    shell = PersistentBashSession(sandbox)
    pid = None
    try:
        shell.start()
        started = time.monotonic()
        result = shell.run(
            "sleep 30 & echo $! > timeout-child.pid; trap '' INT; printf stuck; wait",
            timeout=0.2,
        )
        elapsed = time.monotonic() - started
        pid = int(sandbox.exec("cat timeout-child.pid", timeout=5).strip())
        state = sandbox.exec(f"ps -o stat= -p {pid} 2>/dev/null | tr -d ' '", timeout=5)

        shell.start()
        recovered = shell.run("printf recovered", timeout=5)
    finally:
        shell.close()
        if pid is not None:
            sandbox.exec(f"kill -KILL {pid} 2>/dev/null || true", timeout=5)
        sandbox.close()

    assert result.output == "stuck"
    assert result.timed_out
    assert elapsed < 6
    assert not state or state.startswith("Z")
    assert recovered.output == "recovered"
    assert recovered.exit_code == 0


@requires_tmux
def test_shell_self_exit_returns_partial_output_and_recovers_automatically():
    sandbox = LocalSandbox("native-react-shell-exit")
    shell = PersistentBashSession(sandbox)
    try:
        shell.start()
        started = time.monotonic()
        result = shell.run("printf before-exit; exit 7", timeout=5)
        elapsed = time.monotonic() - started
        recovered = shell.run("printf recovered", timeout=5)
    finally:
        shell.close()
        sandbox.close()

    assert elapsed < 2
    assert result.output == "before-exit"
    assert result.exit_code == -1
    assert result.recovered
    assert result.recovery_reason == "shell exited while running the command"
    assert recovered.output == "recovered"
    assert recovered.exit_code == 0


@requires_tmux
def test_shell_recovery_restores_its_configured_workdir(tmp_path):
    sandbox = LocalSandbox("native-react-shell-workdir")
    shell = PersistentBashSession(sandbox)
    try:
        shell.start(workdir=str(tmp_path))
        result = shell.run("cd /; exit 9", timeout=5)
        recovered = shell.run("pwd", timeout=5)
    finally:
        shell.close()
        sandbox.close()

    assert result.recovered
    assert recovered.output == f"{tmp_path}\n"


@requires_tmux
def test_shell_killed_between_commands_is_restarted_before_next_dispatch(tmp_path):
    sandbox = LocalSandbox("native-react-shell-killed-between-commands")
    shell = PersistentBashSession(sandbox)
    try:
        shell.start(workdir=str(tmp_path))
        first = shell.run("export LOST_STATE=yes; touch persisted", timeout=5)
        sandbox.exec(
            f"{shell._tmux()} kill-session -t {shell.session_name}",
            timeout=5,
        )

        recovered = shell.run(
            'printf \'pwd=%s state=%s file=%s\' "$PWD" "${LOST_STATE-unset}" "$([ -f persisted ] && echo kept || echo missing)"',
            timeout=5,
        )
    finally:
        shell.close()
        sandbox.close()

    assert first.exit_code == 0
    assert recovered.output == f"pwd={tmp_path} state=unset file=kept"
    assert recovered.exit_code == 0
    assert recovered.recovered
    assert recovered.recovery_reason == "shell was not alive before command dispatch"


@requires_tmux
def test_persistent_shells_use_isolated_tmux_sockets_and_state():
    sandbox = LocalSandbox("native-react-isolated-shells")
    first = PersistentBashSession(sandbox)
    second = PersistentBashSession(sandbox)
    try:
        first.start()
        second.start()
        first.run("export ISOLATED_STATE=first", timeout=5)
        second_result = second.run("printf '%s' \"${ISOLATED_STATE-unset}\"", timeout=5)
        first_result = first.run("printf '%s' \"$ISOLATED_STATE\"", timeout=5)
    finally:
        first.close()
        second.close()
        sandbox.close()

    assert first.socket_name != second.socket_name
    assert first_result.output == "first"
    assert second_result.output == "unset"


@requires_tmux
def test_persistent_shell_mixed_command_stress_preserves_state_and_boundaries():
    sandbox = LocalSandbox("native-react-mixed-command-stress")
    shell = PersistentBashSession(sandbox)
    try:
        shell.start()
        shell.run("counter=0", timeout=5)
        for turn in range(15):
            background = f"(sleep 0.5; printf late-{turn}) & " if turn % 5 == 0 else ""
            result = shell.run(
                f"counter=$((counter + 1)); {background}printf '\\033[32mturn-{turn}:雪\\033[0m\\rturn-{turn}:ok'",
                timeout=5,
            )
            assert result.output == f"turn-{turn}:ok"
            assert result.exit_code == 0

        final = shell.run("printf '%s' \"$counter\"", timeout=5)
    finally:
        shell.close()
        sandbox.close()

    assert final.output == "15"


def test_persistent_shell_failures_are_tagged_as_retryable_infrastructure():
    error = PersistentShellProtocolError("broken frame")

    assert isinstance(error, PersistentShellError)
    assert error._rllm_termination_reason is TerminationReason.SANDBOX_ERROR


@pytest.mark.parametrize("max_buffer_size", [0, -1, True, 1.5])
def test_persistent_bash_rejects_invalid_buffer_limits(max_buffer_size):
    with pytest.raises(ValueError, match="positive integer"):
        PersistentBashSession(SimpleNamespace(), max_buffer_size=max_buffer_size)


def test_native_react_tmux_install_is_self_contained():
    script = NativeReactHarness().install_script()

    assert "tmux" in script
    assert "harbor" not in script.lower()


@requires_tmux
def test_harness_resends_complete_assistant_and_native_tool_result(monkeypatch):
    import openai

    requests = []
    first_message = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "I should inspect with bash.",
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": '{"command":"printf command-ok"}',
                },
            }
        ],
        "provider_extension": {"keep": None},
    }

    class SDKMessage:
        def __init__(self, data):
            self.data = data

        def model_dump(self, **kwargs):
            assert kwargs == {"exclude_unset": True}
            return copy.deepcopy(self.data)

    class Completions:
        def create(self, **kwargs):
            requests.append(copy.deepcopy(kwargs))
            if len(requests) == 1:
                message = first_message
            else:
                message = {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "The command succeeded.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "submit", "arguments": "{}"},
                        }
                    ],
                }
            return SimpleNamespace(choices=[SimpleNamespace(message=SDKMessage(message))])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            assert kwargs == {
                "base_url": "http://gateway/v1",
                "api_key": "EMPTY",
                "max_retries": 0,
            }
            self.chat = SimpleNamespace(completions=Completions())

        def close(self):
            pass

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    sandbox = LocalSandbox("native-react-loop")
    try:
        episode = NativeReactHarness(max_turns=4).run(
            Task(
                id="task-1",
                instruction="inspect",
                metadata={"workdir": "/tmp", "agent_timeout": 30},
            ),
            AgentConfig(
                base_url="http://gateway/v1",
                model="qwen",
                session_uid="session-1",
            ),
            env=sandbox,
        )
    finally:
        sandbox.close()

    assert len(requests) == 2
    assert requests[0]["tools"] == NATIVE_TOOL_SCHEMAS
    assert requests[1]["tools"] == NATIVE_TOOL_SCHEMAS
    assert requests[0]["tool_choice"] == "required"
    assert requests[1]["tool_choice"] == "required"
    assert requests[0]["max_tokens"] == 32_768
    assert requests[1]["max_tokens"] == 32_768
    assert requests[1]["messages"][-2] == first_message
    assert requests[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_0",
        "content": "command-ok",
    }
    assert episode.task == "task-1"
    assert episode.metadata["native_react"] == {"turns": 2, "parse_errors": 0}


@requires_tmux
def test_harness_leaves_background_service_alive_for_verifier(monkeypatch, tmp_path):
    import openai

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    state_path = tmp_path / "service-state"
    command = (
        f"nohup python3 -m http.server {port} --bind 127.0.0.1 "
        f"> {shlex.quote(str(tmp_path / 'server.log'))} 2>&1 & "
        "service_pid=$!; "
        f'printf \'%s\\n%s\\n\' "$service_pid" "${{TMUX%%,*}}" > {shlex.quote(str(state_path))}; '
        "sleep 0.3"
    )
    responses = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_start_server",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": command}),
                    },
                }
            ],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_submit",
                    "type": "function",
                    "function": {"name": "submit", "arguments": "{}"},
                }
            ],
        },
    ]

    class SDKMessage:
        def __init__(self, data):
            self.data = data

        def model_dump(self, **kwargs):
            return copy.deepcopy(self.data)

    class Completions:
        def __init__(self):
            self.index = 0

        def create(self, **kwargs):
            response = responses[self.index]
            self.index += 1
            return SimpleNamespace(choices=[SimpleNamespace(message=SDKMessage(response))])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=Completions())

        def close(self):
            pass

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    sandbox = LocalSandbox("native-react-verifier-service")
    service_pid = None
    tmux_socket = None
    status_code = None
    try:
        NativeReactHarness(max_turns=3).run(
            Task(
                id="background-service",
                instruction="start the service",
                metadata={"workdir": str(tmp_path), "agent_timeout": 30},
            ),
            AgentConfig(
                base_url="http://gateway/v1",
                model="qwen",
                session_uid="background-service-session",
            ),
            env=sandbox,
        )
        service_pid_text, tmux_socket = state_path.read_text().splitlines()
        service_pid = int(service_pid_text)
        try:
            status_code = httpx.get(f"http://127.0.0.1:{port}", timeout=2, trust_env=False).status_code
        except httpx.HTTPError:
            status_code = None
    finally:
        if service_pid is not None:
            sandbox.exec(f"kill -KILL {service_pid} 2>/dev/null || true", timeout=10)
        if tmux_socket:
            sandbox.exec(f"tmux -S {shlex.quote(tmux_socket)} kill-server 2>/dev/null || true", timeout=10)
        sandbox.close()

    assert status_code == 200


@requires_tmux
def test_parse_retry_uses_standard_protocol_feedback_not_fake_tool_markup(monkeypatch):
    import openai

    requests = []
    responses = [
        {
            "role": "assistant",
            "content": "I forgot to call a tool.",
            "reasoning_content": "reasoning-that-must-remain-visible",
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_submit",
                    "type": "function",
                    "function": {"name": "submit", "arguments": "{}"},
                }
            ],
        },
    ]

    class SDKMessage:
        def __init__(self, data):
            self.data = data

        def model_dump(self, **kwargs):
            return copy.deepcopy(self.data)

    class Completions:
        def create(self, **kwargs):
            requests.append(copy.deepcopy(kwargs))
            return SimpleNamespace(choices=[SimpleNamespace(message=SDKMessage(responses[len(requests) - 1]))])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=Completions())

        def close(self):
            pass

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    sandbox = LocalSandbox("native-react-parse-retry")
    try:
        episode = NativeReactHarness(max_turns=3).run(
            Task(id="parse-retry", instruction="inspect", metadata={"workdir": "/tmp", "agent_timeout": 30}),
            AgentConfig(base_url="http://gateway/v1", model="qwen", session_uid="parse-retry-session"),
            env=sandbox,
        )
    finally:
        sandbox.close()

    retry_messages = requests[1]["messages"]
    assert retry_messages[-2] == responses[0]
    assert retry_messages[-1] == {
        "role": "user",
        "content": "Your last message did not contain a valid tool call. Call the `bash` or `submit` function using the provided tools.",
    }
    assert "<tool_response>" not in retry_messages[-1]["content"]
    assert episode.metadata["native_react"] == {"turns": 2, "parse_errors": 1}


@requires_tmux
def test_parse_retry_limit_counts_consecutive_failures_not_lifetime_total(monkeypatch):
    import openai

    requests = []
    responses = [
        {"role": "assistant", "content": "forgot first tool"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_bash",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command":"printf ok"}'},
                }
            ],
        },
        {"role": "assistant", "content": "forgot another tool later"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_submit",
                    "type": "function",
                    "function": {"name": "submit", "arguments": "{}"},
                }
            ],
        },
    ]

    class SDKMessage:
        def __init__(self, data):
            self.data = data

        def model_dump(self, **kwargs):
            return copy.deepcopy(self.data)

    class Completions:
        def create(self, **kwargs):
            requests.append(copy.deepcopy(kwargs))
            return SimpleNamespace(choices=[SimpleNamespace(message=SDKMessage(responses[len(requests) - 1]))])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=Completions())

        def close(self):
            pass

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    sandbox = LocalSandbox("native-react-nonconsecutive-parse-retries")
    try:
        episode = NativeReactHarness(max_turns=6, max_parse_retries=1).run(
            Task(id="parse-retries", instruction="inspect", metadata={"workdir": "/tmp", "agent_timeout": 30}),
            AgentConfig(base_url="http://gateway/v1", model="qwen", session_uid="parse-retries-session"),
            env=sandbox,
        )
    finally:
        sandbox.close()

    assert len(requests) == 4
    assert episode.termination_reason is None
    assert episode.metadata["native_react"] == {"turns": 4, "parse_errors": 2}


@requires_tmux
def test_parse_retry_exhaustion_is_a_reward_bearing_tool_protocol_error(monkeypatch):
    import openai

    requests = []
    responses = [{"role": "assistant", "content": f"invalid tool response {index}"} for index in range(1, 4)]

    class SDKMessage:
        def __init__(self, data):
            self.data = data

        def model_dump(self, **kwargs):
            return copy.deepcopy(self.data)

    class Completions:
        def create(self, **kwargs):
            requests.append(copy.deepcopy(kwargs))
            return SimpleNamespace(choices=[SimpleNamespace(message=SDKMessage(responses[len(requests) - 1]))])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=Completions())

        def close(self):
            pass

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    sandbox = LocalSandbox("native-react-parse-retry-exhaustion")
    try:
        episode = NativeReactHarness(max_turns=10, max_parse_retries=2).run(
            Task(id="parse-exhaustion", instruction="inspect", metadata={"workdir": "/tmp", "agent_timeout": 30}),
            AgentConfig(base_url="http://gateway/v1", model="qwen", session_uid="parse-exhaustion-session"),
            env=sandbox,
        )
    finally:
        sandbox.close()

    assert len(requests) == 3
    assert episode.termination_reason is TerminationReason.TOOL_PROTOCOL_ERROR
    assert episode.termination_reason not in INFRA_ERROR_REASONS
    assert episode.metadata["native_react"] == {"turns": 3, "parse_errors": 3}
    assert episode.metadata["error"]["error_type"] == "ToolProtocolError"
    assert "3 consecutive responses" in episode.metadata["error"]["message"]


@requires_tmux
def test_command_timeout_resets_shell_and_continues_rollout(monkeypatch, tmp_path):
    import openai

    requests = []
    assistant_messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_timeout",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": ("touch survives; export TRANSIENT_STATE=lost; sleep 30 & echo $! > child.pid; printf partial-output; cd /; wait")}),
                    },
                }
            ],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_after_timeout",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps(
                            {
                                "command": (
                                    "printf 'pwd=%s\\n' \"$PWD\"; "
                                    "printf 'state=%s\\n' \"${TRANSIENT_STATE-unset}\"; "
                                    "test -f survives && echo file=kept || echo file=missing; "
                                    "pid=$(cat child.pid); "
                                    "state=$(ps -o stat= -p \"$pid\" 2>/dev/null | tr -d ' '); "
                                    "case \"$state\" in ''|Z*) echo child=dead ;; *) echo child=alive:$state ;; esac"
                                )
                            }
                        ),
                    },
                }
            ],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_submit",
                    "type": "function",
                    "function": {"name": "submit", "arguments": "{}"},
                }
            ],
        },
    ]

    class SDKMessage:
        def __init__(self, data):
            self.data = data

        def model_dump(self, **kwargs):
            assert kwargs == {"exclude_unset": True}
            return copy.deepcopy(self.data)

    class Completions:
        def create(self, **kwargs):
            requests.append(copy.deepcopy(kwargs))
            return SimpleNamespace(choices=[SimpleNamespace(message=SDKMessage(assistant_messages[len(requests) - 1]))])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=Completions())

        def close(self):
            pass

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    sandbox = LocalSandbox("native-react-command-timeout")
    try:
        episode = NativeReactHarness(command_timeout=1, max_turns=4).run(
            Task(
                id="command-timeout-task",
                instruction="inspect",
                metadata={"workdir": str(tmp_path), "agent_timeout": 20},
            ),
            AgentConfig(
                base_url="http://gateway/v1",
                model="qwen",
                session_uid="command-timeout-session",
            ),
            env=sandbox,
        )
    finally:
        sandbox.close()

    assert len(requests) == 3
    assert requests[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_timeout",
        "content": ("partial-output\n[Command timed out after 1s. The terminal was automatically restarted; filesystem changes were preserved, but shell-local state was lost.]"),
    }
    assert requests[2]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_after_timeout",
        "content": f"pwd={tmp_path}\nstate=unset\nfile=kept\nchild=dead\n",
    }
    assert episode.termination_reason is None
    assert episode.metadata["native_react"] == {"turns": 3, "parse_errors": 0}


def test_sandbox_control_timeout_is_not_mislabeled_as_agent_timeout(monkeypatch):
    import openai

    import rllm.harnesses.native_react as native_react

    class FakeOpenAI:
        def __init__(self, **kwargs):
            pass

        def close(self):
            pass

    class BrokenShell:
        def __init__(self, *args, **kwargs):
            pass

        def start(self, *, workdir=None):
            try:
                raise SandboxCommandTimeout("sandbox control call timed out")
            except SandboxCommandTimeout as error:
                raise PersistentShellError("tmux startup failed") from error

        def close(self):
            pass

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(native_react, "PersistentBashSession", BrokenShell)

    with pytest.raises(PersistentShellError, match="tmux startup failed"):
        NativeReactHarness(run_timeout=60).run(
            Task(id="infra-timeout-task", instruction="inspect", metadata={"workdir": "/tmp"}),
            AgentConfig(
                base_url="http://gateway/v1",
                model="qwen",
                session_uid="infra-timeout-session",
            ),
            env=SimpleNamespace(),
        )


def test_command_timeout_at_rollout_deadline_returns_typed_timeout(monkeypatch):
    import openai

    import rllm.harnesses.native_react as native_react

    requests = []

    class SDKMessage:
        def model_dump(self, **kwargs):
            assert kwargs == {"exclude_unset": True}
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_timeout",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"command":"sleep 30"}'},
                    }
                ],
            }

    class Completions:
        def create(self, **kwargs):
            requests.append(copy.deepcopy(kwargs))
            return SimpleNamespace(choices=[SimpleNamespace(message=SDKMessage())])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=Completions())

        def close(self):
            pass

    class FakeShell:
        def __init__(self, *args, **kwargs):
            self.run_count = 0

        def start(self, *, workdir=None):
            pass

        def run(self, command, timeout):
            self.run_count += 1
            if self.run_count == 1:
                return SimpleNamespace(output="", exit_code=0, timed_out=False)
            return SimpleNamespace(output="partial-output", exit_code=-1, timed_out=True)

        def close(self):
            pass

    monotonic_values = iter([0.0, 1.0, 2.0, 11.0])
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(native_react, "PersistentBashSession", FakeShell)
    monkeypatch.setattr(native_react.time, "monotonic", lambda: next(monotonic_values))

    episode = NativeReactHarness(command_timeout=30, max_turns=4, run_timeout=10).run(
        Task(
            id="rollout-timeout-task",
            instruction="inspect",
            metadata={"workdir": "/tmp"},
        ),
        AgentConfig(
            base_url="http://gateway/v1",
            model="qwen",
            session_uid="rollout-timeout-session",
        ),
        env=SimpleNamespace(),
    )

    assert len(requests) == 1
    assert episode.termination_reason is TerminationReason.TIMEOUT
    assert episode.metadata["error"] == {
        "error_type": "AgentTimeoutError",
        "message": "Agent execution timed out after 10s",
    }


@requires_tmux
def test_api_timeout_returns_typed_timeout_episode(monkeypatch):
    import openai

    class Completions:
        def create(self, **kwargs):
            raise APITimeoutError(httpx.Request("POST", "http://gateway/v1/chat/completions"))

    class FakeOpenAI:
        def __init__(self, **kwargs):
            assert kwargs == {
                "base_url": "http://gateway/v1",
                "api_key": "EMPTY",
                "max_retries": 0,
            }
            self.chat = SimpleNamespace(completions=Completions())

        def close(self):
            pass

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    sandbox = LocalSandbox("native-react-api-timeout")
    try:
        episode = NativeReactHarness(max_turns=1).run(
            Task(
                id="timeout-task",
                instruction="inspect",
                metadata={"workdir": "/tmp", "agent_timeout": 30},
            ),
            AgentConfig(
                base_url="http://gateway/v1",
                model="qwen",
                session_uid="timeout-session",
            ),
            env=sandbox,
        )
    finally:
        sandbox.close()

    assert episode.termination_reason is TerminationReason.TIMEOUT
    assert episode.metadata["error"]["error_type"] == "AgentTimeoutError"
    assert "APITimeoutError" in episode.metadata["error"]["message"]
