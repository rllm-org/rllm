"""Unit tests for the BaseCliHarness family.

A FakeSandbox captures every ``exec`` call so we can assert on the
shape of install scripts, env var construction, and config-file
contents without needing Docker.

These harnesses return ``None`` from ``run()`` — the gateway captures
LLM calls and the engine builds the trajectory during enrichment. The
tests therefore assert on side-effects (sandbox calls, config files,
invocation strings), not on returned Steps/Trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from rllm.harnesses.mini_swe_agent import MiniSweAgentHarness
from rllm.harnesses.opencode import OpenCodeHarness
from rllm.types import AgentConfig, Task


@dataclass
class _ExecCall:
    command: str
    user: str | None
    timeout: float | None


@dataclass
class FakeSandbox:
    """Captures exec() calls and returns canned stdout."""

    stdout: str = "OK"
    calls: list[_ExecCall] = field(default_factory=list)
    fail_on_substring: str | None = None

    def exec(self, command: str, timeout: float | None = None, user: str | None = None) -> str:
        if self.fail_on_substring and self.fail_on_substring in command:
            raise RuntimeError(f"sandbox exec failed: {self.fail_on_substring!r}")
        self.calls.append(_ExecCall(command=command, user=user, timeout=timeout))
        return self.stdout

    def upload_file(self, *_args, **_kwargs) -> None:  # pragma: no cover - unused
        pass

    def upload_dir(self, *_args, **_kwargs) -> None:  # pragma: no cover - unused
        pass

    def close(self) -> None:  # pragma: no cover - unused
        pass


def _make_task(workdir: str | None = None) -> Task:
    metadata = {"workdir": workdir} if workdir else {}
    return Task(id="t-1", instruction="fix the bug", metadata=metadata)


def _make_config(base_url: str = "http://gw:8000/sessions/eval-0/v1", model: str = "openai/gpt-4o") -> AgentConfig:
    return AgentConfig(base_url=base_url, model=model, session_uid="eval-0")


# ---------------------------------------------------------------------------
# Lifecycle: install on sandbox-ready, run() returns None and execs the CLI
# ---------------------------------------------------------------------------


def test_on_sandbox_ready_runs_install_script_as_root():
    h = OpenCodeHarness()
    sandbox = FakeSandbox()
    h.set_sandbox(sandbox)
    h.on_sandbox_ready({}, _make_config())

    assert len(sandbox.calls) == 1
    call = sandbox.calls[0]
    assert call.user == "root"
    assert "opencode-ai" in call.command


def test_run_returns_none_so_gateway_drives_trajectory():
    """Harnesses don't build Episodes — the gateway captures LLM calls and
    the engine's enrichment pass populates Steps from those traces."""
    h = OpenCodeHarness()
    sandbox = FakeSandbox(stdout="opencode said hi")
    h.set_sandbox(sandbox)

    result = h.run(_make_task(), _make_config())

    assert result is None


def test_run_execs_cli_against_agent_user_with_env_prefix():
    h = OpenCodeHarness()
    sandbox = FakeSandbox()
    h.set_sandbox(sandbox)

    h.run(_make_task(), _make_config())

    # Last call is the CLI invocation; preceding calls are write_configs.
    invocation = sandbox.calls[-1]
    # Invocation runs as the agent user (None on the default image).
    assert invocation.user is None
    # Env vars are prefixed onto the command, not exported by the harness.
    assert "OPENAI_BASE_URL=" in invocation.command
    assert "opencode --model=" in invocation.command


def test_run_swallows_cli_failure_and_returns_none():
    """A CLI failure should not raise — the gateway-captured traces (up to
    the point of failure) plus the verifier still drive the reward."""
    h = OpenCodeHarness()
    sandbox = FakeSandbox(fail_on_substring="opencode --model")
    h.set_sandbox(sandbox)

    result = h.run(_make_task(), _make_config())

    assert result is None


def test_invocation_omits_cd_when_no_workdir_in_metadata():
    """No explicit workdir → don't override the Dockerfile's WORKDIR.

    Hardcoding ``cd /workspace`` was breaking tasks (like harbor's
    hello-world, WORKDIR=/app) because the agent created files in
    /workspace but the verifier checked /app.
    """
    h = OpenCodeHarness()
    cmd = h.build_invocation("hi", _make_task(), _make_config())
    assert not cmd.startswith("cd ")
    # Allow ". $HOME/.nvm/nvm.sh" but not a leading directory change.
    assert "cd '/" not in cmd and not cmd.startswith("cd /")


def test_invocation_includes_cd_when_workdir_set():
    h = OpenCodeHarness()
    cmd = h.build_invocation("hi", _make_task(workdir="/app"), _make_config())
    # shlex.quote leaves simple paths unquoted; quotes only kick in for shell metachars.
    assert cmd.startswith("cd /app && ")
    cmd_with_space = h.build_invocation("hi", _make_task(workdir="/work space"), _make_config())
    assert cmd_with_space.startswith("cd '/work space' && ")


# ---------------------------------------------------------------------------
# OpenCodeHarness — env + config file
# ---------------------------------------------------------------------------


def test_heredoc_write_rejects_paths_with_shell_variables():
    """$HOME inside _heredoc_write becomes a literal dir name due to single-quoting.
    The helper must reject these so opencode-style ``$HOME/.config/...`` paths
    don't silently land in the wrong location."""
    from rllm.harnesses.cli_harness import BaseCliHarness

    with pytest.raises(ValueError, match=r"\$VAR expansion"):
        BaseCliHarness._heredoc_write("$HOME/.config/foo.json", "{}")


def test_opencode_writes_config_via_unquoted_path_so_home_expands():
    """opencode's config file goes under $HOME/.config/opencode/opencode.json,
    so the write command must leave $HOME *unquoted* in the shell. Otherwise
    the file lands in a literal ``$HOME`` dir under cwd and opencode never
    finds it."""
    h = OpenCodeHarness()
    sandbox = FakeSandbox()
    h.set_sandbox(sandbox)
    config = _make_config(model="gpt-5.4-mini")
    env = h.build_env(_make_task(), config)

    h.write_configs(_make_task(), config, env)

    write_cmd = sandbox.calls[-1].command
    # $HOME left unquoted so the shell expands it at exec time.
    assert "mkdir -p $HOME/.config/opencode" in write_cmd
    assert "cat > $HOME/.config/opencode/opencode.json" in write_cmd
    # Defensive: path must not be single-quoted around $HOME.
    assert "'$HOME" not in write_cmd


def test_opencode_writes_config_with_custom_provider_to_bypass_models_dev():
    """opencode validates known-provider model ids against its bundled models.dev
    registry — ``gpt-5.4-mini`` (or any ``rllm setup`` custom name) raises
    ProviderModelNotFoundError under provider="openai". The harness must
    register the gateway as a custom provider via npm:@ai-sdk/openai-compatible
    so arbitrary model ids work."""
    h = OpenCodeHarness()
    sandbox = FakeSandbox()
    h.set_sandbox(sandbox)
    config = _make_config(base_url="http://gw:8000/sessions/eval-0/v1", model="openai/gpt-4o")
    env = h.build_env(_make_task(), config)

    h.write_configs(_make_task(), config, env)

    written = sandbox.calls[-1].command
    assert sandbox.calls[-1].user is None  # written as agent user
    assert "opencode.json" in written
    # Custom provider id, not the inferred "openai" — that's the whole point.
    assert '"rllm-gateway"' in written
    # npm package marks it as a generic OpenAI-shaped endpoint.
    assert '"npm": "@ai-sdk/openai-compatible"' in written
    # baseURL nested under provider.<name>.options.
    assert '"baseURL": "http://gw:8000/sessions/eval-0/v1"' in written


def test_opencode_invocation_uses_custom_provider_prefix():
    """--model flag must carry the custom provider id, not the inferred openai/anthropic."""
    h = OpenCodeHarness()
    cmd = h.build_invocation(
        instruction="hi",
        task=_make_task(),
        config=_make_config(model="gpt-5.4-mini"),
    )
    assert "--model=rllm-gateway/gpt-5.4-mini" in cmd


def test_opencode_invocation_detaches_stdin():
    """opencode blocks on its initial stdin read on Modal sandboxes — the
    invocation must redirect stdin from /dev/null."""
    h = OpenCodeHarness()
    cmd = h.build_invocation("hi", _make_task(), _make_config())
    assert "</dev/null" in cmd


@pytest.mark.parametrize(
    ("bare_model", "expected_provider", "expected_qualified"),
    [
        ("gpt-4o", "openai", "openai/gpt-4o"),
        ("gpt-5.4-mini", "openai", "openai/gpt-5.4-mini"),
        ("o1-preview", "openai", "openai/o1-preview"),
        ("claude-opus-4-1", "anthropic", "anthropic/claude-opus-4-1"),
        ("claude-haiku-4", "anthropic", "anthropic/claude-haiku-4"),
        ("gemini-1.5-pro", "google", "google/gemini-1.5-pro"),
        ("deepseek-coder", "deepseek", "deepseek/deepseek-coder"),
        ("mistral-large", "mistral", "mistral/mistral-large"),
        # Pre-qualified names round-trip unchanged.
        ("openai/gpt-4o", "openai", "openai/gpt-4o"),
        ("anthropic/claude-opus-4-1", "anthropic", "anthropic/claude-opus-4-1"),
    ],
)
def test_provider_inference(bare_model: str, expected_provider: str, expected_qualified: str):
    """rllm setup gives bare model names; opencode/mini-swe-agent require
    provider/model. Harness must bridge that."""
    h = OpenCodeHarness()
    provider, _, qualified = h._split_provider(bare_model)
    assert provider == expected_provider
    assert qualified == expected_qualified


def test_opencode_writes_provider_config_with_model_id_only():
    """opencode.json registers the bare model id under the custom provider —
    the inferred upstream provider (openai/anthropic) is informational and
    used for env-var key selection, not for opencode routing."""
    h = OpenCodeHarness()
    sandbox = FakeSandbox()
    h.set_sandbox(sandbox)
    config = _make_config(model="claude-opus-4-1")  # bare anthropic name
    env = h.build_env(_make_task(), config)
    h.write_configs(_make_task(), config, env)

    written = sandbox.calls[-1].command
    assert '"rllm-gateway"' in written
    assert '"claude-opus-4-1"' in written  # model id registered under rllm-gateway


# ---------------------------------------------------------------------------
# MiniSweAgentHarness — provider key derivation, MSWEA_* env, dotenv
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "expected_var"),
    [
        ("anthropic/claude-opus-4-1", "ANTHROPIC_API_KEY"),
        ("claude-opus-4-1", "ANTHROPIC_API_KEY"),
        ("openai/gpt-4o", "OPENAI_API_KEY"),
        ("gpt-4o", "OPENAI_API_KEY"),
        ("o1-preview", "OPENAI_API_KEY"),
        ("deepseek/deepseek-coder", "DEEPSEEK_API_KEY"),
        ("groq/llama-3", "GROQ_API_KEY"),
        # Unknown → defaults to OPENAI_API_KEY.
        ("totally-unknown", "OPENAI_API_KEY"),
    ],
)
def test_mini_swe_agent_provider_key_var(model: str, expected_var: str):
    h = MiniSweAgentHarness()
    env = h.build_env(_make_task(), _make_config(model=model))
    assert expected_var in env


def test_mini_swe_agent_anthropic_base_url_strips_v1():
    """Anthropic's litellm client appends ``/v1/messages`` itself, so
    ``ANTHROPIC_BASE_URL`` must NOT end in ``/v1`` or requests double up
    to ``/v1/v1/messages`` and Anthropic 404s."""
    h = MiniSweAgentHarness()
    env = h.build_env(_make_task(), _make_config(base_url="http://gw:8000/sessions/eval-0/v1"))
    assert env["ANTHROPIC_BASE_URL"] == "http://gw:8000/sessions/eval-0"
    assert env["OPENAI_BASE_URL"] == "http://gw:8000/sessions/eval-0/v1"


def test_mini_swe_agent_writes_dotenv_at_home_path():
    """v2 reads the dotenv on startup; the base URL must be in the file
    (not just env) because v2 loads dotenv with override=True."""
    h = MiniSweAgentHarness()
    sandbox = FakeSandbox()
    h.set_sandbox(sandbox)
    config = _make_config(base_url="http://gw:8000/sessions/eval-0/v1", model="claude-opus-4-1")
    env = h.build_env(_make_task(), config)

    h.write_configs(_make_task(), config, env)

    written = sandbox.calls[-1].command
    assert "$HOME/.config/mini-swe-agent/.env" in written
    assert "MSWEA_GLOBAL_MODEL=anthropic/claude-opus-4-1" in written
    assert "OPENAI_API_BASE=http://gw:8000/sessions/eval-0/v1" in written
    assert "ANTHROPIC_BASE_URL=http://gw:8000/sessions/eval-0" in written


def test_mini_swe_agent_invocation_uses_qualified_model():
    h = MiniSweAgentHarness()
    cmd = h.build_invocation("hi", _make_task(), _make_config(model="gpt-4o"))
    assert "--model=openai/gpt-4o" in cmd
    assert "--yolo" in cmd
    assert "--exit-immediately" in cmd


# ---------------------------------------------------------------------------
# Container URL rewrite (docker → host.docker.internal)
# ---------------------------------------------------------------------------


def test_container_url_rewrites_loopback_for_docker_backend():
    h = OpenCodeHarness()
    h.sandbox_backend = "docker"
    assert h._container_url("http://127.0.0.1:8000/v1") == "http://host.docker.internal:8000/v1"
    assert h._container_url("http://localhost:9001/sessions/x/v1") == "http://host.docker.internal:9001/sessions/x/v1"


def test_container_url_passthrough_for_non_docker_backends():
    h = OpenCodeHarness()
    h.sandbox_backend = "modal"
    assert h._container_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/v1"
    h.sandbox_backend = "local"
    assert h._container_url("http://localhost:9000/v1") == "http://localhost:9000/v1"


# ---------------------------------------------------------------------------
# Gateway auth: bearer token from config.metadata wins over env fallback
# ---------------------------------------------------------------------------


def test_gateway_api_key_uses_metadata_token_when_present():
    config = AgentConfig(
        base_url="http://gw/v1",
        model="gpt-4o",
        session_uid="eval-0",
        metadata={"gateway_auth_token": "tok-abc"},
    )
    from rllm.harnesses.cli_harness import BaseCliHarness

    assert BaseCliHarness.gateway_api_key(config, "OPENAI_API_KEY") == "tok-abc"


def test_gateway_api_key_falls_back_to_env_var(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-user-key")
    config = AgentConfig(base_url="http://gw/v1", model="gpt-4o", session_uid="eval-0")

    from rllm.harnesses.cli_harness import BaseCliHarness

    assert BaseCliHarness.gateway_api_key(config, "OPENAI_API_KEY") == "sk-real-user-key"


def test_gateway_api_key_returns_placeholder_when_env_unset(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = AgentConfig(base_url="http://gw/v1", model="gpt-4o", session_uid="eval-0")

    from rllm.harnesses.cli_harness import BaseCliHarness

    assert BaseCliHarness.gateway_api_key(config, "OPENAI_API_KEY") == "sk-rllm-gateway"


def test_opencode_uses_gateway_token_in_config_when_metadata_set():
    h = OpenCodeHarness()
    sandbox = FakeSandbox()
    h.set_sandbox(sandbox)
    config = AgentConfig(
        base_url="http://gw/sessions/eval-0/v1",
        model="gpt-4o",
        session_uid="eval-0",
        metadata={"gateway_auth_token": "tok-xyz"},
    )
    env = h.build_env(_make_task(), config)
    h.write_configs(_make_task(), config, env)

    written = sandbox.calls[-1].command
    assert '"apiKey": "tok-xyz"' in written
