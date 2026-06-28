"""Unit tests for the BaseCliHarness family.

A FakeSandbox captures every ``exec`` call so we can assert on the
shape of install scripts, env var construction, and config-file
contents without needing Docker.

These harnesses return a lightweight outcome ``Episode`` from ``run()`` (one
empty-step Trajectory the engine enriches from gateway traces) carrying the
``termination_reason`` the run observed — ``TIMEOUT`` on the agent's wall-clock
budget, ``ERROR`` on a sandbox/exec failure, or ``None`` on a clean exit for the
engine to classify. The tests assert on that reason plus the usual side-effects
(sandbox calls, config files, invocation strings).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from rllm.harnesses.aider import AiderHarness
from rllm.harnesses.claude_code import ClaudeCodeHarness
from rllm.harnesses.codex import CodexHarness
from rllm.harnesses.kimi_cli import KimiCliHarness
from rllm.harnesses.mini_swe_agent import MiniSweAgentHarness
from rllm.harnesses.opencode import OpenCodeHarness
from rllm.harnesses.qwen_code import QwenCodeHarness
from rllm.sandbox.protocol import SandboxCommandTimeout
from rllm.types import AgentConfig, Episode, Task, TerminationReason

ALL_HARNESSES = [
    AiderHarness,
    ClaudeCodeHarness,
    CodexHarness,
    KimiCliHarness,
    MiniSweAgentHarness,
    OpenCodeHarness,
    QwenCodeHarness,
]


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
    timeout_on_substring: str | None = None

    def exec(self, command: str, timeout: float | None = None, user: str | None = None) -> str:
        if self.timeout_on_substring and self.timeout_on_substring in command:
            raise SandboxCommandTimeout(f"command timed out: {self.timeout_on_substring!r}")
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
# Lifecycle: install on sandbox-ready, run() execs the CLI and returns an
# outcome Episode whose termination_reason reflects what the run observed.
# (BaseCliHarness.run is shared by every subclass — tested once here)
# ---------------------------------------------------------------------------


def test_run_returns_empty_outcome_episode_on_clean_exit():
    """A clean exit yields an Episode with one empty-step Trajectory (the
    gateway/engine enrichment fills its Steps) and no termination_reason —
    the engine derives done/length/max-turns from the traces."""
    h = OpenCodeHarness()
    sandbox = FakeSandbox(stdout="opencode said hi")

    result = h.run(_make_task(), _make_config(), env=sandbox)

    assert isinstance(result, Episode)
    assert len(result.trajectories) == 1
    assert result.trajectories[0].name == h.name
    assert result.trajectories[0].steps == []
    assert result.termination_reason is None


def test_run_execs_cli_against_agent_user_with_env_exported():
    """Env vars must be ``export``ed (not inline ``K=V cmd`` prefix), because
    invocations like ``cd /work && claude ...`` are compound commands and
    inline assignments only bind to the first command — the auth env would
    be set for ``cd`` and lost before the CLI runs."""
    h = OpenCodeHarness()
    sandbox = FakeSandbox()

    h.run(_make_task(), _make_config(), env=sandbox)

    # Last call is the CLI invocation; preceding calls are write_configs.
    invocation = sandbox.calls[-1]
    # Invocation runs as the agent user (None on the default image).
    assert invocation.user is None
    assert "export OPENAI_BASE_URL=" in invocation.command
    assert "opencode --model=" in invocation.command


def test_run_marks_error_on_cli_failure():
    """A CLI failure should not raise — the gateway-captured traces (up to the
    point of failure) plus the verifier still drive the reward — but the
    Episode is marked ERROR (with details) so compact filtering can drop it."""
    h = OpenCodeHarness()
    sandbox = FakeSandbox(fail_on_substring="opencode --model")

    result = h.run(_make_task(), _make_config(), env=sandbox)

    assert isinstance(result, Episode)
    assert result.termination_reason == TerminationReason.ERROR
    assert result.metadata["error"]["error_type"] == "RuntimeError"
    assert "message" in result.metadata["error"]


def test_run_marks_sandbox_error_when_box_died_mid_run():
    """A dead sandbox surfaces as a generic exec failure; the is_alive() probe
    distinguishes it from a benign CLI crash and marks SANDBOX_ERROR (infra)."""
    h = OpenCodeHarness()
    sandbox = FakeSandbox(fail_on_substring="opencode --model")
    sandbox.is_alive = lambda: False  # box is gone

    result = h.run(_make_task(), _make_config(), env=sandbox)

    assert result.termination_reason == TerminationReason.SANDBOX_ERROR
    assert result.metadata["error"]["error_type"] == "RuntimeError"


def test_run_marks_error_when_cli_failed_but_box_alive():
    """A non-zero CLI exit on a *live* box stays ERROR, not SANDBOX_ERROR."""
    h = OpenCodeHarness()
    sandbox = FakeSandbox(fail_on_substring="opencode --model")
    sandbox.is_alive = lambda: True  # box is healthy; the agent just failed

    result = h.run(_make_task(), _make_config(), env=sandbox)

    assert result.termination_reason == TerminationReason.ERROR


def test_run_marks_timeout_on_budget_exhaustion():
    """Hitting the wall-clock budget (SandboxCommandTimeout) is expected, not a
    failure: the captured steps are still scored, and the run is marked TIMEOUT
    so compact filtering avoids punishing it."""
    h = OpenCodeHarness()
    sandbox = FakeSandbox(timeout_on_substring="opencode --model")

    result = h.run(_make_task(), _make_config(), env=sandbox)

    assert isinstance(result, Episode)
    assert result.termination_reason == TerminationReason.TIMEOUT
    assert "error" not in result.metadata


# ---------------------------------------------------------------------------
# Outcome sentinel (in-sandbox driver verdict) + wall-clock backstop. A driver
# like terminus2 records why it stopped; run() trusts that over the exit code
# (which ``| tee`` masks). Without a sentinel, an exec that ran ~to the budget
# is classified TIMEOUT regardless of a masked clean-looking exit.
# ---------------------------------------------------------------------------


class _DriverHarness(OpenCodeHarness):
    """OpenCode harness with a stubbed driver outcome, to drive run()'s sentinel path."""

    outcome: dict | None = None

    def _read_outcome(self, sandbox):  # type: ignore[override]
        return self.outcome


def test_run_sentinel_agent_timeout_maps_to_timeout():
    h = _DriverHarness()
    h.outcome = {"exception_type": "AgentTimeoutError", "message": "Agent execution timed out after 60.0s"}
    result = h.run(_make_task(), _make_config(), env=FakeSandbox())
    assert result.termination_reason == TerminationReason.TIMEOUT


def test_run_sentinel_verifier_timeout_distinct_from_agent_timeout():
    h = _DriverHarness()
    h.outcome = {"exception_type": "VerifierTimeoutError", "message": "verifier timed out"}
    result = h.run(_make_task(), _make_config(), env=FakeSandbox())
    assert result.termination_reason == TerminationReason.VERIFIER_TIMEOUT
    assert result.metadata["error"]["error_type"] == "VerifierTimeoutError"


def test_run_sentinel_clean_finish_beats_elapsed_backstop(monkeypatch):
    """A driver that finished cleanly is ENV_DONE even past the wall clock —
    the sentinel is authoritative, so the elapsed heuristic must not fire."""
    import rllm.harnesses.cli_harness as mod

    ticks = iter([0.0, 10_000.0])
    monkeypatch.setattr(mod.time, "monotonic", lambda: next(ticks))
    h = _DriverHarness()
    h.outcome = {}  # sentinel present, no exception → clean finish
    result = h.run(_make_task(), _make_config(), env=FakeSandbox())
    assert result.termination_reason is None  # ENV_DONE, not TIMEOUT


def test_run_elapsed_backstop_marks_timeout_when_exit_masked(monkeypatch):
    """No sentinel + ran ~to the budget with a clean-looking exit (the ``| tee``
    masking case) → TIMEOUT, reconstructed from the clock."""
    import rllm.harnesses.cli_harness as mod

    ticks = iter([0.0, 10_000.0])  # elapsed >> 0.95 * run_timeout
    monkeypatch.setattr(mod.time, "monotonic", lambda: next(ticks))
    h = OpenCodeHarness()  # base harness writes no sentinel
    result = h.run(_make_task(), _make_config(), env=FakeSandbox())
    assert result.termination_reason == TerminationReason.TIMEOUT


@pytest.mark.parametrize(
    ("workdir", "expected_prefix"),
    [
        # No explicit workdir → don't override the Dockerfile's WORKDIR.
        # Hardcoding ``cd /workspace`` was breaking tasks (like harbor's
        # hello-world, WORKDIR=/app) because the agent created files in
        # /workspace but the verifier checked /app.
        (None, None),
        # shlex.quote leaves simple paths unquoted; quotes only kick in
        # for shell metachars.
        ("/app", "cd /app && "),
        ("/work space", "cd '/work space' && "),
    ],
)
def test_invocation_cd_prefix_follows_task_workdir(workdir: str | None, expected_prefix: str | None):
    h = OpenCodeHarness()
    cmd = h.build_invocation("hi", _make_task(workdir=workdir), _make_config())
    if expected_prefix is None:
        assert not cmd.startswith("cd ")
        # Allow ". $HOME/.nvm/nvm.sh" but not a leading directory change.
        assert "cd '/" not in cmd and not cmd.startswith("cd /")
    else:
        assert cmd.startswith(expected_prefix)


@pytest.mark.parametrize("harness_cls", ALL_HARNESSES)
def test_install_script_is_nonempty_and_idempotent(harness_cls):
    """Every harness ships an install script that no-ops when the CLI is
    already present (``command -v`` guard) — re-running install on a warm
    sandbox must not reinstall or fail."""
    script = harness_cls().install_script()
    assert isinstance(script, str)
    assert script.strip()
    assert "command -v" in script


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


def test_opencode_writes_config_with_custom_provider_to_bypass_models_dev():
    """opencode validates known-provider model ids against its bundled models.dev
    registry — ``gpt-5.4-mini`` (or any ``rllm setup`` custom name) raises
    ProviderModelNotFoundError under provider="openai". The harness must
    register the gateway as a custom provider via npm:@ai-sdk/openai-compatible
    so arbitrary model ids work. Also covers the $HOME-expansion and
    gateway-token plumbing of the same config write."""
    h = OpenCodeHarness()
    sandbox = FakeSandbox()
    config = AgentConfig(
        base_url="http://gw:8000/sessions/eval-0/v1",
        model="openai/gpt-4o",
        session_uid="eval-0",
        metadata={"gateway_auth_token": "tok-xyz"},
    )
    env = h.build_env(_make_task(), config)

    h.write_configs(sandbox, _make_task(), config, env)

    written = sandbox.calls[-1].command
    assert sandbox.calls[-1].user is None  # written as agent user
    assert "opencode.json" in written
    # The config lives under $HOME/.config/opencode — $HOME must stay
    # *unquoted* so the shell expands it at exec time; single-quoting it
    # lands the file in a literal ``$HOME`` dir under cwd and opencode
    # never finds it.
    assert "mkdir -p $HOME/.config/opencode" in written
    assert "cat > $HOME/.config/opencode/opencode.json" in written
    assert "'$HOME" not in written
    # Custom provider id, not the inferred "openai" — that's the whole point.
    assert '"rllm-gateway"' in written
    # The bare model id is registered under the custom provider — the
    # inferred upstream provider is only used for env-var key selection.
    assert '"gpt-4o"' in written
    # npm package marks it as a generic OpenAI-shaped endpoint.
    assert '"npm": "@ai-sdk/openai-compatible"' in written
    # baseURL nested under provider.<name>.options.
    assert '"baseURL": "http://gw:8000/sessions/eval-0/v1"' in written
    # Gateway bearer token from config.metadata flows into apiKey.
    assert '"apiKey": "tok-xyz"' in written


def test_opencode_invocation_uses_custom_provider_prefix_and_detaches_stdin():
    """--model must carry the custom provider id, not the inferred
    openai/anthropic. And opencode blocks on its initial stdin read on
    Modal sandboxes — the invocation must redirect stdin from /dev/null."""
    h = OpenCodeHarness()
    cmd = h.build_invocation(
        instruction="hi",
        task=_make_task(),
        config=_make_config(model="gpt-5.4-mini"),
    )
    assert "--model=rllm-gateway/gpt-5.4-mini" in cmd
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


# ---------------------------------------------------------------------------
# Cross-harness: ANTHROPIC_BASE_URL must not end in /v1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("harness_cls", [MiniSweAgentHarness, ClaudeCodeHarness, AiderHarness, OpenCodeHarness])
def test_anthropic_base_url_strips_v1_suffix(harness_cls):
    """Anthropic clients (SDK / litellm) append ``/v1/messages`` themselves,
    so ``ANTHROPIC_BASE_URL`` must NOT end in ``/v1`` or requests double up
    to ``/v1/v1/messages`` and Anthropic 404s. OpenAI-style vars keep the
    ``/v1`` suffix — only the Anthropic route doubles it."""
    env = harness_cls().build_env(_make_task(), _make_config(base_url="http://gw:8000/sessions/eval-0/v1"))
    assert env["ANTHROPIC_BASE_URL"] == "http://gw:8000/sessions/eval-0"
    if "OPENAI_BASE_URL" in env:
        assert env["OPENAI_BASE_URL"] == "http://gw:8000/sessions/eval-0/v1"


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


def test_mini_swe_agent_writes_dotenv_at_home_path():
    """v2 reads the dotenv on startup; the base URL must be in the file
    (not just env) because v2 loads dotenv with override=True."""
    h = MiniSweAgentHarness()
    sandbox = FakeSandbox()
    config = _make_config(base_url="http://gw:8000/sessions/eval-0/v1", model="claude-opus-4-1")
    env = h.build_env(_make_task(), config)

    h.write_configs(sandbox, _make_task(), config, env)

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
# ClaudeCodeHarness — bypassPermissions sandbox mode
# ---------------------------------------------------------------------------


def test_claude_code_build_env_configures_unattended_sandbox_run():
    """One build_env call must establish the whole unattended contract:

    - ``IS_SANDBOX=1``: ``--permission-mode=bypassPermissions`` is a no-op
      without it — the CLI ignores the flag and exits silently on the first
      filesystem tool call (the original 'ran but no traces' symptom).
    - ``CLAUDE_CONFIG_DIR``: keeps debug logs / project sessions / statsig
      writes out of ``$HOME/.claude`` so concurrent runs don't stomp each
      other and read-only $HOME images don't fail.
    - ``ANTHROPIC_DEFAULT_{SONNET,OPUS,HAIKU}_MODEL`` + subagent alias:
      sub-agent dispatch and resumed-session continuations use these
      internal aliases; when the gateway routes to a non-Anthropic id they
      must point at the chosen model or sub-agents try to call
      api.anthropic.com directly and 404 against the bare model id.
    """
    h = ClaudeCodeHarness()
    env = h.build_env(_make_task(), _make_config(model="gpt-4o"))
    assert env["IS_SANDBOX"] == "1"
    assert env["CLAUDE_CONFIG_DIR"].startswith("/")
    for k in (
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "CLAUDE_CODE_SUBAGENT_MODEL",
    ):
        assert env[k] == "gpt-4o"


def test_claude_code_invocation_uses_print_bypass_permissions_and_local_bin_path():
    """Replaces the old ``--bare -p ... --permission-mode acceptEdits``
    shape: ``--bare`` no longer exists in recent CLI versions, and
    ``acceptEdits`` still prompts on shell tool calls. ``--print``
    is the non-interactive entrypoint; ``bypassPermissions`` (gated by
    ``IS_SANDBOX=1`` from build_env) is the unattended-eval contract.
    The official installer places ``claude`` at ``$HOME/.local/bin``,
    which isn't always on the agent shell's default PATH — the
    invocation must export it before running the binary."""
    h = ClaudeCodeHarness()
    cmd = h.build_invocation("fix the bug", _make_task(), _make_config(model="claude-opus-4-1"))
    assert "claude --verbose" in cmd
    assert "--output-format=stream-json" in cmd
    assert "--permission-mode=bypassPermissions" in cmd
    assert "--print" in cmd
    # ``--`` separator so prompts starting with ``-`` aren't reparsed as flags.
    assert "-- 'fix the bug'" in cmd
    # Old ``--bare`` flag is gone in current CLI versions.
    assert "--bare" not in cmd
    assert '"$HOME/.local/bin:$PATH"' in cmd


# ---------------------------------------------------------------------------
# CodexHarness — custom-provider config + non-interactive exec
# ---------------------------------------------------------------------------


def test_codex_build_env_sets_codex_home_and_credentials():
    """Codex reads auth from ``$CODEX_HOME/auth.json`` and the gateway
    URL from ``$CODEX_HOME/config.toml``. The env must point CODEX_HOME
    at our writable dir and keep OPENAI_API_KEY in sync with auth.json."""
    h = CodexHarness()
    env = h.build_env(_make_task(), _make_config(model="gpt-5"))
    assert env["CODEX_HOME"] == "/tmp/codex-home"
    assert env["OPENAI_API_KEY"]
    assert env["OPENAI_BASE_URL"] == "http://gw:8000/sessions/eval-0/v1"


def test_codex_writes_auth_json_and_config_toml():
    """Codex 0.118+ requires BOTH:
      - ``$CODEX_HOME/auth.json`` with ``{"OPENAI_API_KEY": "..."}``
        (env var alone is not picked up by ``codex exec``)
      - ``$CODEX_HOME/config.toml`` with ``openai_base_url = "..."``
        (the env var is ignored by recent versions)
    The earlier custom ``model_provider`` block was a silent no-op."""
    h = CodexHarness()
    sandbox = FakeSandbox()
    config = _make_config(base_url="http://gw:8000/sessions/eval-0/v1", model="gpt-5")
    env = h.build_env(_make_task(), config)

    h.write_configs(sandbox, _make_task(), config, env)

    written = sandbox.calls[-1].command
    assert sandbox.calls[-1].user is None  # written as agent user
    # $CODEX_HOME left unquoted so the shell expands it at exec time.
    assert 'mkdir -p "$CODEX_HOME"' in written
    assert 'cat > "$CODEX_HOME/auth.json"' in written
    assert 'cat > "$CODEX_HOME/config.toml"' in written
    # auth.json schema: JSON with OPENAI_API_KEY at top level.
    assert '"OPENAI_API_KEY"' in written
    # config.toml schema: just openai_base_url — no custom-provider blocks.
    assert 'openai_base_url = "http://gw:8000/sessions/eval-0/v1"' in written
    # Custom-provider keys must NOT appear — they're a silent no-op and
    # caused the original "codex runs, no traces captured" symptom.
    assert "model_provider" not in written
    assert "model_providers" not in written


def test_codex_invocation_uses_exec_with_bypass_flags_and_bare_model_id():
    """``codex exec`` is the non-interactive entrypoint. The bypass flag
    skips Codex's inner approval gates and seatbelt (we're already inside
    the rLLM sandbox). ``--json --enable unified_exec`` are required for
    non-interactive runs against the new exec engine. The ``--`` separator
    is load-bearing — without it Codex tries to parse the instruction as
    flags when it begins with ``-``. And ``rllm setup`` may give bare or
    pre-qualified names; the ``--model`` flag expects the bare id (no
    ``provider/`` prefix)."""
    h = CodexHarness()
    cmd = h.build_invocation("fix the bug", _make_task(), _make_config(model="openai/gpt-5"))
    assert "codex exec" in cmd
    assert "--dangerously-bypass-approvals-and-sandbox" in cmd
    assert "--skip-git-repo-check" in cmd
    assert "--model gpt-5" in cmd
    assert "openai/gpt-5" not in cmd
    assert "--json" in cmd
    assert "--enable unified_exec" in cmd
    assert "-- 'fix the bug'" in cmd
    assert "</dev/null" in cmd


# ---------------------------------------------------------------------------
# QwenCodeHarness — env-only routing, --yolo for unattended runs
# ---------------------------------------------------------------------------


def test_qwen_code_build_env_routes_through_openai_compat_vars():
    """Qwen Code is a gemini-cli fork that natively reads
    OPENAI_BASE_URL/OPENAI_API_KEY/OPENAI_MODEL — so unlike Codex it
    needs no config file, just env vars."""
    h = QwenCodeHarness()
    env = h.build_env(_make_task(), _make_config(model="qwen3-coder-plus"))
    assert env["OPENAI_API_KEY"]
    assert env["OPENAI_BASE_URL"] == "http://gw:8000/sessions/eval-0/v1"
    # OPENAI_MODEL carries the bare model id, not a provider-prefixed name.
    assert env["OPENAI_MODEL"] == "qwen3-coder-plus"


def test_qwen_code_invocation_uses_prompt_flag_with_yolo():
    """``-p`` is the non-interactive prompt flag (gemini-cli inheritance).
    ``--yolo`` auto-approves tool calls so the run finishes unattended."""
    h = QwenCodeHarness()
    cmd = h.build_invocation("fix the bug", _make_task(), _make_config(model="qwen3-coder-plus"))
    assert "qwen --yolo" in cmd
    assert "--model qwen3-coder-plus" in cmd
    assert " -p 'fix the bug'" in cmd
    assert "</dev/null" in cmd


# ---------------------------------------------------------------------------
# AiderHarness — litellm-backed, --yes for unattended
# ---------------------------------------------------------------------------


def test_aider_build_env_routes_via_openai_base_url_vars():
    """aider uses litellm under the hood — both OPENAI_API_BASE and
    OPENAI_BASE_URL are honored across litellm versions (Anthropic
    routing is covered by the cross-harness strip-/v1 test above)."""
    h = AiderHarness()
    env = h.build_env(_make_task(), _make_config(base_url="http://gw:8000/sessions/eval-0/v1", model="openai/gpt-4o"))
    assert env["OPENAI_API_BASE"] == "http://gw:8000/sessions/eval-0/v1"
    assert env["OPENAI_BASE_URL"] == "http://gw:8000/sessions/eval-0/v1"


def test_aider_invocation_uses_yes_no_stream_no_git():
    """``--yes`` auto-confirms diffs/shell prompts (mandatory for
    unattended). ``--no-stream`` so the gateway sees one response per
    turn rather than a chunk-per-token stream. ``--no-git`` because
    Harbor task workdirs aren't always git repos and aider would
    refuse to start."""
    h = AiderHarness()
    cmd = h.build_invocation("fix it", _make_task(), _make_config(model="gpt-4o"))
    assert "aider --yes --no-stream --no-git" in cmd
    assert "--model=openai/gpt-4o" in cmd
    assert "--message='fix it'" in cmd


# ---------------------------------------------------------------------------
# KimiCliHarness — JSON-RPC wire protocol, config-file routing
# ---------------------------------------------------------------------------


def test_kimi_cli_writes_config_with_custom_provider_block():
    """kimi reads provider+model routing from a JSON config; the harness
    registers the rLLM gateway as a custom ``openai_legacy`` provider
    (kimi's name for any plain /v1/chat/completions endpoint) and points
    the chosen model at it."""
    h = KimiCliHarness()
    sandbox = FakeSandbox()
    config = _make_config(base_url="http://gw:8000/sessions/eval-0/v1", model="gpt-4o")
    env = h.build_env(_make_task(), config)
    h.write_configs(sandbox, _make_task(), config, env)
    written = sandbox.calls[-1].command
    assert "/tmp/kimi-config.json" in written
    assert '"rllm-gateway"' in written
    # type=openai_legacy is the wire-compatible setting for plain
    # /v1/chat/completions backends (which is what the gateway exposes).
    assert '"type": "openai_legacy"' in written
    assert '"base_url": "http://gw:8000/sessions/eval-0/v1"' in written


def test_kimi_cli_invocation_pipes_jsonrpc_prompt_and_breaks_loop_on_response_id():
    """kimi reads the prompt from stdin as a JSON-RPC ``prompt`` method,
    not from argv. The field name is ``user_input`` (NOT ``text`` — the
    old field is silently dropped). ``--yolo`` auto-approves tool calls;
    ``--wire`` enables the JSON-RPC protocol.

    kimi also doesn't exit on EOF — the while-read loop watches stdout
    for the response with ``"id":"1"``, ``kill 0``s the process group
    (sleep + kimi + loop). ``trap 'exit 0' TERM`` converts the SIGTERM
    that ``kill 0`` raises into a clean exit so the engine doesn't see
    a 143 and treat the run as failed."""
    h = KimiCliHarness()
    cmd = h.build_invocation("hi", _make_task(), _make_config(model="gpt-4o"))
    assert '"jsonrpc": "2.0"' in cmd
    assert '"method": "prompt"' in cmd
    assert '"user_input": "hi"' in cmd
    # id is a string, not an int — kimi's wire parser expects "1".
    assert '"id": "1"' in cmd
    assert "--wire" in cmd
    assert "--yolo" in cmd
    assert "--config-file /tmp/kimi-config.json" in cmd
    assert "sleep 86400" in cmd
    assert "while IFS= read -r line" in cmd
    assert '*\'"id":"1"\'*' in cmd
    assert "kill 0" in cmd
    # SIGTERM → exit 0 so docker exec / engine see success.
    assert "trap 'exit 0' TERM" in cmd


# ---------------------------------------------------------------------------
# Gateway auth: bearer token from config.metadata wins over env fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("metadata", "env_value", "expected"),
    [
        # metadata token wins even when the env var is set.
        ({"gateway_auth_token": "tok-abc"}, "sk-real-user-key", "tok-abc"),
        # No metadata → fall back to the env var.
        ({}, "sk-real-user-key", "sk-real-user-key"),
        # Neither → placeholder so the CLI still starts.
        ({}, None, "sk-rllm-gateway"),
    ],
)
def test_gateway_api_key_resolution(monkeypatch, metadata: dict, env_value: str | None, expected: str):
    from rllm.harnesses.cli_harness import BaseCliHarness

    if env_value is None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", env_value)
    config = AgentConfig(base_url="http://gw/v1", model="gpt-4o", session_uid="eval-0", metadata=metadata)

    assert BaseCliHarness.gateway_api_key(config, "OPENAI_API_KEY") == expected
