"""Unit tests for the BaseCliHarness family.

A FakeSandbox captures every ``exec`` call so we can assert on the
shape of install scripts, env var construction, and config-file
contents without needing Docker.
"""

from __future__ import annotations

import json
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
    commits: list[str] = field(default_factory=list)

    def exec(self, command: str, timeout: float | None = None, user: str | None = None) -> str:
        if self.fail_on_substring and self.fail_on_substring in command:
            raise RuntimeError(f"sandbox exec failed: {self.fail_on_substring!r}")
        self.calls.append(_ExecCall(command=command, user=user, timeout=timeout))
        return self.stdout

    def commit(self, tag: str) -> None:
        """Record commits so caching tests can assert on tag shape."""
        self.commits.append(tag)

    def upload_file(self, *_args, **_kwargs) -> None:  # pragma: no cover - unused
        pass

    def upload_dir(self, *_args, **_kwargs) -> None:  # pragma: no cover - unused
        pass

    def start_agent_process(self, *_args, **_kwargs) -> None:  # pragma: no cover - unused
        pass

    def get_endpoint(self, *_args, **_kwargs):  # pragma: no cover - unused
        return ("", {})

    def close(self) -> None:  # pragma: no cover - unused
        pass


def _make_task(workdir: str = "/workspace") -> Task:
    return Task(id="t-1", instruction="fix the bug", metadata={"workdir": workdir})


def _make_config(base_url: str = "http://gw:8000/sessions/eval-0/v1", model: str = "openai/gpt-4o") -> AgentConfig:
    return AgentConfig(base_url=base_url, model=model, session_uid="eval-0")


# ---------------------------------------------------------------------------
# Lifecycle: install on sandbox-ready, run produces an Episode
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


def test_run_returns_episode_with_session_uid_and_one_step():
    h = OpenCodeHarness()
    sandbox = FakeSandbox(stdout="opencode said hi")
    h.set_sandbox(sandbox)
    config = _make_config()

    episode = h.run(_make_task(), config)

    assert episode.id == "eval-0"
    assert episode.task == "t-1"
    assert len(episode.trajectories) == 1
    traj = episode.trajectories[0]
    assert traj.uid == "eval-0"
    assert traj.name == "opencode"
    assert len(traj.steps) == 1
    assert "hi" in str(traj.steps[0].output)


def test_run_swallows_cli_failure_and_records_error_in_step():
    h = OpenCodeHarness()
    sandbox = FakeSandbox(fail_on_substring="opencode --model")
    h.set_sandbox(sandbox)

    episode = h.run(_make_task(), _make_config())

    assert len(episode.trajectories) == 1
    output = str(episode.trajectories[0].steps[0].output)
    assert "execution failed" in output


def test_invocation_omits_cd_when_no_workdir_in_metadata():
    """No explicit workdir → don't override the Dockerfile's WORKDIR.

    Hardcoding `cd /workspace` was breaking tasks (like harbor's
    hello-world, WORKDIR=/app) because the agent created files in
    /workspace but the verifier checked /app.
    """
    h = OpenCodeHarness()
    task = Task(id="t-1", instruction="hi", metadata={})  # no workdir set
    cmd = h.build_invocation("hi", task, _make_config())
    assert not cmd.startswith("cd ")
    # Allow ". $HOME/.nvm/nvm.sh" but not a leading directory change.
    assert "cd '/" not in cmd and "cd /" not in cmd


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
def test_opencode_accepts_bare_or_qualified_model(bare_model: str, expected_provider: str, expected_qualified: str):
    """rllm setup gives bare model names; opencode requires provider/model. Harness must bridge that."""
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
# MiniSweAgentHarness — provider key derivation, MSWEA_* env
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "expected_var"),
    [
        ("anthropic/claude-opus-4-1", "ANTHROPIC_API_KEY"),
        ("openai/gpt-4o", "OPENAI_API_KEY"),
        ("claude-opus-4-1", "ANTHROPIC_API_KEY"),  # bare claude name
        ("gpt-4o", "OPENAI_API_KEY"),  # bare gpt name
    ],
)
def test_mini_swe_agent_picks_provider_key_var(model: str, expected_var: str):
    h = MiniSweAgentHarness()
    env = h.build_env(_make_task(), _make_config(model=model))

    assert expected_var in env
    assert env["MSWEA_CONFIGURED"] == "true"
    assert env["MSWEA_COST_TRACKING"] == "ignore_errors"
    assert env["OPENAI_API_BASE"] == "http://gw:8000/sessions/eval-0/v1"


def test_mini_swe_agent_writes_dotenv_to_skip_v2_setup_wizard():
    """v2's setup wizard fires when ~/.config/mini-swe-agent/.env is missing,
    even with MSWEA_CONFIGURED=true in the environment. The harness must
    pre-seed this file or the agent aborts before any LLM call."""
    h = MiniSweAgentHarness()
    sandbox = FakeSandbox()
    h.set_sandbox(sandbox)
    config = _make_config(model="claude-opus-4-1")
    env = h.build_env(_make_task(), config)

    h.write_configs(_make_task(), config, env)

    write_call = next(c for c in sandbox.calls if "mini-swe-agent/.env" in c.command)
    # The dotenv must include the qualified model name, the provider API
    # key var, and the wizard-bypass flag.
    assert "MSWEA_GLOBAL_MODEL=anthropic/claude-opus-4-1" in write_call.command
    assert "ANTHROPIC_API_KEY=" in write_call.command
    assert "MSWEA_CONFIGURED=true" in write_call.command


def test_mini_swe_agent_invocation_qualifies_bare_model_name():
    """mini-swe-agent insists on provider/model — bare rllm names must be auto-prefixed."""
    h = MiniSweAgentHarness()
    cmd = h.build_invocation(
        instruction="hi",
        task=_make_task(),
        config=_make_config(model="claude-opus-4-1"),  # bare
    )
    assert "--model=anthropic/claude-opus-4-1" in cmd


def test_mini_swe_agent_dotenv_contains_gateway_base_url():
    """v2 loads the dotenv with override=True, so OPENAI_API_BASE in process
    env alone gets unset. The base URL must live in the dotenv itself,
    otherwise calls bypass the gateway and traces never appear."""
    h = MiniSweAgentHarness()
    sandbox = FakeSandbox()
    h.set_sandbox(sandbox)
    config = _make_config(base_url="http://gw:8000/sessions/eval-0/v1", model="gpt-4o")
    env = h.build_env(_make_task(), config)

    h.write_configs(_make_task(), config, env)

    write_call = next(c for c in sandbox.calls if "mini-swe-agent/.env" in c.command)
    assert "OPENAI_API_BASE=http://gw:8000/sessions/eval-0/v1" in write_call.command


def test_mini_swe_agent_invocation_does_not_break_default_config():
    """`-c key=value` replaces (rather than augments) mini.yaml in v2 — that
    breaks model construction with Pydantic ValidationError. Stay env-var driven."""
    h = MiniSweAgentHarness()
    cmd = h.build_invocation(
        instruction="hi",
        task=_make_task(),
        config=_make_config(),
    )
    assert "-c " not in cmd


def test_mini_swe_agent_parses_trajectory_into_per_turn_steps():
    """parse_episode must convert mini-swe-agent's messages array into rLLM Steps —
    one Step per assistant turn — instead of a single dump of stdout."""
    h = MiniSweAgentHarness()
    fake_traj = {
        "messages": [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "create hello.txt"},
            {"role": "assistant", "content": "I'll create it."},
            {"role": "tool", "content": "ok"},
            {"role": "assistant", "content": "Done."},
        ]
    }
    sandbox = FakeSandbox(stdout=json.dumps(fake_traj))
    h.set_sandbox(sandbox)
    steps = h.parse_episode("ignored stdout", _make_task(), _make_config())

    assert len(steps) == 2  # one per assistant turn
    assert steps[0].input == "create hello.txt"
    assert steps[0].output == "I'll create it."
    assert (steps[0].metadata or {}).get("tool_output") == "ok"
    assert steps[1].input == ""  # no pending user before the second assistant turn
    assert steps[1].output == "Done."


def test_mini_swe_agent_parse_episode_falls_back_when_trajectory_missing():
    """Read/parse errors must not lose the trial — fall back to the base one-Step dump."""
    import json as _json

    class _FailingSandbox(FakeSandbox):
        def exec(self, command, timeout=None, user=None):  # noqa: ARG002
            raise RuntimeError("file not found")

    h = MiniSweAgentHarness()
    h.set_sandbox(_FailingSandbox())
    steps = h.parse_episode("raw stdout dump", _make_task(), _make_config())
    assert len(steps) == 1
    assert "raw stdout dump" in str(steps[0].output)
    # silence the unused-import warning
    _ = _json


# ---------------------------------------------------------------------------
# Gateway bearer-token overload (cloud sandbox auth)
# ---------------------------------------------------------------------------


def test_opencode_overloads_provider_key_with_gateway_bearer_token():
    """When the gateway is publicly exposed, every provider key in the sandbox
    must be the bearer token. The gateway re-stamps with the real upstream key
    server-side. Without this, a tunnel URL is wide open to anyone."""
    h = OpenCodeHarness()
    config = AgentConfig(
        base_url="https://x.trycloudflare.com/sessions/eval-0/v1",
        model="openai/gpt-4o",
        session_uid="eval-0",
        metadata={"gateway_auth_token": "tok_abc123"},
    )
    env = h.build_env(_make_task(), config)
    assert env["OPENAI_API_KEY"] == "tok_abc123"


def test_mini_swe_agent_overloads_provider_key_with_gateway_bearer_token():
    h = MiniSweAgentHarness()
    config = AgentConfig(
        base_url="https://x.trycloudflare.com/sessions/eval-0/v1",
        model="anthropic/claude-opus-4-1",
        session_uid="eval-0",
        metadata={"gateway_auth_token": "tok_abc123"},
    )
    env = h.build_env(_make_task(), config)
    assert env["ANTHROPIC_API_KEY"] == "tok_abc123"


def test_mini_swe_agent_dotenv_carries_bearer_token():
    """v2 reads OPENAI_API_KEY from ~/.config/mini-swe-agent/.env (override=True),
    so the bearer token must land THERE — not just in process env."""
    h = MiniSweAgentHarness()
    sandbox = FakeSandbox()
    h.set_sandbox(sandbox)
    config = AgentConfig(
        base_url="https://x.trycloudflare.com/sessions/eval-0/v1",
        model="openai/gpt-4o",
        session_uid="eval-0",
        metadata={"gateway_auth_token": "tok_abc123"},
    )
    env = h.build_env(_make_task(), config)
    h.write_configs(_make_task(), config, env)

    write_call = next(c for c in sandbox.calls if "mini-swe-agent/.env" in c.command)
    assert "OPENAI_API_KEY=tok_abc123" in write_call.command


def test_no_gateway_token_falls_back_to_real_provider_key(monkeypatch):
    """Loopback gateway (no token) keeps the legacy behaviour — pass the user's real key."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-user-key")
    h = OpenCodeHarness()
    config = _make_config(model="openai/gpt-4o")  # no gateway_auth_token in metadata
    env = h.build_env(_make_task(), config)
    assert env["OPENAI_API_KEY"] == "sk-real-user-key"


# ---------------------------------------------------------------------------
# Container URL rewrite — host loopback → host.docker.internal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("input_url", "expected"),
    [
        ("http://127.0.0.1:8000/v1", "http://host.docker.internal:8000/v1"),
        ("http://localhost:8000/sessions/eval-0/v1", "http://host.docker.internal:8000/sessions/eval-0/v1"),
        ("https://api.anthropic.com/v1", "https://api.anthropic.com/v1"),  # public host: no rewrite
        ("http://192.168.1.5:8000/v1", "http://192.168.1.5:8000/v1"),  # LAN IP: no rewrite
    ],
)
def test_container_url_rewrite_for_docker_backend(input_url: str, expected: str):
    h = OpenCodeHarness()  # docker backend
    assert h._container_url(input_url) == expected


def test_container_url_passthrough_for_local_backend():
    h = OpenCodeHarness()
    h.sandbox_backend = "local"
    assert h._container_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/v1"


def test_opencode_env_uses_container_url():
    """Regression: build_env on docker backend must rewrite loopback."""
    h = OpenCodeHarness()
    config = _make_config(base_url="http://127.0.0.1:8000/sessions/eval-0/v1")
    env = h.build_env(_make_task(), config)
    assert "host.docker.internal" in env["OPENAI_BASE_URL"]


# ---------------------------------------------------------------------------
# Heredoc helper — paths and unique markers
# ---------------------------------------------------------------------------


def test_heredoc_write_creates_parent_dir_and_uses_unique_marker():
    from rllm.harnesses.cli_harness import BaseCliHarness

    cmd = BaseCliHarness._heredoc_write("/foo/bar/baz.json", '{"a": 1}')

    # shlex.quote leaves safe paths un-quoted, but the parent dir is
    # always created and the file path is exact.
    assert "mkdir -p /foo/bar" in cmd
    assert "cat > /foo/bar/baz.json" in cmd
    # Marker is per-call unique so embedded EOF in content can't terminate it.
    cmd2 = BaseCliHarness._heredoc_write("/x/y.txt", "EOF")
    assert cmd != cmd2  # different markers


# ---------------------------------------------------------------------------
# Image caching — first task installs + commits, second task reuses
# ---------------------------------------------------------------------------


def test_installed_image_tag_is_deterministic_per_base_image_and_install():
    h = OpenCodeHarness()
    t1 = h._installed_image_tag("ubuntu:24.04")
    t2 = h._installed_image_tag("ubuntu:24.04")
    t3 = h._installed_image_tag("ubuntu:22.04")
    assert t1 == t2  # stable input → stable tag
    assert t1 != t3  # different base → different tag
    assert t1.startswith("rllm-cli-opencode-")
    assert t1.endswith(":installed")


def test_installed_image_tag_changes_when_install_script_changes(monkeypatch):
    h = OpenCodeHarness()
    base_tag = h._installed_image_tag("ubuntu:24.04")

    # Pretend the install script changed (e.g., new opencode version pin).
    monkeypatch.setattr(h, "install_script", lambda: "echo NEW INSTALL && true")
    new_tag = h._installed_image_tag("ubuntu:24.04")

    assert base_tag != new_tag, "tag must invalidate when the install script changes"


def test_maybe_use_cached_image_returns_base_when_image_missing(monkeypatch):
    h = OpenCodeHarness()
    monkeypatch.setattr(h, "_docker_image_exists", lambda _tag: False)
    assert h.maybe_use_cached_image("ubuntu:24.04", "docker") == "ubuntu:24.04"


def test_maybe_use_cached_image_returns_cached_tag_when_present(monkeypatch):
    h = OpenCodeHarness()
    monkeypatch.setattr(h, "_docker_image_exists", lambda _tag: True)
    chosen = h.maybe_use_cached_image("ubuntu:24.04", "docker")
    assert chosen.startswith("rllm-cli-opencode-")
    assert chosen.endswith(":installed")


@pytest.mark.parametrize("backend", ["modal", "local", "agentcore"])
def test_pre_setup_skips_commit_for_remote_backends(backend, monkeypatch):
    """Modal/local/agentcore have no docker-commit equivalent. Image caching
    must short-circuit so we don't crash, and so cold-install runs every task
    (acceptable for MVP; future Phase 6 lifts install into Modal Image)."""
    h = OpenCodeHarness()
    sandbox = FakeSandbox()
    monkeypatch.setattr(h, "_docker_image_exists", lambda _tag: False)

    h.pre_setup(sandbox, "python:3.11-slim", backend)

    # Install still ran (we still need the CLI in the live container)…
    assert any("opencode-ai" in c.command for c in sandbox.calls)
    # …but no commit attempt.
    assert sandbox.commits == []


def test_container_url_passes_through_for_modal_backend():
    """Modal sandboxes can't reach host loopback by any name. The harness
    must hand back the URL it was given — the eval CLI's --gateway-public-url
    is responsible for making it reachable."""
    h = OpenCodeHarness()
    h.sandbox_backend = "modal"
    public = "https://abc-def.trycloudflare.com/sessions/eval-0/v1"
    assert h._container_url(public) == public  # no rewrite, no mangling
    # Loopback URLs in modal mode are also passed through unchanged —
    # rewriting them would still leave them unreachable. Caller must supply
    # a public URL.
    local = "http://127.0.0.1:8000/v1"
    assert h._container_url(local) == local


def test_maybe_use_cached_image_passes_through_for_non_docker_backend(monkeypatch):
    h = OpenCodeHarness()
    monkeypatch.setattr(h, "_docker_image_exists", lambda _tag: True)
    # Even with a "cached" image present, non-docker backends ignore it.
    assert h.maybe_use_cached_image("ubuntu:24.04", "local") == "ubuntu:24.04"
    assert h.maybe_use_cached_image("ubuntu:24.04", "modal") == "ubuntu:24.04"


def test_pre_setup_runs_install_and_commits_when_image_missing(monkeypatch):
    h = OpenCodeHarness()
    sandbox = FakeSandbox()
    monkeypatch.setattr(h, "_docker_image_exists", lambda _tag: False)

    h.pre_setup(sandbox, "ubuntu:24.04", "docker")

    # Install ran as root.
    install_calls = [c for c in sandbox.calls if "opencode-ai" in c.command]
    assert install_calls and install_calls[0].user == "root"
    # And committed exactly once with the deterministic tag.
    assert len(sandbox.commits) == 1
    assert sandbox.commits[0] == h._installed_image_tag("ubuntu:24.04")


def test_pre_setup_skips_commit_when_already_running_on_cached_image(monkeypatch):
    h = OpenCodeHarness()
    sandbox = FakeSandbox()
    monkeypatch.setattr(h, "_docker_image_exists", lambda _tag: True)

    h.pre_setup(sandbox, "ubuntu:24.04", "docker")

    # Install still runs (idempotent on a cached image — `command -v opencode` short-circuits)…
    assert any("opencode-ai" in c.command for c in sandbox.calls)
    # …but no commit, because we're already on the cached image.
    assert sandbox.commits == []


def test_pre_setup_does_not_commit_for_non_docker_backend(monkeypatch):
    h = OpenCodeHarness()
    sandbox = FakeSandbox()
    monkeypatch.setattr(h, "_docker_image_exists", lambda _tag: False)

    h.pre_setup(sandbox, "ubuntu:24.04", "local")

    assert sandbox.commits == []  # backend != docker → no commit attempt


def test_pre_setup_swallows_commit_failure(monkeypatch):
    """Commit is a perf optimization; a failure must not break the trial."""
    h = OpenCodeHarness()

    class _CommitFailSandbox(FakeSandbox):
        def commit(self, tag: str) -> None:
            raise RuntimeError("disk full")

    sandbox = _CommitFailSandbox()
    monkeypatch.setattr(h, "_docker_image_exists", lambda _tag: False)

    # Must not raise — caching is optional.
    h.pre_setup(sandbox, "ubuntu:24.04", "docker")
    assert sandbox.commits == []  # the override raises before recording


# ---------------------------------------------------------------------------
# Registry — every harness in agents.json is importable
# ---------------------------------------------------------------------------


def test_all_registered_cli_harnesses_load():
    """Smoke test: every CLI harness in agents.json imports cleanly."""
    import importlib
    import json
    from pathlib import Path

    registry_path = Path("rllm/registry/agents.json")
    data = json.loads(registry_path.read_text())

    for name in ("opencode", "mini-swe-agent"):
        entry = data["agents"][name]
        module = importlib.import_module(entry["module"])
        cls = getattr(module, entry["function"])
        assert cls.name == name
        # Subclass of the base.
        from rllm.harnesses.cli_harness import BaseCliHarness

        assert issubclass(cls, BaseCliHarness)
