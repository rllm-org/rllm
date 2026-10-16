"""MiniSweAgentHarness: runs the mini-swe-agent CLI inside the sandbox.

mini-swe-agent uses litellm under the hood, so it picks up
``OPENAI_API_BASE`` for OpenAI-shaped backends and a model-derived
provider key (``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` / …). The
gateway routes by model name regardless.

``run()`` returns ``None``; the gateway captures every LLM call and
the engine builds the trajectory.
"""

from __future__ import annotations

import json
import logging
import shlex
from contextlib import nullcontext
from urllib.parse import urlparse

from rllm.harnesses.cli_harness import BaseCliHarness
from rllm.sandbox.protocol import Sandbox
from rllm.sandbox.timeouts import sandbox_control_timeout_s
from rllm.types import AgentConfig, Episode, Task, TerminationReason, termination_reason_from_error

logger = logging.getLogger(__name__)

# Model-name prefix → provider env-var mapping. Mirrors litellm's logic
# without taking the dependency.
_PROVIDER_KEYS = (
    ("anthropic/", "ANTHROPIC_API_KEY"),
    ("claude", "ANTHROPIC_API_KEY"),  # bare claude-* model names
    ("openai/", "OPENAI_API_KEY"),
    ("gpt-", "OPENAI_API_KEY"),
    ("o1", "OPENAI_API_KEY"),
    ("deepseek/", "DEEPSEEK_API_KEY"),
    ("groq/", "GROQ_API_KEY"),
)


def _provider_key_var(model: str) -> str:
    name = model.lower()
    for prefix, var in _PROVIDER_KEYS:
        if name.startswith(prefix) or prefix in name:
            return var
    return "OPENAI_API_KEY"


# Shared by both install steps. Installing the agent needs the network even when
# the task image blocks it for the agent: some images point ``http(s)_proxy`` at a
# closed port, and some set ``SSL_CERT_DIR``/``SSL_CERT_FILE`` to paths that don't
# exist, which makes uv trust no certificates at all. Neither applies to the agent's
# own run — only to these install commands.
_INSTALL_ENV_PREAMBLE = r"""
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
if [ -n "${SSL_CERT_DIR:-}" ] && [ ! -d "$SSL_CERT_DIR" ]; then unset SSL_CERT_DIR; fi
if [ -n "${SSL_CERT_FILE:-}" ] && [ ! -f "$SSL_CERT_FILE" ]; then unset SSL_CERT_FILE; fi
"""

# System packages: always installed as root.
_SYSTEM_DEPS_SCRIPT = r"""
set -e
""" + _INSTALL_ENV_PREAMBLE + r"""
# DEBIAN_FRONTEND=noninteractive is mandatory: ``apt-get install python3``
# pulls in ``tzdata``, which triggers a debconf timezone prompt and
# hangs forever on Modal sandboxes (Docker exec falls back to
# Noninteractive automatically when there's no TTY; Modal does not).
export DEBIAN_FRONTEND=noninteractive
# Only fall back to apt when curl is missing — swebench testbeds
# have expired Ubuntu jammy repo signatures, so unconditional
# ``apt-get update`` fails with GPG errors and stops the script
# before uv even gets a chance. Not fatal: an image whose package mirror
# is gone can still get uv through python3 (see the agent install).
if ! command -v curl >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
        { apt-get update -qq && apt-get install -y -qq curl ca-certificates git; } \
            || echo "could not install curl with apt; uv will be fetched with python3" >&2
    elif command -v apk >/dev/null 2>&1; then
        apk add --no-cache curl bash ca-certificates git \
            || echo "could not install curl with apk; uv will be fetched with python3" >&2
    fi
fi
"""

# uv + mini-swe-agent: installed into ``$HOME/.local`` of whichever user the
# agent runs as (the task's ``[agent].user`` when scoped, as harbor does).
_AGENT_INSTALL_SCRIPT = r"""
set -e
""" + _INSTALL_ENV_PREAMBLE + r"""
export PATH="$HOME/.local/bin:$PATH"
if ! command -v mini-swe-agent >/dev/null 2>&1; then
    if ! command -v uv >/dev/null 2>&1; then
        # Sandbox->GitHub egress (astral.sh redirects to
        # release-assets.githubusercontent.com) intermittently resets the
        # connection on cloud backends (Modal/Daytona). Without a retry a
        # single ``curl: (35/56) Connection reset by peer`` aborts the whole
        # install under ``set -e`` and the rollout scores 0. Retry a few times
        # with backoff before giving up. Download to a file rather than
        # ``curl | sh``: a pipeline's status is ``sh``'s, and ``sh`` on empty
        # input exits 0, so a failed download would count as a success.
        # Without curl (and the installer needs curl or wget itself), fetch the
        # static uv build straight from its release with python3.
        uv_installed=0
        for attempt in 1 2 3 4 5; do
            if command -v curl >/dev/null 2>&1; then
                uv_script=$(mktemp)
                if curl -LsSf --retry 3 -o "$uv_script" https://astral.sh/uv/install.sh && sh "$uv_script"; then
                    uv_installed=1
                    break
                fi
            elif command -v python3 >/dev/null 2>&1; then
                if python3 - <<'PY'
import io, os, platform, tarfile, urllib.request
arch = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}[platform.machine().lower()]
url = "https://github.com/astral-sh/uv/releases/latest/download/uv-%s-unknown-linux-musl.tar.gz" % arch
data = urllib.request.urlopen(url, timeout=300).read()
dest = os.path.expanduser("~/.local/bin")
os.makedirs(dest, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(data)) as archive:
    for member in archive.getmembers():
        name = os.path.basename(member.name)
        if member.isfile() and name in ("uv", "uvx"):
            path = os.path.join(dest, name)
            with open(path, "wb") as out:
                out.write(archive.extractfile(member).read())
            os.chmod(path, 0o755)
PY
                then
                    uv_installed=1
                    break
                fi
            else
                echo "neither curl nor python3 is available to fetch uv" >&2
                exit 1
            fi
            echo "uv install attempt ${attempt} failed; retrying in $((attempt * 3))s" >&2
            sleep $((attempt * 3))
        done
        if [ "$uv_installed" -ne 1 ]; then
            echo "uv install failed after 5 attempts" >&2
            exit 1
        fi
    fi
    export PATH="$HOME/.local/bin:$PATH"
    # Pin a modern interpreter for the tool's ISOLATED venv. mini-swe-agent
    # uses PEP 604 ``X | Y`` type syntax (requires Python >=3.10), but
    # ``uv tool install`` otherwise builds the venv with whatever python it
    # discovers — which on many task images (e.g. R2E-Gym's Python 3.9
    # testbeds) is too old, so the CLI crashes on import with
    # ``TypeError: unsupported operand type(s) for |`` and exits before any
    # LLM call (zero traces, score 0). uv fetches a managed CPython 3.12 if
    # one isn't present; this venv is independent of the task's own python.
    #
    # Pinned: unpinned installs pick up whatever litellm is newest (1.105.1
    # crashed at launch with a circular import in ``litellm.rust_bridge``) and
    # drift mid-training. litellm and its extras match harbor's install.
    # Retried: this step downloads Python 3.12 and every wheel from the network.
    for attempt in 1 2 3; do
        if uv tool install --python 3.12 'mini-swe-agent==2.4.6' \
            --with 'litellm==1.93.2' --with orjson --with fastapi; then
            break
        fi
        if [ "$attempt" -eq 3 ]; then
            echo "mini-swe-agent install failed after 3 attempts" >&2
            exit 1
        fi
        echo "mini-swe-agent install attempt ${attempt} failed; retrying in $((attempt * 5))s" >&2
        sleep $((attempt * 5))
    done
fi
"""

_INSTALL_SCRIPT = _SYSTEM_DEPS_SCRIPT + _AGENT_INSTALL_SCRIPT


def _scoped_agent_user(task: Task, sandbox: Sandbox) -> str | None:
    """The task's ``[agent].user`` when the sandbox can scope a default user (Modal)."""
    agent_user = task.metadata.get("agent_user")
    return agent_user if agent_user and callable(getattr(sandbox, "with_default_user", None)) else None


class MiniSweAgentHarness(BaseCliHarness):
    """Run mini-swe-agent inside the sandbox."""

    name = "mini-swe-agent"
    sandbox_backend = "docker"
    stdout_log_path = "/tmp/mini-swe-agent.log"
    max_turns: int | None = None
    max_consecutive_format_errors: int | None = None
    command_timeout: int | None = None
    capture_exit_status: bool = False
    cost_limit: float | None = None
    trajectory_output_path: str = "/tmp/rllm-mini-swe-trajectory.json"

    def configure(self, overrides: dict) -> dict:
        leftovers = super().configure(overrides)
        for name in ("max_turns", "max_consecutive_format_errors", "command_timeout"):
            value = leftovers.pop(name, None)
            if value is not None:
                setattr(self, name, int(value))
        return leftovers

    def install_script(self) -> str:
        return _INSTALL_SCRIPT

    def install_in_sandbox(self, sandbox: Sandbox, task: Task, *, timeout: float) -> None:
        """Install like harbor: system packages as root, the CLI as the agent user.

        Called by ``SandboxTaskHooks`` on a cold sandbox. Without a scoped agent
        user, the whole script runs as root (the agent runs as root too).
        """
        agent_user = _scoped_agent_user(task, sandbox)
        if agent_user is None:
            sandbox.exec(_INSTALL_SCRIPT, timeout=timeout, user="root")
            return
        sandbox.exec(_SYSTEM_DEPS_SCRIPT, timeout=timeout, user="root")
        sandbox.exec(_AGENT_INSTALL_SCRIPT, timeout=timeout, user=agent_user)

    def build_env(self, task: Task, config: AgentConfig) -> dict[str, str]:
        gateway_url = config.base_url
        env: dict[str, str] = {
            # Legacy v1 wizard-skip (still honoured by some forks); v2
            # ignores it and instead checks for ~/.config/mini-swe-agent/.env
            # which we write from ``write_configs``.
            "MSWEA_CONFIGURED": "true",
            # Don't fail when the gateway-routed model isn't in litellm's cost table.
            "MSWEA_COST_TRACKING": "ignore_errors",
            "OPENAI_API_BASE": gateway_url,
            "OPENAI_BASE_URL": gateway_url,
            "ANTHROPIC_BASE_URL": gateway_url.rstrip("/").removesuffix("/v1") or gateway_url,
        }

        # Forward the provider key litellm will look up. When the
        # gateway requires inbound auth, this becomes the bearer token
        # (gateway re-stamps with the real upstream key).
        api_var = _provider_key_var(config.model)
        env[api_var] = self.gateway_api_key(config, api_var)
        return env

    def write_configs(
        self,
        sandbox: Sandbox,
        task: Task,
        config: AgentConfig,
        env: dict[str, str],
    ) -> None:
        """Write ``~/.config/mini-swe-agent/.env`` so mini-swe-agent v2 skips the setup wizard.

        v2's wizard fires whenever this file is missing — even with
        ``MSWEA_CONFIGURED=true`` in the environment. Pre-seeding it
        with the model + provider key is the only reliable bypass
        observed across versions ≥ 2.2.
        """
        _, _, qualified = self.ensure_provider_prefix(config.model)
        api_var = _provider_key_var(config.model)
        api_key = env.get(api_var, self.gateway_api_key(config, api_var))

        # Dotenv lines mini-swe-agent v2 reads on startup. The base
        # URL must live HERE (not just in process env) because v2 loads
        # the dotenv with ``override=True`` — it would otherwise unset
        # ``OPENAI_API_BASE`` we exported in :meth:`build_env`, sending
        # every call to api.openai.com and bypassing the gateway.
        gateway_url = config.base_url
        dotenv_lines = [
            f"MSWEA_GLOBAL_MODEL={qualified}",
            f"{api_var}={api_key}",
            f"OPENAI_API_BASE={gateway_url}",
            f"OPENAI_BASE_URL={gateway_url}",
            f"ANTHROPIC_BASE_URL={gateway_url.rstrip('/').removesuffix('/v1') or gateway_url}",
            "MSWEA_CONFIGURED=true",
            "MSWEA_COST_TRACKING=ignore_errors",
        ]
        content = "\n".join(dotenv_lines)
        path = "$HOME/.config/mini-swe-agent/.env"
        # ``_heredoc_write`` quotes the target path, which kills
        # ``$HOME`` expansion — write the heredoc inline instead.
        self._exec_agent(
            sandbox,
            f"mkdir -p $HOME/.config/mini-swe-agent && cat > {path} << 'MSWEA_DOTENV_EOF'\n{content}\nMSWEA_DOTENV_EOF",
            env=env,
        )

    def _read_exit_outcome(
        self, sandbox: Sandbox
    ) -> tuple[str | None, str | None, dict[str, str] | None]:
        control_timeout = sandbox_control_timeout_s()
        try:
            raw = sandbox.exec(
                f"cat {shlex.quote(self.trajectory_output_path)}",
                timeout=control_timeout,
                user=self.agent_user,
            )
            data = json.loads(raw)
        except Exception as e:
            logger.debug("Could not read mini-SWE trajectory outcome: %s", e)
            return None, None, None

        info = data.get("info", {}) if isinstance(data, dict) else {}
        status = info.get("exit_status") if isinstance(info, dict) else None
        status = str(status).strip() if status else None

        finish_reason = None
        error_extra = info if isinstance(info, dict) and (info.get("exception_str") or info.get("traceback")) else None
        messages = data.get("messages", []) if isinstance(data, dict) else []
        if isinstance(messages, list):
            for message in reversed(messages):
                if not isinstance(message, dict):
                    continue
                extra = message.get("extra")
                if not isinstance(extra, dict):
                    continue
                if status is None and extra.get("exit_status"):
                    status = str(extra["exit_status"]).strip()
                if error_extra is None and (extra.get("exception_str") or extra.get("traceback")):
                    error_extra = extra
                if finish_reason is None and extra.get("interrupt_type") == "FormatError":
                    response = extra.get("response")
                    choices = response.get("choices") if isinstance(response, dict) else None
                    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                        raw_finish_reason = choices[0].get("finish_reason")
                        finish_reason = str(raw_finish_reason).strip() if raw_finish_reason else None

        error = None
        if error_extra is not None:
            error_type = error_extra.get("exit_status") or status or "MiniSweAgentError"
            message = error_extra.get("exception_str") or ""
            traceback_text = error_extra.get("traceback") or ""
            error = {
                "error_type": str(error_type),
                "message": str(message),
            }
            if traceback_text:
                error["traceback"] = str(traceback_text)

        return status, finish_reason, error

    @staticmethod
    def _map_exit_status(
        status: str | None,
        finish_reason: str | None = None,
        error: dict[str, str] | None = None,
    ) -> TerminationReason:
        if error:
            error_text = "\n".join(str(error.get(key, "")) for key in ("message", "traceback"))
            if "context_length_exceeded" in error_text:
                return TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED
        if status == "Submitted":
            return TerminationReason.ENV_DONE
        if status == "LimitsExceeded":
            return TerminationReason.MAX_TURNS_EXCEEDED
        if status == "TimeExceeded":
            return TerminationReason.TIMEOUT
        if status == "RepeatedFormatError":
            if (finish_reason or "").lower() in {"length", "max_tokens", "max_output_tokens"}:
                return TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED
            return TerminationReason.FORMAT_ERROR
        if status in (None, "", "UserInterruption"):
            return TerminationReason.UNKNOWN
        return termination_reason_from_error(status, default=TerminationReason.ERROR)

    def run(self, task: Task, config: AgentConfig, *, env: Sandbox) -> Episode:
        # Run the CLI — and so every command it spawns — as the task's
        # declared ``[agent].user``, as harbor does; otherwise the agent runs as
        # root and can read verifier-only files the task locked to root.
        # Only sandboxes with a scoped default user (Modal) support this.
        agent_user = _scoped_agent_user(task, env)
        with env.with_default_user(agent_user) if agent_user is not None else nullcontext():
            episode = super().run(task, config, env=env)
            if not self.capture_exit_status:
                return episode
            status, finish_reason, error = self._read_exit_outcome(env)
        episode.metadata["miniswe_exit_status"] = status or "missing"
        if error is not None:
            episode.metadata["error"] = error
        if status is not None:
            episode.termination_reason = self._map_exit_status(status, finish_reason, error)
        elif episode.termination_reason is None:
            episode.termination_reason = TerminationReason.UNKNOWN
        return episode

    def build_invocation(
        self,
        instruction: str,
        task: Task,
        config: AgentConfig,
    ) -> str:
        # mini-swe-agent insists on ``provider/model``; infer the prefix
        # when the user passed a bare name from rllm setup.
        _, _, qualified = self.ensure_provider_prefix(config.model)

        # NOTE: gateway routing relies on ``OPENAI_API_BASE`` in the
        # process environment. ``-c key=value`` overrides on the CLI
        # are NOT layered on top of mini.yaml in v2 — they replace it,
        # which breaks the build with missing ``system_template`` etc.
        # The dotenv we write in :meth:`write_configs` carries the base
        # URL into the agent's environment so litellm picks it up.
        config_overrides: list[str] = []
        if self.cost_limit is not None:
            cost_limit = float(self.cost_limit)
            if cost_limit < 0:
                raise ValueError(f"cost_limit must be non-negative, got {self.cost_limit!r}")
            config_overrides.append(f"agent.cost_limit={cost_limit}")
        if self.max_turns is not None:
            max_turns = int(self.max_turns)
            if max_turns <= 0:
                raise ValueError(f"max_turns must be positive, got {self.max_turns!r}")
            config_overrides.append(f"agent.step_limit={max_turns}")
        if self.max_consecutive_format_errors is not None:
            max_format_errors = int(self.max_consecutive_format_errors)
            if max_format_errors < 0:
                raise ValueError(
                    "max_consecutive_format_errors must be non-negative, "
                    f"got {self.max_consecutive_format_errors!r}"
                )
            config_overrides.append(f"agent.max_consecutive_format_errors={max_format_errors}")
        if self.command_timeout is not None:
            command_timeout = int(self.command_timeout)
            if command_timeout <= 0:
                raise ValueError(f"command_timeout must be positive, got {self.command_timeout!r}")
            config_overrides.append(f"environment.timeout={command_timeout}")

        config_args = ""
        if config_overrides:
            # Supplying any -c flag disables mini-SWE's implicit default config,
            # so include mini.yaml explicitly before layering our overrides.
            config_args = (
                "--config=mini.yaml "
                + " ".join(f"--config={value}" for value in config_overrides)
                + " "
            )

        outcome_prefix = ""
        outcome_arg = ""
        if self.capture_exit_status:
            output_path = shlex.quote(self.trajectory_output_path)
            outcome_prefix = f"rm -f {output_path}; "
            outcome_arg = f"--output={output_path} "
        # Some task images set ``http(s)_proxy`` to a closed port so anything the
        # agent downloads fails fast. Add the gateway to the image's own
        # ``NO_PROXY`` (kept, not replaced) so model calls still go through while
        # every other request stays blocked.
        no_proxy_prefix = ""
        gateway_host = urlparse(config.base_url).hostname
        if gateway_host:
            host = shlex.quote(gateway_host)
            no_proxy_prefix = f'export NO_PROXY={host}"${{NO_PROXY:+,$NO_PROXY}}" no_proxy={host}"${{no_proxy:+,$no_proxy}}"; '
        # A cert path that doesn't exist (some images set SSL_CERT_DIR to a
        # missing /etc/ssl/certs) makes httpx trust nothing, so every HTTPS model
        # call fails. Drop only paths that don't exist; valid ones are kept.
        cert_prefix = (
            'if [ -n "${SSL_CERT_DIR:-}" ] && [ ! -d "$SSL_CERT_DIR" ]; then unset SSL_CERT_DIR; fi; '
            'if [ -n "${SSL_CERT_FILE:-}" ] && [ ! -f "$SSL_CERT_FILE" ]; then unset SSL_CERT_FILE; fi; '
        )
        return (
            f"{self._cd_prefix(task)}"
            f'export PATH="$HOME/.local/bin:$PATH"; '
            f"{no_proxy_prefix}"
            f"{cert_prefix}"
            f"{outcome_prefix}"
            f"mini-swe-agent --yolo "
            f"{config_args}"
            f"--model={shlex.quote(qualified)} "
            f"--task={shlex.quote(instruction)} "
            f"{outcome_arg}"
            f"--exit-immediately "
            f"2>&1 | tee {shlex.quote(self.stdout_log_path)}"
        )
