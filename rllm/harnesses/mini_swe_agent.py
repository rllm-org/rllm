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

from rllm.harnesses.cli_harness import BaseCliHarness
from rllm.sandbox.protocol import Sandbox
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


# Read back inside the sandbox: trajectories carry every message and reach tens
# of megabytes, while sandbox backends cap the output of a single command, so
# reading the file whole yields a silently truncated prefix that will not parse.
_TRAJECTORY_SUMMARY_SCRIPT = r"""
import json, sys

with open(sys.argv[1]) as handle:
    data = json.load(handle)
info = data.get("info") if isinstance(data.get("info"), dict) else {}
messages = data.get("messages") if isinstance(data.get("messages"), list) else []
stats = info.get("model_stats") if isinstance(info.get("model_stats"), dict) else {}
last = messages[-1] if messages and isinstance(messages[-1], dict) else {}
summary = {
    "exit_status": info.get("exit_status"),
    "finish_reason": None,
    "error": None,
    "messages": len(messages),
    "last_message_role": last.get("role"),
    "api_calls": stats.get("api_calls"),
}
if info.get("exception_str") or info.get("traceback"):
    summary["error"] = {
        "exit_status": info.get("exit_status"),
        "exception_str": info.get("exception_str"),
        "traceback": info.get("traceback"),
    }
for message in reversed(messages):
    if not isinstance(message, dict):
        continue
    extra = message.get("extra")
    if not isinstance(extra, dict):
        continue
    if not summary["exit_status"] and extra.get("exit_status"):
        summary["exit_status"] = extra["exit_status"]
    if summary["error"] is None and (extra.get("exception_str") or extra.get("traceback")):
        summary["error"] = {
            "exit_status": extra.get("exit_status"),
            "exception_str": extra.get("exception_str"),
            "traceback": extra.get("traceback"),
        }
    if summary["finish_reason"] is None and extra.get("interrupt_type") == "FormatError":
        response = extra.get("response")
        choices = response.get("choices") if isinstance(response, dict) else None
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            summary["finish_reason"] = choices[0].get("finish_reason")
if summary["error"]:
    for key in ("exception_str", "traceback"):
        value = summary["error"].get(key)
        if isinstance(value, str):
            summary["error"][key] = value[-4000:]
print(json.dumps(summary))
"""


def _provider_key_var(model: str) -> str:
    name = model.lower()
    for prefix, var in _PROVIDER_KEYS:
        if name.startswith(prefix) or prefix in name:
            return var
    return "OPENAI_API_KEY"


_INSTALL_SCRIPT = r"""
set -e
# DEBIAN_FRONTEND=noninteractive is mandatory: ``apt-get install python3``
# pulls in ``tzdata``, which triggers a debconf timezone prompt and
# hangs forever on Modal sandboxes (Docker exec falls back to
# Noninteractive automatically when there's no TTY; Modal does not).
export DEBIAN_FRONTEND=noninteractive
if ! command -v mini-swe-agent >/dev/null 2>&1; then
    # Only fall back to apt when curl is missing — swebench testbeds
    # have expired Ubuntu jammy repo signatures, so unconditional
    # ``apt-get update`` fails with GPG errors and stops the script
    # before uv even gets a chance.
    if ! command -v curl >/dev/null 2>&1; then
        if command -v apt-get >/dev/null 2>&1; then
            apt-get update -qq && apt-get install -y -qq curl ca-certificates git
        elif command -v apk >/dev/null 2>&1; then
            apk add --no-cache curl bash ca-certificates git
        fi
    fi
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        # Prefer a package index: PIP_INDEX_URL points at an internal mirror on
        # fabrics whose only egress is a proxy that 504s on astral.sh, and even
        # where astral.sh works this is the faster path. --break-system-packages
        # is the PEP 668 retry for images with an externally-managed python.
        python3 -m pip install --user -q uv 2>/dev/null \
            || python3 -m pip install --user -q --break-system-packages uv 2>/dev/null \
            || true
    fi
    if ! command -v uv >/dev/null 2>&1; then
        # Sandbox->GitHub egress (astral.sh redirects to
        # release-assets.githubusercontent.com) intermittently resets the
        # connection on cloud backends (Modal/Daytona). Without a retry a
        # single ``curl: (35/56) Connection reset by peer`` aborts the whole
        # install under ``set -e`` and the rollout scores 0. Retry a few times
        # with backoff before giving up.
        uv_installed=0
        for attempt in 1 2 3 4 5; do
            if curl -LsSf https://astral.sh/uv/install.sh | sh; then
                uv_installed=1
                break
            fi
            echo "uv install attempt ${attempt} failed; retrying in $((attempt * 3))s" >&2
            sleep $((attempt * 3))
        done
        if [ "$uv_installed" -ne 1 ]; then
            echo "uv install failed after 5 attempts" >&2
            exit 1
        fi
    fi
    # Pin a modern interpreter for the tool's ISOLATED venv: mini-swe-agent
    # needs Python >=3.10 (PEP 604 syntax), but ``uv tool install`` otherwise
    # builds the venv with whatever python it discovers. Prefer the image's own
    # python when it is new enough so uv doesn't download a managed CPython
    # 3.12 on every sandbox; fall back to the managed build otherwise.
    tool_python=3.12
    if command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
        tool_python="$(command -v python3)"
    fi
    uv tool install --python "$tool_python" mini-swe-agent
fi
"""


class MiniSweAgentHarness(BaseCliHarness):
    """Run mini-swe-agent inside the sandbox."""

    name = "mini-swe-agent"
    sandbox_backend = "docker"
    stdout_log_path = "/tmp/mini-swe-agent.log"
    max_turns: int | None = None
    max_consecutive_format_errors: int | None = None
    command_timeout: int | None = None
    capture_exit_status: bool = False
    verify_only_on_env_done: bool = False
    skipped_verifier_reward: float = 0.0
    cost_limit: float | None = None
    trajectory_output_path: str = "/tmp/rllm-mini-swe-trajectory.json"
    exit_code_path: str = "/tmp/rllm-mini-swe-exit-code"

    def install_script(self) -> str:
        return _INSTALL_SCRIPT

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

    def _failure_diagnostics(self, sandbox: Sandbox, diagnostics: dict[str, str]) -> dict[str, str]:
        """Add mini-SWE's own exit code and stdout tail to ``diagnostics``.

        Reached only when no exit status could be determined, which is otherwise
        indistinguishable from the agent being killed mid-loop: the invocation
        pipes through ``tee``, so the exec's status is ``tee``'s and is always 0.
        The recorded code is mini-swe's (128+N when a signal killed it, e.g. 137
        for an OOM kill). Each probe is independent so one failing doesn't hide
        the other.
        """
        try:
            code = sandbox.exec(f"cat {shlex.quote(self.exit_code_path)}", timeout=10, user=self.agent_user)
            diagnostics["exit_code"] = code.strip() or "empty"
        except Exception as e:
            diagnostics["exit_code"] = f"unreadable: {type(e).__name__}: {e}"
        try:
            tail = sandbox.exec(f"tail -c 2000 {shlex.quote(self.stdout_log_path)}", timeout=20, user=self.agent_user)
            diagnostics["stdout_tail"] = tail.strip() or "empty"
        except Exception as e:
            diagnostics["stdout_tail"] = f"unreadable: {type(e).__name__}: {e}"
        logger.warning("mini-SWE finished without an exit status; diagnostics: %s", diagnostics)
        return diagnostics

    def _read_exit_outcome(
        self, sandbox: Sandbox
    ) -> tuple[str | None, str | None, dict[str, str] | None, dict[str, str]]:
        diagnostics: dict[str, str] = {}
        # Any interpreter will do; mini-swe-agent's own is guaranteed to exist
        # here even in images without a system python, since it just ran.
        command = (
            'export PATH="$HOME/.local/bin:$PATH"; '
            'PY="$(command -v python3 || true)"; '
            '[ -n "$PY" ] || PY="$(head -1 "$(command -v mini-swe-agent)" | sed "s/^#!//")"; '
            f"\"$PY\" -c {shlex.quote(_TRAJECTORY_SUMMARY_SCRIPT)} {shlex.quote(self.trajectory_output_path)}"
        )
        try:
            raw = sandbox.exec(command, timeout=120, user=self.agent_user)
        except Exception as e:
            logger.warning("Could not read the mini-SWE trajectory at %s: %s", self.trajectory_output_path, e)
            diagnostics["trajectory"] = f"unreadable: {type(e).__name__}: {e}"
            return None, None, None, self._failure_diagnostics(sandbox, diagnostics)
        try:
            summary = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning("mini-SWE trajectory summary at %s is not valid JSON (%d bytes): %s", self.trajectory_output_path, len(raw), e)
            diagnostics["trajectory"] = f"unparsable ({len(raw)} bytes): {type(e).__name__}: {e}"
            return None, None, None, self._failure_diagnostics(sandbox, diagnostics)

        status = summary.get("exit_status")
        status = str(status).strip() if status else None
        finish_reason = summary.get("finish_reason")
        finish_reason = str(finish_reason).strip() if finish_reason else None

        error = None
        error_source = summary.get("error")
        if isinstance(error_source, dict):
            error_type = error_source.get("exit_status") or status or "MiniSweAgentError"
            error = {
                "error_type": str(error_type),
                "message": str(error_source.get("exception_str") or ""),
            }
            traceback_text = error_source.get("traceback")
            if traceback_text:
                error["traceback"] = str(traceback_text)

        if status is None:
            # mini-swe saves the trajectory in a ``finally`` on every iteration
            # but only writes ``info.exit_status`` once its loop appends an
            # ``exit`` message, so an empty status means it stopped mid-loop.
            # Record how far it got alongside the process-level diagnostics.
            diagnostics["trajectory"] = "present, no exit status"
            diagnostics["last_message_role"] = str(summary.get("last_message_role"))
            diagnostics["messages"] = str(summary.get("messages"))
            diagnostics["api_calls"] = str(summary.get("api_calls"))
            diagnostics = self._failure_diagnostics(sandbox, diagnostics)

        return status, finish_reason, error, diagnostics

    @staticmethod
    def _map_exit_status(status: str | None, finish_reason: str | None = None) -> TerminationReason:
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
        # Any other status is mini-swe reporting an uncaught exception by class
        # name, so the shared taxonomy classifies it (context-window rejections
        # become MAX_PROMPT_LENGTH_EXCEEDED); unknown names stay ERROR.
        return termination_reason_from_error(status, default=TerminationReason.ERROR)

    def run(self, task: Task, config: AgentConfig, *, env: Sandbox) -> Episode:
        episode = super().run(task, config, env=env)
        if not self.capture_exit_status:
            return episode

        status, finish_reason, error, diagnostics = self._read_exit_outcome(env)
        episode.metadata["miniswe_exit_status"] = status or "missing"
        if diagnostics:
            episode.metadata["miniswe_diagnostics"] = diagnostics
        if error is not None:
            episode.metadata["error"] = error
        if status is not None:
            episode.termination_reason = self._map_exit_status(status, finish_reason)
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
        # ``tee`` owns the pipeline's exit status, so record mini-swe's own before
        # it is masked. It is the only evidence that separates a signal kill
        # (128+N) from a non-zero exit or a clean one when no exit status lands
        # in the trajectory. The redirect below belongs to the group, and this
        # write goes to a file, so the log stays free of it.
        exit_code_capture = ""
        if self.capture_exit_status:
            output_path = shlex.quote(self.trajectory_output_path)
            code_path = shlex.quote(self.exit_code_path)
            outcome_prefix = f"rm -f {output_path} {code_path}; "
            outcome_arg = f"--output={output_path} "
            exit_code_capture = f"; echo $? > {code_path}"
        return (
            f"{self._cd_prefix(task)}"
            f'export PATH="$HOME/.local/bin:$PATH"; '
            f"{outcome_prefix}"
            f"{{ mini-swe-agent --yolo "
            f"{config_args}"
            f"--model={shlex.quote(qualified)} "
            f"--task={shlex.quote(instruction)} "
            f"{outcome_arg}"
            f"--exit-immediately"
            f"{exit_code_capture}; }} "
            f"2>&1 | tee {shlex.quote(self.stdout_log_path)}"
        )
