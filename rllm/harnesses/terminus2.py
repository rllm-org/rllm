"""Terminus2Harness: runs Harbor's Terminus-2 agent *inside* the sandbox.

Harbor ships Terminus-2 as a "mono-tool" terminal agent: it drives a single
tmux session and calls an LLM via LiteLLM. In Harbor's own runtime the agent
process runs on the host and talks to a *separate* container through a
``BaseEnvironment`` abstraction (every interaction is an ``environment.exec``).

rLLM instead owns the sandbox (Modal here) and runs the agent *in* it — the
mini-swe-agent pattern (``llm_inside_env=True``). The trick that makes this
work without Harbor's runtime: Terminus-2 only ever touches the container
through ``environment.exec()`` / ``upload`` / ``download``. So the in-sandbox
driver hands Terminus-2 a tiny ``BaseEnvironment`` whose ``exec`` runs commands
*locally* (this very container). The agent's tmux session, edits, and the
verifier's ``tests/test.sh`` all land in the same filesystem — exactly what the
per-task sandbox-shell verifier (``/logs/verifier/reward.txt``) reads afterward.

Harbor is only used here for (a) loading the dataset and (b) the Terminus-2
agent code itself; Harbor's environment/verifier/registry runtime is bypassed.

``run()`` returns ``None``; the gateway captures every LLM call and the engine
builds the trajectory. Reward comes from rLLM's per-task verifier, not here.
"""

from __future__ import annotations

import logging
import shlex

from rllm.harnesses.cli_harness import BaseCliHarness
from rllm.sandbox.protocol import Sandbox
from rllm.types import AgentConfig, Task

logger = logging.getLogger(__name__)

# Where the isolated harbor install + driver live inside the sandbox.
_VENV_DIR = "/opt/terminus2-venv"
_DRIVER_PATH = "/opt/terminus2/driver.py"
_INSTRUCTION_PATH = "/tmp/terminus2/instruction.txt"
_LOGS_DIR = "/tmp/terminus2/logs"
# The driver records a structured outcome here (Harbor ExceptionInfo-shaped) so
# the harness can label TIMEOUT / AGENT_SETUP_TIMEOUT / ... by the agent's actual
# verdict instead of an exit code the ``| tee`` pipeline would mask.
_OUTCOME_PATH = "/tmp/terminus2/outcome.json"


def _install_script(harbor_version: str, py_version: str) -> str:
    """Idempotent install of harbor (Terminus-2) into an isolated py venv.

    harbor requires Python >= 3.12, which task images rarely ship, so we let
    ``uv`` provision a private interpreter + venv decoupled from the container's
    system Python. tmux/util-linux(`script`)/procps are baked here too so the
    agent doesn't pay an apt round-trip on its first turn (Terminus-2 would
    otherwise auto-install tmux at runtime).
    """
    return rf"""
set -e
export DEBIAN_FRONTEND=noninteractive
# Marker keeps this a no-op on a snapshot that already baked the install.
if [ -f {_VENV_DIR}/.terminus2-ready ]; then
    exit 0
fi
# tmux + `script` (util-linux) + ps (procps) are what Terminus-2's TmuxSession
# needs. Only apt when curl is missing OR tmux is missing (avoid an
# unconditional apt-get update on images with stale/expired repo signatures).
if ! command -v tmux >/dev/null 2>&1 || ! command -v script >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update -qq && apt-get install -y -qq tmux util-linux procps curl ca-certificates git
    elif command -v apk >/dev/null 2>&1; then
        apk add --no-cache tmux util-linux procps curl bash ca-certificates git
    fi
fi
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
# Private interpreter + venv; harbor and its deps live only here.
uv venv --python {shlex.quote(py_version)} {_VENV_DIR}
uv pip install --python {_VENV_DIR}/bin/python {shlex.quote("harbor==" + harbor_version)}
touch {_VENV_DIR}/.terminus2-ready
"""


class Terminus2Harness(BaseCliHarness):
    """Run Harbor's Terminus-2 agent inside the (Modal) sandbox."""

    name = "terminus2"
    # Default to Modal for this integration; override via --sandbox-backend.
    sandbox_backend = "modal"
    stdout_log_path = "/tmp/terminus2.log"

    # ---- Terminus-2 knobs (overridable via agent kwargs / configure) ----
    harbor_version: str = "0.3.0"
    terminus_python: str = "3.12"
    parser_name: str = "json"  # "json" or "xml"
    # Temperature the agent requests; in `rllm eval`/training the gateway
    # enforces its own sampling params on top (default 1.0), so this is the
    # value used only when sampling isn't gateway-enforced. Kept at 1.0 to match.
    temperature: float = 1.0
    # Per-rollout turn cap. ``None`` = don't impose one — let Harbor's own
    # (effectively unbounded) default apply, so the agent isn't artificially
    # cut short; the per-rollout run timeout (RLLM_HARNESS_RUN_TIMEOUT_S) still
    # bounds wall-clock. Set a value (e.g. via the train_*.sh scripts) to cap.
    max_turns: int | None = None
    # Asciinema recording needs an extra dep and a writable trial dir; off by
    # default since rLLM scores from the verifier, not the cast.
    record_terminal_session: bool = False

    def install_script(self) -> str:
        return _install_script(self.harbor_version, self.terminus_python)

    def build_env(self, task: Task, config: AgentConfig) -> dict[str, str]:
        gateway_url = config.base_url
        # Force the OpenAI-compatible provider so LiteLLM posts to the gateway's
        # /chat/completions. Prepend "openai/" unconditionally: LiteLLM strips
        # that one prefix and forwards the remainder verbatim as the model id —
        # which must equal what the gateway/proxy routes on (the eval proxy
        # serves exactly ``config.model``). Stripping an existing "openai/"
        # here would drop it from the forwarded id and miss the route.
        terminus_model = f"openai/{config.model}"

        env: dict[str, str] = {
            # LiteLLM reads OPENAI_* for the openai provider; api_base is also
            # passed explicitly to the agent, but set both for belt-and-braces.
            "OPENAI_API_BASE": gateway_url,
            "OPENAI_BASE_URL": gateway_url,
            "OPENAI_API_KEY": self.gateway_api_key(config, "OPENAI_API_KEY"),
            # Driver inputs.
            "RLLM_TERMINUS_MODEL": terminus_model,
            "RLLM_TERMINUS_API_BASE": gateway_url,
            "RLLM_TERMINUS_WORKDIR": str(task.metadata.get("workdir") or ""),
            "RLLM_TERMINUS_PARSER": self.parser_name,
            "RLLM_TERMINUS_TEMPERATURE": str(self.temperature),
            "RLLM_TERMINUS_RECORD": "1" if self.record_terminal_session else "0",
            "RLLM_TERMINUS_INSTRUCTION_FILE": _INSTRUCTION_PATH,
            "RLLM_TERMINUS_LOGS_DIR": _LOGS_DIR,
            "RLLM_TERMINUS_OUTCOME_FILE": _OUTCOME_PATH,
            # The driver self-limits the agent loop at this budget and records an
            # AgentTimeoutError, so the timeout is labelled correctly even though
            # the outer ``| tee`` exec masks the kill's exit code. The harness's
            # exec gets a grace window beyond this (see BaseCliHarness.run).
            "RLLM_TERMINUS_AGENT_TIMEOUT_S": str(self._effective_timeout(task)),
        }
        # Only pass a turn cap when one is set; absent var = no artificial limit.
        if self.max_turns is not None:
            env["RLLM_TERMINUS_MAX_TURNS"] = str(self.max_turns)
        return env

    def _read_outcome(self, sandbox: Sandbox) -> dict | None:
        """Read the driver's outcome sentinel (Harbor ExceptionInfo-shaped).

        Returns ``{"exception_type", "message"}`` when the driver recorded a
        failure, ``{}`` (present, no exception) on a clean finish, or ``None``
        when the file is absent/unreadable — e.g. the driver was SIGKILLed
        before it could write, in which case run() applies its elapsed backstop.
        """
        import json

        try:
            raw = sandbox.exec(f"cat {shlex.quote(_OUTCOME_PATH)} 2>/dev/null || true", timeout=15, user=self.agent_user).strip()
        except Exception as e:
            logger.debug("terminus2 outcome read failed: %s", e)
            return None
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except Exception as e:
            logger.debug("terminus2 outcome parse failed: %s", e)
            return None
        info = data.get("exception_info")
        if info:
            return {"exception_type": info.get("exception_type"), "message": info.get("exception_message", "")}
        return {}

    def write_configs(
        self,
        sandbox: Sandbox,
        task: Task,
        config: AgentConfig,
        env: dict[str, str],
    ) -> None:
        """Write the in-sandbox driver + the instruction file.

        The driver is written per-run (not baked) so it can be iterated on
        without rebuilding snapshots; it's tiny and the write is cheap.
        """
        instruction = str(task.instruction).strip()
        self._exec_agent(sandbox, self._heredoc_write(_DRIVER_PATH, _DRIVER_SCRIPT), env=env)
        self._exec_agent(sandbox, self._heredoc_write(_INSTRUCTION_PATH, instruction), env=env)

    def build_invocation(
        self,
        instruction: str,
        task: Task,
        config: AgentConfig,
    ) -> str:
        return f"{_VENV_DIR}/bin/python {_DRIVER_PATH} 2>&1 | tee {shlex.quote(self.stdout_log_path)}"


# ---------------------------------------------------------------------------
# In-sandbox driver. Runs under the isolated py3.12 venv (has harbor). Builds a
# local-exec BaseEnvironment, instantiates Terminus-2, and runs the agent loop.
# Pure Python: no shell $-expansion needed (read via os.environ), so it's safe
# inside a quoted heredoc.
# ---------------------------------------------------------------------------
_DRIVER_SCRIPT = r'''
import asyncio
import json
import logging
import os
import shutil
import traceback
from datetime import datetime
from pathlib import Path

from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.agent.context import AgentContext
from harbor.models.environment_type import EnvironmentType
from harbor.trial.errors import AgentSetupTimeoutError, AgentTimeoutError

log = logging.getLogger("terminus2-driver")


class LocalEnvironment(BaseEnvironment):
    """A BaseEnvironment whose exec runs commands on THIS machine (the sandbox).

    Skips Harbor's heavy __init__/validation — Terminus-2 only consumes the
    documented BaseEnvironment surface (exec, upload/download, default_user,
    session_id, is_dir). Commands run as the current user (root on Modal),
    matching ModalSandbox.exec which ignores the user arg.
    """

    def __init__(self, workdir=None, session_id="terminus2-local"):
        self.session_id = session_id
        self.environment_name = session_id
        self.default_user = None
        self._workdir = workdir or None

    @staticmethod
    def type():
        return EnvironmentType.MODAL  # arbitrary; Terminus-2 never branches on it

    @property
    def is_mounted(self):
        return True

    @property
    def supports_gpus(self):
        return False

    @property
    def can_disable_internet(self):
        return False

    def _validate_definition(self):
        return None

    async def start(self, force_build=False):
        return None

    async def stop(self, delete=False):
        return None

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        run_cwd = cwd or self._workdir or None
        full_env = dict(os.environ)
        if env:
            full_env.update({k: str(v) for k, v in env.items()})
        proc = await asyncio.create_subprocess_exec(
            "bash", "-c", command,
            cwd=run_cwd,
            env=full_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
            return ExecResult(stdout="", stderr="command timed out", return_code=124)
        return ExecResult(
            stdout=out.decode("utf-8", "replace"),
            stderr=err.decode("utf-8", "replace"),
            return_code=proc.returncode if proc.returncode is not None else -1,
        )

    async def upload_file(self, source_path, target_path):
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(source_path), str(target_path))

    async def upload_dir(self, source_dir, target_dir):
        shutil.copytree(str(source_dir), str(target_dir), dirs_exist_ok=True)

    async def download_file(self, source_path, target_path):
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(source_path), str(target_path))

    async def download_dir(self, source_dir, target_dir):
        shutil.copytree(str(source_dir), str(target_dir), dirs_exist_ok=True)


async def _main():
    instruction = Path(os.environ["RLLM_TERMINUS_INSTRUCTION_FILE"]).read_text()
    model = os.environ["RLLM_TERMINUS_MODEL"]
    api_base = os.environ.get("RLLM_TERMINUS_API_BASE") or None
    workdir = os.environ.get("RLLM_TERMINUS_WORKDIR") or None
    parser = os.environ.get("RLLM_TERMINUS_PARSER", "json")
    # Unset/empty = no artificial cap; Harbor's Terminus2 treats max_turns=None
    # as its own (effectively unbounded) default.
    _max_turns = os.environ.get("RLLM_TERMINUS_MAX_TURNS")
    max_turns = int(_max_turns) if _max_turns else None
    temperature = float(os.environ.get("RLLM_TERMINUS_TEMPERATURE", "1.0"))
    record = os.environ.get("RLLM_TERMINUS_RECORD", "0") == "1"
    logs_dir = Path(os.environ.get("RLLM_TERMINUS_LOGS_DIR", "/tmp/terminus2/logs"))
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Permissive model_info: gateway-routed model names aren't in LiteLLM's cost
    # table, which would otherwise spam warnings and mis-estimate context.
    model_info = {"max_input_tokens": 200000, "max_output_tokens": 16384}

    env = LocalEnvironment(workdir=workdir)
    agent = Terminus2(
        logs_dir=logs_dir,
        model_name=model,
        api_base=api_base,
        parser_name=parser,
        max_turns=max_turns,
        temperature=temperature,
        record_terminal_session=record,
        model_info=model_info,
        suppress_max_turns_warning=True,
    )
    ctx = AgentContext()
    # Budgets: self-limit the agent loop so a wall-clock kill is recorded as a
    # typed AgentTimeoutError (mirrors harbor Trial), instead of relying on the
    # outer exec's exit code which the `| tee` pipeline masks. 0/unset = no cap.
    agent_timeout = float(os.environ.get("RLLM_TERMINUS_AGENT_TIMEOUT_S", "0")) or None
    setup_timeout = float(os.environ.get("RLLM_TERMINUS_SETUP_TIMEOUT_S", "0")) or None
    outcome_file = os.environ.get("RLLM_TERMINUS_OUTCOME_FILE")
    log.info("terminus2-driver: model=%s workdir=%s parser=%s max_turns=%s agent_timeout=%s", model, workdir, parser, max_turns, agent_timeout)

    exc_info = None
    try:
        if setup_timeout:
            try:
                await asyncio.wait_for(agent.setup(env), timeout=setup_timeout)
            except asyncio.TimeoutError:
                raise AgentSetupTimeoutError(f"Agent setup timed out after {setup_timeout}s")
        else:
            await agent.setup(env)
        if agent_timeout:
            try:
                await asyncio.wait_for(agent.run(instruction, env, ctx), timeout=agent_timeout)
            except asyncio.TimeoutError:
                raise AgentTimeoutError(f"Agent execution timed out after {agent_timeout}s")
        else:
            await agent.run(instruction, env, ctx)
    except BaseException as e:  # noqa: BLE001 — record the verdict, never crash the exec
        # Captures harbor's typed timeouts plus ContextLengthExceededError /
        # OutputLengthExceededError raised inside agent.run.
        exc_info = {
            "exception_type": type(e).__name__,
            "exception_message": str(e),
            "exception_traceback": traceback.format_exc(),
            "occurred_at": datetime.now().isoformat(),
        }
        log.warning("terminus2-driver agent phase failed: %s: %s", type(e).__name__, e)

    # Mirror harbor's ExceptionInfo; exit 0 regardless so the agent's partial
    # container state survives for the verifier (a timeout is still graded).
    if outcome_file:
        try:
            Path(outcome_file).parent.mkdir(parents=True, exist_ok=True)
            Path(outcome_file).write_text(json.dumps({"exception_info": exc_info, "finished": exc_info is None}))
        except Exception as e:
            log.warning("terminus2-driver failed to write outcome file: %s", e)

    log.info(
        "terminus2-driver done: episodes=%s in_tokens=%s out_tokens=%s exc=%s",
        (ctx.metadata or {}).get("n_episodes"), ctx.n_input_tokens, ctx.n_output_tokens,
        (exc_info or {}).get("exception_type"),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(_main())
'''
