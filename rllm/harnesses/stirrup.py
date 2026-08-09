"""Run Stirrup inside an rLLM-managed sandbox.

Stirrup normally provisions its own Docker or E2B environment.  This harness
instead starts Stirrup inside the task sandbox and gives it Stirrup's local
code-execution backend.  "Local" is therefore local to the isolated task
container: the agent and the verifier share one filesystem, while rLLM keeps
ownership of sandbox creation, teardown, tracing, and artifact collection.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from pathlib import Path
from typing import Any

from rllm import paths
from rllm.harnesses.cli_harness import BaseCliHarness
from rllm.sandbox.protocol import Sandbox
from rllm.types import AgentConfig, Episode, Task, Trajectory

_VENV_DIR = "/opt/stirrup-venv"
_DRIVER_PATH = "/opt/stirrup/driver.py"
_INSTRUCTION_PATH = "/tmp/stirrup/instruction.txt"
_RUN_METADATA_PATH = "/tmp/stirrup/run.json"

_INSTALL_SCRIPT = rf"""
set -e
export DEBIAN_FRONTEND=noninteractive
if [ -f {_VENV_DIR}/.stirrup-ready ]; then
    exit 0
fi
if ! command -v curl >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update -qq && apt-get install -y -qq curl ca-certificates
    elif command -v apk >/dev/null 2>&1; then
        apk add --no-cache curl ca-certificates bash
    fi
fi
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 {_VENV_DIR}
uv pip install --python {_VENV_DIR}/bin/python stirrup==0.1.12
touch {_VENV_DIR}/.stirrup-ready
"""


class StirrupHarness(BaseCliHarness):
    """Run the stock Stirrup agent in the task's existing sandbox."""

    name = "stirrup"
    sandbox_backend = "docker"
    stdout_log_path = "/tmp/stirrup.log"
    run_timeout = 14_400

    max_turns: int = 250
    max_output_tokens: int = 16_384
    max_context_tokens: int = 200_000
    enable_web: bool = True
    enable_vision: bool = True

    def install_script(self) -> str:
        return _INSTALL_SCRIPT

    def build_env(self, task: Task, config: AgentConfig) -> dict[str, str]:
        workdir = str(task.metadata.get("workdir") or "/workspace")
        reference_files = [str(name) for name in task.metadata.get("reference_files", [])]
        reasoning_effort = config.sampling_params.get("reasoning_effort")

        env = {
            "OPENAI_BASE_URL": config.base_url,
            "OPENAI_API_KEY": self.gateway_api_key(config, "OPENAI_API_KEY"),
            "RLLM_STIRRUP_MODEL": config.model,
            "RLLM_STIRRUP_WORKDIR": workdir,
            "RLLM_STIRRUP_REFERENCE_FILES": json.dumps(reference_files),
            "RLLM_STIRRUP_MAX_TURNS": str(self.max_turns),
            "RLLM_STIRRUP_MAX_OUTPUT_TOKENS": str(self.max_output_tokens),
            "RLLM_STIRRUP_MAX_CONTEXT_TOKENS": str(self.max_context_tokens),
            "RLLM_STIRRUP_ENABLE_WEB": "1" if self.enable_web else "0",
            "RLLM_STIRRUP_ENABLE_VISION": "1" if self.enable_vision else "0",
        }
        if reasoning_effort is not None:
            if not isinstance(reasoning_effort, str):
                raise ValueError("reasoning_effort sampling parameter must be a string")
            env["RLLM_STIRRUP_REASONING_EFFORT"] = reasoning_effort

        # Stirrup consumes this key itself. Agent shell commands do not inherit
        # the parent environment because Agent.share_parent_exec_env defaults
        # to False.
        brave_key = os.environ.get("BRAVE_API_KEY")
        if brave_key:
            env["BRAVE_API_KEY"] = brave_key
        return env

    def write_configs(
        self,
        sandbox: Sandbox,
        task: Task,
        config: AgentConfig,
        env: dict[str, str],
    ) -> None:
        self._exec_agent(sandbox, self._heredoc_write(_DRIVER_PATH, _DRIVER_SCRIPT), env=env)
        self._exec_agent(sandbox, self._heredoc_write(_INSTRUCTION_PATH, str(task.instruction).strip()), env=env)

    def build_invocation(self, instruction: str, task: Task, config: AgentConfig) -> str:
        del instruction, task, config
        return f"{_VENV_DIR}/bin/python {_DRIVER_PATH} 2>&1 | tee {shlex.quote(self.stdout_log_path)}"

    def run(self, task: Task, config: AgentConfig, *, env: Sandbox) -> Episode:
        """Run Stirrup, preserve deliverables, and expose its usage metadata."""
        super().run(task, config, env=env)
        run_data = self._read_run_data(env)
        metrics = _usage_metrics(run_data, config.model)
        artifacts = self._collect_deliverables(env, task, config, run_data)
        return Episode(task=task.metadata, trajectories=[Trajectory(name=self.name, steps=[])], metrics=metrics, artifacts=artifacts)

    @staticmethod
    def _read_run_data(sandbox: Sandbox) -> dict[str, Any]:
        try:
            raw = sandbox.exec(f"cat {shlex.quote(_RUN_METADATA_PATH)}", user="root")
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _collect_deliverables(
        sandbox: Sandbox,
        task: Task,
        config: AgentConfig,
        run_data: dict[str, Any],
    ) -> dict[str, Any]:
        workdir = str(task.metadata.get("workdir") or "/workspace")
        remote_dir = f"{workdir.rstrip('/')}/deliverables"
        safe_uid = re.sub(r"[^A-Za-z0-9_.-]+", "_", config.session_uid).strip("._") or "run"
        local_dir = Path(paths.rllm_path("agent_outputs", safe_uid))
        download = getattr(sandbox, "download_dir", None)
        downloaded: list[str] = []
        if callable(download):
            try:
                downloaded = [str(path) for path in download(remote_dir, str(local_dir))]
            except Exception:
                downloaded = []

        finish = run_data.get("finish") if isinstance(run_data.get("finish"), dict) else {}
        submitted = [str(path) for path in finish.get("paths") or []]
        return {
            "deliverable_dir": str(local_dir) if downloaded else None,
            "deliverables": downloaded,
            "submitted_paths": submitted,
            "remote_deliverable_dir": remote_dir,
        }


def _usage_metrics(run_data: dict[str, Any], model: str) -> dict[str, Any]:
    metadata = run_data.get("metadata") if isinstance(run_data.get("metadata"), dict) else {}
    raw_usage = metadata.get("token_usage")
    usage_entries = raw_usage if isinstance(raw_usage, list) else [raw_usage]
    usage = [entry for entry in usage_entries if isinstance(entry, dict)]
    input_tokens = sum(int(entry.get("input") or 0) for entry in usage)
    answer_tokens = sum(int(entry.get("answer") or 0) for entry in usage)
    reasoning_tokens = sum(int(entry.get("reasoning") or 0) for entry in usage)
    output_tokens = answer_tokens + reasoning_tokens
    metrics: dict[str, Any] = {
        "input_tokens": input_tokens,
        "answer_tokens": answer_tokens,
        "reasoning_tokens": reasoning_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }

    pricing_path = os.environ.get("RLLM_PRICING_FILE") or os.environ.get("GDPVAL_PRICING_FILE")
    if pricing_path:
        try:
            pricing = json.loads(Path(pricing_path).expanduser().read_text(encoding="utf-8"))
            models = pricing.get("models") if isinstance(pricing, dict) else {}
            rates = models.get(model) or models.get(model.removeprefix("openrouter/"))
            if isinstance(rates, dict):
                metrics["cost_usd"] = (
                    input_tokens * float(rates.get("input") or 0)
                    + answer_tokens * float(rates.get("answer") or rates.get("output") or 0)
                    + reasoning_tokens * float(rates.get("reasoning") or rates.get("answer") or rates.get("output") or 0)
                ) / 1_000_000
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    return metrics


_DRIVER_SCRIPT = r'''
import asyncio
import json
import os
from pathlib import Path

from stirrup import Agent, aggregate_metadata
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.tools.code_backends.local import LocalCodeExecToolProvider
from stirrup.tools.view_image import ViewImageToolProvider
from stirrup.tools.web import WebToolProvider


def _input_files(workdir: Path) -> list[str]:
    configured = json.loads(os.environ.get("RLLM_STIRRUP_REFERENCE_FILES", "[]"))
    if configured:
        candidates = [workdir / name for name in configured]
    else:
        candidates = [path for path in workdir.iterdir() if path.is_file()]
    return [str(path) for path in candidates if path.is_file()]


async def main() -> None:
    workdir = Path(os.environ.get("RLLM_STIRRUP_WORKDIR", "/workspace"))
    output_dir = workdir / "deliverables"
    output_dir.mkdir(parents=True, exist_ok=True)

    client = ChatCompletionsClient(
        model=os.environ["RLLM_STIRRUP_MODEL"],
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ.get("OPENAI_API_KEY", "sk-rllm-gateway"),
        max_tokens=int(os.environ.get("RLLM_STIRRUP_MAX_CONTEXT_TOKENS", "200000")),
        reasoning_effort=os.environ.get("RLLM_STIRRUP_REASONING_EFFORT"),
        kwargs={"max_tokens": int(os.environ.get("RLLM_STIRRUP_MAX_OUTPUT_TOKENS", "16384"))},
    )
    tools = [LocalCodeExecToolProvider(temp_base_dir="/tmp/stirrup-exec")]
    if os.environ.get("RLLM_STIRRUP_ENABLE_WEB", "1") == "1":
        tools.append(WebToolProvider())
    if os.environ.get("RLLM_STIRRUP_ENABLE_VISION", "1") == "1":
        tools.append(ViewImageToolProvider())

    agent = Agent(
        client=client,
        name="rllm-stirrup",
        max_turns=int(os.environ.get("RLLM_STIRRUP_MAX_TURNS", "250")),
        tools=tools,
    )
    instruction = Path("/tmp/stirrup/instruction.txt").read_text(encoding="utf-8")
    prompt = (
        instruction
        + "\n\nWhen finished, call the finish tool with a short summary and the relative paths "
          "of every deliverable that should be graded."
    )
    inputs = _input_files(workdir)
    async with agent.session(output_dir=output_dir, input_files=inputs or None) as session:
        finish_params, _history, metadata = await session.run(prompt)
    run_data = {
        "finish": finish_params.model_dump(mode="json") if finish_params is not None else None,
        "metadata": aggregate_metadata(metadata, return_json_serializable=True),
    }
    metadata_path = Path("/tmp/stirrup/run.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(run_data), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
'''
