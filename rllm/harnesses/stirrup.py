"""Run Stirrup inside an rLLM-managed sandbox.

Stirrup normally provisions its own Docker or E2B environment. This harness
instead starts Stirrup inside the task sandbox, so rLLM keeps ownership of
sandbox creation, teardown, tracing and artifact collection while the agent
code stays stock.

Three things Stirrup's shipped local backend cannot provide, supplied here:

* **Whole-sandbox filesystem.** ``LocalCodeExecToolProvider`` confines commands
  to a private temp directory and rejects any command mentioning ``/home``,
  ``/tmp`` or ``~`` — the paths a sandboxed benchmark actually uses. The driver
  therefore supplies its own :class:`CodeExecToolProvider` rooted at the task's
  working directory, which is what an E2B sandbox looks like to the agent.
* **A submission contract.** Stirrup's default ``finish`` takes a ``reason`` and
  validates only existence. This one takes a summary plus *absolute* paths, and
  has a sibling ``abandon`` for a task the model judges impossible. Both are
  defined here.
* **Non-root identity.** The solver runs as an ordinary user rather than root,
  which is what a benchmark's prompt normally promises it.

The harness is benchmark-agnostic: prompt, working directory, submission
contract and limits are class attributes a benchmark subclasses to supply.

Everything it records is provenance for a later grading stage. It does not
score, rank, or compare anything.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shlex
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import tomllib

from rllm import paths
from rllm.env import env_int
from rllm.harnesses.cli_harness import BaseCliHarness
from rllm.sandbox.protocol import Sandbox
from rllm.types import AgentConfig, Episode, Task, Trajectory

logger = logging.getLogger(__name__)

#: Pinned so a Stirrup release cannot silently change the runtime contract.
#: Recorded in every submission manifest.
#:
#: 0.2.0 is the floor for any strictly-validating provider: through 0.1.12,
#: ``to_openai_messages`` dumped Stirrup's internal ToolCall model and then
#: layered the OpenAI fields on top, so every tool call also carried
#: ``tool_call_id``/``arguments``/``signature``. OpenAI and OpenRouter ignore
#: the extras; Fireworks rejects the request with "Extra inputs are not
#: permitted". 0.2.0 builds the tool-call payload explicitly.
STIRRUP_VERSION = "0.2.0"

_VENV_DIR = "/opt/stirrup-venv"
_UV_PYTHON_DIR = "/opt/uv-python"
_CONFIG_DIR = "/opt/stirrup"
_DRIVER_PATH = f"{_CONFIG_DIR}/driver.py"
_SYSTEM_PROMPT_PATH = f"{_CONFIG_DIR}/system_prompt.txt"
_INSTRUCTION_PATH = f"{_CONFIG_DIR}/instruction.txt"

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
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin INSTALLER_NO_MODIFY_PATH=1 sh
fi
export PATH="/usr/local/bin:$HOME/.local/bin:$PATH"
# The install runs as root but the solver runs as a non-root user, so a
# uv-managed interpreter must not land under root's home (mode 700) — the
# venv's python symlink would be unusable. Accept any 3.12+ interpreter so an
# image that already ships one is reused instead of downloading another.
export UV_PYTHON_INSTALL_DIR={_UV_PYTHON_DIR}
uv venv --python '>=3.12' {_VENV_DIR}
uv pip install --python {_VENV_DIR}/bin/python stirrup=={STIRRUP_VERSION}
chmod -R a+rX {_VENV_DIR} {_UV_PYTHON_DIR} 2>/dev/null || true
touch {_VENV_DIR}/.stirrup-ready
"""


class StirrupHarness(BaseCliHarness):
    """Run the stock Stirrup agent in the task's existing sandbox."""

    name = "stirrup"
    sandbox_backend = "docker"
    stdout_log_path = "/tmp/stirrup.log"
    run_timeout = 14_400

    #: Written into the sandbox as the agent's system prompt. A benchmark
    #: subclass supplies its own; empty is refused at setup rather than run.
    system_prompt: str = ""
    #: Free-text label for the contract this harness reproduces, recorded in
    #: every submission manifest so a result can be traced to its methodology.
    methodology: str = ""
    #: Default when the task does not carry ``workdir`` in its metadata.
    workdir: str = "/home/user"
    #: Roots a submitted file may live under. Anything outside is rejected.
    submittable_roots: tuple[str, ...] = ("/home/user", "/tmp")
    #: Benchmark-specific provenance the builder wrote next to the task, read
    #: into the submission manifest. None means the harness expects none.
    provenance_filename: str | None = None
    #: Sandbox root the driver stages its submission and run report under. The
    #: harness creates both and reads them back, so it owns the paths; a
    #: benchmark overrides only to keep an established layout.
    submission_root: str = "/tmp/stirrup"

    #: Name Stirrup gives the agent it constructs. Cosmetic, but it lands in
    #: Stirrup's own logs, so a benchmark may want its own.
    agent_name: str = "stirrup-solver"

    max_turns: int = 250
    shell_timeout: int = 600
    #: Stirrup's own default (``stirrup.constants.CONTEXT_SUMMARIZATION_CUTOFF``).
    context_summarization_cutoff: float = 0.7
    # A benchmark that publishes no token budget leaves these at Stirrup's own
    # defaults; they are overridable per model. Setting the output cap too low is not a soft
    # truncation: Stirrup raises OutputTokenLimitError and the run dies, which
    # 16k reliably does for a reasoning model mid-tool-call.
    max_output_tokens: int = env_int("RLLM_STIRRUP_MAX_OUTPUT_TOKENS", 64_000)
    # Drives the 70% compaction threshold, so it should match the model's real
    # context window rather than being left at a generic default.
    max_context_tokens: int = env_int("RLLM_STIRRUP_MAX_CONTEXT_TOKENS", 200_000)
    enable_web: bool = True
    # View Image belongs only to vision-capable models. rLLM has no vision
    # capability metadata and a model slug does not imply it, so this is an
    # operator switch: set RLLM_STIRRUP_ENABLE_VISION=0 for a text-only model.
    # Offering the tool to one that cannot accept images fails the run — the
    # provider rejects the image content block mid-trajectory.
    enable_vision: bool = env_int("RLLM_STIRRUP_ENABLE_VISION", 1) == 1
    #: Newest images kept in the conversation; 0 disables pruning. See
    #: ``_PrunedImageClient`` -- every turn re-sends the whole history, so View
    #: Image results accumulate until the provider rejects the request.
    max_history_images: int = env_int("RLLM_STIRRUP_MAX_HISTORY_IMAGES", 0)

    @property
    def submission_dir(self) -> str:
        return f"{self.submission_root}/submission"

    @property
    def run_metadata_path(self) -> str:
        return f"{self.submission_root}/run.json"

    @property
    def run_id(self) -> str:
        """Identifier for one ``rllm eval`` invocation.

        ``AgentConfig`` carries no run-scoped id — ``session_uid`` is per task —
        and the flow instance is built once per run, so mint it here and reuse
        it for every task. ``RLLM_ARENA_RUN_ID`` pins it for a resumed run that
        should land in the same directory.
        """
        cached = getattr(self, "_run_id", None)
        if cached is None:
            pinned = os.environ.get("RLLM_ARENA_RUN_ID")
            cached = _slug(pinned, "run") if pinned else f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"
            self._run_id = cached
        return cached

    def install_script(self) -> str:
        return _INSTALL_SCRIPT

    def build_env(self, task: Task, config: AgentConfig) -> dict[str, str]:
        workdir = str(task.metadata.get("workdir") or self.workdir)
        reasoning_effort = config.sampling_params.get("reasoning_effort")

        env = {
            "OPENAI_BASE_URL": config.base_url,
            "OPENAI_API_KEY": self.gateway_api_key(config, "OPENAI_API_KEY"),
            "RLLM_STIRRUP_MODEL": config.model,
            "RLLM_STIRRUP_WORKDIR": workdir,
            "RLLM_STIRRUP_SUBMISSION_DIR": self.submission_dir,
            "RLLM_STIRRUP_RUN_METADATA_PATH": self.run_metadata_path,
            "RLLM_STIRRUP_AGENT_NAME": self.agent_name,
            "RLLM_STIRRUP_SUBMITTABLE_ROOTS": json.dumps(list(self.submittable_roots)),
            "RLLM_STIRRUP_MAX_TURNS": str(self.max_turns),
            "RLLM_STIRRUP_SHELL_TIMEOUT": str(self.shell_timeout),
            "RLLM_STIRRUP_CONTEXT_CUTOFF": str(self.context_summarization_cutoff),
            "RLLM_STIRRUP_MAX_OUTPUT_TOKENS": str(self.max_output_tokens),
            "RLLM_STIRRUP_MAX_CONTEXT_TOKENS": str(self.max_context_tokens),
            "RLLM_STIRRUP_ENABLE_WEB": "1" if self.enable_web else "0",
            "RLLM_STIRRUP_ENABLE_VISION": "1" if self.enable_vision else "0",
            "RLLM_STIRRUP_MAX_HISTORY_IMAGES": str(self.max_history_images),
            "RLLM_STIRRUP_SYSTEM_PROMPT_PATH": _SYSTEM_PROMPT_PATH,
            "RLLM_STIRRUP_INSTRUCTION_PATH": _INSTRUCTION_PATH,
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

    def build_invocation(self, instruction: str, task: Task, config: AgentConfig) -> str:
        del instruction, task, config
        return f"{_VENV_DIR}/bin/python {_DRIVER_PATH} 2>&1 | tee {shlex.quote(self.stdout_log_path)}"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self, task: Task, config: AgentConfig, *, env: Sandbox) -> Episode:
        """Run Stirrup as the task's agent user, then preserve its submission.

        ``BaseCliHarness.run`` is not reused: it execs as the class-level
        ``agent_user``, but the solver identity is a per-task property
        (``[agent] user`` in task.toml) and the flow instance is shared across
        concurrent rollouts, so it cannot be stashed on ``self``.
        """
        sandbox = env
        agent_user = task.metadata.get("agent_user") or self.agent_user
        env_vars = self.build_env(task, config)

        # An empty prompt would run the agent with no instructions at all and
        # still produce plausible-looking output, so refuse rather than let a
        # misconfigured harness quietly score a whole benchmark.
        if not self.system_prompt:
            raise ValueError(f"{type(self).__name__} defines no system_prompt; a benchmark subclass must supply one")

        # Config files are written as root: /opt is not writable by the
        # solver, and the solver only ever reads them.
        self._exec_agent(sandbox, self._heredoc_write(_DRIVER_PATH, _DRIVER_SCRIPT), env=env_vars)
        self._exec_agent(sandbox, self._heredoc_write(_SYSTEM_PROMPT_PATH, self.system_prompt), env=env_vars)
        self._exec_agent(sandbox, self._heredoc_write(_INSTRUCTION_PATH, str(task.instruction)), env=env_vars)
        self._exec_agent(sandbox, f"chmod -R a+rX {shlex.quote(_CONFIG_DIR)}", env=env_vars)

        timeout = float(task.metadata.get("agent_timeout", self.run_timeout))
        driver_output = ""
        try:
            driver_output = self._exec_agent(sandbox, self.build_invocation("", task, config), timeout=timeout, env=env_vars, user=agent_user)
        except Exception as e:
            # The submission bundle and run metadata may still exist (e.g. the
            # driver finished but the tee pipe failed), so keep collecting.
            logger.warning("%s execution failed: %s", type(self).__name__, e)

        run_data = self._read_run_data(sandbox)
        # A failed task otherwise surfaces only as a bare "EmptyCompletion" in
        # the eval summary, with the reason left behind in the torn-down sandbox
        # or buried in a manifest nobody reads mid-sweep.
        termination = run_data.get("termination") or {}
        if termination.get("type") == "error":
            # The driver caught this one and recorded it, so the log is enough.
            logger.warning("Stirrup run for task %s failed: %s", task.id, termination.get("reason"))
        elif not run_data:
            # No metadata at all means the driver never reached its epilogue,
            # and its output is the only account of why.
            self._log_driver_failure(sandbox, task, driver_output)
        metrics = _usage_metrics(run_data)
        artifacts = self._collect_submission(sandbox, task, config, run_data, metrics)
        return Episode(task=task.metadata, trajectories=[Trajectory(name=self.name, steps=[])], metrics=metrics, artifacts=artifacts)

    def _log_driver_failure(self, sandbox: Sandbox, task: Task, driver_output: str) -> None:
        """Log the tail of the driver's output for a task that produced nothing.

        Prefers the in-sandbox log over the captured stdout: a timeout or a
        broken pipe truncates what ``_exec_agent`` returns, while ``tee`` has
        already written whatever the driver managed to print.
        """
        tail = ""
        try:
            tail = sandbox.exec(f"tail -c 4000 {shlex.quote(self.stdout_log_path)}", user="root") or ""
        except Exception:
            logger.debug("No driver log at %s", self.stdout_log_path, exc_info=True)
        if not tail.strip():
            tail = driver_output[-4000:]
        logger.warning(
            "Stirrup produced no run metadata for task %s; driver log tail:\n%s",
            task.id,
            tail.strip() or "(empty)",
        )

    def _read_run_data(self, sandbox: Sandbox) -> dict[str, Any]:
        try:
            raw = sandbox.exec(f"cat {shlex.quote(self.run_metadata_path)}", user="root")
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except Exception:
            logger.debug("No run metadata at %s", self.run_metadata_path, exc_info=True)
            return {}

    def _collect_submission(
        self,
        sandbox: Sandbox,
        task: Task,
        config: AgentConfig,
        run_data: dict[str, Any],
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """Download the submission bundle and write the arena submission record.

        Runs before the sandbox is torn down. The bundle is staged in-sandbox
        by the driver, so this is a single directory download regardless of
        where the solver actually wrote its files.
        """
        local_dir = _submission_dir(task, config, self.name, self.run_id)

        downloaded: list[str] = []
        download = getattr(sandbox, "download_dir", None)
        if callable(download):
            try:
                downloaded = [str(path) for path in download(self.submission_dir, str(local_dir))]
            except Exception:
                logger.warning("Could not download the submission bundle from %s", self.submission_dir, exc_info=True)

        manifest = self._write_manifest(local_dir, task, config, run_data, metrics)
        termination = manifest["termination"]
        return {
            "submission_dir": str(local_dir) if downloaded else None,
            "submission_manifest": str(local_dir / "submission_manifest.json"),
            "deliverables": [entry["local_path"] for entry in manifest["artifacts"]],
            "submitted_paths": list(termination.get("submitted_paths") or []),
            "remote_submission_dir": self.submission_dir,
        }

    def _write_manifest(
        self,
        local_dir: Path,
        task: Task,
        config: AgentConfig,
        run_data: dict[str, Any],
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge sandbox-side and host-side records into one immutable manifest.

        Always written, including when the run produced nothing: a crashed or
        timed-out solver is a fact the corpus needs to record, and a missing
        file is indistinguishable from a task that was never run.
        """
        provenance = _task_provenance(task, self.provenance_filename)
        sandbox_manifest = _read_json(local_dir / "manifest.json")
        termination = run_data.get("termination") if isinstance(run_data.get("termination"), dict) else {}

        artifacts = []
        for entry in sandbox_manifest.get("artifacts") or []:
            local_path = local_dir / "files" / str(entry.get("bundle_path") or "").removeprefix("files/")
            if not local_path.is_file():
                logger.warning("Submitted file %s was not preserved locally", entry.get("submitted_path"))
                continue
            artifacts.append(
                {
                    "submitted_path": entry.get("submitted_path"),
                    "local_path": str(local_path),
                    "sha256": _sha256_file(local_path),
                    "size_bytes": local_path.stat().st_size,
                    "sandbox_sha256": entry.get("sha256"),
                }
            )

        manifest = {
            "schema_version": 1,
            "benchmark": _benchmark_name(task.dataset_dir),
            "methodology": self.methodology,
            "stage": "solver_generation",
            "graded": False,
            "task_id": provenance.get("task_id") or task.id,
            "solver_model": config.model,
            "run_id": config.session_uid,
            "dataset_repo": provenance.get("dataset_repo"),
            "dataset_revision": provenance.get("dataset_revision"),
            "sandbox_image_digest": provenance.get("sandbox_image_digest"),
            "sandbox_platform": provenance.get("sandbox_platform"),
            "stirrup_version": STIRRUP_VERSION,
            "system_prompt_sha256": _sha256_text(self.system_prompt),
            "task_prompt_sha256": _sha256_text(str(task.instruction)),
            "reference_files": provenance.get("reference_files") or [],
            "termination": termination or {"type": "unknown", "reason": "the solver produced no run metadata"},
            "rejected_paths": sandbox_manifest.get("rejected_paths") or [],
            "artifacts": artifacts,
            "metrics": metrics,
            "sampling_parameters": dict(config.sampling_params or {}),
        }
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "submission_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return manifest


def _slug(value: str, fallback: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or fallback


def _benchmark_name(dataset_dir: Path) -> str:
    """Benchmark name from ``dataset.toml``, else the directory name.

    Prefer the declared name: a dataset materialized to a temp directory should
    still file its submissions under the benchmark it belongs to.
    """
    config = Path(dataset_dir) / "dataset.toml"
    if config.exists():
        try:
            name = tomllib.loads(config.read_text()).get("dataset", {}).get("name")
            if name:
                return _slug(name, "benchmark")
        except (OSError, tomllib.TOMLDecodeError):
            pass
    return _slug(Path(dataset_dir).name, "benchmark")


def _player_id(model: str, agent: str) -> str:
    """Directory-safe competitor identity.

    Identity is (model, agent, label): the same model behind a different
    scaffold is a different competitor, and ``RLLM_ARENA_PLAYER_LABEL``
    separates runs of the same pair that should rank independently (different
    sampling or tool settings). Without the label they pool into one player and
    the configurations can never be compared.
    """
    parts = [_slug(model, "model"), _slug(agent, "agent")]
    label = os.environ.get("RLLM_ARENA_PLAYER_LABEL")
    return "__".join(parts) + (f"@{_slug(label, 'run')}" if label else "")


def _submission_dir(task: Task, config: AgentConfig, agent: str, run_id: str) -> Path:
    """Where this task's submission is preserved.

    Keyed by benchmark, player and run so a second model — or a second run of
    the same model — accumulates alongside the first instead of replacing it.
    The previous scheme keyed on ``session_uid`` alone (``<task>:<attempt>``),
    which silently destroyed the earlier model's files and left its episode
    JSON pointing at the newer model's output.
    """
    attempt = 0
    _, _, tail = str(config.session_uid).rpartition(":")
    if tail.isdigit():
        attempt = int(tail)
    return Path(
        paths.rllm_path(
            "agent_outputs",
            _benchmark_name(task.dataset_dir),
            _player_id(config.model, agent),
            run_id,
            f"{_slug(str(task.id), 'task')}__{attempt}",
        )
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _sha256_text(text: str) -> str:
    """Hex SHA-256 of *text* as UTF-8, for prompt provenance."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _task_provenance(task: Task, filename: str | None) -> dict[str, Any]:
    """Read the builder's provenance sidecar for this task, if the harness
    declares one and it exists."""
    task_dir = getattr(task, "task_dir", None)
    if task_dir is None or not filename:
        return {}
    return _read_json(Path(task_dir) / filename)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _usage_metrics(run_data: dict[str, Any]) -> dict[str, Any]:
    """Solver token usage for one rollout.

    Tokens only, deliberately. Cost is a function of vendor rates that change
    without notice and vary per account, so computing it here would bake a
    dated snapshot into results that read as authoritative. Multiply these
    counts by the rates actually in effect instead.
    """
    metadata = run_data.get("metadata") if isinstance(run_data.get("metadata"), dict) else {}
    raw_usage = metadata.get("token_usage")
    usage_entries = raw_usage if isinstance(raw_usage, list) else [raw_usage]
    usage = [entry for entry in usage_entries if isinstance(entry, dict)]
    input_tokens = sum(int(entry.get("input") or 0) for entry in usage)
    answer_tokens = sum(int(entry.get("answer") or 0) for entry in usage)
    reasoning_tokens = sum(int(entry.get("reasoning") or 0) for entry in usage)
    output_tokens = answer_tokens + reasoning_tokens
    metrics: dict[str, Any] = {
        "turns": int(run_data.get("turns") or 0),
        "input_tokens": input_tokens,
        "answer_tokens": answer_tokens,
        "reasoning_tokens": reasoning_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    return metrics


_DRIVER_SCRIPT = r'''"""In-sandbox Stirrup driver.

Runs as the solver user (UID 1000) inside the task sandbox. Builds the agent
with the configured system prompt, tool set and limits, then stages whatever the model
submitted into a single bundle so the harness can copy it out before teardown.
"""

import asyncio
import contextlib
import hashlib
import json
import os
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Annotated, Any

import anyio
from pydantic import BaseModel, Field

from stirrup import Agent, aggregate_metadata
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.core.models import AssistantMessage, ImageContentBlock, Tool, ToolResult, ToolUseCountMetadata
from stirrup.tools.code_backends.base import CodeExecToolProvider, CodeExecutionParams, CommandResult
from stirrup.tools.view_image import ViewImageToolProvider
from stirrup.tools.web import WebToolProvider

WORKDIR = Path(os.environ.get("RLLM_STIRRUP_WORKDIR", "/home/user"))
SUBMISSION_DIR = Path(os.environ["RLLM_STIRRUP_SUBMISSION_DIR"])
RUN_METADATA_PATH = Path(os.environ["RLLM_STIRRUP_RUN_METADATA_PATH"])
# Roots are resolved once: submitted paths are compared after resolution, so an
# unresolved root (/tmp is a symlink on some systems) would reject every file.
SUBMITTABLE_ROOTS = [Path(p).resolve() for p in json.loads(os.environ.get("RLLM_STIRRUP_SUBMITTABLE_ROOTS", '["/home/user", "/tmp"]'))]
SHELL_TIMEOUT = int(os.environ.get("RLLM_STIRRUP_SHELL_TIMEOUT", "600"))
# Newest images kept in the conversation; 0 disables pruning.
#
# Every turn re-sends the whole conversation, so View Image results accumulate
# and Fireworks rejects the request outright once it carries more than 60
# images ("we currently limit the number of images per conversation to 60") --
# a 400 that kills the rollout rather than degrading it. Summarization is the
# usual safety valve -- clearing earlier turn history drops those images with it
# -- but it only fires at a fraction of the context window, so sizing that window
# to the model's own real capacity pushes the threshold past the provider's image
# ceiling and a vision-heavy task hits the cap first. 0 disables the pruning.
_IMAGE_KEEP = int(os.environ.get("RLLM_STIRRUP_MAX_HISTORY_IMAGES", "0") or 0)


class _PrunedImageClient(ChatCompletionsClient):
    """Client that bounds the number of images carried in the conversation.

    Every turn re-sends the whole conversation, so View Image results accumulate
    until the provider rejects the request outright. Keeps the newest
    ``_IMAGE_KEEP`` and replaces older ones with a text note, in place. Older
    images are the ones the model has already reasoned about -- its written
    conclusions stay in the transcript -- so the cost is losing the ability to
    re-inspect, against losing the entire rollout to a 400. The block is replaced
    rather than deleted: the assistant's tool call and the tool message promising
    an image both remain, so removing it outright would leave the model looking at
    a request that was never answered. Text blocks here are bare strings.
    """

    async def generate(self, messages, tools):  # type: ignore[override]
        if _IMAGE_KEEP > 0:
            blocks = [(m, i, b) for m in messages for i, b in enumerate(getattr(m, "content", None) or []) if isinstance(b, ImageContentBlock)]
            for m, i, _ in blocks[: max(0, len(blocks) - _IMAGE_KEEP)]:
                m.content[i] = "[earlier image omitted to stay within the provider's per-conversation image limit]"
        return await super().generate(messages, tools)


class SandboxCodeExecToolProvider(CodeExecToolProvider):
    """Whole-sandbox code execution rooted at the task's working directory.

    Stirrup's LocalCodeExecToolProvider is designed for an *unsandboxed* host:
    it confines every command to a private temp directory and rejects commands
    that mention /home, /tmp or ~. rLLM already isolates the task in a
    container, so this provider gives the agent the same view an E2B sandbox
    would — the whole filesystem, with /home/user as the working directory.

    Each call runs a fresh ``bash -c``, so no working directory, environment
    variable or other shell state survives between calls, which is what a
    sandboxed benchmark's prompt normally tells the model to expect.
    """

    def __init__(self, workdir, *, shell_timeout, env=None):
        # No allowlist: this is an unrestricted shell inside an already-isolated
        # sandbox, so the provider is the boundary, not the command list.
        super().__init__(allowed_commands=None, shell_timeout=shell_timeout)
        self._workdir = Path(workdir)
        self._env = dict(env) if env is not None else None

    @property
    def temp_dir(self):
        return self._workdir

    async def __aenter__(self):
        self._workdir.mkdir(parents=True, exist_ok=True)
        return self.get_code_exec_tool()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None

    def _resolve(self, path):
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self._workdir / candidate

    async def run_command(self, cmd, *, timeout=None):
        if timeout is None:
            timeout = self._shell_timeout
        process = None
        try:
            with anyio.fail_after(timeout):
                # start_new_session puts bash at the head of its own process
                # group so a timeout can kill the whole tree, not just bash.
                process = await anyio.open_process(
                    ["bash", "-c", cmd],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=str(self._workdir),
                    env=self._env,
                    start_new_session=True,
                )
                stdout_chunks = []
                stderr_chunks = []

                async def read_stdout():
                    if process.stdout:
                        stdout_chunks.extend([chunk async for chunk in process.stdout])

                async def read_stderr():
                    if process.stderr:
                        stderr_chunks.extend([chunk async for chunk in process.stderr])

                async with anyio.create_task_group() as tg:
                    tg.start_soon(read_stdout)
                    tg.start_soon(read_stderr)
                await process.wait()
                return CommandResult(
                    exit_code=process.returncode or 0,
                    stdout=b"".join(stdout_chunks).decode("utf-8", errors="replace"),
                    stderr=b"".join(stderr_chunks).decode("utf-8", errors="replace"),
                )
        except TimeoutError:
            if process:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                with anyio.move_on_after(5):
                    await process.wait()
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr=f"Command timed out after {timeout} seconds",
                error_kind="timeout",
            )
        except Exception as exc:
            return CommandResult(exit_code=1, stdout="", stderr=str(exc), error_kind="execution_error")

    async def read_file_bytes(self, path):
        resolved = self._resolve(path)
        if not resolved.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        return resolved.read_bytes()

    async def write_file_bytes(self, path, content):
        resolved = self._resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_bytes(content)

    async def file_exists(self, path):
        return self._resolve(path).is_file()

    async def is_directory(self, path):
        return self._resolve(path).is_dir()

    async def list_files(self, path):
        resolved = self._resolve(path)
        if not resolved.is_dir():
            return []
        return [str(child.relative_to(resolved)) for child in sorted(resolved.rglob("*")) if child.is_file()]

    async def view_image(self, path):
        resolved = self._resolve(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        if resolved.is_dir():
            raise ValueError(f"Path is a directory, not an image: {path}")
        # Downscaling to one megapixel happens in ImageContentBlock, which
        # defaults to stirrup.constants.RESOLUTION_1MP.
        return ImageContentBlock(data=resolved.read_bytes())


class FinishParams(BaseModel):
    """Completed work, submitted as absolute file paths."""

    summary: Annotated[str, Field(description="A brief summary of what you accomplished.")]
    paths: Annotated[
        list[str],
        Field(description="A list of ABSOLUTE file paths for the required output files. Do not submit folders, only files."),
    ]


class AbandonParams(BaseModel):
    """Explanation for why the task cannot be completed."""

    reason: Annotated[str, Field(description="A brief reason why the task cannot be completed.")]


def path_problem(raw):
    """Return why *raw* is not a submittable file, or None if it is valid."""
    if not isinstance(raw, str) or not raw:
        return "is empty"
    if not raw.startswith("/"):
        return "is not an absolute path"
    try:
        resolved = Path(raw).resolve()
    except OSError as exc:
        return f"could not be resolved ({exc})"
    if not any(resolved == root or root in resolved.parents for root in SUBMITTABLE_ROOTS):
        allowed = ", ".join(str(root) for root in SUBMITTABLE_ROOTS)
        return f"resolves outside the writable roots ({allowed})"
    if not resolved.exists():
        return "does not exist"
    if resolved.is_dir():
        return "is a directory, not a file"
    if not resolved.is_file():
        return "is not a regular file"
    return None


def partition_paths(paths):
    accepted, rejected = [], []
    for raw in paths:
        problem = path_problem(raw)
        if problem is None:
            accepted.append(raw)
        else:
            rejected.append({"path": raw, "reason": problem})
    return accepted, rejected


async def finish_executor(params):
    _, rejected = partition_paths(params.paths)
    if rejected:
        details = "; ".join(f"{entry['path']} {entry['reason']}" for entry in rejected)
        return ToolResult(
            content=f"ERROR: these submitted paths are not valid deliverables: {details}. Submit absolute paths to existing files.",
            metadata=ToolUseCountMetadata(),
            success=False,
        )
    if not params.paths:
        return ToolResult(
            content="ERROR: no files submitted. Provide absolute paths to every deliverable, or use abandon_task_finish.",
            metadata=ToolUseCountMetadata(),
            success=False,
        )
    return ToolResult(content=params.summary, metadata=ToolUseCountMetadata(), success=True)


async def abandon_executor(params):
    return ToolResult(content=params.reason, metadata=ToolUseCountMetadata(), success=True)


FINISH_TOOL = Tool[FinishParams, ToolUseCountMetadata](
    name="finish",
    description=(
        "Signal task completion and submit your work. Provide a brief summary and the absolute path of every "
        "deliverable file. Note that you will need a separate turn to finish."
    ),
    parameters=FinishParams,
    executor=finish_executor,
)

ABANDON_TOOL = Tool[AbandonParams, ToolUseCountMetadata](
    name="abandon_task_finish",
    description=(
        "Signal that you cannot complete the task, with a brief reason, instead of submitting files. "
        "Use only when required inputs are missing, a hard dependency is unavailable, or the request is incoherent."
    ),
    parameters=AbandonParams,
    executor=abandon_executor,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_submission(paths):
    """Copy submitted files into the bundle, preserving path→file mapping.

    Basenames collide across directories, so each file keeps its full source
    path under the bundle rather than being flattened.
    """
    # Created even when nothing is accepted, so the harness always has a
    # bundle to download and can tell "submitted nothing" from "never ran".
    (SUBMISSION_DIR / "files").mkdir(parents=True, exist_ok=True)
    accepted, rejected = partition_paths(paths)
    artifacts = []
    for raw in accepted:
        source = Path(raw).resolve()
        bundle_relative = Path("files") / source.relative_to("/")
        destination = SUBMISSION_DIR / bundle_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(source, destination)
        except OSError as exc:
            rejected.append({"path": raw, "reason": f"could not be copied ({exc})"})
            continue
        artifacts.append(
            {
                "submitted_path": raw,
                "bundle_path": str(bundle_relative),
                "sha256": sha256_file(destination),
                "size_bytes": destination.stat().st_size,
            }
        )
    return artifacts, rejected


def count_turns(history):
    return sum(1 for messages in history for message in messages if isinstance(message, AssistantMessage))


def write_run_metadata(payload):
    RUN_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUN_METADATA_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


async def main():
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    client = _PrunedImageClient(
        model=os.environ["RLLM_STIRRUP_MODEL"],
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ.get("OPENAI_API_KEY", "sk-rllm-gateway"),
        # Two distinct budgets: max_tokens caps generation, while
        # context_window_tokens is the capacity the 70% compaction threshold is
        # a fraction of.
        max_tokens=int(os.environ.get("RLLM_STIRRUP_MAX_OUTPUT_TOKENS", "16384")),
        context_window_tokens=int(os.environ.get("RLLM_STIRRUP_MAX_CONTEXT_TOKENS", "200000")),
        reasoning_effort=os.environ.get("RLLM_STIRRUP_REASONING_EFFORT"),
        # Sampling is set on the gateway session, not here: SessionRoutingMiddleware
        # does payload.update(session_params), so anything this client sends for a
        # key the session also defines is discarded. Pass temperature via
        # `rllm eval --sampling-params temperature=...` instead.
    )

    exec_env = SandboxCodeExecToolProvider(
        WORKDIR,
        shell_timeout=SHELL_TIMEOUT,
        env={**os.environ, "HOME": str(WORKDIR), "PWD": str(WORKDIR)},
    )
    tools = [exec_env]
    if os.environ.get("RLLM_STIRRUP_ENABLE_WEB", "1") == "1":
        tools.append(WebToolProvider())
    if os.environ.get("RLLM_STIRRUP_ENABLE_VISION", "1") == "1":
        tools.append(ViewImageToolProvider(exec_env))

    agent = Agent(
        client=client,
        name=os.environ.get("RLLM_STIRRUP_AGENT_NAME", "stirrup-solver"),
        max_turns=int(os.environ.get("RLLM_STIRRUP_MAX_TURNS", "250")),
        system_prompt=Path(os.environ["RLLM_STIRRUP_SYSTEM_PROMPT_PATH"]).read_text(encoding="utf-8"),
        tools=tools,
        finish_tool=[FINISH_TOOL, ABANDON_TOOL],
        context_summarization_cutoff=float(os.environ.get("RLLM_STIRRUP_CONTEXT_CUTOFF", "0.7")),
    )

    prompt = Path(os.environ["RLLM_STIRRUP_INSTRUCTION_PATH"]).read_text(encoding="utf-8")
    # No output_dir and no input_files: reference files are already staged at
    # the absolute paths the prompt quotes, and submitted files are bundled
    # here rather than flattened into an output directory by basename.
    finish_params, history, metadata, error = None, [], {}, None
    try:
        async with agent.session(cache_on_interrupt=False) as session:
            finish_params, history, metadata = await session.run(prompt)
    except Exception as exc:
        # A run that died still belongs in the corpus; losing the record would
        # make it indistinguishable from a task that was never attempted.
        error = f"{type(exc).__name__}: {exc}"

    artifacts, rejected = [], []
    if error is not None:
        termination = {"type": "error", "reason": error}
    elif isinstance(finish_params, FinishParams):
        artifacts, rejected = stage_submission(finish_params.paths)
        termination = {"type": "finish", "summary": finish_params.summary, "submitted_paths": list(finish_params.paths)}
    elif isinstance(finish_params, AbandonParams):
        termination = {"type": "abandon_task_finish", "reason": finish_params.reason}
    else:
        termination = {"type": "max_turns_exhausted"}

    manifest = {
        "schema_version": 1,
        "termination": termination,
        "artifacts": artifacts,
        "rejected_paths": rejected,
    }
    (SUBMISSION_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_run_metadata(
        {
            "termination": termination,
            "turns": count_turns(history),
            "metadata": aggregate_metadata(metadata, return_json_serializable=True),
        }
    )


if __name__ == "__main__":
    asyncio.run(main())
'''
