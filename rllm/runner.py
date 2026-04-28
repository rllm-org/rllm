"""Runner: unified orchestrator for AgentFlow + Evaluator over a Task.

The Runner replaces the split between ``EvalRunner`` (for catalog datasets)
and ``TaskRunner`` (for sandbox tasks). It:

1. Reads the task's verifier configuration.
2. Builds the sandbox if the task requires one.
3. Lets the AgentFlow run on the task — producing an Episode (1 or many trajectories).
4. Resolves an Evaluator from the task config and runs it.
5. Writes rewards back onto the trajectories.

A single code path. No ``episode.artifacts["_sandbox"]`` hack — sandbox-using
evaluators carry their sandbox reference internally (constructed at resolve time).
"""

from __future__ import annotations

import importlib
import inspect
import logging
import re
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import tomllib

from rllm.eval.module_evaluator import PythonModuleEvaluator, _coerce_eval_result
from rllm.eval.script_evaluator import ShellScriptEvaluator
from rllm.eval.types import EvalOutput
from rllm.sandbox.protocol import Sandbox
from rllm.types import AgentConfig, AgentFlow, Episode, Evaluator, Task

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class Runner:
    """Run an AgentFlow on a Task, then resolve and run its verifier.

    Args:
        agent_flow: The :class:`AgentFlow` to drive the task.
        sandbox_backend: Optional override for the sandbox backend
            (``"docker"``, ``"local"``, ``"modal"``, ...).
        evaluator_override: If provided, bypass per-task verifier
            resolution and use this :class:`Evaluator` for every task.
            The CLI's ``--evaluator`` flag flows through here.
    """

    def __init__(
        self,
        agent_flow: AgentFlow,
        sandbox_backend: str | None = None,
        evaluator_override: Evaluator | None = None,
    ):
        self.agent_flow = agent_flow
        self.sandbox_backend = sandbox_backend
        self.evaluator_override = evaluator_override

    async def run(self, task: Task, config: AgentConfig) -> Episode:
        from rllm.sandbox.sandboxed_flow import SandboxedAgentFlow

        verifier_kind, verifier_config = _detect_verifier(task)
        # Force a sandbox if the agent_flow needs one, even when the
        # evaluator override doesn't.
        needs_sandbox = _needs_sandbox(task, verifier_kind) or isinstance(self.agent_flow, SandboxedAgentFlow)

        sandbox: Sandbox | None = None
        try:
            if needs_sandbox:
                sandbox = _create_sandbox_for_task(task, self.sandbox_backend)
                _setup_task_environment(task, sandbox)
                if isinstance(self.agent_flow, SandboxedAgentFlow):
                    self.agent_flow.set_sandbox(sandbox)
                    self.agent_flow.on_sandbox_ready({"task_path": str(task.task_dir)}, config)

            # AgentFlow runs the agent → Episode
            episode = await _run_agent_flow(self.agent_flow, task, config)

            # Evaluator: explicit override > per-task resolution
            if self.evaluator_override is not None:
                evaluator = _adapt_legacy_evaluator(self.evaluator_override)
            else:
                evaluator = _resolve_evaluator(task, sandbox, verifier_kind, verifier_config)
            eval_output = evaluator.evaluate(task, episode)

            # Write rewards back
            for traj in episode.trajectories:
                traj.reward = eval_output.reward
                traj.signals = {s.name: s.value for s in eval_output.signals}
            episode.is_correct = eval_output.is_correct
            return episode
        finally:
            if sandbox is not None:
                if isinstance(self.agent_flow, SandboxedAgentFlow):
                    try:
                        self.agent_flow.teardown_sandbox()
                    except Exception:
                        logger.exception("teardown_sandbox failed")
                else:
                    try:
                        sandbox.close()
                    except Exception:
                        logger.exception("sandbox close failed")


# ---------------------------------------------------------------------------
# Verifier resolution
# ---------------------------------------------------------------------------


def _detect_verifier(task: Task) -> tuple[str, dict]:
    """Inspect task.toml/dataset.toml + filesystem; return (kind, config).

    Kinds: ``"sandbox-shell"``, ``"python-host"``, ``"python-hybrid"``,
    ``"registered"``, ``"import"``.
    """
    config = _read_verifier_config(task)
    task_dir = task.task_dir
    has_dockerfile = (task_dir / "environment" / "Dockerfile").exists() or (task.dataset_dir / "environment" / "Dockerfile").exists()

    if "script" in config:
        return "sandbox-shell", config
    if "module" in config:
        return ("python-hybrid" if has_dockerfile else "python-host"), config
    if "name" in config:
        return "registered", config
    if "import_path" in config:
        return "import", config

    # Auto-detect by file presence
    if (task_dir / "tests" / "test.sh").exists():
        return "sandbox-shell", {"script": "tests/test.sh"}
    if (task_dir / "tests" / "evaluate.py").exists():
        return ("python-hybrid" if has_dockerfile else "python-host"), {"module": "tests.evaluate"}
    # Shared verifier at benchmark level (rows-with-shared-verifier shape)
    if (task.dataset_dir / "tests" / "evaluate.py").exists():
        return ("python-hybrid" if has_dockerfile else "python-host"), {"module": "tests.evaluate"}
    if (task.dataset_dir / "tests" / "test.sh").exists():
        return "sandbox-shell", {"script": "tests/test.sh"}

    return "missing", {}


def _read_verifier_config(task: Task) -> dict:
    """Read ``[verifier]`` from task.toml (per-task) or dataset.toml (shared)."""
    candidates = []
    if task.sub_dir is not None:
        candidates.append(task.dataset_dir / task.sub_dir / "task.toml")
    candidates.append(task.dataset_dir / "dataset.toml")
    for cfg_path in candidates:
        if cfg_path.exists():
            try:
                raw = tomllib.loads(cfg_path.read_text())
            except Exception:
                continue
            verifier = raw.get("verifier", {})
            if verifier:
                return verifier
    return {}


def _resolve_evaluator(
    task: Task,
    sandbox: Sandbox | None,
    kind: str,
    verifier_config: dict,
) -> Evaluator:
    """Construct an Evaluator instance for this task."""
    if kind == "sandbox-shell":
        if sandbox is None:
            raise RuntimeError("sandbox-shell verifier requires an active sandbox")
        return ShellScriptEvaluator(
            sandbox=sandbox,
            script_path=verifier_config.get("script", "tests/test.sh"),
            verifier_user=task.metadata.get("verifier_user"),
            verifier_timeout=float(task.metadata.get("verifier_timeout", 600.0)),
            reward_file_override=verifier_config.get("reward_file"),
        )

    if kind in ("python-host", "python-hybrid"):
        # Look in the task's own dir first, then the shared benchmark dir
        module = verifier_config.get("module", "tests.evaluate")
        function = verifier_config.get("function", "evaluate")
        for base in (task.task_dir, task.dataset_dir):
            try:
                ev = PythonModuleEvaluator.from_module(base, module, function)
                return _wrap_with_sandbox_if_needed(ev, sandbox)
            except FileNotFoundError:
                continue
        raise FileNotFoundError(f"Verifier module '{module}' not found in {task.task_dir} or {task.dataset_dir}")

    if kind == "registered":
        from rllm.eval.evaluator_loader import load_evaluator

        return _adapt_legacy_evaluator(load_evaluator(verifier_config["name"]))

    if kind == "import":
        ev = _load_callable(verifier_config["import_path"])
        if isinstance(ev, type):
            ev = ev()
        if hasattr(ev, "evaluate"):
            return _adapt_legacy_evaluator(ev)
        # Bare function — wrap as a thin Evaluator
        return _FunctionEvaluator(ev)

    raise RuntimeError(f"No verifier configured for task '{task.id}' (dataset_dir={task.dataset_dir})")


def build_dataset_evaluator(dataset_dir: Path, sub_dir: Path | None = None) -> Evaluator | None:
    """Build a single :class:`Evaluator` from a dataset's ``[verifier]`` config.

    Supports the host-only verifier kinds (``module``, ``name``,
    ``import_path``, plus auto-detected ``tests/evaluate.py``) so the
    trainer — which expects one Evaluator for the whole dataset — can
    reuse the same per-task resolution that :class:`Runner` performs for
    eval. Sandbox-shell verifiers return ``None`` because they need a
    per-task sandbox lifecycle that lives inside :class:`Runner`.
    """
    probe = Task(id="", instruction="", metadata={}, dataset_dir=dataset_dir, sub_dir=sub_dir)
    kind, config = _detect_verifier(probe)
    if kind in ("sandbox-shell", "python-hybrid", "missing"):
        return None
    return _resolve_evaluator(probe, sandbox=None, kind=kind, verifier_config=config)


def _wrap_with_sandbox_if_needed(ev: PythonModuleEvaluator, sandbox: Sandbox | None) -> Evaluator:
    """If the user's evaluate() signature includes ``sandbox``, inject it."""
    if sandbox is None:
        return ev
    if any(p.name == "sandbox" for p in ev._params):  # noqa: SLF001
        # Bind the sandbox into kwargs at call time
        original_build = ev._build_kwargs  # noqa: SLF001

        def build_with_sandbox(task: Task, episode: Episode) -> dict:
            kwargs = original_build(task, episode)
            kwargs["sandbox"] = sandbox
            return kwargs

        ev._build_kwargs = build_with_sandbox  # type: ignore[method-assign]
    return ev


# ---------------------------------------------------------------------------
# Sandbox setup (extracted from rllm/tasks/runner.py)
# ---------------------------------------------------------------------------


def _needs_sandbox(task: Task, verifier_kind: str) -> bool:
    """Decide whether the Runner should set up a sandbox."""
    if verifier_kind in ("sandbox-shell", "python-hybrid"):
        return True
    # If the task ships an environment/, treat it as sandboxed
    if (task.task_dir / "environment").is_dir() or (task.dataset_dir / "environment").is_dir():
        return True
    return False


def _create_sandbox_for_task(task: Task, sandbox_backend: str | None) -> Sandbox:
    from rllm.sandbox.sandboxed_flow import create_sandbox

    backend = sandbox_backend or task.metadata.get("sandbox_backend") or "docker"
    image = _resolve_image(task, backend)

    safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "-", task.id)
    name = f"rllm-{safe_id}-{uuid.uuid4().hex[:6]}"
    return create_sandbox(backend, name=name, image=image)


def _resolve_image(task: Task, backend: str) -> str:
    """Build from Dockerfile if present and backend is docker, else config default."""
    env_config = task.metadata.get("environment", {}) or {}
    configured = env_config.get("docker_image", "python:3.11-slim")

    dockerfile = task.task_dir / "environment" / "Dockerfile"
    if not dockerfile.exists():
        dockerfile = task.dataset_dir / "environment" / "Dockerfile"

    if dockerfile.exists() and backend == "docker":
        return _build_docker_image(dockerfile.parent, task.id)
    return configured


def _build_docker_image(context_dir: Path, task_id: str) -> str:
    """Build via subprocess (avoids docker-py credential helper issues on macOS)."""
    import subprocess

    tag = "rllm-task-" + re.sub(r"[^a-zA-Z0-9_.-]", "-", task_id).lower()
    logger.info("Building Docker image '%s' from %s", tag, context_dir)
    result = subprocess.run(
        ["docker", "build", "-t", tag, "--rm", "."],
        cwd=str(context_dir),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Docker build failed for {task_id}:\n{result.stderr[:1000]}")
    return tag


def _setup_task_environment(task: Task, sandbox: Sandbox) -> None:
    """Upload environment/files/, run setup.sh and [rllm].setup_commands.

    Honors the agent/verifier user split when configured in task.toml.
    """
    workdir = task.metadata.get("workdir", "/workspace")
    env_root = task.task_dir / "environment"
    if not env_root.is_dir():
        env_root = task.dataset_dir / "environment"

    _safe_exec(sandbox, f"mkdir -p {workdir}", timeout=30)

    files_dir = env_root / "files"
    if files_dir.is_dir():
        sandbox.upload_dir(str(files_dir), workdir)

    setup_script = env_root / "setup.sh"
    if setup_script.exists():
        sandbox.upload_file(str(setup_script), "/tmp/rllm_setup.sh")
        _safe_exec(sandbox, "chmod +x /tmp/rllm_setup.sh && /tmp/rllm_setup.sh", timeout=300)

    for cmd in task.metadata.get("setup_commands", []) or []:
        _safe_exec(sandbox, cmd, timeout=300)

    agent_user = task.metadata.get("agent_user")
    if agent_user:
        _safe_exec(sandbox, "mkdir -p /logs/verifier /tmp/rllm /tests", timeout=10)
        _safe_exec(sandbox, "chmod 700 /logs/verifier /tmp/rllm /tests", timeout=10)
        _safe_exec(sandbox, "chown root:root /logs/verifier /tmp/rllm /tests", timeout=10)
        _safe_exec(sandbox, f"chown -R {agent_user} {workdir}", timeout=30)

    env_vars = task.metadata.get("env_vars", {}) or task.metadata.get("environment", {}).get("env", {})
    if env_vars:
        exports = " && ".join(f"export {k}='{v}'" for k, v in env_vars.items())
        _safe_exec(sandbox, exports, timeout=10)


def _safe_exec(sandbox: Sandbox, command: str, timeout: float | None = None, user: str | None = None) -> str:
    try:
        return sandbox.exec(command, timeout=timeout, user=user)
    except Exception as e:
        logger.debug("exec failed (suppressed): %s — %s", command[:200], e)
        return ""


# ---------------------------------------------------------------------------
# AgentFlow invocation
# ---------------------------------------------------------------------------


async def _run_agent_flow(agent: AgentFlow, task: Task, config: AgentConfig) -> Episode:
    """Call agent.arun() if available, else agent.run() — adapter for sync/async."""
    import asyncio

    if hasattr(agent, "arun") and inspect.iscoroutinefunction(agent.arun):
        return await agent.arun(task, config)

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, agent.run, task, config)


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


class _FunctionEvaluator:
    """Wrap a bare ``evaluate(task, episode)`` callable as an Evaluator."""

    def __init__(self, fn: Callable):
        self.fn = fn

    def evaluate(self, task: Task, episode: Episode) -> EvalOutput:
        result = self.fn(task, episode)
        return _coerce_eval_result(result) if not isinstance(result, EvalOutput) else result


def _adapt_legacy_evaluator(ev: Any) -> Evaluator:
    """Adapt evaluators with ``evaluate(task: dict, episode)`` to ``evaluate(task: Task, episode)``.

    With ``from __future__ import annotations`` annotations are strings,
    so we compare both ``is dict`` and string forms.
    """
    sig = inspect.signature(ev.evaluate)
    params = list(sig.parameters.values())
    if not params:
        return ev
    first = params[0]
    annotation = first.annotation if first.annotation is not inspect.Parameter.empty else None

    is_dict_annotation = annotation is dict or annotation == "dict" or (isinstance(annotation, str) and annotation.startswith("dict"))
    if is_dict_annotation or first.name in ("task_data", "task_info") or annotation is None:
        return _LegacyDictAdapter(ev)
    return ev


class _LegacyDictAdapter:
    """Pass ``task.metadata`` (dict) to an old-style Evaluator."""

    def __init__(self, inner: Any):
        self.inner = inner

    def evaluate(self, task: Task, episode: Episode) -> EvalOutput:
        return self.inner.evaluate(task.metadata, episode)


def _load_callable(import_path: str) -> Callable:
    """Resolve ``module.path:attr`` to a Python object."""
    if ":" not in import_path:
        raise ValueError(f"import_path must be 'module:attr', got {import_path!r}")
    module_path, attr_name = import_path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)
