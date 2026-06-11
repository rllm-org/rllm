"""Per-task verifier resolution + sandbox lifecycle helpers.

These were originally part of the ``rllm.runner.Runner`` per-task driver
that drove ``rllm eval`` before eval was unified onto
:class:`rllm.engine.agentflow_engine.AgentFlowEngine`.
``Runner`` is gone; the helpers live here because:

* :class:`rllm.hooks.SandboxTaskHooks` calls them on every rollout to set
  up the sandbox and resolve the per-task evaluator.
* :func:`build_dataset_evaluator` is the train CLI's entry point for
  resolving a single dataset-wide evaluator from a ``[verifier]`` block.

Module is private (``_resolution``) — external callers should go through
:class:`rllm.hooks.SandboxTaskHooks` or :func:`build_dataset_evaluator`.
"""

from __future__ import annotations

import base64
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
from rllm.types import Episode, Evaluator, Task

logger = logging.getLogger(__name__)


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
                ev.sandbox = sandbox
                return ev
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


def dataset_verifier_kind(dataset_dir: Path, sub_dir: Path | None = None) -> str:
    """The dataset-level verifier kind (``"missing"`` when none is configured).

    Used by the train CLI to distinguish env-style verifiers (resolved per
    task inside the sandbox — leave the trainer's ``evaluator`` unset) from a
    genuinely missing verifier (fail fast).
    """
    probe = Task(id="", instruction="", metadata={}, dataset_dir=dataset_dir, sub_dir=sub_dir)
    kind, _ = _detect_verifier(probe)
    return kind


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


# ---------------------------------------------------------------------------
# Sandbox setup (extracted from rllm/tasks/runner.py)
# ---------------------------------------------------------------------------


def _resolve_backend(task: Task, sandbox_backend: str | None) -> str:
    """Resolve the effective sandbox backend for a task."""
    return sandbox_backend or task.metadata.get("sandbox_backend") or "docker"


def _create_base_sandbox(task: Task, backend: str, *, image: str | None = None, name: str | None = None) -> Sandbox:
    """Create a sandbox from a base ``image`` — no Dockerfile RUN replay.

    ``image`` defaults to the task's resolved base image; pass a snapshot
    ref to boot from a pre-warmed environment instead.
    """
    from rllm.sandbox.sandboxed_flow import create_sandbox

    image = image if image is not None else _resolve_image(task, backend)
    if name is None:
        safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "-", task.id)
        name = f"rllm-{safe_id}-{uuid.uuid4().hex[:6]}"
    return create_sandbox(backend, name=name, image=image, **_sandbox_resource_kwargs(task, backend))


def _replay_dockerfile(task: Task, sandbox: Sandbox, backend: str) -> None:
    """Replay the Dockerfile RUN steps on a live sandbox (stage C).

    Non-docker backends pull the Dockerfile's FROM base instead of building
    it, so the RUN steps (e.g. swebench's ``uv``, needed by the grader) must
    be replayed. Best-effort: a failed step shouldn't abort the task. Docker
    builds the image, so its RUN steps already ran — skip.
    """
    if backend == "docker":
        return
    for cmd in _dockerfile_run_commands(task):
        _safe_exec(sandbox, cmd, timeout=900)


def _create_sandbox_for_task(task: Task, sandbox_backend: str | None) -> Sandbox:
    """Cold-path sandbox creation: base image + RUN replay (today's behavior)."""
    backend = _resolve_backend(task, sandbox_backend)
    sandbox = _create_base_sandbox(task, backend)
    _replay_dockerfile(task, sandbox, backend)
    return sandbox


def _sandbox_resource_kwargs(task: Task, backend: str) -> dict:
    """Map a harbor task's declared ``[environment]`` resources to backend kwargs.

    Harbor task.toml declares ``cpus`` / ``memory_mb`` / ``storage_mb``; without
    these a remote sandbox runs at the backend default (Daytona: 1 GiB), which
    OOM-kills compile-heavy graders (e.g. ``go test ./...``). Modal takes memory
    in MB; Daytona takes memory/disk in GB. Docker/local ignore the values.
    """
    env = task.metadata.get("environment", {}) or {}
    cpus, mem_mb, disk_mb = env.get("cpus"), env.get("memory_mb"), env.get("storage_mb")
    kw: dict = {}
    if backend == "modal":
        if cpus:
            kw["cpu"] = float(cpus)
        if mem_mb:
            kw["memory"] = int(mem_mb)
    elif backend == "daytona":
        if cpus:
            kw["cpu"] = int(cpus)
        if mem_mb:
            kw["memory"] = max(1, round(mem_mb / 1024))
        if disk_mb:
            kw["disk"] = max(1, round(disk_mb / 1024))
    return kw


def _dockerfile_run_commands(task: Task) -> list[str]:
    """Return a task's ``environment/Dockerfile`` ``RUN`` shell steps (joining
    ``\\``-continuations). Non-``RUN`` directives — ``COPY``/``ADD`` etc. — are
    skipped; only ``RUN`` is replayable on a live sandbox.
    """
    dockerfile = task.task_dir / "environment" / "Dockerfile"
    if not dockerfile.exists():
        dockerfile = task.dataset_dir / "environment" / "Dockerfile"
    if not dockerfile.exists():
        return []
    try:
        lines = dockerfile.read_text().splitlines()
    except OSError:
        return []

    commands: list[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.upper().startswith("RUN "):
            parts = [stripped[4:]]
            while parts[-1].rstrip().endswith("\\"):
                parts[-1] = parts[-1].rstrip()[:-1]
                i += 1
                if i >= len(lines):
                    break
                parts.append(lines[i])
            cmd = "\n".join(parts).strip()
            if cmd:
                commands.append(cmd)
        i += 1
    return commands


def _as_single_run_line(cmd: str) -> str:
    """Collapse a multi-line shell command into one line for a Dockerfile ``RUN``.

    Daytona builds snapshots declaratively: each command becomes a raw
    ``RUN <command>`` line, which a multi-line script breaks. ``bash`` (not
    ``sh``) matches how :meth:`Sandbox.exec` runs the same scripts live.
    """
    if "\n" not in cmd:
        return cmd
    encoded = base64.b64encode(cmd.encode("utf-8")).decode("ascii")
    return f"echo {encoded} | base64 -d | bash"


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
    # ``workdir`` is unset for tasks (e.g. swesmith) whose Dockerfile
    # already declares a meaningful WORKDIR (``/testbed``) — forcing
    # ``/workspace`` would override it and break verifiers that
    # ``cd``-into-cwd or ``git checkout``. The ``mkdir`` / ``chown``
    # / ``upload_dir(files)`` steps below only fire when a workdir is
    # explicitly declared.
    workdir = task.metadata.get("workdir")
    env_root = task.task_dir / "environment"
    if not env_root.is_dir():
        env_root = task.dataset_dir / "environment"

    if workdir:
        _safe_exec(sandbox, f"mkdir -p {workdir}", timeout=30)

    files_dir = env_root / "files"
    if files_dir.is_dir():
        # Falls back to ``/workspace`` when files/ ships but no workdir
        # is declared — preserves the historical default for tasks that
        # actually rely on it.
        sandbox.upload_dir(str(files_dir), workdir or "/workspace")

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
        if workdir:
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

    Only an explicit ``dict`` annotation (string form included — with
    ``from __future__ import annotations`` annotations are strings) or a
    legacy parameter name opts into the dict calling convention; an
    unannotated evaluator gets the ``Task``.
    """
    sig = inspect.signature(ev.evaluate)
    params = list(sig.parameters.values())
    if not params:
        return ev
    first = params[0]
    annotation = first.annotation if first.annotation is not inspect.Parameter.empty else None

    is_dict_annotation = annotation is dict or annotation == "dict" or (isinstance(annotation, str) and annotation.startswith("dict"))
    if is_dict_annotation or first.name in ("task_data", "task_info"):
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
