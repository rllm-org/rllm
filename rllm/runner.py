"""Runner: unified orchestrator for AgentFlow + Evaluator over a Task.

Single code path for both data tasks (gsm8k-style rows) and sandbox tasks
(Harbor-style task directories):

1. Read the task's verifier configuration.
2. Build the sandbox if the task requires one.
3. Let the AgentFlow run on the task — producing an Episode (1 or many trajectories).
4. Resolve an Evaluator from the task config (or use ``evaluator_override``) and run it.
5. Write rewards back onto the trajectories.

No ``episode.artifacts["_sandbox"]`` hack — sandbox-using evaluators
carry their sandbox reference internally (constructed at resolve time).
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
        import asyncio

        from rllm.sandbox.sandboxed_flow import SandboxedAgentFlow

        # Every sandbox lifecycle call below is sync and can do real I/O
        # (Modal API roundtrips, docker exec, file uploads, verifier
        # script execution). Running them directly inside this async
        # method blocks the event loop — Modal's SDK even surfaces this
        # as ``AsyncUsageWarning`` — and serialises every concurrent
        # task on the slowest setup. ``asyncio.to_thread`` offloads to
        # a worker thread where Modal's "in async context" check fails
        # cleanly and other in-flight tasks keep making progress.

        # Skip per-task verifier detection when an override is provided —
        # the override fully dictates scoring, so we shouldn't probe the
        # task dir for verifier scripts (Harbor task dirs contain
        # tests/test.sh + environment/ that the harbor trial runs itself).
        if self.evaluator_override is not None:
            verifier_kind, verifier_config = "override", {}
            needs_sandbox = isinstance(self.agent_flow, SandboxedAgentFlow)
        else:
            verifier_kind, verifier_config = _detect_verifier(task)
            needs_sandbox = _needs_sandbox(task, verifier_kind) or isinstance(self.agent_flow, SandboxedAgentFlow)

        sandbox: Sandbox | None = None
        try:
            if needs_sandbox:
                sandbox, base_image, backend = await asyncio.to_thread(
                    _create_sandbox_for_task,
                    task,
                    self.sandbox_backend,
                    self.agent_flow,
                )
                if isinstance(self.agent_flow, SandboxedAgentFlow):
                    self.agent_flow.set_sandbox(sandbox)  # local assignment, no I/O
                    # Install + (on docker) commit *before* per-task setup
                    # so the cached image stays free of task-specific files.
                    await asyncio.to_thread(
                        self.agent_flow.pre_setup,
                        sandbox,
                        base_image,
                        backend,
                    )
                await asyncio.to_thread(_setup_task_environment, task, sandbox)
                if isinstance(self.agent_flow, SandboxedAgentFlow):
                    await asyncio.to_thread(
                        self.agent_flow.on_sandbox_ready,
                        {"task_path": str(task.task_dir)},
                        config,
                    )

            # AgentFlow runs the agent → Episode (already runs ``run`` in
            # a thread pool via ``_run_agent_flow``, so its sandbox.exec
            # calls don't block the event loop either).
            episode = await _run_agent_flow(self.agent_flow, task, config)

            # Stamp session_uid onto every trajectory so the UI can join
            # back to traces in the shared gateway db
            # (filtered by run_id). Harnesses may pre-set this; we only
            # fill in blanks.
            if config.session_uid:
                for traj in episode.trajectories:
                    if traj.session_id is None:
                        traj.session_id = config.session_uid

            # Evaluator: explicit override > per-task resolution
            if self.evaluator_override is not None:
                evaluator = _adapt_legacy_evaluator(self.evaluator_override)
            else:
                evaluator = _resolve_evaluator(task, sandbox, verifier_kind, verifier_config)
            eval_output = await asyncio.to_thread(evaluator.evaluate, task, episode)

            # Write rewards back
            for traj in episode.trajectories:
                traj.reward = eval_output.reward
                traj.signals = {s.name: s.value for s in eval_output.signals}
            episode.is_correct = eval_output.is_correct
            return episode
        finally:
            if sandbox is not None:
                from rllm.sandbox.cleanup import deregister

                # Let the harness drop its sandbox reference first
                # (its own ``teardown_sandbox`` is a no-op for externally-
                # managed sandboxes, but it clears internal state — e.g.,
                # the harness's view of the sandbox handle — that matters
                # if ``create_instance()`` reuses the harness instance).
                if isinstance(self.agent_flow, SandboxedAgentFlow):
                    try:
                        self.agent_flow.teardown_sandbox()  # local state only
                    except Exception:
                        logger.exception("teardown_sandbox failed")
                # The runner owns the sandbox lifecycle whenever it
                # created the sandbox (which is always, on this code
                # path). Close + deregister unconditionally so cloud
                # backends (Modal, Daytona) don't leak duration-billed
                # containers when running through SandboxedAgentFlow.
                try:
                    await asyncio.to_thread(sandbox.close)
                except Exception:
                    logger.exception("sandbox close failed")
                finally:
                    deregister(sandbox)


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


def _create_sandbox_for_task(
    task: Task,
    sandbox_backend: str | None,
    agent_flow: AgentFlow | None = None,
) -> tuple[Sandbox, str, str]:
    """Create a sandbox for *task* and return ``(sandbox, base_image, backend)``.

    The base image is the result of building the task's environment
    Dockerfile (or the configured fallback). When *agent_flow* exposes
    :meth:`maybe_use_cached_image`, a post-install derived image is used
    instead — but the *returned* base image is unchanged so the harness
    can compute the same derived tag in :meth:`pre_setup`.

    The image's ``WORKDIR`` (read from the Dockerfile) is propagated as
    the sandbox's working directory so backends that don't honour image
    metadata at exec time (Modal) still land commands in the right
    place. Falls back to ``[environment].workdir`` when set explicitly.
    """
    from rllm.sandbox.sandboxed_flow import SandboxedAgentFlow, create_sandbox

    backend = sandbox_backend or task.metadata.get("sandbox_backend") or "docker"
    base_image = _resolve_image(task, backend)
    image = base_image
    if isinstance(agent_flow, SandboxedAgentFlow):
        image = agent_flow.maybe_use_cached_image(base_image, backend)

    safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "-", task.id)
    name = f"rllm-{safe_id}-{uuid.uuid4().hex[:6]}"
    sandbox_kwargs: dict[str, Any] = {}
    workdir = _resolve_workdir(task)
    if workdir:
        sandbox_kwargs["workdir"] = workdir
    sandbox_kwargs.update(_resolve_resource_kwargs(task, backend))
    base_image_str = base_image if isinstance(base_image, str) else "<modal.Image>"
    return create_sandbox(backend, name=name, image=image, **sandbox_kwargs), base_image_str, backend


def _resolve_image(task: Task, backend: str) -> Any:
    """Resolve the sandbox image for *task* on *backend*.

    Returns either:
    - A locally-built ``rllm-task-<id>`` tag string (docker backend with
      a task ``Dockerfile``).
    - A ``modal.Image`` object built from the task's ``Dockerfile``
      (modal backend with a task ``Dockerfile``). Modal's ``Sandbox``
      accepts both strings and ``modal.Image`` objects directly.
    - The configured registry image string (everything else).

    The previous implementation silently dropped the task's Dockerfile
    on non-docker backends, so Modal runs always boots ``python:3.11-slim``
    regardless of what the task asked for. That breaks every
    Dockerfile-driven task on Modal.
    """
    env_config = task.metadata.get("environment", {}) or {}
    configured = env_config.get("docker_image", "python:3.11-slim")

    dockerfile = task.task_dir / "environment" / "Dockerfile"
    if not dockerfile.exists():
        dockerfile = task.dataset_dir / "environment" / "Dockerfile"

    if not dockerfile.exists():
        return configured

    if backend == "docker":
        return _build_docker_image(dockerfile.parent, task.id)
    if backend == "modal":
        return _build_modal_image(dockerfile)
    return configured


def _resolve_workdir(task: Task) -> str | None:
    """Resolve the sandbox's working directory for *task*.

    Order of precedence:
    1. ``[environment].workdir`` in task.toml — explicit override.
    2. Last ``WORKDIR`` directive in the task's ``Dockerfile``.
    3. ``None`` — let the backend use its default cwd.

    Returning a value guarantees both Docker and Modal exec commands
    in the same directory, removing image-WORKDIR-vs-explicit-workdir
    drift between backends.
    """
    env_config = task.metadata.get("environment", {}) or {}
    explicit = env_config.get("workdir") or task.metadata.get("workdir")
    if isinstance(explicit, str) and explicit:
        return explicit

    dockerfile = task.task_dir / "environment" / "Dockerfile"
    if not dockerfile.exists():
        dockerfile = task.dataset_dir / "environment" / "Dockerfile"
    if dockerfile.exists():
        return _parse_dockerfile_workdir(dockerfile)
    return None


def _resolve_resource_kwargs(task: Task, backend: str) -> dict[str, Any]:
    """Translate Harbor-style ``[environment]`` resource hints into backend kwargs.

    Reads ``cpus`` / ``memory_mb`` / ``gpus`` / ``gpu_types`` from
    ``task.metadata['environment']`` and emits ``cpu`` / ``memory`` / ``gpu``
    kwargs in the shape Modal's ``Sandbox.create()`` expects. Other backends
    (docker, local) accept ``**kwargs`` and silently ignore unknown keys, so
    we only emit when the backend is ``modal`` to avoid noisy warnings.

    GPU encoding follows Modal's API:
      - ``gpus = N`` + no types → ``gpu = "any:N"`` (or ``"any"`` when N==1).
      - ``gpus = N`` + ``gpu_types = ["A100", ...]`` → first type wins,
        ``gpu = "A100:N"`` (or ``"A100"`` when N==1).
    """
    if backend != "modal":
        return {}

    env_config = task.metadata.get("environment", {}) or {}
    out: dict[str, Any] = {}

    cpus = env_config.get("cpus")
    if isinstance(cpus, int | float) and cpus > 0:
        out["cpu"] = float(cpus)

    memory_mb = env_config.get("memory_mb")
    if isinstance(memory_mb, int | float) and memory_mb > 0:
        out["memory"] = int(memory_mb)

    gpus = env_config.get("gpus")
    if isinstance(gpus, int | float) and gpus > 0:
        gpu_types = env_config.get("gpu_types") or []
        gpu_kind = str(gpu_types[0]) if isinstance(gpu_types, list) and gpu_types else "any"
        gpu_count = int(gpus)
        out["gpu"] = gpu_kind if gpu_count == 1 else f"{gpu_kind}:{gpu_count}"

    return out


def _parse_dockerfile_workdir(dockerfile: Path) -> str | None:
    """Return the last ``WORKDIR`` directive in *dockerfile*, or ``None``.

    Tolerates whitespace, trailing comments, and empty lines. Doesn't
    handle ARG/ENV expansion — if a Dockerfile uses ``WORKDIR $FOO``
    we leave the literal value alone (the caller will probably pass it
    to the sandbox unchanged, which is rare enough not to special-case).
    """
    try:
        text = dockerfile.read_text()
    except OSError:
        return None
    workdir: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not line.upper().startswith("WORKDIR"):
            continue
        rest = line[len("WORKDIR") :].strip()
        if rest:
            workdir = rest
    return workdir


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


def _build_modal_image(dockerfile: Path) -> Any:
    """Build a ``modal.Image`` from the task's Dockerfile.

    Modal builds remotely on its own infra and caches by content, so
    repeated runs against the same Dockerfile are cheap. The returned
    ``modal.Image`` is passed straight through to ``Sandbox.create()``.

    ``context_dir`` is pinned to the Dockerfile's parent so ``COPY``
    directives resolve against the task's ``environment/`` directory.
    Modal otherwise defaults to ``Path.cwd()``, which makes every
    ``COPY Gemfile ...`` (or any other relative path) silently fail
    when ``rllm eval`` is invoked from the repo root.
    """
    import modal

    logger.info("Building Modal image from %s", dockerfile)
    return modal.Image.from_dockerfile(dockerfile, context_dir=dockerfile.parent)


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
