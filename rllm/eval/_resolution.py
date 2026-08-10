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
import os
import re
import shlex
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


def _effective_verifier_timeout(task: Task) -> float | None:
    """Per-task verifier timeout (s), with RLLM_HARNESS_VERIFIER_TIMEOUT_S as a hard cap
    (mirrors RLLM_HARNESS_RUN_TIMEOUT_S for the agent). Returns None only when the task
    declares no verifier_timeout and no cap is set (callers apply their own default)."""
    from rllm.env import env_int

    declared = task.metadata.get("verifier_timeout")
    cap = env_int("RLLM_HARNESS_VERIFIER_TIMEOUT_S", 0)
    if declared is None:
        return float(cap) if cap > 0 else None
    if cap > 0:
        return min(float(declared), float(cap))
    return float(declared)


def _capture_git_heads(task: Task, sandbox: Sandbox) -> dict[str, str]:
    """Map ``repo root -> HEAD sha`` for the sandbox's git repos, before the agent runs.

    Handed to :class:`ShellScriptEvaluator`, which puts HEAD back (soft) right
    before grading — see :meth:`ShellScriptEvaluator._restore_git_heads` for why
    an agent's commits break in-sandbox verifiers. Called at evaluator-resolution
    time, which the hook does after environment setup and before the agent, so
    what's recorded is the image's own commit.

    Scans the declared ``workdir`` plus the top-level directories, so it covers
    ``/app``, ``/testbed``, ``/workspace`` and friends without knowing which task
    family it's looking at. Best-effort: a git-less environment yields an empty
    map and the restore is a no-op.

    Only for tasks that declared ``[verifier].environment_mode = "separate"``
    and are being graded in the agent's container anyway — i.e. exactly where
    rLLM knowingly deviates from the contract, which is what makes the verifier's
    pristine-HEAD assumption false. Moving HEAD is wrong for a task whose agent is
    *supposed* to move it (a shared-mode task that builds commits: terminal-bench's
    git-object-builder asserts on ``refs/heads/main`` and ``git ls-tree HEAD``), so
    shared-mode tasks are left alone. ``RLLM_VERIFIER_RESTORE_GIT_HEAD`` forces it
    on (``1``) for a shared task whose verifier does assume a pristine HEAD, or off
    (``0``) entirely.
    """
    from rllm.env import env_int
    from rllm.eval.script_evaluator import _GIT
    from rllm.tasks.loader import VERIFIER_MODE_SEPARATE

    override = os.environ.get("RLLM_VERIFIER_RESTORE_GIT_HEAD")
    if override is not None:
        if not env_int("RLLM_VERIFIER_RESTORE_GIT_HEAD", 1):
            return {}
    elif task.metadata.get("verifier_mode") != VERIFIER_MODE_SEPARATE:
        return {}
    roots = "/*"
    workdir = task.metadata.get("workdir")
    if workdir:
        roots = f"{shlex.quote(str(workdir))} /*"
    script = (
        f'for d in {roots}; do [ -d "$d" ] || continue; '
        f't=$({_GIT} -C "$d" rev-parse --show-toplevel 2>/dev/null) || continue; '
        f'h=$({_GIT} -C "$t" rev-parse HEAD 2>/dev/null) || continue; '
        f'printf "%s %s\\n" "$t" "$h"; done | sort -u'
    )
    heads: dict[str, str] = {}
    for line in _safe_exec(sandbox, script, timeout=60).splitlines():
        parts = line.split()
        if len(parts) == 2 and len(parts[1]) == 40:
            heads.setdefault(parts[0], parts[1])
    if heads:
        logger.debug("Captured pre-agent git HEADs for %s: %s", task.id, heads)
    return heads


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
        from rllm.tasks.loader import VERIFIER_MODE_SEPARATE

        separate = task.metadata.get("verifier_mode") == VERIFIER_MODE_SEPARATE and _separate_verifier_enabled()
        return ShellScriptEvaluator(
            sandbox=sandbox,
            script_path=verifier_config.get("script", "tests/test.sh"),
            verifier_user=task.metadata.get("verifier_user"),
            verifier_timeout=(_effective_verifier_timeout(task) or 600.0),
            reward_file_override=verifier_config.get("reward_file"),
            git_heads=_capture_git_heads(task, sandbox),
            verifier_sandbox_factory=(lambda: _create_verifier_sandbox(task, task.metadata.get("sandbox_backend"))) if separate else None,
            verifier_tests_baked=bool(separate and _verifier_dockerfile(task)),
            collect_commands=task.metadata.get("verifier_collect") if separate else None,
            artifacts=task.metadata.get("artifacts") if separate else None,
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


def _create_base_sandbox(
    task: Task,
    backend: str,
    *,
    image: str | None = None,
    name: str | None = None,
    env_override: dict | None = None,
    **backend_kwargs,
) -> Sandbox:
    """Create a sandbox from a base ``image`` — no Dockerfile RUN replay.

    ``image`` defaults to the task's resolved base image; pass a snapshot
    ref to boot from a pre-warmed environment instead. ``env_override``
    replaces the ``[environment]`` section resources are read from (used for a
    separate verifier container). ``backend_kwargs`` pass through to the backend
    constructor (e.g. Modal's ``timeout``).
    """
    from rllm.sandbox.sandboxed_flow import create_sandbox

    image = image if image is not None else _resolve_image(task, backend)
    if name is None:
        safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "-", task.id)
        name = f"rllm-{safe_id}-{uuid.uuid4().hex[:6]}"
    # Explicit caller kwargs override the per-task resource defaults — both may
    # carry Modal's ``timeout`` (e.g. the snapshot builder's build_timeout).
    kwargs = {**_sandbox_resource_kwargs(task, backend, env_override), **backend_kwargs}
    return create_sandbox(backend, name=name, image=image, **kwargs)


def _should_replay_dockerfile(task: Task) -> bool:
    """Whether to replay the Dockerfile's ``RUN`` steps on non-docker backends.

    Two task conventions are supported via ``[environment].replay_dockerfile``:

    * **SWE-bench style** (default, ``true``): the configured ``docker_image``
      is a *base*; the Dockerfile's ``RUN`` steps (e.g. ``uv``, needed by the
      grader) are not in that image and must be replayed on top.
    * **Terminal-bench / Harbor style** (``false``): the configured
      ``docker_image`` is the *fully built* task image, so replaying its ``RUN``
      steps double-applies the build (``git clone ... already exists``, missing
      ``COPY``'d files, etc.). These tasks set
      ``[environment]\nreplay_dockerfile = false`` to boot the image as-is.
    """
    env = task.metadata.get("environment", {}) or {}
    return bool(env.get("replay_dockerfile", True))


def _replay_dockerfile(task: Task, sandbox: Sandbox, backend: str) -> None:
    """Replay the Dockerfile RUN steps on a live sandbox (stage C).

    Non-docker backends pull the Dockerfile's FROM base instead of building
    it, so the RUN steps (e.g. swebench's ``uv``, needed by the grader) must
    be replayed. Best-effort: a failed step shouldn't abort the task. Docker
    builds the image, so its RUN steps already ran — skip. Tasks that boot a
    fully-built image opt out via ``[environment].replay_dockerfile = false``
    (see :func:`_should_replay_dockerfile`).
    """
    if backend == "docker":
        return
    if not _should_replay_dockerfile(task):
        return
    for cmd in _dockerfile_run_commands(task):
        _safe_exec(sandbox, cmd, timeout=900)


def _task_dockerfile(task: Task) -> Path | None:
    """Locate a task's ``environment/Dockerfile`` (task dir first, then dataset dir)."""
    for base in (task.task_dir, task.dataset_dir):
        df = base / "environment" / "Dockerfile"
        if df.exists():
            return df
    return None


# Remote backends that build images themselves and so can build the *real* Dockerfile
# (COPY/ENV/WORKDIR/RUN) instead of pulling FROM + replaying RUN. ``docker`` is excluded
# because it already builds via ``docker build``; ``local`` cannot build. ``modal`` is a
# tracked follow-up (it accepts a ``modal.Image`` and already keepalive-overrides the
# entrypoint, but the from_dockerfile path there is untested — see _dockerfile_image).
_FROM_DOCKERFILE_BACKENDS = ("daytona",)


def _builds_from_dockerfile(task: Task, backend: str) -> Path | None:
    """Return the Dockerfile to build directly on a remote backend, else ``None``.

    For remote backends that build images themselves, building the real Dockerfile keeps
    ``COPY``/``ENV``/``WORKDIR`` (which ``_replay_dockerfile`` silently drops) — required by
    COPY-then-RUN tasks like honeycomb's AWS/LocalStack tasks (``COPY start_localstack.sh``
    + ``ready.d``). Only when the task is Dockerfile-based (``replay_dockerfile`` true);
    prebuilt-image tasks (``replay_dockerfile = false``) boot their image as-is.
    """
    if backend not in _FROM_DOCKERFILE_BACKENDS or not _should_replay_dockerfile(task):
        return None
    return _task_dockerfile(task)


def _dockerfile_image(backend: str, dockerfile: Path):
    """Backend-native build spec for the real Dockerfile (COPY context = its directory)."""
    if backend == "daytona":
        from daytona import Image  # daytona.Image.from_dockerfile bundles the Dockerfile dir as context

        return Image.from_dockerfile(str(dockerfile))
    raise ValueError(f"from_dockerfile build unsupported for backend {backend!r}")


def _dockerfile_context_fingerprint(dockerfile: Path) -> str:
    """Stable hash of a Dockerfile's build context (its directory) for snapshot identity.

    Tasks built via ``Image.from_dockerfile`` must key on the *whole* context, not just
    ``FROM``+``RUN``: two tasks that share a base image and RUN block but differ in COPYed
    data (e.g. AWS tasks with different ``ready.d`` seeds) would otherwise collide on one
    snapshot / warm-queue sandbox. Hashes the Dockerfile text plus every file under its
    directory (relative path + bytes).
    """
    import hashlib

    ctx = dockerfile.parent
    h = hashlib.sha256()
    for p in sorted(ctx.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(ctx)).encode("utf-8") + b"\0")
            try:
                h.update(p.read_bytes())
            except OSError:
                pass
    return h.hexdigest()[:16]


def _create_sandbox_for_task(
    task: Task,
    sandbox_backend: str | None,
    *,
    name: str | None = None,
    env_override: dict | None = None,
) -> Sandbox:
    """Cold-path sandbox creation.

    When a Dockerfile-based task runs on a remote backend that builds images itself
    (``_builds_from_dockerfile``), build the *real* Dockerfile (full COPY/ENV/WORKDIR/RUN
    fidelity — the same primitive harbor's own environment uses) so nothing needs replaying.
    Otherwise create from the base/prebuilt image and replay the Dockerfile's RUN steps
    (``_replay_dockerfile``; a no-op for docker, which already built the image, and for
    prebuilt-image tasks that set ``replay_dockerfile = false``).
    """
    backend = _resolve_backend(task, sandbox_backend)
    compose_file = _task_compose_file(task)
    if compose_file is not None:
        if name is None:
            safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "-", task.id)
            name = f"rllm-{safe_id}-{uuid.uuid4().hex[:6]}"
        env = env_override if env_override is not None else (task.metadata.get("environment", {}) or {})
        compose_kwargs = {
            "name": name,
            "environment_dir": compose_file.parent,
            "compose_file": compose_file,
            "resources": env,
            "build_timeout": float(env.get("build_timeout_sec") or 600.0),
        }
        if backend == "docker":
            from rllm.sandbox.backends.docker_compose import DockerComposeSandbox

            return DockerComposeSandbox(**compose_kwargs)
        if backend == "modal":
            from rllm.sandbox.backends.modal_compose import ModalComposeSandbox

            return ModalComposeSandbox(**compose_kwargs, **_sandbox_resource_kwargs(task, "modal", env_override))
        raise RuntimeError(f"task {task.id!r} requires Docker Compose, which rLLM supports with --sandbox-backend docker or modal")
    dockerfile = _builds_from_dockerfile(task, backend)
    if dockerfile is not None:
        return _create_base_sandbox(task, backend, image=_dockerfile_image(backend, dockerfile), name=name, env_override=env_override)
    sandbox = _create_base_sandbox(task, backend, name=name, env_override=env_override)
    _replay_dockerfile(task, sandbox, backend)
    return sandbox


def _separate_verifier_enabled() -> bool:
    """Whether to honour ``environment_mode = "separate"`` by grading in a fresh box.

    The task's own declaration is the gate: only ``environment_mode = "separate"``
    tasks grade in a fresh box, which today means harbor SWE benchmarks like
    deepswe. Every other benchmark resolves to shared and never reaches this.

    ``RLLM_SEPARATE_VERIFIER_ENV=0`` is an escape hatch back to in-place grading
    for a task that declares separate — worth knowing about because the contract
    is *stricter*: it carries the agent's work as ``git diff <base_commit> HEAD``,
    so uncommitted edits don't count and a harness whose agent never commits
    scores 0. Harbor tasks instruct the agent to commit for exactly this reason.
    """
    from rllm.env import env_int

    return bool(env_int("RLLM_SEPARATE_VERIFIER_ENV", 1))


def _verifier_env_section(task: Task) -> dict:
    """The ``[environment]`` a separate-mode verifier container runs under.

    Harbor's ``resolve_effective_verifier_env_config`` prefers the task's
    ``[verifier.environment]`` and otherwise copies ``[environment]``. Tasks
    routinely declare only resources there (deepswe sets cpus/memory/storage and
    no image), so layer it *over* the task's section rather than replacing it —
    replacing would drop the image and leave nothing to boot.
    """
    return {**(task.metadata.get("environment") or {}), **(task.metadata.get("verifier_environment") or {})}


def _verifier_dockerfile(task: Task) -> Path | None:
    """Locate the Dockerfile whose build context is the task's tests directory."""
    for base in (task.task_dir, task.dataset_dir):
        dockerfile = base / "tests" / "Dockerfile"
        if dockerfile.exists():
            return dockerfile
    return None


def _create_verifier_sandbox(task: Task, sandbox_backend: str | None) -> Sandbox:
    """A fresh container to grade a separate-mode task in.

    Harbor builds this image from ``tests/`` as the build context. The verifier
    Dockerfile may install dependencies or transform files, so uploading the
    raw tests into the agent image is not equivalent.
    """
    backend = _resolve_backend(task, sandbox_backend)
    dockerfile = _verifier_dockerfile(task)
    if dockerfile is None:
        # Preserve legacy separate-verifier tasks that predate Harbor's
        # tests/Dockerfile contract.
        safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "-", task.id)
        return _create_sandbox_for_task(
            task,
            backend,
            name=f"rllm-verify-{safe_id}-{uuid.uuid4().hex[:6]}",
            env_override=_verifier_env_section(task),
        )
    if backend == "docker":
        image = _build_docker_image(dockerfile.parent, f"{task.id}-verifier")
    elif backend == "daytona":
        image = _dockerfile_image(backend, dockerfile)
    elif backend == "modal":
        import modal

        image = modal.Image.from_dockerfile(str(dockerfile), context_dir=str(dockerfile.parent))
    else:
        raise RuntimeError(f"building tests/Dockerfile is not supported by the {backend!r} sandbox backend")
    safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "-", task.id)
    sandbox = _create_base_sandbox(
        task,
        backend,
        image=image,
        name=f"rllm-verify-{safe_id}-{uuid.uuid4().hex[:6]}",
        env_override=_verifier_env_section(task),
    )
    verifier_env = {
        **(_verifier_env_section(task).get("env") or {}),
        **(task.metadata.get("verifier_env_vars") or {}),
    }
    set_env = getattr(sandbox, "set_env", None)
    if verifier_env and callable(set_env):
        set_env(verifier_env)
    return sandbox


def _sandbox_resource_kwargs(task: Task, backend: str, env_override: dict | None = None) -> dict:
    """Map a harbor task's declared ``[environment]`` resources to backend kwargs.

    Harbor task.toml declares ``cpus`` / ``memory_mb`` / ``storage_mb``; without
    these a remote sandbox runs at the backend default (Daytona: 1 GiB), which
    OOM-kills compile-heavy graders (e.g. ``go test ./...``). Modal takes memory
    in MB; Daytona takes memory/disk in GB. Docker applies CPU and memory
    limits directly; local has no resource-isolation primitive.

    Those per-task values are baked into ``task.toml`` at dataset-build time, so
    an over-provisioned default can only be shrunk by rebuilding the dataset.
    Three provider-agnostic *caps* let an operator clamp every sandbox down at
    runtime instead — each lowers the task's declared value (``min``) when set
    (>0); a task already at or below the cap is untouched, and a task that
    declares nothing is left at the backend default (a cap only lowers — it never
    raises a task above what it declared, nor introduces a value where there is
    none):

    * ``RLLM_SANDBOX_MAX_CPUS`` (float) — max physical cores.
    * ``RLLM_SANDBOX_MAX_MEMORY_MB`` (int) — max memory in MB.
    * ``RLLM_SANDBOX_MAX_STORAGE_MB`` (int) — max disk in MB (Daytona only; Modal
      never receives ``storage`` and bills scratch disk as part of compute).

    Modal Sandboxes bill on reserved CPU+memory per second, so capping cuts the
    per-rollout bill proportionally (e.g. ``RLLM_SANDBOX_MAX_CPUS=2`` halves the
    CPU term of a 4-core task).

    The sandbox lifetime is sized to this task's own budget so the box always
    outlives the agent + verifier it hosts (both run inside it). A flat default
    could be shorter than agent+verifier and reap the box mid-rollout ("Sandbox
    already shut down" / ENOSPC mid-run). The provider-agnostic
    ``RLLM_SANDBOX_TIMEOUT_S`` (seconds) is a *floor* on top of that, applied the
    same way for every backend; each backend then expresses it in its own unit
    (Modal's hard ``timeout`` in seconds; Daytona's idle ``auto_stop_interval``
    in minutes).
    """
    from rllm.env import env_float, env_int, sandbox_timeout_override_s

    env = env_override if env_override is not None else (task.metadata.get("environment", {}) or {})
    cpus, mem_mb, disk_mb = env.get("cpus"), env.get("memory_mb"), env.get("storage_mb")
    gpus = env.get("gpus")
    gpu_types = env.get("gpu_types") or []

    # Operator caps clamp the baked-in task.toml values down (see docstring):
    # shrink an over-provisioned sandbox at runtime without rebuilding. A cap
    # only lowers — never raises a task above its declared value, nor sets one
    # where the task declares none (a min against a missing value would be wrong).
    cpu_cap = env_float("RLLM_SANDBOX_MAX_CPUS", 0.0)
    mem_cap = env_int("RLLM_SANDBOX_MAX_MEMORY_MB", 0)
    disk_cap = env_int("RLLM_SANDBOX_MAX_STORAGE_MB", 0)
    if cpu_cap > 0 and cpus:
        cpus = min(float(cpus), cpu_cap)
    if mem_cap > 0 and mem_mb:
        mem_mb = min(int(mem_mb), mem_cap)
    if disk_cap > 0 and disk_mb:
        disk_mb = min(int(disk_mb), disk_cap)

    # Per-task lifetime floor (seconds), shared across backends: agent + verifier
    # + install + teardown/scheduling slack, raised to the operator override. The
    # sandbox lifetime tracks the *effective* agent timeout: RLLM_HARNESS_RUN_TIMEOUT_S,
    # when set, is a hard CAP on a task's own agent_timeout (matching
    # cli_harness._effective_timeout) — otherwise a task's large baked-in agent_timeout
    # (e.g. SWE-bench Verified) keeps the sandbox alive far past the operator cap.
    _run_cap = env_int("RLLM_HARNESS_RUN_TIMEOUT_S", 0)
    _per_task = task.metadata.get("agent_timeout")
    if _per_task is None:
        agent_t = float(_run_cap or 3600)
    elif _run_cap > 0:
        agent_t = min(float(_per_task), float(_run_cap))
    else:
        agent_t = float(_per_task)
    # Sandbox must outlast the full rollout (agent + verifier + teardown). When the task
    # declares a verifier_timeout, budget exactly that plus 300s teardown/scheduling slack;
    # otherwise a flat 600s cushion. Raised to the operator override (RLLM_SANDBOX_TIMEOUT_S).
    verifier_t = _effective_verifier_timeout(task)
    if verifier_t is not None:
        lifetime_s = int(agent_t + float(verifier_t) + 300)
    else:
        lifetime_s = int(agent_t + 600)
    lifetime_s = max(lifetime_s, sandbox_timeout_override_s())

    kw: dict = {}
    if backend == "docker":
        if cpus:
            kw["cpus"] = float(cpus)
        if mem_mb:
            kw["memory"] = int(mem_mb) * 1024 * 1024
        if gpus:
            kw["gpus"] = int(gpus)
    elif backend == "modal":
        if cpus:
            kw["cpu"] = float(cpus)
        if mem_mb:
            kw["memory"] = int(mem_mb)
        if gpus:
            gpu_type = str(gpu_types[0]) if gpu_types else "any"
            kw["gpu"] = f"{gpu_type}:{int(gpus)}" if int(gpus) > 1 else gpu_type
        kw["timeout"] = lifetime_s  # Modal's hard lifetime, in seconds
    elif backend == "daytona":
        if cpus:
            kw["cpu"] = max(1, int(cpus))  # Daytona cores are ints; a fractional Modal-style cap floors to 1
        if mem_mb:
            kw["memory"] = max(1, round(mem_mb / 1024))
        if disk_mb:
            kw["disk"] = max(1, round(disk_mb / 1024))
        if gpus:
            kw["gpu"] = int(gpus)
        # First boot of a from-image sandbox includes the registry pull, which
        # for multi-GB SWE images routinely exceeds the SDK's 120s default.
        # Honor the task's declared build timeout, with a pull-friendly floor.
        kw["create_timeout"] = float(env.get("build_timeout_sec") or 600.0)
        # Daytona's lifetime knob is an idle auto-stop in minutes (its default
        # 30-min idle can reap a long task, e.g. during a stalled LLM call that
        # looks idle). Express the shared lifetime floor in minutes, rounded up.
        kw["auto_stop_interval"] = (lifetime_s + 59) // 60
    return kw


def _dockerfile_run_commands(task: Task) -> list[str]:
    """Return a task's ``environment/Dockerfile`` ``RUN`` shell steps.

    ``\\``-continuations are joined into a single logical command with a space
    (matching shell line-continuation semantics) so multi-line ``RUN`` steps
    stay valid when re-executed via ``bash -c``. Non-``RUN`` directives —
    ``COPY``/``ADD`` etc. — are skipped; only ``RUN`` is replayable on a live
    sandbox.
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
            cmd = " ".join(part.strip() for part in parts).strip()
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


def _task_compose_file(task: Task) -> Path | None:
    """Locate a Harbor-style Docker Compose overlay for the task environment."""
    for base in (task.task_dir, task.dataset_dir):
        for name in ("docker-compose.yaml", "docker-compose.yml", "compose.yaml", "compose.yml"):
            path = base / "environment" / name
            if path.exists():
                return path
    return None


def _validate_task_runtime(task: Task, backend: str) -> None:
    """Fail before agent execution when a task needs an unsupported capability."""
    if _task_compose_file(task) is not None and backend not in {"docker", "modal"}:
        raise RuntimeError(f"task {task.id!r} requires Docker Compose; select --sandbox-backend docker or modal")
    if task.metadata.get("verifier_mode") == "separate" and _verifier_dockerfile(task) is not None and backend not in {"docker", "daytona", "modal"}:
        raise RuntimeError(f"task {task.id!r} requires a verifier image build unsupported by backend {backend!r}")


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


def _run_healthcheck(task: Task, sandbox: Sandbox) -> None:
    """Boot a task's declared service and wait for readiness before the agent.

    Harbor-format tasks may declare ``[environment.healthcheck]`` (command +
    interval/timeout/retries/start-period) that boots an in-image service and
    blocks until it is ready — e.g. honeycomb's AWS/LocalStack tasks run
    ``bash /usr/local/bin/start_localstack.sh`` to start LocalStack and seed
    ``ready.d``. rLLM boots the container with ``sleep infinity`` and never runs
    the image CMD/entrypoint, so without this the service never starts and the
    agent (and verifier) hit a dead endpoint. Mirrors harbor's ``run_healthcheck``
    (run after environment setup, before the agent).

    No-op when the task declares no healthcheck — so non-service eval *and*
    training tasks are unaffected. Raises on exhaustion so an unbootable service
    surfaces as an explicit infra error rather than a silent reward 0.0.
    """
    import time

    hc = (task.metadata.get("environment", {}) or {}).get("healthcheck")
    if not isinstance(hc, dict):
        return
    command = hc.get("command")
    if not command:
        return

    timeout_s = float(hc.get("timeout_sec") or 300.0)
    interval_s = float(hc.get("interval_sec") or 5.0)
    retries = int(hc.get("retries") if hc.get("retries") is not None else 3)
    start_period_s = float(hc.get("start_period_sec") or 0.0)

    if start_period_s > 0:
        time.sleep(start_period_s)

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            sandbox.exec(command, timeout=timeout_s)
            logger.info("Healthcheck passed (attempt %d/%d): %s", attempt + 1, retries + 1, command)
            return
        except Exception as e:  # non-zero exit / timeout → service not ready yet
            last_err = e
            if attempt < retries:
                time.sleep(interval_s)
    raise RuntimeError(f"Healthcheck failed after {retries + 1} attempt(s) for command {command!r}: {last_err}")


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
        # Lock the verifier/reward dirs away from the (sandboxed) agent user, but
        # keep them owned by whoever runs the verifier — root unless the task set
        # a distinct verifier_user — so the verifier (now actually switched to
        # that user via the backend's su emulation) can still write reward files.
        verifier_owner = task.metadata.get("verifier_user") or "root"
        _safe_exec(sandbox, "mkdir -p /logs/verifier /tmp/rllm /tests", timeout=10)
        _safe_exec(sandbox, "chmod 700 /logs/verifier /tmp/rllm /tests", timeout=10)
        _safe_exec(sandbox, f"chown {verifier_owner} /logs/verifier /tmp/rllm /tests", timeout=10)
        if workdir:
            _safe_exec(sandbox, f"chown -R {agent_user} {workdir}", timeout=30)

    env_vars = task.metadata.get("env_vars", {}) or task.metadata.get("environment", {}).get("env", {})
    if env_vars:
        # Make declared env present in *every* later exec (agent + verifier), the
        # way Harbor injects [environment].env as a per-exec Secret. A one-shot
        # ``export`` wouldn't survive — each exec is a fresh shell. Backends that
        # expose ``set_env`` (e.g. Modal) honor it; others fall back to export.
        set_env = getattr(sandbox, "set_env", None)
        if callable(set_env):
            set_env(env_vars)
        else:
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
