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


def _create_base_sandbox(task: Task, backend: str, *, image: str | None = None, name: str | None = None, **backend_kwargs) -> Sandbox:
    """Create a sandbox from a base ``image`` — no Dockerfile RUN replay.

    ``image`` defaults to the task's resolved base image; pass a snapshot
    ref to boot from a pre-warmed environment instead. ``backend_kwargs``
    pass through to the backend constructor (e.g. Modal's ``timeout``).
    """
    from rllm.sandbox.sandboxed_flow import create_sandbox

    image = image if image is not None else _resolve_image(task, backend)
    if name is None:
        safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "-", task.id)
        name = f"rllm-{safe_id}-{uuid.uuid4().hex[:6]}"
    return create_sandbox(backend, name=name, image=image, **_sandbox_resource_kwargs(task, backend), **backend_kwargs)


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


def _create_sandbox_for_task(task: Task, sandbox_backend: str | None) -> Sandbox:
    """Cold-path sandbox creation.

    When a Dockerfile-based task runs on a remote backend that builds images itself
    (``_builds_from_dockerfile``), build the *real* Dockerfile (full COPY/ENV/WORKDIR/RUN
    fidelity — the same primitive harbor's own environment uses) so nothing needs replaying.
    Otherwise create from the base/prebuilt image and replay the Dockerfile's RUN steps
    (``_replay_dockerfile``; a no-op for docker, which already built the image, and for
    prebuilt-image tasks that set ``replay_dockerfile = false``).
    """
    backend = _resolve_backend(task, sandbox_backend)
    dockerfile = _builds_from_dockerfile(task, backend)
    if dockerfile is not None:
        return _create_base_sandbox(task, backend, image=_dockerfile_image(backend, dockerfile))
    sandbox = _create_base_sandbox(task, backend)
    _replay_dockerfile(task, sandbox, backend)
    return sandbox


def _sandbox_resource_kwargs(task: Task, backend: str) -> dict:
    """Map a harbor task's declared ``[environment]`` resources to backend kwargs.

    Harbor task.toml declares ``cpus`` / ``memory_mb`` / ``storage_mb``; without
    these a remote sandbox runs at the backend default (Daytona: 1 GiB), which
    OOM-kills compile-heavy graders (e.g. ``go test ./...``). Modal takes memory
    in MB; Daytona takes memory/disk in GB. Docker/local ignore the values.

    The sandbox lifetime is sized to this task's own budget so the box always
    outlives the agent + verifier it hosts (both run inside it). A flat default
    could be shorter than agent+verifier and reap the box mid-rollout ("Sandbox
    already shut down" / ENOSPC mid-run). The provider-agnostic
    ``RLLM_SANDBOX_TIMEOUT_S`` (seconds) is a *floor* on top of that, applied the
    same way for every backend; each backend then expresses it in its own unit
    (Modal's hard ``timeout`` in seconds; Daytona's idle ``auto_stop_interval``
    in minutes).
    """
    from rllm.env import env_int, sandbox_timeout_override_s

    env = task.metadata.get("environment", {}) or {}
    cpus, mem_mb, disk_mb = env.get("cpus"), env.get("memory_mb"), env.get("storage_mb")

    # Per-task lifetime floor (seconds), shared across backends: agent + verifier
    # + install + teardown/scheduling slack, raised to the operator override.
    agent_t = float(task.metadata.get("agent_timeout") or env_int("RLLM_HARNESS_RUN_TIMEOUT_S", 3600))
    verifier_t = float(task.metadata.get("verifier_timeout") or 600.0)
    install_t = float(env_int("RLLM_HARNESS_INSTALL_TIMEOUT_S", 600))
    lifetime_s = max(int(agent_t + verifier_t + install_t + 600), sandbox_timeout_override_s())

    kw: dict = {}
    if backend == "modal":
        if cpus:
            kw["cpu"] = float(cpus)
        if mem_mb:
            kw["memory"] = int(mem_mb)
        kw["timeout"] = lifetime_s  # Modal's hard lifetime, in seconds
    elif backend == "daytona":
        if cpus:
            kw["cpu"] = int(cpus)
        if mem_mb:
            kw["memory"] = max(1, round(mem_mb / 1024))
        if disk_mb:
            kw["disk"] = max(1, round(disk_mb / 1024))
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
