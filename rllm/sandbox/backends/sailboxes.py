"""Sailboxes sandbox backend.

Uses Sail Research Sailboxes (https://docs.sailresearch.com/sailboxes) —
persistent, stateful cloud VMs built for long-horizon agents — to run agent
code remotely.

Requires the ``sail`` package::

    pip install sail

Authentication is via the ``SAIL_API_KEY`` environment variable
(``export SAIL_API_KEY=sk_...``) or a prior ``sail auth login``.

Two things set this backend apart from Modal/Daytona:

* **No OCI registry pull.** Sailboxes images are built from a fixed Debian base
  via a builder (``pip_install``/``apt_install``/``run_commands``); there is no
  ``from_registry``. A Docker image string (``python:3.11-slim``, a prebuilt
  SWE-bench image, …) therefore can't be pulled — we boot the Debian base and
  rely on the cold-path Dockerfile ``RUN`` replay for setup. Tasks whose
  environment lives *in* a prebuilt registry image are not reproducible here.
* **Snapshots are checkpoints.** ``build_snapshot`` captures a live Sailbox via
  ``checkpoint()`` (durable, TTL-bounded) and boots from it via
  ``from_checkpoint()`` — the checkpoint/fork workflow Sailboxes documents for
  RL rollouts. The SDK exposes no checkpoint GET/list/delete, so the prune-side
  probes (:func:`_sailboxes_ref_absent`, :func:`delete_sailboxes_snapshot`) are
  conservative and lean on the checkpoint's own ``ttl_seconds`` for GC.
"""

from __future__ import annotations

import atexit
import io
import logging
import os
import shlex
import tarfile
import threading
import time
import weakref
from pathlib import Path

from rllm.env import env_int
from rllm.sandbox.protocol import SandboxCommandTimeout, SnapshotNotFound

logger = logging.getLogger(__name__)

# Default Sail App that owns rLLM's Sailboxes (grouping / billing scope).
_DEFAULT_APP_NAME = "rllm-sandbox"

# Checkpoint TTL: how long a built snapshot stays bootable backend-side before
# Sailboxes garbage-collects it. Must comfortably exceed the local registry's
# trust horizon (RLLM_SNAPSHOT_TTL_HOURS, default 7d) so a locally-trusted ref
# never resolves to a GC'd checkpoint. Mirrors the docs' 30-day example.
_CHECKPOINT_TTL_S = env_int("RLLM_SAILBOX_CHECKPOINT_TTL_S", 30 * 86400)

# Seconds to wait for a Sailbox to boot (create/from_checkpoint). Sailboxes are
# persistent (no lifetime cap), so this bounds boot only, not the run.
_DEFAULT_BOOT_TIMEOUT = env_int("RLLM_SAILBOX_BOOT_TIMEOUT_S", 600)

# atexit-tracked sandboxes; terminated on process exit to avoid leaks.
_LIVE_SANDBOXES: weakref.WeakSet = weakref.WeakSet()
_LIVE_LOCK = threading.Lock()


def _terminate_all_live() -> None:
    """atexit hook: terminate every still-alive SailboxesSandbox."""
    with _LIVE_LOCK:
        survivors = list(_LIVE_SANDBOXES)
    if not survivors:
        return
    logger.warning("atexit: terminating %d unreleased Sailbox(es)", len(survivors))
    for sb in survivors:
        try:
            sb.close()
        except Exception:
            logger.debug("atexit: error closing %s", getattr(sb, "name", "<unknown>"), exc_info=True)


atexit.register(_terminate_all_live)


def _looks_like_docker_image(image: str) -> bool:
    """Heuristic: treat strings containing ``:`` or ``/`` as Docker image refs.

    ``python:3.11-slim`` / ``ghcr.io/foo/bar:tag`` → Docker image (which
    Sailboxes cannot pull; we fall back to the Debian base). A bare token with
    neither char is treated as a checkpoint id (Sailboxes' snapshot ref).
    """
    return ":" in image or "/" in image


def _build_exec_command(command: str, persistent_env: dict[str, str] | None, user: str | int | None) -> str:
    """Wrap *command* with persistent-env exports and an optional ``su`` user-switch.

    Pure string transform (mirrors ``modal_backend._build_exec_command`` /
    ``daytona._build_exec_command``) so the exec contract is unit-testable
    without a live Sailbox:

    * ``persistent_env`` is exported ahead of the command so every exec sees the
      task's declared environment — each Sailbox exec is a fresh shell, so a
      one-shot ``export`` wouldn't persist. Mirrors harbor's per-exec env.
    * ``user`` switches via ``su <user> -s /bin/bash -c <cmd>`` — Sailboxes'
      ``run`` has no ``user=`` — with the env exports placed *inside* the
      switched shell so they reach the target user. An ``int`` is resolved as a
      uid via ``getent``; a ``str`` is used verbatim.
    """
    run = command
    if persistent_env:
        prefix = "".join(f"export {k}={shlex.quote(str(v))}; " for k, v in persistent_env.items())
        run = prefix + run
    if user is not None:
        user_arg = f"$(getent passwd {user} | cut -d: -f1)" if isinstance(user, int) else shlex.quote(str(user))
        run = f"su {user_arg} -s /bin/bash -c {shlex.quote(run)}"
    return run


class SailboxesSandbox:
    """Sandbox implementation using Sail Research Sailboxes.

    Finds/creates a Sail ``App``, creates a ``Sailbox`` via ``Sailbox.create()``,
    executes commands via ``sailbox.run()``, and moves files through the native
    ``sailbox.fs`` API.

    The ``image`` parameter accepts:

    - A **checkpoint id** (bare token, no ``:`` or ``/``) — booted via
      ``Sailbox.from_checkpoint()``; a vanished checkpoint raises
      :class:`SnapshotNotFound` for cold fallback.
    - A **Docker image string** (``python:3.11-slim`` etc.) — Sailboxes can't
      pull registries, so the Debian base is used instead (a warning is logged);
      the caller's Dockerfile ``RUN`` replay installs the real dependencies.
    - a ``sail.ImageDefinition`` object — used directly as the build spec.

    Optional kwargs:

    - ``app_name``: Sail App name (default ``"rllm-sandbox"``).
    - ``arch``: ``"amd64"`` (default) or ``"arm64"`` base architecture.
    - ``timeout``: boot-wait seconds (default 600).
    - ``size``: Sailbox size tier ``"s"`` | ``"m"``.
    - ``memory_gib`` / ``disk_gib``: resource overrides.
    - ``private``: restrict the Sailbox to the creating user (default False).
    """

    def __init__(self, name: str, image: str = "python:3.11-slim", **kwargs):
        # Lazy import so users without the SDK can still import this module.
        try:
            import sail
            from sail import App, Image, Sailbox
            from sail.errors import NotFoundError, SailboxCreationError
        except ImportError as e:
            raise ImportError("The Sailboxes sandbox backend requires the 'sail' package. Install with: pip install sail  and set SAIL_API_KEY in your environment (or run 'sail auth login').") from e

        self.name = name
        self._image_spec = image
        self._closed = False
        # Env exported into every exec (populated via set_env); see exec().
        self._persistent_env: dict[str, str] = {}
        self._boot_timeout = int(kwargs.pop("timeout", _DEFAULT_BOOT_TIMEOUT))
        app_name = kwargs.pop("app_name", _DEFAULT_APP_NAME)
        arch = kwargs.pop("arch", "amd64")

        self._app = App.find(name=app_name, mint_if_missing=True)

        create_kwargs: dict = {"app": self._app, "name": name, "timeout": self._boot_timeout}
        for key in ("size", "memory_gib", "disk_gib", "private", "ingress_ports", "ssh"):
            if key in kwargs:
                create_kwargs[key] = kwargs.pop(key)

        from_checkpoint = isinstance(image, str) and not _looks_like_docker_image(image)

        if isinstance(image, sail.ImageDefinition):
            image_label = "<sail.ImageDefinition>"
            self._sandbox = Sailbox.create(image=image, **create_kwargs)
        elif from_checkpoint:
            # Bare token → a checkpoint id we previously stored. A gone/expired
            # checkpoint surfaces as NotFound (or a creation error naming it) →
            # SnapshotNotFound so get_sandbox falls back to cold.
            image_label = f"checkpoint:{image}"
            try:
                self._sandbox = Sailbox.from_checkpoint(image, name=name, timeout=self._boot_timeout)
            except (NotFoundError, SailboxCreationError) as e:
                raise SnapshotNotFound(f"sailboxes checkpoint {image} unavailable: {e}") from e
        else:
            # A Docker image string: Sailboxes can't pull it. Boot the Debian
            # base; the Dockerfile RUN replay (cold path) installs the real deps.
            logger.warning(
                "Sailboxes cannot pull Docker image %r; booting the Debian %s base instead (dependencies come from the Dockerfile RUN replay, not the image).",
                image,
                arch,
            )
            base = Image.debian_arm64 if arch == "arm64" else Image.debian_amd64
            image_label = f"debian_{arch}"
            self._sandbox = Sailbox.create(image=base, **create_kwargs)

        self._sandbox_id = self._sandbox.sailbox_id

        with _LIVE_LOCK:
            _LIVE_SANDBOXES.add(self)

        logger.info(
            "SailboxesSandbox %s created (id: %s, image: %s)",
            name,
            self._sandbox_id,
            image_label,
        )

    def set_env(self, env: dict[str, str]) -> None:
        """Register env vars exported into every subsequent :meth:`exec`.

        Sailbox execs are independent shells, so persistent task env (Harbor's
        ``[environment].env``) must be re-applied per command. Mirrors
        ModalSandbox / DaytonaSandbox ``set_env``.
        """
        for k, v in (env or {}).items():
            self._persistent_env[str(k)] = str(v)

    def exec(self, command: str, timeout: float | None = None, user: str | int | None = None) -> str:
        """Execute a command inside the Sailbox and return stdout.

        Runs ``bash -c <command>``. Raises :class:`SandboxCommandTimeout` when
        the command exceeds ``timeout`` (Sailboxes enforces it server-side and
        flags ``ExecResult.timed_out``), and ``RuntimeError`` on any other
        non-zero exit (matching DockerSandbox behavior).

        ``user`` switches the command to another OS user via ``su`` (Sailboxes'
        ``run`` has no ``user=``), and env registered via :meth:`set_env` is
        exported into every command. See :func:`_build_exec_command`.
        """
        import sail

        run = _build_exec_command(command, self._persistent_env, user)
        run_kwargs: dict = {"check": False}
        if timeout is not None:
            run_kwargs["timeout"] = int(timeout)

        try:
            result = self._sandbox.run(["bash", "-c", run], **run_kwargs)
        except sail.TimeoutError as e:
            raise SandboxCommandTimeout(f"Command hit its {int(timeout) if timeout else '?'}s timeout in sandbox {self.name}: {command[:200]}") from e

        if result.timed_out:
            logger.warning("Command hit its %ss timeout in sandbox %s: %s", int(timeout) if timeout else "?", self.name, command[:200])
            raise SandboxCommandTimeout(f"Command hit its {int(timeout) if timeout else '?'}s timeout in sandbox {self.name}")

        stdout = result.stdout or ""
        if result.exit_code != 0:
            stderr = result.stderr or ""
            logger.debug(
                "Command failed in sandbox %s: %s\nstderr: %s",
                self.name,
                command,
                stderr[:500],
            )
            raise RuntimeError(f"Command failed (exit {result.exit_code}) in sandbox {self.name}: {command}\n{stderr[:500]}")
        return stdout

    def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload a single file via Sailboxes' native ``fs.write`` (creates parents)."""
        with open(local_path, "rb") as f:
            content = f.read()
        self._sandbox.fs.write(remote_path, content, create_parents=True)
        logger.debug("Uploaded %s -> %s in sandbox %s", local_path, remote_path, self.name)

    def upload_dir(self, local_path: str, remote_path: str) -> None:
        """Upload a directory tree.

        Sailboxes' ``fs`` API has no native recursive upload, so we package the
        tree into a single ``.tar.gz`` locally, write that one file with the
        native API, then extract it inside the Sailbox — one transfer + one
        exec, regardless of tree size. Mirrors the Daytona/Modal backends.
        """
        local = Path(local_path)
        if not local.exists():
            raise FileNotFoundError(f"upload_dir: local path {local_path} does not exist")

        remote_parent = os.path.dirname(remote_path.rstrip("/")) or "/"
        remote_name = os.path.basename(remote_path.rstrip("/")) or local.name

        self._exec_unchecked(f"mkdir -p {remote_parent}")

        tar_buf = io.BytesIO()
        with tarfile.open(fileobj=tar_buf, mode="w:gz") as tar:
            tar.add(local_path, arcname=remote_name)
        tar_buf.seek(0)

        remote_tar = f"/tmp/_upload_{self.name}_{int(time.time() * 1000)}.tar.gz"
        self._sandbox.fs.write(remote_tar, tar_buf.read(), create_parents=True)
        # --no-same-owner: don't restore the host's uid/gid (root extraction
        # would otherwise chown to nonexistent ids and fail). Permissions are
        # kept, so executables stay +x.
        self.exec(f"tar xzf {remote_tar} --no-same-owner -C {remote_parent} && rm -f {remote_tar}")
        logger.debug("Uploaded dir %s -> %s in sandbox %s", local_path, remote_path, self.name)

    def is_alive(self) -> bool:
        """One API GET: fetch the Sailbox and check it is runnable.

        ``running`` and ``sleeping`` count as alive — a sleeping Sailbox
        auto-wakes on the next exec. ``paused``/``failed``/``terminated`` (and
        any error querying) count as dead. Never raises.
        """
        if self._closed:
            return False
        try:
            from sail import Sailbox

            status = Sailbox.get(self._sandbox_id).status
            return str(status).lower() in ("running", "sleeping")
        except Exception:
            logger.debug("SailboxesSandbox %s is_alive check failed — treating as dead", self.name, exc_info=True)
            return False

    def close(self) -> None:
        """Terminate the Sailbox and release resources."""
        if self._closed:
            return
        try:
            self._sandbox.terminate()
        except Exception:
            logger.warning("Sailbox %s terminate failed — it may be orphaned until its checkpoint TTL", self.name, exc_info=True)
        with _LIVE_LOCK:
            _LIVE_SANDBOXES.discard(self)
        self._closed = True
        logger.info("SailboxesSandbox %s closed", self.name)

    def _exec_unchecked(self, command: str) -> str:
        """Execute a command without raising on non-zero exit."""
        try:
            return self.exec(command)
        except RuntimeError:
            return ""


def create_sailboxes_sandbox(name: str, image: str = "python:3.11-slim", **kwargs) -> SailboxesSandbox:
    """Factory function for creating a SailboxesSandbox."""
    return SailboxesSandbox(name=name, image=image, **kwargs)


def build_sailboxes_snapshot(task, key: str, prior_ref: str | None = None, *, force: bool = False, install_script: str = "") -> str | None:
    """Capture ``task``'s environment as a durable checkpoint; return its checkpoint id.

    Boots a base Sailbox, replays the Dockerfile ``RUN`` steps, runs the install
    script (if any), then ``checkpoint()``s the live filesystem into a
    TTL-bounded, bootable snapshot — the checkpoint/fork pattern Sailboxes
    documents for RL rollouts (analogous to Modal's ``snapshot_filesystem``).

    Reuse: the SDK exposes no checkpoint-liveness probe, so a known
    ``prior_ref`` is reused optimistically unless ``force`` (checkpoints are
    durable within their TTL, and a stale ref falls back to cold at boot). A
    failed install fails the build — a snapshot keyed on the install must
    actually contain it.
    """
    from rllm.eval._resolution import _create_base_sandbox, _dockerfile_run_commands, _replay_dockerfile, _should_replay_dockerfile

    if prior_ref and not force:
        logger.info("sailboxes snapshot %s: reusing prior checkpoint %s (no liveness probe available)", key, prior_ref)
        return prior_ref

    n_replay = len(_dockerfile_run_commands(task)) if _should_replay_dockerfile(task) else 0
    logger.info("sailboxes snapshot %s: building (%d RUN steps%s)", key, n_replay, ", +install" if install_script else "")
    # Sailboxes are persistent (no lifetime cap), so unlike Modal the build box
    # can't be reaped mid-build — each replay/install step is bounded by its own
    # per-exec timeout, and the boot-wait uses the backend default.
    sb = _create_base_sandbox(task, "sailboxes", name=f"{key}-build")
    try:
        _replay_dockerfile(task, sb, "sailboxes")
        if install_script:
            sb.exec(install_script, timeout=env_int("RLLM_HARNESS_INSTALL_TIMEOUT_S", 900), user="root")
        checkpoint = sb._sandbox.checkpoint(name=key, ttl_seconds=_CHECKPOINT_TTL_S)  # noqa: SLF001 — sail.SailboxCheckpoint
        logger.info("sailboxes snapshot built: %s -> %s", key, checkpoint.checkpoint_id)
        return checkpoint.checkpoint_id
    finally:
        try:
            sb.close()
        except Exception:
            logger.debug("build sandbox close failed", exc_info=True)


def _sailboxes_ref_absent(ref: str) -> bool:  # noqa: ARG001
    """No-boot prune probe. Always ``False`` (keep): Sailboxes exposes no checkpoint
    GET, so absence can't be confirmed without booting — the conservative contract
    for ``registry.sync`` is never to prune a record it can't verify is gone.
    Checkpoints self-expire via their ``ttl_seconds``.
    """
    return False


def delete_sailboxes_snapshot(ref: str) -> bool:
    """Delete a checkpoint. The SDK exposes no checkpoint-delete API, so this is a
    no-op returning ``False`` (keep the local record); the checkpoint's
    ``ttl_seconds`` garbage-collects it backend-side.
    """
    logger.info("sailboxes has no checkpoint-delete API; checkpoint %s will expire via its TTL", ref)
    return False
