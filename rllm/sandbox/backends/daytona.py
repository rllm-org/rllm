"""Daytona sandbox backend.

Uses Daytona Cloud Sandboxes (https://www.daytona.io/docs) to run agent
code in remote cloud containers.

Requires the ``daytona`` package::

    pip install daytona

Authentication is handled via the ``DAYTONA_API_KEY``, ``DAYTONA_API_URL``,
and ``DAYTONA_TARGET`` environment variables, or by passing an explicit
``DaytonaConfig`` via the ``daytona_config`` kwarg.
"""

from __future__ import annotations

import atexit
import io
import logging
import os
import tarfile
import threading
import time
import weakref
from pathlib import Path

from rllm.sandbox.protocol import SnapshotNotFound

logger = logging.getLogger(__name__)

# Default auto-stop interval (minutes). 0 disables auto-stop entirely;
# we rely on explicit ``close()`` plus the atexit hook below. Set to a
# finite value to belt-and-suspenders against leaked sandboxes if the
# process is SIGKILL'd (atexit won't fire then).
_DEFAULT_AUTO_STOP_INTERVAL = 30  # minutes; mirrors ModalSandbox's 30-min timeout

# atexit-tracked sandboxes; terminated on process exit to avoid leaks.
_LIVE_SANDBOXES: weakref.WeakSet = weakref.WeakSet()
_LIVE_LOCK = threading.Lock()


def _terminate_all_live() -> None:
    """atexit hook: delete every still-alive DaytonaSandbox."""
    with _LIVE_LOCK:
        survivors = list(_LIVE_SANDBOXES)
    if not survivors:
        return
    logger.warning("atexit: deleting %d unreleased Daytona sandbox(es)", len(survivors))
    for sb in survivors:
        try:
            sb.close()
        except Exception:
            logger.debug("atexit: error closing %s", getattr(sb, "name", "<unknown>"), exc_info=True)


atexit.register(_terminate_all_live)


def _looks_like_docker_image(image: str) -> bool:
    """Heuristic: treat strings containing ``:`` or ``/`` as Docker image refs.

    ``python:3.11-slim`` / ``ghcr.io/foo/bar:tag`` → Docker image (Daytona
    will pull from a registry). ``rllm-worker-v0`` → Daytona snapshot name.
    """
    return ":" in image or "/" in image


class DaytonaSandbox:
    """Sandbox implementation using Daytona Cloud Sandboxes.

    Creates a Daytona Sandbox via ``daytona.create()``, executes commands
    via ``sandbox.process.exec()``, uploads files via the native
    ``sandbox.fs`` API, and exposes ports via ``sandbox.get_preview_link()``.

    The ``image`` parameter accepts either:
    - A Daytona snapshot name (e.g. ``"rllm-worker-v0"``) — passed via
      ``CreateSandboxFromSnapshotParams``.
    - A Docker image reference (e.g. ``"python:3.11-slim"``) — passed via
      ``CreateSandboxFromImageParams`` so Daytona pulls it from the
      registry.
    - A ``daytona.Image`` object — used directly as a custom build spec.

    Optional kwargs:
    - ``daytona_config``: explicit ``DaytonaConfig`` (overrides env vars).
    - ``auto_stop_interval``: minutes; default ``30``. Pass ``0`` to disable.
    - ``auto_archive_interval``: minutes; default Daytona's own default.
    - ``cpu`` / ``memory`` / ``disk`` / ``gpu``: resource overrides (only
      effective with the from-image path).
    - ``env_vars``: dict of env vars baked into the sandbox.
    - ``labels``: dict of labels for filtering / billing.
    - ``os_user``: default user inside the sandbox.
    - ``create_timeout``: seconds to wait for ``create()``; default ``120``.
    """

    def __init__(self, name: str, image: str = "python:3.11-slim", **kwargs):
        # Lazy import so users without the SDK can still import this module.
        try:
            from daytona import (
                CreateSandboxFromImageParams,
                CreateSandboxFromSnapshotParams,
                Daytona,
                DaytonaNotFoundError,
                DaytonaValidationError,
                Image,
                Resources,
            )
        except ImportError as e:
            raise ImportError(
                "The Daytona sandbox backend requires the 'daytona' package. "
                "Install with: pip install daytona  "
                "(or, for harbor agents: pip install 'harbor[daytona]'). "
                "Also set DAYTONA_API_KEY in your environment."
            ) from e

        self.name = name
        self._image_spec = image
        self._closed = False
        self._auto_stop_interval = int(kwargs.pop("auto_stop_interval", _DEFAULT_AUTO_STOP_INTERVAL))
        self._auto_archive_interval = kwargs.pop("auto_archive_interval", None)
        self._create_timeout = float(kwargs.pop("create_timeout", 120.0))

        # Client init
        daytona_config = kwargs.pop("daytona_config", None)
        self._client = Daytona(daytona_config) if daytona_config is not None else Daytona()

        # Build common parameters
        base_kwargs: dict = {
            "name": name,
            "auto_stop_interval": self._auto_stop_interval,
        }
        if self._auto_archive_interval is not None:
            base_kwargs["auto_archive_interval"] = int(self._auto_archive_interval)
        for key in ("env_vars", "labels", "os_user", "public", "volumes", "auto_delete_interval"):
            if key in kwargs:
                base_kwargs[key] = kwargs.pop(key)

        # Resources only apply to the from-image path
        resources = None
        resource_keys = {"cpu", "memory", "disk", "gpu"}
        if resource_keys & kwargs.keys():
            resources = Resources(**{k: kwargs.pop(k) for k in list(kwargs.keys()) if k in resource_keys})

        # Decide create path: snapshot name vs Docker image vs Image object
        if isinstance(image, Image):
            params = CreateSandboxFromImageParams(image=image, resources=resources, **base_kwargs)
            self._sandbox = self._client.create(params, timeout=self._create_timeout)
            image_label = "<daytona.Image>"
        elif isinstance(image, str) and _looks_like_docker_image(image):
            params = CreateSandboxFromImageParams(image=image, resources=resources, **base_kwargs)
            self._sandbox = self._client.create(params, timeout=self._create_timeout)
            image_label = image
        else:
            # Treat as Daytona snapshot name. Resources are baked into the
            # snapshot at build time, so any re-passed here are ignored (debug,
            # not a warning — booting from a snapshot is the expected path).
            if resources is not None:
                logger.debug("Resources ignored when creating from snapshot %s; snapshot resources win.", image)
            params = CreateSandboxFromSnapshotParams(snapshot=image, **base_kwargs)
            try:
                self._sandbox = self._client.create(params, timeout=self._create_timeout)
            except (DaytonaNotFoundError, DaytonaValidationError) as e:
                # A gone/removing snapshot surfaces as 404 (NotFound) or 400
                # (Validation, e.g. "Snapshot <name> is removing") — both mean
                # cold fallback. Validation errors that don't name this snapshot
                # (bad resources, etc.) are real and must propagate.
                msg = str(e).lower()
                if isinstance(e, DaytonaNotFoundError) or image.lower() in msg or any(w in msg for w in ("not found", "does not exist", "removing", "removed")):
                    raise SnapshotNotFound(f"daytona snapshot {image} unavailable: {e}") from e
                raise
            image_label = f"snapshot:{image}"

        self._sandbox_id = getattr(self._sandbox, "id", None) or getattr(self._sandbox, "sandbox_id", None)

        with _LIVE_LOCK:
            _LIVE_SANDBOXES.add(self)

        logger.info(
            "DaytonaSandbox %s created (id: %s, image: %s)",
            name,
            self._sandbox_id,
            image_label,
        )

    def exec(self, command: str, timeout: float | None = None, user: str | None = None) -> str:  # noqa: ARG002
        """Execute a command inside the Daytona sandbox.

        ``user`` is accepted for protocol compatibility but ignored —
        Daytona sets the default user at sandbox creation time
        (``os_user`` kwarg), not per exec.

        Wraps the command in ``bash -c`` for shell-feature parity with
        DockerSandbox / ModalSandbox. Returns stdout. Raises
        ``RuntimeError`` on non-zero exit.
        """
        exec_kwargs: dict = {}
        if timeout is not None:
            exec_kwargs["timeout"] = int(timeout)

        response = self._sandbox.process.exec(f"bash -c {_shell_quote(command)}", **exec_kwargs)

        stdout = response.result or ""
        exit_code = response.exit_code

        if exit_code != 0:
            tail = stdout[-500:]
            logger.warning(
                "Command failed in sandbox %s: %s\noutput tail: %s",
                self.name,
                command,
                tail,
            )
            raise RuntimeError(f"Command failed (exit {exit_code}) in sandbox {self.name}: {command}\n{tail}")
        return stdout

    def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload a single file via Daytona's native ``fs.upload_file``."""
        remote_dir = os.path.dirname(remote_path)
        if remote_dir:
            self._exec_unchecked(f"mkdir -p {remote_dir}")
        self._sandbox.fs.upload_file(local_path, remote_path)
        logger.debug("Uploaded %s -> %s in sandbox %s", local_path, remote_path, self.name)

    def upload_dir(self, local_path: str, remote_path: str) -> None:
        """Upload a directory tree.

        Daytona's ``fs`` API has no native recursive upload, so we package
        the tree into a single ``.tar.gz`` locally, upload that one file
        with the native API, then extract it inside the sandbox. One HTTP
        upload + one exec, regardless of tree size.
        """
        local = Path(local_path)
        if not local.exists():
            raise FileNotFoundError(f"upload_dir: local path {local_path} does not exist")

        remote_parent = os.path.dirname(remote_path.rstrip("/")) or "/"
        remote_name = os.path.basename(remote_path.rstrip("/")) or local.name

        self._exec_unchecked(f"mkdir -p {remote_parent}")

        # Build tar.gz in memory, write to a temp file, upload, extract.
        tar_buf = io.BytesIO()
        with tarfile.open(fileobj=tar_buf, mode="w:gz") as tar:
            tar.add(local_path, arcname=remote_name)
        tar_buf.seek(0)

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp.write(tar_buf.read())
            tmp_path = tmp.name

        try:
            remote_tar = f"/tmp/_upload_{self.name}_{int(time.time() * 1000)}.tar.gz"
            self._sandbox.fs.upload_file(tmp_path, remote_tar)
            # --no-same-owner: don't restore the host's uid/gid (root extraction
            # would otherwise chown to nonexistent ids and fail). Permissions are
            # kept, so executables stay +x.
            self.exec(f"tar xzf {remote_tar} --no-same-owner -C {remote_parent} && rm -f {remote_tar}")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        logger.debug("Uploaded dir %s -> %s in sandbox %s", local_path, remote_path, self.name)

    def is_alive(self) -> bool:
        """One API GET: refresh sandbox data and check it is still started.

        A Daytona sandbox auto-stops after ``auto_stop_interval`` idle
        minutes; a stopped (or errored/deleted) sandbox fails its first
        exec with "no IP address found", so anything not ``started``
        counts as dead here.
        """
        if self._closed:
            return False
        try:
            self._sandbox.refresh_data()
            state = getattr(self._sandbox, "state", None)
            return str(getattr(state, "value", state) or "").lower() == "started"
        except Exception:
            logger.debug("DaytonaSandbox %s is_alive check failed — treating as dead", self.name, exc_info=True)
            return False

    def close(self) -> None:
        """Delete the Daytona sandbox and release resources."""
        if self._closed:
            return
        try:
            self._sandbox.delete()
        except Exception:
            # A failed delete (e.g. API key without delete scope) leaks a billed
            # sandbox — surface it, then stop() so compute halts now, not at auto_stop.
            logger.warning("Sandbox %s delete failed; attempting stop() fallback", self.name, exc_info=True)
            try:
                self._sandbox.stop()
            except Exception:
                logger.warning("Sandbox %s stop() fallback also failed — it may be orphaned until auto_stop", self.name, exc_info=True)
        with _LIVE_LOCK:
            _LIVE_SANDBOXES.discard(self)
        self._closed = True
        logger.info("DaytonaSandbox %s closed", self.name)

    def _exec_unchecked(self, command: str) -> str:
        """Execute a command without raising on non-zero exit."""
        try:
            return self.exec(command)
        except RuntimeError:
            return ""


def _shell_quote(s: str) -> str:
    """Single-quote a string for safe interpolation into a shell command."""
    return "'" + s.replace("'", "'\\''") + "'"


def create_daytona_sandbox(name: str, image: str = "python:3.11-slim", **kwargs) -> DaytonaSandbox:
    """Factory function for creating a DaytonaSandbox."""
    return DaytonaSandbox(name=name, image=image, **kwargs)


def build_daytona_snapshot(task, key: str, *, force: bool = False, install_script: str = "") -> str | None:
    """Declaratively bake ``task``'s base image + RUN steps into a named snapshot; return the name.

    Idempotent: an already-registered snapshot is reused unless ``force``, which
    deletes the existing snapshot first so the rebuild replaces it.
    """
    from daytona import CreateSnapshotParams, Daytona, DaytonaNotFoundError, Image, Resources

    from rllm.eval._resolution import _as_single_run_line, _dockerfile_run_commands, _resolve_image, _sandbox_resource_kwargs

    client = Daytona()
    try:
        existing = client.snapshot.get(key)
        if not force:
            logger.info("daytona snapshot %s already registered", key)
            return key
        client.snapshot.delete(existing)  # force: replace the existing snapshot
        logger.info("daytona snapshot %s deleted for forced rebuild", key)
    except DaytonaNotFoundError:
        pass

    img = Image.base(_resolve_image(task, "daytona"))
    run_commands = [_as_single_run_line(c) for c in _dockerfile_run_commands(task)]
    if install_script:
        run_commands.append(_as_single_run_line(install_script))
    if run_commands:
        img = img.run_commands(*run_commands)

    res_kw = _sandbox_resource_kwargs(task, "daytona")
    resources = Resources(**res_kw) if res_kw else None
    client.snapshot.create(
        CreateSnapshotParams(name=key, image=img, resources=resources),
        on_logs=lambda chunk: logger.debug("daytona build[%s]: %s", key, chunk.rstrip()),
    )
    logger.info("daytona snapshot built: %s", key)
    return key


def _daytona_ref_absent(ref: str) -> bool:
    """True only when the snapshot is verifiably gone (NotFound / terminal state); ``False`` (keep) on any unconfirmed error."""
    from daytona import (
        Daytona,
        DaytonaAuthenticationError,
        DaytonaAuthorizationError,
        DaytonaNotFoundError,
        DaytonaRateLimitError,
    )

    try:
        snapshot = Daytona().snapshot.get(ref)
        state = getattr(snapshot, "state", None)
        name = str(getattr(state, "value", state) or "").lower()  # state is a SnapshotState enum
        return name in {"error", "build_failed", "removing", "removed"}
    except DaytonaNotFoundError:
        return True  # confirmed gone
    except (DaytonaAuthenticationError, DaytonaAuthorizationError, DaytonaRateLimitError):
        return False  # cannot confirm — keep
    except Exception:
        logger.debug("daytona ref probe failed for %s — treating as unknown", ref, exc_info=True)
        return False


def delete_daytona_snapshot(ref: str) -> bool:
    """Delete a Daytona snapshot by name; ``True`` only when confirmed gone, ``False`` (keep the local record) otherwise."""
    from daytona import (
        Daytona,
        DaytonaAuthenticationError,
        DaytonaAuthorizationError,
        DaytonaNotFoundError,
        DaytonaRateLimitError,
    )

    try:
        client = Daytona()
        client.snapshot.delete(client.snapshot.get(ref))
        return True
    except DaytonaNotFoundError:
        return True  # already gone — safe to drop
    except (DaytonaAuthenticationError, DaytonaAuthorizationError, DaytonaRateLimitError):
        logger.warning("no permission to delete daytona snapshot %s — keeping local record", ref)
        return False
    except Exception:
        logger.warning("failed to delete daytona snapshot %s — keeping local record", ref, exc_info=True)
        return False
