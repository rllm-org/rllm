"""OpenSandbox sandbox backend.

Uses OpenSandbox (https://github.com/alibaba/opensandbox, https://open-sandbox.ai)
to run agent code in remote isolated environments.

Requires the ``opensandbox`` package::

    pip install opensandbox

Authentication is handled via the ``OPEN_SANDBOX_API_KEY`` and ``OPEN_SANDBOX_DOMAIN``
environment variables (or an explicit ``ConnectionConfigSync`` via the
``connection_config`` kwarg) — the SDK's own defaults, so anything the SDK
accepts works here unchanged.

Snapshot mapping: OpenSandbox snapshots are captures of a *live* sandbox's
filesystem (like Modal, unlike Daytona's declarative bakes), so
:func:`build_opensandbox_snapshot` boots a build sandbox, replays the task's
Dockerfile ``RUN`` steps plus the install script, captures a named snapshot,
and stores its ``snap-*`` id as the registry ref. Booting from a ref goes
through ``SandboxSync.create(snapshot_id=...)``; a ref that no longer resolves
raises :class:`~rllm.sandbox.protocol.SnapshotNotFound` so the caller falls
back to the cold path.

Warm-queue note: OpenSandbox sandboxes have a server-side lifetime (default
30 minutes here, configurable) rather than Daytona-style idle auto-stop, and
no provider-side restart — a sandbox that outlives its lifetime is gone, and
:meth:`OpenSandboxSandbox.is_alive` (an execd ping) reports that so the warm
queue transparently replaces it. OpenSandbox's ``pause``/``resume`` could
support warm-queue *parking* of unused sandboxes, but the rllm protocol has
no park/resume surface, so it is out of scope here.
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
from datetime import timedelta
from pathlib import Path

from rllm.env import env_int
from rllm.sandbox.protocol import SnapshotNotFound

logger = logging.getLogger(__name__)

# Default sandbox lifetime (seconds). The SDK defaults to 10 minutes, which a
# warm-queue-parked sandbox or a long rollout can outlive; 30 minutes mirrors
# DaytonaSandbox's auto-stop default and ModalSandbox's timeout. 0 disables the
# lifetime entirely (manual cleanup only — leaks survive until killed by hand).
_DEFAULT_LIFETIME_S = env_int("RLLM_OPEN_SANDBOX_TIMEOUT_S", 30 * 60)

# Default ready wait: the SDK's own default is 30s, but a first boot from a
# fresh image includes the registry pull, which routinely exceeds that.
_DEFAULT_READY_TIMEOUT_S = 120.0

# How long create → poll-until-Ready may take after the capture request
# (capture is asynchronous server-side; the SDK call only enqueues it).
_DEFAULT_SNAPSHOT_CAPTURE_TIMEOUT_S = env_int("RLLM_OPEN_SANDBOX_SNAPSHOT_TIMEOUT_S", 900)

_POLL_INTERVAL_S = 2.0

# Terminal snapshot states: no further transition to Ready is possible.
_SNAPSHOT_TERMINAL_STATES = {"failed", "deleting"}

# atexit-tracked sandboxes; terminated on process exit to avoid leaks.
_LIVE_SANDBOXES: weakref.WeakSet = weakref.WeakSet()
_LIVE_LOCK = threading.Lock()


def _terminate_all_live() -> None:
    """atexit hook: destroy every still-alive OpenSandboxSandbox."""
    with _LIVE_LOCK:
        survivors = list(_LIVE_SANDBOXES)
    if not survivors:
        return
    logger.warning("atexit: destroying %d unreleased OpenSandbox sandbox(es)", len(survivors))
    for sb in survivors:
        try:
            sb.close()
        except Exception:
            logger.debug("atexit: error closing %s", getattr(sb, "name", "<unknown>"), exc_info=True)


atexit.register(_terminate_all_live)


def _looks_like_docker_image(image: str) -> bool:
    """Heuristic: treat strings containing ``:`` or ``/`` as Docker image refs.

    ``python:3.11-slim`` / ``ghcr.io/foo/bar:tag`` → image (pulled from a
    registry). Anything else — in practice a stored ``snap-*`` id from the
    snapshot registry — boots via ``snapshot_id``.
    """
    return ":" in image or "/" in image


def _snapshot_state(info) -> str:
    """Lowercased snapshot lifecycle state (``Creating``/``Ready``/``Failed``/``Deleting``)."""
    state = getattr(getattr(info, "status", None), "state", None)
    return str(getattr(state, "value", state) or "").lower()


class OpenSandboxSandbox:
    """Sandbox implementation using OpenSandbox managed environments.

    Creates a sandbox via ``SandboxSync.create()``, executes commands via
    ``sandbox.commands.run()`` (the server runs the command through its
    preferred shell, so shell features work without a ``bash -c`` wrapper),
    uploads files via the native ``sandbox.files`` API, and deletes the
    sandbox on :meth:`close`.

    The ``image`` parameter accepts either:
    - A container image reference (e.g. ``"python:3.11-slim"``) — passed via
      ``SandboxSync.create(image=...)``.
    - A snapshot id from the snapshot registry (e.g. ``"snap-..."``) — passed
      via ``SandboxSync.create(snapshot_id=...)``. A ref that no longer
      resolves raises :class:`SnapshotNotFound` (cold fallback).

    Optional kwargs:
    - ``lifetime_seconds``: sandbox lifetime; default
      ``RLLM_OPEN_SANDBOX_TIMEOUT_S`` (1800). ``0`` disables the lifetime.
    - ``ready_timeout``: seconds to wait for the sandbox to become healthy;
      default ``120``.
    - ``env_vars``: dict of env vars baked into the sandbox.
    - ``metadata``: dict of metadata (``name`` is always recorded here — the
      API has no sandbox-name field, so ``rllm``'s name survives as metadata).
    - ``resource``: ``{"cpu": "2", "memory": "4Gi"}`` resource limits.
    - ``entrypoint``: command to run as the sandbox entrypoint.
    - ``connection_config``: explicit ``ConnectionConfigSync`` (overrides env
      vars).
    """

    def __init__(self, name: str, image: str = "python:3.11-slim", **kwargs):
        # Lazy import so users without the SDK can still import this module.
        try:
            from opensandbox.exceptions import SandboxApiException
            from opensandbox.models.sandboxes import SandboxImageSpec
            from opensandbox.sync import SandboxSync
        except ImportError as e:
            raise ImportError(
                "The OpenSandbox sandbox backend requires the 'opensandbox' package. Install with: pip install opensandbox  Also set OPEN_SANDBOX_API_KEY / OPEN_SANDBOX_DOMAIN in your environment."
            ) from e

        self.name = name
        self._image_spec = image
        self._closed = False
        self._lifetime_s = int(kwargs.pop("lifetime_seconds", _DEFAULT_LIFETIME_S))
        self._ready_timeout_s = float(kwargs.pop("ready_timeout", _DEFAULT_READY_TIMEOUT_S))
        self._connection_config = kwargs.pop("connection_config", None)
        self._resource = kwargs.pop("resource", None)
        self._entrypoint = kwargs.pop("entrypoint", None)

        metadata = dict(kwargs.pop("metadata", None) or {})
        metadata.setdefault("rllm_name", name)
        env_vars = kwargs.pop("env_vars", None)

        lifetime = timedelta(seconds=self._lifetime_s) if self._lifetime_s > 0 else None

        # Decide create path: registry snapshot ref vs container image.
        if isinstance(image, str) and _looks_like_docker_image(image):
            self._sandbox = SandboxSync.create(
                image=SandboxImageSpec(image=image),
                timeout=lifetime,
                ready_timeout=timedelta(seconds=self._ready_timeout_s),
                env=env_vars,
                metadata=metadata,
                resource=self._resource,
                entrypoint=self._entrypoint,
                connection_config=self._connection_config,
            )
            image_label = image
        else:
            # Registry ref: a snapshot id. The SDK requires exactly one of
            # image / snapshot_id, so env/resource knobs that only apply to
            # a fresh image boot are dropped here (debug, not a warning —
            # booting from a snapshot is the expected path).
            if self._resource is not None or env_vars:
                logger.debug(
                    "env/resource overrides ignored when creating from snapshot %s; snapshot contents win.",
                    image,
                )
            try:
                self._sandbox = SandboxSync.create(
                    snapshot_id=image,
                    timeout=lifetime,
                    ready_timeout=timedelta(seconds=self._ready_timeout_s),
                    metadata=metadata,
                    connection_config=self._connection_config,
                )
            except SandboxApiException as e:
                # A gone snapshot surfaces as HTTP 404 — that means cold
                # fallback. Only the snapshot path maps it: a 404 on a plain
                # image ref is a real miss and must propagate as-is.
                if getattr(e, "status_code", None) == 404:
                    raise SnapshotNotFound(f"opensandbox snapshot {image} unavailable: {e}") from e
                raise
            image_label = f"snapshot:{image}"

        self._sandbox_id = self._sandbox.id

        with _LIVE_LOCK:
            _LIVE_SANDBOXES.add(self)

        logger.info(
            "OpenSandboxSandbox %s created (id: %s, image: %s)",
            name,
            self._sandbox_id,
            image_label,
        )

    def exec(self, command: str, timeout: float | None = None, user: str | None = None) -> str:
        """Execute a command inside the OpenSandbox environment.

        The server runs ``command`` through its preferred login shell (bash
        when available, else sh), so shell features need no ``bash -c``
        wrapper. Returns stdout. Raises ``RuntimeError`` on non-zero exit
        (or a missing exit code — e.g. the command never spawned).

        ``user`` is honored when it is a numeric UID (the API takes
        ``uid``/``gid`` ints); a username cannot be resolved client-side and
        is ignored with a debug log, like on the backends with no per-exec
        user at all. When the server-side ``timeout`` elapses it kills the
        command, which surfaces here as a non-zero exit.
        """
        from opensandbox.models.execd import RunCommandOpts

        opts_kwargs: dict = {}
        if timeout is not None:
            opts_kwargs["timeout"] = timedelta(seconds=float(timeout))
        if user is not None:
            uid = str(user).strip()
            if uid.isdigit():
                opts_kwargs["uid"] = int(uid)
            else:
                logger.debug(
                    "user=%r ignored: OpenSandbox exec takes a numeric uid, not a username.",
                    user,
                )

        execution = self._sandbox.commands.run(command, opts=RunCommandOpts(**opts_kwargs))

        stdout = "".join(msg.text for msg in execution.logs.stdout)
        exit_code = execution.exit_code

        if exit_code is None or exit_code != 0:
            stderr = "".join(msg.text for msg in execution.logs.stderr)
            tail = (stderr or stdout)[-500:]
            logger.warning(
                "Command failed in sandbox %s: %s\noutput tail: %s",
                self.name,
                command,
                tail,
            )
            raise RuntimeError(f"Command failed (exit {exit_code}) in sandbox {self.name}: {command}\n{tail}")
        return stdout

    def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload a single file via OpenSandbox's native ``files.write_file``."""
        remote_dir = os.path.dirname(remote_path)
        if remote_dir:
            self._exec_unchecked(f"mkdir -p {_shell_quote(remote_dir)}")
        with open(local_path, "rb") as f:
            self._sandbox.files.write_file(remote_path, f.read())
        logger.debug("Uploaded %s -> %s in sandbox %s", local_path, remote_path, self.name)

    def upload_dir(self, local_path: str, remote_path: str) -> None:
        """Upload a directory tree.

        The ``files`` API has no native recursive upload, so we package the
        tree into a single ``.tar.gz`` locally, write it through the native
        API (byte payload — no argv limit), then extract it inside the
        sandbox. One upload + one exec, regardless of tree size.
        """
        local = Path(local_path)
        if not local.exists():
            raise FileNotFoundError(f"upload_dir: local path {local_path} does not exist")

        remote_parent = os.path.dirname(remote_path.rstrip("/")) or "/"
        remote_name = os.path.basename(remote_path.rstrip("/")) or local.name

        self._exec_unchecked(f"mkdir -p {_shell_quote(remote_parent)}")

        tar_buf = io.BytesIO()
        with tarfile.open(fileobj=tar_buf, mode="w:gz") as tar:
            tar.add(local_path, arcname=remote_name)

        remote_tar = f"/tmp/_upload_{self._sandbox_id}_{int(time.time() * 1000)}.tar.gz"
        self._sandbox.files.write_file(remote_tar, tar_buf.getvalue())
        # --no-same-owner: don't restore the host's uid/gid (root extraction
        # would otherwise chown to nonexistent ids and fail). Permissions are
        # kept, so executables stay +x.
        self.exec(f"tar xzf {_shell_quote(remote_tar)} --no-same-owner -C {_shell_quote(remote_parent)} && rm -f {_shell_quote(remote_tar)}")

        logger.debug("Uploaded dir %s -> %s in sandbox %s", local_path, remote_path, self.name)

    def is_alive(self) -> bool:
        """One execd ping: the SDK's own health check, which never raises.

        An OpenSandbox sandbox whose server-side lifetime elapsed (or which
        was killed out-of-band) stops answering the ping, so anything not
        healthy counts as dead here.
        """
        if self._closed:
            return False
        try:
            return bool(self._sandbox.is_healthy())
        except Exception:
            logger.debug("OpenSandboxSandbox %s is_alive check failed — treating as dead", self.name, exc_info=True)
            return False

    def close(self) -> None:
        """Destroy the OpenSandbox sandbox and release resources."""
        if self._closed:
            return
        try:
            # destroy() = kill (remote) + close (local HTTP resources), with
            # local cleanup guaranteed even when the kill fails.
            self._sandbox.destroy()
        except Exception:
            # A failed kill leaks a billed sandbox — surface it. The finite
            # lifetime bounds the leak (unless lifetime was disabled).
            logger.warning(
                "Sandbox %s destroy failed — it may be orphaned until its lifetime expires",
                self.name,
                exc_info=True,
            )
        with _LIVE_LOCK:
            _LIVE_SANDBOXES.discard(self)
        self._closed = True
        logger.info("OpenSandboxSandbox %s closed", self.name)

    def _exec_unchecked(self, command: str) -> str:
        """Execute a command without raising on non-zero exit."""
        try:
            return self.exec(command)
        except RuntimeError:
            return ""


def _shell_quote(s: str) -> str:
    """Single-quote a string for safe interpolation into a shell command."""
    return "'" + s.replace("'", "'\\''") + "'"


def create_opensandbox_sandbox(name: str, image: str = "python:3.11-slim", **kwargs) -> OpenSandboxSandbox:
    """Factory function for creating an OpenSandboxSandbox."""
    return OpenSandboxSandbox(name=name, image=image, **kwargs)


def build_opensandbox_snapshot(task, key: str, prior_ref: str | None = None, *, force: bool = False, install_script: str = "") -> str | None:
    """Build an OpenSandbox snapshot of ``task``'s environment; return its ``snap-*`` id.

    Mirrors Modal's live-FS capture flow (OpenSandbox snapshots are captures of
    a running sandbox, not declarative bakes): boot a build sandbox, replay the
    Dockerfile ``RUN`` steps, run the install script, capture a named snapshot,
    and wait for it to reach ``Ready``. A known-live ``prior_ref`` is reused
    unless ``force`` — every fresh capture orphans the old snapshot forever. A
    failed install fails the build: a snapshot keyed on the install must
    actually contain it.
    """
    from rllm.eval._resolution import _create_base_sandbox, _dockerfile_run_commands, _replay_dockerfile

    if prior_ref and not force and _opensandbox_ref_alive(prior_ref):
        logger.info("opensandbox snapshot %s already live (%s) — reusing", key, prior_ref)
        return prior_ref

    # Size the build sandbox's lifetime to the worst-case replay: each RUN is
    # bounded at 900s (a step that hangs against a prebuilt image burns its
    # full bound), plus the install bound and the async capture wait.
    install_budget = env_int("RLLM_HARNESS_INSTALL_TIMEOUT_S", 900) if install_script else 0
    build_lifetime = max(
        _DEFAULT_LIFETIME_S,
        900 * len(_dockerfile_run_commands(task)) + install_budget + _DEFAULT_SNAPSHOT_CAPTURE_TIMEOUT_S,
    )
    sb = _create_base_sandbox(task, "opensandbox", name=f"{key}-build", lifetime_seconds=build_lifetime)
    try:
        _replay_dockerfile(task, sb, "opensandbox")
        if install_script:
            sb.exec(install_script, timeout=env_int("RLLM_HARNESS_INSTALL_TIMEOUT_S", 900), user="root")
        return _capture_and_wait(sb, key)
    finally:
        try:
            sb.close()
        except Exception:
            logger.debug("build sandbox close failed", exc_info=True)


def _capture_and_wait(sb: OpenSandboxSandbox, key: str) -> str:
    """Request a named snapshot from the build sandbox and poll until it is Ready.

    ``create_snapshot`` only enqueues the capture (the SDK call returns once
    the request is accepted); the artifact becomes bootable at ``Ready``.
    """
    info = sb._sandbox.create_snapshot(name=key)  # noqa: SLF001 — the SDK handle
    snapshot_id = info.id
    deadline = time.monotonic() + _DEFAULT_SNAPSHOT_CAPTURE_TIMEOUT_S
    while time.monotonic() < deadline:
        state = _snapshot_state(info)
        if state == "ready":
            logger.info("opensandbox snapshot built: %s -> %s", key, snapshot_id)
            return snapshot_id
        if state in _SNAPSHOT_TERMINAL_STATES:
            raise RuntimeError(f"opensandbox snapshot {key} ({snapshot_id}) reached terminal state {state!r}")
        time.sleep(_POLL_INTERVAL_S)
        info = _get_snapshot(snapshot_id)
    raise RuntimeError(f"opensandbox snapshot {key} ({snapshot_id}) not Ready after {_DEFAULT_SNAPSHOT_CAPTURE_TIMEOUT_S}s")


def _get_snapshot(snapshot_id: str):
    """Fetch one snapshot via the no-boot manager path.

    Typed ``SandboxApiException``s pass through unwrapped (callers match on
    ``status_code``); anything else — transport failures, missing config — is
    wrapped so callers only ever see SDK exception types or ``RuntimeError``.
    """
    from opensandbox.exceptions import SandboxApiException
    from opensandbox.sync import SandboxManagerSync

    try:
        return SandboxManagerSync.create().get_snapshot(snapshot_id)
    except SandboxApiException:
        raise
    except Exception as e:
        raise RuntimeError(f"opensandbox get_snapshot {snapshot_id} failed: {e}") from e


def _opensandbox_ref_alive(ref: str) -> bool:
    """True only when the snapshot is confirmed present and Ready, for build reuse.

    Unsure (auth/unknown/still Creating) returns ``False`` so the caller
    rebuilds rather than reuse a ref it cannot confirm.
    """
    from opensandbox.exceptions import SandboxApiException

    try:
        return _snapshot_state(_get_snapshot(ref)) == "ready"
    except SandboxApiException as e:
        if getattr(e, "status_code", None) == 404:
            return False
        logger.debug("opensandbox ref probe failed for %s — treating as unknown", ref, exc_info=True)
        return False
    except Exception:
        logger.debug("opensandbox ref probe failed for %s — treating as unknown", ref, exc_info=True)
        return False


def _opensandbox_ref_absent(ref: str) -> bool:
    """True only when the snapshot is verifiably gone (404 / terminal state); ``False`` (keep) on any unconfirmed error."""
    from opensandbox.exceptions import SandboxApiException

    try:
        return _snapshot_state(_get_snapshot(ref)) in _SNAPSHOT_TERMINAL_STATES
    except SandboxApiException as e:
        return getattr(e, "status_code", None) == 404  # confirmed gone only
    except Exception:
        logger.debug("opensandbox ref probe failed for %s — treating as unknown (keep)", ref, exc_info=True)
        return False


def delete_opensandbox_snapshot(ref: str) -> bool:
    """Delete an OpenSandbox snapshot by id; ``True`` only when confirmed gone, ``False`` (keep the local record) otherwise."""
    from opensandbox.exceptions import SandboxApiException
    from opensandbox.sync import SandboxManagerSync

    try:
        SandboxManagerSync.create().delete_snapshot(ref)
        return True
    except SandboxApiException as e:
        if getattr(e, "status_code", None) == 404:
            return True  # already gone — safe to drop
        logger.warning("failed to delete opensandbox snapshot %s — keeping local record: %s", ref, e)
        return False
    except Exception:
        logger.warning("failed to delete opensandbox snapshot %s — keeping local record", ref, exc_info=True)
        return False
