"""Runta sandbox backend.

Uses Runta runtimes (https://runta.com/docs/) to run agent code in remote
cloud machines.

Requires the ``runta-sdk`` package::

    pip install runta-sdk

Authentication is handled via the ``RUNTA_TOKEN`` environment variable or the
Runta config file written by ``runta login``; ``RUNTA_ENDPOINT`` overrides the
API endpoint.

A Runta runtime is a full machine rather than a container, and it is not
created from a Docker image. To keep parity with the other backends, the task
image runs as a long-lived container *inside* the runtime (Docker is
preinstalled there) and every command is routed through ``docker exec``.
"""

from __future__ import annotations

import atexit
import logging
import os
import shlex
import threading
import uuid
import weakref

from rllm.env import env_int
from rllm.sandbox.protocol import SnapshotNotFound

logger = logging.getLogger(__name__)

# Server-side lifetime backstop: Runta shuts the runtime down after this many
# seconds even if this process dies without reaching ``close()`` (SIGKILL, OOM),
# then deletes the stopped remains. Raise via the env var for workloads whose
# single rollout can exceed it.
_DEFAULT_TIMEOUT = env_int("RLLM_RUNTA_SANDBOX_TIMEOUT_S", 30 * 60)
_DELETE_AFTER_SHUTDOWN_S = 10 * 60

# A stored snapshot ref is a Runta checkpoint id behind this prefix, so it can
# never be mistaken for a Docker image reference.
_CHECKPOINT_PREFIX = "runta-checkpoint:"

# Name of the task container inside the runtime.
_CONTAINER = "rllm-sandbox"

# Runta can intercept outbound TLS; its merged CA bundle is present at this
# path inside nested containers. Pointing the common tools at it keeps
# ``pip`` / ``curl`` / ``git`` working under an egress policy.
_CA_BUNDLE = "/run/runta/ca-bundle.pem"
_CA_ENV_VARS = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "PIP_CERT", "GIT_SSL_CAINFO")

# Runta's exec timeout kills only the ``docker exec`` client on the runtime; the
# process inside the task container keeps running. A command with a timeout
# therefore runs under ``_TRACK``, which records its process group in a PID
# file (removed again when the command ends in time), and ``_KILL`` stops a
# timed-out command through that file. ``setsid -w`` gives the command a group
# of its own where the exec'd shell does not already lead one.
_TRACKED = '{ echo $$ > "$1"; } 2>/dev/null; bash -c "$2"; rc=$?; rm -f "$1"; exit $rc'
_TRACK = f'if setsid -w true >/dev/null 2>&1; then exec setsid -w bash -c {shlex.quote(_TRACKED)} rllm-exec "$@"; fi; {_TRACKED}'
_KILL = 'pid=$(cat "$1" 2>/dev/null) || exit 0; rm -f "$1"; kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null; exit 0'

# atexit-tracked sandboxes; deleted on process exit to avoid leaks.
_LIVE_SANDBOXES: weakref.WeakSet = weakref.WeakSet()
_LIVE_LOCK = threading.Lock()


def _terminate_all_live() -> None:
    """atexit hook: delete every still-alive RuntaSandbox."""
    with _LIVE_LOCK:
        survivors = list(_LIVE_SANDBOXES)
    if not survivors:
        return
    logger.warning("atexit: deleting %d unreleased Runta sandbox(es)", len(survivors))
    for sb in survivors:
        try:
            sb.close()
        except Exception:
            logger.debug("atexit: error closing %s", getattr(sb, "name", "<unknown>"), exc_info=True)


atexit.register(_terminate_all_live)


def _import_runta():
    """Lazy import so users without the SDK can still import this module."""
    try:
        import runta
    except ImportError as e:
        raise ImportError("The Runta sandbox backend requires the 'runta-sdk' package. Install with: pip install runta-sdk  Also set RUNTA_TOKEN in your environment (or run `runta login`).") from e
    return runta


def _is_not_found(error: Exception) -> bool:
    return getattr(error, "status_code", None) == 404


class RuntaSandbox:
    """Sandbox implementation using Runta runtimes.

    The ``image`` parameter accepts either:
    - A Docker image reference (e.g. ``"python:3.11-slim"``) — pulled and run
      as a container inside a fresh runtime.
    - A snapshot ref returned by :func:`build_runta_snapshot`
      (``"runta-checkpoint:<id>"``) — the runtime is restored from that
      checkpoint with the task container already running.
    - ``None`` / ``""`` — no container; commands run directly on the runtime.

    Optional kwargs:
    - ``vcpus`` / ``memory_mib`` / ``memory_max_mib``: runtime resources.
    - ``timeout``: runtime lifetime in seconds (default: 1800,
      ``RLLM_RUNTA_SANDBOX_TIMEOUT_S``); the server-side leak backstop.
    - ``create_timeout``: seconds to wait for the runtime and the image pull
      (default: 600).
    - ``endpoint``: API endpoint override (default: ``RUNTA_ENDPOINT`` / config).
    """

    def __init__(self, name: str, image: str | None = "python:3.11-slim", **kwargs):
        runta = _import_runta()

        self.name = name
        self._image_spec = image
        self._closed = False
        self._create_timeout = float(kwargs.pop("create_timeout", 600.0))
        lifetime = int(kwargs.pop("timeout", _DEFAULT_TIMEOUT))
        endpoint = kwargs.pop("endpoint", None)

        create_kwargs: dict = {
            "auto_shutdown_after_secs": lifetime,
            "auto_delete_after_secs": _DELETE_AFTER_SHUTDOWN_S,
        }
        for key in ("vcpus", "memory_mib", "memory_max_mib"):
            if key in kwargs:
                create_kwargs[key] = kwargs.pop(key)

        from_snapshot = isinstance(image, str) and image.startswith(_CHECKPOINT_PREFIX)
        if from_snapshot:
            create_kwargs["checkpoint_id"] = image[len(_CHECKPOINT_PREFIX) :]

        self._client = runta.Runta(endpoint=endpoint) if endpoint else runta.Runta()
        try:
            # wait=False: take the handle before waiting, so a runtime that
            # never becomes ready is deleted below rather than left running.
            self._runtime = self._client.runtimes.create(name, wait=False, **create_kwargs)
        except runta.ApiError as e:
            self._client.close()
            # A deleted checkpoint surfaces as 404; translate it so get_sandbox
            # can fall back to cold. Any other failure is real and propagates.
            if from_snapshot and _is_not_found(e):
                raise SnapshotNotFound(f"runta checkpoint {image} no longer exists") from e
            raise
        except BaseException:
            self._client.close()
            raise
        self._sandbox_id = self._runtime.id

        with _LIVE_LOCK:
            _LIVE_SANDBOXES.add(self)

        try:
            self._runtime.wait_until_ready(timeout=self._create_timeout)
            if from_snapshot:
                # A checkpoint restores the whole machine, container included;
                # ``docker start`` is a no-op when it is already running and
                # tells a container snapshot from a direct-mode one.
                self._in_container = self._runtime.exec(f"docker start {_CONTAINER}", timeout=120).exit_code == 0
            elif image:
                self._start_container(image)
                self._in_container = True
            else:
                self._in_container = False
        except BaseException:
            self.close()
            raise

        logger.info("RuntaSandbox %s created (id: %s, image: %s)", name, self._sandbox_id, image or "<none>")

    def _start_container(self, image: str) -> None:
        """Pull ``image`` and run it as the long-lived task container."""
        # --network host: container ports are the runtime's ports, and the
        # runtime's egress policy applies to the container unchanged.
        command = shlex.join(["docker", "run", "-d", "--name", _CONTAINER, "--network", "host", image, "sleep", "infinity"])
        result = self._runtime.exec(command, timeout=self._create_timeout)
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to start image {image} in Runta sandbox {self.name} (exit {result.exit_code}):\n{result.stderr_text[-600:]}")

    def exec(self, command: str, timeout: float | None = None, user: str | None = None) -> str:
        """Execute a command inside the sandbox.

        Runs ``bash -c <command>`` in the task container (``docker exec``,
        honoring ``user``) and returns stdout. Raises ``RuntimeError`` on
        non-zero exit, matching DockerSandbox; a command that exceeds
        ``timeout`` exits 124 and is killed. In direct mode (no image) the
        command runs on the runtime itself and ``user`` is ignored.
        """
        pidfile = None
        if self._in_container:
            argv = ["docker", "exec"]
            for var in _CA_ENV_VARS:
                argv += ["-e", f"{var}={_CA_BUNDLE}"]
            if user is not None:
                argv += ["-u", str(user)]
            if timeout is None:
                argv += [_CONTAINER, "bash", "-c", command]
            else:
                pidfile = f"/tmp/rllm-exec-{uuid.uuid4().hex[:12]}.pid"
                argv += [_CONTAINER, "bash", "-c", _TRACK, "rllm-exec", pidfile, command]
        else:
            argv = ["bash", "-c", command]

        result = self._runtime.exec(argv, timeout=timeout)
        if pidfile and result.exit_code == 124:
            self._kill_in_container(pidfile)

        stdout = result.stdout_text
        if result.stdout_truncated:
            # Runta keeps only the head of an oversized stream; the other
            # backends return everything, so never pass this off as complete.
            logger.warning("stdout of a command in sandbox %s exceeded Runta's capture limit and was truncated to its first %d bytes: %s", self.name, len(result.stdout), command[:200])
        if result.exit_code != 0:
            stderr = result.stderr_text
            # Short tail in the raised message (callers print it at WARNING for
            # every failed verifier); the full tails go to debug.
            logger.debug(
                "Command failed (exit %d) in sandbox %s: %s\nstdout (tail):\n%s\nstderr (tail):\n%s",
                result.exit_code,
                self.name,
                command,
                stdout[-8000:],
                stderr[-8000:],
            )
            raise RuntimeError(f"Command failed (exit {result.exit_code}) in sandbox {self.name}: {command}\nstderr (tail):\n{stderr[-600:]}")
        return stdout

    def _kill_in_container(self, pidfile: str) -> None:
        """Kill the process group a timed-out command left running in the task container."""
        try:
            self._runtime.exec(["docker", "exec", "-u", "0", _CONTAINER, "bash", "-c", _KILL, "rllm-kill", pidfile], timeout=30)
        except Exception:
            logger.warning("Could not stop the timed-out command in sandbox %s — it may still be running", self.name, exc_info=True)

    def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload a single file into the sandbox."""
        remote_dir = os.path.dirname(remote_path)
        if remote_dir:
            self._exec_unchecked(f"mkdir -p {shlex.quote(remote_dir)}")
        if not self._in_container:
            self._runtime.files.upload(local_path, remote_path)
        else:
            staged = self._stage_path()
            self._runtime.files.upload(local_path, staged)
            self._copy_into_container(staged, remote_path)
        logger.debug("Uploaded %s -> %s in sandbox %s", local_path, remote_path, self.name)

    def upload_dir(self, local_path: str, remote_path: str) -> None:
        """Upload a directory tree so that it becomes ``remote_path``."""
        if not os.path.isdir(local_path):
            raise FileNotFoundError(f"upload_dir: local path {local_path} does not exist")
        remote = remote_path.rstrip("/")
        self._exec_unchecked(f"mkdir -p {shlex.quote(remote)}")
        # Stage on the runtime first, then merge the staged directory's contents
        # ("/.") into the (already created) destination: the same semantics in
        # both modes, and no ``tar`` needed in the task image.
        staged = self._stage_path()
        self._runtime.files.upload(local_path, staged)
        if self._in_container:
            self._copy_into_container(f"{staged}/.", remote)
        else:
            self._finish_staged(f"cp -a {shlex.quote(staged)}/. {shlex.quote(remote)}/", staged, remote)
        logger.debug("Uploaded dir %s -> %s in sandbox %s", local_path, remote_path, self.name)

    def _stage_path(self) -> str:
        return f"/tmp/rllm-upload-{uuid.uuid4().hex[:12]}"

    def _copy_into_container(self, staged: str, remote_path: str) -> None:
        """``docker cp`` a path staged on the runtime into the task container."""
        copy = f"docker cp {shlex.quote(staged)} {shlex.quote(f'{_CONTAINER}:{remote_path}')}"
        self._finish_staged(copy, staged.removesuffix("/."), remote_path)

    def _finish_staged(self, copy_command: str, stage_root: str, remote_path: str) -> None:
        """Run ``copy_command`` on the runtime, then always remove the staged path."""
        command = f"{copy_command}; rc=$?; rm -rf {shlex.quote(stage_root)}; exit $rc"
        result = self._runtime.exec(command, timeout=600)
        if result.exit_code != 0:
            raise RuntimeError(f"Upload to {remote_path} failed in sandbox {self.name}:\n{result.stderr_text[-600:]}")

    def is_alive(self) -> bool:
        """One API GET: refresh runtime metadata and check it is still running.

        A runtime past its lifetime is shut down (and later deleted) by Runta,
        so anything not ``running`` counts as dead here.
        """
        if self._closed:
            return False
        try:
            status = self._runtime.refresh_info().status
            return str(getattr(status, "value", status)).lower() == "running"
        except Exception:
            logger.debug("RuntaSandbox %s is_alive check failed — treating as dead", self.name, exc_info=True)
            return False

    def close(self) -> None:
        """Delete the Runta runtime and release resources."""
        if self._closed:
            return
        self._closed = True
        try:
            self._runtime.delete()
        except Exception:
            # The server-side lifetime still reclaims it; surface the leak window.
            logger.warning("Sandbox %s delete failed — Runta reclaims it when its lifetime expires", self.name, exc_info=True)
        try:
            self._client.close()
        except Exception:
            logger.debug("Sandbox %s client close failed", self.name, exc_info=True)
        with _LIVE_LOCK:
            _LIVE_SANDBOXES.discard(self)
        logger.info("RuntaSandbox %s closed", self.name)

    def checkpoint(self, name: str) -> str:
        """Checkpoint the running sandbox; return a snapshot ref for ``image=``."""
        # wait=False: take the id before waiting, so a checkpoint that never
        # becomes ready is deleted rather than left behind untracked.
        info = self._runtime.checkpoints.create(name=name, wait=False)
        try:
            self._client.checkpoints.wait_until_ready(info.id, timeout=self._create_timeout)
        except BaseException:
            try:
                self._client.checkpoints.delete(info.id)
            except Exception as e:
                if not _is_not_found(e):
                    logger.warning("Runta checkpoint %s (%s) did not become ready and could not be deleted — delete it manually", info.id, name, exc_info=True)
            raise
        return f"{_CHECKPOINT_PREFIX}{info.id}"

    def _exec_unchecked(self, command: str) -> str:
        """Execute a command without raising on non-zero exit."""
        try:
            return self.exec(command)
        except RuntimeError:
            return ""


def create_runta_sandbox(name: str, image: str | None = "python:3.11-slim", **kwargs) -> RuntaSandbox:
    """Factory function for creating a RuntaSandbox."""
    return RuntaSandbox(name=name, image=image, **kwargs)


def _checkpoint_id(ref: str) -> str:
    return ref.removeprefix(_CHECKPOINT_PREFIX)


def _runta_ref_alive(ref: str) -> bool:
    """True only when the checkpoint is confirmed ready, for build reuse.

    Unsure (auth/unknown) returns ``False`` so the caller rebuilds rather than
    reuse a ref it cannot confirm.
    """
    runta = _import_runta()
    try:
        with runta.Runta() as client:
            state = client.checkpoints.get(_checkpoint_id(ref)).state
        return str(getattr(state, "value", state)).lower() == "ready"
    except Exception:
        logger.debug("runta ref probe failed for %s — treating as not reusable", ref, exc_info=True)
        return False


def _runta_ref_absent(ref: str) -> bool:
    """True only when the checkpoint is verifiably gone (404 / failed); ``False`` (keep) on any unconfirmed error."""
    runta = _import_runta()
    try:
        with runta.Runta() as client:
            state = client.checkpoints.get(_checkpoint_id(ref)).state
        return str(getattr(state, "value", state)).lower() == "failed"
    except runta.ApiError as e:
        return _is_not_found(e)  # 404 = confirmed gone; auth/rate-limit/other = keep
    except Exception:
        logger.debug("runta ref probe failed for %s — treating as unknown", ref, exc_info=True)
        return False


def build_runta_snapshot(task, key: str, prior_ref: str | None = None, *, force: bool = False, install_script: str = "") -> str | None:
    """Checkpoint ``task``'s prepared environment; return its snapshot ref.

    Mirrors the Modal backend: a known-live ``prior_ref`` is reused unless
    ``force``. Otherwise create a base sandbox, replay the Dockerfile RUN
    steps, run the install script (if any), and checkpoint the runtime. A
    failed install fails the build — a snapshot keyed on the install must
    actually contain it.
    """
    from rllm.eval._resolution import _create_base_sandbox, _dockerfile_run_commands, _replay_dockerfile

    if prior_ref and not force and _runta_ref_alive(prior_ref):
        logger.info("runta snapshot %s already live (%s) — reusing", key, prior_ref)
        return prior_ref

    # Size the build runtime's lifetime to the worst-case replay (each RUN is
    # bounded at 900s), plus the install bound and pull/checkpoint slack.
    install_budget = env_int("RLLM_HARNESS_INSTALL_TIMEOUT_S", 900) if install_script else 0
    build_timeout = max(_DEFAULT_TIMEOUT, 900 * len(_dockerfile_run_commands(task)) + install_budget + 600)
    sb = _create_base_sandbox(task, "runta", name=f"{key}-build", timeout=build_timeout)
    try:
        _replay_dockerfile(task, sb, "runta")
        if install_script:
            sb.exec(install_script, timeout=env_int("RLLM_HARNESS_INSTALL_TIMEOUT_S", 900), user="root")
        ref = sb.checkpoint(key)
        logger.info("runta snapshot built: %s -> %s", key, ref)
        return ref
    finally:
        try:
            sb.close()
        except Exception:
            logger.debug("build sandbox close failed", exc_info=True)


def delete_runta_snapshot(ref: str) -> bool:
    """Delete a Runta checkpoint; ``True`` only when confirmed gone, ``False`` (keep the local record) otherwise."""
    runta = _import_runta()
    try:
        with runta.Runta() as client:
            client.checkpoints.delete(_checkpoint_id(ref))
        return True
    except runta.ApiError as e:
        if _is_not_found(e):
            return True  # already gone — safe to drop
        logger.warning("failed to delete runta checkpoint %s — keeping local record", ref, exc_info=True)
        return False
    except Exception:
        logger.warning("failed to delete runta checkpoint %s — keeping local record", ref, exc_info=True)
        return False
