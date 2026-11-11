"""boxd sandbox backend.

Uses boxd machines (https://boxd.sh/?ref=rllm) to run agent code in remote
microVMs. A machine boots straight from the task's Docker image, so the
agent sees the same filesystem it would see under the docker backend.

Requires the ``boxd`` package::

    pip install boxd

Authentication is handled via the ``BOXD_API_KEY`` environment variable
(or an explicit ``api_key`` kwarg). ``BOXD_ORG`` and ``BOXD_REGION`` pick
the organization and region; both are optional.
"""

from __future__ import annotations

import atexit
import io
import logging
import os
import re
import tarfile
import threading
import time
import uuid
import weakref

from rllm.env import env_int, env_str
from rllm.sandbox.protocol import SnapshotNotFound

logger = logging.getLogger(__name__)

# boxd machine sizes (vCPU, GiB). Either value names the whole size.
_SIZES = ((1, 4), (2, 8), (4, 16), (8, 32))

# Statuses in which a machine still answers exec. Standby and hibernated
# machines wake on the next exec, so a parked sandbox in those states is usable.
_ALIVE_STATUSES = {"pending", "starting", "running", "suspended", "hibernated", "migrating"}

# A long command that prints nothing (an agent CLI logging to a file, a quiet
# build) still writes one NUL byte to stderr this often, so the exec stream is
# never idle; the bytes are dropped before stderr is used.
_KEEPALIVE_INTERVAL_S = 60

_DEFAULT_CREATE_TIMEOUT = 600.0  # seconds; a cold boot includes the image pull
_SNAPSHOT_READY_TIMEOUT = 900.0  # seconds to wait for a snapshot capture

# atexit-tracked sandboxes; deleted on process exit to avoid leaks.
_LIVE_SANDBOXES: weakref.WeakSet = weakref.WeakSet()
_LIVE_LOCK = threading.Lock()


def _terminate_all_live() -> None:
    """atexit hook: delete every still-alive BoxdSandbox."""
    with _LIVE_LOCK:
        survivors = list(_LIVE_SANDBOXES)
    if not survivors:
        return
    logger.warning("atexit: deleting %d unreleased boxd sandbox(es)", len(survivors))
    for sb in survivors:
        try:
            sb.close()
        except Exception:
            logger.debug("atexit: error closing %s", getattr(sb, "name", "<unknown>"), exc_info=True)


atexit.register(_terminate_all_live)


def _import_boxd():
    """Lazy import so users without the SDK can still import this module."""
    try:
        import boxd
        import boxd.errors
    except ImportError as e:
        raise ImportError("The boxd sandbox backend requires the 'boxd' package. Install with: pip install boxd  Also set BOXD_API_KEY in your environment.") from e
    return boxd


def _make_client(api_key: str | None = None, region: str | None = None):
    """Build a ``boxd.Boxd`` client from an explicit key or ``BOXD_API_KEY``.

    A missing key is a configuration error, raised before any API call.
    """
    boxd = _import_boxd()
    api_key = api_key or os.environ.get("BOXD_API_KEY")
    token = None if api_key else os.environ.get("BOXD_TOKEN")
    if not api_key and not token:
        raise ValueError("The boxd sandbox backend needs an API key: set BOXD_API_KEY (create one with `boxd auth keys create`).")
    if api_key:
        return boxd.Boxd(api_key=api_key, region=region)
    return boxd.Boxd(token=token, region=region)


def _looks_like_docker_image(image: str) -> bool:
    """Heuristic: treat strings containing ``:`` or ``/`` as Docker image refs.

    ``python:3.11-slim`` / ``ghcr.io/foo/bar:tag`` boot a machine from the
    image. ``rllm-env-<hash>`` is a boxd snapshot name (what
    :func:`build_boxd_snapshot` returns).
    """
    return ":" in image or "/" in image


def _machine_name(name: str, prefix: str = "") -> str:
    """A legal machine name: lowercase letters, digits and dashes, at most 63 chars.

    The tail of the name carries rLLM's per-sandbox uuid suffix, so a long
    task id is cut from the middle rather than the end.
    """
    raw = f"{prefix}{name}".lower()
    safe = re.sub(r"[^a-z0-9-]+", "-", raw)
    safe = re.sub(r"-{2,}", "-", safe).strip("-") or f"rllm-{uuid.uuid4().hex[:6]}"
    if len(safe) > 63:
        safe = (safe[:40].rstrip("-") + "-" + safe[-22:].lstrip("-")).strip("-")
    if not safe[0].isalpha():
        safe = f"r{safe}"[:63]
    return safe


def _size_for(cpu: float | None = None, memory_gb: float | None = None) -> tuple[int, int] | None:
    """Smallest boxd size with at least ``cpu`` vCPUs and ``memory_gb`` GiB, or ``None``."""
    if not cpu and not memory_gb:
        return None
    for vcpu, mem in _SIZES:
        if (not cpu or vcpu >= cpu) and (not memory_gb or mem >= memory_gb):
            return vcpu, mem
    logger.warning("Requested %s vCPU / %s GiB exceeds the largest boxd size; using 8 vCPU / 32 GiB", cpu, memory_gb)
    return _SIZES[-1]


def _parse_labels(value: str | None) -> dict[str, str]:
    """Parse ``k=v,k2=v2`` (the ``RLLM_BOXD_LABELS`` format) into a dict."""
    out: dict[str, str] = {}
    for part in (value or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            if k.strip():
                out[k.strip()] = v.strip()
    return out


def _status(machine) -> str:
    status = getattr(machine, "status", None)
    return str(getattr(status, "value", status) or "").lower()


class BoxdSandbox:
    """Sandbox implementation using boxd machines.

    Creates a machine via ``machines.create()``, waits until it answers
    exec, runs commands via ``machines.exec()``, and moves files with the
    native ``machines.files`` API. Every call addresses the machine by id.

    The ``image`` parameter accepts either:
    - A Docker image reference (e.g. ``"python:3.11-slim"``): the machine
      boots from that image.
    - A boxd snapshot name (e.g. ``"rllm-env-0123456789ab"``, as built by
      ``rllm snapshot create --sandbox-backend boxd``): the machine restores
      the snapshot, with the environment already installed.

    Optional kwargs:
    - ``api_key``: boxd API key (default: ``BOXD_API_KEY``).
    - ``org``: organization name (default: ``BOXD_ORG``, else the key's org).
    - ``region``: ``"fra"`` or ``"nyc"`` (default: ``BOXD_REGION``, else the
      account default).
    - ``isolated``: default ``True``. An isolated machine joins no private
      network and holds no boxd credentials, which is what agent-written
      code should run in.
    - ``cpu`` / ``memory``: vCPUs and GiB; rounded up to the next boxd size
      (1x4, 2x8, 4x16, 8x32). Only effective on the from-image path; a
      snapshot keeps its own size.
    - ``env``: dict of env vars set for every command run in the sandbox.
    - ``labels``: dict of machine labels, merged over ``RLLM_BOXD_LABELS``
      (``k=v,k2=v2``).
    - ``name_prefix``: prepended to the machine name (default:
      ``RLLM_BOXD_NAME_PREFIX``).
    - ``auto_suspend_timeout``: idle seconds before standby; default ``0``
      (off), so a long silent command is never put to sleep mid-run.
    - ``auto_destroy_timeout``: seconds after which boxd deletes the
      machine; a backstop for a process that is SIGKILL'd before ``close()``.
    - ``create_timeout``: seconds to wait for the machine to be ready;
      default ``600``.
    """

    def __init__(self, name: str, image: str = "python:3.11-slim", **kwargs):
        boxd = _import_boxd()
        errors = boxd.errors

        self.name = name
        self._image_spec = image
        self._closed = False
        self._machine_id: str | None = None
        self._create_timeout = float(kwargs.pop("create_timeout", _DEFAULT_CREATE_TIMEOUT))
        self._region = kwargs.pop("region", None) or os.environ.get("BOXD_REGION") or None
        self._org = kwargs.pop("org", None) or os.environ.get("BOXD_ORG") or None
        self._client = _make_client(kwargs.pop("api_key", None), self._region)

        isolated = bool(kwargs.pop("isolated", True))
        labels = {"app": "rllm", **_parse_labels(env_str("RLLM_BOXD_LABELS", "")), **(kwargs.pop("labels", None) or {})}
        prefix = kwargs.pop("name_prefix", None)
        if prefix is None:
            prefix = env_str("RLLM_BOXD_NAME_PREFIX", "")
        self.machine_name = _machine_name(name, prefix)

        create_kwargs: dict = {
            "org": self._org,
            "isolated": isolated,
            "auto_suspend_timeout": int(kwargs.pop("auto_suspend_timeout", 0)),
        }
        if self._region:
            create_kwargs["region"] = self._region
        auto_destroy = kwargs.pop("auto_destroy_timeout", None)

        size = _size_for(kwargs.pop("cpu", None), kwargs.pop("memory", None))
        # Env goes with every exec, so it holds on both the image and the snapshot path.
        self._env = {str(k): str(v) for k, v in (kwargs.pop("env", None) or {}).items()}
        if kwargs:
            logger.debug("BoxdSandbox %s ignoring unsupported kwargs: %s", name, sorted(kwargs))

        t0 = time.monotonic()
        if isinstance(image, str) and _looks_like_docker_image(image):
            if size is not None:
                create_kwargs["vcpu"] = size[0]
            create_kwargs["labels"] = labels
            if auto_destroy is not None:
                create_kwargs["auto_destroy_timeout"] = int(auto_destroy)
            machine = self._client.machines.create(self.machine_name, image=image, **create_kwargs)
            image_label = image
        else:
            # A snapshot restores with its own size, env, labels and timers.
            if size is not None:
                logger.debug("Size ignored when creating from snapshot %s; the snapshot's size wins.", image)
            try:
                machine = self._client.machines.create(self.machine_name, from_snapshot=image, **create_kwargs)
            except errors.NotFoundError as e:
                raise SnapshotNotFound(f"boxd snapshot {image} unavailable: {e}") from e
            image_label = f"snapshot:{image}"

        self._machine_id = machine.id
        with _LIVE_LOCK:
            _LIVE_SANDBOXES.add(self)
        try:
            self._client.machines.wait_until_ready(machine.id, timeout=self._create_timeout, region=self._region)
        except Exception:
            self.close()
            raise

        logger.info(
            "BoxdSandbox %s created (id: %s, image: %s, %.1fs)",
            name,
            self._machine_id,
            image_label,
            time.monotonic() - t0,
        )

    @property
    def machine_id(self) -> str | None:
        """The boxd machine id this sandbox runs on."""
        return self._machine_id

    def exec(self, command: str, timeout: float | None = None, user: str | None = None) -> str:
        """Execute a command inside the machine.

        Wraps the command in ``bash -c`` for shell-feature parity with
        DockerSandbox / ModalSandbox. Returns stdout. Raises
        ``RuntimeError`` on non-zero exit, and on a timeout (the remote
        process group is killed). ``user`` runs the command as that user
        via ``runuser`` (as root) or ``sudo -n`` (as a non-root user).
        """
        errors = _import_boxd().errors
        line = _as_user(command, user)
        try:
            result = self._exec_once(line, command, timeout)
        except errors.ConflictError:
            # A stopped machine refuses exec: start it and run the command once more.
            logger.info("BoxdSandbox %s is stopped; starting it", self.name)
            self._client.machines.start(self._machine_id, region=self._region)
            self._client.machines.wait_until_ready(self._machine_id, timeout=self._create_timeout, region=self._region)
            result = self._exec_once(line, command, timeout)

        if result.exit_code != 0:
            stderr = (result.stderr or "").replace("\0", "")
            tail = (stderr or result.stdout or "")[-500:]
            logger.warning("Command failed in sandbox %s: %s\nstderr: %s", self.name, command, tail)
            raise RuntimeError(f"Command failed (exit {result.exit_code}) in sandbox {self.name}: {command}\n{tail}")
        return result.stdout or ""

    def _exec_once(self, line: list[str], command: str, timeout: float | None):
        errors = _import_boxd().errors
        try:
            return self._client.machines.exec(self._machine_id, line, env=self._env or None, timeout=timeout, region=self._region)
        except errors.ConflictError:
            raise
        except errors.BoxdError as e:
            if timeout is not None and "timed out" in str(e):
                raise RuntimeError(f"Command timed out after {timeout}s in sandbox {self.name}: {command}") from e
            raise

    def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload a single file via the native ``machines.files.upload`` API."""
        with open(local_path, "rb") as f:
            data = f.read()
        self._client.machines.files.upload(self._machine_id, remote_path, data, region=self._region)
        logger.debug("Uploaded %s -> %s in sandbox %s", local_path, remote_path, self.name)

    def upload_dir(self, local_path: str, remote_path: str) -> None:
        """Upload a directory tree.

        Packs the tree into one ``.tar.gz`` in memory, uploads it with the
        native files API, then extracts it inside the machine. One upload
        plus one exec, regardless of tree size.
        """
        local = os.path.abspath(local_path)
        if not os.path.exists(local):
            raise FileNotFoundError(f"upload_dir: local path {local_path} does not exist")

        remote_parent = os.path.dirname(remote_path.rstrip("/")) or "/"
        remote_name = os.path.basename(remote_path.rstrip("/")) or os.path.basename(local)

        tar_buf = io.BytesIO()
        with tarfile.open(fileobj=tar_buf, mode="w:gz") as tar:
            tar.add(local, arcname=remote_name)

        remote_tar = f"/tmp/_upload_{uuid.uuid4().hex[:12]}.tar.gz"
        self._client.machines.files.upload(self._machine_id, remote_tar, tar_buf.getvalue(), region=self._region)
        # --no-same-owner: don't restore the host's uid/gid. Permissions are
        # kept, so executables stay +x.
        self.exec(f"mkdir -p {_shell_quote(remote_parent)} && tar xzf {remote_tar} --no-same-owner -C {_shell_quote(remote_parent)}; rc=$?; rm -f {remote_tar}; exit $rc")
        logger.debug("Uploaded dir %s -> %s in sandbox %s", local_path, remote_path, self.name)

    def is_alive(self) -> bool:
        """One API GET: the machine exists and is in a state that answers exec.

        Standby and hibernated machines count as alive: the next exec
        wakes them. Stopped, failed and deleted machines do not.
        """
        if self._closed or self._machine_id is None:
            return False
        try:
            machine = self._client.machines.get(self._machine_id, region=self._region)
            return _status(machine) in _ALIVE_STATUSES
        except Exception:
            logger.debug("BoxdSandbox %s is_alive check failed; treating as dead", self.name, exc_info=True)
            return False

    def close(self) -> None:
        """Delete the machine and release resources."""
        if self._closed:
            return
        self._closed = True
        with _LIVE_LOCK:
            _LIVE_SANDBOXES.discard(self)
        if self._machine_id is None:
            return
        errors = _import_boxd().errors
        try:
            self._client.machines.delete(self._machine_id, region=self._region)
        except errors.NotFoundError:
            pass
        except Exception:
            # A failed delete leaves a billed machine; hibernate it so it stops using compute.
            logger.warning("Sandbox %s delete failed; attempting hibernate() fallback", self.name, exc_info=True)
            try:
                self._client.machines.hibernate(self._machine_id, region=self._region)
            except Exception:
                logger.warning("Sandbox %s hibernate() fallback also failed; machine %s may be orphaned", self.name, self._machine_id, exc_info=True)
        logger.info("BoxdSandbox %s closed", self.name)


def _shell_quote(s: str) -> str:
    """Single-quote a string for safe interpolation into a shell command."""
    return "'" + s.replace("'", "'\\''") + "'"


def _as_user(command: str, user: str | None) -> list[str]:
    """The argv for ``command``, run by ``bash -c``, as ``user`` when given.

    The switch is a no-op when the session already runs as ``user``. The
    command runs next to a keepalive loop that writes a NUL byte to stderr
    every ``_KEEPALIVE_INTERVAL_S`` seconds and is stopped when it exits.
    """
    c = _shell_quote(command)
    if user is None:
        run = f"bash -c {c}"
    else:
        u = _shell_quote(str(user))
        run = (
            f'want={u}; case "$want" in *[!0-9]*) uid=$(id -u "$want" 2>/dev/null) ;; *) uid="$want" ;; esac; '
            f'if [ "$(id -u)" = "$uid" ]; then bash -c {c}; '
            f'elif [ "$(id -u)" = 0 ]; then runuser -u "$(id -nu "$uid" 2>/dev/null || echo "$want")" -- bash -c {c}; '
            f'else sudo -n -u "#$uid" -- bash -c {c}; fi'
        )
    keepalive = f"( while sleep {_KEEPALIVE_INTERVAL_S} </dev/null >/dev/null 2>&1; do printf '\\0' >&2; done ) & _ka=$!"
    return ["bash", "-c", f"{keepalive}; {run}; _rc=$?; kill $_ka 2>/dev/null; exit $_rc"]


def create_boxd_sandbox(name: str, image: str = "python:3.11-slim", **kwargs) -> BoxdSandbox:
    """Factory function for creating a BoxdSandbox."""
    return BoxdSandbox(name=name, image=image, **kwargs)


def _wait_snapshot_ready(client, name: str, org: str | None, region: str | None, timeout: float = _SNAPSHOT_READY_TIMEOUT):
    deadline = time.monotonic() + timeout
    while True:
        snap = client.snapshots.get(name, org=org, region=region)
        if snap.status == "ready":
            return snap
        if snap.status == "failed":
            raise RuntimeError(f"boxd snapshot {name} failed to capture")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"boxd snapshot {name} not ready after {timeout:.0f}s")
        time.sleep(1.0)


def build_boxd_snapshot(task, key: str, prior_ref: str | None = None, *, force: bool = False, install_script: str = "") -> str | None:
    """Bake ``task``'s environment into a boxd snapshot named ``key``; return the name.

    Boots a machine from the task's base image, replays the Dockerfile RUN
    steps, runs the install script (if any), then captures the machine as a
    snapshot. Idempotent: a ready snapshot named ``key`` is reused unless
    ``force``, which captures a new version under the same name. A failed
    install fails the build, since a snapshot keyed on the install must
    actually contain it.
    """
    from rllm.eval._resolution import _create_base_sandbox, _replay_dockerfile

    errors = _import_boxd().errors
    org = os.environ.get("BOXD_ORG") or None
    region = os.environ.get("BOXD_REGION") or None
    client = _make_client(region=region)

    if not force:
        try:
            snap = client.snapshots.get(key, org=org, region=region)
            if snap.status == "ready":
                logger.info("boxd snapshot %s already registered", key)
                return key
        except errors.NotFoundError:
            pass

    sb = _create_base_sandbox(task, "boxd", name=f"{key}-build")
    try:
        _replay_dockerfile(task, sb, "boxd")
        if install_script:
            sb.exec(install_script, timeout=env_int("RLLM_HARNESS_INSTALL_TIMEOUT_S", 900), user="root")
        client.snapshots.create(sb.machine_id, key, region=region)
        _wait_snapshot_ready(client, key, org, region)
        logger.info("boxd snapshot built: %s (from machine %s)", key, sb.machine_id)
        return key
    finally:
        try:
            sb.close()
        except Exception:
            logger.debug("build sandbox close failed", exc_info=True)


def _boxd_ref_absent(ref: str) -> bool:
    """True only when the snapshot is verifiably gone (NotFound / failed); ``False`` (keep) on any unconfirmed error."""
    errors = _import_boxd().errors
    try:
        client = _make_client(region=os.environ.get("BOXD_REGION") or None)
        snap = client.snapshots.get(ref, org=os.environ.get("BOXD_ORG") or None)
        return snap.status == "failed"
    except errors.NotFoundError:
        return True  # confirmed gone
    except Exception:
        logger.debug("boxd ref probe failed for %s; treating as unknown (keep)", ref, exc_info=True)
        return False


def delete_boxd_snapshot(ref: str) -> bool:
    """Delete a boxd snapshot by name; ``True`` only when confirmed gone, ``False`` (keep the local record) otherwise."""
    errors = _import_boxd().errors
    try:
        client = _make_client(region=os.environ.get("BOXD_REGION") or None)
        client.snapshots.delete(ref, org=os.environ.get("BOXD_ORG") or None)
        return True
    except errors.NotFoundError:
        return True  # already gone, safe to drop
    except (errors.AuthenticationError, errors.PermissionDeniedError):
        logger.warning("no permission to delete boxd snapshot %s; keeping local record", ref)
        return False
    except Exception:
        logger.warning("failed to delete boxd snapshot %s; keeping local record", ref, exc_info=True)
        return False
