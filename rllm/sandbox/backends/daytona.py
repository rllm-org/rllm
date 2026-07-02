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
import shlex
import tarfile
import threading
import time
import uuid
import weakref
from pathlib import Path

from rllm.env import env_float, env_int
from rllm.sandbox.protocol import SandboxCommandTimeout, SnapshotNotFound

logger = logging.getLogger(__name__)

# Default auto-stop interval (minutes). 0 disables auto-stop entirely;
# we rely on explicit ``close()`` plus the atexit hook below. Set to a
# finite value to belt-and-suspenders against leaked sandboxes if the
# process is SIGKILL'd (atexit won't fire then).
_DEFAULT_AUTO_STOP_INTERVAL = 30  # minutes; mirrors ModalSandbox's 30-min timeout

# Commands allowed to run longer than this ride a session (async submit +
# short polls) instead of the one-shot exec endpoint. The one-shot exec
# carries the whole command on a single HTTP request that stays byte-silent
# until completion, and Daytona's toolbox proxy silently drops idle flows
# after ~4 minutes (measured 2026-07-01: sleep 240 delivered, sleep 270/330
# lost) — the completion is then unreachable and the call blocks until the
# SDK read timeout, mislabeling long-finished commands as timeouts.
_SESSION_EXEC_THRESHOLD_S = env_float("RLLM_DAYTONA_SESSION_EXEC_THRESHOLD_S", 180.0)
_SESSION_EXEC_POLL_S = env_float("RLLM_DAYTONA_SESSION_EXEC_POLL_S", 15.0)
_SESSION_EXEC_MAX_POLL_FAILURES = 5


def _enable_tcp_keepalive(client) -> None:
    """Inject TCP keepalive into the SDK's HTTP pools (defense-in-depth).

    An L4 middlebox on the toolbox path silently drops flows idle for
    ~245-250s (measured 2026-07-01), black-holing responses of long
    requests. Kernel keepalive probes every 60s reset that idle timer
    (validated: a 400s one-shot exec delivers with these options, and is
    always lost without them). Sessions + polling remain the primary
    defense for long execs; this also protects every other SDK call.

    Reaches into SDK internals (no public hook: ``DaytonaConfig`` doesn't
    expose ``socket_options`` and the toolbox client is cloned at
    construction), so it is strictly best-effort — never fails the caller.
    """
    import socket as _socket

    opts = [(_socket.SOL_SOCKET, _socket.SO_KEEPALIVE, 1)]
    for name, val in (("TCP_KEEPIDLE", 60), ("TCP_KEEPINTVL", 60), ("TCP_KEEPCNT", 5)):
        if hasattr(_socket, name):  # Linux; macOS/Windows lack some of these
            opts.append((_socket.IPPROTO_TCP, getattr(_socket, name), val))
    try:
        import urllib3.connection

        base = list(urllib3.connection.HTTPConnection.default_socket_options)
        from daytona_api_client.rest import RESTClientObject as _ApiREST
        from daytona_toolbox_api_client.rest import RESTClientObject as _ToolboxREST

        for attr, rest_cls in (("_api_client", _ApiREST), ("_toolbox_api_client", _ToolboxREST)):
            api = getattr(client, attr, None)
            if api is None or not hasattr(api, "configuration"):
                continue
            api.configuration.socket_options = base + opts
            api.rest_client = rest_cls(api.configuration)
        logger.debug("TCP keepalive enabled on Daytona SDK HTTP pools")
    except Exception:
        logger.warning("Could not enable TCP keepalive on Daytona SDK pools (SDK internals changed?)", exc_info=True)

# atexit-tracked sandboxes; terminated on process exit to avoid leaks.
_LIVE_SANDBOXES: weakref.WeakSet = weakref.WeakSet()
_LIVE_LOCK = threading.Lock()


class _CreateRateLimiter:
    """Process-global token bucket pacing ``Daytona.create()``.

    A GRPO training step spins up many sandboxes at once (one per group copy).
    Daytona enforces a server-side create rate limit; a large concurrent burst
    trips it and those creates fail. This mirrors Modal's backend limiter: up to
    ``burst`` creates go through immediately, after which creates are paced at
    ``rate``/s — so the startup burst is absorbed and only the long tail queues
    locally instead of failing. Retry-with-backoff (:func:`_is_transient_daytona_error`)
    is the second line of defence for anything that still slips through.

    Thread-safe; token accounting is serialized under the lock while the wait
    sleeps outside it. Scope is per-process (all creates happen in the driver).
    """

    def __init__(self, rate: float, burst: float) -> None:
        self._rate = rate
        self._capacity = max(1.0, burst)
        self._tokens = self._capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        if self._rate <= 0:  # disabled via RLLM_DAYTONA_SANDBOX_CREATE_RPS=0
            return
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self._capacity, self._tokens + (now - self._last) * self._rate)
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self._rate
            time.sleep(wait)


# Create pacing + retry knobs. Generous defaults: the limiter only smooths large
# training bursts, and retry-with-linear-backoff is the real safety net. Set
# RLLM_DAYTONA_SANDBOX_CREATE_RPS=0 to disable pacing entirely.
_CREATE_RATE_RPS = env_float("RLLM_DAYTONA_SANDBOX_CREATE_RPS", 8.0)
_CREATE_BURST = env_float("RLLM_DAYTONA_SANDBOX_CREATE_BURST", 100.0)
_CREATE_LIMITER = _CreateRateLimiter(_CREATE_RATE_RPS, _CREATE_BURST)
_CREATE_MAX_ATTEMPTS = env_int("RLLM_DAYTONA_SANDBOX_CREATE_RETRIES", 6)
_CREATE_BACKOFF_BASE_S = env_float("RLLM_DAYTONA_SANDBOX_CREATE_BACKOFF_S", 5.0)
_CREATE_BACKOFF_CAP_S = 60.0


def _is_transient_daytona_error(exc: Exception) -> bool:
    """Whether a Daytona create error is worth retrying (rate-limit / capacity / 5xx / connection).

    A vanished snapshot (``DaytonaNotFoundError``) or a real validation error is
    *not* transient and must propagate so the caller can cold-fall back or fail.
    """
    try:
        from daytona import DaytonaRateLimitError

        if isinstance(exc, DaytonaRateLimitError):
            return True
    except Exception:  # noqa: S110 — SDK shape varies; fall through to message sniffing
        pass
    msg = str(exc).lower()
    return any(
        p in msg
        for p in (
            "rate limit",
            "too many requests",
            "429",
            "capacity",
            "try again",
            "temporarily",
            "503",
            "502",
            "connection reset",
            "connection aborted",
        )
    )


def _looks_like_timeout(exc: Exception) -> bool:
    """Heuristic: did this SDK exception come from a command/exec timeout?"""
    msg = str(exc).lower()
    return "timed out" in msg or "timeout" in msg


def _build_exec_command(command: str, persistent_env: dict[str, str] | None, user: str | int | None) -> str:
    """Wrap *command* with persistent-env exports and an optional ``su`` user-switch.

    Pure string transform (mirrors ``modal_backend._build_exec_command``):

    * ``persistent_env`` is exported ahead of the command so every exec sees the
      task's declared environment — each Daytona exec is an independent shell, so
      a one-shot ``export`` wouldn't persist. Mirrors harbor's per-exec env.
    * ``user`` switches via ``su <user> -s /bin/bash -c <cmd>`` — Daytona sets the
      default user only at create time (``os_user``), not per exec. An ``int`` is
      resolved as a uid via ``getent``; a ``str`` is used verbatim. Matches how
      harbor's Daytona environment runs the agent vs. verifier under their users.
    """
    run = command
    if persistent_env:
        prefix = "".join(f"export {k}={shlex.quote(str(v))}; " for k, v in persistent_env.items())
        run = prefix + run
    if user is not None:
        user_arg = f"$(getent passwd {user} | cut -d: -f1)" if isinstance(user, int) else shlex.quote(str(user))
        run = f"su {user_arg} -s /bin/bash -c {shlex.quote(run)}"
    return run


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
    via ``sandbox.process.exec()``, and uploads files via the native
    ``sandbox.fs`` API. Sandboxes are **ephemeral** by default (single-use RL
    rollouts): deleted when stopped, on the higher ephemeral quota.

    The ``image`` parameter accepts either:
    - A Daytona snapshot name (e.g. ``"rllm-worker-v0"``) — passed via
      ``CreateSandboxFromSnapshotParams``.
    - A Docker image reference (e.g. ``"python:3.11-slim"``) — passed via
      ``CreateSandboxFromImageParams`` so Daytona pulls it from the
      registry.
    - A ``daytona.Image`` object — used directly as a custom build spec.

    Optional kwargs:
    - ``daytona_config``: explicit ``DaytonaConfig`` (overrides env vars).
    - ``ephemeral``: delete (not archive) on stop; default ``True``. Implies
      ``auto_delete_interval=0`` unless overridden.
    - ``auto_stop_interval``: minutes; default ``30``. Pass ``0`` to disable.
    - ``auto_delete_interval``: minutes after stop to delete; default ``0``
      (immediate) when ephemeral, else ``-1`` (never).
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
        # Env exported into every exec (the task's [environment].env). Daytona
        # execs are independent shells, so this is re-applied per command rather
        # than relying on a one-shot export. Populated via set_env().
        self._persistent_env: dict[str, str] = {}
        # RL rollout/warm-queue sandboxes are strictly single-use (create → run →
        # verify → delete), so default to ephemeral. Two payoffs: (1) Daytona's
        # ephemeral tier carries a higher per-sandbox quota (disk/cpu/mem), so
        # tasks declaring more than the standard 10 GB disk (e.g. tmax's 20 GB)
        # schedule instead of 400-ing; (2) leak-proofing — an ephemeral sandbox is
        # *deleted* (not left stopped) when it stops, so a SIGKILL'd run that never
        # reaches close()/atexit doesn't strand a stopped box forever. Mirrors how
        # harbor's Daytona environment always runs ephemeral.
        self._ephemeral = bool(kwargs.pop("ephemeral", True))
        self._auto_stop_interval = int(kwargs.pop("auto_stop_interval", _DEFAULT_AUTO_STOP_INTERVAL))
        self._auto_archive_interval = kwargs.pop("auto_archive_interval", None)
        # `ephemeral=True` ⟺ `auto_delete_interval=0` ("delete immediately on
        # stop"); keep them consistent so an idle auto-stop reaps the box.
        self._auto_delete_interval = int(kwargs.pop("auto_delete_interval", 0 if self._ephemeral else -1))
        self._create_timeout = float(kwargs.pop("create_timeout", 120.0))

        # Client init
        daytona_config = kwargs.pop("daytona_config", None)
        self._client = Daytona(daytona_config) if daytona_config is not None else Daytona()
        _enable_tcp_keepalive(self._client)

        # Build common parameters
        base_kwargs: dict = {
            "name": name,
            "ephemeral": self._ephemeral,
            "auto_stop_interval": self._auto_stop_interval,
            "auto_delete_interval": self._auto_delete_interval,
        }
        if self._auto_archive_interval is not None:
            base_kwargs["auto_archive_interval"] = int(self._auto_archive_interval)
        for key in ("env_vars", "labels", "os_user", "public", "volumes"):
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
            self._sandbox = self._create(params)
            image_label = "<daytona.Image>"
        elif isinstance(image, str) and _looks_like_docker_image(image):
            params = CreateSandboxFromImageParams(image=image, resources=resources, **base_kwargs)
            self._sandbox = self._create(params)
            image_label = image
        else:
            # Treat as Daytona snapshot name. Resources are baked into the
            # snapshot at build time, so any re-passed here are ignored (debug,
            # not a warning — booting from a snapshot is the expected path).
            if resources is not None:
                logger.debug("Resources ignored when creating from snapshot %s; snapshot resources win.", image)
            params = CreateSandboxFromSnapshotParams(snapshot=image, **base_kwargs)
            try:
                self._sandbox = self._create(params)
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

    def _create(self, params):
        """Create the sandbox with rate-limit pacing + retry on transient errors.

        ``Daytona.create()`` can fail under a concurrent-create burst (GRPO step)
        with a server-side rate-limit/capacity error. The limiter paces the burst
        and this loop retries transient failures with linear backoff (mirroring
        harbor's tenacity policy). A vanished snapshot / real validation error is
        not transient, so it raises on the first attempt for the caller to handle.
        """
        last_exc: Exception | None = None
        for attempt in range(1, _CREATE_MAX_ATTEMPTS + 1):
            _CREATE_LIMITER.acquire()
            try:
                return self._client.create(params, timeout=self._create_timeout)
            except Exception as e:  # noqa: BLE001 — classify, then retry-or-reraise
                last_exc = e
                if attempt < _CREATE_MAX_ATTEMPTS and _is_transient_daytona_error(e):
                    backoff = min(_CREATE_BACKOFF_CAP_S, _CREATE_BACKOFF_BASE_S * attempt)
                    logger.warning(
                        "daytona create %s attempt %d/%d failed (transient: %s); retrying in %.0fs",
                        self.name,
                        attempt,
                        _CREATE_MAX_ATTEMPTS,
                        e,
                        backoff,
                    )
                    time.sleep(backoff)
                    continue
                raise
        assert last_exc is not None  # loop always raises or returns
        raise last_exc

    def set_env(self, env: dict[str, str]) -> None:
        """Register env vars exported into every subsequent :meth:`exec`.

        Daytona execs are independent shells, so persistent task env (Harbor's
        ``[environment].env``) must be re-applied per command — otherwise the
        verifier (and any agent step) wouldn't observe it. Mirrors how harbor
        injects the same vars per exec, and ModalSandbox.set_env.
        """
        for k, v in (env or {}).items():
            self._persistent_env[str(k)] = str(v)

    def exec(self, command: str, timeout: float | None = None, user: str | int | None = None) -> str:
        """Execute a command inside the Daytona sandbox.

        Wraps the command in ``bash -c`` for shell-feature parity with
        DockerSandbox / ModalSandbox. Returns stdout. Raises ``RuntimeError``
        on non-zero exit, or :class:`SandboxCommandTimeout` when the command is
        killed for exceeding *timeout*.

        ``user`` switches the command to another OS user via ``su`` (Daytona
        sets the default user only at create time, not per exec), and env
        registered via :meth:`set_env` is exported into every command. See
        :func:`_build_exec_command`.

        When *timeout* is set the command is wrapped in coreutils ``timeout`` so
        a runaway process is hard-killed (SIGTERM, then SIGKILL after a grace
        window) and reports a clean exit code with captured stdout — rather than
        hanging until the sandbox idle-stops or surfacing an opaque SDK error.
        """
        run = _build_exec_command(command, self._persistent_env, user)
        wrapped = f"bash -c {_shell_quote(run)}"

        if timeout is not None:
            t = int(timeout)
            # SIGTERM at t, SIGKILL 10s later if it ignores TERM. Exit 124
            # (TERM) / 137 (KILL) flag the timeout below.
            wrapped = f"timeout -k 10 {t} {wrapped}"

        if timeout is not None and timeout > _SESSION_EXEC_THRESHOLD_S:
            # Long command: a one-shot exec would ride a single idle HTTP
            # request that NAT middleboxes drop after ~4 minutes of silence,
            # so its completion could never be delivered (see
            # _SESSION_EXEC_THRESHOLD_S). Untimed execs stay one-shot: they
            # are quick housekeeping calls in practice, and the TCP-keepalive
            # layer (_enable_tcp_keepalive) protects any that straggle.
            stdout, exit_code = self._exec_session(wrapped, timeout, command)
        else:
            exec_kwargs: dict = {}
            if timeout is not None:
                # SDK timeout as a backstop, set beyond the shell timeout so the
                # deterministic shell `timeout` is what fires first.
                exec_kwargs["timeout"] = int(timeout) + 60
            try:
                response = self._sandbox.process.exec(wrapped, **exec_kwargs)
            except Exception as e:  # noqa: BLE001 — map an SDK-level timeout to the protocol type
                if timeout is not None and _looks_like_timeout(e):
                    # SDK-level timeout = NO response was received (unlike the
                    # exit-124 path below, where the shell `timeout` killed the
                    # command and reported back). The command may have finished
                    # and its completion been lost in transport — say so.
                    raise SandboxCommandTimeout(
                        f"No exec response within {int(timeout) + 60}s in sandbox {self.name} "
                        f"(command may have completed; response lost in transport): {command[:200]}"
                    ) from e
                raise
            stdout = response.result or ""
            exit_code = response.exit_code

        if timeout is not None and exit_code in (124, 137):
            # `timeout` killed it: 124 = exited on SIGTERM, 137 = SIGKILL'd after
            # the grace window. Distinct from a real failure so an agent that
            # simply spent its time budget isn't reported as one.
            logger.warning(
                "Command hit its %ds timeout in sandbox %s (exit %d): %s",
                int(timeout),
                self.name,
                exit_code,
                command[:200],
            )
            raise SandboxCommandTimeout(f"Command hit its {int(timeout)}s timeout in sandbox {self.name} (exit {exit_code})")

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

    def _exec_session(self, wrapped: str, timeout: float | None, command: str) -> tuple[str, int]:
        """Run ``wrapped`` via a session command: async submit, then short polls.

        Each poll is its own short-lived HTTP request, so no connection ever
        sits idle long enough for the toolbox proxy to drop it — unlike
        ``process.exec``, which waits for the completion on one idle long-poll.
        The in-sandbox coreutils ``timeout`` (already baked into ``wrapped``)
        remains the deterministic killer; the poll deadline below (same +60s
        grace the one-shot path gives the SDK) only catches a daemon that
        lost the command entirely.
        """
        from daytona.common.process import SessionExecuteRequest

        proc = self._sandbox.process
        session_id = f"rllm-exec-{uuid.uuid4().hex[:12]}"
        proc.create_session(session_id)
        try:
            submitted = proc.execute_session_command(session_id, SessionExecuteRequest(command=wrapped, run_async=True))
            cmd_id = submitted.cmd_id
            deadline = None if timeout is None else time.monotonic() + float(timeout) + 60.0
            poll_failures = 0
            # Adaptive poll: fast at first so short-lived commands that merely
            # *allow* a long budget (cached install, quick verifier) don't pay
            # a full poll interval, decaying to the steady-state interval.
            poll_s = 1.0
            while True:
                time.sleep(poll_s)
                poll_s = min(poll_s * 2, _SESSION_EXEC_POLL_S)
                try:
                    cmd = proc.get_session_command(session_id, cmd_id)
                    poll_failures = 0
                except Exception as e:  # noqa: BLE001 — a transient API blip must not kill a long rollout
                    poll_failures += 1
                    if poll_failures >= _SESSION_EXEC_MAX_POLL_FAILURES:
                        raise
                    logger.warning("Session exec poll %d/%d failed in sandbox %s: %s", poll_failures, _SESSION_EXEC_MAX_POLL_FAILURES, self.name, e)
                    continue
                if cmd.exit_code is not None:
                    exit_code = int(cmd.exit_code)
                    break
                if deadline is not None and time.monotonic() > deadline:
                    raise SandboxCommandTimeout(f"Command hit its {int(timeout)}s timeout in sandbox {self.name} (poll deadline): {command[:200]}")
            try:
                logs = proc.get_session_command_logs(session_id, cmd_id)
                stdout = (logs.stdout or "") + (logs.stderr or "")
            except Exception as e:  # noqa: BLE001 — output is best-effort; the exit code is what gates
                logger.warning("Session exec log fetch failed in sandbox %s: %s", self.name, e)
                stdout = ""
            return stdout, exit_code
        finally:
            try:
                proc.delete_session(session_id)
            except Exception:
                logger.debug("Session cleanup failed for %s", session_id, exc_info=True)

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

    from rllm.eval._resolution import (
        _as_single_run_line,
        _builds_from_dockerfile,
        _dockerfile_run_commands,
        _resolve_image,
        _sandbox_resource_kwargs,
        _should_replay_dockerfile,
    )

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

    dockerfile = _builds_from_dockerfile(task, "daytona")
    if dockerfile is not None:
        # Build the real Dockerfile (COPY/ENV/WORKDIR/RUN) so a snapshotted task is
        # identical to a cold-built one; only the install script layers on top.
        img = Image.from_dockerfile(str(dockerfile))
        run_commands = [_as_single_run_line(install_script)] if install_script else []
    else:
        img = Image.base(_resolve_image(task, "daytona"))
        # Honor [environment].replay_dockerfile: fully-built task images opt out so
        # their RUN steps aren't double-applied on top of the prebuilt image.
        run_commands = [_as_single_run_line(c) for c in _dockerfile_run_commands(task)] if _should_replay_dockerfile(task) else []
        if install_script:
            run_commands.append(_as_single_run_line(install_script))
    if run_commands:
        img = img.run_commands(*run_commands)

    # _sandbox_resource_kwargs also carries sandbox-lifecycle knobs
    # (create_timeout, auto_stop_interval) that Resources() doesn't accept; a
    # snapshot bakes only the compute shape, so keep just the resource fields.
    res_kw = {k: v for k, v in _sandbox_resource_kwargs(task, "daytona").items() if k in ("cpu", "memory", "disk", "gpu")}
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
