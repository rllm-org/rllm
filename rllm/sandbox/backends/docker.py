"""Docker container sandbox backend."""

from __future__ import annotations

import io
import logging
import os
import shlex
import tarfile
import threading
import time

from rllm.sandbox.protocol import SandboxExecTimeout

logger = logging.getLogger(__name__)

# Grace between the in-container SIGTERM and SIGKILL, and how long the client
# waits past the deadline before giving up on the container entirely.
_EXEC_KILL_GRACE_SEC = 15
_EXEC_CLIENT_SLACK_SEC = 30


class _ExecFailed(RuntimeError):
    """Non-zero exit from a container command, carrying the code.

    Subclasses RuntimeError so callers that already catch RuntimeError -- the
    evaluators do -- are unaffected.
    """

    def __init__(self, message: str, exit_code: int):
        super().__init__(message)
        self.exit_code = exit_code


# Linux Docker does not define ``host.docker.internal`` unless the container
# is started with ``--add-host=host.docker.internal:host-gateway``.
# CLI harnesses inside sandboxes reach the rLLM gateway via that hostname
# after ``container_reachable_url`` rewrites loopback addresses.
_DOCKER_EXTRA_HOSTS = {"host.docker.internal": "host-gateway"}


class DockerSandbox:
    """Sandbox implementation using Docker containers.

    Creates a container with ``sleep infinity``, uploads files via tar archives,
    executes commands via ``exec_run()``, and runs agent processes with ``nohup``.

    Requires the ``docker`` Python package (not in ``[sdk]`` extra — install
    separately when using ``backend=docker``).
    """

    def __init__(
        self,
        name: str,
        image: str = "python:3.11-slim",
        *,
        mounts: list[dict] | None = None,
        mem_limit: str | int | None = None,
        nano_cpus: int | None = None,
        **kwargs,
    ):
        import docker

        self.name = name
        self.image = image
        self._timeout_fmt: str | None = None  # probed once per container, see _timeout_wrapper
        self._client = docker.from_env()
        run_kwargs: dict = {
            "command": "sleep infinity",
            "name": f"rllm-sandbox-{name}",
            "detach": True,
            "remove": False,
            "extra_hosts": _DOCKER_EXTRA_HOSTS,
        }
        if mounts:
            run_kwargs["mounts"] = mounts
        # Unbounded until now: a runaway verifier reached 519 GB RSS on a host
        # that happened to have 2 TB. These are the two limits Harbor's compose
        # applies (``deploy.resources.limits.cpus`` / ``memory``); the values
        # come from the task's ``[environment]`` via
        # ``_sandbox_resource_kwargs``.
        if mem_limit:
            run_kwargs["mem_limit"] = mem_limit
        if nano_cpus:
            run_kwargs["nano_cpus"] = int(nano_cpus)
        self._container = self._client.containers.run(image, **run_kwargs)
        agent_mounts = [m.get("source") for m in mounts or [] if m.get("type") == "image"]
        if agent_mounts:
            logger.info(
                "DockerSandbox %s created (container: %s, image: %s, agent_mount=%s)",
                name,
                self._container.short_id,
                image,
                agent_mounts[0],
            )
        else:
            logger.info("DockerSandbox %s created (container: %s, image: %s)", name, self._container.short_id, image)

    def exec(self, command: str, timeout: float | None = None, user: str | None = None) -> str:
        """Execute a command inside the container.

        Args:
            command: Shell command to run.
            timeout: Optional per-call timeout in seconds. Enforced in two
                layers -- see below. ``None`` waits forever, as before.
            user: Optional UID/username to run as (e.g., ``"agent"``, ``"1000"``).
                Maps to ``docker exec --user``. If ``None``, runs as the
                container's default user.

        Raises:
            SandboxExecTimeout: the command exceeded ``timeout``.

        ``timeout`` used to be accepted and dropped ("currently unused by
        Docker SDK"), so ``task.toml``'s ``[verifier] timeout_sec`` was a no-op
        here and ``exec_run`` blocked forever. One verifier pytest that looped
        stalled a whole training run for 2h58m with the GPUs idle and the
        container at 519 GB RSS. The SDK still has no timeout, so this enforces
        it in two layers:

        1. ``timeout --kill-after`` inside the container. Cheap, and the
           container survives, so the episode fails and is scored as unsolved.
        2. A client-side deadline. This is the layer that actually guarantees
           the caller unblocks: in the incident above a ``timeout`` process was
           already defunct while its *grandchild* pytest kept running, because
           GNU timeout signals only its direct child. Killing the container is
           the only thing that reliably stops every descendant, so on expiry
           that is what happens -- which also stops the runaway from burning
           CPU and memory while the episode unwinds.
        """
        if timeout is None:
            return self._exec_now(command, user)

        deadline = float(timeout)
        wrapper = self._timeout_wrapper()
        if wrapper:
            command = wrapper.format(grace=_EXEC_KILL_GRACE_SEC, deadline=f"{deadline:.0f}") + f" bash -c {shlex.quote(command)}"

        box: dict = {}

        def _run() -> None:
            try:
                box["ok"] = self._exec_now(command, user)
            except BaseException as e:  # noqa: BLE001 - handed to the caller thread
                box["err"] = e

        worker = threading.Thread(target=_run, name=f"docker-exec-{self.name}", daemon=True)
        started = time.monotonic()
        worker.start()
        worker.join(deadline + _EXEC_KILL_GRACE_SEC + _EXEC_CLIENT_SLACK_SEC)
        if worker.is_alive():
            logger.warning(
                "Sandbox %s: exec exceeded %.0fs and the in-container timeout did not free it — killing the container. Command: %s",
                self.name,
                deadline,
                command[:200],
            )
            try:
                self._container.kill()
            except Exception:
                logger.debug("Sandbox %s: kill after exec timeout failed", self.name, exc_info=True)
            raise SandboxExecTimeout(f"exec exceeded {deadline:.0f}s in container {self.name} (container killed)")
        err = box.get("err")
        if err is not None:
            # GNU timeout exits 124 when SIGTERM did the job, but 137 (128+9)
            # when it had to escalate to the SIGKILL -- and 137 is also what
            # an OOM kill looks like. BusyBox timeout has no 124: it reports
            # the child's death by signal, 143 (128+15) after SIGTERM. Elapsed
            # time separates a timeout from an OOM or a self-inflicted signal:
            # a timeout runs out the clock, the others usually do not.
            code = getattr(err, "exit_code", None)
            timed_out = code == 124 or (code in (137, 143) and time.monotonic() - started >= deadline)
            if timed_out:
                raise SandboxExecTimeout(f"exec exceeded {deadline:.0f}s in container {self.name} (exit {code})") from err
            raise err
        return box["ok"]

    # Candidate ``timeout`` invocations, most specific first. GNU coreutils
    # takes ``--kill-after=15s``; BusyBox (Alpine images) only ``-k 15`` and
    # rejects the long option with exit 1 -- which, wrapped around a verifier,
    # was indistinguishable from the verifier failing and scored the task 0.
    _TIMEOUT_CANDIDATES = (
        "timeout --kill-after={grace}s {deadline}",
        "timeout -k {grace} {deadline}",
    )

    def _timeout_wrapper(self) -> str:
        """The ``timeout`` invocation this image accepts, or ``""`` for none.

        Probed once per container by *running* each candidate on a no-op, not
        by checking that a ``timeout`` binary exists: the binary's dialect is
        what varies between images. With no working candidate the exec runs
        unwrapped and the client-side deadline (which kills the container) is
        the only enforcement.
        """
        if self._timeout_fmt is None:
            self._timeout_fmt = ""
            for fmt in self._TIMEOUT_CANDIDATES:
                probe = fmt.format(grace=1, deadline=5) + " true"
                try:
                    code, _ = self._container.exec_run(["bash", "-c", probe], demux=True)
                except Exception:
                    break
                if code == 0:
                    self._timeout_fmt = fmt
                    break
            if not self._timeout_fmt:
                logger.warning("Sandbox %s: no usable `timeout` in image — exec deadlines fall back to killing the container", self.name)
        return self._timeout_fmt

    def _exec_now(self, command: str, user: str | None) -> str:
        kwargs: dict = {"demux": True}
        if user is not None:
            kwargs["user"] = user
        exit_code, output = self._container.exec_run(
            ["bash", "-c", command],
            **kwargs,
        )
        stdout = (output[0] or b"").decode("utf-8", errors="replace")
        stderr = (output[1] or b"").decode("utf-8", errors="replace")
        if exit_code != 0:
            # The raised message stays short — callers print it at WARNING
            # for *every* failed verifier, and dumping kilobytes of pytest
            # output for each agent that didn't solve a task spams the
            # terminal. The full tail goes to ``logger.debug`` so it's
            # available with ``--log-level=debug`` without polluting the
            # default run.
            short_tail = 600
            err_tail = stderr[-short_tail:] if len(stderr) > short_tail else stderr
            full_tail = 8000
            logger.debug(
                "Command failed (exit %d) in container %s: %s\nstdout (tail):\n%s\nstderr (tail):\n%s",
                exit_code,
                self.name,
                command,
                stdout[-full_tail:] if len(stdout) > full_tail else stdout,
                stderr[-full_tail:] if len(stderr) > full_tail else stderr,
            )
            raise _ExecFailed(
                f"Command failed (exit {exit_code}) in container {self.name}: {command}\nstderr (tail):\n{err_tail}",
                exit_code,
            )
        return stdout

    def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload a single file into the container via tar archive."""
        remote_dir = os.path.dirname(remote_path)
        remote_name = os.path.basename(remote_path)
        self._container.exec_run(["mkdir", "-p", remote_dir])

        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            tar.add(local_path, arcname=remote_name)
        tar_stream.seek(0)
        self._container.put_archive(remote_dir, tar_stream)

    def upload_dir(self, local_path: str, remote_path: str) -> None:
        """Upload a directory tree into the container via tar archive."""
        remote_parent = os.path.dirname(remote_path.rstrip("/"))
        remote_name = os.path.basename(remote_path.rstrip("/"))
        self._container.exec_run(["mkdir", "-p", remote_parent])

        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            tar.add(local_path, arcname=remote_name)
        tar_stream.seek(0)
        self._container.put_archive(remote_parent, tar_stream)

    def is_alive(self) -> bool:
        """Refresh container state from the Docker daemon and check it is running."""
        try:
            self._container.reload()
            return self._container.status == "running"
        except Exception:
            logger.debug("DockerSandbox %s is_alive check failed — treating as dead", self.name, exc_info=True)
            return False

    def close(self) -> None:
        """Stop and remove the container."""
        try:
            self._container.stop(timeout=5)
        except Exception:
            try:
                self._container.kill()
            except Exception:
                pass
        try:
            self._container.remove(force=True)
        except Exception:
            pass
        logger.info("DockerSandbox %s closed", self.name)


def create_docker_sandbox(name: str, image: str = "python:3.11-slim", **kwargs) -> DockerSandbox:
    """Factory function for creating a DockerSandbox."""
    return DockerSandbox(name=name, image=image, **kwargs)
