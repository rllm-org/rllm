"""Modal sandbox backend.

Uses Modal Sandboxes (https://modal.com/docs/guide/sandboxes) to run
agent code in serverless cloud containers.

Requires the ``modal`` package::

    pip install modal

Authentication is handled via ``modal setup`` or the ``MODAL_TOKEN_ID``
and ``MODAL_TOKEN_SECRET`` environment variables.
"""

from __future__ import annotations

import atexit
import io
import logging
import os
import tarfile
import threading
import weakref

from rllm.env import env_int
from rllm.sandbox.protocol import SnapshotNotFound

logger = logging.getLogger(__name__)

# Headroom the sandbox lifetime keeps *beyond* the agent run timeout + install,
# to cover the post-run verifier, teardown, and scheduling slack.
_SANDBOX_LIFETIME_HEADROOM_S = 10 * 60


def _default_sandbox_timeout() -> int:
    """Default Modal sandbox lifetime (seconds) — must outlast a full rollout.

    The lifetime clock starts at ``Sandbox.create()``, *before* the cold-path
    install, the agent run, and the verifier. If it doesn't exceed their sum the
    container is reaped mid-rollout (exit 137 + "Sandbox already shut down").
    So derive it from the agent run-timeout knob with headroom for install +
    verify + teardown. ``RLLM_MODAL_SANDBOX_TIMEOUT_S`` overrides outright.
    (Modal's own default is only 5 min.)
    """
    explicit = env_int("RLLM_MODAL_SANDBOX_TIMEOUT_S", 0)
    if explicit > 0:
        return explicit
    run_timeout = env_int("RLLM_HARNESS_RUN_TIMEOUT_S", 3600)
    install_timeout = env_int("RLLM_HARNESS_INSTALL_TIMEOUT_S", 600)
    return run_timeout + install_timeout + _SANDBOX_LIFETIME_HEADROOM_S


# Modal caps an exec's total argv at 64 KiB (ARG_MAX); payloads above this go
# through a chunked temp-file path instead of being inlined in the command.
_B64_ARGV_LIMIT = 50_000

# atexit-tracked sandboxes; terminated on process exit to avoid leaks.
_LIVE_SANDBOXES: weakref.WeakSet = weakref.WeakSet()
_LIVE_LOCK = threading.Lock()


def _terminate_all_live() -> None:
    """atexit hook: terminate every still-alive ModalSandbox."""
    with _LIVE_LOCK:
        survivors = list(_LIVE_SANDBOXES)
    if not survivors:
        return
    logger.warning("atexit: terminating %d unreleased Modal sandbox(es)", len(survivors))
    for sb in survivors:
        try:
            sb.close()
        except Exception:
            logger.debug("atexit: error closing %s", getattr(sb, "name", "<unknown>"), exc_info=True)


atexit.register(_terminate_all_live)


class ModalSandbox:
    """Sandbox implementation using Modal serverless containers.

    Creates a Modal Sandbox via ``modal.Sandbox.create()``, executes
    commands via ``sandbox.exec()``, and manages file operations through
    exec-based helpers (Modal's file API is alpha).

    The ``image`` parameter accepts either:
    - A Docker Hub image name (e.g. ``"python:3.11-slim"``) — will be
      wrapped with ``modal.Image.from_registry()``.
    - A ``modal.Image`` object directly.

    Optional kwargs:
    - ``app_name``: Modal App name (default: ``"rllm-sandbox"``).
    - ``timeout``: Sandbox lifetime in seconds (default: 1800).
    - ``secrets``: List of ``modal.Secret`` objects.
    - ``volumes``: Dict mapping mount paths to ``modal.Volume`` objects.
    - ``workdir``: Working directory inside the container.
    - ``gpu``: GPU spec (e.g. ``"T4"``, ``"A10G"``).
    - ``cpu``: CPU count (float).
    - ``memory``: Memory in MB (int).
    """

    def __init__(self, name: str, image: str = "python:3.11-slim", **kwargs):
        import modal

        self.name = name
        self._image_spec = image
        self._timeout = kwargs.pop("timeout", None) or _default_sandbox_timeout()
        self._app_name = kwargs.pop("app_name", "rllm-sandbox")

        # A stored snapshot ref is a bare Modal image id ("im-…", no registry/tag);
        # the ":" / "/" guard keeps real docker images off the from_id path.
        from_snapshot = isinstance(image, str) and image.startswith("im-") and ":" not in image and "/" not in image
        self._app = modal.App.lookup(self._app_name, create_if_missing=True)

        # A missing snapshot surfaces as NotFoundError either at from_id (if it
        # ever resolves eagerly) or at create; translate both so get_sandbox can
        # fall back to cold. A registry-image NotFoundError is a genuine bad
        # image — let it raise.
        try:
            if from_snapshot:
                modal_image = modal.Image.from_id(image)
            elif isinstance(image, str):
                modal_image = modal.Image.from_registry(image)
            else:
                modal_image = image  # already a modal.Image

            create_kwargs: dict = {"app": self._app, "image": modal_image, "timeout": self._timeout}
            for key in ("secrets", "volumes", "workdir", "gpu", "cpu", "memory"):
                if key in kwargs:
                    create_kwargs[key] = kwargs.pop(key)

            self._sandbox = modal.Sandbox.create(**create_kwargs)
        except modal.exception.NotFoundError as e:
            if from_snapshot:
                raise SnapshotNotFound(f"modal snapshot {image} no longer exists") from e
            raise
        self._sandbox_id = self._sandbox.object_id

        with _LIVE_LOCK:
            _LIVE_SANDBOXES.add(self)

        logger.info(
            "ModalSandbox %s created (id: %s, image: %s)",
            name,
            self._sandbox_id,
            image if isinstance(image, str) else "<modal.Image>",
        )

    def exec(self, command: str, timeout: float | None = None, user: str | None = None) -> str:  # noqa: ARG002
        """Execute a command inside the Modal sandbox.

        ``user`` is accepted for protocol compatibility but currently ignored.

        Runs ``bash -c <command>`` and returns stdout. Raises
        ``RuntimeError`` on non-zero exit code (matching DockerSandbox
        behavior).
        """
        exec_kwargs = {}
        if timeout is not None:
            exec_kwargs["timeout"] = int(timeout)

        process = self._sandbox.exec("bash", "-c", command, **exec_kwargs)

        stdout = process.stdout.read()
        stderr = process.stderr.read()

        # Wait for the process to complete and get exit code
        process.wait()
        exit_code = process.returncode

        if exit_code != 0:
            logger.warning(
                "Command failed in sandbox %s: %s\nstderr: %s",
                self.name,
                command,
                stderr[:500],
            )
            raise RuntimeError(f"Command failed (exit {exit_code}) in sandbox {self.name}: {command}\n{stderr[:500]}")
        return stdout

    def _push_b64(self, b64: str, consume: str) -> None:
        """Feed a base64 payload to ``consume`` (a shell command reading stdin).

        Modal rejects execs whose argv exceeds 64 KiB, so a small payload is
        inlined while a large one is appended to a sandbox temp file in
        argv-sized chunks and streamed from there.
        """
        if len(b64) <= _B64_ARGV_LIMIT:
            self._exec_unchecked(f"echo '{b64}' | {consume}")
            return
        import uuid as _uuid

        tmp = f"/tmp/.rllm-b64-{_uuid.uuid4().hex[:8]}"
        self._exec_unchecked(f": > {tmp}")
        for i in range(0, len(b64), _B64_ARGV_LIMIT):
            self._exec_unchecked(f"printf %s '{b64[i : i + _B64_ARGV_LIMIT]}' >> {tmp}")
        # Brace group so the stdin redirect feeds the FIRST pipeline member
        # (`cmd1 | cmd2 < f` would bind f to cmd2 and leave cmd1 blocked).
        self._exec_unchecked(f"{{ {consume}; }} < {tmp}; rc=$?; rm -f {tmp}; exit $rc")

    def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload a single file into the Modal sandbox.

        Uses exec to write file contents since Modal's file API is in alpha.
        """
        remote_dir = os.path.dirname(remote_path)
        if remote_dir:
            self._exec_unchecked(f"mkdir -p {remote_dir}")

        with open(local_path, "rb") as f:
            content = f.read()

        # Use base64 encoding for safe binary transfer
        import base64

        b64 = base64.b64encode(content).decode("ascii")
        self._push_b64(b64, f"base64 -d > {remote_path}")
        logger.debug("Uploaded %s -> %s in sandbox %s", local_path, remote_path, self.name)

    def upload_dir(self, local_path: str, remote_path: str) -> None:
        """Upload a directory tree into the Modal sandbox.

        Creates a tar archive locally, base64-encodes it, and extracts
        it inside the sandbox.
        """
        remote_parent = os.path.dirname(remote_path.rstrip("/"))
        remote_name = os.path.basename(remote_path.rstrip("/"))

        if remote_parent:
            self._exec_unchecked(f"mkdir -p {remote_parent}")

        # Create tar in memory
        tar_buf = io.BytesIO()
        with tarfile.open(fileobj=tar_buf, mode="w:gz") as tar:
            tar.add(local_path, arcname=remote_name)
        tar_buf.seek(0)

        import base64

        b64 = base64.b64encode(tar_buf.read()).decode("ascii")

        # Write tar to sandbox and extract. --no-same-owner: don't restore the
        # host's uid/gid (root extraction would otherwise chown to nonexistent
        # ids and error); permissions are kept so executables stay +x.
        self._push_b64(b64, f"base64 -d | tar xzf - --no-same-owner -C {remote_parent}")
        logger.debug("Uploaded dir %s -> %s in sandbox %s", local_path, remote_path, self.name)

    def is_alive(self) -> bool:
        """One API call: ``poll()`` returns ``None`` while the sandbox is still running.

        A Modal sandbox dies for good when its ``timeout`` (total lifetime,
        not idle time) elapses or it is terminated; ``poll()`` then returns
        an exit code (verified live: a box past its lifetime polls ``124``).
        Caveat: ``poll()`` can lag a *just*-issued terminate by a few
        seconds, so this is a "has it died" check, not a fence against
        in-flight termination — fine for callers like the warm queue,
        whose dead boxes have been dead for minutes by the time they're
        checked.
        """
        try:
            return self._sandbox.poll() is None
        except Exception:
            logger.debug("ModalSandbox %s is_alive check failed — treating as dead", self.name, exc_info=True)
            return False

    def close(self) -> None:
        """Terminate and detach from the Modal sandbox."""
        try:
            self._sandbox.terminate()
        except Exception:
            logger.debug("Sandbox %s terminate error (may already be stopped)", self.name)
        try:
            self._sandbox.detach()
        except Exception:
            pass
        with _LIVE_LOCK:
            _LIVE_SANDBOXES.discard(self)
        logger.info("ModalSandbox %s closed", self.name)

    def _exec_unchecked(self, command: str) -> str:
        """Execute a command without raising on non-zero exit."""
        try:
            return self.exec(command)
        except RuntimeError:
            return ""


def create_modal_sandbox(name: str, image: str = "python:3.11-slim", **kwargs) -> ModalSandbox:
    """Factory function for creating a ModalSandbox."""
    return ModalSandbox(name=name, image=image, **kwargs)


def _modal_ref_alive(ref: str) -> bool:
    """True only when the image is confirmed present, for build reuse (one no-boot ``ImageFromId`` RPC).

    ``hydrate()`` resolves the id without creating a sandbox. Unsure (auth/unknown)
    returns ``False`` so the caller rebuilds rather than reuse a ref it cannot confirm.
    """
    import modal
    from modal.exception import NotFoundError

    try:
        modal.Image.from_id(ref).hydrate()
        return True
    except NotFoundError:
        return False
    except Exception:
        logger.debug("modal ref probe failed for %s — treating as unknown", ref, exc_info=True)
        return False


def _modal_ref_absent(ref: str) -> bool:
    """True only when the image is confirmed gone, for sync prune.

    Not the inverse of :func:`_modal_ref_alive`: both default to ``False`` when unsure,
    for opposite-safe reasons — here unsure (auth/unknown) means keep the local record
    rather than prune one we cannot confirm is absent.
    """
    import modal
    from modal.exception import NotFoundError

    try:
        modal.Image.from_id(ref).hydrate()
        return False  # present
    except NotFoundError:
        return True  # confirmed gone
    except Exception:
        logger.debug("modal ref probe failed for %s — treating as unknown (keep)", ref, exc_info=True)
        return False


def build_modal_snapshot(task, key: str, prior_ref: str | None = None, *, force: bool = False, install_script: str = "") -> str | None:
    """Build a Modal filesystem snapshot of ``task``'s environment; return its image id.

    Mirrors Daytona's idempotency: when a ``prior_ref`` is known and still live
    (probed via :func:`_modal_ref_alive`), reuse it instead of capturing a fresh
    filesystem — every fresh capture orphans the old image forever. ``force``
    bypasses reuse and always rebuilds.

    Create a base sandbox, replay the Dockerfile RUN steps, run the install
    script (if any), then capture the live filesystem as a ``modal.Image``
    (stored as a diff from the base). A failed install fails the build — a
    snapshot keyed on the install must actually contain it.
    """
    from rllm.eval._resolution import (
        _create_base_sandbox,
        _dockerfile_run_commands,
        _replay_dockerfile,
        _should_replay_dockerfile,
    )

    if prior_ref and not force and _modal_ref_alive(prior_ref):
        logger.info("modal snapshot %s already live (%s) — reusing", key, prior_ref)
        return prior_ref

    # Size the build sandbox's lifetime to the worst-case replay: each RUN is
    # bounded at 900s (a step that hangs against a prebuilt image burns its
    # full bound), plus the install bound and pull/capture slack — floored at
    # the rollout default. Without this floor, two hung steps killed the
    # sandbox mid-build.
    install_budget = env_int("RLLM_HARNESS_INSTALL_TIMEOUT_S", 900) if install_script else 0
    n_replay = len(_dockerfile_run_commands(task)) if _should_replay_dockerfile(task) else 0
    build_timeout = max(_default_sandbox_timeout(), 900 * n_replay + install_budget + 600)
    sb = _create_base_sandbox(task, "modal", name=f"{key}-build", timeout=build_timeout)
    try:
        _replay_dockerfile(task, sb, "modal")
        if install_script:
            sb.exec(install_script, timeout=env_int("RLLM_HARNESS_INSTALL_TIMEOUT_S", 900), user="root")
        image = sb._sandbox.snapshot_filesystem()  # noqa: SLF001 — modal.Image
        logger.info("modal snapshot built: %s -> %s", key, image.object_id)
        return image.object_id
    finally:
        try:
            sb.close()
        except Exception:
            logger.debug("build sandbox close failed", exc_info=True)


def delete_modal_snapshot(ref: str) -> bool:
    """Delete a Modal snapshot by image id; ``True`` only when confirmed gone, ``False`` (keep) otherwise."""
    from modal.exception import AuthError, NotFoundError, PermissionDeniedError
    from modal.experimental import image_delete

    try:
        image_delete(ref)
        return True
    except NotFoundError:
        return True  # already gone — safe to drop
    except (AuthError, PermissionDeniedError):
        logger.warning("no permission to delete modal snapshot %s — keeping local record", ref)
        return False
    except Exception:
        logger.warning("failed to delete modal snapshot %s — keeping local record", ref, exc_info=True)
        return False
