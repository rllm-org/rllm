"""SandboxedAgentFlow: base class for agents that run against a sandbox.

The flow holds no sandbox state. :class:`rllm.hooks.SandboxTaskHooks` owns
the lifecycle: it creates the sandbox, runs the harness install (when not
already baked into the image), and closes it after evaluation. The engine
passes the live sandbox into ``run(task, config, *, env)`` as a call
argument, so parallel rollouts can share one flow instance safely.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from rllm.sandbox.protocol import Sandbox
from rllm.types import AgentConfig, Episode, Task

logger = logging.getLogger(__name__)


class SandboxedAgentFlow(ABC):
    """Base class for agents that run against a sandboxed execution environment.

    The sandbox backend is pluggable via ``sandbox_backend``:
    ``"docker"`` | ``"local"`` | ``"modal"`` | ``"daytona"``.

    Subclasses implement :meth:`run` and may override :meth:`get_image`
    for per-task images.
    """

    # Declared env requirement — read by rllm.hooks.resolve_rollout_plan.
    needs_env: bool = True
    # Where the flow's LLM client runs. Host-side loops (bash/oracle) keep the
    # local gateway URL; CLI harnesses (BaseCliHarness) override to True so
    # the in-sandbox process gets the publicly-reachable (tunneled) URL.
    llm_inside_env: bool = False
    sandbox_backend: str = "docker"
    image: str = "python:3.11-slim"
    # Default cap on concurrent sandboxes. The eval/train runner clamps
    # effective concurrency to this value, so it is the single source of
    # truth for sandboxed flows. Subclasses override only to deviate;
    # ``--sandbox-concurrency`` overrides it per-run.
    max_concurrent: int = 64

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def configure(self, overrides: dict) -> dict:
        """Apply caller/CLI overrides this flow understands; return the rest.

        The wiring layer warns about anything returned, so a flag that a
        given agent can't honor is visible instead of a silent no-op.
        """
        leftovers = dict(overrides)
        backend = leftovers.pop("sandbox_backend", None)
        if backend is not None:
            self.sandbox_backend = backend
        concurrency = leftovers.pop("sandbox_concurrency", None)
        if concurrency is not None:
            self.max_concurrent = concurrency
        return leftovers

    def get_image(self, task: dict) -> str:
        """Return container image for this task. Override for per-task images."""
        return self.image

    @abstractmethod
    def run(self, task: Task, config: AgentConfig, *, env: Sandbox) -> Episode: ...


def create_sandbox(backend: str, name: str, image: str, **kwargs) -> Sandbox:
    """Factory: create a Sandbox from a backend name. Lazy imports."""
    if backend == "docker":
        from rllm.sandbox.backends.docker import DockerSandbox

        return DockerSandbox(name=name, image=image, **kwargs)
    elif backend == "local":
        from rllm.sandbox.backends.local import LocalSandbox

        return LocalSandbox(name=name, **kwargs)
    elif backend == "modal":
        from rllm.sandbox.backends.modal_backend import ModalSandbox

        return ModalSandbox(name=name, image=image, **kwargs)
    elif backend == "daytona":
        from rllm.sandbox.backends.daytona import DaytonaSandbox

        return DaytonaSandbox(name=name, image=image, **kwargs)
    elif backend == "bwrap":
        from rllm.sandbox.backends.bwrap import BwrapSandbox

        return BwrapSandbox(name=name, **kwargs)
    else:
        raise ValueError(f"Unknown sandbox backend: {backend}. Available: docker, local, modal, daytona, bwrap")


def build_snapshot(backend: str, task: Task, key: str, prior_ref: str | None = None, *, force: bool = False, install_script: str = "") -> str | None:
    """Build a snapshot of ``task``'s environment; return a backend ref, or ``None``.

    Each backend owns its mechanism (Modal: live-FS capture; Daytona:
    declarative bake). Backends without snapshots (docker/local) return ``None``.
    A known-live ``prior_ref`` is reused unless ``force``, which always rebuilds.
    ``install_script`` is baked on top of the task's RUN steps — ``key`` must
    have been computed with the same script.
    """
    if backend == "modal":
        from rllm.sandbox.backends.modal_backend import build_modal_snapshot

        return build_modal_snapshot(task, key, prior_ref, force=force, install_script=install_script)
    elif backend == "daytona":
        from rllm.sandbox.backends.daytona import build_daytona_snapshot

        return build_daytona_snapshot(task, key, force=force, install_script=install_script)
    return None


def delete_snapshot(backend: str, ref: str) -> bool:
    """Delete a snapshot from its backend. Returns ``True`` on success."""
    if backend == "modal":
        from rllm.sandbox.backends.modal_backend import delete_modal_snapshot

        return delete_modal_snapshot(ref)
    elif backend == "daytona":
        from rllm.sandbox.backends.daytona import delete_daytona_snapshot

        return delete_daytona_snapshot(ref)
    return False


def snapshot_absent(backend: str, ref: str) -> bool:
    """No-boot probe for ``registry.sync``: ``True`` only when ``ref`` is verifiably gone.

    Conservative by construction — auth/permission/rate-limit/unknown errors return
    ``False`` so sync never prunes a record it cannot confirm is absent.
    """
    if backend == "modal":
        from rllm.sandbox.backends.modal_backend import _modal_ref_absent

        return _modal_ref_absent(ref)
    elif backend == "daytona":
        from rllm.sandbox.backends.daytona import _daytona_ref_absent

        return _daytona_ref_absent(ref)
    return False
