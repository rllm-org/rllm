"""Shared Compose *spec* logic for the Compose-backed sandboxes.

Both Compose sandboxes drive the same substrate — the ``docker compose``
CLI — so the artifacts they assemble are identical: rLLM's base-overlay
YAML (the ``main`` service build + keepalive + resource limits) and the
``docker compose`` argument list. Only *where* those run differs (host
subprocess vs. docker-in-docker inside a Modal VM). This base holds the
shared assembly; each subclass keeps its own transport.

Scope rule: **spec only — never transport, never public API.** No exec,
no uploads, no lifecycle. The public contract is
:class:`rllm.sandbox.protocol.ComposeSandbox`, which the subclasses
satisfy structurally; this class is an implementation detail and is not
itself a Sandbox.
"""

from __future__ import annotations

import re


class BaseComposeSandbox:
    """Compose-spec helpers shared by ``DockerComposeSandbox`` / ``ModalComposeSandbox``.

    Subclasses must set the invocation-view paths in ``__init__`` — the paths
    *as the ``docker compose`` process will see them* (host paths for the
    docker backend; in-VM paths for the Modal backend):

    - ``_project``:            Compose project name (:meth:`_compose_project_name`).
    - ``_compose_env_dir``:    the task's ``environment/`` dir — build context
                               and ``--project-directory``.
    - ``_compose_base_file``:  where rLLM's base overlay YAML lives.
    - ``_compose_task_file``:  the task-authored compose file.
    """

    _project: str
    _compose_env_dir: str
    _compose_base_file: str
    _compose_task_file: str

    @staticmethod
    def _compose_project_name(name: str) -> str:
        """Return an isolated, Compose-safe project name."""
        normalized = re.sub(r"[^a-z0-9_-]+", "-", name.lower()).strip("-_")
        return (normalized or "rllm")[:63]

    def _base_overlay(self, resources: dict) -> dict:
        """rLLM's base Compose file, as a dict: the ``main`` service the agent lives in.

        Task compose files are overlays on top of this — they add sidecars,
        networks, healthchecks, volumes, and may override ``main``. The
        ``sleep infinity`` keepalive replaces the image CMD so rLLM drives
        the container, mirroring the single-image sandboxes.
        """
        main: dict = {
            "build": {"context": self._compose_env_dir, "dockerfile": "Dockerfile"},
            "command": ["sleep", "infinity"],
        }
        cpus = resources.get("cpus")
        memory_mb = resources.get("memory_mb")
        env = resources.get("env")
        if cpus:
            main["cpus"] = float(cpus)
        if memory_mb:
            main["mem_limit"] = f"{int(memory_mb)}m"
        if env:
            main["environment"] = {str(k): str(v) for k, v in env.items()}
        return {"services": {"main": main}}

    def _compose_parts(self, *args: str) -> list[str]:
        """The ``docker compose`` argv: base + task overlay, in merge order."""
        return [
            "docker",
            "compose",
            "--project-name",
            self._project,
            "--project-directory",
            self._compose_env_dir,
            "-f",
            self._compose_base_file,
            "-f",
            self._compose_task_file,
            *args,
        ]
