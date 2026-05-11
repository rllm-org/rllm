"""Minimal patches for SWE-ReX Modal deployment.

Fixes:
1. Startup command: /bin/sh instead of /usr/bin/env bash
2. GCR/GAR image support
3. Install pipx via pyenv (system python3-venv often missing)
4. Fix stop() to always terminate sandboxes (fixes FD leak)
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any


def apply_swerex_modal_minimal_patch(secret_name: str = "turing-swe-bench"):
    """Apply minimal patches to ModalDeployment."""
    import modal
    from swerex.deployment import modal as swerex_modal

    if getattr(swerex_modal.ModalDeployment, "_minimal_patch_applied", False):
        return

    image_builder_cls = swerex_modal._ImageBuilder
    modal_deployment_cls = swerex_modal.ModalDeployment

    def _read_positive_float_env(name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None:
            return default

        value = float(raw)
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")
        return value

    def _run_coro_blocking(coro: Any) -> Any:
        """Run a coroutine from sync code, including if an event loop exists."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coro).result()

    # Patch 0: Bound Modal App.lookup.
    # The upstream sync wrapper can block indefinitely in transient Modal
    # control-plane stalls. Raise a normal timeout so the existing sandbox retry
    # loop in agent_flow.py can move on instead of pinning a rollout thread.
    def _init_with_bounded_lookup(
        self: Any,
        *,
        logger: Any | None = None,
        image: Any,
        startup_timeout: float = 0.4,
        runtime_timeout: float = 3600.0,
        modal_sandbox_kwargs: dict[str, Any] | None = None,
        install_pipx: bool = True,
        deployment_timeout: float = 3600.0,
    ) -> None:
        self._image = image_builder_cls(install_pipx=install_pipx, logger=logger).auto(image)
        self._runtime = None
        self._startup_timeout = startup_timeout
        self._sandbox = None
        self._port = 8880
        self.logger = logger or swerex_modal.get_logger("rex-deploy")

        lookup_default = min(max(float(startup_timeout), 1.0), 60.0)
        lookup_timeout = _read_positive_float_env(
            "SWE_REX_MODAL_APP_LOOKUP_TIMEOUT_S",
            _read_positive_float_env("SWE_REX_MODAL_CONTROL_TIMEOUT_S", lookup_default),
        )

        async def _lookup_app() -> Any:
            return await asyncio.wait_for(
                modal.App.lookup.aio("swe-rex", create_if_missing=True),
                timeout=lookup_timeout,
            )

        try:
            self._app = _run_coro_blocking(_lookup_app())
        except TimeoutError as exc:
            raise TimeoutError(f"Modal App.lookup timed out after {lookup_timeout:.1f}s") from exc
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"Modal App.lookup timed out after {lookup_timeout:.1f}s") from exc

        self._user = swerex_modal._get_modal_user()
        self._runtime_timeout = runtime_timeout
        self._deployment_timeout = deployment_timeout
        self._modal_kwargs = modal_sandbox_kwargs or {}
        self._hooks = swerex_modal.CombinedDeploymentHook()

    modal_deployment_cls.__init__ = _init_with_bounded_lookup  # type: ignore

    # Patch 1: Add GCR image support
    original_auto = image_builder_cls.auto

    def _auto_with_gcr(self: Any, image_spec: Any) -> Any:
        if isinstance(image_spec, str) and ("pkg.dev" in image_spec or "gcr.io" in image_spec):
            secret = modal.Secret.from_name(secret_name)
            image = modal.Image.from_gcp_artifact_registry(image_spec, secret=secret)
            if getattr(self, "_install_pipx", False):
                image = self.ensure_pipx_installed(image)
            return image
        return original_auto(self, image_spec)

    image_builder_cls.auto = _auto_with_gcr  # type: ignore

    # Patch 2: Install pipx via pyenv Python 3.11 (from SWE-bench_Pro-os)
    # The default swerex `apt_install("pipx")` fails on images without python3-venv.
    def _ensure_pipx_via_pyenv(self: Any, image: Any) -> Any:
        return image \
            .run_commands("pip config unset global.index-url || true") \
            .run_commands("(apt update && apt install -y curl) || (apk update && apk add --no-cache curl bash)") \
            .run_commands("curl https://pyenv.run | bash") \
            .run_commands(
                "(apt update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata"
                " && apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev"
                " libreadline-dev libsqlite3-dev curl git libncursesw5-dev xz-utils tk-dev"
                " libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev)"
                " || (apk add --no-cache make build-base openssl-dev zlib-dev bzip2-dev"
                " readline-dev sqlite-dev git ncurses-dev xz tk-dev libxml2-dev xmlsec-dev"
                " libffi-dev xz-dev)"
            ) \
            .run_commands("~/.pyenv/bin/pyenv install 3.11.13") \
            .run_commands("~/.pyenv/versions/3.11.13/bin/python3.11 -m pip install pipx") \
            .run_commands("~/.pyenv/versions/3.11.13/bin/python3.11 -m pipx ensurepath") \
            .run_commands("ln -sf ~/.pyenv/versions/3.11.13/bin/pipx /usr/local/bin/pipx") \
            .entrypoint([])

    image_builder_cls.ensure_pipx_installed = _ensure_pipx_via_pyenv  # type: ignore

    # Patch 3: Fix start() to use /bin/sh with ~/.local/bin on PATH
    async def _start_with_bash(self: Any) -> None:
        if self._runtime is not None and self._sandbox is not None:
            return

        import asyncio
        import time
        from swerex.runtime.remote import RemoteRuntime

        self.logger.info("Starting modal sandbox")
        self._hooks.on_custom_step("Starting modal sandbox")
        t0 = time.time()
        token = self._get_token()

        cmd = self._start_swerex_cmd(token)
        create_timeout = _read_positive_float_env(
            "SWE_REX_MODAL_SANDBOX_CREATE_TIMEOUT_S",
            _read_positive_float_env("SWE_REX_MODAL_CONTROL_TIMEOUT_S", max(float(self._startup_timeout), 1.0)),
        )
        self._sandbox = await asyncio.wait_for(
            modal.Sandbox.create.aio(
                "/bin/sh",
                "-c",
                f"export PATH=\"$HOME/.local/bin:$PATH\" && {cmd}",
                image=self._image,
                timeout=int(self._deployment_timeout),
                unencrypted_ports=[self._port],
                app=self._app,
                **self._modal_kwargs,
            ),
            timeout=create_timeout,
        )

        tunnels_timeout = _read_positive_float_env(
            "SWE_REX_MODAL_TUNNELS_TIMEOUT_S",
            _read_positive_float_env("SWE_REX_MODAL_CONTROL_TIMEOUT_S", min(max(float(self._startup_timeout), 1.0), 60.0)),
        )
        tunnels = await asyncio.wait_for(self._sandbox.tunnels.aio(), timeout=tunnels_timeout)
        tunnel = tunnels[self._port]
        elapsed = time.time() - t0
        log_url = await self.get_modal_log_url()
        self.logger.info(f"Sandbox created in {elapsed:.2f}s: {log_url}")

        await asyncio.sleep(1)
        self._runtime = RemoteRuntime(
            host=tunnel.url,
            timeout=self._runtime_timeout,
            auth_token=token,
            logger=self.logger,
        )

        remaining = max(0, self._startup_timeout - elapsed)
        await self._wait_until_alive(timeout=remaining)

    modal_deployment_cls.start = _start_with_bash  # type: ignore

    # Patch 4: Fix stop() to always terminate sandboxes
    # Original swerex has inverted logic: only terminates already-exited processes.
    async def _stop_always_terminate(self: Any) -> None:
        if self._runtime is not None:
            try:
                await self._runtime.close()
            except Exception:
                pass
            self._runtime = None
        if self._sandbox is not None:
            try:
                await self._sandbox.terminate.aio()
            except Exception:
                pass
        self._sandbox = None
        self._app = None

    modal_deployment_cls.stop = _stop_always_terminate  # type: ignore
    modal_deployment_cls._minimal_patch_applied = True  # type: ignore
