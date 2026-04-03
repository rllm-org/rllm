"""Minimal patches for SWE-ReX Modal deployment.

Fixes:
1. Startup command: /bin/sh instead of /usr/bin/env bash
2. Install pipx via pyenv (system python3-venv often missing)
3. Fix stop() to always terminate sandboxes (fixes FD leak)
"""

from __future__ import annotations

from typing import Any


def apply_swerex_modal_minimal_patch():
    """Apply minimal patches to ModalDeployment."""
    import modal
    from swerex.deployment import modal as swerex_modal

    if getattr(swerex_modal.ModalDeployment, "_minimal_patch_applied", False):
        return

    image_builder_cls = swerex_modal._ImageBuilder
    modal_deployment_cls = swerex_modal.ModalDeployment

    # Patch 1: Install pipx via pyenv Python 3.11 (from SWE-bench_Pro-os)
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

    # Patch 2: Fix start() to use /bin/sh with ~/.local/bin on PATH
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
        self._sandbox = await modal.Sandbox.create.aio(
            "/bin/sh",
            "-c",
            f"export PATH=\"$HOME/.local/bin:$PATH\" && {cmd}",
            image=self._image,
            timeout=int(self._deployment_timeout),
            unencrypted_ports=[self._port],
            app=self._app,
            **self._modal_kwargs,
        )

        tunnels = await self._sandbox.tunnels.aio()
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

    # Patch 3: Fix stop() to always terminate sandboxes
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
