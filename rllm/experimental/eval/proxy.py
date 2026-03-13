"""EvalProxyManager: LiteLLM proxy for external providers (OpenAI, etc.).

Extends ``ProxyManager`` to generate a LiteLLM configuration that routes
requests to an external provider, following the same pattern as
``VerlProxyManager`` and ``TinkerProxyManager``.

The caller controls the proxy lifecycle::

    pm = EvalProxyManager(provider="openai", model_name="gpt-4o", api_key="sk-...")
    pm.start_proxy_subprocess(pm.build_proxy_config())
    base_url = pm.get_proxy_url()  # http://127.0.0.1:4000/v1
    # ... run eval ...
    pm.shutdown_proxy()
"""

from __future__ import annotations

import atexit
import logging
import os
import resource
import subprocess
import sys
import tempfile
import time
from typing import Any

import requests

from rllm.sdk.proxy.proxy_manager import ProxyManager

logger = logging.getLogger(__name__)


class EvalProxyManager(ProxyManager):
    """Manages a LiteLLM proxy that routes to an external provider."""

    def __init__(
        self,
        provider: str,
        model_name: str,
        api_key: str,
        proxy_host: str = "127.0.0.1",
        proxy_port: int = 4000,
    ) -> None:
        super().__init__(proxy_host=proxy_host, proxy_port=proxy_port)
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self._stderr_path: str | None = None

    def _generate_litellm_config(self) -> dict[str, Any]:
        """Generate LiteLLM configuration for the configured provider."""
        from rllm.experimental.eval.config import get_provider_info

        # Use the registry's litellm_prefix (e.g. "together_ai" for "together")
        info = get_provider_info(self.provider)
        prefix = info.litellm_prefix if info else self.provider
        litellm_model = f"{prefix}/{self.model_name}"

        return {
            "model_list": [
                {
                    "model_name": self.model_name,
                    "litellm_params": {
                        "model": litellm_model,
                        "api_key": self.api_key,
                    },
                }
            ],
            "litellm_settings": {
                "drop_params": True,
                "num_retries": 3,
            },
        }

    def build_proxy_config(self) -> dict[str, Any]:
        """Return a fresh LiteLLM configuration."""
        return self._generate_litellm_config()

    def start_proxy_subprocess(self, config: dict[str, Any], **kwargs) -> str:
        """Start LiteLLM proxy, capturing stderr for diagnostics on failure."""
        if self._proxy_process is not None:
            logger.warning("Proxy subprocess already running")
            return ""

        snapshot_path = self._snapshot_config_to_file(config, directory=kwargs.get("snapshot_directory"))
        if not snapshot_path or not os.path.exists(snapshot_path):
            raise RuntimeError("Config snapshot not available. Cannot start proxy.")

        cmd = [
            sys.executable,
            "-m",
            "rllm.sdk.proxy.litellm_server",
            "--host",
            self.proxy_host,
            "--port",
            str(self.proxy_port),
        ]

        env = os.environ.copy()
        try:
            import certifi

            ca_path = certifi.where()
            env["SSL_CERT_FILE"] = ca_path
            env["REQUESTS_CA_BUNDLE"] = ca_path
            env["CURL_CA_BUNDLE"] = ca_path
            env["OPENAI_CA_BUNDLE"] = ca_path
        except Exception:
            pass

        def set_limits() -> None:
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))
            except (ValueError, OSError):
                pass

        # Capture stderr to a temp file so we can show errors on failure
        stderr_file = tempfile.NamedTemporaryFile(mode="w", prefix="rllm_proxy_", suffix=".log", delete=False)
        self._stderr_path = stderr_file.name

        logger.info("Starting proxy subprocess: %s", " ".join(cmd))
        self._proxy_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=stderr_file,
            env=env,
            preexec_fn=set_limits,
        )

        try:
            self._wait_for_proxy(timeout=30.0)
            logger.info("Proxy server started, sending configuration...")
            self.reload_proxy_config(config=config)
            logger.info("Proxy configuration loaded successfully")
        except Exception:
            self._report_proxy_failure()
            self.shutdown_proxy()
            raise

        atexit.register(self.shutdown_proxy)
        logger.info("Proxy subprocess ready (PID: %s)", self._proxy_process.pid)
        return snapshot_path

    def _wait_for_proxy(self, timeout: float = 30.0) -> None:
        """Wait for proxy to start, with stderr-aware error reporting."""
        if self._proxy_process is None:
            raise RuntimeError("Proxy process not started")

        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._proxy_process.poll() is not None:
                exit_code = self._proxy_process.returncode
                stderr_tail = self._read_stderr_tail()
                msg = f"Proxy process died during startup with exit code {exit_code}"
                if stderr_tail:
                    msg += f"\n\n{stderr_tail}"
                raise RuntimeError(msg)

            try:
                requests.get(f"http://{self.proxy_host}:{self.proxy_port}/", timeout=0.5)
                logger.info("Proxy server accepting connections")
                return
            except requests.RequestException:
                pass

            time.sleep(0.3)

        raise TimeoutError(f"Proxy server did not start within {timeout}s")

    def _read_stderr_tail(self, max_lines: int = 30) -> str:
        """Read the last N lines from the stderr log file."""
        if not self._stderr_path or not os.path.exists(self._stderr_path):
            return ""
        try:
            with open(self._stderr_path) as f:
                lines = f.readlines()
            tail = lines[-max_lines:] if len(lines) > max_lines else lines
            return "".join(tail).strip()
        except OSError:
            return ""

    def _report_proxy_failure(self) -> None:
        """Print stderr output to help diagnose proxy startup failures."""
        stderr_output = self._read_stderr_tail()
        if stderr_output:
            logger.error("Proxy stderr output:\n%s", stderr_output)

    def shutdown_proxy(self) -> None:
        """Shutdown proxy and clean up stderr log."""
        super().shutdown_proxy()
        if self._stderr_path:
            try:
                os.unlink(self._stderr_path)
            except OSError:
                pass
            self._stderr_path = None

    def __repr__(self) -> str:
        mode = "subprocess" if self._proxy_process else "external"
        return f"EvalProxyManager(provider={self.provider}, model={self.model_name}, proxy={self.get_proxy_url()}, mode={mode})"
