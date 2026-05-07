"""EvalProxyManager: LiteLLM proxy for external providers (OpenAI, etc.).

The caller controls the proxy lifecycle::

    pm = EvalProxyManager(provider="openai", model_name="gpt-4o", api_key="sk-...")
    pm.start_proxy_subprocess(pm.build_proxy_config())
    base_url = pm.get_proxy_url()  # http://127.0.0.1:4000/v1
    # ... run eval ...
    pm.shutdown_proxy()

This proxy is the bridge to commercial providers during ``rllm eval``. For
training and for vLLM-backed eval, requests go through ``rllm-model-gateway``
instead.
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
import yaml

logger = logging.getLogger(__name__)


class _ProxyManagerBase:
    """LiteLLM proxy lifecycle helpers shared between subclasses."""

    def __init__(
        self,
        proxy_host: str = "127.0.0.1",
        proxy_port: int = 4000,
        admin_token: str | None = None,
    ) -> None:
        self.proxy_host = proxy_host
        self.proxy_port = proxy_port
        self.admin_token = admin_token
        self._proxy_process: subprocess.Popen | None = None

    def _snapshot_config_to_file(self, config: dict[str, Any], directory: str | None = None) -> str | None:
        base_dir = directory or os.getenv("RLLM_PROXY_CONFIG_DIR") or os.getcwd()
        os.makedirs(base_dir, exist_ok=True)
        snapshot_path = os.path.join(base_dir, "litellm_proxy_config_autogen.yaml")
        with open(snapshot_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info("LiteLLM config snapshot written to %s", snapshot_path)
        return snapshot_path

    def reload_proxy_config(
        self,
        config: dict[str, Any],
        reload_url: str | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        url = reload_url or f"http://{self.proxy_host}:{self.proxy_port}/admin/reload"
        if config is None:
            raise RuntimeError("LiteLLM config must be provided when reloading.")

        payload = {"config_yaml": yaml.dump(config, default_flow_style=False)}

        headers = {"Content-Type": "application/json"}
        if self.admin_token:
            token = self.admin_token if self.admin_token.lower().startswith("bearer ") else f"Bearer {self.admin_token}"
            headers["Authorization"] = token

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Failed to reload LiteLLM proxy via {url}: {exc}") from exc

        try:
            return resp.json()
        except ValueError:
            return {"status": "ok", "raw": resp.text}

    def get_proxy_url(self, include_v1: bool = True) -> str:
        base = f"http://{self.proxy_host}:{self.proxy_port}"
        return f"{base}/v1" if include_v1 else base

    def shutdown_proxy(self) -> None:
        if self._proxy_process is None:
            return

        logger.info("Shutting down proxy subprocess...")
        self._proxy_process.terminate()
        try:
            self._proxy_process.wait(timeout=5.0)
            logger.info("Proxy shutdown gracefully")
        except subprocess.TimeoutExpired:
            logger.warning("Proxy did not terminate gracefully, forcing kill")
            self._proxy_process.kill()
            self._proxy_process.wait()

        self._proxy_process = None


class EvalProxyManager(_ProxyManagerBase):
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
        from rllm.eval.config import get_provider_info

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
        return self._generate_litellm_config()

    def start_proxy_subprocess(self, config: dict[str, Any], **kwargs) -> str:
        if self._proxy_process is not None:
            logger.warning("Proxy subprocess already running")
            return ""

        snapshot_path = self._snapshot_config_to_file(config, directory=kwargs.get("snapshot_directory"))
        if not snapshot_path or not os.path.exists(snapshot_path):
            raise RuntimeError("Config snapshot not available. Cannot start proxy.")

        cmd = [
            sys.executable,
            "-m",
            "rllm.eval._litellm_server",
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
        stderr_output = self._read_stderr_tail()
        if stderr_output:
            logger.error("Proxy stderr output:\n%s", stderr_output)

    def shutdown_proxy(self) -> None:
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
