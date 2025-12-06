"""LiteLLM proxy manager for VERL rollout engines.

This module provides utilities to:
1. Extract vLLM server addresses from VERL rollout engines
2. Configure LiteLLM proxy with multiple vLLM backends for load balancing
3. Provide a unified OpenAI-compatible endpoint with metadata routing
"""

from __future__ import annotations

import atexit
import logging
import os
import resource
import subprocess
import sys
import time
from typing import Any

import aiohttp
import requests
import yaml

logger = logging.getLogger(__name__)


class ProxyManager:
    """Generic LiteLLM proxy manager with shared lifecycle helpers."""

    def __init__(
        self,
        proxy_host: str = "127.0.0.1",
        proxy_port: int = 4000,
        admin_token: str | None = None,
        proxy_access_log: bool = False,
    ) -> None:
        self.proxy_host = proxy_host
        self.proxy_port = proxy_port
        self.proxy_access_log = proxy_access_log
        self.admin_token = admin_token
        self._proxy_process: subprocess.Popen | None = None

    async def flush_tracer(self, timeout: float = 30.0) -> bool:
        """Ask LiteLLM proxy to flush the tracer queue."""

        url = f"http://{self.proxy_host}:{self.proxy_port}/admin/flush-tracer"
        headers = {"Content-Type": "application/json"}
        if self.admin_token:
            headers["Authorization"] = f"Bearer {self.admin_token}"

        try:
            request_timeout = aiohttp.ClientTimeout(total=timeout + 5.0)
            async with aiohttp.ClientSession(timeout=request_timeout) as session:
                async with session.post(url, json={"timeout": timeout}, headers=headers) as resp:
                    resp.raise_for_status()
                    result = await resp.json()
                    logger.info("Tracer flush succeeded: %s", result)
            return True
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Failed to flush tracer via proxy: %s", exc)
            return False

    def _snapshot_config_to_file(self, config: dict[str, Any], directory: str | None = None) -> str | None:
        """Persist the provided config to a readable path for debugging."""
        base_dir = directory or os.getenv("RLLM_PROXY_CONFIG_DIR") or os.getcwd()
        os.makedirs(base_dir, exist_ok=True)
        snapshot_path = os.path.join(base_dir, "litellm_proxy_config_autogen.yaml")
        with open(snapshot_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info("📄 LiteLLM config snapshot written to %s", snapshot_path)
        return snapshot_path

    def reload_proxy_config(
        self,
        config: dict[str, Any],
        reload_url: str | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Ask a LiteLLM proxy endpoint to reload the given configuration."""

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
        except requests.exceptions.RequestException as exc:  # pragma: no cover - network dependent
            raise RuntimeError(f"Failed to reload LiteLLM proxy via {url}: {exc}") from exc

        try:
            return resp.json()
        except ValueError:
            return {"status": "ok", "raw": resp.text}

    def get_proxy_url(self, include_v1: bool = True) -> str:
        """Get the unified proxy endpoint URL."""

        base = f"http://{self.proxy_host}:{self.proxy_port}"
        return f"{base}/v1" if include_v1 else base

    def start_proxy_subprocess(
        self,
        config: dict[str, Any],
        db_path: str | None = None,
        project: str | None = None,
        snapshot_directory: str | None = None,
        sync_tracer: bool = False,
        add_logprobs: bool = False,
        add_return_token_ids: bool = False,
    ) -> str:
        """Start LiteLLM proxy as subprocess (no GIL contention).

        Args:
            config: LiteLLM configuration dict.
            db_path: Path to SQLite database file.
            project: Project name/namespace for the tracer.
            snapshot_directory: Directory to save config snapshot.
            sync_tracer: If True, enable synchronous tracer persistence (waits for traces to be stored before returning response).
            add_logprobs: If True, automatically add logprobs=True to requests.

        Returns:
            Path to the config snapshot on disk.
        """

        if self._proxy_process is not None:
            logger.warning("Proxy subprocess already running")
            return ""

        snapshot_path = self._snapshot_config_to_file(config, directory=snapshot_directory)
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

        if self.admin_token:
            cmd.extend(["--admin-token", self.admin_token])

        if db_path:
            cmd.extend(["--db-path", db_path])
        if project:
            cmd.extend(["--project", project])

        if sync_tracer:
            cmd.extend(["--sync-tracer"])

        if add_logprobs:
            cmd.extend(["--add-logprobs"])

        if add_return_token_ids:
            cmd.extend(["--add-return-token-ids"])

        env = os.environ.copy()
        env["AIOHTTP_CONNECTOR_LIMIT"] = "4096"
        env["AIOHTTP_KEEPALIVE_TIMEOUT"] = "60"
        # Ensure the proxy subprocess has a working CA bundle for outbound HTTPS
        # (fixes SSLCertVerificationError on hosts with incomplete system CAs)
        try:
            import certifi  # type: ignore

            ca_path = certifi.where()
            env["SSL_CERT_FILE"] = ca_path
            env["REQUESTS_CA_BUNDLE"] = ca_path
            env["CURL_CA_BUNDLE"] = ca_path
            env["OPENAI_CA_BUNDLE"] = ca_path
        except Exception:
            # Best-effort; if certifi is unavailable, rely on system CA store
            pass

        def set_limits() -> None:
            """Set resource limits for the proxy subprocess."""

            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))
            except (ValueError, OSError) as e:  # pragma: no cover - platform dependent
                logger.warning("Could not set file descriptor limit: %s", e)

        logger.info("Starting proxy subprocess: %s", " ".join(cmd))
        self._proxy_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            preexec_fn=set_limits,
        )

        try:
            self._wait_for_server_start(timeout=30.0)
            logger.info("Proxy server started, sending configuration...")
            self.reload_proxy_config(config=config)
            logger.info("Proxy configuration loaded successfully")
        except Exception:
            self.shutdown_proxy()
            raise

        atexit.register(self.shutdown_proxy)
        logger.info("✅ Proxy subprocess ready (PID: %s)", self._proxy_process.pid)
        return snapshot_path

    def _wait_for_server_start(self, timeout: float = 10.0) -> None:
        """Wait for proxy server process to start accepting connections."""

        if self._proxy_process is None:
            raise RuntimeError("Proxy process not started")

        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._proxy_process.poll() is not None:
                exit_code = self._proxy_process.returncode
                raise RuntimeError(f"Proxy process died during startup with exit code {exit_code}")

            try:
                requests.get(f"http://{self.proxy_host}:{self.proxy_port}/", timeout=0.5)
                logger.info("Proxy server accepting connections")
                return
            except requests.RequestException:
                pass

            time.sleep(0.3)

        raise TimeoutError(f"Proxy server did not start within {timeout}s")

    def shutdown_proxy(self) -> None:
        """Gracefully shutdown proxy subprocess."""

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

    def __repr__(self) -> str:
        mode = "subprocess" if self._proxy_process else "external"
        return f"{self.__class__.__name__}(proxy={self.get_proxy_url()}, mode={mode})"


class VerlProxyManager(ProxyManager):
    """Manages LiteLLM proxy configuration for VERL rollout engines.

    This class:
    - Extracts all vLLM server addresses from VERL's AgentLoopManager
    - Generates LiteLLM config with load balancing across all replicas
    - Provides the unified proxy endpoint URL for OpenAI clients

    Example:
        ```python
        # Create VERL engine
        verl_engine = VerlEngine(config, rollout_manager, tokenizer)

        # Setup proxy manager
        proxy_mgr = VerlProxyManager(
            rollout_engine=verl_engine,
            model_name="Qwen/Qwen2.5-7B-Instruct",
            proxy_port=4000
        )

        # Get the unified endpoint
        base_url = proxy_mgr.get_proxy_url()  # http://localhost:4000/v1

        # Use with OpenAI client
        from openai import AsyncOpenAI
        client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
        ```
    """

    def __init__(
        self,
        rollout_engine,
        model_name: str,
        proxy_host: str = "127.0.0.1",
        proxy_port: int = 4000,
        admin_token: str | None = None,
        auto_instrument_vllm: bool = True,
        proxy_access_log: bool = False,
        add_logprobs: bool = False,
    ):
        """Initialize the proxy manager for a VERL rollout engine.

        Args:
            rollout_engine: The VERL rollout engine instance.
            model_name: Model name for the proxy configuration.
            proxy_host: Host address for the proxy server.
            proxy_port: Port number for the proxy server.
            admin_token: Optional admin token for proxy authentication.
            auto_instrument_vllm: If True, automatically instrument vLLM servers.
            proxy_access_log: If True, enable proxy access logging.
            add_logprobs: If True, automatically add logprobs=True to requests.
        """

        if type(rollout_engine).__name__ != "VerlEngine":
            raise TypeError(f"VerlProxyManager only supports VerlEngine, got {type(rollout_engine).__name__}")

        super().__init__(
            proxy_host=proxy_host,
            proxy_port=proxy_port,
            admin_token=admin_token,
            proxy_access_log=proxy_access_log,
        )

        self.model_name = model_name
        self.rollout_engine = rollout_engine
        self.auto_instrument_vllm = auto_instrument_vllm
        self.add_logprobs = add_logprobs
        self._server_addresses: list[str] = []

        if auto_instrument_vllm:
            self._instrument_vllm_servers()

        self._server_addresses = self._extract_server_addresses()
        logger.info("Extracted %d vLLM server addresses from VERL", len(self._server_addresses))

        self._snapshot_config_to_file(self._generate_litellm_config())

    def _instrument_vllm_servers(self) -> None:
        """Instrument vLLM servers to return token IDs.

        WARNING: This method cannot instrument already-running VERL servers!

        VERL servers run in separate Ray worker processes, and monkey patches
        applied in the main process do NOT propagate to Ray workers.

        To enable token IDs for vLLM < 0.10.2, you must instrument BEFORE
        creating the AgentLoopManager. See docs/howto/instrument_verl_vllm_for_token_ids.md

        This method only logs a warning if instrumentation is needed but servers
        are already running.
        """
        try:
            from rllm.patches.vllm_instrumentation import check_vllm_instrumentation_status, get_vllm_token_ids_support

            support = get_vllm_token_ids_support()
            print(f"[PROXY_MANAGER] vLLM token IDs support: {support}")

            # Get detailed status for debugging
            status = check_vllm_instrumentation_status()
            print(f"[PROXY_MANAGER] Detailed instrumentation status: {status}")

            if support == "none":
                logger.warning("vLLM < 0.10.2 detected, but VERL servers are already running. Token IDs will NOT be available! To enable token IDs, call instrument_vllm() BEFORE creating AgentLoopManager. See docs/howto/instrument_verl_vllm_for_token_ids.md for details.")
            elif support == "native":
                logger.info("vLLM >= 0.10.2 detected, token IDs available via native support")
            elif support == "instrumented":
                logger.info("vLLM already instrumented, token IDs should be available")
            else:
                logger.debug("vLLM not available in main process (expected for Ray workers)")
        except Exception as e:
            logger.debug(f"Could not check vLLM instrumentation status: {e}")

    def _extract_server_addresses(self) -> list[str]:
        """Extract all vLLM server addresses from VERL's AgentLoopManager."""
        server_addresses = self.rollout_engine.rollout_manager.server_addresses
        return server_addresses

    def _generate_litellm_config(self) -> dict[str, Any]:
        """Generate LiteLLM configuration with all vLLM replicas.

        Creates a model_list with one entry per vLLM replica for load balancing.
        LiteLLM will automatically round-robin across all entries with the same model_name.
        """
        model_list = []

        for idx, server_address in enumerate(self._server_addresses):
            # Each replica gets its own entry in the model list
            # LiteLLM will load balance across all entries with the same model_name
            # SamplingParametersCallback will detect vLLM from litellm_params (hosted_vllm prefix)
            model_list.append(
                {
                    "model_name": self.model_name,
                    "litellm_params": {
                        "model": f"hosted_vllm/{self.model_name}",
                        "api_base": f"http://{server_address}/v1",
                        "drop_params": True,
                    },
                    # Optional: Add replica identifier for debugging
                    "model_info": {
                        "id": f"verl-replica-{idx}",
                        "replica_rank": idx,
                    },
                }
            )

        config = {
            "model_list": model_list,
            "litellm_settings": {
                "drop_params": True,
                "num_retries": 3,
                # Enable load balancing across replicas
                "routing_strategy": "simple-shuffle",
                # # Cooldown policy - back    # for t
                # "allowed_fails": 5,  # Allow 5 failures per minute before cooldown
                # "cooldown_time": 10,  # Cooldown for 10 seconds, then auto-retry
            },
        }

        return config

    def get_server_addresses(self) -> list[str]:
        """Return the list of vLLM server addresses."""
        return self._server_addresses.copy()

    def build_proxy_config(self) -> dict[str, Any]:
        """Return a fresh LiteLLM configuration for the current VERL servers."""

        return self._generate_litellm_config()

    def __repr__(self) -> str:
        mode = "subprocess" if self._proxy_process else "external"
        return f"VerlProxyManager(model={self.model_name}, replicas={len(self._server_addresses)}, proxy={self.get_proxy_url()}, mode={mode})"
