"""Lifecycle manager for the pinned MCP-Atlas Docker environment and harness."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import requests

from rllm.integrations.mcp_atlas.constants import GATEWAY_API_KEY, IMAGE, SOURCE_REVISION, UVX_MCP_VERSION
from rllm.integrations.mcp_atlas.source import ensure_harness_build, ensure_source, patch_sha256

logger = logging.getLogger(__name__)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _tail(path: Path, limit: int = 4000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-limit:]


class MCPAtlasServiceManager:
    def __init__(
        self,
        *,
        image: str = IMAGE,
        env_file: str | None = None,
        preflight: str = "strict",
        external_harness_url: str | None = None,
        startup_timeout: float = 180.0,
        health_timeout: float = 30.0,
        run_dir: str | Path | None = None,
        required_servers: set[str] | None = None,
    ) -> None:
        if preflight not in {"strict", "smoke"}:
            raise ValueError("MCP-Atlas preflight must be 'strict' or 'smoke'")
        self.image = image
        self.env_file = str(Path(env_file).expanduser()) if env_file else None
        self.preflight = preflight
        self.external_harness_url = external_harness_url.rstrip("/") if external_harness_url else None
        self.startup_timeout = startup_timeout
        self.health_timeout = health_timeout
        self.run_dir = Path(run_dir).expanduser() if run_dir else None
        self.required_servers = set(required_servers or ())
        self.harness_url: str | None = self.external_harness_url
        self.sandbox_url: str | None = None
        self.container_name: str | None = None
        self.image_digest: str | None = None
        self.server_health: dict[str, Any] = {}
        self._harness_process: subprocess.Popen | None = None
        self._harness_log = None
        self._docker_log = None
        self._tempdir: tempfile.TemporaryDirectory | None = None

    def _check_programs(self) -> None:
        required = [] if self.external_harness_url else ["git", "node", "npm", "docker"]
        missing = [program for program in required if shutil.which(program) is None]
        if missing:
            raise RuntimeError(f"MCP-Atlas requires these programs: {', '.join(missing)}")
        if self.env_file and not Path(self.env_file).is_file():
            raise FileNotFoundError(f"MCP-Atlas env file not found: {self.env_file}")

    def _wait_json(self, url: str, process: subprocess.Popen | None, log_path: Path) -> dict:
        deadline = time.monotonic() + self.startup_timeout
        last_error = "not ready"
        while time.monotonic() < deadline:
            if process is not None and process.poll() is not None:
                raise RuntimeError(f"MCP-Atlas service exited with code {process.returncode}: {_tail(log_path)}")
            try:
                response = requests.get(url, timeout=min(self.health_timeout, 10.0))
                if response.ok:
                    return response.json()
                last_error = f"HTTP {response.status_code}"
            except (requests.RequestException, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.5)
        raise TimeoutError(f"Timed out waiting for MCP-Atlas service at {url}: {last_error}; {_tail(log_path)}")

    def _render_container_env(self, source: Path) -> Path | None:
        values: dict[str, str] = {}
        if self.env_file:
            for raw in Path(self.env_file).read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip()

        template = (source / "env.template").read_text(encoding="utf-8")
        allowed = set(re.findall(r"^([A-Z][A-Z0-9_]*)=", template, flags=re.MULTILINE))
        values = {key: value for key, value in values.items() if key in allowed}
        for key in allowed:
            if key not in values and os.environ.get(key):
                values[key] = os.environ[key]
        if not values:
            return None
        assert self._tempdir is not None
        path = Path(self._tempdir.name) / "container.env"
        path.write_text("".join(f"{key}={value}\n" for key, value in sorted(values.items())), encoding="utf-8")
        path.chmod(0o600)
        return path

    def _render_uvx_compatibility(self) -> tuple[Path, Path]:
        """Keep the image's open-ended uvx dependencies on MCP SDK 1.x."""
        assert self._tempdir is not None
        tempdir = Path(self._tempdir.name)
        constraints = tempdir / "uvx-constraints.txt"
        constraints.write_text(f"mcp=={UVX_MCP_VERSION}\n", encoding="utf-8")
        wrapper = tempdir / "uvx"
        wrapper.write_text(
            '#!/bin/sh\nexec uv tool run --constraints /tmp/rllm-mcp-atlas-constraints.txt "$@"\n',
            encoding="utf-8",
        )
        wrapper.chmod(0o755)
        return constraints, wrapper

    def _ensure_image(self) -> None:
        inspect = subprocess.run(["docker", "image", "inspect", self.image], text=True, capture_output=True)
        if inspect.returncode != 0:
            subprocess.run(["docker", "pull", self.image], check=True)
        digest = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", self.image],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        try:
            digests = json.loads(digest)
            self.image_digest = digests[0] if digests else self.image
        except (json.JSONDecodeError, TypeError):
            self.image_digest = self.image

    def _official_health_check(self, source: Path) -> None:
        script = source / "services" / "mcp_eval" / "test_servers.py"
        spec = importlib.util.spec_from_file_location("rllm_mcp_atlas_test_servers", script)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load MCP-Atlas server health check: {script}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        module.BASE_URL = f"{self.sandbox_url}/call-tool"
        if self.env_file:
            module.ENV_PATH = Path(self.env_file)
        servers, _required_vars = module.load_servers()
        unknown = self.required_servers - set(servers)
        if unknown:
            raise RuntimeError(f"MCP-Atlas tasks reference unknown servers: {', '.join(sorted(unknown))}")
        missing_tests = self.required_servers - set(module.TEST_CALLS)
        if missing_tests:
            raise RuntimeError(f"MCP-Atlas has no official health probe for: {', '.join(sorted(missing_tests))}")
        if self.preflight == "smoke":
            keyed = sorted(server for server in self.required_servers if servers[server])
            if keyed:
                raise RuntimeError(f"MCP-Atlas smoke mode only permits no-key servers; selected: {', '.join(keyed)}")

        import asyncio

        passed: list[str] = []
        failed: list[str] = []
        for server in sorted(self.required_servers):
            try:
                asyncio.run(module.main(timeout=self.health_timeout, concurrency=1, only_server=server))
            except SystemExit as exc:
                failed.append(server)
                if self.preflight == "strict":
                    raise RuntimeError(f"MCP-Atlas official health probe failed for {server}") from exc
            else:
                passed.append(server)
        self.server_health["official_probes"] = passed
        self.server_health["official_probe_failures"] = failed

    def start(self) -> None:
        self._check_programs()
        if self.external_harness_url:
            log_path = Path(os.devnull)
            self._wait_json(f"{self.external_harness_url}/health", None, log_path)
            self.harness_url = self.external_harness_url
            return

        source = ensure_source()
        harness_dir = ensure_harness_build()
        self._ensure_image()
        self._tempdir = tempfile.TemporaryDirectory(prefix="rllm-mcp-atlas-")
        log_dir = self.run_dir or Path(self._tempdir.name)
        log_dir.mkdir(parents=True, exist_ok=True)
        docker_log_path = log_dir / "mcp_atlas_sandbox.log"
        harness_log_path = log_dir / "mcp_atlas_harness.log"
        self._docker_log = docker_log_path.open("a", encoding="utf-8")
        self._harness_log = harness_log_path.open("a", encoding="utf-8")

        sandbox_port = _free_port()
        harness_port = _free_port()
        self.sandbox_url = f"http://127.0.0.1:{sandbox_port}"
        self.harness_url = f"http://127.0.0.1:{harness_port}"
        self.container_name = f"rllm-mcp-atlas-{uuid.uuid4().hex[:12]}"

        docker_cmd = [
            "docker",
            "run",
            "--rm",
            "--name",
            self.container_name,
            "-p",
            f"127.0.0.1:{sandbox_port}:1984",
        ]
        rendered_env = self._render_container_env(source)
        if rendered_env is not None:
            docker_cmd += ["--env-file", str(rendered_env)]
        constraints, uvx_wrapper = self._render_uvx_compatibility()
        docker_cmd += [
            "-v",
            f"{constraints}:/tmp/rllm-mcp-atlas-constraints.txt:ro",
            "-v",
            f"{uvx_wrapper}:/usr/local/bin/uvx:ro",
        ]
        docker_cmd.append(self.image)
        docker_process = subprocess.Popen(docker_cmd, stdout=self._docker_log, stderr=subprocess.STDOUT, text=True)

        self._wait_json(f"{self.sandbox_url}/health", docker_process, docker_log_path)
        health = self._wait_json(f"{self.sandbox_url}/enabled-servers", docker_process, docker_log_path)
        self.server_health = health
        statuses = dict(health.get("servers") or [])
        offline = sorted(server for server in self.required_servers if statuses.get(server) != "OK")
        if offline:
            raise RuntimeError(f"MCP-Atlas selected tasks require offline servers: {', '.join(offline)}")
        self._official_health_check(source)

        env = os.environ.copy()
        env.update(
            {
                "PORT": str(harness_port),
                "MCP_SANDBOX_URL": self.sandbox_url,
                # The rLLM patch makes the request-level llm_base_url
                # authoritative; these satisfy upstream startup validation.
                "LLM_BASE_URL": "http://127.0.0.1:1",
                "LLM_API_KEY": GATEWAY_API_KEY,
                "LOG_LEVEL": env.get("MCP_ATLAS_LOG_LEVEL", "info"),
            }
        )
        self._harness_process = subprocess.Popen(
            ["node", "dist/index.js"],
            cwd=harness_dir,
            env=env,
            stdout=self._harness_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._wait_json(f"{self.harness_url}/health", self._harness_process, harness_log_path)

    def stop(self) -> None:
        if self._harness_process is not None:
            self._harness_process.terminate()
            try:
                self._harness_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._harness_process.kill()
            self._harness_process = None
        if self.container_name:
            subprocess.run(["docker", "stop", "--time", "10", self.container_name], text=True, capture_output=True)
            self.container_name = None
        for handle_name in ("_harness_log", "_docker_log"):
            handle = getattr(self, handle_name)
            if handle is not None:
                handle.close()
                setattr(self, handle_name, None)
        if self._tempdir is not None:
            self._tempdir.cleanup()
            self._tempdir = None

    def metadata(self) -> dict[str, Any]:
        return {
            "source_revision": SOURCE_REVISION,
            "session_base_url_patch_sha256": patch_sha256(),
            "image": self.image,
            "image_digest": self.image_digest,
            "preflight": self.preflight,
            "external_harness": bool(self.external_harness_url),
            "server_health": self.server_health,
            "required_servers": sorted(self.required_servers),
            "uvx_mcp_version": UVX_MCP_VERSION,
        }


__all__ = ["MCPAtlasServiceManager"]
