"""Sandbox orchestrator for managing sandbox lifecycle and task dispatch.

Supports two modes:
- **persistent**: Pool of long-lived workers reused across batches.
- **per_task**: Fresh sandbox per task, destroyed after completion.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

from rllm.sdk.sandbox.protocol import ExecutionResult, SandboxConfig

logger = logging.getLogger(__name__)

# Paths relative to the rllm package
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent  # rllm/
RUNNER_PATH = str(Path(__file__).resolve().parent / "worker_server.py")
RLLM_PACKAGE_DIR = str(_PACKAGE_ROOT.parent)  # the directory containing rllm/ and pyproject.toml


@dataclass
class Worker:
    """A running sandbox worker."""

    sandbox: Any  # Sandbox protocol instance
    name: str
    port: int


class SandboxOrchestrator:
    """Orchestrates sandbox lifecycle and task dispatch."""

    def __init__(self, sandbox_factory: Any, config: SandboxConfig):
        self._sandbox_factory = sandbox_factory
        self._config = config
        self._workers: list[Worker] = []
        self._available: asyncio.Queue[Worker] = asyncio.Queue()
        self._semaphore: asyncio.Semaphore | None = None
        self._proxy_url: str = ""
        self._result_store: Any = None
        self._initialized = False
        # Atomic counter for assigning unique ports in per-task mode
        self._port_counter: int = 0

    async def initialize(self, proxy_url: str, result_store: Any) -> None:
        """Initialize the orchestrator with proxy URL and result store.

        For persistent mode, creates and warms up all workers.
        For per-task mode, just stores config and creates a semaphore.
        """
        self._proxy_url = proxy_url
        self._result_store = result_store

        if self._config.pool_mode == "persistent":
            await self._init_persistent_pool()
        else:
            self._semaphore = asyncio.Semaphore(self._config.max_concurrent)

        self._initialized = True
        logger.info(
            "SandboxOrchestrator initialized (mode=%s, backend=%s)",
            self._config.pool_mode,
            self._config.backend,
        )

    async def _init_persistent_pool(self) -> None:
        """Create and warm up all workers for persistent mode."""
        for i in range(self._config.num_workers):
            port = self._config.worker_port + i
            worker = await self._create_worker(f"worker-{i}", port)
            self._workers.append(worker)
            self._available.put_nowait(worker)
        logger.info("Persistent pool ready: %d workers", len(self._workers))

    async def _create_worker(self, name: str, port: int) -> Worker:
        """Create one sandbox, upload code, install deps, start runner."""
        sandbox = self._sandbox_factory(
            name=name,
            image=self._config.image,
        )

        await self._setup_sandbox(sandbox)

        # Start the worker server
        cmd = (
            f"python /app/runner/worker_server.py "
            f"--agent-module {self._config.agent_module} "
            f"--agent-func {self._config.agent_func} "
            f"--port {port} "
            f"--agent-dir /app/agent"
        )
        sandbox.start_agent_process(cmd, port=port)
        return Worker(sandbox=sandbox, name=name, port=port)

    async def _setup_sandbox(self, sandbox: Any) -> None:
        """Upload agent code, runner, and install dependencies."""
        is_local = self._config.backend == "local"

        # Upload agent project
        if self._config.agent_dir:
            agent_dir = os.path.expanduser(self._config.agent_dir)
            sandbox.upload_dir(agent_dir, "/app/agent")

        # Upload the runner script
        sandbox.upload_file(RUNNER_PATH, "/app/runner/worker_server.py")

        # Install rllm[sdk] if requested (skip for local backend — packages
        # are already available in the parent Python environment)
        if self._config.install_rllm_sdk and not is_local:
            sandbox.upload_dir(RLLM_PACKAGE_DIR, "/app/rllm-pkg")
            sandbox.exec("pip install -e '/app/rllm-pkg[sdk]'")

        # Install agent requirements (skip for local — deps already installed)
        if self._config.requirements_file and not is_local:
            sandbox.exec(f"pip install -r /app/agent/{self._config.requirements_file}")

        # Run custom setup commands
        for cmd in self._config.setup_commands:
            sandbox.exec(cmd)

    async def execute(self, task: dict, agent_config: dict) -> ExecutionResult:
        """Dispatch a task to a sandbox worker and wait for the result."""
        if not self._initialized:
            raise RuntimeError("SandboxOrchestrator not initialized. Call initialize() first.")

        if self._config.pool_mode == "persistent":
            return await self._execute_persistent(task, agent_config)
        else:
            return await self._execute_per_task(task, agent_config)

    async def _execute_persistent(self, task: dict, agent_config: dict) -> ExecutionResult:
        """Pick an idle worker, send task, wait for result, return worker."""
        execution_id = str(uuid.uuid4())
        self._result_store.register(execution_id)

        worker = await self._available.get()
        try:
            url, headers = worker.sandbox.get_endpoint(worker.port)
            async with aiohttp.ClientSession() as session:
                await session.post(
                    f"{url}/execute",
                    json={
                        "execution_id": execution_id,
                        "proxy_url": self._proxy_url,
                        "task": task,
                        "agent_config": agent_config,
                    },
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                )

            # Run the blocking wait in a thread to avoid blocking the event loop
            # (the event loop must stay free to handle the worker's result POST)
            return await asyncio.to_thread(
                self._result_store.wait,
                execution_id,
                self._config.execution_timeout,
            )
        finally:
            self._available.put_nowait(worker)

    async def _execute_per_task(self, task: dict, agent_config: dict) -> ExecutionResult:
        """Create sandbox, run task, destroy sandbox."""
        execution_id = str(uuid.uuid4())
        self._result_store.register(execution_id)

        assert self._semaphore is not None
        async with self._semaphore:
            sandbox = None
            # Assign a unique port per task to avoid conflicts when
            # multiple per-task sandboxes run concurrently on local backend.
            port = self._config.worker_port + self._port_counter
            self._port_counter += 1
            try:
                sandbox = self._sandbox_factory(
                    name=f"task-{execution_id[:8]}",
                    image=self._config.image,
                )

                await self._setup_sandbox(sandbox)

                # Run task-specific setup commands (template variables from task dict)
                for cmd_template in self._config.task_setup_commands:
                    cmd = cmd_template.format(**task)
                    sandbox.exec(cmd, timeout=self._config.task_setup_timeout)

                # Start runner
                cmd = (
                    f"python /app/runner/worker_server.py "
                    f"--agent-module {self._config.agent_module} "
                    f"--agent-func {self._config.agent_func} "
                    f"--port {port} "
                    f"--agent-dir /app/agent"
                )
                sandbox.start_agent_process(cmd, port=port)

                # Send task
                url, headers = sandbox.get_endpoint(port)
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        f"{url}/execute",
                        json={
                            "execution_id": execution_id,
                            "proxy_url": self._proxy_url,
                            "task": task,
                            "agent_config": agent_config,
                        },
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=10),
                    )

                return await asyncio.to_thread(
                    self._result_store.wait,
                    execution_id,
                    self._config.execution_timeout,
                )
            finally:
                if sandbox and self._config.teardown_on_complete:
                    try:
                        sandbox.close()
                    except Exception:
                        logger.warning("Failed to close sandbox task-%s", execution_id[:8], exc_info=True)

    async def shutdown(self) -> None:
        """Stop all workers and destroy sandboxes."""
        for worker in self._workers:
            try:
                worker.sandbox.close()
            except Exception:
                logger.warning("Failed to close sandbox %s", worker.name, exc_info=True)
        self._workers.clear()

        # Drain the queue
        while not self._available.empty():
            try:
                self._available.get_nowait()
            except asyncio.QueueEmpty:
                break

        if self._result_store is not None:
            self._result_store.close()

        self._initialized = False
        logger.info("SandboxOrchestrator shut down")
