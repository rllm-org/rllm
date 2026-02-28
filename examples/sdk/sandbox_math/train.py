"""Train a math agent running inside a sandboxed environment.

Demonstrates the sandbox execution feature: the agent code lives in a
separate ``agent/`` directory, gets uploaded to a sandbox (local subprocess
by default, Docker with ``--backend docker``), and communicates results back
through the proxy's result store.

This is the sandbox counterpart of ``examples.sdk.adk_math``.  Instead of
running ``agent_run_func`` inside the trainer process, the trainer:

1. Starts the LiteLLM proxy with ``--enable-result-store``
2. Creates sandbox workers (local subprocesses or Docker containers)
3. Uploads ``agent/`` + ``worker_server.py`` into each sandbox
4. Dispatches tasks via HTTP to the worker server inside the sandbox
5. Workers call ``agent.rollout(task, config)`` → push results to the proxy
6. Trainer polls the result store and builds Episodes from traces + results

Prerequisites:
    1. Prepare the countdown dataset:
        python -m examples.countdown.prepare_countdown_data

Usage (local subprocess sandbox — no Docker required):
    python -m examples.sdk.sandbox_math.train \
        rllm/backend=tinker \
        model.name=Qwen/Qwen3-8B \
        training.group_size=8 \
        data.train_batch_size=4 \
        rllm.trainer.test_freq=5 \
        rllm.trainer.val_before_train=false

    The sandbox config defaults are in base.yaml and can be overridden:
        rllm.sdk.sandbox.backend=docker \
        rllm.sdk.sandbox.image=python:3.11-slim \
        rllm.sdk.sandbox.num_workers=16

Standalone sandbox test (without training, for quick validation):
    python -m examples.sdk.sandbox_math.train --test-sandbox
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

import hydra

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer

logger = logging.getLogger(__name__)

# Silence noisy loggers
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)

# Resolve the agent directory relative to this file
AGENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent")


# ---------------------------------------------------------------------------
# Standalone sandbox smoke test (no training infrastructure needed)
# ---------------------------------------------------------------------------


async def _test_sandbox() -> None:
    """Quick smoke test: create a local sandbox, upload agent, run one task.

    Spins up a minimal aiohttp server that accepts result submissions (the
    same endpoint the LiteLLM proxy exposes with ``--enable-result-store``)
    so we can validate the full fire-and-forget → push-to-store loop
    without needing the real proxy or an inference backend.
    """
    from rllm.sdk.sandbox import create_sandbox_orchestrator
    from rllm.sdk.sandbox.protocol import SandboxConfig
    from rllm.sdk.sandbox.result_store import ExecutionResultStore

    import tempfile

    from aiohttp import web

    print(f"Agent directory: {AGENT_DIR}")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_results.db")
        result_store = ExecutionResultStore(db_path=db_path)

        # --- Minimal result-accepting server (stands in for the real proxy) ---
        async def _handle_result_post(request: web.Request) -> web.Response:
            eid = request.match_info["execution_id"]
            data = await request.json()
            result_store.store_result(eid, data)
            return web.json_response({"status": "stored", "execution_id": eid})

        # Also handle chat completions with a dummy response so the agent
        # doesn't crash on the LLM call.
        async def _handle_chat(request: web.Request) -> web.Response:
            return web.json_response({
                "id": "test",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "<answer> 4 </answer>"},
                    "finish_reason": "stop",
                }],
                "model": "test",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            })

        app = web.Application()
        app.router.add_post("/rllm/results/{execution_id}", _handle_result_post)
        # Catch chat completion requests on any path containing /chat/completions
        app.router.add_post("/{path:.*chat/completions.*}", _handle_chat)

        runner = web.AppRunner(app)
        await runner.setup()
        test_port = 18999
        site = web.TCPSite(runner, "127.0.0.1", test_port)
        await site.start()
        proxy_url = f"http://127.0.0.1:{test_port}/v1"
        print(f"Test server listening on {proxy_url}")

        # --- Create orchestrator ---
        print("Creating local sandbox orchestrator...")
        config = SandboxConfig(
            enabled=True,
            backend="local",
            agent_dir=AGENT_DIR,
            agent_module="agent",
            agent_func="rollout",
            pool_mode="persistent",
            num_workers=1,
            worker_port=8199,
            install_rllm_sdk=False,  # Already installed in the local env
            execution_timeout=30.0,
        )

        orchestrator = create_sandbox_orchestrator(config)
        await orchestrator.initialize(proxy_url, result_store)

        # --- Dispatch a task ---
        task = {
            "question": "Using the numbers 2, 2, find a way to reach the target number 4.",
            "target": 4,
            "ground_truth": "4",
        }
        agent_config = {"model_id": "default"}

        print(f"Dispatching test task...")
        result = await orchestrator.execute(task, agent_config)

        print(f"Result: success={result.success}, session_uid={result.session_uid}")
        if result.success:
            n_trajs = len(result.trajectories or [])
            print(f"  trajectories: {n_trajs} returned")
            for i, traj in enumerate(result.trajectories or []):
                print(f"  traj[{i}]: name={traj.get('name')}, reward={traj.get('reward')}, steps={len(traj.get('steps', []))}")
        else:
            print(f"  error: {result.error}")

        # --- Cleanup ---
        await orchestrator.shutdown()
        await runner.cleanup()
        result_store.close()

    passed = result.success and result.trajectories is not None and len(result.trajectories) > 0
    if passed:
        reward = result.trajectories[0].get("reward", 0)
        print(f"\nSandbox smoke test PASSED (reward={reward})")
    else:
        print(f"\nSandbox smoke test FAILED")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Trainer entry point
# ---------------------------------------------------------------------------


@hydra.main(
    config_path="pkg://rllm.experimental.config",
    config_name="unified",
    version_base=None,
)
def main(config) -> None:
    # Inject sandbox config — use agent_dir relative to this script
    from omegaconf import OmegaConf

    sandbox_overrides = {
        "rllm": {
            "sdk": {
                "sandbox": {
                    "enabled": True,
                    "agent_dir": AGENT_DIR,
                    "pool_mode": "persistent",  # Reuse workers across tasks (stateless agent)
                }
            }
        }
    }
    config = OmegaConf.merge(config, OmegaConf.create(sandbox_overrides))

    train_dataset = DatasetRegistry.load_dataset("countdown", "train")
    test_dataset = DatasetRegistry.load_dataset("countdown", "test")

    assert train_dataset, "Train dataset not found. Run: python -m examples.countdown.prepare_countdown_data"
    assert test_dataset, "Test dataset not found. Run: python -m examples.countdown.prepare_countdown_data"

    trainer = AgentTrainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        backend="tinker",
        # No agent_run_func — sandbox handles execution
    )
    trainer.train()


if __name__ == "__main__":
    if "--test-sandbox" in sys.argv:
        sys.argv.remove("--test-sandbox")
        asyncio.run(_test_sandbox())
    else:
        main()
