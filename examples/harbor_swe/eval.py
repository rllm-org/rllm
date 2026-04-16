"""Eval through RemoteAgentFlowEngine with Tinker backend.

Stands up a TinkerEngine + gateway in-process, builds a HarborRuntime,
wraps it in a RemoteAgentFlowEngine, and runs N tasks. The engine creates a
gateway session per task, hands the per-session URL to the agent, collects the
gateway-side traces, and returns rllm Episodes built from traces + verifier
rewards.

Usage:
    python examples/harbor-swe/eval.py
"""

import asyncio
import logging
import sys

import tinker
from omegaconf import OmegaConf
from tinker_cookbook.tokenizer_utils import get_tokenizer

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.engine.gateway_manager import GatewayManager
from rllm.experimental.engine.remote_agent_flow_engine import RemoteAgentFlowEngine
from rllm.experimental.engine.remote_runtime import (
    RemoteRuntimeConfig,
    create_remote_runtime,
)
from rllm.experimental.rollout.tinker_engine import TinkerEngine

logging.basicConfig(level=logging.INFO)

# --- config -----------------------------------------------------------------
DATASET = "swebench_verified_harbor"
SPLIT = "test"
LIMIT = 16
N_PARALLEL = 128
GATEWAY_PORT = 9090
PUBLIC_URL = None
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
AGENT = "mini-swe-agent"
ENVIRONMENT_TYPE = "daytona"
SESSION_TIMEOUT = 1800.0
# ----------------------------------------------------------------------------


def create_tinker_engine(model_name: str) -> TinkerEngine:
    tokenizer = get_tokenizer(model_name)
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=model_name)
    engine = TinkerEngine(
        base_url="",
        model_name=model_name,
        tokenizer=tokenizer,
        service_client=service_client,
        max_prompt_length=32768,
        max_response_length=4096,
        max_model_length=32768,
        sampling_params={
            "train": {"temperature": 1.0, "top_p": 1.0},
            "val": {"temperature": 0.7, "top_p": 0.8},
        },
        bypass_render_with_parser=False,
        renderer_name="qwen3_instruct",
    )
    engine.set_sampling_client(sampling_client)
    return engine


async def main() -> int:
    config = OmegaConf.create(
        {
            "rllm": {
                "gateway": {
                    "port": GATEWAY_PORT,
                    "public_url": PUBLIC_URL,
                    "sampling_params_priority": "session",
                }
            },
            "model": {"name": MODEL_NAME},
        }
    )
    gateway = GatewayManager(config, mode="thread")

    tinker_engine = create_tinker_engine(MODEL_NAME)
    gateway.start(tinker_engine)

    runtime_cfg = RemoteRuntimeConfig(
        enabled=True,
        backend="harbor",
        harbor={
            "agent": AGENT,
            "environment_type": ENVIRONMENT_TYPE,
        },
        session_timeout=SESSION_TIMEOUT,
    )
    runtime = create_remote_runtime(runtime_cfg, exp_id="harbor-swesmith-eval")
    runtime.initialize()

    engine = RemoteAgentFlowEngine(
        runtime=runtime,
        gateway=gateway,
        session_timeout=SESSION_TIMEOUT,
        n_parallel_tasks=N_PARALLEL,
    )

    try:
        ds = DatasetRegistry.load_dataset(DATASET, SPLIT)
        if ds is None:
            raise RuntimeError(f"Dataset {DATASET}/{SPLIT} not found. Run prepare_data.py first.")
        rows = [ds[i] for i in range(min(LIMIT, len(ds)))]
        tasks = [{"task_path": r["task_path"], "id": r["id"]} for r in rows]
        task_ids = [r["id"] for r in rows]

        episodes = await engine.execute_tasks(tasks=tasks, task_ids=task_ids, is_validation=True)
    finally:
        runtime.shutdown()
        gateway.stop()

    n_total = len(episodes)
    n_correct = sum(1 for ep in episodes if ep.is_correct)
    print(f"Accuracy: {n_correct / n_total:.3f}  ({n_correct}/{n_total})")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
