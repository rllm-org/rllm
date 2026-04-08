"""Eval through RemoteAgentFlowEngine — exercises the same path the trainer uses.

Stands up the gateway in-process, registers a vLLM worker, builds a HarborRuntime,
wraps it in a RemoteAgentFlowEngine, and runs N tasks. The engine creates a
gateway session per task, hands the per-session URL to the agent, collects the
gateway-side traces, and returns rllm Episodes built from traces + verifier
rewards.

Spin up vLLM first (e.g. on port 30000), then run:
    python examples/harbor-swe/eval_with_engine.py
"""

import asyncio
import shlex
import sys

from omegaconf import OmegaConf

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.engine.gateway_manager import GatewayManager
from rllm.experimental.engine.remote_agent_flow_engine import RemoteAgentFlowEngine
from rllm.experimental.engine.remote_runtime import (
    RemoteRuntimeConfig,
    create_remote_runtime,
)


# --- podman rootless workaround (POC-local; NOT in the runtime) -------------
def _apply_podman_upload_patch() -> None:
    from harbor.environments.docker.docker import (
        DockerEnvironment,
        _sanitize_docker_compose_project_name,
    )

    async def _upload_dir_tar_pipe(self, source_dir, target_dir):
        source = shlex.quote(str(source_dir))
        target = shlex.quote(str(target_dir))
        project_name = _sanitize_docker_compose_project_name(self.session_id)
        proj_dir = shlex.quote(str(self.environment_dir.resolve().absolute()))
        compose_files = " ".join(f"-f {shlex.quote(str(p.resolve().absolute()))}" for p in self._docker_compose_paths)
        base = f"docker compose --project-name {project_name} --project-directory {proj_dir} {compose_files}"
        mkdir_cmd = f"{base} exec -T main mkdir -p {target}"
        pipe_cmd = f"tar -C {source} -cf - . | {base} exec -T main tar --no-same-owner -xf - -C {target}"
        env = self._env_vars.to_env_dict(include_os_env=True)
        for cmd in (mkdir_cmd, pipe_cmd):
            proc = await asyncio.create_subprocess_shell(cmd, env=env)
            rc = await proc.wait()
            if rc != 0:
                raise RuntimeError(f"patched upload_dir failed (rc={rc}): {cmd}")

    DockerEnvironment.upload_dir = _upload_dir_tar_pipe


_apply_podman_upload_patch()
# ----------------------------------------------------------------------------


# --- config -----------------------------------------------------------------
DATASET = "swesmith_harbor"
SPLIT = "train"
LIMIT = 16
N_PARALLEL = 8
GATEWAY_PORT = 9090
VLLM_URL = "http://localhost:30000"
MODEL_NAME = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
AGENT = "qwen-coder"
ENVIRONMENT_TYPE = "docker"
SESSION_TIMEOUT = 1800.0
# ----------------------------------------------------------------------------


async def main() -> int:
    # Build a minimal config so GatewayManager has somewhere to read from.
    config = OmegaConf.create(
        {
            "rllm": {
                "gateway": {
                    "port": GATEWAY_PORT,
                    "sampling_params_priority": "session",
                }
            },
            # GatewayManager pulls the served model name from here and pins it
            # via the gateway's body-rewrite, so harbor scaffolds can ship any
            # placeholder and vLLM still gets the right model.
            "model": {"name": MODEL_NAME},
        }
    )
    gateway = GatewayManager(config, mode="thread")

    # Bypass GatewayManager.start(rollout_engine) since we don't have a verl
    # rollout engine — we register the standalone vLLM worker manually.
    gateway._start_thread()
    gateway.client.add_worker(url=VLLM_URL)

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

        episodes = await engine.execute_tasks(tasks=tasks, task_ids=task_ids, is_validation=False, show_progress=True)
    finally:
        runtime.shutdown()
        gateway.stop()

    n_total = len(episodes)
    n_correct = sum(1 for ep in episodes if ep.is_correct)
    print(f"Accuracy: {n_correct / n_total:.3f}  ({n_correct}/{n_total})")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
