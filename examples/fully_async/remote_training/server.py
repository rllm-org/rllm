"""
Remote Training Server
======================

Deploy this on a GPU cluster (Modal, RunPod, etc.).
It starts the rLLM training infrastructure and exposes an HTTP API
that remote agent code can connect to.

Usage (with the built-in fully-async config):

    python server.py \
        actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct-2507

You can pass any Hydra override on the command line, just like the
colocated training scripts:

    python server.py \
        actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct-2507 \
        trainer.n_gpus_per_node=8 \
        rollout.n_gpus_per_node=4 \
        +server.host=0.0.0.0 \
        +server.port=8000

The server will:
1. Wait for a client to connect and optionally send config overrides
2. Initialise SGLang inference servers, trainer, and parameter sync
3. Accept generation requests and trajectory submissions via HTTP
"""

import hydra
import ray
from omegaconf import DictConfig, OmegaConf

from rllm.experimental.fully_async.remote import TrainingServer


@hydra.main(
    config_path="pkg://rllm.experimental.fully_async.config",
    config_name="fully_async_ppo_trainer",
    version_base=None,
)
def main(config: DictConfig):
    # Allow +server.host and +server.port overrides (not part of training config)
    host = OmegaConf.select(config, "server.host", default="0.0.0.0")
    port = OmegaConf.select(config, "server.port", default=8000)

    # Remove the server key (if present) so it doesn't confuse training components
    if "server" in config:
        OmegaConf.set_struct(config, False)
        del config["server"]
        OmegaConf.set_struct(config, True)

    # Initialise Ray (connect to existing cluster or start local)
    if not ray.is_initialized():
        ray.init()

    # Config is already fully resolved by Hydra (including ppo_trainer defaults)
    server = TrainingServer(config=config)
    print(f"Starting rLLM Training Server on {host}:{port}")
    server.run(host=host, port=int(port))


if __name__ == "__main__":
    main()
