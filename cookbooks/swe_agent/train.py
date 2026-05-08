"""Train mini-swe-agent on swesmith, validate on swebench-verified.

Tinker (single-machine LoRA) by default. Override ``rllm/backend=verl``
on the CLI for distributed verl training.

The Python-API equivalent of::

    rllm train harbor:swesmith --agent mini-swe-agent \\
        --val-dataset swebench-verified \\
        --model Qwen/Qwen3-30B-A3B \\
        --batch-size 2 --group-size 4

``AgentTrainer`` auto-detects the SandboxedAgentFlow harness and wires
:class:`rllm.hooks.SandboxTaskHooks` (per-task sandbox lifecycle +
verifier resolution from each harbor task's ``tests/test.sh``) plus
gateway loopback (so docker containers can reach the gateway via the
harness's ``host.docker.internal`` rewrite). The dataset's ``as_tasks=True``
flag wraps each harbor row as a Task rooted at its ``task_path`` so the
hook can find the right verifier.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from rllm.data.dataset import DatasetRegistry
from rllm.eval.agent_loader import load_agent
from rllm.experimental.unified_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    train_dataset = DatasetRegistry.load_dataset("swesmith", "default", as_tasks=True)
    val_dataset = DatasetRegistry.load_dataset("swebench-verified", "default", as_tasks=True)

    if train_dataset is None:
        raise RuntimeError("swesmith not found. Run: rllm dataset pull harbor:swesmith")
    if val_dataset is None:
        raise RuntimeError("swebench-verified not found. Run: rllm dataset pull harbor:swebench-verified")

    AgentTrainer(
        backend=config.rllm.get("backend", "tinker"),
        agent_flow=load_agent("mini-swe-agent"),
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    ).train()


if __name__ == "__main__":
    main()
