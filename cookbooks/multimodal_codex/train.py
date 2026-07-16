"""Train multimodal_codex — Codex CLI + Qwen3.5-9B VLM via AgentTrainer.

Usage (from repo root):
    python cookbooks/multimodal_codex/train.py rllm.backend=verl

Requires a running gateway with ``RLLM_API_FORMAT=responses --cumulative-token-mode
--renderer-family qwen3.5``, plus a vLLM server behind it. See
``cookbooks/multimodal_codex/train_smoke.sh`` for the full pod bring-up.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

# Repo root on sys.path so ``cookbooks.multimodal_codex.*`` imports work when
# invoked as ``python cookbooks/multimodal_codex/train.py``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rllm.data.dataset import DatasetRegistry
from rllm.trainer import AgentTrainer

from cookbooks.multimodal_codex.harness import MultimodalCodexHarness
from cookbooks.multimodal_codex.multimodal_codex_eval import multimodal_codex_evaluator
from cookbooks.multimodal_codex.prepare_data import prepare_multimodal_codex_data


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    prepare_multimodal_codex_data()
    train_dataset = DatasetRegistry.load_dataset("multimodal_codex", "train")
    test_dataset = DatasetRegistry.load_dataset("multimodal_codex", "test")

    if train_dataset is None or test_dataset is None:
        raise RuntimeError("multimodal_codex splits not found after prepare_multimodal_codex_data()")

    sandbox_backend = config.get("sandbox_backend", "bwrap")

    trainer = AgentTrainer(
        backend=config.rllm.get("backend", "tinker"),
        agent_flow=MultimodalCodexHarness(),
        evaluator=multimodal_codex_evaluator,
        sandbox_backend=sandbox_backend,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
