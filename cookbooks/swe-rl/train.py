"""Train an SWE agent on rllm-swesmith, validate on SWE-bench Verified.

This cookbook deliberately ships no custom AgentFlow or evaluator:

* The **agent** is the in-tree ``mini-swe-agent`` harness — a CLI scaffold
  installed inside each task's sandbox. The rLLM gateway intercepts every
  LLM call, so the trainer sees full trajectories without the harness
  knowing it's being trained.
* The **evaluator** is each task's own ``tests/test.sh`` (sandbox-shell
  verifier) — pytest for rllm-swesmith, the SWE-bench harness for the
  Verified split. The verifier writes a reward to
  ``/logs/verifier/reward.txt`` and rLLM reads it back.

So ``train.py`` only wires the two datasets into :class:`AgentTrainer`;
everything else is configured by Hydra overrides on the command line
(see ``train_tinker.sh`` / ``train_verl.sh`` for working defaults).

Usage (from rllm repo root)::

    python cookbooks/swe-rl/train.py rllm/backend=tinker
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from rllm.data.dataset import DatasetRegistry
from rllm.trainer import AgentTrainer

TRAIN_DATASET = "rllm-swesmith"
VAL_DATASET = "swebench-verified"


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="unified", version_base=None)
def main(config: DictConfig) -> None:
    train_dataset = DatasetRegistry.load_dataset(TRAIN_DATASET, "train")
    val_dataset = DatasetRegistry.load_dataset(VAL_DATASET, "test")

    if train_dataset is None:
        raise RuntimeError(f"Dataset '{TRAIN_DATASET}' not found. Run: rllm dataset pull {TRAIN_DATASET} (or: python cookbooks/swe-rl/prepare_data.py)")
    if val_dataset is None:
        raise RuntimeError(f"Dataset '{VAL_DATASET}' not found. Run: rllm dataset pull harbor:swebench-verified (or: python cookbooks/swe-rl/prepare_data.py)")

    trainer = AgentTrainer(
        backend=config.rllm.get("backend", "tinker"),
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
