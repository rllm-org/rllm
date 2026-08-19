"""Mini-SWE terminal-RL entrypoint used by W&B run z6xi3xkh.

This restores the run's original ``MiniSweAgentHarness`` path on top of the
current ``terminal-rl`` branch.  The normal cookbook now uses Terminus-2, which
would produce a different request stream and therefore is not suitable for the
raw-TraceRecord/TraceGraph comparison.
"""

from __future__ import annotations

import os

import hydra
from omegaconf import DictConfig

from rllm.data.dataset import DatasetRegistry
from rllm.harnesses.mini_swe_agent import MiniSweAgentHarness
from rllm.trainer import AgentTrainer

TRAIN_DATASET = os.environ.get("TB_TRAIN_DATASET", "tb-v2-debug")

# Terminal-Bench eval version (Harbor registry). Must match prepare_data.py;
# both read TB_EVAL_VERSION so the pulled and loaded dataset names agree.
EVAL_VERSION = os.environ.get("TB_EVAL_VERSION", "2.0")
VAL_DATASET = f"terminal-bench@{EVAL_VERSION}"

# Sandbox backend for the SandboxedAgentFlow path: docker | local | modal | daytona.
SANDBOX_BACKEND = os.environ.get("TERMINAL_SANDBOX_BACKEND", "modal")

# Optional cap on the validation set size. Terminal-Bench 2.0 is 89 tasks;
# validation runs ALL of them every time it fires, which is slow. Set
# TB_VAL_MAX=N to validate on the first N tasks instead (0/unset = all).
TB_VAL_MAX = int(os.environ.get("TB_VAL_MAX", "0"))

# These values match the source launcher at d501e357. They are passed into
# mini-SWE's own config, so they affect the raw request stream rather than just
# host-side bookkeeping.
MINISWE_MAX_TURNS = int(os.environ.get("MINISWE_MAX_TURNS", "64"))
if MINISWE_MAX_TURNS <= 0:
    raise ValueError(f"MINISWE_MAX_TURNS must be positive, got {MINISWE_MAX_TURNS}")

MINISWE_MAX_CONSECUTIVE_FORMAT_ERRORS = int(os.environ.get("MINISWE_MAX_CONSECUTIVE_FORMAT_ERRORS", "1"))
if MINISWE_MAX_CONSECUTIVE_FORMAT_ERRORS < 0:
    raise ValueError(f"MINISWE_MAX_CONSECUTIVE_FORMAT_ERRORS must be non-negative, got {MINISWE_MAX_CONSECUTIVE_FORMAT_ERRORS}")

MINISWE_COMMAND_TIMEOUT = int(os.environ.get("MINISWE_COMMAND_TIMEOUT", "300"))
if MINISWE_COMMAND_TIMEOUT <= 0:
    raise ValueError(f"MINISWE_COMMAND_TIMEOUT must be positive, got {MINISWE_COMMAND_TIMEOUT}")


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="unified", version_base=None)
def main(config: DictConfig) -> None:
    train_dataset = DatasetRegistry.load_dataset(TRAIN_DATASET, "train")
    val_dataset = DatasetRegistry.load_dataset(VAL_DATASET, "default")

    if train_dataset is None:
        raise RuntimeError(f"Dataset '{TRAIN_DATASET}' not found. Run: rllm dataset pull {TRAIN_DATASET}")
    if val_dataset is None:
        raise RuntimeError(f"Dataset '{VAL_DATASET}' not found. Run: rllm dataset pull harbor:{VAL_DATASET} (or: python cookbooks/terminal-rl/prepare_data.py)")

    if TB_VAL_MAX > 0 and TB_VAL_MAX < len(val_dataset):
        val_dataset = val_dataset.select(range(TB_VAL_MAX))

    agent_flow = MiniSweAgentHarness(
        sandbox_backend=SANDBOX_BACKEND,
        max_turns=MINISWE_MAX_TURNS,
        max_consecutive_format_errors=MINISWE_MAX_CONSECUTIVE_FORMAT_ERRORS,
        command_timeout=MINISWE_COMMAND_TIMEOUT,
        capture_exit_status=True,
        cost_limit=0.0,
    )

    trainer = AgentTrainer(
        backend=config.rllm.get("backend", "tinker"),
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        agent_flow=agent_flow,
        sandbox_backend=SANDBOX_BACKEND,
    )
    trainer.train()


if __name__ == "__main__":
    main()
