"""Train a terminal agent on local tasks and evaluate on Terminal-Bench.

This cookbook deliberately ships no custom AgentFlow or evaluator:

* The **agent** is the in-tree ``mini-swe-agent`` harness
  (:class:`rllm.harnesses.mini_swe_agent.MiniSweAgentHarness`) — a
  :class:`~rllm.sandbox.sandboxed_flow.SandboxedAgentFlow` that runs the
  Mini-SWE CLI inside each task's sandbox. The rLLM gateway
  intercepts every LLM call, so the trainer sees full trajectories without the
  harness knowing it's being trained.
* The **evaluator** is each task's own verifier (sandbox-shell), resolved
  per-task by :class:`rllm.hooks.SandboxTaskHooks`. Each task ships a
  ``tests/test.sh`` that writes ``1.0``/``0.0`` to
  ``/logs/verifier/reward.txt``; rLLM reads that back as the RL reward.

Because we pass an ``agent_flow`` (and no explicit ``evaluator``/``hooks``),
:class:`AgentTrainer` runs the **rLLM-native SandboxedAgentFlow path**
(``AgentFlowEngine``) — sandboxes are created locally by ``SandboxTaskHooks``
via a pluggable ``sandbox_backend`` (``docker`` | ``local`` | ``modal`` |
``daytona``). This is NOT the remote-runtime / ``RemoteAgentFlowEngine`` path.

The sandbox backend is selected by the ``TERMINAL_SANDBOX_BACKEND`` env var
(default ``modal``). For ``modal`` install ``pip install modal`` and run
``modal token new``; for ``daytona`` install ``pip install daytona`` and set
``DAYTONA_API_KEY``. Everything else is configured by Hydra overrides on the
command line (see ``train_tinker.sh`` / ``train_verl.sh`` for working defaults).

Usage (from rllm repo root)::

    TERMINAL_SANDBOX_BACKEND=modal python cookbooks/terminal-rl/train.py rllm/backend=verl
"""

from __future__ import annotations

import os

import hydra
from omegaconf import DictConfig

from rllm.data.dataset import DatasetRegistry
from rllm.harnesses.mini_swe_agent import MiniSweAgentHarness
from rllm.trainer import AgentTrainer

TRAIN_DATASET = os.environ.get("TB_TRAIN_DATASET", "tb-opus-pass")
EVAL_VERSION = os.environ.get("TB_EVAL_VERSION", "2.0")
VAL_DATASET = f"terminal-bench@{EVAL_VERSION}"

# Sandbox backend for the SandboxedAgentFlow path: docker | local | modal | daytona.
SANDBOX_BACKEND = os.environ.get("TERMINAL_SANDBOX_BACKEND", "modal")

# Set TB_VAL_MAX=N to cap validation; 0 or unset evaluates the full split.
TB_VAL_MAX = int(os.environ.get("TB_VAL_MAX", "0"))

# Per-rollout Mini-SWE model-call cap. Each step is one agent turn.
MINISWE_MAX_TURNS = int(os.environ.get("MINISWE_MAX_TURNS", "64"))
if MINISWE_MAX_TURNS <= 0:
    raise ValueError(f"MINISWE_MAX_TURNS must be positive, got {MINISWE_MAX_TURNS}")

MINISWE_MAX_CONSECUTIVE_FORMAT_ERRORS = int(os.environ.get("MINISWE_MAX_CONSECUTIVE_FORMAT_ERRORS", "1"))
if MINISWE_MAX_CONSECUTIVE_FORMAT_ERRORS < 0:
    raise ValueError(
        "MINISWE_MAX_CONSECUTIVE_FORMAT_ERRORS must be non-negative, "
        f"got {MINISWE_MAX_CONSECUTIVE_FORMAT_ERRORS}"
    )

MINISWE_COMMAND_TIMEOUT = int(os.environ.get("MINISWE_COMMAND_TIMEOUT", "300"))
if MINISWE_COMMAND_TIMEOUT <= 0:
    raise ValueError(f"MINISWE_COMMAND_TIMEOUT must be positive, got {MINISWE_COMMAND_TIMEOUT}")


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="unified", version_base=None)
def main(config: DictConfig) -> None:
    train_dataset = DatasetRegistry.load_dataset(TRAIN_DATASET, "train")
    val_dataset = DatasetRegistry.load_dataset(VAL_DATASET, "default")

    if train_dataset is None:
        raise RuntimeError(f"Dataset '{TRAIN_DATASET}' not found. Run: python cookbooks/terminal-rl/prepare_data.py")
    if val_dataset is None:
        raise RuntimeError(
            f"Dataset '{VAL_DATASET}' not found. "
            f"Run: rllm dataset pull harbor:{VAL_DATASET} "
            "(or: python cookbooks/terminal-rl/prepare_data.py)"
        )
    if TB_VAL_MAX > 0 and TB_VAL_MAX < len(val_dataset):
        val_dataset = val_dataset.select(range(TB_VAL_MAX))

    # Mini-SWE as a SandboxedAgentFlow. Passing ``agent_flow`` (with no
    # explicit evaluator/hooks) makes AgentTrainer auto-wire SandboxTaskHooks
    # for the sandbox lifecycle + per-task verifier, and route rollouts through
    # AgentFlowEngine — rLLM's own runtime, not the remote Harbor runtime.
    agent_flow = MiniSweAgentHarness(
        sandbox_backend=SANDBOX_BACKEND,
        max_turns=MINISWE_MAX_TURNS,
        max_consecutive_format_errors=MINISWE_MAX_CONSECUTIVE_FORMAT_ERRORS,
        command_timeout=MINISWE_COMMAND_TIMEOUT,
        capture_exit_status=True,
        verify_only_on_env_done=True,
        skipped_verifier_reward=0.0,
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
