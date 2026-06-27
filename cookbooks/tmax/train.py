"""Reproduce Ai2's Tmax terminal-agent RL run on the tmax-15k corpus, eval on
Terminal-Bench.

This cookbook is the sibling of ``cookbooks/terminal-rl`` — same machinery, but
the training data is Ai2's published ~14.6K-task ``tmax-15k`` corpus and the
hyperparameters track their DPPO recipe (arXiv:2606.23321). Like terminal-rl it
ships no custom AgentFlow or evaluator:

* The **agent** is an in-tree terminal harness run as a
  :class:`~rllm.sandbox.sandboxed_flow.SandboxedAgentFlow` inside each task's
  sandbox. ``TMAX_HARNESS`` selects it:

  - ``terminus2`` (default) — Harbor's Terminus-2 tmux agent. The proven
    training path in this repo and the agent the ``tmax/TMax-15K-Harbor``
    dataset is documented to run with (``harbor run --agent terminus-2``).
  - ``mini-swe-agent`` — the architectural match to Tmax's *official* harness
    (their "Vanillux2" is a mini-SWE-agent-derived bash-tool agent). Use this
    for the highest-fidelity reproduction of their rollout shape.

  The rLLM gateway intercepts every LLM call, so the trainer sees full
  trajectories without the harness knowing it's being trained.
* The **evaluator** is each task's own verifier (sandbox-shell), resolved
  per-task by :class:`rllm.hooks.SandboxTaskHooks`. Both the tmax-15k training
  tasks and the Terminal-Bench eval tasks ship a ``tests/test.sh`` that writes
  ``1.0``/``0.0`` to ``/logs/verifier/reward.txt``; rLLM reads that back as the
  RL reward (Tmax's outcome-only ``verification_reward``).

Because we pass an ``agent_flow`` (and no explicit ``evaluator``/``hooks``),
:class:`AgentTrainer` runs the rLLM-native SandboxedAgentFlow path
(``AgentFlowEngine``) — sandboxes are created locally by ``SandboxTaskHooks``
via a pluggable ``sandbox_backend`` (``docker`` | ``local`` | ``modal`` |
``daytona``). This is NOT the remote-runtime / Harbor-runtime path.

The sandbox backend is selected by ``TERMINAL_SANDBOX_BACKEND`` (default
``modal``). Everything else is Hydra overrides on the command line — see
``train_verl.sh`` (full fine-tuning, the faithful reproduction) and
``train_fireworks.sh`` / ``train_tinker.sh`` (managed/single-machine LoRA).

Usage (from the rllm repo root)::

    TERMINAL_SANDBOX_BACKEND=modal python cookbooks/tmax/train.py rllm/backend=verl
"""

from __future__ import annotations

import os

import hydra
from omegaconf import DictConfig

from rllm.data.dataset import DatasetRegistry
from rllm.trainer import AgentTrainer

TRAIN_DATASET = "tmax-15k"
TRAIN_SPLIT = "train"

# Terminal-Bench eval version (Harbor registry). Must match prepare_data.py;
# both read TB_EVAL_VERSION so the pulled and loaded dataset names agree.
EVAL_VERSION = os.environ.get("TB_EVAL_VERSION", "2.0")
VAL_DATASET = f"terminal-bench@{EVAL_VERSION}"

# Sandbox backend for the SandboxedAgentFlow path: docker | local | modal | daytona.
SANDBOX_BACKEND = os.environ.get("TERMINAL_SANDBOX_BACKEND", "modal")

# Agent harness: "terminus2" (default; proven training path, runs the
# TMax-15K-Harbor tasks) or "mini-swe-agent" (closest match to Tmax's official
# Vanillux2 harness for fidelity).
TMAX_HARNESS = os.environ.get("TMAX_HARNESS", "terminus2")

# Optional cap on the validation set size. Terminal-Bench 2.0 is 89 tasks;
# validation runs ALL of them every time it fires, which is slow. Set
# TB_VAL_MAX=N to validate on the first N tasks instead (0/unset = all).
TB_VAL_MAX = int(os.environ.get("TB_VAL_MAX", "0"))

# Per-rollout cap. Terminus calls this max_turns; Mini-SWE calls it step_limit.
_max_turns = os.environ.get("TERMINUS_MAX_TURNS")
TERMINUS_MAX_TURNS = int(_max_turns) if _max_turns and int(_max_turns) > 0 else None
_mini_swe_step_limit = os.environ.get("MINI_SWE_STEP_LIMIT", "64")
MINI_SWE_STEP_LIMIT = int(_mini_swe_step_limit) if _mini_swe_step_limit and int(_mini_swe_step_limit) > 0 else None
_mini_swe_tool_timeout = os.environ.get("MINI_SWE_TOOL_TIMEOUT", "120")
MINI_SWE_TOOL_TIMEOUT = int(_mini_swe_tool_timeout) if _mini_swe_tool_timeout and int(_mini_swe_tool_timeout) > 0 else None


def _build_agent_flow():
    """Construct the selected harness as a SandboxedAgentFlow."""
    if TMAX_HARNESS == "mini-swe-agent":
        from rllm.harnesses.mini_swe_agent import MiniSweAgentHarness

        return MiniSweAgentHarness(
            sandbox_backend=SANDBOX_BACKEND,
            step_limit=MINI_SWE_STEP_LIMIT,
            tool_timeout=MINI_SWE_TOOL_TIMEOUT,
        )
    if TMAX_HARNESS == "terminus2":
        from rllm.harnesses.terminus2 import Terminus2Harness

        return Terminus2Harness(sandbox_backend=SANDBOX_BACKEND, max_turns=TERMINUS_MAX_TURNS)
    raise ValueError(f"Unknown TMAX_HARNESS={TMAX_HARNESS!r} (expected 'terminus2' or 'mini-swe-agent')")


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="unified", version_base=None)
def main(config: DictConfig) -> None:
    train_dataset = DatasetRegistry.load_dataset(TRAIN_DATASET, TRAIN_SPLIT)
    # val_dataset = DatasetRegistry.load_dataset(VAL_DATASET, "default")

    if train_dataset is None:
        raise RuntimeError(f"Dataset '{TRAIN_DATASET}' not found. Run: python cookbooks/tmax/prepare_data.py")

    keep = set(open("cookbooks/tmax/tmax_train_task_ids.txt", encoding="utf-8").read().splitlines())
    train_dataset.data = [row for row in train_dataset.data if row["task_id"] in keep]
    print(f"TMax task-id filter: kept {len(train_dataset)}/{len(keep)} rows", flush=True)
    # if val_dataset is None:
    #     raise RuntimeError(f"Dataset '{VAL_DATASET}' not found. Run: rllm dataset pull harbor:{VAL_DATASET} (or: python cookbooks/tmax/prepare_data.py)")

    # if TB_VAL_MAX > 0 and TB_VAL_MAX < len(val_dataset):
    #     val_dataset = val_dataset.select(range(TB_VAL_MAX))

    agent_flow = _build_agent_flow()

    trainer = AgentTrainer(
        backend=config.rllm.get("backend", "verl"),
        config=config,
        train_dataset=train_dataset,
        # val_dataset=val_dataset,
        agent_flow=agent_flow,
        sandbox_backend=SANDBOX_BACKEND,
    )
    trainer.train()


if __name__ == "__main__":
    main()
