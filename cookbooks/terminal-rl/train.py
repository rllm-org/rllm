"""Train a terminal agent on a local set of terminal-agent tasks, eval on
Terminal-Bench.

This cookbook deliberately ships no custom AgentFlow or evaluator:

* The **agent** is selected with ``TB_HARNESS``: ``terminus-2`` runs Harbor's
  Terminus-2 tmux/terminal agent and ``opencode`` runs the pinned OpenCode CLI.
  Both are in-tree :class:`~rllm.sandbox.sandboxed_flow.SandboxedAgentFlow`
  harnesses and run inside each task's sandbox. The rLLM gateway intercepts
  every LLM call, so the trainer sees full trajectories without the harness
  knowing it's being trained.
* The **evaluator** is each task's own verifier (sandbox-shell), resolved
  per-task by :class:`rllm.hooks.SandboxTaskHooks`. Both the local training
  tasks and the Terminal-Bench eval tasks ship a ``tests/test.sh`` that writes
  ``1.0``/``0.0`` to ``/logs/verifier/reward.txt``; rLLM reads that back as the
  RL reward.

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

    TERMINAL_SANDBOX_BACKEND=modal python cookbooks/terminal-rl/train.py rllm/backend=tinker
"""

from __future__ import annotations

import os

import hydra
from omegaconf import DictConfig

from rllm.data.dataset import DatasetRegistry
from rllm.harnesses.opencode import OpenCodeHarness
from rllm.harnesses.terminus2 import Terminus2Harness
from rllm.trainer import AgentTrainer

TRAIN_DATASET = os.environ.get("TB_TRAIN_DATASET", "tb-opus-pass")
TRAIN_SPLIT = os.environ.get("TB_TRAIN_SPLIT", "train")

# The periodic validation suite and boundary-only benchmark are independent.
# Production uses the full pinned Terminal-Bench 2.1 suite for validation at
# step 0, every 10 optimizer steps, and final weights. The separate benchmark
# path remains available to profiles such as the one-step LoRA sanity check.
EVAL_VERSION = os.environ.get("TB_EVAL_VERSION", "2.0")
VAL_DATASET = os.environ.get("TB_VAL_DATASET", f"terminal-bench@{EVAL_VERSION}")
VAL_SPLIT = os.environ.get("TB_VAL_SPLIT", "default")
VAL_EXPECTED_TASKS = int(os.environ.get("TB_VAL_EXPECTED_TASKS", "0"))
BENCHMARK_DATASET = os.environ.get("TB_BENCHMARK_DATASET", "").strip()
BENCHMARK_SPLIT = os.environ.get("TB_BENCHMARK_SPLIT", "default")
BENCHMARK_EXPECTED_TASKS = int(os.environ.get("TB_BENCHMARK_EXPECTED_TASKS", "0"))

# Sandbox backend for the SandboxedAgentFlow path: docker | local | modal | daytona.
SANDBOX_BACKEND = os.environ.get("TERMINAL_SANDBOX_BACKEND", "modal")

# Optional cap on the validation set size. Terminal-Bench 2.0 is 89 tasks;
# validation runs ALL of them every time it fires, which is slow. Set
# TB_VAL_MAX=N to validate on the first N tasks instead (0/unset = all).
TB_VAL_MAX = int(os.environ.get("TB_VAL_MAX", "0"))

# Per-rollout turn cap for the terminus2 agent. Unset = no artificial cap
# (Harbor's own default); the per-rollout RLLM_HARNESS_RUN_TIMEOUT_S still
# bounds wall-clock. The train_*.sh scripts default this to 100; set
# TERMINUS_MAX_TURNS=N to override (empty/0 = uncapped).
_terminus_max_turns = os.environ.get("TERMINUS_MAX_TURNS")
TERMINUS_MAX_TURNS = int(_terminus_max_turns) if _terminus_max_turns and int(_terminus_max_turns) > 0 else None

# Terminus-2 context compaction (summarization). Harbor enables it by default.
# Compaction creates a fresh linear-history segment; the gateway and trainer
# preserve those segments rather than treating the reset as one token prefix.
# Unset = Harbor's default (on).
TERMINUS_ENABLE_SUMMARIZE = os.environ.get("TERMINUS_ENABLE_SUMMARIZE", "1").strip().lower() not in ("0", "false", "no", "off")
# Harbor uses this advertised input limit to trigger proactive compaction.
# Launchers must align it with rllm.data.max_prompt_length so the gateway cannot
# reject a prompt before Terminus reaches its compaction threshold.
TERMINUS_MAX_INPUT_TOKENS = int(os.environ.get("TERMINUS_MAX_INPUT_TOKENS", "200000"))

# Terminal harness: terminus-2 | opencode.
TB_HARNESS = os.environ.get("TB_HARNESS", "terminus-2").strip().lower()


def _build_agent_flow():
    if TB_HARNESS in ("terminus-2", "terminus2"):
        return Terminus2Harness(
            sandbox_backend=SANDBOX_BACKEND,
            max_turns=TERMINUS_MAX_TURNS,
            enable_summarize=TERMINUS_ENABLE_SUMMARIZE,
            max_input_tokens=TERMINUS_MAX_INPUT_TOKENS,
        )
    if TB_HARNESS == "opencode":
        return OpenCodeHarness(sandbox_backend=SANDBOX_BACKEND)
    raise ValueError(f"Unsupported TB_HARNESS={TB_HARNESS!r}; expected 'terminus-2' or 'opencode'")


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="unified", version_base=None)
def main(config: DictConfig) -> None:
    train_dataset = DatasetRegistry.load_dataset(TRAIN_DATASET, TRAIN_SPLIT)
    val_dataset = DatasetRegistry.load_dataset(VAL_DATASET, VAL_SPLIT)
    benchmark_dataset = DatasetRegistry.load_dataset(BENCHMARK_DATASET, BENCHMARK_SPLIT) if BENCHMARK_DATASET else None

    if train_dataset is None:
        raise RuntimeError(f"Dataset '{TRAIN_DATASET}/{TRAIN_SPLIT}' not found. Run: python cookbooks/terminal-rl/prepare_data.py")
    if val_dataset is None:
        raise RuntimeError(f"Dataset '{VAL_DATASET}/{VAL_SPLIT}' not found. Run: python cookbooks/terminal-rl/prepare_data.py")
    if BENCHMARK_DATASET and benchmark_dataset is None:
        raise RuntimeError(f"Dataset '{BENCHMARK_DATASET}/{BENCHMARK_SPLIT}' not found. Run: python cookbooks/terminal-rl/prepare_data.py")
    if benchmark_dataset is not None and BENCHMARK_EXPECTED_TASKS > 0:
        if len(benchmark_dataset) != BENCHMARK_EXPECTED_TASKS:
            raise RuntimeError(f"Benchmark '{BENCHMARK_DATASET}/{BENCHMARK_SPLIT}' has {len(benchmark_dataset)} tasks; expected {BENCHMARK_EXPECTED_TASKS}")

    if TB_VAL_MAX > 0 and TB_VAL_MAX < len(val_dataset):
        val_dataset = val_dataset.select(range(TB_VAL_MAX))
    if VAL_EXPECTED_TASKS > 0 and len(val_dataset) != VAL_EXPECTED_TASKS:
        raise RuntimeError(f"Validation '{VAL_DATASET}/{VAL_SPLIT}' has {len(val_dataset)} tasks; expected {VAL_EXPECTED_TASKS}")

    train_ids = {str(row.get("task_id")) for row in train_dataset.get_data()}
    val_ids = {str(row.get("task_id")) for row in val_dataset.get_data()}
    overlap = sorted(train_ids & val_ids)
    if overlap:
        raise RuntimeError(f"Training and validation datasets overlap on {len(overlap)} task_id values; first overlap: {overlap[0]}")

    if benchmark_dataset is not None:
        benchmark_ids = {str(row.get("task_id")) for row in benchmark_dataset.get_data()}
        benchmark_overlap = sorted(train_ids & benchmark_ids)
        if benchmark_overlap:
            raise RuntimeError(f"Training and benchmark datasets overlap on {len(benchmark_overlap)} task_id values; first overlap: {benchmark_overlap[0]}")

    # Selected CLI as a SandboxedAgentFlow. Passing ``agent_flow`` (with no
    # explicit evaluator/hooks) makes AgentTrainer auto-wire SandboxTaskHooks
    # for the sandbox lifecycle + per-task verifier, and route rollouts through
    # AgentFlowEngine — rLLM's own runtime, not the remote Harbor runtime.
    # enable_summarize controls Terminus-2 context compaction. Prefix resets
    # caused by compaction are represented as separate trainable segments.
    agent_flow = _build_agent_flow()

    trainer = AgentTrainer(
        backend=config.rllm.get("backend", "tinker"),
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        benchmark_dataset=benchmark_dataset,
        agent_flow=agent_flow,
        sandbox_backend=SANDBOX_BACKEND,
    )
    trainer.train()


if __name__ == "__main__":
    main()
