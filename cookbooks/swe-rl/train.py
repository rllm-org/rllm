"""Train an SWE agent on R2E-Gym, validate on SWE-bench Verified.

This cookbook deliberately ships no custom AgentFlow or evaluator:

* The **agent** is the in-tree ``terminus2`` harness
  (:class:`rllm.harnesses.terminus2.Terminus2Harness`) — a
  :class:`~rllm.sandbox.sandboxed_flow.SandboxedAgentFlow` that runs the
  terminus2 CLI agent inside each task's sandbox. The rLLM gateway
  intercepts every LLM call, so the trainer sees full trajectories without
  the harness knowing it's being trained.
* The **evaluator** is each task's own verifier (sandbox-shell), resolved
  per-task by :class:`rllm.hooks.SandboxTaskHooks`. For r2egym it runs the
  image's own ``/testbed/run_tests.sh`` and checks pytest-output equality
  against the row's expected output; for the Verified split it runs the
  task's bundled ``tests/test.sh``. The verifier writes a reward that rLLM
  reads back.

Because we pass an ``agent_flow`` (and no explicit ``evaluator``/``hooks``),
:class:`AgentTrainer` runs the **rLLM-native SandboxedAgentFlow path**
(``AgentFlowEngine``) — sandboxes are created locally by ``SandboxTaskHooks``
via a pluggable ``sandbox_backend`` (``docker`` | ``local`` | ``modal`` |
``daytona``). This is NOT the remote-runtime / ``RemoteAgentFlowEngine`` path.

The sandbox backend is selected by the ``SWE_SANDBOX_BACKEND`` env var
(default ``modal``). For ``modal`` install ``pip install modal`` and run
``modal token new``; for ``daytona`` install ``pip install daytona`` and set
``DAYTONA_API_KEY``. Everything else is configured by Hydra overrides on the
command line (see ``train_tinker.sh`` / ``train_verl.sh`` for working defaults).

Usage (from rllm repo root)::

    SWE_SANDBOX_BACKEND=modal python cookbooks/swe-rl/train.py rllm/backend=tinker
"""

from __future__ import annotations

import os

import hydra
from omegaconf import DictConfig

from rllm.data.dataset import DatasetRegistry
from rllm.harnesses.terminus2 import Terminus2Harness
from rllm.trainer import AgentTrainer

TRAIN_DATASET = "r2egym"
VAL_DATASET = "swebench-verified"

# Sandbox backend for the SandboxedAgentFlow path: docker | local | modal | daytona.
SANDBOX_BACKEND = os.environ.get("SWE_SANDBOX_BACKEND", "modal")

# Optional cap on the validation set size. SWE-bench Verified is 500 tasks;
# validation runs ALL of them every time it fires, which is slow. Set
# SWE_VAL_MAX=N to validate on the first N tasks instead (0/unset = all 500).
SWE_VAL_MAX = int(os.environ.get("SWE_VAL_MAX", "0"))

# Per-rollout turn cap for the terminus2 agent. Unset = no artificial cap
# (Harbor's own default); the per-rollout RLLM_HARNESS_RUN_TIMEOUT_S still
# bounds wall-clock. The train_*.sh scripts default this to 100; set
# TERMINUS_MAX_TURNS=N to override (empty/0 = uncapped).
_terminus_max_turns = os.environ.get("TERMINUS_MAX_TURNS")
TERMINUS_MAX_TURNS = int(_terminus_max_turns) if _terminus_max_turns and int(_terminus_max_turns) > 0 else None


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="unified", version_base=None)
def main(config: DictConfig) -> None:
    train_dataset = DatasetRegistry.load_dataset(TRAIN_DATASET, "train")
    val_dataset = DatasetRegistry.load_dataset(VAL_DATASET, "default")

    if train_dataset is None:
        raise RuntimeError(f"Dataset '{TRAIN_DATASET}' not found. Run: rllm dataset pull {TRAIN_DATASET} (or: python cookbooks/swe-rl/prepare_data.py)")
    if val_dataset is None:
        raise RuntimeError(f"Dataset '{VAL_DATASET}' not found. Run: rllm dataset pull harbor:swebench-verified (or: python cookbooks/swe-rl/prepare_data.py)")

    if SWE_VAL_MAX > 0 and SWE_VAL_MAX < len(val_dataset):
        val_dataset = val_dataset.select(range(SWE_VAL_MAX))

    # terminus2 as a SandboxedAgentFlow. Passing ``agent_flow`` (with no
    # explicit evaluator/hooks) makes AgentTrainer auto-wire SandboxTaskHooks
    # for the sandbox lifecycle + per-task verifier, and route rollouts through
    # AgentFlowEngine — rLLM's own runtime, not the remote Harbor runtime.
    agent_flow = Terminus2Harness(sandbox_backend=SANDBOX_BACKEND, max_turns=TERMINUS_MAX_TURNS)

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
