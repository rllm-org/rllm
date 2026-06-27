"""torchrun entry point for the verl SFT backend.

The dispatcher (:meth:`AgentSFTTrainer._launch_distributed`) materializes the
curated data to parquet, serializes the resolved verl config, then spawns::

    torchrun --standalone --nnodes=1 --nproc_per_node=<gpus> \
        -m rllm.trainer.sft.verl_entry --config <verl_sft_config.yaml>

Each rank loads the same serialized config and runs verl's FSDP SFT loop. The
config is already fully composed (verl defaults + SFTSpec overrides + the
parquet ``data.train_files``), so no hydra composition happens here — we mirror
verl's own ``sft_trainer.main`` (``auto_set_device`` then ``run_sft``).
"""

from __future__ import annotations

import argparse
import os


def _isolate_compile_caches() -> None:
    """Give each torchrun rank its own Triton/Inductor cache dir.

    All ranks otherwise compile the same kernels (e.g. Triton's ``cuda_utils``)
    into one shared ``~/.triton/cache`` path concurrently and race, surfacing as
    ``cuda_utils...so: cannot open shared object file``. Per-rank dirs make each
    compile independent. Must run before torch/verl/triton are imported.
    """
    rank = os.environ.get("LOCAL_RANK") or os.environ.get("RANK") or "0"
    base = os.path.join("/tmp", f"rllm_sft_compile_rank{rank}")
    os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(base, "triton"))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(base, "inductor"))


def main() -> None:
    parser = argparse.ArgumentParser(description="rLLM verl SFT torchrun entry")
    parser.add_argument("--config", required=True, help="Path to the serialized verl SFT config (yaml).")
    args = parser.parse_args()

    _isolate_compile_caches()

    from omegaconf import OmegaConf

    config = OmegaConf.load(args.config)

    # Imported here (not at module top) so the module stays importable without
    # the verl stack, and so torch/verl init happens inside the process group.
    from verl.trainer.sft_trainer import run_sft
    from verl.utils.device import auto_set_device

    auto_set_device(config)
    run_sft(config)


if __name__ == "__main__":
    main()
