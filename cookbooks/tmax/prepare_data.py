"""Pull the train + eval datasets for the tmax cookbook.

Both sides are sandbox-format benchmarks (self-contained per-task environment +
``tests/test.sh`` verifier) in Harbor's directory-per-task layout. They differ
only in *where* the tasks come from:

* **Train** — ``tmax-15k``, a first-class rLLM dataset registered in
  ``rllm/registry/datasets.json`` and materialized by
  ``rllm.data.tmax_builder`` from Ai2's Hugging Face corpus: ``allenai/TMax-15K``
  (the ``test_final_state`` pytest verifier + task description) joined by
  ``task_id`` with ``allenai/tmax-15k-open-instruct`` (the per-task prebuilt
  image ``hamishi740/swerl-tmax-v3:<digest>``). This is the full ~14.6K
  compositional terminal-agent RL corpus behind the Tmax models
  (arXiv:2606.23321). ``rllm dataset pull tmax-15k`` builds the Harbor task
  dirs and registers them under ``tmax-15k/train``. Each task's verifier runs
  ``test_final_state.py`` in the prebuilt image and writes ``/tmp/rllm/reward.json``
  — the signal rLLM's per-task verifier reads back as the RL reward.

  (Ai2's README points at a Harbor-registry copy, ``tmax/TMax-15K-Harbor``, but
  that dataset is not present in the public Harbor registry, so we build from
  Hugging Face instead. Pulling the per-task images at full 15K scale may need
  a Docker Hub business account; ``--train-limit`` keeps smoke runs small but
  note the builder still downloads both full HF datasets to do the join.)
* **Eval** — ``harbor:terminal-bench@<version>`` pulled straight from the
  Harbor registry (the same path the Terminal-Bench eval cookbook uses).
  ``TB_EVAL_VERSION`` selects the version (default ``2.0``, which is the
  ``TB 2.0`` number Tmax reports in its model card). The registry only
  publishes ``2.0`` today, so set ``TB_EVAL_VERSION=2.1`` once it lands.

Re-runs are cheap: the Harbor pulls are no-ops once the tasks are cached
locally.

Usage::

    python cookbooks/tmax/prepare_data.py
    # smoke run with a small training cap (re-registers a truncated split):
    python cookbooks/tmax/prepare_data.py --train-limit 50
    # evaluate against a different Terminal-Bench version:
    TB_EVAL_VERSION=2.1 python cookbooks/tmax/prepare_data.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

# First-class rLLM dataset name (catalog entry in rllm/registry/datasets.json,
# source: harbor:tmax/TMax-15K-Harbor@latest). Registered under the "train" split.
TRAIN_DATASET = "tmax-15k"
TRAIN_SPLIT = "train"

# Terminal-Bench eval version (Harbor registry). 2.0 is what the registry
# publishes today and the number Tmax reports; flip via TB_EVAL_VERSION.
EVAL_VERSION = os.environ.get("TB_EVAL_VERSION", "2.0")
EVAL_DATASET = f"terminal-bench@{EVAL_VERSION}"


def _pull(name: str) -> None:
    """``rllm dataset pull <name>`` as a subprocess (uses the catalog/Harbor path)."""
    cmd = [sys.executable, "-m", "rllm.cli.main", "dataset", "pull", name]
    print(f"[tmax] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _pull_train() -> None:
    """Build + register the tmax-15k corpus from Hugging Face (via the catalog builder)."""
    try:
        _pull(TRAIN_DATASET)
    except subprocess.CalledProcessError as e:
        raise SystemExit(
            f"[tmax] Failed to build '{TRAIN_DATASET}'. It is materialized by "
            f"rllm.data.tmax_builder from allenai/TMax-15K + allenai/tmax-15k-open-instruct; "
            f"set HF_TOKEN if rate-limited and ensure network access to the HF Hub."
        ) from e


def _truncate_train(limit: int) -> int:
    """Re-register only the first ``limit`` tasks of the train split (smoke runs)."""
    from rllm.data import DatasetRegistry

    ds = DatasetRegistry.load_dataset(TRAIN_DATASET, TRAIN_SPLIT)
    if ds is None:
        raise RuntimeError(f"'{TRAIN_DATASET}/{TRAIN_SPLIT}' not found after pull")
    rows = list(ds.get_data() if hasattr(ds, "get_data") else ds.data)
    rows = rows[:limit]
    DatasetRegistry.register_dataset(
        name=TRAIN_DATASET,
        data=rows,
        split=TRAIN_SPLIT,
        source="allenai/TMax-15K",
        description=f"TMax-15K (truncated to {len(rows)} tasks for a smoke run)",
        category="agentic",
    )
    return len(rows)


def _pull_eval() -> None:
    """Pull the Terminal-Bench eval split from the Harbor registry."""
    name = f"harbor:{EVAL_DATASET}"
    try:
        _pull(name)
    except subprocess.CalledProcessError as e:
        raise SystemExit(
            f"[tmax] Failed to pull '{name}'. The Harbor registry currently "
            f"publishes terminal-bench@2.0; if you requested a version it does not "
            f"have, set TB_EVAL_VERSION to an available one (e.g. 2.0)."
        ) from e


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Cap training tasks (default: all ~14.6K). Useful for smoke runs.",
    )
    args = ap.parse_args()

    _pull_train()
    n_train = None
    if args.train_limit is not None:
        n_train = _truncate_train(args.train_limit)
        print(f"[tmax] Truncated {TRAIN_DATASET}/{TRAIN_SPLIT} to {n_train} tasks", flush=True)

    _pull_eval()

    suffix = f" ({n_train})" if n_train is not None else ""
    print(
        f"\n[tmax] Done. Train: {TRAIN_DATASET}{suffix}   Eval: {EVAL_DATASET}\n"
        f"        Run `bash cookbooks/tmax/train_verl.sh` to reproduce the 9B run,\n"
        f"        or `bash cookbooks/tmax/train_fireworks.sh` for the managed LoRA variant.",
        flush=True,
    )


if __name__ == "__main__":
    main()
