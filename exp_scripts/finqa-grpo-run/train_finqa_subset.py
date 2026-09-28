"""FinQA GRPO driver with subsampled splits (2x RTX 5000 Ada).

Identical to cookbooks/finqa/train.py except that the train/val splits are
subsampled, so that "a few epochs" is a few hours instead of a few weeks.
Everything else -- finqa_flow, finqa_evaluator, AgentTrainer, the unified
hydra config -- is reused untouched.

    FINQA_TRAIN_N=256 FINQA_VAL_N=64 python train_finqa_subset.py rllm/backend=verl ...

FINQA_TRAIN_N / FINQA_VAL_N unset or 0 -> use the full split.
"""

import os

import hydra
from finqa_eval import finqa_evaluator
from finqa_flow import finqa_flow
from omegaconf import DictConfig

from rllm.data.dataset import DatasetRegistry
from rllm.trainer import AgentTrainer


SUBSET_SEED = int(os.getenv("FINQA_SUBSET_SEED", "0"))


def _subsample(ds, n, what):
    """Take a RANDOM n-subset, not the first n.

    The FinQA CSVs are grouped by company: the first 640 train rows cover only
    27 of the 165 companies, so a head-slice trains on a narrow set of table
    schemas and any conclusion drawn from it fails to generalise. Shuffling
    with a fixed seed keeps runs reproducible while covering ~162 companies.
    """
    total = len(ds)
    if not n or n <= 0 or n >= total:
        print(f"[subset] {what}: using all {total} examples", flush=True)
        return ds
    sub = ds.shuffle(seed=SUBSET_SEED).select(range(n))
    try:
        companies = {r.get("company") for r in sub.get_data()}
        print(f"[subset] {what}: {n}/{total} examples (random, seed={SUBSET_SEED}, "
              f"{len(companies)} distinct companies)", flush=True)
    except Exception:
        print(f"[subset] {what}: {n}/{total} examples (random, seed={SUBSET_SEED})", flush=True)
    return sub


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    train_dataset = DatasetRegistry.load_dataset("finqa", "train")
    val_dataset = DatasetRegistry.load_dataset("finqa", "val")

    if train_dataset is None or val_dataset is None:
        raise RuntimeError("FinQA dataset not found. Run: python cookbooks/finqa/prepare_finqa_data.py")

    train_dataset = _subsample(train_dataset, int(os.getenv("FINQA_TRAIN_N", "0")), "train")
    val_dataset = _subsample(val_dataset, int(os.getenv("FINQA_VAL_N", "0")), "val")

    trainer = AgentTrainer(
        backend=config.rllm.get("backend", "verl"),
        agent_flow=finqa_flow,
        evaluator=finqa_evaluator,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
