"""Experiment-local FinQA multi-table training entrypoint.

Selects ``multi_train``/``multi_val`` and delegates all training mechanics to
rLLM's public AgentTrainer.  No trainer, loss, sampler, or checkpoint logic is
implemented here.
"""

from __future__ import annotations

import json

import hydra
from finqa_eval import finqa_evaluator
from omegaconf import DictConfig

from multitable_v2_flow import build_policy_visible_initial_messages, finqa_multitable_v2
from rllm.data.dataset import DatasetRegistry
from rllm.trainer import AgentTrainer

EXPECTED_ROWS = {"multi_train": 991, "multi_val": 126}
PRIVATE_FIELDS = ("ground_truth", "explanation")


def _audit_split(dataset, split: str) -> None:
    """Fail closed on the wrong data or private supervision in policy input."""
    rows = dataset.get_data()
    expected = EXPECTED_ROWS[split]
    if len(rows) != expected:
        raise RuntimeError(f"finqa/{split}: expected {expected} rows, found {len(rows)}")

    for index, row in enumerate(rows):
        qtype = str(row.get("question_type") or "")
        tables = row.get("table_name")
        if not qtype.startswith("multi_table"):
            raise RuntimeError(f"finqa/{split}[{index}] is not multi-table: {qtype!r}")
        if not isinstance(tables, list) or not 2 <= len(tables) <= 5:
            raise RuntimeError(f"finqa/{split}[{index}] has invalid table_name list: {tables!r}")
        for field in ("question", "company", *PRIVATE_FIELDS):
            if not str(row.get(field) or "").strip():
                raise RuntimeError(f"finqa/{split}[{index}] missing {field}")

    # Inject unique values into private fields and exercise the exact message
    # builder used by real rollouts. Any future accidental serialization fails.
    probe = dict(rows[0])
    sentinels = {
        "ground_truth": "PRIVATE_GROUND_TRUTH_SENTINEL_7f91cdb0",
        "explanation": "PRIVATE_EXPLANATION_SENTINEL_451a2ec9",
    }
    probe.update(sentinels)
    visible = json.dumps(
        build_policy_visible_initial_messages(probe, probe["question"], f"{split}-leak-probe"),
        ensure_ascii=False,
        sort_keys=True,
    )
    leaked = [field for field, sentinel in sentinels.items() if sentinel in visible]
    if leaked:
        raise RuntimeError(f"policy-visible prompt leaked private fields: {leaked}")


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="unified", version_base=None)
def main(config: DictConfig) -> None:
    train_dataset = DatasetRegistry.load_dataset("finqa", "multi_train")
    val_dataset = DatasetRegistry.load_dataset("finqa", "multi_val")
    if train_dataset is None or val_dataset is None:
        raise RuntimeError("FinQA multi_train/multi_val are not registered in the active RLLM_HOME")

    _audit_split(train_dataset, "multi_train")
    _audit_split(val_dataset, "multi_val")
    print("[multi-table preflight] rows=991/126; private-field sentinel leak check passed", flush=True)

    trainer = AgentTrainer(
        backend=config.rllm.get("backend", "verl"),
        agent_flow=finqa_multitable_v2,
        evaluator=finqa_evaluator,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
