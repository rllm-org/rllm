"""Register the multi-table FinQA splits that prepare_finqa_data.py leaves on disk.

The cookbook's prepare_finqa_data.py only loads data/{split}_finqa.csv, so the
991/126/131 rows under data/multi_table_data/ are never registered. That is
deliberate for TRAINING -- finqa.md Finding 1 shows adding multi-table data
hurts (66.3% single-only vs 61.6% mixed). But the multi-table test set is
exactly the held-out generalization probe the paper reports (base 13.9% ->
trained 26.6% on FinQA-Reasoning), so it is worth having as an EVAL split.

Registers them under the same `finqa` dataset with distinct split names, so
`rllm eval finqa --split multi_test` picks them up without touching the repo.

  python register_multi_table.py
"""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd
from rllm.data.dataset import DatasetRegistry

DATA = Path("/home/yanan/agents/rllm/cookbooks/finqa/data/multi_table_data")


def _parse_json_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            parsed = s
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, str):
            cleaned = parsed.strip()
            return [cleaned] if cleaned else []
        return []
    return []


# Identical shape to prepare_finqa_data.preprocess_fn -- in particular
# question_type must survive verbatim, since finqa_eval switches to the
# 6-component weighted rubric on qtype.startswith("multi_table").
def preprocess_fn(example: dict) -> dict:
    return {
        "question": example["user_query"],
        "ground_truth": example["answer"],
        "data_source": "finqa",
        "company": example["company"],
        "question_id": str(example["id"]),
        "question_type": example["question_type"],
        "core_question": example["question"],
        "table_name": _parse_json_list(example.get("table_name")),
        "columns_used": _parse_json_list(example.get("columns_used_json")),
        "rows_used": _parse_json_list(example.get("rows_used_json")),
        "explanation": example["explanation"],
    }


if __name__ == "__main__":
    for src, split in (("test", "multi_test"), ("val", "multi_val"), ("train", "multi_train")):
        df = pd.read_csv(DATA / f"{src}_finqa.csv")
        rows = [preprocess_fn(r) for _, r in df.iterrows()]
        ds = DatasetRegistry.register_dataset("finqa", rows, split)
        kinds = sorted({r["question_type"] for r in rows})
        n_multi = sum(1 for r in rows if str(r["question_type"]).startswith("multi_table"))
        print(f"  finqa/{split:12s} {len(ds.get_data()):5d} rows | "
              f"multi_table-typed: {n_multi}/{len(rows)} | kinds={kinds}")
