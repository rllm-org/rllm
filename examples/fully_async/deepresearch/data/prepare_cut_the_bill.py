import random

from datasets import load_dataset

from rllm.data import DatasetRegistry

ds = load_dataset("aidenjhwu/ASearcher_en_no-math_Qwen3-8B-reject-sample")

data_ls = []
for sample in ds["train"]:
    data_raw = sample["extra_info"]
    data_raw.pop("query_id")
    data_raw.pop("id")
    data_ls.append(data_raw)

random.seed(42)
random.shuffle(data_ls)

DatasetRegistry.register_dataset("cutthebill", data_ls, "train")
