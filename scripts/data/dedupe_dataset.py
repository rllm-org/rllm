# https://github.com/huggingface/open-r1/blob/main/scripts/decontaminate.py
"""
Usage:

python scripts/data/dedupe_dataset.py \
    --dedupe_dataset <This is the dataset that gets deduped> \
    --dataset <RAG over this dataset, unmodified> \
    --problem_column <name of column> 
"""

import collections
import os

from tqdm import tqdm
from rllm.utils import RAG

import json
from rllm.data.utils import load_dataset
from rllm.data.dataset_types import TrainDataset

def normalize_string(text: str) -> str:
    """Basic string normalization."""
    # Convert to lowercase and normalize whitespace
    text = text.lower().strip()
    # Replace multiple spaces with single space
    text = " ".join(text.split())
    return text

def get_prompt_from_chat_template(text: str) -> str:
    """Extract the prompt from a chat template."""
    if isinstance(text, str):
        return text
    elif isinstance(text, list):
        return text[0]['content'] if text else ""
    else:
        raise ValueError(f"Unsupported type for text: {type(text)}. Expected str or list.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dedupe_dataset", type=str, required=True, help="Path of the first dataset to check for duplicates")
    parser.add_argument("--dataset", type=str, required=True, help="Paths of 2nd dataset to check for duplicates against.")
    parser.add_argument(
        "--problem_column", type=str, default="problem", help="Name of the column containing the problem (prompt)."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.expanduser("~/rllm/rllm/data/train/code"),
        help="Data directory to save the deduped dataset. If not provided, will use the default data directory.",
    )
    parser.add_argument(
        "--new_dataset_name",
        type=str,
        default=None,
        help="New name for the dataset. If not provided, will reuse the name and add a `_dedupe` to the name."
    )
    args = parser.parse_args()

    # Load the dataset to check for contamination

    # open dataset from json
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory {args.data_dir} does not exist.")
    

    
    # read the dataset as json
    ds_name = TrainDataset.Code[args.dedupe_dataset.upper()]
    orig_ds_name = TrainDataset.Code[args.dataset.upper()]

    ds = load_dataset(ds_name)
    orig_ds = load_dataset(orig_ds_name)


    # get the column as a list
    problem_col = [prob[args.problem_column] for prob in ds]

    # init rag
    rag = RAG(docs=problem_col)


    # loop through the dataset and check for duplicates
    # using the rag
    dupe_idx = set()
    for prob_desc in tqdm(orig_ds, desc="Checking for duplicates"):
        # get the problem description
        desc = prob_desc[args.problem_column]
        if not isinstance(desc, str):
            print(f"Skipping due to non-string question: {desc}")
            continue
        # normalize the question
        normalized_question = normalize_string(desc)
        # check if the question is in the rag
        top_3 = rag.top_k(normalized_question, k=3)

        # loop through the 3 top results and check if the score is greater than 0.95
        if top_3:
            for top in top_3:
                if top["score"] > 0.95:
                    dupe_idx.add(top["idx"])  # add the index to the set

    # remove the dupe idx rows from ds
    
    if dupe_idx:
        print(f"Found {len(dupe_idx)} duplicates in the dataset.")
        ds = [p for i, p in enumerate(ds) if i not in dupe_idx]  # remove the duplicates from the dataset
        print(len(ds), "remaining after removing duplicates.")
    else:
        print("No duplicates found.")

    # save the dataset to a new file
    new_ds_name = args.new_dataset_name or f"{os.path.splitext(os.path.basename(args.dedupe_dataset))[0]}_dedupe"
    if not new_ds_name.endswith(".json"):
        new_ds_name += ".json"

    # write to json path
    with open(os.path.join(args.data_dir, new_ds_name), "w", encoding="utf-8") as f:
        json.dump(ds, f)

    print(f"All done! Saving the deduped dataset to {new_ds_name} in {args.data_dir}")
