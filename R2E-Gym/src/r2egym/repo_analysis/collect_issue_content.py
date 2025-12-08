import re
import os
import json
import subprocess
from time import sleep
from concurrent.futures import ThreadPoolExecutor

import tqdm
from ghapi.all import GhApi

from datasets import load_dataset, Dataset


GHAPI_TOKEN = os.environ.get("GHAPI_TOKEN")
api = GhApi(token=GHAPI_TOKEN)

repo_name_map = {
    "pandas": "pandas-dev/pandas",
    "numpy": "numpy/numpy",
    "pillow": "python-pillow/Pillow",
    "Pillow": "python-pillow/Pillow",
    "orange3": "biolan/orange3",
    "datalad": "datalad/datalad",
    "coveragepy": "nedbat/coveragepy",
    "aiohttp": "aio-libs/aiohttp",
    "tornado": "tornadoweb/tornado",
    "pyramid": "Pylons/pyramid",
    "scrapy": "scrapy/scrapy",
}


def collect_issue_contents(row):
    issue_contents = []
    for issue_num in row["resolved_issues"]:
        try:
            issue_data = api.issues.get(
                row["repo_owner"], row["repo_name"], int(issue_num)
            )
            issue_contents.append(
                {
                    "title": issue_data.title,
                    "body": issue_data.body,
                }
            )
        except Exception as e:
            if "API rate limit exceeded" in str(e):
                print("API rate limit exceeded. Waiting for 60 seconds...")
                print(repr(e))
                sleep(60)
                return collect_issue_contents(row)
            print(
                f"Error fetching issue {issue_num} from {row['repo_owner']}/{row['repo_name']}: {e}"
            )
            continue

    return issue_contents


def main():
    ds = load_dataset("r2e-edits/r2e-dockers-v2")["train"].to_pandas()
    ds["repo_owner"] = ds["repo_name"].apply(lambda x: repo_name_map[x].split("/")[0])
    ds["repo_name"] = ds["repo_name"].apply(lambda x: repo_name_map[x].split("/")[1])
    ds["issue_contents"] = None

    rows = ds.iterrows()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(collect_issue_contents, row): idx for idx, row in rows
        }

        for future in tqdm.tqdm(
            futures,
            desc="Processing rows",
            total=len(futures),
            unit="row",
        ):
            idx = futures[future]
            try:
                ds.at[idx, "issue_contents"] = future.result()
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
    print("Done")
    print(ds["issue_contents"])
    print("Issue contents counts:")
    print(ds["issue_contents"].notna().sum())
    print("Issue contents counts:")
    print(
        ds["issue_contents"].apply(lambda x: len(x) if isinstance(x, list) else 0).sum()
    )
    print("Issue contents counts:")
    print(
        ds["issue_contents"]
        .apply(lambda x: len(x) > 0 if isinstance(x, list) else 0)
        .sum()
    )

    dataset = Dataset.from_pandas(ds)
    dataset.push_to_hub(
        "r2e-edits/r2e-dockers-v3",
        split="train",
    )


if __name__ == "__main__":
    main()
