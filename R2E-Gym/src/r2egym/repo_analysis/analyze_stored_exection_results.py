import glob
import json
from collections import Counter

from tqdm import tqdm

from r2egym.repo_analysis.execution_result_analysis import ExecutionResult

files = glob.glob(
    "/home/gcpuser/buckets/local_repoeval_bucket/repos/numpy*/execution_result.json"
)

counts = Counter()

for file in tqdm(files):
    with open(file, "r") as f:
        data = json.load(f)
        execution_result = ExecutionResult(**data)
        counts[execution_result.is_good_exec()[0]] += 1

print(counts)
