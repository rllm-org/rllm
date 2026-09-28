"""Compare all four full-test runs, including task-level pass disagreements."""
import itertools
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
runs = {}
print("| Run | Correct/687 | Accuracy | Errors |")
print("|---|---:|---:|---:|")
for gpu, rep in itertools.product(range(2), (1, 2)):
    name = f"gpu{gpu}-repeat{rep}"
    path = root / name / "result.json"
    if not path.exists():
        print(f"| {name} | pending | pending | pending |")
        continue
    result = json.loads(path.read_text())
    assert result["total"] == 687
    runs[name] = result
    print(f'| {name} | {result["correct"]}/687 | {result["score"]:.6f} | {result["errors"]} |')
for (a, ra), (b, rb) in itertools.combinations(runs.items(), 2):
    aa = {item["idx"]: item["is_correct"] for item in ra["items"]}
    bb = {item["idx"]: item["is_correct"] for item in rb["items"]}
    assert aa.keys() == bb.keys()
    different = sum(aa[k] != bb[k] for k in aa)
    print(f"{a} vs {b}: task pass/fail disagreements {different}/687 ({different/687:.2%})")
if len(runs) == 4:
    scores = [run["score"] for run in runs.values()]
    print(f"Accuracy range: {min(scores):.6f}–{max(scores):.6f}; spread {(max(scores)-min(scores))*100:.3f} percentage points")
