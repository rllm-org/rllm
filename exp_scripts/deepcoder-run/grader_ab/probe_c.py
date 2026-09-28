"""Grader A/B across hosts: grade the SAME reference solutions with the SAME grader.

Every host reads one shared manifest (fixed task list), grades the dataset's own
reference solution for each task through the training grader path
(forkserver context -> RewardCodeFn, identical to compatible_grader/deepcoder_eval),
and records wall time + pass/fail per task. Reference solutions should pass;
a timeout or failure is a property of the host, not of a model.
"""
import json, os, socket, sys, time
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, "/home/yanan/agents/rllm/cookbooks/deepcoder")
PARQUET = f"{ROOT}/runtime/datasets/deepcoder/train.parquet"
MANIFEST = f"{ROOT}/grader_ab/manifest.json"
N_TASKS, STRIDE, CONCURRENCY = 48, 503, int(os.environ.get("PROBE_CONC","8"))

def build_manifest():
    pf = pq.ParquetFile(PARQUET); rg = pf.metadata.row_group(0).num_rows
    rows = []
    for i in range(N_TASKS):
        idx = (i * STRIDE) % pf.metadata.num_rows
        r = pf.read_row_group(idx // rg).to_pylist()[idx % rg]
        assert r["solutions"], f"row {idx} has no reference solution"
        rows.append({"row": idx, "uid": r["uid"], "data_source": r["data_source"]})
    json.dump(rows, open(MANIFEST, "w"), indent=1)
    return rows

def load_row(pf, idx):
    rg = pf.metadata.row_group(0).num_rows
    return pf.read_row_group(idx // rg).to_pylist()[idx % rg]

def main():
    from compatible_grader import install_grader_context
    install_grader_context()                      # same forkserver path as training
    from rllm.rewards.code_reward import RewardCodeFn
    from rllm.rewards.reward_types import RewardConfig
    grader = RewardCodeFn(RewardConfig())
    manifest = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else build_manifest()
    pf = pq.ParquetFile(PARQUET)
    tasks = [load_row(pf, m["row"]) for m in manifest]
    for m, r in zip(manifest, tasks): assert r["uid"] == m["uid"], "manifest/parquet mismatch"
    tasks *= int(os.environ.get("PROBE_REPEAT_TASKS", "1"))
    def one(r):
        t0 = time.time(); out = grader(task_info=r, action=r["solutions"][0]); dt = time.time() - t0
        return {"uid": r["uid"], "data_source": r["data_source"], "passed": bool(out.is_correct),
                "sec": round(dt, 2), "n_tests": (out.metadata or {}).get("total_tests")}
    t0 = time.time()
    with ThreadPoolExecutor(CONCURRENCY) as ex: res = list(ex.map(one, tasks))
    wall = time.time() - t0
    host = socket.gethostbyname(socket.gethostname()); host = os.environ.get("PROBE_HOST", host)
    secs = sorted(x["sec"] for x in res)
    summ = {"host": host, "tasks": len(res), "concurrency": CONCURRENCY, "wall_s": round(wall, 1),
            "passed": sum(x["passed"] for x in res), "ge12s": sum(x["sec"] >= 12 for x in res),
            "p50": secs[len(secs)//2], "p90": secs[int(len(secs)*.9)], "max": secs[-1], "per_task": res}
    json.dump(summ, open(f"{ROOT}/grader_ab/result_{host}.json", "w"), indent=1)
    print(f"[probe] host={host} passed={summ['passed']}/{len(res)} ge12s={summ['ge12s']} p50={summ['p50']}s p90={summ['p90']}s max={summ['max']}s wall={summ['wall_s']}s", flush=True)

if __name__ == "__main__":
    main()
