"""Test-retest stability of the multi-table rubric judge: gpt-5.4-nano versus gpt-5.4-mini.

Fixed answers only (never reruns the policy): N recorded multi-table answers, each scored
K times by each judge with the unchanged FinQA multi-table rubric (finqa_eval._call_judge,
multi_table=True: 6 weighted 0-100 scores -> [0, 1]).

Per judge it reports:
  mean score; failed calls (no rubric back after the evaluator's own retries)
  test-retest: mean |score_i - score_j| over repeat pairs of the same answer
  within-answer SD (pooled) and between-answer SD
  ICC(1): between-answer variance / (between + within); 1 = scores track the answer, 0 = noise
  threshold flips: answers whose repeats fall on both sides of 0.9 (the is_correct cut)
and across judges: Spearman correlation of the per-answer mean scores.

usage: probe_judge_stability.py EVAL_DIR OUTPUT_DIR [N_PER_SPLIT] [K] [WORKERS]
  EVAL_DIR holds multi_val.json / multi_test.json and episodes_<split>/episodes/*.json
Run with reef's .venv-finqa (the judge module is the byte-identical rllm prpo-branch copy there).
"""

import concurrent.futures
import itertools
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "/home/yanan/agents/reef/exp_scripts/finqa")
import finqa_env  # noqa: E402,F401  installs rllm stand-ins and loads the judge credentials
import finqa_eval as evaluator  # noqa: E402

MODELS = ("gpt-5.4-nano", "gpt-5.4-mini")


def judge(model: str, sample: dict) -> dict:
    evaluator.MULTI_TABLE_JUDGE_MODEL = model
    task = sample["episode"]["task"]
    answer = sample["episode"]["artifacts"]["answer"]
    prompt = f"question : {task.get('core_question') or task['question']}\nmodel response : {answer}\nlabel : {task['ground_truth']}"
    value, rubric = evaluator._call_judge(evaluator.MULTI_TABLE_CORRECTNESS_PROMPT, prompt, multi_table=True)
    return {"model": model, "key": sample["key"], "score": float(value), "ok": bool(rubric), "rubric": rubric}


def run_model(model: str, samples: list[dict], k: int, workers: int, out: Path) -> list[dict]:
    jobs = [(s, r) for r in range(k) for s in samples]
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool, open(out / f"{model}.jsonl", "w") as fh:
        for row, (sample, repeat) in zip(pool.map(lambda job: judge(model, job[0]), jobs), jobs):
            row["repeat"] = repeat
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
    return rows


def spearman(a: list[float], b: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for m in range(i, j + 1):
                r[order[m]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    return cov / (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5


def stats(rows: list[dict], k: int) -> tuple[dict, dict[str, float]]:
    by: dict[str, list[float]] = {}
    for r in rows:
        if r["ok"]:
            by.setdefault(r["key"], []).append(r["score"])
    full = {key: v for key, v in by.items() if len(v) == k}
    pair_diffs = [abs(a - b) for v in full.values() for a, b in itertools.combinations(v, 2)]
    within = statistics.mean(statistics.variance(v) for v in full.values())
    means = {key: statistics.mean(v) for key, v in full.items()}
    between = max(statistics.variance(means.values()) - within / k, 0.0)
    return {
        "answers_complete": len(full),
        "calls": len(rows),
        "failed_calls": sum(not r["ok"] for r in rows),
        "mean_score": round(statistics.mean(means.values()), 4),
        "test_retest_mean_abs_diff": round(statistics.mean(pair_diffs), 4),
        "within_answer_sd": round(within ** 0.5, 4),
        "between_answer_sd": round(between ** 0.5, 4),
        "icc1": round(between / (between + within), 3) if between + within > 0 else None,
        "threshold_0.9_flips": sum(min(v) < 0.9 <= max(v) for v in full.values()),
        "answers_ever_ge_0.9": sum(max(v) >= 0.9 for v in full.values()),
    }, means


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(20260929)
    samples = []
    for split in ("multi_val", "multi_test"):
        files = sorted((root / f"episodes_{split}" / "episodes").glob("*.json"))
        for file in rng.sample(files, n):
            episode = json.loads(file.read_text())
            samples.append({"key": f"{split}/{episode['eval_idx']}", "episode": episode})
    (out / "samples.json").write_text(json.dumps([{"key": s["key"]} for s in samples], indent=1))
    print(f"{len(samples)} answers x {k} repeats x {len(MODELS)} judges", flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(MODELS)) as pool:  # a process per judge: model global
        futures = {m: pool.submit(run_model, m, samples, k, workers, out) for m in MODELS}
        rows = {m: f.result() for m, f in futures.items()}
    summary, means = {}, {}
    for m in MODELS:
        summary[m], means[m] = stats(rows[m], k)
    common = sorted(set(means[MODELS[0]]) & set(means[MODELS[1]]))
    summary["spearman_nano_vs_mini_means"] = round(spearman([means[MODELS[0]][c] for c in common],
                                                            [means[MODELS[1]][c] for c in common]), 3)
    summary["settings"] = {"eval_dir": str(root), "answers": len(samples), "repeats": k}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
