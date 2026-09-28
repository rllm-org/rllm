"""Four-run offline evaluation uses upstream flow, runner and compatible grader."""
import argparse
import os
import asyncio
from dataclasses import asdict
import json
from pathlib import Path

from compatible_grader import ForkserverEvaluator


CONC = int(os.environ.get("EVAL_CONCURRENCY", "8"))  # 8 = the original protocol
# Sampled evaluation (e.g. the training temperature with k samples per task).
# Unset = the greedy protocol, byte for byte: temperature 0, seed 1234, 1 attempt.
# EVAL_SEED=none drops the seed: with a fixed seed every attempt of a task would
# repeat the same sample, so k>1 at temperature>0 requires it.
TEMPERATURE = float(os.environ.get("EVAL_TEMPERATURE", "0"))
ATTEMPTS = int(os.environ.get("EVAL_ATTEMPTS", "1"))
SEED = os.environ.get("EVAL_SEED", "1234")


async def main(args):
    # Warm scoring before gateway threads acquire any file locks.
    evaluator = ForkserverEvaluator()
    from rllm.data import DatasetRegistry
    from rllm.cli.eval import _dict_rows_to_tasks
    from rllm.eval.runner import run_dataset
    from deepcoder_flow import deepcoder_flow

    dataset = DatasetRegistry.load_dataset("deepcoder", "test")
    assert dataset is not None and len(dataset.data) == 687
    rows = list(dataset.data)
    # Sharding splits the 687 test tasks across GPUs; merge_shards.py reassembles
    # them. Strided (not contiguous) so each shard sees the same difficulty mix,
    # and every task lands in exactly one shard. Defaults reproduce the
    # single-GPU protocol byte for byte.
    selected = list(range(len(rows)))[args.shard_index :: args.num_shards]
    rows = [rows[i] for i in selected]
    tasks = _dict_rows_to_tasks(rows)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    if ATTEMPTS > 1 and TEMPERATURE > 0 and SEED != "none":
        raise SystemExit("EVAL_ATTEMPTS>1 at EVAL_TEMPERATURE>0 needs EVAL_SEED=none, or every attempt repeats one sample")
    sampling = dict(temperature=0 if TEMPERATURE == 0 else TEMPERATURE, top_p=1.0, max_tokens=16384)
    if SEED != "none":
        sampling = dict(temperature=sampling["temperature"], top_p=1.0, seed=int(SEED), max_tokens=16384)
    (out / "protocol.json").write_text(json.dumps(dict(model=args.model, sampling=sampling,
        tasks=len(tasks), split="test", concurrency=CONC, max_model_len=32768,
        num_shards=args.num_shards, shard_index=args.shard_index, attempts=ATTEMPTS,
        global_indices=selected), indent=2))
    print(f"[eval] starting {len(tasks)} tasks (shard {args.shard_index}/{args.num_shards}) sampling={sampling}", flush=True)
    result, _episodes = await run_dataset(tasks, deepcoder_flow, args.url, args.model,
        concurrency=CONC, sandbox_backend="local", evaluator=evaluator,
        sampling_params=sampling, dataset_name="deepcoder", agent_name="deepcoder", attempts=ATTEMPTS)
    assert result.total == len(tasks) * ATTEMPTS
    # Re-index items from shard-local to global task ids so the merge is a plain
    # concatenation and a missing or duplicated task is detectable.
    from dataclasses import replace
    result = replace(result, items=[replace(it, idx=selected[it.idx]) for it in result.items])
    (out / "result.json").write_text(json.dumps(asdict(result), indent=2))
    # Checkpoint sweeps consume only aggregate success-rate metrics.  Do not
    # serialize the returned episodes: each file repeats private tests and can
    # exceed 16 GiB, while the lane deletes it immediately after completion.
    print(f"[eval] COMPLETE correct={result.correct}/{len(tasks) * ATTEMPTS} score={result.score:.6f} errors={result.errors}"
          + (f" pass_at={result.pass_at}" if ATTEMPTS > 1 else ""), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    ns = parser.parse_args()
    if ns.num_shards < 1:
        parser.error("--num-shards must be >=1")
    if not 0 <= ns.shard_index < ns.num_shards:
        parser.error("--shard-index must be in [0, num_shards)")
    asyncio.run(main(ns))
