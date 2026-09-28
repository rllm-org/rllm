"""Generate verifier-scored DeepCoder trajectories in rLLM's eval layout.

This is orchestration around the upstream DeepCoder flow and deterministic
verifier.  It deliberately writes the native ``results.json`` plus one Episode
JSON per rollout so ``rllm.eval.curation`` can consume the run without a format
conversion step.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from dataclasses import replace
from pathlib import Path

from compatible_grader import ForkserverEvaluator

# The gateway binds a TCP port per ``run_dataset`` call. Left to the OS
# (``_find_free_port``) it lands in the ephemeral range (32768-60999 here), which
# the shard's own outbound HTTP sockets also draw from -- the probe/bind window is
# a TOCTOU race that killed shard0 on 2026-09-19 after 76 chunks ("address already
# in use" -> "Gateway thread did not start within 30.0s"). One long-lived gateway
# on a fixed port below the ephemeral floor removes both the repetition and the race.
GATEWAY_PORT_FLOOR = 1024
GATEWAY_PORT_CEIL = 32767


def _start_gateway(args: argparse.Namespace):
    from rllm.gateway.manager import EvalGatewayManager

    gateway = EvalGatewayManager(
        upstream_url=args.url,
        model=args.model,
        port=args.gateway_port or None,
    )
    gateway.start()
    print(f"[gateway] serving {gateway.gateway_url} for the whole run", flush=True)
    return gateway


def _load_resume_state(out, selected_indices: list[int], args: argparse.Namespace):
    """Validate an interrupted run directory and return (first_unfinished_task, items).

    Every check is fatal: resuming onto a directory whose task selection, attempt
    count or rollout count disagrees with this invocation would silently splice two
    different experiments into one results.json.
    """
    from rllm.eval.results import EvalResult

    progress_path = out / "progress.json"
    meta_path = out / "meta.json"
    results_path = out / "results.json"
    for path in (progress_path, meta_path, results_path):
        if not path.exists():
            raise RuntimeError(f"--resume needs {path}; this run directory is not resumable")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("selected_source_indices") != selected_indices:
        raise RuntimeError(
            "--resume refused: the task selection differs from the original run "
            "(check --split, --selection-seed, --max-examples, --num-shards, --shard-index)"
        )
    if int(meta.get("attempts", -1)) != args.attempts:
        raise RuntimeError(
            f"--resume refused: original run used attempts={meta.get('attempts')}, this one {args.attempts}"
        )

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    done = int(progress["completed_tasks"])
    if int(progress["selected_tasks"]) != len(selected_indices):
        raise RuntimeError(
            f"--resume refused: progress.json has selected_tasks={progress['selected_tasks']}, "
            f"this invocation selected {len(selected_indices)}"
        )
    if done % args.flush_tasks:
        raise RuntimeError(
            f"--resume refused: completed_tasks={done} is not a multiple of --flush-tasks="
            f"{args.flush_tasks}; the durable boundary and the chunk boundary must agree"
        )

    result = EvalResult.load(str(results_path))
    expected = done * args.attempts
    if len(result.items) != expected:
        raise RuntimeError(
            f"--resume refused: results.json holds {len(result.items)} rollouts, "
            f"expected {expected} for {done} completed tasks x {args.attempts} attempts"
        )

    print(
        f"[resume] {out} continues at task {done}/{len(selected_indices)}; "
        f"{len(result.items)} rollouts kept",
        flush=True,
    )
    return done, list(result.items)


async def main(args: argparse.Namespace) -> None:
    # Warm the forkserver before vLLM/gateway activity creates file-locking
    # threads.  This is the same grader compatibility path used by training.
    evaluator = ForkserverEvaluator()

    from deepcoder_flow import deepcoder_flow
    from rllm.cli.eval import _dict_rows_to_tasks
    from rllm.data import DatasetRegistry
    from rllm.eval.episode_store import EvalEpisodeStore, _json_default, _sanitize
    from rllm.eval.results import EvalResult
    from rllm.eval.runner import run_dataset

    dataset = DatasetRegistry.load_dataset("deepcoder", args.split)
    if dataset is None:
        raise RuntimeError(f"DeepCoder split {args.split!r} is not registered")

    rows = [dict(row) for row in dataset.data]
    source_size = len(rows)
    selected_indices = list(range(source_size))
    if args.max_examples is not None and args.max_examples < source_size:
        rng = random.Random(args.selection_seed)
        selected_indices = rng.sample(selected_indices, args.max_examples)
        rows = [rows[i] for i in selected_indices]

    # Shard only after optional seeded sampling. Each selected task belongs to
    # exactly one shard, while all k attempts for that task stay together.
    if args.num_shards > 1:
        paired = list(zip(selected_indices, rows, strict=True))
        paired = paired[args.shard_index :: args.num_shards]
        selected_indices = [idx for idx, _ in paired]
        rows = [row for _, row in paired]

    # rLLM's generic wrapper otherwise uses the post-selection row number.  A
    # stable source UID is required when attempts/runs are pooled by curation.
    for source_idx, row in zip(selected_indices, rows, strict=True):
        row["task_id"] = str(row.get("uid") or row.get("index") or source_idx)

    tasks = _dict_rows_to_tasks(rows)
    # Keep an immutable copy before ``run_dataset`` expands attempts and the
    # engine gives each Episode an attempt-qualified id (``task:attempt``).
    # Curation groups attempts by the task id embedded in the episode filename,
    # so that filename must contain the source task id, not the Episode id.
    stable_task_ids = [task.id for task in tasks]
    out = Path(args.output).expanduser().resolve()
    resume_from = 0
    resumed_items: list = []
    if out.exists() and any(out.iterdir()):
        if not args.resume:
            raise RuntimeError(f"refusing to reuse non-empty eval directory: {out}")
        resume_from, resumed_items = _load_resume_state(out, selected_indices, args)
    store = EvalEpisodeStore(out)

    # Do not pin one identical request seed across the eight attempts.  The
    # curation signal requires independently sampled completions.  Selection of
    # tasks remains deterministic through selection_seed.
    sampling = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    store.write_meta(
        {
            "benchmark": "deepcoder",
            "split": args.split,
            "model": args.model,
            "agent": "deepcoder",
            "attempts": args.attempts,
            "source_size": source_size,
            "selected_tasks": len(tasks),
            "selected_source_indices": selected_indices,
            "selection_seed": args.selection_seed,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "sampling": sampling,
            "verifier": "ForkserverEvaluator(deepcoder_evaluator)",
        }
    )

    def save_episode(global_idx, episode):
        task_pos, attempt = divmod(global_idx, args.attempts)
        if task_pos >= len(stable_task_ids):
            raise RuntimeError(
                f"episode index {global_idx} maps past {len(stable_task_ids)} source tasks"
            )
        expected_attempt = global_idx % args.attempts
        episode_attempt = int(str(episode.id).rsplit(":", 1)[-1])
        if episode_attempt != expected_attempt:
            raise RuntimeError(
                f"episode order mismatch at {global_idx}: expected attempt "
                f"{expected_attempt}, got {episode.id!r}"
            )

        # EvalEpisodeStore currently falls back to Episode.id when Episode.task
        # is a dict rather than a Task dataclass. That would write
        # ``..._<task>_<attempt>.json`` and make native curation treat the eight
        # attempts as eight different tasks. Write the same native JSON payload
        # under an explicitly stable filename instead.
        store.episodes_dir.mkdir(parents=True, exist_ok=True)
        path = store.episodes_dir / (
            f"episode_{global_idx:06d}_{_sanitize(stable_task_ids[task_pos])}.json"
        )
        data = episode.model_dump(mode="json")
        data["eval_idx"] = global_idx
        data["stable_task_id"] = stable_task_ids[task_pos]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=_json_default)

    # Upstream run_dataset invokes its callback only after execute_tasks has
    # returned the *entire* input. On a 24,287 x 8 formal run that would retain
    # every Episode in RAM and write nothing for days. Run bounded task chunks
    # and atomically publish a curation-compatible results.json after each one.
    all_items = list(resumed_items)
    gateway = _start_gateway(args)
    for chunk_start in range(resume_from, len(tasks), args.flush_tasks):
        chunk = tasks[chunk_start : chunk_start + args.flush_tasks]

        def save_chunk_episode(local_idx, episode, *, offset=chunk_start):
            save_episode(offset * args.attempts + local_idx, episode)

        chunk_result, _ = await run_dataset(
            tasks=chunk,
            agent_flow=deepcoder_flow,
            base_url=args.url,
            model=args.model,
            concurrency=args.concurrency,
            sandbox_backend="local",
            evaluator=evaluator,
            sampling_params=sampling,
            dataset_name="deepcoder",
            agent_name="deepcoder",
            attempts=args.attempts,
            on_episode_complete=save_chunk_episode,
            gateway=gateway,
        )
        all_items.extend(replace(item, idx=item.idx + chunk_start) for item in chunk_result.items)
        result = EvalResult.from_items(
            "deepcoder", args.model, "deepcoder", all_items, attempts=args.attempts
        )
        temporary = out / "results.json.tmp"
        result.save(str(temporary))
        temporary.replace(out / "results.json")
        completed_tasks = min(chunk_start + len(chunk), len(tasks))
        progress_tmp = out / "progress.json.tmp"
        progress_tmp.write_text(
            json.dumps(
                {
                    "status": "complete" if completed_tasks == len(tasks) else "running",
                    "completed_tasks": completed_tasks,
                    "selected_tasks": len(tasks),
                    "completed_rollouts": result.total,
                    "expected_rollouts": len(tasks) * args.attempts,
                    "flush_tasks": args.flush_tasks,
                },
                indent=2,
            )
            + "\n"
        )
        progress_tmp.replace(out / "progress.json")
        print(
            f"[durable-progress] tasks={completed_tasks}/{len(tasks)} "
            f"rollouts={result.total}/{len(tasks) * args.attempts}",
            flush=True,
        )

    gateway.stop()

    expected = len(tasks) * args.attempts
    if result.total != expected:
        raise RuntimeError(f"incomplete eval: expected {expected} rollouts, got {result.total}")
    episode_files = list(store.episodes_dir.glob("episode_*.json"))
    if len(episode_files) != expected:
        raise RuntimeError(
            "incomplete episode store: "
            f"expected {expected} per-rollout JSON files, found {len(episode_files)}"
        )
    result.save(str(out / "results.json"))
    print(
        json.dumps(
            {
                "run_dir": str(out),
                "tasks": len(tasks),
                "attempts": args.attempts,
                "rollouts": result.total,
                "correct": result.correct,
                "errors": result.errors,
                "score": result.score,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--attempts", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--selection-seed", type=int, default=1234)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument(
        "--gateway-port",
        type=int,
        default=0,
        help="fixed gateway port (0 = OS-assigned; a fixed port below the ephemeral "
        "floor avoids the bind race that kills long runs)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue an interrupted run directory from its last durable chunk",
    )
    parser.add_argument(
        "--flush-tasks",
        type=int,
        default=32,
        help="durably publish episodes/results after this many source tasks",
    )
    ns = parser.parse_args()
    if ns.attempts < 2:
        parser.error("--attempts must be >=2 for difficulty-band curation")
    if ns.max_examples is not None and ns.max_examples < 1:
        parser.error("--max-examples must be positive")
    if ns.num_shards < 1:
        parser.error("--num-shards must be >=1")
    if not 0 <= ns.shard_index < ns.num_shards:
        parser.error("--shard-index must be in [0, num_shards)")
    if ns.flush_tasks < 1:
        parser.error("--flush-tasks must be positive")
    if ns.gateway_port and not GATEWAY_PORT_FLOOR <= ns.gateway_port <= GATEWAY_PORT_CEIL:
        parser.error(
            f"--gateway-port must be in [{GATEWAY_PORT_FLOOR}, {GATEWAY_PORT_CEIL}]; "
            "the ephemeral range is reachable by outbound sockets and will race"
        )
    asyncio.run(main(ns))
