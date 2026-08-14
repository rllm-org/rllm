#!/usr/bin/env python3
"""Prepare verifier-correct DeepSWE trajectories for 256K Nemotron SFT.

The eval archive stores cumulative chat histories.  This script takes the final
history from every verifier-correct episode, canonicalizes it through rLLM's SFT
bridge, and keeps every accepted assistant message trainable exactly once.

Later harness retry messages start a new cumulative interval so a renderer may
rewrite earlier reasoning without changing a supervised target. Rows above the
context limit are split at assistant/tool-cycle boundaries, filling spare space
with the longest preceding context suffix. No row is silently truncated.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import random
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from rllm.data.sft_bridges import bridge_messages
from rllm.data.sft_schema import normalize_messages
from rllm.renderers import resolve
from rllm.trainer.sft.tinker_dataset import (
    _to_renderer_message,
    _validate_rendered_attribution,
    _validate_tools,
)

SOURCE_REPO = "mobius-lab/deepswe-sft"
SOURCE_REVISION = "bf260a695a1ce108a1605bf7dc955a251d5fd549"
SOURCE_RESULTS_SHA256 = "1187b71e5b98ea989a8de45080680580ef5bbcf680497ae76458df10c3aff8e6"
SOURCE_META_SHA256 = "f9dcdd4a902455c0627bc9151744e9e21f2a6e80bb65f51c19984abea2511619"
MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
TOKENIZER_REVISION = "d51eab0d1f979ebc26b546e634a04f450d99158e"
RLLM_REVISION = "38207dc67e80ff2b7e2de721141b5f8fcbf0347b"
MAX_LENGTH = 262_144
TOKENIZER_SHA256 = {
    "chat_template.jinja": "575fb74f54ed264df9047d0ecce3c98938aae953fb4f50356675706264cbb68a",
    "config.json": "699f34f0fc645d29ebffa5767fb59e6ae6ec98e3a4605485eb9913256d0df7e6",
    "special_tokens_map.json": "e9435fefd6d838fd9fcbbc44b97a8e3ff322be7f6dfb7e4fd2468586574bb52b",
    "tokenizer.json": "623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7",
    "tokenizer_config.json": "10f93eabcb9b1602fbb991d6308e787ce1df28ee9cd7a1c6d1e8c3f338b957bc",
}

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _verify_tokenizer(tokenizer_dir: Path, revision: str) -> dict[str, str]:
    if revision != TOKENIZER_REVISION:
        raise ValueError(f"expected tokenizer revision {TOKENIZER_REVISION}, got {revision}")
    if not tokenizer_dir.is_dir():
        raise ValueError("--tokenizer must be the pinned local tokenizer directory")
    actual = {name: _sha256(tokenizer_dir / name) for name in TOKENIZER_SHA256}
    if actual != TOKENIZER_SHA256:
        differences = {name: (TOKENIZER_SHA256[name], actual[name]) for name in actual if actual[name] != TOKENIZER_SHA256[name]}
        raise ValueError(f"tokenizer SHA-256 mismatch: {differences}")
    return actual


def _episode_path(run_dir: Path, eval_idx: int, attempt: int) -> Path:
    matches = sorted((run_dir / "episodes").glob(f"episode_{eval_idx:06d}_*_{attempt}.json"))
    if len(matches) != 1:
        raise ValueError(f"expected one episode for eval_idx={eval_idx}, attempt={attempt}; found {len(matches)}")
    return matches[0]


def _canonical_correct_histories(run_dir: Path) -> list[dict]:
    results_path = run_dir / "results.json"
    meta_path = run_dir / "meta.json"
    if _sha256(results_path) != SOURCE_RESULTS_SHA256:
        raise ValueError(f"unexpected results.json hash: {_sha256(results_path)}")
    if _sha256(meta_path) != SOURCE_META_SHA256:
        raise ValueError(f"unexpected meta.json hash: {_sha256(meta_path)}")

    results = json.loads(results_path.read_text())
    histories: list[dict] = []
    for item in sorted(results["items"], key=lambda value: (value["idx"], value["attempt"])):
        if item.get("is_correct") is not True:
            continue
        episode = json.loads(_episode_path(run_dir, item["idx"], item["attempt"]).read_text())
        trajectories = [trajectory for trajectory in episode["trajectories"] if trajectory["name"] == "mini-swe-agent"]
        if len(trajectories) != 1:
            raise ValueError(f"eval_idx={item['idx']} has {len(trajectories)} mini-swe-agent trajectories")
        trajectory = trajectories[0]
        if episode["eval_idx"] != item["idx"] or episode["is_correct"] is not True:
            raise ValueError(f"results/episode correctness mismatch at eval_idx={item['idx']}")
        if float(item["reward"]) != 1.0 or float(trajectory["reward"]) != 1.0:
            raise ValueError(f"correct episode has non-unit reward at eval_idx={item['idx']}")
        if not trajectory["steps"]:
            raise ValueError(f"correct episode has no steps at eval_idx={item['idx']}")

        raw_messages = copy.deepcopy(trajectory["steps"][-1]["chat_completions"])
        for message in raw_messages:
            message["trainable"] = message.get("role") == "assistant"
        task_id = episode["task"]["task"]["name"]
        row = bridge_messages(
            [
                {
                    "messages": raw_messages,
                    "tools": [BASH_TOOL],
                    "source_run": "glm-5.2",
                    "source_eval_idx": item["idx"],
                    "task_id": task_id,
                    "attempt": item["attempt"],
                    "reward": 1.0,
                    "is_correct": True,
                    "termination_reason": episode["termination_reason"],
                }
            ]
        )[0].to_record()
        row["source_message_count"] = len(row["messages"])
        row["raw_step_target_count"] = len(trajectory["steps"])
        histories.append(row)
    return histories


def _interval_rows(history: dict) -> list[dict]:
    """Emit one cumulative prefix per user interval.

    Only assistant messages after that interval's user are targets.  The row
    ends at its last accepted assistant, so a later harness-retry user cannot
    rewrite an earlier supervised prefix.
    """
    messages = history["messages"]
    users = [index for index, message in enumerate(messages) if message["role"] == "user"]
    if not users:
        raise ValueError(f"task {history['task_id']} has no user message")

    rows: list[dict] = []
    base = {key: copy.deepcopy(value) for key, value in history.items() if key != "messages"}
    for interval_index, user_index in enumerate(users):
        next_user = users[interval_index + 1] if interval_index + 1 < len(users) else len(messages)
        targets = [index for index in range(user_index + 1, next_user) if messages[index]["role"] == "assistant"]
        if not targets:
            raise ValueError(f"task {history['task_id']} user interval {interval_index} has no assistant target")
        row_messages = copy.deepcopy(messages[: targets[-1] + 1])
        target_set = set(targets)
        for index, message in enumerate(row_messages):
            message["trainable"] = index in target_set
        rows.append(
            {
                **copy.deepcopy(base),
                "messages": row_messages,
                "source_interval": interval_index,
                "source_target_message_indices": targets,
                "source_target_message_sha256": [_json_sha256(messages[index]) for index in targets],
                "target_count": len(targets),
            }
        )
    return rows


def _render_stats(row: dict, renderer) -> tuple[int, int]:
    messages = normalize_messages(row["messages"])
    if messages[-1].role != "assistant" or messages[-1].trainable is not True:
        raise ValueError(f"task {row['task_id']} row must end at a trainable assistant")
    invalid_role = next(
        (index for index, message in enumerate(messages) if message.trainable and message.role != "assistant"),
        None,
    )
    if invalid_role is not None:
        raise ValueError(f"task {row['task_id']} trains non-assistant message {invalid_role}")
    seen_calls: set[str] = set()
    for index, message in enumerate(messages):
        seen_calls.update(call.id for call in message.tool_calls or [] if call.id)
        if message.role == "tool" and message.tool_call_id not in seen_calls:
            raise ValueError(f"task {row['task_id']} has orphan tool result at message {index}")
    rendered = renderer.render(
        [_to_renderer_message(message) for message in messages],
        tools=_validate_tools(row.get("tools")),
    )
    _validate_rendered_attribution(rendered, len(messages))
    content = getattr(rendered, "is_content", None) or [True] * len(rendered.token_ids)
    per_message = Counter(index for index, is_content in zip(rendered.message_indices, content, strict=True) if index >= 0 and messages[index].trainable and is_content)
    missing = [index for index, message in enumerate(messages) if message.trainable and per_message[index] == 0]
    if missing:
        raise ValueError(f"task {row['task_id']} has targets with no rendered loss tokens: {missing}")
    return len(rendered.token_ids), sum(per_message.values())


def _window_candidate(
    row: dict,
    target_positions: list[int],
    start: int,
    end: int,
    *,
    context_ordinal: int | None = None,
) -> dict:
    first_target = target_positions[0]
    if context_ordinal is None:
        context_ordinal = start if start == 0 else start - 1
    context_start = target_positions[context_ordinal]
    indexed_messages = [
        *enumerate(row["messages"][:first_target]),
        *enumerate(row["messages"][context_start : target_positions[end] + 1], start=context_start),
    ]
    selected_positions = set(target_positions[start : end + 1])
    messages = []
    for source_index, source_message in indexed_messages:
        message = copy.deepcopy(source_message)
        message["trainable"] = source_index in selected_positions
        messages.append(message)

    candidate = {key: copy.deepcopy(value) for key, value in row.items() if key != "messages"}
    candidate["messages"] = messages
    candidate["source_target_message_indices"] = [row["source_target_message_indices"][ordinal] for ordinal in range(start, end + 1)]
    candidate["source_target_message_sha256"] = [row["source_target_message_sha256"][ordinal] for ordinal in range(start, end + 1)]
    candidate["target_count"] = end - start + 1
    candidate["window_target_ordinal_start"] = start
    candidate["window_target_ordinal_end"] = end
    candidate["context_target_ordinal_start"] = context_ordinal
    candidate["overlap_target_message_index"] = row["source_target_message_indices"][start - 1] if start else None
    return candidate


def _fit_row(row: dict, renderer, max_length: int) -> list[dict]:
    rendered_tokens, trainable_tokens = _render_stats(row, renderer)
    if rendered_tokens <= max_length:
        fitted = copy.deepcopy(row)
        fitted["window_index"] = 0
        fitted["window_count"] = 1
        fitted["window_target_ordinal_start"] = 0
        fitted["window_target_ordinal_end"] = row["target_count"] - 1
        fitted["context_target_ordinal_start"] = 0
        fitted["overlap_target_message_index"] = None
        fitted["rendered_tokens"] = rendered_tokens
        fitted["trainable_tokens"] = trainable_tokens
        return [fitted]

    target_positions = [index for index, message in enumerate(row["messages"]) if message.get("trainable") is True]
    if len(target_positions) != row["target_count"]:
        raise ValueError(f"task {row['task_id']} target metadata does not match its mask")

    windows: list[dict] = []
    start = 0
    while start < len(target_positions):
        low, high = start, len(target_positions) - 1
        best: tuple[dict, int, int] | None = None
        while low <= high:
            end = (low + high) // 2
            candidate = _window_candidate(row, target_positions, start, end)
            length, loss_tokens = _render_stats(candidate, renderer)
            if length <= max_length:
                best = (candidate, length, loss_tokens)
                low = end + 1
            else:
                high = end - 1
        if best is None:
            raise ValueError(f"task {row['task_id']} target ordinal {start} cannot fit in {max_length} tokens")
        candidate, length, loss_tokens = best

        # Spend the remaining context budget on the longest immediately
        # preceding assistant/tool-cycle suffix. This keeps continuation
        # windows as cumulative as the 256K ceiling allows.
        minimal_context = candidate["context_target_ordinal_start"]
        low, high = 0, minimal_context
        expanded = (candidate, length, loss_tokens)
        while low <= high:
            context_ordinal = (low + high) // 2
            context_candidate = _window_candidate(
                row,
                target_positions,
                start,
                candidate["window_target_ordinal_end"],
                context_ordinal=context_ordinal,
            )
            context_length, context_loss = _render_stats(context_candidate, renderer)
            if context_length <= max_length:
                expanded = (context_candidate, context_length, context_loss)
                high = context_ordinal - 1
            else:
                low = context_ordinal + 1
        candidate, length, loss_tokens = expanded
        candidate["rendered_tokens"] = length
        candidate["trainable_tokens"] = loss_tokens
        windows.append(candidate)
        start = candidate["window_target_ordinal_end"] + 1

    for index, window in enumerate(windows):
        window["window_index"] = index
        window["window_count"] = len(windows)
    return windows


def _task_split(task_ids: list[str], val_fraction: float, seed: int) -> set[str]:
    shuffled = sorted(set(task_ids))
    random.Random(seed).shuffle(shuffled)
    n_val = int(len(shuffled) * val_fraction)
    if val_fraction > 0 and n_val == 0:
        n_val = 1
    return set(shuffled[:n_val])


def _semantic_sha256(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        canonical = copy.deepcopy(row)
        canonical["messages"] = [message.model_dump(exclude_none=True) for message in normalize_messages(canonical["messages"])]
        digest.update(json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _write_parquet(path: Path, rows: list[dict]) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="zstd", version="2.6", use_dictionary=True)


def _split_stats(rows: list[dict]) -> dict:
    return {
        "rows": len(rows),
        "tasks": len({row["task_id"] for row in rows}),
        "targets": sum(row["target_count"] for row in rows),
        "rendered_tokens": sum(row["rendered_tokens"] for row in rows),
        "trainable_tokens": sum(row["trainable_tokens"] for row in rows),
        "max_rendered_tokens": max(row["rendered_tokens"] for row in rows),
        "task_list_sha256": _json_sha256(sorted({row["task_id"] for row in rows})),
        "semantic_sha256": _semantic_sha256(rows),
    }


def prepare(args: argparse.Namespace) -> dict:
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    if args.source_revision != SOURCE_REVISION:
        raise ValueError(f"expected source revision {SOURCE_REVISION}, got {args.source_revision}")

    renderers_version = importlib.metadata.version("renderers")
    if renderers_version != "0.1.9":
        raise ValueError(f"renderers 0.1.9 is required, got {renderers_version}")

    tokenizer_sha256 = _verify_tokenizer(args.tokenizer, args.tokenizer_revision)
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), trust_remote_code=False)
    resolution = resolve(MODEL, tokenizer, renderer_name="nemotron3")
    if (resolution.source, resolution.name) != ("prime", "nemotron-3"):
        raise ValueError(f"expected Prime nemotron-3 renderer, got {resolution.source}:{resolution.name}")

    histories = _canonical_correct_histories(run_dir)
    print(f"selected {len(histories)} verifier-correct final histories", flush=True)
    accepted_targets = {row["source_eval_idx"]: [(index, _json_sha256(message)) for index, message in enumerate(row["messages"]) if message["role"] == "assistant"] for row in histories}
    val_tasks = _task_split([row["task_id"] for row in histories], args.val_fraction, args.seed)

    interval_rows = [interval for history in histories for interval in _interval_rows(history)]
    print(f"rendering and fitting {len(interval_rows)} cumulative user intervals", flush=True)
    fitted_rows: list[dict] = []
    for index, row in enumerate(interval_rows, start=1):
        fitted_rows.extend(_fit_row(row, resolution.renderer, args.max_length))
        if index % 25 == 0 or index == len(interval_rows):
            print(f"  fitted {index}/{len(interval_rows)} intervals", flush=True)
    emitted_targets: dict[int, list[tuple[int, str]]] = {}
    for row in fitted_rows:
        emitted_targets.setdefault(row["source_eval_idx"], []).extend(
            zip(
                row["source_target_message_indices"],
                row["source_target_message_sha256"],
                strict=True,
            )
        )
    for eval_idx, expected in accepted_targets.items():
        actual = sorted(emitted_targets.get(eval_idx, []))
        if actual != expected:
            raise ValueError(f"eval_idx={eval_idx} target coverage mismatch: expected {len(expected)}, got {len(actual)}")

    train_rows = [row for row in fitted_rows if row["task_id"] not in val_tasks]
    validation_rows = [row for row in fitted_rows if row["task_id"] in val_tasks]
    if {row["task_id"] for row in train_rows} & {row["task_id"] for row in validation_rows}:
        raise ValueError("task leakage between train and validation")

    train_path = output_dir / "train.parquet"
    validation_path = output_dir / "validation.parquet"
    canary_path = output_dir / "canary-longest-two.parquet"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet(train_path, train_rows)
    _write_parquet(validation_path, validation_rows)
    _write_parquet(
        canary_path,
        sorted(fitted_rows, key=lambda row: row["rendered_tokens"], reverse=True)[:2],
    )

    # Prove the persisted representation retains every semantic value.
    reread_train = pq.read_table(train_path).to_pylist()
    reread_validation = pq.read_table(validation_path).to_pylist()
    if _semantic_sha256(reread_train) != _semantic_sha256(train_rows):
        raise ValueError("train Parquet semantic round-trip mismatch")
    if _semantic_sha256(reread_validation) != _semantic_sha256(validation_rows):
        raise ValueError("validation Parquet semantic round-trip mismatch")

    manifest = {
        "schema_version": "rllm.deepswe-nemotron-super-sft.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "source": {
            "repo": SOURCE_REPO,
            "revision": args.source_revision,
            "run": run_dir.name,
            "results_sha256": _sha256(run_dir / "results.json"),
            "meta_sha256": _sha256(run_dir / "meta.json"),
            "teacher_model": "accounts/fireworks/models/glm-5p2",
            "selection": "is_correct == true and reward == 1.0",
            "correct_episodes": len(histories),
            "raw_step_targets": sum(row["raw_step_target_count"] for row in histories),
            "accepted_final_history_targets": sum(len(value) for value in accepted_targets.values()),
            "rejected_outputs_omitted": sum(row["raw_step_target_count"] for row in histories) - sum(len(value) for value in accepted_targets.values()),
        },
        "conversion": {
            "format": "canonical SFTRow with explicit per-message trainable mask",
            "trajectory_policy": "final cumulative chat_completions only",
            "retry_policy": "one cumulative prefix per user interval; only that interval trains",
            "window_policy": "assistant/tool-cycle boundaries; maximal preceding context suffix",
            "max_length": args.max_length,
            "val_fraction": args.val_fraction,
            "seed": args.seed,
            "task_grouped_split": True,
            "tool_semantic_sha256": _json_sha256([BASH_TOOL]),
        },
        "rendering": {
            "model": MODEL,
            "tokenizer_revision": args.tokenizer_revision,
            "tokenizer_sha256": tokenizer_sha256,
            "renderer": f"{resolution.source}:{resolution.name}",
            "renderers_version": renderers_version,
            "rllm_revision": RLLM_REVISION,
        },
        "runtime": {
            "python": ".".join(map(str, sys.version_info[:3])),
            "transformers": importlib.metadata.version("transformers"),
            "tokenizers": importlib.metadata.version("tokenizers"),
            "pyarrow": importlib.metadata.version("pyarrow"),
        },
        "total": _split_stats(fitted_rows),
        "train": {**_split_stats(train_rows), "file": train_path.name, "sha256": _sha256(train_path)},
        "validation": {
            **_split_stats(validation_rows),
            "file": validation_path.name,
            "sha256": _sha256(validation_path),
        },
        "canary": {
            "file": canary_path.name,
            "rows": 2,
            "purpose": "authenticated two-longest-row 256K capacity check only",
            "sha256": _sha256(canary_path),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    sums_path = output_dir / "SHA256SUMS"
    sums_path.write_text("".join(f"{_sha256(path)}  {path.name}\n" for path in (train_path, validation_path, canary_path, manifest_path)))
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    parser.add_argument("--tokenizer-revision", default=TOKENIZER_REVISION)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = prepare(args)
    print(json.dumps({"output_dir": str(args.output_dir), **manifest["total"]}, indent=2))


if __name__ == "__main__":
    main()
