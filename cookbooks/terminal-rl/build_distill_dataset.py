#!/usr/bin/env python3
"""Build a thinking-SFT distillation dataset from an rllm eval run.

Turns deepseek-v4-pro terminus2 trajectories into per-step SFT rows that
PRESERVE reasoning. Stock `rllm dataset from-eval` drops reasoning_content
(rllm/eval/curation.py `_clean_message`), which would give thinking-free SFT of
a thinking model — so we curate here instead and register the result exactly
like curation does (`DatasetRegistry.register_dataset(name, rows)`), making it a
drop-in `rllm sft <name>` dataset.

Format (verified against the tinker_cookbook `deepseekv3_thinking` renderer):
  - ONE ROW PER STEP. That renderer lacks the sequence-extension property, so
    multi-turn conversations must be split and trained with LAST_ASSISTANT_MESSAGE
    (i.e. `rllm sft --tokenize-method stepwise`).
  - Each row's `messages` = that step's chat_completions prefix (already ends at
    the step's assistant turn, with earlier turns' thinking stripped, matching
    deepseek's history format). The LAST assistant message's content is rewritten
    to `<think>{reasoning}</think>{content}` so reasoning renders into a TRAINED
    `<think>` block (confirmed: the renderer puts weight>0 on the think tokens).
  - Only CORRECT episodes by default.

Usage:
  python build_distill_dataset.py <eval_run_dir_or_id> --name tb2-dsv4-distill
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from rllm import paths
from rllm.data import DatasetRegistry

# Tokenization for length filtering — MUST match the SFT config so a row that
# survives here also fits at train time (same tokenizer + renderer + masking).
_RENDERER = "deepseekv3_thinking"
_TOKENIZER_DIR = os.path.expanduser("~/.rllm/tokenizers/deepseek-v4-flash")


def _fold_reasoning(msg: dict) -> dict:
    reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
    content = msg.get("content") or ""
    if reasoning and "<think>" not in content:
        content = f"<think>{reasoning}</think>{content}"
    out = {"role": "assistant", "content": content}
    if msg.get("tool_calls"):
        out["tool_calls"] = msg["tool_calls"]
    return out


def _clean(msg: dict) -> dict:
    out = {"role": msg["role"], "content": msg.get("content") or ""}
    for k in ("tool_calls", "tool_call_id", "name"):
        if msg.get(k):
            out[k] = msg[k]
    return out


def _resolve_run_dir(ref: str) -> str:
    if os.path.isdir(os.path.join(ref, "episodes")):
        return ref
    cand = os.path.join(paths.eval_results_dir(), ref)
    if os.path.isdir(os.path.join(cand, "episodes")):
        return cand
    raise SystemExit(f"no episodes/ under {ref!r} (or under eval_results/{ref})")


def build(run_dir: str, include_incorrect: bool, max_length: int) -> list[dict]:
    # datum truncation is right-side, so a row longer than max_length loses its
    # TARGET (weights all zero) and trains on nothing. Drop such rows here so the
    # dataset is clean at any max_length rather than silently no-op'ing 30%+.
    from tinker_cookbook.renderers import TrainOnWhat, get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    renderer = get_renderer(_RENDERER, get_tokenizer(_TOKENIZER_DIR))

    rows: list[dict] = []
    n_ep = n_kept = n_dropped_len = 0
    for f in sorted(glob.glob(os.path.join(run_dir, "episodes", "*.json"))):
        ep = json.load(open(f))
        n_ep += 1
        if not include_incorrect and not ep.get("is_correct"):
            continue
        n_kept += 1
        task_id = (ep.get("task") or {}).get("harbor_task_name") or ep.get("id")
        for traj in ep.get("trajectories") or []:
            for step in traj.get("steps") or []:
                cc = step.get("chat_completions") or []
                last = max((i for i, m in enumerate(cc) if m.get("role") == "assistant"), default=-1)
                if last < 0:
                    continue
                messages = [_clean(m) for m in cc[:last]] + [_fold_reasoning(cc[last])]
                if not any(m["role"] != "assistant" for m in messages[:-1]):
                    continue  # need a user/tool message before the target
                mi, _ = renderer.build_supervised_example(messages, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE)
                if mi.length > max_length:
                    n_dropped_len += 1
                    continue
                rows.append({"messages": messages, "task_id": task_id, "reward": traj.get("reward")})
    print(f"episodes seen={n_ep} kept={n_kept} -> {len(rows)} step-rows (dropped {n_dropped_len} over max_length={max_length})")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="eval run dir or run-id under ~/.rllm/eval_results")
    ap.add_argument("--name", required=True, help="registered dataset name for `rllm sft`")
    ap.add_argument("--split", default="train")
    ap.add_argument("--include-incorrect", action="store_true")
    ap.add_argument("--max-length", type=int, default=131072, help="drop rows whose rendered length exceeds this (MUST match `rllm sft --max-length`)")
    args = ap.parse_args()

    run_dir = _resolve_run_dir(args.run)
    rows = build(run_dir, args.include_incorrect, args.max_length)
    if not rows:
        raise SystemExit("no rows produced")
    DatasetRegistry.register_dataset(
        args.name, rows, split=args.split, source=f"distill-from-eval:{os.path.basename(run_dir)}",
        description="thinking-SFT distillation from deepseek-v4-pro terminus2 trajectories",
        category="agentic",
    )
    print(f"registered dataset '{args.name}' (split={args.split}) with {len(rows)} rows")
    print(f"next: rllm sft {args.name} --backend fireworks --renderer-name deepseekv3_thinking --tokenize-method stepwise ...")


if __name__ == "__main__":
    main()
