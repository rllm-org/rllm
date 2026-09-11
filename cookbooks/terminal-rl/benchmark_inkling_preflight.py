"""Compare cached/uncached Inkling preflight on canonical JSONL rows, offline.

Uses a locally cached tokenizer and runs every prefix check in both variants.
No provider client or training job is created. Example:

    python cookbooks/terminal-rl/benchmark_inkling_preflight.py \
        --train-file train.jsonl --output comparison.json

Requires an installed renderers version that provides InklingRenderer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

from renderers.configs import InklingRendererConfig
from renderers.inkling import InklingRenderer
from transformers import AutoTokenizer

from rllm.trainer.sft import tinker_dataset as td


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--tokenizer", default="thinkingmachines/Inkling-Small")
    parser.add_argument("--max-length", type=int, default=262144)
    parser.add_argument("--reasoning-effort", type=float, default=0.99)
    args = parser.parse_args()
    with args.train_file.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not 1 <= args.sample_count <= len(rows):
        parser.error(f"--sample-count must be between 1 and {len(rows)}")
    indices = [0] if args.sample_count == 1 else [round(i * (len(rows) - 1) / (args.sample_count - 1)) for i in range(args.sample_count)]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    renderer = InklingRenderer(tokenizer, InklingRendererConfig(reasoning_effort=args.reasoning_effort))
    # Warm tokenizer/runtime initialization before timing either variant.
    td.conversation_to_datum([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}], renderer, args.max_length)
    cache_renderer = td._preflight_renderer
    records = []
    totals = {"uncached": 0.0, "cached": 0.0}
    for sample_index, row_index in enumerate(indices):
        row = rows[row_index]
        hashes = []
        variants = [("uncached", lambda value: value), ("cached", cache_renderer)]
        # Alternate first variant to limit systematic warm-up/order bias.
        for variant, wrapper in variants[:: 1 if sample_index % 2 == 0 else -1]:
            with patch.object(td, "_preflight_renderer", wrapper):
                started = time.perf_counter()
                datum = td.conversation_to_datum(row["messages"], renderer, args.max_length, tools=row.get("tools"), overlength_policy="error", validate_prefix_stability=True)
                elapsed = time.perf_counter() - started
            payload = {
                "input_tokens": datum.model_input.to_ints(),
                "targets": datum.loss_fn_inputs["target_tokens"].data,
                "weights": datum.loss_fn_inputs["weights"].data,
            }
            digest = hashlib.sha256(json.dumps(payload).encode()).hexdigest()
            hashes.append(digest)
            totals[variant] += elapsed
            record = {"variant": variant, "row_index": row_index, "input_tokens": datum.model_input.length, "seconds": elapsed, "datum_sha256": digest}
            records.append(record)
            print(json.dumps(record), flush=True)
        if len(set(hashes)) != 1:
            raise RuntimeError(f"Cache changed input tokens, targets, or loss weights for row {row_index}")
    summary = {"rows_compared": len(indices), "seconds": totals, "speedup": totals["uncached"] / totals["cached"]}
    result = {
        "summary": summary,
        "versions": {name: version(name) for name in ("renderers", "tinker", "tinker-cookbook", "transformers")},
        "records": records,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
