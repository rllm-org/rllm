#!/usr/bin/env python3
"""Launch a Fireworks SFT of deepseek-v4-flash on distilled terminus2 traces.

Self-contained: every deepseek-specific var is baked in via SFTSpec.overrides
(the `rllm sft` CLI can't set renderer_name / tokenizer_model / training-shape,
and its defaults — role_colon + qwen tokenizer — would silently corrupt a
deepseek think-model). Verified pieces:

  - renderer_name = deepseekv3_thinking  → correct <｜User｜>/<｜Assistant｜>/<think>
    format, and it TRAINS the think block (weight>0).
  - tokenize_method = stepwise (LAST_ASSISTANT_MESSAGE) → required because that
    renderer lacks the sequence-extension property; the dataset is one row per step.
  - tokenizer_model = a LOCAL v4 tokenizer dir (tokenizer.json only, no model
    config so transformers can load it) — its <think> id is 128821, which R1's
    tokenizer gets WRONG (128798), so R1 must NOT be used.
  - shape = deepseek-v4-flash-256k-lora (auto-selected via the Fireworks SDK).

Run:
  python sft_deepseek_flash.py <distill_dataset_name>
Needs FIREWORKS_API_KEY.
"""

from __future__ import annotations

import argparse
import os
import shutil

MODEL = "accounts/fireworks/models/deepseek-v4-flash"
SHAPE = "accounts/fireworks/trainingShapes/deepseek-v4-flash-256k-lora"
RENDERER = "deepseekv3_thinking"
TOKENIZER_REPO = "deepseek-ai/DeepSeek-V4-Flash"
TOKENIZER_DIR = os.path.expanduser("~/.rllm/tokenizers/deepseek-v4-flash")


def ensure_tokenizer() -> str:
    """Materialize a config-free v4 tokenizer dir so get_tokenizer() can load it."""
    if os.path.exists(os.path.join(TOKENIZER_DIR, "tokenizer.json")):
        return TOKENIZER_DIR
    from huggingface_hub import hf_hub_download

    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    for fn in ("tokenizer.json", "tokenizer_config.json"):
        try:
            shutil.copy(hf_hub_download(TOKENIZER_REPO, fn), os.path.join(TOKENIZER_DIR, fn))
        except Exception as e:  # noqa: BLE001
            print(f"  (skip {fn}: {str(e)[:60]})")
    return TOKENIZER_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="registered distill dataset (from build_distill_dataset.py)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=32768)
    ap.add_argument("--replica-count", type=int, default=1)
    ap.add_argument("--experiment", default=None)
    args = ap.parse_args()

    if not os.environ.get("FIREWORKS_API_KEY"):
        raise SystemExit("FIREWORKS_API_KEY not set")

    tok_dir = ensure_tokenizer()
    print(f"tokenizer dir: {tok_dir}")

    from rllm.data import DatasetRegistry
    from rllm.trainer.agent_sft_trainer import AgentSFTTrainer
    from rllm.trainer.sft import SFTSpec

    train = DatasetRegistry.load_dataset(args.dataset, "train")
    if train is None:
        raise SystemExit(f"dataset '{args.dataset}' (split train) not found — run build_distill_dataset.py first")
    val = DatasetRegistry.load_dataset(args.dataset, "test")  # optional

    spec = SFTSpec(
        model=MODEL,
        train_dataset=train,
        val_dataset=val,
        lr=args.lr,
        lr_schedule="cosine",
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        tokenize_method="stepwise",  # -> LAST_ASSISTANT_MESSAGE
        lora_rank=args.lora_rank,
        save_freq=20,
        val_freq=10,
        project="terminal-rl-sft",
        experiment=args.experiment or f"dsv4flash-distill-{args.dataset}",
        # The escape hatch: deep-merged last over the Fireworks template, so it
        # overrides the qwen/role_colon defaults with deepseek-correct values.
        overrides={
            "model": {"tokenizer_model": tok_dir},
            "data": {"renderer_name": RENDERER},
            "fireworks_config": {
                "policy_trainer_shape_id": SHAPE,
                "policy_trainer_replica_count": args.replica_count,
            },
        },
    )

    print(f"launching SFT: model={MODEL} shape={SHAPE} renderer={RENDERER} rows={len(train)}")
    AgentSFTTrainer(spec, backend="fireworks").train()


if __name__ == "__main__":
    main()
