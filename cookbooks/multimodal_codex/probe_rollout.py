"""Pod-side rollout probe for multimodal_codex — no training loop.

Sends a single Responses API request (Codex-CLI-shaped) with one input_image
data URL through the gateway → vLLM. Verifies the multimodal path end to end
without spawning verl's actor / Ray workers, so a single-GPU debug pod can
smoke-test it.

Usage (from /data/work/rllm on the pod, after vLLM + gateway are up):

    /tmp/uv-venv/bin/python cookbooks/multimodal_codex/probe_rollout.py \\
        --gateway http://localhost:8080 \\
        --model Qwen/Qwen3.5-9B
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests


def _pick_sample_row(dataset_dir: Path) -> dict:
    """Load one row from the prepared multimodal_codex train split.

    Prefers Arrow IPC (raw bytes). Falls back to verl parquet which stores
    ``image_bytes`` as ``{"bytes": ..., "path": ""}`` inside ``extra_info``.
    """
    arrow = dataset_dir / "multimodal_codex" / "train.arrow"
    verl = dataset_dir / "multimodal_codex" / "train_verl.parquet"
    if arrow.exists():
        df = pd.read_feather(arrow)
        row = df.iloc[0].to_dict()
        return {
            "task_type": row["task_type"],
            "question": row["question"],
            "ground_truth": row["ground_truth"],
            "image_bytes": row["image_bytes"],
        }
    if verl.exists():
        df = pd.read_parquet(verl)
        extra = df.iloc[0]["extra_info"]
        img = extra["image_bytes"]
        if isinstance(img, dict):
            img = img["bytes"]
        return {
            "task_type": extra["task_type"],
            "question": extra["question"],
            "ground_truth": extra["ground_truth"],
            "image_bytes": img,
        }
    raise FileNotFoundError(f"No dataset found in {dataset_dir}")


def _make_responses_request(model: str, question: str, image_bytes: bytes) -> dict:
    """Codex-CLI-shaped Responses API payload with one input_image."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"
    return {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": question},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
        "max_output_tokens": 200,
        "temperature": 0.2,
        "stream": False,
    }


def _extract_model_text(body: dict) -> str:
    """Best-effort assistant text extraction from a Responses API response."""
    for item in body.get("output") or []:
        if item.get("type") == "message" and item.get("role") == "assistant":
            for c in item.get("content") or []:
                if c.get("type") in ("output_text", "text"):
                    return c.get("text") or ""
    # Older schema fallback
    return body.get("output_text") or ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", default="http://localhost:8080")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--dataset-dir", default="/local-ssd/rllm_home/datasets")
    args = ap.parse_args()

    print(f"[probe] dataset dir: {args.dataset_dir}")
    sample = _pick_sample_row(Path(args.dataset_dir))
    print(f"[probe] task_type: {sample['task_type']}")
    print(f"[probe] question: {sample['question'][:100]}")
    print(f"[probe] ground_truth: {sample['ground_truth']}")
    print(f"[probe] image_bytes: {len(sample['image_bytes'])} B")

    payload = _make_responses_request(args.model, sample["question"], sample["image_bytes"])
    url = f"{args.gateway}/v1/responses"
    print(f"[probe] POST {url}")

    t0 = time.time()
    r = requests.post(url, json=payload, timeout=180)
    dt = time.time() - t0
    print(f"[probe] status: {r.status_code}  time: {dt:.1f}s")

    if not r.ok:
        print(f"[probe] body: {r.text[:800]}")
        return 1

    body = r.json()
    print(f"[probe] response keys: {list(body.keys())}")

    model_text = _extract_model_text(body)
    print(f"[probe] model output: {model_text[:300]!r}")

    gt = str(sample["ground_truth"]).strip()
    hit = gt in model_text
    print(f"[probe] ground_truth={gt!r}  contained_in_output={hit}")
    print("[probe] === DONE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
