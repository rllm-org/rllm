"""Prepare multimodal_codex dataset: procedurally-generated visual tasks.

Renders task images ONCE at prepare-time and stores raw PNG bytes in the
parquet row. Rollout picks them up from ``task.metadata['image_bytes']`` and
``MultimodalCodexHarness.write_configs`` uploads to sandbox.

Seed is fixed (42) so re-running yields the same dataset.
"""

from __future__ import annotations

import random

from rllm.data.dataset import DatasetRegistry

from cookbooks.multimodal_codex.renderer import _render_image_from_params
from cookbooks.multimodal_codex.tasks import GENERATORS

import os as _os

TRAIN_SIZE = int(_os.environ.get("MM_CODEX_TRAIN_SIZE", "256"))
TEST_SIZE = int(_os.environ.get("MM_CODEX_TEST_SIZE", "8"))
SEED = 42

# Absolute path inside sandbox — Codex CLI --image accepts absolute paths
# directly, so we don't need to know the sandbox workdir.
IMAGE_PATH = "/tmp/multimodal_codex_input.png"

# Codex CLI works better with a short lead-in that reminds the model an image
# is attached; the harness passes ``--image`` so the wire has the file, but
# the model needs to be told to LOOK.
PROMPT_TEMPLATE = "Look at the image at {image_path} and answer:\n\n{question}\n\nRespond with just the answer, no explanation."


def _build_row(idx: int, rng: random.Random) -> dict:
    gen = GENERATORS[idx % len(GENERATORS)]
    row = gen(rng, idx)
    png_bytes = _render_image_from_params(row)

    return {
        "uid": row["uid"],
        "task_type": row["task_type"],
        "data_source": "multimodal_codex",
        "question": row["question"],
        "instruction": PROMPT_TEMPLATE.format(image_path=IMAGE_PATH, question=row["question"]),
        "prompt": PROMPT_TEMPLATE.format(image_path=IMAGE_PATH, question=row["question"]),
        "ground_truth": row["ground_truth"],
        # Raw PNG bytes; MultimodalCodexHarness uploads to sandbox at image_file.
        "image_bytes": png_bytes,
        "image_file": IMAGE_PATH,
        # Keep original generator params for reproducibility / evaluator inspection.
        "generator_params": {k: v for k, v in row.items() if k not in {"uid", "task_type", "question", "ground_truth"}},
    }


def prepare_multimodal_codex_data() -> None:
    rng = random.Random(SEED)
    train_rows = [_build_row(i, rng) for i in range(TRAIN_SIZE)]
    test_rows = [_build_row(i + TRAIN_SIZE, rng) for i in range(TEST_SIZE)]

    DatasetRegistry.register_dataset(
        "multimodal_codex",
        train_rows,
        "train",
        description="Procedurally-generated multimodal chart/shape/table reading tasks for Codex CLI RL.",
        category="multimodal",
    )
    DatasetRegistry.register_dataset(
        "multimodal_codex",
        test_rows,
        "test",
        description="Held-out eval set for multimodal_codex.",
        category="multimodal",
    )


if __name__ == "__main__":
    prepare_multimodal_codex_data()
    print(f"Registered multimodal_codex: {TRAIN_SIZE} train + {TEST_SIZE} test rows")
