#!/usr/bin/env python3
"""Prepare GEO3K multimodal geometry dataset for RLLM training."""

import base64
from io import BytesIO
from typing import Iterable, List

from datasets import load_dataset
from PIL import Image

from rllm.data.dataset import DatasetRegistry

DATA_URI_PREFIX = "data:image/png;base64,"


def _serialize_images(images: Iterable[Image.Image]) -> List[str]:
    """Serialize a list of PIL images into base64 data URIs for Parquet storage."""

    serialized: list[str] = []
    for image in images or []:
        if not isinstance(image, Image.Image):
            continue
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        serialized.append(f"{DATA_URI_PREFIX}{encoded}")
    return serialized


def prepare_geo3k_data():
    """
    Prepare GEO3K dataset following RLLM conventions.

    Returns:
        Tuple of (train_dataset, test_dataset) registered with DatasetRegistry
    """
    print("📥 Loading GEO3K dataset from HuggingFace...")
    data_source = "hiyouga/geometry3k"
    dataset = load_dataset(data_source)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    print(f"✅ Dataset loaded:")
    print(f"   - Train samples: {len(train_dataset)}")
    print(f"   - Test samples: {len(test_dataset)}")

    # Instruction template based on Verl's GEO3K processing
    instruction_following = (
        "You must strictly follow this answer template: "
        "(1) write all reasoning as an internal monologue inside <think>...</think>; "
        "(2) after </think>, output exactly one line in the form `Final answer: \\boxed{value}` with the numeric solution."
    )

    formatting_example = (
        "<think>Reason step by step here....</think>\n"
        "Final answer: \\boxed{42}"
    )

    def preprocess_fn(example, idx):
        """
        Preprocess function to convert GEO3K data to RLLM format.

        This follows the pattern from verl/examples/data_preprocess/geo3k.py
        but adapts it for RLLM's simpler format requirements.
        """
        problem = example["problem"]
        answer = example["answer"]
        images = _serialize_images(example.get("images", []))

        # Create the full prompt with instruction following
        prompt = problem + "\n\n" + instruction_following + "\nExample format:\n" + formatting_example

        # Return RLLM-compatible format
        return {
            "question": prompt,  # RLLM expects 'question' field
            "ground_truth": answer,  # RLLM expects 'ground_truth' field
            "data_source": "hiyouga/geometry3k",  # Data source identifier matching verl's reward function
            "images": images,  # Serialized data URIs; reconstructed during training/inference
            "extra_info": {
                "original_problem": problem,
                "answer": answer,
                "has_images": len(images) > 0,
                "num_images": len(images),
                "formatting_example": formatting_example,
            }
        }

    print("🔄 Preprocessing datasets...")

    # Apply preprocessing to both splits
    train_dataset = train_dataset.map(preprocess_fn, with_indices=True)
    test_dataset = test_dataset.map(preprocess_fn, with_indices=True)

    # Register datasets with RLLM DatasetRegistry
    print("📋 Registering datasets with RLLM...")

    train_dataset = DatasetRegistry.register_dataset("geo3k_train", train_dataset, "train")
    test_dataset = DatasetRegistry.register_dataset("geo3k_test", test_dataset, "test")

    print("✅ Datasets registered:")
    print("   - geo3k_train (training data)")
    print("   - geo3k_test (evaluation data)")

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_geo3k_data()
    print(f"\n📊 Dataset Statistics:")
    print(f"   - Training samples: {len(train_dataset)}")
    print(f"   - Test samples: {len(test_dataset)}")

    # Show a sample to verify format
    sample = train_dataset[0]
    print(f"\n📋 Sample data format:")
    print(f"   - Question length: {len(sample['question'])} chars")
    print(f"   - Has images: {len(sample['images']) > 0}")
    print(f"   - Number of images: {len(sample['images'])}")
    print(f"   - Ground truth: {sample['ground_truth'][:100]}...")

    print(f"\n🎉 GEO3K dataset preparation complete!")
    print(f"\nNext steps:")
    print(f"1. Configure multimodal model in training config")
    print(f"2. Run training: python train_geo3k_agent.py")
    print(f"3. Test inference: python run_geo3k_agent.py")
