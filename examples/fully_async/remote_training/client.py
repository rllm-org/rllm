"""
Remote Agent Client
===================

Run this on your local desktop. It connects to a remote TrainingServer,
runs agent code locally, and streams trajectories to the server for training.

Usage:
    python client.py \
        --server-url https://my-gpu-server:8000 \
        --model-name Qwen/Qwen2.5-7B \
        --dataset-name my_dataset \
        --dataset-split train

This example implements a simple single-turn QA agent that:
1. Sends a question to the model via the remote server
2. Computes a reward based on the model's answer
3. Submits the trajectory for training
"""

import argparse

from rllm.experimental.fully_async.protocol import Trajectory
from rllm.experimental.fully_async.remote import AgentTrainerClient


# ── Your agent code ──────────────────────────────────────────────────────────


async def rollout_fn(client, tokenizer, **kwargs):
    """Example rollout function.

    This is where your agent logic lives. The ``client`` is a
    :class:`RemoteRolloutClient` that talks to the remote server's SGLang
    inference engine. Use ``client.chat_completion()`` exactly as you would
    with the colocated ``RolloutClient``.

    Args:
        client: RemoteRolloutClient instance.
        tokenizer: HuggingFace tokenizer (loaded locally).
        **kwargs: A single row from your dataset.

    Returns:
        A :class:`Trajectory` with reward and metadata.
    """
    question = kwargs.get("question", "")
    ground_truth = kwargs.get("ground_truth", "")

    # ── Step 1: Generate a response ──
    messages = [{"role": "user", "content": question}]
    response_msg, output = await client.chat_completion(
        messages,
        sampling_params={"temperature": 0.7, "max_new_tokens": 1024},
    )

    # ── Step 2: Compute reward ──
    answer = response_msg.get("content", "")
    # Simple exact-match reward (replace with your own reward logic!)
    reward = 1.0 if ground_truth.lower().strip() in answer.lower().strip() else 0.0

    # ── Step 3: Build trajectory ──
    sequence = output.to_sequence()
    trajectory = Trajectory(sequences=[sequence], reward=reward)
    trajectory.metadata = {
        "question": question[:100],
        "reward": reward,
        "param_version": client.cur_version,
    }

    return trajectory


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="rLLM Remote Agent Client")
    parser.add_argument("--server-url", type=str, required=True, help="URL of the remote TrainingServer")
    parser.add_argument("--model-name", type=str, required=True, help="HuggingFace model name for tokenizer")
    parser.add_argument("--dataset-name", type=str, required=True, help="Dataset name (registered in DatasetRegistry)")
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--n", type=int, default=4, help="Rollouts per prompt (GRPO group size)")
    parser.add_argument("--max-concurrency", type=int, default=128)
    args = parser.parse_args()

    # Load dataset
    from rllm.data.dataset import DatasetRegistry

    dataset = DatasetRegistry.load_dataset(args.dataset_name, args.dataset_split)
    if dataset is None:
        raise ValueError(f"Failed to load dataset '{args.dataset_name}' split '{args.dataset_split}'")
    print(f"Loaded dataset with {len(dataset)} examples")

    # Create the client trainer and run
    trainer = AgentTrainerClient(
        server_url=args.server_url,
        rollout_fn=rollout_fn,
        model_name=args.model_name,
        dataset=dataset,
        n=args.n,
        max_concurrency=args.max_concurrency,
        # Override training config from the client side
        config_overrides={
            "trainer.project_name": "remote-training-example",
            "trainer.experiment_name": "run-1",
            "algorithm.adv_estimator": "grpo",
        },
    )

    trainer.train()


if __name__ == "__main__":
    main()
