"""
SDK + Unified Trainer + Tinker backend example.

Demonstrates using an SDK `agent_run_func` (instead of a Workflow class) with
the unified trainer and the Tinker backend.  The unified trainer automatically
wraps the agent function in an `SdkWorkflow` adapter so you get the full
pipeline: rejection sampling, advantage computation, and Tinker RL training.

Prerequisites:
    1. Prepare the countdown dataset:
        python -m examples.countdown.prepare_countdown_data

    2. Start a Tinker service (or use local mode with `tinker_base_url: null`).

Usage:
    python -m examples.sdk.unified_tinker.train_countdown_sdk_tinker \\
        rllm/backend=tinker \\
        model.name=Qwen/Qwen3-8B \\
        training.group_size=8 \\
        validation.group_size=1 \\
        training.learning_rate=2e-5 \\
        data.train_batch_size=16 \\
        data.max_prompt_length=2048 \\
        data.max_response_length=2048 \\
        rllm.trainer.test_freq=5 \\
        rllm.trainer.val_before_train=true
"""

import os

import hydra

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer
from rllm.rewards.countdown_reward import countdown_reward_fn
from rllm.sdk.shortcuts import get_chat_client


def countdown_rollout(**kwargs):
    """SDK agent function for the countdown task.

    Receives task fields as keyword arguments (from the dataset), makes one LLM
    call via the SDK traced client, and returns a scalar reward.

    The ``SdkWorkflowFactory`` sets ``RLLM_SDK_BASE_URL`` to the address of the
    inference server wrapping the training model (InferenceAPIServer for tinker,
    LiteLLM proxy for verl).  We read it here so the function remains portable
    across backends.
    """
    question = kwargs["question"]

    # The unified trainer sets RLLM_SDK_BASE_URL to the inference endpoint.
    base_url = os.environ.get("RLLM_SDK_BASE_URL", "http://localhost:8089/v1")
    client = get_chat_client(base_url=base_url, api_key="EMPTY")

    response = client.chat.completions.create(
        model="default",  # model name is handled by the proxy / inference server
        messages=[{"role": "user", "content": question}],
    )

    response_text = response.choices[0].message.content or ""
    reward = countdown_reward_fn(kwargs, response_text).reward
    return reward * 1.0


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("countdown", "train")
    test_dataset = DatasetRegistry.load_dataset("countdown", "test")

    assert train_dataset, "Train dataset not found. Run: python -m examples.countdown.prepare_countdown_data"
    assert test_dataset, "Test dataset not found. Run: python -m examples.countdown.prepare_countdown_data"

    trainer = AgentTrainer(
        agent_run_func=countdown_rollout,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        backend="tinker",
    )
    trainer.train()


if __name__ == "__main__":
    main()
