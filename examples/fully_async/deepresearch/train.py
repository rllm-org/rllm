import random
import time

import hydra

from rllm.experimental.fully_async.runner import AsyncAgentTrainer
from rllm.utils import colorful_print

from .search_agent import rollout
from .tool import LocalRetrievalTool

# URL file paths for auto-refresh support
TRAIN_RAG_URL_FILE = "/data/user/rllm/examples/fully_async/deep_research/.url/rag_url"
VAL_RAG_URL_FILE = "/data/user/rllm/examples/fully_async/deep_research/.url/val_url"

# Create retriever tool with URL file for training (auto-refreshes every 5 min)
train_retriever_tool = LocalRetrievalTool(
    url_file=TRAIN_RAG_URL_FILE,
    max_results=10,
    timeout=90.0,
    format_style="original",
    max_content_length=4000,
)

# Validation uses browsecomp URL file
val_retriever_tool = LocalRetrievalTool(
    url_file=VAL_RAG_URL_FILE,
    max_results=10,
    timeout=90.0,
    format_style="original",
    max_content_length=4000,
)


def colorful_messages(messages: list[dict], reward=None) -> None:
    """Print messages with color based on role and reward."""
    assistant_color = "green" if reward is not None and reward > 0 else "red"
    for m in messages:
        role = m.get("role", "")
        if role == "assistant":
            colorful_print(m, fg=assistant_color)
        elif role == "tool":
            colorful_print(m, fg="yellow")
        elif role == "user":
            colorful_print(m, fg="cyan")
        elif role == "system":
            colorful_print(m, fg="magenta")
        else:
            colorful_print(m, fg="white")


async def rollout_fn(
    client,
    tokenizer,
    **kwargs,
):
    start_time = time.time()
    param_version_start = client.cur_version

    # Use multi-endpoint train_retriever_tool for training rollouts
    reward, metric = await rollout(client=client, tool=train_retriever_tool, **kwargs)

    trajectory = metric.pop("trajectory")
    trajectory.reward = reward
    messages = metric.pop("messages")

    # Capture timing and version info
    end_time = time.time()
    param_version_end = client.cur_version
    processing_time = end_time - start_time

    if random.random() < 0.002:
        print("\n" + "=" * 70)
        print(f"Question: {kwargs.get('question', 'N/A')}")
        print(f"Ground Truth: {kwargs.get('ground_truth', 'N/A')}")
        print(f"Reward: {reward}")
        print("=" * 70)
        colorful_messages(messages, reward=reward)
        print("=" * 70 + "\n")

    # Store metadata for statistics tracking
    # Count tool calls from messages (supports both OpenAI-style and generic dicts)
    tool_calls_count = 0
    try:
        # messages is expected to be a list of dicts
        for msg in messages or []:
            # OpenAI-style: message with 'tool_calls' list
            if isinstance(msg, dict) and isinstance(msg.get("tool_calls"), list):
                tool_calls_count += len(msg["tool_calls"])
    except Exception:
        # Be robust: if structure is unexpected, fall back to zero
        tool_calls_count = 0

    metadata = {
        "processing_time": processing_time,
        "param_version_start": param_version_start,
        "param_version_end": param_version_end,
        "param_version": param_version_end,
        "is_partial": param_version_start != param_version_end,
        "tool_calls_time": int(tool_calls_count),
    }

    metric.update(metadata)
    trajectory.metadata = metric

    return trajectory


async def val_rollout_fn(
    client,
    tokenizer,
    **kwargs,
):
    start_time = time.time()
    param_version_start = client.cur_version

    reward, metric = await rollout(client=client, tool=val_retriever_tool, **kwargs)

    trajectory = metric.pop("trajectory")
    trajectory.reward = reward
    messages = metric.pop("messages")

    # Capture timing and version info
    end_time = time.time()
    param_version_end = client.cur_version
    processing_time = end_time - start_time

    if random.random() < 0.002:
        print("\n" + "=" * 70)
        print(f"Question: {kwargs.get('question', 'N/A')}")
        print(f"Ground Truth: {kwargs.get('ground_truth', 'N/A')}")
        print(f"Reward: {reward}")
        print("=" * 70)
        colorful_messages(messages, reward=reward)
        print("=" * 70 + "\n")

    # Store metadata for statistics tracking
    # Count tool calls from messages (supports both OpenAI-style and generic dicts)
    tool_calls_count = 0
    try:
        # messages is expected to be a list of dicts
        for msg in messages or []:
            # OpenAI-style: message with 'tool_calls' list
            if isinstance(msg, dict) and isinstance(msg.get("tool_calls"), list):
                tool_calls_count += len(msg["tool_calls"])
    except Exception:
        # Be robust: if structure is unexpected, fall back to zero
        tool_calls_count = 0

    metadata = {
        "processing_time": processing_time,
        "param_version_start": param_version_start,
        "param_version_end": param_version_end,
        "param_version": param_version_end,
        "is_partial": param_version_start != param_version_end,
        "tool_calls_time": int(tool_calls_count),
    }

    metric.update(metadata)
    trajectory.metadata = metric

    return trajectory


@hydra.main(config_path="pkg://rllm.experimental.fully_async.config", config_name="fully_async_ppo_trainer", version_base=None)
def main(config):
    """
    Main entry point for DAPO fully async training.

    The config is loaded from:
    - Base: rllm/experimental/fully_async/config/fully_async_ppo_trainer.yaml
    - Which inherits from: verl/trainer/config/ppo_trainer.yaml

    You can override any config value from the command line.
    """

    trainer = AsyncAgentTrainer(
        config=config,
        rollout_fn=rollout_fn,
        val_rollout_fn=val_rollout_fn,
    )

    trainer.train()


if __name__ == "__main__":
    main()
