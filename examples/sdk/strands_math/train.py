"""Train a Strands math agent via the SDK agent_run_func pattern.

Uses the rLLM unified trainer with the Tinker backend.  Instead of running a
remote endpoint, the Strands agent executes **locally** inside the trainer
process.  The ``agent_run_func`` is automatically wrapped in an ``SdkWorkflow``
which handles trace collection, advantage computation, and RL training.

All LLM calls are routed through the trainer's proxy (via ``RLLM_SDK_BASE_URL``)
so that token IDs, logprobs, and messages are recorded for zero-retokenization
training.

The ``RLLMTrajectoryHookProvider`` captures trajectory structure and provides
debugging visibility.  The ``SdkWorkflow`` merges these user-provided
trajectories with the proxy-captured traces (which contain token IDs and
logprobs) using trace-ID-based lookup.

Prerequisites:
    1. Prepare the countdown dataset:
        python -m examples.countdown.prepare_countdown_data

    2. Install Strands:
        pip install 'strands-agents[openai]'

Usage:
    python -m examples.sdk.strands_math.train \\
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

from __future__ import annotations

import logging
import os
import re

import hydra

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer
from rllm.types import Trajectory

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a math solver. Solve the given problem step by step.
Show your reasoning clearly, then provide the final answer inside
<answer>...</answer> tags.

For example: <answer> 42 </answer>"""


# ---------------------------------------------------------------------------
# Reward helpers
# ---------------------------------------------------------------------------


def _extract_answer(text: str) -> str | None:
    """Extract content from <answer>...</answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _check_countdown(task: dict, response_text: str) -> float:
    """Countdown reward: 1.0 if the extracted answer matches the target."""
    answer_str = _extract_answer(response_text)
    if answer_str is None:
        return 0.0
    target = task.get("target") or task.get("ground_truth")
    if target is None:
        return 0.0
    try:
        if not re.match(r"^[\d+\-*/().\s]+$", answer_str):
            return 0.0
        result = eval(answer_str)  # noqa: S307
        return 1.0 if abs(float(result) - float(target)) < 1e-6 else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Agent run function (agent_run_func pattern)
# ---------------------------------------------------------------------------


def strands_math_rollout(**kwargs) -> list[Trajectory]:
    """Strands math agent that runs locally inside the trainer process.

    Receives task fields as keyword arguments (from the dataset), creates a
    Strands agent pointing at the trainer's inference proxy, runs it, and
    returns a list of Trajectory objects with rewards.

    The ``SdkWorkflowFactory`` sets ``RLLM_SDK_BASE_URL`` to the address of the
    inference server.  The proxy's ``TracingCallback`` captures token IDs and
    logprobs; the ``RLLMTrajectoryHookProvider`` captures trajectory structure
    for debugging visibility.  The ``SdkWorkflow`` merges both via trace-ID
    lookup.
    """
    from strands import Agent
    from strands.models.litellm import LiteLLMModel

    from rllm.sdk.integrations.strands import RLLMTrajectoryHookProvider
    from rllm.sdk.proxy.metadata_slug import assemble_routing_metadata, build_proxied_base_url

    # 1. Read the inference URL set by the unified trainer
    base_url = os.environ.get("RLLM_SDK_BASE_URL", "http://localhost:8089/v1")

    # 2. Encode session metadata into the URL so the proxy's TracingCallback
    #    can associate traces with the current session (same mechanism used by
    #    get_chat_client() internally).
    routing_metadata = assemble_routing_metadata()
    if routing_metadata:
        base_url = build_proxied_base_url(base_url, routing_metadata)

    model = LiteLLMModel(
        client_args={
            "api_base": base_url,
            "api_key": "EMPTY",
        },
        model_id="openai/default",
    )

    # 3. Create hook provider + agent
    hook_provider = RLLMTrajectoryHookProvider()
    agent = Agent(
        name="solver",
        model=model,
        system_prompt=SYSTEM_PROMPT,
        hooks=[hook_provider],
        callback_handler=None,
    )

    # 4. Run the agent (catch max-tokens and other recoverable errors)
    question = kwargs.get("question", "")
    try:
        result = agent(question)
        response_text = str(result)
    except Exception as e:
        # Strands raises MaxTokensReachedException when the model hits the
        # token limit.  Return partial trajectory with 0 reward.
        logger.warning("Agent error (returning partial trajectory): %s", type(e).__name__)
        response_text = ""

    # 5. Get trajectory and assign reward
    try:
        traj = hook_provider.get_trajectory()
    except ValueError:
        # No LLM calls completed (agent failed before first model response)
        traj = Trajectory(name="solver", steps=[], reward=0.0, output=response_text or None)
    traj.reward = _check_countdown(kwargs, response_text)

    logger.info(f"Trajectory: {traj.model_dump_json(indent=2)}")

    logger.info("reward=%.1f steps=%d response=%s", traj.reward, len(traj.steps), response_text[:100])
    return [traj]


# ---------------------------------------------------------------------------
# Trainer entry point
# ---------------------------------------------------------------------------


@hydra.main(
    config_path="pkg://rllm.experimental.config",
    config_name="unified",
    version_base=None,
)
def main(config) -> None:
    train_dataset = DatasetRegistry.load_dataset("countdown", "train")
    test_dataset = DatasetRegistry.load_dataset("countdown", "test")

    assert train_dataset, "Train dataset not found. Run: python -m examples.countdown.prepare_countdown_data"
    assert test_dataset, "Test dataset not found. Run: python -m examples.countdown.prepare_countdown_data"

    trainer = AgentTrainer(
        agent_run_func=strands_math_rollout,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        backend="tinker",
    )
    trainer.train()


if __name__ == "__main__":
    main()
