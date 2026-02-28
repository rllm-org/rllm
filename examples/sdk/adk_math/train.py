"""Train a Google ADK math agent via the SDK agent_run_func pattern.

Uses the rLLM unified trainer with the Tinker backend.  Instead of running a
remote endpoint, the ADK agent executes **locally** inside the trainer
process.  The ``agent_run_func`` is automatically wrapped in an ``SdkWorkflow``
which handles trace collection, advantage computation, and RL training.

All LLM calls are routed through the trainer's proxy (via ``RLLM_SDK_BASE_URL``)
so that token IDs, logprobs, and messages are recorded for zero-retokenization
training.

Prerequisites:
    1. Prepare the countdown dataset:
        python -m examples.countdown.prepare_countdown_data

    2. Install Google ADK with LiteLLM:
        pip install 'google-adk[extensions]'

Usage:
    python -m examples.sdk.adk_math.train \
        rllm/backend=tinker \
        model.name=Qwen/Qwen3-8B \
        training.group_size=8 \
        validation.group_size=1 \
        training.learning_rate=2e-5 \
        data.train_batch_size=16 \
        data.max_prompt_length=2048 \
        data.max_response_length=2048 \
        rllm.trainer.test_freq=5 \
        rllm.trainer.val_before_train=false
"""

from __future__ import annotations

import logging
import os
import re
import uuid

import hydra

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer
from rllm.types import Trajectory

logger = logging.getLogger(__name__)

# Silence noisy third-party loggers
logging.getLogger("google_adk").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)

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


async def adk_math_rollout(**kwargs) -> list[Trajectory]:
    """ADK math agent that runs locally inside the trainer process.

    Receives task fields as keyword arguments (from the dataset), creates a
    Google ADK agent pointing at the trainer's inference proxy via LiteLLM,
    runs it, and returns a list of Trajectory objects with rewards.

    This function is async so the SdkWorkflow calls it directly on the event
    loop (via ``await``) instead of dispatching to a ThreadPoolExecutor.  This
    avoids the double-threading issue where ADK's sync ``runner.run()`` spawns
    its own background thread per call, which can cause thread exhaustion and
    hangs at high parallelism.

    The ``SdkWorkflowFactory`` sets ``RLLM_SDK_BASE_URL`` to the address of the
    inference server.  The proxy's ``TracingCallback`` captures token IDs and
    logprobs; the ``RLLMTrajectoryPlugin`` captures trajectory structure
    for debugging visibility.  The ``SdkWorkflow`` merges both via trace-ID
    lookup.
    """
    from google.adk.agents import Agent
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    from rllm.sdk.integrations.adk import RLLMTrajectoryPlugin
    from rllm.sdk.proxy.metadata_slug import assemble_routing_metadata, build_proxied_base_url

    # 1. Read the inference URL set by the unified trainer
    base_url = os.environ.get("RLLM_SDK_BASE_URL", "http://localhost:8089/v1")

    # 2. Encode session metadata into the URL so the proxy's TracingCallback
    #    can associate traces with the current session (same mechanism used by
    #    get_chat_client() internally).
    routing_metadata = assemble_routing_metadata()
    if routing_metadata:
        base_url = build_proxied_base_url(base_url, routing_metadata)

    # 3. Create LiteLLM model pointing at the trainer's proxy.
    #    The "openai/" prefix tells LiteLLM to use the OpenAI-compatible API.
    model = LiteLlm(
        model="openai/default",
        api_key="EMPTY",
        api_base=base_url,
    )

    # 4. Create plugin + agent + runner
    plugin = RLLMTrajectoryPlugin()

    agent = Agent(
        name="solver",
        model=model,
        instruction=SYSTEM_PROMPT,
        description="Solves math problems with step-by-step reasoning.",
    )

    runner = Runner(
        app_name="adk_math",
        agent=agent,
        session_service=InMemorySessionService(),
        plugins=[plugin],
        auto_create_session=True,
    )

    # 5. Run the agent using the async API directly on the event loop.
    #    This avoids spawning extra threads (ADK's sync runner.run() creates
    #    a background thread per call, which causes thread exhaustion at high
    #    parallelism).
    question = kwargs.get("question", "")
    user_message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=question)],
    )

    response_text = ""
    try:
        async for event in runner.run_async(
            user_id="trainer",
            session_id=f"session_{uuid.uuid4().hex[:12]}",
            new_message=user_message,
        ):
            if event.content and event.content.parts:
                text = "".join(p.text for p in event.content.parts if getattr(p, "text", None))
                if text:
                    response_text = text
    except Exception as e:
        logger.warning("Agent error (returning partial trajectory): %s", type(e).__name__)

    # 6. Get trajectory and assign reward
    try:
        traj = plugin.get_trajectory()
    except ValueError:
        traj = Trajectory(name="solver", steps=[], reward=0.0)
    traj.reward = _check_countdown(kwargs, response_text)

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
        agent_run_func=adk_math_rollout,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        backend="tinker",
    )
    trainer.train()


if __name__ == "__main__":
    main()
