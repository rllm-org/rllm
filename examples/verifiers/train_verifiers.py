"""Train with Verifiers environments using rLLM + VERL.

Verifiers handles everything:
- Dataset loading
- Rollouts
- Scoring

We just bridge to VERL for training.

Usage:
    python -m examples.verifiers.train_verifiers \
        --env math_group \
        actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
        trainer.n_gpus_per_node=8
"""

from __future__ import annotations

import asyncio
import logging

import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


async def train_loop(config: DictConfig) -> None:
    """Main training loop."""
    from verifiers import load_environment

    from rllm.integrations.verifiers import VerifiersIntegration, transform_episodes_for_verl

    # Get env name from config
    env_name = config.get("verifiers", {}).get("env_name", "math_group")
    n_rollouts = config.actor_rollout_ref.rollout.get("n", 8)

    # Load Verifiers environment (has dataset internally)
    logger.info(f"Loading Verifiers environment: {env_name}")
    env = load_environment(env_name)

    # Setup VERL engine
    # Note: This requires VERL to be properly set up with Ray
    from rllm.trainer.verl.verl_trainer import create_verl_engine

    verl_engine = create_verl_engine(config)

    # Create integration (sets up proxy)
    integration = VerifiersIntegration(
        verl_engine=verl_engine,
        model_name=config.actor_rollout_ref.model.path,
    )

    try:
        num_epochs = config.trainer.get("total_epochs", 100)

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            # Generate rollouts (Verifiers does everything)
            episodes = await integration.generate(
                env=env,
                n_rollouts=n_rollouts,
                sampling_args={
                    "temperature": config.actor_rollout_ref.rollout.get("temperature", 0.7),
                    "max_tokens": config.data.get("max_response_length", 2048),
                },
            )

            if not episodes:
                logger.warning("No episodes generated, skipping epoch")
                continue

            # Convert to VERL format
            tokenizer = verl_engine.tokenizer
            data_proto = transform_episodes_for_verl(
                episodes=episodes,
                tokenizer=tokenizer,
                max_prompt_length=config.data.max_prompt_length,
                max_response_length=config.data.max_response_length,
            )

            # Training step
            metrics = verl_engine.train_step(data_proto)

            # Log metrics
            rewards = [e.trajectories[0].reward for e in episodes if e.trajectories]
            avg_reward = sum(rewards) / len(rewards) if rewards else 0
            accuracy = sum(1 for e in episodes if e.is_correct) / len(episodes) if episodes else 0

            logger.info(
                f"Epoch {epoch + 1}: "
                f"reward={avg_reward:.3f}, "
                f"accuracy={accuracy:.1%}, "
                f"episodes={len(episodes)}"
            )

    finally:
        integration.shutdown()


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config: DictConfig) -> None:
    """Entry point."""
    asyncio.run(train_loop(config))


if __name__ == "__main__":
    main()
