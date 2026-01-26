"""
Test script for SkyRL backend with simple math workflow.

To run this test:
    python test_skyrl_simple_math.py rllm/backend=skyrl [additional config overrides]

Example with minimal config:
    python test_skyrl_simple_math.py rllm/backend=skyrl \
        trainer.policy.model.path=Qwen/Qwen2.5-1.5B-Instruct \
        trainer.placement.policy_num_gpus_per_node=1 \
        generator.num_inference_engines=1 \
        generator.inference_engine_tensor_parallel_size=1

Required configs for SkyRL:
    - trainer.policy.model.path: Model path (e.g., "Qwen/Qwen2.5-1.5B-Instruct")
    - trainer.placement.policy_num_gpus_per_node: Number of GPUs per node
    - generator.num_inference_engines: Number of inference engines
    - generator.inference_engine_tensor_parallel_size: Tensor parallel size for inference

Optional configs:
    - trainer.strategy: Training strategy (fsdp2, fsdp, megatron, deepspeed)
    - trainer.algorithm.advantage_estimator: Advantage estimator (grpo, gae, rloo, reinforce++)
    - algorithm.use_rllm: Whether to use rLLM-native advantage computation (default: false)
    - rllm.stepwise_advantage.enable: Enable stepwise advantage computation
    - rllm.stepwise_advantage.mode: Stepwise mode (broadcast or per_step)
"""

import hydra

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.skyrl.skyrl_launcher import SkyRLTrainerLauncher
from rllm.rewards.reward_fn import math_reward_fn
from rllm.workflows.simple_workflow import SimpleWorkflow


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("hendrycks_math", "train")
    test_dataset = DatasetRegistry.load_dataset("math500", "test")

    trainer = SkyRLTrainerLauncher(
        workflow_class=SimpleWorkflow,
        workflow_args={
            "reward_function": math_reward_fn,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()

