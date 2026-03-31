"""Train a math agent on GSM8K using Tinker backend + AgentCore remote runtime.

The agent runs inside an AgentCore container (strands_math_agent) and calls back
to the rllm-model-gateway for model inference. The gateway captures all traces
(token IDs, logprobs) while the agent returns only the reward.

Usage:
    # First prepare data:
    python -m examples.agentcore_math.prepare_gsm8k_data

    # Then train:
    bash examples/agentcore_math/train_agentcore_math_tinker.sh
"""

import hydra

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import UnifiedTrainer
from rllm.trainer.tinker.tinker_backend import TinkerBackend


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("gsm8k_agentcore", "train")
    test_dataset = DatasetRegistry.load_dataset("gsm8k_agentcore", "test")

    trainer = UnifiedTrainer(
        backend_cls=TinkerBackend,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    try:
        trainer.fit()
    finally:
        trainer.shutdown()


if __name__ == "__main__":
    main()
