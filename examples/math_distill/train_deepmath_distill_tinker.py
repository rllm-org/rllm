import hydra
import tinker
from omegaconf import DictConfig
from transformers import AutoTokenizer

from rllm.data.dataset import DatasetRegistry
from rllm.engine.rollout.tinker_engine import TinkerEngine
from rllm.experimental.unified_trainer import AgentTrainer
from rllm.rewards.reward_fn import math_reward_fn
from rllm.workflows.distillation_workflow import DistillationWorkflow


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    """Main training function for simple math distillation."""
    # Load datasets
    train_dataset = DatasetRegistry.load_dataset("deepmath_opd", "train")
    test_dataset = DatasetRegistry.load_dataset("deepmath_opd", "test")

    teacher_model = "Qwen/Qwen3-32B"
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model)
    teacher_service_client = tinker.ServiceClient()
    teacher_sampling_client = teacher_service_client.create_sampling_client(base_model=teacher_model)
    teacher_engine = TinkerEngine(
        model_name=teacher_model,
        tokenizer=teacher_tokenizer,
        service_client=teacher_service_client,
        sampling_client=teacher_sampling_client,
        bypass_render_with_parser=True,
    )

    trainer = AgentTrainer(
        workflow_class=DistillationWorkflow,
        workflow_args={
            "reward_function": math_reward_fn,
            "teacher_engine": teacher_engine,
            "shared_tokenizer": True,
            "clip_min": -5.0,
            "clip_max": 5.0,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        backend="tinker",
    )

    trainer.train()


if __name__ == "__main__":
    main()
