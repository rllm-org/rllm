import hydra

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.tinker.tinker_backend import TinkerBackend
from rllm.experimental.unified_trainer import UnifiedTrainer

from .math_opsd_workflow import MathOPSDWorkflow


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("deepscaler_math", "train")
    test_dataset = DatasetRegistry.load_dataset("aime2024", "test")

    try:
        # Since we use a custom backend, we will directly interface with the UnifiedTrainer instead of using the AgentTrainer
        trainer = UnifiedTrainer(
            backend_cls=TinkerBackend,
            config=config,
            workflow_class=MathOPSDWorkflow,
            train_dataset=train_dataset,
            val_dataset=test_dataset,
        )
        trainer.fit()
    except Exception as e:
        print(f"Error training Tinker Math OPSD: {e}")
        raise e
    finally:
        if trainer is not None:
            trainer.shutdown()
    print("Training completed successfully")


if __name__ == "__main__":
    main()
