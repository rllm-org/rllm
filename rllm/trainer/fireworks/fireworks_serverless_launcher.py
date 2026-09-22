"""Launcher for pooled Fireworks serverless training."""

from rllm.trainer.fireworks.fireworks_serverless_backend import (
    FireworksServerlessBackend,
)
from rllm.trainer.unified_trainer import TrainerLauncher, UnifiedTrainer


class FireworksServerlessTrainerLauncher(TrainerLauncher):
    def train(self):
        trainer = None
        try:
            trainer = UnifiedTrainer(
                backend_cls=FireworksServerlessBackend,
                config=self.config,
                workflow_class=self.workflow_class,
                train_dataset=self.train_dataset,
                val_dataset=self.val_dataset,
                workflow_args=self.workflow_args,
                store=self.store,
                **self.kwargs,
            )
            trainer.fit()
        except Exception as e:
            print(f"Error training with Fireworks serverless: {e}")
            raise
        finally:
            if trainer is not None:
                trainer.shutdown()
