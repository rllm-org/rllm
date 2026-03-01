from omegaconf import DictConfig

from rllm.data import Dataset
from rllm.trainer.tinker.tinker_backend import TinkerBackend
from rllm.experimental.unified_trainer import TrainerLauncher, UnifiedTrainer
from rllm.workflows.workflow import Workflow


class TinkerTrainerLauncher(TrainerLauncher):
    """
    Tinker trainer launcher that scaffolds the tinker backend.
    """

    def __init__(
        self,
        config: DictConfig,
        workflow_class: type[Workflow],
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        workflow_args: dict | None = None,
        **kwargs,
    ):
        """Initialize the TinkerTrainerLauncher. Nothing special here, just use the parent class's init."""
        super().__init__(config, workflow_class, train_dataset, val_dataset, workflow_args, **kwargs)

    def train(self):
        trainer = None
        try:
            trainer = UnifiedTrainer(backend_cls=TinkerBackend, config=self.config, workflow_class=self.workflow_class, train_dataset=self.train_dataset, val_dataset=self.val_dataset, workflow_args=self.workflow_args, **self.kwargs)
            trainer.fit()
        except Exception as e:
            print(f"Error training Tinker: {e}")
            raise e
        finally:
            if trainer is not None:
                trainer.shutdown()
