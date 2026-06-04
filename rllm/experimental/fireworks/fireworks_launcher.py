from omegaconf import DictConfig

from rllm.data import Dataset
from rllm.experimental.fireworks.fireworks_backend import FireworksBackend
from rllm.experimental.unified_trainer import TrainerLauncher, UnifiedTrainer
from rllm.workflows.store import Store
from rllm.workflows.workflow import Workflow


class FireworksTrainerLauncher(TrainerLauncher):
    """
    Fireworks trainer launcher that scaffolds the fireworks backend.
    """

    def __init__(
        self,
        config: DictConfig,
        workflow_class: type[Workflow] | None = None,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        workflow_args: dict | None = None,
        store: Store | None = None,
        **kwargs,
    ):
        """Initialize the FireworksTrainerLauncher. Nothing special here, just use the parent class's init."""
        super().__init__(config, workflow_class, train_dataset, val_dataset, workflow_args, store=store, **kwargs)

    def train(self):
        trainer = None
        try:
            trainer = UnifiedTrainer(
                backend_cls=FireworksBackend,
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
            print(f"Error training with Fireworks: {e}")
            raise e
        finally:
            if trainer is not None:
                trainer.shutdown()
