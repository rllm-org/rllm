from collections.abc import Callable

from omegaconf import DictConfig

from rllm.data import Dataset
from rllm.experimental.tinker.tinker_backend import TinkerBackend
from rllm.experimental.unified_trainer import TrainerLauncher, UnifiedTrainer
from rllm.workflows.workflow import Workflow


class TinkerTrainerLauncher(TrainerLauncher):
    """
    Tinker trainer launcher that scaffolds the tinker backend.
    """

    def __init__(
        self,
        config: DictConfig,
        workflow_class: type[Workflow] | None = None,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        workflow_args: dict | None = None,
        agent_run_func: Callable | None = None,
        **kwargs,
    ):
        """Initialize the TinkerTrainerLauncher. Nothing special here, just use the parent class's init."""
        super().__init__(config, workflow_class, train_dataset, val_dataset, workflow_args, agent_run_func=agent_run_func, **kwargs)

    def train(self):
        trainer = None
        try:
            trainer = UnifiedTrainer(backend_cls=TinkerBackend, config=self.config, workflow_class=self.workflow_class, train_dataset=self.train_dataset, val_dataset=self.val_dataset, workflow_args=self.workflow_args, agent_run_func=self.agent_run_func, **self.kwargs)
            trainer.fit()
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
        except Exception as e:
            print(f"Error training Tinker: {e}")
            raise e
        finally:
            if trainer is not None:
                trainer.shutdown()
