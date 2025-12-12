"""
Base classes for defining a backend protocol to be used by the UnifiedTrainer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Generic, TypeVar

from omegaconf import DictConfig

from rllm.agents.agent import Episode, TrajectoryGroup
from rllm.data import Dataset
from rllm.engine.rollout import RolloutEngine
from rllm.trainer.common.advantage import AlgorithmConfig, compute_advantage_from_trajectory_groups

if TYPE_CHECKING:
    from rllm.experimental.unified_trainer import TrainerState

TDataset = TypeVar("TDataset", bound=Iterable)  # backend-specific dataset type
TBatch = TypeVar("TBatch")  # backend-specific batch type


class BackendProtocol(ABC, Generic[TDataset, TBatch]):
    """Protocol for defining a backend.

    Attributes:
        - requires_loop: Whether the backend requires an event loop (often in a different thread).
        - requires_preprocess_dataset: Whether the backend requires the dataset to be preprocessed.
    """

    name: str = "base_backend"
    requires_loop: bool = False
    requires_preprocess_dataset: bool = False

    def __init__(self, config: DictConfig, **kwargs):
        """Initialize the backend.

        Args:
            config: The backend configuration.
        """
        self.config = config

    @abstractmethod
    def init_rollout_engine(self) -> RolloutEngine:
        """Initialize the workflow engine.

        Returns:
            The rollout engine.
        """
        raise NotImplementedError("Subclasses must implement this method to return a UnifiedWorkflowEngine.")

    @abstractmethod
    def validate_config(self) -> None:
        """Validate and setup the backend configuration.

        Args:
            config: The backend configuration.
        """
        pass

    @abstractmethod
    def get_dataloader(self, dataset: Dataset) -> TDataset:
        """Preprocess the dataset for the backend.

        Args:
            dataset: The dataset to preprocess.

        Returns:
            The preprocessed dataset of type TDataset.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the backend.

        Args:
            config: The backend configuration.
        """
        pass

    # =========================================================================
    # Required methods for the training/validation loop (must implement)
    # =========================================================================

    @abstractmethod
    def generate_episodes(self, batch: TBatch, **kwargs) -> list[Episode]:
        """Generate episodes from the batch."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def transform_trajectory_groups_to_backend_batch(self, trajectory_groups: list[TrajectoryGroup], **kwargs) -> TBatch:
        """Transform trajectory groups to backend-specific batch."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def process_backend_batch(self, batch: TBatch, **kwargs) -> TBatch:
        """Process the backend-specific batch."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def compute_advantages(self, trainer_state: TrainerState, algorithm_config: AlgorithmConfig, **kwargs) -> None:
        """Compute advantages from trajectory groups and (optionally) info from the backend batch (e.g. entropy)."""
        assert trainer_state.trajectory_groups is not None, "Trajectory groups are not set"
        compute_advantage_from_trajectory_groups(trainer_state.trajectory_groups, algorithm_config)
        return

    @abstractmethod
    def update_policy(self, trainer_state: TrainerState) -> None:
        """Update the policy."""
        raise NotImplementedError("Subclasses must implement this method.")

    # =========================================================================
    # Hook methods for the training/validation loop
    # =========================================================================

    @abstractmethod
    def on_train_start(self, trainer_state: TrainerState) -> None:
        """Hook method called at the start of training."""
        pass

    @abstractmethod
    def on_train_end(self, trainer_state: TrainerState) -> None:
        """Hook method called at the end of training."""
        pass

    @abstractmethod
    def on_batch_start(self, trainer_state: TrainerState) -> None:
        """Hook method called at the start of a batch."""
        pass

    @abstractmethod
    def on_batch_end(self, trainer_state: TrainerState) -> None:
        """Hook method called at the end of a batch."""
        pass

    @abstractmethod
    def on_epoch_start(self, trainer_state: TrainerState) -> None:
        """Hook method called at the start of an epoch."""
        pass

    @abstractmethod
    def on_epoch_end(self, trainer_state: TrainerState) -> None:
        """Hook method called at the end of an epoch."""
        pass

    @abstractmethod
    def on_validation_start(self, trainer_state: TrainerState) -> None:
        """Hook method called at the start of validation."""
        trainer_state.is_training = False

    @abstractmethod
    def on_validation_end(self, trainer_state: TrainerState) -> None:
        """Hook method called at the end of validation."""
        trainer_state.is_training = True
