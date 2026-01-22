"""
Base classes for defining a backend protocol to be used by the UnifiedTrainer.

The protocol is async-prioritized to accommodate backends that naturally use
async operations (like Tinker). Sync backends can implement async methods
that simply wrap their sync operations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Generic, TypeVar

from omegaconf import DictConfig

from rllm.agents.agent import Episode
from rllm.data import Dataset
from rllm.experimental.common.advantage import AlgorithmConfig, compute_advantage_from_trajectory_groups
from rllm.experimental.rollout import RolloutEngine

if TYPE_CHECKING:
    from rllm.experimental.engine.unified_workflow_engine import UnifiedWorkflowEngine
    from rllm.experimental.unified_trainer import TrainerState

TDataset = TypeVar("TDataset", bound=Iterable)  # backend-specific dataset type
TBatch = TypeVar("TBatch")  # backend-specific data batch type


class BackendProtocol(ABC, Generic[TDataset, TBatch]):
    """Protocol for defining a backend.

    Attributes:
        name: Backend identifier for config lookup.
        requires_loop: Whether the backend requires an event loop (often in a different thread).
    """

    name: str = "base_backend"
    requires_loop: bool = False

    def __init__(self, config: DictConfig, **kwargs):
        """Initialize the backend.

        Args:
            config: The backend configuration.
        """
        self.config = config

    @abstractmethod
    def init_rollout_engine(self, **kwargs) -> RolloutEngine:
        """Initialize the workflow engine.

        Args:
            **kwargs: Additional arguments, including the various configurations

        Returns:
            The rollout engine.
        """
        raise NotImplementedError("Subclasses must implement this method to return a RolloutEngine.")

    @abstractmethod
    def validate_config(self) -> None:
        """Validate and setup the backend configuration."""
        pass

    @abstractmethod
    def get_dataloader(self, dataset: Dataset | None, trainer_state: TrainerState) -> TDataset:
        """Get the dataloader for the backend.

        Args:
            dataset: The dataset to get the dataloader from.
            trainer_state: The trainer state.

        Returns:
            The dataloader of type TDataset.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the backend and cleanup resources."""
        pass

    # =========================================================================
    # Required async methods for the training/validation loop
    # =========================================================================

    @abstractmethod
    async def generate_episodes(
        self,
        batch: TBatch,
        agent_workflow_engine: UnifiedWorkflowEngine,
        **kwargs,
    ) -> list[Episode]:
        """Generate episodes from the batch using the agent workflow engine.

        Args:
            batch: The input batch.
            agent_workflow_engine: The workflow engine to use.
            **kwargs: Additional arguments.

        Returns:
            List of generated episodes.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def transform_to_backend_batch(
        self,
        trainer_state: TrainerState,
        **kwargs,
    ) -> TBatch:
        """Transform rllm-native data structures to backend-specific batch.

        This is typically a sync operation as it's just data transformation.

        Args:
            trainer_state: The trainer state containing rllm-native data structures (episodes, trajectory groups, etc.).
            **kwargs: Additional arguments.

        Returns:
            Backend-specific batch.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    async def process_backend_batch(
        self,
        trainer_state: TrainerState,
        **kwargs,
    ) -> None:
        """Process the backend-specific batch.

        This may include computing log probs, critic values, running forward pass, etc.
        This is async to accommodate backends that use async operations.

        Args:
            trainer_state: The trainer state.
            **kwargs: Additional arguments.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    async def compute_advantages(
        self,
        trainer_state: TrainerState,
        algorithm_config: AlgorithmConfig,
        **kwargs,
    ) -> None:
        """Compute advantages from trajectory groups.

        Default implementation uses rLLM-native advantage computation.
        Backends can override for custom advantage computation.

        Args:
            trainer_state: The trainer state.
            algorithm_config: Algorithm configuration.
            **kwargs: Additional arguments.
        """
        assert trainer_state.trajectory_groups is not None, "Trajectory groups are not set"
        adv_metrics = compute_advantage_from_trajectory_groups(trainer_state.trajectory_groups, algorithm_config)
        trainer_state.metrics.update(adv_metrics)

    @abstractmethod
    async def update_policy(
        self,
        trainer_state: TrainerState,
        **kwargs,
    ) -> None:
        """Update the policy.

        Args:
            trainer_state: The trainer state.
            **kwargs: Additional arguments.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    # =========================================================================
    # Async hook methods for the training/validation loop
    # =========================================================================

    async def on_train_start(self, trainer_state: TrainerState) -> None:
        """Hook method called at the start of training."""
        pass

    async def on_train_end(self, trainer_state: TrainerState) -> None:
        """Hook method called at the end of training."""
        pass

    async def on_batch_start(self, trainer_state: TrainerState) -> None:
        """Hook method called at the start of a batch."""
        pass

    async def on_batch_end(self, trainer_state: TrainerState) -> None:
        """Hook method called at the end of a batch."""
        pass

    async def on_epoch_start(self, trainer_state: TrainerState) -> None:
        """Hook method called at the start of an epoch."""
        pass

    async def on_epoch_end(self, trainer_state: TrainerState) -> None:
        """Hook method called at the end of an epoch."""
        pass

    async def on_validation_start(self, trainer_state: TrainerState) -> bool:
        """Hook method called at the start of validation.

        Returns:
            bool: True if validation should proceed, False to skip.
        """
        trainer_state.is_training = False
        return True

    async def on_validation_end(self, trainer_state: TrainerState) -> None:
        """Hook method called at the end of validation."""
        trainer_state.is_training = True
