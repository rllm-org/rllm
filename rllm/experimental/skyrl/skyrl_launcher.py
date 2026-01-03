"""
SkyRL launcher implementation for the UnifiedTrainer.

This launcher handles the necessary setup for the SkyRL backend, including
initializing Ray, creating inference engines, generators, and trainers.

Follows the BasePPOExp pattern from skyrl_train.entrypoints.main_base.
"""

import os
import socket
from pprint import pprint

import ray
from omegaconf import DictConfig, OmegaConf
from ray.util.placement_group import placement_group, PlacementGroup
from transformers import AutoTokenizer

from rllm.data import Dataset
from rllm.experimental.unified_trainer import TrainerLauncher, UnifiedTrainer
from rllm.experimental.skyrl.skyrl_backend import SkyRLBackend
from rllm.workflows.workflow import Workflow

# Import SkyRL components
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.generators.base import GeneratorInterface
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.utils.utils import get_ray_pg_ready_with_timeout
from skyrl_train.utils.constants import SKYRL_RAY_PG_TIMEOUT_IN_S
from skyrl_train.entrypoints.main_base import (
    create_ray_wrapped_inference_engines_from_config,
    create_remote_inference_engines_from_config,
)
from loguru import logger


class SkyRLExp:
    """
    SkyRL experiment class for UnifiedTrainer.

    Follows the BasePPOExp pattern from skyrl_train.entrypoints.main_base,
    but adapted for use with UnifiedTrainer instead of direct RayPPOTrainer.
    """

    def __init__(self, cfg: DictConfig, workflow_class: type[Workflow], workflow_args: dict | None = None, train_dataset: Dataset | None = None, val_dataset: Dataset | None = None, **kwargs):
        """
        Initializes a SkyRL UnifiedTrainer experiment.

        The `cfg` passed here will be the final config from Hydra, including CLI overrides.

        Args:
            cfg: Training configuration
            workflow_class: Workflow class for training
            workflow_args: Optional workflow arguments
            train_dataset: Optional training dataset (if None, uses config.data paths)
            val_dataset: Optional validation dataset (if None, uses config.data paths)
            **kwargs: Additional arguments
        """
        self.cfg = cfg
        self.workflow_class = workflow_class
        self.workflow_args = workflow_args or {}
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.kwargs = kwargs
        self.tokenizer = self.get_tokenizer()
        self.colocate_pg = self.get_colocate_pg()

    @staticmethod
    def get_cfg_as_str(dict_cfg: DictConfig) -> str:
        """Convert config to YAML string."""
        return OmegaConf.to_yaml(dict_cfg)

    def get_tokenizer(self, padding_side="left"):
        """Initializes a tokenizer for the given model."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.trainer.policy.model.path,
            trust_remote_code=True,
            use_fast=not self.cfg.trainer.get("disable_fast_tokenizer", False),
        )
        tokenizer.padding_side = padding_side
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    def get_colocate_pg(self, timeout: int = SKYRL_RAY_PG_TIMEOUT_IN_S) -> PlacementGroup | None:
        """Initializes a placement group for colocated training.

        A single placement group that packs all the inference engines together is created.

        Args:
            timeout (int): The timeout for the placement group to be ready.

        Returns:
            PlacementGroup: The placement group for colocated training, or None.
        """
        if self.cfg.trainer.placement.get("colocate_all", False):
            pg = placement_group(
                [{"GPU": 1, "CPU": 1}]
                * self.cfg.generator.num_inference_engines
                * self.cfg.generator.inference_engine_tensor_parallel_size
                * self.cfg.generator.inference_engine_pipeline_parallel_size
                * self.cfg.generator.inference_engine_data_parallel_size,
                strategy="PACK",
            )
            get_ray_pg_ready_with_timeout(pg, timeout=timeout)
            return pg
        else:
            return None

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """Initializes the generator.

        Returns:
            GeneratorInterface: The generator.
        """
        from skyrl_train.generators.skyrl_gym_generator import SkyRLGymGenerator

        return SkyRLGymGenerator(
            generator_cfg=cfg.generator,
            skyrl_gym_cfg=cfg.environment.get("skyrl_gym", {}),
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            model_name=cfg.trainer.policy.model.path,
        )

    def get_tracker(self):
        """Initializes the tracker for experiment tracking.

        Returns:
            Tracking: The tracker.
        """
        from skyrl_train.utils.tracking import Tracking

        return Tracking(
            project_name=self.cfg.trainer.project_name,
            experiment_name=self.cfg.trainer.run_name,
            backends=self.cfg.trainer.logger,
            config=self.cfg,
        )

    def _setup_trainer(self):
        """Setup and return the UnifiedTrainer.

        Instantiates the UnifiedTrainer and all the associated models for training.

        Returns:
            UnifiedTrainer: The trainer.
        """
        logger.info(self.get_cfg_as_str(self.cfg))
        os.makedirs(self.cfg.trainer.export_path, exist_ok=True)
        os.makedirs(self.cfg.trainer.ckpt_path, exist_ok=True)

        if self.cfg.trainer.strategy == "deepspeed":
            from skyrl_train.workers.deepspeed.deepspeed_worker import (
                PolicyWorker,
                CriticWorker,
                RefWorker,
            )
        elif self.cfg.trainer.strategy in ("fsdp", "fsdp2"):
            from skyrl_train.workers.fsdp.fsdp_worker import PolicyWorker, CriticWorker, RefWorker
        elif self.cfg.trainer.strategy == "megatron":
            from skyrl_train.workers.megatron.megatron_worker import PolicyWorker, CriticWorker, RefWorker
        else:
            raise ValueError(f"Unknown strategy type: {self.cfg.trainer.strategy}")

        # NOTE (sumanthrh): Instantiate tracker before trainer init.
        # We have custom validation before this step to give better error messages.
        tracker = self.get_tracker()

        tokenizer = self.tokenizer
        if self.cfg.generator.get("run_engines_locally", True):
            inference_engines = create_ray_wrapped_inference_engines_from_config(self.cfg, self.colocate_pg, tokenizer)
        else:
            inference_engines = create_remote_inference_engines_from_config(self.cfg, tokenizer)

        inference_engine_client = InferenceEngineClient(inference_engines, tokenizer, self.cfg)

        generator: GeneratorInterface = self.get_generator(self.cfg, tokenizer, inference_engine_client)

        # Create SkyRL trainer (RayPPOTrainer) for building models
        # Note: We don't pass train_dataset/eval_dataset here as UnifiedTrainer handles that
        skyrl_trainer = RayPPOTrainer(
            cfg=self.cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=None,  # UnifiedTrainer handles dataset loading
            eval_dataset=None,  # UnifiedTrainer handles dataset loading
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=self.colocate_pg,
        )

        # Build the models (policy, critic, ref)
        skyrl_trainer.build_models(PolicyWorker, CriticWorker, RefWorker)

        # Assemble backend-specific arguments for initializing the SkyRL backend
        backend_args = {
            "tokenizer": tokenizer,
            "train_dataset": None,  # UnifiedTrainer handles dataset loading
            "eval_dataset": None,  # UnifiedTrainer handles dataset loading
            "inference_engine_client": inference_engine_client,
            "generator": generator,
            "colocate_pg": self.colocate_pg,
            "tracker": tracker,
        }

        trainer = UnifiedTrainer(
            backend_cls=SkyRLBackend,
            config=self.cfg,
            workflow_class=self.workflow_class,
            train_dataset=self.train_dataset,  # Pass through Dataset objects if provided
            val_dataset=self.val_dataset,  # Pass through Dataset objects if provided
            workflow_args=self.workflow_args,
            backend_args=backend_args,
            **self.kwargs,
        )

        return trainer

    def run(self):
        """Run the training loop."""
        trainer = self._setup_trainer()
        try:
            # Start the training loop
            trainer.fit()
        finally:
            trainer.shutdown()


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
def skyrl_entrypoint(cfg: DictConfig, workflow_class: type[Workflow], workflow_args: dict | None = None, train_dataset: Dataset | None = None, val_dataset: Dataset | None = None, **kwargs):
    """Ray remote entrypoint for SkyRL UnifiedTrainer.

    Makes sure that the training loop is not run on the head node.
    """
    print(f"SkyRL entrypoint hostname: {socket.gethostname()}, PID: {os.getpid()}")
    OmegaConf.register_new_resolver("mul", lambda x, y: int(x) * int(y))
    OmegaConf.resolve(cfg)
    pprint(OmegaConf.to_container(cfg))

    exp = SkyRLExp(cfg, workflow_class, workflow_args, train_dataset=train_dataset, val_dataset=val_dataset, **kwargs)
    exp.run()


class SkyRLTrainerLauncher(TrainerLauncher):
    """
    SkyRL trainer launcher that handles the necessary setup for the SkyRL backend.

    This launcher:
    1. Initializes Ray if needed
    2. Launches the UnifiedTrainer with the SkyRL backend via SkyRLExp
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
        """Initialize the SkyRLTrainerLauncher.

        Args:
            config: Training configuration
            workflow_class: Workflow class for training
            train_dataset: Optional training dataset (if None, uses config.data paths)
            val_dataset: Optional validation dataset (if None, uses config.data paths)
            workflow_args: Optional workflow arguments
            **kwargs: Additional arguments
        """
        super().__init__(config, workflow_class, train_dataset, val_dataset, workflow_args, **kwargs)

        # For SkyRL, datasets can be passed as Dataset objects (like Tinker) or via config.data paths.
        # When Dataset objects are provided, they are used directly by UnifiedTrainer (no conversion needed).
        # The workflow engine expects generic dicts, so no SkyRL-specific format conversion is required.

    def train(self):
        """Launch the training process."""
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            # Read ray_init settings from config
            if self.config is not None and hasattr(self.config, "ray_init"):
                ray_init_settings = {k: v for k, v in self.config.ray_init.items() if v is not None}
            else:
                ray_init_settings = {}

            # Initialize Ray with default settings if no runtime_env is specified
            if "runtime_env" not in ray_init_settings:
                # Use a basic runtime env - SkyRL may have its own requirements
                ray_init_settings["runtime_env"] = {}

            ray.init(**ray_init_settings)

        # Launch the training via Ray remote entrypoint
        ray.get(
            skyrl_entrypoint.remote(  # type: ignore
                cfg=self.config,
                workflow_class=self.workflow_class,
                workflow_args=self.workflow_args,
                train_dataset=self.train_dataset,
                val_dataset=self.val_dataset,
                **self.kwargs,
            )
        )

