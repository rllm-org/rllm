"""
SkyRL launcher implementation for the UnifiedTrainer.

This launcher handles the necessary setup for the SkyRL backend, including
initializing Ray, creating inference engines, generators, and trainers.

Follows the BasePPOExp pattern from skyrl_train.entrypoints.main_base.
"""

import os
import socket
import tempfile
from pprint import pprint

import ray
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from ray.util.placement_group import PlacementGroup, placement_group
from skyrl_train.entrypoints.main_base import (
    create_ray_wrapped_inference_engines_from_config,
    create_remote_inference_engines_from_config,
)
# Import SkyRL components
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.utils.constants import SKYRL_RAY_PG_TIMEOUT_IN_S
from skyrl_train.utils.tracking import Tracking
from skyrl_train.utils.utils import get_ray_pg_ready_with_timeout, initialize_ray
from transformers import AutoTokenizer

from rllm.data import Dataset
from rllm.experimental.skyrl.skyrl_backend import SkyRLBackend
from rllm.experimental.unified_trainer import TrainerLauncher, UnifiedTrainer
from rllm.workflows.workflow import Workflow


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
                [{"GPU": 1, "CPU": 1}] * self.cfg.generator.num_inference_engines * self.cfg.generator.inference_engine_tensor_parallel_size * self.cfg.generator.inference_engine_pipeline_parallel_size * self.cfg.generator.inference_engine_data_parallel_size,
                strategy="PACK",
            )
            get_ray_pg_ready_with_timeout(pg, timeout=timeout)
            return pg
        else:
            return None

    def _convert_rllm_dataset_to_skyrl_file(self, rllm_dataset: Dataset | None) -> str | None:
        """Convert rLLM Dataset to SkyRL format and save to file.

        Following Verl pattern: save to file and return path.

        Note: PromptDataset will load the data into memory (keep_in_memory=True) when created,
        so the file is only needed during PromptDataset initialization. The dataloader state
        checkpointing is separate and handled by StatefulDataLoader.

        Args:
            rllm_dataset: rLLM Dataset object or None

        Returns:
            File path to saved dataset or None
        """
        if rllm_dataset is None:
            return None

        import pandas as pd

        # Convert rLLM Dataset data to format expected by PromptDataset
        # PromptDataset expects: prompt (chat format), env_class, and other fields
        data = rllm_dataset.get_data()

        # Save to temporary parquet file
        temp_fd, temp_path = tempfile.mkstemp(suffix=".parquet")
        os.close(temp_fd)  # Close file descriptor, we'll use pandas to write

        try:
            # Convert data to DataFrame and save
            # Ensure data has required fields for PromptDataset
            df_data = []
            for item in data:
                # Ensure prompt is in chat format (list of dicts)
                if "prompt" not in item:
                    # If no prompt, create from available fields
                    prompt = [{"role": "user", "content": str(item.get("question", item.get("input", "")))}]
                elif isinstance(item["prompt"], str):
                    # Convert string to chat format
                    prompt = [{"role": "user", "content": item["prompt"]}]
                else:
                    prompt = item["prompt"]

                df_item = {
                    "prompt": prompt,
                    "env_class": item.get("env_class", self.cfg.environment.get("env_class", "BaseTextEnv")),
                }
                # Add any other fields
                for key, value in item.items():
                    if key not in ["prompt", "env_class"]:
                        df_item[key] = value

                df_data.append(df_item)

            df = pd.DataFrame(df_data)
            df.to_parquet(temp_path)

            return temp_path
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise RuntimeError(f"Failed to convert rLLM Dataset to SkyRL format: {e}") from e

    def _setup_trainer(self):
        """Setup and return the UnifiedTrainer.

        Instantiates the UnifiedTrainer and all the associated models for training.

        Returns:
            UnifiedTrainer: The trainer.
        """
        logger.info(self.get_cfg_as_str(self.cfg))
        os.makedirs(self.cfg.trainer.export_path, exist_ok=True)
        os.makedirs(self.cfg.trainer.ckpt_path, exist_ok=True)

        tokenizer = self.tokenizer
        if self.cfg.generator.get("run_engines_locally", True):
            inference_engines = create_ray_wrapped_inference_engines_from_config(self.cfg, self.colocate_pg, tokenizer)
        else:
            inference_engines = create_remote_inference_engines_from_config(self.cfg, tokenizer)

        inference_engine_client = InferenceEngineClient(inference_engines, tokenizer, self.cfg)

        # Convert rLLM Datasets to SkyRL format and build PromptDatasets
        # SkyRL's RayPPOTrainer expects PromptDataset objects (unlike Verl which reads from config)
        from skyrl_train.dataset.dataset import PromptDataset

        train_prompt_dataset = None
        eval_prompt_dataset = None

        if self.train_dataset is not None:
            train_file_path = self._convert_rllm_dataset_to_skyrl_file(self.train_dataset)
            if train_file_path is not None:
                train_prompt_dataset = PromptDataset(
                    datasets=train_file_path,
                    tokenizer=tokenizer,
                    max_prompt_length=self.cfg.data.max_prompt_length,
                    num_workers=8,
                )

        if self.val_dataset is not None:
            val_file_path = self._convert_rllm_dataset_to_skyrl_file(self.val_dataset)
            if val_file_path is not None:
                eval_prompt_dataset = PromptDataset(
                    datasets=val_file_path,
                    tokenizer=tokenizer,
                    max_prompt_length=self.cfg.data.max_prompt_length,
                    num_workers=8,
                )

        # Create tracker for RayPPOTrainer (SkyRL's Tracking, not rLLM's)
        # NOTE: We disable wandb here to avoid duplicate logging - rLLM's UnifiedTrainer
        # will handle all wandb logging via its own Tracking instance.
        # SkyRL's tracker is still needed for RayPPOTrainer initialization, but we only
        # use console logging to avoid duplicate wandb runs.
        tracker = Tracking(
            project_name=self.cfg.trainer.project_name,
            experiment_name=self.cfg.trainer.run_name,
            backends=["console"],  # Only console, wandb is handled by UnifiedTrainer
            config=self.cfg,
        )

        # Assemble backend-specific arguments for initializing the SkyRL backend
        # SkyRLBackend inherits from RayPPOTrainer, so it needs all RayPPOTrainer init parameters
        backend_args = {
            "tracker": tracker,
            "tokenizer": tokenizer,
            "train_dataset": train_prompt_dataset,
            "eval_dataset": eval_prompt_dataset,
            "inference_engine_client": inference_engine_client,
            "generator": None,  # Not used — generate_episodes() calls workflow engine directly
            "colocate_pg": self.colocate_pg,
        }

        # Create UnifiedTrainer (which creates SkyRLBackend)
        # SkyRLBackend inherits from RayPPOTrainer, so it IS the trainer
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

        # Build the models (policy, critic, ref) on the backend
        # This must be called after backend initialization
        if self.cfg.trainer.strategy == "deepspeed":
            from skyrl_train.workers.deepspeed.deepspeed_worker import (
                CriticWorker,
                PolicyWorker,
                RefWorker,
            )
        elif self.cfg.trainer.strategy in ("fsdp", "fsdp2"):
            from skyrl_train.workers.fsdp.fsdp_worker import CriticWorker, PolicyWorker, RefWorker
        elif self.cfg.trainer.strategy == "megatron":
            from skyrl_train.workers.megatron.megatron_worker import CriticWorker, PolicyWorker, RefWorker
        else:
            raise ValueError(f"Unknown strategy type: {self.cfg.trainer.strategy}")

        trainer.backend.build_models(PolicyWorker, CriticWorker, RefWorker)

        return trainer

    def run(self):
        """Run the training loop."""
        print("Starting training loop")
        print("train dataset is ", self.train_dataset)
        print("val dataset is ", self.val_dataset)
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

    def train(self):
        """Launch the training process."""
        # Initialize Ray if not already initialized
        # Use SkyRL's initialize_ray() which handles runtime_env setup and sync_registries()
        if not ray.is_initialized():
            # If Ray was previously initialized but is in a bad state, try to reset it
            try:
                initialize_ray(self.config)
            except Exception as e:
                # If initialization fails, try shutting down and reinitializing
                logger.warning(f"Ray initialization failed: {e}. Attempting to reset Ray...")
                try:
                    ray.shutdown()
                except Exception:
                    pass  # Ignore errors during shutdown
                # Wait a moment for cleanup
                import time

                time.sleep(1)
                # Retry initialization
                initialize_ray(self.config)

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
