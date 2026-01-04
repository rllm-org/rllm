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

import tempfile
import os

from rllm.data import Dataset
from rllm.experimental.unified_trainer import TrainerLauncher, UnifiedTrainer
from rllm.experimental.skyrl.skyrl_backend import SkyRLBackend
from rllm.workflows.workflow import Workflow

# Import SkyRL components
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.generators.base import GeneratorInterface
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.utils.utils import get_ray_pg_ready_with_timeout, initialize_ray
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
        from rllm.experimental.skyrl.rllm_generator import RLLMGenerator

        max_response_length = cfg.data.get("max_response_length", 4096)
        # workflow_engine will be set later in the backend when it's available
        return RLLMGenerator(
            workflow_engine=None,
            tokenizer=tokenizer,
            max_response_length=max_response_length,
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
        temp_fd, temp_path = tempfile.mkstemp(suffix='.parquet')
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

        # Assemble backend-specific arguments for initializing the SkyRL backend
        backend_args = {
            "tokenizer": tokenizer,
            "train_dataset": train_prompt_dataset,
            "eval_dataset": eval_prompt_dataset,
            "inference_engine_client": inference_engine_client,
            "generator": generator,
            "colocate_pg": self.colocate_pg,
            "tracker": tracker,
        }

        # Create UnifiedTrainer (which creates SkyRLBackend)
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
        # SkyRLBackend inherits from RayPPOTrainer, so it has build_models method
        trainer.backend.build_models(PolicyWorker, CriticWorker, RefWorker)

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



    def train(self):
        """Launch the training process."""
        # Initialize Ray if not already initialized
        # Use SkyRL's initialize_ray() which handles runtime_env setup and sync_registries()
        if not ray.is_initialized():
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

