"""Tinker-based SFT trainer for rLLM.

This module provides supervised fine-tuning using Tinker's hosted GPU service.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass

import tinker
from omegaconf import OmegaConf
from tinker.lib.public_interfaces import APIFuture
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.display import colorize_example
from tinker_cookbook.renderers import TrainOnWhat, get_renderer
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier
from tinker_cookbook.utils.misc_utils import timed

from rllm.trainer.tinker.tinker_sft_dataset import create_tinker_sft_datasets

logger = logging.getLogger(__name__)


@dataclass
class SubmittedBatch:
    """Represents a batch that has been submitted for training."""

    fwd_bwd_future: APIFuture[tinker.ForwardBackwardOutput]
    optim_step_future: APIFuture[tinker.OptimStepResponse]
    metrics: dict[str, int | float | str]
    data: list
    step: int
    epoch_idx: int
    batch_idx: int
    batch_start_time: float


class TinkerSFTTrainer:
    """
    Supervised fine-tuning trainer using Tinker backend.

    This trainer:
    - Loads rLLM parquet datasets and converts to Tinker format
    - Uses Tinker's hosted GPU service for training
    - Implements pipelined batch submission for efficiency
    - Supports checkpointing and resuming
    - Logs metrics using rLLM's Tracking system
    """

    def __init__(
        self,
        config,
        train_dataset=None,
        val_dataset=None,
    ):
        """
        Initialize the Tinker SFT trainer.

        Args:
            config: Training configuration (OmegaConf)
            train_dataset: Training dataset (rLLM format, optional)
            val_dataset: Validation dataset (rLLM format, optional)
        """
        self.config = config
        self.train_dataset_raw = train_dataset
        self.val_dataset_raw = val_dataset

        # Ensure checkpoint directory exists
        os.makedirs(self.config.trainer.default_local_dir, exist_ok=True)

        # Initialize Tinker service client
        self.service_client = tinker.ServiceClient(base_url=self.config.get("tinker_base_url", None))
        self.training_client = None
        self.tokenizer = None

    def fit_sft(self):
        """Main training loop - sync wrapper for async training."""
        asyncio.run(self._fit_sft_async())

    async def _fit_sft_async(self):
        """Async main training loop."""
        from rllm.utils.tracking import Tracking

        # Setup logging
        logger_backend = self.config.trainer.logger
        if isinstance(logger_backend, str):
            logger_backend = [logger_backend]

        self.tracking_logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=logger_backend,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        # Initialize tokenizer and renderer
        self.tokenizer = get_tokenizer(self.config.model.name)

        # Determine renderer name
        renderer_name = self.config.data.get("renderer_name", "role_colon")
        renderer = get_renderer(renderer_name, self.tokenizer)

        # Map rLLM tokenization method to Tinker's TrainOnWhat
        tokenize_method = self.config.data.get("rllm", {}).get("tokenize_and_mask_method", "cumulative")
        if tokenize_method == "cumulative":
            train_on_what = TrainOnWhat.ALL_ASSISTANT_MESSAGES
        elif tokenize_method == "stepwise":
            train_on_what = TrainOnWhat.LAST_ASSISTANT_MESSAGE
        else:
            logger.warning(f"Unknown tokenize_and_mask_method '{tokenize_method}', defaulting to ALL_ASSISTANT_MESSAGES")
            train_on_what = TrainOnWhat.ALL_ASSISTANT_MESSAGES

        logger.info(f"Using renderer: {renderer_name}, train_on_what: {train_on_what}")

        # Create datasets
        max_length = self.config.data.get("max_length", None)
        train_batch_size = self.config.data.get("train_batch_size", 32)
        val_batch_size = self.config.data.get("micro_batch_size_per_gpu", train_batch_size)

        # Use datasets passed to __init__ if available (from DatasetRegistry),
        # otherwise fall back to loading from file paths in config
        if self.train_dataset_raw is not None:
            # Datasets provided directly (e.g., from DatasetRegistry)
            logger.info("Using datasets provided directly (e.g., from DatasetRegistry)")
            train_data = self.train_dataset_raw
            val_data = self.val_dataset_raw
        else:
            # Load from file paths in config (backward compatibility)
            logger.info("Loading datasets from file paths in config")
            train_data = self.config.data.train_files
            val_data = self.config.data.get("val_files", None)

            if isinstance(train_data, str):
                train_data = [train_data]
            if val_data and isinstance(val_data, str):
                val_data = [val_data]

        train_dataset, val_dataset = create_tinker_sft_datasets(
            train_data=train_data,
            val_data=val_data,
            renderer=renderer,
            batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            max_length=max_length,
            train_on_what=train_on_what,
            max_train_samples=self.config.data.get("train_max_samples", -1),
            max_val_samples=self.config.data.get("val_max_samples", -1),
        )

        # Initialize or resume training client
        resume_info = checkpoint_utils.get_last_checkpoint(self.config.trainer.default_local_dir)

        if resume_info:
            logger.info(f"Resuming from checkpoint: {resume_info}")
            self.training_client = await self.service_client.create_training_client_from_state_async(resume_info["state_path"])
            start_epoch = resume_info.get("epoch", 0)
            start_batch = resume_info.get("batch", 0)
        else:
            logger.info("Starting training from scratch")
            # Configure which layers to train
            train_unembed = OmegaConf.select(self.config, "model.train_unembed", default=True)
            train_attn = OmegaConf.select(self.config, "model.train_attn", default=True)
            train_mlp = OmegaConf.select(self.config, "model.train_mlp", default=True)

            self.training_client = await self.service_client.create_lora_training_client_async(
                base_model=self.config.model.name,
                rank=self.config.model.get("lora_rank", 32),
                train_unembed=train_unembed,
                train_attn=train_attn,
                train_mlp=train_mlp,
            )
            start_epoch = 0
            start_batch = 0

        # Training parameters
        n_batches = len(train_dataset)
        total_epochs = self.config.trainer.get("total_epochs", 1)
        total_steps = n_batches * total_epochs
        progress_denominator = total_steps if total_steps > 0 else 1

        logger.info(f"Training for {n_batches} batches x {total_epochs} epochs = {total_steps} steps")

        # Get training hyperparameters
        base_learning_rate = self.config.get("optim", {}).get("lr", 1e-5)
        lr_schedule = self.config.get("optim", {}).get("lr_scheduler", "constant")
        adam_betas = self.config.get("optim", {}).get("betas", [0.9, 0.95])
        adam_eps = self.config.get("optim", {}).get("eps", 1e-8)

        save_every = self.config.trainer.get("save_freq", 20)
        eval_every = self.config.trainer.get("test_freq", 10)

        async def submit_batch(epoch_idx: int, batch_idx: int) -> SubmittedBatch:
            """Submit a batch for training (forward-backward + optim step)."""
            step = epoch_idx * n_batches + batch_idx
            batch_start_time = time.time()
            metrics: dict[str, int | float | str] = {"epoch": epoch_idx}
            metrics["progress"] = step / progress_denominator

            # Compute learning rate with schedule
            learning_rate = base_learning_rate * compute_schedule_lr_multiplier(
                lr_schedule=lr_schedule,
                step=step,
                total_steps=total_steps,
            )
            metrics["learning_rate"] = learning_rate

            adam_params = tinker.AdamParams(
                learning_rate=learning_rate,
                beta1=adam_betas[0],
                beta2=adam_betas[1],
                eps=adam_eps,
            )

            with timed("get_batch", metrics):
                data = train_dataset.get_batch(batch_idx)

            if data:
                logger.info(colorize_example(data[0], self.tokenizer))

            # Submit forward-backward and optimizer step in pipeline
            fwd_bwd_future = await self.training_client.forward_backward_async(data, loss_fn="cross_entropy")
            optim_step_future = await self.training_client.optim_step_async(adam_params)

            return SubmittedBatch(
                fwd_bwd_future=fwd_bwd_future,
                optim_step_future=optim_step_future,
                metrics=metrics,
                data=data,
                step=step,
                epoch_idx=epoch_idx,
                batch_idx=batch_idx,
                batch_start_time=batch_start_time,
            )

        async def finish_batch(submitted: SubmittedBatch):
            """Wait for batch to complete and log metrics."""
            metrics = submitted.metrics
            metrics["progress"] = min((submitted.step + 1) / progress_denominator, 1.0)

            # Save checkpoint if needed
            if save_every > 0 and submitted.step % save_every == 0 and submitted.step > 0:
                with timed("save_checkpoint", metrics):
                    await checkpoint_utils.save_checkpoint_async(
                        training_client=self.training_client,
                        name=f"{submitted.step:06d}",
                        log_path=self.config.trainer.default_local_dir,
                        loop_state={"epoch": submitted.epoch_idx, "batch": submitted.batch_idx},
                        kind="both",
                    )

            # Wait for training to complete
            with timed("step", metrics):
                fwd_bwd_result = await submitted.fwd_bwd_future.result_async()
                await submitted.optim_step_future.result_async()

            # Compute metrics
            logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
            weights = [datum.loss_fn_inputs["weights"] for datum in submitted.data]
            train_nll = compute_mean_nll(logprobs, weights)

            metrics.update(
                num_sequences=len(submitted.data),
                num_tokens=sum(datum.model_input.length for datum in submitted.data),
                num_loss_tokens=sum(sum(datum.loss_fn_inputs["weights"].data) for datum in submitted.data),
                train_mean_nll=train_nll,
            )
            metrics["time/total"] = time.time() - submitted.batch_start_time

            # Run validation if needed
            if val_dataset and eval_every > 0 and submitted.step % eval_every == 0 and submitted.step > 0:
                with timed("validation", metrics):
                    val_metrics = await self.validate(val_dataset)
                metrics.update(val_metrics)

            # Log metrics
            self.tracking_logger.log(data=metrics, step=submitted.step)
            logger.info(f"Step {submitted.step}: train_nll={train_nll:.4f}, lr={metrics['learning_rate']:.2e}")

        # Training loop with pipelining
        pending_batch: SubmittedBatch | None = None

        for epoch_idx in range(start_epoch, total_epochs):
            logger.info(f"Starting epoch {epoch_idx}")
            train_dataset.set_epoch(seed=epoch_idx)

            start_batch_idx = start_batch if epoch_idx == start_epoch else 0
            for batch_idx in range(start_batch_idx, n_batches):
                # Submit current batch
                submitted_batch = await submit_batch(epoch_idx, batch_idx)

                # Finish previous batch (pipeline overlap)
                if pending_batch is not None:
                    await finish_batch(pending_batch)

                pending_batch = submitted_batch

        # Finish last batch
        if pending_batch is not None:
            await finish_batch(pending_batch)

        # Save final checkpoint
        if start_epoch < total_epochs:
            await checkpoint_utils.save_checkpoint_async(
                training_client=self.training_client,
                name="final",
                log_path=self.config.trainer.default_local_dir,
                kind="both",
                loop_state={"epoch": total_epochs, "batch": n_batches},
            )
        else:
            logger.info("Training was already complete; nothing to do")

        self.tracking_logger.log(data={"status": "completed"}, step=total_steps)
        del self.tracking_logger
        logger.info("Training completed successfully")

    async def validate(self, val_dataset) -> dict[str, float]:
        """Run validation on the validation dataset."""
        logger.info("Running validation...")

        total_nll = 0.0
        total_tokens = 0
        n_val_batches = len(val_dataset)

        for batch_idx in range(n_val_batches):
            data = val_dataset.get_batch(batch_idx)

            # Run forward pass only (no backward)
            fwd_bwd_future = await self.training_client.forward_backward_async(data, loss_fn="cross_entropy")
            fwd_bwd_result = await fwd_bwd_future.result_async()

            # Compute NLL
            logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
            weights = [datum.loss_fn_inputs["weights"] for datum in data]
            batch_nll = compute_mean_nll(logprobs, weights)
            batch_tokens = sum(sum(datum.loss_fn_inputs["weights"].data) for datum in data)

            total_nll += batch_nll * batch_tokens
            total_tokens += batch_tokens

        val_nll = total_nll / total_tokens if total_tokens > 0 else 0.0

        logger.info(f"Validation NLL: {val_nll:.4f}")
        return {"test/mean_nll": val_nll}
