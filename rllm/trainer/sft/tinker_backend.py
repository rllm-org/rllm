"""Tinker SFT backend.

Owns the full Tinker SFT loop (migrated from the removed
``rllm.trainer.deprecated.tinker_sft_trainer``). ``tinker``/``tinker_cookbook``
are imported lazily inside :meth:`fit` so the module — and the dispatcher that
imports it — stay importable without the tinker stack installed.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig, OmegaConf

from rllm.trainer.sft.backend import SFTBackend, validate_messages_dataset

if TYPE_CHECKING:
    import tinker
    from tinker.lib.public_interfaces import APIFuture

logger = logging.getLogger(__name__)

_CONFIG_FILE = Path(__file__).resolve().parent / "config" / "tinker.yaml"


def resolve_train_on_what(tokenize_method: str):
    """Map rLLM's tokenize_and_mask_method to tinker's TrainOnWhat."""
    from tinker_cookbook.renderers import TrainOnWhat

    if tokenize_method == "stepwise":
        return TrainOnWhat.LAST_ASSISTANT_MESSAGE
    if tokenize_method not in ("cumulative", "hf_template"):
        logger.warning(f"Unknown tokenize_and_mask_method '{tokenize_method}', defaulting to ALL_ASSISTANT_MESSAGES")
    return TrainOnWhat.ALL_ASSISTANT_MESSAGES


def build_sft_data(config, train_data, val_data):
    """Build (tokenizer, train_dataset, val_dataset) from a backend config.

    Shared by the tinker and fireworks SFT backends: both render rLLM
    ``messages`` rows into tinker Datums via tinker_cookbook renderers.
    """
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    from rllm.trainer.sft.tinker_dataset import create_tinker_sft_datasets

    # Fireworks' model.name is a FW model path (accounts/fireworks/models/...),
    # not HF-resolvable, so render/tokenize from the HF tokenizer_model when set.
    tokenizer_name = config.model.get("tokenizer_model") or config.model.name
    tokenizer = get_tokenizer(tokenizer_name)
    renderer_name = config.data.get("renderer_name", "role_colon")
    renderer = get_renderer(renderer_name, tokenizer)
    tokenize_method = config.data.get("rllm", {}).get("tokenize_and_mask_method", "cumulative")
    train_on_what = resolve_train_on_what(tokenize_method)
    logger.info(f"Using renderer: {renderer_name}, train_on_what: {train_on_what}")

    train_batch_size = config.data.get("train_batch_size", 32)
    val_batch_size = config.data.get("micro_batch_size_per_gpu", train_batch_size)
    train_dataset, val_dataset = create_tinker_sft_datasets(
        train_data=train_data,
        val_data=val_data,
        renderer=renderer,
        batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        max_length=config.data.get("max_length", None),
        train_on_what=train_on_what,
        max_train_samples=config.data.get("train_max_samples", -1),
        max_val_samples=config.data.get("val_max_samples", -1),
    )
    return tokenizer, train_dataset, val_dataset


@dataclass
class _SubmittedBatch:
    """A batch submitted to Tinker (forward-backward + optim step in flight)."""

    fwd_bwd_future: APIFuture[tinker.ForwardBackwardOutput]
    optim_step_future: APIFuture[tinker.OptimStepResponse]
    metrics: dict[str, Any]
    data: list
    step: int
    epoch_idx: int
    batch_idx: int
    batch_start_time: float


class TinkerSFTBackend(SFTBackend):
    """Supervised fine-tuning on Tinker's hosted GPU service."""

    name = "tinker"
    requires_distributed = False

    def __init__(self, spec):
        super().__init__(spec)
        self._config: DictConfig | None = None

    # -- contract -----------------------------------------------------------

    def validate_spec(self) -> None:
        validate_messages_dataset(self.spec.train_dataset, "train")
        if self.spec.val_dataset is not None:
            validate_messages_dataset(self.spec.val_dataset, "val")

    def _config_template(self) -> Path:
        """Path to the backend's native config template (overridden per backend)."""
        return _CONFIG_FILE

    def build_config(self) -> DictConfig:
        """SFTSpec → the DictConfig shape the tinker/fireworks loop reads."""
        spec = self.spec
        base = OmegaConf.load(str(self._config_template()))
        overrides = OmegaConf.create(
            {
                "model": {"name": spec.model, "lora_rank": spec.lora_rank},
                "data": {
                    "train_batch_size": spec.batch_size,
                    "micro_batch_size_per_gpu": spec.batch_size,
                    "max_length": spec.max_length,
                    "rllm": {"tokenize_and_mask_method": spec.tokenize_method},
                },
                "optim": {"lr": spec.lr, "lr_scheduler": spec.lr_schedule},
                "trainer": {
                    "total_epochs": spec.epochs,
                    "save_freq": spec.save_freq,
                    "test_freq": spec.val_freq,
                    "project_name": spec.project,
                    "experiment_name": spec.experiment or "default",
                },
            }
        )
        cfg = OmegaConf.merge(base, overrides)
        if spec.output_dir:
            cfg = OmegaConf.merge(cfg, OmegaConf.create({"trainer": {"default_local_dir": spec.output_dir}}))
        if spec.overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.create(spec.overrides))
        self._config = cfg
        return cfg

    def prepare_data(self) -> None:
        # Tinker consumes the in-memory Dataset objects directly; nothing to do.
        pass

    @property
    def checkpoint_dir(self) -> str:
        cfg = self._config or self.build_config()
        return cfg.trainer.default_local_dir

    def fit(self) -> None:
        if self._config is None:
            self.build_config()
        asyncio.run(self._fit_async())

    # -- training loop (migrated) ------------------------------------------

    async def _fit_async(self) -> None:
        import tinker
        from tinker_cookbook import checkpoint_utils
        from tinker_cookbook.display import colorize_example
        from tinker_cookbook.supervised.common import compute_mean_nll
        from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier
        from tinker_cookbook.utils.misc_utils import timed

        from rllm.utils.tracking import Tracking

        config = self._config
        os.makedirs(config.trainer.default_local_dir, exist_ok=True)
        service_client = tinker.ServiceClient(base_url=config.get("tinker_base_url", None))

        logger_backend = config.trainer.logger
        if isinstance(logger_backend, str):
            logger_backend = [logger_backend]
        tracking_logger = Tracking(
            project_name=config.trainer.project_name,
            experiment_name=config.trainer.experiment_name,
            default_backend=logger_backend,
            config=OmegaConf.to_container(config, resolve=True),
        )

        tokenizer, train_dataset, val_dataset = build_sft_data(config, self.spec.train_dataset, self.spec.val_dataset)

        resume_info = checkpoint_utils.get_last_checkpoint(config.trainer.default_local_dir)
        if resume_info:
            logger.info(f"Resuming from checkpoint: {resume_info}")
            training_client = await service_client.create_training_client_from_state_async(resume_info["state_path"])
            start_epoch = resume_info.get("epoch", 0)
            start_batch = resume_info.get("batch", 0)
        else:
            logger.info("Starting training from scratch")
            training_client = await service_client.create_lora_training_client_async(
                base_model=config.model.name,
                rank=config.model.get("lora_rank", 32),
                train_unembed=OmegaConf.select(config, "model.train_unembed", default=True),
                train_attn=OmegaConf.select(config, "model.train_attn", default=True),
                train_mlp=OmegaConf.select(config, "model.train_mlp", default=True),
            )
            start_epoch = 0
            start_batch = 0

        n_batches = len(train_dataset)
        total_epochs = config.trainer.get("total_epochs", 1)
        total_steps = n_batches * total_epochs
        progress_denominator = total_steps if total_steps > 0 else 1
        logger.info(f"Training for {n_batches} batches x {total_epochs} epochs = {total_steps} steps")

        base_learning_rate = config.get("optim", {}).get("lr", 1e-5)
        lr_schedule = config.get("optim", {}).get("lr_scheduler", "constant")
        adam_betas = config.get("optim", {}).get("betas", [0.9, 0.95])
        adam_eps = config.get("optim", {}).get("eps", 1e-8)
        save_every = config.trainer.get("save_freq", 20)
        eval_every = config.trainer.get("test_freq", 10)

        async def submit_batch(epoch_idx: int, batch_idx: int) -> _SubmittedBatch:
            step = epoch_idx * n_batches + batch_idx
            batch_start_time = time.time()
            metrics: dict[str, Any] = {"epoch": epoch_idx, "progress": step / progress_denominator}
            learning_rate = base_learning_rate * compute_schedule_lr_multiplier(lr_schedule=lr_schedule, step=step, total_steps=total_steps)
            metrics["learning_rate"] = learning_rate
            adam_params = tinker.AdamParams(learning_rate=learning_rate, beta1=adam_betas[0], beta2=adam_betas[1], eps=adam_eps)

            with timed("get_batch", metrics):
                data = train_dataset.get_batch(batch_idx)
            if data:
                logger.info(colorize_example(data[0], tokenizer))

            fwd_bwd_future = await training_client.forward_backward_async(data, loss_fn="cross_entropy")
            optim_step_future = await training_client.optim_step_async(adam_params)
            return _SubmittedBatch(fwd_bwd_future, optim_step_future, metrics, data, step, epoch_idx, batch_idx, batch_start_time)

        async def finish_batch(submitted: _SubmittedBatch) -> None:
            metrics = submitted.metrics
            metrics["progress"] = min((submitted.step + 1) / progress_denominator, 1.0)
            if save_every > 0 and submitted.step % save_every == 0 and submitted.step > 0:
                with timed("save_checkpoint", metrics):
                    await checkpoint_utils.save_checkpoint_async(
                        training_client=training_client,
                        name=f"{submitted.step:06d}",
                        log_path=config.trainer.default_local_dir,
                        loop_state={"epoch": submitted.epoch_idx, "batch": submitted.batch_idx},
                        kind="both",
                    )
            with timed("step", metrics):
                fwd_bwd_result = await submitted.fwd_bwd_future.result_async()
                await submitted.optim_step_future.result_async()

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

            if val_dataset and eval_every > 0 and submitted.step % eval_every == 0 and submitted.step > 0:
                with timed("validation", metrics):
                    val_metrics = await self._validate(training_client, val_dataset, compute_mean_nll)
                metrics.update(val_metrics)

            tracking_logger.log(data=metrics, step=submitted.step)
            logger.info(f"Step {submitted.step}: train_nll={train_nll:.4f}, lr={metrics['learning_rate']:.2e}")

        pending: _SubmittedBatch | None = None
        for epoch_idx in range(start_epoch, total_epochs):
            logger.info(f"Starting epoch {epoch_idx}")
            train_dataset.set_epoch(seed=epoch_idx)
            start_batch_idx = start_batch if epoch_idx == start_epoch else 0
            for batch_idx in range(start_batch_idx, n_batches):
                submitted = await submit_batch(epoch_idx, batch_idx)
                if pending is not None:
                    await finish_batch(pending)
                pending = submitted
        if pending is not None:
            await finish_batch(pending)

        if start_epoch < total_epochs:
            await checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name="final",
                log_path=config.trainer.default_local_dir,
                kind="both",
                loop_state={"epoch": total_epochs, "batch": n_batches},
            )
        else:
            logger.info("Training was already complete; nothing to do")

        tracking_logger.log(data={"status": "completed"}, step=total_steps)
        try:
            tracking_logger.finish()
        except Exception:
            pass
        logger.info("Training completed successfully")

    @staticmethod
    async def _validate(training_client, val_dataset, compute_mean_nll) -> dict[str, float]:
        logger.info("Running validation...")
        total_nll = 0.0
        total_tokens = 0
        for batch_idx in range(len(val_dataset)):
            data = val_dataset.get_batch(batch_idx)
            fwd_bwd_future = await training_client.forward_backward_async(data, loss_fn="cross_entropy")
            fwd_bwd_result = await fwd_bwd_future.result_async()
            logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
            weights = [datum.loss_fn_inputs["weights"] for datum in data]
            batch_nll = compute_mean_nll(logprobs, weights)
            batch_tokens = sum(sum(datum.loss_fn_inputs["weights"].data) for datum in data)
            total_nll += batch_nll * batch_tokens
            total_tokens += batch_tokens
        val_nll = total_nll / total_tokens if total_tokens > 0 else 0.0
        logger.info(f"Validation NLL: {val_nll:.4f}")
        return {"test/mean_nll": val_nll}
