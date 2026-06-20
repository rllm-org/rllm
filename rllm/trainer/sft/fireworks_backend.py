"""Fireworks SFT backend.

Mirrors how the RL stack's ``FireworksBackend`` extends ``TinkerBackend``:
:class:`FireworksSFTBackend` subclasses :class:`TinkerSFTBackend` and reuses
``validate_spec`` / ``build_config`` / ``prepare_data`` and the shared
tinker-cookbook data pipeline (``build_sft_data``). It overrides only what
differs — training-client creation (Fireworks' SDK-managed client instead of a
``tinker.ServiceClient``) and checkpointing — and runs a **synchronous**
pipelined loop because Fireworks' ``ReconnectableClient`` is sync (blocking
``forward_backward``/``optim_step`` plus future-returning ``submit_*``), unlike
tinker's async client.

Requires ``FIREWORKS_API_KEY`` in the environment. Imports of the Fireworks SDK
are deferred to :meth:`fit` so the dispatcher/CLI import without it installed.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from pathlib import Path

from omegaconf import OmegaConf

from rllm.trainer.sft.backend import SFTConfigError
from rllm.trainer.sft.tinker_backend import TinkerSFTBackend, build_sft_data

logger = logging.getLogger(__name__)

_CONFIG_FILE = Path(__file__).resolve().parent / "config" / "fireworks.yaml"


class FireworksSFTBackend(TinkerSFTBackend):
    """Supervised fine-tuning on Fireworks' managed training service."""

    name = "fireworks"
    requires_distributed = False

    def _config_template(self) -> Path:
        return _CONFIG_FILE

    def fit(self) -> None:
        if self._config is None:
            self.build_config()
        config = self._config

        # -- Fireworks SDK (deferred import) -------------------------------
        try:
            import tinker
            from tinker_cookbook.supervised.common import compute_mean_nll
            from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier
            from training.utils import ReconnectableClient, TrainerConfig, build_service_client
            from training.utils.checkpoints import TrainingCheckpoints
            from training.utils.client import DEFAULT_TIMEOUT_S
        except ImportError as e:
            raise SFTConfigError(f"Fireworks SFT backend requires the Fireworks training SDK: {e}") from None

        from rllm.utils.tracking import Tracking

        api_key = os.environ.get("FIREWORKS_API_KEY", "")
        if not api_key:
            raise SFTConfigError("FIREWORKS_API_KEY is not set; required for the fireworks SFT backend.")
        base_url = os.environ.get("FIREWORKS_BASE_URL", config.get("fireworks_base_url", "https://api.fireworks.ai"))

        os.makedirs(config.trainer.default_local_dir, exist_ok=True)
        lora_rank = config.model.get("lora_rank", 32)
        max_length = config.data.get("max_length", None)

        logger_backend = config.trainer.logger
        if isinstance(logger_backend, str):
            logger_backend = [logger_backend]
        tracking_logger = Tracking(
            project_name=config.trainer.project_name,
            experiment_name=config.trainer.experiment_name,
            default_backend=logger_backend,
            config=OmegaConf.to_container(config, resolve=True),
        )

        # -- Data (shared tinker-cookbook pipeline) ------------------------
        _tokenizer, train_dataset, val_dataset = build_sft_data(config, self.spec.train_dataset, self.spec.val_dataset)

        # -- Provision the SDK-managed training client ---------------------
        service = build_service_client(
            api_key=api_key,
            base_url=base_url,
            additional_headers=None,
            base_model=config.model.name,
            tokenizer_model=config.model.name,
            lora_rank=lora_rank,
            max_context_length=max_length,
            learning_rate=config.optim.lr,
            trainer=TrainerConfig(),
        )
        try:
            training_client = service.create_training_client(
                config.model.name,
                lora_rank=lora_rank,
                train_mlp=OmegaConf.select(config, "model.train_mlp", default=True),
                train_attn=OmegaConf.select(config, "model.train_attn", default=True),
                train_unembed=OmegaConf.select(config, "model.train_unembed", default=True),
            )
            client = ReconnectableClient.from_training_client(
                training_client,
                base_model=config.model.name,
                lora_rank=lora_rank,
                job_id=service.trainer_job_id,
                default_timeout=DEFAULT_TIMEOUT_S,
                service=service,
            )
            ckpt = TrainingCheckpoints(
                client,
                service,
                trainer_id=service.trainer_job_id,
                log_path=config.trainer.default_local_dir,
                lora_rank=lora_rank,
            )

            # Auto-resume from the newest resumable checkpoint, if any.
            resume = ckpt.resume()
            start_step = resume.step if resume else 0

            n_batches = len(train_dataset)
            total_epochs = config.trainer.get("total_epochs", 1)
            total_steps = n_batches * total_epochs
            progress_denominator = total_steps if total_steps > 0 else 1
            logger.info(f"Training for {n_batches} batches x {total_epochs} epochs = {total_steps} steps")

            base_lr = config.optim.lr
            lr_schedule = config.optim.get("lr_scheduler", "constant")
            betas = config.optim.get("betas", [0.9, 0.95])
            eps = config.optim.get("eps", 1e-8)
            save_every = config.trainer.get("save_freq", 20)
            eval_every = config.trainer.get("test_freq", 10)

            # Pipelined sync loop: keep one (fwd_bwd, optim) pair in flight.
            in_flight: deque = deque()

            def submit(step: int):
                lr = base_lr * compute_schedule_lr_multiplier(lr_schedule=lr_schedule, step=step, total_steps=total_steps)
                adam = tinker.AdamParams(learning_rate=lr, beta1=betas[0], beta2=betas[1], eps=eps)
                data = train_dataset.get_batch(step % n_batches)
                fb_fut = client.submit_forward_backward(data, loss_fn="cross_entropy")
                opt_fut = client.submit_optim_step(adam)
                in_flight.append((step, lr, data, fb_fut, opt_fut, time.time()))

            def collect():
                step, lr, data, fb_fut, opt_fut, t0 = in_flight.popleft()
                fb_result = fb_fut.result(timeout=DEFAULT_TIMEOUT_S)
                opt_fut.result(timeout=DEFAULT_TIMEOUT_S)
                logprobs = [x["logprobs"] for x in fb_result.loss_fn_outputs]
                weights = [datum.loss_fn_inputs["weights"] for datum in data]
                train_nll = compute_mean_nll(logprobs, weights)
                metrics = {
                    "learning_rate": lr,
                    "progress": min((step + 1) / progress_denominator, 1.0),
                    "num_sequences": len(data),
                    "num_loss_tokens": sum(sum(d.loss_fn_inputs["weights"].data) for d in data),
                    "train_mean_nll": train_nll,
                    "time/total": time.time() - t0,
                }
                if val_dataset and eval_every > 0 and step % eval_every == 0 and step > 0:
                    metrics.update(self._validate(client, val_dataset, compute_mean_nll, DEFAULT_TIMEOUT_S))
                tracking_logger.log(data=metrics, step=step)
                logger.info(f"Step {step}: train_nll={train_nll:.4f}, lr={lr:.2e}")
                if save_every > 0 and step % save_every == 0 and step > 0:
                    logger.info(f"Saving checkpoint at step {step}")
                    ckpt.save(f"step-{step}", resumable=True, promotable=False)

            for step in range(start_step, total_steps):
                if step % n_batches == 0:
                    train_dataset.set_epoch(seed=step // n_batches)
                submit(step)
                if len(in_flight) > 1:
                    collect()
            while in_flight:
                collect()

            if total_steps > start_step:
                logger.info(f"Saving final checkpoint at step {total_steps}")
                ckpt.save(f"step-{total_steps}", resumable=True, promotable=False)

            tracking_logger.log(data={"status": "completed"}, step=total_steps)
            try:
                tracking_logger.finish()
            except Exception:
                pass
            logger.info("Training completed successfully")
        finally:
            service.close()

    @staticmethod
    def _validate(client, val_dataset, compute_mean_nll, timeout) -> dict[str, float]:
        logger.info("Running validation...")
        total_nll = 0.0
        total_tokens = 0
        for batch_idx in range(len(val_dataset)):
            data = val_dataset.get_batch(batch_idx)
            fb_result = client.submit_forward_backward(data, loss_fn="cross_entropy").result(timeout=timeout)
            logprobs = [x["logprobs"] for x in fb_result.loss_fn_outputs]
            weights = [datum.loss_fn_inputs["weights"] for datum in data]
            batch_tokens = sum(sum(d.loss_fn_inputs["weights"].data) for d in data)
            total_nll += compute_mean_nll(logprobs, weights) * batch_tokens
            total_tokens += batch_tokens
        val_nll = total_nll / total_tokens if total_tokens > 0 else 0.0
        logger.info(f"Validation NLL: {val_nll:.4f}")
        return {"test/mean_nll": val_nll}
