"""Fireworks SFT backend.

Mirrors how the RL stack's ``FireworksBackend`` extends ``TinkerBackend``:
:class:`FireworksSFTBackend` subclasses :class:`TinkerSFTBackend`, reuses the
shared tinker-cookbook data pipeline (``build_sft_data``) and ``validate_spec``,
and overrides only what differs — provisioning and checkpointing.

Provisioning is identical to the RL backend's: a ``fireworks_infra`` provision
document (carrying ``trainers.policy.training_shape_id``) is parsed by
``training.provision.load_yaml_provision`` and handed to
``init_fireworks_infra("sft", ...)``. Because the document names a training
shape, the SDK takes the **training-shape path** (not the manual-infra path), so
it works on standard accounts. ``infra.policy`` is the same sync
``ReconnectableClient`` the RL path uses; the training loop is a synchronous
pipeline over it (Fireworks has no async client, unlike tinker).

Requires ``FIREWORKS_API_KEY``. Fireworks SDK imports are deferred to
:meth:`fit` so the dispatcher/CLI import without it installed.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from collections import deque
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

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

    def build_config(self) -> DictConfig:
        """SFTSpec → Fireworks config.

        Unlike tinker, Fireworks needs a FW model path + HF tokenizer + matching
        training shape (configured in the template). So ``--model`` only replaces
        the FW base model when it is itself a FW path (``accounts/...``);
        otherwise the template's ``model.name``/``tokenizer_model`` are kept.
        """
        spec = self.spec
        base = OmegaConf.load(str(self._config_template()))
        model_override = {"lora_rank": spec.lora_rank}
        if str(spec.model).startswith("accounts/"):
            model_override["name"] = spec.model
        overrides = OmegaConf.create(
            {
                "model": model_override,
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

    def _provision(self, config, api_key: str, base_url: str):
        """Provision a dedicated SFT trainer via the shape path (like RL)."""
        import yaml
        from training.provision import init_fireworks_infra, load_yaml_provision

        # Parse the fireworks_infra provision document; inject runtime knobs the
        # way the RL backend does (learning rate, optional max_seq_len).
        doc = OmegaConf.to_container(config.fireworks_infra, resolve=True)
        common = doc.setdefault("common", {})
        common["learning_rate"] = float(config.optim.lr)
        if config.data.get("max_length"):
            common["max_seq_len"] = config.data.max_length

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
            yaml.safe_dump(doc, fh)
            doc_path = Path(fh.name)
        try:
            _mode, provision_cfg = load_yaml_provision(mode="sft", recipe=None, path=doc_path)
        finally:
            doc_path.unlink(missing_ok=True)

        # cleanup_existing/cleanup_on_close mirror the RL backend: rllm provisions
        # a fresh trainer per run and tears it down on exit.
        return init_fireworks_infra(
            "sft",
            provision_cfg,
            base_url=base_url,
            cleanup_on_close=True,
            cleanup_existing=True,
        )

    def fit(self) -> None:
        if self._config is None:
            self.build_config()
        config = self._config

        try:
            import tinker
            from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier
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

        logger_backend = config.trainer.logger
        if isinstance(logger_backend, str):
            logger_backend = [logger_backend]
        tracking_logger = Tracking(
            project_name=config.trainer.project_name,
            experiment_name=config.trainer.experiment_name,
            default_backend=logger_backend,
            config=OmegaConf.to_container(config, resolve=True),
        )

        _tokenizer, train_dataset, val_dataset = build_sft_data(config, self.spec.train_dataset, self.spec.val_dataset)

        infra = self._provision(config, api_key, base_url)
        try:
            client = infra.policy
            ckpt = TrainingCheckpoints(
                client,
                infra.service,
                trainer_id=infra.policy_job_id,
                log_path=config.trainer.default_local_dir,
                lora_rank=lora_rank,
            )

            # Auto-resume from the newest resumable checkpoint, if any.
            resume = ckpt.resume()
            start_step = resume.step if resume else 0

            # len(dataset) floors examples//batch_size; keep the final partial
            # batch when the dataset is smaller than one batch (else 0 steps).
            n_batches = max(1, len(train_dataset))
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
                # Fireworks' cross_entropy forward_backward returns aggregate
                # metrics (loss:sum / response_tokens), not per-token logprobs.
                fb_metrics = getattr(fb_result, "metrics", {}) or {}
                n_loss_tokens = fb_metrics.get("response_tokens") or 0
                train_loss = (fb_metrics.get("loss:sum", 0.0) / n_loss_tokens) if n_loss_tokens else 0.0
                metrics = {
                    "learning_rate": lr,
                    "progress": min((step + 1) / progress_denominator, 1.0),
                    "num_sequences": len(data),
                    "num_loss_tokens": n_loss_tokens,
                    "train_loss": train_loss,
                    "time/total": time.time() - t0,
                }
                if val_dataset and eval_every > 0 and step % eval_every == 0 and step > 0:
                    metrics.update(self._validate(client, val_dataset, DEFAULT_TIMEOUT_S))
                tracking_logger.log(data=metrics, step=step)
                logger.info(f"Step {step}: train_loss={train_loss:.4f}, lr={lr:.2e}")
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
            infra.close()

    @staticmethod
    def _validate(client, val_dataset, timeout) -> dict[str, float]:
        logger.info("Running validation...")
        total_loss = 0.0
        total_tokens = 0
        for batch_idx in range(len(val_dataset)):
            data = val_dataset.get_batch(batch_idx)
            fb_result = client.submit_forward_backward(data, loss_fn="cross_entropy").result(timeout=timeout)
            m = getattr(fb_result, "metrics", {}) or {}
            total_loss += m.get("loss:sum", 0.0)
            total_tokens += m.get("response_tokens") or 0
        val_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        logger.info(f"Validation loss: {val_loss:.4f}")
        return {"test/loss": val_loss}
