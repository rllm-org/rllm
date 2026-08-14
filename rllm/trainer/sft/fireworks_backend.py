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
import re
import tempfile
import time
from collections import deque
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from rllm.trainer.sft.backend import SFTConfigError
from rllm.trainer.sft.tinker_backend import (
    TinkerSFTBackend,
    build_sft_data,
    should_validate_step,
)

logger = logging.getLogger(__name__)

_CONFIG_FILE = Path(__file__).resolve().parent / "config" / "fireworks.yaml"


class FireworksSFTBackend(TinkerSFTBackend):
    """Supervised fine-tuning on Fireworks' managed training service."""

    name = "fireworks"
    requires_distributed = False
    # rank 0 → POLICY_TRAINER full-parameter shapes (tinker is LoRA-only).
    supports_full_finetune = True

    def _config_template(self) -> Path:
        return _CONFIG_FILE

    def build_config(self) -> DictConfig:
        """SFTSpec → Fireworks config.

        Unlike tinker, Fireworks needs a FW model path + HF tokenizer + matching
        training shape (configured in the template). So ``--model`` only replaces
        the FW base model when it is itself a FW path (``accounts/...``);
        otherwise the template's ``model.name``/``tokenizer_model`` are kept.
        Swapping to a *different* FW base model requires ``model.tokenizer_model``
        + ``fireworks_config.policy_trainer_shape_id`` overrides to move with it
        (enforced below — the template's values belong to the template's model).
        """
        spec = self.spec
        base = OmegaConf.load(str(self._config_template()))
        model_override = {"lora_rank": spec.lora_rank}
        if str(spec.model).startswith("accounts/"):
            model_override["name"] = spec.model
        trainer_overrides: dict = {
            "total_epochs": spec.epochs,
            "save_freq": spec.save_freq,
            "test_freq": spec.val_freq,
            "project_name": spec.project,
            "experiment_name": spec.experiment or "default",
        }
        # spec.logger=None keeps the yaml default (['console']); a set list selects
        # tracking backends for rllm.utils.tracking.Tracking (wandb/mlflow/ui/...).
        if spec.logger is not None:
            trainer_overrides["logger"] = list(spec.logger)
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
                "trainer": trainer_overrides,
            }
        )
        cfg = OmegaConf.merge(base, overrides)
        if spec.output_dir:
            cfg = OmegaConf.merge(cfg, OmegaConf.create({"trainer": {"default_local_dir": spec.output_dir}}))
        if spec.overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.create(spec.overrides))
        self._align_shape_with_lora_rank(cfg)
        self._require_consistent_model_swap(base, cfg)
        self._config = cfg
        return cfg

    def _align_shape_with_lora_rank(self, cfg: DictConfig) -> None:
        """Keep the training shape's trainer mode consistent with ``lora_rank``.

        Fireworks publishes each model's shapes per trainer mode as sibling
        resources named ``<base>`` (POLICY_TRAINER = full-parameter) and
        ``<base>-lora`` (LORA_TRAINER); ``lora_rank`` alone selects the mode
        (rank 0 → POLICY_TRAINER). A rank-0 run pointed at a ``-lora`` shape
        would provision the wrong trainer mode, so derive the full-parameter
        sibling by stripping the suffix; explicit non-``-lora`` shapes pass through.
        """
        shape = str(OmegaConf.select(cfg, "fireworks_config.policy_trainer_shape_id") or "")
        lora_rank = int(cfg.model.get("lora_rank") or 0)
        if lora_rank == 0 and shape.endswith("-lora"):
            full_shape = shape.removesuffix("-lora")
            logger.info("Full-parameter mode (lora_rank=0): training shape %s -> %s (POLICY_TRAINER)", shape, full_shape)
            cfg.fireworks_config.policy_trainer_shape_id = full_shape
        elif lora_rank > 0 and shape and not shape.endswith("-lora"):
            logger.warning(
                "lora_rank=%d (LoRA mode) with training shape %r, which does not follow the '-lora' naming "
                "convention for LORA_TRAINER shapes. If provisioning fails with a trainer-mode mismatch, point "
                "fireworks_config.policy_trainer_shape_id at the model's LoRA-validated shape.",
                lora_rank,
                shape,
            )

    def _require_consistent_model_swap(self, base: DictConfig, cfg: DictConfig) -> None:
        """Fail fast when the FW base model is swapped (via ``--model`` or an
        overrides ``model.name``) but the HF tokenizer and training shape are
        left at the template's values.

        Rendering/tokenization would silently use the wrong tokenizer and
        provisioning would request a shape validated for a different model.
        Detection is by explicit intent: the user must set both knobs in
        ``spec.overrides`` (e.g. via ``rllm sft --config``).
        """
        new_model = str(cfg.model.name)
        if new_model == str(base.model.name):
            return
        user = OmegaConf.to_container(OmegaConf.create(self.spec.overrides), resolve=False) if self.spec.overrides else {}
        user_model = user.get("model") if isinstance(user.get("model"), dict) else {}
        user_fw = user.get("fireworks_config") if isinstance(user.get("fireworks_config"), dict) else {}
        missing = []
        if not user_model.get("tokenizer_model"):
            missing.append("model.tokenizer_model (the HF tokenizer the rows are rendered with)")
        if not user_fw.get("policy_trainer_shape_id"):
            missing.append("fireworks_config.policy_trainer_shape_id (a training shape validated for the new model)")
        if missing:
            raise SFTConfigError(
                f"The Fireworks base model was swapped to {new_model!r}, but the config still carries the "
                f"template's values for: {'; '.join(missing)}. Set them together via overrides "
                "(e.g. rllm sft --config overrides.yaml with a model: {tokenizer_model: ...} and "
                "fireworks_config: {policy_trainer_shape_id: ...} section)."
            )

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

        # <local dir>/<experiment>/<run stamp>/ — the template default is a fixed
        # shared /tmp path, and even an explicit --output collides across
        # relaunches; a per-run stamp keeps every run's logs/metadata separate.
        # Safe here: Fireworks SFT resume is server-side (job DCP), not local.
        run_stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        config.trainer.default_local_dir = os.path.join(
            config.trainer.default_local_dir,
            config.trainer.get("experiment_name") or "default",
            run_stamp,
        )
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

        # Outer try/finally so tracking_logger.finish() runs even on failure: the
        # 'ui' backend tees stdout/stderr and holds an open session until finish().
        try:
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

                if should_validate_step(
                    0,
                    eval_every=eval_every,
                    has_validation=val_dataset is not None,
                    include_initial=start_step == 0,
                ):
                    tracking_logger.log(
                        data=self._validate(client, val_dataset, DEFAULT_TIMEOUT_S),
                        step=0,
                    )

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
                    completed_steps = step + 1
                    if should_validate_step(
                        completed_steps,
                        eval_every=eval_every,
                        has_validation=val_dataset is not None,
                    ):
                        metrics.update(self._validate(client, val_dataset, DEFAULT_TIMEOUT_S))
                    tracking_logger.log(data=metrics, step=completed_steps)
                    logger.info(f"Step {completed_steps}: train_loss={train_loss:.4f}, lr={lr:.2e}")
                    if save_every > 0 and step % save_every == 0 and step > 0:
                        logger.info(f"Saving checkpoint at step {step}")
                        ckpt.save(f"step-{step}", resumable=True, promotable=False)

                for step in range(start_step, total_steps):
                    if step % n_batches == 0:
                        train_dataset.set_epoch(seed=step // n_batches)
                    if in_flight and should_validate_step(
                        in_flight[0][0] + 1,
                        eval_every=eval_every,
                        has_validation=val_dataset is not None,
                    ):
                        collect()
                    submit(step)
                    if len(in_flight) > 1:
                        collect()
                while in_flight:
                    collect()

                if total_steps > start_step:
                    logger.info(f"Saving final checkpoint at step {total_steps}")
                    # promotable=True writes the servable sampler row that
                    # ``promote_latest`` needs. Without it the weights are a
                    # resumable-only DCP blob, GC'd after the job's ~30-day retention
                    # window — so promote BEFORE ``finally: infra.close()`` deletes the
                    # trainer job. Mirrors the RL path and the SDK sft recipe.
                    ckpt.save(f"step-{total_steps}", resumable=True, promotable=True)
                    artifact = "LoRA adapter" if lora_rank else "full-weight model"
                    experiment = config.trainer.get("experiment_name") or "default"
                    output_model_id = re.sub(r"[^a-z0-9-]+", "-", f"{config.trainer.get('project_name', 'rllm-sft')}-{experiment}".lower()).strip("-")[:63]
                    try:
                        model = ckpt.promote_latest(output_model_id, config.model.name)
                        logger.info("Promoted final %s -> %s", artifact, (model or {}).get("name", output_model_id))
                    except Exception:
                        logger.exception(
                            "Final %s promotion failed. The promotable sampler checkpoint for job %s "
                            "survives job deletion for ~30 days; promote it manually via "
                            "TrainerJobManager.promote_checkpoint(name=<row from list_checkpoints>, "
                            "output_model_id=%r, base_model=%r).",
                            artifact,
                            getattr(infra, "policy_job_id", "<job>"),
                            output_model_id,
                            config.model.name,
                        )

                tracking_logger.log(data={"status": "completed"}, step=total_steps)
                logger.info("Training completed successfully")
            finally:
                infra.close()
        finally:
            try:
                tracking_logger.finish()
            except Exception:
                pass

    @staticmethod
    def _validate(client, val_dataset, _timeout=None) -> dict[str, float]:
        """Compute held-out NLL without adding validation gradients."""
        from tinker_cookbook.supervised.common import compute_mean_nll

        logger.info("Running validation...")
        total_nll = 0.0
        total_tokens = 0
        for batch_idx in range(len(val_dataset)):
            data = val_dataset.get_batch(batch_idx)
            weights = [datum.loss_fn_inputs["weights"] for datum in data]
            batch_tokens = sum(sum(weight.data) for weight in weights)
            if not batch_tokens:
                continue
            forward_result = client.forward(data, "cross_entropy")
            logprobs = [output["logprobs"] for output in forward_result.loss_fn_outputs]
            total_nll += compute_mean_nll(logprobs, weights) * batch_tokens
            total_tokens += batch_tokens
        if total_tokens <= 0:
            raise SFTConfigError("The validation dataset has no trainable tokens after rendering and masking.")
        val_loss = total_nll / total_tokens
        logger.info(f"Validation loss: {val_loss:.4f}")
        return {"test/loss": val_loss}
