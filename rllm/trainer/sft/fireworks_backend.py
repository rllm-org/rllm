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
import math
import os
import re
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from rllm.trainer.sft.backend import SFTConfigError
from rllm.trainer.sft.tinker_backend import (
    PreparedResumeManifest,
    SFTOptimizerSettings,
    SFTResumeContract,
    TinkerSFTBackend,
    _is_step_boundary,
    build_adam_params,
    build_hosted_resume_contract,
    build_sft_data,
    iter_training_batches_from_step,
    prepare_hosted_resume_manifest,
    resolve_sft_optimizer_settings,
    resolve_training_steps,
    sft_lr_multiplier,
    should_validate_step,
    update_hosted_resume_manifest,
)

logger = logging.getLogger(__name__)

_CONFIG_FILE = Path(__file__).resolve().parent / "config" / "fireworks.yaml"
_RESUME_NOT_CHECKED = object()


def _fireworks_mean_loss(result: Any, loss_weight: Real) -> float:
    """Compute mean NLL from strict provider loss and local Datum weight mass."""
    metrics = getattr(result, "metrics", None)
    value = metrics.get("loss:sum") if isinstance(metrics, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise SFTConfigError(f"Fireworks forward/backward returned invalid loss:sum metric {value!r}.")
    if isinstance(loss_weight, bool) or not isinstance(loss_weight, Real) or not math.isfinite(float(loss_weight)) or loss_weight <= 0:
        raise SFTConfigError(f"Fireworks batch has invalid loss-weight mass {loss_weight!r}.")
    return float(value) / float(loss_weight)


def _fireworks_optimizer_metrics(result: Any) -> dict[str, Real]:
    """Preserve finite numeric optimizer telemetry under a provider namespace."""
    raw = getattr(result, "metrics", None)
    if not isinstance(raw, Mapping):
        return {}
    metrics = {}
    for key, value in raw.items():
        if isinstance(key, str) and not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(float(value)):
            metrics[f"fireworks/optimizer/{key}"] = value
    return metrics


@dataclass
class _SubmittedFireworksBatch:
    step: int
    data: list[Any]
    learning_rate: float
    started_at: float
    fb_future: Any
    opt_future: Any


def _configured_policy_job_id(config) -> str | None:
    value = OmegaConf.select(config, "fireworks_infra.trainers.policy.job_id")
    return str(value) if value else None


def build_fireworks_resume_contract(
    config,
    train_dataset,
    optimizer: SFTOptimizerSettings,
    *,
    n_batches: int,
    total_steps: int,
) -> SFTResumeContract:
    """Build the local identity for a Fireworks optimizer-state resume."""
    return build_hosted_resume_contract(
        config,
        train_dataset,
        optimizer,
        backend="fireworks",
        provider={
            "training_shape_id": str(config.fireworks_config.policy_trainer_shape_id),
        },
        n_batches=n_batches,
        total_steps=total_steps,
    )


def prepare_fireworks_resume_contract(
    checkpoint_dir: str,
    contract: SFTResumeContract,
    *,
    configured_job_id: str | None,
) -> PreparedResumeManifest:
    """Validate local identity before provisioning or reattaching a trainer."""
    prepared = prepare_hosted_resume_manifest(
        checkpoint_dir,
        contract,
        require_existing=configured_job_id is not None,
    )
    bound_job_id = prepared.data.get("provider_job_id")
    if bound_job_id is not None and (not isinstance(bound_job_id, str) or not bound_job_id):
        raise SFTConfigError("Fireworks run identity has an invalid provider_job_id. Use a new output directory.")
    if configured_job_id is not None:
        if bound_job_id is None:
            raise SFTConfigError("Cannot reattach the Fireworks trainer: the local run manifest has no provider job identity. Use a new output directory.")
        if bound_job_id != configured_job_id:
            raise SFTConfigError(f"Cannot reattach Fireworks job {configured_job_id!r}: the local run manifest belongs to {bound_job_id!r}. Use the matching output directory.")
    elif bound_job_id is not None:
        raise SFTConfigError(f"The Fireworks output directory belongs to provider job {bound_job_id!r}. Set fireworks_infra.trainers.policy.job_id to reattach it, or use a new output directory.")
    return prepared


def validate_fireworks_resume_contract(
    prepared: PreparedResumeManifest,
    *,
    configured_job_id: str | None,
    actual_job_id: str,
    resume_info=_RESUME_NOT_CHECKED,
) -> PreparedResumeManifest:
    """Bind the provider job, then optionally validate its resume state."""
    if not isinstance(actual_job_id, str) or not actual_job_id:
        raise SFTConfigError("Fireworks provisioning did not return a provider job identity; exact resume is unavailable.")
    if configured_job_id is not None and actual_job_id != configured_job_id:
        raise SFTConfigError(f"Fireworks reattached job {actual_job_id!r}, expected configured job {configured_job_id!r}.")

    bound_job_id = prepared.data.get("provider_job_id")
    if configured_job_id is not None and bound_job_id is None:
        raise SFTConfigError("Cannot reattach the Fireworks trainer: the local run manifest has no provider job identity. Use a new output directory.")
    if bound_job_id is not None and bound_job_id != actual_job_id:
        raise SFTConfigError(f"Fireworks job {actual_job_id!r} does not match the local run manifest job {bound_job_id!r}.")
    if bound_job_id is None:
        prepared = update_hosted_resume_manifest(prepared, provider_job_id=actual_job_id)

    if resume_info is _RESUME_NOT_CHECKED:
        return prepared
    if configured_job_id is not None and resume_info is None:
        raise SFTConfigError(f"Fireworks reattached job {configured_job_id!r}, but it has no resumable checkpoint/cursor. Exact resume is unavailable; use a new output directory.")
    if configured_job_id is None and resume_info is not None:
        raise SFTConfigError("A newly provisioned Fireworks job unexpectedly contained a checkpoint. Refusing an identity-ambiguous resume; use a new output directory.")
    return prepared


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
        user = OmegaConf.to_container(OmegaConf.create(spec.overrides), resolve=False) if spec.overrides else {}
        user_trainer = user.get("trainer") if isinstance(user, dict) else None
        explicit_override = isinstance(user_trainer, dict) and user_trainer.get("default_local_dir") is not None
        self._resume_requested = bool(spec.output_dir) or explicit_override
        configured_job_id = _configured_policy_job_id(cfg)
        if configured_job_id is not None and not self._resume_requested:
            raise SFTConfigError(
                "Reattaching a Fireworks trainer via fireworks_infra.trainers.policy.job_id requires an explicit --output (or trainer.default_local_dir override) containing its local cursor metadata."
            )
        if not self._resume_requested:
            experiment = (
                re.sub(
                    r"[^A-Za-z0-9._-]+",
                    "-",
                    str(cfg.trainer.get("experiment_name") or "default"),
                ).strip("-._")
                or "default"
            )
            cfg.trainer.default_local_dir = os.path.join(
                str(cfg.trainer.default_local_dir),
                experiment,
                self._run_leaf,
            )
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

        # Provider trainer state is the durable half of the resume contract.
        # ``infra.close()`` must release local/client handles without deleting a
        # fresh or reattached trainer after an exception, Ctrl-C, or relaunch.
        return init_fireworks_infra(
            "sft",
            provision_cfg,
            base_url=base_url,
            cleanup_on_close=False,
            cleanup_existing=False,
        )

    def fit(self) -> None:
        if self._config is None:
            self.build_config()
        config = self._config

        try:
            from training.utils.checkpoints import TrainingCheckpoints
            from training.utils.client import DEFAULT_TIMEOUT_S
        except ImportError as e:
            raise SFTConfigError(f"Fireworks SFT backend requires the Fireworks training SDK: {e}") from None

        from rllm.trainer.sft.tinker_dataset import count_loss_tokens, sum_loss_weights
        from rllm.utils.tracking import Tracking

        api_key = os.environ.get("FIREWORKS_API_KEY", "")
        if not api_key:
            raise SFTConfigError("FIREWORKS_API_KEY is not set; required for the fireworks SFT backend.")
        base_url = os.environ.get("FIREWORKS_BASE_URL", config.get("fireworks_base_url", "https://api.fireworks.ai"))

        # A generated path is isolated per backend instance. An explicit path is
        # stable because Fireworks keeps its exact data cursor locally beside the
        # run manifest; changing this path would make provider resume ambiguous.
        if not self._resume_requested and os.path.exists(config.trainer.default_local_dir):
            raise SFTConfigError("The generated Fireworks run directory already exists; create a new backend instance so a fresh isolated directory can be selected.")
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

            n_batches = len(train_dataset)
            total_epochs = config.trainer.get("total_epochs", 1)
            max_steps = config.trainer.get("max_steps")
            total_steps = resolve_training_steps(n_batches, total_epochs, max_steps)
            progress_denominator = total_steps
            optimizer = resolve_sft_optimizer_settings(config.optim, total_steps=total_steps)
            save_every = config.trainer.get("save_freq", 20)
            eval_every = config.trainer.get("test_freq", 10)

            train_dataset.preflight(
                label="train",
                planned_batches=(
                    (epoch_idx, batch_idx)
                    for _step, epoch_idx, batch_idx in iter_training_batches_from_step(
                        n_batches=n_batches,
                        total_epochs=total_epochs,
                        start_step=0,
                        max_steps=max_steps,
                    )
                ),
            )
            if val_dataset is not None:
                val_dataset.preflight(label="validation")

            resume_contract = build_fireworks_resume_contract(
                config,
                train_dataset,
                optimizer,
                n_batches=n_batches,
                total_steps=total_steps,
            )
            configured_job_id = _configured_policy_job_id(config)
            prepared_manifest = prepare_fireworks_resume_contract(
                config.trainer.default_local_dir,
                resume_contract,
                configured_job_id=configured_job_id,
            )

            infra = self._provision(config, api_key, base_url)
            try:
                # Persist the provider identity before any checkpoint call, so a
                # failure during resume discovery cannot orphan a durable job.
                prepared_manifest = validate_fireworks_resume_contract(
                    prepared_manifest,
                    configured_job_id=configured_job_id,
                    actual_job_id=infra.policy_job_id,
                )
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
                validate_fireworks_resume_contract(
                    prepared_manifest,
                    configured_job_id=configured_job_id,
                    actual_job_id=infra.policy_job_id,
                    resume_info=resume,
                )
                if resume is not None:
                    data_consumed = getattr(resume, "data_consumed", None)
                    if isinstance(data_consumed, bool) or not isinstance(data_consumed, int) or data_consumed <= 0:
                        raise SFTConfigError(
                            "Fireworks found a resumable checkpoint without a positive persisted dataset cursor; "
                            "refusing to replay training data. Use a new trainer job or restore its dataloader.json."
                        )
                    # Fireworks may rename the requested checkpoint (for example
                    # step-42 to step-0), so its name-derived ``step`` is not a
                    # data cursor. The raw-row cursor persisted by rLLM is.
                    start_step = train_dataset.step_for_data_cursor(data_consumed)
                else:
                    start_step = 0
                if not 0 <= start_step <= total_steps:
                    raise SFTConfigError(f"Fireworks checkpoint step {start_step!r} is outside the resolved SFT horizon 0..{total_steps}; use a new trainer job or an intact rLLM checkpoint.")

                logger.info(f"Training for {n_batches} batches x {total_epochs} epochs = {total_steps} steps")

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

                current_epoch: int | None = None
                pending: _SubmittedFireworksBatch | None = None

                def submit_batch(step: int, epoch_idx: int, batch_idx: int):
                    nonlocal current_epoch
                    if epoch_idx != current_epoch:
                        logger.info("Starting epoch %d", epoch_idx)
                        train_dataset.set_epoch(seed=epoch_idx)
                        current_epoch = epoch_idx
                    started_at = time.time()
                    lr = optimizer.learning_rate * sft_lr_multiplier(
                        optimizer.lr_schedule,
                        step,
                        total_steps,
                        optimizer.warmup_steps_ratio,
                        optimizer.warmup_steps,
                        optimizer.min_lr_ratio,
                    )
                    adam = build_adam_params(
                        learning_rate=lr,
                        betas=optimizer.betas,
                        eps=optimizer.eps,
                        weight_decay=optimizer.weight_decay,
                        grad_clip_norm=optimizer.grad_clip_norm,
                    )
                    # The pinned Fireworks SDK serializes this Adam field and
                    # returns provider-named global/RMS grad metrics. It does not
                    # add gradient normalization; Datum weights remain the sole
                    # implementation of token_mean.
                    adam = adam.model_copy(update={"emit_grad_norm_metrics": "basic"})
                    data = train_dataset.get_batch(batch_idx)
                    fb_fut = client.submit_forward_backward(data, loss_fn="cross_entropy")
                    # Datum weights encode reduction; provider normalization would double-divide token_mean.
                    opt_fut = client.submit_optim_step(adam)
                    return _SubmittedFireworksBatch(
                        step=step,
                        data=data,
                        learning_rate=lr,
                        started_at=started_at,
                        fb_future=fb_fut,
                        opt_future=opt_fut,
                    )

                def finish_batch(submitted: _SubmittedFireworksBatch):
                    fb_result = submitted.fb_future.result(timeout=DEFAULT_TIMEOUT_S)
                    opt_result = submitted.opt_future.result(timeout=DEFAULT_TIMEOUT_S)
                    # Fireworks exposes only aggregate loss. Divide by the
                    # submitted weight mass, while logging the independent count
                    # of positive-weight tokens (normalization can rescale mass).
                    n_loss_tokens = count_loss_tokens(submitted.data)
                    loss_weight = sum_loss_weights(submitted.data)
                    train_loss = _fireworks_mean_loss(fb_result, loss_weight)
                    completed_steps = submitted.step + 1
                    metrics = {
                        "learning_rate": submitted.learning_rate,
                        "progress": min(completed_steps / progress_denominator, 1.0),
                        "num_sequences": len(submitted.data),
                        "num_loss_tokens": n_loss_tokens,
                        "train_loss": train_loss,
                        "time/total": time.time() - submitted.started_at,
                    }
                    metrics.update(_fireworks_optimizer_metrics(opt_result))
                    if should_validate_step(
                        completed_steps,
                        eval_every=eval_every,
                        has_validation=val_dataset is not None,
                    ):
                        metrics.update(self._validate(client, val_dataset, DEFAULT_TIMEOUT_S))
                    if save_every > 0 and completed_steps % save_every == 0 and completed_steps < total_steps:
                        logger.info("Saving checkpoint at step %d", completed_steps)
                        ckpt.save(
                            f"step-{completed_steps}",
                            resumable=True,
                            promotable=False,
                            data_consumed=train_dataset.data_cursor_for_step(completed_steps),
                        )
                    tracking_logger.log(data=metrics, step=completed_steps)
                    logger.info(
                        "Step %d: train_loss=%.4f, lr=%.2e",
                        completed_steps,
                        train_loss,
                        submitted.learning_rate,
                    )

                for step, epoch_idx, batch_idx in iter_training_batches_from_step(
                    n_batches=n_batches,
                    total_epochs=total_epochs,
                    start_step=start_step,
                    max_steps=max_steps,
                ):
                    if pending is None:
                        pending = submit_batch(step, epoch_idx, batch_idx)
                        continue

                    if _is_step_boundary(
                        pending.step + 1,
                        total_steps,
                        save_every=save_every,
                        eval_every=eval_every,
                        has_validation=val_dataset is not None,
                    ):
                        finish_batch(pending)
                        pending = submit_batch(step, epoch_idx, batch_idx)
                    else:
                        following = submit_batch(step, epoch_idx, batch_idx)
                        finish_batch(pending)
                        pending = following
                if pending is not None:
                    finish_batch(pending)

                if total_steps > start_step:
                    logger.info(f"Saving final checkpoint at step {total_steps}")
                    # promotable=True writes the servable sampler row that
                    # ``promote_latest`` needs. Without it the weights are a
                    # resumable-only DCP blob, GC'd after the job's ~30-day retention
                    # window, so promote it before detaching from the retained
                    # trainer job. Mirrors the RL path and the SDK sft recipe.
                    ckpt.save(
                        f"step-{total_steps}",
                        resumable=True,
                        promotable=True,
                        data_consumed=train_dataset.data_cursor_for_step(total_steps),
                    )
                    artifact = "LoRA adapter" if lora_rank else "full-weight model"
                    experiment = config.trainer.get("experiment_name") or "default"
                    output_model_id = re.sub(r"[^a-z0-9-]+", "-", f"{config.trainer.get('project_name', 'rllm-sft')}-{experiment}".lower()).strip("-")[:63]
                    try:
                        model = ckpt.promote_latest(output_model_id, config.model.name)
                        logger.info("Promoted final %s -> %s", artifact, (model or {}).get("name", output_model_id))
                    except Exception:
                        logger.exception(
                            "Final %s promotion failed. The promotable sampler checkpoint for job %s "
                            "remains available for manual promotion via "
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

        from rllm.trainer.sft.tinker_dataset import count_loss_tokens

        logger.info("Running validation...")
        total_nll = 0.0
        total_tokens = 0
        for batch_idx in range(len(val_dataset)):
            data = val_dataset.get_batch(batch_idx)
            weights = [datum.loss_fn_inputs["weights"] for datum in data]
            batch_tokens = count_loss_tokens(data)
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
