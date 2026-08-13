"""Tinker SFT backend.

Owns the full Tinker SFT loop (migrated from the removed
``rllm.trainer.deprecated.tinker_sft_trainer``). ``tinker``/``tinker_cookbook``
are imported lazily inside :meth:`fit` so the module — and the dispatcher that
imports it — stay importable without the tinker stack installed.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig, OmegaConf

from rllm.trainer.sft.backend import SFTBackend, SFTConfigError, validate_messages_dataset

if TYPE_CHECKING:
    import tinker
    from tinker.lib.public_interfaces import APIFuture

logger = logging.getLogger(__name__)

_CONFIG_FILE = Path(__file__).resolve().parent / "config" / "tinker.yaml"

# Plain-text renderers that cannot represent reasoning (``<think>``) or tool-calls.
_PLAIN_RENDERERS = {"role_colon", "llama3"}


def _guard_renderer_capability(renderer_name: str, renderer_source: str, train_data) -> None:
    """Reject structured rows when the resolved renderer cannot prove support.

    Plain renderers cannot represent reasoning or tool calls. An automatically
    selected chat-template fallback is likewise unverified, so it may train only
    inspectable text-only data; file-backed input must pin a capable renderer.
    """
    fallback = renderer_source == "chat_template"
    if renderer_name not in _PLAIN_RENDERERS and not fallback:
        return
    if not hasattr(train_data, "get_data"):
        raise SFTConfigError(f"Hosted SFT cannot verify structured-message support for the {renderer_name!r} renderer on file-backed data. Pin a capable renderer.")
    try:
        rows = train_data.get_data()
    except Exception as e:  # noqa: BLE001 - convert inspection failure to config error
        raise SFTConfigError(f"Hosted SFT could not inspect data before using the {renderer_name!r} renderer. Pin a capable renderer.") from e
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("tools"):
            raise SFTConfigError(f"The {renderer_name!r} renderer cannot safely represent tool declarations in hosted SFT. Pin a capable renderer.")
        for msg in row.get("messages") or []:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            has_thinking = isinstance(content, list) and any(isinstance(p, dict) and p.get("type") == "thinking" for p in content)
            if has_thinking or msg.get("tool_calls"):
                raise SFTConfigError(
                    f"The {renderer_name!r} renderer cannot represent reasoning (<think>) or tool-calls, "
                    "but the training data contains structured messages (thinking parts / tool_calls). "
                    "Pass --renderer to pin a capable renderer (e.g. qwen3 / qwen3_5 / deepseekv3), or use "
                    "a chat model whose default renderer supports them."
                )


def build_sft_data(config, train_data, val_data):
    """Build (tokenizer, train_dataset, val_dataset) from a backend config.

    Shared by the Tinker and Fireworks SFT backends: both use rLLM's production
    renderer, then package the resulting tokens and loss mask as Tinker Datums.
    """
    tokenize_method = config.data.get("rllm", {}).get("tokenize_and_mask_method", "cumulative")
    if tokenize_method == "hf_template":
        raise SFTConfigError(
            "Hosted SFT does not support tokenize_and_mask_method='hf_template': "
            "it bypasses the canonical renderer and its exact trainable-token attribution. "
            "Use 'cumulative' or 'stepwise', or use the verl backend for hf_template."
        )

    from tinker_cookbook.tokenizer_utils import get_tokenizer

    from rllm.renderers import resolve
    from rllm.trainer.sft.tinker_dataset import create_tinker_sft_datasets

    # Fireworks' model.name is a FW model path (accounts/fireworks/models/...),
    # not HF-resolvable, so render/tokenize from the HF tokenizer_model when set.
    tokenizer_name = config.model.get("tokenizer_model") or config.model.name
    tokenizer = get_tokenizer(tokenizer_name)
    # Training and serving resolve through the same production renderer layer
    # with the same template-faithful history policy.
    explicit = config.data.get("renderer_name", None)
    try:
        if explicit:
            resolution = resolve(
                tokenizer_name,
                tokenizer,
                renderer_name=explicit,
            )
        else:
            resolution = resolve(tokenizer_name, tokenizer)
    except Exception as e:  # noqa: BLE001 - surface renderer setup as config
        raise SFTConfigError(f"Could not initialize SFT renderer for {tokenizer_name!r}: {e}") from e
    _guard_renderer_capability(resolution.name, resolution.source, train_data)
    if val_data is not None:
        _guard_renderer_capability(resolution.name, resolution.source, val_data)
    renderer = resolution.renderer
    # Masking is always CUSTOMIZED, driven by each message's ``trainable`` flag:
    # rows from ``from-eval``'s automerge carry the flags directly; flag-less rows
    # (e.g. an external ``--train-file``) get a derived default in the dataset
    # loader. ``tokenize_and_mask_method=stepwise`` only selects that default
    # (train just the last assistant turn) rather than the all-assistant default.
    last_only = tokenize_method == "stepwise"
    logger.info(
        "Using canonical renderer: %s (%s), masking=trainable (last_only=%s)",
        resolution.name,
        resolution.source,
        last_only,
    )

    train_batch_size = config.data.get("train_batch_size", 32)
    val_batch_size = config.data.get("micro_batch_size_per_gpu", train_batch_size)
    rllm_data = config.data.get("rllm", {})
    train_dataset, val_dataset = create_tinker_sft_datasets(
        train_data=train_data,
        val_data=val_data,
        renderer=renderer,
        batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        max_length=config.data.get("max_length", None),
        last_only=last_only,
        max_train_samples=config.data.get("train_max_samples", -1),
        max_val_samples=config.data.get("val_max_samples", -1),
        overlength_policy=str(rllm_data.get("overlength_policy", "truncate")),
        loss_reduction=str(rllm_data.get("loss_reduction", "token_mean")),
    )
    return tokenizer, train_dataset, val_dataset


def should_validate_step(
    completed_steps: int,
    *,
    eval_every: int,
    has_validation: bool,
    include_initial: bool = False,
) -> bool:
    """Whether validation belongs at this completed-update boundary."""
    if not has_validation or eval_every <= 0:
        return False
    if completed_steps == 0:
        return include_initial
    return completed_steps % eval_every == 0


def resolve_training_steps(
    n_batches: int,
    total_epochs: int,
    max_steps: int | None,
) -> int:
    """Resolve an optional positive step cap against the available batches."""
    if n_batches <= 0:
        raise SFTConfigError("The SFT training dataset contains no batches.")
    if isinstance(total_epochs, bool) or not isinstance(total_epochs, int) or total_epochs <= 0:
        raise SFTConfigError(f"trainer.total_epochs must be a positive integer, got {total_epochs!r}.")
    available_steps = n_batches * total_epochs
    if max_steps is None:
        return available_steps
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise SFTConfigError(f"trainer.max_steps must be a positive integer when set, got {max_steps!r}.")
    return min(available_steps, max_steps)


def iter_training_batches(
    *,
    n_batches: int,
    total_epochs: int,
    start_epoch: int = 0,
    start_batch: int = 0,
    max_steps: int | None = None,
):
    """Yield ``(step, epoch, batch)`` up to the exact effective horizon.

    ``start_epoch`` and ``start_batch`` are an input cursor supplied by the
    existing checkpoint layer. Defining how that cursor is persisted or
    restored is intentionally outside this scheduling helper.
    """
    total_steps = resolve_training_steps(n_batches, total_epochs, max_steps)
    start_step = start_epoch * n_batches + start_batch
    for step in range(start_step, total_steps):
        epoch, batch = divmod(step, n_batches)
        yield step, epoch, batch


def sft_lr_multiplier(
    lr_schedule: str,
    step: int,
    total_steps: int,
    warmup_steps_ratio: float = 0.0,
    warmup_steps: int | None = -1,
    min_lr_ratio: float = 0.0,
) -> float:
    """Apply linear warmup followed by the selected decay schedule."""
    from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier

    if total_steps <= 0:
        raise SFTConfigError(f"total_steps must be positive, got {total_steps}.")
    if step < 0:
        raise SFTConfigError(f"optimizer step must be non-negative, got {step}.")
    if not 0 <= warmup_steps_ratio <= 1:
        raise SFTConfigError(f"optim.warmup_steps_ratio must be in [0, 1], got {warmup_steps_ratio}.")
    if not 0 <= min_lr_ratio <= 1:
        raise SFTConfigError(f"optim.min_lr / optim.lr must be in [0, 1], got {min_lr_ratio}.")
    resolved_warmup = -1 if warmup_steps is None else int(warmup_steps)
    if resolved_warmup < 0:
        resolved_warmup = int(total_steps * (warmup_steps_ratio or 0.0))
    if resolved_warmup > 0 and step < resolved_warmup:
        return step / resolved_warmup
    decay = compute_schedule_lr_multiplier(
        lr_schedule=lr_schedule,
        step=step - resolved_warmup,
        total_steps=max(total_steps - resolved_warmup, 1),
    )
    return min_lr_ratio + (1 - min_lr_ratio) * decay


def build_adam_params(
    *,
    learning_rate: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    grad_clip_norm: float,
):
    import tinker

    return tinker.AdamParams(
        learning_rate=learning_rate,
        beta1=betas[0],
        beta2=betas[1],
        eps=eps,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
    )


@dataclass(frozen=True)
class SFTOptimizerSettings:
    learning_rate: float
    lr_schedule: str
    warmup_steps_ratio: float
    warmup_steps: int
    min_lr_ratio: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    grad_clip_norm: float


def resolve_sft_optimizer_settings(optim_cfg, *, total_steps: int) -> SFTOptimizerSettings:
    """Validate optimizer controls and normalize them for both hosted loops."""
    if total_steps <= 0:
        raise SFTConfigError(f"total_steps must be positive, got {total_steps}.")
    try:
        learning_rate = float(optim_cfg.get("lr", 1e-5))
        min_learning_rate = float(optim_cfg.get("min_lr", 0.0))
    except (TypeError, ValueError) as e:
        raise SFTConfigError("optim.lr and optim.min_lr must be numeric.") from e
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise SFTConfigError(f"optim.lr must be positive, got {learning_rate}.")
    if not math.isfinite(min_learning_rate) or not 0 <= min_learning_rate <= learning_rate:
        raise SFTConfigError(f"optim.min_lr must be between zero and optim.lr, got {min_learning_rate} with lr={learning_rate}.")

    lr_schedule = str(optim_cfg.get("lr_scheduler", "constant"))
    if lr_schedule not in {"constant", "linear", "cosine"}:
        raise SFTConfigError(f"optim.lr_scheduler must be constant, linear, or cosine, got {lr_schedule!r}.")
    try:
        warmup_steps_ratio = float(optim_cfg.get("warmup_steps_ratio", 0.0) or 0.0)
    except (TypeError, ValueError) as e:
        raise SFTConfigError("optim.warmup_steps_ratio must be numeric.") from e
    if not math.isfinite(warmup_steps_ratio) or not 0 <= warmup_steps_ratio <= 1:
        raise SFTConfigError(f"optim.warmup_steps_ratio must be in [0, 1], got {warmup_steps_ratio}.")
    raw_warmup_steps = optim_cfg.get("warmup_steps", -1)
    if isinstance(raw_warmup_steps, bool) or not isinstance(raw_warmup_steps, int) or raw_warmup_steps < -1:
        raise SFTConfigError(f"optim.warmup_steps must be -1 or a non-negative integer, got {raw_warmup_steps!r}.")
    warmup_steps = raw_warmup_steps
    if warmup_steps > total_steps:
        raise SFTConfigError(f"optim.warmup_steps cannot exceed total training steps ({total_steps}), got {warmup_steps}.")

    raw_betas = optim_cfg.get("betas", [0.9, 0.95])
    if isinstance(raw_betas, str | bytes) or not hasattr(raw_betas, "__len__") or len(raw_betas) != 2:
        raise SFTConfigError(f"optim.betas must contain beta1 and beta2, got {raw_betas}.")
    try:
        betas = (float(raw_betas[0]), float(raw_betas[1]))
    except (TypeError, ValueError) as e:
        raise SFTConfigError(f"optim.betas must be numeric, got {raw_betas}.") from e
    if any(not math.isfinite(beta) or not 0 <= beta < 1 for beta in betas):
        raise SFTConfigError(f"optim.betas must each be in [0, 1), got {betas}.")
    try:
        eps = float(optim_cfg.get("eps", 1e-8))
        weight_decay = float(optim_cfg.get("weight_decay", 0.0))
        grad_clip_norm = float(optim_cfg.get("grad_clip_norm", 0.0))
    except (TypeError, ValueError) as e:
        raise SFTConfigError("optim.eps, optim.weight_decay, and optim.grad_clip_norm must be numeric.") from e
    if not math.isfinite(eps) or eps <= 0:
        raise SFTConfigError(f"optim.eps must be positive, got {eps}.")
    if any(not math.isfinite(value) or value < 0 for value in (weight_decay, grad_clip_norm)):
        raise SFTConfigError("optim.weight_decay and optim.grad_clip_norm must be non-negative.")

    return SFTOptimizerSettings(
        learning_rate=learning_rate,
        lr_schedule=lr_schedule,
        warmup_steps_ratio=warmup_steps_ratio,
        warmup_steps=warmup_steps,
        min_lr_ratio=min_learning_rate / learning_rate,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
    )


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
    # Tinker's SDK exposes only LoRA training clients (create_lora_training_client;
    # state-resume asserts is_lora), so full-parameter (rank 0) is impossible here.
    supports_full_finetune = False

    def __init__(self, spec):
        super().__init__(spec)
        self._config: DictConfig | None = None

    # -- contract -----------------------------------------------------------

    def validate_spec(self) -> None:
        if self.spec.tokenize_method == "hf_template":
            raise SFTConfigError(
                f"The {self.name!r} hosted SFT backend does not support tokenize_method='hf_template': "
                "it bypasses the canonical renderer and exact trainable-token attribution. "
                "Use 'cumulative' or 'stepwise', or use the verl backend."
            )
        if self._effective_lora_rank() == 0 and not self.supports_full_finetune:
            raise SFTConfigError(
                f"--lora-rank 0 (full-parameter fine-tuning) is not supported by the {self.name!r} SFT backend: "
                "the tinker SDK only exposes LoRA training clients. Use --lora-rank >= 1 here, or switch to "
                "--backend fireworks (full-weight POLICY_TRAINER shapes) or --backend verl."
            )
        validate_messages_dataset(self.spec.train_dataset, "train")
        if self.spec.val_dataset is not None:
            validate_messages_dataset(self.spec.val_dataset, "val")

    def _effective_lora_rank(self) -> int:
        """The rank the training loop will actually see: an overrides
        ``model.lora_rank`` (e.g. from ``rllm sft --config``) beats the spec
        field, exactly as ``build_config`` merges it."""
        if self.spec.overrides:
            user = OmegaConf.to_container(OmegaConf.create(self.spec.overrides), resolve=False)
            model = user.get("model") if isinstance(user, dict) and isinstance(user.get("model"), dict) else {}
            if model.get("lora_rank") is not None:
                return int(model["lora_rank"])
        return int(self.spec.lora_rank)

    def _config_template(self) -> Path:
        """Path to the backend's native config template (overridden per backend)."""
        return _CONFIG_FILE

    def build_config(self) -> DictConfig:
        """SFTSpec → the DictConfig shape the tinker/fireworks loop reads."""
        spec = self.spec
        base = OmegaConf.load(str(self._config_template()))
        trainer_overrides: dict[str, Any] = {
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
                "model": {"name": spec.model, "lora_rank": spec.lora_rank},
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
        from tinker_cookbook.utils.misc_utils import timed

        from rllm.trainer.sft.tinker_dataset import count_loss_tokens
        from rllm.utils.tracking import Tracking

        config = self._config
        os.makedirs(config.trainer.default_local_dir, exist_ok=True)
        tokenizer, train_dataset, val_dataset = build_sft_data(config, self.spec.train_dataset, self.spec.val_dataset)

        resume_info = checkpoint_utils.get_last_checkpoint(config.trainer.default_local_dir)
        start_epoch = resume_info.get("epoch", 0) if resume_info else 0
        start_batch = resume_info.get("batch", 0) if resume_info else 0
        n_batches = len(train_dataset)
        total_epochs = config.trainer.get("total_epochs", 1)
        max_steps = config.trainer.get("max_steps")
        total_steps = resolve_training_steps(n_batches, total_epochs, max_steps)
        progress_denominator = total_steps
        optimizer = resolve_sft_optimizer_settings(config.get("optim", {}), total_steps=total_steps)
        save_every = config.trainer.get("save_freq", 20)
        eval_every = config.trainer.get("test_freq", 10)

        train_dataset.preflight(
            label="train",
            planned_batches=(
                (epoch_idx, batch_idx)
                for _step, epoch_idx, batch_idx in iter_training_batches(
                    n_batches=n_batches,
                    total_epochs=total_epochs,
                    start_epoch=start_epoch,
                    start_batch=start_batch,
                    max_steps=max_steps,
                )
            ),
        )
        if val_dataset is not None:
            val_dataset.preflight(label="validation")

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

        # Wrap the loop so tracking_logger.finish() runs even on failure: the 'ui'
        # backend tees stdout/stderr and holds an open session until finish().
        try:
            if resume_info:
                logger.info(f"Resuming from checkpoint: {resume_info}")
                training_client = await service_client.create_training_client_from_state_async(resume_info["state_path"])
            else:
                logger.info("Starting training from scratch")
                training_client = await service_client.create_lora_training_client_async(
                    base_model=config.model.name,
                    rank=config.model.get("lora_rank", 32),
                    train_unembed=OmegaConf.select(config, "model.train_unembed", default=True),
                    train_attn=OmegaConf.select(config, "model.train_attn", default=True),
                    train_mlp=OmegaConf.select(config, "model.train_mlp", default=True),
                )
            logger.info(f"Training for {n_batches} batches x {total_epochs} epochs = {total_steps} steps")

            if should_validate_step(
                0,
                eval_every=eval_every,
                has_validation=val_dataset is not None,
                include_initial=start_epoch == 0 and start_batch == 0,
            ):
                initial_metrics: dict[str, Any] = {}
                with timed("validation", initial_metrics):
                    initial_metrics.update(await self._validate(training_client, val_dataset, compute_mean_nll))
                tracking_logger.log(data=initial_metrics, step=0)

            async def submit_batch(epoch_idx: int, batch_idx: int) -> _SubmittedBatch:
                step = epoch_idx * n_batches + batch_idx
                batch_start_time = time.time()
                metrics: dict[str, Any] = {"epoch": epoch_idx, "progress": step / progress_denominator}
                learning_rate = optimizer.learning_rate * sft_lr_multiplier(
                    optimizer.lr_schedule,
                    step,
                    total_steps,
                    optimizer.warmup_steps_ratio,
                    optimizer.warmup_steps,
                    optimizer.min_lr_ratio,
                )
                metrics["learning_rate"] = learning_rate
                adam_params = build_adam_params(
                    learning_rate=learning_rate,
                    betas=optimizer.betas,
                    eps=optimizer.eps,
                    weight_decay=optimizer.weight_decay,
                    grad_clip_norm=optimizer.grad_clip_norm,
                )

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
                    num_loss_tokens=count_loss_tokens(submitted.data),
                    train_mean_nll=train_nll,
                )
                metrics["time/total"] = time.time() - submitted.batch_start_time

                completed_steps = submitted.step + 1
                if should_validate_step(
                    completed_steps,
                    eval_every=eval_every,
                    has_validation=val_dataset is not None,
                ):
                    with timed("validation", metrics):
                        val_metrics = await self._validate(training_client, val_dataset, compute_mean_nll)
                    metrics.update(val_metrics)

                tracking_logger.log(data=metrics, step=completed_steps)
                logger.info(f"Step {completed_steps}: train_nll={train_nll:.4f}, lr={metrics['learning_rate']:.2e}")

            pending: _SubmittedBatch | None = None
            current_epoch: int | None = None
            for _step, epoch_idx, batch_idx in iter_training_batches(
                n_batches=n_batches,
                total_epochs=total_epochs,
                start_epoch=start_epoch,
                start_batch=start_batch,
                max_steps=max_steps,
            ):
                if epoch_idx != current_epoch:
                    logger.info(f"Starting epoch {epoch_idx}")
                    train_dataset.set_epoch(seed=epoch_idx)
                    current_epoch = epoch_idx
                if pending is not None and should_validate_step(
                    pending.step + 1,
                    eval_every=eval_every,
                    has_validation=val_dataset is not None,
                ):
                    await finish_batch(pending)
                    pending = None
                submitted = await submit_batch(epoch_idx, batch_idx)
                if pending is not None:
                    await finish_batch(pending)
                pending = submitted
            if pending is not None:
                await finish_batch(pending)

            start_step = start_epoch * n_batches + start_batch
            if start_step < total_steps:
                final_epoch, final_batch = divmod(total_steps, n_batches)
                await checkpoint_utils.save_checkpoint_async(
                    training_client=training_client,
                    name="final",
                    log_path=config.trainer.default_local_dir,
                    kind="both",
                    loop_state={"epoch": final_epoch, "batch": final_batch},
                )
            else:
                logger.info("Training was already complete; nothing to do")

            tracking_logger.log(data={"status": "completed"}, step=total_steps)
            logger.info("Training completed successfully")
        finally:
            try:
                tracking_logger.finish()
            except Exception:
                pass

    @staticmethod
    async def _validate(training_client, val_dataset, compute_mean_nll) -> dict[str, float]:
        """Compute held-out NLL without adding validation gradients."""
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
            forward_future = await training_client.forward_async(data, loss_fn="cross_entropy")
            forward_result = await forward_future.result_async()
            logprobs = [output["logprobs"] for output in forward_result.loss_fn_outputs]
            batch_nll = compute_mean_nll(logprobs, weights)
            total_nll += batch_nll * batch_tokens
            total_tokens += batch_tokens
        if total_tokens <= 0:
            raise SFTConfigError("The validation dataset has no trainable tokens after rendering and masking.")
        val_nll = total_nll / total_tokens
        logger.info(f"Validation NLL: {val_nll:.4f}")
        return {"test/mean_nll": val_nll}
