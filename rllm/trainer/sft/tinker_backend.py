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

        # Wrap the loop so tracking_logger.finish() runs even on failure: the 'ui'
        # backend tees stdout/stderr and holds an open session until finish().
        try:
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

            # len(dataset) floors examples//batch_size; keep the final partial batch
            # when the dataset is smaller than one batch (else 0 steps).
            n_batches = max(1, len(train_dataset))
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
            for epoch_idx in range(start_epoch, total_epochs):
                logger.info(f"Starting epoch {epoch_idx}")
                train_dataset.set_epoch(seed=epoch_idx)
                start_batch_idx = start_batch if epoch_idx == start_epoch else 0
                for batch_idx in range(start_batch_idx, n_batches):
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
            logger.info("Training completed successfully")
        finally:
            try:
                tracking_logger.finish()
            except Exception:
                pass

    @staticmethod
    async def _validate(training_client, val_dataset, compute_mean_nll) -> dict[str, float]:
        """Compute held-out NLL without adding validation gradients."""
        logger.info("Running validation...")
        total_nll = 0.0
        total_tokens = 0
        for batch_idx in range(len(val_dataset)):
            data = val_dataset.get_batch(batch_idx)
            weights = [datum.loss_fn_inputs["weights"] for datum in data]
            batch_tokens = sum(sum(weight.data) for weight in weights)
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
