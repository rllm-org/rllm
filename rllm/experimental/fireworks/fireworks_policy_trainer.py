"""Policy training module for Firetitan-based RL.

Uses ``FiretitanTrainingClient`` / ``ReconnectableClient`` from the
Fireworks training SDK instead of Tinker's ``ServiceClient``.

This module handles gradient updates, model checkpointing, and data processing.
It does NOT contain any environment or agent logic.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

import tinker
from fireworks.training.sdk import WeightSyncer
from tinker.types import AdamParams
from training.utils.client import ReconnectableClient

from rllm.experimental.common import (
    AlgorithmConfig,
    CompactFilteringConfig,
    TransformConfig,
)
from rllm.trainer.tinker.tinker_policy_trainer import (
    compute_schedule_lr_multiplier,
    require_training_client,
)
from rllm.trainer.tinker.transform import transform_trajectory_groups_to_datums
from rllm.types import TrajectoryGroup

logger = logging.getLogger(__name__)


class FireworksPolicyTrainer:
    """Handles policy updates via gradient descent using Fireworks Firetitan.

    This class handles:
    - Training client management (``ReconnectableClient`` with auto-reconnect)
    - Data processing (filtering, advantages, datum conversion)
    - Forward-backward passes
    - Optimizer steps
    - Checkpoint saving / loading
    - Weight syncing to an inference deployment (``WeightSyncer``)

    It does NOT handle:
    - Environment or agent interactions
    - Trajectory collection
    - Sampling
    """

    _METRIC_SKIP_KEYS = {"step_id", "step"}
    _STEP_CHECKPOINT_RE = re.compile(r"(?:^|/)step-(\d+)$")

    def __init__(
        self,
        config,
        training_client: ReconnectableClient,
        weight_syncer: WeightSyncer | None = None,
        cf_config: CompactFilteringConfig | None = None,
        transform_config: TransformConfig | None = None,
        algorithm_config: AlgorithmConfig | None = None,
        rlor_mgr=None,
        policy_job_id: str | None = None,
    ):
        self.config = config
        self.training_client = training_client
        self.weight_syncer = weight_syncer
        self._rlor_mgr = rlor_mgr
        self._policy_job_id = policy_job_id
        self._resume_checkpoint_name = self.config.training.get("resume_from_dcp_checkpoint")
        self._resume_source_job_id = self.config.training.get("resume_from_fireworks_job_id") or policy_job_id

        self.cf_config = cf_config or CompactFilteringConfig.from_config(self.config.rllm.compact_filtering)
        self.transform_config = transform_config or TransformConfig()
        self.algorithm_config = algorithm_config or AlgorithmConfig.from_config(self.config.rllm.algorithm)
        self.resolve_builtin_loss(self.algorithm_config)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    async def initialize_async(
        self,
        resume_from_checkpoint: bool = True,
        hot_load_before_training: bool = False,
    ) -> int:
        """Initialize or resume training.

        Handles checkpoint resume via ``FiretitanTrainingClient.list_checkpoints``
        and ``load_state_with_optimizer``.

        Args:
            resume_from_checkpoint: If True, attempt to resume from the
                last DCP checkpoint.
            hot_load_before_training: If True, push initial weights to the
                inference deployment before the first training step.

        Returns:
            The starting global step (0 when training from scratch).
        """
        start_step = 0

        if resume_from_checkpoint:
            start_step = await self._try_resume()

        if start_step == 0:
            logger.info("Starting training from scratch with model: %s", self.config.model.name)
            if hot_load_before_training:
                await self._initial_weight_sync()

        return start_step

    async def _try_resume(self) -> int:
        """Attempt to resume from a DCP checkpoint.

        Returns:
            The step to resume from, or 0 if no checkpoint was found.
        """
        inner = self.training_client.inner
        source_job_id = self._resume_source_job_id
        checkpoint_name = self._resume_checkpoint_name

        if checkpoint_name:
            spec_source_job_id, checkpoint_name = self._parse_checkpoint_spec(checkpoint_name)
            source_job_id = spec_source_job_id or source_job_id
            logger.info(
                "Resuming from configured DCP checkpoint: %s (source job: %s)",
                checkpoint_name,
                source_job_id or self._policy_job_id,
            )
        else:
            checkpoints = self._list_resume_checkpoints(source_job_id)
            if not checkpoints:
                logger.info("No existing checkpoints found.")
                return 0

            checkpoint_name = checkpoints[-1]
            logger.info("Resuming from latest DCP checkpoint: %s", checkpoint_name)

        checkpoint_ref = inner.resolve_checkpoint_path(checkpoint_name, source_job_id=source_job_id)
        timeout = self.config.hotload.get("dcp_timeout", 2700)
        await asyncio.to_thread(self.training_client.load_state_with_optimizer, checkpoint_ref, timeout=timeout)

        step = self._parse_checkpoint_step(checkpoint_name)

        await self._sync_weights(f"resume-{step}")
        return step

    @staticmethod
    def _parse_checkpoint_spec(spec: str) -> tuple[str | None, str]:
        """Parse ``job_id:checkpoint_name`` or a plain checkpoint name."""
        if ":" in spec and not spec.startswith(("gs://", "/")):
            source_job_id, checkpoint_name = spec.split(":", 1)
            return source_job_id, checkpoint_name
        return None, spec

    def _list_resume_checkpoints(self, source_job_id: str | None) -> list[str]:
        """List DCP checkpoint names from the source job when available."""
        if self._rlor_mgr is not None and source_job_id:
            rows = self._rlor_mgr.list_checkpoints(source_job_id)
            checkpoints = [(row.get("name") or "").rstrip("/").rsplit("/", 1)[-1] for row in rows if self._is_dcp_checkpoint_row(row)]
            return sorted(checkpoints, key=self._parse_checkpoint_step)

        checkpoints = self.training_client.inner.list_checkpoints()
        if isinstance(checkpoints, tuple):
            checkpoints = checkpoints[0]
        return list(checkpoints)

    @staticmethod
    def _is_dcp_checkpoint_row(row: dict) -> bool:
        checkpoint_type = row.get("checkpointType") or ""
        return checkpoint_type.endswith("TRAINING") or checkpoint_type.endswith("TRAINING_LORA")

    @classmethod
    def _parse_checkpoint_step(cls, checkpoint_name: str) -> int:
        """Parse integer step from DCP names like ``step-50``."""
        match = cls._STEP_CHECKPOINT_RE.search(checkpoint_name)
        if not match:
            logger.warning("Could not parse step from checkpoint name: %s", checkpoint_name)
            return 0
        return int(match.group(1))

    async def _initial_weight_sync(self) -> None:
        """Push initial base weights to the inference deployment."""
        await self._sync_weights("step-0-base", checkpoint_type="base")

    async def _sync_weights(self, name: str, checkpoint_type: str | None = None) -> str | None:
        """Save sampler weights and hot-load them into the deployment.

        Returns the snapshot_name on success, None on failure."""
        if self.weight_syncer is None:
            return None
        snapshot_name = await asyncio.to_thread(
            self.weight_syncer.save_and_hotload,
            name,
            checkpoint_type=checkpoint_type,
        )
        logger.debug("Weights synced to deployment: %s", name)
        return snapshot_name

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _process_datums(
        raw_datums: list[tinker.Datum],
    ) -> tuple[list[tinker.Datum], list[float], list[list[float]], list[int], list[int]]:
        """Extract rollout data and rebuild clean datums matching cookbook format.

        Returns (clean_datums, advantages, inf_logprobs, prompt_lens, num_loss_tokens).
        """
        clean_datums: list[tinker.Datum] = []
        advantages: list[float] = []
        inf_logprobs: list[list[float]] = []
        prompt_lens: list[int] = []
        num_loss_tokens: list[int] = []
        for datum in raw_datums:
            # Extract rollout data
            inf_logprobs.append(list(datum.loss_fn_inputs["logprobs"].data))
            mask = datum.loss_fn_inputs["mask"].data
            adv_data = datum.loss_fn_inputs["advantages"].data
            prompt_len = len(mask) + 1
            scalar = 0.0
            for i, m in enumerate(mask):
                if m != 0:
                    prompt_len = i + 1
                    scalar = float(adv_data[i])
                    break
            prompt_lens.append(prompt_len)
            advantages.append(scalar)
            num_loss_tokens.append(int(sum(mask)))

            # Rebuild clean datum: target_tokens + loss_mask only
            inputs = {"target_tokens": datum.loss_fn_inputs["target_tokens"]}
            if "mask" in datum.loss_fn_inputs:
                inputs["loss_mask"] = datum.loss_fn_inputs["mask"]
            clean_datums.append(tinker.Datum(model_input=datum.model_input, loss_fn_inputs=inputs))

        return clean_datums, advantages, inf_logprobs, prompt_lens, num_loss_tokens

    async def _compute_proximal_logprobs(
        self,
        datums: list[tinker.Datum],
    ) -> list[list[float]]:
        """Compute proximal (pi_old) logprobs via policy.forward().

        Only called when ``bypass_mode=False`` (3-policy / decoupled PPO).
        """
        prox_fwd = await asyncio.to_thread(
            self.training_client.forward,
            datums,
            "cross_entropy",
        )
        return [out["logprobs"].data for out in prox_fwd.loss_fn_outputs]

    @staticmethod
    def _compute_rollout_entropy_metrics(datums: list[tinker.Datum]) -> dict[str, float]:
        total_logprob = 0.0
        total_tokens = 0
        for datum in datums:
            logprobs = datum.loss_fn_inputs["logprobs"].data
            mask = datum.loss_fn_inputs["mask"].data
            for lp, m in zip(logprobs, mask, strict=True):
                if m:
                    total_logprob += float(lp)
                    total_tokens += 1

        if total_tokens == 0:
            return {}

        entropy = -total_logprob / total_tokens
        return {
            "train/entropy": entropy,
            "train/perplexity": math.exp(entropy),
        }

    @staticmethod
    def _compute_offpolicy_metrics(
        old_logprobs: list[list[float]],
        rollout_logprobs: list[list[float]],
        masks: list[list[int]],
    ) -> dict[str, float]:
        import torch

        safety_bound = 20.0
        training_means = []
        rollout_means = []
        log_ratio_sums = []
        token_old = []
        token_rollout = []

        for old_lp, rollout_lp, mask in zip(old_logprobs, rollout_logprobs, masks, strict=False):
            active_old = []
            active_rollout = []
            for old, rollout, m in zip(old_lp, rollout_lp, mask, strict=False):
                if m:
                    active_old.append(float(old))
                    active_rollout.append(float(rollout))
            if not active_old:
                continue

            old_t = torch.tensor(active_old, dtype=torch.float32)
            rollout_t = torch.tensor(active_rollout, dtype=torch.float32)
            training_means.append(old_t.mean())
            rollout_means.append(rollout_t.mean())
            log_ratio_sums.append((old_t - rollout_t).sum())
            token_old.append(old_t)
            token_rollout.append(rollout_t)

        if not token_old:
            return {}

        mean_log_prob_training = torch.stack(training_means)
        mean_log_prob_rollout = torch.stack(rollout_means)
        old_flat = torch.cat(token_old)
        rollout_flat = torch.cat(token_rollout)
        log_ratio = old_flat - rollout_flat
        logprob_abs_diff = log_ratio.abs()
        old_prob = torch.exp(old_flat)
        rollout_prob = torch.exp(rollout_flat)
        prob_abs_diff = (old_prob - rollout_prob).abs()
        log_ratio_safe = torch.clamp(log_ratio, min=-safety_bound, max=safety_bound)
        ratio = torch.exp(log_ratio_safe)
        log_ppl_diff = mean_log_prob_rollout - mean_log_prob_training

        metrics = {
            "offpolicy/kl": (rollout_flat - old_flat).mean().item(),
            "offpolicy/k3_kl": (torch.exp(log_ratio) - log_ratio - 1).mean().item(),
            "offpolicy/logprob_abs_diff/mean": logprob_abs_diff.mean().item(),
            "offpolicy/logprob_abs_diff/min": logprob_abs_diff.min().item(),
            "offpolicy/logprob_abs_diff/max": logprob_abs_diff.max().item(),
            "offpolicy/prob_abs_diff/mean": prob_abs_diff.mean().item(),
            "offpolicy/prob_abs_diff/min": prob_abs_diff.min().item(),
            "offpolicy/prob_abs_diff/max": prob_abs_diff.max().item(),
            "offpolicy/training_ppl": torch.exp(-mean_log_prob_training).mean().item(),
            "offpolicy/training_log_ppl": (-mean_log_prob_training).mean().item(),
            "offpolicy/rollout_ppl": torch.exp(-mean_log_prob_rollout).mean().item(),
            "offpolicy/rollout_log_ppl": (-mean_log_prob_rollout).mean().item(),
            "offpolicy/log_ppl_diff": log_ppl_diff.mean().item(),
            "offpolicy/log_ppl_abs_diff": log_ppl_diff.abs().mean().item(),
            "offpolicy/log_ppl_diff_min": log_ppl_diff.min().item(),
            "offpolicy/log_ppl_diff_max": log_ppl_diff.max().item(),
            "offpolicy/ppl_ratio": torch.exp(log_ppl_diff).mean().item(),
            "offpolicy/ratio/mean": ratio.mean().item(),
            "offpolicy/ratio/min": ratio.min().item(),
            "offpolicy/ratio/max": ratio.max().item(),
        }

        if old_prob.numel() > 1:
            old_centered = old_prob - old_prob.mean()
            rollout_centered = rollout_prob - rollout_prob.mean()
            denom = torch.sqrt(old_centered.square().sum() * rollout_centered.square().sum())
            if denom.item() > 0.0:
                metrics["offpolicy/prob_pearson_corr"] = ((old_centered * rollout_centered).sum() / denom).item()

        metrics["offpolicy/chi2_token"] = (ratio.square().mean() - 1.0).item()

        log_ratio_sum = torch.stack(log_ratio_sums)
        log_ratio_sum_safe = torch.clamp(log_ratio_sum, min=-safety_bound, max=safety_bound)
        metrics["offpolicy/chi2_seq"] = (torch.exp(2.0 * log_ratio_sum_safe).mean() - 1.0).item()

        return metrics

    def resolve_builtin_loss(self, algorithm_config: AlgorithmConfig, profile=None):
        """Resolve the builtin server-side loss kernel at setup time.

        Must be called before the first forward-backward pass.
        Raises ValueError if the loss has no builtin kernel or PP > 1.
        """
        from training.utils.rl.cispo import CISPOConfig
        from training.utils.rl.dapo import DAPOConfig
        from training.utils.rl.dro import DROConfig
        from training.utils.rl.gspo import GSPOConfig
        from training.utils.rl.losses import resolve_builtin_loss

        eps = algorithm_config.eps_clip
        eps_high = algorithm_config.eps_clip_high
        loss_fn_name = algorithm_config.loss_fn or "grpo"

        result = resolve_builtin_loss(
            loss_fn_name,
            profile,
            dapo_config=DAPOConfig(
                eps_clip=eps,
                eps_clip_high=eps_high if eps_high is not None else 0.28,
            ),
            dro_config=DROConfig(),
            gspo_config=GSPOConfig(
                clip_ratio_low=eps,
                clip_ratio_high=eps_high,
            ),
            cispo_config=CISPOConfig(
                eps_low=eps,
                eps_high=eps_high if eps_high is not None else 0.28,
            ),
            eps_clip=eps,
            eps_clip_high=eps_high,
        )
        if result is None:
            from training.utils.rl.losses import SUPPORTED_POLICY_LOSSES

            raise ValueError(f"loss_fn='{loss_fn_name}' has no builtin server-side kernel. Supported: {', '.join(SUPPORTED_POLICY_LOSSES)}")
        self._builtin_loss = result
        logger.info("Resolved builtin loss: kernel=%s, config=%s", result[0], result[1])

    # ------------------------------------------------------------------
    # Forward-backward
    # ------------------------------------------------------------------

    @require_training_client
    async def forward_backward_from_trajectory_groups(
        self,
        trajectory_groups: list[TrajectoryGroup],
        algorithm_config: AlgorithmConfig | None = None,
    ) -> tuple[list[tinker.Datum] | dict[str, list[tinker.Datum]], list[torch.Tensor], dict]:
        """Run forward-backward pass using the builtin server-side loss kernel.

        Args:
            trajectory_groups: List of TrajectoryGroup objects (already filtered/transformed).
            algorithm_config: Algorithm config (uses ``self.algorithm_config`` if None).

        Returns:
            ``(training_datums, training_logprobs, adv_metrics)``
        """
        from training.utils.rl.losses import build_builtin_loss_datums
        from training.utils.rl.tis import TISConfig

        if algorithm_config is None:
            algorithm_config = self.algorithm_config

        raw_datums, adv_metrics = transform_trajectory_groups_to_datums(
            trajectory_groups,
            algorithm_config=algorithm_config,
        )

        adv_metrics["train/num_sequences"] = len(raw_datums)
        adv_metrics["train/active_tokens"] = sum(int(sum(datum.loss_fn_inputs["mask"].data)) for datum in raw_datums)
        adv_metrics.update(self._compute_rollout_entropy_metrics(raw_datums))

        rc = algorithm_config.rollout_correction
        clean_datums, advantages, inf_logprobs, prompt_lens, num_loss_tokens = self._process_datums(raw_datums)

        # seq-mean-token-mean: normalize advantages by number of loss tokens so that
        # token-sum within each sequence equals token-mean, then NUM_SEQUENCES
        # at optim_step gives seq-mean-token-mean overall.
        if algorithm_config.loss_agg_mode == "seq-mean-token-mean":
            for i in range(len(advantages)):
                advantages[i] /= max(1, num_loss_tokens[i])

        # Proximal logprobs
        t0 = time.perf_counter()
        if rc.bypass_mode:
            prox_logprobs = inf_logprobs
        else:
            prox_logprobs = await self._compute_proximal_logprobs(clean_datums)
        adv_metrics.update(
            self._compute_offpolicy_metrics(
                old_logprobs=prox_logprobs,
                rollout_logprobs=inf_logprobs,
                masks=[list(datum.loss_fn_inputs["mask"].data) for datum in raw_datums],
            )
        )
        adv_metrics["time/proximal_forward"] = time.perf_counter() - t0

        # Build datums for the builtin kernel.
        tis_config = TISConfig(level=rc.tis_mode or "token", cap=rc.tis_cap) if rc.tis_mode else None
        builtin_datums = build_builtin_loss_datums(
            clean_datums,
            advantages,
            prox_logprobs,
            inf_logprobs,
            prompt_lens,
            tis_config=tis_config,
            policy_loss=algorithm_config.loss_fn or "grpo",
        )

        kernel_loss, kernel_config = self._builtin_loss
        fwd_bwd_result = await asyncio.to_thread(
            self.training_client.forward_backward,
            builtin_datums,
            kernel_loss,
            loss_fn_config=kernel_config,
        )

        # Merge remote fwd/bwd metrics (e.g. loss) into adv_metrics
        if hasattr(fwd_bwd_result, "metrics") and fwd_bwd_result.metrics:
            for k, v in fwd_bwd_result.metrics.items():
                if k not in self._METRIC_SKIP_KEYS:
                    adv_metrics[f"train/{k}"] = v

        return raw_datums, [], adv_metrics

    # ------------------------------------------------------------------
    # Optimizer step
    # ------------------------------------------------------------------

    @require_training_client
    async def optim_step(
        self,
        step: int,
        total_steps: int,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        grad_clip_norm: float = 1.0,
    ) -> tuple[float, dict]:
        """Run optimizer step. Returns (scheduled_lr, metrics)."""
        scheduled_lr = learning_rate * compute_schedule_lr_multiplier(
            lr_schedule=self.algorithm_config.lr_schedule,
            warmup_steps_ratio=self.algorithm_config.warmup_steps_ratio,
            step=step,
            total_steps=total_steps,
        )

        adam_params = AdamParams(
            learning_rate=scheduled_lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
        )
        from fireworks.training.sdk.client import GradAccNormalization

        _LOSS_AGG_MAP = {
            "token-mean": GradAccNormalization.NUM_LOSS_TOKENS,
            "seq-mean-token-sum": GradAccNormalization.NUM_SEQUENCES,
            "seq-mean-token-mean": GradAccNormalization.NUM_SEQUENCES,
        }
        grad_norm = _LOSS_AGG_MAP.get(self.algorithm_config.loss_agg_mode)
        optim_result = await asyncio.to_thread(
            self.training_client.optim_step,
            adam_params,
            grad_accumulation_normalization=grad_norm,
        )

        metrics = {}
        if hasattr(optim_result, "metrics") and optim_result.metrics:
            for k, v in optim_result.metrics.items():
                if k not in self._METRIC_SKIP_KEYS:
                    metrics[f"train/{k}"] = v

        return scheduled_lr, metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    @require_training_client
    async def sync_weights(self, step: int, checkpoint_type: str | None = None) -> str | None:
        """Hot-load current weights into the inference deployment.

        Returns the snapshot_name on success, None on failure."""
        return await self._sync_weights(f"step-{step}", checkpoint_type=checkpoint_type)

    async def promote_checkpoint(self, snapshot_name: str, output_model_id: str) -> None:
        """Promote a sampler checkpoint to a deployable Fireworks model."""
        if self._rlor_mgr is None or self._policy_job_id is None:
            logger.warning("Cannot promote: rlor_mgr or policy_job_id not set")
            return
        await asyncio.to_thread(
            self._rlor_mgr.promote_checkpoint,
            self._policy_job_id,
            snapshot_name,
            output_model_id,
            self.config.model.name,
        )
        logger.info("Promoted checkpoint '%s' -> model '%s'", snapshot_name, output_model_id)

    @require_training_client
    async def save_dcp_checkpoint(self, step: int) -> None:
        """Save a DCP checkpoint for resume."""
        name = f"step-{step}"
        timeout = self.config.hotload.get("dcp_timeout", 2700)
        try:
            await asyncio.to_thread(self.training_client.save_state, name, timeout=timeout)
            logger.info("DCP checkpoint saved: %s", name)
        except Exception:
            logger.exception("Failed to save DCP checkpoint %s", name)
            raise
