"""
Fireworks backend implementation for the UnifiedTrainer.

Inherits from ``TinkerBackend`` and overrides only what differs:
infrastructure setup (via the cookbook's ``training.provision`` API),
rollout engine (FireworksEngine), and checkpoint lifecycle hooks
(weight syncing via ``WeightSyncer`` instead of Tinker sampler paths).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fireworks.training.sdk import (
    DeploymentSampler,
    TrainerJobManager,
    WeightSyncer,
)
from omegaconf import DictConfig, OmegaConf
from training.provision import FireworksProvisionInfra, init_fireworks_infra
from training.utils import ReconnectableClient, load_deployment_tokenizer

from rllm.engine.rollout import FireworksEngine, RolloutEngine
from rllm.trainer.algorithms import AlgorithmConfig, simple_timer

# fix:fireworks - tinker 0.15.0 sends project_id=None, server rejects it
# from tinker.types import CreateSessionRequest
# _orig_model_dump = CreateSessionRequest.model_dump
# def _patched_model_dump(self, **kwargs):
#     result = _orig_model_dump(self, **kwargs)
#     if result.get("project_id") is None:
#         result.pop("project_id", None)
#     return result
# CreateSessionRequest.model_dump = _patched_model_dump
# fix:fireworks - optim_step sends grad_accumulation_normalization via extra_body, server rejects it
# from fireworks.training.sdk.client import FiretitanTrainingClient
# from tinker.lib.public_interfaces.training_client import TrainingClient
# FiretitanTrainingClient.optim_step = TrainingClient.optim_step
# from training.utils.client import ReconnectableClient as _RC
# _orig_rc_optim_step = _RC.optim_step
# def _patched_rc_optim_step(self, params, grad_accumulation_normalization=None):
#     return self._client.optim_step(params).result(timeout=self._default_timeout)
# _RC.optim_step = _patched_rc_optim_step
from rllm.trainer.fireworks.fireworks_policy_trainer import FireworksPolicyTrainer, builtin_loss_args
from rllm.trainer.tinker.tinker_backend import TinkerBackend
from rllm.trainer.tinker.tinker_metrics_utils import (
    update_training_metrics,
)

if TYPE_CHECKING:
    from rllm.trainer.unified_trainer import TrainerState

logger = logging.getLogger(__name__)
logging.getLogger("fireworks.training.sdk.deployment").setLevel(logging.WARNING)


class FireworksBackend(TinkerBackend):
    """Fireworks backend for the unified trainer.

    Extends ``TinkerBackend`` with Fireworks-specific infrastructure:
        - ``FireworksEngine`` for rollout (via ``DeploymentSampler``)
        - ``FireworksPolicyTrainer`` for gradient updates (via ``ReconnectableClient``)
        - ``WeightSyncer`` for hot-loading checkpoints into an inference deployment

    Inherited unchanged from ``TinkerBackend``:
        - ``get_dataloader``, ``shutdown``
        - ``generate_episodes``, ``transform_to_backend_batch``
        - ``process_backend_batch``, ``compute_advantages``, ``update_policy``
        - ``on_epoch_start/end``, ``on_validation_start/end``
    """

    name: str = "fireworks"

    def __init__(self, config: DictConfig, **kwargs):
        # Intentionally skip TinkerBackend.__init__ to avoid creating a
        # tinker.ServiceClient; we set up Fireworks-specific clients instead.
        from rllm.trainer.backend_protocol import BackendProtocol

        BackendProtocol.__init__(self, config, **kwargs)

        self.full_config = config

        self.policy_trainer: FireworksPolicyTrainer | None = None
        self.tokenizer = None
        self.rollout_engine: FireworksEngine | None = None

        # In TinkerBackend this is a tinker.SamplingClient; here it's a
        # DeploymentSampler, but both get passed to set_sampling_client().
        self.sampling_client: DeploymentSampler | None = None
        self._algorithm_config = None

        self._policy_updated_this_step: bool = False

        self.learning_rate = config.training.get("learning_rate", 1e-6)
        self.beta1 = config.training.get("beta1", 0.9)
        self.beta2 = config.training.get("beta2", 0.95)
        self.eps = config.training.get("eps", 1e-8)
        self.weight_decay = config.training.get("weight_decay", 0.01)
        self.grad_clip_norm = config.training.get("grad_clip_norm", 1.0)

        # Fireworks-specific handles (populated in _init_fireworks_infra)
        self.weight_syncer: WeightSyncer | None = None
        self._policy_rc: ReconnectableClient | None = None
        self._rlor_mgr: TrainerJobManager | None = None
        self._infra: FireworksProvisionInfra | None = None
        self._sample_timeout: int = 600

    # ------------------------------------------------------------------
    # Fireworks infrastructure setup
    # ------------------------------------------------------------------

    def _build_provision_config(self, algorithm_config: AlgorithmConfig):
        """Parse ``fireworks_infra`` (cookbook provision-document format) into
        the flat recipe config ``init_fireworks_infra`` expects.

        The cookbook's ``load_yaml_provision`` resolves trainer/deployment
        references and builds the typed ``rl_loop.Config``; it only accepts a
        file path, so the resolved OmegaConf section round-trips through a
        temp file.
        """
        import tempfile
        from pathlib import Path

        import yaml
        from training.provision import load_yaml_provision

        cfg = self.full_config
        doc = OmegaConf.to_container(cfg.fireworks_infra, resolve=True)
        common = doc.setdefault("common", {})
        # rllm owns these knobs; mirror them into the provision document.
        common["kl_beta"] = float(algorithm_config.kl_beta)
        common["learning_rate"] = self.learning_rate
        if cfg.get("concurrency") is not None:
            common["concurrency"] = OmegaConf.to_container(cfg.concurrency, resolve=True)
        if cfg.training.get("max_length") is not None:
            common["max_seq_len"] = cfg.training.max_length

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
            yaml.safe_dump(doc, fh)
            doc_path = Path(fh.name)
        try:
            _, provision_cfg = load_yaml_provision(mode=None, recipe=None, path=doc_path)
        finally:
            doc_path.unlink(missing_ok=True)
        return provision_cfg

    def _init_fireworks_infra(self, **kwargs) -> None:
        """Provision the trainer job, deployment, and sampler via the
        cookbook's ``training.provision.init_fireworks_infra``."""
        cfg = self.full_config
        # Fail fast on loss misconfiguration before provisioning any
        # (expensive, slow-to-create) remote infrastructure.
        from training.utils.rl.losses import validate_loss_path

        algorithm_config = kwargs.get("algorithm_config") or AlgorithmConfig.from_config(cfg.rllm.algorithm)
        validate_loss_path(builtin_loss_args(algorithm_config))

        provision_cfg = self._build_provision_config(algorithm_config)

        # cleanup_existing=True: rllm names a fresh deployment per run, so
        # shutdown deletes the trainer job and deployment even when an
        # explicit deployment_id is set (parity with the pre-provision path).
        infra = init_fireworks_infra(
            "rl",
            provision_cfg,
            base_url=cfg.get("fireworks_base_url", "https://api.fireworks.ai"),
            cleanup_on_close=True,
            cleanup_existing=True,
            cleanup_deployment_on_close="delete",
        )
        self._infra = infra

        try:
            profile = infra.training_profile
            if profile is not None:
                pp = getattr(profile, "pipeline_parallelism", 1)
                if pp > 1:
                    raise ValueError(f"Pipeline parallelism (PP={pp}) is not supported. Use a training shape with PP=1.")

            if infra.max_seq_len and not cfg.training.get("max_length"):
                cfg.training.max_length = infra.max_seq_len
                logger.info("Auto-derived max_length from training shape: %d", cfg.training.max_length)

            self._rlor_mgr = infra.trainer_manager
            self._policy_job_id = infra.policy_job_id
            self._policy_rc = infra.policy
            self._sample_timeout = provision_cfg.deployment.sample_timeout or 600

            self.sampling_client = infra.sampler
            self.tokenizer = getattr(infra.sampler, "tokenizer", None) or load_deployment_tokenizer(provision_cfg.deployment)
            self.weight_syncer = WeightSyncer(
                policy_client=infra.training_client,
                deploy_mgr=infra.deployment_manager,
                deployment_id=infra.deployment_id,
                base_model=provision_cfg.base_model,
                hotload_timeout=provision_cfg.weight_sync_timeout,
                lora_rank=provision_cfg.lora_rank,
            )
        except BaseException:
            infra.close()
            raise

    # ------------------------------------------------------------------
    # BackendProtocol overrides
    # ------------------------------------------------------------------

    def init_rollout_engine(self, **kwargs) -> RolloutEngine:
        self._init_fireworks_infra(**kwargs)

        # Anything that fails past this point must tear down the provisioned
        # trainer job and deployment, or they keep running (and billing).
        try:
            self.policy_trainer = FireworksPolicyTrainer(
                config=self.full_config,
                training_client=self._policy_rc,
                weight_syncer=self.weight_syncer,
                cf_config=kwargs.get("cf_config"),
                transform_config=kwargs.get("transform_config"),
                algorithm_config=kwargs.get("algorithm_config"),
                rlor_mgr=self._rlor_mgr,
                policy_job_id=self._policy_job_id,
            )

            cfg = self.full_config
            rollout_extra = dict(cfg.get("rollout_engine", {}))
            self.rollout_engine = FireworksEngine(
                tokenizer=self.tokenizer,
                sampler=self.sampling_client,
                max_prompt_length=cfg.data.max_prompt_length,
                max_response_length=cfg.data.max_response_length,
                max_model_length=cfg.training.max_length,
                sampling_params=cfg.rllm.rollout,
                disable_thinking=rollout_extra.pop("disable_thinking", False),
                accumulate_reasoning=rollout_extra.pop("accumulate_reasoning", False),
                reasoning_effort=rollout_extra.pop("reasoning_effort", "medium"),
                sample_timeout=self._sample_timeout,
                router_replay=cfg.rllm.algorithm.get("router_replay", "disabled") == "R3",
                **rollout_extra,
            )
            return self.rollout_engine
        except BaseException:
            self.shutdown()
            raise

    def validate_config(self) -> None:
        if self.full_config.get("fuse_forward_backward_and_optim_step", False):
            raise ValueError("fuse_forward_backward_and_optim_step is not supported by the Fireworks backend. Set fuse_forward_backward_and_optim_step: false in your config.")

        rollout_cfg = self.full_config.rllm.rollout
        for split in ("train", "val"):
            sp = rollout_cfg.get(split, {}) or {}
            if sp.get("temperature", 1.0) != 1.0 or sp.get("top_p", 1.0) != 1.0:
                logger.warning(
                    "rllm.rollout.%s.{temperature,top_p} are set away from 1.0; this can cause issues with logprobs accuracy.",
                    split,
                )

        # --- Algorithm / loss function validation ---
        # Loss function validation is handled by resolve_builtin_loss() at setup time.
        alg = self.full_config.rllm.algorithm
        loss_fn = alg.get("loss_fn", None)
        loss_agg_mode = alg.get("loss_agg_mode", None)
        eps_clip_high = alg.get("eps_clip_high", None)
        router_replay = alg.get("router_replay", "disabled")
        rc = alg.get("rollout_correction", {})
        tis_mode = rc.get("tis_mode", None)
        bypass_mode = rc.get("bypass_mode", True)

        # eps_clip_high only meaningful for dapo/cispo (asymmetric clipping)
        if eps_clip_high is not None and loss_fn not in ("dapo", "cispo"):
            logger.warning(
                "eps_clip_high is set but loss_fn='%s' does not use asymmetric clipping. eps_clip_high is only used by 'dapo' and 'cispo'.",
                loss_fn,
            )

        valid_loss_agg_modes = {None, "token-mean", "seq-mean-token-sum", "seq-mean-token-mean"}
        if loss_agg_mode not in valid_loss_agg_modes:
            raise ValueError(f"rllm.algorithm.loss_agg_mode must be null, 'token-mean', 'seq-mean-token-sum', or 'seq-mean-token-mean' for the Fireworks backend, got {loss_agg_mode!r}")
        logger.info("Fireworks loss aggregation mode: %s", loss_agg_mode or "backend default")

        if router_replay == "R2":
            raise ValueError("rllm.algorithm.router_replay='R2' is not supported by the Fireworks backend; use 'R3' or 'disabled'.")

        # rollout_correction.tis_mode validation
        if tis_mode is not None and tis_mode not in ("token", "sequence"):
            raise ValueError(f"rollout_correction.tis_mode must be null, 'token', or 'sequence', got '{tis_mode}'")

        # TIS with bypass is a no-op (prox = inf, weight = 1.0)
        if tis_mode is not None and bypass_mode:
            logger.warning(
                "rollout_correction.tis_mode='%s' with bypass_mode=true; TIS weight will be 1.0 (no correction). Set bypass_mode=false for active TIS.",
                tis_mode,
            )

        # save_freq must be a multiple of sync interval (save requires a sampler snapshot from sync)
        save_freq = self.full_config.rllm.trainer.get("save_freq", -1)
        if save_freq > 0:
            async_cfg = self.full_config.rllm.get("async_training", {})
            if async_cfg.get("enable", False):
                sync_interval = async_cfg.get("trigger_parameter_sync_step", 1)
                if sync_interval > 0 and save_freq % sync_interval != 0:
                    raise ValueError(f"save_freq ({save_freq}) must be a multiple of trigger_parameter_sync_step ({sync_interval}). Promotion requires a sampler snapshot created at sync time.")

    # ------------------------------------------------------------------
    # Policy update (override: no fused path, uses ReconnectableClient)
    # ------------------------------------------------------------------

    async def process_backend_batch(
        self,
        trainer_state: TrainerState,
        **kwargs,
    ) -> None:
        await super().process_backend_batch(trainer_state, **kwargs)
        trainer_state.extra_info.pop("training_logprobs", None)

    async def update_policy(self, trainer_state: TrainerState, **kwargs) -> None:
        assert self.policy_trainer is not None, "policy_trainer is not initialized"

        with simple_timer("optim_step", trainer_state.timing_dict):
            scheduled_lr, optim_metrics = await self.policy_trainer.optim_step(
                step=trainer_state.global_step,
                total_steps=trainer_state.total_steps,
                learning_rate=self.learning_rate,
                beta1=self.beta1,
                beta2=self.beta2,
                eps=self.eps,
                weight_decay=self.weight_decay,
                grad_clip_norm=self.grad_clip_norm,
            )
            trainer_state.extra_info["scheduled_learning_rate"] = scheduled_lr
            trainer_state.metrics.update(optim_metrics)

    # ------------------------------------------------------------------
    # Train lifecycle hooks (overrides)
    # ------------------------------------------------------------------

    async def on_train_start(self, trainer_state: TrainerState) -> None:
        assert self.policy_trainer is not None, "policy_trainer is not initialized"

        start_step = await self.policy_trainer.initialize_async(
            resume_from_checkpoint=True,
        )
        trainer_state.global_step = start_step

    async def _save_and_sync(
        self,
        trainer_state: TrainerState,
        should_save: bool = False,
        should_sync: bool = False,
    ) -> None:
        """Unified save + sync + promote logic.

        Args:
            trainer_state: Current trainer state.
            should_save: Save a DCP checkpoint and promote if sync also happens.
            should_sync: Sync (hotload) weights to the inference deployment.
        """
        global_step = trainer_state.global_step

        if should_save and not should_sync:
            logger.warning(
                "save_freq triggered at step %d but no sync; skipping save/promote (save_freq must be a multiple of sync interval)",
                global_step,
            )

        if should_sync:
            checkpoint_type = "base" if should_save else None
            snapshot_name = await self.policy_trainer.sync_weights(
                global_step,
                checkpoint_type=checkpoint_type,
            )

            if should_save:
                with simple_timer("save_checkpoint", trainer_state.timing_dict):
                    await self.policy_trainer.save_dcp_checkpoint(global_step)
                if snapshot_name:
                    experiment = self.full_config.rllm.trainer.get("experiment_name", "default")
                    output_model_id = f"{experiment}-step-{global_step}"
                    try:
                        await self.policy_trainer.promote_checkpoint(
                            snapshot_name,
                            output_model_id,
                        )
                    except Exception as exc:
                        logger.exception(
                            "Checkpoint promotion failed for '%s' -> '%s'; continuing because the DCP checkpoint was saved. Error: %s",
                            snapshot_name,
                            output_model_id,
                            exc,
                        )

    async def on_train_end(self, trainer_state: TrainerState) -> None:
        assert self.policy_trainer is not None, "policy_trainer is not initialized"
        logger.info("Saving final checkpoint at step %d", trainer_state.global_step)
        await self._save_and_sync(trainer_state, should_save=True, should_sync=True)

    async def on_policy_updated(self, trainer_state: TrainerState) -> None:
        """Called in async mode after optimizer step when coordinator triggers sync."""
        assert self.policy_trainer is not None
        self._policy_updated_this_step = True
        save_freq = self.full_config.rllm.trainer.save_freq
        step = trainer_state.global_step
        await self._save_and_sync(
            trainer_state,
            should_save=save_freq > 0 and step % save_freq == 0,
            should_sync=True,
        )

    async def on_batch_end(self, trainer_state: TrainerState) -> None:
        assert self.policy_trainer is not None, "policy_trainer is not initialized"

        # In async mode, on_policy_updated already handled save/sync
        if not self._policy_updated_this_step:
            step = trainer_state.global_step
            save_freq = self.full_config.rllm.trainer.save_freq
            await self._save_and_sync(
                trainer_state,
                should_save=save_freq > 0 and step % save_freq == 0,
                should_sync=True,
            )
        self._policy_updated_this_step = False

        learning_rate = trainer_state.extra_info.get("scheduled_learning_rate", self.learning_rate)
        update_training_metrics(trainer_state, learning_rate, trainer_state.total_steps)
        if trainer_state.backend_batch:
            trainer_state.metrics.update(self.policy_trainer._compute_rollout_entropy_metrics(trainer_state.backend_batch))

    def shutdown(self) -> None:
        """Tear down provisioned Fireworks resources."""
        if self._infra is not None:
            self._infra.close()
