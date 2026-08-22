"""Miles backend for the UnifiedTrainer.

rLLM drives the loop; Miles owns the GPUs. Miles' own ``train.py`` is never used --
this reproduces its spine (generate, train, update_weights) with rLLM's workflow
engine standing in for the generate half. See ``design/miles-training-backend.md``.

Miles' ``RolloutManager`` is still created, because it owns the SGLang engine fleet,
the router, and the weight-update broker that ``RayTrainGroup.update_weights`` goes
through. Its rollout function is pinned to ``sleep_rollout`` and never invoked.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig

from rllm.data.utils import interleave_tasks
from rllm.engine.rollout import RolloutEngine
from rllm.trainer.algorithms.advantage import AlgorithmConfig, collect_reward_and_advantage_from_trajectory_groups
from rllm.trainer.algorithms.performance import simple_timer
from rllm.trainer.backend_protocol import BackendProtocol
from rllm.trainer.miles.miles_config import build_miles_args, validate_pinned
from rllm.trainer.miles.transform import payloads_to_samples, trajectory_groups_to_payloads
from rllm.types import Episode

if TYPE_CHECKING:
    from rllm.engine.unified_workflow_engine import UnifiedWorkflowEngine
    from rllm.trainer.unified_trainer import TrainerState

logger = logging.getLogger(__name__)


@dataclass
class MilesBatch:
    """What ``RayTrainGroup.train`` consumes, plus what we need for metrics.

    ``data_ref`` is the object-store handle Miles' train workers pull from;
    ``sample_indices`` mirrors what its own rollout path returns alongside it.
    """

    data_ref: Any
    sample_indices: Any = None
    num_samples: int = 0
    metrics: dict = field(default_factory=dict)

    def __len__(self) -> int:
        """Trainable sequences in this batch.

        The async training loop sums ``len(backend_batch)`` across forward-backward
        passes and skips the optimizer step when the total is zero.
        """
        return self.num_samples

    @property
    def is_empty(self) -> bool:
        return self.num_samples == 0


class MilesBackend(BackendProtocol[Iterable, MilesBatch]):
    """Backend that trains through Miles' Ray actors."""

    name: str = "miles"

    def __init__(self, config: DictConfig, **kwargs):
        BackendProtocol.__init__(self, config, **kwargs)
        self.full_config = config
        self.miles_args = None
        self.rollout_manager = None
        self.actor_model = None
        self.rollout_engine = None
        self.tokenizer = None
        self.algorithm_config: AlgorithmConfig | None = None
        # Async training moves weight sync from on_batch_end to on_policy_updated,
        # which the trainer calls inside its own pause/drain window.
        self._engine_closed = False
        self.is_async = bool((config.rllm.get("async_training", {}) or {}).get("enable", False))
        # Miles indexes everything by rollout_id; rLLM's global_step is the same clock.
        self._num_rollout_per_epoch = None
        # Advantages are computed during transform (stage 4), but the trainer's
        # compute_advantages hook runs at stage 6; carry the metrics across.
        self._pending_adv_metrics: dict | None = None

    # =====================================================================
    # setup
    # =====================================================================

    def validate_config(self) -> None:
        """Fail before any GPU is claimed."""
        try:
            import miles  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"The miles backend needs the `miles` package ({e}). Install it from source:\n"
                "  git clone https://github.com/radixark/miles && cd miles\n"
                "  pip install -r requirements.txt && pip install -e . --no-deps\n"
                "See design/miles-training-backend.md for the full recipe."
            ) from e

        from omegaconf import OmegaConf

        node = self.full_config.get("miles", None)
        raw = OmegaConf.to_container(node, resolve=True) if node is not None else {}
        raw = raw if isinstance(raw, dict) else {}
        validate_pinned(raw)

        async_cfg = self.full_config.rllm.get("async_training", {}) or {}
        if async_cfg.get("enable", False):
            # The trainer's async loop accumulates gradient across `num_fwd_bwd_passes`
            # calls to process_backend_batch, then takes one optimizer step in
            # update_policy. Miles' RayTrainGroup.train() is monolithic -- log-probs, loss
            # and the optimizer steps happen in one call -- so it cannot accumulate across
            # passes. Constrain the loop to one pass per step, the same restriction verl's
            # separated mode carries.
            mini_batch = async_cfg.get("mini_batch_size", 1)
            fwd_bwd = async_cfg.get("fwd_bwd_group_size") or mini_batch
            if fwd_bwd != mini_batch:
                raise ValueError(
                    f"The miles backend needs rllm.async_training.fwd_bwd_group_size ({fwd_bwd}) "
                    f"== mini_batch_size ({mini_batch}): Miles takes its optimizer step inside "
                    "RayTrainGroup.train(), so it cannot accumulate gradient across passes."
                )
            # The trainer asserts this too, but only once _fit_fully_async starts --
            # after Ray, the SGLang engines and the model are all up, which is minutes of
            # setup to reach a config error. validate_config runs before any of that.
            if self.full_config.rllm.get("workflow", {}).get("raise_on_error", True):
                raise ValueError("Async training needs rllm.workflow.raise_on_error=false so a failed task still yields an (error) episode instead of killing the generation loop.")
            if raw.get("pause_generation_mode") == "abort":
                raise ValueError(
                    "miles.pause_generation_mode=abort kills in-flight generation on every "
                    "weight update, which async training does continuously. Use 'retract' "
                    "(Miles' default): it requeues those requests and resumes them after."
                )

    def init_rollout_engine(self, **kwargs) -> RolloutEngine:
        """Build Miles' args, bring up its Ray actors, and return the rollout engine.

        Called from ``UnifiedTrainer.__init__`` after the dataloaders exist, which is
        the first point ``total_training_steps`` is known -- and Miles needs it as
        ``--num-rollout``, because ``--num-epoch`` asserts the global dataset we disable.
        """
        from miles.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
        from miles.utils import object_store
        from miles.utils.async_utils import run as miles_run
        from miles.utils.tracking_utils.tracking import init_tracking

        from rllm.trainer.miles.patch import apply_all_miles_patches

        # Worker processes get these through runtime_env.worker_process_setup_hook
        # (see MilesTrainerLauncher); the driver needs them applied here.
        apply_all_miles_patches()

        self.algorithm_config = kwargs.get("algorithm_config")
        total_steps = kwargs.get("total_training_steps")

        self.miles_args = build_miles_args(self.full_config, total_steps=total_steps)
        logger.info(
            "Miles args: train_backend=%s num_rollout=%s compute_advantages=%s global_dataset=%s",
            self.miles_args.train_backend,
            self.miles_args.num_rollout,
            self.miles_args.compute_advantages_and_returns,
            self.miles_args.rollout_global_dataset,
        )

        pgs = create_placement_groups(self.miles_args)
        object_store.init_instance(self.miles_args, contribute_segment=False)
        init_tracking(self.miles_args)

        self.rollout_manager, self._num_rollout_per_epoch = create_rollout_manager(self.miles_args, pgs["rollout"])
        # create_training_models is async and we are in sync __init__; miles' own
        # helper runs it on a background loop, which is safe either way.
        self.actor_model, critic_model = miles_run(create_training_models(self.miles_args, pgs, self.rollout_manager))
        if critic_model is not None:
            raise ValueError("The miles backend does not support a critic yet (advantage_estimator=ppo).")

        # Publish the freshly loaded training weights before the first rollout, so
        # generation starts on-policy. Mirrors miles' train.py.
        miles_run(self.actor_model.update_weights())

        # Diagnostic, mirroring miles' train.py: assert the engines actually hold the
        # trainer's weights. A silent divergence here shows up much later as a large
        # train/inference logprob gap, which wrecks any importance correction (TIS)
        # without ever failing a run. create_rollout_manager already took the snapshot
        # and reset the engine tensors when this flag is set.
        if getattr(self.miles_args, "check_weight_update_equal", False):
            import ray

            logger.info("Comparing engine weights against the trainer (--check-weight-update-equal)")
            result = ray.get(
                self.rollout_manager.check_weights.remote(
                    action="compare",
                    allow_quant_error=self.miles_args.check_weight_update_allow_quant_error,
                    selector=self.miles_args.check_weight_update_selector,
                    skip_list=self.miles_args.check_weight_update_skip_list,
                )
            )
            logger.info("Weight comparison result: %s", result)

        self.tokenizer = self._load_tokenizer()
        self.rollout_engine = self._build_engine()
        return self.rollout_engine

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.miles_args.hf_checkpoint, trust_remote_code=True)

    def _build_engine(self) -> RolloutEngine:
        import ray

        from rllm.engine.rollout.miles_engine import MilesEngine

        host, port = ray.get(self.rollout_manager.get_router_address.remote())
        router_url = f"http://{host}:{port}"
        logger.info("Miles SGLang router at %s", router_url)
        return MilesEngine(config=self.full_config, router_url=router_url, tokenizer=self.tokenizer, miles_args=self.miles_args)

    async def on_train_end(self, trainer_state: TrainerState) -> None:
        """Close the engine's HTTP client on the loop that opened it.

        httpx binds its connections to the running loop, so closing them anywhere else
        (Miles' background loop, or after ``fit()``'s loop has been torn down) raises
        "Event loop is closed". This hook is awaited on the trainer's own loop, which is
        the only safe place.
        """
        await self._close_engine()

    async def _close_engine(self) -> None:
        engine = self.rollout_engine
        if engine is None or self._engine_closed or not hasattr(engine, "close"):
            return
        self._engine_closed = True
        try:
            await engine.close()
        except Exception:
            logger.warning("MilesEngine HTTP client did not close cleanly", exc_info=True)

    def shutdown(self) -> None:
        # on_train_end already closed the engine on the right loop. Reaching here with it
        # still open means fit() never ran (a bring-up failure), and there is no live loop
        # to close it on -- the process is exiting anyway, so leave it.
        if self.rollout_engine is not None and not self._engine_closed:
            logger.debug("MilesEngine HTTP client left open; no event loop available at shutdown.")
        try:
            if self.rollout_manager is not None:
                import ray

                ray.get(self.rollout_manager.dispose.remote())
        except Exception:
            logger.exception("Miles rollout manager did not dispose cleanly")
        try:
            from miles.utils.tracking_utils.tracking import finish_tracking

            finish_tracking()
        except Exception:
            logger.exception("Miles tracking did not finish cleanly")

    # =====================================================================
    # training loop
    # =====================================================================

    async def generate_episodes(self, batch: Any, agent_workflow_engine: UnifiedWorkflowEngine, is_validation: bool = False, **kwargs) -> list[Episode]:
        """rLLM generates; Miles is not involved beyond serving the engines."""
        repeat_times = self.full_config.rllm.rollout.n_val if is_validation else self.full_config.rllm.rollout.n
        tasks, task_ids = interleave_tasks(batch, repeat_times)
        episodes = await agent_workflow_engine.execute_tasks(tasks, task_ids, is_validation=is_validation, **kwargs)
        for episode, task in zip(episodes, tasks, strict=True):
            data_source = task.get("data_source") if isinstance(task, dict) else None
            if data_source is not None:
                episode.info["data_source"] = data_source
        return episodes

    def transform_to_backend_batch(self, trainer_state: TrainerState, **kwargs) -> MilesBatch:
        """Trajectory groups -> Miles Samples -> the train-data pack its workers pull."""
        from miles.ray.rollout.train_data_conversion import ROLLOUT_DATA_VALUE_SPEC, convert_samples_to_train_data, split_train_data_by_dp
        from miles.utils import object_store

        groups = trainer_state.trajectory_groups
        assert groups is not None, "trajectory_groups must be set before transform_to_backend_batch"

        # The trainer calls this at stage 4 but compute_advantages only at stage 6, so
        # the per-token advantages the transform needs do not exist yet. Compute them
        # here when absent -- the same thing the tinker transform does -- and hand the
        # metrics to compute_advantages rather than computing twice.
        needs_advantages = any(step.advantage is None for group in groups for traj in group.trajectories for step in traj.steps)
        if needs_advantages:
            assert self.algorithm_config is not None, "algorithm_config was not provided to init_rollout_engine"
            self._pending_adv_metrics = collect_reward_and_advantage_from_trajectory_groups(groups, self.algorithm_config)

        grouped = trajectory_groups_to_payloads(groups)
        samples, advantages = payloads_to_samples(grouped)
        if not samples:
            # Every group was dropped (e.g. all responses empty). The async loop treats a
            # zero-length batch as "this pass contributes no gradient" and carries on, so
            # returning empty is more useful than raising -- update_policy skips it.
            logger.warning("No trainable samples in this batch; %d trajectory groups produced nothing.", len(groups))
            return MilesBatch(data_ref=None, num_samples=0, metrics={"batch/miles_samples": 0, "batch/miles_groups": 0})

        samples, advantages, dynamic_gbs = self._fit_to_dp(samples, advantages)

        train_data = convert_samples_to_train_data(
            self.miles_args,
            samples,
            metadata={"dynamic_global_batch_size": dynamic_gbs},
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )
        # rLLM owns advantage computation, so they ride along as a top-level key.
        # Miles CP-slices it beside rollout_log_probs; see rllm/trainer/miles/patch.py.
        train_data["advantages"] = advantages

        sample_indices = train_data.get("sample_indices")
        if self.miles_args.delay_split_train_data_by_dp:
            data_ref = object_store.get_instance().put(value=train_data, value_spec=ROLLOUT_DATA_VALUE_SPEC)
        else:
            data_ref = split_train_data_by_dp(self.miles_args, train_data, self._train_parallel_config())

        return MilesBatch(
            data_ref=data_ref,
            sample_indices=sample_indices,
            num_samples=len(samples),
            metrics={"batch/miles_samples": len(samples), "batch/miles_groups": len(grouped)},
        )

    def _fit_to_dp(self, samples: list, advantages: list[list[float]]) -> tuple[list, list[list[float]], int]:
        """Trim the batch to a multiple of the DP size and return that as the batch size.

        Miles computes ``num_steps_per_rollout = num_local_samples // num_local_gbs`` with
        integer division, so a static ``--global-batch-size`` would either drop the
        remainder silently or take several optimizer steps inside one rLLM step. Reusing
        Miles' own ``_compute_dynamic_global_batch_size`` (rounded down to ``dp_size``)
        pins it to exactly one step over exactly the samples we keep, whatever survived
        rejection sampling and compact filtering.
        """
        from miles.ray.rollout.rollout_data_conversion import _compute_dynamic_global_batch_size

        parallel_config = self._train_parallel_config()
        dynamic_gbs = _compute_dynamic_global_batch_size(self.miles_args, parallel_config, len(samples))

        if dynamic_gbs > len(samples):
            # Fewer samples than dp_size: Miles pads the batch size up and would index
            # past the end. Nothing trainable here.
            raise ValueError(
                f"Only {len(samples)} trainable samples for dp_size={parallel_config['dp_size']}; "
                "raise rllm.async_training.mini_batch_size (or rllm.data.train_batch_size) "
                "or rllm.rollout.n so every DP rank gets at least one sample."
            )
        if dynamic_gbs < len(samples):
            logger.info(
                "Trimming batch from %d to %d samples (a multiple of dp_size=%d) so Miles takes exactly one optimizer step.",
                len(samples),
                dynamic_gbs,
                parallel_config["dp_size"],
            )
            samples, advantages = samples[:dynamic_gbs], advantages[:dynamic_gbs]
        return samples, advantages, dynamic_gbs

    def _train_parallel_config(self):
        """Read the DP layout the train actor pushed into the RolloutManager.

        ``train_actor.set_rollout_manager`` calls
        ``rollout_manager.set_train_parallel_config(...)`` during init, so the value
        exists by the time ``create_training_models`` returns -- but RolloutManager
        exposes only the setter. It is derived from live torch.distributed state
        (``get_parallel_state().intra_dp.size``), so the driver cannot recompute it;
        ``__ray_call__`` reads the attribute off the actor instead of patching Miles.
        """
        import ray

        return ray.get(self.rollout_manager.__ray_call__.remote(lambda mgr: mgr.train_parallel_config))

    async def process_backend_batch(self, trainer_state: TrainerState, **kwargs) -> None:
        """No-op: Miles' train actor runs its own log-prob and reference forward passes."""
        batch = trainer_state.backend_batch
        if isinstance(batch, MilesBatch) and batch.metrics:
            trainer_state.metrics.update(batch.metrics)

    async def compute_advantages(self, trainer_state: TrainerState, algorithm_config: AlgorithmConfig, **kwargs) -> None:
        """Publish the advantage metrics. Miles' own estimator is off via
        ``--disable-compute-advantages-and-returns``.

        The advantages themselves were computed in transform_to_backend_batch, which
        the trainer runs earlier; this only reports. Recomputing here would be
        wasteful and would double-count the reward metrics.
        """
        assert trainer_state.trajectory_groups is not None, "Trajectory groups are not set"
        metrics = self._pending_adv_metrics
        self._pending_adv_metrics = None
        if metrics is None:
            metrics = collect_reward_and_advantage_from_trajectory_groups(trainer_state.trajectory_groups, algorithm_config)
        trainer_state.metrics.update(metrics)

    async def update_policy(self, trainer_state: TrainerState, **kwargs) -> None:
        batch: MilesBatch = trainer_state.backend_batch  # type: ignore[assignment]
        if batch is None or batch.is_empty:
            logger.warning("Skipping the optimizer step: no trainable sequences in this batch.")
            return
        rollout_id = trainer_state.global_step
        with simple_timer("update_actor", trainer_state.timing_dict):
            await self.actor_model.train(rollout_id, {"data_ref": batch.data_ref, "sample_indices": batch.sample_indices})

        from miles.utils.data import remove_rollout_data_refs

        remove_rollout_data_refs(self.miles_args, {"data_ref": batch.data_ref, "sample_indices": batch.sample_indices})

    # =====================================================================
    # hooks
    # =====================================================================

    async def on_batch_end(self, trainer_state: TrainerState) -> None:
        rollout_id = trainer_state.global_step
        save_interval = self.miles_args.save_interval
        if save_interval and rollout_id % save_interval == 0:
            with simple_timer("save_checkpoint", trainer_state.timing_dict):
                await self.actor_model.save_model(rollout_id)
                await self.rollout_manager.save.remote(rollout_id)

        # Sync mode publishes weights at the end of the step; async mode syncs in
        # on_policy_updated instead, inside the trainer's pause/drain window. Doing both
        # would push the same weights twice per step.
        if not self.is_async:
            with simple_timer("update_weights", trainer_state.timing_dict):
                await self.actor_model.update_weights(rollout_id=rollout_id)

    async def on_policy_updated(self, trainer_state: TrainerState) -> None:
        """Publish new weights to the SGLang engines (async path).

        Called from the trainer's ``_perform_weight_sync``. With ``partial_rollout=True``
        the trainer does *not* drain generation first, which is safe because Miles' own
        update pauses every engine, flushes, loads and resumes
        (``update_weight_utils.update_weights``). Under the default
        ``--pause-generation-mode retract`` in-flight requests are requeued rather than
        killed, so a rollout crossing a sync boundary survives it.
        """
        if not self.is_async:
            return
        with simple_timer("weight_sync", trainer_state.timing_dict):
            await self.actor_model.update_weights(rollout_id=trainer_state.global_step)

    async def on_validation_start(self, trainer_state: TrainerState) -> bool:
        trainer_state.is_training = False
        if self.rollout_engine is not None:
            self.rollout_engine.is_validation = True
        return True

    async def on_validation_end(self, trainer_state: TrainerState) -> None:
        trainer_state.is_training = True
        if self.rollout_engine is not None:
            self.rollout_engine.is_validation = False
