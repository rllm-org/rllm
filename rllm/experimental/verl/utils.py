"""Helpers for the VerlBackend that previously came from RayPPOTrainer."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from verl import DataProto
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance

logger = logging.getLogger(__name__)


def _explicit_override_keys(hydra_overrides: list[str] | None = None) -> set[str]:
    """Return the set of dotted paths the user explicitly set on the Hydra CLI.

    If ``hydra_overrides`` is provided (typically captured in the main Hydra-
    decorated process and forwarded into a Ray actor), use it. Otherwise fall
    back to ``HydraConfig.get().overrides.task``, which only works when running
    inside the Hydra entry-point process.
    """
    if hydra_overrides is None:
        try:
            from hydra.core.hydra_config import HydraConfig

            hydra_overrides = list(HydraConfig.get().overrides.task)
        except (ValueError, AttributeError, ImportError):
            return set()
    keys: set[str] = set()
    for o in hydra_overrides:
        if "=" not in o:
            continue
        key = o.split("=", 1)[0].lstrip("+~")
        keys.add(key)
    return keys


# (verl_native_path, rllm_path) — value is the same on both sides; only the
# location differs. Used by ``sync_config`` to keep both
# namespaces in sync regardless of which side the user typed on the CLI.
_SHARED_KEYS: list[tuple[str, str]] = [
    ("algorithm.adv_estimator", "rllm.algorithm.adv_estimator"),
    ("algorithm.norm_adv_by_std_in_grpo", "rllm.algorithm.norm_adv_by_std_in_grpo"),
    ("algorithm.rollout_correction.bypass_mode", "rllm.algorithm.rollout_correction.bypass_mode"),
    ("algorithm.rollout_correction.rollout_is", "rllm.algorithm.rollout_correction.tis_mode"),
    ("algorithm.rollout_correction.rollout_is_threshold", "rllm.algorithm.rollout_correction.tis_cap"),
    ("actor_rollout_ref.actor.kl_loss_coef", "rllm.algorithm.kl_beta"),
    ("actor_rollout_ref.actor.policy_loss.loss_mode", "rllm.algorithm.loss_fn"),
    ("actor_rollout_ref.actor.loss_agg_mode", "rllm.algorithm.loss_agg_mode"),
    ("actor_rollout_ref.actor.clip_ratio_high", "rllm.algorithm.eps_clip_high"),
    ("actor_rollout_ref.actor.router_replay.mode", "rllm.algorithm.router_replay"),
    ("actor_rollout_ref.rollout.n", "rllm.rollout.n"),
    ("actor_rollout_ref.rollout.val_kwargs.n", "rllm.rollout.n_val"),
    ("trainer.save_freq", "rllm.trainer.save_freq"),
    ("trainer.test_freq", "rllm.trainer.test_freq"),
    ("trainer.val_before_train", "rllm.trainer.val_before_train"),
    ("trainer.val_only", "rllm.trainer.val_only"),
    ("trainer.total_epochs", "rllm.trainer.total_epochs"),
    ("trainer.logger", "rllm.trainer.logger"),
    ("trainer.project_name", "rllm.trainer.project_name"),
    ("trainer.experiment_name", "rllm.trainer.experiment_name"),
]

_TOTAL_TRAINING_STEPS_KEY: tuple[str, str] = ("trainer.total_training_steps", "rllm.trainer.total_batches")


def sync_config(config: DictConfig, hydra_overrides: list[str] | None = None) -> None:
    """Keep verl-native and rllm-namespaced config in sync.

    Precedence per shared key: rllm CLI explicit > verl CLI explicit > rllm
    yaml default > verl yaml default. ``None`` rllm values are treated as
    "no rllm default", letting verl's yaml default stand. Verl-native shared
    CLI overrides still work for backward compatibility, but warn because
    shared keys should be set through their rllm.* paths going forward.

    ``hydra_overrides`` is the list of CLI overrides captured in the main
    Hydra-decorated process (``HydraConfig.get().overrides.task``). It must be
    passed in when this function runs inside a Ray actor (Hydra context isn't
    available across processes). Outside Ray, it can be omitted.

    Also derives ``actor.use_kl_loss = (kl_beta > 0)`` from rllm values.
    """
    explicit = _explicit_override_keys(hydra_overrides)

    def warn_verl_override(verl_path: str, rllm_path: str, *, conflict: bool = False) -> None:
        if conflict:
            logger.warning(
                "Verl-native shared config %s conflicts with %s; using %s. Setting shared rLLM/verl knobs through Verl-native paths is deprecated; use %s=... instead.",
                verl_path,
                rllm_path,
                rllm_path,
                rllm_path,
            )
            return
        logger.warning(
            "Verl-native shared config %s is deprecated and will be removed in a future release; use %s=... instead.",
            verl_path,
            rllm_path,
        )

    def maybe_warn_verl_override(verl_path: str, rllm_path: str, *, conflict: bool = False) -> None:
        if verl_path in explicit:
            warn_verl_override(verl_path, rllm_path, conflict=conflict)

    def plain(value: Any) -> Any:
        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=False)
        return value

    def same(left: Any, right: Any) -> bool:
        return plain(left) == plain(right)

    def sync_pair(verl_path: str, rllm_path: str) -> None:
        if rllm_path in explicit:
            rllm_value = OmegaConf.select(config, rllm_path)
            verl_value = OmegaConf.select(config, verl_path)
            maybe_warn_verl_override(verl_path, rllm_path, conflict=not same(verl_value, rllm_value))
            OmegaConf.update(config, verl_path, rllm_value, merge=False)
        elif verl_path in explicit:
            maybe_warn_verl_override(verl_path, rllm_path)
            OmegaConf.update(config, rllm_path, OmegaConf.select(config, verl_path), merge=False)
        else:
            value = OmegaConf.select(config, rllm_path)
            if value is None:
                return
            OmegaConf.update(config, verl_path, value, merge=False)

    def sync_total_training_steps() -> None:
        verl_path, rllm_path = _TOTAL_TRAINING_STEPS_KEY

        def to_verl(value: int | None) -> int | None:
            return None if value is None or value <= 0 else value

        def to_rllm(value: int | None) -> int:
            return -1 if value is None else value

        if rllm_path in explicit:
            rllm_value = OmegaConf.select(config, rllm_path)
            verl_value = OmegaConf.select(config, verl_path)
            rllm_as_verl = to_verl(rllm_value)
            maybe_warn_verl_override(verl_path, rllm_path, conflict=not same(verl_value, rllm_as_verl))
            OmegaConf.update(config, verl_path, rllm_as_verl, merge=False)
        elif verl_path in explicit:
            maybe_warn_verl_override(verl_path, rllm_path)
            OmegaConf.update(config, rllm_path, to_rllm(OmegaConf.select(config, verl_path)), merge=False)
        else:
            OmegaConf.update(config, verl_path, to_verl(OmegaConf.select(config, rllm_path)), merge=False)

    def sync_clip_ratio() -> None:
        eps_clip_path = "rllm.algorithm.eps_clip"
        clip_ratio_path = "actor_rollout_ref.actor.clip_ratio"
        clip_ratio_low_path = "actor_rollout_ref.actor.clip_ratio_low"

        if eps_clip_path in explicit:
            eps_clip = OmegaConf.select(config, eps_clip_path)
            for verl_path in (clip_ratio_low_path, clip_ratio_path):
                maybe_warn_verl_override(
                    verl_path,
                    eps_clip_path,
                    conflict=not same(OmegaConf.select(config, verl_path), eps_clip),
                )
            OmegaConf.update(config, clip_ratio_path, eps_clip, merge=False)
            OmegaConf.update(config, clip_ratio_low_path, eps_clip, merge=False)
        elif clip_ratio_low_path in explicit:
            maybe_warn_verl_override(clip_ratio_low_path, eps_clip_path)
            if clip_ratio_path in explicit:
                maybe_warn_verl_override(clip_ratio_path, eps_clip_path)
            OmegaConf.update(config, eps_clip_path, OmegaConf.select(config, clip_ratio_low_path), merge=False)
        elif clip_ratio_path in explicit:
            maybe_warn_verl_override(clip_ratio_path, eps_clip_path)
            OmegaConf.update(config, eps_clip_path, OmegaConf.select(config, clip_ratio_path), merge=False)
        else:
            eps_clip = OmegaConf.select(config, eps_clip_path)
            if eps_clip is not None:
                OmegaConf.update(config, clip_ratio_path, eps_clip, merge=False)
                OmegaConf.update(config, clip_ratio_low_path, eps_clip, merge=False)

        eps_clip = OmegaConf.select(config, eps_clip_path)
        eps_clip_high = OmegaConf.select(config, "rllm.algorithm.eps_clip_high")
        if eps_clip_high is None:
            eps_clip_high = eps_clip
        if eps_clip_high is not None:
            OmegaConf.update(config, "actor_rollout_ref.actor.clip_ratio_high", eps_clip_high, merge=False)

    for verl_path, rllm_path in _SHARED_KEYS:
        sync_pair(verl_path, rllm_path)

    sync_total_training_steps()

    # Derived verl-only keys
    if "actor_rollout_ref.actor.use_kl_loss" not in explicit:
        kl_beta = OmegaConf.select(config, "rllm.algorithm.kl_beta")
        if kl_beta is None:
            kl_beta = 0.0
        OmegaConf.update(config, "actor_rollout_ref.actor.use_kl_loss", kl_beta > 0, merge=False)

    # Router replay: derive verl's rollout-side flag from the rllm mode (R3 records at rollout).
    router_replay_mode = config.rllm.algorithm.get("router_replay", "disabled")
    if router_replay_mode == "R3":
        OmegaConf.update(config, "actor_rollout_ref.rollout.enable_rollout_routing_replay", True, merge=False)

    # clip_ratio family: verl uses clip_ratio_{low,high} when set, else falls back to clip_ratio.
    # Mirror the effective low bound to/from rllm.algorithm.eps_clip.
    sync_clip_ratio()


def save_checkpoint(
    config: DictConfig,
    global_steps: int,
    actor_rollout_wg,
    train_dataloader=None,
) -> None:
    """Save actor and dataloader checkpoints."""
    from verl.utils.fs import local_mkdir_safe

    local_global_step_folder = os.path.join(config.trainer.default_local_dir, f"global_step_{global_steps}")
    print(f"local_global_step_folder: {local_global_step_folder}")

    actor_local_path = os.path.join(local_global_step_folder, "actor")
    actor_remote_path = None if config.trainer.default_hdfs_dir is None else os.path.join(config.trainer.default_hdfs_dir, f"global_step_{global_steps}", "actor")

    remove_previous = config.trainer.get("remove_previous_ckpt_in_save", False)
    if remove_previous:
        print("Warning: remove_previous_ckpt_in_save is deprecated, set max_actor_ckpt_to_keep=1 instead")
    max_actor_ckpt = config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous else 1

    actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, global_steps, max_ckpt_to_keep=max_actor_ckpt)

    if train_dataloader is not None:
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        torch.save(train_dataloader.state_dict(), dataloader_local_path)

    async_save = False
    ckpt_cfg = config.actor_rollout_ref.actor.get("checkpoint", {})
    if isinstance(ckpt_cfg, dict):
        async_save = ckpt_cfg.get("async_save", False)
    elif hasattr(ckpt_cfg, "async_save"):
        async_save = ckpt_cfg.async_save

    if not async_save:
        latest_path = os.path.join(config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(latest_path, "w") as f:
            f.write(str(global_steps))


def load_checkpoint(
    config: DictConfig,
    actor_rollout_wg,
    train_dataloader=None,
) -> int:
    """Load checkpoint and return global step to resume from (0 if training from scratch)."""
    from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path

    if config.trainer.resume_mode == "disable":
        return 0

    if config.trainer.default_hdfs_dir is not None:
        raise NotImplementedError("load from hdfs is not implemented yet")

    checkpoint_folder = config.trainer.default_local_dir
    if not os.path.isabs(checkpoint_folder):
        checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
    global_step_folder = find_latest_ckpt_path(checkpoint_folder)

    if config.trainer.resume_mode == "auto":
        if global_step_folder is None:
            print("Training from scratch")
            return 0
    elif config.trainer.resume_mode == "resume_path":
        assert isinstance(config.trainer.resume_from_path, str), "resume ckpt must be str type"
        assert "global_step_" in config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
        global_step_folder = config.trainer.resume_from_path
        if not os.path.isabs(global_step_folder):
            global_step_folder = os.path.join(os.getcwd(), global_step_folder)

    print(f"Load from checkpoint folder: {global_step_folder}")
    global_steps = int(global_step_folder.split("global_step_")[-1])
    print(f"Setting global step to {global_steps}")

    actor_path = os.path.join(global_step_folder, "actor")
    actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=config.trainer.del_local_ckpt_after_load)

    if train_dataloader is not None:
        dataloader_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_path):
            state_dict = torch.load(dataloader_path, weights_only=False)
            train_dataloader.load_state_dict(state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_path}, will start from scratch")

    return global_steps


def balance_batch(
    batch: DataProto,
    actor_rollout_wg,
    metrics: dict,
    use_prefix_grouper: bool = False,
    logging_prefix: str = "global_seqlen",
) -> None:
    """Reorder the batch so each DP rank gets similar total tokens.

    Mutates ``batch`` in-place via ``batch.reorder()``. Mirrors the semantics of
    ``RayPPOTrainer._balance_batch``.
    """
    attention_mask = batch.batch["attention_mask"]
    batch_size = attention_mask.shape[0]
    global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1)
    workload_lst = calculate_workload(global_seqlen_lst)

    role_key = "actor"
    if role_key not in actor_rollout_wg._dispatch_info:
        dp_rank_mapping = actor_rollout_wg._query_dispatch_info(role_key)
        actor_rollout_wg._dispatch_info[role_key] = dp_rank_mapping
    else:
        dp_rank_mapping = actor_rollout_wg._dispatch_info[role_key]
    dp_size = max(dp_rank_mapping) + 1

    if use_prefix_grouper and "uid" in batch.non_tensor_batch:
        from verl.utils.seqlen_balancing import get_group_balanced_partitions

        uid_list = list(batch.non_tensor_batch["uid"])
        seqlen_list = global_seqlen_lst.tolist()
        num_groups = len(set(uid_list))
        if num_groups % dp_size != 0:
            raise ValueError(f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) % dp_size ({dp_size}) == 0.")
        global_partition_lst = get_group_balanced_partitions(
            seqlen_list=seqlen_list,
            uid_list=uid_list,
            k_partitions=dp_size,
        )
    else:
        global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)

    if not use_prefix_grouper:
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (workload_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition

    global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
    batch.reorder(global_idx)
    global_balance_stats = log_seqlen_unbalance(
        seqlen_list=global_seqlen_lst.tolist(),
        partitions=global_partition_lst,
        prefix=logging_prefix,
    )
    metrics.update(global_balance_stats)


def start_profiling(global_steps, actor_rollout_wg, ref_policy_wg=None, use_reference_policy=False) -> None:
    actor_rollout_wg.start_profile(role="e2e", profile_step=global_steps)
    if use_reference_policy and ref_policy_wg is not None:
        ref_policy_wg.start_profile(profile_step=global_steps)


def stop_profiling(actor_rollout_wg, ref_policy_wg=None, use_reference_policy=False) -> None:
    actor_rollout_wg.stop_profile()
    if use_reference_policy and ref_policy_wg is not None:
        ref_policy_wg.stop_profile()


def build_wg_kwargs(config: DictConfig, device_name: str) -> dict[str, Any]:
    """Build kwargs for RayWorkerGroup construction."""
    wg_kwargs: dict[str, Any] = {"device_name": device_name}
    if OmegaConf.select(config.trainer, "ray_wait_register_center_timeout") is not None:
        wg_kwargs["ray_wait_register_center_timeout"] = config.trainer.ray_wait_register_center_timeout
    if OmegaConf.select(config, "global_profiler.steps") is not None:
        wg_kwargs["profile_steps"] = OmegaConf.select(config, "global_profiler.steps")
        if OmegaConf.select(config, "global_profiler.tool") == "nsys":
            assert OmegaConf.select(config, "global_profiler.global_tool_config.nsys.worker_nsight_options") is not None
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(OmegaConf.select(config, "global_profiler.global_tool_config.nsys.worker_nsight_options"))
    return wg_kwargs


def create_dataloaders(config: DictConfig, tokenizer, processor=None):
    """Create train and validation dataloaders.

    Returns (train_dataloader, val_dataloader, total_training_steps).
    """
    from torchdata.stateful_dataloader import StatefulDataLoader
    from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
    from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

    train_dataset = create_rl_dataset(
        config.data.train_files,
        config.data,
        tokenizer,
        processor,
        max_samples=config.data.get("train_max_samples", -1),
    )
    val_dataset = create_rl_dataset(
        config.data.val_files,
        config.data,
        tokenizer,
        processor,
        max_samples=config.data.get("val_max_samples", -1),
    )

    train_sampler = create_rl_sampler(config.data, train_dataset)
    num_workers = config.data["dataloader_num_workers"]

    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=config.data.get("gen_batch_size", config.data.train_batch_size),
        num_workers=num_workers,
        drop_last=True,
        collate_fn=default_collate_fn,
        sampler=train_sampler,
    )

    val_batch_size = config.data.val_batch_size
    if val_batch_size is None or val_batch_size == -1:
        val_batch_size = len(val_dataset)

    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        num_workers=num_workers,
        shuffle=config.data.get("validation_shuffle", True),
        drop_last=False,
        collate_fn=default_collate_fn,
    )

    assert len(train_dataloader) >= 1, "Train dataloader is empty!"
    assert len(val_dataloader) >= 1, "Validation dataloader is empty!"
    print(f"Size of train dataloader: {len(train_dataloader)}, Size of val dataloader: {len(val_dataloader)}")

    total_training_steps = len(train_dataloader) * config.trainer.total_epochs
    if config.trainer.total_training_steps is not None:
        total_training_steps = config.trainer.total_training_steps

    # Propagate total_training_steps into actor.optim so the LR scheduler (cosine / lr_warmup_steps_ratio) sees a valid count.
    try:
        OmegaConf.set_struct(config, True)
        with open_dict(config):
            if OmegaConf.select(config, "actor_rollout_ref.actor.optim"):
                config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
    except Exception as e:
        logger.warning(f"Could not set total_training_steps on actor.optim: {e}")

    return train_dataloader, val_dataloader, total_training_steps
