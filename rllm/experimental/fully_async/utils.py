# http utils

import asyncio
import os
import time
import uuid
from collections import defaultdict

import httpx
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from verl import DataProto
from verl.utils.torch_functional import pad_sequence_to_length

from .protocol import TrajectoryGroup

_client: httpx.AsyncClient | None = None


# Checkpoint utils


async def save_dataloader_checkpoint(dataloader, dataloader_lock, checkpoint_folder: str):
    """Save dataloader state to checkpoint folder.

    Args:
        dataloader: StatefulDataLoader instance
        dataloader_lock: asyncio.Lock for thread-safe access
        checkpoint_folder: Path to checkpoint folder (e.g., .../global_step_100)

    Note: Due to the asynchronous nature, there may be some in-flight samples
    (in pending/result queues). These samples will be regenerated on resume.
    """
    from verl.utils.fs import local_mkdir_safe

    local_mkdir_safe(checkpoint_folder)
    dataloader_path = os.path.join(checkpoint_folder, "data.pt")

    async with dataloader_lock:
        dataloader_state_dict = dataloader.state_dict()

    torch.save(dataloader_state_dict, dataloader_path)
    print(f"[Checkpoint] Saved dataloader state to {dataloader_path}")


def load_dataloader_checkpoint(dataloader, config) -> int:
    """Load dataloader state from checkpoint based on resume mode.

    Args:
        dataloader: StatefulDataLoader instance to load state into
        config: Config object with trainer.resume_mode, trainer.default_local_dir, etc.

    Returns:
        trainer_global_steps from checkpoint (0 if no checkpoint found or resume disabled)
    """
    from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path

    if config.trainer.resume_mode == "disable":
        print("[Checkpoint] Resume mode is disabled, starting from scratch")
        return 0

    # Determine checkpoint folder path
    if config.trainer.default_hdfs_dir is not None:
        raise NotImplementedError("[Checkpoint] Load from hdfs is not implemented yet")

    checkpoint_folder = config.trainer.default_local_dir
    if not os.path.isabs(checkpoint_folder):
        working_dir = os.getcwd()
        checkpoint_folder = os.path.join(working_dir, checkpoint_folder)

    global_step_folder = find_latest_ckpt_path(checkpoint_folder)

    # Find and validate global_step_folder based on resume mode
    if config.trainer.resume_mode == "auto":
        if global_step_folder is None:
            print("[Checkpoint] Training from scratch (no checkpoint found)")
            return 0
    elif config.trainer.resume_mode == "resume_path":
        assert isinstance(config.trainer.resume_from_path, str), "[Checkpoint] resume_from_path must be str type"
        assert "global_step_" in config.trainer.resume_from_path, "[Checkpoint] resume_from_path must specify the global_steps"
        global_step_folder = config.trainer.resume_from_path
        if not os.path.isabs(global_step_folder):
            working_dir = os.getcwd()
            global_step_folder = os.path.join(working_dir, global_step_folder)
    else:
        raise ValueError(f"[Checkpoint] Unknown resume_mode: {config.trainer.resume_mode}")

    print(f"[Checkpoint] Loading from: {global_step_folder}")

    # Extract trainer_global_steps from folder name
    trainer_global_steps = int(global_step_folder.split("global_step_")[-1])

    # Load dataloader state
    dataloader_path = os.path.join(global_step_folder, "data.pt")
    if os.path.exists(dataloader_path):
        dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
        dataloader.load_state_dict(dataloader_state_dict)
        print(f"[Checkpoint] Loaded dataloader state from {dataloader_path}")
    else:
        print(f"[Checkpoint] Warning: No dataloader state found at {dataloader_path}")

    return trainer_global_steps


def calculate_rollout_global_steps(trainer_global_steps: int, config) -> int:
    """Calculate rollout global_steps from trainer_global_steps.

    Formula: rollout_steps = trainer_steps * required_samples * sync_frequency + 1

    Args:
        trainer_global_steps: The trainer's global step count
        config: Config object with async_training settings

    Returns:
        The rollout global_steps value
    """
    # Use config.async_training.required_samples directly (consistent with fully_async_rollouter.py)
    required_samples = config.async_training.required_samples
    return trainer_global_steps * required_samples * config.async_training.trigger_parameter_sync_step + 1


def calculate_max_concurrency(config) -> int:
    """
    Calculate max HTTP concurrency: sglang_server_concurrency * num_engines
    Matches slime's approach.
    """
    sglang_server_concurrency = config.async_training.get("sglang_server_concurrency", 512)
    rollout_n_gpus = config.rollout.nnodes * config.rollout.n_gpus_per_node
    tensor_parallel_size = config.actor_rollout_ref.rollout.get("tensor_model_parallel_size", 1)
    num_engines = rollout_n_gpus // tensor_parallel_size
    return sglang_server_concurrency * num_engines


def get_client(max_connections: int = 100) -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max_connections),
            timeout=httpx.Timeout(None),
        )
    return _client


async def get(url: str):
    client = get_client()
    response = await client.get(url)
    response.raise_for_status()
    return response.json()


async def post(url: str, payload: dict = None, max_retries: int = 3, expect_json: bool = True):
    client = get_client()
    for attempt in range(max_retries):
        try:
            response = await client.post(url, json=payload or {})
            response.raise_for_status()
            if expect_json:
                return response.json() if response.content else {}
            return response
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(1)


# Sglang utils


async def abort_async(router_url):
    """Abort all requests on all workers behind the router and WAIT for completion.

    This uses SGLang's /pause_generation endpoint with mode="abort" which:
    1. Aborts all ongoing requests
    2. Waits until all requests are actually completed (not fire-and-forget)
    3. Pauses the scheduler to prevent new requests from being processed

    This should be called from within an async context (e.g., a Ray actor's async method)
    to avoid event loop issues with the global HTTP client.
    """
    response = await get(f"{router_url.strip('/')}/workers")
    urls = [worker["url"] for worker in response["workers"]]

    # Use /pause_generation with mode="abort" which WAITS for all requests to complete
    # This is different from /abort_request which is fire-and-forget
    await asyncio.gather(*[post(f"{url}/pause_generation", {"mode": "abort"}, expect_json=True) for url in urls])


async def continue_generation_async(router_url):
    """Resume generation on all workers behind the router.

    This should be called after abort_async to allow the workers to process new requests.
    """
    response = await get(f"{router_url.strip('/')}/workers")
    urls = [worker["url"] for worker in response["workers"]]

    await asyncio.gather(*[post(f"{url}/continue_generation", {}, expect_json=True) for url in urls])


# Sample utils


def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    traj_uuids: np.ndarray,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        config: `(Optional[AlgoConfig])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    traj_uuid2score = dict()

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            traj_uuid = traj_uuids[i]
            if traj_uuid not in traj_uuid2score:
                id2score[index[i]].append(scores[i])
                traj_uuid2score[traj_uuid] = scores[i]
            else:
                assert scores[i] == traj_uuid2score[traj_uuid], f"Score for traj_uuid {traj_uuid} is not the same"

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def padding(tensor_ls, max_len, pad_value, padding_side="left"):
    """
    Pad a list of 1D tensors to max_len.

    Args:
        tensor_ls: List of 1D tensors with varying lengths
        max_len: Target length to pad to
        pad_value: Value to use for padding (e.g., pad_token_id)
        padding_side: 'left' or 'right' padding

    Returns:
        Stacked tensor of shape [batch_size, max_len]
    """
    # First, use pad_sequence to pad to the max length in the batch (right-padded)
    # pad_sequence expects a list of tensors and pads them to equal length
    padded = pad_sequence(tensor_ls, batch_first=True, padding_value=pad_value)

    # Then use pad_sequence_to_length to ensure we reach max_len
    # pad_sequence_to_length pads a 2D tensor [bs, seq_len] -> [bs, max_len]
    left_pad = padding_side == "left"
    padded = pad_sequence_to_length(padded, max_len, pad_value, left_pad=left_pad)

    return padded


def apply_rejection_sampling(
    trajectory_group_ls: list[TrajectoryGroup],
    config,
) -> tuple[list[TrajectoryGroup], dict[str, float]]:
    """
    Apply rejection sampling to filter trajectory groups based on is_correct field.

    This follows the same logic as agent_ppo_trainer.py:
    - Filter out groups where ALL trajectories are correct (solve_all)
    - Filter out groups where ALL trajectories are incorrect (solve_none)
    - Keep only groups with mixed results (some correct, some incorrect)

    Args:
        trajectory_group_ls: List of TrajectoryGroup objects to filter
        config: Configuration object with rollout.rejection_sample (bool) setting

    Returns:
        Tuple of:
        - Filtered list of TrajectoryGroup objects (filtered only if enabled)
        - Statistics dict with solve_none, solve_all, solve_partial counts (always computed)
    """
    enable = config.rollout.get("rejection_sample", False)

    stats: dict[str, float] = {
        "rejection_sample/solve_none": 0,
        "rejection_sample/solve_all": 0,
        "rejection_sample/solve_partial": 0,
        "rejection_sample/total_groups": len(trajectory_group_ls),
        "rejection_sample/filtered_groups": 0,
        "rejection_sample/enabled": int(enable),
    }

    # Collect ALL rewards BEFORE filtering for unfiltered metrics
    all_rewards = []
    for trajectory_group in trajectory_group_ls:
        for trajectory in trajectory_group.trajectories:
            all_rewards.append(trajectory.reward)

    # Track unfiltered reward statistics (before any filtering)
    # NOTE: We store both _sum and _count to enable correct weighted averaging
    # across batches in MetricsAggregator. The mean is computed from sum/count.
    if all_rewards:
        stats["rejection_sample/unfiltered_reward_sum"] = float(np.sum(all_rewards))
        stats["rejection_sample/unfiltered_reward_count"] = len(all_rewards)
        stats["rejection_sample/unfiltered_reward_mean"] = float(np.mean(all_rewards))
        stats["rejection_sample/unfiltered_reward_std"] = float(np.std(all_rewards))
        stats["rejection_sample/unfiltered_reward_min"] = float(np.min(all_rewards))
        stats["rejection_sample/unfiltered_reward_max"] = float(np.max(all_rewards))
        stats["rejection_sample/unfiltered_num_trajectories"] = len(all_rewards)

    filtered_groups = []

    for trajectory_group in trajectory_group_ls:
        # Extract is_correct from each trajectory's metadata
        is_correct_list = []
        for trajectory in trajectory_group.trajectories:
            metadata = trajectory.metadata or {}
            # is_correct can be in metadata directly or under different keys
            is_correct = metadata.get("is_correct", None)
            if is_correct is None:
                # Fallback: infer from reward if is_correct not explicitly set
                is_correct = trajectory.reward > 0
            is_correct_list.append(bool(is_correct))

        if not is_correct_list:
            # Empty trajectory group, skip
            continue

        # Check if all correct or all incorrect
        all_correct = all(is_correct_list)
        all_incorrect = not any(is_correct_list)

        if all_incorrect:
            stats["rejection_sample/solve_none"] += 1
            if not enable:
                # Keep all groups when not filtering
                filtered_groups.append(trajectory_group)
        elif all_correct:
            stats["rejection_sample/solve_all"] += 1
            if not enable:
                # Keep all groups when not filtering
                filtered_groups.append(trajectory_group)
        else:
            # Mixed results - always keep this group
            stats["rejection_sample/solve_partial"] += 1
            filtered_groups.append(trajectory_group)

    stats["rejection_sample/filtered_groups"] = len(filtered_groups)

    # Collect filtered rewards for comparison
    filtered_rewards = []
    for trajectory_group in filtered_groups:
        for trajectory in trajectory_group.trajectories:
            filtered_rewards.append(trajectory.reward)

    # Track filtered reward statistics (after filtering)
    # NOTE: We store both _sum and _count to enable correct weighted averaging
    # across batches in MetricsAggregator. The mean is computed from sum/count.
    if filtered_rewards:
        stats["rejection_sample/filtered_reward_sum"] = float(np.sum(filtered_rewards))
        stats["rejection_sample/filtered_reward_count"] = len(filtered_rewards)
        stats["rejection_sample/filtered_reward_mean"] = float(np.mean(filtered_rewards))
        stats["rejection_sample/filtered_reward_std"] = float(np.std(filtered_rewards))
        stats["rejection_sample/filtered_num_trajectories"] = len(filtered_rewards)

    # Log rejection sampling results
    unfiltered_mean = stats.get("rejection_sample/unfiltered_reward_mean", 0)
    filtered_mean = stats.get("rejection_sample/filtered_reward_mean", 0)
    if enable:
        print(f"[RejectionSampling] Applied rejection sampling: solve_none={stats['rejection_sample/solve_none']}, solve_all={stats['rejection_sample/solve_all']}, solve_partial={stats['rejection_sample/solve_partial']}, kept {len(filtered_groups)}/{len(trajectory_group_ls)} groups, unfiltered_reward={unfiltered_mean:.4f}, filtered_reward={filtered_mean:.4f}")
    else:
        print(f"[RejectionSampling] Stats (filtering disabled): solve_none={stats['rejection_sample/solve_none']}, solve_all={stats['rejection_sample/solve_all']}, solve_partial={stats['rejection_sample/solve_partial']}, reward_mean={unfiltered_mean:.4f}")

    return filtered_groups, stats


def assemble_batch_from_trajectory_group_ls(trajectory_group_ls: list[TrajectoryGroup], config, tokenizer, balance_batch=None) -> DataProto:
    """
    Assemble gen_batch_output from TrajectoryGroup objects.
    Optimized version: pre-allocates tensors and fills in single pass.

    Args:
        trajectory_group_ls: List of TrajectoryGroup objects
        config: Configuration object with data.max_prompt_length and data.max_response_length
        tokenizer: Tokenizer instance
        balance_batch: Optional function to balance the batch

    Returns:
        DataProto: Assembled gen_batch_output

    Raises:
        ValueError: If trajectory_group_ls is empty
    """
    max_prompt_length = config.data.max_prompt_length
    max_response_length = config.data.max_response_length

    start_time = time.time()

    if not trajectory_group_ls:
        raise ValueError("Empty trajectory group provided for batch assembly")

    # Apply rejection sampling at the top
    filtered_groups, rejection_stats = apply_rejection_sampling(trajectory_group_ls, config)

    # Handle edge case: all groups were filtered out
    # Keep the first group but mark for masking all response tokens
    mask_all_responses = False
    if not filtered_groups and trajectory_group_ls:
        print("[BatchUtils] All trajectory groups filtered out by rejection sampling. Keeping first group with masked responses.")
        filtered_groups = [trajectory_group_ls[0]]
        mask_all_responses = True
        rejection_stats["rejection_sample/all_filtered_fallback"] = 1
    else:
        rejection_stats["rejection_sample/all_filtered_fallback"] = 0

    # Use filtered_groups for the rest of the processing
    trajectory_group_ls = filtered_groups

    # Pre-collect all data in single pass
    all_data = []  # List of (uid, trajectory_uid, seq)
    trajectory_uuid2reward = {}

    # Collect metadata for statistics (initialize with None to track which trajectories have metadata)
    processing_times = []
    tool_calls_times = []
    param_versions = []
    param_version_starts = []
    param_version_ends = []
    # Store all user-defined custom metrics
    custom_metrics: dict[str, list] = defaultdict(list)

    for trajectory_group in trajectory_group_ls:
        uid = str(uuid.uuid4())
        for trajectory in trajectory_group.trajectories:
            trajectory_uid = str(uuid.uuid4())
            trajectory_uuid2reward[trajectory_uid] = trajectory.reward

            # Extract metadata if available, use defaults if not provided
            # This ensures all lists have consistent length matching the number of trajectories
            metadata = trajectory.metadata or {}
            processing_times.append(metadata.get("processing_time", 0.0))
            tool_calls_times.append(metadata.get("tool_calls_time", 0.0))
            param_versions.append(metadata.get("param_version", 0))
            param_version_starts.append(metadata.get("param_version_start", 0))
            param_version_ends.append(metadata.get("param_version_end", 0))

            # Collect user-defined custom metrics
            # Built-in keys are handled separately above, so we skip them here
            builtin_keys = {"processing_time", "tool_calls_time", "param_version", "param_version_start", "param_version_end"}
            for key, value in metadata.items():
                if key not in builtin_keys:
                    # Add custom/ prefix if not already present
                    metric_key = key if key.startswith("custom/") else f"custom/{key}"
                    custom_metrics[metric_key].append(value)

            for seq in trajectory.merge():
                seq = seq.resize_prompt_length(max_prompt_length)
                all_data.append((uid, trajectory_uid, seq))

    num_sequences = len(all_data)
    print(f"[BatchUtils] Assembling batch from {num_sequences} sequences")

    # Find max lengths in single pass
    max_prompt_len = 0
    max_response_len = 0
    for _, _, seq in all_data:
        max_prompt_len = max(max_prompt_len, len(seq.prompt_ids))
        max_response_len = max(max_response_len, len(seq.response_ids))
    max_response_len = min(max_response_len, max_response_length)

    pad_token_id = tokenizer.pad_token_id

    # Pre-allocate tensors (MUCH faster than repeated torch.tensor + padding)
    prompts_t = torch.full((num_sequences, max_prompt_len), pad_token_id, dtype=torch.long)
    prompt_attention_masks_t = torch.zeros((num_sequences, max_prompt_len), dtype=torch.long)
    responses_t = torch.full((num_sequences, max_response_len), pad_token_id, dtype=torch.long)
    response_attention_masks_t = torch.zeros((num_sequences, max_response_len), dtype=torch.long)
    response_masks_t = torch.zeros((num_sequences, max_response_len), dtype=torch.long)
    rollout_log_probs_t = torch.zeros((num_sequences, max_response_len), dtype=torch.float32)

    uids = []
    trajectory_uuids = []
    response_lens = []

    # Fill tensors in single pass (left-pad prompts, right-pad responses)
    for i, (uid, traj_uid, seq) in enumerate(all_data):
        uids.append(uid)
        trajectory_uuids.append(traj_uid)

        p_ids = seq.prompt_ids
        r_ids = seq.response_ids
        r_masks = seq.response_masks
        r_logprobs = seq.response_logprobs

        p_len = len(p_ids)
        r_len_original = len(r_ids)
        r_len = min(r_len_original, max_response_len)
        response_lens.append(r_len_original)  # Original length before clipping

        # Left-pad prompts
        prompts_t[i, max_prompt_len - p_len :] = torch.as_tensor(p_ids, dtype=torch.long)
        prompt_attention_masks_t[i, max_prompt_len - p_len :] = 1

        # Right-pad responses (clip to max_response_len)
        responses_t[i, :r_len] = torch.as_tensor(r_ids[:r_len], dtype=torch.long)
        response_attention_masks_t[i, :r_len] = 1
        # Clip masks and logprobs to their actual lengths (in case they differ from r_ids)
        assert len(r_masks) == len(r_ids), "{} != {}".format(len(r_masks), len(r_ids))
        r_mask_len = min(len(r_masks), r_len)
        r_logprob_len = min(len(r_logprobs), r_len)
        response_masks_t[i, :r_mask_len] = torch.as_tensor(r_masks[:r_mask_len], dtype=torch.long)
        # Ensure logprobs are Python floats (not nested structures or tensors)
        # This prevents "unsupported format string passed to Tensor.__format__" errors
        logprobs_as_floats = [float(lp) for lp in r_logprobs[:r_logprob_len]]
        rollout_log_probs_t[i, :r_logprob_len] = torch.tensor(logprobs_as_floats, dtype=torch.float32)

    # Multiply response_mask by attention_mask (consistent with verl's agent_loop.py:586)
    # This ensures padding positions have mask=0
    response_masks_t = response_masks_t * response_attention_masks_t

    # If all groups were filtered out by rejection sampling, mask all response tokens
    if mask_all_responses:
        print("[BatchUtils] Masking all response tokens due to rejection sampling fallback.")
        response_masks_t = torch.zeros_like(response_masks_t)

    # Token level rewards (place the reward at the last response token)
    token_level_scores = torch.zeros_like(responses_t, dtype=torch.float32)
    trajectory_rewards = torch.tensor([trajectory_uuid2reward[traj_uid] for traj_uid in trajectory_uuids], dtype=torch.float32)
    response_lens_t = torch.tensor(response_lens, dtype=torch.long)
    # clamp to max_response_len - 1 in case response was truncated
    last_token_idx = (response_lens_t - 1).clamp(0, max_response_len - 1)
    token_level_scores[torch.arange(num_sequences, dtype=torch.long), last_token_idx] = trajectory_rewards

    # Concatenate and compute derived tensors
    attention_masks_t = torch.cat([prompt_attention_masks_t, response_attention_masks_t], dim=-1)
    position_ids_t = torch.clip(torch.cumsum(attention_masks_t, dim=-1) - 1, min=0, max=None)
    input_ids_t = torch.cat([prompts_t, responses_t], dim=-1)

    tensor_dict = {
        "attention_mask": attention_masks_t,
        "input_ids": input_ids_t,
        "position_ids": position_ids_t,
        "prompts": prompts_t,
        "response_mask": response_masks_t,
        "responses": responses_t,
        "rollout_log_probs": rollout_log_probs_t,
        "token_level_scores": token_level_scores,
    }

    # Calculate global_token_num for MFU calculation
    # This should be a list of sequence lengths (one per sample), not a single total
    # Each sequence length = number of non-padded tokens (sum of attention mask for that row)
    batch_seqlens = attention_masks_t.sum(dim=1).tolist()

    batch = DataProto.from_dict(
        tensors=tensor_dict,
        non_tensors={
            "uids": np.array(uids),
            "trajectory_uuids": np.array(trajectory_uuids),
            "response_clipped": np.array([l > max_response_length for l in response_lens]),
            "ignore_in_loss": np.array([False] * num_sequences),
            "trajectory_rewards": np.array([trajectory_uuid2reward[traj_uid] for traj_uid in trajectory_uuids]),
            # NOTE: processing_times, tool_calls_times, param_version_start/end are NOT included here
            # because they are per-trajectory (1024) while batch is per-sequence (1076).
            # They are used only for statistics in meta_info below.
        },
    )

    # Pad batch to actor_world_size for distributed training
    from verl.protocol import pad_dataproto_to_divisor

    actor_world_size = config.trainer.nnodes * config.trainer.n_gpus_per_node
    original_batch_size = len(batch)
    batch, pad_size = pad_dataproto_to_divisor(batch, actor_world_size)
    batch.meta_info["pad_size"] = pad_size

    # Mark padded samples as ignore_in_loss so they don't contribute to training
    if pad_size > 0:
        batch.non_tensor_batch["ignore_in_loss"][original_batch_size:] = True

    # Set meta_info for downstream processing
    # global_token_num should be a list of sequence lengths for flops_counter.estimate_flops()
    batch.meta_info["global_token_num"] = batch_seqlens

    # Calculate and add statistics to meta_info
    if processing_times:
        processing_times_arr = np.array(processing_times)
        batch.meta_info.update(
            {
                "fully_async/processing_time/avg": np.mean(processing_times_arr),
                "fully_async/processing_time/max": np.max(processing_times_arr),
                "fully_async/processing_time/min": np.min(processing_times_arr),
                "fully_async/processing_time/tp50": np.percentile(processing_times_arr, 50),
                "fully_async/processing_time/tp95": np.percentile(processing_times_arr, 95),
                "fully_async/processing_time/tp99": np.percentile(processing_times_arr, 99),
            }
        )

    # Tool calls stats
    tool_calls_arr = np.array([t for t in tool_calls_times if t > 0])
    if len(tool_calls_arr) > 0:
        batch.meta_info.update(
            {
                "timing_s/agent_loop/tool_calls/max": np.max(tool_calls_arr),
                "timing_s/agent_loop/tool_calls/min": np.min(tool_calls_arr),
                "timing_s/agent_loop/tool_calls/mean": np.mean(tool_calls_arr),
            }
        )

    # Partial rollout stats (param version changed during generation)
    if param_version_starts and param_version_ends:
        param_version_diff = [abs(a - b) for a, b in zip(param_version_ends, param_version_starts, strict=False)]
        num_diff0 = param_version_diff.count(0)
        batch.meta_info.update(
            {
                "fully_async/partial/total_partial_num": len(param_version_diff) - num_diff0,
                "fully_async/partial/partial_ratio": (len(param_version_diff) - num_diff0) / len(param_version_diff) if param_version_diff else 0,
                "fully_async/partial/max_partial_span": max(param_version_diff) if param_version_diff else 0,
            }
        )

    # Parameter version tracking
    if param_versions:
        batch.meta_info["rollout_param_versions"] = param_versions
        batch.meta_info["param_version_diversity"] = len(set(param_versions))
        batch.meta_info["trajectory_param_versions"] = param_version_ends

    # Add user-defined custom metrics to meta_info
    # Users can define metrics with "custom/" prefix in trajectory.metadata
    for metric_key, values in custom_metrics.items():
        if values:
            values_arr = np.array(values)
            batch.meta_info[f"{metric_key}/avg"] = float(np.mean(values_arr))
            batch.meta_info[f"{metric_key}/max"] = float(np.max(values_arr))
            batch.meta_info[f"{metric_key}/min"] = float(np.min(values_arr))

    if balance_batch:
        balance_batch(batch, metrics={})

    # Add rejection sampling stats to meta_info for logging
    batch.meta_info.update(rejection_stats)

    print(f"[BatchUtils] Batch assembly completed in {time.time() - start_time:.2f}s")

    return batch