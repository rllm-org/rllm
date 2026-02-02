# http utils

import asyncio
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


def assemble_batch_from_trajectory_group_ls(trajectory_group_ls: list[TrajectoryGroup], config, tokenizer, balance_batch=None) -> DataProto:
    """
    Assemble gen_batch_output from TrajectoryGroup objects.

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

    total_trajectories = sum(len(tg.trajectories) for tg in trajectory_group_ls)
    print(f"[BatchUtils] Assembling batch from {total_trajectories} Trajectories objects")

    uids = []
    trajectory_uuids = []
    prompts = []
    response_masks = []
    responses = []
    rollout_log_probs = []
    sequences = []

    trajectory_uuid2reward = {}

    # Collect metadata for statistics
    processing_times = []
    tool_calls_times = []
    param_versions = []
    param_version_starts = []
    param_version_ends = []

    # not used by verl, but we need to use it to track the padding
    prompt_attention_masks = []
    response_attention_masks = []

    # Collect all sequences from trajectory groups
    for trajectory_group in trajectory_group_ls:
        uid = str(uuid.uuid4())
        for trajectory in trajectory_group.trajectories:
            trajectory_uid = str(uuid.uuid4())
            trajectory_uuid2reward[trajectory_uid] = trajectory.reward

            # Extract metadata if available
            if trajectory.metadata:
                processing_times.append(trajectory.metadata.get("processing_time", 0.0))
                tool_calls_times.append(trajectory.metadata.get("tool_calls_time", 0.0))
                param_versions.append(trajectory.metadata.get("param_version", 0))
                param_version_starts.append(trajectory.metadata.get("param_version_start", 0))
                param_version_ends.append(trajectory.metadata.get("param_version_end", 0))

            seqs = trajectory.merge()
            sequences.extend(seqs)
            uids.extend([uid] * len(seqs))
            trajectory_uuids.extend([trajectory_uid] * len(seqs))

    # Format sequences
    for seq in sequences:
        seq = seq.resize_prompt_length(max_prompt_length)
        prompts.append(seq.prompt_ids)
        responses.append(seq.response_ids)
        response_masks.append(seq.response_masks)
        rollout_log_probs.append(seq.response_logprobs)
        prompt_attention_masks.append([1] * len(seq.prompt_ids))
        response_attention_masks.append([1] * len(seq.response_ids))

    prompt_lens = [len(p) for p in prompts]
    response_lens = [len(r) for r in responses]
    max_prompt_len = max(prompt_lens)
    max_response_len = max(response_lens)

    pad_token_id = tokenizer.pad_token_id
    prompts_t = padding([torch.tensor(p) for p in prompts], max_prompt_len, pad_token_id, padding_side="left")
    prompt_attention_masks_t = padding([torch.tensor(pm) for pm in prompt_attention_masks], max_prompt_len, 0, padding_side="left")

    responses_t = padding([torch.tensor(r) for r in responses], max_response_len, pad_token_id, padding_side="right")
    response_attention_masks_t = padding([torch.tensor(rm) for rm in response_attention_masks], max_response_len, 0, padding_side="right")
    response_masks_t = padding([torch.tensor(rm) for rm in response_masks], max_response_len, 0, padding_side="right").long()
    rollout_log_probs_t = padding([torch.tensor(lp) for lp in rollout_log_probs], max_response_len, -100.0, padding_side="right")

    # clip the length
    responses_t = responses_t[:, :max_response_length]
    response_attention_masks_t = response_attention_masks_t[:, :max_response_length]
    response_masks_t = response_masks_t[:, :max_response_length]
    rollout_log_probs_t = rollout_log_probs_t[:, :max_response_length]

    cur_response_len = responses_t.shape[1]

    # token level rewards (place the reward at the last response token)
    token_level_scores = torch.zeros_like(responses_t, dtype=torch.float32)
    trajectory_rewards = torch.tensor([trajectory_uuid2reward[traj_uid] for traj_uid in trajectory_uuids], dtype=torch.float32)
    response_lens_t = torch.tensor(response_lens)
    # clamp to max_response_length - 1 in case response was truncated
    last_token_idx = (response_lens_t - 1).clamp(0, cur_response_len - 1)
    token_level_scores[torch.arange(len(sequences)), last_token_idx] = trajectory_rewards

    attention_masks_t = torch.cat([prompt_attention_masks_t, response_attention_masks_t], dim=-1)
    position_ids_t = torch.cumsum(attention_masks_t, dim=-1) - 1
    position_ids_t = position_ids_t.masked_fill(attention_masks_t == 0, 0)
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
            "ignore_in_loss": np.array([False] * len(uids)),
            "trajectory_rewards": np.array([trajectory_uuid2reward[traj_uid] for traj_uid in trajectory_uuids]),
            # Store metadata arrays for statistics calculation
            "processing_times": np.array(processing_times) if processing_times else np.array([0.0]),
            "tool_calls_times": np.array(tool_calls_times) if tool_calls_times else np.array([]),
            "param_version_start": np.array(param_version_starts) if param_version_starts else np.array([0]),
            "param_version_end": np.array(param_version_ends) if param_version_ends else np.array([0]),
        },
    )

    # Set meta_info for downstream processing
    # global_token_num should be a list of sequence lengths for flops_counter.estimate_flops()
    batch.meta_info["global_token_num"] = batch_seqlens

    # Calculate and add statistics to meta_info
    if processing_times:
        processing_times_arr = np.array(processing_times)
        processing_time_stats = {
            "fully_async/processing_time/avg": np.mean(processing_times_arr),
            "fully_async/processing_time/max": np.max(processing_times_arr),
            "fully_async/processing_time/min": np.min(processing_times_arr),
            "fully_async/processing_time/tp50": np.percentile(processing_times_arr, 50),
            "fully_async/processing_time/tp95": np.percentile(processing_times_arr, 95),
            "fully_async/processing_time/tp99": np.percentile(processing_times_arr, 99),
        }
        batch.meta_info.update(processing_time_stats)

    # Tool calls stats
    tool_calls_arr = np.array([t for t in tool_calls_times if t > 0])
    if len(tool_calls_arr) > 0:
        tool_calls_stats = {
            "timing_s/agent_loop/tool_calls/max": np.max(tool_calls_arr),
            "timing_s/agent_loop/tool_calls/min": np.min(tool_calls_arr),
            "timing_s/agent_loop/tool_calls/mean": np.mean(tool_calls_arr),
        }
        batch.meta_info.update(tool_calls_stats)

    # Partial rollout stats (param version changed during generation)
    if param_version_starts and param_version_ends:
        param_version_diff = [abs(a - b) for a, b in zip(param_version_ends, param_version_starts)]
        num_diff0 = param_version_diff.count(0)
        partial_stats = {
            "fully_async/partial/total_partial_num": len(param_version_diff) - num_diff0,
            "fully_async/partial/partial_ratio": (len(param_version_diff) - num_diff0) / len(param_version_diff) if param_version_diff else 0,
            "fully_async/partial/max_partial_span": max(param_version_diff) if param_version_diff else 0,
        }
        batch.meta_info.update(partial_stats)

    # Parameter version tracking
    if param_versions:
        batch.meta_info["rollout_param_versions"] = param_versions
        batch.meta_info["param_version_diversity"] = len(set(param_versions))
        batch.meta_info["trajectory_param_versions"] = param_version_ends

    if balance_batch:
        balance_batch(batch, metrics={})

    print(f"[BatchUtils] Batch assembly completed in {time.time() - start_time:.2f}s")

    return batch
