import random
import time

import hydra

from rllm.experimental.fully_async.protocol import Trajectory
from rllm.experimental.fully_async.runner import AsyncAgentTrainer
from verl.utils.reward_score import default_compute_score

# DAPO overlong buffer settings - defaults (will be overridden in main via closure)
_DEFAULT_OVERLONG_BUFFER_ENABLED = True
_DEFAULT_MAX_RESP_LEN = 8192
_DEFAULT_OVERLONG_BUFFER_LEN = 2048
_DEFAULT_OVERLONG_PENALTY_FACTOR = 1.0


async def _rollout_fn_impl(
    client,
    tokenizer,
    overlong_buffer_enabled=_DEFAULT_OVERLONG_BUFFER_ENABLED,
    max_resp_len=_DEFAULT_MAX_RESP_LEN,
    overlong_buffer_len=_DEFAULT_OVERLONG_BUFFER_LEN,
    overlong_penalty_factor=_DEFAULT_OVERLONG_PENALTY_FACTOR,
    **kwargs,
):
    """
    Rollout function for DAPO training with rllm dataset format.

    Expected datum format from DatasetRegistry:
    {
        'data_source': 'math_dapo',
        'prompt': [{'content': '...', 'role': 'user'}],
        'ability': 'MATH',
        'reward_model': {'ground_truth': '34', 'style': 'rule-lighteval/MATH_v2'},
        'extra_info': {'index': '...'}
    }
    """
    start_time = time.time()
    param_version_start = client.cur_version

    # Extract prompt from dataset (chat format: [{'content': '...', 'role': 'user'}])
    messages = kwargs["prompt"]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
    )

    # Get sampling params from config or use defaults
    # Match verl's fully_async_policy/agent_loop/agent_loop.py:101-106
    # Note: verl does NOT set top_k in sampling_params (uses server default from config)
    # Note: verl hardcodes repetition_penalty=1.0 (not from config)
    sampling_params = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.0,
        "max_new_tokens": 8192,
    }

    output = await client.generate(prompt_ids, sampling_params=sampling_params)

    # Capture timing and version info
    end_time = time.time()
    param_version_end = client.cur_version
    processing_time = end_time - start_time

    # Extract response_ids from output_chunks (OutputWithVersion protocol)
    response_ids = []
    for chunk in output.output_chunks:
        response_ids.extend(chunk.response_ids)

    # Decode the response for reward calculation
    response_str = tokenizer.decode(response_ids, skip_special_tokens=False)
    if random.random() < 0.001:
        prompt_str = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        print(f"[FullyAsyncRollouter DEBUG] Prompt: {prompt_str}")
        print(f"[FullyAsyncRollouter DEBUG] Response: {response_str}")

    # Extract ground_truth and data_source from kwargs (rllm dataset format)
    reward_model_info = kwargs["reward_model"]
    ground_truth = reward_model_info.get("ground_truth", "")
    data_source = kwargs["data_source"]

    # Compute reward using default_compute_score (same as DAPORewardManager)
    try:
        result = default_compute_score(
            data_source=data_source,
            solution_str=tokenizer.decode(response_ids, skip_special_tokens=True),
            ground_truth=ground_truth,
        )
        if isinstance(result, dict):
            score = result["score"]
        else:
            score = float(result)
    except Exception as e:
        print(f"[RolloutFn] Error computing reward: {e}, using default score -1.0")
        score = -1.0

    reward = score

    # Apply overlong penalty (DAPO-specific feature)
    if overlong_buffer_enabled:
        valid_response_length = len(response_ids)
        expected_len = max_resp_len - overlong_buffer_len
        exceed_len = valid_response_length - expected_len
        overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
        reward += overlong_reward

    # Store metadata for statistics tracking
    metadata = {
        "processing_time": processing_time,
        "param_version_start": param_version_start,
        "param_version_end": param_version_end,
        "param_version": param_version_end,
        "is_partial": param_version_start != param_version_end,
        "tool_calls_time": 0.0,
    }

    return Trajectory(sequences=[output.to_sequence()], reward=reward, metadata=metadata)


@hydra.main(config_path="config", config_name="fully_async_ppo_trainer", version_base=None)
def main(config):
    """
    Main entry point for DAPO fully async training.

    The config is loaded from:
    - Base: rllm/experimental/fully_async/config/fully_async_ppo_trainer.yaml
    - Which inherits from: verl/trainer/config/ppo_trainer.yaml

    You can override any config value from the command line.
    """
    # Extract overlong buffer settings from config
    overlong_cfg = config.reward_model.get("reward_kwargs", {}).get("overlong_buffer_cfg", {})
    _overlong_buffer_enabled = overlong_cfg.get("enable", True) if overlong_cfg else True
    _overlong_buffer_len = overlong_cfg.get("len", 2048) if overlong_cfg else 2048
    _overlong_penalty_factor = overlong_cfg.get("penalty_factor", 1.0) if overlong_cfg else 1.0
    _max_resp_len = config.data.get("max_response_length", 8192)

    # Create rollout_fn as a closure to capture config values
    # This ensures the values are serialized correctly when sent to Ray workers
    async def rollout_fn_with_config(client, tokenizer, **kwargs):
        return await _rollout_fn_impl(
            client,
            tokenizer,
            overlong_buffer_enabled=_overlong_buffer_enabled,
            max_resp_len=_max_resp_len,
            overlong_buffer_len=_overlong_buffer_len,
            overlong_penalty_factor=_overlong_penalty_factor,
            **kwargs,
        )

    trainer = AsyncAgentTrainer(
        config=config,
        dataset_name=config.async_training.dataset_name,
        rollout_fn=rollout_fn_with_config,
    )

    trainer.train()


if __name__ == "__main__":
    main()