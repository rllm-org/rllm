#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export FIREWORKS_API_KEY="${FIREWORKS_API_KEY:-$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.rllm/config.json')))['api_keys']['fireworks'])" 2>/dev/null || true)}"
if [ -z "${FIREWORKS_API_KEY}" ]; then
    echo "FIREWORKS_API_KEY is not set and ~/.rllm/config.json has no api_keys.fireworks (run: rllm model setup)" >&2
    exit 1
fi

export TERMINAL_SANDBOX_BACKEND="${TERMINAL_SANDBOX_BACKEND:-modal}"
export TERMINAL_AGENT="${TERMINAL_AGENT:-native_react}"
export TB_TRAIN_DATASET="${TB_TRAIN_DATASET:-tb-v2-debug}"
export NATIVE_REACT_MAX_TURNS="${NATIVE_REACT_MAX_TURNS:-100}"
export RLLM_SANDBOX_MAX_CPUS="${RLLM_SANDBOX_MAX_CPUS:-0.125}"
export RLLM_SANDBOX_MAX_MEMORY_MB="${RLLM_SANDBOX_MAX_MEMORY_MB:-256}"
export RLLM_TUNNEL_SPEC="${RLLM_TUNNEL_SPEC:-null}"
export RLLM_HARNESS_RUN_TIMEOUT_S="${RLLM_HARNESS_RUN_TIMEOUT_S:-3600}"
export RLLM_SANDBOX_TIMEOUT_S="${RLLM_SANDBOX_TIMEOUT_S:-4800}"

python -u train.py \
    rllm/backend=fireworks \
    model.name=accounts/fireworks/models/qwen3p5-35b-a3b \
    model.tokenizer_model=Qwen/Qwen3.5-35B-A3B \
    model.lora_rank=0 \
    fireworks_config.policy_trainer_shape_id=accounts/fireworks/trainingShapes/qwen3p5-35b-a3b-256k \
    fireworks_config.policy_trainer_replica_count=2 \
    fireworks_config.rollout_deployment_replica_count=6 \
    training.group_size=32 \
    training.learning_rate=1e-6 \
    training.beta2=0.999 \
    training.max_length=133072 \
    rllm.rollout.train.temperature=1.0 \
    rllm.rollout.train.top_p=1.0 \
    rllm.rollout.val.temperature=1.0 \
    rllm.rollout.val.top_p=1.0 \
    data.max_prompt_length=100000 \
    data.max_response_length=32768 \
    data.train_batch_size=1 \
    data.val_batch_size=-1 \
    rllm.data.max_prompt_length=100000 \
    rllm.data.max_response_length=32768 \
    rllm.data.train_batch_size=1 \
    rllm.data.val_batch_size=-1 \
    rllm.compact_filtering.enable=true \
    rllm.compact_filtering.mask_max_prompt_length_exceeded=false \
    rllm.compact_filtering.mask_max_response_length_exceeded=false \
    rllm.compact_filtering.mask_max_turns_exceeded=false \
    rllm.compact_filtering.mask_timeout=false \
    rllm.algorithm.adv_estimator=grpo \
    rllm.algorithm.router_replay=R3 \
    rllm.algorithm.loss_fn=dppo_tv \
    rllm.algorithm.loss_agg_mode=token-mean \
    '+rllm.algorithm.loss_params.delta=0.2' \
    rllm.algorithm.eps_clip=0.2 \
    rllm.algorithm.rollout_correction.bypass_mode=true \
    rllm.algorithm.norm_adv_by_std_in_grpo=false \
    rllm.async_training.enable=true \
    rllm.async_training.mini_batch_size=8 \
    rllm.async_training.fwd_bwd_group_size=8 \
    rllm.async_training.staleness_threshold=3.0 \
    rllm.async_training.trigger_parameter_sync_step=1 \
    rllm.async_training.partial_rollout=true \
    rllm.workflow.n_parallel_tasks=256 \
    rllm.workflow.raise_on_error=false \
    rllm.rejection_sample.filter_uniform_groups=true \
    rllm.gateway.port=9201 \
    "rllm.gateway.tunnel=${RLLM_TUNNEL_SPEC}" \
    rllm.gateway.num_workers=4 \
    rllm.gateway.cumulative_token_mode=true \
    rllm.gateway.renderer_family=qwen3.5 \
    rllm.trainer.total_epochs=500 \
    rllm.trainer.dump_batch_dir=null \
    rllm.trainer.logger='[wandb]' \
    rllm.trainer.project_name='terminal-rl' \
    rllm.trainer.experiment_name='qwen3p5-35b-a3b-tb-v2-fft' \
    rllm.trainer.val_before_train=false \
    rllm.trainer.test_freq=-1 \
    rllm.trainer.save_freq=10 \
    "$@"
