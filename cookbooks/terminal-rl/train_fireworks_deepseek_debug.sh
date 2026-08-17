set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export FIREWORKS_API_KEY="${FIREWORKS_API_KEY:-$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.rllm/config.json')))['api_keys']['fireworks'])" 2>/dev/null || true)}"
if [ -z "${FIREWORKS_API_KEY}" ]; then
    echo "FIREWORKS_API_KEY is not set and ~/.rllm/config.json has no api_keys.fireworks (run: rllm model setup)" >&2
    exit 1
fi

export TERMINAL_SANDBOX_BACKEND="${TERMINAL_SANDBOX_BACKEND:-modal}"
export TB_TRAIN_DATASET="${TB_TRAIN_DATASET:-tb_v2_deepseek_v4_debug}"
export MINISWE_MAX_TURNS="${MINISWE_MAX_TURNS:-64}"
export MINISWE_MAX_CONSECUTIVE_FORMAT_ERRORS="${MINISWE_MAX_CONSECUTIVE_FORMAT_ERRORS:-1}"
export MINISWE_COMMAND_TIMEOUT="${MINISWE_COMMAND_TIMEOUT:-300}"
export RLLM_SANDBOX_MAX_CPUS="${RLLM_SANDBOX_MAX_CPUS:-0.125}"
export RLLM_SANDBOX_MAX_MEMORY_MB="${RLLM_SANDBOX_MAX_MEMORY_MB:-256}"
export RLLM_MODAL_SANDBOX_CREATE_RPS="${RLLM_MODAL_SANDBOX_CREATE_RPS:-2}"
export RLLM_HARNESS_INSTALL_TIMEOUT_S="${RLLM_HARNESS_INSTALL_TIMEOUT_S:-300}"
export RLLM_HARNESS_RUN_TIMEOUT_S="${RLLM_HARNESS_RUN_TIMEOUT_S:-2400}"
export RLLM_HARNESS_VERIFIER_TIMEOUT_S="${RLLM_HARNESS_VERIFIER_TIMEOUT_S:-300}"
export RLLM_SANDBOX_TIMEOUT_S="${RLLM_SANDBOX_TIMEOUT_S:-3000}"
python -u train_debug.py \
    rllm/backend=fireworks \
    model.name=accounts/fireworks/models/deepseek-v4-flash-0731 \
    model.tokenizer_model=deepseek-ai/DeepSeek-V4-Flash-0731 \
    model.lora_rank=128 \
    fireworks_config.policy_trainer_shape_id=accounts/fireworks/trainingShapes/deepseek-v4-flash-0731-256k-lora \
    fireworks_config.policy_trainer_replica_count=2 \
    fireworks_config.rollout_deployment_replica_count=4  \
    training.group_size=16 \
    training.learning_rate=5e-5 \
    training.grad_clip_norm=0.0 \
    training.beta2=0.999 \
    training.eps=1e-10 \
    training.max_length=67584 \
    rllm.rollout.train.temperature=1.0 \
    rllm.rollout.train.top_p=1.0 \
    rllm.rollout.val.temperature=1.0 \
    rllm.rollout.val.top_p=1.0 \
    rllm.data.max_prompt_length=67584 \
    rllm.data.max_response_length=16384 \
    rllm.data.train_batch_size=1 \
    rllm.data.val_batch_size=-1 \
    rllm.compact_filtering.enable=false \
    rllm.algorithm.adv_estimator=grpo \
    rllm.algorithm.norm_adv_by_std_in_grpo=false \
    rllm.algorithm.router_replay=R3 \
    rllm.algorithm.loss_fn=dppo_tv \
    +rllm.algorithm.loss_params='{delta: 0.1}' \
    rllm.algorithm.loss_agg_mode=token-mean \
    rllm.algorithm.rollout_correction.bypass_mode=true \
    rllm.async_training.enable=true \
    rllm.async_training.mini_batch_size=8 \
    rllm.async_training.fwd_bwd_group_size=8 \
    rllm.async_training.staleness_threshold=3.0 \
    rllm.async_training.trigger_parameter_sync_step=1 \
    rllm.async_training.partial_rollout=true \
    rllm.workflow.n_parallel_tasks=384 \
    rllm.workflow.raise_on_error=false \
    rllm.workflow.verify_only_on_env_done=true \
    rllm.rejection_sample.filter_uniform_groups=true \
    rllm.rejection_sample.refill_filtered_uniform_groups=true \
    rllm.gateway.tunnel=http://5.78.144.17:19090 \
    rllm.gateway.port=9200 \
    rllm.gateway.num_workers=4 \
    rllm.gateway.cumulative_token_mode=true \
    rllm.gateway.renderer_family=auto \
    rllm.trainer.total_epochs=1000 \
    rllm.episode_logging.log_episodes=true \
    rllm.trainer.logger='[wandb]' \
    rllm.trainer.project_name='terminal-rl' \
    rllm.trainer.experiment_name='deepseek-v4-flash-0731-lora-rank-128-lr-5e-5-eps-1e-10-tb-v2-debug' \
    rllm.trainer.val_before_train=false \
    rllm.trainer.test_freq=-1 \
    rllm.trainer.save_freq=20 \
    "$@"
