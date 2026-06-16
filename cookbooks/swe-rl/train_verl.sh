#!/usr/bin/env bash
# Train an SWE agent on R2E-Gym with the verl (distributed) backend.
#
# Prerequisites:
#   1. Install rllm with verl extras:     uv pip install -e ".[verl]"
#   2. Install megatron:                   bash scripts/install_megatron.sh <cu128|cu129|...>
#   3. Install this cookbook:              uv pip install --no-deps -e cookbooks/swe-rl
#   4. Pull the datasets:                  python cookbooks/swe-rl/prepare_data.py
#
# Verl runs vLLM rollouts + FSDP/Megatron training across N GPUs. Mini-swe-agent
# still executes inside per-task sandboxes via rLLM's own SandboxedAgentFlow path
# (AgentFlowEngine, not the remote Harbor runtime); only the policy rollout
# traffic flows through the gateway back to the verl-hosted vLLM engine.
#
# Sandbox backend is chosen by SWE_SANDBOX_BACKEND (docker | local | modal |
# daytona; default modal). modal needs `pip install modal` + `modal token new`;
# daytona needs `pip install daytona` + DAYTONA_API_KEY. Per-rollout agent
# timeout: RLLM_HARNESS_RUN_TIMEOUT_S.

set -euo pipefail

unset ROCR_VISIBLE_DEVICES 2>/dev/null || true

export SWE_SANDBOX_BACKEND="${SWE_SANDBOX_BACKEND:-modal}"
export RLLM_HARNESS_RUN_TIMEOUT_S="${RLLM_HARNESS_RUN_TIMEOUT_S:-1800}"

MODEL_PATH=Qwen/Qwen3.5-9B

python -u train.py \
    rllm/backend=verl \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=true \
    rllm.algorithm.use_rllm=true \
    data.train_batch_size=16 \
    data.val_batch_size=-1 \
    data.max_prompt_length=32768 \
    data.max_response_length=8192 \
    +model.name=$MODEL_PATH \
    actor_rollout_ref.model.path=$MODEL_PATH \
    +actor_rollout_ref.model.lora.rank=32 \
    +actor_rollout_ref.model.lora.alpha=32 \
    +actor_rollout_ref.model.lora.merge=true \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.enforce_eager=False \
    +actor_rollout_ref.rollout.max_model_len=40960 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    rllm.workflow.n_parallel_tasks=64 \
    rllm.workflow.raise_on_error=false \
    rllm.gateway.port=9090 \
    trainer.logger="['console','wandb']" \
    trainer.project_name=swe-rl \
    trainer.experiment_name=r2egym-mini-swe-agent-qwen3.5-9b-verl \
    trainer.val_before_train=true \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=20 \
    trainer.total_epochs=1 \
    trainer.default_hdfs_dir=null \
    trainer.resume_mode=disable \
    "$@"
