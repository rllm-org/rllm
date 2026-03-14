#!/bin/bash
set -x

# Solver-judge workflow training on Modal with rLLM unified trainer + SkyRL backend.
#
# Usage:
#   modal run rllm/experimental/test/modal_run.py
#   modal run rllm/experimental/test/modal_run.py \
#       --command "bash rllm/experimental/test/run_skyrl_solver_judge_modal.sh trainer.train_batch_size=128"

: "${NUM_GPUS:=2}"
: "${LOGGER:=wandb,ui}"  # options: wandb, console, ui

LOGGER_HYDRA="$LOGGER"
if [[ "$LOGGER_HYDRA" != \[*\] ]]; then
    LOGGER_HYDRA="[$(printf '%s' "$LOGGER_HYDRA" | tr -d '[:space:]')]"
fi

cd /root/rllm

python3 -m rllm.experimental.test.test_skyrl_solver_judge \
    rllm/backend=skyrl \
    trainer.policy.model.path=Qwen/Qwen3-0.6B \
    trainer.strategy=fsdp2 \
    trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
    trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
    trainer.placement.colocate_all=true \
    generator.num_inference_engines=$NUM_GPUS \
    generator.inference_engine_tensor_parallel_size=1 \
    generator.n_samples_per_prompt=5 \
    generator.eval_n_samples_per_prompt=1 \
    generator.backend=vllm \
    generator.run_engines_locally=true \
    generator.weight_sync_backend=nccl \
    generator.async_engine=true \
    generator.batched=true \
    generator.gpu_memory_utilization=0.8 \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    generator.sampling_params.max_generate_length=1024 \
    trainer.epochs=10 \
    trainer.eval_before_train=false \
    trainer.eval_interval=5 \
    trainer.ckpt_interval=10 \
    trainer.ckpt_path=/root/data/ckpts/solver_judge \
    trainer.export_path=/root/data/export/solver_judge \
    trainer.project_name=rllm-solver-judge \
    trainer.run_name=skyrl-countdown \
    trainer.policy.optimizer_config.lr=1e-6 \
    trainer.algorithm.use_kl_loss=false \
    rllm.episode_logging.log_episodes=true \
    rllm.rejection_sample.min_trajs_per_group=1 \
    trainer.logger="$LOGGER_HYDRA" \
    "$@"
