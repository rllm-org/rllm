#!/usr/bin/env bash
#
# GLM-5.2 Terminal-Bench RL launcher for the Fireworks backend.
#
# Usage:
#   train_fireworks_glm5p2.sh <lora|full> <opencode|terminus-2> <debug|train|sanity|production>
#
# The debug phase uses the eight-task tb_v2_debug split, one optimizer batch,
# and two Terminal-Bench 2.0 validation tasks. The train phase uses the full
# tb-opus-pass training split and the complete Terminal-Bench 2.0 validation
# split. The sanity and production phases train on tb-opus-pass/train and use
# the pinned 89-task Terminal-Bench 2.1 boundary benchmark. Sanity is a one-step
# LoRA + OpenCode smoke test: it evaluates the base checkpoint, skips the
# 20-minute mid-test, and stops after one optimizer batch. Production is
# full-parameter + OpenCode with the complete 89-task suite at step 0, every
# ten optimizer steps, and final weights. Production uses one validation suite
# instead of also scheduling the same tasks as a boundary benchmark.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

mode="${1:?usage: $0 <lora|full> <opencode|terminus-2> <debug|train|sanity|production>}"
harness="${2:?usage: $0 <lora|full> <opencode|terminus-2> <debug|train|sanity|production>}"
phase="${3:?usage: $0 <lora|full> <opencode|terminus-2> <debug|train|sanity|production>}"

case "$mode" in
    lora)
        lora_rank=128
        shape_id="accounts/fireworks/trainingShapes/glm-5p2-200k-lora"
        learning_rate="${TB_LORA_LEARNING_RATE:-2e-5}"
        ;;
    full)
        lora_rank=0
        shape_id="accounts/fireworks/trainingShapes/glm-5p2-200k"
        learning_rate="${TB_FULL_LEARNING_RATE:-1e-6}"
        ;;
    *)
        echo "unsupported mode '$mode' (expected lora or full)" >&2
        exit 2
        ;;
esac

case "$harness" in
    opencode|terminus-2) ;;
    *)
        echo "unsupported harness '$harness' (expected opencode or terminus-2)" >&2
        exit 2
        ;;
esac

case "$phase" in
    debug)
        train_dataset="tb_v2_debug"
        train_split="train"
        val_dataset="terminal-bench@2.0"
        val_split="default"
        benchmark_dataset=""
        benchmark_split="default"
        benchmark_expected_tasks=0
        benchmark_before_train=false
        benchmark_after_train=false
        val_max="${TB_DEBUG_VAL_MAX:-2}"
        val_expected_tasks=0
        total_batches="${TB_DEBUG_TOTAL_BATCHES:-1}"
        total_epochs=1
        test_freq=1
        val_before_train=true
        trainer_replicas="${TB_TRAINER_REPLICAS:-1}"
        rollout_replicas="${TB_ROLLOUT_REPLICAS:-1}"
        n_parallel_tasks="${TB_DEBUG_N_PARALLEL_TASKS:-16}"
        async_mini_batch_size="${TB_DEBUG_ASYNC_MINI_BATCH_SIZE:-1}"
        ;;
    train)
        train_dataset="tb-opus-pass"
        train_split="train"
        val_dataset="terminal-bench@2.0"
        val_split="default"
        benchmark_dataset=""
        benchmark_split="default"
        benchmark_expected_tasks=0
        benchmark_before_train=false
        benchmark_after_train=false
        val_max=0
        val_expected_tasks=0
        total_batches=-1
        total_epochs=1
        test_freq=50
        val_before_train=false
        trainer_replicas="${TB_TRAINER_REPLICAS:-1}"
        rollout_replicas="${TB_ROLLOUT_REPLICAS:-1}"
        # Keep local Docker sandbox creation aligned with the rollout engine's
        # default 64-way concurrency. Higher values mostly pre-create queued
        # containers and can overload a shared Docker host when four matrix
        # runs launch together.
        n_parallel_tasks="${TB_TRAIN_N_PARALLEL_TASKS:-64}"
        async_mini_batch_size="${TB_TRAIN_ASYNC_MINI_BATCH_SIZE:-8}"
        ;;
    sanity|production)
        if [ "$harness" != "opencode" ]; then
            echo "$phase phase requires the OpenCode harness" >&2
            exit 2
        fi
        if [ "$phase" = "sanity" ] && [ "$mode" != "lora" ]; then
            echo "sanity phase requires: lora opencode sanity" >&2
            exit 2
        fi
        if [ "$phase" = "production" ] && [ "$mode" != "full" ]; then
            echo "production phase requires: full opencode production" >&2
            exit 2
        fi
        train_dataset="tb-opus-pass"
        train_split="train"
        val_dataset="terminal-bench@2.1"
        benchmark_split="default"
        val_max=0
        total_batches=-1
        total_epochs=1
        trainer_replicas="${TB_TRAINER_REPLICAS:-4}"
        if [ "$phase" = "sanity" ]; then
            val_split="midtest"
            val_expected_tasks=0
            benchmark_dataset="terminal-bench@2.1"
            benchmark_expected_tasks=89
            benchmark_before_train=true
            # The step-0 boundary benchmark already exercises the evaluation
            # stack. Keep this smoke test to one LoRA optimizer batch and avoid
            # another ~20 minutes for the periodic mid-test.
            test_freq=-1
            val_before_train=false
            benchmark_after_train=false
            total_batches="${TB_SANITY_TOTAL_BATCHES:-1}"
            rollout_replicas="${TB_ROLLOUT_REPLICAS:-4}"
        else
            # Use the full suite as validation so step 0 and every tenth step
            # share one metric namespace. Disable the separate benchmark path
            # to avoid evaluating the same tasks twice at step 0. The trainer's
            # final-validation hook still evaluates this suite at final weights.
            val_split="default"
            val_expected_tasks=89
            benchmark_dataset=""
            benchmark_expected_tasks=0
            benchmark_before_train=false
            benchmark_after_train=false
            test_freq=10
            val_before_train=true
            rollout_replicas="${TB_ROLLOUT_REPLICAS:-12}"
        fi
        n_parallel_tasks="${TB_TRAIN_N_PARALLEL_TASKS:-64}"
        async_mini_batch_size="${TB_TRAIN_ASYNC_MINI_BATCH_SIZE:-8}"
        ;;
    *)
        echo "unsupported phase '$phase' (expected debug, train, sanity, or production)" >&2
        exit 2
        ;;
esac

if [ -z "${FIREWORKS_API_KEY:-}" ]; then
    echo "FIREWORKS_API_KEY is required" >&2
    exit 1
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WANDB_API_KEY is required" >&2
    exit 1
fi

run_stamp="${TB_RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
comparison="${TB_COMPARISON_ID:-glm5p2-tb-${run_stamp}}"
run_name="${TB_RUN_NAME:-${comparison}-${phase}-${mode}-${harness}}"
gateway_port="${RLLM_GATEWAY_PORT:-9200}"
python_bin="${RLLM_PYTHON:-python}"
state_root="${TB_STATE_ROOT:-${HOME}/.rllm/glm5p2-terminal-rl}"
trainer_region="${TB_TRAINER_REGION:-}"
stamp_slug="${run_stamp,,}"
stamp_slug="${stamp_slug//[^a-z0-9-]/-}"
stamp_slug="${stamp_slug:0:20}"
deployment_nonce="$(tr -d '-' </proc/sys/kernel/random/uuid | cut -c1-10)"
deployment_id="${TB_DEPLOYMENT_ID:-tb-glm5p2-${phase}-${mode}-${harness}-${stamp_slug}-${deployment_nonce}}"

export TERMINAL_SANDBOX_BACKEND="${TERMINAL_SANDBOX_BACKEND:-docker}"
export TB_HARNESS="$harness"
export TB_TRAIN_DATASET="$train_dataset"
export TB_TRAIN_SPLIT="$train_split"
export TB_VAL_DATASET="$val_dataset"
export TB_VAL_SPLIT="$val_split"
export TB_VAL_MAX="$val_max"
export TB_VAL_EXPECTED_TASKS="$val_expected_tasks"
export TB_BENCHMARK_DATASET="$benchmark_dataset"
export TB_BENCHMARK_SPLIT="$benchmark_split"
export TB_BENCHMARK_EXPECTED_TASKS="$benchmark_expected_tasks"
export TERMINUS_MAX_TURNS="${TERMINUS_MAX_TURNS:-100}"
export TERMINUS_ENABLE_SUMMARIZE="${TERMINUS_ENABLE_SUMMARIZE:-0}"
export RLLM_HARNESS_INSTALL_TIMEOUT_S="${RLLM_HARNESS_INSTALL_TIMEOUT_S:-900}"
export RLLM_HARNESS_RUN_TIMEOUT_S="${RLLM_HARNESS_RUN_TIMEOUT_S:-2400}"
export RLLM_HARNESS_VERIFIER_TIMEOUT_S="${RLLM_HARNESS_VERIFIER_TIMEOUT_S:-300}"
export RLLM_SANDBOX_TIMEOUT_S="${RLLM_SANDBOX_TIMEOUT_S:-3000}"
export RLLM_HOME="${RLLM_HOME:-${state_root}/state}"
export HF_HOME="${HF_HOME:-${state_root}/hf-home}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${state_root}/uv-cache}"
export WANDB_MODE=online
export WANDB_DIR="${WANDB_DIR:-${state_root}/wandb}"
export WANDB_RUN_GROUP="$comparison"
export WANDB_JOB_TYPE="${phase}-${mode}-${harness}"
export WANDB_TAGS="terminal-bench,glm-5.2,${mode},${harness},${phase},${trainer_region:-auto-region}"

mkdir -p "$RLLM_HOME" "$WANDB_DIR" "${state_root}/logs"

printf 'run=%s mode=%s harness=%s phase=%s train=%s/%s val=%s/%s benchmark=%s region=%s trainer_replicas=%s rollout_replicas=%s gateway_port=%s deployment=%s async_mini_batch_size=%s\n' \
    "$run_name" "$mode" "$harness" "$phase" "$train_dataset" "$train_split" \
    "$val_dataset" "$val_split" "${benchmark_dataset:-disabled}" "${trainer_region:-control-plane-selected}" \
    "$trainer_replicas" "$rollout_replicas" "$gateway_port" "$deployment_id" "$async_mini_batch_size"

region_override=()
if [ -n "$trainer_region" ]; then
    region_override+=("fireworks_infra.trainers.policy.region=$trainer_region")
fi

exec "$python_bin" -u train.py \
    rllm/backend=fireworks \
    model.name=accounts/fireworks/models/glm-5p2-fp8 \
    model.tokenizer_model=zai-org/GLM-5.2 \
    model.lora_rank="$lora_rank" \
    fireworks_config.policy_trainer_shape_id="$shape_id" \
    fireworks_config.policy_trainer_replica_count="$trainer_replicas" \
    fireworks_config.rollout_deployment_replica_count="$rollout_replicas" \
    fireworks_infra.deployments.rollout.deployment_id="$deployment_id" \
    "${region_override[@]}" \
    training.group_size=16 \
    training.learning_rate="$learning_rate" \
    training.beta2=0.999 \
    training.max_length=67584 \
    rllm.rollout.train.temperature=1.0 \
    rllm.rollout.train.top_p=1.0 \
    rllm.rollout.val.temperature=1.0 \
    rllm.rollout.val.top_p=1.0 \
    rllm.data.max_prompt_length=51200 \
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
    rllm.async_training.mini_batch_size="$async_mini_batch_size" \
    rllm.async_training.fwd_bwd_group_size=1 \
    rllm.async_training.staleness_threshold=3.0 \
    rllm.async_training.trigger_parameter_sync_step=1 \
    rllm.async_training.partial_rollout=true \
    rllm.workflow.n_parallel_tasks="$n_parallel_tasks" \
    rllm.workflow.raise_on_error=false \
    rllm.rejection_sample.filter_uniform_groups=true \
    rllm.gateway.port="$gateway_port" \
    rllm.gateway.num_workers=4 \
    rllm.gateway.cumulative_token_mode=true \
    rllm.gateway.renderer_family=auto \
    rllm.trainer.total_epochs="$total_epochs" \
    rllm.trainer.total_batches="$total_batches" \
    rllm.trainer.logger='[console,wandb]' \
    rllm.trainer.project_name=terminal-rl \
    rllm.trainer.experiment_name="$run_name" \
    rllm.trainer.val_before_train="$val_before_train" \
    rllm.trainer.benchmark_before_train="$benchmark_before_train" \
    rllm.trainer.benchmark_after_train="$benchmark_after_train" \
    rllm.trainer.test_freq="$test_freq" \
    rllm.trainer.save_freq=20 \
    "${@:4}"
