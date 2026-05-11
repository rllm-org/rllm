#!/usr/bin/env bash
# SWE veRL Training — Qwen3.5-9B (dense), Megatron-Core (TP=2 CP=2 PP=1), hard test
# 1:1 port of run_swe_training_9b.sh (FSDP2) except for parallelism.
#
# Env knobs:
#   NNODES=1|2           (default 2)
#   NGPUS_PER_NODE=8     (default 8)
#   ACTOR_TP=2 ACTOR_CP=2 ACTOR_PP=1 ROLLOUT_TP=1  (Megatron parallelism)
#   ACTOR_LR_WARMUP_STEPS=20  (set to 0 for one-step smoke tests)
#   ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU=1  (B200 full-token safe default)
#   RLLM_RAY_MASTER_PORT_RANGE=25000:25100  (avoid ephemeral-port collisions)
#   RLLM_VLLM_PORT_BASE=46000 RLLM_VLLM_PORT_STRIDE=100  (per-rollout vLLM ports)
#   LOGGER='[console, wandb]' or '[console]'  (default "[console, wandb]")
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COOKBOOK_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RLLM_REPO_ROOT="$(cd "$COOKBOOK_DIR/../.." && pwd)"
cd "$COOKBOOK_DIR"

VENV_ROOT="${VENV_ROOT:-$COOKBOOK_DIR/.venv-verl-vllm018}"
VENV_DIR="${VENV_DIR:-$VENV_ROOT/.venv}"

if [ -n "${VIRTUAL_ENV:-}" ] && [ -d "$VIRTUAL_ENV/bin" ]; then
    :
elif [ -d "$VENV_DIR/bin" ]; then
    source "$VENV_DIR/bin/activate"
elif [ -d /tmp/verl_venv/bin ]; then
    # Copied venvs keep absolute paths in activate scripts; set these directly.
    export VIRTUAL_ENV=/tmp/verl_venv
    export PATH="$VIRTUAL_ENV/bin:$PATH"
elif [ -d "$RLLM_REPO_ROOT/verl/.venv/bin" ]; then
    source "$RLLM_REPO_ROOT/verl/.venv/bin/activate"
else
    echo "No verl virtualenv found. Activate one, set VENV_DIR, or run cookbooks/swe/scripts/setup_verl_vllm018_qwen35.sh." >&2
    exit 1
fi

if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    CLEAN_LD_LIBRARY_PATH=""
    IFS=":" read -r -a LD_LIBRARY_PATH_PARTS <<< "$LD_LIBRARY_PATH"
    for LD_LIBRARY_PATH_PART in "${LD_LIBRARY_PATH_PARTS[@]}"; do
        if [ -z "$LD_LIBRARY_PATH_PART" ]; then
            continue
        fi
        case ":$CLEAN_LD_LIBRARY_PATH:" in
            *":$LD_LIBRARY_PATH_PART:"*) ;;
            *) CLEAN_LD_LIBRARY_PATH="${CLEAN_LD_LIBRARY_PATH:+$CLEAN_LD_LIBRARY_PATH:}$LD_LIBRARY_PATH_PART" ;;
        esac
    done
    export LD_LIBRARY_PATH="$CLEAN_LD_LIBRARY_PATH"
fi
if [ -d "$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia" ]; then
    NVIDIA_PKG_ROOT="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia"
    export NVRTC_HOME="${NVRTC_HOME:-$NVIDIA_PKG_ROOT/cuda_nvrtc}"
    export CUBLAS_HOME="${CUBLAS_HOME:-$NVIDIA_PKG_ROOT/cublas}"
    export CUDNN_HOME="${CUDNN_HOME:-$NVIDIA_PKG_ROOT/cudnn}"
    export CURAND_HOME="${CURAND_HOME:-$NVIDIA_PKG_ROOT/curand}"
    export CUFFT_HOME="${CUFFT_HOME:-$NVIDIA_PKG_ROOT/cufft}"
    export CUSOLVER_HOME="${CUSOLVER_HOME:-$NVIDIA_PKG_ROOT/cusolver}"
    export CUSPARSE_HOME="${CUSPARSE_HOME:-$NVIDIA_PKG_ROOT/cusparse}"
    NVIDIA_SITE_LIBS=$(find "$NVIDIA_PKG_ROOT" -name "lib" -type d | paste -sd: -)
    export LD_LIBRARY_PATH="${NVIDIA_SITE_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# ---- CUDA driver library selection ----
# Some B200 images put CUDA compat libcuda ahead of the real driver in
# ldconfig. Torch 2.10+cu129 fails with CUDA error 803 in that state, so prefer
# the worker's real driver library when it is available.
REAL_LIBCUDA=/usr/lib/x86_64-linux-gnu/libcuda.so.1
REAL_NVML_LIB=""
for f in /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.*; do
    if [ -f "$f" ] && [ -s "$f" ] && [ ! -L "$f" ]; then
        REAL_NVML_LIB="$f"
        break
    fi
done
if [ -f "$REAL_LIBCUDA" ] && [ -s "$REAL_LIBCUDA" ]; then
    if [ -n "$REAL_NVML_LIB" ]; then
        export LD_PRELOAD="$REAL_LIBCUDA:$REAL_NVML_LIB${LD_PRELOAD:+:$LD_PRELOAD}"
    else
        export LD_PRELOAD="$REAL_LIBCUDA${LD_PRELOAD:+:$LD_PRELOAD}"
    fi
    export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
fi

# ---- CUDA Forward Compat fallback ----
if [ ! -f "$REAL_LIBCUDA" ] && { [ -z "${LD_LIBRARY_PATH:-}" ] || [[ ! "$LD_LIBRARY_PATH" == *"cuda-compat"* ]]; }; then
    COMPAT=/tmp/cuda-compat-13/usr/local/cuda-13.0/compat
    if [ -f "$COMPAT/libcuda.so" ]; then
        export CUDA_HOME=$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13
        export PATH=$CUDA_HOME/bin:$CUDA_HOME/nvvm/bin:$PATH
        NVIDIA_LIBS=$(find "$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia" -name "lib" -type d | tr "\n" ":")
        export LD_LIBRARY_PATH="${COMPAT}:${NVIDIA_LIBS}/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
    fi
fi

# ---- Environment ----
export RAY_ADDRESS=${RAY_ADDRESS:-auto}
export RLLM_RUN_TASK_RUNNER_LOCAL=${RLLM_RUN_TASK_RUNNER_LOCAL:-1}
export RAY_BACKEND_LOG_LEVEL=${RAY_BACKEND_LOG_LEVEL:-info}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
export RAY_LOG_TO_DRIVER=${RAY_LOG_TO_DRIVER:-1}
export RAY_CLIENT_RECONNECT_GRACE_PERIOD=${RAY_CLIENT_RECONNECT_GRACE_PERIOD:-300}
export RAY_CLIENT_MAX_CONNECTION_TIMEOUT_S=${RAY_CLIENT_MAX_CONNECTION_TIMEOUT_S:-60}
export RAY_PREFLIGHT_TIMEOUT_S=${RAY_PREFLIGHT_TIMEOUT_S:-90}
export RLLM_RAY_MASTER_PORT_RANGE=${RLLM_RAY_MASTER_PORT_RANGE:-25000:25100}
export RLLM_VLLM_PORT_BASE=${RLLM_VLLM_PORT_BASE:-46000}
export RLLM_VLLM_PORT_STRIDE=${RLLM_VLLM_PORT_STRIDE:-100}
export WANDB_MODE=${WANDB_MODE:-online}
export HF_HOME=${HF_HOME:-/tmp/hf_cache}
export RLLM_GATEWAY_HEALTH_TIMEOUT=${RLLM_GATEWAY_HEALTH_TIMEOUT:-180}
export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-INFO}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}

# transformers 5.3 _patch_mistral_regex calls HF /api/models on every tokenizer
# load. 8 parallel vLLM EngineCore workers on anon IP triggers 429 and an
# unhelpful "Unable to load vocabulary" error. Forcing offline mode makes
# is_offline_mode()=True which short-circuits the API call.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Expose the HF cache at the default location for Ray actors that don't
# inherit HF_HOME.
mkdir -p ~/.cache/huggingface/hub
if [ ! -e ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B ] \
    && [ -d /tmp/hf_cache/hub/models--Qwen--Qwen3.5-9B ]; then
    ln -sf /tmp/hf_cache/hub/models--Qwen--Qwen3.5-9B \
        ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B
fi

export NCCL_CUMEM_ENABLE=0
# verl propagates the driver's NCCL env to every actor via ray
# runtime_env.env_vars (see rllm/trainer/verl/ray_runtime_env.py),
# so a wrong NCCL_SOCKET_IFNAME here will poison every worker. Auto-detect by
# probing which interface owns a global IPv6 — bond0 on the older B200 fleet,
# eth0 on the newer one. This overrides any inherited value from the shell.
if ip -6 addr show bond0 2>/dev/null | grep -q 'inet6.*global'; then
    export NCCL_SOCKET_IFNAME=bond0
elif ip -6 addr show eth0 2>/dev/null | grep -q 'inet6.*global'; then
    export NCCL_SOCKET_IFNAME=eth0
fi
export NCCL_SOCKET_FAMILY=AF_INET6
echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-<unset>}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TRITON_CACHE_DIR=/tmp/triton_cache
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/xdg_cache}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/tmp/torch_extensions}
export PYTHONPATH="$COOKBOOK_DIR:$RLLM_REPO_ROOT:${PYTHONPATH:-}"
mkdir -p "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR"

# Required for Megatron comm/compute overlap
export CUDA_DEVICE_MAX_CONNECTIONS=1

MODEL_REPO=${MODEL_REPO:-Qwen/Qwen3.5-9B}
MODEL_CACHE_DIR=${MODEL_CACHE_DIR:-/tmp/hf_cache/hub/models--Qwen--Qwen3.5-9B}
if [ -z "${MODEL_PATH:-}" ]; then
    if [ -d "$MODEL_CACHE_DIR/snapshots" ]; then
        MODEL_PATH=$(find -L "$MODEL_CACHE_DIR/snapshots" -mindepth 1 -maxdepth 1 -type d -print -quit)
    else
        MODEL_PATH=""
    fi
fi
if [ -z "$MODEL_PATH" ] || [ ! -d "$MODEL_PATH" ]; then
    echo "Missing local model snapshot. Expected one under: $MODEL_CACHE_DIR/snapshots" >&2
    echo "Set MODEL_PATH explicitly, or stage $MODEL_REPO into /tmp/hf_cache on each Ray node." >&2
    exit 1
fi
MODEL_NAME="$MODEL_PATH"

if [[ "$RAY_ADDRESS" == ray://* ]]; then
    RAY_STATUS_LINE=$(
        timeout "$RAY_PREFLIGHT_TIMEOUT_S" python -c 'import os, ray; ray.init(address=os.environ["RAY_ADDRESS"]); print(ray.cluster_resources()); ray.shutdown()' 2>&1 | tail -1
    ) || RAY_STATUS_LINE="<unavailable>"
else
    RAY_STATUS_LINE=$(ray status 2>&1 | head -1) || RAY_STATUS_LINE="<unavailable>"
fi

# Parallelism for the two-node 16xB200 setup. Qwen3.5 context parallelism
# requires Megatron Gated DeltaNet THD support and remove-padding enabled.
ACTOR_TP=${ACTOR_TP:-2}
ACTOR_CP=${ACTOR_CP:-2}
ACTOR_PP=${ACTOR_PP:-1}
ROLLOUT_TP=${ROLLOUT_TP:-1}
MODEL_USE_REMOVE_PADDING=${MODEL_USE_REMOVE_PADDING:-true}
NNODES=${NNODES:-2}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
ACTOR_LR_WARMUP_STEPS=${ACTOR_LR_WARMUP_STEPS:-20}
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU=${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LOGGER=${LOGGER:-"[console,wandb]"}
SWE_STEP_LIMIT=${SWE_STEP_LIMIT:-200}
SWE_AGENT_TIMEOUT=${SWE_AGENT_TIMEOUT:-360}
SWE_COMMAND_TIMEOUT=${SWE_COMMAND_TIMEOUT:-120}
SWE_SANDBOX_TIMEOUT=${SWE_SANDBOX_TIMEOUT:-480}
SWE_STARTUP_JITTER_S=${SWE_STARTUP_JITTER_S:-25.0}
SWE_VAL_STEP_LIMIT=${SWE_VAL_STEP_LIMIT:-300}
SWE_VAL_AGENT_TIMEOUT=${SWE_VAL_AGENT_TIMEOUT:-900}
SWE_VAL_COMMAND_TIMEOUT=${SWE_VAL_COMMAND_TIMEOUT:-$SWE_COMMAND_TIMEOUT}
SWE_VAL_SANDBOX_TIMEOUT=${SWE_VAL_SANDBOX_TIMEOUT:-1020}
SWE_VAL_STARTUP_JITTER_S=${SWE_VAL_STARTUP_JITTER_S:-30}
MODEL_MAX_TOKENS=${MODEL_MAX_TOKENS:-4096}
DEFAULT_TRAJ_DIR="$COOKBOOK_DIR/trajectories/\${trainer.experiment_name}"
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/tmp/turing-swe/checkpoints}
MAIN_CHECKPOINT_DIR=${MAIN_CHECKPOINT_DIR:-$CHECKPOINT_ROOT/swe-verl-9b-megatron/qwen35-9b-swe-smith}
LORA_CHECKPOINT_DIR=${LORA_CHECKPOINT_DIR:-$CHECKPOINT_ROOT/swe-verl-9b-megatron-lora/qwen35-9b-swe-smith-lora-r16}
TRAJECTORY_OUTPUT_DIR=${TRAJ_DIR:-$DEFAULT_TRAJ_DIR}
ROLLOUT_CORRECTION_BYPASS=${ROLLOUT_CORRECTION_BYPASS:-true}
ROLLOUT_CORRECTION_IS=${ROLLOUT_CORRECTION_IS:-null}
ROLLOUT_CORRECTION_IS_THRESHOLD=${ROLLOUT_CORRECTION_IS_THRESHOLD:-2.0}

LORA_ARGS=(
    actor_rollout_ref.model.lora.type=lora
    actor_rollout_ref.model.lora.merge=false
    actor_rollout_ref.model.lora.rank=16
    actor_rollout_ref.model.lora.alpha=32
    'actor_rollout_ref.model.lora.target_modules=[language_model.decoder.layers.*.self_attention.linear_qkv,language_model.decoder.layers.*.self_attention.linear_proj,language_model.decoder.layers.*.mlp.linear_fc1,language_model.decoder.layers.*.mlp.linear_fc2]'
    actor_rollout_ref.model.lora.lora_A_init_method=kaiming
    actor_rollout_ref.actor.megatron.use_mbridge=true
    actor_rollout_ref.actor.megatron.vanilla_mbridge=false
    actor_rollout_ref.ref.megatron.vanilla_mbridge=false
    trainer.project_name=swe-verl-9b-megatron-lora
    trainer.experiment_name=qwen35-9b-swe-smith-megatron-lora-r16
    trainer.default_local_dir="$LORA_CHECKPOINT_DIR"
    ++trainer.hf_repo_id=JWei05/qwen35-9b-swe-smith-megatron-lora-r16
)

echo "============================================================"
echo "SWE veRL Training — Qwen3.5-9B (Megatron TP=${ACTOR_TP} CP=${ACTOR_CP} PP=${ACTOR_PP}) hard TEST"
echo "============================================================"
echo "Model:    $MODEL_NAME"
echo "Repo:     $MODEL_REPO"
echo "Dataset:  swe_smith_filtered_mix"
echo "Topology: ${NNODES} node(s) × ${NGPUS_PER_NODE} GPU  |  Megatron TP=${ACTOR_TP} CP=${ACTOR_CP} PP=${ACTOR_PP}  |  vLLM TP=${ROLLOUT_TP}"
echo "Actor:    ppo_micro_batch_size_per_gpu=${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU} lr_warmup_steps=${ACTOR_LR_WARMUP_STEPS}"
echo "SWE:      train steps=${SWE_STEP_LIMIT} timeout=${SWE_AGENT_TIMEOUT}s cmd=${SWE_COMMAND_TIMEOUT}s sandbox=${SWE_SANDBOX_TIMEOUT}s jitter=${SWE_STARTUP_JITTER_S}s max_tokens=${MODEL_MAX_TOKENS}"
echo "SWE val:  steps=${SWE_VAL_STEP_LIMIT} timeout=${SWE_VAL_AGENT_TIMEOUT}s cmd=${SWE_VAL_COMMAND_TIMEOUT}s sandbox=${SWE_VAL_SANDBOX_TIMEOUT}s jitter=${SWE_VAL_STARTUP_JITTER_S}s"
echo "Ckpt:     $MAIN_CHECKPOINT_DIR"
echo "Traj:     save=${SAVE_TRAJ:-false} dir=${TRAJECTORY_OUTPUT_DIR}"
echo "Logger:   $LOGGER"
echo "Ray:      $RAY_STATUS_LINE"
echo "Debug:    RAY_DEDUP_LOGS=$RAY_DEDUP_LOGS RAY_LOG_TO_DRIVER=$RAY_LOG_TO_DRIVER RAY_CLIENT_RECONNECT_GRACE_PERIOD=$RAY_CLIENT_RECONNECT_GRACE_PERIOD VERL_LOGGING_LEVEL=$VERL_LOGGING_LEVEL"
echo "============================================================"

python -u -m swe.scripts.train_swe_verl \
    --config-name=verl_swe_trainer \
    train_dataset=swe_smith_filtered_mix \
    val_dataset=swe_bench_multilingual \
    train_max_samples=1500 \
    actor_rollout_ref.model.path="$MODEL_NAME" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=${MODEL_USE_REMOVE_PADDING} \
    +model.name="$MODEL_NAME" \
    \
    data.train_batch_size=16 \
    data.gen_batch_size=16 \
    data.max_prompt_length=30720 \
    data.max_response_length=32768 \
    data.filter_overlong_prompts=true \
    data.filter_overlong_prompts_workers=8 \
    \
    ++trainer.use_legacy_worker_impl=disable \
    actor_rollout_ref.actor.strategy=megatron \
    actor_rollout_ref.actor.megatron.use_mbridge=true \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=false \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${ACTOR_CP} \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.calculate_per_token_loss=true \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.mtp_num_layers=0 \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.sequence_parallel=true \
    actor_rollout_ref.actor.megatron.use_distributed_optimizer=true \
    actor_rollout_ref.actor.megatron.param_offload=true \
    actor_rollout_ref.actor.megatron.grad_offload=true \
    actor_rollout_ref.actor.megatron.optimizer_offload=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    ++actor_rollout_ref.actor.optim.lr_warmup_steps=${ACTOR_LR_WARMUP_STEPS} \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=k1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.calculate_entropy=true \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.actor.use_dynamic_bsz=false \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.model.use_fused_kernels=${USE_FUSED_KERNELS:-false} \
    actor_rollout_ref.rollout.enable_prefix_caching=false \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.max_model_len=32768 \
     actor_rollout_ref.rollout.multi_stage_wake_up=true \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=49152 \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=6144 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=true \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=qwen3_coder \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    ++actor_rollout_ref.ref.strategy=megatron \
    actor_rollout_ref.ref.megatron.use_mbridge=true \
    actor_rollout_ref.ref.megatron.vanilla_mbridge=false \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${ACTOR_TP} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${ACTOR_PP} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${ACTOR_CP} \
    actor_rollout_ref.ref.megatron.param_offload=true \
    \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=false \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    algorithm.rollout_correction.bypass_mode=${ROLLOUT_CORRECTION_BYPASS} \
    algorithm.rollout_correction.rollout_is=${ROLLOUT_CORRECTION_IS} \
    algorithm.rollout_correction.rollout_is_threshold=${ROLLOUT_CORRECTION_IS_THRESHOLD} \
    \
    ++rllm.gateway.db_path=/tmp/gateway_traces_9b_megatron.db \
    ++rllm.gateway.strip_vllm_fields=false \
    rllm.workflow.n_parallel_tasks=64 \
    rllm.workflow.retry_limit=1 \
    rllm.workflow.raise_on_error=false \
    rllm.algorithm.rollout_correction.bypass_mode=${ROLLOUT_CORRECTION_BYPASS} \
    rllm.algorithm.rollout_correction.tis_mode=${ROLLOUT_CORRECTION_IS} \
    rllm.algorithm.rollout_correction.tis_cap=${ROLLOUT_CORRECTION_IS_THRESHOLD} \
    \
    swe.save_trajectories=${SAVE_TRAJ:-false} \
    swe.trajectory_output_dir="${TRAJECTORY_OUTPUT_DIR}" \
    swe.step_limit=${SWE_STEP_LIMIT} \
    swe.agent_timeout=${SWE_AGENT_TIMEOUT} \
    swe.command_timeout=${SWE_COMMAND_TIMEOUT} \
    swe.sandbox_timeout=${SWE_SANDBOX_TIMEOUT} \
    +swe.val_step_limit=${SWE_VAL_STEP_LIMIT} \
    +swe.val_agent_timeout=${SWE_VAL_AGENT_TIMEOUT} \
    +swe.val_command_timeout=${SWE_VAL_COMMAND_TIMEOUT} \
    +swe.val_sandbox_timeout=${SWE_VAL_SANDBOX_TIMEOUT} \
    +swe.val_startup_jitter_s=${SWE_VAL_STARTUP_JITTER_S} \
    swe.verbose=true \
    +swe.compaction_enabled=true \
    +swe.compaction_token_trigger=25600 \
    +swe.compaction_keep_recent_turns=1 \
    +swe.startup_jitter_s=${SWE_STARTUP_JITTER_S} \
    +swe.sandbox_retry_backoff_min_s=5.0 \
    +swe.sandbox_retry_backoff_max_s=15.0 \
    +swe.model_max_tokens=${MODEL_MAX_TOKENS} \
    +swe.model_return_token_ids=true \
    \
    trainer.total_epochs=100 \
    trainer.save_freq=100 \
    trainer.test_freq=1000 \
    trainer.val_before_train=false \
    ++rllm.trainer.val_before_train=false \
    trainer.default_local_dir="$MAIN_CHECKPOINT_DIR" \
    +trainer.hf_repo_id=JWei05/qwen35-9b-swe-smith-megatron \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    "trainer.logger=${LOGGER}" \
    trainer.project_name=swe-verl-9b-megatron \
    trainer.experiment_name=qwen35-9b-swe-smith-megatron \
    "$@"
