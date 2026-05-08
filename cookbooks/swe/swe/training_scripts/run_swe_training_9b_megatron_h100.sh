#!/usr/bin/env bash
# =============================================================================
#  SWE veRL Training — Qwen3.5-9B (dense), Megatron-Core, H100 variant
# =============================================================================
#
#  This is the H100 port of run_swe_training_9b_megatron.sh.  The training
#  config is identical; only the host-side CUDA / NCCL / venv plumbing changes
#  for the H100 hardware tier (cloudnative-maliva H100-SXM-80GB nodes,
#  driver 535.x / CUDA 12.9 toolkit, no bond0).
#
#  -----------------------------------------------------------------------------
#  WHY AN H100 FORK IS NEEDED (vs. the original B200 script)
#  -----------------------------------------------------------------------------
#
#  1. NETWORK INTERFACE — H100 nodes have no `bond0`.  Only `eth0` carries a
#     global IPv6 address (1420 MTU); eth1..eth8 are RDMA NICs without v6.
#     `NCCL_SOCKET_IFNAME=bond0` therefore must become `eth0`, both here and
#     in the head/worker startup scripts.
#
#  2. CUDA TOOLKIT — some shared veRL venvs do NOT contain a
#     bundled `nvidia/cu13/` toolkit wheel — only the per-component runtime
#     wheels (cublas, cudnn, cufft, ...).  The original script set
#     `CUDA_HOME=$VIRTUAL_ENV/.../nvidia/cu13` and added `$CUDA_HOME/bin` to
#     PATH expecting `nvcc` there.  That path doesn't exist, so flashinfer's
#     JIT compile of `gdn_prefill_sm90` (used by qwen3_next-style models in
#     vLLM 0.18) fails with `nvcc: not found` and one or more vLLM workers
#     get marked dead.  Fix: point CUDA_HOME at the system toolkit at
#     `/usr/local/cuda` (CUDA 12.9.86) which actually has nvcc.
#
#  3. CUDA-COMPAT-13 — H100 nodes ship with NVIDIA driver 535.129.x which is
#     fully compatible with CUDA 12.9 (the toolkit torch was built against),
#     so the cuda-compat-13 forward-compat shim from the B200 launch path is
#     unnecessary and we skip the wget/dpkg step entirely.
#
#  -----------------------------------------------------------------------------
#  LAUNCH INSTRUCTIONS (4-node H100 cluster from a Tiger devbox)
#  -----------------------------------------------------------------------------
#
#  0) PICK 4 H100 NODES.  All MALIVA H100s share a single /64 IPv6 subnet, so
#     bandwidth between any pair is fine; still prefer adjacent pod numbers
#     (the middle field in their `n124-{POD}-{NODE}` hostnames) for locality.
#
#       mlx worker list | grep H100-SXM-80GB
#
#     Pick four IDs.  In what follows, $HEAD is the chosen head node and
#     $WORKERS is the other three.
#
#  1) PRE-DOWNLOAD THE MODEL onto LOCAL /tmp on every node.  Don't use any
#     subdirectory of `/mlx_devbox` — that filesystem IS the 125 GB NFS
#     rootfs shared by the devbox and every worker, with only ~20 GB free.
#     The H100 local /tmp is `/dev/nvme0n2p1`, ~270-490 GB free, per node.
#
#  2) FIX HF CACHE LAYOUT.  `_download_qwen35_9b.sh` writes weights flat at
#     /tmp/hf_cache/models--Qwen--Qwen3.5-9B/, but HF_HUB_OFFLINE expects the
#     hub/ subdir layout. Make sure every node exposes the model under
#     /tmp/hf_cache/hub/models--Qwen--Qwen3.5-9B.
#
#  3) DEFENSIVE CLEANUP — stop any leftover Ray + tmux on every node:
#
#       for n in $HEAD $WORKERS; do
#         mlx worker login $n -- 'ray stop --force 2>/dev/null; tmux kill-server 2>/dev/null; true'
#       done
#
#  4) START OR ATTACH TO RAY, then run this script on the driver/head with
#     NNODES set to the cluster size and RAY_ADDRESS pointing at the cluster.
#
#  5) MONITOR.  All training output is printed by this script. Redirect to a
#     log file or run it inside tmux for long jobs.
#
#       mlx worker login $HEAD -- 'tail -f /tmp/swe_training_9b_megatron_h100.log'
#       mlx worker login $HEAD -- 'tmux capture-pane -t train -p -S -200 | tail'
#
#     wandb project: swe-verl-9b-megatron-h100 (set below).
#
#  6) STOP everything when done:
#
#       for n in $HEAD $WORKERS; do
#         mlx worker login $n -- 'ray stop --force; tmux kill-server'
#       done
#
#  -----------------------------------------------------------------------------
#  ENV KNOBS (override at the CLI before invoking this script)
#  -----------------------------------------------------------------------------
#
#    NNODES=2|4                       (default 2 — must match cluster size)
#    NGPUS_PER_NODE=8                 (default 8)
#    ACTOR_TP=2 ACTOR_PP=1 ACTOR_CP=2 (Megatron parallelism — DP = world / TPxPPxCP)
#    ROLLOUT_TP=1                     (vLLM tensor parallelism)
#    LOGGER='[console,wandb]'|'[console]'
#
# =============================================================================
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
    source /tmp/verl_venv/bin/activate
elif [ -d "$RLLM_REPO_ROOT/verl/.venv/bin" ]; then
    source "$RLLM_REPO_ROOT/verl/.venv/bin/activate"
else
    echo "No verl virtualenv found. Activate one, set VENV_DIR, or run cookbooks/swe/scripts/setup_verl_vllm018_qwen35.sh." >&2
    exit 1
fi

# ---- CUDA toolkit (H100) ----
# The venv has no bundled nvidia/cu13/ toolkit wheel — only per-component
# runtime libs.  Use the SYSTEM CUDA 12.9 toolkit at /usr/local/cuda for
# nvcc (flashinfer JIT) and the venv's nvidia/*/lib paths for runtime libs.
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:${PATH:-}"
NVIDIA_LIBS=$(find "$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia" -name "lib" -type d 2>/dev/null | tr "\n" ":")
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# ---- Environment ----
export RAY_ADDRESS=${RAY_ADDRESS:-auto}
export RLLM_RUN_TASK_RUNNER_LOCAL=${RLLM_RUN_TASK_RUNNER_LOCAL:-1}
export RAY_BACKEND_LOG_LEVEL=${RAY_BACKEND_LOG_LEVEL:-info}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
export RAY_LOG_TO_DRIVER=${RAY_LOG_TO_DRIVER:-1}
export WANDB_MODE=${WANDB_MODE:-online}
export HF_HOME=${HF_HOME:-/tmp/hf_cache}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-INFO}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}

# transformers 5.3 _patch_mistral_regex calls HF /api/models on every tokenizer
# load.  8 parallel vLLM EngineCore workers on anon IP triggers 429 and an
# unhelpful "Unable to load vocabulary" error.  Forcing offline mode makes
# is_offline_mode()=True which short-circuits the API call.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p ~/.cache/huggingface/hub
if [ ! -e ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B ] \
    && [ -d /tmp/hf_cache/hub/models--Qwen--Qwen3.5-9B ]; then
    ln -sf /tmp/hf_cache/hub/models--Qwen--Qwen3.5-9B \
        ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B
fi

export NCCL_CUMEM_ENABLE=0
# H100 nodes have no bond0 — eth0 is the only iface with a global IPv6.
# Unconditional export overrides the devbox-shell preset and is propagated
# to actors via verl runtime_env.env_vars.
export NCCL_SOCKET_IFNAME=eth0
export NCCL_SOCKET_FAMILY=AF_INET6
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_DISABLE_CUMEM=1
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TRITON_CACHE_DIR=/tmp/triton_cache
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/xdg_cache}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/tmp/torch_extensions}
export PYTHONPATH="$COOKBOOK_DIR:$RLLM_REPO_ROOT:${PYTHONPATH:-}"
mkdir -p "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR"

# On CPU Driver
export RLLM_RUN_TASK_RUNNER_LOCAL=1

# Required for Megatron comm/compute overlap
export CUDA_DEVICE_MAX_CONNECTIONS=1

# H100 has 80 GB HBM vs B200; Triton autotune scratch allocations during the
# first backward pass push past the limit.  Disable exhaustive autotuning so
# the first valid kernel config is used without benchmarking alternatives.
# Note: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is intentionally
# NOT set — vLLM's memory pool asserts against it (pytorch#147851).
export TORCHINDUCTOR_MAX_AUTOTUNE=0
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=0

MODEL_REPO=${MODEL_REPO:-Qwen/Qwen3.5-9B}
MODEL_CACHE_DIR=${MODEL_CACHE_DIR:-/tmp/hf_cache/hub/models--Qwen--Qwen3.5-9B}
if [ -z "${MODEL_PATH:-}" ]; then
    if [ -d "$MODEL_CACHE_DIR/snapshots" ]; then
        MODEL_PATH=$(find -L "$MODEL_CACHE_DIR/snapshots" -mindepth 1 -maxdepth 1 -type d | head -1)
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

# Parallelism for the two-node 16xH100 setup:
# world size 16 / (TP=2 × CP=2 × PP=1) = DP=4.
ACTOR_TP=${ACTOR_TP:-4}
ACTOR_CP=${ACTOR_CP:-4}
ACTOR_PP=${ACTOR_PP:-1}
ROLLOUT_TP=${ROLLOUT_TP:-1}
MODEL_MAX_TOKENS=${MODEL_MAX_TOKENS:-4096}
NNODES=${NNODES:-4}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
LOGGER=${LOGGER:-"[console,wandb]"}
SWE_STEP_LIMIT=${SWE_STEP_LIMIT:-200}
SWE_AGENT_TIMEOUT=${SWE_AGENT_TIMEOUT:-600}
SWE_COMMAND_TIMEOUT=${SWE_COMMAND_TIMEOUT:-120}
SWE_SANDBOX_TIMEOUT=${SWE_SANDBOX_TIMEOUT:-780}
SWE_STARTUP_JITTER_S=${SWE_STARTUP_JITTER_S:-15.0}
SWE_VAL_STEP_LIMIT=${SWE_VAL_STEP_LIMIT:-300}
SWE_VAL_AGENT_TIMEOUT=${SWE_VAL_AGENT_TIMEOUT:-900}
SWE_VAL_COMMAND_TIMEOUT=${SWE_VAL_COMMAND_TIMEOUT:-$SWE_COMMAND_TIMEOUT}
SWE_VAL_SANDBOX_TIMEOUT=${SWE_VAL_SANDBOX_TIMEOUT:-1020}
SWE_VAL_STARTUP_JITTER_S=${SWE_VAL_STARTUP_JITTER_S:-30}
DEFAULT_TRAJ_DIR="$COOKBOOK_DIR/trajectories/\${trainer.experiment_name}"
TRAJECTORY_OUTPUT_DIR=${TRAJ_DIR:-$DEFAULT_TRAJ_DIR}
ROLLOUT_CORRECTION_IS=${ROLLOUT_CORRECTION_IS:-token}
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
    trainer.default_local_dir=/tmp/turing-swe/checkpoints/swe-verl-9b-megatron-lora/qwen35-9b-swe-smith-lora-r16
    ++trainer.hf_repo_id=JWei05/qwen35-9b-swe-smith-megatron-lora-r16
)

echo "============================================================"
echo "SWE veRL Training — Qwen3.5-9B (Megatron TP=${ACTOR_TP} CP=${ACTOR_CP} PP=${ACTOR_PP})  H100"
echo "============================================================"
echo "Model:    $MODEL_NAME"
echo "Repo:     $MODEL_REPO"
echo "Dataset:  swe_smith_filtered_mix"
echo "Topology: ${NNODES} node(s) × ${NGPUS_PER_NODE} GPU  |  Megatron TP=${ACTOR_TP} CP=${ACTOR_CP} PP=${ACTOR_PP}  |  vLLM TP=${ROLLOUT_TP}"
echo "SWE:      train steps=${SWE_STEP_LIMIT} timeout=${SWE_AGENT_TIMEOUT}s cmd=${SWE_COMMAND_TIMEOUT}s sandbox=${SWE_SANDBOX_TIMEOUT}s jitter=${SWE_STARTUP_JITTER_S}s max_tokens=${MODEL_MAX_TOKENS}"
echo "SWE val:  steps=${SWE_VAL_STEP_LIMIT} timeout=${SWE_VAL_AGENT_TIMEOUT}s cmd=${SWE_VAL_COMMAND_TIMEOUT}s sandbox=${SWE_VAL_SANDBOX_TIMEOUT}s jitter=${SWE_VAL_STARTUP_JITTER_S}s"
echo "Traj:     save=${SAVE_TRAJ:-false} dir=${TRAJECTORY_OUTPUT_DIR}"
echo "Logger:   $LOGGER"
echo "Ray:      $(ray status 2>&1 | head -1)"
echo "Debug:    RAY_DEDUP_LOGS=$RAY_DEDUP_LOGS RAY_LOG_TO_DRIVER=$RAY_LOG_TO_DRIVER VERL_LOGGING_LEVEL=$VERL_LOGGING_LEVEL"
echo "============================================================"

python -u -m swe.scripts.train_swe_verl \
    --config-name=verl_swe_trainer \
    train_dataset=swe_smith_filtered_mix \
    val_dataset=swe_bench_multilingual \
    train_max_samples=1500 \
    actor_rollout_ref.model.path="$MODEL_NAME" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=false \
    +model.name="$MODEL_NAME" \
    \
    data.train_batch_size=8 \
    data.gen_batch_size=8 \
    data.max_prompt_length=30720 \
    data.max_response_length=32768 \
    data.filter_overlong_prompts=true \
    data.filter_overlong_prompts_workers=8 \
    \
    ++trainer.use_legacy_worker_impl=disable \
    actor_rollout_ref.actor.strategy=megatron \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${ACTOR_CP} \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.calculate_per_token_loss=true \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.mtp_num_layers=0 \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.sequence_parallel=false \
    actor_rollout_ref.actor.megatron.use_distributed_optimizer=true \
    actor_rollout_ref.actor.megatron.param_offload=true \
    actor_rollout_ref.actor.megatron.grad_offload=true \
    actor_rollout_ref.actor.megatron.optimizer_offload=true \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    ++actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=k1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.calculate_entropy=true \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.actor.use_dynamic_bsz=false \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.free_cache_engine=false \
    actor_rollout_ref.rollout.enable_prefix_caching=false \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=6144 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=true \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=qwen3_xml \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    ++actor_rollout_ref.ref.strategy=megatron \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${ACTOR_TP} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${ACTOR_PP} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${ACTOR_CP} \
    actor_rollout_ref.ref.megatron.param_offload=true \
    \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=false \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    algorithm.rollout_correction.bypass_mode=true \
    algorithm.rollout_correction.rollout_is=${ROLLOUT_CORRECTION_IS} \
    algorithm.rollout_correction.rollout_is_threshold=${ROLLOUT_CORRECTION_IS_THRESHOLD} \
    \
    ++rllm.gateway.db_path=/tmp/gateway_traces_9b_megatron.db \
    ++rllm.gateway.strip_vllm_fields=false \
    rllm.workflow.n_parallel_tasks=64 \
    rllm.workflow.retry_limit=1 \
    rllm.workflow.raise_on_error=false \
    rllm.algorithm.rollout_correction.bypass_mode=true \
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
    +swe.compaction_token_trigger=18000 \
    +swe.compaction_keep_recent_turns=1 \
    +swe.startup_jitter_s=${SWE_STARTUP_JITTER_S} \
    +swe.sandbox_retry_backoff_min_s=5.0 \
    +swe.sandbox_retry_backoff_max_s=10.0 \
    +swe.model_max_tokens=${MODEL_MAX_TOKENS} \
    +swe.model_return_token_ids=true \
    \
    trainer.total_epochs=100 \
    trainer.save_freq=100 \
    trainer.test_freq=1000 \
    trainer.val_before_train=false \
    ++rllm.trainer.val_before_train=false \
    trainer.default_local_dir=/tmp/turing-swe/checkpoints/swe-verl-9b-megatron/qwen35-9b-swe-smith \
    +trainer.hf_repo_id=JWei05/qwen35-9b-swe-smith-megatron \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    "trainer.logger=${LOGGER}" \
    trainer.project_name=swe-verl-9b-megatron \
    trainer.experiment_name=qwen35-9b-swe-smith-megatron \
    "${LORA_ARGS[@]}" \
    "$@"
