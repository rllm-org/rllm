#!/usr/bin/env bash
# Train an SWE agent on Scale-SWE, eval on SWE-bench Verified — Fireworks ASYNC
# backend, Qwen3.5-35B-A3B (MoE) + LoRA, with the **DPPO** policy loss.
#
# This is the a35b (35B MoE) sibling of train_fireworks.sh (Qwen3.5-9B), adapted
# from cookbooks/terminal-rl/train_fireworks_a35b.sh but swapping the algorithm
# from plain GRPO to DPPO.
#
# DPPO (Divergence Proximal Policy Optimization, arXiv:2602.04879): keeps GRPO's
# group-normalized advantages but replaces PPO's ratio-clipping with a per-token
# total-variation divergence mask — it zeroes a token's gradient only when the
# update pushes it away from the behavior policy AND |exp(pi)-exp(mu)| exceeds a
# threshold delta. With mu = the inference (sampling) log-probs (the default on
# the managed backends), the same mask also absorbs the train/inference log-prob
# mismatch. DPPO is an rLLM-native loss and Fireworks has no native DPPO kernel,
# so it runs on the client-side forward_backward_custom path (one extra forward
# pass per step vs a native fused kernel).
#
# Prerequisites:
#   1. Install rllm with fireworks extras:  uv pip install -e ".[fireworks]"
#   2. Install this cookbook:                uv pip install --no-deps -e cookbooks/swe-rl
#   3. Pull the datasets:                    python cookbooks/swe-rl/prepare_data.py
#   4. Set your API key:                     export FIREWORKS_API_KEY=...
#
# The trainer job and inference deployment are provisioned on Fireworks at
# startup and torn down on shutdown.
#
# Model: Qwen3.5-35B-A3B (MoE) + LoRA-32 on the qwen3p5-35b-a3b-256k-lora training
# shape. To change the model, swap model.name / model.tokenizer_model /
# fireworks_config.policy_trainer_shape_id together (see docs/backends/fireworks.mdx).
#
# The DPPO knobs (rllm.algorithm.*):
#   loss_fn=dppo_tv        select the DPPO total-variation loss (dppo_kl is the KL variant)
#   eps_clip=0.15          reused as the TV divergence threshold delta (paper's TV value for the
#                          MoE-Base+LoRA setup, which this run is; TV is robust over 0.10-0.20)
#   adv_estimator=grpo     DPPO keeps GRPO's group-normalized advantages
# To A/B against plain GRPO, just drop loss_fn (null = backend-native GRPO clip).
#
# Sandbox backend is chosen by SWE_SANDBOX_BACKEND (docker | local | modal |
# daytona; default modal). modal needs `pip install modal` + `modal token new`.
# Per-rollout agent timeout: RLLM_HARNESS_RUN_TIMEOUT_S; the Modal sandbox
# lifetime (RLLM_MODAL_SANDBOX_TIMEOUT_S) sits above it so a capped rollout is
# torn down cleanly rather than reaped mid-run.
#
# Qwen3.5 is a reasoning family; if the harness can't parse reasoning output,
# disable it by appending: rollout_engine.reasoning_effort=none
#
# Override anything by passing extra Hydra args after the script:
#   bash train_fireworks_a35b_dppo.sh training.learning_rate=1e-5

set -euo pipefail

export SWE_SANDBOX_BACKEND="${SWE_SANDBOX_BACKEND:-modal}"
# Sandbox resources: pin to Modal's minimum reservation. Every scaleswe task.toml
# uniformly declares cpus=4 / memory_mb=16384 / storage_mb=30720; these caps clamp
# those baked-in values DOWN at runtime (min(declared, cap)) with no dataset rebuild.
# rLLM passes cpu/memory to Modal as scalars => SOFT requests (milli_cpu_max=None,
# memory_mb_max=0 = no hard limit), so the container still BURSTS to whatever CPU/RAM
# the host has spare — this only lowers the reserved/billed floor, it cannot OOM or
# hard-throttle. 0.125 cores is Modal's minimum reservation. Modal ignores storage
# (billed as part of compute), so RLLM_SANDBOX_MAX_STORAGE_MB is a no-op here.
export RLLM_SANDBOX_MAX_CPUS="${RLLM_SANDBOX_MAX_CPUS:-0.125}"
export RLLM_SANDBOX_MAX_MEMORY_MB="${RLLM_SANDBOX_MAX_MEMORY_MB:-128}"
# Eval budget: validate on the first N swebench-verified tasks (0/unset = all 500).
export SWE_VAL_MAX="${SWE_VAL_MAX:-250}"
# Per-rollout turn cap for terminus2 (read by train.py). Empty = uncapped.
export TERMINUS_MAX_TURNS="${TERMINUS_MAX_TURNS:-50}"
# Disable Terminus-2 context summarization/compaction. swe-rl train.py reads this and
# passes enable_summarize to Terminus2Harness, which forwards RLLM_TERMINUS_ENABLE_SUMMARIZE
# into the sandbox where the driver honors it (needs terminal-rl >= b342c9c7, this branch's base).
export TERMINUS_ENABLE_SUMMARIZE="${TERMINUS_ENABLE_SUMMARIZE:-0}"
# Keep prior-turn reasoning in the agent's chat history instead of dropping it
# (Harbor's interleaved_thinking; off by default in Terminus2Harness). Read at import
# from RLLM_TERMINUS_INTERLEAVED_THINKING (train.py doesn't forward it via a ctor arg).
# NOTE: with cumulative_token_mode on (below), this is mostly belt-and-braces. The
# gateway rebuilds each turn's prompt from the PRIOR turn's *sampled* token ids
# (prev_completion_ids = choices[0].token_ids, which already include the generated
# <think>) and DROPS the assistant message from the delta it re-renders — so prior
# reasoning stays in both the model's input and the training tokens regardless of this
# flag. It only bites on the fallback chat path taken after an accumulator reset
# (compaction/prefix-change), where the full message array is re-rendered.
export RLLM_TERMINUS_INTERLEAVED_THINKING="${RLLM_TERMINUS_INTERLEAVED_THINKING:-1}"
export RLLM_HARNESS_RUN_TIMEOUT_S="${RLLM_HARNESS_RUN_TIMEOUT_S:-1800}"
# Cap the verifier at 300s. SWE-bench Verified tasks declare verifier.timeout_sec=3000,
# which is far longer than needed for fast iteration and bloats the sandbox lifetime.
export RLLM_HARNESS_VERIFIER_TIMEOUT_S="${RLLM_HARNESS_VERIFIER_TIMEOUT_S:-300}"
# Sandbox lifetime is auto-derived (no env needed): the effective agent timeout
# (RLLM_HARNESS_RUN_TIMEOUT_S caps the task's own) + verifier + install + slack —
# ~3600s here from the 1800s agent cap. Set RLLM_SANDBOX_TIMEOUT_S to override.

python -u train.py \
    rllm/backend=fireworks \
    model.name=accounts/fireworks/models/qwen3p5-35b-a3b \
    model.tokenizer_model=Qwen/Qwen3.5-35B-A3B \
    model.lora_rank=32 \
    fireworks_config.policy_trainer_shape_id=accounts/fireworks/trainingShapes/qwen3p5-35b-a3b-256k-lora \
    fireworks_config.policy_trainer_replica_count=2 \
    fireworks_config.rollout_deployment_replica_count=4 \
    training.group_size=16 \
    training.learning_rate=2e-5 \
    training.max_length=65536 \
    rllm.rollout.train.temperature=1.0 \
    rllm.rollout.train.top_p=0.95 \
    rllm.rollout.val.temperature=1.0 \
    rllm.rollout.val.top_p=0.95 \
    data.max_prompt_length=57344 \
    data.max_response_length=8192 \
    data.train_batch_size=1 \
    data.val_batch_size=-1 \
    rllm.data.max_prompt_length=57344 \
    rllm.data.max_response_length=8192 \
    rllm.data.train_batch_size=1 \
    rllm.data.val_batch_size=-1 \
    rllm.compact_filtering.enable=true \
    rllm.algorithm.adv_estimator=grpo \
    rllm.algorithm.norm_adv_by_std_in_grpo=true \
    rllm.algorithm.loss_fn=dppo_tv \
    rllm.algorithm.loss_agg_mode=seq-mean-token-mean \
    rllm.algorithm.eps_clip=0.15 \
    rllm.async_training.enable=true \
    rllm.async_training.mini_batch_size=16 \
    rllm.async_training.fwd_bwd_group_size=1 \
    rllm.async_training.staleness_threshold=3.0 \
    rllm.async_training.trigger_parameter_sync_step=1 \
    rllm.async_training.partial_rollout=true \
    rllm.workflow.n_parallel_tasks=378 \
    rllm.workflow.raise_on_error=false \
    rllm.rejection_sample.filter_uniform_groups=false \
    rllm.gateway.port=9090 \
    rllm.gateway.num_workers=4 \
    rllm.gateway.cumulative_token_mode=true \
    rllm.gateway.renderer_family=qwen3.5 \
    rllm.trainer.total_epochs=1 \
    rllm.trainer.logger='[wandb]' \
    rllm.trainer.project_name='swe-rl' \
    rllm.trainer.experiment_name='swe-rl-scaleswe-qwen3p5-35b-a3b-dppo-fireworks' \
    rllm.trainer.val_before_train=false \
    rllm.trainer.test_freq=20 \
    rllm.trainer.save_freq=20 \
    "$@"
