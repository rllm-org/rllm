#!/usr/bin/env bash
# FinQA GRPO on .16 (tai): 2x RTX 6000 Ada 48GB, sm_89.
#
# Reuses the cookbook untouched (finqa_flow / finqa_evaluator / AgentTrainer /
# unified hydra config). The only non-repo file is train_finqa_subset.py, which
# is cookbooks/finqa/train.py plus split subsampling.
#
#   ./train_16.sh smoke    # 2 steps, 20 train tasks, no val  -> link + timing check
#   ./train_16.sh epoch    # full 4030 tasks, checkpoint once per epoch
set -euo pipefail
RD="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:-smoke}"; shift || true

source "$RD/env.sh"
source "$VENV/bin/activate"
cd /home/yanan/agents/rllm/cookbooks/finqa   # finqa_constants resolves data/ and prompts/ from here

MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507

# train_batch_size=32 matches the paper GRADIENT batch exactly: finqa.md uses
#   mini_batch 32 x n 8 = 256 sequences per update; so do we. Raising the batch
#   costs nothing in total epoch compute (4030 tasks x 8 rollouts is fixed) and
#   cuts sleep/wake + weight-sync overhead from 403 cycles to 126.
#   4030 // 32 = 125 steps per epoch (drop_last=True drops 30 samples).

case "$MODE" in
  smoke)
    export FINQA_TRAIN_N=20 FINQA_VAL_N=8
    MODE_OVERRIDES=(
      trainer.total_epochs=1
      trainer.total_training_steps=2
      trainer.test_freq=-1
      trainer.save_freq=-1
      trainer.val_before_train=false
      trainer.experiment_name=qwen3-4b-16-smoke
    )
    ;;
  epoch)
    # 0 = full training split. Formal validation is performed offline so that
    # training is not interrupted and all checkpoints use one evaluation protocol.
    export FINQA_TRAIN_N=0 FINQA_VAL_N=0
    MODE_OVERRIDES=(
      trainer.total_epochs=1
      trainer.save_freq=125                   # one checkpoint per complete epoch
      # One epoch is 125 steps, NOT 126: the train dataloader is built with
      # drop_last=True (ray_trainer.py:409), so 4030 // 32 = 125 and the tail
      # 30 samples are dropped each epoch. Epoch boundaries are therefore
      # 125 / 250 / 375 / ... and total_training_steps = 125 x 10 = 1250.
      trainer.test_freq=-1                    # disable in-training validation
      trainer.val_before_train=false          # base is evaluated by the offline pipeline
      trainer.experiment_name=qwen3-4b-16-epoch
      # Unattended: verl's own default is "auto"; the cookbook's train_verl.sh
      # sets "disable", which for a ~46 h run means any crash restarts from zero.
      # With auto, a restart resumes from the newest epoch-boundary checkpoint.
      trainer.resume_mode=auto
    )
    ;;
  stability)
    # Longer shake-out before committing to the full epoch: 20 steps x 32 tasks,
    # and it deliberately exercises BOTH the validation path and checkpoint
    # writing, neither of which the 2-step smoke touches.
    export FINQA_TRAIN_N=640 FINQA_VAL_N=64
    MODE_OVERRIDES=(
      trainer.total_epochs=1
      trainer.total_training_steps=20     # 20 x 32 = 640 tasks
      trainer.test_freq=10
      trainer.save_freq=20                # one checkpoint at the end of this epoch
      trainer.val_before_train=false
      trainer.experiment_name=qwen3-4b-16-stability
    )
    ;;
  literal)
    # Control arm: finqa.md's Context & Generation lengths taken LITERALLY
    # (prompt 2048 / response 16384). Those numbers describe the OLD stack
    # (projects/finqa, AgentExecutionEngine) where response = the whole 20-turn
    # trajectory; in the AgentFlow cookbook each LLM call is a Step, so our
    # measured prompts are 2591-2898 and responses only 206-341.
    # This arm settles empirically whether 2048 truncates in practice.
    #   watch: batch/termination_reason/max_prompt_length_exceeded
    export FINQA_TRAIN_N=96 FINQA_VAL_N=0
    MODE_OVERRIDES=(
      trainer.total_epochs=1
      trainer.total_training_steps=3
      trainer.test_freq=-1
      trainer.save_freq=-1
      trainer.val_before_train=false
      trainer.experiment_name=qwen3-4b-16-literal
      data.max_prompt_length=2048
      data.max_response_length=16384
      # both must clear max_seq_len = 2048+16384 = 18432
      # (seqlen_balancing.py:384 asserts max_token_len >= max_seq_len)
      actor_rollout_ref.rollout.max_model_len=18432
      actor_rollout_ref.actor.ppo_max_token_len_per_gpu=18432
      actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=18432
      actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=18432
    )
    ;;
  *) echo "usage: $0 {smoke|stability|literal|epoch} [hydra overrides...]" >&2; exit 2 ;;
esac

exec python -u "$RD/train_finqa_subset.py" \
    rllm/backend=verl \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=true \
    rllm.algorithm.use_rllm=true \
    +model.name=$MODEL_PATH \
    actor_rollout_ref.model.path=$MODEL_PATH \
    `# LoRA: the FSDP engine reads the FLAT model.lora_rank; the nested` \
    `# model.lora.* dict is the Megatron config and is silently ignored here.` \
    `# Assert on the log line: PeftModelForCausalLM, not Qwen3ForCausalLM.` \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.lora.merge=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    `# finqa.md Training Configuration: KL 0.001, entropy 0.002.` \
    `# use_kl_loss=True pulls in a reference policy (need_reference_policy),` \
    `# entropy_coeff!=0 turns on entropy -> fp32 logits; chunking+checkpointing` \
    `# keep that bounded (the fsdp engine honours both, transformer_impl.py:158).` \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.002 \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    `# Offload is back ON. Rationale for turning it off ("LoRA optimizer is tiny")` \
    `# held for the optimizer but NOT for params: param_offload moves the 3.87 GB` \
    `# of base weights off the GPU and empties the torch cache, which is exactly` \
    `# what vLLM needs to reclaim on wake_up. Without it, adding the KL reference` \
    `# model pushed the peak past what the caching allocator would return to the` \
    `# driver and vLLMHttpServer.wake_up() died with` \
    `#   CUDA Error: out of memory at cumem_allocator.cpp:139` \
    `# Measured cost of offload on this box: update_actor 164.4s -> 171.5s (~4%).` \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=12288 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=12288 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=true \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=hermes \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    trainer.nnodes=1 \
    trainer.logger="['console']" \
    trainer.project_name=finqa \
    trainer.default_hdfs_dir=null \
    trainer.resume_mode=disable \
    trainer.default_local_dir="$RD/checkpoints/\${trainer.experiment_name}" \
    `# ---- 2x RTX 6000 Ada 48GB ----` \
    trainer.n_gpus_per_node=2 \
    data.train_batch_size=32 \
    data.val_batch_size=16 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    `# max_model_len keeps 2048 of headroom over prompt+response: on .29 an exact` \
    `# sum (4096+1024=5120) let a 4097-token prompt trip vLLM's context check.` \
    actor_rollout_ref.rollout.max_model_len=12288 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.42 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    `# must be >= longest single sequence (seqlen_balancing.py asserts it);` \
    `# dynamic bsz packs whole sequences and never splits one.` \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
    "${MODE_OVERRIDES[@]}" \
    "$@"
