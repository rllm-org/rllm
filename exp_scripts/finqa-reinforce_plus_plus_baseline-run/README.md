# FinQA REINFORCE++-baseline comparison arm

This directory contains only experiment-layer launch and audit artifacts. It
does not modify the rLLM, veRL, vLLM, FinQA flow, or evaluator implementations.

The comparison target is the completed FinQA GRPO arm. The launcher delegates
to `../finqa-grpo-run/train_16.sh` and changes only the advantage estimator:

```yaml
algorithm.adv_estimator: reinforce_plus_plus_baseline
rllm.algorithm.adv_estimator: reinforce_plus_plus_baseline
actor_rollout_ref.rollout.n: 8
rllm.rollout.n: 8
```

TIS remains disabled. Batch size, model and LoRA configuration, optimiser,
policy loss, clipping, KL, entropy, rollout sampling, judge, checkpoint cadence
and validation protocol are inherited from the GRPO arm.

The formal arm targets the 2x48GB `.24` host and retains the GRPO arm's
`actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288`. A prior local 32GB
smoke OOMed during backward and is not evidence against the 48GB configuration.
Formal checkpoints and validation both run once per 125-step epoch.

Commands:

```bash
bash train_local_reinforce_plus_plus.sh smoke
bash train_local_reinforce_plus_plus.sh formal
```

The smoke run must show 32 groups of exactly 8 trajectories per step, nonzero
reward variance, finite nonzero advantages/gradient norm, and no CUDA, Ray,
vLLM or judge failure before the formal run is launched.

This experiment should be reported as **rLLM
`reinforce_plus_plus_baseline` advantage estimator with the existing veRL PPO
policy-loss protocol**, not as a complete reproduction of every component of
the REINFORCE++ paper.
