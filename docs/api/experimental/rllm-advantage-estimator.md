# rLLM Advantage Estimator API (Experimental)

Reference for the rLLM-native advantage estimator interfaces, registry APIs, and
related config types.

---

## Registry and Collection

::: rllm.experimental.common.advantage.register_rllm_adv_estimator

::: rllm.experimental.common.advantage.get_rllm_adv_estimator

::: rllm.experimental.common.advantage.collect_reward_and_advantage_from_trajectory_groups

---

## Built-in Estimators

::: rllm.experimental.common.advantage.calculate_grpo_advantages

::: rllm.experimental.common.advantage.calculate_reinforce_advantages

::: rllm.experimental.common.advantage.calculate_reinforce_plus_plus_baseline_advantages

::: rllm.experimental.common.advantage.calculate_rloo_advantages

---

## Supporting Config Types

::: rllm.experimental.common.config.rLLMAdvantageEstimator

::: rllm.experimental.common.config.AlgorithmConfig

---

## Per-Group Algo Helpers

::: rllm.experimental.common.rl_algo.calculate_grpo_advantages_per_group

::: rllm.experimental.common.rl_algo.calculate_rloo_advantages_per_group

---

## UnifiedTrainer Entry Point

`traj_group_adv_estimator_map` is provided through trainer construction kwargs and
wired into `AlgorithmConfig.estimator_map` in `UnifiedTrainer`.

::: rllm.experimental.unified_trainer.UnifiedTrainer
