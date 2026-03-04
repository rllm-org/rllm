# rLLM RL Advantage Estimator

> **Module**: `rllm.experimental.common.advantage`

!!! warning "Backend behavior differs"
    The phrase "rLLM advantage estimator" does **not** mean the same runtime path for all backends:

    - **Tinker backend**: advantage computation is entirely rLLM-based. So the following sections naturally apply.
    - **Verl backend**: by default (`rllm.algorithm.use_rllm=False`), it uses Verl-native advantage computation; rLLM-based computation is used only when `rllm.algorithm.use_rllm=True`.

    Also, the rLLM estimator set is intentionally smaller than Verl's full native set. The tradeoff is a unified interface with easier role-level customization.

This page walks through:

1. How the rLLM estimator interface works
2. How behavior differs between Tinker and Verl
3. Why role-level estimator overrides are powerful
4. How to register and use custom estimators

---

## Core Concept

In the unified trainer, reward comparison happens on `TrajectoryGroup`s.
Groups are partitioned by `group_role` (for example, `solver` and `judge`), and each role's estimator receives:

- a batch (list) of reward arrays (`list[np.ndarray]`), from a batch of `TrajectoryGroup`s with the same `group_role`.
- each reward array contains all the trajectories' rewards in a single `TrajectoryGroup`.

Expected estimator signature:

```python
def my_adv_estimator(
    rewards: list[np.ndarray],
    **kwargs,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    ...
```

Where outputs are aligned to input groups:

- `advantages_by_group[i]` corresponds to `rewards[i]`
- `returns_by_group[i]` has the same shape/alignment as `advantages_by_group[i]`

---

## Backend-Specific Behavior

### Tinker Backend

Tinker uses the rLLM-based advantage path in the unified training flow.
That means your configured `rllm.algorithm.adv_estimator` and any role overrides apply directly.

#### Tinker loss mapping

For Tinker, rLLM also maps estimators to default loss functions in
`rllm/trainer/tinker/tinker_policy_trainer.py`:

| Advantage estimator | Default Tinker loss fn |
|---|---|
| `REINFORCE` | `importance_sampling` |
| `REINFORCE_PLUS_PLUS_BASELINE` | `importance_sampling` |
| `GRPO` | `ppo` |
| `RLOO` | `importance_sampling` |
| `OTHER` / unknown | `importance_sampling` (safe fallback) |

Override anytime via:

```yaml
rllm:
  algorithm:
    loss_fn: ppo  # or importance_sampling / cispo / dro / cross_entropy
```

### Verl Backend

Verl has two possible paths:

- `rllm.algorithm.use_rllm=false` (default): use Verl-native advantage computation.
- `rllm.algorithm.use_rllm=true`: use rLLM-based advantage computation and then inject it back into the Verl batch.

So, role-level estimator overrides (`traj_group_adv_estimator_map`) are an rLLM-path feature and require `use_rllm=true`.

---

## Built-in rLLM Estimators

| Estimator enum | Value | Behavior |
|---|---|---|
| `GRPO` | `grpo` | Per-group reward centering; optional std normalization (`norm_adv_by_std_in_grpo`) |
| `REINFORCE` | `reinforce` | No baseline (`adv = reward`) |
| `REINFORCE_PLUS_PLUS_BASELINE` | `reinforce_plus_plus_baseline` | Per-group centering, then role-batch std normalization |
| `RLOO` | `rloo` | Leave-one-out baseline per group |

For `REINFORCE++ baseline`, normalization uses batch-level statistics to calculate the standard deviation of all centered rewards.

Please refer to the [API reference](../api/experimental/rllm-advantage-estimator.md) for more details.

---

## Setting Role-Level Advantage Estimators via Mapping

The most powerful feature of the rLLM path is **assigning different estimators to different trajectory roles** in one training job.

This is especially useful for multi-agent workflows where roles have different reward/statistical properties. For instance, in a solver-judge workflow, we have abundant number of solver trajectories (2 solver trajectories per rollout * N rollouts = 2N solver trajectories), so **GRPO** can be a good choice for optimizing the solver's performance. The judge, on the other hand, depends on the rollout's own solver answers, so it can be less reasonable to compare their relative advantage by grouping across rollouts. So we might want to use a more vanilla **REINFORCE** for the judge.

This seemingly complicated setup can be easily achieved by simply configuring the `traj_group_adv_estimator_map` in the trainer constructor.

Example from `rllm/experimental/test_examples/test_tinker_solver_judge.py`:

```python
from rllm.experimental.common.config import rLLMAdvantageEstimator
from rllm.experimental.unified_trainer import AgentTrainer

traj_group_adv_estimator_map = {
    "solver": rLLMAdvantageEstimator.GRPO,
    "judge": rLLMAdvantageEstimator.REINFORCE,
}

trainer = AgentTrainer(
    ...,
    backend="tinker",
    traj_group_adv_estimator_map=traj_group_adv_estimator_map,
)
```

!!! tip
    If you pass `traj_group_adv_estimator_map`, set `rllm.algorithm.use_rllm=true`.
    `UnifiedTrainer` validates this.

Global default remains:

```yaml
rllm:
  algorithm:
    use_rllm: true
    adv_estimator: grpo
  stepwise_advantage:
    mode: broadcast
    norm_adv_by_std_in_grpo: true
```

---

## Custom Estimators: Register and Use

Use the registry helpers:

- `register_rllm_adv_estimator(name)`
- `get_rllm_adv_estimator(name)`

```python
import numpy as np
from rllm.experimental.common.advantage import (
    register_rllm_adv_estimator,
    get_rllm_adv_estimator,
)


@register_rllm_adv_estimator("my_custom_adv")
def my_custom_adv(rewards: list[np.ndarray], **kwargs):
    advantages_by_group = [group_rewards - np.mean(group_rewards) for group_rewards in rewards]
    returns_by_group = advantages_by_group
    return advantages_by_group, returns_by_group


fn = get_rllm_adv_estimator("my_custom_adv")
```

You can use the custom estimator as:

1. Global default:

```yaml
rllm:
  algorithm:
    use_rllm: true
    adv_estimator: my_custom_adv
```

2. Role-specific override in `traj_group_adv_estimator_map`:

```python
traj_group_adv_estimator_map = {
    "solver": "my_custom_adv",
    "judge": "reinforce",
}
```

---

## Related References

- [Pre-computing Advantage in Workflow](rllm-precompute-advantage.md)
- [API: rLLM Advantage Estimator](../api/experimental/rllm-advantage-estimator.md)
- [Unified Trainer](unified-trainer.md)
- [rLLM and Backend Config](rllm-and-backend-config.md)
