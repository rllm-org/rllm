# Pre-computing Advantage in Workflow

> **Related Modules**: `rllm.agents.agent.Step`, `rllm.experimental.common.advantage`, `rllm.experimental.common.config.AlgorithmConfig`

This page explains how to set `step.advantage` during workflow rollout and let the
unified trainer consume it directly, instead of computing advantages later from
trajectory rewards.

!!! warning "Required configuration"
    To enable precomputed advantages:

    - set `rllm.algorithm.use_precomputed_advantage: true`
    - if using **Verl backend**, also set `rllm.algorithm.use_rllm: true`

    With `use_precomputed_advantage=true`, rLLM will use your `step.advantage`
    values for groups that provide them.

---

## Why This Exists

In standard RL estimators (for example GRPO), advantages are computed after grouping
trajectories into `TrajectoryGroup`s. This requires rollout results from multiple
samples, so the calculation naturally happens in the trainer pipeline.

For other post-training setups, however, "advantage" is better treated as a generic
per-token training signal that can be produced directly inside workflow logic.
Common examples:

1. **SFT-like supervision**: token-level signals can be derived from demonstration targets.
2. **On-policy distillation (OPD)**: token-level reverse-KL-style signals can be computed
   from teacher/student log-probabilities.

In those cases, precomputing at workflow time gives you more control and removes the
need for group-based advantage computation on those trajectories.

---

## How It Works in rLLM

The advantage collection logic is in
`rllm.experimental.common.advantage.collect_reward_and_advantage_from_trajectory_groups`.

For each `TrajectoryGroup`:

- if any step has `step.advantage != None` **and** `use_precomputed_advantage=true`,
  rLLM consumes precomputed values from steps in that group
- otherwise, rLLM falls back to RL estimator computation from trajectory rewards

This means you can run **mixed training in one job**:

- some roles/groups use precomputed step-level signals
- other roles/groups still use RL estimators (GRPO/REINFORCE/RLOO/custom)

---

## Data Contract for `step.advantage`

When precompute mode is active for a group:

- `step.advantage` can be:
  - `float`: broadcast to all tokens in `step.response_ids`
  - `list[float]`: must match `len(step.response_ids)`
- unsupported types raise an error
- length mismatches are replaced by zeros with a warning

If precomputed values exist but `use_precomputed_advantage=false`, rLLM logs a warning
and overwrites with the configured RL estimator.

---

## Solver-Judge Example (Mixed Mode)

The solver-judge workflow is a good example of why this is useful. Suppose each episode
has two `solver` trajectories and one `judge` trajectory, the workflow (with some simplification from `examples.solver_judge.solver_judge_flow`) looks like this:

```python
async def run(self, task: dict, uid: str, **kwargs) -> Episode:
    solver_trajectories = await self.solver.generate_solutions(problem, n_solutions=2)
    judge_trajectory = await self.judge.judge_solutions(problem, solutions)
    return Episode(
        id=uid,
        task=task,
        trajectories=[*solver_trajectories, judge_trajectory],
        is_correct=is_correct,
        metrics={"solver_acc": solver_acc, "judge_acc": judge_acc},
    )
```

??? info "Grouping intuition"
    If you sample `N` rollouts per prompt:

    - solver group size is approximately `2N` trajectories
    - judge group size is approximately `N` trajectories

    (assuming one judge trajectory per episode and two solver trajectories per episode)

In a classic setup, both roles are trained with the same RL estimator (e.g. GRPO).  
With precomputed advantages, you can do something more flexible:

- keep `solver` on GRPO (group-based RL)
- precompute `judge` step advantages using OPD-style teacher signals

### Configure the advantage estimator for `solver` (no precomputed advantage)

```yaml
rllm:
  algorithm:
    use_precomputed_advantage: true
    use_rllm: true   # required for Verl;
    adv_estimator: grpo
```

### Precompute judge advantage in workflow

Here we assume that you have access to a generic "teacher" client that can evaluate the log probabilities of any token sequence.
In reality, this might depend on the backend you use -- for instance, in Tinker this can be a simple `tinker.SamplingClient` instance.

We can then assign a reverse-KL-style advantage to the `judge` step in the workflow:

```python
from rllm.trainer.distill.advantage import compute_distill_reverse_kl

async def run(self, task: dict, uid: str, **kwargs) -> Episode:
    # we obtain the solver & judge trajectories as usual
    ...
    judge_step = judge_trajectory.steps[0]
    teacher_logprobs = await self.teacher_client.compute_logprobs(judge_step.response_ids)
    student_logprobs = judge_step.logprobs
    # judge_step.advantage is now precomputed (per-token list)
    judge_step.advantage = compute_distill_reverse_kl(teacher_logprobs, student_logprobs)
    ...
```

With this, during training, the `judge` group will bypass the usual RL advantage computation process during the training loop, and directly use the precomputed advantages.
While the `solver` group will receive the usual group-based RL advantages from the trainer.

### Configure trainer for mixed mode

Following the example above, we can take one step further by considering a scenario where we have an extra role in our workflow, let's say a `validator` role, that we want to equip with a different RL advantage estimator (e.g. REINFORCE, or a custom advantage estimator you registered). We can combine the precomputed advantage with the role-level advantage estimator (introduced in [rLLM RL Advantage Estimator](rllm-rl-advantage-estimator.md)) to achieve this:

```python
from rllm.experimental.common.config import rLLMAdvantageEstimator

traj_group_adv_estimator_map = {
    "solver": rLLMAdvantageEstimator.GRPO,       # RL estimator
    "validator": rLLMAdvantageEstimator.REINFORCE,  # another RL role (if present)
}

trainer = AgentTrainer(
    workflow_class=SolverJudgeWorkflow,
    ...,
    backend="tinker",
    traj_group_adv_estimator_map=traj_group_adv_estimator_map,
)
```

In this setup:

- `judge` uses your precomputed `step.advantage`
- `solver` and `validator` use estimator-based RL advantages

---

## Responsibility and Sanity Checks

Precompute mode increases flexibility, but correctness becomes your responsibility.
rLLM performs basic validation (type/length checks, fallback warnings), but it cannot
verify whether your mathematical signal is "correct" for your intended algorithm.

Recommended practice:

1. Start with small runs and inspect `advantage/*` metrics.
2. Log representative `step.advantage` samples from workflow.
3. Validate shape/value ranges before scaling experiments.

---

## Related Docs

- [rLLM RL Advantage Estimator](rllm-rl-advantage-estimator.md)
- [Unified Trainer](unified-trainer.md)
- [rLLM and Backend Config](rllm-and-backend-config.md)
