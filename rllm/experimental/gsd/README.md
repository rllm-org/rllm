# Generalized Self-Distillation (GSD)

GSD is a post-training algorithm that constructs a **hint-conditioned pseudo-teacher** from the model's own accumulated experience and distills its improved behavior back into the student policy. Unlike existing self-distillation methods (OPSD, SDPO) that condition the teacher on ground-truth solutions, GSD's teacher uses **solution-independent** strategic hints -- avoiding epistemic suppression and information leakage.

## Algorithm Overview

Each training step processes a batch of problems. For each problem:

```
                     question
                        |
                  [1. Self-Hinting]
                   generate hint z
                        |
            +-----------+-----------+
            |                       |
     [2a. Student]           [2b. Teacher]
     N rollouts              N rollouts
     (no hint)               (with hint z)
            |                       |
            +-----------+-----------+
                        |
                  [3. Gating]
              R_T_avg >= R_S_avg ?
                   /         \
                 YES          NO
                  |            |
          [Case 1: Distill]  [Case 2: GRPO]
          on-policy (RKL)    standard RL on
          + supervised (FKL) student rollouts
```

### Case 1: Teacher Valid (Distillation)

Two complementary loss terms, following the CE + IS decomposition from [SDPO](https://arxiv.org/abs/2601.20802):

**On-policy distillation (reverse KL)** -- Student's own rollouts scored under the teacher. Per-token advantage = `kl_coeff * (teacher_lp - student_lp)`, trained via `importance_sampling`. This is a REINFORCE proxy for reverse KL that provides mode-seeking pressure.

**Supervised distillation (forward KL)** -- Teacher's correct rollouts with the hint removed from the prompt. The teacher's Top-K token distribution serves as soft targets via `cross_entropy`. This provides dense, multi-token distributional signal.

### Case 2: Teacher Invalid (GRPO Fallback)

When the hint doesn't help (teacher solves fewer problems), fall back to standard GRPO on the student's rollouts. The teacher's rollouts are discarded.

### Experience Buffer (Phase 2, not yet implemented)

Cross-problem strategic memory via embedding-based retrieval. Successful hints are stored and retrieved for similar future problems.

### Meta-RL on Hints (Phase 3, not yet implemented)

Train the hinting process itself via REINFORCE, using teacher improvement as the meta-reward.

## File Structure

```
rllm/experimental/gsd/
|-- __init__.py       Public API exports
|-- workflow.py       GsdConfig + GsdWorkflow (the main training loop)
|-- losses.py         Loss utilities: teacher scoring, reverse KL advantages,
|                     forward KL datum builder, estimator map
|-- transform.py      Custom trajectory-to-datum transform that dispatches
|                     CE datums (supervised) vs IS datums (on-policy/GRPO)
|-- prompts.py        Prompt templates: hint, teacher, student, extraction
```

### `workflow.py`

The central module. `GsdWorkflow` subclasses `Workflow` and implements:

- `run()` -- dispatches to `_do_training()` or `_do_validation()`
- `_do_training()` -- the 4-phase loop (hint -> rollouts -> gate -> route)
- `_do_validation()` -- N_val student + N_val teacher rollouts with pass@1 metrics
- `_case1_distillation()` -- teacher scoring + advantage computation + trajectory routing
- `_case2_grpo_fallback()` -- names student trajectories for GRPO

`GsdConfig` holds all hyperparameters: `N`, `N_val`, `distill_topk`, `kl_coeff`, etc.

### `losses.py`

Loss computation primitives, designed to mirror the [SDPO recipe](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/sdpo):

- `score_teacher_for_response()` -- single `sample_async` call that returns both Top-K logprobs (for CE) and scalar logprobs (for IS) via teacher-forcing
- `compute_sampled_rkl_advantages()` -- wraps `compute_distill_reverse_kl` for the on-policy loss
- `build_topk_fkl_datum()` -- builds a `tinker.Datum` with `(N, K)` shaped soft targets for `cross_entropy`
- `DEFAULT_GSD_ADV_ESTIMATOR_MAP` -- per-role loss routing: `gsd_distill_onpolicy` -> `importance_sampling`, `gsd_distill_supervised` -> `cross_entropy`, `gsd_student` -> `importance_sampling` (GRPO)

### `transform.py`

`gsd_transform_trajectory_groups_to_datums()` -- dispatches to `build_topk_fkl_datum()` for the `gsd_distill_supervised` role and to the standard `trajectory_to_datums()` for everything else. Must be passed as `transform_fn` to `AgentTrainer`.

### `prompts.py`

Four prompt builders and a hint extractor:

- `build_hint_prompt(question, experiences)` -- strategist prompt with optional past-experience context
- `build_teacher_prompt(question, hint)` -- solver system prompt + hint + problem
- `build_student_prompt(question)` -- solver system prompt + problem (no hint)
- `extract_hint(response)` -- parses `<hint>...</hint>` tags

## Key Design Decisions

**Why two separate losses instead of a combined custom loss?** SDPO uses `forward_backward_custom` with a single Python loss function. We use rLLM's per-role loss routing (`estimator_map` + `loss_fn_map`) to dispatch separate `forward_backward_async` calls. This is slightly less efficient (~2x forward passes instead of 1.5x) but integrates cleanly with rLLM's existing multi-role training infrastructure.

**Why `group_size=1`?** GSD manages its own N rollouts per task internally. The Tinker `training.group_size` must be 1 so the engine dispatches one task at a time to the workflow. This is forced in the entry point via `OmegaConf.update`.

**Why `use_precomputed_advantage=true`?** The on-policy distillation role (`gsd_distill_onpolicy`) has advantages pre-computed in the workflow via `compute_sampled_rkl_advantages`. The advantage pipeline must respect these rather than overwriting with GRPO.
