# GSD Legacy Archive

This directory preserves the original GSD implementation that was used in the
first iteration of the project. **It is not imported by the active GSD
pipeline** — the top-level `rllm.experimental.gsd` package now provides a
different algorithm (SFT-style CE + frozen reference teacher + hint GRPO).

## What's in here

| File                 | Legacy name(s) exported from `legacy/__init__.py`        |
|----------------------|----------------------------------------------------------|
| `workflow_topk.py`   | `LegacyGsdWorkflow`, `LegacyGsdConfig`                   |
| `transform_topk.py`  | `legacy_gsd_transform_trajectory_groups_to_datums`, `LEGACY_COMBINED_DISTILL_ROLES` |
| `losses_topk.py`     | `build_combined_gsd_datum`, `build_topk_fkl_datum`, `make_gsd_combined_loss`, `score_teacher_for_response`, `compute_sampled_rkl_advantages`, `compute_topk_rkl_at_position`, `compute_student_logprobs_for_teacher_topk`, `build_legacy_gsd_estimator_map`, `LEGACY_GSD_ADV_ESTIMATOR_MAP` |
| `prompts_legacy.py`  | `legacy_build_hint_prompt`, `legacy_build_student_prompt`, `legacy_build_teacher_prompt`, `legacy_extract_hint`, `LEGACY_HINT_SYSTEM_PROMPT`, `LEGACY_SOLVER_SYSTEM_PROMPT` |

## Why it's kept

1. **Reproducibility** — some experiments in project notebooks reference the
   Top-K CE + custom-loss code path; being able to rerun them requires the
   original code to be importable.
2. **Reference implementation** — it is the only in-tree example of building
   `(N, K+1)` shaped datums for `forward_backward_custom_async`, which is
   useful for anyone experimenting with custom losses.
3. **Unit tests** — `tests/test_gsd_losses.py` exercises these builders and
   is retained as a regression suite.

## Philosophy

- **No modifications.** The files are verbatim copies of the code that used
  to live at `rllm/experimental/gsd/{workflow,transform,losses,prompts}.py`;
  the only edits are the intra-module imports that were redirected from
  `rllm.experimental.gsd.losses` → `rllm.experimental.gsd.legacy.losses_topk`
  (and similarly for `prompts`).
- **No imports from the new code.** Nothing in the active pipeline reaches
  into `legacy/`. If you find yourself wanting to import something from
  here into a new module, consider instead porting the piece you need.
- **Explicit names.** Public symbols are re-exported under `Legacy…` / 
  `legacy_…` prefixes so that any caller has to acknowledge they are using
  the archive.
