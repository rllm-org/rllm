"""Legacy GSD implementation — top-K CE + combined custom loss.

This subpackage preserves the original GSD approach that used:

* **Teacher Top-K scoring** via ``sample_async`` + ``topk_prompt_logprobs``
  to obtain Top-K token probabilities at every response position.
* **Combined CE + IS datum** with ``(N, K+1)`` shaped ``target_tokens``
  (columns 0..K-1 for CE, column K for IS).
* **Custom loss function** invoked through ``forward_backward_custom_async``
  that operated on both terms in a single forward pass.
* **Moving teacher** — the teacher was the live training model conditioned
  on the hint (logprobs drift every training step).

This path is **no longer used by the active GSD pipeline** (see the top-level
:mod:`rllm.experimental.gsd` modules for the current SFT-style CE + frozen
reference teacher + hint-GRPO architecture).  It is kept here for
reproducibility and reference, and so that existing unit tests can continue
to exercise the original behaviour.

All public symbols are re-exported under explicit ``Legacy``-prefixed names
so that importing code has to acknowledge it is reaching into the archive.
"""

from rllm.experimental.gsd.legacy.losses_topk import (
    DEFAULT_GSD_ADV_ESTIMATOR_MAP as LEGACY_GSD_ADV_ESTIMATOR_MAP,
)
from rllm.experimental.gsd.legacy.losses_topk import (
    build_combined_gsd_datum,
    build_topk_fkl_datum,
    compute_sampled_rkl_advantages,
    compute_student_logprobs_for_teacher_topk,
    compute_topk_rkl_at_position,
    make_gsd_combined_loss,
    score_teacher_for_response,
)
from rllm.experimental.gsd.legacy.losses_topk import (
    build_gsd_estimator_map as build_legacy_gsd_estimator_map,
)
from rllm.experimental.gsd.legacy.prompts_legacy import (
    HINT_SYSTEM_PROMPT as LEGACY_HINT_SYSTEM_PROMPT,
)
from rllm.experimental.gsd.legacy.prompts_legacy import (
    SOLVER_SYSTEM_PROMPT as LEGACY_SOLVER_SYSTEM_PROMPT,
)
from rllm.experimental.gsd.legacy.prompts_legacy import (
    build_hint_prompt as legacy_build_hint_prompt,
)
from rllm.experimental.gsd.legacy.prompts_legacy import (
    build_student_prompt as legacy_build_student_prompt,
)
from rllm.experimental.gsd.legacy.prompts_legacy import (
    build_teacher_prompt as legacy_build_teacher_prompt,
)
from rllm.experimental.gsd.legacy.prompts_legacy import (
    extract_hint as legacy_extract_hint,
)
from rllm.experimental.gsd.legacy.transform_topk import (
    COMBINED_DISTILL_ROLES as LEGACY_COMBINED_DISTILL_ROLES,
)
from rllm.experimental.gsd.legacy.transform_topk import (
    gsd_transform_trajectory_groups_to_datums as legacy_gsd_transform_trajectory_groups_to_datums,
)
from rllm.experimental.gsd.legacy.workflow_topk import (
    GsdConfig as LegacyGsdConfig,
)
from rllm.experimental.gsd.legacy.workflow_topk import (
    GsdWorkflow as LegacyGsdWorkflow,
)

__all__ = [
    # Maps & workflow
    "LEGACY_GSD_ADV_ESTIMATOR_MAP",
    "LEGACY_COMBINED_DISTILL_ROLES",
    "LegacyGsdConfig",
    "LegacyGsdWorkflow",
    "build_legacy_gsd_estimator_map",
    "legacy_gsd_transform_trajectory_groups_to_datums",
    # Loss primitives (kept under their original names since tests reference them)
    "build_combined_gsd_datum",
    "build_topk_fkl_datum",
    "compute_sampled_rkl_advantages",
    "compute_student_logprobs_for_teacher_topk",
    "compute_topk_rkl_at_position",
    "make_gsd_combined_loss",
    "score_teacher_for_response",
    # Prompts
    "LEGACY_HINT_SYSTEM_PROMPT",
    "LEGACY_SOLVER_SYSTEM_PROMPT",
    "legacy_build_hint_prompt",
    "legacy_build_student_prompt",
    "legacy_build_teacher_prompt",
    "legacy_extract_hint",
]
