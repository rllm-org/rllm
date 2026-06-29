"""Managed-backend executor for token-level auxiliary losses (tinker / fireworks).

Both managed backends compute their primary loss in a fixed server-side kernel,
so an auxiliary loss (see ``rllm.trainer.algorithms.aux_loss`` and
``design/auxiliary-losses.md``) is expressed as an extra gradient-accumulated
``cross_entropy`` pass over the same rollout datums, weighting the selected token
subset. Tinker's ``cross_entropy`` loss is ``-sum_t weights_t * logprob(target_t)``,
so weighting position ``t`` by ``w_t`` contributes ``w_t * (-logprob(target_t))``
to the (accumulated) gradient.

This module holds the parts shared by tinker and fireworks: the token-subset
selector and the datum builder. The per-token weight *scale* differs by backend
normalization (tinker accumulates raw; fireworks folds normalization into the
weights and disables grad-accumulation normalization) and is supplied by the
caller.
"""

from __future__ import annotations

import tinker
from tinker.types.tensor_data import TensorData

from rllm.trainer.algorithms.loss import MASK_ACTION, MASK_OBSERVATION


def aux_positions(term, mask_values) -> list[bool]:
    """Boolean per-token selector for an additive term from a datum's action mask.

    ``term`` is a ``ResolvedTerm`` whose ``.mask`` names its static token region. The
    datum ``mask`` is ``1.0`` on action tokens and ``0.0`` on environment observation
    tokens (built by ``rllm.trainer.tinker.transform``).
    """
    is_obs = [float(m) == 0.0 for m in mask_values]
    if term.mask == MASK_OBSERVATION:
        return is_obs
    if term.mask == MASK_ACTION:
        return [not o for o in is_obs]
    raise ValueError(f"Additive term mask {term.mask!r} is not supported on managed (tinker/fireworks) backends; use algorithm.losses (forward_backward_custom) for non-CE terms")


def build_aux_ce_datum(model_input, target_tokens, positions: list[bool], scale) -> tinker.Datum | None:
    """Build a ``cross_entropy`` datum weighting ``positions`` by ``scale``.

    ``scale`` is either a float (the same weight on every selected position) or a
    per-token sequence aligned with ``positions``. Returns ``None`` when no
    position is selected (e.g. a single-turn rollout has no observation tokens),
    so the caller can skip empty datums.
    """
    if not any(positions):
        return None
    if isinstance(scale, int | float):
        weights = [float(scale) if p else 0.0 for p in positions]
    else:
        weights = [float(scale[i]) if p else 0.0 for i, p in enumerate(positions)]
    return tinker.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": target_tokens,
            "weights": TensorData(data=weights, dtype="float32"),
        },
    )
