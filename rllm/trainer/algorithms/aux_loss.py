"""DEPRECATED — the standalone auxiliary-loss framework.

Superseded by the unified loss-term system in ``rllm.trainer.algorithms.loss``. There is
now **one** registry and one decorator:

* ECHO is the ``env_prediction`` term (``@register_loss("env_prediction",
  aux_mask=MASK_OBSERVATION)``) — not a separate ``EnvPredictionLoss`` class.
* Define a custom additive loss with ``@rllm.register_loss(name,
  aux_mask=MASK_OBSERVATION)`` (a cross-entropy term over the observation tokens) and
  select it via ``algorithm.losses`` (or the back-compat ``algorithm.aux_losses`` /
  ``env_loss_coef``).

This module remains as a thin, warning-emitting shim so existing imports and configs keep
working. ``build_aux_losses`` now returns unified ``ResolvedTerm`` additive terms.
"""

from __future__ import annotations

import warnings

from rllm.trainer.algorithms.loss import (  # noqa: F401  (re-exported for back-compat)
    MASK_ACTION,
    MASK_OBSERVATION,
    register_loss,
    resolve_additive_terms,
)

_DEPRECATION = (
    "rllm.trainer.algorithms.aux_loss is deprecated. Use @rllm.register_loss(name, "
    "aux_mask=MASK_OBSERVATION) and algorithm.losses; ECHO is the 'env_prediction' term. "
    "See design/unified-custom-loss.md."
)

# Legacy registry kept for back-compat; the source of truth is loss.RLLM_LOSS_REGISTRY.
AUX_LOSS_REGISTRY: dict[str, type] = {}


def register_aux_loss(name: str):
    """DEPRECATED. Bridges a legacy ``AuxiliaryLoss`` subclass into the unified term
    registry as a uniform cross-entropy term, so old code keeps resolving."""
    warnings.warn(_DEPRECATION, DeprecationWarning, stacklevel=2)

    def deco(cls):
        cls.name = name
        AUX_LOSS_REGISTRY[name] = cls
        mask = getattr(cls, "mask", MASK_OBSERVATION)

        @register_loss(name, aux_mask=mask)
        def _bridged_term(ctx, _mask=mask):
            sel = ctx.obs_mask if _mask == MASK_OBSERVATION else ctx.action_mask
            return -ctx.pi, sel, {}

        return cls

    return deco


def get_aux_loss(name: str):
    """DEPRECATED."""
    warnings.warn(_DEPRECATION, DeprecationWarning, stacklevel=2)
    if name not in AUX_LOSS_REGISTRY:
        raise ValueError(f"Unknown auxiliary loss '{name}'. Use @rllm.register_loss instead.")
    return AUX_LOSS_REGISTRY[name]


class AuxiliaryLoss:
    """DEPRECATED base class. Use ``@rllm.register_loss(name, aux_mask=...)`` instead."""

    name: str = ""
    mask: str = MASK_OBSERVATION
    requires: tuple = ()

    def __init__(self, coef: float, **cfg):
        warnings.warn(_DEPRECATION, DeprecationWarning, stacklevel=2)
        self.coef = float(coef)
        self.cfg = cfg
        if self.mask not in (MASK_ACTION, MASK_OBSERVATION):
            raise ValueError(f"Auxiliary loss '{self.name or type(self).__name__}' has unknown mask '{self.mask}'")
        if self.requires:
            raise NotImplementedError(f"Auxiliary loss '{self.name}' declares requires={self.requires!r}; AuxForward is not implemented.")

    def weight(self, ctx):
        return None


def build_aux_losses(algorithm_config):
    """DEPRECATED alias for :func:`resolve_additive_terms`.

    Returns the additive (ECHO-style) terms configured via ``aux_losses`` /
    ``env_loss_coef`` as unified ``ResolvedTerm`` objects (``.name``, ``.coef``,
    ``.mask``, ``.fn``). Backends consume these directly; new code should call
    ``resolve_additive_terms`` (or use ``algorithm.losses``)."""
    return resolve_additive_terms(algorithm_config)
