"""Auxiliary token-level losses added on top of the policy-gradient loss.

See ``design/auxiliary-losses.md``. A growing family of agent-RL algorithms keep
GRPO as the optimization backbone and add a token-level auxiliary loss of the
form ``coef * Agg_mask(weight_t * [-log p_theta(token_t)])`` — e.g. ECHO (predict
environment-observation tokens) and SDAR (gated on-policy self-distillation).

An :class:`AuxiliaryLoss` declares *what* to add (a named token subset, a
coefficient, and an optional dynamic per-token weight). Backend executors decide
*how* to compute it:

* verl folds the term into the single existing forward/backward pass
  (``rllm.trainer.verl.verl_backend.CustomPPOLoss``).
* tinker / fireworks submit an extra gradient-accumulated ``cross_entropy`` pass
  (``rllm.trainer.tinker.aux_loss``).

This module is backend-agnostic (no torch / tinker imports): it only defines the
spec, the registry, and the config plumbing.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Named token subsets. Resolved to concrete masks by each backend executor
# (verl: tensor masks from response_mask/responses; tinker/fireworks: the datum
# ``mask`` field). Defining the names here keeps the vocabulary in one place.
# ---------------------------------------------------------------------------
MASK_ACTION = "action"  # assistant/action tokens (what GRPO trains)
MASK_OBSERVATION = "observation"  # environment-observation tokens (tool/terminal output)
_KNOWN_MASKS = {MASK_ACTION, MASK_OBSERVATION}


AUX_LOSS_REGISTRY: dict[str, type[AuxiliaryLoss]] = {}


def register_aux_loss(name: str):
    """Register an :class:`AuxiliaryLoss` subclass under ``name`` (the config ``type``)."""

    def deco(cls: type[AuxiliaryLoss]) -> type[AuxiliaryLoss]:
        cls.name = name
        AUX_LOSS_REGISTRY[name] = cls
        return cls

    return deco


def get_aux_loss(name: str) -> type[AuxiliaryLoss]:
    if name not in AUX_LOSS_REGISTRY:
        raise ValueError(f"Unknown auxiliary loss '{name}'. Registered: {sorted(AUX_LOSS_REGISTRY)}. Register one with @register_aux_loss.")
    return AUX_LOSS_REGISTRY[name]


class AuxiliaryLoss:
    """Declarative spec for a token-level auxiliary loss.

    Subclasses set class attributes and (optionally) override :meth:`weight`.
    The contributed loss is ``coef * Agg_mask(weight_t * [-log p_theta(token_t)])``
    over the tokens selected by ``mask``; ``token_t`` is the rollout's own token
    at each masked position (already scored by the forward pass).

    Attributes:
        name: registry key (set by :func:`register_aux_loss`).
        mask: which token subset the loss applies to (``MASK_*``).
        requires: declared auxiliary forward passes (e.g. SDAR's teacher branch).
            Not yet implemented — a non-empty value raises (design step 4).
    """

    name: str = ""
    mask: str = MASK_OBSERVATION
    requires: tuple = ()

    def __init__(self, coef: float, **cfg):
        self.coef = float(coef)
        self.cfg = cfg
        if self.mask not in _KNOWN_MASKS:
            raise ValueError(f"Auxiliary loss '{self.name or type(self).__name__}' has unknown mask '{self.mask}' (known: {sorted(_KNOWN_MASKS)})")
        if self.requires:
            # AuxForward (dynamic-weight losses needing an extra teacher/privileged
            # forward pass, e.g. SDAR) is the next milestone; see the design doc.
            raise NotImplementedError(f"Auxiliary loss '{self.name}' declares requires={self.requires!r}, but AuxForward is not implemented yet (design step 4).")

    def weight(self, ctx) -> None:
        """Per-token weight tensor, or ``None`` for uniform (coef-only) weighting.

        ``ctx`` is the backend-specific context (forward-pass log-probs, entropy,
        masks). Returning ``None`` — the default — means every masked token is
        weighted equally, i.e. plain coef-scaled cross-entropy (ECHO). Dynamic
        weights (e.g. SDAR's detached sigmoid gate) override this.
        """
        return None


@register_aux_loss("env_prediction")
class EnvPredictionLoss(AuxiliaryLoss):
    """ECHO (arXiv:2605.24517): length-normalized cross-entropy on environment
    observation tokens. Reuses the existing forward pass; uniform weight."""

    mask = MASK_OBSERVATION


def build_aux_losses(algorithm_config) -> list[AuxiliaryLoss]:
    """Instantiate the auxiliary losses configured on an ``AlgorithmConfig``.

    Reads ``algorithm_config.aux_losses`` (a list of ``{"type", "coef", ...}``
    specs). For backward compatibility, a positive ``algorithm_config.env_loss_coef``
    is treated as shorthand for an ``env_prediction`` aux loss (this is how
    ``adv_estimator=echo`` enables ECHO), unless ``env_prediction`` is already
    listed explicitly.
    """
    specs = list(getattr(algorithm_config, "aux_losses", None) or [])
    losses: list[AuxiliaryLoss] = []
    seen_types: set[str] = set()
    for spec in specs:
        spec = dict(spec)
        loss_type = spec.pop("type", None)
        if loss_type is None:
            raise ValueError(f"aux_losses entry is missing 'type': {spec}")
        coef = spec.pop("coef", None)
        if coef is None:
            raise ValueError(f"aux_losses entry '{loss_type}' is missing 'coef'")
        losses.append(get_aux_loss(loss_type)(coef=coef, **spec))
        seen_types.add(loss_type)

    # Back-compat sugar: env_loss_coef == an env_prediction aux loss.
    env_coef = float(getattr(algorithm_config, "env_loss_coef", 0.0) or 0.0)
    if env_coef > 0.0 and "env_prediction" not in seen_types:
        losses.append(EnvPredictionLoss(coef=env_coef))

    if losses:
        logger.info("Auxiliary losses enabled: %s", [(loss.name, loss.coef) for loss in losses])
    return losses
