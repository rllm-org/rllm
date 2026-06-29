"""verl adapter for the unified custom-loss abstraction.

Bridges rLLM loss terms (``rllm.trainer.algorithms.loss``) into verl's in-process
``POLICY_LOSS_REGISTRY``. A registered term becomes selectable as
``algorithm.loss_fn = <name>`` (verl ``loss_mode``), exactly like a verl-native loss.

We never shadow a verl-native kernel: verl 0.8 already ships ``dppo_tv``/``dppo_kl``
(and the rLLM terms reproduce their math), so for those we let verl use its own. The
shim exists so a blackbox user's ``@register_loss`` term also runs on verl — one term
definition, every backend.

Note: verl hands a policy-loss fn already-extracted, padded tensors, so there is no
re-extraction risk here. The shim only sees the *action* tokens (``response_mask``);
observation-token (ECHO) terms are applied via the existing aux-loss executor in
``CustomPPOLoss._apply_aux_losses``, which has the observation mask.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _make_verl_shim(term_fn):
    """Wrap an rLLM loss term as a verl ``PolicyLossFn``."""
    import torch

    from verl.trainer.ppo.core_algos import agg_loss

    from rllm.trainer.algorithms.loss import LossContext

    def loss_fn(old_log_prob, log_prob, advantages, response_mask, loss_agg_mode="token-mean", config=None, rollout_is_weights=None):
        # Hyperparameters follow verl convention: clip_ratio doubles as the divergence
        # threshold delta for DPPO-style terms (see verl's dppo_tv).
        clip = getattr(config, "clip_ratio", 0.2) if config is not None else 0.2
        params = {
            "delta": clip,
            "eps_clip": clip,
            "delta_low": getattr(config, "clip_ratio_low", None) or clip,
            "delta_high": getattr(config, "clip_ratio_high", None) or clip,
            "eps_clip_high": getattr(config, "clip_ratio_high", None),
            "clip_ratio_c": getattr(config, "clip_ratio_c", 20.0) if config is not None else 20.0,
        }
        ctx = LossContext(
            pi=log_prob,
            mu=old_log_prob,
            advantages=advantages,
            action_mask=response_mask.to(log_prob.dtype),
            obs_mask=torch.zeros_like(log_prob),  # not available on the verl policy-loss path
            params=params,
            backend="verl",
        )
        per_token, mask, metrics = term_fn(ctx)
        if rollout_is_weights is not None:
            per_token = per_token * rollout_is_weights
        gbi = getattr(config, "global_batch_info", {}) or {}
        loss = agg_loss(loss_mat=per_token, loss_mask=mask, loss_agg_mode=loss_agg_mode, **gbi)
        return loss, {f"actor/{k}": (v if isinstance(v, float) else float(v)) for k, v in metrics.items()}

    return loss_fn


def register_rllm_terms_into_verl() -> list[str]:
    """Register every rLLM loss term into verl's ``POLICY_LOSS_REGISTRY``.

    Idempotent; never overwrites a verl-native kernel of the same name. Returns the
    list of names actually registered. Call once during ``VerlBackend`` init, before
    ``set_loss_fn`` and before loss-name validation.
    """
    from verl.trainer.ppo.core_algos import POLICY_LOSS_REGISTRY, register_policy_loss

    from rllm.trainer.algorithms.loss import RLLM_LOSS_REGISTRY

    registered: list[str] = []
    for name, term_fn in RLLM_LOSS_REGISTRY.items():
        if name in POLICY_LOSS_REGISTRY:
            continue  # verl-native (e.g. dppo_tv/dppo_kl) wins
        register_policy_loss(name)(_make_verl_shim(term_fn))
        registered.append(name)
    if registered:
        logger.info("Registered rLLM loss terms into verl POLICY_LOSS_REGISTRY: %s", registered)
    return registered
