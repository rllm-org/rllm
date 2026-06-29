"""Unified custom policy-loss abstraction across the verl / tinker / fireworks backends.

A *loss term* is a per-token function over a normalized :class:`LossContext`. The
**same** term runs in-process under verl and inside ``forward_backward_custom`` on
tinker / fireworks (those two share one mechanism). The total objective is a sum of
coefficient-weighted terms::

    L = sum_i  coef_i * Agg_mask_i( term_i(ctx) )

This subsumes the older auxiliary-loss concept (see ``aux_loss.py``): an "aux" loss
(e.g. ECHO) is just a term over the observation mask. ``AuxiliaryLoss`` survives as a
thin declarative shorthand that compiles down to a term.

Extensibility (works for blackbox ``pip install rllm`` users) — same decorator style as
``@rllm.rollout`` / ``@rllm.evaluator``:

    import rllm

    @rllm.register_loss("my_dppo")
    def my_dppo(ctx: rllm.LossContext):
        ratio = (ctx.pi - ctx.mu).exp()
        ...
        return per_token_loss, ctx.action_mask, {"metric": ...}

(``rllm.trainer.algorithms.register_loss`` / ``LossContext`` are the same objects.)

then select it in config without editing rllm::

    algorithm:
      loss_plugins: ["my_pkg.my_losses"]    # imported at startup -> fires @register_loss
      losses:
        - {type: my_dppo, coef: 1.0}
        - {type: env_prediction, coef: 0.05}   # ECHO, just another term

A term returns ``(per_token_loss, agg_mask, metrics)``:

* ``per_token_loss`` — tensor shaped like ``ctx.pi`` (it may already fold in an
  internal selector such as DPPO's divergence mask).
* ``agg_mask`` — which tokens this term is aggregated over (``ctx.action_mask`` for
  policy-gradient terms, ``ctx.obs_mask`` for ECHO).
* ``metrics`` — scalar dict for logging.

Each backend adapter applies its own (proven) aggregation/normalization to
``per_token_loss`` under ``agg_mask``; this module owns only the math and the
registry, and stays import-light (torch is imported lazily inside terms).
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:  # avoid a hard torch import at module load (registry/config stay torch-free)
    import torch

logger = logging.getLogger(__name__)


@dataclass
class LossContext:
    """Normalized per-token inputs a loss term sees, identical across backends.

    On verl the tensors are 2-D ``(batch, response_len)``; on the
    ``forward_backward_custom`` path they are 1-D ``(num_tokens,)`` per datum and the
    adapter loops. Terms are written shape-agnostically (elementwise math + masks),
    so the same function works in both.

    Attributes:
        pi: current-policy per-token log-probs (``requires_grad=True``). The only
            differentiable input — gradients flow through this.
        mu: behavior/old-policy per-token log-probs (the importance-ratio denominator).
            On verl this is ``old_log_prob``; on tinker/fireworks it is the sampling
            (inference) log-probs by default, or proximal log-probs — see ``mu_source``.
        advantages: per-token advantage estimates.
        action_mask: 1.0 on assistant/action tokens (what the policy gradient trains).
        obs_mask: 1.0 on environment-observation tokens (what ECHO trains).
        ref: reference-policy per-token log-probs for a KL term, or None.
        params: loss hyperparameters (e.g. ``delta``/``eps_clip``, ``kl_beta``).
        backend: "verl" | "tinker" | "fireworks" (for backend-aware metrics only).
    """

    pi: "torch.Tensor"
    mu: "torch.Tensor"
    advantages: "torch.Tensor"
    action_mask: "torch.Tensor"
    obs_mask: "torch.Tensor"
    ref: Optional["torch.Tensor"] = None
    params: dict[str, Any] = field(default_factory=dict)
    backend: str = ""


# A loss term: (ctx) -> (per_token_loss, agg_mask, metrics)
LossTerm = Callable[[LossContext], "tuple[torch.Tensor, torch.Tensor, dict[str, float]]"]

RLLM_LOSS_REGISTRY: dict[str, LossTerm] = {}


def register_loss(name: str) -> Callable[[LossTerm], LossTerm]:
    """Register a loss term under ``name`` (its config ``type``).

    Public API for users who install rllm as a package: define a term function and
    decorate it, then reference ``name`` from ``algorithm.losses`` in config. Use
    ``algorithm.loss_plugins`` to have rllm import the defining module at startup.
    """

    def deco(fn: LossTerm) -> LossTerm:
        if name in RLLM_LOSS_REGISTRY and RLLM_LOSS_REGISTRY[name] is not fn:
            logger.warning("Overriding already-registered loss term %r", name)
        RLLM_LOSS_REGISTRY[name] = fn
        return fn

    return deco


def get_loss(name: str) -> LossTerm:
    if name not in RLLM_LOSS_REGISTRY:
        raise ValueError(f"Unknown loss term '{name}'. Registered: {sorted(RLLM_LOSS_REGISTRY)}. Register one with @register_loss, and list its module under algorithm.loss_plugins.")
    return RLLM_LOSS_REGISTRY[name]


def is_custom_loss(name: str | None) -> bool:
    """True if ``name`` is handled by the unified (rLLM) loss path rather than a
    backend-native kernel (verl ``vanilla``/``gspo``, tinker ``ppo``, fireworks ``grpo``…)."""
    return name is not None and name in RLLM_LOSS_REGISTRY


def load_loss_plugins(modules: list[str]) -> None:
    """Import each module so its ``@register_loss`` decorators run. Idempotent."""
    for mod in modules or []:
        try:
            importlib.import_module(mod)
            logger.info("Loaded loss plugin module %r", mod)
        except Exception as e:  # surface a clear, actionable error
            raise ImportError(f"Failed to import loss plugin module {mod!r} (algorithm.loss_plugins). Is it installed/importable on this process (and on verl Ray workers)?") from e


@dataclass
class ResolvedTerm:
    """A loss term resolved from config, ready for a backend adapter to evaluate."""

    name: str
    fn: LossTerm
    coef: float
    params: dict[str, Any]


def resolve_loss_terms(algorithm_config) -> list[ResolvedTerm]:
    """Resolve the configured unified loss terms from an ``AlgorithmConfig``.

    Precedence:
      1. ``algorithm.losses`` — explicit ``[{type, coef, ...params}]`` list (front door).
      2. otherwise, back-compat: the main ``loss_fn`` (if rLLM-registered) at coef 1.0,
         plus ``aux_losses`` / ``env_loss_coef`` (ECHO) as additional terms.

    Returns an empty list when no rLLM-registered loss is selected (caller then uses
    the backend-native path). First imports ``algorithm.loss_plugins`` so user terms
    are registered before resolution.
    """
    load_loss_plugins(list(getattr(algorithm_config, "loss_plugins", None) or []))

    base_params = {
        "eps_clip": getattr(algorithm_config, "eps_clip", 0.2),
        "eps_clip_high": getattr(algorithm_config, "eps_clip_high", None),
        "kl_beta": getattr(algorithm_config, "kl_beta", 0.0),
    }

    explicit = list(getattr(algorithm_config, "losses", None) or [])
    terms: list[ResolvedTerm] = []

    if explicit:
        for spec in explicit:
            spec = dict(spec)
            name = spec.pop("type", None)
            if name is None:
                raise ValueError(f"algorithm.losses entry is missing 'type': {spec}")
            coef = float(spec.pop("coef", 1.0))
            params = {**base_params, **spec}
            terms.append(ResolvedTerm(name=name, fn=get_loss(name), coef=coef, params=params))
        return terms

    # Back-compat path: only engaged when the main loss_fn is an rLLM term.
    main = getattr(algorithm_config, "loss_fn", None)
    if not is_custom_loss(main):
        return []
    terms.append(ResolvedTerm(name=main, fn=get_loss(main), coef=1.0, params=dict(base_params)))

    # Compose configured auxiliary losses (ECHO, ...) as terms over the same registry.
    from rllm.trainer.algorithms.aux_loss import build_aux_losses

    for aux in build_aux_losses(algorithm_config):
        if aux.name not in RLLM_LOSS_REGISTRY:
            logger.warning("Aux loss %r has no registered term; skipping on the unified loss path", aux.name)
            continue
        terms.append(ResolvedTerm(name=aux.name, fn=get_loss(aux.name), coef=aux.coef, params={**base_params, **getattr(aux, "cfg", {})}))
    return terms


# ---------------------------------------------------------------------------
# Built-in loss terms. Math kept identical to verl 0.8's POLICY_LOSS_REGISTRY so a
# loss runs the same whether on verl-native kernels or the forward_backward_custom
# path. See https://arxiv.org/pdf/2602.04879 (DPPO) and arXiv:2605.24517 (ECHO).
# ---------------------------------------------------------------------------
_RATIO_CLAMP = 20.0


def _ratio(ctx: LossContext):
    import torch

    return torch.exp(torch.clamp(ctx.pi - ctx.mu, min=-_RATIO_CLAMP, max=_RATIO_CLAMP))


def _truncated_is(ratio, params):
    """Truncated importance-sampling surrogate weight (detached). Loss uses
    ``ratio.detach() * logprob`` so the gradient equals the policy gradient."""
    import torch

    cap = params.get("clip_ratio_c", _RATIO_CLAMP)
    return torch.clamp(ratio, max=cap).detach()


@register_loss("dppo_tv")
def dppo_tv(ctx: LossContext):
    """DPPO with a binary total-variation divergence mask (Eq. 12, TV variant).

    Replaces PPO ratio-clipping with a per-token mask: zero the gradient only when the
    update pushes the token *away* from the behavior policy AND the TV divergence
    ``|exp(pi) - exp(mu)|`` already exceeds ``delta``. ``delta`` defaults to ``eps_clip``.
    """
    import torch

    delta = float(ctx.params.get("delta", ctx.params.get("eps_clip", 0.2)))
    delta_lo = float(ctx.params.get("delta_low", delta))
    delta_hi = float(ctx.params.get("delta_high", delta))
    tr = _truncated_is(_ratio(ctx), ctx.params)
    pi_p, mu_p = ctx.pi.exp(), ctx.mu.exp()
    keep = torch.where(ctx.advantages > 0, (pi_p - mu_p) <= delta_hi, (pi_p - mu_p) >= -delta_lo)
    keep = keep.detach().to(ctx.pi.dtype)
    per_token = -ctx.advantages * tr * ctx.pi * keep
    am = ctx.action_mask
    denom = am.sum().clamp(min=1.0)
    metrics = {"dppo_tv/mask_frac": ((1.0 - keep) * am).sum().div(denom).item()}
    return per_token, am, metrics


@register_loss("dppo_kl")
def dppo_kl(ctx: LossContext):
    """DPPO with a binary-KL divergence mask (Eq. 12, KL variant).

    Bernoulli-KL between behavior ``mu`` and current ``pi`` at the sampled token:
    ``D = q·log(q/p) + (1-q)·log((1-q)/(1-p))`` with ``q=exp(mu)``, ``p=exp(pi)``.
    Mask when moving away from behavior and ``D > delta``.
    """
    import torch

    delta = float(ctx.params.get("delta", ctx.params.get("eps_clip", 0.2)))
    eps = 1e-6
    tr = _truncated_is(_ratio(ctx), ctx.params)
    p = ctx.pi.exp().clamp(eps, 1.0 - eps)
    q = ctx.mu.exp().clamp(eps, 1.0 - eps)
    d_kl = q * (q / p).log() + (1.0 - q) * ((1.0 - q) / (1.0 - p)).log()
    moving_away = torch.where(ctx.advantages > 0, p > q, p < q)
    keep = ~(moving_away & (d_kl > delta))
    keep = keep.detach().to(ctx.pi.dtype)
    per_token = -ctx.advantages * tr * ctx.pi * keep
    am = ctx.action_mask
    denom = am.sum().clamp(min=1.0)
    metrics = {"dppo_kl/mask_frac": ((1.0 - keep) * am).sum().div(denom).item()}
    return per_token, am, metrics


@register_loss("ppo_clip")
def ppo_clip(ctx: LossContext):
    """Standard PPO/GRPO clipped surrogate — the baseline DPPO replaces.

    Registered so a full GRPO+ECHO objective is expressible purely as unified terms.
    """
    import torch

    eps = float(ctx.params.get("eps_clip", 0.2))
    eps_hi = ctx.params.get("eps_clip_high")
    eps_hi = float(eps_hi) if eps_hi is not None else eps
    ratio = _ratio(ctx)
    clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps_hi)
    per_token = -torch.minimum(ratio * ctx.advantages, clipped * ctx.advantages)
    am = ctx.action_mask
    denom = am.sum().clamp(min=1.0)
    metrics = {"ppo/clip_frac": ((clipped != ratio).to(ctx.pi.dtype) * am).sum().div(denom).item()}
    return per_token, am, metrics


@register_loss("env_prediction")
def env_prediction(ctx: LossContext):
    """ECHO (arXiv:2605.24517): cross-entropy on environment-observation tokens.

    The aux-loss family unified into a term: uniform-weight ``-log p_theta`` over the
    observation mask. No advantage, no ratio.
    """
    return -ctx.pi, ctx.obs_mask, {}
