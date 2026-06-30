"""Custom policy losses across the verl / tinker / fireworks backends.

Mirrors verl's model: a **single** loss selected by name (``algorithm.loss_fn`` →
verl's ``policy_loss.loss_mode``), not a list. A loss is one function that returns the
**complete** scalar objective and does its own masking + aggregation — exactly like a
verl ``POLICY_LOSS_REGISTRY`` function. There is no separate auxiliary-loss framework:
a loss that wants an extra term (e.g. ECHO's cross-entropy on observation tokens) simply
adds it inside its own body (see ``ppo_clip_env``).

The same function runs in-process under verl and inside ``forward_backward_custom`` on
tinker/fireworks. Each backend injects ``ctx.aggregate(per_token, mask) -> scalar`` (verl:
``agg_loss`` with global-batch normalization; managed: seq-mean-token-mean), so the loss
body is backend-agnostic.

Public API — same decorator style as ``@rllm.rollout`` / ``@rllm.evaluator``:

    import rllm

    @rllm.register_loss("my_dppo")
    def my_dppo(ctx: rllm.LossContext):
        ratio = (ctx.pi - ctx.mu).exp()
        keep = (...).detach()
        pg = -ctx.advantages * ratio.clamp(max=20).detach() * ctx.pi * keep
        return ctx.aggregate(pg, ctx.action_mask), {"mask_frac": ...}

    # config:  algorithm: { loss_fn: my_dppo, loss_params: {delta: 0.2} }
    # for a blackbox install, list the module:  algorithm.loss_plugins: ["my_pkg.losses"]
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:  # avoid a hard torch import at module load
    import torch

logger = logging.getLogger(__name__)


@dataclass
class LossContext:
    """Normalized inputs a loss function sees, identical across backends.

    On verl the tensors are 2-D ``(batch, response_len)``; on the
    ``forward_backward_custom`` path they are 1-D ``(num_tokens,)`` per datum (the adapter
    loops). Loss bodies are written shape-agnostically (elementwise math + ``aggregate``).

    Attributes:
        pi: current-policy per-token log-probs (``requires_grad=True``) — the only
            differentiable input.
        mu: behavior/old-policy per-token log-probs (importance-ratio denominator). verl:
            ``old_log_probs``; managed: sampling (inference) log-probs by default.
        advantages: per-token advantage estimates.
        action_mask: 1.0 on assistant/action tokens (the policy gradient).
        obs_mask: 1.0 on environment-observation tokens (e.g. for ECHO).
        aggregate: ``(per_token_loss, mask) -> scalar`` reducer injected by the backend.
        ref: reference-policy log-probs for a KL term, or None.
        params: loss hyperparameters (``delta``/``eps_clip``, ``env_loss_coef``, ...).
        backend: "verl" | "tinker" | "fireworks".
    """

    pi: "torch.Tensor"
    mu: "torch.Tensor"
    advantages: "torch.Tensor"
    action_mask: "torch.Tensor"
    obs_mask: "torch.Tensor"
    aggregate: Callable[["torch.Tensor", "torch.Tensor"], "torch.Tensor"]
    ref: Optional["torch.Tensor"] = None
    params: dict[str, Any] = field(default_factory=dict)
    backend: str = ""


# A loss: (ctx) -> (scalar_loss, metrics)
LossFn = Callable[[LossContext], "tuple[torch.Tensor, dict[str, float]]"]

RLLM_LOSS_REGISTRY: dict[str, LossFn] = {}


def register_loss(name: str) -> Callable[[LossFn], LossFn]:
    """Register a loss under ``name`` (its ``algorithm.loss_fn`` value).

    Public API for blackbox ``pip install rllm`` users: decorate a function and select it
    by name. Use ``algorithm.loss_plugins`` to have rllm import the defining module at
    startup so the decorator runs.
    """

    def deco(fn: LossFn) -> LossFn:
        if name in RLLM_LOSS_REGISTRY and RLLM_LOSS_REGISTRY[name] is not fn:
            logger.warning("Overriding already-registered loss %r", name)
        RLLM_LOSS_REGISTRY[name] = fn
        return fn

    return deco


def get_loss(name: str) -> LossFn:
    if name not in RLLM_LOSS_REGISTRY:
        raise ValueError(f"Unknown loss {name!r}. Registered: {sorted(RLLM_LOSS_REGISTRY)}. Register one with @rllm.register_loss and list its module under algorithm.loss_plugins.")
    return RLLM_LOSS_REGISTRY[name]


def is_custom_loss(name: str | None) -> bool:
    """True if ``name`` is an rLLM loss (vs a backend-native one like verl ``vanilla``)."""
    return name is not None and name in RLLM_LOSS_REGISTRY


def load_loss_plugins(modules: list[str]) -> None:
    """Import each module so its ``@register_loss`` decorators run. Idempotent."""
    for mod in modules or []:
        try:
            importlib.import_module(mod)
            logger.info("Loaded loss plugin module %r", mod)
        except Exception as e:
            raise ImportError(f"Failed to import loss plugin module {mod!r} (algorithm.loss_plugins). Importable on this process (and on verl Ray workers)?") from e


@dataclass
class ResolvedLoss:
    """The single loss selected from config, ready for a backend to run."""

    name: str
    fn: LossFn
    params: dict[str, Any]


def resolve_loss(algorithm_config) -> ResolvedLoss | None:
    """Resolve ``algorithm.loss_fn`` to an rLLM loss, or None for a backend-native loss.

    First imports ``algorithm.loss_plugins`` so user losses are registered. Params passed
    to the loss are the standard clip/kl fields plus ``env_loss_coef`` and anything under
    ``algorithm.loss_params`` (verl-style loss-specific config)."""
    load_loss_plugins(list(getattr(algorithm_config, "loss_plugins", None) or []))
    name = getattr(algorithm_config, "loss_fn", None)
    if not is_custom_loss(name):
        return None
    params = {
        "eps_clip": getattr(algorithm_config, "eps_clip", 0.2),
        "eps_clip_high": getattr(algorithm_config, "eps_clip_high", None),
        "kl_beta": getattr(algorithm_config, "kl_beta", 0.0),
        "env_loss_coef": float(getattr(algorithm_config, "env_loss_coef", 0.0) or 0.0),
        **dict(getattr(algorithm_config, "loss_params", None) or {}),
    }
    return ResolvedLoss(name=name, fn=get_loss(name), params=params)


# ---------------------------------------------------------------------------
# Built-in losses. Math kept identical to verl 0.8's POLICY_LOSS_REGISTRY so a loss runs
# the same on verl-native kernels or the forward_backward_custom path. Each returns a
# scalar (own aggregation via ctx.aggregate) — the verl convention.
# See https://arxiv.org/pdf/2602.04879 (DPPO) and arXiv:2605.24517 (ECHO).
# ---------------------------------------------------------------------------
_RATIO_CLAMP = 20.0


def _ratio(ctx: LossContext):
    import torch

    return torch.exp(torch.clamp(ctx.pi - ctx.mu, min=-_RATIO_CLAMP, max=_RATIO_CLAMP))


def _truncated_is(ratio, params):
    import torch

    return torch.clamp(ratio, max=params.get("clip_ratio_c", _RATIO_CLAMP)).detach()


@register_loss("ppo_clip")
def ppo_clip(ctx: LossContext):
    """Standard PPO/GRPO clipped surrogate."""
    import torch

    eps = float(ctx.params.get("eps_clip", 0.2))
    eps_hi = ctx.params.get("eps_clip_high")
    eps_hi = float(eps_hi) if eps_hi is not None else eps
    ratio = _ratio(ctx)
    clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps_hi)
    pg = -torch.minimum(ratio * ctx.advantages, clipped * ctx.advantages)
    am = ctx.action_mask
    clip_frac = ((clipped != ratio).to(ctx.pi.dtype) * am).sum() / am.sum().clamp(min=1.0)
    return ctx.aggregate(pg, am), {"ppo/clip_frac": clip_frac.item()}


@register_loss("dppo_tv")
def dppo_tv(ctx: LossContext):
    """DPPO with a binary total-variation divergence mask (Eq. 12, TV variant).

    Replaces PPO ratio-clipping: zero the gradient only when the update pushes the token
    *away* from the behavior policy AND ``|exp(pi)-exp(mu)|`` exceeds ``delta`` (defaults
    to ``eps_clip``)."""
    import torch

    delta = float(ctx.params.get("delta", ctx.params.get("eps_clip", 0.2)))
    delta_lo = float(ctx.params.get("delta_low", delta))
    delta_hi = float(ctx.params.get("delta_high", delta))
    tr = _truncated_is(_ratio(ctx), ctx.params)
    pi_p, mu_p = ctx.pi.exp(), ctx.mu.exp()
    keep = torch.where(ctx.advantages > 0, (pi_p - mu_p) <= delta_hi, (pi_p - mu_p) >= -delta_lo).detach().to(ctx.pi.dtype)
    pg = -ctx.advantages * tr * ctx.pi * keep
    am = ctx.action_mask
    mask_frac = ((1.0 - keep) * am).sum() / am.sum().clamp(min=1.0)
    return ctx.aggregate(pg, am), {"dppo_tv/mask_frac": mask_frac.item()}


@register_loss("dppo_kl")
def dppo_kl(ctx: LossContext):
    """DPPO with a binary-KL divergence mask (Eq. 12, KL variant)."""
    import torch

    delta = float(ctx.params.get("delta", ctx.params.get("eps_clip", 0.2)))
    eps = 1e-6
    tr = _truncated_is(_ratio(ctx), ctx.params)
    p = ctx.pi.exp().clamp(eps, 1.0 - eps)
    q = ctx.mu.exp().clamp(eps, 1.0 - eps)
    d_kl = q * (q / p).log() + (1.0 - q) * ((1.0 - q) / (1.0 - p)).log()
    moving_away = torch.where(ctx.advantages > 0, p > q, p < q)
    keep = (~(moving_away & (d_kl > delta))).detach().to(ctx.pi.dtype)
    pg = -ctx.advantages * tr * ctx.pi * keep
    am = ctx.action_mask
    mask_frac = ((1.0 - keep) * am).sum() / am.sum().clamp(min=1.0)
    return ctx.aggregate(pg, am), {"dppo_kl/mask_frac": mask_frac.item()}


@register_loss("cispo")
def cispo(ctx: LossContext):
    """CISPO (arXiv:2506.13585, MiniMax-M1): clip the importance-sampling weight with a
    stop-gradient, but keep **every** token's gradient through ``log_prob`` — no token is
    dropped (unlike PPO clip, which zeros the gradient of clipped tokens)."""
    import torch

    eps_lo = float(ctx.params.get("eps_clip", 0.2))
    eps_hi = ctx.params.get("eps_clip_high")
    eps_hi = float(eps_hi) if eps_hi is not None else eps_lo
    ratio = _ratio(ctx)
    clipped = torch.clamp(ratio, 1.0 - eps_lo, 1.0 + eps_hi)
    pg = -clipped.detach() * ctx.advantages * ctx.pi
    am = ctx.action_mask
    clip_frac = ((ratio != clipped).to(ctx.pi.dtype) * am).sum() / am.sum().clamp(min=1.0)
    return ctx.aggregate(pg, am), {"cispo/clip_frac": clip_frac.item()}


@register_loss("gpg")
def gpg(ctx: LossContext):
    """GPG (Group Policy Gradient): clip-free REINFORCE-style policy gradient with
    group-normalized advantages — ``-advantages * log_prob``."""
    return ctx.aggregate(-ctx.advantages * ctx.pi, ctx.action_mask), {}


@register_loss("ppo_clip_env")
def ppo_clip_env(ctx: LossContext):
    """ECHO (arXiv:2605.24517) in the verl-style single-loss model: PPO/GRPO plus a
    length-normalized cross-entropy term on observation tokens, composed *inside the loss*
    (no auxiliary-loss framework). ``env_loss_coef`` (default 0.05) scales the term;
    set it to 0 to recover plain ``ppo_clip``. This is how an additive term is done now —
    add it in your loss body, exactly as verl would.
    """
    loss, metrics = ppo_clip(ctx)
    coef = float(ctx.params.get("env_loss_coef", 0.05))
    if coef:
        loss = loss + coef * ctx.aggregate(-ctx.pi, ctx.obs_mask)
        metrics["echo/coef"] = coef
    return loss, metrics
