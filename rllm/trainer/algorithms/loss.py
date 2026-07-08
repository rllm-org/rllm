"""Custom policy losses across the verl / tinker / fireworks backends.

Mirrors verl's model: a **single** loss selected by name (``algorithm.loss_fn`` →
verl's ``policy_loss.loss_mode``), not a list. A loss is one function that returns the
**complete** scalar objective and does its own masking + aggregation — exactly like a
verl ``POLICY_LOSS_REGISTRY`` function. There is no separate auxiliary-loss framework:
a loss that wants an extra term (e.g. ECHO's cross-entropy on observation tokens) simply
adds it inside its own body (see ``echo``).

The same function runs in-process under verl and inside ``forward_backward_custom`` on
tinker/fireworks. Each backend injects ``ctx.aggregate(per_token, mask) -> scalar`` realizing
the configured ``algorithm.loss_agg_mode`` (see ``LOSS_AGG_MODES``) with **global**
normalization spanning the whole optimizer step — verl via ``agg_loss`` + global counts,
Fireworks via a raw-sum client loss + server-side ``GradAccNormalization``, Tinker via a
client-side mean over its (single) pass — so the loss body is backend-agnostic.

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
        aggregate: ``(per_token_loss, mask, mode=None) -> scalar`` reducer injected by the
            backend. ``mode`` overrides the aggregation (e.g. GSPO forces
            "seq-mean-token-mean"); None uses the backend/config default.
        ref: reference-policy log-probs for a KL term, or None.
        params: loss hyperparameters (``delta``/``eps_clip``, ``env_loss_coef``, ...).
        backend: "verl" | "tinker" | "fireworks".
    """

    pi: "torch.Tensor"
    mu: "torch.Tensor"
    advantages: "torch.Tensor"
    action_mask: "torch.Tensor"
    obs_mask: "torch.Tensor"
    aggregate: Callable[..., "torch.Tensor"]
    ref: Optional["torch.Tensor"] = None
    params: dict[str, Any] = field(default_factory=dict)
    backend: str = ""

    def seq_reduce(self, values: "torch.Tensor", mask: "torch.Tensor", reduction: str = "mean") -> "torch.Tensor":
        """Reduce ``values`` over each sequence (masked) and broadcast back to per-token.

        A "sequence" is one row: verl tensors are ``(batch, seq_len)`` so this reduces over
        the last dim per row; on the managed per-datum path a datum is one 1-D sequence, so
        it reduces over the whole vector. Enables sequence-level losses (GSPO/GMPO). The
        returned tensor has the same shape as ``values`` (each token holds its sequence's
        reduced value)."""
        import torch

        summed = (values * mask).sum(dim=-1, keepdim=True)
        if reduction == "mean":
            summed = summed / mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return summed.expand_as(values)


# A loss: (ctx) -> (scalar_loss, metrics)
LossFn = Callable[[LossContext], "tuple[torch.Tensor, dict[str, float]]"]

# Canonical loss-aggregation modes, shared across backends (verl's names). A loss body stays
# agnostic and just calls ``ctx.aggregate``; each backend's injected ``aggregate`` (+ its
# optimizer-step normalization) realizes these semantics with GLOBAL normalization spanning
# the whole optimizer step (all micro-batches / grad-accumulation passes / DP ranks):
#   token-mean           Σ_tokens(loss·mask) / Σ_tokens(mask)          — every token equal
#   seq-mean-token-mean  mean within a sequence, then mean over sequences — every seq equal
#   seq-mean-token-sum   sum within a sequence, then mean over sequences
LOSS_AGG_MODES = ("token-mean", "seq-mean-token-mean", "seq-mean-token-sum")
DEFAULT_LOSS_AGG_MODE = "token-mean"  # matches verl's default and Fireworks' RL guidance

RLLM_LOSS_REGISTRY: dict[str, LossFn] = {}
# Optional per-loss aggregation-mode override. A sequence-level loss (e.g. GSPO) must
# aggregate a fixed way regardless of ``algorithm.loss_agg_mode``; it declares that here.
RLLM_LOSS_AGG_MODE: dict[str, str] = {}


def register_loss(name: str, *, agg_mode: str | None = None) -> Callable[[LossFn], LossFn]:
    """Register a loss under ``name`` (its ``algorithm.loss_fn`` value).

    Public API for blackbox ``pip install rllm`` users: decorate a function and select it
    by name. Use ``algorithm.loss_plugins`` to have rllm import the defining module at
    startup so the decorator runs.

    ``agg_mode``: pin this loss's aggregation mode (one of :data:`LOSS_AGG_MODES`), overriding
    ``algorithm.loss_agg_mode``. Use only for losses whose math *requires* a specific reduction
    (GSPO → ``seq-mean-token-mean``); leave None to inherit the configured/default mode.
    """
    if agg_mode is not None and agg_mode not in LOSS_AGG_MODES:
        raise ValueError(f"register_loss({name!r}): agg_mode={agg_mode!r} not in {LOSS_AGG_MODES}")

    def deco(fn: LossFn) -> LossFn:
        if name in RLLM_LOSS_REGISTRY and RLLM_LOSS_REGISTRY[name] is not fn:
            logger.warning("Overriding already-registered loss %r", name)
        RLLM_LOSS_REGISTRY[name] = fn
        if agg_mode is not None:
            RLLM_LOSS_AGG_MODE[name] = agg_mode
        return fn

    return deco


_ENTRY_POINTS_LOADED = False


def _discover_entry_point_losses() -> None:
    """Load losses advertised by installed packages via the ``rllm.losses`` entry-point group.

    A package declares in its ``pyproject.toml``::

        [project.entry-points."rllm.losses"]
        my_dppo = "my_pkg.losses:my_dppo"

    and ``pip install`` makes it discoverable with no config — ``ep.load()`` imports the
    module, firing its ``@register_loss`` decorator. Mirrors ``rllm.eval.agent_loader`` /
    ``evaluator_loader``. Runs once per process (idempotent); triggered lazily on a registry
    miss, so it also populates the registry on verl Ray workers (which call ``get_loss`` there).
    """
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True
    try:
        from importlib.metadata import entry_points

        for ep in entry_points(group="rllm.losses"):
            try:
                ep.load()  # imports the module → registers via @register_loss
                logger.info("Loaded loss from entry point %r", ep.name)
            except Exception as e:  # one bad plugin shouldn't break the run
                logger.warning("Failed to load rllm.losses entry point %r: %s", ep.name, e)
    except Exception as e:
        logger.debug("rllm.losses entry-point discovery skipped: %s", e)


def get_loss(name: str) -> LossFn:
    if name not in RLLM_LOSS_REGISTRY:
        _discover_entry_point_losses()  # lazy: only pay discovery cost on a miss
    if name not in RLLM_LOSS_REGISTRY:
        raise ValueError(f"Unknown loss {name!r}. Registered: {sorted(RLLM_LOSS_REGISTRY)}. Define one with @rllm.register_loss and make it importable — inline in your script, an `rllm.losses` entry point, or algorithm.loss_plugins.")
    return RLLM_LOSS_REGISTRY[name]


def is_custom_loss(name: str | None) -> bool:
    """True if ``name`` is an rLLM loss (vs a backend-native one like verl ``vanilla``).

    Triggers entry-point discovery on a miss so installed loss packages are recognized even
    when the name hasn't been imported yet."""
    if name is None:
        return False
    if name in RLLM_LOSS_REGISTRY:
        return True
    _discover_entry_point_losses()
    return name in RLLM_LOSS_REGISTRY


def load_loss_plugins(modules: list[str]) -> None:
    """Import each module so its ``@register_loss`` decorators run. Idempotent."""
    for mod in modules or []:
        try:
            importlib.import_module(mod)
            logger.info("Loaded loss plugin module %r", mod)
        except Exception as e:
            raise ImportError(f"Failed to import loss plugin module {mod!r} (algorithm.loss_plugins). Importable on this process (and on verl Ray workers)?") from e


def native_loss_names(backend: str) -> set[str]:
    """The loss names a backend has a **native fused kernel** for — the map used for
    native-first routing (``resolve_loss``'s ``native_losses``).

    Read from each backend's own source of truth (not hardcoded) so it never drifts from the
    installed version — e.g. verl 0.7 lacks ``dppo_tv`` while 0.8 has it:

    * ``verl``     → ``verl.trainer.ppo.core_algos.POLICY_LOSS_REGISTRY``
      (vanilla, dppo_tv, dppo_kl, gspo, sapo, gpg, clip_cov, kl_cov, geo_mean, cispo, …)
    * ``tinker``   → ``tinker.types.LossFnType``
      (cross_entropy, importance_sampling, ppo, cispo, dro)
    * ``fireworks``→ ``training.utils.rl.builtin_losses.BUILTIN_LOSSES``
      (grpo, importance_sampling, dapo, dro, gspo, cispo)

    Returns an empty set if the backend isn't importable in this process (→ everything falls
    back to the rLLM custom path)."""
    try:
        if backend == "verl":
            from verl.trainer.ppo.core_algos import POLICY_LOSS_REGISTRY

            return set(POLICY_LOSS_REGISTRY)
        if backend == "tinker":
            from typing import get_args

            from tinker.types import LossFnType

            return set(get_args(LossFnType))
        if backend == "fireworks":
            from training.utils.rl.builtin_losses import BUILTIN_LOSSES

            return set(BUILTIN_LOSSES)
    except Exception as e:  # backend not installed here → treat as "no native kernels"
        logger.debug("native_loss_names(%r): backend not available (%s)", backend, e)
    return set()


@dataclass
class ResolvedLoss:
    """The single loss selected from config, ready for a backend to run."""

    name: str
    fn: LossFn
    params: dict[str, Any]
    agg_mode: str = DEFAULT_LOSS_AGG_MODE  # one of LOSS_AGG_MODES; drives each backend's aggregate + normalization


def resolve_loss(algorithm_config, native_losses: "set[str] | None" = None) -> ResolvedLoss | None:
    """Resolve ``algorithm.loss_fn`` to an rLLM loss, or None to let the backend run it.

    Routing is **native-first**: if ``native_losses`` (the backend's own fused-kernel menu)
    contains ``loss_fn``, return None so the backend uses its fast native kernel — even when
    an rLLM loss of the same name also exists (e.g. verl-native ``dppo_tv``/``gspo``/``cispo``,
    Fireworks builtin ``gspo``/``cispo``). Only losses the backend can't run natively fall
    back to the rLLM custom path (verl in-process; tinker/fireworks ``forward_backward_custom``).

    First imports ``algorithm.loss_plugins`` so user losses are registered. Params passed to
    the loss are the standard clip/kl fields plus ``env_loss_coef`` and anything under
    ``algorithm.loss_params`` (verl-style loss-specific config). Note: a loss that routes to a
    native kernel takes its hyperparameters from the backend-native config (e.g.
    ``eps_clip``→``clip_ratio``), not ``loss_params``."""
    load_loss_plugins(list(getattr(algorithm_config, "loss_plugins", None) or []))
    name = getattr(algorithm_config, "loss_fn", None)
    if name is None:
        return None
    if native_losses is not None and name in native_losses:
        return None  # prefer the backend's native fused kernel
    if not is_custom_loss(name):
        return None
    params = {
        "eps_clip": getattr(algorithm_config, "eps_clip", 0.2),
        "eps_clip_high": getattr(algorithm_config, "eps_clip_high", None),
        "kl_beta": getattr(algorithm_config, "kl_beta", 0.0),
        "env_loss_coef": float(getattr(algorithm_config, "env_loss_coef", 0.0) or 0.0),
        **dict(getattr(algorithm_config, "loss_params", None) or {}),
    }
    # Aggregation mode: the loss's own pin (GSPO) wins; else the config; else the canonical
    # default. Same value feeds verl's agg_loss and the managed adapter, so all backends agree.
    agg_mode = RLLM_LOSS_AGG_MODE.get(name) or getattr(algorithm_config, "loss_agg_mode", None) or DEFAULT_LOSS_AGG_MODE
    return ResolvedLoss(name=name, fn=get_loss(name), params=params, agg_mode=agg_mode)


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


@register_loss("gspo", agg_mode="seq-mean-token-mean")
def gspo(ctx: LossContext):
    """GSPO (arXiv:2507.18071, Qwen): a **sequence-level** importance ratio
    ``s_i = (π_θ(y_i)/π_old(y_i))^(1/|y_i|)`` (length-normalized), clipped PPO-style and
    applied to every token. Uses verl's stop-gradient identity so the value is the
    detached sequence ratio while the gradient flows per-token through ``log_prob``.
    Aggregated seq-mean-token-mean (forced)."""
    import torch

    eps_lo = float(ctx.params.get("eps_clip", 0.2))
    eps_hi = ctx.params.get("eps_clip_high")
    eps_hi = float(eps_hi) if eps_hi is not None else eps_lo
    seq_log_ratio = ctx.seq_reduce(ctx.pi - ctx.mu, ctx.action_mask, "mean")  # per-token = seq mean log-ratio
    # s_{i,t} = sg[s_i] · π_θ,t / sg[π_θ,t]  ⇒  value = seq ratio, gradient flows through pi.
    log_s = torch.clamp(ctx.pi - ctx.pi.detach() + seq_log_ratio.detach(), max=10.0)
    s = torch.exp(log_s)
    pg = torch.maximum(-ctx.advantages * s, -ctx.advantages * torch.clamp(s, 1.0 - eps_lo, 1.0 + eps_hi))
    am = ctx.action_mask
    clip_frac = ((-ctx.advantages * torch.clamp(s, 1.0 - eps_lo, 1.0 + eps_hi) > -ctx.advantages * s).to(ctx.pi.dtype) * am).sum() / am.sum().clamp(min=1.0)
    return ctx.aggregate(pg, am, mode="seq-mean-token-mean"), {"gspo/clip_frac": clip_frac.item()}


@register_loss("reinforce")
def reinforce(ctx: LossContext):
    """REINFORCE: the advantage-weighted policy gradient ``-advantages * log_prob`` — no
    importance ratio, no clip, no trust region (on-policy). This is the exact loss verl
    registers as ``gpg`` and Fireworks as ``reinforce``. Pair with any advantage estimator
    (grpo/rloo/reinforce); the group normalization, if any, lives in the estimator."""
    return ctx.aggregate(-ctx.advantages * ctx.pi, ctx.action_mask), {}


@register_loss("echo")
def echo(ctx: LossContext):
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
