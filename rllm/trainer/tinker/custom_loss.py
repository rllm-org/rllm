"""Managed-backend adapter for the unified custom-loss abstraction.

Tinker and Fireworks share one mechanism for arbitrary differentiable losses:
``forward_backward_custom(data, loss_fn)``. It runs a forward pass, hands back
per-token log-probs with ``requires_grad=True``, calls ``loss_fn(data, logprobs_list)``,
then reproduces the gradient server-side via a weighted-CE pass. (Fireworks' "client"
loss path is literally built on this.) So a single adapter here serves both backends.

This builds the ``loss_fn`` closure that evaluates rLLM loss terms
(``rllm.trainer.algorithms.loss``) over a per-datum :class:`LossContext`. The rollout
arrays (advantages, behavior log-probs μ, masks) are *captured in the closure* — the
``forward_backward_custom`` API only lets the forward datums carry ``target_tokens``.

Normalization: each term is aggregated token-mean over its mask within a datum, then
averaged across datums (seq-mean-token-mean). On the custom path the gradient is taken
directly from ``loss.backward()``, so the caller must NOT let ``optim_step`` re-normalize
it (Tinker's optim_step is already a no-op here; the Fireworks trainer forces
``GradAccNormalization.NONE`` for this pass).
"""

from __future__ import annotations

import logging
from collections import defaultdict

import tinker
from tinker.types.tensor_data import TensorData

from rllm.trainer.algorithms.loss import LossContext, ResolvedTerm

logger = logging.getLogger(__name__)


def _strip_to_target_tokens(datum: tinker.Datum) -> tinker.Datum:
    """forward_backward_custom only accepts loss_fn_inputs ⊆ {target_tokens, weights}."""
    return tinker.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={"target_tokens": datum.loss_fn_inputs["target_tokens"]},
    )


def _agg_token_mean(per_token, mask):
    import torch

    denom = mask.sum().clamp(min=1.0)
    return (per_token * mask).sum() / denom


def build_custom_loss(
    terms: list[ResolvedTerm],
    datums: list[tinker.Datum],
    *,
    mu_arrays: list[list[float]] | None = None,
):
    """Prepare the ``forward_backward_custom`` payload for a set of rLLM loss terms.

    Args:
        terms: resolved loss terms (from ``resolve_loss_terms``).
        datums: rLLM datums with ``loss_fn_inputs`` = {target_tokens, logprobs(μ),
            advantages, mask} (1.0 = action token, 0.0 = observation/prompt).
        mu_arrays: optional override for the behavior log-probs μ per datum (e.g.
            Fireworks proximal log-probs). Defaults to each datum's sampling
            ``logprobs`` (inference μ — the tmax DPPO default).

    Returns:
        ``(stripped_datums, loss_fn)`` — pass both to ``forward_backward_custom``.
    """
    # Capture rollout arrays by datum index (the closure can't read them off the
    # stripped forward datums).
    mu_list = mu_arrays if mu_arrays is not None else [list(d.loss_fn_inputs["logprobs"].data) for d in datums]
    adv_list = [list(d.loss_fn_inputs["advantages"].data) for d in datums]
    action_mask_list = [list(d.loss_fn_inputs["mask"].data) for d in datums]
    stripped = [_strip_to_target_tokens(d) for d in datums]

    def loss_fn(data, logprobs_list):
        import torch

        total = torch.zeros((), dtype=logprobs_list[0].dtype)
        metric_sums: dict[str, float] = defaultdict(float)
        n = len(logprobs_list)
        for i, pi in enumerate(logprobs_list):
            mu = torch.tensor(mu_list[i], dtype=pi.dtype)
            adv = torch.tensor(adv_list[i], dtype=pi.dtype)
            action_mask = torch.tensor(action_mask_list[i], dtype=pi.dtype)
            obs_mask = 1.0 - action_mask
            ctx = LossContext(pi=pi, mu=mu, advantages=adv, action_mask=action_mask, obs_mask=obs_mask, backend="managed")
            for term in terms:
                ctx.params = term.params
                per_token, mask, metrics = term.fn(ctx)
                total = total + term.coef * _agg_token_mean(per_token, mask)
                for k, v in metrics.items():
                    metric_sums[k] += float(v)
        loss = total / max(1, n)
        out_metrics = {k: v / max(1, n) for k, v in metric_sums.items()}
        out_metrics["custom_loss/num_datums"] = float(n)
        return loss, out_metrics

    return stripped, loss_fn
