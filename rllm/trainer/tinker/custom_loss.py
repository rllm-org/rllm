"""Managed-backend adapter for rLLM custom losses (tinker / fireworks).

Both share ``forward_backward_custom(data, loss_fn)``: the service runs a forward pass,
hands back per-token log-probs with ``requires_grad=True``, calls ``loss_fn(data,
logprobs_list)`` on the host, then reproduces the gradient remotely via a weighted-CE
pass. (Fireworks' "client" loss path is built on this.) One adapter serves both.

This builds the ``loss_fn`` closure that evaluates a single rLLM loss
(``rllm.trainer.algorithms.loss``) over a per-datum :class:`LossContext`. The loss returns
a scalar and aggregates via ``ctx.aggregate`` — here, token-mean over the mask within a
datum, averaged across datums (seq-mean-token-mean). The rollout arrays (advantages,
behavior log-probs μ, masks) are captured in the closure (the forward datums may only
carry ``target_tokens``).
"""

from __future__ import annotations

import logging
from collections import defaultdict

import tinker

from rllm.trainer.algorithms.loss import LossContext, ResolvedLoss

logger = logging.getLogger(__name__)


def _strip_to_target_tokens(datum: tinker.Datum) -> tinker.Datum:
    """forward_backward_custom only accepts loss_fn_inputs ⊆ {target_tokens, weights}."""
    return tinker.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={"target_tokens": datum.loss_fn_inputs["target_tokens"]},
    )


def build_custom_loss(
    resolved: ResolvedLoss,
    datums: list[tinker.Datum],
    *,
    mu_arrays: list[list[float]] | None = None,
):
    """Prepare the ``forward_backward_custom`` payload for one rLLM loss.

    Args:
        resolved: the loss to run (from ``resolve_loss``).
        datums: rLLM datums with ``loss_fn_inputs`` = {target_tokens, logprobs(μ),
            advantages, mask} (1.0 = action token, 0.0 = observation/prompt).
        mu_arrays: optional override for μ per datum (e.g. Fireworks proximal log-probs);
            defaults to each datum's sampling ``logprobs`` (inference μ — tmax default).

    Returns:
        ``(stripped_datums, loss_fn)`` — pass both to ``forward_backward_custom``.
    """
    mu_list = mu_arrays if mu_arrays is not None else [list(d.loss_fn_inputs["logprobs"].data) for d in datums]
    adv_list = [list(d.loss_fn_inputs["advantages"].data) for d in datums]
    action_mask_list = [list(d.loss_fn_inputs["mask"].data) for d in datums]
    stripped = [_strip_to_target_tokens(d) for d in datums]

    def loss_fn(data, logprobs_list):
        import torch

        # token-mean within a datum; the cross-datum average below makes the overall
        # reduction seq-mean-token-mean. ``mode`` is accepted for API parity (GSPO passes
        # "seq-mean-token-mean", which is already what this composes to).
        def aggregate(per_token, mask, mode=None):
            return (per_token * mask).sum() / mask.sum().clamp(min=1.0)

        total = torch.zeros((), dtype=logprobs_list[0].dtype)
        metric_sums: dict[str, float] = defaultdict(float)
        n = len(logprobs_list)
        for i, pi in enumerate(logprobs_list):
            action_mask = torch.tensor(action_mask_list[i], dtype=pi.dtype)
            ctx = LossContext(
                pi=pi,
                mu=torch.tensor(mu_list[i], dtype=pi.dtype),
                advantages=torch.tensor(adv_list[i], dtype=pi.dtype),
                action_mask=action_mask,
                obs_mask=1.0 - action_mask,
                aggregate=aggregate,
                params=resolved.params,
                backend="managed",
            )
            loss_i, metrics_i = resolved.fn(ctx)
            total = total + loss_i
            for k, v in metrics_i.items():
                metric_sums[k] += float(v)
        loss = total / max(1, n)
        out = {k: v / max(1, n) for k, v in metric_sums.items()}
        out["custom_loss/num_datums"] = float(n)
        return loss, out

    return stripped, loss_fn
