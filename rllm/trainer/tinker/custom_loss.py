"""Managed-backend adapter for rLLM custom losses (tinker / fireworks).

Both share ``forward_backward_custom(data, loss_fn)``: the service runs a forward pass,
hands back per-token log-probs with ``requires_grad=True``, calls ``loss_fn(data,
logprobs_list)`` on the host, then reproduces the gradient remotely via a weighted-CE
pass. (Fireworks' "client" loss path is built on this.) One adapter serves both.

This builds the ``loss_fn`` closure that evaluates a single rLLM loss
(``rllm.trainer.algorithms.loss``) over a per-datum :class:`LossContext`, honoring the
resolved ``loss_agg_mode``. ``ctx.aggregate`` does only the **within-sequence** reduction
(token-mean for ``seq-mean-token-mean``; masked sum otherwise); the closure then sums those
across datums. Where the final divisor is applied depends on the backend's accumulation model:

* **Fireworks** (``server_normalized=True``): each grad-accumulation pass returns a **raw
  sum**; the caller sets ``GradAccNormalization`` (NUM_LOSS_TOKENS for token-mean,
  NUM_SEQUENCES for seq-mean-*) so the server divides once by the total counted across the
  *whole* window. Invariant to how the mini-batch is split into passes.
* **Tinker** (``server_normalized=False``, the default): no server-side normalization, so the
  closure divides by this pass's global count. Correct when Tinker runs a single pass over the
  whole mini-batch (``fwd_bwd_group_size == mini_batch_size``); under grad accumulation the
  divisor is per-pass (a known limitation — keep the two equal, or use Fireworks).

The rollout arrays (advantages, behavior log-probs μ, masks) are captured in the closure
(the forward datums may only carry ``target_tokens``).
"""

from __future__ import annotations

import logging
from collections import defaultdict

import tinker

from rllm.trainer.algorithms.loss import LossContext, ResolvedLoss, offpolicy_metrics

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
    server_normalized: bool = False,
):
    """Prepare the ``forward_backward_custom`` payload for one rLLM loss.

    Args:
        resolved: the loss to run (from ``resolve_loss``); ``resolved.agg_mode`` selects the
            aggregation mode.
        datums: rLLM datums with ``loss_fn_inputs`` = {target_tokens, logprobs(μ),
            advantages, mask} (1.0 = action token, 0.0 = observation/prompt).
        mu_arrays: optional override for μ per datum (e.g. Fireworks proximal log-probs);
            defaults to each datum's sampling ``logprobs`` (inference μ — tmax default).
            ``ctx.logp_rollout`` always exposes the raw sampling ``logprobs`` regardless of this
            override, so a loss can reach the behavior policy even when μ is a proximal.
        server_normalized: True on Fireworks — return a raw cross-sequence sum and let the
            server divide (via GradAccNormalization) across the whole accumulation window.
            False on Tinker — divide by this pass's global count client-side.

    Returns:
        ``(stripped_datums, loss_fn)`` — pass both to ``forward_backward_custom``.
    """
    rollout_list = [list(d.loss_fn_inputs["logprobs"].data) for d in datums]  # raw inference log-probs (behavior policy)
    mu_list = mu_arrays if mu_arrays is not None else rollout_list
    adv_list = [list(d.loss_fn_inputs["advantages"].data) for d in datums]
    action_mask_list = [list(d.loss_fn_inputs["mask"].data) for d in datums]
    stripped = [_strip_to_target_tokens(d) for d in datums]
    agg_mode = resolved.agg_mode

    def loss_fn(data, logprobs_list):
        import torch

        # Within-sequence reduction only (token-mean for seq-mean-token-mean, else masked
        # sum). The cross-sequence combination + global division is applied to ``total``
        # below (or deferred to the server when server_normalized). ``mode`` lets a loss pin
        # its own reduction (GSPO).
        def aggregate(per_token, mask, mode=None):
            s = (per_token * mask).sum()
            if (mode or agg_mode) == "seq-mean-token-mean":
                s = s / mask.sum().clamp(min=1.0)
            return s

        total = torch.zeros((), dtype=logprobs_list[0].dtype)
        metric_sums: dict[str, float] = defaultdict(float)
        n = len(logprobs_list)
        num_tokens = 0.0
        num_seqs = 0.0
        offp_curr, offp_rollout, offp_mask = [], [], []  # for off-policy diagnostics
        for i, logp_curr in enumerate(logprobs_list):
            action_mask = torch.tensor(action_mask_list[i], dtype=logp_curr.dtype)
            ctx = LossContext(
                logp_curr=logp_curr,
                logp_old=torch.tensor(mu_list[i], dtype=logp_curr.dtype),
                logp_rollout=torch.tensor(rollout_list[i], dtype=logp_curr.dtype),
                advantages=torch.tensor(adv_list[i], dtype=logp_curr.dtype),
                action_mask=action_mask,
                obs_mask=1.0 - action_mask,
                aggregate=aggregate,
                params=resolved.params,
                backend="managed",
            )
            loss_i, metrics_i = resolved.fn(ctx)
            total = total + loss_i
            tok = float(action_mask.sum())
            num_tokens += tok
            num_seqs += 1.0 if tok > 0 else 0.0
            for k, v in metrics_i.items():
                metric_sums[k] += float(v)
            offp_curr.append(logp_curr.detach())
            offp_rollout.append(ctx.logp_rollout)
            offp_mask.append(action_mask)

        if server_normalized:
            loss = total  # server divides by NUM_LOSS_TOKENS / NUM_SEQUENCES across the window
        elif agg_mode == "token-mean":
            loss = total / max(num_tokens, 1.0)
        else:  # seq-mean-token-mean / seq-mean-token-sum
            loss = total / max(num_seqs, 1.0)

        out = {k: v / max(1, n) for k, v in metric_sums.items()}
        out["custom_loss/num_datums"] = float(n)
        # off-policy gap: current policy (logp_curr) vs rollout — captures staleness +
        # train/inference mismatch. Free here (no proximal forward) and non-zero even under
        # bypass_mode, where logp_old == logp_rollout.
        out.update(offpolicy_metrics(offp_curr, offp_rollout, offp_mask))
        return loss, out

    return stripped, loss_fn
