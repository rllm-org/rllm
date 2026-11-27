"""Score centering for the opt-in verl/FSDP path.

For each response token the sampler records a head ``H`` (top-k token IDs with
their actual sampler log-probs ``log q``). Outside the head we model ``q`` as
proportional to the detached trainer tail: ``q(v) = Q_T / P_T * p(v)`` for
``v not in H``, where ``Q_T = 1 - sum_H q`` and ``P_T = 1 - sum_H p``.

Under that model the expected score ``E_q[w * grad log p]`` collapses to a
closed form with a single "lumped tail" bucket::

    center = sum_H sg(c_v) * log p_v + sg(c_T) * log P_T
    c_v = q_v * w(p_v / q_v) = min(p_v, cap * q_v)     (q_v without TIS)
    c_T = Q_T * w(P_T / Q_T) = min(P_T, cap * Q_T)     (Q_T without TIS)

because every tail token shares the ratio ``p/q = P_T/Q_T`` and
``sum_T p_v grad log p_v = P_T grad log P_T``. This is division-free, needs no
clamping, and is exact when the head covers the sampler support (``Q_T = 0``).
``log P_T`` is computed directly by a masked log-sum-exp, avoiding the
cancellation in ``1 - sum_H p``.

The head mass is never renormalized. IS weights use the current trainer /
sampler ratio and are detached. There is no PPO clipping.
See https://arxiv.org/abs/2609.20807 and verl's score_centering.py.
"""

from __future__ import annotations

import math

import torch
from torch.utils.checkpoint import checkpoint

SC_FIELDS = ("rollout_topk_ids", "rollout_topk_log_probs")


def _gather_chunk(chunk: torch.Tensor, labels: torch.Tensor, head_ids: torch.Tensor) -> torch.Tensor:
    work = chunk.float()
    lse = torch.logsumexp(work, dim=-1, keepdim=True)
    label_lp = work.gather(-1, labels.unsqueeze(-1)) - lse
    head_lp = work.gather(-1, head_ids) - lse
    # finfo.min (not -inf) keeps the backward finite even if the head covers the vocabulary.
    tail = work.scatter(-1, head_ids, torch.finfo(work.dtype).min)
    tail_lp = torch.logsumexp(tail, dim=-1, keepdim=True) - lse
    return torch.cat((label_lp, head_lp, tail_lp), dim=-1)


def candidate_log_probs(logits: torch.Tensor, labels: torch.Tensor, head_ids: torch.Tensor, chunk_size: int = 128):
    """Full-vocabulary normalized gathers; fp32 chunks are recomputed in backward.

    Returns ``(label_logp [N], head_logp [N, K], tail_log_mass [N])``. The
    workspace is a few ``chunk_size x vocab`` fp32 buffers, plus the original
    logits and their gradient. All operations are out-of-place.
    """
    if chunk_size < 1 or logits.ndim != 2 or head_ids.ndim != 2 or labels.ndim != 1 or not (logits.shape[0] == head_ids.shape[0] == labels.shape[0]):
        raise ValueError("Invalid logits/labels/head_ids shape or chunk_size")
    labels, head_ids = labels.long(), head_ids.long()
    pieces = []
    for start in range(0, len(logits), chunk_size):
        args = (logits[start : start + chunk_size], labels[start : start + chunk_size], head_ids[start : start + chunk_size])
        if torch.is_grad_enabled() and logits.requires_grad:
            pieces.append(checkpoint(_gather_chunk, *args, use_reentrant=False))
        else:
            pieces.append(_gather_chunk(*args))
    out = torch.cat(pieces) if pieces else logits.new_empty((0, head_ids.shape[1] + 2), dtype=torch.float32)
    return out[:, 0], out[:, 1:-1], out[:, -1]


def centered_token_loss(sample_logp, sample_logq, head_logp, head_logq, tail_logp, advantages, tis_cap=None):
    """Return per-token SC PG loss and head masses (sampler, trainer), unaggregated.

    Slots with ``head_logq = -inf`` are treated as zero-q head entries.
    """
    if tis_cap is not None and (not math.isfinite(tis_cap) or tis_cap <= 0):
        raise ValueError("Score centering TIS cap must be finite and positive")
    if not torch.isfinite(sample_logq).all():
        raise ValueError("Sampled tokens require finite actual sampler log probabilities")
    sample_logp, head_logp, tail_logp = sample_logp.float(), head_logp.float(), tail_logp.float()
    q = head_logq.detach().float().exp()
    q_mass = q.sum(-1)
    if (q_mass > 1.0001).any() or (q_mass <= 0).any():
        raise ValueError("Invalid sampler head probability mass")
    with torch.no_grad():
        q_tail = (1 - q_mass).clamp_min(0)
        p_head, p_tail = head_logp.exp(), tail_logp.exp()
        if tis_cap is None:
            c_head, c_tail, w_sample = q, q_tail, torch.ones_like(sample_logp)
        else:
            c_head = torch.minimum(p_head, tis_cap * q)
            c_tail = torch.minimum(p_tail, tis_cap * q_tail)
            w_sample = (sample_logp - sample_logq.float()).clamp(max=math.log(tis_cap)).exp()
    # A head spanning the whole vocabulary leaves an empty tail (finfo.min sentinel); drop it.
    has_tail = tail_logp > torch.finfo(torch.float32).min / 2
    tail_term = torch.where(has_tail, c_tail * tail_logp, torch.zeros_like(tail_logp))
    center = (c_head * head_logp).sum(-1) + tail_term
    loss = -advantages.detach() * (w_sample * sample_logp - center)
    return loss, q_mass.detach(), p_head.sum(-1)


def validate_head(ids, log_probs, response_length):
    """Validate persisted sampler heads before padding or cropping; return head width."""
    if ids is None or log_probs is None or len(ids) != response_length or len(log_probs) != response_length:
        raise ValueError("Score centering candidate fields must match response length")
    k = len(ids[0]) if ids else 0
    if response_length and not k:
        raise ValueError("Empty score centering candidate head")
    for token_ids, lps in zip(ids, log_probs, strict=True):
        if len(token_ids) != k or len(lps) != k or len(set(token_ids)) != k or any(i < 0 for i in token_ids):
            raise ValueError("Score centering heads require distinct nonnegative IDs and a fixed width")
        if any(math.isnan(lp) or lp > 0 for lp in lps):
            raise ValueError("Invalid sampler candidate log probabilities")
        mass = sum(math.exp(lp) for lp in lps)
        if mass <= 0 or mass > 1.0001:
            raise ValueError("Invalid sampler head mass")
    return k


def _dummy_head(rows: int, k: int, device=None):
    """Distinct IDs with uniform unit mass: valid for padding rows, which are masked out of the loss."""
    return torch.arange(k, device=device).expand(rows, k).clone(), torch.full((rows, k), -math.log(k), device=device)


def batch_rollout_heads(heads, responses, max_length):
    if len(heads) != len(responses) or any(head is None for head in heads):
        raise ValueError("Missing score centering candidate distribution in batch")
    widths = [validate_head(*head, len(response)) for head, response in zip(heads, responses, strict=True)]
    k = max(widths, default=0)
    if not k:
        raise ValueError("Score centering batch has no response candidates")
    if any(width not in (0, k) for width in widths):
        raise ValueError("Mixed score centering candidate widths")
    ids, lps = (t.expand(len(heads), max_length, k).clone() for t in _dummy_head(1, k))
    for row, (head, response) in enumerate(zip(heads, responses, strict=True)):
        n = min(len(response), max_length)
        if n:
            ids[row, :n] = torch.tensor(head[0][:n], dtype=torch.int64)
            lps[row, :n] = torch.tensor(head[1][:n], dtype=torch.float32)
    return dict(zip(SC_FIELDS, (ids, lps), strict=True))


def heads_to_no_padding(data):
    """Move response heads to the rows that *predict* response tokens.

    Response token ``r`` is predicted by full-sequence position ``p - 1 + r``,
    matching verl's ``no_padding_2_padding`` shift. Invoke before verl removes
    padding. Candidate tensors use full-sequence offsets and survive microbatch
    slicing/DP dispatch as nested tensors.
    """
    if not any(key in data for key in SC_FIELDS):
        return
    if not all(key in data for key in SC_FIELDS):
        raise ValueError("Incomplete score centering batch fields")
    mask, prompt_width = data["attention_mask"], data["prompts"].shape[1]
    k = data[SC_FIELDS[0]].shape[-1]
    rows = {key: [] for key in SC_FIELDS}
    for row in range(mask.shape[0]):
        p = int(mask[row, :prompt_width].sum())
        r = int(mask[row, prompt_width:].sum())
        if p < 1:
            raise ValueError("Score centering requires a nonempty prompt")
        for key, value in zip(SC_FIELDS, _dummy_head(p + r, k, mask.device), strict=True):
            value[p - 1 : p - 1 + r] = data[key][row, :r]
            rows[key].append(value)
    for key in SC_FIELDS:
        data[key] = torch.nested.as_nested_tensor(rows[key], layout=torch.jagged)
