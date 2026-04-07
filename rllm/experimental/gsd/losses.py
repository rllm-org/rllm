"""GSD distillation loss utilities.

Provides two complementary loss primitives for Generalized Self-Distillation:

* **On-policy loss (reverse KL)** — sampled-token REINFORCE proxy via
  ``importance_sampling``.  The per-token advantage is
  ``kl_coeff * (teacher_lp - student_lp)`` for the single generated token.
  This is an unbiased estimator of the reverse-KL gradient::

      ∇_θ KL(p_θ || q) = E_{y~p_θ}[ (log p_θ(y) - log q(y)) · ∇_θ log p_θ(y) ]

* **Supervised loss (forward KL)** — Top-K soft-target cross-entropy via
  Tinker's native ``cross_entropy`` loss.  Teacher's Top-K token probabilities
  are stored as ``(T, K)`` shaped targets in the datum::

      L_t = -Σ_{k=1..K}  q̃_T(v_k) · log p_θ(v_k | x, y_{<t})

  which equals forward KL up to the constant ``H(q_T)``.

Design follows the SDPO recipe (``tinker_cookbook.sdpo``) which uses the same
CE + IS decomposition.  Key conventions inherited from SDPO:

* ``TensorData.from_torch()`` for multi-dimensional loss_fn_inputs.
* A single ``sample_async`` call returns *both* ``topk_prompt_logprobs`` (for
  the CE term) and ``prompt_logprobs`` (for the IS term).
* Explicit ``mask`` field in CE datums to zero out prompt positions.
* All tensor operations use pure torch (no numpy in hot paths).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import tinker
import torch

from rllm.experimental.common.config import rLLMAdvantageEstimator

if TYPE_CHECKING:
    from rllm.experimental.rollout.types import TinkerTokenInput


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default estimator / loss-fn map
# ---------------------------------------------------------------------------

# On-policy distillation trajectories get reverse-KL advantages fed through
# importance_sampling; supervised distillation trajectories get forward-KL
# via cross_entropy with (T, K) teacher soft targets.
DEFAULT_GSD_ADV_ESTIMATOR_MAP: dict[str, Any] = {
    # Case 2 fallback: standard GRPO
    "gsd_student": (rLLMAdvantageEstimator.GRPO, "importance_sampling"),
    # Case 1 on-policy distillation: reverse KL → importance_sampling
    "gsd_distill_onpolicy": ("precomputed", "importance_sampling"),
    # Case 1 supervised distillation: forward KL → cross_entropy
    "gsd_distill_supervised": ("precomputed", "cross_entropy"),
    # Meta-RL on hints (Phase 3, optional)
    "gsd_hint": (rLLMAdvantageEstimator.REINFORCE, "importance_sampling"),
}


def build_gsd_estimator_map(
    *,
    train_hint: bool = False,
) -> dict[str, Any]:
    """Build the estimator map for GSD.

    Args:
        train_hint: Whether to include the hint trajectory for meta-RL training.
    """
    m = dict(DEFAULT_GSD_ADV_ESTIMATOR_MAP)
    if not train_hint:
        m.pop("gsd_hint", None)
    return m


# ---------------------------------------------------------------------------
# Teacher scoring — single call returns both Top-K and sampled-token data
# ---------------------------------------------------------------------------


async def score_teacher_for_response(
    sampling_client: tinker.SamplingClient,
    teacher_prompt_ids: list[int],
    response_ids: list[int],
    topk: int = 20,
    max_context_length: int = 32768,
) -> dict:
    """Score a response under the teacher via teacher-forcing.

    A single ``sample_async`` call with ``topk_prompt_logprobs=K`` returns:

    * ``topk_prompt_logprobs`` — teacher's Top-K token IDs and logprobs
      (used for the forward-KL / CE term).
    * ``prompt_logprobs`` — teacher's scalar logprob of the actual token at
      each position (used for the reverse-KL / IS term).

    Args:
        sampling_client: A Tinker ``SamplingClient``.
        teacher_prompt_ids: Tokenized teacher prompt (with hint).
        response_ids: Tokenized response to score.
        topk: Number of top tokens to retrieve per position.
        max_context_length: Max context length; response is truncated if
            teacher_prompt + response exceeds this.

    Returns:
        Dict with keys:

        * ``"topk_ids"``     — ``list[list[int]]``, shape ``(T, K)``
        * ``"topk_logprobs"`` — ``list[list[float]]``, shape ``(T, K)``
        * ``"sampled_logprobs"`` — ``list[float]``, shape ``(T,)`` —
          teacher's logprob for the actual response token at each position.
        * ``"response_len"``  — int, actual number of response positions scored
          (may be < len(response_ids) if truncated).
    """
    prompt_len = len(teacher_prompt_ids)
    available = max_context_length - prompt_len
    if available <= 0:
        return {
            "topk_ids": [],
            "topk_logprobs": [],
            "sampled_logprobs": [],
            "response_len": 0,
        }

    truncated_response = list(response_ids[:available])
    full_ids = list(teacher_prompt_ids) + truncated_response

    result = await sampling_client.sample_async(
        prompt=tinker.ModelInput.from_ints(full_ids),
        num_samples=1,
        sampling_params=tinker.SamplingParams(max_tokens=1),
        include_prompt_logprobs=True,
        topk_prompt_logprobs=topk,
    )

    T = len(truncated_response)
    topk_all = result.topk_prompt_logprobs
    scalar_all = result.prompt_logprobs

    topk_ids: list[list[int]] = []
    topk_logprobs: list[list[float]] = []
    sampled_logprobs: list[float] = []

    for t in range(T):
        teacher_pos = prompt_len + t

        # --- Scalar logprob of the actual token (for IS) ---
        lp_val = 0.0
        if scalar_all is not None and teacher_pos < len(scalar_all):
            v = scalar_all[teacher_pos]
            if v is not None:
                lp_val = v
        sampled_logprobs.append(lp_val)

        # --- Top-K entries (for CE) ---
        pos_ids: list[int] = []
        pos_lps: list[float] = []
        if topk_all is not None and teacher_pos < len(topk_all):
            entries = topk_all[teacher_pos]
            if entries is not None:
                for tok_id, lp in entries[:topk]:
                    pos_ids.append(tok_id)
                    pos_lps.append(lp)
        # Pad to K if fewer returned
        while len(pos_ids) < topk:
            pos_ids.append(0)
            pos_lps.append(float("-inf"))
        topk_ids.append(pos_ids)
        topk_logprobs.append(pos_lps)

    return {
        "topk_ids": topk_ids,
        "topk_logprobs": topk_logprobs,
        "sampled_logprobs": sampled_logprobs,
        "response_len": T,
    }


# ---------------------------------------------------------------------------
# On-policy loss: sampled-token reverse KL (uses compute_distill_reverse_kl)
# ---------------------------------------------------------------------------


def compute_sampled_rkl_advantages(
    teacher_logprobs: list[float],
    student_logprobs: list[float],
    *,
    kl_coeff: float = 1.0,
    clip_min: float = -5.0,
    clip_max: float = 5.0,
    kl_discount_factor: float = 0.0,
) -> list[float]:
    """Compute per-token reverse-KL advantages from pre-computed logprobs.

    Thin wrapper around :func:`compute_distill_reverse_kl` from
    ``rllm.trainer.distill.advantage``.  Unlike the higher-level
    ``compute_step_distill_advantage`` (which handles cross-tokenizer
    distillation), this function operates directly on aligned logprob
    sequences — appropriate for self-distillation where teacher and student
    share a tokenizer.

    Per-token advantage::

        advantage_t = kl_coeff * (teacher_lp[y_t] - student_lp[y_t])

    Args:
        teacher_logprobs: Teacher's logprob for the actual token at each
            response position (from ``score_teacher_for_response``'s
            ``"sampled_logprobs"``).
        student_logprobs: Student's logprob for the actual token at each
            response position (from ``step.logprobs``).
        kl_coeff: Coefficient for the KL penalty.
        clip_min: Lower clip bound.
        clip_max: Upper clip bound.
        kl_discount_factor: Discount factor for future sum (0 = none).

    Returns:
        Per-token advantages as a list of floats.
    """
    from rllm.trainer.distill.advantage import compute_distill_reverse_kl

    return compute_distill_reverse_kl(
        teacher_logprobs=teacher_logprobs,
        student_logprobs=student_logprobs,
        clip_min=clip_min,
        clip_max=clip_max,
        kl_penalty_coef=kl_coeff,
        kl_discount_factor=kl_discount_factor,
    )


# ---------------------------------------------------------------------------
# Supervised loss: Top-K forward KL datum builder (fully vectorized)
# ---------------------------------------------------------------------------


def build_topk_fkl_datum(
    prompt_ids: TinkerTokenInput,
    response_ids: list[int],
    teacher_topk: dict[str, list[list[int]] | list[list[float]]],
    loss_clamp: float | None = None,
):
    """Build a Tinker Datum for ``cross_entropy`` with Top-K teacher soft targets.

    Following the SDPO recipe (``tinker_cookbook.sdpo.data``), uses
    ``TensorData.from_torch()`` for multi-dimensional loss_fn_inputs and
    includes a ``mask`` field to zero out prompt positions.

    All tensor operations are pure torch — no per-position Python loops.

    Args:
        prompt_ids: Tokenized student prompt (``TinkerTokenInput``).
        response_ids: Response token IDs.
        teacher_topk: ``{"topk_ids": (T, K), "topk_logprobs": (T, K)}`` from
            :func:`score_teacher_for_response`.
        loss_clamp: If set, cap the maximum weight per position to limit
            any single token's loss contribution.

    Returns:
        A ``tinker.Datum`` ready for ``forward_backward_async(loss_fn="cross_entropy")``.
    """
    import tinker
    from tinker_cookbook.supervised.common import (
        create_rightshifted_model_input_and_leftshifted_targets,
    )

    from rllm.experimental.rollout.tinker_engine import _flat_token_input_to_model_input

    T = len(response_ids)
    topk_ids_TK = torch.tensor(teacher_topk["topk_ids"], dtype=torch.long)  # (T, K)
    topk_lps_TK = torch.tensor(teacher_topk["topk_logprobs"], dtype=torch.float64)  # (T, K)
    K = topk_ids_TK.shape[1]
    assert topk_ids_TK.shape[0] == T

    # Build full token sequence and right-shift for autoregressive targets
    all_model_input = _flat_token_input_to_model_input(list(prompt_ids) + list(response_ids))
    input_tokens, _ = create_rightshifted_model_input_and_leftshifted_targets(
        list(all_model_input.chunks),
    )
    N = input_tokens.length  # len(all_tokens) - 1
    prompt_len = N - T

    # --- Renormalize teacher logprobs → probabilities (batched) ---
    # torch.logsumexp handles -inf correctly: exp(-inf) = 0 contributes nothing.
    # Guard against all-inf rows (no valid tokens) → logsumexp returns -inf → exp = 0.
    topk_lps_TK -= torch.logsumexp(topk_lps_TK, dim=1, keepdim=True)
    probs_TK = topk_lps_TK.exp().float()  # (T, K), float32
    # NaN guard: all-inf rows produce nan after -inf - (-inf); zero them out.
    probs_TK = torch.nan_to_num(probs_TK, nan=0.0)

    # --- Optional per-position weight clamping ---
    if loss_clamp is not None:
        max_per_pos = probs_TK.max(dim=1, keepdim=True).values  # (T, 1)
        scale = torch.where(
            max_per_pos > loss_clamp,
            loss_clamp / max_per_pos.clamp(min=1e-12),
            torch.ones_like(max_per_pos),
        )
        probs_TK = probs_TK * scale

    # --- Assemble (N, K) tensors with zero-padded prompt portion ---
    target_tokens_NK = torch.zeros(N, K, dtype=torch.long)
    weights_NK = torch.zeros(N, K, dtype=torch.float32)
    mask_N = torch.zeros(N, dtype=torch.float32)

    target_tokens_NK[prompt_len:] = topk_ids_TK
    weights_NK[prompt_len:] = probs_TK
    mask_N[prompt_len:] = 1.0

    return tinker.Datum(
        model_input=input_tokens,
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_torch(target_tokens_NK),
            "weights": tinker.TensorData.from_torch(weights_NK),
            "mask": tinker.TensorData.from_torch(mask_N),
        },
    )


# ---------------------------------------------------------------------------
# Top-K reverse KL helper (standalone, for future ablation)
# ---------------------------------------------------------------------------


def compute_topk_rkl_at_position(
    teacher_logprobs_k: list[float],
    student_logprobs_k: list[float],
) -> float:
    """Compute reverse KL at a single position over K tokens.

    Both inputs are raw (un-renormalized) logprobs.  They are renormalized
    over the K tokens before computing::

        D_RKL = Σ_k  p̃_S(v_k) · [log p̃_S(v_k) - log q̃_T(v_k)]

    Returns:
        Scalar reverse KL for this position (non-negative by Gibbs' inequality).
    """
    t_lps = torch.tensor(teacher_logprobs_k, dtype=torch.float64)
    s_lps = torch.tensor(student_logprobs_k, dtype=torch.float64)

    # Renormalize via logsumexp
    t_lps = t_lps - torch.logsumexp(t_lps, dim=0)
    s_lps = s_lps - torch.logsumexp(s_lps, dim=0)

    p_s = s_lps.exp()
    rkl = (p_s * (s_lps - t_lps)).sum()
    return max(rkl.item(), 0.0)  # clamp numerical noise


async def compute_student_logprobs_for_teacher_topk(
    sampling_client: tinker.SamplingClient,
    student_prompt_ids: list[int],
    response_ids: list[int],
    teacher_topk: dict[str, list[list[int]] | list[list[float]]],
    student_k: int = 40,
    floor_logprob: float = -18.42,  # ≈ log(1e-8)
) -> dict[str, list[list[float]]]:
    """Get student logprobs for the teacher's Top-K token IDs.

    Calls :func:`score_teacher_for_response` on the student (with a larger K
    for coverage), then looks up the teacher's token IDs in the student's
    results.  Tokens outside the student's Top-``student_k`` get
    ``floor_logprob``.

    Args:
        sampling_client: A Tinker ``SamplingClient``.
        student_prompt_ids: Student prompt (no hint).
        response_ids: Shared response token IDs.
        teacher_topk: Teacher's Top-K data from :func:`score_teacher_for_response`.
        student_k: K for the student scoring pass (should be >= 2x teacher K).
        floor_logprob: Floor logprob for tokens not in student's Top-K.

    Returns:
        ``{"logprobs": list[list[float]]}`` with shape ``(T, teacher_K)``
        aligned to ``teacher_topk["topk_ids"]``.
    """
    student_data = await score_teacher_for_response(
        sampling_client,
        student_prompt_ids,
        response_ids,
        topk=student_k,
    )

    teacher_ids = teacher_topk["topk_ids"]
    T = len(teacher_ids)

    # Vectorized lookup: build (T, student_K) tensor, scatter into lookup
    student_ids_t = torch.tensor(student_data["topk_ids"], dtype=torch.long)  # (T, student_K)
    student_lps_t = torch.tensor(student_data["topk_logprobs"], dtype=torch.float64)  # (T, student_K)
    teacher_ids_t = torch.tensor(teacher_ids, dtype=torch.long)  # (T, teacher_K)

    # For each position, find teacher tokens in student's top-K
    result_logprobs: list[list[float]] = []
    for t in range(T):
        # Boolean mask: which student positions match each teacher token
        # student_ids_t[t] is (student_K,), teacher_ids_t[t] is (teacher_K,)
        matches = student_ids_t[t].unsqueeze(0) == teacher_ids_t[t].unsqueeze(1)  # (teacher_K, student_K)
        # For each teacher token, find the first matching student position
        has_match = matches.any(dim=1)  # (teacher_K,)
        match_idx = matches.float().argmax(dim=1)  # (teacher_K,) — index into student_K
        lps = torch.where(
            has_match,
            student_lps_t[t][match_idx],
            torch.tensor(floor_logprob, dtype=torch.float64),
        )
        result_logprobs.append(lps.tolist())

    return {"logprobs": result_logprobs}
