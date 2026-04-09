"""GSD loss helpers — SFT-style CE + sampled-token IS.

The new GSD pipeline decomposes distillation into two standard Tinker losses
that run through the built-in ``forward_backward_async``:

* **CE (forward KL, supervised)** — student is trained on the teacher's
  correct response via Tinker's classic ``cross_entropy`` loss.  The datum
  is a normal SFT example: student's prompt + teacher's response with a
  ``weights`` mask that is 1 on the response tokens and 0 on the prompt.

* **IS (reverse KL, on-policy)** — student is trained on its own response
  with a per-token advantage ``teacher_lp - student_lp`` via Tinker's
  ``importance_sampling`` loss.  The teacher logprobs come from a frozen
  reference :class:`tinker.SamplingClient` called with
  ``compute_logprobs_async`` (no Top-K, no extra ``sample_async`` call).

The legacy Top-K CE variant (combined ``(N, K+1)`` custom-loss path) lives
under :mod:`rllm.experimental.gsd.legacy` and is no longer imported here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import tinker
from tinker.types.tensor_data import TensorData
from tinker_cookbook.supervised.common import (
    create_rightshifted_model_input_and_leftshifted_targets,
)

from rllm.experimental.common.config import rLLMAdvantageEstimator
from rllm.experimental.rollout.tinker_engine import _flat_token_input_to_model_input

if TYPE_CHECKING:
    from rllm.experimental.rollout.types import TinkerTokenInput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Role names — single source of truth for transform + workflow + estimator map
# ---------------------------------------------------------------------------

CE_ROLE = "gsd_ce"  # student_prompt + teacher_response → cross_entropy
IS_ROLE = "gsd_is"  # student_prompt + student_response with precomputed advantages → importance_sampling
GRPO_ROLE = "gsd_grpo"  # Case 2 student fallback → ppo via REINFORCE++ baseline
HINT_ROLE = "gsd_hint"  # hint generation trajectories → REINFORCE (GRPO) across tasks

GSD_ROLES = (CE_ROLE, IS_ROLE, GRPO_ROLE, HINT_ROLE)


# ---------------------------------------------------------------------------
# Estimator / loss-fn map
# ---------------------------------------------------------------------------


def build_gsd_estimator_map(*, train_hint: bool = True) -> dict[str, Any]:
    """Build the per-role ``estimator_map`` for the new GSD pipeline.

    The per-role dispatch in ``TinkerPolicyTrainer._get_forward_backward_futures``
    fires one standard ``forward_backward_async`` call per role, so every
    role here maps to a built-in Tinker loss (no custom loss functions).

    Args:
        train_hint: If True, include ``gsd_hint`` in the map so that hint
            trajectories are optimized via REINFORCE / GRPO.  If False,
            hint trajectories (if any) are still built by the workflow but
            fall through to ``importance_sampling`` as a passthrough.

    Returns:
        A dict of ``role → (estimator, loss_fn)`` tuples.  The trainer's
        ``AlgorithmConfig.__post_init__`` will split these into separate
        ``estimator_map`` and ``loss_fn_map`` entries.
    """
    m: dict[str, Any] = {
        CE_ROLE: ("precomputed", "cross_entropy"),
        IS_ROLE: ("precomputed", "importance_sampling"),
        GRPO_ROLE: (rLLMAdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE, "ppo"),
    }
    if train_hint:
        m[HINT_ROLE] = (rLLMAdvantageEstimator.REINFORCE, "importance_sampling")
    return m


# ---------------------------------------------------------------------------
# Teacher logprob evaluation on a student response
# ---------------------------------------------------------------------------


async def compute_teacher_logprobs_for_response(
    teacher_client: tinker.SamplingClient,
    teacher_prompt_ids: list[int],
    response_ids: list[int],
    max_context_length: int = 32768,
) -> tuple[list[float], int]:
    """Evaluate the frozen teacher's per-token logprobs on ``response_ids``.

    Uses ``compute_logprobs_async`` — a single cheap call that returns a
    logprob for every token in the concatenated ``teacher_prompt + response``
    sequence.  The slice ``[len(prompt):]`` gives the teacher's logprob for
    each response token (conditioned on the teacher prompt as context),
    matching the convention used in ``rllm.experimental.opsd.advantage``.

    Returns ``(response_logprobs, T)`` where ``T`` is the actual number of
    response tokens scored (may be less than ``len(response_ids)`` if the
    sequence had to be truncated to fit ``max_context_length``).
    """
    prompt_len = len(teacher_prompt_ids)
    available = max_context_length - prompt_len
    if available <= 0:
        return [], 0

    truncated = list(response_ids[:available])
    T = len(truncated)
    if T == 0:
        return [], 0

    full_ids = list(teacher_prompt_ids) + truncated
    mi = tinker.ModelInput.from_ints(full_ids)

    raw = await teacher_client.compute_logprobs_async(mi)

    # ``compute_logprobs_async`` returns an object that is directly
    # sliceable — match the usage pattern in rLLM's existing OPSD code
    # (``rllm/experimental/opsd/advantage.py``): slice at ``prompt_len``
    # to get exactly the response logprobs.
    response_slice = raw[prompt_len : prompt_len + T]
    response_lp = [float(v) if v is not None else 0.0 for v in response_slice]

    # Guard against a short return (shouldn't happen, but the API is
    # documented as possibly returning None at some positions).
    if len(response_lp) < T:
        response_lp = response_lp + [0.0] * (T - len(response_lp))
    return response_lp, T


def kl_advantages_from_logprobs(
    teacher_logprobs: list[float],
    student_logprobs: list[float],
    *,
    kl_coeff: float = 1.0,
    clip_min: float = -5.0,
    clip_max: float = 5.0,
) -> list[float]:
    """Per-token reverse-KL advantages: ``clip(kl_coeff * (t - s), min, max)``.

    Operates directly on aligned scalar logprob sequences.  This is the
    cheap replacement for the old ``compute_sampled_rkl_advantages`` helper
    that went through ``compute_distill_reverse_kl`` — we intentionally keep
    it self-contained here to avoid a soft dependency on
    :mod:`rllm.trainer.distill.advantage`.
    """
    n = min(len(teacher_logprobs), len(student_logprobs))
    out: list[float] = []
    for i in range(n):
        a = kl_coeff * (teacher_logprobs[i] - student_logprobs[i])
        if a < clip_min:
            a = clip_min
        elif a > clip_max:
            a = clip_max
        out.append(float(a))
    return out


# ---------------------------------------------------------------------------
# Datum builders — standard Tinker loss shapes
# ---------------------------------------------------------------------------


def build_sft_style_ce_datum(
    prompt_ids: TinkerTokenInput,
    response_ids: list[int],
) -> tinker.Datum:
    """Build a cross-entropy datum for supervised fine-tuning on a response.

    The datum mirrors ``renderer.build_supervised_example`` — one
    right-shifted input, a left-shifted ``target_tokens`` vector, and a
    ``weights`` mask that is ``1.0`` on response positions and ``0.0`` on
    prompt positions.  This is the standard Tinker format for
    ``forward_backward_async(loss_fn="cross_entropy")``.

    Args:
        prompt_ids: The prompt as a :data:`TinkerTokenInput` — can be a
            flat list of ints or a list of ``ModelInputChunk``.  In GSD
            this is the *student* prompt (no hint), because the student
            is the model being trained.
        response_ids: The response tokens (typically from a teacher
            rollout that reached a correct answer).

    Returns:
        A :class:`tinker.Datum` ready to feed into ``forward_backward_async``.
    """
    all_tokens = list(prompt_ids) + list(response_ids)
    all_model_input = _flat_token_input_to_model_input(all_tokens)
    input_tokens, target_tokens = create_rightshifted_model_input_and_leftshifted_targets(
        list(all_model_input.chunks),
    )
    N = input_tokens.length  # len(all_tokens) - 1
    T = len(response_ids)
    prompt_len = N - T

    weights = [0.0] * prompt_len + [1.0] * T
    assert len(weights) == N, f"weights length {len(weights)} != N {N}"

    return tinker.Datum(
        model_input=input_tokens,
        loss_fn_inputs={
            "target_tokens": TensorData(data=target_tokens, dtype="int64"),
            "weights": TensorData(data=weights, dtype="float32"),
        },
    )


def build_is_datum(
    prompt_ids: TinkerTokenInput,
    response_ids: list[int],
    logprobs: list[float],
    advantages: list[float] | float,
) -> tinker.Datum:
    """Build an importance-sampling datum with precomputed per-token advantages.

    Matches the shape produced by :func:`rllm.trainer.tinker.transform.trajectory_to_datums`
    (``target_tokens`` / ``logprobs`` / ``advantages`` / ``mask``) so it can
    be fed to ``forward_backward_async(loss_fn="importance_sampling")``.

    Args:
        prompt_ids: The prompt as a :data:`TinkerTokenInput`.  In GSD this
            is the student's own prompt — the response was sampled from
            the student, so the IS ratio is computed against student's
            own sampling logprobs (``logprobs``) and the advantage is the
            precomputed ``teacher_lp - student_lp`` from the frozen
            reference teacher.
        response_ids: The response tokens (from the student rollout).
        logprobs: Sampling-time student logprobs for each response token
            (length must equal ``len(response_ids)``).
        advantages: Per-token reverse-KL advantages (length must equal
            ``len(response_ids)``), or a scalar broadcast across tokens.
    """
    T = len(response_ids)
    assert len(logprobs) == T, f"logprobs length {len(logprobs)} != response length {T}"

    if isinstance(advantages, int | float):
        adv_list = [float(advantages)] * T
    else:
        adv_list = list(advantages)
    assert len(adv_list) == T, f"advantages length {len(adv_list)} != response length {T}"

    all_tokens = list(prompt_ids) + list(response_ids)
    all_model_input = _flat_token_input_to_model_input(all_tokens)
    input_tokens, target_tokens = create_rightshifted_model_input_and_leftshifted_targets(
        list(all_model_input.chunks),
    )
    N = input_tokens.length  # len(all_tokens) - 1
    prompt_len = N - T

    # The prompt contribution is zero / masked; response positions carry
    # the sampled logprobs and advantages.  Note that `logprobs[0]` aligns
    # with the first response target token (which in the right-shifted
    # input sits at position `prompt_len`).
    sampled_logprobs_N = [0.0] * prompt_len + list(logprobs)
    advantages_N = [0.0] * prompt_len + adv_list
    mask_N = [0.0] * prompt_len + [1.0] * T

    assert len(sampled_logprobs_N) == N, f"sampled_logprobs length {len(sampled_logprobs_N)} != N {N}"

    return tinker.Datum(
        model_input=input_tokens,
        loss_fn_inputs={
            "target_tokens": TensorData(data=target_tokens, dtype="int64"),
            "logprobs": TensorData(data=sampled_logprobs_N, dtype="float32"),
            "advantages": TensorData(data=advantages_N, dtype="float32"),
            "mask": TensorData(data=mask_N, dtype="float32"),
        },
    )


__all__ = [
    "CE_ROLE",
    "IS_ROLE",
    "GRPO_ROLE",
    "HINT_ROLE",
    "GSD_ROLES",
    "build_gsd_estimator_map",
    "compute_teacher_logprobs_for_response",
    "kl_advantages_from_logprobs",
    "build_sft_style_ce_datum",
    "build_is_datum",
]
