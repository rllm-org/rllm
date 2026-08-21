"""Monkey-patches applied to Miles from the rLLM backend.

Same mechanism rLLM already uses for verl (``rllm/trainer/verl/patch.py``): applied
lazily, idempotent, and applied on the train workers as well as the driver.

Each patch here should end up upstream in Miles; the docstrings say what the
upstream change would be so they can be dropped as it lands.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_ADVANTAGES_CP_SLICE_PATCHED = False

# The key tuple in miles/backends/training_utils/data.py:get_rollout_data that gets
# CP-sliced per sample. Asserted so a Miles upgrade that renames or reorders these
# fails loudly here instead of silently skipping our advantages.
_EXPECTED_CP_SLICED_KEYS = ("rollout_log_probs", "teacher_log_probs", "opd_reverse_kl")


def patch_advantages_cp_slice() -> None:
    """Let driver-supplied per-token advantages ride the ``rollout_log_probs`` path.

    rLLM computes advantages itself and ships them as a top-level
    ``rollout_data["advantages"]``. Miles' own estimator is off (via
    ``--disable-compute-advantages-and-returns``), and its
    ``policy_loss_function`` reads ``batch["advantages"]`` directly -- so the only
    thing missing is the per-sample context-parallel slice that every other
    response-aligned float array gets.

    ``get_rollout_data`` applies ``slice_log_prob_with_cp`` to a hardcoded tuple of
    keys. This wraps it so ``advantages`` is sliced the same way. At CP size 1 the
    slice is the identity, so this is a no-op there and only matters for CP > 1.

    Upstream equivalent: add ``"advantages"`` to that tuple, plus a
    ``ROLLOUT_DATA_VALUE_SPEC`` entry (needed only for the Mooncake object store;
    the Ray store ignores the spec).
    """
    global _ADVANTAGES_CP_SLICE_PATCHED
    if _ADVANTAGES_CP_SLICE_PATCHED:
        return

    import torch
    from miles.backends.training_utils import data as miles_data

    original = miles_data.get_rollout_data

    def get_rollout_data(args, rollout_data_ref, witness_info=None):
        rollout_data, store_get_result = original(args, rollout_data_ref, witness_info=witness_info)

        advantages = rollout_data.get("advantages")
        if advantages is not None and not isinstance(advantages[0], torch.Tensor):
            rollout_data["advantages"] = [
                torch.as_tensor(
                    miles_data.slice_log_prob_with_cp(
                        value,
                        total_length,
                        response_length,
                        args.qkv_format,
                        rollout_data["max_seq_lens"][i] if args.qkv_format == "bshd" else None,
                    ),
                    device=torch.cuda.current_device(),
                    dtype=torch.float32,
                )
                for i, (value, total_length, response_length) in enumerate(zip(advantages, rollout_data["total_lengths"], rollout_data["response_lengths"], strict=True))
            ]
        return rollout_data, store_get_result

    miles_data.get_rollout_data = get_rollout_data
    _ADVANTAGES_CP_SLICE_PATCHED = True
    logger.info("Patched miles get_rollout_data to CP-slice rLLM's per-token advantages")


def assert_cp_slice_contract() -> None:
    """Fail loudly if Miles moved the ground this patch stands on."""
    import inspect

    from miles.backends.training_utils import data as miles_data

    source = inspect.getsource(miles_data)
    expected = 'for key in ("rollout_log_probs", "teacher_log_probs", "opd_reverse_kl"):'
    if expected not in source:
        raise RuntimeError(
            "miles get_rollout_data no longer CP-slices the expected key tuple "
            f"{_EXPECTED_CP_SLICED_KEYS}. rllm/trainer/miles/patch.py assumes that shape "
            "for per-token advantages; re-check it against the installed miles version."
        )


def apply_all_miles_patches() -> None:
    """Entry point for both the driver and the train workers."""
    assert_cp_slice_contract()
    patch_advantages_cp_slice()
