"""Monkey-patches applied to Miles from the rLLM backend.

Same mechanism rLLM already uses for verl (``rllm/trainer/verl/patch.py``): applied
lazily, idempotent, and applied on the train workers as well as the driver.

Each patch here should end up upstream in Miles; the docstrings say what the
upstream change would be so they can be dropped as it lands.
"""

from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)

_ADVANTAGES_CP_SLICE_PATCHED = False
_PACKAGE_SHARDS_PATCHED = False
_RESPECT_DISABLE_ADV_PATCHED = False

# The key tuple in miles/backends/training_utils/data.py:get_rollout_data that gets
# CP-sliced per sample. Asserted so a Miles upgrade that renames or reorders these
# fails loudly here instead of silently skipping our advantages.
_EXPECTED_CP_SLICED_KEYS = ("rollout_log_probs", "teacher_log_probs", "opd_reverse_kl")


# Both actors do `from ...module import name`, which binds the function *by value* at
# import time. Rebinding the source module therefore only reaches an actor that has not
# been imported yet -- and anything that imports an actor early (assert_patch_contracts
# did) silently defeats the patch. Always repoint the holders too.
_FUNCTION_HOLDERS = ("miles.backends.fsdp_utils.actor", "miles.backends.megatron_utils.actor")


def _rebind_everywhere(name: str, replacement) -> None:
    for module_path in _FUNCTION_HOLDERS:
        try:
            module = importlib.import_module(module_path)
        except Exception:  # the megatron actor needs megatron installed
            continue
        if hasattr(module, name):
            setattr(module, name, replacement)


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
    _rebind_everywhere("get_rollout_data", get_rollout_data)
    _ADVANTAGES_CP_SLICE_PATCHED = True
    logger.info("Patched miles get_rollout_data to CP-slice rLLM's per-token advantages")


def patch_package_shards_forwards_advantages() -> None:
    """Stop the DP split from dropping rLLM's per-token advantages.

    ``_package_shards`` copies a **hardcoded allowlist** of keys into each DP rank's
    shard (``miles/ray/rollout/train_data_conversion.py``), so any extra key the
    driver attaches -- ours included -- is silently discarded on the way to the train
    workers. Silently: the loss then falls back to whatever Miles computed itself, and
    the run looks healthy while ignoring rLLM's advantages entirely.

    Upstream equivalent: add ``"advantages"`` to that per-sample list.
    """
    global _PACKAGE_SHARDS_PATCHED
    if _PACKAGE_SHARDS_PATCHED:
        return

    from miles.ray.rollout import train_data_conversion as tdc

    original = tdc._package_shards

    def _package_shards(args, data, partitions):
        shards = original(args, data, partitions)
        advantages = data.get("advantages")
        if advantages is not None:
            for shard, partition in zip(shards, partitions, strict=True):
                shard["advantages"] = [advantages[j] for j in partition]
        return shards

    tdc._package_shards = _package_shards
    # split_train_data_by_dp_raw closed over the module-level name at def time only
    # if it had been imported by value; it calls it through the module, so this is enough.
    _PACKAGE_SHARDS_PATCHED = True
    logger.info("Patched miles _package_shards to forward rLLM's advantages to the DP shards")


def patch_respect_disable_compute_advantages() -> None:
    """Make ``--disable-compute-advantages-and-returns`` actually hold on the FSDP path.

    The Megatron actor gates its ``compute_advantages_and_returns`` call on that flag;
    the FSDP actor (``miles/backends/fsdp_utils/actor.py``) calls it unconditionally, so
    Miles recomputes advantages from the scalar rewards and overwrites the per-token
    values rLLM shipped. This wraps the function to no-op when the flag is off and
    advantages are already present, leaving ``returns`` aliased to them (GRPO's
    identity, and the only estimator this backend supports today).

    Upstream equivalent: wrap the FSDP call site in the same ``if`` the Megatron one uses.
    """
    global _RESPECT_DISABLE_ADV_PATCHED
    if _RESPECT_DISABLE_ADV_PATCHED:
        return

    from miles.backends.training_utils import loss as miles_loss

    original = miles_loss.compute_advantages_and_returns

    def compute_advantages_and_returns(args, rollout_data):
        if not getattr(args, "compute_advantages_and_returns", True) and rollout_data.get("advantages") is not None:
            rollout_data.setdefault("returns", rollout_data["advantages"])
            return
        return original(args, rollout_data)

    miles_loss.compute_advantages_and_returns = compute_advantages_and_returns
    _rebind_everywhere("compute_advantages_and_returns", compute_advantages_and_returns)

    _RESPECT_DISABLE_ADV_PATCHED = True
    logger.info("Patched miles compute_advantages_and_returns to respect --disable-compute-advantages-and-returns")


def assert_patch_contracts() -> None:
    """Fail loudly if Miles moved the ground any of these patches stands on.

    Every one of them is load-bearing for rLLM's advantages reaching the loss, and
    every one fails *silently* if it stops applying -- the run completes and the loss
    quietly uses Miles' own advantages. So the structural assumptions are asserted
    rather than assumed, and an upstream fix is reported so the patch can be dropped.
    """
    import inspect

    from miles.backends.fsdp_utils import actor as fsdp_actor
    from miles.backends.training_utils import data as miles_data
    from miles.ray.rollout import train_data_conversion as tdc

    # 1. the CP-slice key tuple get_rollout_data walks
    expected = 'for key in ("rollout_log_probs", "teacher_log_probs", "opd_reverse_kl"):'
    if expected not in inspect.getsource(miles_data):
        raise RuntimeError(
            "miles get_rollout_data no longer CP-slices the expected key tuple "
            f"{_EXPECTED_CP_SLICED_KEYS}. rllm/trainer/miles/patch.py assumes that shape "
            "for per-token advantages; re-check it against the installed miles version."
        )

    # 2. the DP-shard allowlist we append to
    if not hasattr(tdc, "_package_shards"):
        raise RuntimeError("miles train_data_conversion._package_shards is gone; the DP-shard advantages patch cannot apply.")
    if '"advantages"' in inspect.getsource(tdc._package_shards):
        logger.info("miles _package_shards now forwards advantages itself; the rLLM patch is redundant.")

    # 3. the ungated FSDP call site the wrapper compensates for
    fsdp_source = inspect.getsource(fsdp_actor)
    if "compute_advantages_and_returns(" not in fsdp_source:
        raise RuntimeError("the miles FSDP actor no longer calls compute_advantages_and_returns; re-check whether rLLM's advantages still survive to the loss.")
    if "if self.args.compute_advantages_and_returns" in fsdp_source:
        logger.info("miles' FSDP actor now honours --disable-compute-advantages-and-returns; the rLLM wrapper is redundant.")


def apply_all_miles_patches() -> None:
    """Entry point for both the driver and the train workers."""
    assert_patch_contracts()
    patch_advantages_cp_slice()
    patch_package_shards_forwards_advantages()
    patch_respect_disable_compute_advantages()
