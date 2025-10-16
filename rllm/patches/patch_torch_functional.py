"""Patches for verl.utils.experimental.torch_functional module.

This module patches the FusedLinearForPPOFunction class to:
- Move `output_requires_grad` calculation to immediately after `ctx.set_materialize_grads(False)`

This helps address the issue of vanishing gradients during LoRA (which turns `vocab_weights.requires_grad == False`).

Reference: https://github.com/volcengine/verl/pull/3765
"""

from ._utils import get_bounded_args, wrap_class_method_once

TARGET_MODULE = "verl.utils.experimental.torch_functional"

# ============================================================================
# Patches for FusedLinearForPPOFunction class
# ============================================================================


def forward_wrapper(wrapped, instance, args, kwargs):
    """
    Patch the forward method to move output_requires_grad calculation earlier.
    """
    from verl.utils.experimental.torch_functional import _fused_linear_for_ppo_fwd

    # Extract arguments - note that for staticmethod, first arg is ctx
    ba = get_bounded_args(wrapped, args, kwargs)
    ctx = ba["ctx"]
    hidden_states = ba["hidden_states"]
    vocab_weights = ba["vocab_weights"]
    input_ids = ba["input_ids"]
    temperature = ba.get("temperature", 1.0)
    chunk_size = ba.get("chunk_size", 512)

    ctx.set_materialize_grads(False)
    # Fix: move this calculation to immediately after set_materialize_grads
    output_requires_grad = hidden_states.requires_grad or vocab_weights.requires_grad

    # Cast to a 2D tensor of the shape [T, D] for ease of working
    orig_ndim = hidden_states.ndim
    assert orig_ndim in (2, 3), f"Invalid hidden_states shape, received {hidden_states.shape}"

    orig_batch_size = -1
    if orig_ndim == 3:
        assert input_ids.ndim == 2, f"input_ids shape doesn't match, {hidden_states.shape} {input_ids.shape}"
        orig_batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.flatten(0, 1)
        input_ids = input_ids.flatten(0, 1)

    T = hidden_states.shape[0]

    # Allocate memory for outputs - use the pre-calculated output_requires_grad
    log_probs = hidden_states.new_zeros(T, requires_grad=output_requires_grad)
    entropy = hidden_states.new_zeros(T, requires_grad=output_requires_grad)

    # Perform forward one chunk at a time
    for chunk_start in range(0, T, chunk_size):
        chunk_end = min(chunk_start + chunk_size, T)

        chunk_log_probs, chunk_entropy = _fused_linear_for_ppo_fwd(
            hidden_states=hidden_states[chunk_start:chunk_end],
            vocab_weights=vocab_weights,
            input_ids=input_ids[chunk_start:chunk_end],
            temperature=temperature,
        )
        log_probs[chunk_start:chunk_end] = chunk_log_probs
        entropy[chunk_start:chunk_end] = chunk_entropy

    # Cast the output back to the original input dimension
    if orig_ndim == 3:
        log_probs = log_probs.view(orig_batch_size, -1)
        entropy = entropy.view(orig_batch_size, -1)

    ctx.save_for_backward(hidden_states, vocab_weights, input_ids)
    ctx.orig_batch_size = orig_batch_size
    ctx.orig_ndim = orig_ndim
    ctx.temperature = temperature
    ctx.chunk_size = chunk_size

    return log_probs, entropy


def register():
    cls_name = "FusedLinearForPPOFunction"
    wrap_class_method_once(TARGET_MODULE, cls_name, "forward", forward_wrapper)
