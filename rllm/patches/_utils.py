"""Shared utilities for verl patches using wrapt."""

import inspect
import sys

from wrapt import wrap_function_wrapper
from wrapt.importer import when_imported

# These are magic numbers used to ensure we identify a single lora throughout the training process.
# TODO: In the future, we might want to support multiple loras and this need to be updated.
LORA_ID_MAGIC = 123
LORA_PATH_MAGIC = "simon_lora_path"


def get_bounded_args(wrapped, args, kwargs):
    sig = inspect.signature(wrapped)

    bounded_args = sig.bind(*args, **kwargs)
    bounded_args.apply_defaults()
    return bounded_args.arguments


_logger_registry = {}


def get_logger(module_name: str):
    """Get a logger for a module. Use the registry to avoid creating multiple loggers for the same module."""
    if module_name in _logger_registry:
        return _logger_registry[module_name]

    import importlib as _importlib
    import logging
    import os

    mod = _importlib.import_module(module_name)
    logger = getattr(mod, "logger", None) or logging.getLogger(mod.__name__)
    if not getattr(logger, "level", None):
        logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

    _logger_registry[module_name] = logger
    return logger


def vllm_max_lora_rank(lora_rank: int):
    """
    Fix: for vLLM, the smallest `max_lora_rank` is 8, and allowed values are (8, 16, 32, 64, 128, 256, 320, 512)
    verl mistakenly set `max_lora_rank = config.lora_rank`,this prevents us from using very small lora_rank (e.g. 1).
    """
    assert lora_rank > 0, "lora_rank must be greater than 0 to invoke this function."
    vllm_max_lora_ranks = [8, 16, 32, 64, 128, 256, 320, 512]
    max_lora_idx = 0
    while max_lora_idx < len(vllm_max_lora_ranks) and vllm_max_lora_ranks[max_lora_idx] < lora_rank:
        max_lora_idx += 1

    max_lora_rank = vllm_max_lora_ranks[max_lora_idx]

    if lora_rank > max_lora_rank:
        print(
            f"Warning: vLLM only supports lora_rank up to {max_lora_rank}, \
            your lora_rank {lora_rank} might have the risk of causing OOM issues."
        )

    return max_lora_rank


def wrap_class_method_once(module_name: str, cls_name: str, method_name: str, wrapper):
    """
    Wrap a class method once, descriptor-safe, even if hot-reloaded/imported twice.

    Args:
        module_name: Full module path (e.g., "verl.workers.rollout.vllm_rollout.vllm_rollout_spmd")
        cls_name: Name of the class to patch
        method_name: Name of the method to wrap
        wrapper: Wrapper function with signature (wrapped, instance, args, kwargs)
    """
    sentinel = f"__patched_{cls_name}_{method_name}__"

    def apply(mod):
        cls = getattr(mod, cls_name, None)
        if cls is None or getattr(cls, sentinel, False):
            return
        wrap_function_wrapper(cls, method_name, wrapper)
        setattr(cls, sentinel, True)

    @when_imported(module_name)
    def _on_import(mod):
        apply(mod)

    # If already imported, patch immediately
    if module_name in sys.modules:
        apply(sys.modules[module_name])


def wrap_function_once(module_name: str, function_name: str, wrapper):
    """
    Wrap a module-level function once, even if hot-reloaded/imported twice.

    Args:
        module_name: Full module path
        function_name: Name of the function to wrap
        wrapper: Wrapper function with signature (wrapped, instance, args, kwargs)
    """
    sentinel = f"__patched_fn_{function_name}__"

    def apply(mod):
        if getattr(mod, sentinel, False):
            return
        wrap_function_wrapper(mod, function_name, wrapper)
        setattr(mod, sentinel, True)

    @when_imported(module_name)
    def _on_import(mod):
        apply(mod)

    # If already imported, patch immediately
    if module_name in sys.modules:
        apply(sys.modules[module_name])
