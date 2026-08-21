"""OmegaConf -> Miles ``argparse.Namespace``.

Miles builds one flat Namespace on top of Megatron's own ``parse_args``
(``miles/utils/arguments.py``), so every Megatron flag shares the Miles CLI. We
render an argv list and let Miles parse it, rather than hand-building a
Namespace: ``miles_validate_args`` / ``megatron_validate_args`` /
``set_default_megatron_args`` do real derivation we want to run.

Config keys are **flag names** in snake_case, not argparse dests. That matters
for negated flags: Miles spells "don't load a dataset" as
``--disable-rollout-global-dataset`` (``action="store_false"``,
``dest="rollout_global_dataset"``), so keying on the flag avoids having to model
the dest/flag inversion.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

# Flags rLLM owns. The user cannot set these; validate_pinned() rejects attempts.
PINNED_FLAGS: dict[str, Any] = {
    # rLLM drives generation. The RolloutManager stays up only to own the SGLang
    # fleet, the router, and the weight-update broker -- it must never generate.
    "rollout_function_path": "miles.rollout.sleep_rollout.sleep",
    "eval_function_path": "miles.rollout.sleep_rollout.sleep",
    # rLLM owns the dataset; skips Miles' Dataset + tokenizer load.
    "disable_rollout_global_dataset": True,
    # Advantages arrive per-token from rLLM inside the batch.
    "disable_compute_advantages_and_returns": True,
}

# Flags that configure *Miles'* generation path, which rLLM bypasses entirely.
# Several of them also select a rollout function and would silently override -- or
# assert against -- our sleep_rollout pin (see resolve_rollout_function_paths and
# _resolve_rollout_functions in miles/utils/arguments.py), so reject them with an
# explanation instead of letting Miles emit a confusing conflict.
REJECTED_GENERATION_FLAGS: dict[str, str] = {
    "fully_async": "overrides rollout_function_path with FullyAsyncRolloutFn, and asserts when both are set",
    "multi_lora": "selects a rollout function of its own",
    "partial_rollout": "resumes Miles-side generation, which never runs here",
    "custom_generate_function_path": "configures Miles' generate path",
    "custom_agent_function_path": "configures Miles' TITO agent path",
    "use_session_server": "the TITO session server serves Miles-side agents; rLLM owns tokenization",
    "rollout_global_dataset": "spelled --disable-rollout-global-dataset; the dataset lives on the rLLM side",
    # miles_validate_args: "Evaluation datasets must be configured when eval_interval is set."
    # rLLM runs validation through its own workflow engine, so Miles never evaluates.
    "eval_interval": "rLLM runs validation itself; Miles' eval path is pinned to sleep_rollout",
    "eval_prompt_data": "eval datasets live on the rLLM side",
    "eval_config": "eval datasets live on the rLLM side",
}


# (miles_flag, rllm_config_path) -- mirrored so users set these once on the rllm.* side.
SHARED_KEYS: list[tuple[str, str]] = [
    ("hf_checkpoint", "model.name"),
    ("rollout_batch_size", "rllm.data.train_batch_size"),
    ("rollout_max_prompt_len", "rllm.data.max_prompt_length"),
    ("save_interval", "rllm.trainer.save_freq"),
]


def _flag(key: str) -> str:
    return "--" + key.replace("_", "-")


def miles_arity(argv: list[str] | None = None) -> dict[str, int | str]:
    """Value count per Miles flag, introspected from Miles' own parser.

    0 for store_true/store_false, 1 for a scalar, "+" for nargs.

    Parts of Miles' argument provider partial-parse ``sys.argv`` while registering
    flags, so pass ``argv`` (without the program name) to install a valid one for
    the duration. Otherwise the host process's argv leaks in and argparse prints a
    spurious "the following arguments are required" usage error to stderr.
    """
    import argparse

    from miles.utils.arguments import get_miles_extra_args_provider

    saved = sys.argv
    try:
        if argv is not None:
            sys.argv = ["rllm-miles", *argv]
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
    finally:
        sys.argv = saved

    arity: dict[str, int | str] = {}
    for action in parser._actions:
        n = 0 if action.nargs == 0 or isinstance(action.const, bool) else (action.nargs or 1)
        if action.nargs in ("+", "*"):
            n = "+"
        elif not isinstance(action.nargs, int) and action.nargs is not None:
            n = 1
        for opt in action.option_strings:
            if opt.startswith("--"):
                arity[opt[2:].replace("-", "_")] = n
    return arity


def _infer_arity(value: Any) -> int | str:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (list, tuple)):
        return "+"
    return 1


def render_argv(block: dict[str, Any], arity: dict[str, int | str] | None = None) -> list[str]:
    """Render a ``{flag_name: value}`` mapping to argv.

    ``None`` values are omitted (let Miles' own default stand). A zero-arity flag
    is emitted bare when truthy and omitted when falsy. Flags absent from
    ``arity`` fall back to inferring it from the Python type, which is why the
    Megatron half of the CLI works without introspecting Megatron.
    """
    arity = arity or {}
    argv: list[str] = []
    for key, value in block.items():
        if value is None:
            continue
        n = arity.get(key, _infer_arity(value))
        if n == 0:
            if value:
                argv.append(_flag(key))
            continue
        if n == "+":
            if not isinstance(value, (list, tuple)):
                raise TypeError(f"miles.{key} takes a list, got {type(value).__name__}: {value!r}")
            if not value:
                continue
            argv.append(_flag(key))
            argv.extend(str(v) for v in value)
            continue
        if isinstance(value, (list, tuple)):
            raise TypeError(f"miles.{key} takes a single value, got a list: {value!r}")
        argv.extend([_flag(key), str(value)])
    return argv


def validate_pinned(block: dict[str, Any]) -> None:
    """Reject user-set flags that rLLM owns."""
    clashes = sorted(set(block) & set(PINNED_FLAGS))
    if clashes:
        raise ValueError("These Miles flags are set by the rLLM backend and cannot be overridden: " + ", ".join(f"miles.{c}" for c in clashes))
    for flag, why in REJECTED_GENERATION_FLAGS.items():
        if block.get(flag):
            raise ValueError(f"miles.{flag} is not supported by the rLLM backend: {why}. rLLM drives generation, so Miles' rollout path is pinned to sleep_rollout.")

    if block.get("colocate"):
        raise ValueError("miles.colocate is not supported yet: rLLM generates while Miles trains, so the two must not share GPUs.")
    if block.get("num_epoch") is not None:
        # Miles asserts num_epoch requires rollout_global_dataset, which we disable.
        raise ValueError("Set epochs on the rLLM side (rllm.trainer.total_epochs); miles.num_epoch needs Miles' own dataset.")


def build_block(config: DictConfig, total_steps: int | None = None) -> tuple[dict[str, Any], list[Any]]:
    """Assemble the flag mapping: user's ``miles:`` block + shared keys + pinned."""
    node = config.get("miles", None)
    # An empty DictConfig is falsy, so `or {}` here would hand to_container a plain dict.
    raw = OmegaConf.to_container(node, resolve=True) if node is not None else {}
    if not isinstance(raw, dict):
        raise TypeError(f"`miles:` must be a mapping of flag names to values, got {type(raw).__name__}")
    extra_args = raw.pop("extra_args", None) or []
    validate_pinned(raw)

    block: dict[str, Any] = dict(raw)
    for miles_flag, rllm_path in SHARED_KEYS:
        if block.get(miles_flag) is not None:
            continue  # explicit miles.* wins
        value = OmegaConf.select(config, rllm_path)
        if value is not None:
            block[miles_flag] = value

    # rLLM owns the schedule. num_rollout is required because --num-epoch asserts
    # Miles' global dataset, which we disable.
    if total_steps is not None:
        block["num_rollout"] = total_steps
    block.update(PINNED_FLAGS)
    return block, extra_args


def build_miles_args(config: DictConfig, total_steps: int | None = None):
    """Build the Miles ``Namespace``. Imports miles, so needs the Miles image."""
    from miles.utils.arguments import parse_args

    block, extra_args = build_block(config, total_steps=total_steps)
    tail = [str(a) for a in extra_args]

    # Two passes, because building argv needs the arity map and introspecting the
    # arity map needs a valid argv: parts of Miles' argument provider partial-parse
    # sys.argv while registering flags, and against the host process's argv that
    # fails on Miles' required flags (printing a bogus usage error to stderr).
    # Pass 1 infers arity from config value types -- enough to render a valid argv.
    saved = sys.argv
    try:
        pass1 = render_argv(block) + tail
        sys.argv = ["rllm-miles", *pass1]
        try:
            arity = miles_arity(pass1)
        except Exception as e:  # pragma: no cover - introspection is best-effort
            logger.warning("Could not introspect Miles' parser (%s); inferring flag arity from config types.", e)
            arity = {}

        # Pass 2 re-renders against Miles' declared arity, which is authoritative.
        argv = render_argv(block, arity) + tail
        if arity and argv != pass1:
            logger.warning("Inferred flag arity disagreed with Miles' parser; using Miles'. Check miles.* value types.")
        logger.info("miles argv: %s", " ".join(argv))

        sys.argv = ["rllm-miles", *argv]
        return parse_args()
    finally:
        sys.argv = saved
