"""Resolve the system-prompt hint for a task's verifier.

Each score_fn module may define a ``SYSTEM_PROMPT`` string describing the
output format the grader expects (e.g. ``\\boxed{}`` for math). Harnesses
call :func:`get_verifier_system_prompt` to inject that hint into their
system prompt, so the LLM produces output the grader can parse.
"""

from __future__ import annotations

import importlib
import logging

import tomllib

from rllm.types import Task

logger = logging.getLogger(__name__)


# Map of legacy reward-fn names → score_fn module paths
_REWARD_FN_TO_SCORE_FN: dict[str, str] = {
    "math_reward_fn": "rllm.eval.reward_fns.math",
    "mcq_reward_fn": "rllm.eval.reward_fns.mcq",
    "f1_reward_fn": "rllm.eval.reward_fns.f1",
    "code_reward_fn": "rllm.eval.reward_fns.code",
    "countdown_reward_fn": "rllm.eval.reward_fns.countdown",
    "iou_reward_fn": "rllm.eval.reward_fns.iou",
    "point_in_mask_reward_fn": "rllm.eval.reward_fns.point_in_mask",
    "depth_reward_fn": "rllm.eval.reward_fns.depth",
    "bfcl_reward_fn": "rllm.eval.reward_fns.bfcl",
    "ifeval_reward_fn": "rllm.eval.reward_fns.ifeval",
    "llm_equality_reward_fn": "rllm.eval.reward_fns.llm_equality",
    "llm_judge_reward_fn": "rllm.eval.reward_fns.llm_judge",
    "translation_reward_fn": "rllm.eval.reward_fns.translation",
    "widesearch_reward_fn": "rllm.eval.reward_fns.widesearch",
}


def get_verifier_system_prompt(task: Task) -> str | None:
    """Return the SYSTEM_PROMPT exported by this task's verifier module, if any.

    Reads ``[verifier]`` from ``task.dataset_dir/dataset.toml`` (or
    ``task.toml`` if per-task), resolves the module, and returns its
    ``SYSTEM_PROMPT`` attribute. Returns ``None`` if the verifier doesn't
    expose one (e.g. shell-script verifiers).
    """
    cfg = _read_verifier_config(task)
    if not cfg:
        return None

    module_name = None
    if "import_path" in cfg:
        module_name = cfg["import_path"].split(":", 1)[0]
    elif "name" in cfg:
        module_name = _REWARD_FN_TO_SCORE_FN.get(cfg["name"])

    if not module_name:
        return None

    try:
        module = importlib.import_module(module_name)
        return getattr(module, "SYSTEM_PROMPT", None)
    except Exception as e:
        logger.debug("Failed to resolve system prompt for %s: %s", module_name, e)
        return None


def _read_verifier_config(task: Task) -> dict:
    candidates = []
    if task.sub_dir is not None:
        candidates.append(task.dataset_dir / task.sub_dir / "task.toml")
    candidates.append(task.dataset_dir / "dataset.toml")
    for cfg_path in candidates:
        if cfg_path.exists():
            try:
                raw = tomllib.loads(cfg_path.read_text())
            except Exception:
                continue
            verifier = raw.get("verifier", {})
            if verifier:
                return verifier
    return {}
