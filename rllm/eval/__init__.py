"""rLLM evaluation package.

Re-exports the public eval API. Imports are lazy (PEP 562): heavy submodules
(``runner``, ``proxy``, ``rllm.types`` and their transitive torch/verl deps) are
only loaded when the corresponding name is first accessed. This keeps lightweight
consumers — e.g. the ``rllm model`` CLI, which only needs ``config`` — fast.
"""

from __future__ import annotations

import importlib

# name -> submodule that defines it
_EXPORTS = {
    "load_agent": "rllm.eval.agent_loader",
    "load_evaluator": "rllm.eval.evaluator_loader",
    "resolve_evaluator_from_catalog": "rllm.eval.evaluator_loader",
    "run_dataset": "rllm.eval.runner",
    "EvalResult": "rllm.eval.results",
    "EvalItem": "rllm.eval.results",
    "RllmConfig": "rllm.eval.config",
    "load_config": "rllm.eval.config",
    "save_config": "rllm.eval.config",
    "EvalProxyManager": "rllm.eval.proxy",
    "AgentConfig": "rllm.types",
    "AgentFlow": "rllm.types",
    "Evaluator": "rllm.types",
    "EvalOutput": "rllm.eval.types",
    "Signal": "rllm.eval.types",
    "run_agent_flow": "rllm.types",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_path), name)


def __dir__() -> list[str]:
    return __all__
