"""Single source of truth for rLLM's on-disk home directory and the paths under it.

rLLM stores user-level state (config, registered agents/evaluators, materialised
datasets, eval results, sandbox traces) under a single home directory. That
directory is ``$RLLM_HOME`` when set, otherwise ``~/.rllm``.

Every helper resolves the environment at call time, so:

* there is exactly one place that knows the default location and how ``~`` /
  ``$RLLM_HOME`` are expanded, and
* tests can redirect all storage with a single
  ``monkeypatch.setenv("RLLM_HOME", tmp_path)`` instead of patching scattered
  module constants.

Prefer these helpers over re-deriving ``os.environ.get("RLLM_HOME", ...)`` inline.
"""

from __future__ import annotations

import os

# Location used when ``$RLLM_HOME`` is not set. ``~`` is expanded by rllm_home().
DEFAULT_RLLM_HOME = "~/.rllm"


def rllm_home() -> str:
    """Return the rLLM home directory: ``$RLLM_HOME`` if set, else ``~/.rllm``.

    The result is always expanded (any leading ``~`` is resolved), so callers
    can pass it straight to ``open``/``os.makedirs`` without further handling.
    """
    return os.path.expanduser(os.environ.get("RLLM_HOME", DEFAULT_RLLM_HOME))


def rllm_path(*parts: str) -> str:
    """Return an absolute path under the rLLM home directory.

    ``rllm_path("datasets", "registry.json")`` -> ``<rllm_home>/datasets/registry.json``.
    """
    return os.path.join(rllm_home(), *parts)


def datasets_dir() -> str:
    """Return the datasets root (``<rllm_home>/datasets``)."""
    return rllm_path("datasets")


def eval_results_dir() -> str:
    """Return the eval-results root (``<rllm_home>/eval_results``)."""
    return rllm_path("eval_results")


def config_path() -> str:
    """Return the user-level config file path (``<rllm_home>/config.json``)."""
    return rllm_path("config.json")
