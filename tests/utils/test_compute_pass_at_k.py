"""``rllm.utils.compute_pass_at_k`` is advertised in ``rllm.utils.__all__``.

Importing it must hand back the function, and must work on a core install
(the symbol is lazy precisely because the module used to need torch).
"""

from __future__ import annotations

import importlib
import sys


def _forget_rllm_utils(monkeypatch):
    """Drop the cached package so the lazy ``__getattr__`` path runs from scratch."""
    for name in [n for n in list(sys.modules) if n == "rllm.utils" or n.startswith("rllm.utils.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)


def test_from_import_yields_the_function(monkeypatch):
    """``from rllm.utils import compute_pass_at_k`` must not resolve to the submodule."""
    _forget_rllm_utils(monkeypatch)

    from rllm.utils import compute_pass_at_k

    assert callable(compute_pass_at_k), f"expected a function, got {compute_pass_at_k!r}"


def test_repeated_attribute_access_yields_the_function(monkeypatch):
    _forget_rllm_utils(monkeypatch)

    import rllm.utils

    assert callable(rllm.utils.compute_pass_at_k)
    assert callable(rllm.utils.compute_pass_at_k)


def test_module_imports_without_torch(monkeypatch):
    """Core ``pip install rllm`` has no torch; only save_trajectories needs it."""
    _forget_rllm_utils(monkeypatch)
    monkeypatch.setitem(sys.modules, "torch", None)

    module = importlib.import_module("rllm.utils.compute_pass_at_k")

    assert callable(module.compute_pass_at_k)
