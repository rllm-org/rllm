"""Smoke tests for the terminal-rl cookbook.

These tests don't boot sandboxes, run agents, or train. They only verify the
wiring: the terminus2 harness is importable, the Harbor task loader is
reachable, and ``train.py`` / ``prepare_data.py`` import without side effects
and expose the expected dataset names.

Run::

    pytest cookbooks/terminal-rl/test.py -v
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_COOKBOOK_DIR = Path(__file__).resolve().parent


def _import_cookbook_module(name: str):
    if str(_COOKBOOK_DIR) not in sys.path:
        sys.path.insert(0, str(_COOKBOOK_DIR))
    return importlib.import_module(name)


# -- Harness wiring -----------------------------------------------------------


def test_terminus2_harness_importable():
    """``terminus2`` is the harness this cookbook drives; it must import."""
    mod = importlib.import_module("rllm.harnesses.terminus2")
    assert hasattr(mod, "Terminus2Harness")
    assert mod.Terminus2Harness.name == "terminus2"


def test_harbor_loader_importable():
    """The local training tarball is ingested via the Harbor task loader."""
    mod = importlib.import_module("rllm.integrations.harbor.dataset_loader")
    assert hasattr(mod, "harbor_task_to_row")


# -- Cookbook scripts ---------------------------------------------------------


def test_train_module_imports():
    """``train.py`` must import without triggering Hydra or starting training."""
    mod = _import_cookbook_module("train")
    assert mod.TRAIN_DATASET == "tb-opus-pass"
    assert mod.VAL_DATASET.startswith("terminal-bench@")
    assert callable(mod.main)


def test_prepare_data_module_imports():
    """``prepare_data.py`` must import and expose its dataset names."""
    mod = _import_cookbook_module("prepare_data")
    assert mod.TRAIN_DATASET == "tb-opus-pass"
    assert mod.EVAL_DATASET.startswith("terminal-bench@")
    assert callable(mod.main)
