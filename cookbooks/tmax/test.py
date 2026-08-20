"""Smoke tests for the tmax cookbook.

These tests don't boot sandboxes, run agents, or train. They only verify the
wiring: the terminal harnesses are importable, the Harbor loader is reachable,
the ``tmax-15k`` dataset is registered in the in-repo catalog with a Harbor
source, and ``train.py`` / ``prepare_data.py`` import without side effects and
expose the expected dataset names.

Run::

    pytest cookbooks/tmax/test.py -v
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

_COOKBOOK_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _COOKBOOK_DIR.parent.parent
_CATALOG = _REPO_ROOT / "rllm" / "registry" / "datasets.json"


def _import_cookbook_module(name: str):
    if str(_COOKBOOK_DIR) not in sys.path:
        sys.path.insert(0, str(_COOKBOOK_DIR))
    return importlib.import_module(name)


# -- Harness wiring -----------------------------------------------------------


def test_terminus2_harness_importable():
    """``terminus2`` is the default harness this cookbook drives; it must import."""
    mod = importlib.import_module("rllm.harnesses.terminus2")
    assert hasattr(mod, "Terminus2Harness")
    assert mod.Terminus2Harness.name == "terminus2"


def test_mini_swe_agent_harness_importable():
    """``mini-swe-agent`` is the high-fidelity (Vanillux2-like) harness option."""
    mod = importlib.import_module("rllm.harnesses.mini_swe_agent")
    assert hasattr(mod, "MiniSweAgentHarness")
    assert mod.MiniSweAgentHarness.name == "mini-swe-agent"


def test_harbor_loader_importable():
    """tmax-15k is pulled + flattened via the Harbor dataset loader."""
    mod = importlib.import_module("rllm.integrations.harbor.dataset_loader")
    assert hasattr(mod, "harbor_task_to_row")


# -- Dataset catalog ----------------------------------------------------------


def test_tmax15k_in_catalog():
    """``tmax-15k`` must be a first-class catalog entry built from the HF corpus."""
    catalog = json.loads(_CATALOG.read_text(encoding="utf-8"))
    entry = catalog["datasets"].get("tmax-15k")
    assert entry is not None, "tmax-15k missing from rllm/registry/datasets.json"
    assert entry["source"] == "allenai/TMax-15K", entry["source"]
    assert entry["builder"] == "rllm.data.tmax_builder:build_benchmark", entry.get("builder")
    assert "train" in entry["splits"]


def test_tmax_builder_importable():
    """The native builder that materializes TMax-15K must import and expose build_benchmark."""
    mod = importlib.import_module("rllm.data.tmax_builder")
    assert callable(mod.build_benchmark)
    assert mod.DEFAULT_HF_REPO_ID == "allenai/TMax-15K"
    assert mod.IMAGE_HF_REPO_ID == "allenai/tmax-15k-open-instruct"


# -- Cookbook scripts ---------------------------------------------------------


def test_train_module_imports():
    """``train.py`` must import without triggering Hydra or starting training."""
    mod = _import_cookbook_module("train")
    assert mod.TRAIN_DATASET == "tmax-15k"
    assert mod.VAL_DATASET.startswith("terminal-bench@")
    assert mod.TMAX_HARNESS in ("terminus2", "mini-swe-agent")
    assert callable(mod.main)


def test_prepare_data_module_imports():
    """``prepare_data.py`` must import and expose its dataset names."""
    mod = _import_cookbook_module("prepare_data")
    assert mod.TRAIN_DATASET == "tmax-15k"
    assert mod.EVAL_DATASET.startswith("terminal-bench@")
    assert callable(mod.main)
