"""Smoke tests for the swe-rl cookbook.

These tests don't boot sandboxes, run agents, or train. They only
verify the wiring: the dataset names resolve in the catalog, the
``mini-swe-agent`` harness is importable, and ``train.py`` can be
imported without side effects.

Run::

    pytest cookbooks/swe-rl/test.py -v
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

# -- Dataset catalog entries --------------------------------------------------


def _load_catalog() -> dict:
    registry_path = Path(__file__).resolve().parents[2] / "rllm" / "registry" / "datasets.json"
    return json.loads(registry_path.read_text())


def test_train_dataset_in_catalog():
    """r2egym must be registered as a sandbox dataset using mini-swe-agent."""
    catalog = _load_catalog()
    entry = catalog["datasets"]["r2egym"]
    assert entry["category"] == "code"
    assert entry["default_agent"] == "mini-swe-agent"
    assert "train" in entry["splits"]


def test_val_dataset_resolvable():
    """SWE-bench Verified must be reachable either as the native row dataset
    (``swebench_verified``) or as the Harbor sandbox dataset
    (``harbor:swebench-verified``)."""
    catalog = _load_catalog()
    assert "swebench_verified" in catalog["datasets"], "native swebench_verified entry missing from registry"


# -- Harness wiring -----------------------------------------------------------


def test_mini_swe_agent_harness_importable():
    """``mini-swe-agent`` is the harness this cookbook drives; it must import."""
    mod = importlib.import_module("rllm.harnesses.mini_swe_agent")
    assert hasattr(mod, "MiniSweAgentHarness")
    assert mod.MiniSweAgentHarness.name == "mini-swe-agent"


# -- Train script -------------------------------------------------------------


def test_train_module_imports():
    """``train.py`` must be importable without triggering Hydra or starting training."""
    import sys

    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    mod = importlib.import_module("train")
    assert mod.TRAIN_DATASET == "r2egym"
    assert mod.VAL_DATASET == "swebench-verified"
    assert callable(mod.main)
