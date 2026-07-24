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
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

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


def test_harbor_loader_selects_package_client_for_namespaced_dataset():
    mod = importlib.import_module("rllm.integrations.harbor.dataset_loader")

    package_client = mod._registry_client_for("terminal-bench/terminal-bench-2-1@6")
    legacy_client = mod._registry_client_for("terminal-bench@2.0")

    assert type(package_client).__name__ == "PackageDatasetClient"
    assert type(legacy_client).__name__ != "PackageDatasetClient"


# -- Cookbook scripts ---------------------------------------------------------


def test_train_module_imports():
    """``train.py`` must import without triggering Hydra or starting training."""
    mod = _import_cookbook_module("train")
    assert mod.TRAIN_DATASET == "tb-opus-pass"
    assert mod.VAL_DATASET.startswith("terminal-bench@")
    assert mod.VAL_EXPECTED_TASKS == 0
    assert mod.BENCHMARK_DATASET == ""
    assert callable(mod.main)


def test_prepare_data_module_imports():
    """``prepare_data.py`` must import and expose its dataset names."""
    mod = _import_cookbook_module("prepare_data")
    assert mod.TRAIN_DATASET == "tb-opus-pass"
    assert mod.DEBUG_DATASET == "tb_v2_debug"
    assert mod.LEGACY_EVAL_DATASET == "terminal-bench@2.0"
    assert mod.LEGACY_EVAL_EXPECTED_TASKS == 89
    assert mod.MIDTEST_SPLIT == "midtest"
    assert mod.EVAL_DATASET == "terminal-bench@2.1"
    assert mod.EVAL_SOURCE == "terminal-bench/terminal-bench-2-1@6"
    assert mod.EVAL_EXPECTED_TASKS == 89
    assert mod.DEFAULT_MIDTEST_SIZE == 8
    assert mod._tasks_root() != mod._debug_tasks_root()
    assert callable(mod.main)


def test_prepare_data_midtest_is_deterministic_fixed_benchmark_subset():
    mod = _import_cookbook_module("prepare_data")
    rows = [{"task_id": f"task-{idx:03d}"} for idx in range(100)]

    midtest_a = mod._select_fixed_subset(rows, subset_size=8, subset_seed=20260723)
    midtest_b = mod._select_fixed_subset(list(reversed(rows)), subset_size=8, subset_seed=20260723)

    midtest_ids_a = {row["task_id"] for row in midtest_a}
    assert len(midtest_a) == 8
    assert midtest_ids_a <= {row["task_id"] for row in rows}
    assert midtest_ids_a == {row["task_id"] for row in midtest_b}


def test_prepare_data_extracts_wrapped_zip_and_ignores_macosx(tmp_path):
    mod = _import_cookbook_module("prepare_data")
    archive = tmp_path / "tasks.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("tb_tasks/example/task.toml", "version = '1.0'\n")
        zf.writestr("__MACOSX/tb_tasks/._example", "metadata")

    tasks_root = mod._extract_archive(archive, tmp_path / "extracted")

    assert tasks_root.name == "tb_tasks"
    assert (tasks_root / "example" / "task.toml").is_file()


def test_prepare_data_rejects_zip_parent_traversal(tmp_path):
    mod = _import_cookbook_module("prepare_data")
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape", "unsafe")

    with pytest.raises(ValueError, match="Unsafe ZIP member path"):
        mod._extract_archive(archive, tmp_path / "extracted")


def test_glm5p2_production_profile_is_full_opencode_full_suite_every_ten_steps(tmp_path):
    script = (_COOKBOOK_DIR / "train_fireworks_glm5p2.sh").read_text()

    assert "production phase requires: full opencode production" in script
    assert 'val_dataset="terminal-bench@2.1"' in script

    env = os.environ.copy()
    env.update(
        {
            "FIREWORKS_API_KEY": "dry-run",
            "WANDB_API_KEY": "dry-run",
            "RLLM_PYTHON": "/bin/echo",
            "TB_STATE_ROOT": str(tmp_path),
            "TB_RUN_STAMP": "dryrun",
            "TB_TRAINER_REGION": "AP_MALAYSIA_2",
            "TB_TRAINER_REPLICAS": "4",
            "TB_ROLLOUT_REPLICAS": "4",
        }
    )
    result = subprocess.run(
        ["bash", str(_COOKBOOK_DIR / "train_fireworks_glm5p2.sh"), "full", "opencode", "production"],
        cwd=_COOKBOOK_DIR,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "val=terminal-bench@2.1/default benchmark=disabled" in result.stdout
    assert "fireworks_config.policy_trainer_replica_count=4" in result.stdout
    assert "fireworks_config.rollout_deployment_replica_count=4" in result.stdout
    assert "fireworks_infra.trainers.policy.region=AP_MALAYSIA_2" in result.stdout
    assert "rllm.trainer.val_before_train=true" in result.stdout
    assert "rllm.trainer.benchmark_before_train=false" in result.stdout
    assert "rllm.trainer.benchmark_after_train=false" in result.stdout
    assert "rllm.trainer.test_freq=10" in result.stdout


def test_glm5p2_sanity_profile_is_one_step_lora_opencode_without_midtest():
    script = (_COOKBOOK_DIR / "train_fireworks_glm5p2.sh").read_text()

    assert "sanity phase requires: lora opencode sanity" in script
    assert 'if [ "$phase" = "sanity" ]; then' in script
    assert "test_freq=-1" in script
    assert "val_before_train=false" in script
    assert "benchmark_after_train=false" in script
    assert 'total_batches="${TB_SANITY_TOTAL_BATCHES:-1}"' in script
    assert 'trainer_replicas="${TB_TRAINER_REPLICAS:-4}"' in script
    assert 'rollout_replicas="${TB_ROLLOUT_REPLICAS:-4}"' in script
