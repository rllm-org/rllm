"""Tests for DatasetRegistry v1 -> v2 migration."""

import json
import os

import pytest

from rllm.data.dataset import DatasetRegistry


@pytest.fixture
def tmp_rllm_home(monkeypatch, tmp_path):
    """Set up a temporary RLLM_HOME directory."""
    rllm_home = str(tmp_path / ".rllm")
    monkeypatch.setattr(DatasetRegistry, "_RLLM_HOME", rllm_home)
    monkeypatch.setattr(DatasetRegistry, "_REGISTRY_FILE", os.path.join(rllm_home, "datasets", "registry.json"))
    monkeypatch.setattr(DatasetRegistry, "_DATASET_DIR", os.path.join(rllm_home, "datasets"))
    return rllm_home


def test_fresh_registry_is_v2(tmp_rllm_home):
    """A fresh load should return v2 format."""
    registry = DatasetRegistry._load_registry()
    assert registry["version"] == 2
    assert "datasets" in registry
    assert isinstance(registry["datasets"], dict)


def test_register_creates_v2(tmp_rllm_home):
    """Registering a dataset should create a v2 registry."""
    data = [{"x": 1}, {"x": 2}]
    DatasetRegistry.register_dataset("test_ds", data, split="train")

    registry = DatasetRegistry._load_registry()
    assert registry["version"] == 2
    assert "test_ds" in registry["datasets"]
    assert "train" in registry["datasets"]["test_ds"]["splits"]
    assert registry["datasets"]["test_ds"]["splits"]["train"]["num_examples"] == 2
    assert "x" in registry["datasets"]["test_ds"]["splits"]["train"]["fields"]


def test_v1_migration(tmp_rllm_home, tmp_path):
    """A v1 registry should be auto-migrated to v2."""
    # Create a fake v1 registry at the new location
    datasets_dir = os.path.join(tmp_rllm_home, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)

    # Create a fake parquet file
    import pandas as pd

    ds_dir = os.path.join(datasets_dir, "myds")
    os.makedirs(ds_dir, exist_ok=True)
    fake_parquet = os.path.join(ds_dir, "train.parquet")
    pd.DataFrame([{"q": "hello"}]).to_parquet(fake_parquet)

    # Write v1 format to the new location
    v1_registry = {"myds": {"train": fake_parquet}}
    registry_file = os.path.join(datasets_dir, "registry.json")
    with open(registry_file, "w") as f:
        json.dump(v1_registry, f)

    # Load should auto-migrate
    registry = DatasetRegistry._load_registry()
    assert registry["version"] == 2
    assert "myds" in registry["datasets"]
    assert "train" in registry["datasets"]["myds"]["splits"]


def test_load_dataset_v2(tmp_rllm_home):
    """Load dataset should work with v2 registry."""
    data = [{"question": "1+1", "answer": "2"}, {"question": "2+2", "answer": "4"}]
    DatasetRegistry.register_dataset("math_test", data, split="test")

    ds = DatasetRegistry.load_dataset("math_test", "test")
    assert ds is not None
    assert len(ds) == 2
    assert ds[0]["question"] == "1+1"


def test_dataset_info(tmp_rllm_home):
    """get_dataset_info should return metadata and splits."""
    data = [{"q": "x"}]
    DatasetRegistry.register_dataset("info_test", data, split="train", source="test-source", category="test-cat")

    info = DatasetRegistry.get_dataset_info("info_test")
    assert info is not None
    assert info["metadata"]["source"] == "test-source"
    assert info["metadata"]["category"] == "test-cat"
    assert "train" in info["splits"]


def test_remove_split_v2(tmp_rllm_home):
    """Removing a split should work with v2 format."""
    data = [{"q": "x"}]
    DatasetRegistry.register_dataset("rm_test", data, split="train")
    DatasetRegistry.register_dataset("rm_test", data, split="test")

    assert DatasetRegistry.remove_dataset_split("rm_test", "train")
    assert not DatasetRegistry.dataset_exists("rm_test", "train")
    assert DatasetRegistry.dataset_exists("rm_test", "test")


def test_remove_dataset_v2(tmp_rllm_home):
    """Removing a dataset should work with v2 format."""
    data = [{"q": "x"}]
    DatasetRegistry.register_dataset("rm_all", data, split="test")

    assert DatasetRegistry.remove_dataset("rm_all")
    assert not DatasetRegistry.dataset_exists("rm_all")


def test_backward_compat_load(tmp_rllm_home):
    """The public load_dataset API should work unchanged."""
    data = [{"question": "hi", "ground_truth": "there"}]
    DatasetRegistry.register_dataset("compat", data, split="train")

    ds = DatasetRegistry.load_dataset("compat", "train")
    assert ds is not None
    assert ds.name == "compat"
    assert ds.split == "train"
    assert len(ds) == 1
