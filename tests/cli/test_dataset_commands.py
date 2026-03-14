"""Tests for rllm dataset CLI commands."""

import csv
import json
import os

import pytest
from click.testing import CliRunner

from rllm.experimental.cli.main import cli


@pytest.fixture
def tmp_rllm_home(monkeypatch, tmp_path):
    """Set up a temporary RLLM_HOME directory."""
    rllm_home = str(tmp_path / ".rllm")
    monkeypatch.setenv("RLLM_HOME", rllm_home)
    # Patch DatasetRegistry class variables (both new and legacy paths)
    from rllm.data.dataset import DatasetRegistry

    monkeypatch.setattr(DatasetRegistry, "_RLLM_HOME", rllm_home)
    monkeypatch.setattr(DatasetRegistry, "_REGISTRY_FILE", os.path.join(rllm_home, "datasets", "registry.json"))
    monkeypatch.setattr(DatasetRegistry, "_DATASET_DIR", os.path.join(rllm_home, "datasets"))
    # Point legacy paths to nonexistent dir to prevent migration from real data
    legacy_dir = str(tmp_path / "legacy_registry")
    monkeypatch.setattr(DatasetRegistry, "_LEGACY_REGISTRY_DIR", legacy_dir)
    monkeypatch.setattr(DatasetRegistry, "_LEGACY_REGISTRY_FILE", os.path.join(legacy_dir, "dataset_registry.json"))
    monkeypatch.setattr(DatasetRegistry, "_LEGACY_DATASET_DIR", os.path.join(legacy_dir, "datasets"))
    return rllm_home


@pytest.fixture
def runner():
    return CliRunner()


def test_dataset_list_empty(runner, tmp_rllm_home):
    """List with no datasets should show a helpful message."""
    result = runner.invoke(cli, ["dataset", "list"])
    assert result.exit_code == 0
    assert "No datasets pulled" in result.output


def test_dataset_list_all(runner, tmp_rllm_home):
    """List --all should show catalog datasets."""
    result = runner.invoke(cli, ["dataset", "list", "--all"])
    assert result.exit_code == 0
    assert "gsm8k" in result.output
    assert "available" in result.output


def test_dataset_info_catalog_only(runner, tmp_rllm_home):
    """Info for a catalog dataset that's not pulled should show catalog info."""
    result = runner.invoke(cli, ["dataset", "info", "gsm8k"])
    assert result.exit_code == 0
    assert "gsm8k" in result.output
    assert "math" in result.output.lower()
    assert "not pulled" in result.output


def test_dataset_info_not_found(runner, tmp_rllm_home):
    """Info for a non-existent dataset should error."""
    result = runner.invoke(cli, ["dataset", "info", "nonexistent_dataset_xyz"])
    assert result.exit_code == 1


def test_dataset_remove_not_found(runner, tmp_rllm_home):
    """Remove a non-existent dataset should show error."""
    result = runner.invoke(cli, ["dataset", "remove", "nonexistent"])
    assert result.exit_code == 0
    assert "not found" in result.output


def test_dataset_register_and_list(runner, tmp_rllm_home):
    """Register a dataset manually and list it."""
    from rllm.data import DatasetRegistry

    data = [{"question": "1+1", "ground_truth": "2"}]
    DatasetRegistry.register_dataset("test_ds", data, split="test")

    result = runner.invoke(cli, ["dataset", "list"])
    assert result.exit_code == 0
    assert "test_ds" in result.output


def test_dataset_register_inspect_remove(runner, tmp_rllm_home):
    """Full lifecycle: register, inspect, remove."""
    from rllm.data import DatasetRegistry

    data = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(5)]
    DatasetRegistry.register_dataset("lifecycle_ds", data, split="test")

    # Inspect
    result = runner.invoke(cli, ["dataset", "inspect", "lifecycle_ds", "--split", "test", "-n", "2"])
    assert result.exit_code == 0
    assert "q0" in result.output
    assert "q1" in result.output

    # Remove
    result = runner.invoke(cli, ["dataset", "remove", "lifecycle_ds"])
    assert result.exit_code == 0
    assert "Removed" in result.output

    # Confirm removal
    result = runner.invoke(cli, ["dataset", "list"])
    assert "lifecycle_ds" not in result.output


class TestDatasetRegister:
    """Tests for 'rllm dataset register' CLI command."""

    def test_register_json(self, runner, tmp_rllm_home, tmp_path):
        """Register a dataset from a JSON file."""
        data = [{"q": "1+1", "a": "2"}, {"q": "2+2", "a": "4"}]
        f = tmp_path / "data.json"
        f.write_text(json.dumps(data))

        result = runner.invoke(cli, ["dataset", "register", "my_ds", "--file", str(f)])
        assert result.exit_code == 0
        assert "Registered" in result.output
        assert "2 examples" in result.output

        # Should appear in list
        result = runner.invoke(cli, ["dataset", "list"])
        assert "my_ds" in result.output

    def test_register_jsonl(self, runner, tmp_rllm_home, tmp_path):
        """Register a dataset from a JSONL file."""
        f = tmp_path / "data.jsonl"
        lines = [json.dumps({"x": i}) for i in range(5)]
        f.write_text("\n".join(lines))

        result = runner.invoke(cli, ["dataset", "register", "jsonl_ds", "--file", str(f), "--split", "train"])
        assert result.exit_code == 0
        assert "5 examples" in result.output
        assert "train" in result.output

    def test_register_csv(self, runner, tmp_rllm_home, tmp_path):
        """Register a dataset from a CSV file."""
        f = tmp_path / "data.csv"
        with open(f, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["question", "answer"])
            writer.writeheader()
            writer.writerow({"question": "hi", "answer": "hello"})
            writer.writerow({"question": "bye", "answer": "goodbye"})
        result = runner.invoke(cli, ["dataset", "register", "csv_ds", "--file", str(f)])
        assert result.exit_code == 0
        assert "2 examples" in result.output

    def test_register_with_category_and_description(self, runner, tmp_rllm_home, tmp_path):
        """Register with optional metadata."""
        data = [{"q": "test"}]
        f = tmp_path / "data.json"
        f.write_text(json.dumps(data))

        result = runner.invoke(
            cli,
            [
                "dataset",
                "register",
                "meta_ds",
                "--file",
                str(f),
                "--category",
                "qa",
                "--description",
                "A test dataset",
            ],
        )
        assert result.exit_code == 0

        # Verify metadata via info command
        result = runner.invoke(cli, ["dataset", "info", "meta_ds"])
        assert result.exit_code == 0

    def test_register_missing_file(self, runner, tmp_rllm_home):
        """Registering a non-existent file should fail."""
        result = runner.invoke(cli, ["dataset", "register", "bad_ds", "--file", "/nonexistent/path.json"])
        assert result.exit_code != 0

    def test_register_inspect_roundtrip(self, runner, tmp_rllm_home, tmp_path):
        """Registered dataset can be inspected."""
        data = [{"query": "Best pizza?", "cuisine": "italian"}]
        f = tmp_path / "data.json"
        f.write_text(json.dumps(data))

        runner.invoke(cli, ["dataset", "register", "rt_ds", "--file", str(f), "--split", "test"])
        result = runner.invoke(cli, ["dataset", "inspect", "rt_ds", "--split", "test", "-n", "1"])
        assert result.exit_code == 0
        assert "Best pizza?" in result.output
