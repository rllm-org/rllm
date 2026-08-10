"""Tests for the first-class Terminal-Bench 3 catalog entry."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from rllm.cli._pull import load_dataset_catalog, pull_dataset
from rllm.integrations.harbor.dataset_loader import get_harbor_dataset_client

HARBOR_DATASET_ID = "terminal-bench/terminal-bench@3.0.0"


def test_terminal_bench_3_catalog_entry_uses_harbor_registry() -> None:
    entry = load_dataset_catalog()["datasets"]["terminal-bench-3"]

    assert entry["source"] == f"harbor:{HARBOR_DATASET_ID}"
    assert entry["splits"] == ["default"]
    assert entry["eval_split"] == "default"
    assert entry["default_agent"] == "claude-code"
    assert not entry["default_agent"].startswith("harbor:")
    assert "reward_fn" not in entry
    assert "builder" not in entry


def test_terminal_bench_3_uses_hub_package_client() -> None:
    pytest.importorskip("harbor")
    from harbor.registry.client.package import PackageDatasetClient

    assert isinstance(get_harbor_dataset_client(HARBOR_DATASET_ID), PackageDatasetClient)


def test_terminal_bench_3_pull_uses_exact_versioned_identifier() -> None:
    entry = load_dataset_catalog()["datasets"]["terminal-bench-3"]
    rows = [{"task_id": "example", "task_path": "/tmp/example"}]

    with (
        patch("rllm.integrations.harbor.dataset_loader.load_harbor_dataset", return_value=rows) as load_harbor,
        patch("rllm.data.DatasetRegistry.register_dataset") as register_dataset,
    ):
        pull_dataset("terminal-bench-3", entry)

    load_harbor.assert_called_once_with(HARBOR_DATASET_ID)
    register_dataset.assert_called_once_with(
        name="terminal-bench-3",
        data=rows,
        split="default",
        source=f"harbor:{HARBOR_DATASET_ID}",
        description=entry["description"],
        category="agentic",
    )
