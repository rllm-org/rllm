"""Tests for the Datasets panel.

These run against the real registry shipped in ``rllm/registry/`` —
the panel is a read-only browser, so coupling to fixtures isn't worth
the maintenance cost. Assertions stick to invariants that hold across
registry edits (e.g. "gsm8k exists and is in the math category").
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rllm.console import mount_console


@pytest.fixture
def client(tmp_path) -> TestClient:
    app = FastAPI()
    mount_console(app, eval_results_root=tmp_path)
    return TestClient(app)


def test_list_returns_registry(client: TestClient) -> None:
    r = client.get("/console/api/panels/datasets")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["datasets"], list)
    assert len(body["datasets"]) > 10  # sanity floor — registry has ~60

    names = {d["name"] for d in body["datasets"]}
    # Spot-check well-known anchors.
    assert {"gsm8k", "math500", "humaneval"} <= names

    # Categories ordered, "math" present.
    assert "math" in body["categories"]


def test_dataset_detail_known_entry(client: TestClient) -> None:
    r = client.get("/console/api/panels/datasets/gsm8k")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "gsm8k"
    assert body["source"] == "openai/gsm8k"
    assert body["category"] == "math"
    assert {s["name"] for s in body["splits_detail"]} == set(body["splits"])


def test_dataset_detail_404(client: TestClient) -> None:
    assert client.get("/console/api/panels/datasets/no-such-dataset").status_code == 404


def test_dataset_name_validation(client: TestClient) -> None:
    # Slashes / colons / spaces rejected before hitting the registry.
    for bad in ("with space", "evil/path"):
        r = client.get(f"/console/api/panels/datasets/{bad}")
        # Multi-segment via URL-encoded slash 404s at routing; bare bad
        # chars 400 from validator.
        assert r.status_code in (400, 404)


def test_entries_requires_split(client: TestClient) -> None:
    r = client.get("/console/api/panels/datasets/gsm8k/entries")
    assert r.status_code == 422  # FastAPI: missing required query param


def test_entries_unknown_split_404(client: TestClient) -> None:
    r = client.get("/console/api/panels/datasets/gsm8k/entries?split=nonexistent")
    assert r.status_code == 404


def test_entries_for_local_split_paginates(client: TestClient) -> None:
    """Skip if the test machine doesn't have parquet cached — these
    files aren't versioned in the repo."""
    detail = client.get("/console/api/panels/datasets/gsm8k").json()
    local = next((s for s in detail["splits_detail"] if s["is_local"]), None)
    if not local:
        pytest.skip("no local parquet cached for gsm8k on this machine")

    r = client.get(
        f"/console/api/panels/datasets/gsm8k/entries?split={local['name']}&limit=3",
    )
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == local["n_rows"]
    assert body["limit"] == 3
    assert body["offset"] == 0
    assert len(body["rows"]) == 3
    assert set(body["columns"]) >= {"question"}

    # Offset past the end yields empty rows but a valid response.
    r2 = client.get(
        f"/console/api/panels/datasets/gsm8k/entries?split={local['name']}&offset={body['total'] + 100}&limit=3",
    )
    assert r2.status_code == 200
    assert r2.json()["rows"] == []


def test_entries_limit_clamped(client: TestClient) -> None:
    """limit > 200 rejected at validation time."""
    r = client.get("/console/api/panels/datasets/gsm8k/entries?split=train&limit=999")
    assert r.status_code == 422
