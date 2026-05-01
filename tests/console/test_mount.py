"""Tests for ``rllm.console.mount_console`` — the FastAPI integration layer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rllm.console import mount_console


@pytest.fixture
def app(tmp_path: Path) -> FastAPI:
    eval_root = tmp_path / "eval_results"
    eval_root.mkdir()
    a = FastAPI()
    mount_console(a, eval_results_root=eval_root)
    return a


def test_shell_info_lists_default_panels(app: FastAPI) -> None:
    r = TestClient(app).get("/console/api/shell/info")
    assert r.status_code == 200
    body = r.json()
    assert body["url_prefix"] == "/console"
    assert body["eval_results_root"].endswith("eval_results")
    ids = [p["id"] for p in body["panels"]]
    # Default panels self-register at import. The exact order matters
    # less than the set being present — this test pins both.
    assert ids == ["sessions", "runs", "sandboxes", "eval_launcher", "training"]
    placeholder = {p["id"]: p["placeholder"] for p in body["panels"]}
    assert placeholder == {
        "sessions": False,
        "runs": False,
        "sandboxes": True,
        "eval_launcher": True,
        "training": True,
    }


def test_panel_routes_registered_under_prefix(app: FastAPI) -> None:
    paths = {getattr(r, "path", "") for r in app.routes}
    assert "/console/api/shell/info" in paths
    assert "/console/api/panels/sessions/runs" in paths
    assert "/console/api/panels/runs/{run_id}/index" in paths


def test_static_dir_missing_does_not_block_api(tmp_path: Path) -> None:
    """When ``rllm/console/static`` is absent, API still serves; only
    SPA routes 404. Important: dev installs without a built frontend
    must still be able to call the panel APIs."""
    a = FastAPI()
    mount_console(
        a,
        eval_results_root=tmp_path,
        static_dir=tmp_path / "does-not-exist",
    )
    client = TestClient(a)
    assert client.get("/console/api/shell/info").status_code == 200
    # No SPA mount, so /console/ has no handler.
    assert client.get("/console/").status_code == 404


def test_static_dir_serves_spa(tmp_path: Path) -> None:
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<html>spa</html>", encoding="utf-8")
    (static / "assets").mkdir()
    (static / "assets" / "main.js").write_text("console.log('hi')", encoding="utf-8")

    a = FastAPI()
    mount_console(a, eval_results_root=tmp_path, static_dir=static)
    client = TestClient(a)

    # Root → index.html
    r = client.get("/console/")
    assert r.status_code == 200
    assert "spa" in r.text

    # Asset served from /console/assets/
    r = client.get("/console/assets/main.js")
    assert r.status_code == 200
    assert "console.log" in r.text

    # Unknown sub-path → SPA fallback (so client routing works)
    r = client.get("/console/p/sessions")
    assert r.status_code == 200
    assert "spa" in r.text


def test_url_prefix_override(tmp_path: Path) -> None:
    a = FastAPI()
    mount_console(a, eval_results_root=tmp_path, url_prefix="/foo/bar")
    r = TestClient(a).get("/foo/bar/api/shell/info")
    assert r.status_code == 200
    assert r.json()["url_prefix"] == "/foo/bar"


def test_eval_results_root_optional() -> None:
    """When omitted, the Runs panel returns 503; everything else still works."""
    a = FastAPI()
    mount_console(a)  # no eval_results_root
    client = TestClient(a)

    info = client.get("/console/api/shell/info").json()
    assert info["eval_results_root"] is None

    # Sessions panel doesn't need the eval root.
    assert client.get("/console/api/panels/sessions/runs").status_code == 200

    # Runs panel does.
    r = client.get("/console/api/panels/runs")
    assert r.status_code == 503
    assert "eval_results_root" in json.dumps(r.json())
