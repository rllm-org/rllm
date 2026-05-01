"""Tests for the Settings panel — config + env-var management."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rllm.console import mount_console
from rllm.console.panels.settings import store


@pytest.fixture
def env_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate console.env to a tmp dir + clear any stray env vars
    that would leak into the test (re-running locally, etc.)."""
    monkeypatch.setenv("RLLM_HOME", str(tmp_path))
    p = tmp_path / "console.env"
    for k in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DAYTONA_API_KEY",
        "MY_CUSTOM_VAR",
        "MY_CUSTOM_TOKEN",
    ):
        monkeypatch.delenv(k, raising=False)
    return p


@pytest.fixture
def client(env_path: Path, tmp_path: Path) -> TestClient:
    app = FastAPI()
    mount_console(app, eval_results_root=tmp_path)
    return TestClient(app)


def test_config_endpoint(client: TestClient) -> None:
    r = client.get("/console/api/panels/settings/config")
    assert r.status_code == 200
    body = r.json()
    assert "version" in body
    assert body["url_prefix"] == "/console"
    assert body["console_env_file"].endswith("console.env")


def test_env_lists_known_categories(client: TestClient) -> None:
    r = client.get("/console/api/panels/settings/env")
    assert r.status_code == 200
    body = r.json()
    assert set(body["categories"]) >= {"Model providers", "Sandbox providers", "rLLM", "Other"}
    keys = {row["key"] for row in body["rows"]}
    # Spot-check the canonical providers are present.
    assert {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DAYTONA_API_KEY"} <= keys


def test_set_persists_and_applies(client: TestClient, env_path: Path) -> None:
    r = client.post(
        "/console/api/panels/settings/env",
        json={"key": "OPENAI_API_KEY", "value": "sk-test-abcdefghij"},
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"key": "OPENAI_API_KEY", "ok": True}

    # Persisted to file
    assert env_path.is_file()
    assert "OPENAI_API_KEY=" in env_path.read_text()

    # Applied to os.environ
    assert os.environ["OPENAI_API_KEY"] == "sk-test-abcdefghij"

    # Reflected in /env (masked, since OpenAI is secret-shaped).
    rows = client.get("/console/api/panels/settings/env").json()["rows"]
    row = next(r for r in rows if r["key"] == "OPENAI_API_KEY")
    assert row["is_set"] is True
    assert row["in_console_file"] is True
    assert row["masked_value"] == "sk-t…ghij"


def test_reveal_returns_full_value(client: TestClient) -> None:
    client.post(
        "/console/api/panels/settings/env",
        json={"key": "OPENAI_API_KEY", "value": "sk-test-abcdefghij"},
    )
    r = client.get("/console/api/panels/settings/env/OPENAI_API_KEY?reveal=true")
    assert r.status_code == 200
    assert r.json() == {
        "key": "OPENAI_API_KEY",
        "value": "sk-test-abcdefghij",
        "secret": True,
        "revealed": True,
        "in_console_file": True,
    }
    # Without reveal, masked.
    r2 = client.get("/console/api/panels/settings/env/OPENAI_API_KEY")
    assert r2.json()["value"] == "sk-t…ghij"
    assert r2.json()["revealed"] is False


def test_delete_removes_from_file_and_env(client: TestClient, env_path: Path) -> None:
    client.post(
        "/console/api/panels/settings/env",
        json={"key": "OPENAI_API_KEY", "value": "sk-zzz"},
    )
    assert os.environ.get("OPENAI_API_KEY") == "sk-zzz"

    r = client.delete("/console/api/panels/settings/env/OPENAI_API_KEY")
    assert r.status_code == 200
    body = r.json()
    assert body["removed_from_file"] is True
    assert body["removed_from_env"] is True
    assert "OPENAI_API_KEY" not in os.environ
    assert "OPENAI_API_KEY" not in env_path.read_text()


def test_unknown_var_appears_in_other_category(client: TestClient) -> None:
    client.post(
        "/console/api/panels/settings/env",
        json={"key": "MY_CUSTOM_TOKEN", "value": "supersecretvalue"},
    )
    rows = client.get("/console/api/panels/settings/env").json()["rows"]
    row = next(r for r in rows if r["key"] == "MY_CUSTOM_TOKEN")
    assert row["category"] == "Other"
    # *_TOKEN matches the secret-shape heuristic.
    assert row["secret"] is True
    assert row["masked_value"] == "supe…alue"


def test_unknown_non_secret_var_not_masked(client: TestClient) -> None:
    client.post(
        "/console/api/panels/settings/env",
        json={"key": "MY_CUSTOM_VAR", "value": "plainvalue"},
    )
    rows = client.get("/console/api/panels/settings/env").json()["rows"]
    row = next(r for r in rows if r["key"] == "MY_CUSTOM_VAR")
    assert row["secret"] is False
    assert row["masked_value"] == "plainvalue"


def test_invalid_key_rejected(client: TestClient) -> None:
    for bad in ("with space", "1starts-numeric", "kebab-case", ""):
        r = client.post(
            "/console/api/panels/settings/env",
            json={"key": bad, "value": "x"},
        )
        # 400 from validator or 422 from pydantic, depending on the
        # specific failure mode (empty key is min_length=1, others hit
        # the regex). Both are "rejected".
        assert r.status_code in (400, 422), f"{bad!r} should be rejected"


def test_load_into_environ_overrides_shell(env_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point of override=True: UI-set values beat shell."""
    monkeypatch.setenv("OPENAI_API_KEY", "from-shell")
    store.write_assignment("OPENAI_API_KEY", "from-console-ui", path=env_path)
    loaded = store.load_into_environ(env_path)
    assert loaded == {"OPENAI_API_KEY": "from-console-ui"}
    assert os.environ["OPENAI_API_KEY"] == "from-console-ui"


def test_quote_handling_roundtrip(env_path: Path) -> None:
    """Spaces, hashes, quotes round-trip through the parser correctly."""
    tricky = 'value with "quotes" and # and = signs'
    store.write_assignment("TRICKY", tricky, path=env_path)
    assert store.read_file(env_path) == {"TRICKY": tricky}


def test_existing_lines_preserved_on_update(env_path: Path) -> None:
    """Editing one key doesn't reshuffle the rest."""
    store.write_assignment("KEY_A", "1", path=env_path)
    store.write_assignment("KEY_B", "2", path=env_path)
    store.write_assignment("KEY_C", "3", path=env_path)
    store.write_assignment("KEY_B", "updated", path=env_path)

    # Read back: order preserved, value updated.
    text = env_path.read_text().strip().splitlines()
    assert text[0].startswith("KEY_A=")
    assert text[1] == "KEY_B=updated"
    assert text[2].startswith("KEY_C=")
