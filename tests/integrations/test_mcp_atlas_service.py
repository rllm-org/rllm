from __future__ import annotations

import stat
import tempfile

import pytest

from rllm.integrations.mcp_atlas.service import MCPAtlasServiceManager


class _HealthResponse:
    ok = True
    status_code = 200

    @staticmethod
    def json():
        return {"status": "ok"}


def test_external_harness_mode_does_not_bootstrap_local_stack(monkeypatch):
    monkeypatch.setattr("rllm.integrations.mcp_atlas.service.requests.get", lambda *args, **kwargs: _HealthResponse())
    monkeypatch.setattr(
        "rllm.integrations.mcp_atlas.service.ensure_source",
        lambda: (_ for _ in ()).throw(AssertionError("source bootstrap is local-mode only")),
    )
    manager = MCPAtlasServiceManager(external_harness_url="http://external.example/")

    manager.start()

    assert manager.harness_url == "http://external.example"
    assert manager.metadata()["external_harness"] is True


def test_env_file_is_filtered_and_written_private(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "env.template").write_text("GITHUB_TOKEN=\nNOTION_TOKEN=\n")
    env_file = tmp_path / "atlas.env"
    env_file.write_text("GITHUB_TOKEN=secret-value\nIGNORED_SECRET=do-not-pass\n")
    manager = MCPAtlasServiceManager(env_file=str(env_file))
    manager._tempdir = tempfile.TemporaryDirectory(dir=tmp_path)
    try:
        rendered = manager._render_container_env(source)
        assert rendered is not None
        assert rendered.read_text() == "GITHUB_TOKEN=secret-value\n"
        assert stat.S_IMODE(rendered.stat().st_mode) == 0o600
    finally:
        manager._tempdir.cleanup()


def test_invalid_preflight_is_rejected():
    with pytest.raises(ValueError, match="strict.*smoke"):
        MCPAtlasServiceManager(preflight="skip")


def test_missing_env_file_is_rejected(tmp_path):
    manager = MCPAtlasServiceManager(env_file=str(tmp_path / "missing.env"), external_harness_url="http://external")
    with pytest.raises(FileNotFoundError, match="env file"):
        manager._check_programs()
