import importlib.util
from pathlib import Path


def _load_ray_init_utils():
    module_path = Path(__file__).resolve().parents[2] / "rllm" / "trainer" / "ray_init_utils.py"
    spec = importlib.util.spec_from_file_location("rllm_ray_init_utils_test", module_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_get_ray_init_settings_defaults_to_local_cluster(monkeypatch, tmp_path):
    monkeypatch.delenv("RAY_ADDRESS", raising=False)

    ray_init_utils = _load_ray_init_utils()
    monkeypatch.setattr(ray_init_utils, "_ray_current_cluster_path", lambda: tmp_path / "missing")

    settings = ray_init_utils.get_ray_init_settings(config=None)
    assert "address" not in settings


def test_get_ray_init_settings_attaches_when_ray_address_env_set(monkeypatch, tmp_path):
    monkeypatch.setenv("RAY_ADDRESS", "ray://dummy")

    ray_init_utils = _load_ray_init_utils()
    monkeypatch.setattr(ray_init_utils, "_ray_current_cluster_path", lambda: tmp_path / "missing")

    settings = ray_init_utils.get_ray_init_settings(config=None)
    assert settings["address"] == "auto"


def test_get_ray_init_settings_attaches_when_ray_current_cluster_file_exists(monkeypatch, tmp_path):
    monkeypatch.delenv("RAY_ADDRESS", raising=False)

    marker = tmp_path / "ray_current_cluster"
    marker.write_text("dummy")

    ray_init_utils = _load_ray_init_utils()
    monkeypatch.setattr(ray_init_utils, "_ray_current_cluster_path", lambda: marker)

    settings = ray_init_utils.get_ray_init_settings(config=None)
    assert settings["address"] == "auto"


def test_config_address_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("RAY_ADDRESS", "ray://dummy")

    class Cfg:
        ray_init = {"address": "ray://explicit"}

    ray_init_utils = _load_ray_init_utils()
    monkeypatch.setattr(ray_init_utils, "_ray_current_cluster_path", lambda: tmp_path / "missing")

    settings = ray_init_utils.get_ray_init_settings(Cfg())
    assert settings["address"] == "ray://explicit"
