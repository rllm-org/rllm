"""Tests for rllm eval config (load/save/validate)."""

import json
import os
import stat

import pytest

from rllm.experimental.eval.config import (
    DEFAULT_MODELS,
    PROVIDER_REGISTRY,
    SUPPORTED_PROVIDERS,
    RllmConfig,
    get_provider_info,
    load_config,
    save_config,
)


@pytest.fixture
def tmp_rllm_home(monkeypatch, tmp_path):
    """Redirect RLLM_HOME to a temp directory."""
    rllm_home = str(tmp_path / ".rllm")
    monkeypatch.setenv("RLLM_HOME", rllm_home)
    return rllm_home


class TestRllmConfig:
    def test_empty_config_not_configured(self):
        config = RllmConfig()
        assert not config.is_configured()

    def test_partial_config_not_configured(self):
        config = RllmConfig(provider="openai", api_keys={"openai": "sk-test"})
        assert not config.is_configured()

    def test_full_config_is_configured(self):
        config = RllmConfig(provider="openai", api_keys={"openai": "sk-test"}, model="gpt-4o")
        assert config.is_configured()

    def test_validate_empty(self):
        config = RllmConfig()
        errors = config.validate()
        assert len(errors) == 3
        assert any("provider" in e for e in errors)
        assert any("api_key" in e for e in errors)
        assert any("model" in e for e in errors)

    def test_validate_unsupported_provider(self):
        config = RllmConfig(provider="unsupported", api_keys={"unsupported": "key"}, model="m")
        errors = config.validate()
        assert len(errors) == 1
        assert "unsupported" in errors[0]

    def test_validate_valid(self):
        config = RllmConfig(provider="openai", api_keys={"openai": "sk-test"}, model="gpt-4o")
        assert config.validate() == []

    def test_custom_provider_requires_base_url(self):
        config = RllmConfig(provider="custom", model="my-model")
        assert not config.is_configured()
        errors = config.validate()
        assert any("base_url" in e for e in errors)

    def test_custom_provider_configured_with_base_url(self):
        config = RllmConfig(provider="custom", model="my-model", base_url="http://localhost:8000/v1")
        assert config.is_configured()
        assert config.validate() == []

    def test_custom_provider_api_key_optional(self):
        config = RllmConfig(provider="custom", model="my-model", base_url="http://localhost:8000/v1")
        assert config.is_configured()
        errors = config.validate()
        assert not any("api_key" in e for e in errors)

    def test_custom_provider_requires_model(self):
        config = RllmConfig(provider="custom", base_url="http://localhost:8000/v1")
        assert not config.is_configured()
        errors = config.validate()
        assert any("model" in e for e in errors)


class TestLoadSaveConfig:
    def test_load_missing_file(self, tmp_rllm_home):
        config = load_config()
        assert not config.is_configured()

    def test_save_and_load(self, tmp_rllm_home):
        original = RllmConfig(provider="openai", api_keys={"openai": "sk-secret"}, model="gpt-4o-mini")
        path = save_config(original)
        assert os.path.exists(path)

        loaded = load_config()
        assert loaded.provider == "openai"
        assert loaded.api_key == "sk-secret"
        assert loaded.model == "gpt-4o-mini"
        assert loaded.is_configured()

    def test_save_creates_directories(self, tmp_rllm_home):
        config = RllmConfig(provider="openai", api_keys={"openai": "sk-test"}, model="gpt-4o")
        path = save_config(config)
        assert os.path.isdir(os.path.dirname(path))

    def test_save_sets_permissions(self, tmp_rllm_home):
        config = RllmConfig(provider="openai", api_keys={"openai": "sk-test"}, model="gpt-4o")
        path = save_config(config)
        mode = stat.S_IMODE(os.stat(path).st_mode)
        assert mode == 0o600

    def test_load_corrupt_file(self, tmp_rllm_home):
        path = os.path.join(tmp_rllm_home, "config.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("not valid json{{{")
        config = load_config()
        assert not config.is_configured()

    def test_save_overwrites(self, tmp_rllm_home):
        save_config(RllmConfig(provider="openai", api_keys={"openai": "key1"}, model="m1"))
        save_config(RllmConfig(provider="openai", api_keys={"openai": "key2"}, model="m2"))
        loaded = load_config()
        assert loaded.api_key == "key2"
        assert loaded.model == "m2"

    def test_base_url_persisted(self, tmp_rllm_home):
        original = RllmConfig(provider="custom", model="my-model", base_url="http://localhost:8000/v1")
        save_config(original)
        loaded = load_config()
        assert loaded.base_url == "http://localhost:8000/v1"
        assert loaded.provider == "custom"
        assert loaded.model == "my-model"

    def test_base_url_absent_loads_empty(self, tmp_rllm_home):
        """Old config files without base_url should load with empty base_url."""
        config_path = os.path.join(tmp_rllm_home, "config.json")
        os.makedirs(tmp_rllm_home, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump({"provider": "openai", "api_keys": {"openai": "sk-test"}, "model": "gpt-4o"}, f)
        loaded = load_config()
        assert loaded.base_url == ""
        assert loaded.is_configured()

    def test_base_url_not_written_when_empty(self, tmp_rllm_home):
        """When base_url is empty, it should not be in the JSON file."""
        config = RllmConfig(provider="openai", api_keys={"openai": "sk-test"}, model="gpt-4o")
        path = save_config(config)
        with open(path) as f:
            data = json.load(f)
        assert "base_url" not in data


class TestConstants:
    def test_openai_in_supported_providers(self):
        assert "openai" in SUPPORTED_PROVIDERS

    def test_default_models_has_openai(self):
        assert "openai" in DEFAULT_MODELS

    def test_new_providers_in_supported(self):
        for pid in ["openrouter", "deepseek", "together", "fireworks", "groq", "cerebras", "xai", "zhipu", "kimi", "minimax", "custom"]:
            assert pid in SUPPORTED_PROVIDERS, f"{pid} not in SUPPORTED_PROVIDERS"

    def test_original_providers_first(self):
        """Original providers must remain at indices 0, 1, 2 for backward compat."""
        assert SUPPORTED_PROVIDERS[0] == "openai"
        assert SUPPORTED_PROVIDERS[1] == "anthropic"
        assert SUPPORTED_PROVIDERS[2] == "gemini"

    def test_custom_is_last(self):
        assert SUPPORTED_PROVIDERS[-1] == "custom"

    def test_get_provider_info(self):
        info = get_provider_info("openai")
        assert info is not None
        assert info.label == "OpenAI"
        assert info.litellm_prefix == "openai"
        assert info.env_key == "OPENAI_API_KEY"

    def test_get_provider_info_together(self):
        info = get_provider_info("together")
        assert info is not None
        assert info.litellm_prefix == "together_ai"

    def test_get_provider_info_unknown(self):
        assert get_provider_info("nonexistent") is None

    def test_registry_has_labels(self):
        for p in PROVIDER_REGISTRY:
            assert p.label, f"Provider {p.id} has no label"

    def test_minimax_provider_info(self):
        """MiniMax provider should have correct metadata."""
        info = get_provider_info("minimax")
        assert info is not None
        assert info.label == "MiniMax"
        assert info.litellm_prefix == "minimax"
        assert info.env_key == "MINIMAX_API_KEY"

    def test_minimax_default_model_is_m27(self):
        """MiniMax default model should be the latest M2.7."""
        info = get_provider_info("minimax")
        assert info is not None
        assert info.default_model == "MiniMax-M2.7"

    def test_minimax_models_include_m27_family(self):
        """MiniMax model list should include M2.7 and M2.7-highspeed."""
        info = get_provider_info("minimax")
        assert info is not None
        assert "MiniMax-M2.7" in info.models
        assert "MiniMax-M2.7-highspeed" in info.models

    def test_minimax_models_include_m25_family(self):
        """MiniMax model list should still include M2.5 models for backward compat."""
        info = get_provider_info("minimax")
        assert info is not None
        assert "MiniMax-M2.5" in info.models
        assert "MiniMax-M2.5-highspeed" in info.models

    def test_minimax_m27_is_first_model(self):
        """M2.7 should be listed before M2.5 (newest first)."""
        info = get_provider_info("minimax")
        assert info is not None
        m27_idx = info.models.index("MiniMax-M2.7")
        m25_idx = info.models.index("MiniMax-M2.5")
        assert m27_idx < m25_idx

    def test_minimax_config_validates(self):
        """A complete MiniMax config should pass validation."""
        config = RllmConfig(
            provider="minimax",
            api_keys={"minimax": "test-key"},
            model="MiniMax-M2.7",
        )
        assert config.is_configured()
        assert config.validate() == []

    def test_minimax_in_default_models(self):
        """MiniMax should have an entry in DEFAULT_MODELS."""
        assert "minimax" in DEFAULT_MODELS
        assert DEFAULT_MODELS["minimax"] == "MiniMax-M2.7"
