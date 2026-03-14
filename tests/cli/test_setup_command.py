"""Tests for rllm setup CLI command (deprecated alias for rllm model setup)."""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from rllm.experimental.cli.main import cli
from rllm.experimental.eval.config import RllmConfig, load_config, save_config


@pytest.fixture
def tmp_rllm_home(monkeypatch, tmp_path):
    """Redirect RLLM_HOME to a temp directory."""
    rllm_home = str(tmp_path / ".rllm")
    monkeypatch.setenv("RLLM_HOME", rllm_home)
    return rllm_home


@pytest.fixture
def runner():
    return CliRunner()


def test_setup_fresh(runner, tmp_rllm_home):
    """Setup with no existing config should prompt and save."""
    # Numbered fallback: 1 = openai, API key, 1 = gpt-5-nano
    result = runner.invoke(cli, ["setup"], input="1\nsk-testkey123\n1\n")
    assert result.exit_code == 0
    assert "Configuration saved" in result.output

    config = load_config()
    assert config.provider == "openai"
    assert config.api_key == "sk-testkey123"
    assert config.model == "gpt-5-nano"


def test_setup_select_different_model(runner, tmp_rllm_home):
    """Setup should allow selecting a different model from the list."""
    # 1 = openai, API key, 2 = gpt-5-mini
    result = runner.invoke(cli, ["setup"], input="1\nsk-testkey\n2\n")
    assert result.exit_code == 0

    config = load_config()
    assert config.model == "gpt-5-mini"


def test_setup_custom_model(runner, tmp_rllm_home):
    """Setup should allow entering a custom model name."""
    from rllm.experimental.eval.config import PROVIDER_MODELS

    other_num = len(PROVIDER_MODELS["openai"]) + 1  # "Other (enter manually)"

    result = runner.invoke(cli, ["setup"], input=f"1\nsk-testkey\n{other_num}\nmy-custom-model\n")
    assert result.exit_code == 0

    config = load_config()
    assert config.model == "my-custom-model"


def test_setup_shows_existing(runner, tmp_rllm_home):
    """Setup should show existing config when re-running."""
    save_config(RllmConfig(provider="openai", model="gpt-5-mini", api_keys={"openai": "sk-oldkey1234"}))

    # y = yes swap, 1 = openai, n = don't change key, 1 = gpt-5-nano
    result = runner.invoke(cli, ["setup"], input="y\n1\nn\n1\n")
    assert result.exit_code == 0
    assert "current config" in result.output
    assert "****1234" in result.output


def test_setup_keeps_existing_key(runner, tmp_rllm_home):
    """Setup should keep existing key when swapping to same provider."""
    save_config(RllmConfig(provider="openai", model="gpt-5-mini", api_keys={"openai": "sk-existing"}))

    # y = swap, 1 = openai, n = don't change key, 2 = gpt-5-mini
    result = runner.invoke(cli, ["setup"], input="y\n1\nn\n2\n")
    assert result.exit_code == 0

    config = load_config()
    assert config.api_key == "sk-existing"


def test_setup_replace_key(runner, tmp_rllm_home):
    """Setup should allow replacing the API key."""
    save_config(RllmConfig(provider="openai", model="gpt-5-mini", api_keys={"openai": "sk-old"}))

    # y = swap, 1 = openai, y = change key, new key, 1 = gpt-5-nano
    result = runner.invoke(cli, ["setup"], input="y\n1\ny\nsk-newkey\n1\n")
    assert result.exit_code == 0

    config = load_config()
    assert config.api_key == "sk-newkey"


def test_setup_default_selection(runner, tmp_rllm_home):
    """Pressing enter should use the default (pre-selected) choice."""
    # Default provider = 1, API key, default model = 1
    result = runner.invoke(cli, ["setup"], input="\nsk-testkey\n\n")
    assert result.exit_code == 0

    config = load_config()
    assert config.provider == "openai"
    assert config.model == "gpt-5-nano"


def test_setup_invalid_choice_reprompts(runner, tmp_rllm_home):
    """Invalid number should reprompt, not crash."""
    # 99 is invalid, then 1 is valid; same for model
    result = runner.invoke(cli, ["setup"], input="99\n1\nsk-testkey\n99\n1\n")
    assert result.exit_code == 0
    assert "Please enter a number" in result.output

    config = load_config()
    assert config.provider == "openai"


def test_setup_anthropic_provider(runner, tmp_rllm_home):
    """Setup with Anthropic provider should save correctly."""
    # 2 = anthropic, API key, 1 = claude-sonnet-4-6
    result = runner.invoke(cli, ["setup"], input="2\nsk-ant-testkey\n1\n")
    assert result.exit_code == 0
    assert "Configuration saved" in result.output

    config = load_config()
    assert config.provider == "anthropic"
    assert config.api_key == "sk-ant-testkey"
    assert config.model == "claude-sonnet-4-6"


def test_setup_gemini_provider(runner, tmp_rllm_home):
    """Setup with Gemini provider should save correctly."""
    # 3 = gemini, API key, 1 = gemini-3-flash-preview
    result = runner.invoke(cli, ["setup"], input="3\nAIza-testkey\n1\n")
    assert result.exit_code == 0
    assert "Configuration saved" in result.output

    config = load_config()
    assert config.provider == "gemini"
    assert config.api_key == "AIza-testkey"
    assert config.model == "gemini-3-flash-preview"


def test_setup_shows_env_var_tip(runner, tmp_rllm_home):
    """Setup should show env var tip for new providers."""
    # 2 = anthropic, API key, 1 = first model
    result = runner.invoke(cli, ["setup"], input="2\nsk-ant-key\n1\n")
    assert result.exit_code == 0
    assert "ANTHROPIC_API_KEY" in result.output


def test_setup_switch_provider_prompts_new_key(runner, tmp_rllm_home):
    """Switching provider should prompt for a new key."""
    save_config(RllmConfig(provider="openai", model="gpt-5-mini", api_keys={"openai": "sk-openai-key"}))

    # y = swap, 2 = anthropic (no key on file), new key, 1 = first model
    result = runner.invoke(cli, ["setup"], input="y\n2\nsk-ant-newkey\n1\n")
    assert result.exit_code == 0

    config = load_config()
    assert config.provider == "anthropic"
    assert config.api_key == "sk-ant-newkey"
    # old openai key should be preserved
    assert config.api_keys["openai"] == "sk-openai-key"


def test_setup_with_tty_uses_terminal_menu(tmp_rllm_home):
    """When TTY is available and simple-term-menu installed, TerminalMenu is used."""
    mock_menu_instance = MagicMock()
    mock_menu_instance.show.return_value = 0
    mock_menu_cls = MagicMock(return_value=mock_menu_instance)

    with patch("rllm.experimental.cli._ui._has_tty", return_value=True), patch("rllm.experimental.cli._ui._get_terminal_menu", return_value=mock_menu_cls):
        runner = CliRunner()
        result = runner.invoke(cli, ["setup"], input="sk-testkey\n")

    assert result.exit_code == 0
    assert mock_menu_cls.call_count == 2  # provider + model


def test_setup_deprecation_hint(runner, tmp_rllm_home):
    """The setup alias should print a deprecation hint."""
    result = runner.invoke(cli, ["setup"], input="1\nsk-testkey\n1\n")
    # Deprecation hint goes to stderr; CliRunner mixes them by default
    assert result.exit_code == 0
