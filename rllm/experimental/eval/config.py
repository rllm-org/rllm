"""rLLM configuration: persistent provider/model settings for eval.

Stores configuration in ``~/.rllm/config.json`` (or ``$RLLM_HOME/config.json``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


@dataclass
class ProviderInfo:
    """Metadata for a supported LLM provider."""

    id: str  # e.g. "openai", "deepseek", "custom"
    label: str  # Display name, e.g. "OpenAI", "Deepseek"
    litellm_prefix: str  # LiteLLM routing prefix, e.g. "together_ai"
    env_key: str  # Environment variable for API key, e.g. "OPENAI_API_KEY"
    default_model: str  # Default model name
    models: list[str] = field(default_factory=list)  # Curated model list


PROVIDER_REGISTRY: list[ProviderInfo] = [
    # --- Original providers (preserve ordering for backward compat) ---
    ProviderInfo(
        id="openai",
        label="OpenAI",
        litellm_prefix="openai",
        env_key="OPENAI_API_KEY",
        default_model="gpt-5-mini",
        models=[
            # GPT-5 family
            "gpt-5-nano",
            "gpt-5-mini",
            "gpt-5",
            "gpt-5.1",
            "gpt-5.2",
            # GPT-4 family
            "gpt-4.1-nano",
            "gpt-4.1-mini",
            "gpt-4.1",
            "gpt-4o-mini",
            "gpt-4o",
            # o-series reasoning
            "o3-mini",
            "o3",
            "o3-pro",
            "o4-mini",
        ],
    ),
    ProviderInfo(
        id="anthropic",
        label="Anthropic",
        litellm_prefix="anthropic",
        env_key="ANTHROPIC_API_KEY",
        default_model="claude-sonnet-4-6",
        models=[
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
            "claude-opus-4-6",
        ],
    ),
    ProviderInfo(
        id="gemini",
        label="Google Gemini",
        litellm_prefix="gemini",
        env_key="GEMINI_API_KEY",
        default_model="gemini-2.5-flash",
        models=[
            # Gemini 3 family
            "gemini-3-flash-preview",
            "gemini-3-pro-preview",
            "gemini-3.1-pro-preview",
            # Gemini 2.5 family
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-pro",
            # Gemini 2.0
            "gemini-2.0-flash",
        ],
    ),
    # --- Western providers ---
    ProviderInfo(
        id="openrouter",
        label="OpenRouter",
        litellm_prefix="openrouter",
        env_key="OPENROUTER_API_KEY",
        default_model="anthropic/claude-sonnet-4-6",
        models=[
            "anthropic/claude-sonnet-4-6",
            "openai/gpt-5-mini",
            "google/gemini-2.5-flash",
            "deepseek/deepseek-chat",
            "meta-llama/llama-4-maverick",
        ],
    ),
    ProviderInfo(
        id="deepseek",
        label="Deepseek",
        litellm_prefix="deepseek",
        env_key="DEEPSEEK_API_KEY",
        default_model="deepseek-chat",
        models=[
            "deepseek-chat",
            "deepseek-reasoner",
        ],
    ),
    ProviderInfo(
        id="together",
        label="Together",
        litellm_prefix="together_ai",
        env_key="TOGETHER_API_KEY",
        default_model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        models=[
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "Qwen/Qwen2.5-72B-Instruct-Turbo",
            "deepseek-ai/DeepSeek-R1",
        ],
    ),
    ProviderInfo(
        id="fireworks",
        label="Fireworks",
        litellm_prefix="fireworks_ai",
        env_key="FIREWORKS_API_KEY",
        default_model="accounts/fireworks/models/llama4-maverick-instruct-basic",
        models=[
            "accounts/fireworks/models/llama4-maverick-instruct-basic",
            "accounts/fireworks/models/deepseek-r1",
            "accounts/fireworks/models/qwen2p5-72b-instruct",
        ],
    ),
    ProviderInfo(
        id="groq",
        label="Groq",
        litellm_prefix="groq",
        env_key="GROQ_API_KEY",
        default_model="llama-3.3-70b-versatile",
        models=[
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "deepseek-r1-distill-llama-70b",
            "gemma2-9b-it",
        ],
    ),
    ProviderInfo(
        id="cerebras",
        label="Cerebras",
        litellm_prefix="cerebras",
        env_key="CEREBRAS_API_KEY",
        default_model="llama-3.3-70b",
        models=[
            "llama-3.3-70b",
            "llama-3.1-8b",
            "deepseek-r1-distill-llama-70b",
        ],
    ),
    ProviderInfo(
        id="xai",
        label="xAI",
        litellm_prefix="xai",
        env_key="XAI_API_KEY",
        default_model="grok-3-mini",
        models=[
            "grok-3-mini",
            "grok-3",
        ],
    ),
    # --- Chinese providers ---
    ProviderInfo(
        id="zhipu",
        label="Zhipu (GLM)",
        litellm_prefix="zai",
        env_key="ZAI_API_KEY",
        default_model="glm-4.5",
        models=[
            "glm-5",
            "glm-4.7",
            "glm-4.5",
            "glm-4.5-flash",
        ],
    ),
    ProviderInfo(
        id="kimi",
        label="Kimi (Moonshot)",
        litellm_prefix="moonshot",
        env_key="MOONSHOT_API_KEY",
        default_model="kimi-k2.5",
        models=[
            "kimi-k2.5",
            "kimi-k2-thinking",
        ],
    ),
    ProviderInfo(
        id="minimax",
        label="MiniMax",
        litellm_prefix="minimax",
        env_key="MINIMAX_API_KEY",
        default_model="MiniMax-M2.7",
        models=[
            "MiniMax-M2.7",
            "MiniMax-M2.7-highspeed",
            "MiniMax-M2.5",
            "MiniMax-M2.5-highspeed",
        ],
    ),
    # --- Custom endpoint (last) ---
    ProviderInfo(
        id="custom",
        label="Custom (OpenAI-compatible)",
        litellm_prefix="",
        env_key="",
        default_model="",
        models=[],
    ),
]

# Derive backward-compatible constants from registry
SUPPORTED_PROVIDERS = [p.id for p in PROVIDER_REGISTRY]
DEFAULT_MODELS: dict[str, str] = {p.id: p.default_model for p in PROVIDER_REGISTRY if p.default_model}
PROVIDER_MODELS: dict[str, list[str]] = {p.id: p.models for p in PROVIDER_REGISTRY if p.models}
PROVIDER_ENV_KEYS: dict[str, str] = {p.id: p.env_key for p in PROVIDER_REGISTRY if p.env_key}

# Index for fast lookup
_PROVIDER_INDEX: dict[str, ProviderInfo] = {p.id: p for p in PROVIDER_REGISTRY}


def get_provider_info(provider_id: str) -> ProviderInfo | None:
    """Look up a provider by ID. Returns None if not found."""
    return _PROVIDER_INDEX.get(provider_id)


def _rllm_home() -> str:
    return os.path.expanduser(os.environ.get("RLLM_HOME", "~/.rllm"))


def _config_path() -> str:
    return os.path.join(_rllm_home(), "config.json")


@dataclass
class RllmConfig:
    """User-level rLLM configuration."""

    provider: str = ""
    model: str = ""
    api_keys: dict[str, str] = field(default_factory=dict)
    base_url: str = ""

    @property
    def api_key(self) -> str:
        """Return the API key for the active provider."""
        return self.api_keys.get(self.provider, "")

    def is_configured(self) -> bool:
        """Return True if all required fields are set."""
        if self.provider == "custom":
            return bool(self.base_url and self.model)
        return bool(self.provider and self.api_key and self.model)

    def validate(self) -> list[str]:
        """Return a list of validation error strings (empty if valid)."""
        errors: list[str] = []
        if not self.provider:
            errors.append("provider is required")
        elif self.provider not in SUPPORTED_PROVIDERS:
            errors.append(f"unsupported provider '{self.provider}' (supported: {', '.join(SUPPORTED_PROVIDERS)})")

        if self.provider == "custom":
            if not self.base_url:
                errors.append("base_url is required for custom provider")
            if not self.model:
                errors.append("model is required")
        else:
            if not self.api_key:
                errors.append("api_key is required")
            if not self.model:
                errors.append("model is required")
        return errors


def load_ui_config() -> dict:
    """Return UI-specific config (``ui_api_key``) from ``~/.rllm/config.json``."""
    path = _config_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        result = {}
        if data.get("ui_api_key"):
            result["ui_api_key"] = data["ui_api_key"]
        return result
    except (json.JSONDecodeError, OSError):
        return {}


def save_ui_config(ui_api_key: str | None) -> None:
    """Merge or remove ``ui_api_key`` in ``~/.rllm/config.json``."""
    path = _config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data: dict = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}
    if ui_api_key is None:
        data.pop("ui_api_key", None)
    else:
        data["ui_api_key"] = ui_api_key
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.chmod(path, 0o600)


def load_config() -> RllmConfig:
    """Load configuration from ``~/.rllm/config.json``.

    Handles both old format (``{"api_key": "..."}``)) and new format
    (``{"api_keys": {...}}``), migrating transparently.

    Returns an empty ``RllmConfig`` if the file is missing or corrupt.
    """
    path = _config_path()
    if not os.path.exists(path):
        return RllmConfig()
    try:
        with open(path) as f:
            data = json.load(f)
        provider = data.get("provider", "")
        model = data.get("model", "")
        base_url = data.get("base_url", "")

        # New format: api_keys dict
        api_keys: dict[str, str] = dict(data.get("api_keys", {}))

        # Backward compat: old format had a single "api_key" field
        if not api_keys and data.get("api_key") and provider:
            api_keys[provider] = data["api_key"]

        return RllmConfig(provider=provider, model=model, api_keys=api_keys, base_url=base_url)
    except (json.JSONDecodeError, OSError, TypeError):
        return RllmConfig()


def save_config(config: RllmConfig) -> str:
    """Persist configuration to ``~/.rllm/config.json``.

    Creates parent directories as needed and sets file permissions to 0o600
    (owner read/write only) since the file contains API keys.

    Returns the path that was written.
    """
    path = _config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data: dict[str, object] = {
        "provider": config.provider,
        "model": config.model,
        "api_keys": dict(config.api_keys),
    }
    if config.base_url:
        data["base_url"] = config.base_url
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.chmod(path, 0o600)
    return path
