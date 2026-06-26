"""rLLM configuration: persistent provider/model settings for eval.

Stores configuration in ``~/.rllm/config.json`` (or ``$RLLM_HOME/config.json``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from rllm import paths

# Tinker's (beta) OpenAI-compatible inference endpoint. Used as a fixed
# ``api_base`` so the Tinker provider routes through LiteLLM's OpenAI adapter
# exactly like any other provider. See
# https://tinker-docs.thinkingmachines.ai/tinker/compatible-apis/openai/
TINKER_OAI_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"

# Fireworks' control-plane model catalog. ``GET {base}/v1/accounts/fireworks/models``
# lists the full public model library (~280 entries) with rich metadata
# (``supportsServerless``, ``kind``, ``contextLength``). We filter this to
# serverless chat LLMs. (The OpenAI-compatible ``/inference/v1/models`` endpoint
# only returns a small curated subset, so it's not used here.) See
# https://docs.fireworks.ai/api-reference/list-models
FIREWORKS_CONTROL_BASE_URL = "https://api.fireworks.ai/v1"


@dataclass
class ProviderInfo:
    """Metadata for a supported LLM provider."""

    id: str  # e.g. "openai", "deepseek", "custom"
    label: str  # Display name, e.g. "OpenAI", "Deepseek"
    litellm_prefix: str  # LiteLLM routing prefix, e.g. "together_ai"
    env_key: str  # Environment variable for API key, e.g. "OPENAI_API_KEY"
    default_model: str  # Default model name
    models: list[str] = field(default_factory=list)  # Curated model list
    base_url: str = ""  # Fixed OpenAI-compatible api_base (e.g. Tinker); blank = use provider default


PROVIDER_REGISTRY: list[ProviderInfo] = [
    # --- Original providers (preserve ordering for backward compat) ---
    ProviderInfo(
        id="openai",
        label="OpenAI",
        litellm_prefix="openai",
        env_key="OPENAI_API_KEY",
        default_model="gpt-5.5",
        models=[
            # GPT-5 family
            "gpt-5.5-pro",
            "gpt-5.5",
            "gpt-5.5-chat-latest",
            "gpt-5.4",
            "gpt-5.3",
            "gpt-5.2-pro",
            "gpt-5.2",
            "gpt-5.1",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            # GPT-4 family
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "gpt-4o",
            "gpt-4o-mini",
            # o-series reasoning
            "o4-mini",
            "o3-pro",
            "o3",
            "o3-mini",
        ],
    ),
    ProviderInfo(
        id="anthropic",
        label="Anthropic",
        litellm_prefix="anthropic",
        env_key="ANTHROPIC_API_KEY",
        default_model="claude-opus-4-8",
        models=[
            "claude-opus-4-8",
            "claude-opus-4-7",
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
        ],
    ),
    ProviderInfo(
        id="gemini",
        label="Google Gemini",
        litellm_prefix="gemini",
        env_key="GEMINI_API_KEY",
        default_model="gemini-3.5-flash",
        models=[
            # Gemini 3.5 family
            "gemini-3.5-pro",
            "gemini-3.5-flash",
            # Gemini 3 family
            "gemini-3.1-pro-preview",
            "gemini-3-pro-preview",
            "gemini-3-flash-preview",
            # Gemini 2.5 family
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
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
        default_model="anthropic/claude-opus-4-8",
        models=[
            "anthropic/claude-opus-4-8",
            "anthropic/claude-sonnet-4-6",
            "openai/gpt-5.5",
            "google/gemini-3.5-flash",
            "deepseek/deepseek-v4-pro",
            "x-ai/grok-4.3",
            "meta-llama/llama-4-maverick",
        ],
    ),
    ProviderInfo(
        id="deepseek",
        label="Deepseek",
        litellm_prefix="deepseek",
        env_key="DEEPSEEK_API_KEY",
        default_model="deepseek-v4-flash",
        models=[
            "deepseek-v4-pro",
            "deepseek-v4-flash",
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
        default_model="accounts/fireworks/models/deepseek-v4-flash",
        # Curated fallback shown when the live catalog can't be fetched
        # (offline / no key). ``rllm model setup`` pulls the full, live list of
        # serverless chat models from the control-plane catalog when possible.
        models=[
            "accounts/fireworks/models/deepseek-v4-flash",
            "accounts/fireworks/models/deepseek-v4-pro",
            "accounts/fireworks/models/glm-5p2",
            "accounts/fireworks/models/gpt-oss-120b",
            "accounts/fireworks/models/gpt-oss-20b",
            "accounts/fireworks/models/kimi-k2p6",
            "accounts/fireworks/models/minimax-m3",
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
        default_model="grok-4",
        models=[
            "grok-4.3",
            "grok-4.20",
            "grok-4.1",
            "grok-4-heavy",
            "grok-4",
            "grok-3",
            "grok-3-mini",
        ],
    ),
    # --- Chinese providers ---
    ProviderInfo(
        id="zhipu",
        label="Zhipu (GLM)",
        litellm_prefix="zai",
        env_key="ZAI_API_KEY",
        default_model="glm-5",
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
        default_model="kimi-k2.6",
        models=[
            "kimi-k2.6",
            "kimi-k2.5",
            "kimi-k2-thinking",
        ],
    ),
    ProviderInfo(
        id="minimax",
        label="MiniMax",
        litellm_prefix="minimax",
        env_key="MINIMAX_API_KEY",
        default_model="MiniMax-M3",
        models=[
            "MiniMax-M3",
            "MiniMax-M2.7",
            "MiniMax-M2.7-highspeed",
        ],
    ),
    # --- Tinker (Thinking Machines) — beta OpenAI-compatible endpoint ---
    ProviderInfo(
        id="tinker",
        label="Tinker (Thinking Machines)",
        # Route through LiteLLM's OpenAI-compatible adapter pointed at the
        # Tinker endpoint (``base_url`` below). The OAI endpoint accepts both
        # base model IDs (below) and ``tinker://`` sampler checkpoint paths.
        litellm_prefix="openai",
        env_key="TINKER_API_KEY",
        default_model="Qwen/Qwen3-8B",
        # Curated fallback shown when the live catalog can't be fetched
        # (offline / no key). ``rllm model setup`` pulls the full, live list
        # from ``ServiceClient.get_server_capabilities()`` when possible.
        models=[
            "Qwen/Qwen3-8B",
            "Qwen/Qwen3-32B",
            "Qwen/Qwen3-4B-Instruct-2507",
            "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct",
            "deepseek-ai/DeepSeek-V3.1",
            "openai/gpt-oss-20b",
        ],
        base_url=TINKER_OAI_BASE_URL,
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

# Fixed OpenAI-compatible api_base per provider (only set for e.g. Tinker)
PROVIDER_BASE_URLS: dict[str, str] = {p.id: p.base_url for p in PROVIDER_REGISTRY if p.base_url}

# Index for fast lookup
_PROVIDER_INDEX: dict[str, ProviderInfo] = {p.id: p for p in PROVIDER_REGISTRY}


def get_provider_info(provider_id: str) -> ProviderInfo | None:
    """Look up a provider by ID. Returns None if not found."""
    return _PROVIDER_INDEX.get(provider_id)


def fetch_tinker_models(api_key: str | None = None) -> list[str]:
    """Return Tinker's live base-model catalog (best-effort).

    Queries ``ServiceClient.get_server_capabilities()`` for the models the
    Tinker server currently exposes. Any base model here can be sampled via
    the OpenAI-compatible endpoint without training (a ``tinker://`` sampler
    checkpoint path also works, but isn't enumerable here).

    Returns an empty list on any failure (tinker not installed, no API key,
    network error) so callers can fall back to the curated static list.
    """
    key = api_key or os.environ.get("TINKER_API_KEY", "")
    if not key:
        return []
    # ServiceClient reads TINKER_API_KEY from the environment; set it for the
    # duration of the call without clobbering any pre-existing value.
    prev = os.environ.get("TINKER_API_KEY")
    os.environ["TINKER_API_KEY"] = key
    try:
        import tinker

        caps = tinker.ServiceClient().get_server_capabilities()
        names = [m.model_name for m in caps.supported_models if m.model_name]
        return sorted(names)
    except Exception:
        return []
    finally:
        if prev is None:
            os.environ.pop("TINKER_API_KEY", None)
        else:
            os.environ["TINKER_API_KEY"] = prev


def _fireworks_get(path: str, key: str, params: dict | None = None) -> dict:
    """GET a Fireworks control-plane path and return the parsed JSON dict."""
    import urllib.parse
    import urllib.request

    url = f"{FIREWORKS_CONTROL_BASE_URL}/{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.load(resp)


def _fireworks_paged(path: str, key: str, list_key: str) -> list[dict]:
    """Return every item under a paged control-plane list endpoint.

    ``path`` is e.g. ``accounts/fireworks/models``; ``list_key`` is the array
    field in the response (e.g. ``models``, ``deployedModels``).
    """
    items: list[dict] = []
    page_token = ""
    # Bound the loop so a misbehaving cursor can't spin forever (200/page).
    for _ in range(20):
        params = {"pageSize": "200"}
        if page_token:
            params["pageToken"] = page_token
        payload = _fireworks_get(path, key, params)
        items.extend(payload.get(list_key, []))
        page_token = payload.get("nextPageToken") or ""
        if not page_token:
            break
    return items


def _fireworks_account_ids(key: str) -> list[str]:
    """Return the caller's account IDs, excluding the shared public namespace."""
    accounts = _fireworks_get("accounts", key).get("accounts", [])
    ids = [a["name"].split("/", 1)[1] for a in accounts if a.get("name", "").startswith("accounts/")]
    return [i for i in ids if i != "fireworks"]


def fetch_fireworks_models(api_key: str | None = None) -> list[str]:
    """Return Fireworks' live serverless chat-model catalog (best-effort).

    Pages through the control-plane ``accounts/fireworks/models`` endpoint and
    keeps only models that are (a) serverless-available and (b) text base
    models (``kind == "HF_BASE_MODEL"``) — i.e. the ones you can sample without
    spinning up a dedicated deployment. IDs are returned in the
    ``accounts/fireworks/models/<slug>`` form the inference engine and config
    already use.

    Returns an empty list on any failure (no API key, network error, bad
    response) so callers can fall back to the curated static list.
    """
    key = api_key or os.environ.get("FIREWORKS_API_KEY", "")
    if not key:
        return []
    try:
        models = _fireworks_paged("accounts/fireworks/models", key, "models")
        return sorted(m["name"] for m in models if m.get("supportsServerless") and m.get("kind") == "HF_BASE_MODEL" and m.get("name"))
    except Exception:
        return []


def fetch_fireworks_deployed_models(api_key: str | None = None) -> list[str]:
    """Return the caller's runnable dedicated-deployment IDs (best-effort).

    Discovers the caller's account ID(s) via ``GET /v1/accounts``, then collects
    two complementary sources (the shared public ``fireworks`` namespace is
    excluded — see :func:`fetch_fireworks_models`):

    * ``deployedModels`` in state ``DEPLOYED`` — a custom/fine-tuned model bound
      to a deployment; addressed for inference by its ``model`` id.
    * ``deployments`` in state ``READY`` — a dedicated deployment of a base
      model has no ``deployedModel`` entry (``model`` is null) and is addressed
      for inference by its own ``accounts/<acct>/deployments/<id>`` name. Without
      this, base-model deployments are invisible to the picker even though
      that's the exact id evals run against.

    Returns an empty list on any failure so callers can degrade gracefully.
    """
    key = api_key or os.environ.get("FIREWORKS_API_KEY", "")
    if not key:
        return []
    try:
        names: list[str] = []
        for account in _fireworks_account_ids(key):
            for dm in _fireworks_paged(f"accounts/{account}/deployedModels", key, "deployedModels"):
                if dm.get("state") == "DEPLOYED" and dm.get("model"):
                    names.append(dm["model"])
            for dep in _fireworks_paged(f"accounts/{account}/deployments", key, "deployments"):
                if dep.get("state") == "READY" and dep.get("name"):
                    names.append(dep["name"])
        return sorted(set(names))
    except Exception:
        return []


def _config_path() -> str:
    return paths.config_path()


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


def load_tunnel_config() -> dict:
    """Return the gateway-tunnel config (``backend``/``domain``/``port``) from ``~/.rllm/config.json``.

    Written by ``rllm tunnel setup``; consumed by ``rllm tunnel up`` and the
    quick-tunnel fallback warning. Empty dict if unset or unreadable.
    """
    path = _config_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        tunnel = data.get("tunnel")
        return dict(tunnel) if isinstance(tunnel, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_tunnel_config(backend: str | None, *, domain: str | None = None, port: int | None = None) -> None:
    """Merge or remove the ``tunnel`` block in ``~/.rllm/config.json``.

    ``backend=None`` removes the block. Merges into the existing file so the
    provider/model and ``ui_api_key`` entries survive.
    """
    path = _config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data: dict = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}
    if backend is None:
        data.pop("tunnel", None)
    else:
        entry: dict = {"backend": backend}
        if domain:
            entry["domain"] = domain
        if port:
            entry["port"] = port
        data["tunnel"] = entry
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

    Merges into any existing file so unrelated keys (e.g. ``ui_api_key``
    written by ``rllm login``) survive provider/model changes instead of
    being clobbered.

    Returns the path that was written.
    """
    path = _config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data: dict[str, object] = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                data = loaded
        except (json.JSONDecodeError, OSError):
            data = {}
    data["provider"] = config.provider
    data["model"] = config.model
    data["api_keys"] = dict(config.api_keys)
    if config.base_url:
        data["base_url"] = config.base_url
    else:
        data.pop("base_url", None)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.chmod(path, 0o600)
    return path
