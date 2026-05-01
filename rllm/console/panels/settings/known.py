"""Registry of well-known env vars surfaced in the Settings panel.

The list is curated for the rLLM use case: model providers users hit
through the gateway, sandbox providers harbor / rllm.sandbox can launch,
and rLLM-internal vars. Unknown vars set via the UI still work — they're
just shown under "Other" rather than in the canonical list.

Each entry:

* ``key``        — env var name
* ``label``      — short display label
* ``description``— one-line context for the UI tooltip
* ``category``   — bucket header
* ``secret``     — whether to mask the value in API responses
* ``url``        — optional link to where the user can grab a key
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KnownEnvVar:
    key: str
    label: str
    description: str
    category: str
    secret: bool = True
    url: str | None = None


_PROVIDERS = "Model providers"
_SANDBOX = "Sandbox providers"
_RLLM = "rLLM"

KNOWN_ENV_VARS: tuple[KnownEnvVar, ...] = (
    # ---- Model providers (the gateway routes traffic to these) ---------
    KnownEnvVar("OPENAI_API_KEY", "OpenAI", "API key for OpenAI / OpenAI-compatible endpoints.", _PROVIDERS, url="https://platform.openai.com/api-keys"),
    KnownEnvVar("ANTHROPIC_API_KEY", "Anthropic", "API key for Claude models via the native Anthropic API.", _PROVIDERS, url="https://console.anthropic.com/settings/keys"),
    KnownEnvVar("TOGETHER_API_KEY", "Together", "API key for Together AI hosted models.", _PROVIDERS, url="https://api.together.xyz/settings/api-keys"),
    KnownEnvVar("GROQ_API_KEY", "Groq", "API key for Groq's LPU-hosted models.", _PROVIDERS, url="https://console.groq.com/keys"),
    KnownEnvVar("MISTRAL_API_KEY", "Mistral", "API key for Mistral's hosted models.", _PROVIDERS, url="https://console.mistral.ai/api-keys"),
    KnownEnvVar("HF_TOKEN", "Hugging Face", "Token for Hugging Face Hub (datasets, gated models, inference).", _PROVIDERS, url="https://huggingface.co/settings/tokens"),
    KnownEnvVar("GEMINI_API_KEY", "Google Gemini", "API key for Google's Gemini models.", _PROVIDERS, url="https://aistudio.google.com/apikey"),
    KnownEnvVar("FIREWORKS_API_KEY", "Fireworks", "API key for Fireworks AI hosted models.", _PROVIDERS, url="https://fireworks.ai/account/api-keys"),
    # ---- Sandbox providers (harbor / rllm.sandbox launch into these) ---
    KnownEnvVar("DAYTONA_API_KEY", "Daytona", "API key for Daytona cloud sandbox runtime.", _SANDBOX, url="https://app.daytona.io/dashboard/keys"),
    KnownEnvVar("MODAL_TOKEN_ID", "Modal token id", "Token ID for Modal — pair with MODAL_TOKEN_SECRET.", _SANDBOX, secret=False, url="https://modal.com/settings/tokens"),
    KnownEnvVar("MODAL_TOKEN_SECRET", "Modal token secret", "Token secret for Modal.", _SANDBOX, url="https://modal.com/settings/tokens"),
    KnownEnvVar("E2B_API_KEY", "E2B", "API key for E2B sandbox runtime.", _SANDBOX, url="https://e2b.dev/dashboard?tab=keys"),
    KnownEnvVar("RUNLOOP_API_KEY", "Runloop", "API key for Runloop sandbox runtime.", _SANDBOX, url="https://platform.runloop.ai"),
    # ---- rLLM-internal --------------------------------------------------
    KnownEnvVar("RLLM_HOME", "rLLM home", "Override for ~/.rllm — where eval_results and the gateway db live.", _RLLM, secret=False),
    KnownEnvVar("RLLM_GATEWAY_DB", "Gateway DB path", "Override for the gateway's sqlite trace store.", _RLLM, secret=False),
    KnownEnvVar("RLLM_CONSOLE_BASE_PATH", "Console base path", "URL prefix for the console (default /console/).", _RLLM, secret=False),
)


def known_by_key() -> dict[str, KnownEnvVar]:
    return {k.key: k for k in KNOWN_ENV_VARS}


def categories() -> tuple[str, ...]:
    """Stable category order for the UI."""
    return (_PROVIDERS, _SANDBOX, _RLLM)
