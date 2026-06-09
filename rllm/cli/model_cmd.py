"""``rllm model`` — manage provider, API key, and model configuration.

Subcommands:
    rllm model setup  — first-time interactive config
    rllm model swap   — switch provider or model
    rllm model show   — print current configuration
"""

from __future__ import annotations

import os

import click
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from rllm.cli._ui import (
    _mask_key,
    _prompt_base_url,
    _select_model,
    _select_provider,
    console,
    fail,
    info_panel,
)
from rllm.eval.config import (
    PROVIDER_ENV_KEYS,
    RllmConfig,
    get_provider_info,
    load_config,
    save_config,
)


def _env_key_value(env_key: str) -> str:
    """Resolve an API key from the environment, falling back to a ``.env`` file.

    Checks ``os.environ`` first, then a ``.env`` in the current directory so the
    user can just press Enter at the prompt instead of re-typing the key.
    """
    if not env_key:
        return ""
    val = os.environ.get(env_key, "")
    if val:
        return val
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.isfile(env_path):
        try:
            with open(env_path) as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    if key.strip() == env_key:
                        return value.strip().strip('"').strip("'")
        except OSError:
            pass
    return ""


def _prompt_api_key(provider: str) -> str:
    """Prompt for an API key, defaulting to the provider's env var if present."""
    env_key = PROVIDER_ENV_KEYS.get(provider, "")
    env_val = _env_key_value(env_key)
    if env_val:
        console.print(f"  [dim]Found {env_key} in your environment — press Enter to use it.[/]")
    elif env_key:
        console.print(f"  [dim]Tip: you can also set {env_key} in your environment[/]")
    api_key = Prompt.ask("  [label]API key[/]", password=True, default=env_val or None, show_default=False, console=console)
    api_key = (api_key or "").strip()
    if not api_key:
        fail("API key is required.")
    return api_key


def _prompt_optional_api_key(provider: str) -> str:
    """Prompt for an API key that is optional (e.g. for local endpoints)."""
    env_key = PROVIDER_ENV_KEYS.get(provider, "")
    if env_key:
        console.print(f"  [dim]Tip: you can also set {env_key} in your environment[/]")
    console.print("  [dim]API key is optional for local endpoints (press Enter to skip)[/]")
    api_key = Prompt.ask("  [label]API key[/]", password=True, default="", console=console).strip()
    return api_key


def _provider_label(provider_id: str) -> str:
    """Return the display label for a provider, falling back to the raw ID."""
    info = get_provider_info(provider_id)
    return info.label if info else provider_id


def _print_config_table(config: RllmConfig, title: str = "[dim]current config[/]", border: str = "dim") -> None:
    """Print a config summary panel."""
    rows = [("Provider", _provider_label(config.provider))]
    if config.base_url:
        rows.append(("Base URL", f"[dim]{config.base_url}[/]"))
    if config.provider != "custom" or config.api_key:
        rows.append(("API key", f"[key]{_mask_key(config.api_key)}[/]"))
    rows.append(("Model", config.model))
    console.print(info_panel(rows, title=title, border=border, label_width=10))


def _print_saved_summary(config: RllmConfig, path: str) -> None:
    """Print the saved-config summary panel."""
    rows = [("Provider", f"[val]{_provider_label(config.provider)}[/]")]
    if config.base_url:
        rows.append(("Base URL", f"[dim]{config.base_url}[/]"))
    if config.provider != "custom" or config.api_key:
        rows.append(("API key", f"[key]{_mask_key(config.api_key)}[/]"))
    rows.append(("Model", f"[val]{config.model}[/]"))
    rows.append(("Saved to", f"[dim]{path}[/]"))
    console.print(info_panel(rows, title="[success]Configuration saved[/]", border="green", label_width=10))
    console.print()


def _do_swap(existing: RllmConfig) -> None:
    """Core swap logic: pick provider, ensure API key, pick model, save."""
    # Provider
    provider = _select_provider(existing)
    console.print()

    # Base URL — for custom provider
    base_url = ""
    if provider == "custom":
        if existing.base_url:
            console.print(f"  [label]Base URL[/]  [dim]{existing.base_url}[/]  [dim](on file)[/]")
            change = Confirm.ask("  Change URL?", default=False, console=console)
            if change:
                base_url = _prompt_base_url()
            else:
                base_url = existing.base_url
        else:
            base_url = _prompt_base_url()
        console.print()

    # API key — use stored key if available, otherwise prompt
    api_keys = dict(existing.api_keys)
    if provider == "custom":
        # API key is optional for custom endpoints
        if provider in api_keys and api_keys[provider]:
            console.print(f"  [label]API key[/]  [key]{_mask_key(api_keys[provider])}[/]  [dim](on file)[/]")
            change = Confirm.ask("  Change key?", default=False, console=console)
            if change:
                api_keys[provider] = _prompt_optional_api_key(provider)
        else:
            api_keys[provider] = _prompt_optional_api_key(provider)
    elif provider in api_keys:
        console.print(f"  [label]API key[/]  [key]{_mask_key(api_keys[provider])}[/]  [dim](on file)[/]")
        change = Confirm.ask("  Change key?", default=False, console=console)
        if change:
            api_keys[provider] = _prompt_api_key(provider)
    else:
        api_keys[provider] = _prompt_api_key(provider)
    console.print()

    # Model — pre-select current if same provider
    model_existing = RllmConfig(provider=provider, model=existing.model if existing.provider == provider else "")
    model = _select_model(provider, model_existing, api_key=api_keys.get(provider))
    console.print()

    config = RllmConfig(provider=provider, model=model, api_keys=api_keys, base_url=base_url)
    errors = config.validate()
    if errors:
        fail("\n".join(errors))

    path = save_config(config)
    _print_saved_summary(config, path)


@click.group("model")
def model():
    """Manage provider and model configuration."""


@model.command("setup")
def model_setup():
    """First-time configuration (provider, API key, model)."""
    existing = load_config()

    console.print()
    console.print(Panel("[bold]rLLM Setup[/]", subtitle="[dim]configure your provider and model[/]", border_style="cyan", expand=False))
    console.print()

    if existing.is_configured():
        _print_config_table(existing)
        console.print()
        swap = Confirm.ask("  Already configured. Would you like to swap?", default=True, console=console)
        if not swap:
            console.print("  [dim]No changes made.[/]")
            console.print()
            return
        console.print()
        _do_swap(existing)
        return

    # Fresh setup: provider -> (base_url?) -> key -> model
    provider = _select_provider(existing)
    console.print()

    base_url = ""
    if provider == "custom":
        base_url = _prompt_base_url()
        console.print()

    if provider == "custom":
        api_key = _prompt_optional_api_key(provider)
    else:
        api_key = _prompt_api_key(provider)
    console.print()

    model_name = _select_model(provider, existing, api_key=api_key)
    console.print()

    api_keys = {}
    if api_key:
        api_keys[provider] = api_key
    config = RllmConfig(provider=provider, model=model_name, api_keys=api_keys, base_url=base_url)
    errors = config.validate()
    if errors:
        fail("\n".join(errors))

    path = save_config(config)
    _print_saved_summary(config, path)


@model.command("swap")
def model_swap():
    """Switch provider or model (requires prior setup)."""
    existing = load_config()

    console.print()
    if not existing.is_configured():
        fail("Not configured. Run `rllm model setup` first.")

    console.print(Panel("[bold]rLLM Swap[/]", subtitle="[dim]switch provider or model[/]", border_style="cyan", expand=False))
    console.print()

    _do_swap(existing)


@model.command("show")
def model_show():
    """Print current provider and model configuration."""
    config = load_config()

    console.print()
    if not config.is_configured():
        console.print("  [dim]Not configured.[/] Run [bold]rllm model setup[/] to get started.")
        console.print()
        return

    _print_config_table(config, title="[bold]rLLM Config[/]", border="cyan")
    console.print()
