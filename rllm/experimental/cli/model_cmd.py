"""``rllm model`` — manage provider, API key, and model configuration.

Subcommands:
    rllm model setup  — first-time interactive config
    rllm model swap   — switch provider or model
    rllm model show   — print current configuration
"""

from __future__ import annotations

import click
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from rllm.experimental.cli._ui import (
    _mask_key,
    _prompt_base_url,
    _select_model,
    _select_provider,
    console,
)
from rllm.experimental.eval.config import (
    PROVIDER_ENV_KEYS,
    RllmConfig,
    get_provider_info,
    load_config,
    save_config,
)


def _prompt_api_key(provider: str) -> str:
    """Prompt for an API key, showing env var tip."""
    env_key = PROVIDER_ENV_KEYS.get(provider, "")
    if env_key:
        console.print(f"  [dim]Tip: you can also set {env_key} in your environment[/]")
    api_key = Prompt.ask("  [label]API key[/]", password=True, console=console).strip()
    if not api_key:
        console.print("  [error]API key is required.[/]")
        raise SystemExit(1)
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
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="label", width=10)
    table.add_column()
    table.add_row("Provider", _provider_label(config.provider))
    if config.base_url:
        table.add_row("Base URL", f"[dim]{config.base_url}[/]")
    if config.provider != "custom" or config.api_key:
        table.add_row("API key", f"[key]{_mask_key(config.api_key)}[/]")
    table.add_row("Model", config.model)
    console.print(Panel(table, title=title, border_style=border, expand=False))


def _print_saved_summary(config: RllmConfig, path: str) -> None:
    """Print the saved-config summary panel."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="label", width=10)
    table.add_column()
    table.add_row("Provider", f"[bold]{_provider_label(config.provider)}[/]")
    if config.base_url:
        table.add_row("Base URL", f"[dim]{config.base_url}[/]")
    if config.provider != "custom" or config.api_key:
        table.add_row("API key", f"[key]{_mask_key(config.api_key)}[/]")
    table.add_row("Model", f"[bold]{config.model}[/]")
    table.add_row("Saved to", f"[dim]{path}[/]")
    console.print(Panel(table, title="[success]Configuration saved[/]", border_style="green", expand=False))
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
    model = _select_model(provider, model_existing)
    console.print()

    config = RllmConfig(provider=provider, model=model, api_keys=api_keys, base_url=base_url)
    errors = config.validate()
    if errors:
        for err in errors:
            console.print(f"  [error]Error: {err}[/]")
        raise SystemExit(1)

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

    model_name = _select_model(provider, existing)
    console.print()

    api_keys = {}
    if api_key:
        api_keys[provider] = api_key
    config = RllmConfig(provider=provider, model=model_name, api_keys=api_keys, base_url=base_url)
    errors = config.validate()
    if errors:
        for err in errors:
            console.print(f"  [error]Error: {err}[/]")
        raise SystemExit(1)

    path = save_config(config)
    _print_saved_summary(config, path)


@model.command("swap")
def model_swap():
    """Switch provider or model (requires prior setup)."""
    existing = load_config()

    console.print()
    if not existing.is_configured():
        console.print("  [error]Not configured.[/] Run [bold]rllm model setup[/] first.")
        console.print()
        raise SystemExit(1)

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
