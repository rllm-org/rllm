"""Shared UI helpers for the rLLM CLI."""

from __future__ import annotations

import sys

from rich.console import Console
from rich.prompt import Prompt
from rich.theme import Theme

from rllm.experimental.eval.config import (
    PROVIDER_MODELS,
    PROVIDER_REGISTRY,
    RllmConfig,
)

theme = Theme(
    {
        "option": "cyan",
        "option.selected": "bold cyan",
        "label": "dim",
        "success": "bold green",
        "error": "bold red",
        "key": "yellow",
    }
)
console = Console(theme=theme)


def _mask_key(key: str) -> str:
    """Mask an API key for display, showing only the last 4 characters."""
    if len(key) <= 4:
        return "****"
    return "****" + key[-4:]


def _has_tty() -> bool:
    """Check if we have a real terminal for interactive menus."""
    if not sys.stdin.isatty():
        return False
    try:
        with open("/dev/tty"):
            return True
    except OSError:
        return False


def _get_terminal_menu():
    """Lazy-import TerminalMenu. Returns the class or None if unavailable."""
    try:
        from simple_term_menu import TerminalMenu

        return TerminalMenu
    except ImportError:
        return None


def _select_from_menu(title: str, choices: list[str], cursor: int = 0) -> int | None:
    """Show an interactive menu if possible, otherwise fall back to numbered prompt."""
    TerminalMenu = _get_terminal_menu() if _has_tty() else None

    if TerminalMenu is not None:
        console.print(f"  [label]{title}[/]")
        menu = TerminalMenu(
            choices,
            cursor_index=cursor,
            menu_cursor_style=("fg_cyan", "bold"),
            menu_highlight_style=("fg_cyan", "bold"),
        )
        return menu.show()

    # Fallback: numbered list
    console.print(f"  [label]{title}[/]")
    for i, choice in enumerate(choices):
        if i == cursor:
            console.print(f"    [option.selected]> {i + 1}) {choice}[/]")
        else:
            console.print(f"      {i + 1}) [option]{choice}[/]")
    while True:
        raw = Prompt.ask(f"  Enter choice [dim](1-{len(choices)})[/]", default=str(cursor + 1), console=console)
        try:
            idx = int(raw.strip()) - 1
            if 0 <= idx < len(choices):
                return idx
        except ValueError:
            pass
        console.print(f"  [error]Please enter a number between 1 and {len(choices)}.[/]")


def _select_provider(existing: RllmConfig) -> str:
    """Interactive provider selection using display labels."""
    choices = [p.label for p in PROVIDER_REGISTRY]
    provider_ids = [p.id for p in PROVIDER_REGISTRY]

    cursor = 0
    if existing.provider in provider_ids:
        cursor = provider_ids.index(existing.provider)

    idx = _select_from_menu("Provider", choices, cursor)
    if idx is None:
        console.print("\n  [dim]Aborted.[/]")
        raise SystemExit(1)
    return provider_ids[idx]


def _select_model(provider: str, existing: RllmConfig) -> str:
    """Interactive model selection with option to enter a custom model."""
    models = PROVIDER_MODELS.get(provider, [])

    # For custom provider or providers with no curated list, prompt for free-text
    if provider == "custom" or not models:
        default = existing.model if existing.model else ""
        model = Prompt.ask("  Enter model name", default=default or None, console=console).strip()
        if not model:
            console.print("  [error]Model is required.[/]")
            raise SystemExit(1)
        return model

    choices = list(models) + ["Other (enter manually)"]

    cursor = 0
    if existing.model in models:
        cursor = models.index(existing.model)
    elif existing.model:
        cursor = len(models)

    idx = _select_from_menu("Model", choices, cursor)
    if idx is None:
        console.print("\n  [dim]Aborted.[/]")
        raise SystemExit(1)

    if idx == len(models):
        default = existing.model if existing.model and existing.model not in models else ""
        model = Prompt.ask("  Enter model name", default=default or None, console=console).strip()
        if not model:
            console.print("  [error]Model is required.[/]")
            raise SystemExit(1)
    else:
        model = choices[idx]

    return model


def _prompt_base_url() -> str:
    """Prompt for a custom endpoint URL."""
    console.print("  [dim]Examples: http://localhost:8000/v1, https://my-api.example.com/v1[/]")
    url = Prompt.ask("  [label]Base URL[/]", console=console).strip()
    if not url:
        console.print("  [error]Base URL is required for custom provider.[/]")
        raise SystemExit(1)
    return url
