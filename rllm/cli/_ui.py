"""Shared UI toolkit for the rLLM CLI.

Single source of truth for the CLI's visual language: the themed ``console``,
the brand palette, the error/abort helpers, and the table/panel builders every
command shares. Command modules should import ``console`` (never construct their
own ``Console``) and render through ``fail``/``not_found``/``abort`` and the
table helpers rather than hand-rolling Rich objects or raising ``SystemExit``.
"""

from __future__ import annotations

import sys
from typing import NoReturn

import click
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.theme import Theme

theme = Theme(
    {
        # semantic
        "label": "dim",
        "success": "bold green",
        "error": "bold red",
        "val": "bold",
        "key": "yellow",
        "option": "cyan",
        "option.selected": "bold cyan",
        # brand palette (formerly inline hex in dataset.py / main.py)
        "brand": "#00D4FF",
        "accent": "#00CCFF",
        "header": "bold #00D4FF",
        "border": "dim #0077FF",
        "highlight": "#FFD700",
        "muted": "#88BBFF",
    }
)
console = Console(theme=theme)
_err_console = Console(theme=theme, stderr=True)


# --- errors / exits ---------------------------------------------------------


class CliError(click.ClickException):
    """A CLI error rendered through the themed console (stderr, exit code 1)."""

    def show(self, file=None) -> None:
        # Escape the message: it routinely interpolates user input (paths,
        # dataset names) that must not be parsed as Rich markup.
        _err_console.print(f"  [error]Error:[/] {escape(self.message)}")


def fail(message: str) -> NoReturn:
    """Abort the command with an error message (stderr, exit code 1)."""
    raise CliError(message)


def not_found(thing: str, name: str, hint: str | None = None) -> NoReturn:
    """Abort with a consistent ``<thing> '<name>' not found`` error."""
    fail(f"{thing} '{name}' not found." + (f" {hint}" if hint else ""))


def abort() -> NoReturn:
    """Cancel an interactive flow (prints ``Aborted!``, exit code 1)."""
    raise click.Abort()


def parse_index_spec(spec: str) -> list[int]:
    """Parse an index spec (``"5"``, ``"3,7,12"``, ``"0-9"``, ``"2,5-8,11"``) into a list of ints.

    Raises ``click.BadParameter`` on malformed input (non-integer or reversed range).
    """
    indices: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                lo, hi = (int(x) for x in part.split("-", 1))
                if lo > hi:
                    raise ValueError
                indices.extend(range(lo, hi + 1))
            else:
                indices.append(int(part))
        except ValueError:
            raise click.BadParameter(f"invalid index spec {part!r}; expected N or N-M (e.g. '0', '3,7', '0-9')") from None
    return indices


# --- tables / panels --------------------------------------------------------


def info_panel(
    rows: list[tuple[str, str]],
    *,
    title: str | None = None,
    border: str = "brand",
    label_width: int = 12,
) -> Panel:
    """A key/value detail panel for ``info`` / ``show`` / summary screens."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="label", width=label_width)
    table.add_column()
    for label, value in rows:
        table.add_row(label, value)
    return Panel(table, title=title, border_style=border, expand=False)


def catalog_table(title: str | None = None, *, width: int | None = None) -> Table:
    """An empty ROUNDED catalog table in the brand palette; caller adds the columns."""
    return Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="header",
        border_style="border",
        title=title,
        title_style="bold",
        padding=(0, 1),
        expand=False,
        width=width,
    )


def simple_table(headers: list[str], rows: list[list[str]], *, title: str | None = None) -> Table:
    """A flat themed table (replaces the old plain-ASCII ``format_table``)."""
    table = catalog_table(title)
    for header in headers:
        table.add_column(header)
    for row in rows:
        table.add_row(*row)
    return table


# --- interactive selection --------------------------------------------------


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


def _select_provider(existing) -> str:
    """Interactive provider selection using display labels."""
    from rllm.eval.config import PROVIDER_REGISTRY

    choices = [p.label for p in PROVIDER_REGISTRY]
    provider_ids = [p.id for p in PROVIDER_REGISTRY]

    cursor = 0
    if existing.provider in provider_ids:
        cursor = provider_ids.index(existing.provider)

    idx = _select_from_menu("Provider", choices, cursor)
    if idx is None:
        abort()
    return provider_ids[idx]


def _select_model(provider: str, existing, api_key: str | None = None) -> str:
    """Interactive model selection with option to enter a custom model.

    For Tinker, the model list is pulled live from the server's capabilities
    (``api_key`` lets the fetch authenticate); the curated list is the fallback.
    """
    from rllm.eval.config import PROVIDER_MODELS

    models = PROVIDER_MODELS.get(provider, [])

    if provider == "tinker":
        from rllm.eval.config import fetch_tinker_models

        with console.status("[dim]Fetching Tinker model catalog...[/]"):
            live = fetch_tinker_models(api_key)
        if live:
            models = live
            console.print(f"  [success]Loaded {len(live)} Tinker models[/] [dim](base IDs; you can also paste a tinker:// checkpoint path)[/]")
        else:
            console.print("  [dim]Could not fetch live catalog — showing curated list (or enter a tinker:// path manually).[/]")

    # For custom provider or providers with no curated list, prompt for free-text
    if provider == "custom" or not models:
        default = existing.model if existing.model else ""
        model = Prompt.ask("  Enter model name", default=default or None, console=console).strip()
        if not model:
            fail("Model is required.")
        return model

    choices = list(models) + ["Other (enter manually)"]

    cursor = 0
    if existing.model in models:
        cursor = models.index(existing.model)
    elif existing.model:
        cursor = len(models)

    idx = _select_from_menu("Model", choices, cursor)
    if idx is None:
        abort()

    if idx == len(models):
        default = existing.model if existing.model and existing.model not in models else ""
        model = Prompt.ask("  Enter model name", default=default or None, console=console).strip()
        if not model:
            fail("Model is required.")
    else:
        model = choices[idx]

    return model


def _prompt_base_url() -> str:
    """Prompt for a custom endpoint URL."""
    console.print("  [dim]Examples: http://localhost:8000/v1, https://my-api.example.com/v1[/]")
    url = Prompt.ask("  [label]Base URL[/]", console=console).strip()
    if not url:
        fail("Base URL is required for custom provider.")
    return url
