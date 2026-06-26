"""rLLM CLI: evaluate any model on any benchmark with one command.

Entry point: ``rllm [dataset|eval|agent|model]``
"""

from __future__ import annotations

import click
from rich.align import Align
from rich.console import Group
from rich.panel import Panel
from rich.text import Text

from rllm.cli._ui import catalog_table, console

_BANNER = "        [bold #00CCFF]██╗     [/][bold #00C4FF]██╗     [/][bold #00BBFF]███╗   ███╗[/]\n        [bold #00CCFF]██║     [/][bold #00C4FF]██║     [/][bold #00BBFF]████╗ ████║[/]\n[bold #00AAFF]██████╗ [/][bold #009FFF]██║     [/][bold #0094FF]██║     [/][bold #0088FF]██╔████╔██║[/]\n[bold #00AAFF]██╔═══╝ [/][bold #009FFF]██║     [/][bold #0094FF]██║     [/][bold #0088FF]██║╚██╔╝██║[/]\n[bold #0077FF]██║     [/][bold #006AFF]███████╗[/][bold #005DFF]███████╗[/][bold #0050FF]██║ ╚═╝ ██║[/]\n[bold #0077FF]╚═╝     [/][bold #006AFF]╚══════╝[/][bold #005DFF]╚══════╝[/][bold #0050FF]╚═╝     ╚═╝[/]"


class _LazyGroup(click.Group):
    """A Click group that lazily imports subcommands on first use.

    Avoids importing heavy modules (torch, litellm, transformers) at CLI
    startup by deferring subcommand module imports until a command is
    actually invoked.
    """

    # (module_path, attr_name, short_help, icon)
    _COMMANDS: dict[str, tuple[str, str, str, str] | None] = {
        "agent": ("rllm.cli.agent", "agent", "Manage agent scaffolds.", "🤖"),
        "dataset": ("rllm.cli.dataset", "dataset", "Manage datasets.", "📦"),
        "eval": ("rllm.cli.eval", "eval_cmd", "Evaluate a model on a benchmark dataset.", "📊"),
        "init": ("rllm.cli.init", "init_cmd", "Scaffold a new agent project.", "🚀"),
        "model": ("rllm.cli.model_cmd", "model", "Manage provider and model configuration.", "⚙️"),
        "snapshot": ("rllm.cli.snapshot_cmd", "snapshot", "Manage sandbox environment snapshots.", "📸"),
        "train": ("rllm.cli.train", "train_cmd", "Train a model on a benchmark dataset using RL.", "🏋️"),
        "tunnel": ("rllm.cli.tunnel", "tunnel", "Set up & run the gateway tunnel.", "🛰️"),
        "view": ("rllm.cli.view", "view_cmd", "Browse saved eval episodes in a local web viewer.", "🔍"),
        "login": ("rllm.cli.login", "login_cmd", "Log in to rLLM UI.", "🔑"),
        "setup": None,  # handled inline
    }

    def list_commands(self, ctx):
        # Exclude hidden commands (setup)
        return [name for name in sorted(self._COMMANDS) if name != "setup"]

    def get_command(self, ctx, cmd_name):
        if cmd_name == "setup":
            return setup_alias
        spec = self._COMMANDS.get(cmd_name)
        if spec is None:
            return None
        module_path, attr, _help, _icon = spec
        import importlib

        mod = importlib.import_module(module_path)
        return getattr(mod, attr)

    def format_help(self, ctx, formatter):
        """Render a fancy Rich help screen instead of plain Click output."""
        from importlib.metadata import version as pkg_version

        try:
            ver = pkg_version("rllm")
        except Exception:
            ver = "dev"

        console.print()

        # Banner with logo inside a styled panel
        logo = Text.from_markup(_BANNER)
        tagline = Text(
            "Reinforcement Learning for Language Agents",
            style="muted",
            justify="center",
        )

        panel = Panel(
            Group(Align.center(logo), Text(), tagline),
            title=f"[header]rLLM[/] [dim]v{ver}[/]",
            border_style="border",
            padding=(1, 2),
            expand=False,
        )
        console.print(panel)

        # Usage
        console.print(Text("  Usage: ", style="bold") + Text("rllm ", style="header") + Text("<command> ", style="accent") + Text("[options]", style="dim"))
        console.print()

        # Commands table
        table = catalog_table(title="[bold]Commands[/]")
        table.add_column("Command", style="brand", min_width=14)
        table.add_column("Description", style="muted")

        for name in self.list_commands(ctx):
            spec = self._COMMANDS.get(name)
            if spec is None:
                continue
            _mod, _attr, short_help, icon = spec
            table.add_row(f" {icon} {name}", short_help)

        console.print(table)
        console.print()

        # Footer hints
        console.print(Text("  Options:", style="bold"))
        console.print(Text("    --version  ", style="accent") + Text("Show the version and exit.", style="dim"))
        console.print(Text("    --help     ", style="accent") + Text("Show this message and exit.", style="dim"))
        console.print()
        console.print(Text("  Run ", style="dim") + Text("rllm <command> --help", style="header") + Text(" for more information on a command.", style="dim"))
        console.print()


@click.group(cls=_LazyGroup, invoke_without_command=True)
@click.version_option(package_name="rllm")
@click.pass_context
def cli(ctx):
    """rLLM: Reinforcement Learning for Language Agents."""
    if ctx.invoked_subcommand is None:
        cli.format_help(ctx, None)  # renders via Rich console


@cli.command("setup", hidden=True)
@click.pass_context
def setup_alias(ctx):
    """[deprecated] Use ``rllm model setup`` instead."""
    click.echo("Hint: use `rllm model setup` (the `setup` command is deprecated).\n", err=True)
    from rllm.cli.model_cmd import model_setup

    ctx.invoke(model_setup)


if __name__ == "__main__":
    cli()
