"""Agent management CLI commands.

``rllm agent [list|info|register|unregister]``
"""

from __future__ import annotations

import click

from rllm.experimental.cli._display import format_table
from rllm.experimental.cli._pull import load_agent_catalog, load_dataset_catalog


@click.group()
def agent():
    """Manage agent scaffolds."""


@agent.command(name="list")
def list_agents():
    """List registered agent scaffolds (built-in and agenthub)."""
    from rllm.experimental.eval.agent_loader import list_agents as _list_agents

    agents = _list_agents()

    if not agents:
        click.echo("No agents registered.")
        return

    headers = ["Name", "Source", "Module", "Description"]
    rows = []
    for a in agents:
        rows.append([a["name"], a["source"], a["module"], a["description"]])

    click.echo(format_table(headers, rows))


@agent.command()
@click.argument("name")
def info(name: str):
    """Show agent details and compatible datasets."""
    agent_catalog = load_agent_catalog()
    agents = agent_catalog.get("agents", {})

    if name not in agents:
        # Check if it's a plugin agent
        from rllm.experimental.eval.agent_loader import list_agents as _list_agents

        plugin_agents = {a["name"]: a for a in _list_agents()}
        if name in plugin_agents:
            a = plugin_agents[name]
            click.echo(f"\nAgent: {name}")
            click.echo(f"  Source:       {a['source']}")
            click.echo(f"  Module:       {a['module']}")
            click.echo()
            return

        available = ", ".join(sorted({*agents.keys(), *plugin_agents.keys()}))
        click.echo(f"Error: Agent '{name}' not found. Available: {available}")
        raise SystemExit(1)

    entry = agents[name]
    click.echo(f"\nAgent: {name}")
    click.echo(f"  Description:  {entry.get('description', 'N/A')}")
    click.echo(f"  Module:       {entry.get('module', 'N/A')}")
    click.echo(f"  Function:     {entry.get('function', 'N/A')}")

    # Find compatible datasets
    ds_catalog = load_dataset_catalog()
    compatible = []
    for ds_name, ds_info in ds_catalog.get("datasets", {}).items():
        if ds_info.get("default_agent") == name:
            compatible.append(ds_name)

    if compatible:
        click.echo(f"\n  Compatible datasets: {', '.join(compatible)}")
    click.echo()


@agent.command()
@click.argument("name")
@click.argument("import_path")
def register(name: str, import_path: str):
    """Register a custom agent so it's discoverable by name.

    \b
    NAME is the short name (e.g. "my-agent").
    IMPORT_PATH is "module:object" (e.g. "my_pkg.agent:my_agent").

    After registration, use it with:
        rllm eval gsm8k --agent my-agent
    """
    from rllm.experimental.eval.agent_loader import register_agent

    register_agent(name, import_path)
    click.echo(f"Registered agent '{name}' -> {import_path}")
    click.echo(f"Use it: rllm eval <benchmark> --agent {name}")


@agent.command()
@click.argument("name")
def unregister(name: str):
    """Remove a registered custom agent."""
    from rllm.experimental.eval.agent_loader import unregister_agent

    if unregister_agent(name):
        click.echo(f"Unregistered agent '{name}'.")
    else:
        click.echo(f"Agent '{name}' not found in user registry.")
