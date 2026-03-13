"""Dataset management CLI commands.

``rllm dataset [list|pull|info|inspect|remove]``
"""

from __future__ import annotations

import click
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from rllm.experimental.cli._pull import load_dataset_catalog, pull_dataset

_console = Console()

_CATEGORY_ICONS = {
    "math": "📐",
    "code": "💻",
    "qa": "❓",
    "mcq": "🔘",
    "agentic": "🤖",
    "instruction_following": "📋",
    "translation": "🌐",
    "vlm": "👁️ ",
    "search": "🔍",
}

_CATEGORY_LABELS = {
    "instruction_following": "instruct",
}

_STATUS_STYLES = {
    "pulled": ("bold green", "●"),
    "available": ("dim", "○"),
    "local": ("bold yellow", "◆"),
}


@click.group()
def dataset():
    """Manage datasets."""


@dataset.command(name="list")
@click.option("--local", "local_only", is_flag=True, help="Show only locally pulled datasets.")
def list_datasets(local_only: bool):
    """List datasets."""
    from rllm.data import DatasetRegistry

    catalog = load_dataset_catalog()
    catalog_datasets = catalog.get("datasets", {})
    local_names = set(DatasetRegistry.get_dataset_names())

    if not local_only:
        # Group datasets by category
        by_category: dict[str, list[tuple[str, dict, str]]] = {}
        for name, info in sorted(catalog_datasets.items()):
            status = "pulled" if name in local_names else "available"
            cat = info.get("category", "other")
            by_category.setdefault(cat, []).append((name, info, status))
        for name in sorted(local_names - set(catalog_datasets.keys())):
            ds_info = DatasetRegistry.get_dataset_info(name)
            cat = ds_info.get("metadata", {}).get("category", "other") if ds_info else "other"
            by_category.setdefault(cat, []).append((name, {"description": ""}, "local"))

        total = sum(len(v) for v in by_category.values())
        pulled = sum(1 for entries in by_category.values() for _, _, s in entries if s == "pulled")

        table = Table(
            box=box.ROUNDED,
            show_header=True,
            header_style="bold #00D4FF",
            padding=(0, 1),
            border_style="dim #0077FF",
            title=f"[bold]Dataset Catalog[/]  [dim]({total} datasets, {pulled} pulled)[/]",
            title_style="bold",
            expand=False,
            width=min(_console.width, 96),
        )
        table.add_column("Dataset", style="bold #00CCFF", min_width=18, no_wrap=True)
        table.add_column("Status", justify="center", width=13, no_wrap=True)
        table.add_column("Description", style="dim", overflow="ellipsis", no_wrap=True)

        first_category = True
        for cat in sorted(by_category.keys()):
            entries = by_category[cat]
            icon = _CATEGORY_ICONS.get(cat, "📁")
            label = _CATEGORY_LABELS.get(cat, cat).upper()
            if not first_category:
                table.add_row("", "", "")
            first_category = False
            table.add_row(
                f"[bold #FFD700]{icon} {label}[/]",
                "",
                f"[dim]{len(entries)} dataset{'s' if len(entries) != 1 else ''}[/]",
            )
            for name, info, status in entries:
                style, dot = _STATUS_STYLES.get(status, ("dim", "○"))
                status_text = f"[{style}]{dot} {status}[/]"
                desc = info.get("description", "")
                if len(desc) > 44:
                    desc = desc[:41] + "..."
                table.add_row(f"  {name}", status_text, desc)

        _console.print()
        _console.print(table)
        _console.print()
        _console.print(Text("  Legend: ", style="bold") + Text("● pulled  ", style="bold green") + Text("○ available  ", style="dim") + Text("◆ local", style="bold yellow"))
        _console.print(Text("  Run ", style="dim") + Text("rllm dataset pull <name>", style="bold #00D4FF") + Text(" to download a dataset.", style="dim"))
        _console.print()
    else:
        if not local_names:
            _console.print()
            _console.print(
                Panel(
                    "[dim]No datasets pulled yet.[/]\n\nRun [bold #00D4FF]rllm dataset list --all[/] to see available datasets.",
                    border_style="dim #0077FF",
                    title="[bold]Datasets[/]",
                    expand=False,
                    padding=(1, 3),
                )
            )
            _console.print()
            return

        table = Table(
            box=box.ROUNDED,
            show_header=True,
            header_style="bold #00D4FF",
            padding=(0, 2),
            border_style="dim #0077FF",
            title=f"[bold]Local Datasets[/]  [dim]({len(local_names)} pulled)[/]",
            title_style="bold",
            expand=False,
        )
        table.add_column("Dataset", style="bold #00CCFF", min_width=20)
        table.add_column("Category", justify="center", min_width=10)
        table.add_column("Splits", style="#88BBFF")

        for name in sorted(local_names):
            splits = DatasetRegistry.get_dataset_splits(name)
            ds_info = DatasetRegistry.get_dataset_info(name)
            cat = ds_info.get("metadata", {}).get("category", "") if ds_info else ""
            icon = _CATEGORY_ICONS.get(cat, "📁")
            label = _CATEGORY_LABELS.get(cat, cat)
            table.add_row(name, f"{icon} {label}" if cat else "", ", ".join(splits))

        _console.print()
        _console.print(table)
        _console.print()


@dataset.command()
@click.argument("name")
def pull(name: str):
    """Pull a dataset from HuggingFace."""
    catalog = load_dataset_catalog()
    catalog_datasets = catalog.get("datasets", {})

    if name not in catalog_datasets:
        available = ", ".join(sorted(catalog_datasets.keys()))
        click.echo(f"Error: Dataset '{name}' not found in catalog. Available: {available}")
        raise SystemExit(1)

    click.echo(f"Pulling {name} from {catalog_datasets[name]['source']}...")
    pull_dataset(name, catalog_datasets[name])
    click.echo(f"Done. Use 'rllm dataset info {name}' to view details.")


@dataset.command()
@click.argument("name")
def info(name: str):
    """Show dataset metadata and splits."""
    from rllm.data import DatasetRegistry

    # Check local registry first
    ds_info = DatasetRegistry.get_dataset_info(name)

    # Also check catalog
    catalog = load_dataset_catalog()
    catalog_entry = catalog.get("datasets", {}).get(name)

    if not ds_info and not catalog_entry:
        click.echo(f"Error: Dataset '{name}' not found.")
        raise SystemExit(1)

    click.echo(f"\nDataset: {name}")

    if catalog_entry:
        click.echo(f"  Description:    {catalog_entry.get('description', 'N/A')}")
        click.echo(f"  Source:         {catalog_entry.get('source', 'N/A')}")
        click.echo(f"  Category:       {catalog_entry.get('category', 'N/A')}")
        click.echo(f"  Default agent:  {catalog_entry.get('default_agent', 'N/A')}")
        click.echo(f"  Reward fn:      {catalog_entry.get('reward_fn', 'N/A')}")
        click.echo(f"  Eval split:     {catalog_entry.get('eval_split', 'N/A')}")

    if ds_info:
        click.echo("\n  Local splits:")
        for split, split_info in ds_info.get("splits", {}).items():
            num = split_info.get("num_examples", "?")
            fields = split_info.get("fields", [])
            click.echo(f"    {split}: {num} examples")
            if fields:
                click.echo(f"      fields: {', '.join(fields)}")
    else:
        click.echo("\n  Status: not pulled (use 'rllm dataset pull {name}')".format(name=name))

    click.echo()


@dataset.command()
@click.argument("name")
@click.option("--split", default=None, help="Split to inspect (default: first available or eval_split).")
@click.option("-n", "--num-rows", default=3, help="Number of example rows to show.")
def inspect(name: str, split: str | None, num_rows: int):
    """Show sample data rows from a dataset."""
    from rllm.data import DatasetRegistry

    catalog = load_dataset_catalog()
    catalog_entry = catalog.get("datasets", {}).get(name)

    if split is None:
        if catalog_entry:
            split = catalog_entry.get("eval_split", "test")
        else:
            splits = DatasetRegistry.get_dataset_splits(name)
            split = splits[0] if splits else "default"

    ds = DatasetRegistry.load_dataset(name, split)
    if ds is None:
        click.echo(f"Error: Cannot load '{name}' split '{split}'. Try 'rllm dataset pull {name}' first.")
        raise SystemExit(1)

    click.echo(f"\n{name}/{split} — {len(ds)} examples (showing first {min(num_rows, len(ds))})\n")

    for i in range(min(num_rows, len(ds))):
        row = ds[i]
        click.echo(f"--- Example {i} ---")
        for key, value in row.items():
            if isinstance(value, bytes):
                val_str = f"<{len(value)} bytes (image)>"
            elif isinstance(value, list) and value and isinstance(value[0], bytes):
                total = sum(len(b) for b in value if isinstance(b, bytes))
                val_str = f"<{len(value)} images, {total} bytes total>"
            else:
                val_str = str(value)
                if len(val_str) > 200:
                    val_str = val_str[:200] + "..."
            click.echo(f"  {key}: {val_str}")
        click.echo()


@dataset.command()
@click.argument("name")
@click.option("--file", "file_path", required=True, type=click.Path(exists=True), help="Path to data file (JSON, JSONL, CSV, or Parquet).")
@click.option("--split", default="default", help="Split name (e.g., train, test). Default: 'default'.")
@click.option("--category", default=None, help="Dataset category (e.g., math, qa, code).")
@click.option("--description", default=None, help="Short description of the dataset.")
def register(name: str, file_path: str, split: str, category: str | None, description: str | None):
    """Register a local data file as a dataset."""
    from rllm.data import Dataset, DatasetRegistry

    ds = Dataset.load_data(file_path)
    DatasetRegistry.register_dataset(
        name,
        ds.data,
        split=split,
        category=category,
        description=description,
    )
    click.echo(f"Registered '{name}' split '{split}' ({len(ds)} examples).")


@dataset.command()
@click.argument("name")
@click.option("--split", default=None, help="Remove only this split (default: remove all).")
def remove(name: str, split: str | None):
    """Remove a local dataset."""
    from rllm.data import DatasetRegistry

    if split:
        ok = DatasetRegistry.remove_dataset_split(name, split)
        if ok:
            click.echo(f"Removed {name}/{split}.")
        else:
            click.echo(f"Error: {name}/{split} not found.")
    else:
        ok = DatasetRegistry.remove_dataset(name)
        if ok:
            click.echo(f"Removed {name}.")
        else:
            click.echo(f"Error: Dataset '{name}' not found.")
