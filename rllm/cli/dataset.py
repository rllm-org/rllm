"""Dataset management CLI commands.

``rllm dataset [list|pull|info|inspect|register|remove]``
"""

from __future__ import annotations

import os

import click
from rich.panel import Panel
from rich.text import Text

from rllm.cli._pull import load_dataset_catalog, pull_dataset
from rllm.cli._ui import catalog_table, console, fail, info_panel, not_found

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

        table = catalog_table(
            title=f"[bold]Dataset Catalog[/]  [dim]({total} datasets, {pulled} pulled)[/]",
            width=min(console.width, 96),
        )
        table.add_column("Dataset", style="brand", min_width=18, no_wrap=True)
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
                f"[highlight]{icon} {label}[/]",
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

        console.print()
        console.print(table)
        console.print()
        console.print(Text("  Legend: ", style="bold") + Text("● pulled  ", style="bold green") + Text("○ available  ", style="dim") + Text("◆ local", style="bold yellow"))
        console.print(Text("  Run ", style="dim") + Text("rllm dataset pull <name>", style="header") + Text(" to download a dataset.", style="dim"))
        console.print()
        console.print(Text("  Harbor: ", style="bold highlight") + Text("80+ agent benchmarks available via ", style="dim") + Text("harbor:", style="header") + Text(" prefix", style="dim"))
        console.print(Text("  Browse: ", style="dim") + Text("https://harbor.ai/datasets", style="bold dim") + Text("  |  ", style="dim") + Text("rllm eval harbor:<name>", style="header"))
        console.print()
    else:
        if not local_names:
            console.print()
            console.print(
                Panel(
                    "[dim]No datasets pulled yet.[/]\n\nRun [header]rllm dataset list[/] to see available datasets.",
                    border_style="border",
                    title="[bold]Datasets[/]",
                    expand=False,
                    padding=(1, 3),
                )
            )
            console.print()
            return

        table = catalog_table(title=f"[bold]Local Datasets[/]  [dim]({len(local_names)} pulled)[/]")
        table.add_column("Dataset", style="brand", min_width=20)
        table.add_column("Category", justify="center", min_width=10)
        table.add_column("Splits", style="muted")

        for name in sorted(local_names):
            splits = DatasetRegistry.get_dataset_splits(name)
            ds_info = DatasetRegistry.get_dataset_info(name)
            cat = ds_info.get("metadata", {}).get("category", "") if ds_info else ""
            icon = _CATEGORY_ICONS.get(cat, "📁")
            label = _CATEGORY_LABELS.get(cat, cat)
            table.add_row(name, f"{icon} {label}" if cat else "", ", ".join(splits))

        console.print()
        console.print(table)
        console.print()


@dataset.command()
@click.argument("name")
def pull(name: str):
    """Pull a dataset from HuggingFace or Harbor.

    Use the harbor: prefix for Harbor datasets (e.g., harbor:swebench-verified).
    """
    catalog = load_dataset_catalog()
    catalog_datasets = catalog.get("datasets", {})

    catalog_entry = catalog_datasets.get(name)

    # Explicit Harbor prefix: "harbor:<name>"
    if catalog_entry is None and name.startswith("harbor:"):
        from rllm.cli._pull import resolve_harbor_catalog_entry

        harbor_name = name.removeprefix("harbor:")
        console.print(f"[dim]Looking up '{harbor_name}' in Harbor registry...[/]")
        catalog_entry = resolve_harbor_catalog_entry(harbor_name)
        if catalog_entry:
            console.print(f"[bold green]Found Harbor dataset:[/] {harbor_name}")
            name = harbor_name

    if catalog_entry is None:
        fail(f"Dataset '{name}' not found in catalog. Use the harbor: prefix (e.g. rllm dataset pull harbor:swebench-verified).")

    click.echo(f"Pulling {name} from {catalog_entry['source']}...")
    pull_dataset(name, catalog_entry)
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
        not_found("Dataset", name)

    rows: list[tuple[str, str]] = []

    if catalog_entry:
        rows.append(("Description", str(catalog_entry.get("description", "N/A"))))
        rows.append(("Source", str(catalog_entry.get("source", "N/A"))))
        rows.append(("Category", str(catalog_entry.get("category", "N/A"))))
        rows.append(("Default agent", str(catalog_entry.get("default_agent", "N/A"))))
        rows.append(("Reward fn", str(catalog_entry.get("reward_fn", "N/A"))))
        rows.append(("Eval split", str(catalog_entry.get("eval_split", "N/A"))))

    if ds_info:
        splits = ds_info.get("splits", {})
        split_paths = [DatasetRegistry._resolve_path(s["path"]) for s in splits.values() if s.get("path")]
        if split_paths:
            location = os.path.commonpath(split_paths) if len(split_paths) > 1 else os.path.dirname(split_paths[0])
            rows.append(("Location", str(location)))

        for split, split_info in splits.items():
            num = split_info.get("num_examples", "?")
            fields = split_info.get("fields", [])
            value = f"{num} examples"
            if fields:
                value += f"\nfields: {', '.join(fields)}"
            rows.append((split, value))
    else:
        rows.append(("Status", f"not pulled (use 'rllm dataset pull {name}')"))

    console.print()
    console.print(info_panel(rows, title=f"Dataset: {name}"))
    console.print()


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
        fail(f"Cannot load '{name}' split '{split}'. Try 'rllm dataset pull {name}' first.")

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
