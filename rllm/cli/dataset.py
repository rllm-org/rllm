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


def _split_train_val(rows: list[dict], val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Hold out a fraction of *tasks* (not rows) for validation, to avoid leaking
    sibling trajectories of the same task across the split."""
    if val_fraction <= 0 or not rows:
        return rows, []
    import random

    by_task: dict = {}
    for r in rows:
        by_task.setdefault(r.get("task_id"), []).append(r)
    task_ids = list(by_task.keys())
    random.Random(seed).shuffle(task_ids)
    n_val = int(len(task_ids) * val_fraction)
    if 0 < val_fraction < 1 and n_val == 0:
        n_val = 1  # always hold out at least one task when a fraction was asked for
    val_tasks = set(task_ids[:n_val])
    train_rows, val_rows = [], []
    for tid, task_rows in by_task.items():
        (val_rows if tid in val_tasks else train_rows).extend(task_rows)
    return train_rows, val_rows


@dataset.command(name="from-eval")
@click.argument("runs", nargs=-1, required=True)
@click.option("--name", required=True, help="Name to register the curated SFT dataset under.")
@click.option("--metric", default="is_correct", help="What avg/best/worst aggregate: is_correct (default), reward, or a signal name.")
@click.option("--filter", "filter_expr", default="solved", help='Task-level filter over aggregates, e.g. "0 < avg < 1" or "pass@4 >= 0.5" (default: solved).')
@click.option("--select", default="correct", type=click.Choice(["correct", "best", "best-n", "shortest", "all"]), help="Which trajectories to keep per task (default: correct).")
@click.option("--max-per-task", default=None, type=int, help="Cap trajectories kept per task.")
@click.option("--min-reward", default=None, type=float, help="Per-trajectory passing threshold on the metric (default: use is_correct).")
@click.option("--dedup/--no-dedup", default=False, help="Drop duplicate assistant solutions (default: off).")
@click.option("--trajectory", default=None, help="Named trajectory to extract for multi-agent flows (default: first).")
@click.option("--split", default="train", help="Split name to register the curated rows under (default: train).")
@click.option("--val-fraction", default=0.0, type=float, help="Hold out this fraction of tasks as a validation split.")
@click.option("--val-split", default="test", help="Split name for the held-out validation rows (default: test).")
@click.option("--seed", default=0, type=int, help="Seed for the train/val split.")
@click.option("--category", default=None, help="Dataset category metadata (e.g. math, code).")
@click.option("--description", default=None, help="Dataset description metadata.")
@click.option("--dry-run", is_flag=True, help="Report what would be curated without registering anything.")
def from_eval(
    runs: tuple[str, ...],
    name: str,
    metric: str,
    filter_expr: str,
    select: str,
    max_per_task: int | None,
    min_reward: float | None,
    dedup: bool,
    trajectory: str | None,
    split: str,
    val_fraction: float,
    val_split: str,
    seed: int,
    category: str | None,
    description: str | None,
    dry_run: bool,
):
    """Curate eval trajectories into an SFT dataset.

    RUNS are eval run ids (under ~/.rllm/eval_results) or paths to run dirs.
    Filters tasks by aggregate metrics, selects trajectories, and registers the
    survivors as a `messages` dataset ready for `rllm sft`.

    \b
    Examples:
      rllm dataset from-eval math500_run --name math500-rft --filter "0 < avg < 1"
      rllm dataset from-eval run_a run_b --name pooled --select best --max-per-task 1
      rllm dataset from-eval run --name d --filter "pass@4 >= 0.5" --dry-run
    """
    from rllm.data import DatasetRegistry
    from rllm.eval.curation import CurationConfig, CurationError, curate
    from rllm.eval.filter_dsl import FilterError

    if not 0.0 <= val_fraction < 1.0:
        fail("--val-fraction must be in [0, 1).")

    config = CurationConfig(
        metric=metric,
        filter_expr=filter_expr,
        select=select,
        max_per_task=max_per_task,
        min_reward=min_reward,
        dedup=dedup,
        trajectory=trajectory,
    )

    try:
        rows, stats = curate(list(runs), config)
    except (CurationError, FilterError) as e:
        fail(str(e))

    summary = [
        ("Runs", f"[val]{stats.runs}[/]"),
        ("Tasks kept", f"[val]{stats.tasks_kept}[/] [dim]/ {stats.tasks_total}[/]"),
        ("Attempts pooled", f"[dim]{stats.attempts_total}[/]"),
        ("Filter", f"[dim]{config.filter_expr}  (metric={config.metric})[/]"),
        ("Select", f"[dim]{config.select}{f', max {max_per_task}/task' if max_per_task else ''}[/]"),
        ("Trajectories", f"[val]{stats.rows_emitted}[/]"),
    ]
    if stats.rows_skipped_no_messages:
        summary.append(("Skipped", f"[yellow]{stats.rows_skipped_no_messages}[/] [dim](no usable messages)[/]"))
    if stats.rows_deduped:
        summary.append(("Deduped", f"[dim]{stats.rows_deduped}[/]"))

    console.print()
    console.print(info_panel(summary, title="[bold]Curation[/]", border="brand"))

    if stats.rows_emitted == 0:
        console.print()
        console.print("  [yellow]No trajectories matched.[/] Loosen [bold]--filter[/], lower [bold]--min-reward[/], or check [bold]--metric[/].")
        console.print()
        if not dry_run:
            raise SystemExit(1)
        return

    if dry_run:
        console.print()
        console.print(f"  [dim]Dry run — nothing registered. Re-run without --dry-run to save as [bold]{name}[/].[/]")
        console.print()
        return

    train_rows, val_rows = _split_train_val(rows, val_fraction, seed)

    DatasetRegistry.register_dataset(
        name,
        train_rows,
        split=split,
        source="curated-from-eval",
        category=category,
        description=description,
    )
    registered = [f"{split} ({len(train_rows)})"]
    if val_rows:
        DatasetRegistry.register_dataset(name, val_rows, split=val_split, source="curated-from-eval")
        registered.append(f"{val_split} ({len(val_rows)})")

    console.print()
    console.print(f"  [success]Registered[/] [val]{name}[/]  [dim]splits: {', '.join(registered)}[/]")
    console.print(f"  [dim]Inspect: [bold]rllm dataset inspect {name}[/][/]")
    console.print(f"  [dim]Train:   [bold]rllm sft {name} --model <model> --backend tinker[/][/]")
    console.print()


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
