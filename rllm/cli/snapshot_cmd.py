"""``rllm snapshot``: manage sandbox environment snapshots like datasets.

A snapshot bakes a dataset's environments (base image + Dockerfile RUN steps)
into backend artifacts, so ``rllm eval`` / ``rllm train`` boot from them instead
of paying cold start every rollout. Snapshots persist and are user-managed:

    rllm snapshot create harbor:swebench-verified --sandbox-backend modal --max-examples 5
    rllm snapshot list
    rllm snapshot destroy harbor:swebench-verified --sandbox-backend modal
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import click
from rich.console import Console
from rich.table import Table
from rich.theme import Theme

theme = Theme({"label": "dim", "success": "bold green", "error": "bold red", "val": "bold", "key": "yellow"})
console = Console(theme=theme)

_MAX_BUILD_WORKERS = 4  # snapshot builds are network/IO-bound; a few in parallel is a real win


def _resolve_dataset_tasks(benchmark: str, agent_name: str | None, sandbox_backend: str | None, split: str | None) -> list:
    """Resolve a benchmark to a list of Tasks carrying environment metadata.

    Mirrors the sandboxed branches of ``rllm eval`` (local benchmark dir +
    harbor catalog) — the only datasets that have environments to snapshot.
    """
    from rllm.cli._pull import load_dataset_catalog, pull_dataset
    from rllm.cli.eval import _dict_rows_to_tasks
    from rllm.data import DatasetRegistry
    from rllm.tasks.loader import BenchmarkLoader

    if BenchmarkLoader.is_local_benchmark(benchmark):
        bench = BenchmarkLoader.load(benchmark, sandbox_backend=sandbox_backend, harness_name=agent_name)
        return list(bench.tasks)

    catalog = load_dataset_catalog()
    catalog_entry = catalog.get("datasets", {}).get(benchmark)
    if catalog_entry is None and benchmark.startswith("harbor:"):
        from rllm.cli._pull import resolve_harbor_catalog_entry

        harbor_name = benchmark.removeprefix("harbor:")
        catalog_entry = resolve_harbor_catalog_entry(harbor_name)
        benchmark = harbor_name

    if catalog_entry is None:
        raise click.ClickException(f"Benchmark '{benchmark}' not found in catalog and is not a local benchmark directory.")

    split = split or catalog_entry.get("eval_split", "test")
    dataset = DatasetRegistry.load_dataset(benchmark, split)
    if dataset is None:
        pull_dataset(benchmark, catalog_entry)
        dataset = DatasetRegistry.load_dataset(benchmark, split)
    if dataset is None:
        raise click.ClickException(f"Could not load dataset '{benchmark}' split '{split}'.")

    return _dict_rows_to_tasks(list(dataset.data))


def _slice_tasks(tasks: list, max_examples: int | None, task_indices: str | None) -> list:
    """Apply a ``--task-indices`` (e.g. '0,3-5') or ``--max-examples`` slice."""
    if task_indices is not None:
        idx: list[int] = []
        for part in task_indices.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-", 1)
                idx.extend(range(int(lo), int(hi) + 1))
            else:
                idx.append(int(part))
        return [tasks[i] for i in idx if 0 <= i < len(tasks)]
    if max_examples is not None:
        return tasks[:max_examples]
    return tasks


def _humanize_expiry(iso: str | None) -> str:
    from datetime import datetime, timezone

    if not iso:
        return "-"
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = dt - datetime.now(tz=timezone.utc)
    if delta.total_seconds() <= 0:
        return "[error]expired[/]"
    days, hours = delta.days, delta.seconds // 3600
    return f"in {days}d" if days else f"in {hours}h"


@click.group("snapshot")
def snapshot():
    """Manage sandbox environment snapshots."""


@snapshot.command("create")
@click.argument("benchmark")
@click.option(
    "--sandbox-backend",
    "sandbox_backend",
    required=True,
    type=click.Choice(["modal", "daytona"], case_sensitive=False),
    help="Backend to build snapshots on (only modal/daytona have snapshots).",
)
@click.option("--agent", "agent_name", default=None, help="Agent scaffold (only affects local-benchmark resolution).")
@click.option("--split", default=None, help="Dataset split (default: from catalog eval_split).")
@click.option("--max-examples", default=None, type=int, help="Snapshot only the first N tasks (e.g. the slice you'll eval).")
@click.option("--task-indices", default=None, type=str, help="Snapshot only these task indices (e.g. '0', '3,7,12', '0-9').")
@click.option("--ttl-hours", default=168.0, type=float, help="Local trust horizon in hours (default: 168 = 7 days).")
def create_snapshots(benchmark: str, sandbox_backend: str, agent_name: str | None, split: str | None, max_examples: int | None, task_indices: str | None, ttl_hours: float):
    """Build environment snapshots for a benchmark (or a slice of it)."""
    from rllm.eval._resolution import _resolve_image
    from rllm.sandbox.sandboxed_flow import build_snapshot
    from rllm.sandbox.snapshot import SnapshotRegistry, keys_for_tasks

    tasks = _slice_tasks(_resolve_dataset_tasks(benchmark, agent_name, sandbox_backend, split), max_examples, task_indices)
    by_key = keys_for_tasks(tasks, sandbox_backend)
    if not by_key:
        console.print(f"  [dim]No snapshottable environments for '{benchmark}' on {sandbox_backend}.[/]")
        return

    console.print(f"\n  Building [val]{len(by_key)}[/] snapshot(s) from [val]{len(tasks)}[/] task(s) on [val]{sandbox_backend}[/]\n")
    registry = SnapshotRegistry.load()

    def _build(item: tuple[str, object]) -> tuple[str, str, float]:
        key, task = item
        t0 = time.monotonic()
        try:
            ref = build_snapshot(sandbox_backend, task, key)
            if ref is not None:
                registry.record_env(key, sandbox_backend, ref, _resolve_image(task, sandbox_backend), benchmark, ttl_hours=ttl_hours)
            status = "[success]ok[/]" if ref is not None else "[error]no-snapshot[/]"
        except Exception as e:  # noqa: BLE001
            status = f"[error]failed[/] [dim]{e}[/]"
        return key, status, time.monotonic() - t0

    with ThreadPoolExecutor(max_workers=min(_MAX_BUILD_WORKERS, len(by_key))) as pool:
        results = list(pool.map(_build, by_key.items()))

    table = Table(box=None, padding=(0, 2))
    table.add_column("env_key", style="key")
    table.add_column("status", style="val")
    table.add_column("elapsed", style="dim", justify="right")
    for key, status, elapsed in results:
        table.add_row(key, status, f"{elapsed:.1f}s")
    console.print(table)
    console.print(f"\n  [dim]Registry: ~/.rllm/snapshots.json — reuse with[/] [bold]rllm eval {benchmark} --sandbox-backend {sandbox_backend}[/]\n")


@snapshot.command("list")
@click.option("--sandbox-backend", "sandbox_backend", default=None, type=click.Choice(["modal", "daytona"], case_sensitive=False), help="Filter to one backend.")
@click.option("--verbose", is_flag=True, help="Expand to one row per environment.")
def list_snapshots(sandbox_backend: str | None, verbose: bool):
    """List snapshots by dataset (local registry view)."""
    from rllm.sandbox.snapshot import SnapshotRegistry

    registry = SnapshotRegistry.load()

    if verbose:
        entries = {k: e for k, e in registry.env_entries().items() if sandbox_backend is None or e.get("backend") == sandbox_backend}
        if not entries:
            console.print("  [dim]No snapshots.[/]")
            return
        table = Table(box=None, padding=(0, 2))
        for col in ("env_key", "backend", "ref", "base image", "datasets", "expires"):
            table.add_column(col, style="key" if col == "env_key" else "dim")
        for key, e in entries.items():
            table.add_row(key, e.get("backend", "?"), str(e.get("ref", "?")), e.get("base_image", "?"), ", ".join(e.get("datasets", [])), _humanize_expiry(e.get("expires_at")))
        console.print(table)
        return

    rows = [c for c in registry.collections() if sandbox_backend is None or c["backend"] == sandbox_backend]
    if not rows:
        console.print("  [dim]No snapshots. Build some with[/] [bold]rllm snapshot create <dataset> --sandbox-backend <b>[/]")
        return
    table = Table(box=None, padding=(0, 2))
    table.add_column("dataset", style="val")
    table.add_column("backend", style="key")
    table.add_column("envs", style="dim", justify="right")
    table.add_column("expires", style="dim")
    for c in rows:
        table.add_row(c["dataset"], c["backend"], str(c["envs"]), _humanize_expiry(c["expires_at"]))
    console.print(table)


@snapshot.command("destroy")
@click.argument("benchmark")
@click.option(
    "--sandbox-backend",
    "sandbox_backend",
    default=None,
    type=click.Choice(["modal", "daytona"], case_sensitive=False),
    help="Restrict to one backend (default: all backends for this dataset).",
)
def destroy_snapshots(benchmark: str, sandbox_backend: str | None):
    """Delete a dataset's snapshots from their backend and the registry."""
    from rllm.sandbox.snapshot import SnapshotRegistry

    detached, deleted = SnapshotRegistry.load().destroy(benchmark, sandbox_backend)
    if detached == 0:
        scope = f" on {sandbox_backend}" if sandbox_backend else ""
        console.print(f"  [dim]No snapshots recorded for '{benchmark}'{scope}.[/]")
        return
    console.print(f"  [success]Removed[/] {deleted} backend snapshot(s); detached '{benchmark}' from {detached} environment(s).")
