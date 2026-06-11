"""``rllm snapshot``: manage sandbox environment snapshots like datasets.

A snapshot bakes a task's environment (base image + Dockerfile RUN steps) into a
backend artifact, so ``rllm eval`` / ``rllm train`` boot from it instead of paying
cold start every rollout. Each ``create`` records a named *group*: a thin manifest
of the exact tasks it covered and the env_keys they resolved to. Groups are what a
human names, lists, inspects, and destroys; snapshots are shared freely between them.

    rllm snapshot create harbor:swebench-verified --sandbox-backend modal --max-examples 5
    rllm snapshot list
    rllm snapshot info    swebench-verified-first5-a17c3d9e
    rllm snapshot inspect swebench-verified-first5-a17c3d9e
    rllm snapshot destroy swebench-verified-first5-a17c3d9e
    rllm snapshot renew   swebench-verified-first5-a17c3d9e --ttl-hours 168
    rllm snapshot sync --sandbox-backend modal
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import click
from rich.table import Table

from rllm.cli._ui import console, fail, info_panel, not_found, parse_index_spec

_MAX_BUILD_WORKERS = 4  # snapshot builds are network/IO-bound; a few in parallel is a real win

# Build outcome → cell markup (status is carried as a plain code, styled only at render).
_STATUS_MARKUP = {
    "reused": "[dim]reused[/]",
    "ok": "[success]ok[/]",
    "no-snapshot": "[error]no-snapshot[/]",
    "failed": "[error]failed[/]",
}


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
        fail(f"Benchmark '{benchmark}' not found in catalog and is not a local benchmark directory.")

    split = split or catalog_entry.get("eval_split", "test")
    dataset = DatasetRegistry.load_dataset(benchmark, split)
    if dataset is None:
        pull_dataset(benchmark, catalog_entry)
        dataset = DatasetRegistry.load_dataset(benchmark, split)
    if dataset is None:
        fail(f"Could not load dataset '{benchmark}' split '{split}'.")

    return _dict_rows_to_tasks(list(dataset.data))


def _slice_tasks(tasks: list, max_examples: int | None, task_indices: str | None) -> list:
    """Apply a ``--task-indices`` (e.g. '0,3-5') or ``--max-examples`` slice."""
    if task_indices is not None:
        return [tasks[i] for i in parse_index_spec(task_indices) if 0 <= i < len(tasks)]
    if max_examples is not None:
        return tasks[:max_examples]
    return tasks


def _slice_spec(max_examples: int | None, task_indices: str | None) -> dict:
    """The slice descriptor stored on the group (``kind`` + raw ``value`` for display)."""
    if task_indices is not None:
        return {"kind": "task_indices", "value": task_indices}
    if max_examples is not None:
        return {"kind": "max_examples", "value": max_examples}
    return {"kind": "all", "value": None}


def _slice_display(slice_spec: dict) -> str:
    """Render a slice spec for the ``list``/``info`` columns."""
    kind, value = slice_spec.get("kind"), slice_spec.get("value")
    if kind == "max_examples":
        return f"first {value}"
    if kind == "task_indices":
        return f"idx {value}"
    return "all"


def _humanize_expiry(iso: str | None) -> str:
    """Plain time-to-expiry: ``-`` if unknown, ``expired`` if past, else ``in Nd`` / ``in Nh``."""
    from datetime import datetime, timezone

    if not iso:
        return "-"
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = dt - datetime.now(tz=timezone.utc)
    if delta.total_seconds() <= 0:
        return "expired"
    days, hours = delta.days, delta.seconds // 3600
    return f"in {days}d" if days else f"in {hours}h"


def _is_live(iso: str | None) -> bool:
    """True if an env's expiry is in the future."""
    from rllm.sandbox.snapshot import _expired

    return bool(iso) and not _expired(iso)


def _expiry_cell(iso: str | None) -> str:
    """Expiry text for a table cell — reddened when expired."""
    text = _humanize_expiry(iso)
    return f"[error]{text}[/]" if iso and not _is_live(iso) else text


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
@click.option("--agent", "agent_name", default=None, help="Agent scaffold; its CLI install (if any) is baked into the snapshots.")
@click.option("--split", default=None, help="Dataset split (default: from catalog eval_split).")
@click.option("--max-examples", default=None, type=int, help="Snapshot only the first N tasks (e.g. the slice you'll eval).")
@click.option("--task-indices", default=None, type=str, help="Snapshot only these task indices (e.g. '0', '3,7,12', '0-9').")
@click.option("--ttl-hours", default=168.0, type=float, help="Local trust horizon in hours (default: 168 = 7 days).")
@click.option("--force", is_flag=True, help="Rebuild even when a live snapshot exists, and refresh its TTL.")
def create_snapshots(benchmark: str, sandbox_backend: str, agent_name: str | None, split: str | None, max_examples: int | None, task_indices: str | None, ttl_hours: float, force: bool):
    """Build environment snapshots for a benchmark (or a slice of it) and record a group."""
    from rllm.eval._resolution import _resolve_image
    from rllm.sandbox.sandboxed_flow import build_snapshot
    from rllm.sandbox.snapshot import SnapshotRegistry, env_key_for, install_script_for, keys_for_tasks

    install = ""
    if agent_name is not None:
        from rllm.eval.agent_loader import load_agent

        install = install_script_for(load_agent(agent_name))

    tasks = _slice_tasks(_resolve_dataset_tasks(benchmark, agent_name, sandbox_backend, split), max_examples, task_indices)
    by_key = keys_for_tasks(tasks, sandbox_backend, install)
    if not by_key:
        console.print(f"  [dim]No snapshottable environments for '{benchmark}' on {sandbox_backend}.[/]")
        return

    baked = f" (baking [val]{agent_name}[/] install)" if install else ""
    console.print(f"\n  Building [val]{len(by_key)}[/] snapshot(s) from [val]{len(tasks)}[/] task(s) on [val]{sandbox_backend}[/]{baked}\n")
    registry = SnapshotRegistry.load()

    def _build(item: tuple[str, object]) -> tuple[str, str | None, str, str | None, float]:
        key, task = item
        t0 = time.monotonic()
        err = None
        try:
            prior_ref = registry.lookup_env(key, sandbox_backend)  # a live local ref to reuse when not forced
            ref = build_snapshot(sandbox_backend, task, key, prior_ref, force=force, install_script=install)
            reused = ref is not None and not force and prior_ref == ref
            status = "reused" if reused else "ok" if ref is not None else "no-snapshot"
        except Exception as e:  # noqa: BLE001
            ref, status, err = None, "failed", str(e)
        return key, ref, status, err, time.monotonic() - t0

    with ThreadPoolExecutor(max_workers=min(_MAX_BUILD_WORKERS, len(by_key))) as pool:
        results = {key: (ref, status, err, elapsed) for key, ref, status, err, elapsed in pool.map(_build, by_key.items())}

    # the group's tasks: the exact sliced tasks whose env actually built
    built_tasks = []
    for task in tasks:
        key = env_key_for(task, sandbox_backend, install)
        ref = results.get(key, (None,))[0]
        if ref is not None:
            built_tasks.append({"id": task.id, "env_key": key, "ref": ref, "base_image": _resolve_image(task, sandbox_backend)})

    group_id = None
    if built_tasks:
        group_id = registry.record_group(benchmark, sandbox_backend, _slice_spec(max_examples, task_indices), built_tasks, ttl_hours=ttl_hours, force=force)

    table = Table(box=None, padding=(0, 2))
    table.add_column("env_key", style="key")
    table.add_column("status", style="val")
    table.add_column("elapsed", style="dim", justify="right")
    for key, (ref, status, err, elapsed) in results.items():
        cell = _STATUS_MARKUP[status] + (f" [dim]{err}[/]" if err else "")
        table.add_row(key, cell, f"{elapsed:.1f}s")
    console.print(table)
    if group_id:
        console.print(f"\n  group [val]{group_id}[/]")
    console.print(f"\n  [dim]Registry: ~/.rllm/snapshots.json — reuse with[/] [bold]rllm eval {benchmark} --sandbox-backend {sandbox_backend}[/]\n")


@snapshot.command("list")
@click.option("--sandbox-backend", "sandbox_backend", default=None, type=click.Choice(["modal", "daytona"], case_sensitive=False), help="Filter to one backend.")
@click.option("--verbose", is_flag=True, help="Also expand each group to one row per env_key.")
def list_snapshots(sandbox_backend: str | None, verbose: bool):
    """List snapshot groups (one row per create invocation)."""
    from rllm.sandbox.snapshot import SnapshotRegistry

    registry = SnapshotRegistry.load()
    envs = registry.env_entries()
    groups = {gid: g for gid, g in registry.groups().items() if sandbox_backend is None or g.get("backend") == sandbox_backend}
    if not groups:
        console.print("  [dim]No snapshot groups. Build some with[/] [bold]rllm snapshot create <dataset> --sandbox-backend <b>[/]")
        return

    table = Table(box=None, padding=(0, 2))
    table.add_column("group", style="val")
    table.add_column("dataset", style="dim")
    table.add_column("backend", style="key")
    table.add_column("slice", style="dim")
    table.add_column("envs", style="dim", justify="right")
    table.add_column("live", style="dim", justify="right")
    table.add_column("expires", style="dim")
    for gid, g in sorted(groups.items()):
        members = registry.group_members(gid)
        live = sum(1 for k in members if _is_live(envs.get(k, {}).get("expires_at")))
        expiries = [envs[k]["expires_at"] for k in members if k in envs and envs[k].get("expires_at")]
        soonest = min(expiries) if expiries else None
        table.add_row(gid, g.get("dataset", "?"), g.get("backend", "?"), _slice_display(g.get("slice", {})), str(len(members)), str(live), _expiry_cell(soonest))
    console.print(table)

    if verbose:
        for gid in sorted(groups):
            console.print(f"\n  [val]{gid}[/]")
            sub = Table(box=None, padding=(0, 2))
            for col in ("env_key", "ref", "base image", "expires"):
                sub.add_column(col, style="key" if col == "env_key" else "dim")
            for k in registry.group_members(gid):
                e = envs.get(k, {})
                sub.add_row(k, str(e.get("ref", "?")), e.get("base_image", "?"), _expiry_cell(e.get("expires_at")))
            console.print(sub)


@snapshot.command("info")
@click.argument("group_id")
def info_snapshot(group_id: str):
    """Show one group's metadata (dataset, backend, slice, members, sharing)."""
    from rllm.sandbox.snapshot import SnapshotRegistry

    registry = SnapshotRegistry.load()
    group = registry.groups().get(group_id)
    if group is None:
        not_found("Group", group_id, "See `rllm snapshot list`.")

    envs = registry.env_entries()
    members = registry.group_members(group_id)
    live = sum(1 for k in members if _is_live(envs.get(k, {}).get("expires_at")))
    shared = sorted({g for k in members for g in registry.groups_referencing(k) if g != group_id})

    rows = [
        ("Dataset", group.get("dataset", "?")),
        ("Backend", f"[key]{group.get('backend', '?')}[/]"),
        ("Slice", _slice_display(group.get("slice", {}))),
        ("Created", str(group.get("created_at", "?"))),
        ("TTL", f"{group.get('ttl_hours', '?')}h"),
        ("Members", f"{len(members)} env(s), [success]{live} live[/] / {len(members) - live} expired"),
        ("Shared with", ", ".join(shared) if shared else "[dim]none[/]"),
    ]
    console.print(info_panel(rows, title=f"[val]{group_id}[/]", border="cyan"))


@snapshot.command("inspect")
@click.argument("group_or_env_key")
def inspect_snapshot(group_or_env_key: str):
    """Drill into a group's exact tasks, or a single env_key's groups.

    For a group: the precise tasks it was created from (task id + env_key +
    base image + live status). For a bare env_key: its row plus which groups
    reference it.
    """
    from rllm.sandbox.snapshot import SnapshotRegistry

    registry = SnapshotRegistry.load()
    envs = registry.env_entries()
    group = registry.groups().get(group_or_env_key)

    if group is not None:
        console.print(f"\n  [val]{group_or_env_key}[/]  [dim]{group.get('dataset', '?')} · {group.get('backend', '?')} · {_slice_display(group.get('slice', {}))}[/]\n")
        table = Table(box=None, padding=(0, 2))
        table.add_column("task", style="val")
        table.add_column("env_key", style="key")
        table.add_column("base image", style="dim")
        table.add_column("status", style="dim")
        for t in group.get("tasks", []):
            e = envs.get(t["env_key"], {})
            status = "[success]live[/]" if _is_live(e.get("expires_at")) else "[error]expired/gone[/]"
            table.add_row(str(t.get("id")), t["env_key"], e.get("base_image", "?"), status)
        console.print(table)
        console.print()
        return

    if group_or_env_key in envs:
        e = envs[group_or_env_key]
        in_groups = sorted(registry.groups_referencing(group_or_env_key))
        rows = [
            ("env_key", f"[key]{group_or_env_key}[/]"),
            ("Backend", e.get("backend", "?")),
            ("Ref", str(e.get("ref", "?"))),
            ("Base image", e.get("base_image", "?")),
            ("Expires", _expiry_cell(e.get("expires_at"))),
            ("In groups", ", ".join(in_groups) if in_groups else "[dim]none[/]"),
        ]
        console.print(info_panel(rows, title=f"[val]{group_or_env_key}[/]", border="cyan"))
        return

    not_found("Group or env_key", group_or_env_key, "See `rllm snapshot list`.")


@snapshot.command("destroy")
@click.argument("group_id")
def destroy_snapshot(group_id: str):
    """Delete a group; backend snapshots survive while another group still uses them."""
    from rllm.sandbox.snapshot import SnapshotRegistry

    result = SnapshotRegistry.load().destroy_group(group_id)
    if not result["found"]:
        not_found("Group", group_id, "See `rllm snapshot list`.")

    console.print(f"  [success]Removed group[/] {group_id}")
    if result["deleted"]:
        console.print(f"  Deleted {len(result['deleted'])} backend snapshot(s) no longer referenced.")
    if result["shared"]:
        console.print(f"  [dim]Kept {len(result['shared'])} snapshot(s) still used by other groups.[/]")
    if result["kept"]:
        console.print(f"  [error]Kept {len(result['kept'])} snapshot(s) locally[/] [dim]— backend delete couldn't be confirmed; retry later.[/]")


@snapshot.command("renew")
@click.argument("group_id")
@click.option("--ttl-hours", default=168.0, type=float, help="New trust horizon in hours (default: 168 = 7 days).")
def renew_snapshot(group_id: str, ttl_hours: float):
    """Refresh a group's members' TTL without rebuilding."""
    from rllm.sandbox.snapshot import SnapshotRegistry

    registry = SnapshotRegistry.load()
    if group_id not in registry.groups():
        not_found("Group", group_id, "See `rllm snapshot list`.")
    renewed = registry.renew(group_id, ttl_hours)
    console.print(f"  [success]Renewed[/] {renewed} env(s) of {group_id} to +{ttl_hours}h.")


@snapshot.command("sync")
@click.option(
    "--sandbox-backend",
    "sandbox_backend",
    required=True,
    type=click.Choice(["modal", "daytona"], case_sensitive=False),
    help="Backend to reconcile local records against.",
)
def sync_snapshots(sandbox_backend: str):
    """Reconcile local records against the backend — drop only verified-absent envs."""
    from rllm.sandbox.snapshot import SnapshotRegistry

    result = SnapshotRegistry.load().sync(sandbox_backend)
    pruned = result["pruned"]
    if pruned:
        console.print(f"  [success]Pruned[/] {len(pruned)} absent env(s) on {sandbox_backend}: [dim]{', '.join(pruned)}[/]")
    console.print(f"  [dim]Kept {result['kept']} env(s) (present or unconfirmed).[/]")
