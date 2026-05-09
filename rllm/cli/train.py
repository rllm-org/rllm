"""Train CLI command.

``rllm train <benchmark> --model <name> [OPTIONS]``

Reuses the eval framework's dataset catalog, AgentFlows, and Evaluators to run
RL training via the Tinker backend. Routes every rollout through
``AgentFlowEngine`` (the same engine eval uses); for sandbox-style harnesses
and harbor-sourced datasets, ``EvalHooks`` provides per-task sandbox lifecycle
and per-task verifier resolution.
"""

from __future__ import annotations

import os
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

from rllm.cli._pull import load_dataset_catalog, pull_dataset

theme = Theme({"label": "dim", "success": "bold green", "error": "bold red", "val": "bold", "key": "yellow"})
console = Console(theme=theme)

# Path to the bundled YAML config templates
_CONFIG_PKG = Path(__file__).resolve().parent.parent / "experimental" / "config"


# ---------------------------------------------------------------------------
# 1. build_train_config  — CLI flags → OmegaConf DictConfig
# ---------------------------------------------------------------------------


def build_train_config(
    *,
    model_name: str,
    group_size: int,
    batch_size: int,
    lr: float,
    lora_rank: int,
    total_epochs: int,
    total_steps: int | None,
    val_freq: int,
    save_freq: int,
    project: str,
    experiment: str,
    output_dir: str | None,
    config_file: str | None,
):
    """Build an OmegaConf DictConfig from YAML templates + CLI overrides.

    Produces the same structure that Hydra's ``@hydra.main`` with
    ``unified.yaml`` would produce, without requiring the Hydra runtime.
    """
    from omegaconf import OmegaConf

    # Load the two template files
    base_cfg = OmegaConf.load(str(_CONFIG_PKG / "rllm" / "base.yaml"))
    tinker_cfg = OmegaConf.load(str(_CONFIG_PKG / "rllm" / "backend" / "tinker.yaml"))

    # tinker.yaml has a top-level ``rllm:`` key with backend-specific overrides
    # that should merge into the ``rllm`` namespace.
    tinker_rllm = OmegaConf.to_container(tinker_cfg.get("rllm", {}), resolve=False)
    tinker_top = OmegaConf.to_container(tinker_cfg, resolve=False)
    tinker_top.pop("rllm", None)

    # Merge: base → rllm key, tinker top-level, tinker rllm overrides
    merged = OmegaConf.merge(
        {"rllm": base_cfg},
        OmegaConf.create(tinker_top),
        {"rllm": OmegaConf.create(tinker_rllm)},
    )

    # If user provided a --config file, merge it on top
    if config_file:
        user_cfg = OmegaConf.load(config_file)
        merged = OmegaConf.merge(merged, user_cfg)

    # Apply CLI overrides (only non-default values)
    overrides = OmegaConf.create(
        {
            "model": {"name": model_name, "lora_rank": lora_rank},
            "training": {"group_size": group_size, "learning_rate": lr},
            "validation": {"group_size": group_size},
            "data": {"train_batch_size": batch_size},
            "rllm": {
                "model_name": model_name,
                "trainer": {
                    "total_epochs": total_epochs,
                    "test_freq": val_freq,
                    "save_freq": save_freq,
                    "project_name": project,
                    "experiment_name": experiment,
                },
                "rollout": {
                    "n": group_size,
                },
                "workflow": {
                    "use_workflow": True,
                    "workflow_args": {
                        "timeout": 300,  # 5-minute timeout per rollout
                    },
                },
            },
        }
    )
    merged = OmegaConf.merge(merged, overrides)

    # total_steps overrides epochs
    if total_steps is not None:
        merged = OmegaConf.merge(
            merged,
            OmegaConf.create(
                {
                    "rllm": {"trainer": {"total_batches": total_steps, "total_epochs": 1}},
                }
            ),
        )

    # Output directory
    if output_dir is not None:
        merged = OmegaConf.merge(
            merged,
            OmegaConf.create(
                {
                    "training": {"default_local_dir": output_dir},
                }
            ),
        )

    return merged


# ---------------------------------------------------------------------------
# 2. _run_train  — core training logic
# ---------------------------------------------------------------------------


def _run_train(
    benchmark: str,
    agent_name: str | None,
    evaluator_name: str | None,
    model: str,
    train_dataset_name: str | None,
    train_split: str | None,
    val_dataset_name: str | None,
    val_split: str | None,
    max_examples: int | None,
    group_size: int,
    batch_size: int,
    lr: float,
    lora_rank: int,
    total_epochs: int,
    total_steps: int | None,
    val_freq: int,
    save_freq: int,
    project: str,
    experiment: str,
    output_dir: str | None,
    config_file: str | None,
    enable_ui: bool = False,
    sandbox_backend: str | None = None,
    sandbox_concurrency: int | None = None,
):
    """Core training logic: resolve catalog, load data, build config, launch trainer."""

    try:
        from rllm.eval.agent_loader import load_agent
        from rllm.eval.evaluator_loader import load_evaluator, resolve_evaluator_from_catalog
        from rllm.experimental.unified_trainer import AgentTrainer
    except ImportError as e:
        console.print(f"  [error]Missing training dependencies: {e}[/]")
        console.print("  Install with: [bold]pip install rllm\\[train][/]")
        raise SystemExit(1) from None

    # ------------------------------------------------------------------
    # Local benchmark path: directory with dataset.toml / task.toml
    # ------------------------------------------------------------------
    from rllm.tasks.loader import BenchmarkLoader

    catalog = {}  # needed for Harbor config path later
    catalog_entry = None
    train_ds_name = train_dataset_name or benchmark
    val_ds_name = val_dataset_name or benchmark

    if BenchmarkLoader.is_local_benchmark(benchmark):
        # For local sandbox tasks, --agent picks the AgentFlow.
        bench_result = BenchmarkLoader.load(benchmark, harness_name=agent_name)
        catalog_entry = {
            "description": bench_result.description,
            "category": bench_result.category,
            "default_agent": bench_result.harness_name,
        }
        if agent_name is None:
            agent_name = bench_result.harness_name or "react"

        try:
            agent_flow = load_agent(agent_name)
        except (KeyError, ImportError, AttributeError, TypeError) as e:
            console.print(f"  [error]Cannot load agent '{agent_name}': {e}[/]")
            raise SystemExit(1) from None

        # Evaluator: --evaluator > dataset.toml [verifier].
        # Sandbox-shell verifiers aren't supported here (per-task sandbox
        # lifecycle lives inside Runner) — use --evaluator to override.
        evaluator_display = "N/A"
        if evaluator_name is not None:
            evaluator = load_evaluator(evaluator_name)
            evaluator_display = evaluator_name
        else:
            from pathlib import Path as _Path

            from rllm.eval._resolution import build_dataset_evaluator

            evaluator = build_dataset_evaluator(_Path(benchmark).resolve())
            if evaluator is None:
                console.print(
                    "  [error]Could not resolve a verifier for this benchmark. "
                    "Declare a host-side verifier in dataset.toml ([verifier].name / "
                    ".module / .import_path) or pass --evaluator explicitly. "
                    "Sandbox-shell verifiers aren't supported in training yet.[/]"
                )
                raise SystemExit(1)
            evaluator_display = f"{type(evaluator).__name__} (from dataset.toml)"

        # Datasets: pass the Task list as both train and val for now
        from rllm.data.dataset import Dataset as _Dataset

        if train_split is None:
            train_split = bench_result.split or "train"
        train_dataset = _Dataset(
            data=list(bench_result.tasks),
            name=bench_result.name,
            split=train_split,
        )
        if max_examples is not None and max_examples < len(train_dataset):
            train_dataset = train_dataset.select(range(max_examples))
        val_dataset = _Dataset(
            data=list(bench_result.tasks),
            name=bench_result.name,
            split=bench_result.split or "test",
        )
        if val_split is None:
            val_split = train_dataset.split or "test"

    # ------------------------------------------------------------------
    # Catalog / Harbor path (existing behavior)
    # ------------------------------------------------------------------
    else:
        # ---- Load catalog ----
        catalog = load_dataset_catalog()
        catalog_entry = catalog.get("datasets", {}).get(benchmark)

        # ---- Explicit Harbor prefix: "harbor:<name>" ----
        if catalog_entry is None and benchmark.startswith("harbor:"):
            from rllm.cli._pull import resolve_harbor_catalog_entry

            harbor_name = benchmark.removeprefix("harbor:")
            catalog_entry = resolve_harbor_catalog_entry(harbor_name)
            if catalog_entry:
                console.print(f"  [success]Found Harbor dataset:[/] [val]{harbor_name}[/]")
                benchmark = harbor_name

        # ---- Docker check for Harbor datasets ----
        if catalog_entry and catalog_entry.get("source", "").startswith("harbor:"):
            from rllm.integrations.harbor.utils import diagnose_docker

            ok, reason, hint = diagnose_docker()
            if not ok:
                console.print(f"  [error]Harbor tasks require Docker — {reason}.[/]")
                if hint:
                    console.print(f"  [dim]{hint}[/]")
                raise SystemExit(1)

        # ---- Resolve agent ----
        if agent_name is None:
            if catalog_entry and "default_agent" in catalog_entry:
                agent_name = catalog_entry["default_agent"]
            else:
                console.print(f"  [error]No --agent specified and no default_agent in catalog for '{benchmark}'.[/]")
                raise SystemExit(1)

        try:
            agent_flow = load_agent(agent_name)
        except (KeyError, ImportError, AttributeError, TypeError) as e:
            console.print(f"  [error]Error loading agent '{agent_name}': {e}[/]")
            raise SystemExit(1) from None

        _is_harbor_source = bool(catalog_entry) and catalog_entry.get("source", "").startswith("harbor:")
        _is_harbor_agent = bool(agent_name) and agent_name.startswith("harbor:")

        # ---- Resolve evaluator ----
        # ``harbor_reward_fn`` reads ``episode.artifacts['harbor_reward']``,
        # which is only populated by HarborRuntime (``harbor:<scaffold>``).
        # When a harbor-sourced dataset is run through an rllm-native harness
        # (mini-swe-agent / opencode / oracle / …), the artifact is never set
        # and the evaluator would always score zero. Skip the catalog
        # evaluator in that case and let ``EvalHooks`` resolve a per-task
        # verifier from the harbor task dir (typically ``tests/test.sh``).
        evaluator = None
        evaluator_display = "per-task (from task.toml/dataset.toml)"
        if evaluator_name is not None:
            try:
                evaluator = load_evaluator(evaluator_name)
                evaluator_display = f"{evaluator_name} (overrides per-task verifier)"
            except (KeyError, ImportError, AttributeError, TypeError) as e:
                console.print(f"  [error]Error loading evaluator '{evaluator_name}': {e}[/]")
                raise SystemExit(1) from None
        else:
            _harbor_reward_fn_skipped = _is_harbor_source and catalog_entry.get("reward_fn") == "harbor_reward_fn" and not _is_harbor_agent
            if _harbor_reward_fn_skipped:
                evaluator_display = "per-task (rllm runtime on harbor task)"
            else:
                evaluator = resolve_evaluator_from_catalog(benchmark)
                if evaluator is not None:
                    reward_fn_name = catalog_entry.get("reward_fn", "") if catalog_entry else ""
                    evaluator_display = reward_fn_name or type(evaluator).__name__
                elif catalog_entry and catalog_entry.get("reward_fn"):
                    try:
                        evaluator = load_evaluator(catalog_entry["reward_fn"])
                        evaluator_display = catalog_entry["reward_fn"]
                    except (KeyError, ImportError):
                        pass

        # For non-harbor catalog datasets, training still requires a
        # global evaluator (no per-task verifier files exist for HF rows).
        # Harbor datasets fall back to per-task verifiers via EvalHooks.
        if evaluator is None and not _is_harbor_source:
            console.print(f"  [error]No evaluator found for '{benchmark}'. Specify --evaluator explicitly.[/]")
            raise SystemExit(1)

        # ---- Resolve dataset names ----
        train_ds_name = train_dataset_name or benchmark
        val_ds_name = val_dataset_name or benchmark

        # ---- Resolve catalog entries for train + val datasets ----
        # User may pass ``--val-dataset harbor:<name>`` (or the raw
        # ``<name>`` after a prior ``rllm dataset pull``). Harbor entries
        # aren't written into ``catalog["datasets"]``, so look them up
        # via ``resolve_harbor_catalog_entry`` as a fallback. Cache the
        # benchmark's entry so we don't re-probe Harbor for it.
        train_entry = _resolve_dataset_entry(train_ds_name, catalog, benchmark, catalog_entry)
        val_entry = _resolve_dataset_entry(val_ds_name, catalog, benchmark, catalog_entry)

        # Resolve train split from catalog if not provided. Prefer an
        # explicit ``train_split`` field; otherwise harbor datasets
        # (which only register their single ``eval_split``) need that
        # split for training; for everything else the historical "train"
        # default is correct.
        if train_split is None:
            if train_entry and "train_split" in train_entry:
                train_split = train_entry["train_split"]
            elif train_entry and train_entry.get("source", "").startswith("harbor:"):
                train_split = train_entry.get("eval_split", "default")
            else:
                train_split = "train"

        # Resolve val split from catalog if not provided. Same chain.
        if val_split is None:
            val_split = val_entry.get("eval_split", "test") if val_entry else "test"

        # ---- Load training dataset ----
        # Pass through the resolved catalog_entry so harbor datasets get
        # auto-pulled even when their entry was synthesised at runtime.
        train_dataset = _load_or_pull_dataset(train_ds_name, train_split, catalog, train_entry)
        if train_dataset is None:
            console.print(f"  [error]Could not load training dataset '{train_ds_name}' split '{train_split}'.[/]")
            raise SystemExit(1)

        if max_examples is not None and max_examples < len(train_dataset):
            train_dataset = train_dataset.select(range(max_examples))

        # ---- Load validation dataset ----
        val_dataset = _load_or_pull_dataset(val_ds_name, val_split, catalog, val_entry)
        # val_dataset can be None — training will proceed without validation

        # Harbor-sourced rows carry ``task_path`` pointing at each harbor
        # task dir. Wrap rows as Task objects rooted at that path so the
        # AgentFlowEngine + SandboxTaskHooks can resolve per-task
        # verifiers (``tests/test.sh``) and per-task ``task.toml``.
        # Train and val may be different harbor datasets — wrap
        # independently based on each one's resolved catalog entry.
        from rllm.data.dataset import Dataset as _Dataset
        from rllm.data.dataset import _wrap_rows_as_tasks

        if train_entry and train_entry.get("source", "").startswith("harbor:"):
            train_dataset = _Dataset(data=_wrap_rows_as_tasks(list(train_dataset.data)), name=train_ds_name, split=train_split)
        if val_dataset is not None and val_entry and val_entry.get("source", "").startswith("harbor:"):
            val_dataset = _Dataset(data=_wrap_rows_as_tasks(list(val_dataset.data)), name=val_ds_name, split=val_split)

    # ---- Build config ----
    config = build_train_config(
        model_name=model,
        group_size=group_size,
        batch_size=batch_size,
        lr=lr,
        lora_rank=lora_rank,
        total_epochs=total_epochs,
        total_steps=total_steps,
        val_freq=val_freq,
        save_freq=save_freq,
        project=project,
        experiment=experiment,
        output_dir=output_dir,
        config_file=config_file,
    )

    # ---- Wire UI logging ----
    if enable_ui:
        if not os.environ.get("RLLM_UI_URL"):
            os.environ["RLLM_UI_URL"] = "https://ui.rllm-project.com"
        from omegaconf import OmegaConf

        loggers = list(config.rllm.trainer.logger)
        if "ui" not in loggers:
            loggers.append("ui")
        config = OmegaConf.merge(
            config,
            OmegaConf.create(
                {
                    "rllm": {"trainer": {"logger": loggers}},
                }
            ),
        )

    # ---- Display header ----
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="label", width=14)
    table.add_column()
    table.add_row("Benchmark", f"[val]{benchmark}[/]")
    table.add_row("Model", f"[val]{model}[/]")
    table.add_row("Agent", f"[val]{agent_name}[/]")
    table.add_row("Evaluator", f"[dim]{evaluator_display}[/]")
    table.add_row("Train data", f"[val]{train_ds_name}[/]  [dim]({train_split}, {len(train_dataset)} examples)[/]")
    val_info = f"[val]{val_ds_name}[/]  [dim]({val_split}, {len(val_dataset)} examples)[/]" if val_dataset else "[dim]None[/]"
    table.add_row("Val data", val_info)
    table.add_row("Group size", f"[dim]{group_size}[/]")
    table.add_row("Batch size", f"[dim]{batch_size}[/]")
    table.add_row("Learning rate", f"[dim]{lr}[/]")
    table.add_row("LoRA rank", f"[dim]{lora_rank}[/]")
    epochs_str = f"[dim]{total_epochs}[/]"
    if total_steps is not None:
        epochs_str += f"  [dim](max {total_steps} steps)[/]"
    table.add_row("Epochs", epochs_str)
    if enable_ui:
        table.add_row("Live UI", f"[val]{os.environ['RLLM_UI_URL']}[/]")
    console.print()
    console.print(Panel(table, title="[bold]rLLM Train[/]", border_style="cyan", expand=False))
    console.print()

    # ---- Launch training ----
    # AgentTrainer auto-detects sandbox flows (SandboxedAgentFlow agent
    # or harbor-sourced task dataset) and wires SandboxTaskHooks +
    # gateway loopback. Pass evaluator only when the catalog binds a
    # global one (math_reward_fn / harbor_reward_fn for harbor scaffolds);
    # for harbor-on-rllm-native-harness, evaluator is None and hooks
    # resolve a per-task verifier from each task's tests/test.sh.
    trainer = AgentTrainer(
        backend="tinker",
        agent_flow=agent_flow,
        evaluator=evaluator,
        sandbox_backend=sandbox_backend,
        sandbox_concurrency=sandbox_concurrency,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


def _resolve_dataset_entry(name: str, catalog: dict, benchmark: str | None = None, benchmark_entry: dict | None = None) -> dict | None:
    """Resolve a dataset name to a catalog entry.

    Order: (1) the bundled ``catalog["datasets"]`` dict; (2) the
    benchmark's already-resolved entry when ``name == benchmark``;
    (3) a Harbor registry probe for ``<name>`` (handles
    ``--val-dataset swebench-verified`` where the user pulled it
    earlier and just wants the eval_split looked up).
    """
    entry = catalog.get("datasets", {}).get(name)
    if entry is not None:
        return entry
    if benchmark is not None and name == benchmark and benchmark_entry is not None:
        return benchmark_entry
    try:
        from rllm.cli._pull import resolve_harbor_catalog_entry

        return resolve_harbor_catalog_entry(name)
    except Exception:
        return None


def _load_or_pull_dataset(name: str, split: str, catalog: dict, catalog_entry_override: dict | None = None):
    """Load a dataset, auto-pulling from HuggingFace/Harbor if not cached.

    ``catalog_entry_override`` lets harbor entries resolved at runtime
    (via :func:`resolve_harbor_catalog_entry`) drive the pull, since those
    entries aren't written into ``catalog["datasets"]``.
    """
    from rich.status import Status

    from rllm.data import DatasetRegistry

    dataset = DatasetRegistry.load_dataset(name, split)
    if dataset is None:
        catalog_entry = catalog_entry_override or catalog.get("datasets", {}).get(name)
        if catalog_entry:
            with Status(f"[dim]Pulling {name} from {catalog_entry['source']}...[/]", console=console):
                pull_dataset(name, catalog_entry)
            dataset = DatasetRegistry.load_dataset(name, split)
    return dataset


# ---------------------------------------------------------------------------
# 3. train_cmd  — Click command
# ---------------------------------------------------------------------------


@click.command("train")
@click.argument("benchmark")
# Dataset options
@click.option("--train-dataset", default=None, help="Training dataset name (default: same as <benchmark>).")
@click.option("--train-split", default=None, help="Training split (default: catalog train_split, then 'train' if available, else eval_split).")
@click.option("--val-dataset", default=None, help="Validation dataset name (default: same as <benchmark>).")
@click.option("--val-split", default=None, help="Validation split (default: catalog eval_split).")
@click.option("--max-examples", default=None, type=int, help="Limit training examples.")
# Agent/evaluator options
@click.option("--agent", "agent_name", default=None, help="Agent flow: registry name or module:object path.")
@click.option("--evaluator", "evaluator_name", default=None, help="Evaluator: registry name or module:class path.")
# Model/training options
@click.option("--model", default="Qwen/Qwen3-8B", help="Model name/path (default: Qwen/Qwen3-8B).")
@click.option("--group-size", default=8, type=int, help="Rollouts per prompt for GRPO (default: 8).")
@click.option("--batch-size", default=32, type=int, help="Training batch size (default: 32).")
@click.option("--lr", default=2e-5, type=float, help="Learning rate (default: 2e-5).")
@click.option("--lora-rank", default=32, type=int, help="LoRA rank (default: 32).")
@click.option("--epochs", "total_epochs", default=1, type=int, help="Total training epochs (default: 1).")
@click.option("--max-steps", "total_steps", default=None, type=int, help="Stop after N steps (overrides --epochs).")
@click.option("--val-freq", default=5, type=int, help="Validate every N steps (default: 5).")
@click.option("--save-freq", default=20, type=int, help="Checkpoint every N steps (default: 20).")
# Output/config options
@click.option("--project", default="rllm-train", help="Project name for logging (default: rllm-train).")
@click.option("--experiment", default=None, help="Experiment name (default: <benchmark>).")
@click.option("--output", "output_dir", default=None, help="Checkpoint directory.")
@click.option("--config", "config_file", default=None, type=click.Path(exists=True), help="YAML config file merged on top of base templates. CLI flags override it.")
# UI logging options
@click.option("--ui/--no-ui", "enable_ui", default=None, help="Enable/disable live UI logging. Default: auto-enabled when logged in (see 'rllm login').")
# Sandbox options (sandboxed/harbor agents only — no-op for non-sandbox flows)
@click.option(
    "--sandbox-backend",
    "sandbox_backend",
    default=None,
    type=click.Choice(["docker", "local", "modal", "daytona", "e2b", "runloop", "gke", "apple-container"], case_sensitive=False),
    help="Sandbox backend for SandboxedAgentFlow harnesses (default: per-task or docker). Remote backends auto-spawn a cloudflared tunnel for the gateway.",
)
@click.option("--sandbox-concurrency", "sandbox_concurrency", default=None, type=int, help="Override max concurrent sandboxes (default: agent's max_concurrent — usually 4).")
def train_cmd(
    benchmark: str,
    train_dataset: str | None,
    train_split: str | None,
    val_dataset: str | None,
    val_split: str | None,
    max_examples: int | None,
    agent_name: str | None,
    evaluator_name: str | None,
    model: str,
    group_size: int,
    batch_size: int,
    lr: float,
    lora_rank: int,
    total_epochs: int,
    total_steps: int | None,
    val_freq: int,
    save_freq: int,
    project: str,
    experiment: str | None,
    output_dir: str | None,
    config_file: str | None,
    enable_ui: bool | None,
    sandbox_backend: str | None,
    sandbox_concurrency: int | None,
):
    """Train a model on a benchmark dataset using RL."""
    # Auto-detect UI logging: enable if user is logged in (has ui_api_key or RLLM_API_KEY)
    _ui_explicit = enable_ui is not None
    if enable_ui is None:
        from rllm.eval.config import load_ui_config

        ui_config = load_ui_config()
        enable_ui = bool(os.environ.get("RLLM_API_KEY") or ui_config.get("ui_api_key"))

    if not enable_ui and not _ui_explicit:
        console.print("  [blue]Tip: Try rllm UI for live monitoring! Run [bold]rllm login[/bold] to get started.[/]")

    if experiment is None:
        experiment = benchmark

    _run_train(
        benchmark=benchmark,
        agent_name=agent_name,
        evaluator_name=evaluator_name,
        model=model,
        train_dataset_name=train_dataset,
        train_split=train_split,
        val_dataset_name=val_dataset,
        val_split=val_split,
        max_examples=max_examples,
        group_size=group_size,
        batch_size=batch_size,
        lr=lr,
        lora_rank=lora_rank,
        total_epochs=total_epochs,
        total_steps=total_steps,
        val_freq=val_freq,
        save_freq=save_freq,
        project=project,
        experiment=experiment,
        output_dir=output_dir,
        config_file=config_file,
        enable_ui=enable_ui,
        sandbox_backend=sandbox_backend,
        sandbox_concurrency=sandbox_concurrency,
    )
