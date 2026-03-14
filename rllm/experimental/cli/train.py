"""Train CLI command.

``rllm train <benchmark> --model <name> [OPTIONS]``

Reuses the eval framework's dataset catalog, AgentFlows, and Evaluators to run
RL training via the Tinker backend.  Uses ``AgentTrainer(backend="tinker",
agent_flow=..., evaluator=...)`` with the AgentFlow + Evaluator path.
"""

from __future__ import annotations

import os
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

from rllm.experimental.cli._pull import load_dataset_catalog, pull_dataset

theme = Theme({"label": "dim", "success": "bold green", "error": "bold red", "val": "bold", "key": "yellow"})
console = Console(theme=theme)

# Path to the bundled YAML config templates
_CONFIG_PKG = Path(__file__).resolve().parent.parent / "config"


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
                # model_name is read by SdkWorkflowFactory to register
                # the model in the LiteLLM proxy
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
    train_split: str,
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
):
    """Core training logic: resolve catalog, load data, build config, launch trainer."""

    try:
        from rllm.experimental.eval.agent_loader import load_agent
        from rllm.experimental.eval.evaluator_loader import load_evaluator, resolve_evaluator_from_catalog
        from rllm.experimental.unified_trainer import AgentTrainer
    except ImportError as e:
        console.print(f"  [error]Missing training dependencies: {e}[/]")
        console.print("  Install with: [bold]pip install rllm\\[train][/]")
        raise SystemExit(1) from None

    # ---- Load catalog ----
    catalog = load_dataset_catalog()
    catalog_entry = catalog.get("datasets", {}).get(benchmark)

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

    # ---- Resolve evaluator ----
    evaluator = None
    evaluator_display = "N/A"
    if evaluator_name is not None:
        try:
            evaluator = load_evaluator(evaluator_name)
            evaluator_display = evaluator_name
        except (KeyError, ImportError, AttributeError, TypeError) as e:
            console.print(f"  [error]Error loading evaluator '{evaluator_name}': {e}[/]")
            raise SystemExit(1) from None
    else:
        evaluator = resolve_evaluator_from_catalog(benchmark)
        if evaluator is not None:
            reward_fn_name = catalog_entry.get("reward_fn", "") if catalog_entry else ""
            evaluator_display = reward_fn_name or type(evaluator).__name__

    if evaluator is None:
        console.print(f"  [error]No evaluator found for '{benchmark}'. Specify --evaluator explicitly.[/]")
        raise SystemExit(1)

    # ---- Resolve dataset names ----
    train_ds_name = train_dataset_name or benchmark
    val_ds_name = val_dataset_name or benchmark

    # Resolve val split from catalog if not provided
    if val_split is None:
        val_catalog_entry = catalog.get("datasets", {}).get(val_ds_name)
        val_split = val_catalog_entry.get("eval_split", "test") if val_catalog_entry else "test"

    # ---- Load training dataset ----
    train_dataset = _load_or_pull_dataset(train_ds_name, train_split, catalog)
    if train_dataset is None:
        console.print(f"  [error]Could not load training dataset '{train_ds_name}' split '{train_split}'.[/]")
        raise SystemExit(1)

    if max_examples is not None and max_examples < len(train_dataset):
        train_dataset = train_dataset.select(range(max_examples))

    # ---- Load validation dataset ----
    val_dataset = _load_or_pull_dataset(val_ds_name, val_split, catalog)
    # val_dataset can be None — training will proceed without validation

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
    trainer = AgentTrainer(
        backend="tinker",
        agent_flow=agent_flow,
        evaluator=evaluator,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


def _load_or_pull_dataset(name: str, split: str, catalog: dict):
    """Load a dataset, auto-pulling from HuggingFace if not cached."""
    from rich.status import Status

    from rllm.data import DatasetRegistry

    dataset = DatasetRegistry.load_dataset(name, split)
    if dataset is None:
        catalog_entry = catalog.get("datasets", {}).get(name)
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
@click.option("--train-split", default="train", help="Training split (default: train).")
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
def train_cmd(
    benchmark: str,
    train_dataset: str | None,
    train_split: str,
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
):
    """Train a model on a benchmark dataset using RL."""
    # Auto-detect UI logging: enable if user is logged in (has ui_api_key or RLLM_API_KEY)
    _ui_explicit = enable_ui is not None
    if enable_ui is None:
        from rllm.experimental.eval.config import load_ui_config

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
    )
