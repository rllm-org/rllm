"""SFT CLI command.

``rllm sft [DATASET] --model <name> [--backend tinker|verl|fireworks] [OPTIONS]``

Fine-tunes a model on a conversation (``messages``) dataset. Resolves a
backend-agnostic :class:`~rllm.trainer.sft.spec.SFTSpec` from the flags and hands
it to :class:`~rllm.trainer.agent_sft_trainer.AgentSFTTrainer`, which dispatches
to the chosen backend. Pairs with ``rllm dataset from-eval`` (curated SFT data).
"""

from __future__ import annotations

import os
from pathlib import Path

import click

from rllm.cli._ui import console, fail


@click.command("sft")
@click.argument("dataset", required=False, default=None)
# Data source
@click.option("--train-file", default=None, type=click.Path(exists=True), help="Training data file (parquet/jsonl/json) instead of a registered DATASET.")
@click.option("--val-file", default=None, type=click.Path(exists=True), help="Validation data file.")
@click.option("--train-split", default="train", help="Training split for a registered DATASET (default: train).")
@click.option("--val-split", default=None, help="Validation split for a registered DATASET (default: try 'test').")
@click.option("--max-examples", default=None, type=int, help="Limit number of training examples.")
# Model / backend
@click.option("--model", default="Qwen/Qwen3.5-4B", help="Model name/path (default: Qwen/Qwen3.5-4B).")
@click.option("--backend", default="tinker", type=click.Choice(["tinker", "verl", "fireworks"]), help="SFT backend (default: tinker).")
@click.option("--gpus", default=1, type=int, help="GPUs per node for the distributed (verl) backend's torchrun launcher (default: 1).")
@click.option("--renderer", default=None, help="tinker/fireworks renderer name: qwen3 | qwen3_5 | deepseekv3 | llama3 | role_colon (default: auto-detect from the model).")
@click.option("--lora-rank", default=32, type=int, help="LoRA rank; 0 = full fine-tuning (default: 32).")
# Hyperparameters
@click.option("--lr", default=1e-5, type=float, help="Learning rate (default: 1e-5).")
@click.option("--batch-size", default=32, type=int, help="Training batch size (default: 32).")
@click.option("--epochs", "epochs", default=1, type=int, help="Total training epochs (default: 1).")
@click.option("--max-length", default=2048, type=int, help="Max sequence length (default: 2048).")
@click.option("--tokenize-method", default="cumulative", type=click.Choice(["cumulative", "stepwise", "hf_template"]), help="Tokenization/masking method (default: cumulative).")
@click.option("--lr-schedule", default="constant", type=click.Choice(["constant", "linear", "cosine"]), help="LR schedule (default: constant).")
# Logging / checkpoints
@click.option("--val-freq", default=10, type=int, help="Validate every N steps (default: 10).")
@click.option("--save-freq", default=20, type=int, help="Checkpoint every N steps (default: 20).")
@click.option("--project", default="rllm-sft", help="Project name for logging (default: rllm-sft).")
@click.option("--experiment", default=None, help="Experiment name (default: dataset name).")
@click.option(
    "--logger",
    "loggers",
    multiple=True,
    type=click.Choice(["console", "wandb", "mlflow", "swanlab", "tensorboard", "file", "ui"]),
    help="Tracking backend(s) for training metrics via rllm.utils.tracking (repeatable). 'console' is always on; e.g. --logger wandb (needs WANDB_API_KEY / wandb login).",
)
@click.option("--ui/--no-ui", "enable_ui", default=None, help="Enable/disable live rLLM UI logging. Default: auto-enabled when logged in (see 'rllm login'). Not supported on the verl backend.")
@click.option("--output", "output_dir", default=None, help="Checkpoint directory.")
@click.option(
    "--config",
    "config_file",
    default=None,
    type=click.Path(exists=True),
    help="YAML escape hatch merged ON TOP of the backend config: file keys beat equivalent CLI flags (--renderer/--gpus still win). For backend knobs without a flag.",
)
@click.option("--preflight", is_flag=True, help="Render and validate the dataset, then exit without starting training.")
def sft_cmd(
    dataset: str | None,
    train_file: str | None,
    val_file: str | None,
    train_split: str,
    val_split: str | None,
    max_examples: int | None,
    model: str,
    backend: str,
    gpus: int,
    renderer: str | None,
    lora_rank: int,
    lr: float,
    batch_size: int,
    epochs: int,
    max_length: int,
    tokenize_method: str,
    lr_schedule: str,
    val_freq: int,
    save_freq: int,
    project: str,
    experiment: str | None,
    loggers: tuple[str, ...],
    enable_ui: bool | None,
    output_dir: str | None,
    config_file: str | None,
    preflight: bool,
):
    """Fine-tune a model with supervised learning (SFT).

    Provide either a registered DATASET name or --train-file.

    \b
    Examples:
      rllm sft math500-rft --model Qwen/Qwen3.5-4B --backend tinker --epochs 3
      rllm sft --train-file data.parquet --lr 1e-5
    """
    from rllm.cli._ui import info_panel
    from rllm.data import Dataset, DatasetRegistry
    from rllm.trainer.agent_sft_trainer import AgentSFTTrainer
    from rllm.trainer.sft import SFTSpec
    from rllm.trainer.sft.backend import SFTConfigError

    # ---- resolve datasets ----
    train_dataset = None
    val_dataset = None
    source_label = ""

    if train_file:
        try:
            train_dataset = Dataset.load_data(train_file)
        except Exception as e:
            fail(f"Failed to load training file '{train_file}': {e}")
        source_label = Path(train_file).name
        if val_file:
            try:
                val_dataset = Dataset.load_data(val_file)
            except Exception as e:
                fail(f"Failed to load validation file '{val_file}': {e}")
    elif dataset:
        train_dataset = DatasetRegistry.load_dataset(dataset, train_split)
        if train_dataset is None:
            fail(f"Could not load dataset '{dataset}' split '{train_split}'. Try 'rllm dataset list --local'.")
        source_label = f"{dataset} ({train_split})"
        # Validation: explicit split, else best-effort 'test'.
        want_val = val_split or "test"
        val_dataset = DatasetRegistry.load_dataset(dataset, want_val)
        if val_dataset is None and val_split:
            fail(f"Could not load validation split '{val_split}' for '{dataset}'.")
    else:
        fail("Provide a registered DATASET name or --train-file. See 'rllm sft --help'.")

    if max_examples is not None and max_examples < len(train_dataset):
        train_dataset = train_dataset.select(range(max_examples))

    if experiment is None:
        experiment = dataset or (Path(train_file).stem if train_file else "sft")

    # Resolve UI logging (mirrors `rllm train`): auto-enable when logged in
    # (RLLM_API_KEY or a saved ui_api_key) unless --ui/--no-ui is explicit.
    _ui_explicit = enable_ui is not None
    if enable_ui is None:
        from rllm.eval.config import load_ui_config

        enable_ui = bool(os.environ.get("RLLM_API_KEY") or load_ui_config().get("ui_api_key"))
    if enable_ui and not os.environ.get("RLLM_UI_URL"):
        os.environ["RLLM_UI_URL"] = "https://ui.rllm-project.com"
    if not enable_ui and not _ui_explicit:
        console.print("  [blue]Tip: Try rllm UI for live monitoring! Run [bold]rllm login[/bold] to get started.[/]")

    # Resolve tracking backends: start from console, add any --logger values, and
    # append 'ui' when UI logging is on; dedupe preserving order. If the user asked
    # for nothing and UI is off, leave logger=None so the backend yaml default rules.
    resolved_logger: list[str] | None = None
    if loggers or enable_ui:
        seen: set[str] = set()
        resolved_logger = []
        for name in ["console", *loggers, *(["ui"] if enable_ui else [])]:
            if name not in seen:
                seen.add(name)
                resolved_logger.append(name)

    # Backend-specific spec overrides, lowest → highest precedence:
    #   - --config YAML: the escape hatch for backend-native knobs (see its help);
    #   - verl: route --gpus to trainer.n_gpus_per_node, but only an explicitly
    #     passed --gpus, so the flag's default doesn't override a --config value;
    #   - tinker/fireworks: route --renderer to data.renderer_name (omitted =>
    #     backend auto-detects from the model).
    from click.core import ParameterSource
    from omegaconf import OmegaConf

    override_layers: list[dict] = []
    if config_file:
        loaded = OmegaConf.to_container(OmegaConf.load(config_file), resolve=False)
        if not isinstance(loaded, dict):
            fail(f"--config {config_file} must be a YAML mapping.")
        override_layers.append(loaded)
    ctx = click.get_current_context(silent=True)
    gpus_explicit = ctx is not None and ctx.get_parameter_source("gpus") == ParameterSource.COMMANDLINE
    if backend == "verl" and (gpus_explicit or not config_file):
        override_layers.append({"trainer": {"n_gpus_per_node": gpus}})
    elif backend != "verl" and renderer:
        override_layers.append({"data": {"renderer_name": renderer}})
    overrides = None
    if override_layers:
        overrides = OmegaConf.to_container(OmegaConf.merge(*[OmegaConf.create(layer) for layer in override_layers]), resolve=False)

    spec = SFTSpec(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        lr=lr,
        lr_schedule=lr_schedule,
        epochs=epochs,
        batch_size=batch_size,
        max_length=max_length,
        tokenize_method=tokenize_method,
        lora_rank=lora_rank,
        save_freq=save_freq,
        val_freq=val_freq,
        project=project,
        experiment=experiment,
        logger=resolved_logger,
        output_dir=output_dir,
        overrides=overrides,
    )

    # Build + configure the backend up front (local, no provisioning) so the
    # summary reflects the *resolved* model — e.g. Fireworks maps to its FW
    # model path + HF tokenizer rather than the bare --model default.
    trainer = AgentSFTTrainer(spec, backend=backend)
    try:
        be = trainer.prepare()
    except SFTConfigError as e:
        fail(str(e))
    cfg = be.config
    # tinker/fireworks expose model.name; verl uses model.path. Fall back to the
    # CLI --model for either.
    resolved_model = cfg.model.get("name") or cfg.model.get("path") or model
    tokenizer_model = cfg.model.get("tokenizer_model")

    rows = [
        ("Model", f"[val]{resolved_model}[/]"),
    ]
    if tokenizer_model and tokenizer_model != resolved_model:
        rows.append(("Tokenizer", f"[dim]{tokenizer_model}[/]"))
    # verl's effective GPU count comes from the merged trainer.n_gpus_per_node
    # (a --config value can override the --gpus flag).
    resolved_gpus = (cfg.get("trainer", {}).get("n_gpus_per_node") or gpus) if backend == "verl" else gpus
    rows += [
        ("Backend", f"[val]{backend}[/]" + (f"  [dim]({resolved_gpus} GPU{'s' if resolved_gpus != 1 else ''}, torchrun)[/]" if backend == "verl" else "")),
        ("Train data", f"[val]{source_label}[/]  [dim]({len(train_dataset)} examples)[/]"),
        ("Val data", f"[dim]{len(val_dataset)} examples[/]" if val_dataset else "[dim]none[/]"),
        ("LoRA rank", f"[dim]{lora_rank}[/]"),
        ("LR / schedule", f"[dim]{lr} / {lr_schedule}[/]"),
        ("Batch / epochs", f"[dim]{batch_size} / {epochs}[/]"),
        ("Max length", f"[dim]{max_length}[/]"),
        ("Tokenize", f"[dim]{tokenize_method}[/]"),
        ("Logging", f"[dim]{', '.join(resolved_logger) if resolved_logger else 'console (yaml default)'}[/]"),
    ]
    # Renderer applies to the tinker/fireworks render path only (verl tokenizes
    # via its own dataset). null/omitted => the backend auto-detects it.
    if backend != "verl":
        rows.append(("Renderer", f"[dim]{renderer or 'auto-detect'}[/]"))
    console.print()
    console.print(info_panel(rows, title="[bold]rLLM SFT[/]", border="brand"))
    console.print()

    try:
        if preflight:
            trainer.preflight()
            console.print("[green]Preflight passed.[/green]")
            return
        trainer.train()
    except SFTConfigError as e:
        fail(str(e))
    except ImportError as e:
        fail(f"Missing training dependencies for backend '{backend}': {e}\n  Install with: pip install rllm[train]")
