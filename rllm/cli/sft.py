"""SFT CLI command.

``rllm sft [DATASET] --model <name> [--backend tinker|verl|fireworks] [OPTIONS]``

Fine-tunes a model on a conversation (``messages``) dataset. Resolves a
backend-agnostic :class:`~rllm.trainer.sft.spec.SFTSpec` from the flags and hands
it to :class:`~rllm.trainer.agent_sft_trainer.AgentSFTTrainer`, which dispatches
to the chosen backend. Pairs with ``rllm dataset from-eval`` (curated SFT data).
"""

from __future__ import annotations

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
@click.option("--lora-rank", default=32, type=int, help="LoRA rank (default: 32).")
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
@click.option("--output", "output_dir", default=None, help="Checkpoint directory.")
def sft_cmd(
    dataset: str | None,
    train_file: str | None,
    val_file: str | None,
    train_split: str,
    val_split: str | None,
    max_examples: int | None,
    model: str,
    backend: str,
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
    output_dir: str | None,
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
        output_dir=output_dir,
    )

    rows = [
        ("Model", f"[val]{model}[/]"),
        ("Backend", f"[val]{backend}[/]"),
        ("Train data", f"[val]{source_label}[/]  [dim]({len(train_dataset)} examples)[/]"),
        ("Val data", f"[dim]{len(val_dataset)} examples[/]" if val_dataset else "[dim]none[/]"),
        ("LoRA rank", f"[dim]{lora_rank}[/]"),
        ("LR / schedule", f"[dim]{lr} / {lr_schedule}[/]"),
        ("Batch / epochs", f"[dim]{batch_size} / {epochs}[/]"),
        ("Max length", f"[dim]{max_length}[/]"),
        ("Tokenize", f"[dim]{tokenize_method}[/]"),
    ]
    console.print()
    console.print(info_panel(rows, title="[bold]rLLM SFT[/]", border="brand"))
    console.print()

    try:
        AgentSFTTrainer(spec, backend=backend).train()
    except SFTConfigError as e:
        fail(str(e))
    except ImportError as e:
        fail(f"Missing training dependencies for backend '{backend}': {e}\n  Install with: pip install rllm[train]")
