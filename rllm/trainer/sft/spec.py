"""Backend-agnostic SFT specification.

``SFTSpec`` is the single input to :class:`rllm.trainer.agent_sft_trainer.AgentSFTTrainer`
(the SFT dispatcher). The CLI (``rllm sft``) and the curation flow fill it; each
:class:`~rllm.trainer.sft.backend.SFTBackend` translates it into its own native
config via ``build_config()``. Nothing here is backend-specific.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rllm.data import Dataset

# Tokenization/masking strategies understood by the backends.
TOKENIZE_METHODS = ("cumulative", "stepwise", "hf_template")
LR_SCHEDULES = ("constant", "linear", "cosine")


@dataclass
class SFTSpec:
    """A normalized description of an SFT run.

    Attributes:
        model: model name/path to fine-tune.
        train_dataset: training data (a registered/loaded :class:`Dataset` with a
            ``messages`` column).
        val_dataset: optional validation data.
        overrides: deep-merged into the backend's native config as an escape
            hatch for backend-specific knobs not surfaced as fields.
    """

    model: str = "Qwen/Qwen3.5-4B"
    train_dataset: Dataset | None = None
    val_dataset: Dataset | None = None
    lr: float = 1e-5
    lr_schedule: str = "constant"
    epochs: int = 1
    batch_size: int = 32
    max_length: int = 2048
    tokenize_method: str = "cumulative"
    lora_rank: int = 32
    save_freq: int = 20
    val_freq: int = 10
    project: str = "rllm-sft"
    experiment: str | None = None
    output_dir: str | None = None
    overrides: dict | None = None
