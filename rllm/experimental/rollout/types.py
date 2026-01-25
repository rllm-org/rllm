"""
Type alias for TokenOutput and TokenInput -- need to take different backends into account.
"""

from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.image_processing_utils import BaseImageProcessor

    Tokenizer: TypeAlias = PreTrainedTokenizer
    Processor: TypeAlias = ProcessorMixin
    ImageProcessor: TypeAlias = BaseImageProcessor
else:
    # make it importable from other files as a type in runtime
    Tokenizer: TypeAlias = Any
    Processor: TypeAlias = Any
    ImageProcessor: TypeAlias = Any

# Tinker types. See https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/rl/data_processing.py
# for the rationale behind "FlatObElem" and "FlatOb" types.
try:
    from tinker.types import ModelInputChunk, SampledSequence

    TinkerFlatObElem: TypeAlias = ModelInputChunk | int
    TinkerTokenOutput: TypeAlias = SampledSequence
except ImportError:  # avoid cases when the tinker backend is not used
    TinkerFlatObElem: TypeAlias = Any
    TinkerTokenOutput: TypeAlias = Any

TinkerFlatOb: TypeAlias = list[TinkerFlatObElem]
TinkerTokenInput: TypeAlias = TinkerFlatOb

# Verl types
VerlTokenInput: TypeAlias = list[int]
try:
    from verl.workers.rollout.replica import TokenOutput

    VerlTokenOutput: TypeAlias = TokenOutput
except ImportError:  # avoid cases when the verl backend is not used
    VerlTokenOutput: TypeAlias = Any

# Union everything together
TokenInput: TypeAlias = TinkerTokenInput | VerlTokenInput
TokenOutput: TypeAlias = TinkerTokenOutput | VerlTokenOutput
