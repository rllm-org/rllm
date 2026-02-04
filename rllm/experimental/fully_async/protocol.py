from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OutputChunk:
    response_ids: list[int]
    response_logprobs: list[float]
    version: int
    response_text: str = ""  # Decoded text for this chunk

    @property
    def num_output_tokens(self):
        return len(self.response_ids)


@dataclass
class OutputWithVersion:
    prompt_ids: list[int]
    output_chunks: list[OutputChunk]
    finish_reason: str = "Not Finish"
    prompt_text: str = ""  # Original prompt text (if available)

    @property
    def num_output_tokens(self):
        if not self.output_chunks:
            return 0
        return sum(len(o.response_ids) for o in self.output_chunks)

    @property
    def response_text(self) -> str:
        """Get concatenated response text from all chunks."""
        # Note: For continuation after abort, the text accumulates
        # We return the last chunk's text as it contains the full response
        if not self.output_chunks:
            return ""
        return self.output_chunks[-1].response_text

    @property
    def all_text(self) -> str:
        """Get full text including prompt (if available) and response."""
        return self.prompt_text + self.response_text

    def append(self, chunk: OutputChunk):
        self.output_chunks.append(chunk)

    def all_tokens(self):
        tokens = []
        tokens.extend(self.prompt_ids)
        for chunk in self.output_chunks:
            tokens.extend(chunk.response_ids)
        return tokens

    def all_response_ids(self) -> list[int]:
        """Get all response token IDs from all chunks."""
        ids = []
        for chunk in self.output_chunks:
            ids.extend(chunk.response_ids)
        return ids

    def to_sequence(self):
        response_ids = []
        response_logprobs = []
        for chunk in self.output_chunks:
            response_ids.extend(chunk.response_ids)
            response_logprobs.extend(chunk.response_logprobs)

        start_version = self.output_chunks[0].version if self.output_chunks else None
        end_version = self.output_chunks[-1].version if self.output_chunks else None

        return Sequence(
            prompt_ids=self.prompt_ids,
            response_ids=response_ids,
            response_logprobs=response_logprobs,
            response_masks=[1] * len(response_ids),
            start_version=start_version,
            end_version=end_version,
        )


@dataclass
class Sequence:
    prompt_ids: list[int]
    response_ids: list[int]
    response_logprobs: list[float]
    response_masks: list[int]

    # Auto Generate from .to_sequence()
    start_version: int | None = None
    end_version: int | None = None

    def is_prefix(self, other: "Sequence") -> bool:
        return self.input_ids == other.input_ids[: len(self.input_ids)]

    @property
    def input_ids(self) -> list[int]:
        return self.prompt_ids + self.response_ids

    @property
    def total_length(self) -> int:
        return len(self.input_ids)

    def merge(self, other: "Sequence") -> "Sequence":
        assert self.is_prefix(other), "You can only merge sequence is self is a prefix of other"

        p_len = len(self.prompt_ids)
        other_input_ids = other.input_ids

        other_p_len = len(other.prompt_ids)

        pad_len = other_p_len - self.total_length if other_p_len > self.total_length else 0
        start_idx = max(self.total_length - other_p_len, 0)

        response_ids = other_input_ids[p_len:]
        # Use 0.0 for logprobs padding (consistent with verl), 0 for mask padding
        response_logprobs = self.response_logprobs + [0.0] * pad_len + other.response_logprobs[start_idx:]
        response_masks = self.response_masks + [0] * pad_len + other.response_masks[start_idx:]

        # Preserve version information: start from self, end from other
        start_version = self.start_version if self.start_version is not None else other.start_version
        end_version = other.end_version if other.end_version is not None else self.end_version

        return Sequence(
            prompt_ids=self.prompt_ids,
            response_ids=response_ids,
            response_logprobs=response_logprobs,
            response_masks=response_masks,
            start_version=start_version,
            end_version=end_version,
        )

    def resize_prompt_length(self, max_prompt_length: int) -> "Sequence":
        if len(self.prompt_ids) <= max_prompt_length:
            return self

        new_prompt_ids = self.prompt_ids[:max_prompt_length]
        new_response_ids = self.prompt_ids[max_prompt_length:] + self.response_ids
        new_response_logprobs = [0.0] * (len(self.prompt_ids) - max_prompt_length) + self.response_logprobs
        new_response_masks = [0] * (len(self.prompt_ids) - max_prompt_length) + self.response_masks
        return Sequence(
            prompt_ids=new_prompt_ids,
            response_ids=new_response_ids,
            response_logprobs=new_response_logprobs,
            response_masks=new_response_masks,
            start_version=self.start_version,
            end_version=self.end_version,
        )


@dataclass
class Trajectory:
    sequences: list[Sequence]
    reward: float = 0.0
    metadata: dict | None = None

    def append(self, sequence):
        self.sequences.append(sequence)

    def merge(self):
        merged_sequences = []
        cur_seq = None
        for seq in self.sequences:
            if not cur_seq:
                cur_seq = seq
            elif cur_seq.is_prefix(seq):
                cur_seq = cur_seq.merge(seq)
            else:
                merged_sequences.append(cur_seq)
                cur_seq = seq

        if cur_seq:
            merged_sequences.append(cur_seq)
        return merged_sequences


@dataclass
class TrajectoryGroup:
    trajectories: list[Trajectory]