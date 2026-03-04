"""
Tests for trajectory_to_datums function in rllm.trainer.tinker.transform.

These tests verify that trajectories are correctly converted to Tinker Datum objects,
especially the logic for merging consecutive steps that share a prefix relationship.
"""

from dataclasses import dataclass
from typing import Literal
from unittest.mock import MagicMock

import pytest
import tinker
from tinker.types import ImageChunk

from rllm.agents.agent import Step, Trajectory
from rllm.trainer.tinker.transform import (
    _flatten_token_input,
    _is_prefix,
    trajectory_to_datums,
)

# =============================================================================
# Mock Types
# =============================================================================


@dataclass
class MockSampledSequence:
    """Mock SampledSequence that mimics tinker's SampledSequence interface."""

    tokens: list[int]
    logprobs: list[float]

    def __post_init__(self):
        assert len(self.tokens) == len(self.logprobs), "tokens and logprobs must have same length"


# =============================================================================
# Helper Functions
# =============================================================================


def make_step(
    prompt_ids: list,
    response_tokens: list[int],
    response_logprobs: list[float] | None = None,
    advantage: float | list[float] = 1.0,
) -> Step:
    """Helper to create a Step with mock TinkerTokenOutput."""
    if response_logprobs is None:
        response_logprobs = [0.1] * len(response_tokens)

    mock_output = MockSampledSequence(tokens=response_tokens, logprobs=response_logprobs)

    step = Step(
        prompt_ids=prompt_ids,
        response_ids=mock_output,
        advantage=advantage,
    )
    return step


def make_image_chunk(expected_tokens: int = 256, data: bytes = b"image_data", format: Literal["png", "jpeg"] = "png") -> ImageChunk:
    """Helper to create a ImageChunk."""
    return ImageChunk(expected_tokens=expected_tokens, data=data, format=format)


def verify_datum_structure(datum: tinker.Datum):
    """Verify a Datum has the expected structure and consistent lengths."""
    # Check model_input exists
    assert hasattr(datum, "model_input")
    assert datum.model_input is not None

    # Check all required loss_fn_inputs fields exist
    loss_fn_inputs = datum.loss_fn_inputs
    assert "target_tokens" in loss_fn_inputs
    assert "logprobs" in loss_fn_inputs
    assert "advantages" in loss_fn_inputs
    assert "mask" in loss_fn_inputs

    # Check dtypes
    assert loss_fn_inputs["target_tokens"].dtype == "int64"
    assert loss_fn_inputs["logprobs"].dtype == "float32"
    assert loss_fn_inputs["advantages"].dtype == "float32"
    assert loss_fn_inputs["mask"].dtype == "float32"

    # Check lengths are consistent
    target_len = len(loss_fn_inputs["target_tokens"].data)
    logprobs_len = len(loss_fn_inputs["logprobs"].data)
    advantages_len = len(loss_fn_inputs["advantages"].data)
    mask_len = len(loss_fn_inputs["mask"].data)
    assert target_len == logprobs_len == advantages_len == mask_len, f"Length mismatch: target={target_len}, logprobs={logprobs_len}, advantages={advantages_len}, mask={mask_len}"


# =============================================================================
# Tests for _is_prefix
# =============================================================================


class TestIsPrefix:
    """Tests for the _is_prefix helper function."""

    def test_basic_prefix_cases(self):
        """Test basic prefix relationships with plain int lists."""
        # Empty is prefix of anything
        assert _is_prefix([], [1, 2, 3])
        assert _is_prefix([], [])

        # Identical sequences
        assert _is_prefix([1, 2, 3], [1, 2, 3])

        # Proper prefix
        assert _is_prefix([1, 2], [1, 2, 3, 4])

        # Not a prefix: different elements
        assert not _is_prefix([1, 3], [1, 2, 3])

        # Not a prefix: longer sequence
        assert not _is_prefix([1, 2, 3, 4], [1, 2])

    def test_prefix_with_encoded_text_chunks(self):
        """Test prefix checking with EncodedTextChunk in sequences."""
        chunk1 = tinker.EncodedTextChunk(tokens=[10, 20, 30])
        chunk2 = tinker.EncodedTextChunk(tokens=[10, 20, 30])  # Same content, different object

        # Same chunk object should match
        assert _is_prefix([chunk1, 1, 2], [chunk1, 1, 2, 3, 4])

        # Different chunk objects with same content won't match via object equality
        # (unless tinker implements __eq__ for EncodedTextChunk)
        seq1 = [chunk1, 1, 2]
        seq2 = [chunk2, 1, 2, 3]
        # This depends on how tinker implements equality; test both cases
        result = _is_prefix(seq1, seq2)
        # Just verify it returns a boolean without error
        assert isinstance(result, bool)

    def test_prefix_with_image_chunks(self):
        """Test prefix checking with ImageChunk (non-text multimodal chunk)."""
        img_chunk = make_image_chunk()

        # Mixed sequence with image chunk
        seq1 = [1, 2, img_chunk, 3]
        seq2 = [1, 2, img_chunk, 3, 4, 5]
        assert _is_prefix(seq1, seq2)

        # Different image chunk (different data) should not match
        img_chunk2 = make_image_chunk(data=b"image_data_2")
        seq3 = [1, 2, img_chunk2, 3, 4, 5]
        assert not _is_prefix(seq1, seq3)

    def test_prefix_with_mixed_chunk_types(self):
        """Test prefix with both EncodedTextChunk and ImageChunk."""
        text_chunk = tinker.EncodedTextChunk(tokens=[10, 20])
        img_chunk = make_image_chunk(expected_tokens=128)

        seq1 = [text_chunk, 1, img_chunk, 2]
        seq2 = [text_chunk, 1, img_chunk, 2, 3, 4]
        assert _is_prefix(seq1, seq2)


# =============================================================================
# Tests for _flatten_token_input
# =============================================================================


class TestFlattenTokenInput:
    """Tests for the _flatten_token_input helper function."""

    def test_already_flat(self):
        """Plain int list should remain unchanged."""
        token_input = [1, 2, 3, 4]
        assert _flatten_token_input(token_input) == [1, 2, 3, 4]

    def test_empty_input(self):
        """Empty input should return empty list."""
        assert _flatten_token_input([]) == []

    def test_encoded_text_chunk_is_flattened(self):
        """EncodedTextChunk should be expanded to its tokens."""
        chunk = tinker.EncodedTextChunk(tokens=[10, 20, 30])
        token_input = [1, 2, chunk, 3]
        flattened = _flatten_token_input(token_input)
        assert flattened == [1, 2, 10, 20, 30, 3]

    def test_multiple_encoded_text_chunks(self):
        """Multiple EncodedTextChunks should all be flattened."""
        chunk1 = tinker.EncodedTextChunk(tokens=[10, 20])
        chunk2 = tinker.EncodedTextChunk(tokens=[30, 40])
        token_input = [chunk1, 5, chunk2]
        flattened = _flatten_token_input(token_input)
        assert flattened == [10, 20, 5, 30, 40]

    def test_image_chunk_is_preserved(self):
        """
        ImageChunk (non-EncodedTextChunk) should NOT be flattened.
        It should be kept as-is in the sequence since it doesn't have .tokens.
        """
        img_chunk = make_image_chunk()
        token_input = [1, 2, img_chunk, 3]
        flattened = _flatten_token_input(token_input)
        # ImageChunk should be preserved, not flattened
        assert flattened == [1, 2, img_chunk, 3]

    def test_mixed_chunks_partial_flatten(self):
        """EncodedTextChunks are flattened, ImageChunks are preserved."""
        text_chunk = tinker.EncodedTextChunk(tokens=[10, 20])
        img_chunk = make_image_chunk(expected_tokens=128)

        token_input = [text_chunk, 1, img_chunk, 2]
        flattened = _flatten_token_input(token_input)
        # text_chunk is flattened to [10, 20], img_chunk is preserved
        assert flattened == [10, 20, 1, img_chunk, 2]


# =============================================================================
# Tests for trajectory_to_datums - Single Step
# =============================================================================


class TestTrajectoryToDataSingleStep:
    """Tests for trajectory_to_datums with single-step trajectories."""

    def test_single_step_comprehensive(self):
        """
        Comprehensive test for single-step trajectory.
        Verifies datum count, structure, mask, logprobs, and advantages.
        """
        step = make_step(
            prompt_ids=[1, 2, 3],  # 3 prompt tokens
            response_tokens=[4, 5],  # 2 response tokens
            response_logprobs=[-0.5, -0.8],
            advantage=0.5,
        )
        trajectory = Trajectory(steps=[step])

        datums = trajectory_to_datums(trajectory)

        # Should produce exactly one Datum
        assert len(datums) == 1
        datum = datums[0]

        # Verify structure and dtypes
        verify_datum_structure(datum)

        # Full sequence: [1, 2, 3, 4, 5] (length 5)
        # After [1:] shift, all arrays have length 4
        loss_fn_inputs = datum.loss_fn_inputs

        # Mask: prompt tokens get 0, response tokens get 1
        # Before shift: [0, 0, 0, 1, 1], After shift: [0, 0, 1, 1]
        assert loss_fn_inputs["mask"].data == [0.0, 0.0, 1.0, 1.0]

        # Logprobs: prompt tokens get 0, response tokens get actual logprobs
        # Before shift: [0, 0, 0, -0.5, -0.8], After shift: [0, 0, -0.5, -0.8]
        assert loss_fn_inputs["logprobs"].data == [0.0, 0.0, -0.5, -0.8]

        # Advantages: prompt tokens get 0, response tokens get 0.5 (scalar advantage)
        # Before shift: [0, 0, 0, 0.5, 0.5], After shift: [0, 0, 0.5, 0.5]
        assert loss_fn_inputs["advantages"].data == [0, 0, 0.5, 0.5]

    def test_single_step_with_per_token_advantages(self):
        """Test single step with per-token advantages (list instead of scalar)."""
        step = make_step(
            prompt_ids=[1, 2, 3],
            response_tokens=[4, 5, 6],
            response_logprobs=[-0.1, -0.2, -0.3],
            advantage=[0.5, 0.6, 0.7],  # Per-token advantages
        )
        trajectory = Trajectory(steps=[step])

        datums = trajectory_to_datums(trajectory)

        assert len(datums) == 1
        verify_datum_structure(datums[0])

        # Full sequence: [1, 2, 3, 4, 5, 6]
        # Advantages: [0, 0, 0, 0.5, 0.6, 0.7]
        # After [1:] shift: [0, 0, 0.5, 0.6, 0.7]
        advantages_data = datums[0].loss_fn_inputs["advantages"].data
        assert advantages_data == [0, 0, 0.5, 0.6, 0.7]


# =============================================================================
# Tests for trajectory_to_datums - Prefix Merging
# =============================================================================


class TestTrajectoryToDataPrefixMerging:
    """Tests for trajectory_to_datums when steps should be merged due to prefix relationship."""

    def test_two_steps_with_prefix_comprehensive(self):
        """
        Two steps where step2 extends step1's full sequence should merge into one Datum.
        Also verifies mask and advantages for merged steps.
        """
        # Step 1: prompt=[1,2], response=[3,4]
        step1 = make_step(
            prompt_ids=[1, 2],
            response_tokens=[3, 4],
            response_logprobs=[-0.1, -0.2],
            advantage=0.5,
        )
        # Step 2: prompt=[1,2,3,4,5] (extends [1,2,3,4]), response=[6,7]
        step2 = make_step(
            prompt_ids=[1, 2, 3, 4, 5],
            response_tokens=[6, 7],
            response_logprobs=[-0.3, -0.4],
            advantage=0.6,
        )
        trajectory = Trajectory(steps=[step1, step2])

        datums = trajectory_to_datums(trajectory)

        # Should produce single datum since step2 extends step1
        assert len(datums) == 1
        verify_datum_structure(datums[0])

        # Full sequence: [1, 2, 3, 4, 5, 6, 7]
        # Mask construction:
        #   step1: prompt [1,2] -> mask 0,0; response [3,4] -> mask 1,1
        #   step2: delta prompt [5] -> mask 0; response [6,7] -> mask 1,1
        # Full mask: [0, 0, 1, 1, 0, 1, 1], After [1:]: [0, 1, 1, 0, 1, 1]
        mask_data = datums[0].loss_fn_inputs["mask"].data
        assert mask_data == [0.0, 1.0, 1.0, 0.0, 1.0, 1.0]

    def test_three_steps_all_prefixes(self):
        """Three steps where each extends the previous should produce one Datum."""
        step1 = make_step(prompt_ids=[1, 2], response_tokens=[3, 4], advantage=0.5)
        step2 = make_step(prompt_ids=[1, 2, 3, 4, 5], response_tokens=[6, 7], advantage=0.6)
        step3 = make_step(prompt_ids=[1, 2, 3, 4, 5, 6, 7, 8], response_tokens=[9, 10], advantage=0.7)
        trajectory = Trajectory(steps=[step1, step2, step3])

        datums = trajectory_to_datums(trajectory)

        assert len(datums) == 1
        verify_datum_structure(datums[0])


# =============================================================================
# Tests for trajectory_to_datums - No Prefix (Multiple Datums)
# =============================================================================


class TestTrajectoryToDataNoPrefix:
    """Tests for trajectory_to_datums when steps don't share prefix relationship."""

    def test_two_independent_steps_creates_two_datums(self):
        """Steps without prefix relationship should produce separate Datums."""
        step1 = make_step(prompt_ids=[1, 2, 3], response_tokens=[4, 5], advantage=0.5)
        step2 = make_step(prompt_ids=[10, 11, 12], response_tokens=[13, 14], advantage=0.6)
        trajectory = Trajectory(steps=[step1, step2])

        datums = trajectory_to_datums(trajectory)

        assert len(datums) == 2
        for datum in datums:
            verify_datum_structure(datum)

    def test_mixed_prefix_and_non_prefix(self):
        """
        Test the example from docstring: (O1, A1), (O1+A1+O2, A2), (O3, A3).
        First two steps merge, third is separate.
        """
        step1 = make_step(prompt_ids=[1, 2], response_tokens=[3, 4], advantage=0.5)
        step2 = make_step(prompt_ids=[1, 2, 3, 4, 5], response_tokens=[6, 7], advantage=0.6)
        step3 = make_step(prompt_ids=[100, 101, 102], response_tokens=[103, 104], advantage=0.7)
        trajectory = Trajectory(steps=[step1, step2, step3])

        datums = trajectory_to_datums(trajectory)

        # First two steps merge into one Datum, third step is separate
        assert len(datums) == 2
        for datum in datums:
            verify_datum_structure(datum)


# =============================================================================
# Tests for trajectory_to_datums - With Chunks
# =============================================================================


class TestTrajectoryToDataWithChunks:
    """Tests for trajectory_to_datums with EncodedTextChunk and ImageChunk in prompt_ids."""

    def test_prompt_with_encoded_text_chunk(self):
        """Test that EncodedTextChunk in prompt_ids is flattened and processed correctly."""
        chunk = tinker.EncodedTextChunk(tokens=[1, 2, 3])
        step = Step(
            prompt_ids=[chunk, 4, 5],  # Flattens to [1, 2, 3, 4, 5]
            response_ids=MockSampledSequence(tokens=[6, 7], logprobs=[-0.1, -0.2]),
            advantage=0.5,
        )
        trajectory = Trajectory(steps=[step])

        datums = trajectory_to_datums(trajectory)

        assert len(datums) == 1
        verify_datum_structure(datums[0])

        # Flattened sequence: [1, 2, 3, 4, 5, 6, 7] (5 prompt + 2 response)
        # Mask after [1:] shift should have length 6
        mask_data = datums[0].loss_fn_inputs["mask"].data
        assert len(mask_data) == 6
        # First 4 are prompt (0), last 2 are response (1)
        assert mask_data == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]

    def test_prompt_with_image_chunk(self):
        """
        Test trajectory with ImageChunk in prompt_ids.
        ImageChunk is NOT flattened but its .length is used for token counting.
        """
        img_chunk = make_image_chunk(expected_tokens=3)  # Occupies 3 token positions
        step = Step(
            prompt_ids=[1, img_chunk, 2],  # 1 + 3 (image) + 1 = 5 token positions
            response_ids=MockSampledSequence(tokens=[10, 11], logprobs=[-0.1, -0.2]),
            advantage=0.5,
        )
        trajectory = Trajectory(steps=[step])

        datums = trajectory_to_datums(trajectory)

        assert len(datums) == 1
        verify_datum_structure(datums[0])

    def test_two_steps_with_shared_image_chunk_merge(self):
        """
        Two steps sharing the same ImageChunk (same data) in prefix should merge.
        This tests prefix detection with non-int elements.
        """
        img_chunk = make_image_chunk(expected_tokens=4, data=b"shared_image_data")

        # Step 1: [img_chunk, 1, 2] -> response [3, 4]
        step1 = make_step(
            prompt_ids=[img_chunk, 1, 2],
            response_tokens=[3, 4],
            advantage=0.5,
        )
        # Step 2: extends with [img_chunk, 1, 2, 3, 4, 5] -> response [6, 7]
        step2 = make_step(
            prompt_ids=[img_chunk, 1, 2, 3, 4, 5],  # Extends step1's full sequence
            response_tokens=[6, 7],
            advantage=0.6,
        )
        trajectory = Trajectory(steps=[step1, step2])

        datums = trajectory_to_datums(trajectory)

        # Should merge since step2 extends step1 (same img_chunk object)
        assert len(datums) == 1
        verify_datum_structure(datums[0])

    def test_two_steps_with_different_image_chunks_no_merge(self):
        """
        Two steps with different ImageChunk data should NOT merge,
        even if the image chunks have same expected_tokens.
        """
        img_chunk1 = make_image_chunk(expected_tokens=4, data=b"image_data_1")
        img_chunk2 = make_image_chunk(expected_tokens=4, data=b"image_data_2")

        step1 = make_step(
            prompt_ids=[img_chunk1, 1, 2],
            response_tokens=[3, 4],
            advantage=0.5,
        )
        step2 = make_step(
            prompt_ids=[img_chunk2, 1, 2, 3, 4],  # Different image chunk!
            response_tokens=[5, 6],
            advantage=0.6,
        )
        trajectory = Trajectory(steps=[step1, step2])

        datums = trajectory_to_datums(trajectory)

        # Should NOT merge - different image chunks mean no prefix relationship
        assert len(datums) == 2


# =============================================================================
# Tests for trajectory_to_datums - Edge Cases
# =============================================================================


class TestTrajectoryToDataEdgeCases:
    """Edge case tests for trajectory_to_datums."""

    def test_empty_trajectory(self):
        """Empty trajectory should return empty list."""
        trajectory = Trajectory(steps=[])
        datums = trajectory_to_datums(trajectory)
        assert datums == []

    def test_advantage_list_length_mismatch_raises(self):
        """If advantage list length doesn't match response tokens, should raise."""
        step = Step(
            prompt_ids=[1, 2, 3],
            response_ids=MockSampledSequence(tokens=[4, 5, 6], logprobs=[-0.1, -0.2, -0.3]),
            advantage=[0.5, 0.6],  # Only 2 elements, but 3 response tokens
        )
        trajectory = Trajectory(steps=[step])

        with pytest.raises(AssertionError, match="length mismatch"):
            trajectory_to_datums(trajectory)

    def test_missing_logprobs_raises(self):
        """If logprobs is None, should raise."""
        mock_output = MagicMock()
        mock_output.logprobs = None
        mock_output.tokens = [4, 5]

        step = Step(
            prompt_ids=[1, 2, 3],
            response_ids=mock_output,
            advantage=0.5,
        )
        trajectory = Trajectory(steps=[step])

        with pytest.raises(AssertionError, match="logprobs is None"):
            trajectory_to_datums(trajectory)

    def test_missing_advantage_raises(self):
        """If advantage is None, should raise."""
        step = Step(
            prompt_ids=[1, 2, 3],
            response_ids=MockSampledSequence(tokens=[4, 5], logprobs=[-0.1, -0.2]),
            advantage=None,
        )
        trajectory = Trajectory(steps=[step])

        with pytest.raises(AssertionError, match="advantage is None"):
            trajectory_to_datums(trajectory)

    def test_single_token_prompt_and_response(self):
        """Test minimal case with single token prompt and response."""
        step = make_step(
            prompt_ids=[1],
            response_tokens=[2],
            response_logprobs=[-0.5],
            advantage=1.0,
        )
        trajectory = Trajectory(steps=[step])

        datums = trajectory_to_datums(trajectory)

        assert len(datums) == 1
        verify_datum_structure(datums[0])

        # Sequence: [1, 2], after [1:] shift: length 1
        mask_data = datums[0].loss_fn_inputs["mask"].data
        assert mask_data == [1.0]  # Only the response token remains after shift
