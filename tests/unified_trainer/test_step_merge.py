"""Tests for the backend-agnostic step-merging module.

Covers ``DefaultTokenOps`` (default; Verl) and a mock chunk-aware ``TokenOps``
that mirrors what the Tinker backend ships. Each branch of the merge
(prefix-extend, prefix-break, missing logprobs/advantages, per-token and
per-segment extras) is tested below.
"""

from dataclasses import dataclass

import pytest

from rllm.agents.agent import Step, Trajectory
from rllm.experimental.common.step_merge import (
    DefaultTokenOps,
    PerTokenExtras,
    merge_trajectory_steps,
)

# ---------------------------------------------------------------------------
# Mock chunk classes + ops so this file doesn't require tinker.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextChunk:
    """``EncodedTextChunk`` stand-in: ``flatten_prompt`` unwraps it to ints."""

    tokens: tuple[int, ...]

    @property
    def length(self) -> int:
        return len(self.tokens)


@dataclass(frozen=True)
class ImageChunk:
    """``ImageChunk`` stand-in: occupies ``length`` token slots, kept by reference."""

    length: int
    data: bytes = b""


@dataclass(frozen=True)
class _MockChunkOps:
    """Chunk-aware ``TokenOps`` mirroring what Tinker ships."""

    def flatten_prompt(self, prompt):
        out = []
        for elem in prompt:
            if isinstance(elem, int):
                out.append(elem)
            elif isinstance(elem, TextChunk):
                out.extend(elem.tokens)
            else:
                out.append(elem)
        return out

    def flat_token_length(self, t):
        return sum(1 if isinstance(e, int) else e.length for e in t)


_CHUNK_OPS = _MockChunkOps()


def _make_step(
    prompt_ids,
    response_ids,
    *,
    logprobs=None,
    advantage=1.0,
    routing_matrices=None,
):
    if logprobs is None:
        logprobs = [-0.1] * len(response_ids)
    return Step(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        logprobs=logprobs,
        advantage=advantage,
        routing_matrices=routing_matrices,
    )


# ---------------------------------------------------------------------------
# DefaultTokenOps default (Verl-shaped data)
# ---------------------------------------------------------------------------


class TestSingleStep:
    def test_int_only(self):
        step = _make_step([1, 2, 3], [4, 5], logprobs=[-0.5, -0.8], advantage=0.5)
        seg = merge_trajectory_steps(Trajectory(steps=[step]))[0]

        assert seg.prompt_ids == [1, 2, 3]
        assert seg.response_ids == [4, 5]
        assert seg.response_mask == [1, 1]
        assert seg.response_logprobs == [-0.5, -0.8]
        assert seg.response_advantages == [0.5, 0.5]
        assert seg.extras == {}
        assert seg.num_response_tokens == 2

    def test_per_token_advantage(self):
        step = _make_step([1, 2], [3, 4, 5], logprobs=[-0.1, -0.2, -0.3], advantage=[0.5, 0.6, 0.7])
        seg = merge_trajectory_steps(Trajectory(steps=[step]))[0]
        assert seg.response_advantages == [0.5, 0.6, 0.7]


class TestPrefixMerging:
    def test_two_step_cumulative_int(self):
        # step1 prompt=[1,2] response=[3,4]; step2 prompt=[1,2,3,4,5] extends.
        s1 = _make_step([1, 2], [3, 4], logprobs=[-0.1, -0.2], advantage=0.5)
        s2 = _make_step([1, 2, 3, 4, 5], [6, 7], logprobs=[-0.3, -0.4], advantage=0.6)

        segs = merge_trajectory_steps(Trajectory(steps=[s1, s2]))
        assert len(segs) == 1
        seg = segs[0]
        assert seg.prompt_ids == [1, 2]
        assert seg.response_ids == [3, 4, 5, 6, 7]
        assert seg.response_mask == [1, 1, 0, 1, 1]
        assert seg.response_logprobs == [-0.1, -0.2, 0.0, -0.3, -0.4]
        assert seg.response_advantages == [0.5, 0.5, 0.0, 0.6, 0.6]

    def test_three_step_cumulative_int(self):
        s1 = _make_step([1, 2], [3, 4], advantage=0.5)
        s2 = _make_step([1, 2, 3, 4, 5], [6, 7], advantage=0.6)
        s3 = _make_step([1, 2, 3, 4, 5, 6, 7, 8], [9, 10], advantage=0.7)
        seg = merge_trajectory_steps(Trajectory(steps=[s1, s2, s3]))[0]
        assert seg.response_ids == [3, 4, 5, 6, 7, 8, 9, 10]
        assert seg.response_mask == [1, 1, 0, 1, 1, 0, 1, 1]


class TestPrefixBreak:
    def test_two_independent(self):
        s1 = _make_step([1, 2, 3], [4, 5], advantage=0.5)
        s2 = _make_step([10, 11], [13, 14], advantage=0.6)
        segs = merge_trajectory_steps(Trajectory(steps=[s1, s2]))
        assert [seg.prompt_ids for seg in segs] == [[1, 2, 3], [10, 11]]
        assert [seg.response_ids for seg in segs] == [[4, 5], [13, 14]]

    def test_mixed(self):
        s1 = _make_step([1, 2], [3, 4], advantage=0.5)
        s2 = _make_step([1, 2, 3, 4, 5], [6, 7], advantage=0.6)
        s3 = _make_step([100, 101], [102, 103], advantage=0.7)
        segs = merge_trajectory_steps(Trajectory(steps=[s1, s2, s3]))
        assert len(segs) == 2
        assert segs[0].response_mask == [1, 1, 0, 1, 1]
        assert segs[1].response_mask == [1, 1]


# ---------------------------------------------------------------------------
# Chunk-aware TokenOps (Tinker-shaped data)
# ---------------------------------------------------------------------------


class TestChunkOps:
    def test_text_chunk_flattened_in_prompt(self):
        text = TextChunk(tokens=(1, 2, 3))
        step = _make_step([text, 4, 5], [6, 7], logprobs=[-0.1, -0.2], advantage=0.5)
        seg = merge_trajectory_steps(Trajectory(steps=[step]), token_ops=_CHUNK_OPS)[0]
        assert seg.prompt_ids == [1, 2, 3, 4, 5]
        assert seg.response_ids == [6, 7]
        assert seg.response_mask == [1, 1]

    def test_image_chunk_preserved_in_prompt(self):
        img = ImageChunk(length=3)
        step = _make_step([1, img, 2], [10, 11], logprobs=[-0.1, -0.2], advantage=1.0)
        seg = merge_trajectory_steps(Trajectory(steps=[step]), token_ops=_CHUNK_OPS)[0]
        assert seg.prompt_ids == [1, img, 2]
        # Per-flat-token mask is sized to action tokens, which are pure ints.
        assert seg.response_mask == [1, 1]

    def test_shared_image_chunk_merges(self):
        img = ImageChunk(length=4, data=b"shared")
        s1 = _make_step([img, 1, 2], [3, 4], advantage=0.5)
        s2 = _make_step([img, 1, 2, 3, 4, 5], [6, 7], advantage=0.6)
        segs = merge_trajectory_steps(Trajectory(steps=[s1, s2]), token_ops=_CHUNK_OPS)
        assert len(segs) == 1
        assert segs[0].prompt_ids == [img, 1, 2]
        assert segs[0].response_ids == [3, 4, 5, 6, 7]
        assert segs[0].response_mask == [1, 1, 0, 1, 1]

    def test_different_image_chunks_split(self):
        img1 = ImageChunk(length=4, data=b"one")
        img2 = ImageChunk(length=4, data=b"two")
        s1 = _make_step([img1, 1], [2, 3], advantage=0.5)
        s2 = _make_step([img2, 1, 2, 3, 4], [5, 6], advantage=0.6)
        segs = merge_trajectory_steps(Trajectory(steps=[s1, s2]), token_ops=_CHUNK_OPS)
        assert len(segs) == 2

    def test_delta_obs_with_image_chunk_extends_mask(self):
        # step2's delta_obs contains an ImageChunk → its 5 token slots must
        # show up as 0s in the mask (and per-token arrays).
        img = ImageChunk(length=5)
        s1 = _make_step([1], [2], logprobs=[-0.1], advantage=0.5)
        s2 = _make_step([1, 2, img, 3], [4], logprobs=[-0.2], advantage=0.6)
        seg = merge_trajectory_steps(Trajectory(steps=[s1, s2]), token_ops=_CHUNK_OPS)[0]
        # response = action0(=1 token) + delta_obs([img, 3] = 5+1 tokens) + action1(=1 token)
        assert seg.response_mask == [1] + [0] * 6 + [1]
        assert seg.response_logprobs == [-0.1] + [0.0] * 6 + [-0.2]
        assert seg.response_advantages == [0.5] + [0.0] * 6 + [0.6]


# ---------------------------------------------------------------------------
# Strict / tolerant flags
# ---------------------------------------------------------------------------


class TestStrict:
    def test_require_logprobs_raises(self):
        step = _make_step([1, 2], [3, 4], logprobs=[], advantage=0.5)
        with pytest.raises(AssertionError, match="logprobs"):
            merge_trajectory_steps(Trajectory(steps=[step]), require_logprobs=True)

    def test_require_advantage_raises(self):
        step = _make_step([1, 2], [3, 4], advantage=None)
        with pytest.raises(AssertionError, match="advantage"):
            merge_trajectory_steps(Trajectory(steps=[step]), require_advantage=True)

    def test_advantage_list_length_mismatch_raises(self):
        step = _make_step([1, 2], [3, 4, 5], advantage=[0.1, 0.2])
        with pytest.raises(AssertionError, match="advantage"):
            merge_trajectory_steps(Trajectory(steps=[step]))


class TestTolerant:
    def test_missing_logprobs_yields_empty(self):
        step = _make_step([1, 2], [3, 4], logprobs=[], advantage=0.5)
        seg = merge_trajectory_steps(Trajectory(steps=[step]))[0]
        assert seg.response_logprobs == []
        assert seg.response_mask == [1, 1]

    def test_missing_advantage_yields_zeros(self):
        step = _make_step([1, 2], [3, 4], advantage=None)
        seg = merge_trajectory_steps(Trajectory(steps=[step]))[0]
        assert seg.response_advantages == [0.0, 0.0]

    def test_pad_short_logprobs(self):
        step = Step(prompt_ids=[1, 2], response_ids=[3, 4, 5], logprobs=[], advantage=0.5)
        # Bypass Step's own length-check by setting attribute after init.
        step.logprobs = [-0.1]
        seg = merge_trajectory_steps(Trajectory(steps=[step]), pad_short_logprobs=True)[0]
        assert seg.response_logprobs == [-0.1, 0.0, 0.0]


class TestEdgeCases:
    def test_empty_trajectory(self):
        assert merge_trajectory_steps(Trajectory(steps=[])) == []

    def test_skip_steps_without_model_output(self):
        from rllm.experimental.rollout import ModelOutput

        s_bad = Step(prompt_ids=[1, 2], response_ids=[3, 4], advantage=0.5)
        s_good = Step(
            prompt_ids=[1, 2],
            response_ids=[3, 4],
            logprobs=[-0.1, -0.2],
            advantage=0.6,
            model_output=ModelOutput(prompt_ids=[1, 2], completion_ids=[3, 4], logprobs=[-0.1, -0.2]),
        )
        segs = merge_trajectory_steps(
            Trajectory(steps=[s_bad, s_good]),
            skip_steps_without_model_output=True,
        )
        assert len(segs) == 1
        assert segs[0].prompt_ids == [1, 2]


# ---------------------------------------------------------------------------
# Extras
# ---------------------------------------------------------------------------


class TestPerTokenExtras:
    def test_routing_matrices_passthrough_with_pad(self):
        s1 = _make_step([1, 2], [3, 4], advantage=0.5, routing_matrices=["a", "b"])
        s2 = _make_step([1, 2, 3, 4, 5], [6, 7], advantage=0.6, routing_matrices=["c", "d"])
        per_token = {
            "routing_matrices": PerTokenExtras(
                extractor=lambda s: s.routing_matrices,
                pad_value="",
            )
        }
        seg = merge_trajectory_steps(Trajectory(steps=[s1, s2]), per_token_extras=per_token)[0]
        assert seg.response_mask == [1, 1, 0, 1, 1]
        assert seg.extras["routing_matrices"] == ["a", "b", "", "c", "d"]

    def test_pad_for_step_without_field(self):
        s1 = _make_step([1, 2], [3, 4], advantage=0.5, routing_matrices=["a", "b"])
        s2 = _make_step([1, 2, 3, 4], [5, 6], advantage=0.6, routing_matrices=None)
        per_token = {
            "routing_matrices": PerTokenExtras(
                extractor=lambda s: s.routing_matrices,
                pad_value="<pad>",
            )
        }
        seg = merge_trajectory_steps(Trajectory(steps=[s1, s2]), per_token_extras=per_token)[0]
        assert seg.response_mask == [1, 1, 1, 1]
        assert seg.extras["routing_matrices"] == ["a", "b", "<pad>", "<pad>"]

    def test_extras_absent_when_not_requested(self):
        step = _make_step([1, 2], [3, 4], advantage=0.5, routing_matrices=["a", "b"])
        seg = merge_trajectory_steps(Trajectory(steps=[step]))[0]
        assert "routing_matrices" not in seg.extras


class TestPerSegmentExtras:
    def test_taken_from_first_step(self):
        s1 = _make_step([1, 2], [3, 4], advantage=0.5)
        s2 = _make_step([1, 2, 3, 4, 5], [6, 7], advantage=0.6)
        seg = merge_trajectory_steps(
            Trajectory(steps=[s1, s2]),
            per_segment_extras={
                "multi_modal_inputs": lambda s: {"image_grid_thw": [1, 2, 3]} if s is s1 else {"image_grid_thw": [4, 5, 6]},
            },
        )[0]
        assert seg.extras["multi_modal_inputs"] == {"image_grid_thw": [1, 2, 3]}

    def test_per_segment_when_split(self):
        s1 = _make_step([1], [2], advantage=0.5)
        s2 = _make_step([100], [101], advantage=0.6)
        segs = merge_trajectory_steps(
            Trajectory(steps=[s1, s2]),
            per_segment_extras={"meta": lambda s: {"tag": "first"} if s is s1 else {"tag": "second"}},
        )
        assert [seg.extras["meta"] for seg in segs] == [{"tag": "first"}, {"tag": "second"}]


# ---------------------------------------------------------------------------
# Length invariant
# ---------------------------------------------------------------------------


class TestLengthInvariant:
    def test_int_only(self):
        s1 = _make_step([1, 2], [3, 4], advantage=0.5)
        s2 = _make_step([1, 2, 3, 4, 5, 6], [7], advantage=0.6)
        seg = merge_trajectory_steps(Trajectory(steps=[s1, s2]))[0]
        n = len(seg.response_ids)  # int-only ⇒ flat == len
        assert n == len(seg.response_mask) == len(seg.response_logprobs) == len(seg.response_advantages)

    def test_chunk_ops_flat_length(self):
        img = ImageChunk(length=3)
        s1 = _make_step([1], [2], logprobs=[-0.1], advantage=0.5)
        s2 = _make_step([1, 2, img], [3], logprobs=[-0.2], advantage=0.6)
        seg = merge_trajectory_steps(Trajectory(steps=[s1, s2]), token_ops=_CHUNK_OPS)[0]
        # response: action0(1) + delta_obs([img]=3 slots) + action1(1) = 5
        assert len(seg.response_mask) == 5
        assert len(seg.response_logprobs) == 5
        assert len(seg.response_advantages) == 5


def test_int_token_ops_defaults():
    ops = DefaultTokenOps()
    assert ops.flatten_prompt([1, 2, 3]) == [1, 2, 3]
    assert ops.flat_token_length([1, 2, 3]) == 3
    assert ops.flat_token_length([]) == 0
