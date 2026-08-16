"""Delta-fed verl rows must equal full-list-fed rows exactly.

The full-list form exists ONLY as the parity instrument: the same trajectory
is expressed both ways and the resulting DataProto tensors must match. The
delta feed carries NO full prompt list anywhere (``model_output.prompt_ids``
is None), which is the point — conversion works without the O(n^2)
expansion ever existing.
"""

from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("verl")

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine.rollout import ModelOutput
from rllm.trainer.verl.transform import transform_episodes_to_dataproto


def _engine(pad_token_id: int = 0):
    engine = MagicMock()
    engine.tokenizer.pad_token_id = pad_token_id
    engine.processor = None
    return engine


def _full_step(prompt, response, logprobs):
    return Step(
        prompt_ids=list(prompt),
        response_ids=list(response),
        model_output=ModelOutput(prompt_ids=list(prompt), completion_ids=list(response), logprobs=list(logprobs)),
    )


def _delta_step(lcp, suffix, response, logprobs):
    """A step whose prompt exists ONLY as a delta marker: no full list anywhere."""
    return Step(
        prompt_ids={"__prompt_ids_delta__": [lcp, list(suffix)]},
        response_ids=list(response),
        model_output=ModelOutput(prompt_ids=None, completion_ids=list(response), logprobs=list(logprobs)),
    )


def _both_forms(prompts_full, responses):
    """The same trajectory as (full-list steps, delta-only steps)."""
    lps = [[-0.1 * (i + 1)] * len(r) for i, r in enumerate(responses)]
    full = [_full_step(p, r, lp) for p, r, lp in zip(prompts_full, responses, lps, strict=True)]
    deltas = []
    prev: list[int] = []
    for p, r, lp in zip(prompts_full, responses, lps, strict=True):
        lcp = 0
        while lcp < min(len(prev), len(p)) and prev[lcp] == p[lcp]:
            lcp += 1
        deltas.append(_delta_step(lcp, p[lcp:], r, lp))
        prev = list(p)
    return full, deltas


def _episode(steps, eid="task_0:0"):
    traj = Trajectory(steps=steps, reward=1.0)
    return Episode(id=eid, trajectories=[traj], is_correct=True)


def _to_batch(steps):
    proto = transform_episodes_to_dataproto([_episode(steps)], _engine(), max_prompt_length=64, max_response_length=64)
    return proto.batch


def _assert_equal_batches(a, b):
    assert set(a.keys()) == set(b.keys())
    for k in a.keys():
        assert torch.equal(a[k], b[k]), f"tensor mismatch: {k}"


def _cumulative(turns):
    prompts, responses = [], []
    p = [1, 2, 3]
    for i in range(turns):
        prompts.append(list(p))
        r = [100 + i, 200 + i]
        responses.append(r)
        p = p + r + [50 + i]  # + previous response + one observation token
    return prompts, responses


def test_cumulative_trajectory_delta_equals_full():
    full, deltas = _both_forms(*_cumulative(5))
    a, b = _to_batch(full), _to_batch(deltas)
    _assert_equal_batches(a, b)
    assert a["input_ids"].shape[0] == 1  # the whole lineage merged into one row


def test_fork_trajectory_delta_equals_full():
    """A mid-trajectory context reset (lcp drops) must fork identically."""
    prompts = [[1, 2], [1, 2, 100, 3], [9, 9, 9], [9, 9, 9, 300, 4]]
    responses = [[100], [101], [300], [301]]
    full, deltas = _both_forms(prompts, responses)
    a, b = _to_batch(full), _to_batch(deltas)
    _assert_equal_batches(a, b)
    assert a["input_ids"].shape[0] == 2  # the reset produced exactly one fork


def test_mixed_full_and_delta_steps():
    """Producers may migrate incrementally: mixed forms in one trajectory."""
    prompts = [[1, 2], [1, 2, 100, 3], [1, 2, 100, 3, 101, 4]]
    responses = [[100], [101], [102]]
    full, deltas = _both_forms(prompts, responses)
    mixed = [full[0], deltas[1], full[2]]
    _assert_equal_batches(_to_batch(full), _to_batch(mixed))


def test_delta_feed_carries_no_full_prompt_lists():
    """The delta feed converts with model_output.prompt_ids=None on every
    step: the merge provably never reads a materialized prompt list."""
    _, deltas = _both_forms(*_cumulative(4))
    assert all(s.model_output.prompt_ids is None for s in deltas)
    assert _to_batch(deltas)["input_ids"].shape[0] == 1


def test_conversion_does_not_mutate_steps():
    """Converting twice must give identical results: the incremental prompt
    buffer must never alias or mutate step data."""
    full, deltas = _both_forms(*_cumulative(3))
    mixed = [full[0], deltas[1], deltas[2]]
    first, second = _to_batch(mixed), _to_batch(mixed)
    _assert_equal_batches(first, second)
    assert mixed[0].prompt_ids == full[0].prompt_ids


def test_bad_delta_raises():
    step = _delta_step(5, [1], [7], [-0.1])
    with pytest.raises(ValueError, match="exceeds previous prompt length"):
        _to_batch([step])
