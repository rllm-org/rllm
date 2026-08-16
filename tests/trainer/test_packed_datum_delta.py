"""Delta-fed datum building must equal full-list-fed building exactly.

The conversion to full lists exists ONLY as the parity instrument here: the
same trajectory is expressed both ways and the resulting Datums must match
token-for-token (inputs, targets, logprobs, advantages, masks). In
production the full-list form is never constructed.
"""

import pytest

tinker = pytest.importorskip("tinker")

from rllm.trainer.tinker.transform import trajectory_to_datums  # noqa: E402
from rllm.types import Step, Trajectory  # noqa: E402


def _mk_step(prompt, response, adv=0.5):
    return Step(
        prompt_ids=prompt,
        response_ids=list(response),
        logprobs=[-0.1] * len(response),
        advantage=adv,
    )


def _both_forms(prompts_full, responses):
    """The same trajectory as (full-list steps, delta steps)."""
    full = [_mk_step(list(p), r) for p, r in zip(prompts_full, responses, strict=True)]
    deltas = []
    prev = []
    for p, r in zip(prompts_full, responses, strict=True):
        lcp = 0
        while lcp < min(len(prev), len(p)) and prev[lcp] == p[lcp]:
            lcp += 1
        deltas.append(_mk_step({"__prompt_ids_delta__": [lcp, list(p[lcp:])]}, r))
        prev = p
    return Trajectory(steps=full), Trajectory(steps=deltas)


def _datum_key(d):
    return (
        list(d.model_input.to_ints()),
        list(d.loss_fn_inputs["target_tokens"].data),
        list(d.loss_fn_inputs["logprobs"].data),
        list(d.loss_fn_inputs["advantages"].data),
        list(d.loss_fn_inputs["mask"].data),
    )


def _assert_equal_datums(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert _datum_key(x) == _datum_key(y)


def test_cumulative_trajectory_delta_equals_full():
    """5 turns, each prompt = previous prompt + previous response + new user."""
    prompts, responses = [], []
    p = [1, 2, 3]
    for i in range(5):
        prompts.append(list(p))
        r = [100 + i, 200 + i]
        responses.append(r)
        p = p + r + [50 + i]  # next prompt: + response + one user token
    t_full, t_delta = _both_forms(prompts, responses)
    _assert_equal_datums(trajectory_to_datums(t_full), trajectory_to_datums(t_delta))
    # and the merge really packed: one datum for the whole lineage
    assert len(trajectory_to_datums(t_delta)) == 1


def test_fork_trajectory_delta_equals_full():
    """A mid-trajectory context reset (lcp drops) must fork identically."""
    prompts = [[1, 2], [1, 2, 100, 3], [9, 9, 9], [9, 9, 9, 300, 4]]
    responses = [[100], [101], [300], [301]]
    t_full, t_delta = _both_forms(prompts, responses)
    a, b = trajectory_to_datums(t_full), trajectory_to_datums(t_delta)
    _assert_equal_datums(a, b)
    assert len(a) == 2  # the reset produced exactly one fork both ways


def test_mixed_full_and_delta_steps():
    """Producers may migrate incrementally: mixed forms in one trajectory."""
    prompts = [[1, 2], [1, 2, 100, 3], [1, 2, 100, 3, 101, 4]]
    responses = [[100], [101], [102]]
    t_full, t_delta = _both_forms(prompts, responses)
    mixed = Trajectory(steps=[t_full.steps[0], t_delta.steps[1], t_full.steps[2]])
    _assert_equal_datums(trajectory_to_datums(t_full), trajectory_to_datums(mixed))


def test_bad_delta_raises():
    with pytest.raises(ValueError, match="exceeds previous prompt length"):
        trajectory_to_datums(Trajectory(steps=[_mk_step({"__prompt_ids_delta__": [5, [1]]}, [7])]))


def test_delta_path_never_runs_full_prefix_compare(monkeypatch):
    """Pure-extension delta steps must merge in O(new): the full-sequence
    prefix compare must not run at all."""
    import rllm.trainer.tinker.transform as tf

    def _boom(*a, **k):
        raise AssertionError("_is_prefix called on the delta fast path")

    monkeypatch.setattr(tf, "_is_prefix", _boom)
    prompts, responses = [], []
    p = [1, 2, 3]
    for i in range(4):
        prompts.append(list(p))
        r = [100 + i]
        responses.append(r)
        p = p + r + [50 + i]
    _, t_delta = _both_forms(prompts, responses)
    assert len(trajectory_to_datums(t_delta)) == 1


def test_conversion_does_not_mutate_steps():
    """Converting twice must give identical results: the incremental prompt
    buffer must never alias or mutate Step.prompt_ids."""
    prompts = [[1, 2], [1, 2, 100, 3]]
    responses = [[100], [101]]
    t_full, t_delta = _both_forms(prompts, responses)
    mixed = Trajectory(steps=[t_full.steps[0], t_delta.steps[1]])
    first = [_datum_key(d) for d in trajectory_to_datums(mixed)]
    second = [_datum_key(d) for d in trajectory_to_datums(mixed)]
    assert first == second
    assert mixed.steps[0].prompt_ids == [1, 2]
