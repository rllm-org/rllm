"""pass@k aggregation in EvalResult (--attempts N)."""

import math

from rllm.eval.results import EvalItem, EvalResult, _pass_at_k


def _items_for(per_task: list[list[bool]]) -> list[EvalItem]:
    """One EvalItem per attempt; per_task[i][j] = attempt j of task i passed."""
    return [EvalItem(idx=i, attempt=j, reward=float(ok), is_correct=ok) for i, attempts in enumerate(per_task) for j, ok in enumerate(attempts)]


def test_pass_at_k_unbiased_estimator():
    assert _pass_at_k([(2, 0)], 2) == 0.0
    assert _pass_at_k([(2, 2)], 1) == 1.0
    assert _pass_at_k([(2, 1)], 1) == 0.5  # 1 - C(1,1)/C(2,1)
    assert _pass_at_k([(2, 1)], 2) == 1.0  # any size-2 subset contains the success
    assert math.isclose(_pass_at_k([(4, 1)], 2), 0.5)  # 1 - C(3,2)/C(4,2)


def test_from_items_groups_attempts_by_task_idx():
    result = EvalResult.from_items("d", "m", "a", _items_for([[True, False], [False, False], [True, True]]), attempts=2)
    assert math.isclose(result.pass_at[1], 0.5)  # (0.5 + 0 + 1) / 3
    assert math.isclose(result.pass_at[2], 2 / 3)  # (1 + 0 + 1) / 3
    assert math.isclose(result.score, 0.5)  # 3/6 rollouts; equals unbiased pass@1 at equal n


def test_single_attempt_keeps_legacy_shape():
    result = EvalResult.from_items("d", "m", "a", _items_for([[True], [False]]))
    assert result.attempts == 1 and result.pass_at == {}


def test_save_load_round_trips_pass_at(tmp_path):
    result = EvalResult.from_items("d", "m", "a", _items_for([[True, False]]), attempts=2)
    loaded = EvalResult.load(result.save(str(tmp_path / "r.json")))
    assert loaded.attempts == 2 and loaded.pass_at == result.pass_at
    assert [(i.idx, i.attempt) for i in loaded.items] == [(0, 0), (0, 1)]
