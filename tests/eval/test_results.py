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


def test_solver_usage_reports_tokens_only_and_excludes_judge_usage():
    items = [
        EvalItem(
            idx=0,
            reward=1.0,
            is_correct=True,
            metrics={
                "turns": 2,
                "input_tokens": 100,
                "answer_tokens": 20,
                "reasoning_tokens": 5,
                "output_tokens": 25,
            },
        ),
        EvalItem(
            idx=1,
            reward=0.0,
            is_correct=False,
            metrics={
                "turns": 4,
                "input_tokens": 200,
                "answer_tokens": 40,
                "reasoning_tokens": 10,
                "output_tokens": 50,
                "judge_tokens": 999,
            },
        ),
    ]

    usage = EvalResult.from_items("d", "m", "a", items).usage

    assert usage == {
        "tasks": 2,
        "average_turns_per_task": 3.0,
        "average_answer_tokens_per_task": 30.0,
        "average_reasoning_tokens_per_task": 7.5,
        "average_output_tokens_per_task": 37.5,
        "average_input_tokens_per_task": 150.0,
        "total_input_tokens": 300,
        "total_answer_tokens": 60,
        "total_reasoning_tokens": 15,
        "total_output_tokens": 75,
    }


def test_save_load_round_trips_solver_metrics_and_usage(tmp_path):
    item = EvalItem(
        idx=0,
        reward=1.0,
        is_correct=True,
        metrics={"turns": 3, "answer_tokens": 12, "reasoning_tokens": 4, "output_tokens": 16},
    )
    result = EvalResult.from_items("d", "m", "a", [item])

    loaded = EvalResult.load(result.save(str(tmp_path / "usage.json")))

    assert loaded.items[0].metrics == item.metrics
    assert loaded.usage == result.usage
    assert not any("cost" in key for key in loaded.usage)
