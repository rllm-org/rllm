"""Tests for rllm.eval.curation — eval trajectories → SFT rows."""

import json

import pytest

from rllm.eval.curation import CurationConfig, CurationError, curate
from rllm.eval.results import EvalItem, EvalResult


def _episode(user: str, assistant: str, name: str = "default") -> dict:
    return {
        "trajectories": [
            {
                "name": name,
                "steps": [
                    {
                        "chat_completions": [
                            {"role": "user", "content": user},
                            {"role": "assistant", "content": assistant},
                        ]
                    }
                ],
            }
        ]
    }


def _write_run(run_dir, *, attempts, rollouts):
    """Build a run dir in the on-disk eval format.

    ``rollouts`` is a list of (task_idx, attempt, is_correct, assistant_text).
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir = run_dir / "episodes"
    episodes_dir.mkdir()

    items = []
    for task_idx, attempt, is_correct, assistant in rollouts:
        reward = 1.0 if is_correct else 0.0
        items.append(EvalItem(idx=task_idx, attempt=attempt, reward=reward, is_correct=is_correct, signals={"accuracy": reward}))
        eval_idx = task_idx * attempts + attempt
        ep = _episode(f"question {task_idx}", assistant)
        (episodes_dir / f"episode_{eval_idx:06d}_t{task_idx}.json").write_text(json.dumps(ep))

    result = EvalResult.from_items("bench", "model", "agent", items, attempts=attempts)
    result.save(str(run_dir / "results.json"))
    return run_dir


@pytest.fixture
def run_dir(tmp_path):
    # t0: both wrong (avg=0, unsolved); t1: 1/2 (avg=0.5); t2: 2/2 (avg=1.0)
    return _write_run(
        tmp_path / "run1",
        attempts=2,
        rollouts=[
            (0, 0, False, "wrong a"),
            (0, 1, False, "wrong b"),
            (1, 0, True, "t1 correct"),
            (1, 1, False, "t1 wrong"),
            (2, 0, True, "t2 correct short"),
            (2, 1, True, "t2 correct much longer answer"),
        ],
    )


def test_default_solved_correct(run_dir):
    rows, stats = curate([run_dir], CurationConfig())  # filter=solved, select=correct
    assert stats.tasks_total == 3
    assert stats.tasks_kept == 2  # t1, t2 (t0 unsolved)
    # t1 → 1 correct, t2 → 2 correct
    assert stats.rows_emitted == 3
    task_ids = sorted(r["task_id"] for r in rows)
    assert task_ids == ["t1", "t2", "t2"]
    for r in rows:
        assert r["messages"][0]["role"] == "user"
        assert r["messages"][-1]["role"] == "assistant"
        assert r["source_run"] == "run1"


def test_difficulty_band_filter(run_dir):
    rows, stats = curate([run_dir], CurationConfig(filter_expr="0 < avg < 1"))
    assert stats.tasks_kept == 1  # only t1
    assert {r["task_id"] for r in rows} == {"t1"}


def test_select_best_caps_one_per_task(run_dir):
    rows, _ = curate([run_dir], CurationConfig(select="best"))
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task_id"], []).append(r)
    assert all(len(v) == 1 for v in by_task.values())


def test_max_per_task(run_dir):
    rows, _ = curate([run_dir], CurationConfig(max_per_task=1))
    assert len(rows) == 2  # one each from t1, t2


def test_select_shortest(run_dir):
    rows, _ = curate([run_dir], CurationConfig(filter_expr="avg == 1", select="shortest", max_per_task=1))
    # only t2 has avg==1; shortest of its two correct answers
    assert len(rows) == 1
    assert rows[0]["messages"][-1]["content"] == "t2 correct short"


def test_select_all_includes_failures(run_dir):
    rows, _ = curate([run_dir], CurationConfig(filter_expr="solved", select="all"))
    # t1 (2 attempts) + t2 (2 attempts) = 4, including the t1 failure
    assert len(rows) == 4
    contents = {r["messages"][-1]["content"] for r in rows}
    assert "t1 wrong" in contents


def test_pass_at_k_filter(run_dir):
    # pass@2 == 1.0 for any task with >=1 success
    rows, stats = curate([run_dir], CurationConfig(filter_expr="pass@2 >= 1"))
    assert stats.tasks_kept == 2


def test_dedup(tmp_path):
    rd = _write_run(
        tmp_path / "dup",
        attempts=2,
        rollouts=[
            (0, 0, True, "identical answer"),
            (0, 1, True, "identical answer"),
        ],
    )
    rows, stats = curate([rd], CurationConfig(dedup=True))
    assert stats.rows_deduped == 1
    assert len(rows) == 1


def test_multi_run_pooling(tmp_path):
    # Same task t0 across two runs: each has 1/1 correct -> pooled n=2, n_correct=2.
    r1 = _write_run(tmp_path / "a", attempts=1, rollouts=[(0, 0, True, "ans from run a")])
    r2 = _write_run(tmp_path / "b", attempts=1, rollouts=[(0, 0, True, "ans from run b")])
    rows, stats = curate([r1, r2], CurationConfig())
    assert stats.tasks_total == 1  # pooled by task_id "t0"
    assert stats.attempts_total == 2
    assert len(rows) == 2  # both correct trajectories kept


def test_metric_reward_with_min_reward(tmp_path):
    rd = _write_run(
        tmp_path / "m",
        attempts=2,
        rollouts=[
            (0, 0, True, "good"),
            (0, 1, False, "bad"),
        ],
    )
    # metric=reward, keep tasks with avg>0, select trajectories with reward>=1.0
    rows, _ = curate([rd], CurationConfig(metric="reward", filter_expr="avg > 0", min_reward=1.0))
    assert len(rows) == 1
    assert rows[0]["messages"][-1]["content"] == "good"


def test_skips_missing_messages(tmp_path):
    rd = tmp_path / "empty"
    rd.mkdir()
    (rd / "episodes").mkdir()
    # correct item but no episode file written -> skipped
    items = [EvalItem(idx=0, attempt=0, reward=1.0, is_correct=True, signals={})]
    EvalResult.from_items("b", "m", "a", items, attempts=1).save(str(rd / "results.json"))
    rows, stats = curate([rd], CurationConfig())
    assert rows == []
    assert stats.rows_skipped_no_messages == 1


def test_unknown_run_dir_raises(tmp_path):
    with pytest.raises(CurationError):
        curate([tmp_path / "does-not-exist"], CurationConfig())


def test_named_trajectory(tmp_path):
    rd = tmp_path / "named"
    rd.mkdir()
    (rd / "episodes").mkdir()
    items = [EvalItem(idx=0, attempt=0, reward=1.0, is_correct=True, signals={})]
    EvalResult.from_items("b", "m", "a", items, attempts=1).save(str(rd / "results.json"))
    ep = _episode("q", "from named traj", name="planner")
    (rd / "episodes" / "episode_000000_t0.json").write_text(json.dumps(ep))

    # default (first trajectory) finds it
    rows, _ = curate([rd], CurationConfig())
    assert len(rows) == 1
    # explicit matching name finds it
    rows, _ = curate([rd], CurationConfig(trajectory="planner"))
    assert len(rows) == 1
    # non-matching name -> skipped
    rows, stats = curate([rd], CurationConfig(trajectory="nope"))
    assert rows == []
