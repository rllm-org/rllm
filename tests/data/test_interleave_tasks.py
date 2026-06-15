from pathlib import Path

from rllm.data.utils import interleave_tasks
from rllm.types import Task


def test_dict_and_task_rows_share_uid_when_id_matches():
    """A dict row and a Task with the same id must produce the same GRPO group uid."""
    row = {"id": "abc", "instruction": "x"}
    task = Task(id="abc", instruction="x", metadata=row, dataset_dir=Path("."))

    tasks_d, ids_d = interleave_tasks([row], 3)
    tasks_t, ids_t = interleave_tasks([task], 3)

    assert ids_d == ids_t == ["abc", "abc", "abc"]
    assert tasks_d == [row, row, row]  # same ref repeated group_size times
    assert tasks_t == [task, task, task]


def test_falsy_id_falls_back_to_shared_uuid():
    """Rows without an id get one shared uuid per group (stable within the group)."""
    _, ids = interleave_tasks([{"instruction": "y"}], 2)
    assert len(ids) == 2
    assert ids[0] == ids[1]


def test_each_group_gets_distinct_uid():
    a = Task(id="a", instruction="", metadata={}, dataset_dir=Path("."))
    b = Task(id="b", instruction="", metadata={}, dataset_dir=Path("."))
    _, ids = interleave_tasks([a, b], 2)
    assert ids == ["a", "a", "b", "b"]
