from rllm.data import Dataset, StatefulTaskDataLoader


def _ds(n: int) -> Dataset:
    return Dataset(data=[{"id": i} for i in range(n)], name="t", split="train")


def _run(dataloader, n_epochs):
    """Mimic the trainer: ``for epoch in range(n): for batch in dataloader``. Returns a list per epoch."""
    return [[[d["id"] for d in batch] for batch in dataloader] for _ in range(n_epochs)]


def test_len_drop_last():
    assert len(StatefulTaskDataLoader(_ds(10), 4, drop_last=True)) == 2
    assert len(StatefulTaskDataLoader(_ds(10), 4, drop_last=False)) == 3


def test_single_iter_is_one_epoch_and_advances():
    dataloader = StatefulTaskDataLoader(_ds(4), 2, shuffle=False)
    assert dataloader.epoch == 0
    assert [[d["id"] for d in b] for b in dataloader] == [[0, 1], [2, 3]]  # one epoch
    assert dataloader.epoch == 1  # the pass advanced the epoch
    assert [[d["id"] for d in b] for b in dataloader] == [[0, 1], [2, 3]]  # next pass = next epoch
    assert dataloader.epoch == 2


def test_deterministic_shuffle_per_epoch():
    a = _run(StatefulTaskDataLoader(_ds(10), 2, seed=7), 2)
    b = _run(StatefulTaskDataLoader(_ds(10), 2, seed=7), 2)
    assert a == b
    assert a[0] != a[1]  # different epochs reshuffle


def test_no_shuffle_is_sequential():
    dataloader = StatefulTaskDataLoader(_ds(6), 2, shuffle=False)
    assert _run(dataloader, 1)[0] == [[0, 1], [2, 3], [4, 5]]


def test_resume_mid_epoch():
    flat_ref = [b for ep in _run(StatefulTaskDataLoader(_ds(10), 2, seed=3), 2) for b in ep]

    dataloader = StatefulTaskDataLoader(_ds(10), 2, seed=3)
    seen = []
    it = iter(dataloader)
    for _ in range(3):  # 3 of epoch 0's 5 batches, then "crash"
        seen.append([d["id"] for d in next(it)])
    state = dataloader.state_dict()

    resumed = StatefulTaskDataLoader(_ds(10), 2, seed=3)
    resumed.load_state_dict(state)
    for _epoch in range(resumed.epoch, 2):
        for batch in resumed:
            seen.append([d["id"] for d in batch])

    assert seen == flat_ref


def test_resume_continues_remaining_epochs():
    flat_ref = [b for ep in _run(StatefulTaskDataLoader(_ds(8), 4, seed=1), 3) for b in ep]

    dataloader = StatefulTaskDataLoader(_ds(8), 4, seed=1)
    seen = [[d["id"] for d in batch] for batch in dataloader]  # full epoch 0
    assert dataloader.epoch == 1
    state = dataloader.state_dict()

    resumed = StatefulTaskDataLoader(_ds(8), 4, seed=1)
    resumed.load_state_dict(state)
    for _epoch in range(resumed.epoch, 3):
        for batch in resumed:
            seen.append([d["id"] for d in batch])

    assert seen == flat_ref
