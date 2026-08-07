"""Length-grouped SFT batching.

``TinkerSFTDataset`` can bucket similarly-sized rows into a batch so a batch pads
to little more than its own rows' real length — mirroring the Fireworks training
cookbook's ``group_by_length``. These assert the batching invariants the lever
must never break (full per-epoch coverage, seeded determinism, no silent drops)
and that grouping actually cuts padding versus a plain shuffle.
"""

from __future__ import annotations

import random

import pytest

# tinker_dataset imports tinker / tinker_cookbook at module load.
pytest.importorskip("tinker")
pytest.importorskip("tinker_cookbook")

import rllm.trainer.sft.tinker_dataset as td  # noqa: E402
from rllm.data import Dataset  # noqa: E402
from rllm.trainer.sft.tinker_dataset import TinkerSFTDataset, length_grouped_order  # noqa: E402


def _lengths(n=1000, seed=0):
    r = random.Random(seed)
    return [r.randint(100, 30000) for _ in range(n)]


def _padding_waste(order, lengths, bs, L=32768):
    """Fraction of processed tokens that are padding under pad-to-batch-max."""
    caps = [min(lengths[i], L) for i in order]
    nb = len(order) // bs
    padded = real = 0
    for b in range(nb):
        chunk = caps[b * bs : (b + 1) * bs]
        real += sum(chunk)
        padded += bs * max(chunk)
    return 1 - real / padded


def _messages_ds(n=200):
    """Rows with a unique per-row id (user content) and varied length."""
    rows = [{"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": "a" * (1 + (i * 7) % 60)}]} for i in range(n)]
    return Dataset(data=rows, name="grp", split="train")


# -- sampler invariants -------------------------------------------------------


def test_length_grouped_order_is_full_permutation():
    """Every row appears exactly once — no drop, no duplicate."""
    order = length_grouped_order(_lengths(1000), batch_size=32, factor=50, seed=1)
    assert sorted(order) == list(range(1000))


@pytest.mark.parametrize("seed", [0, 7])
def test_length_grouped_order_deterministic_and_seed_sensitive(seed):
    lengths = _lengths(500)
    a = length_grouped_order(lengths, 16, 25, seed)
    assert a == length_grouped_order(lengths, 16, 25, seed)  # same seed -> same order
    assert a != length_grouped_order(lengths, 16, 25, seed + 1)  # seed matters


def test_length_grouping_cuts_padding_vs_shuffle():
    """The point of the lever: grouped batches waste far less on padding."""
    lengths = _lengths(2000, seed=3)
    bs = 32
    plain = list(range(len(lengths)))
    random.Random(3).shuffle(plain)
    grouped = length_grouped_order(lengths, bs, 50, seed=3)
    assert _padding_waste(grouped, lengths, bs) < 0.5 * _padding_waste(plain, lengths, bs)


# -- dataset integration ------------------------------------------------------


def test_group_by_length_covers_every_row_through_get_batch(monkeypatch):
    """set_epoch(grouped) + get_batch over all batches touches every row once,
    including the final partial batch."""
    ds = TinkerSFTDataset(_messages_ds(203), renderer=object(), batch_size=8, group_by_length=True, length_group_factor=5)
    # Stub the renderer-dependent datum build; return raw messages so coverage
    # is measured through the real get_batch index mapping (not the renderer).
    monkeypatch.setattr(td, "conversation_to_datum", lambda messages, renderer, max_length, last_only, **kwargs: messages)
    monkeypatch.setattr(td, "count_loss_tokens", lambda datums: 1)
    ds.set_epoch(seed=0)
    assert sorted(ds._order) == list(range(203))  # full permutation

    seen = [m[0]["content"] for b in range(len(ds)) for m in ds.get_batch(b)]
    assert len(seen) == 203
    assert len(set(seen)) == len(seen)  # each trained row exactly once


def test_group_by_length_is_deterministic_across_instances():
    a = TinkerSFTDataset(_messages_ds(120), renderer=object(), batch_size=8, group_by_length=True, length_group_factor=4)
    b = TinkerSFTDataset(_messages_ds(120), renderer=object(), batch_size=8, group_by_length=True, length_group_factor=4)
    a.set_epoch(3)
    b.set_epoch(3)
    assert a._order == b._order


def test_default_shuffle_path_is_full_permutation_and_deterministic():
    """Grouping off (default): the explicit-order refactor still yields a seeded
    full-coverage permutation each epoch."""
    ds = TinkerSFTDataset(_messages_ds(120), renderer=object(), batch_size=8)
    ds.set_epoch(seed=1)
    assert sorted(ds._order) == list(range(120))
    first = list(ds._order)
    ds.set_epoch(seed=1)
    assert ds._order == first


def test_raw_row_cursor_round_trips_full_and_partial_batches():
    ds = TinkerSFTDataset(_messages_ds(10), renderer=object(), batch_size=4)

    # Three batches/epoch (4, 4, 2 rows); every checkpoint boundary maps back
    # to the next unseen batch, including across the partial final batch.
    assert [ds.data_cursor_for_step(step) for step in range(7)] == [0, 4, 8, 10, 14, 18, 20]
    for step in range(7):
        assert ds.step_for_data_cursor(ds.data_cursor_for_step(step)) == step
