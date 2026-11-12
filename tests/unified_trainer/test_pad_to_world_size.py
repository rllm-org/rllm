"""Padding a DataProto up to the world size must not add trainable rows.

``verl.protocol.pad_dataproto_to_divisor`` pads by duplicating the leading
rows, so the padded rows arrive carrying real advantages and a real
``response_mask``. ``is_pad_step`` / ``is_valid`` are rLLM-only markers that
verl never reads, so the loss mask is what has to be cleared.
"""

from __future__ import annotations

import numpy as np
import torch
from verl import DataProto

from rllm.trainer.verl.agent_workflow_trainer import AgentWorkflowPPOTrainer


class _WorkerGroup:
    world_size = 4


class _Trainer:
    """Minimal stand-in exposing only what the method under test reads."""

    use_critic = False
    use_reference_policy = False
    use_rm = False
    hybrid_engine = True
    actor_rollout_wg = _WorkerGroup()

    _remove_padding = AgentWorkflowPPOTrainer._remove_padding
    _pad_dataproto_to_world_size = AgentWorkflowPPOTrainer._pad_dataproto_to_world_size


def _make_batch(n_rows: int, resp_len: int = 3) -> DataProto:
    return DataProto.from_dict(
        tensors={
            "prompts": torch.arange(n_rows * 2).reshape(n_rows, 2),
            "response_mask": torch.ones(n_rows, resp_len, dtype=torch.long),
            "advantages": torch.ones(n_rows, resp_len, dtype=torch.float32),
        },
        non_tensors={
            "is_last_step": np.ones(n_rows, dtype=bool),
            "is_pad_step": np.zeros(n_rows, dtype=bool),
            "is_valid": np.ones(n_rows, dtype=bool),
        },
    )


def test_padded_rows_carry_no_loss_mask():
    """6 real rows at world size 4 means 2 duplicated pad rows, which must not train."""
    batch = _make_batch(6)
    out = _Trainer()._pad_dataproto_to_world_size(batch)

    assert len(out) == 8
    assert out.non_tensor_batch["is_pad_step"].tolist() == [False] * 6 + [True] * 2
    # verl pads by duplicating the leading rows, so the advantages are real.
    assert out.batch["advantages"][6].sum().item() == 3.0

    trainable = out.batch["response_mask"][6:].sum().item()
    assert trainable == 0, f"padded duplicate rows still have {trainable} trainable tokens"
    assert out.batch["response_mask"][:6].sum().item() == 18, "real rows must be untouched"


def test_no_padding_needed_is_a_noop():
    out = _Trainer()._pad_dataproto_to_world_size(_make_batch(8))

    assert len(out) == 8
    assert out.batch["response_mask"].sum().item() == 24
    assert not out.non_tensor_batch["is_pad_step"].any()
