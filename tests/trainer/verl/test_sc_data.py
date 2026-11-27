import json
import math
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from rllm.engine.rollout import ModelOutput
from rllm.trainer.verl.sc_fsdp_adapter import ScoreCenteringLoss, prepare_sc_outputs
from rllm.trainer.verl.score_centering import SC_FIELDS, batch_rollout_heads, heads_to_no_padding
from rllm.trainer.verl.transform import transform_episodes_to_dataproto
from rllm.types import Episode, Step, Trajectory
from rllm.workflows.workflow import TerminationReason


def episode(response=(3, 4, 0), name="task:0"):
    output = ModelOutput(
        prompt_ids=[1, 2],
        completion_ids=list(response),
        logprobs=[-0.5] * len(response),
        rollout_topk_ids=[[3, 4] for _ in response],
        rollout_topk_log_probs=[[math.log(0.4), math.log(0.3)] for _ in response],
    )
    return Episode(id=name, termination_reason=TerminationReason.ENV_DONE, trajectories=[Trajectory(steps=[Step.from_model_output(output)], reward=1)])


def transform(episodes, response_length=5):
    engine = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0), processor=None)
    return transform_episodes_to_dataproto(episodes, engine, max_prompt_length=4, max_response_length=response_length)


def test_episode_json_roundtrip_keeps_candidates_and_legacy_defaults():
    original = episode()
    restored = Episode.from_dict(json.loads(json.dumps(original.to_dict())))
    before, after = original.trajectories[0].steps[0], restored.trajectories[0].steps[0]
    for field in SC_FIELDS:
        assert getattr(after.model_output, field) == getattr(before.model_output, field) is not None
    legacy = ModelOutput.from_dict({"completion_ids": [3]})
    assert all(getattr(legacy, field) is None for field in SC_FIELDS)


def test_crop_padding_eos_and_no_padding_predictor_alignment(cpu_unpad):
    from verl.workers.utils.padding import left_right_2_no_padding

    batch = transform([episode(), episode((5,), "task:1")], response_length=2)
    assert batch.batch["rollout_topk_ids"].shape == (2, 2, 2)
    # Verify real EOS/pad-ID token remains attended in the uncropped batch.
    uncropped = transform([episode()])
    assert uncropped.batch["attention_mask"][0].tolist() == [0, 0, 1, 1, 1, 1, 1, 0, 0]
    data = batch.to_tensordict()
    heads_to_no_padding(data)
    heads = data["rollout_topk_ids"].unbind()
    # p=2: candidates belong to positions 1 and 2; final response position is dummy.
    assert heads[0][1:3].tolist() == [[3, 4], [3, 4]]
    assert heads[0][0].tolist() == [0, 1]
    assert heads[0][-1].tolist() == [0, 1]
    original_heads = [x.clone() for x in heads]
    data = left_right_2_no_padding(data)
    # Reorder through the same helper verl uses for balancing/shuffling (rLLM-patched for jagged layouts).
    from verl.utils import tensordict_utils as tu

    from rllm.trainer.verl.patch import patch_verl_tensordict_jagged_layout

    patch_verl_tensordict_jagged_layout()
    reordered = tu.index_select_tensor_dict(data, torch.tensor([1, 0]))
    torch.testing.assert_close(reordered["rollout_topk_ids"].unbind()[0], original_heads[1])
    torch.testing.assert_close(reordered["rollout_topk_ids"].unbind()[1], original_heads[0])


def test_missing_head_mixed_with_present_head_is_rejected():
    with pytest.raises(ValueError, match="Missing"):
        batch_rollout_heads([([[1]], [[0.0]]), None], [torch.ones(1), torch.ones(1)], 2)


def test_multi_turn_is_rejected():
    ep = episode()
    ep.trajectories[0].steps.append(ep.trajectories[0].steps[0].model_copy(deep=True))
    with pytest.raises(ValueError, match="single-turn"):
        transform([ep])


def test_empty_response_beside_nonempty_response_keeps_alignment():
    batch = transform([episode((), "task:empty"), episode()])
    assert batch.batch["response_mask"][0].sum() == 0
    assert batch.batch["rollout_topk_ids"].shape == (2, 5, 2)


@pytest.mark.parametrize("cap", [None, 2.0])
def test_cpu_forward_and_loss_with_real_verl_padding_and_aggregation(cap, cpu_unpad):
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.padding import left_right_2_no_padding

    batch = transform([episode(), episode((4,), "task:1")])
    batch.batch["advantages"] = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0, 0.0]])
    data = batch.to_tensordict()
    heads_to_no_padding(data)
    data = left_right_2_no_padding(data)
    tu.assign_non_tensor(data, use_remove_padding=True, use_fused_kernels=False, calculate_entropy=False, score_centering=True, dp_size=1, batch_num_tokens=4, global_batch_size=2)
    logits = torch.randn(1, data["input_ids"].values().numel(), 7, requires_grad=True)
    engine = SimpleNamespace(use_ulysses_sp=False, model_config=SimpleNamespace(hf_config=SimpleNamespace()))
    loss_fn = ScoreCenteringLoss(SimpleNamespaceConfig(loss_agg_mode="token-mean"), {"chunk_size": 2}, cap)
    labels = data["input_ids"].values().roll(-1)
    output = prepare_sc_outputs(
        engine,
        SimpleNamespace(logits=logits),
        {
            "input_ids_rmpad_rolled": labels,
            "temperature_rmpad": torch.ones(len(labels)),
        },
        data,
        loss_fn,
    )
    loss, metrics = loss_fn(output, data)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0
    assert "score_centering/sampler_head_mass" in metrics
    assert metrics["score_centering/sample_ratio_mean"].aggregate() > 0
    tis_keys = {"score_centering/tis_weight_mean", "score_centering/tis_capped_fraction"}
    assert tis_keys <= metrics.keys() if cap else not tis_keys & metrics.keys()
    if cap:
        assert 0 < metrics["score_centering/tis_weight_mean"].aggregate() <= cap
        assert 0 <= metrics["score_centering/tis_capped_fraction"].aggregate() <= 1
    # Only predicting logits receive gradient, never a prompt-only or final row.
    assert logits.grad[0, 0].abs().sum() == 0
    assert logits.grad[0, 4].abs().sum() == 0
    assert logits.grad[0, -1].abs().sum() == 0


class SimpleNamespaceConfig(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def test_missing_forward_adapter_fails_instead_of_training_plain_pg():
    loss_fn = ScoreCenteringLoss(SimpleNamespaceConfig(loss_agg_mode="token-mean"), {})
    with pytest.raises(RuntimeError, match="did not run"):
        loss_fn({"log_probs": torch.zeros(1)}, TensorDict({}, batch_size=[]))


@pytest.fixture
def cpu_unpad(monkeypatch):
    """Replace only FlashAttention's indexing helper; keep real verl padding/loss.

    flash_attn is unavailable on CPU. This reference performs its unpad_input
    indexing in ordinary Torch; no model forward or loss functions are mocked.
    """
    import verl.workers.utils.padding as padding

    def unpad(values, mask):
        lengths = mask.sum(-1).to(torch.int32)
        indices = mask.reshape(-1).nonzero().flatten()
        offsets = torch.cat((torch.zeros(1, dtype=torch.int32), lengths.cumsum(0).to(torch.int32)))
        return values.flatten(0, 1)[indices], indices, offsets, int(lengths.max())

    monkeypatch.setattr(padding, "unpad_input", unpad)


def test_backend_routes_sc_flag_without_outer_is_or_proximal_forward(cpu_unpad):
    import asyncio
    from unittest.mock import MagicMock

    from omegaconf import OmegaConf
    from verl.utils import tensordict_utils as tu

    from rllm.trainer.algorithms.config import AlgorithmConfig, RolloutCorrectionConfig
    from rllm.trainer.verl.verl_backend import VerlBackend

    backend = VerlBackend.__new__(VerlBackend)
    backend.config = OmegaConf.create(
        {
            "trainer": {"balance_batch": False},
            "rllm": {"algorithm": {"score_centering": {"enabled": True}}},
            "actor_rollout_ref": {
                "rollout": {"temperature": 1.0, "n": 1},
                "actor": {"ppo_mini_batch_size": 2, "entropy_coeff": 0, "ppo_epochs": 1, "data_loader_seed": 1, "shuffle": False},
            },
        }
    )
    backend.algorithm_config = AlgorithmConfig(rollout_correction=RolloutCorrectionConfig(bypass_mode=True, tis_mode="token"))
    backend.use_reference_policy = False
    backend.actor_rollout_wg = MagicMock()
    backend.actor_rollout_wg.compute_log_prob.side_effect = AssertionError("SC must bypass proximal forward")
    backend.actor_rollout_wg.update_actor.return_value = tu.get_tensordict({}, non_tensor_dict={"metrics": {}})
    state = SimpleNamespace(backend_batch=transform([episode()]), metrics={}, timing_dict={})
    asyncio.run(backend.process_backend_batch(state))
    assert "rollout_is_weights" not in state.backend_batch.batch
    torch.testing.assert_close(state.backend_batch.batch["old_log_probs"], state.backend_batch.batch["rollout_log_probs"])
    backend._update_actor_with_loss_routing(state.backend_batch, state)
    sent = backend.actor_rollout_wg.update_actor.call_args.args[0]
    assert tu.get(sent, "score_centering") is True
    assert sent["rollout_topk_ids"].is_nested


def test_ordinary_model_output_creates_no_sc_batch_fields():
    output = ModelOutput(prompt_ids=[1, 2], completion_ids=[3], logprobs=[-1.0])
    ep = Episode(id="plain:0", trajectories=[Trajectory(steps=[Step.from_model_output(output)])])
    assert not any(field in transform([ep]).batch for field in SC_FIELDS)


def test_loss_serialization_installs_pinned_fsdp_adapter_idempotently():
    import cloudpickle
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

    from rllm.trainer.verl.sc_fsdp_adapter import install_fsdp_adapter

    install_fsdp_adapter()
    installed = FSDPEngineWithLMHead.prepare_model_outputs
    loss = ScoreCenteringLoss(SimpleNamespaceConfig(loss_agg_mode="token-mean"), {"chunk_size": 3}, 2.0)
    restored = cloudpickle.loads(cloudpickle.dumps(loss))
    assert FSDPEngineWithLMHead.prepare_model_outputs is installed
    assert restored.chunk_size == 3 and restored.tis_cap == 2.0


def test_eos_equal_to_pad_is_attended_without_score_centering():
    output = ModelOutput(prompt_ids=[0, 2], completion_ids=[3, 0], logprobs=[-1.0, -1.0])
    ep = Episode(id="plain:0", termination_reason=TerminationReason.ENV_DONE, trajectories=[Trajectory(steps=[Step.from_model_output(output)], reward=1)])
    batch = transform([ep], response_length=4)
    # prompt [pad, pad, 0, 2] | response [3, 0(EOS), pad, pad]
    assert batch.batch["attention_mask"][0].tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
    assert batch.batch["response_mask"][0].tolist() == [1, 1, 0, 0]
